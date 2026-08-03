"""A demo is an ordered list of SQL steps, and that ordering *is* the lesson.

Every one of these demos has the same shape, because every anofox-bayes workflow
has the same shape:

    profile the data -> gate it -> fit once -> check the fit -> decide, in SQL

The `Pipeline` below is that sentence made executable. It exists so the seven
demos differ only in their steps, and so a viewer stepping through them sees the
same five-act structure each time rather than seven bespoke scripts.

The one thing worth noticing while reading a pipeline: exactly one step is a
`FIT`. Everything after it is `DECIDE`, and every `DECIDE` step reads the same
persisted draws table. That is the product claim — "a second question costs no
second fit" — and it is visible in the step list rather than asserted in prose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import duckdb


class Kind(str, Enum):
    """What a step is for. Drives the icon and the colour, nothing else."""

    #: Load or shape the fixture. No claims made.
    SETUP = "setup"
    #: Look at the data before modelling it.
    PROFILE = "profile"
    #: A quality or identification check that may refuse. Refusal is a result.
    GATE = "gate"
    #: `anofox_bayes_fit`. There is exactly one of these per pipeline.
    FIT = "fit"
    #: R-hat, ESS, divergences, `__status__`, `__group_status__`.
    DIAGNOSE = "diagnose"
    #: A question answered from the persisted draws. No re-fit.
    DECIDE = "decide"


@dataclass(frozen=True)
class Step:
    """One SQL statement, with the sentence that says why it exists.

    `why` is written for a reader with a business background and no SQL, and it
    is the only part of a demo that is allowed to be discursive. `sql` is shown
    verbatim: these demos exist to make the SQL legible, so nothing is elided,
    generated at display time, or prettified into something the engine did not
    run.
    """

    title: str
    why: str
    sql: str
    kind: Kind = Kind.DECIDE
    #: Optional interpreter turning the result rows into one plain sentence.
    verdict: Callable[[list[tuple]], str] | None = None
    #: Optional chart drawn from the result rows.
    chart: Callable[[list[tuple]], str] | None = None
    #: Steps whose result is not a table worth showing (CREATE TABLE, mostly).
    silent: bool = False


@dataclass
class StepResult:
    """What running a step produced. `error` and `rows` are mutually exclusive."""

    step: Step
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None
    #: True once this step has been run at least once.
    ran: bool = False

    @property
    def ok(self) -> bool:
        return self.ran and self.error is None

    @property
    def icon(self) -> str:
        if not self.ran:
            return "·"
        if self.error is not None:
            return "✗"
        if self.step.kind is Kind.GATE:
            return "⚠" if self.refused else "✅"
        return "✅"

    @property
    def refused(self) -> bool:
        """Whether a GATE step reported a refusal rather than a pass.

        The convention is that a gate's first column is a boolean or a status
        string: `False`, `'REFUSE'`, `'PARTIAL'`, `insufficient_data`,
        `degenerate` and `failed` all count as refused. A refusal is not an
        error — several of these demos exist partly to show that a refusal is a
        deliverable — so it gets its own icon rather than the failure one.
        """
        if not self.rows or self.error is not None:
            return False
        first = self.rows[0][0]
        if isinstance(first, bool):
            return not first
        if isinstance(first, str):
            return first.upper() in {
                "REFUSE",
                "PARTIAL",
                "FALSE",
                "INSUFFICIENT_DATA",
                "DEGENERATE",
                "FAILED",
            }
        return False


class Pipeline:
    """Runs steps against one connection, keeping every result.

    Deliberately not lazy and deliberately not cached: a demo's whole point is
    that you can re-run any step and watch it take the time it takes. The
    `elapsed_ms` on each result is what makes the "no re-fit" claim checkable —
    a `FIT` step takes seconds, and every `DECIDE` step after it takes
    milliseconds against the same table.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, steps: Sequence[Step]) -> None:
        self.con = con
        self.steps = list(steps)
        self.results = [StepResult(step=s) for s in self.steps]

    def __len__(self) -> int:
        return len(self.steps)

    def run_step(self, index: int) -> StepResult:
        """Execute one step, capturing rows or the error text.

        An error is captured rather than raised. A demo that crashed on a
        deliberately-refused fit would be telling the opposite of the story these
        demos are for: several steps here are *expected* to error, and the error
        message is the deliverable.
        """
        step = self.steps[index]
        result = StepResult(step=step)
        started = time.perf_counter()
        try:
            cursor = self.con.sql(step.sql)
            if cursor is not None:
                result.columns = list(cursor.columns)
                result.rows = cursor.fetchall()
        except duckdb.Error as exc:
            result.error = str(exc)
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.ran = True
        self.results[index] = result
        return result

    def run_all(self):
        """Run every step in order, yielding each result as it completes.

        A generator so a TUI can paint after each step rather than freezing for
        the length of a NUTS fit.
        """
        for i in range(len(self.steps)):
            yield i, self.run_step(i)

    def rerun_from(self, index: int):
        """Re-run one step and everything after it.

        Used by the what-if modal: change a service level, and only the `DECIDE`
        steps re-run. The `FIT` step above them is untouched, which is the whole
        argument and is visible in the timings.
        """
        for i in range(index, len(self.steps)):
            yield i, self.run_step(i)

    @property
    def fit_index(self) -> int | None:
        """Where the single `anofox_bayes_fit` call sits, if there is one."""
        for i, s in enumerate(self.steps):
            if s.kind is Kind.FIT:
                return i
        return None

    @property
    def first_decision_index(self) -> int:
        """The first step a what-if may re-run: the first `DECIDE` **after** the fit.

        "After the fit" is the load-bearing part. `freight-audit` answers its
        exact rate-card question *before* fitting anything — that step is a
        genuine `DECIDE`, and it is the majority of the recoverable money — so
        taking the first `DECIDE` outright pointed at index 1, and a what-if
        re-ran the fit sitting at index 2. That silently falsified the one claim
        the whole screen is built to make.

        With no fit at all, every decision step is fair game.
        """
        fit = self.fit_index
        start = 0 if fit is None else fit + 1
        for i in range(start, len(self.steps)):
            if self.steps[i].kind is Kind.DECIDE:
                return i
        return len(self.steps)
