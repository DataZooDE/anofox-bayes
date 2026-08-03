"""What a demo has to provide, and nothing more.

Seven demos share one screen, one set of keybindings and one headless mode. What
differs between them is a fixture, a list of SQL steps and the prose that says
why each step exists — so that is exactly what a `BayesDemo` is, and everything
else lives in `app.py`.

The split matters for a reason beyond tidiness: it makes the seven demos
*comparable*. A viewer who has stepped through the freight audit can step
through the price round without relearning anything, and the differences they
notice are then differences between the models rather than between two people's
idea of a TUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import duckdb

from .steps import Step


@dataclass(frozen=True)
class Param:
    """One knob the what-if modal can turn.

    Every demo has at least one, and it is always a knob that re-runs the
    *decision* steps only. That constraint is the point: a parameter that would
    require a re-fit does not belong here, because the claim being demonstrated
    is precisely that these questions do not need one.
    """

    key: str
    label: str
    default: float | str
    #: Shown under the input in the modal. Say what the number means, not what it is.
    help: str = ""
    #: Inclusive bounds for numeric params; ignored for strings.
    minimum: float | None = None
    maximum: float | None = None

    def parse(self, raw: str) -> float | str:
        """Coerce and clamp a value typed into the modal.

        Clamping rather than rejecting: a service level of 1.4 is a typo with an
        obvious intent, and a modal that refuses it teaches the user about the
        modal instead of about the model.
        """
        if isinstance(self.default, str):
            return raw.strip()
        try:
            value = float(raw)
        except ValueError:
            return self.default
        if self.minimum is not None:
            value = max(self.minimum, value)
        if self.maximum is not None:
            value = min(self.maximum, value)
        return value


class BayesDemo:
    """The specification of one demo. Subclass and fill in.

    Deliberately **not** a dataclass. A subclass declares its fields as plain
    class attributes, and a generated `__init__` would overwrite every one of
    them with the base class's defaults the moment the subclass was instantiated
    — which is exactly what happened the first time this was written, and which
    presents as a demo with no title and no what-if knobs rather than as an
    error.

    `build` is called once at startup and again after every what-if, so it must
    be a pure function of `params` — a demo that mutated state between calls
    would make the "same question, same answer" claim false in the one place a
    viewer can check it.
    """

    #: Console-script name, e.g. `safety-stock`.
    name: str = ""
    #: Shown in the header.
    title: str = ""
    #: The family this demo is about, for the header and the README cross-check.
    family: str = ""
    #: Rich markup: the decision at stake, what is about to happen, "press r".
    intro: str = ""
    #: The table the fit writes, read by the diagnostics modal.
    draws_table: str = "draws"
    #: Optional sibling extensions this demo can use if they are built.
    wants: tuple[str, ...] = ()
    #: What-if knobs.
    params: Sequence[Param] = ()

    def defaults(self) -> dict[str, float | str]:
        return {p.key: p.default for p in self.params}

    # -- to implement -----------------------------------------------------

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        """Create the fixture tables. Called once, before any step runs."""
        raise NotImplementedError

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        """Rich markup summarising the data, with a chart. Two or three lines."""
        raise NotImplementedError

    def build(self, params: dict[str, float | str]) -> list[Step]:
        """The pipeline, as a function of the what-if parameters."""
        raise NotImplementedError

    def summary(self, con: duckdb.DuckDBPyConnection, results) -> str:
        """Rich markup for the closing panel: the decision, in one short table.

        Called after a full run. Returning `""` hides the panel, which is the
        right answer while a run is still in progress.
        """
        return ""
