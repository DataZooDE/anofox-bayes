"""The shared screen: an SQL pipeline you step through.

Written for a viewer with a business background and no SQL. Every panel
translates one thing, and the SQL is on screen unedited underneath it — the
point of these demos is that the mechanism is legible, not that the output is
pretty.

**The one claim the layout is built to make.** Exactly one step in each pipeline
is a `FIT`. Everything after it reads the same persisted draws table, and the
`#events` log timestamps every statement, so pressing `w` and watching the
answer come back in single-digit milliseconds after a fit that took seconds is
the argument — not a sentence in a README saying the same thing.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Iterable

import duckdb
from rich.console import Console, Group
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Static,
)

from . import format as fmt
from .demo import BayesDemo, Param
from .duck import ExtensionMissing, connect, extension_version
from .steps import Kind, Pipeline, StepResult

#: Braille spinner. One glyph per frame, so a stalled UI is visibly stalled.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _duration(seconds: float) -> str:
    """Human units. `0.004 s` and `211 s` should not be formatted the same way."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{int(seconds) // 60}m {int(seconds) % 60:02d}s"


def _work_description(step: Step) -> str:
    """What this step is actually doing, in the viewer's terms.

    A `FIT` step is where the minutes go, so it says what the sampler is being
    asked for — chains and draws, read out of the config in the SQL itself
    rather than duplicated in prose that could drift from it.
    """
    if step.kind is not Kind.FIT:
        return "querying"
    chains = re.search(r"'chains'\s*:\s*(\d+)", step.sql)
    draws = re.search(r"'draws'\s*:\s*(\d+)", step.sql)
    warmup = re.search(r"'warmup'\s*:\s*(\d+)", step.sql)
    if draws and chains:
        total = int(chains.group(1)) * int(draws.group(1))
        extra = f" after {warmup.group(1)} warmup" if warmup else ""
        return (
            f"sampling {chains.group(1)} chains x {draws.group(1)} draws "
            f"({total:,} total){extra}"
        )
    return "fitting"


_KIND_STYLE = {
    Kind.SETUP: "dim",
    Kind.PROFILE: "cyan",
    Kind.GATE: "yellow",
    Kind.FIT: "magenta",
    Kind.DIAGNOSE: "blue",
    Kind.DECIDE: "green",
}


class WhatIfModal(ModalScreen[dict | None]):
    """Change the question, not the model.

    Dismisses with a dict of new parameter values, or `None` on escape. The app
    then rebuilds the pipeline and re-runs from the first `DECIDE` step — never
    from the fit.
    """

    CSS = """
    WhatIfModal { align: center middle; }
    #whatif-panel {
        width: 80; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    /* The fields scroll; the buttons do not. With two parameters the panel is
       taller than an 80x24 terminal, and when Apply lived inside the scrolling
       region it was simply off-screen -- a viewer on a standard terminal could
       not apply a what-if at all on five of the seven demos. */
    #whatif-fields { height: 1fr; }
    #whatif-actions { height: auto; dock: bottom; align-horizontal: right; }
    #whatif-panel Input { margin-bottom: 1; }
    .whatif-help { color: $text-muted; margin-bottom: 1; }
    """
    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def __init__(self, params: Iterable[Param], current: dict) -> None:
        super().__init__()
        self._params = list(params)
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="whatif-panel"):
            with VerticalScroll(id="whatif-fields"):
                yield Label("[b]Ask a different question[/b]")
                yield Label(
                    "These re-run the decision steps only. The fit above them is not "
                    "touched — watch the timings in the activity log.",
                    classes="whatif-help",
                )
                for p in self._params:
                    yield Label(f"[b]{p.label}[/b]")
                    yield Input(value=str(self._current[p.key]), id=f"in-{p.key}")
                    if p.help:
                        yield Label(p.help, classes="whatif-help")
            with Horizontal(id="whatif-actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Apply", variant="primary", id="apply")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "apply":
            self.dismiss(None)
            return
        values = {}
        for p in self._params:
            raw = self.query_one(f"#in-{p.key}", Input).value
            values[p.key] = p.parse(raw)
        self.dismiss(values)


class TableModal(ModalScreen[None]):
    """A scrollable table — diagnostics, refused groups, the full SQL."""

    CSS = """
    TableModal { align: center middle; }
    #table-panel {
        width: 92%; height: 88%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    """
    BINDINGS = [("escape", "dismiss(None)", "Close")]

    def __init__(self, title: str, renderable) -> None:
        super().__init__()
        self._title = title
        self._renderable = renderable

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="table-panel") as panel:
            panel.border_title = self._title
            yield Static(self._renderable)


class BayesDemoApp(App):
    """One screen, seven demos."""

    CSS = """
    #intro   { height: 9;  border: round $accent; padding: 0 1; }
    #dataset { height: 6;  border: round $accent; padding: 0 1; }
    #status  { height: 1;  padding: 0 1; color: $text-muted; }
    #body    { height: 3fr; }
    #steps   { width: 46; border: round $accent; padding: 0 1; }
    #detail  { width: 1fr; border: round $accent; padding: 0 1; }
    #why     { margin-bottom: 1; }
    #sql     { margin-bottom: 1; }
    #result  { margin-bottom: 1; }
    #summary { height: 11; border: round $accent; padding: 0 1; display: none; }
    #summary.visible { display: block; }
    #events  { height: 1fr; min-height: 7; border: round $accent; }
    """

    BINDINGS = [
        ("r", "run", "Run"),
        ("up", "prev_step", "Prev step"),
        ("down", "next_step", "Next step"),
        ("enter", "rerun_step", "Re-run step"),
        ("s", "show_sql", "All SQL"),
        ("d", "show_diagnostics", "Diagnostics"),
        ("w", "what_if", "What-if"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, demo: BayesDemo) -> None:
        super().__init__()
        self.demo = demo
        self.params = demo.defaults()
        self.con: duckdb.DuckDBPyConnection | None = None
        self.pipeline: Pipeline | None = None
        self.selected = 0
        #: Index of the step currently executing, or None between steps.
        self._active_step: int | None = None
        self._active_since = 0.0
        # **Not `_running`.** Textual's `App` already owns an attribute by that
        # name and sets it True for the lifetime of the app, so the guard in
        # `action_run` fired on the very first key press and `r` silently did
        # nothing -- in every demo, for every user. The headless path calls the
        # pipeline directly and never touches this, which is exactly why it went
        # unseen until a pilot-driven test pressed a real key.
        self._pipeline_running = False
        self._status = "Press [b]r[/b] to run the pipeline."

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(self.demo.intro, id="intro")
        yield Static("", id="dataset")
        yield Static(self._status, id="status")
        with Horizontal(id="body"):
            yield Static("", id="steps")
            with VerticalScroll(id="detail"):
                yield Static("", id="why")
                yield Static("", id="sql")
                yield Static("", id="result")
                yield Static("", id="verdict")
        yield Static("", id="summary")
        yield Log(id="events")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.demo.title
        self.sub_title = f"anofox_bayes · {self.demo.family}"
        self.query_one("#events", Log).can_focus = False
        self.query_one("#detail").can_focus = False
        self.set_focus(None)
        self.query_one("#intro", Static).border_title = "📌 The decision at stake"
        self.query_one("#dataset", Static).border_title = "📈 The data"
        self.query_one("#steps", Static).border_title = "🧭 Pipeline"
        self.query_one("#detail").border_title = "🔬 Selected step"
        self.query_one("#summary", Static).border_title = "🏁 The answer"
        self.query_one("#events", Log).border_title = "📜 Every statement, with its timing"

        try:
            self.con, siblings = connect(self.demo.wants)
        except ExtensionMissing as exc:
            self._set_status(f"[red]{exc}[/red]")
            return
        self.demo.load(self.con)
        self.pipeline = Pipeline(self.con, self.demo.build(self.params))
        self.query_one("#dataset", Static).update(self.demo.dataset_panel(self.con))
        log = self.query_one("#events", Log)
        log.write_line(f"anofox_bayes {extension_version(self.con)} loaded")
        for note in siblings.notes:
            log.write_line(note)
        self._render_steps()
        self._show_step(0)
        # Four frames a second: enough that the spinner reads as motion and the
        # elapsed clock as a clock, cheap enough to be free next to a fit.
        self.set_interval(0.25, self._tick)

    # -- rendering --------------------------------------------------------

    def _set_status(self, markup: str) -> None:
        self._status = markup
        try:
            self.query_one("#status", Static).update(markup)
        except Exception:
            pass

    def _render_steps(self) -> None:
        if self.pipeline is None:
            return
        lines = []
        for i, result in enumerate(self.pipeline.results):
            marker = "❯" if i == self.selected else " "
            style = _KIND_STYLE[result.step.kind]
            if i == self._active_step:
                # The running step reports a live clock rather than an icon, so a
                # three-minute fit never looks like a frozen screen.
                elapsed = time.monotonic() - self._active_since
                icon = _SPINNER[int(elapsed * 8) % len(_SPINNER)]
                timing = f"[b]{_duration(elapsed)}[/b]"
            else:
                icon = result.icon
                timing = (
                    f"[dim]{_duration(result.elapsed_ms / 1000.0)}[/dim]"
                    if result.ran
                    else ""
                )
            lines.append(
                f"{marker} {icon} [{style}]{result.step.kind.value:<9}[/{style}] "
                f"{result.step.title}  {timing}"
            )
        fit = self.pipeline.fit_index
        if fit is not None:
            lines.append("")
            lines.append(
                f"[dim]One fit (step {fit + 1}). Every step below it reads the same\n"
                f"draws table — press [b]w[/b] to change the question and watch.[/dim]"
            )
        self.query_one("#steps", Static).update("\n".join(lines))

    def _show_step(self, index: int) -> None:
        if self.pipeline is None or not self.pipeline.results:
            return
        index = max(0, min(len(self.pipeline.results) - 1, index))
        self.selected = index
        result = self.pipeline.results[index]
        step = result.step

        self.query_one("#why", Static).update(f"[b]Why this step[/b]\n{step.why}")
        self.query_one("#sql", Static).update(fmt.sql_panel(step.sql))

        if not result.ran:
            self.query_one("#result", Static).update("[dim](not run yet — press r)[/dim]")
            self.query_one("#verdict", Static).update("")
        elif result.error is not None:
            self.query_one("#result", Static).update(
                f"[red]{result.error}[/red]\n\n"
                "[dim]An error here may be the point: several steps in these demos "
                "exist to show what the extension refuses, and the message is the "
                "deliverable.[/dim]"
            )
            self.query_one("#verdict", Static).update("")
        else:
            if step.silent or not result.columns:
                self.query_one("#result", Static).update(
                    f"[dim]{result.elapsed_ms:,.0f} ms — no rows to show "
                    "(this step writes a table).[/dim]"
                )
            else:
                self.query_one("#result", Static).update(
                    fmt.result_table(result.columns, result.rows)
                )
            parts = []
            if step.chart:
                parts.append(step.chart(result.rows))
            if step.verdict:
                parts.append(step.verdict(result.rows))
            self.query_one("#verdict", Static).update("\n".join(p for p in parts if p))

        self._render_steps()

    # -- running ----------------------------------------------------------

    @work(thread=True, exclusive=True, group="pipeline")
    def _run(self, start: int = 0) -> None:
        """Execute the pipeline on a worker thread.

        DuckDB calls block, and a NUTS fit blocks for **minutes** — the F1 fit in
        the safety-stock demo takes about three and a half on 48 parts. Running
        that on the UI thread would freeze the screen for exactly the step the
        viewer most wants to watch, which is why the worker exists; and it is
        why every step announces itself *before* it runs rather than only after,
        so the screen has something true to show for the whole wait.
        """
        assert self.pipeline is not None
        self._pipeline_running = True
        indices = (
            range(len(self.pipeline))
            if start == 0
            else range(start, len(self.pipeline))
        )
        for index in indices:
            self.call_from_thread(self._on_step_start, index)
            result = self.pipeline.run_step(index)
            self.call_from_thread(self._on_step_done, index, result)
        self.call_from_thread(self._on_run_done, start)
        self._pipeline_running = False

    def _on_step_start(self, index: int) -> None:
        """Announce a step before it runs, and start the clock ticking on it."""
        assert self.pipeline is not None
        self._active_step = index
        self._active_since = time.monotonic()
        self.selected = index
        self._show_step(index)
        self._render_steps()
        self._tick()

    def _tick(self) -> None:
        """Repaint the elapsed time of the step currently running.

        Driven by an interval timer rather than by the worker, because the
        worker is *inside* the long call and cannot report from there. Without
        this the status line said "Running… step 3 of 8" and then did not change
        for three and a half minutes, which is indistinguishable from a hang.
        """
        if self._active_step is None or self.pipeline is None:
            return
        step = self.pipeline.steps[self._active_step]
        elapsed = time.monotonic() - self._active_since
        spinner = _SPINNER[int(elapsed * 8) % len(_SPINNER)]
        detail = _work_description(step)
        self._set_status(
            f"{spinner} step {self._active_step + 1} of {len(self.pipeline)} — "
            f"[b]{step.title}[/b] — {detail} — [b]{_duration(elapsed)}[/b] elapsed"
        )
        self._render_steps()

    def _on_step_done(self, index: int, result: StepResult) -> None:
        self._active_step = None
        log = self.query_one("#events", Log)
        state = "ERROR" if result.error else ("REFUSED" if result.refused else "ok")
        log.write_line(
            f"[{index + 1:>2}] {result.step.kind.value:<9} {result.step.title} "
            f"— {result.elapsed_ms:,.1f} ms — {state}"
        )
        if result.error:
            log.write_line(f"      {result.error.splitlines()[0]}")
        self.selected = index
        self._show_step(index)
        done = sum(1 for r in self.pipeline.results if r.ran) if self.pipeline else 0
        total = len(self.pipeline) if self.pipeline else 0
        self._set_status(
            f"step {done} of {total} done — {result.step.title} took "
            f"{_duration(result.elapsed_ms / 1000.0)}"
        )

    def _on_run_done(self, start: int) -> None:
        assert self.pipeline is not None and self.con is not None
        failures = [r for r in self.pipeline.results if r.error]
        refusals = [r for r in self.pipeline.results if r.refused]
        fit = self.pipeline.fit_index
        fit_ms = self.pipeline.results[fit].elapsed_ms if fit is not None else 0.0
        decide_ms = sum(
            r.elapsed_ms
            for r in self.pipeline.results
            if r.step.kind is Kind.DECIDE and r.ran
        )
        note = (
            f"the fit took [b]{fit_ms:,.0f} ms[/b]; every question after it, "
            f"[b]{decide_ms:,.0f} ms[/b] in total"
        )
        if start > 0:
            self._set_status(
                f"What-if answered in [b]{decide_ms:,.0f} ms[/b] — no re-fit."
            )
        else:
            self._set_status(
                f"Done: {len(self.pipeline)} steps, {len(refusals)} refusal(s), "
                f"{len(failures)} error(s) — {note}."
            )
        summary = self.demo.summary(self.con, self.pipeline.results)
        panel = self.query_one("#summary", Static)
        if summary:
            panel.update(summary)
            panel.add_class("visible")

    # -- actions ----------------------------------------------------------

    def action_run(self) -> None:
        if self.pipeline is None or self._pipeline_running:
            return
        self.query_one("#summary", Static).remove_class("visible")
        self._set_status("Running…")
        self._run(0)

    def action_rerun_step(self) -> None:
        if self.pipeline is None or self._pipeline_running:
            return
        result = self.pipeline.run_step(self.selected)
        self._on_step_done(self.selected, result)

    def action_prev_step(self) -> None:
        self._show_step(self.selected - 1)

    def action_next_step(self) -> None:
        self._show_step(self.selected + 1)

    def action_show_sql(self) -> None:
        """Every statement, as Rich renderables rather than captured text.

        An earlier version rendered this through `Console().capture()` and handed
        the resulting string to `Static`. That string carries ANSI escapes, and
        `Static` parses its argument as *Textual markup* — so pressing `s` raised
        `MarkupError` on every demo. Passing renderables straight through is both
        correct and one step shorter.
        """
        if self.pipeline is None:
            return
        parts: list = []
        for i, step in enumerate(self.pipeline.steps, start=1):
            parts.append(Text.from_markup(f"[b]-- {i}. {step.title} ({step.kind.value})[/b]"))
            parts.append(fmt.sql_panel(step.sql))
            parts.append(Text(""))
        self.push_screen(TableModal("📄 Every statement this demo runs", Group(*parts)))

    def action_show_diagnostics(self) -> None:
        if self.con is None or self.pipeline is None:
            return
        draws = getattr(self.demo, "draws_table", "draws")
        try:
            columns, rows = fmt.diagnostics(self.con, draws)
            table = fmt.result_table(columns, rows, max_rows=40)
            divergent = fmt.divergences(self.con, draws)
            unready = fmt.unready_groups(self.con, draws)
        except duckdb.Error as exc:
            self.push_screen(
                TableModal("🩺 Diagnostics", Text.from_markup(f"[red]{exc}[/red]"))
            )
            return

        parts: list = [Text.from_markup(fmt.status_line(self.con, draws)), Text("")]
        if divergent is None:
            parts.append(
                Text.from_markup(
                    "[dim]No sampler statistics on this table — the fit was served by "
                    "the exact or Laplace engine, so there are no divergences to "
                    "report. That is different from reporting zero.[/dim]"
                )
            )
        else:
            colour = "green" if divergent == 0 else "red"
            parts.append(
                Text.from_markup(f"Divergent transitions: [{colour}]{divergent}[/{colour}]")
            )
        if unready:
            parts.append(Text(""))
            parts.append(Text.from_markup("[b]Groups the fit refused individually:[/b]"))
            for group, verdict in unready:
                parts.append(Text.from_markup(f"  [yellow]{group}[/yellow] — {verdict}"))
        parts.append(Text(""))
        parts.append(
            Text.from_markup(
                "[dim]Gate: R-hat ≤ 1.01 and ESS ≥ 400, from the extension's own "
                "macros.[/dim]"
            )
        )
        parts.append(table)
        self.push_screen(TableModal("🩺 Diagnostics", Group(*parts)))

    @work
    async def action_what_if(self) -> None:
        if self.pipeline is None or self._pipeline_running or not self.demo.params:
            return
        if not any(r.ran for r in self.pipeline.results):
            self._set_status("Run the pipeline first — press [b]r[/b].")
            return
        values = await self.push_screen_wait(WhatIfModal(self.demo.params, self.params))
        if values is None:
            return
        self.params = values
        start = self.pipeline.first_decision_index
        old = self.pipeline
        self.pipeline = Pipeline(self.con, self.demo.build(self.params))
        # Keep the already-run results for everything above the decision steps:
        # those are the fit and its gates, and they have not changed.
        for i in range(start):
            self.pipeline.results[i] = old.results[i]
        self.query_one("#events", Log).write_line(
            f"what-if: {self.params} — re-running from step {start + 1}, no re-fit"
        )
        self._render_steps()
        self._run(start)


# --------------------------------------------------------------------------
# Headless
# --------------------------------------------------------------------------


def run_headless(demo: BayesDemo, params: dict) -> int:
    """Run the pipeline and print it. The CI-safe path, and the fastest review.

    Prints the same SQL, the same prose and the same results the TUI shows, so a
    reviewer can read a whole demo in a scrollback rather than by pressing keys —
    and so the pipelines are testable without a terminal.
    """
    console = Console()
    try:
        con, siblings = connect(demo.wants)
    except ExtensionMissing as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(f"[b]{demo.title}[/b] — anofox_bayes {extension_version(con)}")
    for note in siblings.notes:
        console.print(f"[dim]{note}[/dim]")
    console.print()
    console.print(demo.intro)
    console.print()

    demo.load(con)
    console.print(demo.dataset_panel(con))
    console.print()

    pipeline = Pipeline(con, demo.build(params))
    errors = 0
    for index in range(len(pipeline)):
        step = pipeline.steps[index]
        console.rule(f"{index + 1}. {step.title}  [{step.kind.value}]")
        console.print(step.why)
        console.print()
        console.print(fmt.sql_panel(step.sql))
        console.print()
        # Announce the work *before* doing it. A `FIT` step here runs for
        # minutes, and a headless run that printed nothing until it finished
        # was indistinguishable from a hang -- which is exactly what it looked
        # like the first time anyone timed it.
        console.print(f"[dim]… {_work_description(step)}[/dim]")
        result = pipeline.run_step(index)
        if result.error:
            console.print(f"[red]{result.error}[/red]")
            errors += 1
        elif result.columns and not step.silent:
            console.print(fmt.result_table(result.columns, result.rows))
        else:
            console.print("[dim](no rows)[/dim]")
        if step.chart:
            console.print(step.chart(result.rows))
        if step.verdict:
            console.print(step.verdict(result.rows))
        console.print(f"[dim]{_duration(result.elapsed_ms / 1000.0)}[/dim]")

    summary = demo.summary(con, pipeline.results)
    if summary:
        console.rule("The answer")
        console.print(summary)
    return 1 if errors else 0


def main(demo: BayesDemo, argv: list[str] | None = None) -> int:
    """Console-script entry point shared by all seven demos."""
    parser = argparse.ArgumentParser(description=demo.title)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the pipeline and print it to stdout instead of opening the TUI.",
    )
    for p in demo.params:
        parser.add_argument(
            f"--{p.key.replace('_', '-')}",
            default=None,
            help=f"{p.label} (default: {p.default}). {p.help}",
        )
    args = parser.parse_args(argv)

    params = demo.defaults()
    for p in demo.params:
        raw = getattr(args, p.key, None)
        if raw is not None:
            params[p.key] = p.parse(str(raw))

    if args.headless:
        return run_headless(demo, params)
    app = BayesDemoApp(demo)
    app.params = params
    app.run()
    return 0
