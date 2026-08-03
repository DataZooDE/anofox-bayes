"""Drive the actual TUI, not the pipeline underneath it.

`test_demos.py` runs each `Pipeline` directly, which is the right way to check the
SQL — and it says nothing at all about the application. It would stay green with
a `compose()` that yields the wrong widget ids, a worker that never posts back, a
modal that raises on open, or a what-if that reruns from the wrong index.

Textual's `run_test()` pilot is the fix: it starts the real app against a
headless driver, so key presses go through the real bindings, the real workers
and the real screen stack.
"""

from __future__ import annotations

import asyncio

import pytest

from anofox_bayes_demo import BayesDemoApp
from anofox_bayes_demo.duck import ExtensionMissing, connect

from cash_runway import DEMO as CASH_RUNWAY
from delivery_promise import DEMO as DELIVERY_PROMISE
from dunning import DEMO as DUNNING
from freight_audit import DEMO as FREIGHT_AUDIT
from intervention import DEMO as INTERVENTION
from price_increase import DEMO as PRICE_INCREASE
from safety_stock import DEMO as SAFETY_STOCK

DEMOS = [
    SAFETY_STOCK,
    DELIVERY_PROMISE,
    INTERVENTION,
    CASH_RUNWAY,
    DUNNING,
    PRICE_INCREASE,
    FREIGHT_AUDIT,
]
IDS = [d.name for d in DEMOS]

pytestmark = pytest.mark.asyncio(loop_scope="function")


@pytest.fixture(scope="session", autouse=True)
def extension_available():
    try:
        con, _ = connect()
    except ExtensionMissing as exc:
        pytest.skip(str(exc))
    con.close()


def _text(widget) -> str:
    """The visible text of a `Static`, whatever Textual is calling it this major.

    `Static.renderable` was the obvious accessor and does not exist in 8.x; going
    through `visual` keeps this test about the app rather than about the version.
    """
    for attr in ("visual", "content", "_content"):
        value = getattr(widget, attr, None)
        if value is not None:
            return str(value)
    return ""


async def _run_pipeline(pilot) -> None:
    """Press `r` and wait for the threaded worker to finish.

    The pipeline runs on a `@work(thread=True)` worker precisely so a NUTS fit
    does not freeze the UI, which means the test has to wait for it the same way
    a user does.
    """
    await pilot.press("r")
    await _settle(pilot)


async def _until(pilot, predicate, message: str, timeout: float = 60.0) -> None:
    """Pump the UI until `predicate()` holds, or fail with `message`."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(message)


async def _settle(pilot, timeout: float = 300.0) -> None:
    """Pump the UI until every step has run, or give up loudly.

    **Deliberately does not call `workers.wait_for_complete()`.** The pipeline
    runs on a thread worker that hands each result back with
    `call_from_thread`, which blocks that thread until the event loop runs the
    callback; awaiting the worker from the loop at the same time deadlocks, and
    the first version of this helper hung the whole suite on the second test.
    Polling the pipeline's own state is both simpler and the thing actually
    being waited for.

    Bounded by wall clock rather than by iteration count, because the seven
    demos' fits differ by two orders of magnitude and an iteration budget that
    suits `freight-audit` starves `dunning`.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        pipeline = pilot.app.pipeline
        if pipeline is not None and all(r.ran for r in pipeline.results):
            await pilot.pause()
            return
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"{pilot.app.demo.name}: the pipeline had not finished after {timeout:.0f}s"
    )


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
async def test_the_whole_interactive_journey(demo):
    """Mount, run, step, open every modal, apply a what-if — in one app.

    **Deliberately one test per demo rather than six.** The earlier shape ran a
    full pipeline per assertion: 42 pipelines, and with `safety-stock`'s F1 fit
    alone taking three and a half minutes the suite came to about an hour. A
    suite nobody runs catches nothing. This drives the same interactions against
    a single app instance, which is also closer to what a person actually does.
    """
    app = BayesDemoApp(demo)
    async with app.run_test() as pilot:
        # --- mount ---------------------------------------------------------
        await pilot.pause()
        for widget_id in ("#intro", "#dataset", "#status", "#steps", "#detail",
                          "#summary", "#events"):
            assert app.query(widget_id), f"{demo.name}: {widget_id} is missing"
        assert app.pipeline is not None and app.con is not None
        assert _text(app.query_one("#dataset")).strip(), (
            f"{demo.name}: the dataset panel is empty"
        )

        # --- run, through the real binding and the real worker --------------
        await pilot.press("r")
        await _settle(pilot)

        unrun = [r.step.title for r in app.pipeline.results if not r.ran]
        assert not unrun, f"{demo.name}: steps never ran in the app: {unrun}"
        errored = [(r.step.title, r.error) for r in app.pipeline.results if r.error]
        assert not errored, f"{demo.name}: steps errored in the app: {errored}"
        assert "visible" in app.query_one("#summary").classes, (
            f"{demo.name}: the summary panel never appeared"
        )
        log = app.query_one("#events")
        assert log.line_count >= len(app.pipeline), (
            f"{demo.name}: activity log has {log.line_count} lines for "
            f"{len(app.pipeline)} steps"
        )

        # --- step through every step, painting each -------------------------
        for _ in range(len(app.pipeline) + 2):
            await pilot.press("down")
            await pilot.pause()
        assert app.selected == len(app.pipeline) - 1
        for _ in range(len(app.pipeline) + 2):
            await pilot.press("up")
            await pilot.pause()
        assert app.selected == 0

        # --- the modals ------------------------------------------------------
        depth = len(app.screen_stack)
        await pilot.press("s")
        await pilot.pause()
        assert len(app.screen_stack) == depth + 1, f"{demo.name}: `s` opened nothing"
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("d")
        await pilot.pause()
        assert len(app.screen_stack) == depth + 1, f"{demo.name}: `d` opened nothing"
        rendered = _text(app.screen_stack[-1].query_one("Static"))
        assert "Traceback" not in rendered and "no metadata" not in rendered, (
            f"{demo.name}: diagnostics modal errored:\n{rendered[:400]}"
        )
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == depth

        # --- the what-if, which must not re-fit ------------------------------
        fit_index = app.pipeline.fit_index
        assert fit_index is not None
        fit_ms_before = app.pipeline.results[fit_index].elapsed_ms
        first_decision = app.pipeline.first_decision_index
        rows_before = [r.rows for r in app.pipeline.results]

        param = app.demo.params[0]
        if isinstance(param.default, str):
            return
        default = float(param.default)
        lo = float(param.minimum) if param.minimum is not None else default * 0.5
        hi = float(param.maximum) if param.maximum is not None else default * 2.0
        moved = (
            lo + (default - lo) * 0.5
            if (default - lo) >= (hi - default)
            else default + (hi - default) * 0.5
        )

        await pilot.press("w")
        await pilot.pause()
        assert len(app.screen_stack) > depth, f"{demo.name}: `w` opened nothing"
        app.screen_stack[-1].query_one(f"#in-{param.key}").value = str(moved)
        await pilot.pause()
        await pilot.click("#apply")
        # Wait for the *value* to land before waiting for the pipeline. Applying
        # runs through an async worker, and the previous pipeline still has every
        # step marked `ran` -- so settling on the pipeline alone returns
        # instantly and the assertion below races the modal. Two of the seven
        # demos won that race and five lost it, which is the worst kind of green.
        await _until(
            pilot,
            lambda: app.params[param.key] == pytest.approx(moved),
            f"{demo.name}: the what-if value never reached the app",
        )
        await _settle(pilot)

        assert app.params[param.key] == pytest.approx(moved), (
            f"{demo.name}: the modal did not apply the new value"
        )
        assert app.pipeline.results[fit_index].elapsed_ms == fit_ms_before, (
            f"{demo.name}: the fit re-ran on a what-if; it must be spliced through"
        )
        rows_after = [r.rows for r in app.pipeline.results]
        assert any(
            rows_before[i] != rows_after[i] for i in range(first_decision, len(rows_after))
        ), f"{demo.name}: the what-if left every decision step identical"
        errored = [(r.step.title, r.error) for r in app.pipeline.results if r.error]
        assert not errored, f"{demo.name}: what-if rerun errored: {errored}"
