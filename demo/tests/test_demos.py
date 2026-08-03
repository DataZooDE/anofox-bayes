"""Every demo, run end to end against the real extension.

These are not unit tests of formatting helpers. What they assert is the thing a
demo can silently get wrong: a pipeline that still *runs* after the SQL beneath
it has drifted, or a screen whose prose claims a verdict the query no longer
supports.

Skipping loudly rather than failing when the extension is not built mirrors
`validation/conftest.py` — a developer who has not run `make release` should be
told that, not handed a wall of red.
"""

from __future__ import annotations

import pytest

from anofox_bayes_demo import Kind, Pipeline, connect
from anofox_bayes_demo.duck import ExtensionMissing

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


@pytest.fixture(scope="session")
def extension_available():
    try:
        con, _ = connect()
    except ExtensionMissing as exc:
        pytest.skip(str(exc))
    con.close()
    return True


@pytest.fixture(scope="module")
def _ran(extension_available):
    """Run every pipeline once and cache the results.

    Module-scoped because several of these fits are NUTS and take seconds; the
    assertions below are cheap and there is no reason to pay for the fits more
    than once.
    """
    out = {}
    for demo in DEMOS:
        con, _siblings = connect(demo.wants)
        demo.load(con)
        pipeline = Pipeline(con, demo.build(demo.defaults()))
        for _ in pipeline.run_all():
            pass
        out[demo.name] = (demo, con, pipeline)
    return out


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_every_step_executes(_ran, demo):
    """No step errors, except where the demo deliberately shows a refusal.

    A demo that quietly started erroring on step 6 would still print its intro,
    its dataset panel and its first five steps, which is exactly the failure a
    human skimming the output would miss.
    """
    _demo, _con, pipeline = _ran[demo.name]
    failures = [
        (r.step.title, r.error) for r in pipeline.results if r.error is not None
    ]
    assert not failures, f"{demo.name}: steps errored: {failures}"


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_there_is_exactly_one_fit(_ran, demo):
    """One `anofox_bayes_fit` per pipeline, and every question after it is SQL.

    This is the product claim the whole layout is built to make, so it is worth
    asserting rather than trusting. `dunning` is the documented exception — F5
    has no `group` slot, so a segmented portfolio genuinely needs one fit per
    segment — and it declares that by marking the extra fits `SETUP`.
    """
    _demo, _con, pipeline = _ran[demo.name]
    fits = [r for r in pipeline.results if r.step.kind is Kind.FIT]
    assert len(fits) == 1, f"{demo.name}: expected one FIT step, found {len(fits)}"


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_answering_a_question_is_cheaper_than_fitting(_ran, demo):
    """The "a second question costs no second fit" claim, measured.

    Two things are measured against each other carefully, because a sloppy
    version of this test passes for the wrong reason.

    *Fitting* is the total across every step that calls `anofox_bayes_fit` —
    `dunning` runs three, since F5 has no `group` slot and a segmented portfolio
    genuinely needs one per segment.

    *A question* is a decision step that only **reads**. A step that
    materialises a table is a one-off, not a question: `dunning`'s scoring join
    builds `alive` from 440 debtors × 4 000 draws and takes 136 ms against 87 ms
    of fitting, while every actual question afterwards takes 1.6–2.3 ms. The
    screen claims the *questions* are cheap, and that is what this asserts.

    Deliberately a loose bound. The point is not a benchmark; it is that the
    claim printed on screen is true on whatever machine happens to run this.
    """
    _demo, _con, pipeline = _ran[demo.name]
    fit_ms = sum(
        r.elapsed_ms for r in pipeline.results if "anofox_bayes_fit(" in r.step.sql
    )
    questions = [
        r
        for r in pipeline.results
        if r.step.kind is Kind.DECIDE and "CREATE" not in r.step.sql.upper()
    ]
    assert questions, f"{demo.name}: no read-only decision steps"
    if fit_ms < 200.0:
        pytest.skip(
            f"{demo.name}: fitting took {fit_ms:.0f} ms -- a conjugate fit can be "
            "faster than the queries that read it, and the comparison says nothing "
            "at that scale. The claim on screen is about not needing a re-fit, and "
            "`test_a_what_if_changes_the_answer` is what checks that."
        )
    slowest = max(r.elapsed_ms for r in questions)
    assert slowest < fit_ms, (
        f"{demo.name}: the slowest question took {slowest:.0f} ms against "
        f"{fit_ms:.0f} ms of fitting. The screen claims a further question is cheaper "
        "than re-fitting; if that stops being true the claim has to go."
    )


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_the_fit_reports_the_family_the_demo_advertises(_ran, demo):
    """The header says which family; the draws table has to agree.

    Catches a demo whose prose was updated and whose SQL was not, which is the
    most likely way these files rot.
    """
    _demo, con, _pipeline = _ran[demo.name]
    advertised = demo.family.split(" ")[0]
    reported = con.sql(
        f"SELECT anofox_bayes_family_text(param, value) FROM {demo.draws_table}"
    ).fetchone()
    assert reported is not None
    assert reported[0] == advertised, (
        f"{demo.name}: the header advertises `{advertised}` and the draws table "
        f"reports `{reported[0]}`"
    )


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_a_what_if_changes_the_answer(_ran, demo):
    """Every demo's what-if knob must actually move something.

    A parameter threaded through `build` but not into any SQL would leave the
    modal working, the timings honest, and the answer identical — which is worse
    than not having the knob, because it teaches the viewer the wrong thing about
    what the draws can be asked.
    """
    _demo, con, pipeline = _ran[demo.name]
    assert demo.params, f"{demo.name}: no what-if parameters declared"

    # Keyed by *index*, not by title. Several demos put the parameter's current
    # value in the step title -- which is good on screen and makes a title-keyed
    # comparison silently match nothing, so the test passed for the wrong reason
    # before it was written this way.
    baseline = {
        i: r.rows
        for i, r in enumerate(pipeline.results)
        if r.step.kind is Kind.DECIDE
    }

    moved = dict(demo.defaults())
    param = demo.params[0]
    if isinstance(param.default, str):
        pytest.skip("string parameter; no numeric perturbation defined")
    # Move halfway toward whichever bound is *further away*. Moving toward the
    # nearer one can be a no-op in practice -- a tail threshold of 0.98 nudged to
    # 0.99 flags the same lines -- and the test would then report a knob as
    # unwired when it is merely being asked a question it cannot hear.
    default = float(param.default)
    lo = float(param.minimum) if param.minimum is not None else default * 0.5
    hi = float(param.maximum) if param.maximum is not None else default * 2.0
    moved[param.key] = (
        lo + (default - lo) * 0.5
        if (default - lo) >= (hi - default)
        else default + (hi - default) * 0.5
    )

    altered = Pipeline(con, demo.build(moved))
    start = altered.first_decision_index
    for _ in altered.rerun_from(start):
        pass

    changed = any(
        i in baseline and r.rows != baseline[i]
        for i, r in enumerate(altered.results)
        if r.step.kind is Kind.DECIDE and r.ran
    )
    assert changed, (
        f"{demo.name}: moving `{param.key}` from {default} to {moved[param.key]} left "
        "every decision step's rows identical. The knob is not wired into the SQL."
    )


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_the_fixture_is_deterministic(_ran, demo):
    """Same fixture SQL, same rows — on any machine, with no `setseed()`.

    Every demo builds its data with `anofox_bayes_uniform` / `_std_normal`, which
    are pure functions of `(seed, key, draw)`. A fixture that reached for
    `random()` would still look fine on one run and make every number on screen
    unreproducible.
    """
    con_a, _ = connect(demo.wants)
    con_b, _ = connect(demo.wants)
    demo.load(con_a)
    demo.load(con_b)
    # The first PROFILE step of each demo reads the fixture directly.
    profile = next(
        s for s in demo.build(demo.defaults()) if s.kind in (Kind.PROFILE, Kind.GATE)
    )
    rows_a = con_a.sql(profile.sql).fetchall()
    rows_b = con_b.sql(profile.sql).fetchall()
    assert rows_a == rows_b, f"{demo.name}: the fixture is not deterministic"


@pytest.mark.parametrize("demo", DEMOS, ids=IDS)
def test_every_rendered_string_is_valid_markup(_ran, demo):
    """Every verdict, chart and summary must survive Textual's markup parser.

    Headless printing goes through a plain console that never parses markup, so
    a malformed string is completely invisible there — and in the TUI it raises
    `MarkupError` inside the worker and kills the run mid-pipeline.

    The bug that prompted this: `interval_bar` drew its endpoints as `[` and `]`,
    which Textual reads as a tag, so `[dim]…[/dim]` around a bar failed to
    balance. The endpoints are `├`/`┤` now, and this keeps it that way.
    """
    from textual.markup import MarkupError, to_content

    _demo, con, pipeline = _ran[demo.name]
    problems = []

    def check(label: str, text: str | None) -> None:
        if not text:
            return
        try:
            to_content(text)
        except MarkupError as exc:
            problems.append(f"{label}: {exc}\n    {text[:160]!r}")

    check("intro", demo.intro)
    for i, result in enumerate(pipeline.results, start=1):
        check(f"step{i}.title", result.step.title)
        check(f"step{i}.why", result.step.why)
        if result.step.verdict:
            check(f"step{i}.verdict", result.step.verdict(result.rows))
        if result.step.chart:
            check(f"step{i}.chart", result.step.chart(result.rows))
    check("summary", demo.summary(con, pipeline.results))

    assert not problems, f"{demo.name}: invalid markup:\n" + "\n".join(problems)
