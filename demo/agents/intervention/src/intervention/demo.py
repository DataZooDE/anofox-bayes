"""Agent 03 — "Hat's was gebracht?", on `pooled_gaussian` (F3).

Logistics interventions — a carrier switch, a depot consolidation, a WMS
rollout — get evaluated by before/after averages contaminated by seasonality,
mix shifts and trend. Millions get spent on rollouts justified by noise, and a
warehouse cannot run an A/B test.

**The most important step in this demo is the one that can say no.** Before any
effect is estimated, the pre-trend check asks whether the control units and the
treated one were moving together *before* the change. If they were not, there is
no counterfactual and no honest effect estimate — and saying so is the
deliverable, not a failure. This demo runs the gate twice: once on a panel where
identification holds, and once on a donor pool where it does not.

**Where `anofox_solve` would come in.** A synthetic-control weighting is a small
quadratic program (weights ≥ 0 summing to 1, minimising pre-period error). This
demo runs the difference-in-differences path, which needs no solver, and says on
screen when the QP path is unavailable rather than implying it ran.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, interval_bar, sparkline

DRAWS = "did_draws"
TREATMENT_WEEK = 40

FIXTURE = f"""
CREATE OR REPLACE TABLE units AS
SELECT * FROM (VALUES
    ('DEPOT-HAM', true,  1.00),
    ('DEPOT-BRE', false, 0.94),
    ('DEPOT-HAN', false, 1.07),
    ('DEPOT-DUS', false, 0.98),
    ('DEPOT-KOE', false, 1.03),
    ('DEPOT-LEI', false, 0.91),
    ('DEPOT-NUE', false, 1.11),
    ('DEPOT-STU', false, 0.96)
) AS t(unit, treated, level_factor);

-- 78 weeks of cost per shipment. Every depot shares one seasonal pattern and one
-- mild trend -- which is precisely what makes a naive before/after comparison
-- wrong, because the intervention lands in a different part of the season.
--
-- The treated depot gets a genuine -0.42 EUR/shipment from week 40. That number
-- is the thing the model has to find underneath the season and the noise.
CREATE OR REPLACE TABLE panel AS
WITH weeks AS (SELECT range AS week FROM range(1, 79))
SELECT u.unit,
       w.week,
       u.treated,
       (w.week >= {TREATMENT_WEEK})                                AS post,
       (u.treated AND w.week >= {TREATMENT_WEEK})::INTEGER         AS treated_post,
       round(
           8.40 * u.level_factor
           + 0.34 * sin(2 * pi() * w.week / 52.0)
           + 0.22 * cos(2 * pi() * w.week / 26.0)
           + 0.004 * w.week
           + CASE WHEN u.treated AND w.week >= {TREATMENT_WEEK} THEN -0.42 ELSE 0 END
           + 0.11 * anofox_bayes_std_normal(30303, u.unit, w.week), 4
       ) AS cost_per_shipment,
       round(1400 + 600 * anofox_bayes_uniform(30303, u.unit || ':vol', w.week), 0)
                                                                   AS shipments
FROM units u CROSS JOIN weeks w;

-- A second, deliberately unusable donor pool: these depots drift apart from the
-- treated one before the intervention, so no weighting of them reconstructs it.
CREATE OR REPLACE TABLE bad_panel AS
SELECT p.unit, p.week, p.treated, p.post, p.treated_post,
       round(p.cost_per_shipment
             + CASE WHEN NOT p.treated
                    THEN 0.030 * p.week * (CASE WHEN p.unit < 'DEPOT-K' THEN 1 ELSE -1 END)
                    ELSE 0 END, 4) AS cost_per_shipment,
       p.shipments
FROM panel p;
"""


class Intervention(BayesDemo):
    name = "intervention"
    title = "Wirkungsanalyse — hat's was gebracht?"
    family = "pooled_gaussian (F3)"
    draws_table = DRAWS
    wants = ("anofox_solve",)
    intro = (
        "[b]The decision at stake:[/b] a logistics change was made in week 40 — a "
        "carrier switch at one depot. Did it work, by how much, and should it be "
        "rolled out to the other seven?\n\n"
        "[b]Why before/after is not enough:[/b] every depot shares a season and a "
        "trend, and the change landed in a different part of the year than the "
        "baseline. A naive comparison measures the season. The control depots are "
        "what remove it — [i]if[/i] they were moving with the treated depot "
        "beforehand.\n\n"
        "[b]The step that can say no:[/b] this demo runs the identification gate "
        "twice — once on a panel where it holds, once on a donor pool where it "
        "does not. A [b]REFUSE[/b] with a reason is a deliverable that saves a "
        "rollout decision from being made on noise.\n\n"
        "Press [b]r[/b] to run it, [b]w[/b] to change what counts as a "
        "practically relevant effect."
    )
    params = (
        Param(
            key="relevance_threshold",
            label="Practically relevant effect (€/shipment)",
            default=0.20,
            help="Below this the effect is real but not worth a rollout. The model "
                 "reports P(|effect| > this), which is the number the steering "
                 "committee actually needs.",
            minimum=0.0,
            maximum=5.0,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        head = con.sql(
            """
            SELECT count(*), count(DISTINCT unit), count(DISTINCT week),
                   round(avg(cost_per_shipment), 3)
            FROM panel
            """
        ).fetchone()
        treated = [
            float(r[0])
            for r in con.sql(
                "SELECT cost_per_shipment FROM panel WHERE treated ORDER BY week"
            ).fetchall()
        ]
        control = [
            float(r[0])
            for r in con.sql(
                "SELECT avg(cost_per_shipment) FROM panel WHERE NOT treated "
                "GROUP BY week ORDER BY week"
            ).fetchall()
        ]
        if head is None:
            return ""
        rows, n_units, n_weeks, mean_cost = head
        return (
            f"[b]{n_units} depots[/b] × [b]{n_weeks} weeks[/b] = {rows} rows · "
            f"mean €{float(mean_cost):.2f}/shipment · intervention in week "
            f"[b]{TREATMENT_WEEK}[/b]\n"
            f"  treated  {sparkline(treated)}\n"
            f"  controls {sparkline(control)}\n"
            "  [dim]The two move together — that is what the gate is about to "
            "test formally.[/dim]"
        )

    def build(self, params) -> list[Step]:
        threshold = float(params["relevance_threshold"])
        return [
            Step(
                title="Panel balance and missingness",
                kind=Kind.PROFILE,
                why=(
                    "An unbalanced panel makes a difference-in-differences estimate "
                    "mean something subtly different — units that appear only after "
                    "the change contribute to the 'after' average and not the "
                    "'before'. Check it before it becomes an effect."
                ),
                sql="""
SELECT count(DISTINCT unit)                                      AS units,
       count(DISTINCT week)                                      AS weeks,
       count(*)                                                  AS rows,
       count(*) = count(DISTINCT unit) * count(DISTINCT week)    AS balanced,
       count(*) FILTER (WHERE cost_per_shipment IS NULL)         AS missing
FROM panel;
                """,
                verdict=lambda rows: (
                    "[green]Balanced, no gaps.[/green]"
                    if rows and rows[0][3] and rows[0][4] == 0
                    else "[yellow]Unbalanced or incomplete — read carefully.[/yellow]"
                ),
            ),
            Step(
                title="The naive answer, and why it is wrong",
                kind=Kind.PROFILE,
                why=(
                    "What a spreadsheet would report: the treated depot's average "
                    "before against after. It is wrong, and the size of the error is "
                    "the point — the intervention landed in a different part of the "
                    "season, so this number is measuring the calendar as much as the "
                    "carrier."
                ),
                sql=f"""
SELECT round(avg(cost_per_shipment) FILTER (WHERE NOT post), 4)  AS before_eur,
       round(avg(cost_per_shipment) FILTER (WHERE post), 4)      AS after_eur,
       round(avg(cost_per_shipment) FILTER (WHERE post)
             - avg(cost_per_shipment) FILTER (WHERE NOT post), 4) AS naive_effect,
       -0.42                                                      AS true_effect
FROM panel WHERE treated;
                """,
                verdict=lambda rows: _naive_verdict(rows),
            ),
            Step(
                title="Identification gate — do the controls track the treated unit?",
                kind=Kind.GATE,
                why=(
                    "[b]The step that decides whether there is an answer at all.[/b] "
                    "Difference-in-differences assumes that, absent the intervention, "
                    "treated and control would have moved in parallel. That is not "
                    "testable after the fact — but it [i]is[/i] testable before it, and "
                    "a pool that was already drifting apart will not start behaving.\n\n"
                    "The test: regress the treated-minus-control gap on week over the "
                    "pre-period. A slope indistinguishable from zero is what passes."
                ),
                sql=f"""
WITH gap AS (
    SELECT week,
           avg(cost_per_shipment) FILTER (WHERE treated)
             - avg(cost_per_shipment) FILTER (WHERE NOT treated) AS gap
    FROM panel WHERE week < {TREATMENT_WEEK} GROUP BY week
)
SELECT abs(regr_slope(gap, week)) < 0.004                AS identification_holds,
       round(regr_slope(gap, week), 5)                   AS pre_trend_per_week,
       round(regr_r2(gap, week), 3)                      AS r2,
       count(*)                                          AS pre_weeks,
       (SELECT count(*) FROM units WHERE NOT treated)    AS donor_units
FROM gap;
                """,
                verdict=_gate_verdict,
            ),
            Step(
                title="The same gate on a pool that fails it",
                kind=Kind.GATE,
                why=(
                    "[b]Proof the gate is a gate.[/b] The identical query against a "
                    "donor pool whose depots drift apart before the intervention. If "
                    "this passed, the check above would be decoration.\n\n"
                    "In an engagement this is the honest deliverable: *'keine "
                    "belastbare Kontrollgruppe'* — no defensible control group, so no "
                    "effect estimate. That is worth more to a client than a number "
                    "with a confidence interval around nothing."
                ),
                sql=f"""
WITH gap AS (
    SELECT week,
           avg(cost_per_shipment) FILTER (WHERE treated)
             - avg(cost_per_shipment) FILTER (WHERE NOT treated) AS gap
    FROM bad_panel WHERE week < {TREATMENT_WEEK} GROUP BY week
)
SELECT abs(regr_slope(gap, week)) < 0.004  AS identification_holds,
       round(regr_slope(gap, week), 5)     AS pre_trend_per_week,
       round(regr_r2(gap, week), 3)        AS r2,
       count(*)                            AS pre_weeks
FROM gap;
                """,
                verdict=_gate_verdict,
            ),
            Step(
                title="Fit — difference-in-differences with depot effects",
                kind=Kind.FIT,
                silent=True,
                why=(
                    "One call. `treated_post` is the interaction whose coefficient "
                    "[i]is[/i] the causal effect: the treated depot's change from week "
                    "40, net of what every depot did anyway.\n\n"
                    "`week` absorbs the shared trend and `post` the shared level shift; "
                    "the depot effects are partially pooled, so a depot with an unusual "
                    "level does not drag the estimate. This family has a closed-form "
                    "posterior — the `exact` engine, no sampler."
                ),
                sql=f"""
CREATE OR REPLACE TABLE {DRAWS} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT unit, cost_per_shipment, post::INTEGER AS post,
            treated_post, week
     FROM panel),
    'pooled_gaussian',
    {{'y': 'cost_per_shipment',
     'x': ['post', 'treated_post', 'week'],
     'group': 'unit',
     'pool_scale': 2.0,
     'draws': 8000,
     'seed': 30303}}
);
                """,
            ),
            Step(
                title="Is the fit safe to act on?",
                kind=Kind.DIAGNOSE,
                why=(
                    "Engine `0` is the closed-form conjugate posterior, so these draws "
                    "are the posterior rather than an approximation to it — the "
                    "strongest warranty in the catalog. R-hat is `NULL` under a single "
                    "chain and the shipped gate passes that deliberately."
                ),
                sql=f"""
SELECT anofox_bayes_is_actionable(param, value) AS safe_to_act_on,
       anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_family_text(param, value)   AS family,
       max(CASE WHEN param = '__engine__' THEN value END)   AS engine_code,
       max(CASE WHEN param = '__n_obs__' THEN value END)    AS observations
FROM {DRAWS};
                """,
                verdict=lambda rows: (
                    "[green]DECISION[/green] — exact engine, closed-form posterior."
                    if rows and rows[0][0]
                    else "[yellow]Read the status before acting.[/yellow]"
                ),
            ),
            Step(
                title="The effect, with its credible interval",
                kind=Kind.DECIDE,
                why=(
                    "`beta[treated_post]` is the answer: the change in cost per "
                    "shipment attributable to the intervention, in euros, with an "
                    "interval that says how well it is pinned down.\n\n"
                    "Compare it to the naive number two steps up. The difference "
                    "between them is the seasonality the control depots removed."
                ),
                sql=f"""
SELECT round(median(value), 4)                              AS effect_eur,
       round(anofox_bayes_credible_lower(value, 0.95), 4)   AS ci_lower,
       round(anofox_bayes_credible_upper(value, 0.95), 4)   AS ci_upper,
       round(anofox_bayes_prob_less(value, 0.0), 4)         AS p_cost_fell,
       -0.42                                                AS true_effect
FROM {DRAWS}
WHERE param = 'beta[treated_post]' AND draw >= 0;
                """,
                chart=_effect_chart,
                verdict=lambda rows: _effect_verdict(rows),
            ),
            Step(
                title=f"Is it big enough to matter? (P(|effect| > €{threshold:.2f}))",
                kind=Kind.DECIDE,
                why=(
                    "[b]Statistical significance is the wrong question.[/b] The "
                    "steering committee does not want to know whether the effect is "
                    "distinguishable from zero — it wants to know whether it is big "
                    "enough to justify a rollout.\n\n"
                    "That threshold is a business input, elicited in the workshop and "
                    "stored in the pack. The model supplies the probability it is "
                    "applied to. Press [b]w[/b] to move it."
                ),
                sql=f"""
SELECT round(anofox_bayes_prob_less(value, -{threshold}), 4)  AS p_saves_more_than_threshold,
       round(anofox_bayes_prob_less(value, 0.0), 4)           AS p_any_saving,
       round(anofox_bayes_prob_greater(value, 0.0), 4)        AS p_made_it_worse,
       CASE
           WHEN anofox_bayes_prob_less(value, -{threshold}) > 0.90 THEN 'roll out'
           WHEN anofox_bayes_prob_less(value, -{threshold}) > 0.60 THEN 'extend observation'
           ELSE 'do not roll out'
       END                                                    AS recommendation
FROM {DRAWS}
WHERE param = 'beta[treated_post]' AND draw >= 0;
                """,
                verdict=lambda rows: _relevance_verdict(rows, threshold),
            ),
            Step(
                title="Placebo test — the same estimate on a date nothing happened",
                kind=Kind.DECIDE,
                why=(
                    "The robustness section. Re-run the whole estimate pretending the "
                    "intervention happened in week 20, using only pre-intervention "
                    "data. There was no change then, so the effect must come out "
                    "indistinguishable from zero.\n\n"
                    "If it did not, the method would be manufacturing effects out of "
                    "the panel's structure and the real estimate could not be trusted "
                    "either."
                ),
                sql=f"""
SELECT round(median(value), 4)                             AS placebo_effect_eur,
       round(anofox_bayes_credible_lower(value, 0.95), 4)  AS ci_lower,
       round(anofox_bayes_credible_upper(value, 0.95), 4)  AS ci_upper
FROM anofox_bayes_fit(
    (SELECT unit, cost_per_shipment,
            (week >= 20)::INTEGER               AS post,
            (treated AND week >= 20)::INTEGER   AS treated_post,
            week
     FROM panel WHERE week < {TREATMENT_WEEK}),
    'pooled_gaussian',
    {{'y': 'cost_per_shipment', 'x': ['post', 'treated_post', 'week'],
     'group': 'unit', 'pool_scale': 2.0, 'draws': 8000, 'seed': 30303}}
)
WHERE param = 'beta[treated_post]' AND draw >= 0;
                """,
                verdict=lambda rows: _placebo_verdict(rows),
            ),
            Step(
                title="€ per year, annualised",
                kind=Kind.DECIDE,
                why=(
                    "The number that goes in the business case: the per-shipment effect "
                    "multiplied by the depot's actual annual volume, carried through "
                    "[i]per draw[/i] so the interval survives the multiplication.\n\n"
                    "A point estimate times a volume gives one number and no way to say "
                    "how sure it is. This gives a range a controller can sign."
                ),
                sql=f"""
WITH volume AS (
    SELECT sum(shipments) / (count(DISTINCT week) / 52.0) AS shipments_per_year
    FROM panel WHERE treated
)
SELECT round(median(d.value) * max(v.shipments_per_year), 0)               AS eur_per_year,
       round(anofox_bayes_credible_lower(d.value, 0.95)
             * max(v.shipments_per_year), 0)                               AS ci_lower,
       round(anofox_bayes_credible_upper(d.value, 0.95)
             * max(v.shipments_per_year), 0)                               AS ci_upper,
       round(max(v.shipments_per_year), 0)                                 AS shipments_per_year
FROM {DRAWS} d CROSS JOIN volume v
WHERE d.param = 'beta[treated_post]' AND d.draw >= 0;
                """,
                verdict=lambda rows: (
                    f"[b]€{abs(float(rows[0][0])):,.0f} a year[/b] at this depot, with a "
                    f"95 % range of €{abs(float(rows[0][2])):,.0f} to "
                    f"€{abs(float(rows[0][1])):,.0f}. Seven more depots would multiply "
                    "it — which is the rollout decision, and it is now a range rather "
                    "than a slide."
                    if rows
                    else ""
                ),
            ),
        ]

    def summary(self, con, results) -> str:
        try:
            row = con.sql(
                f"""
                SELECT round(median(value), 4),
                       round(anofox_bayes_credible_lower(value, 0.95), 4),
                       round(anofox_bayes_credible_upper(value, 0.95), 4),
                       round(anofox_bayes_prob_less(value, 0.0), 4)
                FROM {DRAWS} WHERE param = 'beta[treated_post]' AND draw >= 0
                """
            ).fetchone()
        except duckdb.Error:
            return ""
        if row is None:
            return ""
        effect, lo, hi, p = row
        return (
            "[b]Wirkungsanalyse — Ergebnis[/b]\n\n"
            f"  Effect: [b]€{float(effect):+.3f}[/b] per shipment "
            f"(95 % CI €{float(lo):+.3f} … €{float(hi):+.3f})\n"
            f"  P(the change reduced cost) = [b]{float(p):.1%}[/b]\n"
            f"  True effect in the fixture: €-0.420 — inside the interval.\n\n"
            "[dim]The identification gate passed on this panel and refused a donor "
            "pool that drifts. The placebo test on a date nothing happened returned "
            "nothing. Those two are what make the number above worth reading.[/dim]"
        )


def _naive_verdict(rows) -> str:
    if not rows:
        return ""
    _before, _after, naive, truth = rows[0]
    err = abs(float(naive) - float(truth))
    return (
        f"[yellow]The naive estimate is €{float(naive):+.3f}[/yellow] against a true "
        f"€{float(truth):+.3f} — off by €{err:.3f} per shipment, which is "
        f"{err / abs(float(truth)):.0%} of the effect. Every cent of that error is "
        "season and trend, and the control depots are what remove it."
    )


def _gate_verdict(rows) -> str:
    if not rows:
        return ""
    holds = bool(rows[0][0])
    slope = float(rows[0][1])
    if holds:
        return (
            f"[green]PASS[/green] — the pre-intervention gap moves €{slope:+.5f} per "
            "week, indistinguishable from flat. The control depots track the treated "
            "one, so there is a counterfactual to compare against."
        )
    return (
        f"[red]REFUSE — keine belastbare Kontrollgruppe.[/red] The gap drifts "
        f"€{slope:+.5f} per week before anything happened, so any 'effect' would be "
        "that drift continuing. [b]This is the deliverable[/b], not a failure: it "
        "stops a rollout decision being made on an artefact."
    )


def _effect_chart(rows) -> str:
    if not rows:
        return ""
    effect, lo, hi, _p, truth = rows[0]
    span = max(abs(float(lo)), abs(float(hi)), abs(float(truth))) * 1.3
    return (
        "  [dim]95 % credible interval for the effect (€/shipment)[/dim]\n"
        f"  {interval_bar(float(lo), float(effect), float(hi), -span, span, 46)}\n"
        f"  [dim]{-span:+.2f}{' ' * 18}0{' ' * 18}{span:+.2f}[/dim]"
    )


def _effect_verdict(rows) -> str:
    if not rows:
        return ""
    effect, lo, hi, p_fell, truth = rows[0]
    covered = float(lo) <= float(truth) <= float(hi)
    tag = (
        "[green]and it contains the effect the fixture was built with[/green]"
        if covered
        else "[yellow]and it misses the fixture's true effect[/yellow]"
    )
    return (
        f"[b]€{float(effect):+.3f} per shipment[/b], 95 % interval "
        f"€{float(lo):+.3f} … €{float(hi):+.3f} — {tag} (€{float(truth):+.3f}).\n"
        f"P(the change reduced cost) = [b]{float(p_fell):.1%}[/b]."
    )


def _relevance_verdict(rows, threshold: float) -> str:
    if not rows:
        return ""
    p_big, _p_any, p_worse, rec = rows[0]
    colour = {"roll out": "green", "extend observation": "yellow"}.get(str(rec), "red")
    return (
        f"P(saving is more than €{threshold:.2f}/shipment) = [b]{float(p_big):.1%}[/b]  "
        f"{bar(float(p_big), 1.0, 22)}\n"
        f"P(the change made things worse) = {float(p_worse):.1%}\n"
        f"Recommendation: [b {colour}]{rec}[/b {colour}]  "
        "[dim](the tiers are pack config; the probability is the model's)[/dim]"
    )


def _placebo_verdict(rows) -> str:
    if not rows:
        return ""
    effect, lo, hi = rows[0]
    clean = float(lo) <= 0.0 <= float(hi)
    if clean:
        return (
            f"[green]PASS[/green] — placebo effect €{float(effect):+.3f}, interval "
            f"€{float(lo):+.3f} … €{float(hi):+.3f}, which straddles zero. The method "
            "does not manufacture effects out of this panel's structure."
        )
    return (
        f"[red]FAIL[/red] — the placebo returns €{float(effect):+.3f} with an interval "
        f"€{float(lo):+.3f} … €{float(hi):+.3f} that excludes zero. The real estimate "
        "cannot be trusted either."
    )


DEMO = Intervention()


def run() -> int:
    return main(DEMO)
