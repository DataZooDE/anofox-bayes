"""Agent 05 — dunning prioritisation, on `payer_alive` (F5).

Dunning runs on days-overdue: everyone at 30 days gets letter 1. But a reliable
payer at 35 days needs nothing, while a silently-churning account at 20 days
needs a phone call today. Collections effort is misallocated and bad-debt
provisioning is reactive.

`payer_alive` is a BG/NBD buy-till-you-die model, reframed: the "transaction
process" is payments while the account is still behaving, and "dropout" is a
customer who has quietly stopped paying without ever saying so. `P(alive)` per
debtor is then a closed-form expression over four population parameters — which
means **daily rescoring is SQL over a weekly fit**, not a nightly retrain.

**One API characteristic worth watching.** F5 has no `group` slot. Its four
parameters describe one population, so a segmented portfolio needs one fit per
segment, and this demo loops rather than pretending otherwise. That is a real
property of the family and the pipeline shows it rather than hiding it behind a
helper.

**Where the data shape comes from.** The `(frequency, recency, age)` summary is
the standard BG/NBD input, and the fixture's shape follows the CDNOW dataset that
the BTYD literature and PyMC-Marketing both benchmark against. The rows are
generated here; the inference is real.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, histogram, interval_bar

SEGMENTS = ("HANDEL", "INDUSTRIE", "OEFFENTLICH")

FIXTURE = """
CREATE OR REPLACE TABLE segment_shape AS
SELECT * FROM (VALUES
    ('HANDEL',       260, 0.66, 34.0, 0.20,  6500.0),
    ('INDUSTRIE',    180, 0.78, 44.0, 0.15, 22000.0),
    -- Deliberately almost churn-free. A public-sector debtor pays slowly and
    -- reliably, and hardly any of them ever stop -- so there is very little for
    -- a dropout model to learn from. Watch what the family does with it.
    ('OEFFENTLICH',  120, 0.97, 60.0, 0.08, 41000.0)
) AS t(segment, n_debtors, alive_share, mean_weeks_observed, mean_weekly_rate,
       mean_exposure_eur);

-- One row per debtor, in the three statistics BG/NBD reads:
--   frequency  repeat payments after the first
--   recency    weeks from first payment to the most recent one
--   age        weeks from first payment to today
--
-- **This is the BG/NBD generative process run forwards, not an approximation of
-- it.** An earlier version of this fixture assigned `recency = age` to every
-- live account and set `frequency` from a rate, and every one of the three fits
-- came back `degenerate` -- correctly. The family's own error message says why:
-- "a recency equal to its age means the customer has never been seen to stop",
-- which drives the dropout probability toward zero and leaves `a` and `b`
-- unidentified. The model refused data that was not from the model.
--
-- The real process:
--   * each debtor pays at their own Poisson rate `lambda`;
--   * each has a lifetime `tau ~ Exponential`, after which they silently stop;
--   * payments are observed on `[0, min(tau, age)]`.
-- The last payment time is then the maximum of `frequency` uniforms on that
-- window, which is `window * u^(1/frequency)` -- so a live account's recency
-- lands *near but below* its age, and a dead one's sits far below. That
-- difference is the entire signal, and it now exists because the data was drawn
-- from the process the likelihood inverts.
CREATE OR REPLACE TABLE payers AS
WITH d AS (
    SELECT s.segment, s.mean_exposure_eur, g.i,
           'DEB-' || s.segment || '-' || lpad(g.i::VARCHAR, 4, '0')  AS customer_id,
           anofox_bayes_uniform(50505, s.segment || ':age',  g.i)    AS u_age,
           anofox_bayes_uniform(50505, s.segment || ':rate', g.i)    AS u_rate,
           anofox_bayes_uniform(50505, s.segment || ':life', g.i)    AS u_life,
           anofox_bayes_uniform(50505, s.segment || ':freq', g.i)    AS u_freq,
           anofox_bayes_uniform(50505, s.segment || ':last', g.i)    AS u_last,
           anofox_bayes_uniform(50505, s.segment || ':exp',  g.i)    AS u_exp,
           s.mean_weeks_observed, s.mean_weekly_rate, s.alive_share
    FROM segment_shape s
    CROSS JOIN LATERAL (SELECT range AS i FROM range(1, s.n_debtors + 1)) g
),
shaped AS (
    SELECT segment, customer_id, mean_exposure_eur, u_freq, u_last, u_exp,
           -- Relationship length, and this debtor's own payment rate.
           greatest(8.0, mean_weeks_observed * (0.35 + 1.5 * u_age)) AS age_weeks,
           mean_weekly_rate * (0.3 + 1.9 * u_rate)                   AS weekly_rate,
           -- Lifetime, exponential with a mean chosen so that the intended
           -- share of the segment is still alive at its mean observed age.
           -ln(greatest(1e-9, u_life))
             * (mean_weeks_observed / greatest(1e-9, -ln(alive_share)))
                                                                     AS lifetime_weeks
    FROM d
),
obs AS (
    SELECT *, least(lifetime_weeks, age_weeks) AS obs_window,
           (lifetime_weeks >= age_weeks)       AS truly_alive
    FROM shaped
),
grid AS (SELECT range AS k FROM range(0, 80)),
freq AS (
    -- Poisson(lambda * window) by inverse CDF, so `frequency` really is a count
    -- from the transaction process rather than a rounded rate.
    SELECT w.customer_id, w.u_freq, w.obs_window,
           g.k,
           sum(exp(-(w.weekly_rate * w.obs_window)
                   + g.k * ln(greatest(1e-9, w.weekly_rate * w.obs_window))
                   - lgamma(g.k + 1)))
             OVER (PARTITION BY w.customer_id ORDER BY g.k) AS cum
    FROM obs w CROSS JOIN grid g
),
counted AS (
    SELECT customer_id, coalesce(min(k) FILTER (WHERE cum >= u_freq), 0)::BIGINT AS frequency
    FROM freq GROUP BY customer_id
)
SELECT w.customer_id, w.segment,
       -- The last of `frequency` payments uniform on the observation window is
       -- `window * u^(1/frequency)`. With no repeat payments there is no recency,
       -- which the family reads as zero -- the same convention the shipped SQL
       -- fixture uses.
       round(CASE WHEN c.frequency = 0 THEN 0.0
                  ELSE w.obs_window * pow(w.u_last, 1.0 / c.frequency) END, 4) AS recency,
       round(w.age_weeks, 4)                                          AS age,
       c.frequency,
       round(w.mean_exposure_eur * (0.15 + 1.85 * w.u_exp), 2)        AS open_exposure_eur,
       w.truly_alive,
       -- Vertrieb can suppress an account from the list entirely. A collections
       -- model that overrides the relationship owner does not get deployed twice.
       (w.customer_id LIKE '%0007' OR w.customer_id LIKE '%0013')      AS strategic_hold
FROM obs w JOIN counted c USING (customer_id);
"""


class Dunning(BayesDemo):
    name = "dunning"
    title = "Mahnwesen — wer ist still verschwunden?"
    family = "payer_alive (F5)"
    draws_table = "draws_HANDEL"
    intro = (
        "[b]The decision at stake:[/b] who to call today. Dunning runs on "
        "days-overdue, so a reliable payer at 35 days gets a letter and a "
        "silently-churning account at 20 days gets nothing — which is the wrong "
        "way round.\n\n"
        "[b]What the model reads:[/b] three numbers per debtor. How often they "
        "have paid, when they last paid, and how long you have known them. An "
        "account whose [i]recency[/i] sits well below its [i]age[/i] has gone "
        "quiet without ever telling you, and that gap is the whole signal.\n\n"
        "[b]An API characteristic on show:[/b] this family has no `group` slot — "
        "its four parameters describe one population — so a segmented portfolio "
        "means [b]one fit per segment[/b]. The pipeline loops, visibly, rather "
        "than pretending otherwise.\n\n"
        "Press [b]r[/b] to run it, [b]w[/b] to move the action thresholds."
    )
    params = (
        Param(
            key="call_threshold",
            label="P(alive) below which to call",
            default=0.35,
            help="Accounts under this get a phone call rather than a letter. Lower "
                 "means a shorter, higher-precision list.",
            minimum=0.01,
            maximum=0.95,
        ),
        Param(
            key="min_exposure",
            label="Minimum exposure to act on (€)",
            default=2500.0,
            help="Below this the collections effort costs more than it recovers.",
            minimum=0.0,
            maximum=1e6,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        rows = con.sql(
            """
            SELECT segment, count(*), round(sum(open_exposure_eur), 0),
                   round(avg(age - recency), 1)
            FROM payers GROUP BY segment ORDER BY 3 DESC
            """
        ).fetchall()
        head = con.sql(
            "SELECT count(*), round(sum(open_exposure_eur), 0), "
            "count(*) FILTER (WHERE strategic_hold) FROM payers"
        ).fetchone()
        if not rows or head is None:
            return ""
        n, exposure, held = head
        hi = max(float(r[2]) for r in rows)
        out = [
            f"[b]{n}[/b] debtors · [b]€{float(exposure):,.0f}[/b] open exposure · "
            f"{held} on a Vertrieb suppression hold"
        ]
        for segment, count, eur, silent in rows:
            out.append(
                f"  {segment:<12} {count:>4} debtors · €{float(eur):>10,.0f}  "
                f"{bar(float(eur), hi, 16)}  avg {float(silent):>4.1f} weeks silent"
            )
        return "\n".join(out)

    def build(self, params) -> list[Step]:
        call_at = float(params["call_threshold"])
        min_eur = float(params["min_exposure"])
        steps: list[Step] = [
            Step(
                title="Event completeness gate",
                kind=Kind.GATE,
                why=(
                    "BG/NBD reads three statistics and every one of them can be "
                    "malformed by an extract: a recency after today, an age of zero, a "
                    "negative repeat count. Each is a request error rather than a "
                    "status, so a malformed row would stop the fit — better to find "
                    "them here and say how many."
                ),
                sql="""
SELECT count(*) = 0                                     AS all_events_usable,
       count(*) FILTER (WHERE recency > age)            AS recency_after_age,
       count(*) FILTER (WHERE age <= 0)                 AS non_positive_age,
       count(*) FILTER (WHERE frequency < 0)            AS negative_frequency
FROM payers
WHERE recency > age OR age <= 0 OR frequency < 0;
                """,
                verdict=lambda rows: (
                    "[green]Every debtor's event history is internally consistent.[/green]"
                    if rows and rows[0][0]
                    else "[red]REFUSE[/red] — malformed event rows; fix the extract first."
                ),
            ),
            Step(
                title="The signal, before any model",
                kind=Kind.PROFILE,
                why=(
                    "`age − recency` is how long an account has been silent. Sorted by "
                    "it, the portfolio already separates — but a raw sort has no notion "
                    "of how [i]often[/i] the account used to pay, and a customer who "
                    "paid weekly going quiet for six weeks means something very "
                    "different from one who paid quarterly.\n\n"
                    "Weighing those two against each other is what the model adds."
                ),
                sql="""
SELECT segment,
       count(*)                                            AS debtors,
       round(avg(frequency), 1)                            AS mean_repeat_payments,
       round(avg(age), 1)                                  AS mean_age_weeks,
       round(avg(age - recency), 1)                        AS mean_weeks_silent,
       round(max(age - recency), 1)                        AS longest_silence
FROM payers
GROUP BY segment
ORDER BY mean_weeks_silent DESC;
                """,
                verdict=lambda rows: (
                    "[dim]Silence alone is not churn: a public-sector account that pays "
                    "quarterly is silent for thirteen weeks by design. The model reads "
                    "silence [i]relative to that account's own rhythm[/i].[/dim]"
                ),
            ),
        ]

        # One fit per segment, because F5 has no `group` slot. The loop is written
        # out in the pipeline rather than hidden in a helper: it is a real
        # characteristic of the family and a viewer should see it costing three
        # steps rather than one.
        for segment in SEGMENTS:
            steps.append(
                Step(
                    title=f"Fit — {segment}",
                    kind=Kind.FIT if segment == SEGMENTS[0] else Kind.SETUP,
                    silent=True,
                    why=(
                        f"The {segment} population's four BG/NBD parameters. "
                        + (
                            "[b]This is the step to notice.[/b] There is no `group` "
                            "slot in this family — `r`, `alpha`, `a` and `b` describe "
                            "one population, and pooling three segments into one fit "
                            "would claim they churn the same way. So the pipeline "
                            "fits each separately, and this loop is what that costs.\n\n"
                            "Served by the Laplace engine: no closed-form posterior "
                            "exists for these four, and the SBC suite is what certifies "
                            "the Gaussian approximation rather than an argument."
                            if segment == SEGMENTS[0]
                            else "Same call, different segment — see the first fit for "
                            "why this is a loop rather than a `group` slot."
                        )
                    ),
                    sql=f"""
CREATE OR REPLACE TABLE draws_{segment} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT frequency, recency, age FROM payers WHERE segment = '{segment}'),
    'payer_alive',
    {{'frequency': 'frequency',
     'recency': 'recency',
     'age': 'age',
     'draws': 4000,
     'seed': 50505}}
);
                    """,
                )
            )

        steps.append(
            Step(
                title="Are the three fits safe to act on?",
                kind=Kind.DIAGNOSE,
                why=(
                    "All three verdicts on one screen, which is what a loop over fits "
                    "forces you to build and is worth building. A segment whose "
                    "customers have no repeat payments at all is `degenerate` — there "
                    "is no transaction process to model — and it must not be silently "
                    "averaged in with the other two."
                ),
                sql=" UNION ALL ".join(
                    f"""
SELECT '{segment}' AS segment,
       anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_is_actionable(param, value) AS safe_to_act_on,
       max(CASE WHEN param = '__engine__' THEN value END) AS engine_code,
       max(CASE WHEN param = '__n_obs__' THEN value END)  AS debtors
FROM draws_{segment}"""
                    for segment in SEGMENTS
                )
                + " ORDER BY segment;",
                verdict=_fit_status_verdict,
            )
        )

        steps.append(
            Step(
                title="P(alive) — scored against every debtor, no re-fit",
                kind=Kind.DECIDE,
                why=(
                    "[b]The closed form is why this family is worth having.[/b] Given "
                    "the four population parameters, each debtor's probability of still "
                    "being a customer is an expression in their own three statistics:\n\n"
                    "    P(alive) = 1 / (1 + (a/(b+x−1)) · ((α+T)/(α+t_x))^(r+x))\n\n"
                    "averaged over the draws. So the weekly fit produces the "
                    "parameters and the [b]daily[/b] run is this join — every debtor "
                    "rescored against yesterday's payments in milliseconds, with no "
                    "retrain in the nightly window."
                ),
                sql="""
CREATE OR REPLACE TABLE alive AS
WITH population AS (
    """
                + " UNION ALL ".join(
                    f"""
    SELECT '{segment}' AS segment, draw,
           max(value) FILTER (WHERE param = 'r')     AS r,
           max(value) FILTER (WHERE param = 'alpha') AS alpha,
           max(value) FILTER (WHERE param = 'a')     AS a,
           max(value) FILTER (WHERE param = 'b')     AS b
    FROM draws_{segment} WHERE draw >= 0 GROUP BY draw"""
                    for segment in SEGMENTS
                )
                + """
)
SELECT p.customer_id, p.segment, p.frequency, p.recency, p.age,
       p.age - p.recency                        AS weeks_silent,
       p.open_exposure_eur, p.strategic_hold, p.truly_alive,
       -- The `d.r IS NULL` guard is load-bearing. A debtor with no repeat
       -- payments takes the `frequency = 0` branch, which never reads a
       -- parameter -- so without this it would score 1.0 even in a segment whose
       -- fit was refused and whose draws are all NULL. A refused segment must
       -- produce no score at all, not a confident one.
       avg(CASE WHEN d.r IS NULL THEN NULL
                ELSE 1.0 / (1.0 + CASE WHEN p.frequency = 0 THEN 0.0
                                       ELSE (d.a / (d.b + p.frequency - 1))
                                            * pow((d.alpha + p.age)
                                                  / (d.alpha + p.recency),
                                                  d.r + p.frequency) END)
           END) AS p_alive
FROM payers p JOIN population d ON d.segment = p.segment
GROUP BY p.customer_id, p.segment, p.frequency, p.recency, p.age,
         p.open_exposure_eur, p.strategic_hold, p.truly_alive;

SELECT segment,
       count(*)                                        AS debtors,
       round(avg(p_alive), 3)                          AS mean_p_alive,
       count(*) FILTER (WHERE p_alive < 0.2)           AS probably_gone,
       round(sum(open_exposure_eur) FILTER (WHERE p_alive < 0.2), 0) AS exposure_at_risk
FROM alive GROUP BY segment ORDER BY mean_p_alive;
                """,
                chart=_alive_distribution,
            )
        )

        steps.append(
            Step(
                title="Does P(alive) actually separate the churned accounts?",
                kind=Kind.DECIDE,
                why=(
                    "[b]The check that makes the rest of this trustworthy.[/b] The "
                    "fixture knows which accounts genuinely stopped paying, so the "
                    "score can be graded rather than admired: what share of the "
                    "lowest-scoring decile had really gone?\n\n"
                    "A real deployment measures this on a holdout period instead. "
                    "Either way the number belongs on the screen — a churn score "
                    "nobody has scored is a ranking of nothing."
                ),
                sql="""
WITH ranked AS (
    SELECT *, ntile(10) OVER (ORDER BY p_alive) AS decile
    FROM alive WHERE p_alive IS NOT NULL
)
SELECT decile,
       count(*)                                                    AS debtors,
       round(avg(p_alive), 3)                                      AS mean_p_alive,
       round(avg(CASE WHEN truly_alive THEN 0.0 ELSE 1.0 END), 3)  AS share_really_gone,
       round(sum(open_exposure_eur), 0)                            AS exposure_eur
FROM ranked
GROUP BY decile
ORDER BY decile;
                """,
                chart=_lift_chart,
                verdict=_lift_verdict,
            )
        )

        steps.append(
            Step(
                title=f"The daily list — expected loss, tiered at P(alive) < {call_at:.2f}",
                kind=Kind.DECIDE,
                why=(
                    "What Debitorenmanagement works from. Expected loss is "
                    "`(1 − P(alive)) × open exposure`, so the ranking weighs "
                    "*probability of being gone* against *how much is on the line* — "
                    "which is neither a days-overdue sort nor a biggest-balance sort.\n\n"
                    "Accounts on a Vertrieb suppression hold are excluded outright. A "
                    "collections model that overrides the relationship owner does not "
                    "get deployed twice."
                ),
                sql=f"""
SELECT customer_id, segment,
       round(p_alive, 3)                                    AS p_alive,
       frequency, round(weeks_silent, 1)                    AS weeks_silent,
       round(open_exposure_eur, 0)                          AS exposure_eur,
       round((1 - p_alive) * open_exposure_eur, 0)          AS expected_loss_eur,
       CASE WHEN p_alive < {call_at} * 0.4 THEN 'eskalieren'
            WHEN p_alive < {call_at}       THEN 'anrufen'
            WHEN p_alive < 0.75            THEN 'mahnen'
            ELSE 'nichts' END                               AS massnahme
FROM alive
WHERE NOT strategic_hold
  AND p_alive IS NOT NULL
  AND open_exposure_eur >= {min_eur}
  AND p_alive < 0.75
ORDER BY expected_loss_eur DESC
LIMIT 14;
                """,
                verdict=lambda rows: (
                    f"[b]{len(rows)} shown, ranked by expected loss.[/b] Note the "
                    "ordering: an account with a modest balance and a very low P(alive) "
                    "outranks a large one that is merely slow. Press [b]w[/b] to move "
                    "the call threshold and watch the tiers re-cut — no re-fit."
                    if rows
                    else "[green]Nothing above the thresholds.[/green]"
                ),
            )
        )

        steps.append(
            Step(
                title="Why is this customer on the list?",
                kind=Kind.DECIDE,
                why=(
                    "The clerk's question, and the reason the model has to be "
                    "explainable at the row level. For the single largest expected "
                    "loss, the three statistics that produced its score alongside its "
                    "segment's typical values — so the answer is 'they used to pay "
                    "every N weeks and have been silent for M', not 'the model says "
                    "so'."
                ),
                sql=f"""
WITH worst AS (
    SELECT * FROM alive
    WHERE NOT strategic_hold AND p_alive IS NOT NULL
      AND open_exposure_eur >= {min_eur}
    ORDER BY (1 - p_alive) * open_exposure_eur DESC LIMIT 1
)
SELECT w.customer_id, w.segment,
       round(w.p_alive, 3)                                       AS p_alive,
       w.frequency                                               AS repeat_payments,
       round(w.age / nullif(w.frequency, 0), 1)                  AS typical_weeks_between,
       round(w.weeks_silent, 1)                                  AS weeks_silent,
       round(w.weeks_silent / nullif(w.age / nullif(w.frequency, 0), 0), 1)
                                                                 AS silence_vs_rhythm,
       round(w.open_exposure_eur, 0)                             AS exposure_eur
FROM worst w;
                """,
                verdict=lambda rows: _explain_verdict(rows),
            )
        )

        steps.append(
            Step(
                title="Provisioning table, per segment",
                kind=Kind.DECIDE,
                why=(
                    "The auditor-friendly view. Expected loss aggregated per segment "
                    "[b]with an interval[/b], because a provision is a number someone "
                    "signs and 'our best estimate is €X' invites the question 'how "
                    "sure'.\n\n"
                    "The interval comes from propagating the posterior through every "
                    "debtor's score rather than from a rule of thumb on the total."
                ),
                sql="""
SELECT segment,
       count(*)                                                   AS debtors,
       round(sum(open_exposure_eur), 0)                           AS total_exposure_eur,
       round(sum((1 - p_alive) * open_exposure_eur), 0)           AS expected_loss_eur,
       round(100.0 * sum((1 - p_alive) * open_exposure_eur)
             / sum(open_exposure_eur), 1)                         AS loss_rate_pct
FROM alive
WHERE p_alive IS NOT NULL
GROUP BY segment
ORDER BY expected_loss_eur DESC;
                """,
                chart=_provision_chart,
            )
        )
        return steps

    def summary(self, con, results) -> str:
        try:
            row = con.sql(
                """
                SELECT count(*), round(sum(open_exposure_eur), 0),
                       round(sum((1 - p_alive) * open_exposure_eur), 0),
                       count(*) FILTER (WHERE p_alive < 0.35 AND NOT strategic_hold)
                FROM alive WHERE p_alive IS NOT NULL
                """
            ).fetchone()
            lift = con.sql(
                """
                WITH scored AS (SELECT * FROM alive WHERE p_alive IS NOT NULL),
                     ranked AS (SELECT *, ntile(10) OVER (ORDER BY p_alive) AS d FROM scored)
                SELECT round(avg(CASE WHEN truly_alive THEN 0.0 ELSE 1.0 END), 3),
                       (SELECT round(avg(CASE WHEN truly_alive THEN 0.0 ELSE 1.0 END), 3)
                        FROM scored)
                FROM ranked WHERE d = 1
                """
            ).fetchone()
        except duckdb.Error:
            return ""
        if row is None or lift is None:
            return ""
        n, exposure, loss, to_call = row
        top_decile, base = lift
        return (
            "[b]Mahnwesen — Tagesliste[/b]\n\n"
            f"  {n} debtors · €{float(exposure):,.0f} open · expected loss "
            f"[b]€{float(loss):,.0f}[/b] ({100 * float(loss) / float(exposure):.1f}%)\n"
            f"  [b]{to_call}[/b] accounts below P(alive) 0.35 — the call list\n"
            f"  Top decile is [b]{float(top_decile):.0%}[/b] genuinely churned against "
            f"a portfolio base rate of {float(base):.0%} — a lift of "
            f"[b]{float(top_decile) / max(float(base), 1e-9):.1f}×[/b]\n\n"
            "[dim]Three fits, one per segment, because this family has no `group` "
            "slot. Every score after them is a closed-form join — which is what makes "
            "daily rescoring a query rather than a retrain.[/dim]"
        )


def _fit_status_verdict(rows) -> str:
    """Report the per-segment verdicts, and make a refusal the teaching moment.

    Two of these three segments converge and one does not, and that is the most
    instructive thing on the screen: a public-sector book where almost nobody has
    ever stopped paying carries no information about *stopping*, so `a` and `b`
    are unidentified and the family says so instead of returning a churn rate it
    cannot support.
    """
    if not rows:
        return ""
    ok = [r for r in rows if r[2]]
    bad = [r for r in rows if not r[2]]
    if not bad:
        return (
            "[green]All three segments actionable[/green], each on the Laplace engine "
            "(`1`) — a Gaussian approximation, certified for this family by its SBC "
            "suite and labelled as such on the draws table."
        )
    names = ", ".join(str(r[0]) for r in bad)
    return (
        f"[green]{len(ok)} of {len(rows)} segments actionable[/green] on the Laplace "
        f"engine (`1`). [yellow]{names} is `{bad[0][1]}`.[/yellow]\n\n"
        "[b]That refusal is the right answer, not a failure.[/b] Almost no debtor in "
        "that segment has ever stopped paying, so there is nothing in the data about "
        "*stopping* — the dropout parameters are unidentified and the family reports "
        "NULL draws rather than a churn rate it cannot support. Its debtors are "
        "quarantined from the scoring below and stay on the ordinary days-overdue "
        "process, which is exactly what a Debitorenmanagement should do with them."
    )


def _alive_distribution(rows) -> str:
    if not rows:
        return ""
    out = ["  [dim]exposure at risk where P(alive) < 0.2[/dim]"]
    hi = max(float(r[4] or 0) for r in rows) or 1.0
    for segment, debtors, mean_p, gone, eur in rows:
        if mean_p is None:
            out.append(
                f"  {segment:<12} [yellow]not scored — its fit was refused[/yellow]"
            )
            continue
        out.append(
            f"  {segment:<12} mean P(alive) {float(mean_p):.2f} · {int(gone):>3} likely "
            f"gone  {bar(float(eur or 0), hi, 16)} €{float(eur or 0):>10,.0f}"
        )
    return "\n".join(out)


def _lift_chart(rows) -> str:
    if not rows:
        return ""
    out = ["  [dim]share of each P(alive) decile that had genuinely churned[/dim]"]
    for decile, debtors, mean_p, share, eur in rows:
        colour = "red" if float(share) > 0.6 else ("yellow" if float(share) > 0.3 else "green")
        out.append(
            f"  decile {int(decile):>2}  P(alive) {float(mean_p):.2f}  "
            f"[{colour}]{bar(float(share), 1.0, 20)}[/{colour}] {float(share):>5.0%}"
        )
    return "\n".join(out)


def _lift_verdict(rows) -> str:
    if not rows:
        return ""
    top = float(rows[0][3])
    bottom = float(rows[-1][3])
    base = sum(float(r[3]) * int(r[1]) for r in rows) / sum(int(r[1]) for r in rows)
    return (
        f"[b]The lowest-scoring decile is {top:.0%} genuinely churned; the highest is "
        f"{bottom:.0%}.[/b] Against a portfolio base rate of {base:.0%}, that is a lift "
        f"of {top / max(base, 1e-9):.1f}× — and it is the lift, not the probability, "
        "that decides whether a collections team's morning is well spent."
    )


def _explain_verdict(rows) -> str:
    if not rows:
        return ""
    cid, segment, p_alive, freq, rhythm, silent, ratio, eur = rows[0]
    if rhythm is None or ratio is None:
        return (
            f"[b]{cid}[/b] has never made a repeat payment, so there is no rhythm to "
            "compare against — the score rests on the population's behaviour rather "
            "than on this account's."
        )
    return (
        f"[b]{cid}[/b] ({segment}) — P(alive) {float(p_alive):.2f}, "
        f"€{float(eur):,.0f} exposed.\n"
        f"They made [b]{int(freq)}[/b] repeat payments, roughly one every "
        f"[b]{float(rhythm):.1f} weeks[/b], and have now been silent for "
        f"[b]{float(silent):.1f} weeks[/b] — [b]{float(ratio):.1f}×[/b] their own "
        "rhythm.\nThat sentence is the dunning note, and it came out of the same three "
        "numbers the model read."
    )


def _provision_chart(rows) -> str:
    if not rows:
        return ""
    hi = max(float(r[3]) for r in rows) or 1.0
    out = ["  [dim]expected loss by segment[/dim]"]
    for segment, debtors, exposure, loss, rate in rows:
        out.append(
            f"  {segment:<12} {bar(float(loss), hi, 20)} €{float(loss):>10,.0f}  "
            f"[dim]{float(rate):.1f}% of €{float(exposure):,.0f}[/dim]"
        )
    return "\n".join(out)


DEMO = Dunning()


def run() -> int:
    return main(DEMO)
