"""Agent 04 — cash runway, on `payment_delay` (F4).

Mittelstand cash planning is a spreadsheet with due dates taken literally.
Customers pay when they pay. The CFO's question is not "what is the forecast"
but **"what is the probability we cover payroll on the 28th"**, and which
receivables to chase to move that probability.

Two things in this demo are worth watching for:

**The Monte-Carlo is SQL.** Once the delay model is fitted, the forward
simulation is a join: every open invoice crossed with every posterior draw,
sampled to a payment date, aggregated to a daily balance. On this fixture that
is 240 000 invoice-draws collapsing into a 364 000-row cash-path table, in about
130 ms — no Python in the loop, and it scales with the ledger rather than with
the modelling.

**The randomness is keyed, not random.** `anofox_bayes_std_normal(seed, key,
draw)` is a pure function of its three arguments, so the cash path is
reproducible without `setseed()` and identical on every machine. A liquidity
forecast an auditor cannot re-run is not a forecast.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, fan, interval_bar

DRAWS = "delay_draws"
N_DRAWS = 1000
CHAINS = 4

FIXTURE = """
CREATE OR REPLACE TABLE segments AS
SELECT * FROM (VALUES
    ('RETAIL',       22.0, 0.9),
    ('WHOLESALE',    34.0, 1.4),
    ('PUBLIC',       58.0, 2.2),
    ('EXPORT',       41.0, 1.1),
    ('OEM',          27.0, 1.8),
    ('KEY_ACCOUNT',  31.0, 2.6)
) AS t(segment, mean_delay_days, exposure_weight);

-- Cleared items: the history the model learns from. The delay is measured from
-- the **invoice** date, which is the clock `payment_delay` requires -- a delay
-- from the due date goes negative whenever a customer pays early, and fitting
-- only the positive rows would keep exactly the late payers.
--
-- Wilson-Hilferty maps a keyed standard normal onto a Gamma of shape 6, so the
-- history is a genuine right-skewed sample and a pure function of this file.
CREATE OR REPLACE TABLE cleared AS
SELECT s.segment,
       'CL-' || s.segment || '-' || lpad(i::VARCHAR, 3, '0') AS item,
       greatest(0.5, round(
           s.mean_delay_days
           * pow(1.0 - 1.0/54.0
                 + anofox_bayes_std_normal(40404, s.segment, i) / sqrt(54.0), 3), 2
       )) AS delay_days
FROM segments s, generate_series(1, 45) AS g(i);

-- Open items: what is still outstanding today, with an amount and an age.
--
-- **The totals are tuned so the answer is not foregone.** Receivables come to
-- roughly €730k against €809k of obligations and a €150k opening balance, so the
-- horizon is genuinely tight and *timing* decides it. An earlier version of this
-- fixture carried €3.1M of AR, every probability came out at 100.0 %, and a demo
-- whose answer is always "yes" teaches nothing about a model built to say how
-- sure it is.
CREATE OR REPLACE TABLE open_items AS
SELECT s.segment,
       'AR-' || s.segment || '-' || lpad(i::VARCHAR, 3, '0')                AS item,
       round(2000 + 12000 * s.exposure_weight
             * anofox_bayes_uniform(40404, s.segment || ':amt', i), 2)      AS amount_eur,
       -- Days since the invoice was raised. Some are already older than the
       -- segment's habit, which is what makes the conditional question sharp.
       (1 + (i * 7) % 40)::INTEGER                                          AS age_days
FROM segments s, generate_series(1, 10) AS g(i);

-- What has to be paid, and when. Fixed, known, and the thing the probability is
-- computed against.
CREATE OR REPLACE TABLE obligations AS
SELECT * FROM (VALUES
    (12, 'Lohn und Gehalt',      185000.00),
    (20, 'Umsatzsteuer',          64000.00),
    (30, 'Lieferanten (AP)',     142000.00),
    (42, 'Lohn und Gehalt',      185000.00),
    (55, 'Tilgung Darlehen',      48000.00),
    (72, 'Lohn und Gehalt',      185000.00)
) AS t(day, purpose, amount_eur);

CREATE OR REPLACE TABLE opening AS SELECT 150000.00 AS balance_eur;
"""


class CashRunway(BayesDemo):
    name = "cash-runway"
    title = "Cash-Runway-Agent — P(we are covered on the 28th)"
    family = "payment_delay (F4)"
    draws_table = DRAWS
    intro = (
        "[b]The decision at stake:[/b] not a cash forecast — a [b]probability[/b]. "
        "'Do we cover payroll on day 42' is a yes/no question with a number "
        "attached, and the number is what decides whether to draw on the credit "
        "line this week or chase three invoices instead.\n\n"
        "[b]Why a Gamma:[/b] a payment delay is positive and right-skewed. Most "
        "invoices land near the segment's habit and a few land very late — and it "
        "is the few that decide whether payroll clears. A lognormal fits the same "
        "shape and disagrees about the far tail; this demo fits [b]both[/b] and "
        "shows you how far apart they are, because that gap is days of working "
        "capital.\n\n"
        "[b]Watch the simulation step:[/b] the forward Monte-Carlo is a SQL join "
        "over the posterior draws, and the noise is keyed rather than random — so "
        "an auditor can re-run it and get the same answer.\n\n"
        "Press [b]r[/b] to run it, [b]w[/b] to test a different date or threshold."
    )
    params = (
        Param(
            key="cover_day",
            label="Day to test cover on",
            default=72.0,
            help="Days from today. Day 72 is the third payroll — the first date the "
                 "opening balance and the receivables do not obviously cover, and "
                 "therefore the only one where the probability is worth computing.",
            minimum=1.0,
            maximum=90.0,
        ),
        Param(
            key="min_balance",
            label="Minimum acceptable balance (€)",
            default=0.0,
            help="The covenant floor, or zero for plain solvency. The probability "
                 "reported is P(balance stays above this).",
            minimum=-1e9,
            maximum=1e9,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        head = con.sql(
            """
            SELECT (SELECT count(*) FROM cleared),
                   (SELECT count(*) FROM open_items),
                   (SELECT round(sum(amount_eur), 2) FROM open_items),
                   (SELECT round(sum(amount_eur), 2) FROM obligations),
                   (SELECT balance_eur FROM opening)
            """
        ).fetchone()
        rows = con.sql(
            """
            SELECT c.segment, round(avg(c.delay_days), 1) AS observed_mean,
                   round(sum(o.amount_eur), 2) AS open_eur
            FROM cleared c JOIN open_items o USING (segment)
            GROUP BY c.segment ORDER BY observed_mean
            """
        ).fetchall()
        if head is None or not rows:
            return ""
        n_cleared, n_open, open_eur, oblig_eur, opening = head
        hi = max(float(r[2]) for r in rows)
        out = [
            f"[b]{n_cleared}[/b] cleared items (the history) · [b]{n_open}[/b] open "
            f"= [b]€{float(open_eur):,.0f}[/b] receivable · "
            f"[b]€{float(oblig_eur):,.0f}[/b] due over 90 days · "
            f"opening balance [b]€{float(opening):,.0f}[/b]"
        ]
        for segment, mean_delay, open_amount in rows:
            out.append(
                f"  {segment:<12} pays in ~{float(mean_delay):>4.0f} d · "
                f"€{float(open_amount):>9,.0f} open  {bar(float(open_amount), hi, 18)}"
            )
        return "\n".join(out)

    def build(self, params) -> list[Step]:
        day = int(float(params["cover_day"]))
        floor = float(params["min_balance"])
        total_draws = N_DRAWS * CHAINS
        return [
            Step(
                title="Reconciliation gate",
                kind=Kind.GATE,
                why=(
                    "Before modelling anything, check the open items add up. A cash "
                    "forecast built on an AR ledger that does not reconcile to the GL "
                    "is precise about the wrong number, and the finding — 'your open "
                    "items and your GL disagree by €X' — is worth more to the customer "
                    "than the forecast would have been.\n\n"
                    "In a real deployment this compares against the GL balance. Here it "
                    "checks the internal consistency the simulation depends on: every "
                    "open item is positive, dated, and belongs to a segment the model "
                    "will have learned a habit for."
                ),
                sql="""
SELECT count(*) = 0                                     AS reconciles,
       count(*) FILTER (WHERE amount_eur <= 0)          AS non_positive_amounts,
       count(*) FILTER (WHERE age_days < 0)             AS impossible_ages,
       count(*) FILTER (WHERE segment NOT IN (SELECT segment FROM cleared))
                                                        AS segments_without_history
FROM open_items
WHERE amount_eur <= 0
   OR age_days < 0
   OR segment NOT IN (SELECT segment FROM cleared);
                """,
                verdict=lambda rows: (
                    "[green]Reconciled.[/green] Every open item is positive, dated, and "
                    "belongs to a segment with payment history to learn from."
                    if rows and rows[0][0]
                    else "[red]REFUSE[/red] — the ledger does not reconcile. That is the "
                         "deliverable, not the forecast."
                ),
            ),
            Step(
                title="How does each segment actually pay?",
                kind=Kind.PROFILE,
                why=(
                    "The mean sits above the median in every segment. That is the whole "
                    "problem in one number: payment delays are [b]right-skewed[/b], so a "
                    "planning model built on an average — or on a symmetric error around "
                    "one — is wrong in the direction that causes an overdraft."
                ),
                sql="""
SELECT segment,
       count(*)                                   AS cleared_items,
       round(avg(delay_days), 1)                  AS mean_days,
       round(median(delay_days), 1)               AS median_days,
       round(quantile_cont(delay_days, 0.95), 1)  AS p95_days,
       round(avg(delay_days) - median(delay_days), 1) AS skew_days
FROM cleared
GROUP BY segment
ORDER BY mean_days;
                """,
                verdict=lambda rows: (
                    "[b]Every segment's mean is above its median[/b] — that is the skew, "
                    "and it is why the next step fits a Gamma rather than a Gaussian on "
                    "the raw delay."
                    if rows and all(float(r[2]) > float(r[3]) for r in rows)
                    else "[yellow]Not every segment is right-skewed here — read the "
                         "table before trusting the tail.[/yellow]"
                ),
            ),
            Step(
                title="Fit — Gamma delays, pooled across segments",
                kind=Kind.FIT,
                silent=True,
                why=(
                    "One call. `payment_delay` learns each segment's mean delay and how "
                    "much the segments differ, and pools the thin ones toward the "
                    "ledger — `tau` is a parameter with a posterior, not a dial.\n\n"
                    "Note what is refused rather than configured: a delay of zero or "
                    "less is a [i]request error[/i] naming the clock, because an unpaid "
                    "open item is right-censored and belongs to `censored_aft`, not to "
                    "this family. Silently treating it as a zero-day delay would bias "
                    "the forecast in exactly the direction that matters."
                ),
                sql=f"""
CREATE OR REPLACE TABLE {DRAWS} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT segment, delay_days FROM cleared),
    'payment_delay',
    {{'y': 'delay_days',
     'group': 'segment',
     'dist': 'gamma',
     'draws': {N_DRAWS},
     'chains': {CHAINS},
     'warmup': 2000,
     'seed': 40404}}
);
                """,
            ),
            Step(
                title="Is the fit safe to act on?",
                kind=Kind.DIAGNOSE,
                why=(
                    "NUTS is the only engine this family offers, so a divergence is "
                    "possible and a single one grades the fit `degenerate` — draws "
                    "around a divergent trajectory are not from the posterior, and a "
                    "cash probability computed from them would be confident about the "
                    "wrong distribution."
                ),
                sql=f"""
SELECT anofox_bayes_is_actionable(param, value) AS safe_to_act_on,
       anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_family_text(param, value)   AS family,
       max(CASE WHEN param = '__engine__' THEN value END)    AS engine_code,
       sum(CASE WHEN param = '__divergent__' THEN value END) AS divergences
FROM {DRAWS};
                """,
                verdict=lambda rows: (
                    "[green]DECISION[/green] — clean sampler, no divergences."
                    if rows and rows[0][0]
                    else "[yellow]Read the status before using the probabilities.[/yellow]"
                ),
            ),
            Step(
                title="Each segment's payment habit, with its uncertainty",
                kind=Kind.DECIDE,
                why=(
                    "`mu` is the segment's mean delay, reported whole rather than as "
                    "something to exponentiate. The interval is what the pooling buys: "
                    "a segment with 45 cleared items gets a tight one, and a thinner "
                    "segment borrows from the ledger rather than reporting noise."
                ),
                sql=f"""
SELECT group_id                                            AS segment,
       round(median(value), 1)                             AS mean_delay_days,
       round(anofox_bayes_credible_lower(value, 0.90), 1)  AS ci_lower,
       round(anofox_bayes_credible_upper(value, 0.90), 1)  AS ci_upper
FROM {DRAWS}
WHERE param = 'mu' AND draw >= 0
GROUP BY group_id
ORDER BY mean_delay_days;
                """,
                chart=_segment_intervals,
            ),
            Step(
                title="The forward simulation — a cash path per draw",
                kind=Kind.DECIDE,
                silent=True,
                why=(
                    "[b]This is the step worth reading the SQL for.[/b] Every open item "
                    "is crossed with every posterior draw and given a payment day: the "
                    "segment's mean delay for that draw, turned into an actual duration "
                    "by Wilson–Hilferty on a keyed normal, minus the days the invoice "
                    "has already aged.\n\n"
                    f"That is {total_draws:,} draws × the open items, aggregated into a "
                    "daily balance per draw. No Python, no loop, no sampler — the "
                    "posterior is a table and this is a join over it.\n\n"
                    "[b]The noise is keyed, not random.[/b] `anofox_bayes_std_normal` is "
                    "a pure function of `(seed, key, draw)`, so this table is identical "
                    "on every machine and every run. A liquidity forecast an auditor "
                    "cannot reproduce is not a forecast."
                ),
                sql=f"""
CREATE OR REPLACE TABLE cash_path AS
WITH shape AS (
    SELECT chain, draw, value AS k FROM {DRAWS} WHERE param = 'shape' AND draw >= 0
),
level AS (
    SELECT chain, draw, group_id AS segment, value AS mu
    FROM {DRAWS} WHERE param = 'mu' AND draw >= 0
),
paid AS (
    SELECT (l.chain * {N_DRAWS} + l.draw)               AS d,
           o.item, o.amount_eur,
           -- The delay this draw implies, less the age already elapsed. An
           -- invoice cannot be paid in the past, so the floor is today.
           greatest(0, ceil(
               l.mu * pow(1.0 - 1.0 / (9.0 * s.k)
                          + anofox_bayes_std_normal(
                                40404, o.item, (l.chain * {N_DRAWS} + l.draw)::BIGINT)
                            / sqrt(9.0 * s.k), 3)
               - o.age_days))::INTEGER                  AS pay_day
    FROM level l
    JOIN shape s USING (chain, draw)
    JOIN open_items o ON o.segment = l.segment
),
days AS (SELECT range AS day FROM range(0, 91)),
draws AS (SELECT DISTINCT d FROM paid),
inflow AS (
    SELECT d, pay_day AS day, sum(amount_eur) AS eur
    FROM paid WHERE pay_day <= 90 GROUP BY d, pay_day
),
outflow AS (SELECT day, sum(amount_eur) AS eur FROM obligations GROUP BY day)
SELECT dr.d,
       dy.day,
       (SELECT balance_eur FROM opening)
         + coalesce(sum(i.eur) OVER w, 0)
         - coalesce(sum(o.eur) OVER w, 0) AS balance_eur
FROM draws dr
CROSS JOIN days dy
LEFT JOIN inflow  i ON i.d = dr.d AND i.day = dy.day
LEFT JOIN outflow o ON o.day = dy.day
WINDOW w AS (PARTITION BY dr.d ORDER BY dy.day
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

SELECT count(*) AS rows_in_cash_path, count(DISTINCT d) AS draws, max(day) AS horizon_days
FROM cash_path;
                """,
            ),
            Step(
                title=f"P(covered on day {day})",
                kind=Kind.DECIDE,
                why=(
                    "The question the CFO actually asked, as one aggregation over the "
                    "path table. Not 'the forecast balance is €X' — a share of "
                    "simulated futures in which the balance holds.\n\n"
                    "Press [b]w[/b] to move the date or the floor. It re-reads this "
                    "table; the model is not touched."
                ),
                sql=f"""
SELECT day,
       round(anofox_bayes_prob_greater(balance_eur, {floor}), 3) AS p_covered,
       round(median(balance_eur), 2)                             AS median_balance,
       round(anofox_bayes_credible_lower(balance_eur, 0.90), 2)  AS p05_balance,
       round(anofox_bayes_credible_upper(balance_eur, 0.90), 2)  AS p95_balance
FROM cash_path
WHERE day = {day}
GROUP BY day;
                """,
                verdict=lambda rows: _cover_verdict(rows, day, floor),
            ),
            Step(
                title="The 90-day fan, and where it dips",
                kind=Kind.DECIDE,
                why=(
                    "The one-pager. Three quantiles of the cash position across the "
                    "horizon, plus the probability of cover on each obligation date — "
                    "which is the row a Geschäftsführer reads first.\n\n"
                    "The dips are the payroll and tax dates; what matters is not that "
                    "the median dips but whether the [i]lower[/i] band crosses the "
                    "floor, and on which date."
                ),
                sql=f"""
SELECT day,
       round(anofox_bayes_credible_lower(balance_eur, 0.90), 0) AS p05,
       round(median(balance_eur), 0)                            AS median,
       round(anofox_bayes_credible_upper(balance_eur, 0.90), 0) AS p95,
       round(anofox_bayes_prob_greater(balance_eur, {floor}), 3) AS p_covered
FROM cash_path
GROUP BY day
ORDER BY day;
                """,
                chart=_fan_chart,
                verdict=lambda rows: _fan_verdict(rows, floor),
            ),
            Step(
                title="The chase list — which invoice moves the number most",
                kind=Kind.DECIDE,
                why=(
                    "The action. For each open item, how much the probability of cover "
                    "would rise if that one invoice landed today — computed by "
                    "re-aggregating the same draws with that item forced early. Pure "
                    "SQL, no refit.\n\n"
                    "[b]This is not a ranking by amount.[/b] A large invoice from a "
                    "segment that already pays before the obligation date moves nothing; "
                    "a mid-sized one from a slow segment, currently landing just after "
                    "the date, moves everything. That reordering is the product."
                ),
                sql=f"""
WITH shape AS (
    SELECT chain, draw, value AS k FROM {DRAWS} WHERE param = 'shape' AND draw >= 0
),
level AS (
    SELECT chain, draw, group_id AS segment, value AS mu
    FROM {DRAWS} WHERE param = 'mu' AND draw >= 0
),
paid AS (
    SELECT (l.chain * {N_DRAWS} + l.draw) AS d, o.item, o.segment, o.amount_eur,
           greatest(0, ceil(
               l.mu * pow(1.0 - 1.0 / (9.0 * s.k)
                          + anofox_bayes_std_normal(
                                40404, o.item, (l.chain * {N_DRAWS} + l.draw)::BIGINT)
                            / sqrt(9.0 * s.k), 3)
               - o.age_days))::INTEGER AS pay_day
    FROM level l JOIN shape s USING (chain, draw)
    JOIN open_items o ON o.segment = l.segment
),
baseline AS (
    SELECT d, sum(amount_eur) FILTER (WHERE pay_day <= {day}) AS arrived
    FROM paid GROUP BY d
),
outflow AS (SELECT sum(amount_eur) AS due FROM obligations WHERE day <= {day}),
covered AS (
    SELECT b.d,
           (SELECT balance_eur FROM opening) + b.arrived - (SELECT due FROM outflow)
             > {floor} AS ok
    FROM baseline b
),
lifted AS (
    -- The same balance, with item i forced to land on or before the date.
    SELECT p.item, p.segment, max(p.amount_eur) AS amount_eur,
           avg(CASE WHEN (SELECT balance_eur FROM opening)
                         + b.arrived
                         + CASE WHEN p.pay_day > {day} THEN p.amount_eur ELSE 0 END
                         - (SELECT due FROM outflow) > {floor}
                    THEN 1.0 ELSE 0.0 END) AS p_covered_if_chased
    FROM paid p JOIN baseline b ON b.d = p.d
    GROUP BY p.item, p.segment
)
SELECT l.item, l.segment,
       round(l.amount_eur, 2)                                            AS amount_eur,
       round((SELECT avg(CASE WHEN ok THEN 1.0 ELSE 0.0 END) FROM covered), 3)
                                                                         AS p_covered_now,
       round(l.p_covered_if_chased, 3)                                   AS p_if_chased,
       round(l.p_covered_if_chased
             - (SELECT avg(CASE WHEN ok THEN 1.0 ELSE 0.0 END) FROM covered), 4)
                                                                         AS delta_p
FROM lifted l
ORDER BY delta_p DESC, amount_eur DESC
LIMIT 12;
                """,
                chart=_chase_chart,
                verdict=lambda rows: (
                    "[b]Chase from the top.[/b] The ranking is by how much each invoice "
                    "moves the probability of cover — not by size and not by days "
                    "overdue."
                    if rows
                    else ""
                ),
            ),
            Step(
                title="The same ledger as a lognormal — how far apart is the tail?",
                kind=Kind.DECIDE,
                why=(
                    "[b]The catalog has more than one right answer, and the difference "
                    "is the decision.[/b] A lognormal fits the same skew and is what "
                    "`pooled_gaussian` on `log(delay)` would give you. Switching `dist` "
                    "is one config slot — both branches parameterise the mean, so the "
                    "coefficients keep their meaning.\n\n"
                    "They agree closely about the centre. They disagree about the far "
                    "right tail, which is exactly where a covenant test lives. This step "
                    "measures the gap rather than asserting which is right; on a real "
                    "ledger that is an empirical question and the calibration report "
                    "answers it."
                ),
                sql=f"""
CREATE OR REPLACE TABLE lognormal_draws AS
SELECT * FROM anofox_bayes_fit(
    (SELECT segment, delay_days FROM cleared),
    'payment_delay',
    {{'y': 'delay_days', 'group': 'segment', 'dist': 'lognormal',
     'draws': {N_DRAWS}, 'chains': {CHAINS}, 'warmup': 2000, 'seed': 40404}}
);

WITH gamma_tail AS (
    SELECT l.group_id AS segment,
           anofox_bayes_service_level_quantile(
               l.value * pow(1.0 - 1.0 / (9.0 * s.value)
                             + anofox_bayes_std_normal(1, l.group_id, l.draw::BIGINT)
                               / sqrt(9.0 * s.value), 3), 0.99) AS p99_days
    FROM (SELECT group_id, chain, draw, value FROM {DRAWS}
          WHERE param = 'mu' AND draw >= 0) l
    JOIN (SELECT chain, draw, value FROM {DRAWS}
          WHERE param = 'shape' AND draw >= 0) s USING (chain, draw)
    GROUP BY l.group_id
),
lognormal_tail AS (
    SELECT l.group_id AS segment,
           anofox_bayes_service_level_quantile(
               l.value * exp(-0.5 * s.value * s.value
                             + s.value * anofox_bayes_std_normal(
                                   1, l.group_id, l.draw::BIGINT)), 0.99) AS p99_days
    FROM (SELECT group_id, chain, draw, value FROM lognormal_draws
          WHERE param = 'mu' AND draw >= 0) l
    JOIN (SELECT chain, draw, value FROM lognormal_draws
          WHERE param = 'sigma' AND draw >= 0) s USING (chain, draw)
    GROUP BY l.group_id
)
SELECT g.segment,
       round(g.p99_days, 1)                        AS gamma_p99_days,
       round(n.p99_days, 1)                        AS lognormal_p99_days,
       round(n.p99_days - g.p99_days, 1)           AS gap_days
FROM gamma_tail g JOIN lognormal_tail n USING (segment)
ORDER BY abs(n.p99_days - g.p99_days) DESC;
                """,
                verdict=_branch_verdict,
            ),
        ]

    def summary(self, con, results) -> str:
        try:
            rows = con.sql(
                """
                SELECT o.day, o.purpose, o.amount_eur,
                       round(anofox_bayes_prob_greater(c.balance_eur, 0.0), 3)
                FROM obligations o JOIN cash_path c ON c.day = o.day
                GROUP BY o.day, o.purpose, o.amount_eur
                ORDER BY o.day
                """
            ).fetchall()
        except duckdb.Error:
            return ""
        if not rows:
            return ""
        out = ["[b]Liquiditäts-Einseiter[/b] — probability of solvency on each due date\n"]
        for day, purpose, amount, p in rows:
            p = float(p)
            colour = "green" if p >= 0.95 else ("yellow" if p >= 0.80 else "red")
            out.append(
                f"  day {day:>2}  {purpose:<20} €{float(amount):>10,.0f}  "
                f"[{colour}]P(covered) = {p:.1%}[/{colour}]  {bar(p, 1.0, 18)}"
            )
        out.append(
            "\n[dim]One fit. Every probability above, the fan, and the chase ranking "
            "came from the same draws table. Press [b]w[/b] to move the date or the "
            "covenant floor.[/dim]"
        )
        return "\n".join(out)


def _segment_intervals(rows) -> str:
    if not rows:
        return ""
    lo = min(float(r[2]) for r in rows)
    hi = max(float(r[3]) for r in rows)
    out = ["  [dim]90 % credible interval for the segment's mean delay (days)[/dim]"]
    for segment, med, a, b in rows:
        out.append(
            f"  {segment:<12} {interval_bar(float(a), float(med), float(b), lo, hi, 30)} "
            f"[dim]{float(a):.0f} – {float(b):.0f} d[/dim]"
        )
    return "\n".join(out)


def _cover_verdict(rows, day: int, floor: float) -> str:
    if not rows:
        return ""
    _, p, median, p05, _p95 = rows[0]
    p = float(p)
    colour = "green" if p >= 0.95 else ("yellow" if p >= 0.80 else "red")
    floor_text = "solvent" if floor == 0 else f"above €{floor:,.0f}"
    return (
        f"[{colour}]P(covered on day {day}) = {p:.1%}[/{colour}]  "
        f"{bar(p, 1.0, 24)}\n"
        f"Median balance €{float(median):,.0f}; the unlucky 5 % of futures land at "
        f"€{float(p05):,.0f} or worse. The decision is about staying {floor_text}, and "
        "it is that lower band — not the median — that decides it."
    )


def _fan_chart(rows) -> str:
    if not rows:
        return ""
    series = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows]
    lines = fan(series, width=76)
    return (
        "  [dim]90 % fan of the cash balance over 90 days[/dim]\n"
        f"  p95    {lines[0]}\n"
        f"  median {lines[1]}\n"
        f"  p05    {lines[2]}"
    )


def _fan_verdict(rows, floor: float) -> str:
    if not rows:
        return ""
    worst = min(rows, key=lambda r: float(r[4]))
    breach = [r for r in rows if float(r[1]) <= floor]
    text = (
        f"Lowest probability of cover across the horizon: "
        f"[b]{float(worst[4]):.1%}[/b] on day {worst[0]}."
    )
    if breach:
        first = breach[0]
        return (
            text
            + f"\n[yellow]The lower band crosses the floor first on day {first[0]}[/yellow] "
            "— that is the date to act before, not the date the median dips."
        )
    return text + "\n[green]The lower band stays above the floor for the whole horizon.[/green]"


def _chase_chart(rows) -> str:
    if not rows:
        return ""
    hi = max(float(r[5]) for r in rows) or 1.0
    if hi <= 0:
        return (
            "  [dim]No single invoice moves the probability at this date — either "
            "cover is already near-certain, or no one invoice is large enough. Move "
            "the date with [b]w[/b] to find one where it bites.[/dim]"
        )
    out = ["  [dim]Δ probability of cover if this invoice landed today[/dim]"]
    for item, segment, amount, _now, _if_chased, delta in rows:
        out.append(
            f"  {item:<22} {segment:<12} €{float(amount):>8,.0f}  "
            f"{bar(float(delta), hi, 18)} [b]+{float(delta):.3f}[/b]"
        )
    return "\n".join(out)


def _branch_verdict(rows) -> str:
    if not rows:
        return ""
    gaps = [abs(float(r[3])) for r in rows]
    worst = rows[0]
    return (
        f"[b]The two branches disagree by up to {max(gaps):.0f} days[/b] at the 99th "
        f"percentile ({worst[0]}: Gamma {float(worst[1]):.0f} d vs lognormal "
        f"{float(worst[2]):.0f} d).\n"
        "They agree about the centre — that is what makes this a statement about the "
        "tail. On a thirty-day cycle that gap is working capital, and it is the reason "
        "`payment_delay` offers both rather than picking one for you."
    )


DEMO = CashRunway()


def run() -> int:
    return main(DEMO)
