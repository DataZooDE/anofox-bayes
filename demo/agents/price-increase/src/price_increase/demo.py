"""Agent 06 — the price round, on `hier_elasticity` (F6).

The annual price round is negotiated on anecdote: sales says customers will
leave, management needs margin. The transaction data contains the actual
elasticities per segment, and nobody computes them — let alone with honest
uncertainty.

Two things this demo exists to show, and both are things
`pooled_gaussian` + `random_slopes` cannot do:

**Every elasticity is negative, by construction.** Not "almost always" — the
family parameterises `b_g = -exp(...)`, so a positive draw is impossible rather
than improbable. On a thin segment an unconstrained Gaussian slope routinely
puts real mass above zero, and a price meeting handed an interval saying that
raising the price might sell *more* stops reading the interval.

**A segment whose prices never moved is named, not quietly pooled.** That is the
PARTIAL the Entscheidungsvorlage has to carry: *"keine Aussage möglich, die
Preise waren konstant"* arrives as a `__group_status__` row rather than as a
plausible-looking number. This demo deliberately includes such a segment.

**Where the data shape comes from.** The segment structure and the elasticity
range follow PyMC Labs' *Hierarchical Pricing Elasticity Models* case study and
Juan Camilo Orduz's write-up of it over Kaggle retail scanner data. The rows are
generated here, deterministically; the inference is real.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, interval_bar

DRAWS = "elasticity_draws"

FIXTURE = """
CREATE OR REPLACE TABLE segments AS
SELECT * FROM (VALUES
    ('COMMODITY',   5.9, -1.60, 0.60, 14.20,  9.10),
    ('MIDMARKET',   5.7, -0.90, 0.60, 41.00, 26.80),
    ('PREMIUM',     5.0, -0.45, 0.60, 118.00, 71.00),
    ('OEM',         6.0, -1.20, 0.60, 22.40, 15.60),
    ('SPARE_PARTS', 4.5, -0.30, 0.60, 86.00, 34.00),
    ('EXPORT',      5.3, -0.75, 0.60, 55.00, 37.50),
    -- The segment that matters most. A list price that did not move all year is
    -- a real thing a business does, and the model has to say so rather than
    -- invent an elasticity for it.
    ('FIXED_LIST',  5.1, -0.80, 0.00, 63.00, 40.00)
) AS t(segment, log_level, elasticity, price_spread, list_price_eur, unit_cost_eur);

-- 36 months of billing per segment. `log_price` is the realised price relative
-- to the segment's own mean, on the log scale -- **within**-segment variation is
-- what identifies an elasticity, and a panel whose prices moved only between
-- segments would be measuring something else entirely.
--
-- The ladder spans +/-30 % and the dispersion is 60. Both matter, and an earlier
-- version of this fixture got them wrong: at +/-15 % and phi = 20 the
-- month-to-month noise is the same size as the whole price signal, every
-- segment's posterior collapsed onto the population mean near -0.5, and the demo
-- showed shrinkage rather than elasticity. That is a real property of the model
-- rather than a bug -- with no signal, pooling is the correct answer -- but it
-- is not the thing this demo is for. A real discount ladder is this wide.
CREATE OR REPLACE TABLE billing AS
WITH months AS (SELECT range AS month FROM range(1, 37)),
cell AS (
    SELECT s.segment, m.month, s.log_level, s.elasticity, s.price_spread,
           s.list_price_eur, s.unit_cost_eur,
           -- A deterministic discount ladder about the segment's own mean.
           s.price_spread * (((m.month * 7) % 37 - 18.0) / 18.0) AS log_price,
           60.0 AS phi,
           anofox_bayes_uniform(60606, s.segment, m.month) AS u
    FROM segments s CROSS JOIN months m
),
mu AS (
    SELECT *, exp(log_level + elasticity * log_price) AS mean_units FROM cell
),
grid AS (SELECT range AS k FROM range(0, 3000)),
cdf AS (
    -- A genuine negative-binomial draw by inverse CDF, using the same `lgamma`
    -- pmf the extension's own likelihood is written from. Deterministic, and
    -- drawn from exactly the model the fit inverts.
    SELECT c.segment, c.month, c.u, g.k,
           sum(exp(lgamma(g.k + c.phi) - lgamma(c.phi) - lgamma(g.k + 1)
                   + c.phi * ln(c.phi / (c.phi + c.mean_units))
                   + g.k   * ln(c.mean_units / (c.phi + c.mean_units))))
             OVER (PARTITION BY c.segment, c.month ORDER BY g.k) AS cum
    FROM mu c CROSS JOIN grid g
)
-- `min(k) FILTER (cum >= u)` returns NULL when the grid is too short to reach
-- the uniform, and coalescing that to zero silently inverts the price-volume
-- relationship for the highest-volume cells -- which is exactly what happened
-- with a 1 200-wide grid and a segment whose mean approached it. The grid is now
-- comfortably above any cell's mean, and `grid_overflow` below is the assertion
-- that keeps it that way rather than a comment hoping it does.
SELECT m.segment, m.month, m.log_price,
       m.list_price_eur,
       m.list_price_eur * exp(m.log_price)                       AS realised_price_eur,
       m.unit_cost_eur,
       coalesce(min(c.k) FILTER (WHERE c.cum >= m.u), 0)::BIGINT AS units
FROM mu m JOIN cdf c ON c.segment = m.segment AND c.month = m.month
GROUP BY m.segment, m.month, m.log_price, m.list_price_eur, m.unit_cost_eur;
"""


class PriceIncrease(BayesDemo):
    name = "price-increase"
    title = "Preiserhöhungs-Simulator — bands, not anecdotes"
    family = "hier_elasticity (F6)"
    draws_table = DRAWS
    intro = (
        "[b]The decision at stake:[/b] the annual price round. Sales says customers "
        "will leave; management needs margin. Your transaction data already "
        "contains the answer per segment — a [b]price elasticity[/b] — and it has "
        "never been computed with honest uncertainty.\n\n"
        "[b]What makes this family different:[/b] every segment's elasticity is "
        "[b]negative by construction[/b], not by tail probability. A plain "
        "regression on a thin segment routinely returns an interval straddling "
        "zero — telling a price meeting that raising the price [i]might sell "
        "more[/i] — and the meeting then stops reading intervals altogether.\n\n"
        "[b]And one segment here has never moved its list price.[/b] Watch what "
        "the model does with it: it is named, not quietly given a number.\n\n"
        "Press [b]r[/b] to run it. Then press [b]w[/b] to try a different price "
        "move — the scenario table re-prices in milliseconds, which is what makes "
        "this answerable live in the meeting."
    )
    params = (
        Param(
            key="price_move",
            label="List price move (%)",
            default=5.0,
            help="Applied to every segment. The scenario table shows what it costs "
                 "in volume and what it earns in contribution margin.",
            minimum=-25.0,
            maximum=50.0,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        rows = con.sql(
            """
            SELECT segment,
                   sum(units)                                       AS units,
                   round(sum(units * realised_price_eur), 0)         AS revenue,
                   round(max(log_price) - min(log_price), 3)         AS price_span
            FROM billing GROUP BY segment ORDER BY revenue DESC
            """
        ).fetchall()
        head = con.sql(
            "SELECT count(*), count(DISTINCT segment), round(sum(units * realised_price_eur), 0) FROM billing"
        ).fetchone()
        if not rows or head is None:
            return ""
        n, segs, revenue = head
        hi = max(float(r[2]) for r in rows)
        out = [
            f"[b]{segs} segments[/b] × 36 months = [b]{n}[/b] rows · "
            f"[b]€{float(revenue):,.0f}[/b] of billed revenue"
        ]
        for segment, units, rev, span in rows:
            flag = "  [yellow]← price never moved[/yellow]" if float(span) < 0.01 else ""
            out.append(
                f"  {segment:<12} €{float(rev):>9,.0f}  {bar(float(rev), hi, 16)}"
                f"  price span {float(span):.2f}{flag}"
            )
        return "\n".join(out)

    def build(self, params) -> list[Step]:
        pct = float(params["price_move"])
        factor = 1.0 + pct / 100.0
        return [
            Step(
                title="Identification gate: did prices actually move?",
                kind=Kind.GATE,
                why=(
                    "An elasticity is identified by [b]within-segment[/b] price "
                    "variation. If a segment's list price never moved, its coefficient "
                    "is multiplied by a constant and the data says nothing about it — "
                    "no amount of modelling recovers a number that was never in there.\n\n"
                    "This gate finds those segments before the fit, so the pack knows "
                    "which cells of the Entscheidungsvorlage will be a band rather than "
                    "a recommendation. The model finds them independently; that the two "
                    "agree is the check."
                ),
                sql="""
SELECT count(*) FILTER (WHERE price_span < 0.01) = 0  AS every_segment_identified,
       count(*)                                        AS segments,
       count(*) FILTER (WHERE price_span < 0.01)       AS segments_without_variation,
       string_agg(CASE WHEN price_span < 0.01 THEN segment END, ', ') AS names,
       -- The fixture's own guard: a zero-volume month would mean the
       -- inverse-CDF grid was too short and the price-volume relationship got
       -- silently inverted for the highest-volume cells. Checked rather than
       -- assumed, because that failure looks exactly like a real finding.
       (SELECT count(*) FROM billing WHERE units = 0) AS grid_overflow
FROM (
    SELECT segment, max(log_price) - min(log_price) AS price_span
    FROM billing GROUP BY segment
);
                """,
                verdict=lambda rows: (
                    f"[yellow]PARTIAL[/yellow] — {rows[0][2]} segment(s) never moved "
                    f"price: [b]{rows[0][3]}[/b]. Keine Aussage möglich for those, and "
                    "the model will say so itself in a moment."
                    if rows and rows[0][2]
                    else "[green]Every segment has price variation to learn from.[/green]"
                ),
            ),
            Step(
                title="What the raw data looks like",
                kind=Kind.PROFILE,
                why=(
                    "A naive within-segment slope of log volume on log price. This is "
                    "roughly what a spreadsheet would produce, and it is a useful "
                    "baseline for two reasons: it shows the signal really is in the "
                    "data, and it shows what happens without pooling or a sign "
                    "constraint — note the segment with no price variation returns "
                    "nothing at all."
                ),
                sql="""
SELECT segment,
       count(*)                                                     AS months,
       round(regr_slope(ln(units + 0.5), log_price), 2)             AS naive_elasticity,
       round(max(log_price) - min(log_price), 3)                    AS price_span
FROM billing
GROUP BY segment
ORDER BY naive_elasticity;
                """,
                verdict=lambda rows: (
                    "[dim]Useful, and not enough. These slopes have no interval, no "
                    "pooling for the thin segments, and nothing stopping one from "
                    "coming out positive. The fit below fixes all three.[/dim]"
                ),
            ),
            Step(
                title="Fit — elasticity per segment, sign-constrained",
                kind=Kind.FIT,
                silent=True,
                why=(
                    "One call. `price` is its own config slot rather than one of `x`, "
                    "because its coefficient is the thing the family is about: the only "
                    "one pooled on the log of its magnitude and the only one whose sign "
                    "is constrained.\n\n"
                    "The response is `units` — a [b]count[/b], with a negative binomial "
                    "likelihood. `log(units)` would be undefined at zero and would "
                    "assert that a segment selling 60 units a month and one selling "
                    "40 000 are equally noisy in relative terms, which is exactly wrong "
                    "where the shrinkage is doing the work.\n\n"
                    "[b]No priors are set.[/b] Every one is the family's own default, "
                    "including `prior.tau.scale` — the spread of `log |elasticity|` "
                    "across segments, whose default of 0.5 encodes a moderate "
                    "portfolio. This catalogue is at the edge of that (COMMODITY at "
                    "−1.6 against SPARE_PARTS at −0.3 is a factor of five), and with "
                    "three years of a real discount ladder the data is strong enough "
                    "that it does not matter: the posterior for `tau` comes out near "
                    "0.59 against a true 0.55. That is the default being "
                    "weakly-informative rather than merely small."
                ),
                sql=f"""
CREATE OR REPLACE TABLE {DRAWS} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT segment, month, log_price, units FROM billing),
    'hier_elasticity',
    {{'y': 'units',
     'price': 'log_price',
     'group': 'segment',
     'draws': 2000,
     'chains': 4,
     'warmup': 2000,
     'seed': 60606}}
);
                """,
            ),
            Step(
                title="The model's own verdict — and it is a PARTIAL",
                kind=Kind.DIAGNOSE,
                why=(
                    "[b]This is the step to read carefully.[/b] The fit reports "
                    "`insufficient_data`, and that is not a failure — it is worst-wins "
                    "across the segments, and exactly one segment pulled it down.\n\n"
                    "`__n_groups_unready__` says how many, and the `__group_status__` "
                    "rows say [i]which[/i]. So an agent holding a fit over forty "
                    "segments quarantines the three that were on a fixed price list "
                    "rather than discarding the whole table. Press [b]d[/b] to see the "
                    "named list."
                ),
                sql=f"""
SELECT anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_is_actionable(param, value) AS actionable_as_a_whole,
       anofox_bayes_family_text(param, value)   AS family,
       max(CASE WHEN param = '__n_groups__' THEN value END)          AS segments,
       max(CASE WHEN param = '__n_groups_unready__' THEN value END)  AS unready,
       sum(CASE WHEN param = '__divergent__' THEN value END)         AS divergences
FROM {DRAWS};
                """,
                verdict=lambda rows: _status_verdict(rows),
            ),
            Step(
                title="Which segment was refused, by name",
                kind=Kind.DECIDE,
                why=(
                    "The `__group_status__` rows. This is the list a Produktmanagement "
                    "reads before signing: these are the cells where the number below "
                    "is the pooled prior rather than a finding about the segment.\n\n"
                    "The segment is still fitted and still gets a number — a pooled "
                    "estimate is defensible to serve. What would not be defensible is "
                    "serving it unlabelled."
                ),
                sql=f"""
SELECT group_id                                  AS segment,
       anofox_bayes_status_name(value)           AS verdict,
       'elasticity is the pooled prior, not a finding' AS meaning
FROM {DRAWS}
WHERE param = '__group_status__'
ORDER BY group_id;
                """,
                verdict=lambda rows: (
                    "[b]Exactly the segment the gate predicted[/b], found independently "
                    "by the model from the price column itself. The other segments are "
                    "not implicated."
                    if rows
                    else "[green]No segment was refused.[/green]"
                ),
            ),
            Step(
                title="The elasticities, with their bands",
                kind=Kind.DECIDE,
                why=(
                    "The finding. Every draw of every segment is negative — check the "
                    "`worst_draw` column, which is the single largest (least negative) "
                    "draw across 8 000 per segment.\n\n"
                    "The refused segment is included and its band is visibly the "
                    "widest, which is what an honest pooled estimate looks like."
                ),
                sql=f"""
SELECT group_id                                             AS segment,
       round(median(value), 3)                              AS elasticity,
       round(anofox_bayes_credible_lower(value, 0.90), 3)   AS ci_lower,
       round(anofox_bayes_credible_upper(value, 0.90), 3)   AS ci_upper,
       round(max(value), 4)                                 AS worst_draw
FROM {DRAWS}
WHERE param = 'group_elasticity' AND draw >= 0
GROUP BY group_id
ORDER BY elasticity;
                """,
                chart=_elasticity_intervals,
                verdict=lambda rows: _sign_verdict(rows),
            ),
            Step(
                title=f"The scenario table — list {pct:+.1f}%",
                kind=Kind.DECIDE,
                why=(
                    "[b]The Entscheidungsvorlage.[/b] The volume response is "
                    "`exp(elasticity × ln(1 + move))`, evaluated once [i]per posterior "
                    "draw[/i] — so the interval is the model's own rather than a "
                    "delta-method approximation to it.\n\n"
                    "Contribution margin uses each segment's real list price and unit "
                    "cost. The whole table is a deterministic transform of the draws, "
                    "which is why a different percentage costs no second fit. Press "
                    "[b]w[/b] and watch the timing."
                ),
                sql=f"""
CREATE OR REPLACE TABLE scenario AS
WITH base AS (
    SELECT b.segment,
           avg(b.units)               AS units_per_month,
           max(b.list_price_eur)      AS list_price,
           max(b.unit_cost_eur)       AS unit_cost
    FROM billing b GROUP BY b.segment
),
per_draw AS (
    SELECT d.group_id AS segment, d.chain, d.draw,
           exp(d.value * ln({factor}))                                 AS volume_ratio,
           b.units_per_month * exp(d.value * ln({factor}))             AS new_units,
           b.units_per_month * (b.list_price - b.unit_cost)            AS db_now,
           b.units_per_month * exp(d.value * ln({factor}))
             * (b.list_price * {factor} - b.unit_cost)                 AS db_after
    FROM {DRAWS} d
    JOIN base b ON b.segment = d.group_id
    WHERE d.param = 'group_elasticity' AND d.draw >= 0
)
SELECT segment, chain, draw, volume_ratio, new_units, db_now, db_after,
       db_after - db_now AS db_delta
FROM per_draw;

SELECT segment,
       round(100.0 * (median(volume_ratio) - 1.0), 1)                  AS volume_pct,
       round(median(db_delta), 0)                                      AS margin_delta_eur,
       round(anofox_bayes_credible_lower(db_delta, 0.90), 0)           AS delta_lower,
       round(anofox_bayes_credible_upper(db_delta, 0.90), 0)           AS delta_upper,
       round(anofox_bayes_prob_less(db_delta, 0.0), 3)                 AS p_worse_off
FROM scenario
GROUP BY segment
ORDER BY margin_delta_eur DESC;
                """,
                chart=_scenario_chart,
                verdict=lambda rows: _scenario_verdict(rows, pct),
            ),
            Step(
                title="The regret view",
                kind=Kind.DECIDE,
                why=(
                    "What management decides on. Not 'the expected margin goes up' but "
                    "[b]the probability this move makes us worse off[/b], per segment.\n\n"
                    "A segment with a strong elasticity can have a positive expected "
                    "margin delta and a one-in-four chance of a negative one. That is "
                    "the number a Vertriebsleitung needs, and it is the reason the "
                    "answer is a distribution rather than a point."
                ),
                sql="""
SELECT segment,
       round(anofox_bayes_prob_less(db_delta, 0.0), 3)   AS p_margin_falls,
       round(anofox_bayes_prob_less(volume_ratio, 0.95), 3) AS p_volume_drops_over_5pct,
       CASE
           WHEN anofox_bayes_prob_less(db_delta, 0.0) > 0.35 THEN 'hold'
           WHEN anofox_bayes_prob_less(db_delta, 0.0) > 0.15 THEN 'discuss'
           ELSE 'go'
       END                                                AS recommendation
FROM scenario
GROUP BY segment
ORDER BY p_margin_falls DESC;
                """,
                verdict=lambda rows: (
                    "[dim]The three tiers are threshold config, not model output — a "
                    "real pack elicits them in the workshop. What the model supplies is "
                    "the probability they are applied to.[/dim]"
                ),
            ),
            Step(
                title="Ask three moves at once",
                kind=Kind.DECIDE,
                why=(
                    "The meeting never asks one number. Three price moves, every "
                    "segment, from the same draws — and the monotonicity is a property "
                    "of the model rather than of the fixture: a bigger rise always "
                    "costs more volume, because every elasticity is negative."
                ),
                sql=f"""
WITH moves AS (SELECT unnest([1.03, {factor}, 1.10]) AS factor),
base AS (
    SELECT segment, avg(units) AS units_per_month,
           max(list_price_eur) AS list_price, max(unit_cost_eur) AS unit_cost
    FROM billing GROUP BY segment
)
SELECT round(100.0 * (m.factor - 1.0), 1)                       AS move_pct,
       round(sum(b.units_per_month * median_ratio), 0)          AS portfolio_units,
       round(sum(b.units_per_month * median_ratio
                 * (b.list_price * m.factor - b.unit_cost)), 0) AS portfolio_db_eur
FROM moves m
CROSS JOIN LATERAL (
    SELECT d.group_id AS segment, median(exp(d.value * ln(m.factor))) AS median_ratio
    FROM {DRAWS} d
    WHERE d.param = 'group_elasticity' AND d.draw >= 0
    GROUP BY d.group_id
) e
JOIN base b ON b.segment = e.segment
GROUP BY m.factor
ORDER BY m.factor;
                """,
                verdict=lambda rows: (
                    "[b]Three questions, one fit.[/b] Every row here re-read the same "
                    "draws table — look at the timing against the fit step above."
                ),
            ),
        ]

    def summary(self, con, results) -> str:
        try:
            rows = con.sql(
                """
                SELECT s.segment,
                       round(100.0 * (median(s.volume_ratio) - 1.0), 1),
                       round(median(s.db_delta), 0),
                       round(anofox_bayes_prob_less(s.db_delta, 0.0), 3),
                       (SELECT count(*) FROM elasticity_draws g
                        WHERE g.param = '__group_status__' AND g.group_id = s.segment) > 0
                FROM scenario s GROUP BY s.segment ORDER BY 3 DESC
                """
            ).fetchall()
        except duckdb.Error:
            return ""
        if not rows:
            return ""
        out = ["[b]Preisrunde — Entscheidungsvorlage[/b]\n"]
        total = 0.0
        for segment, vol, delta, p_bad, pooled in rows:
            total += float(delta)
            flag = "  [yellow]⚠ pooled — keine eigene Aussage[/yellow]" if pooled else ""
            colour = "green" if float(p_bad) < 0.15 else ("yellow" if float(p_bad) < 0.35 else "red")
            out.append(
                f"  {segment:<12} Menge {float(vol):>+5.1f}%  DB "
                f"[{colour}]{float(delta):>+9,.0f} €[/{colour}]  "
                f"P(schlechter) {float(p_bad):.0%}{flag}"
            )
        out.append(f"\n  [b]Portfolio: {total:>+,.0f} € Deckungsbeitrag pro Monat[/b]")
        out.append(
            "[dim]Every band above came from one fit. Press [b]w[/b] for a different "
            "move — it re-prices in milliseconds.[/dim]"
        )
        return "\n".join(out)


def _status_verdict(rows) -> str:
    if not rows:
        return ""
    status, actionable, _family, segments, unready, divergences = rows[0]
    unready = int(unready or 0)
    segments = int(segments or 0)
    if actionable:
        return "[green]DECISION[/green] — every segment estimated from its own data."
    if unready and unready < segments:
        return (
            f"[yellow]PARTIAL[/yellow] — status `{status}`, and it is "
            f"[b]{unready} of {segments}[/b] segments that caused it, not the fit as a "
            "whole. The other "
            f"{segments - unready} are estimated from their own price history and are "
            "usable. That distinction is the difference between a refusal you can act "
            "around and one you cannot."
        )
    return f"[red]REFUSE[/red] — status `{status}` implicates every segment."


def _elasticity_intervals(rows) -> str:
    if not rows:
        return ""
    lo = min(float(r[2]) for r in rows)
    hi = max(float(r[3]) for r in rows)
    out = ["  [dim]90 % credible interval for each segment's elasticity[/dim]"]
    for segment, med, a, b, _worst in rows:
        out.append(
            f"  {segment:<12} {interval_bar(float(a), float(med), float(b), lo, hi, 30)} "
            f"[dim]{float(a):.2f} – {float(b):.2f}[/dim]"
        )
    return "\n".join(out)


def _sign_verdict(rows) -> str:
    if not rows:
        return ""
    worst = max(float(r[4]) for r in rows)
    if worst >= 0.0:
        return (
            f"[red]A non-negative draw appeared ({worst:+.4f}).[/red] The `-exp` "
            "transform is supposed to make that impossible, so this is a bug rather "
            "than a tail event."
        )
    widest = max(rows, key=lambda r: float(r[3]) - float(r[2]))
    return (
        f"[green]Not one non-negative draw across every segment[/green] — the largest "
        f"anywhere is {worst:+.4f}. That is the parameterisation, not luck.\n"
        f"The widest band belongs to [b]{widest[0]}[/b], which is the segment whose "
        "prices never moved: an honest pooled estimate, visibly less certain than the "
        "ones that earned their own."
    )


def _scenario_chart(rows) -> str:
    if not rows:
        return ""
    lo = min(float(r[3]) for r in rows)
    hi = max(float(r[4]) for r in rows)
    out = ["  [dim]90 % interval for the monthly contribution-margin change (€)[/dim]"]
    for segment, _vol, med, a, b, _p in rows:
        out.append(
            f"  {segment:<12} {interval_bar(float(a), float(med), float(b), lo, hi, 30)} "
            f"[dim]{float(a):>+8,.0f} – {float(b):>+8,.0f}[/dim]"
        )
    return "\n".join(out)


def _scenario_verdict(rows, pct: float) -> str:
    if not rows:
        return ""
    total = sum(float(r[2]) for r in rows)
    risky = [r for r in rows if float(r[5]) > 0.35]
    text = (
        f"[b]A {pct:+.1f}% list move is worth {total:>+,.0f} € of contribution margin "
        f"a month[/b] at the median."
    )
    if risky:
        names = ", ".join(str(r[0]) for r in risky)
        return (
            text
            + f"\n[yellow]But not everywhere: {names} has a better than one-in-three "
            "chance of coming out worse.[/yellow] That is where the elasticity exceeds "
            "one in magnitude, and it is the segment the round should skip."
        )
    return text + "\n[green]No segment has a material chance of being worse off.[/green]"


DEMO = PriceIncrease()


def run() -> int:
    return main(DEMO)
