"""Agent 01 — C-parts safety stock, on `hier_negbin` (F1).

The business problem, in one sentence: reorder points for cheap, slow-moving
parts are set by rules of thumb on a point forecast, and for a part that moves
three to thirty times a year a point forecast is the wrong object entirely —
the reorder point is a **quantile**, so the interval *is* the decision.

What makes this hard is not the arithmetic, it is the catalogue. Most C-parts
have a handful of observations each. Fit them one at a time and the thin ones
get intervals so wide they are useless; pool them all and the fast movers get
the catalogue's average. `hier_negbin` does neither: it learns how much the
parts differ and shrinks each one by that much, which is a number estimated
from the data rather than chosen by an analyst.

**Where the data shape comes from.** The mix of fast, medium, slow and
intermittent movers follows the pattern the anofox-evolve replenishment pilot
uses (10 fast / 30 medium / 60 slow / 25 intermittent), which in turn follows
the intermittent-demand literature PyMC Labs writes about under the
Teunter–Syntetos–Babai model. The numbers are generated, deterministically, from
this file's own text; the inference is real.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, interval_bar, sparkline

DRAWS = "draws"

# One fixture, generated in SQL so that the demo's data is as inspectable as its
# model. `anofox_bayes_std_normal` is a keyed, pure function of
# `(seed, key, draw)`, so this is a function of the file rather than of the run:
# no `setseed()`, no `random()`, the same rows on every machine.
FIXTURE = """
CREATE OR REPLACE TABLE parts AS
SELECT
    tier.name || '-' || lpad(i::VARCHAR, 3, '0') AS part,
    tier.warengruppe,
    tier.weekly_rate,
    tier.unit_cost
FROM (VALUES
        ('FAST',   'Verbindungselemente',  9.0, 0.42),
        ('MED',    'Dichtungen',           2.2, 1.85),
        ('SLOW',   'Lager',                0.55, 7.40),
        ('INTERM', 'Sonderteile',          0.14, 23.10)
     ) AS tier(name, warengruppe, weekly_rate, unit_cost),
     generate_series(1, 12) AS g(i);

-- A genuine negative-binomial draw, by inverse CDF.
--
-- The first version of this fixture cut a corner -- scale a Gamma weight, round
-- to an integer -- and it produced counts that were *under*-dispersed relative
-- to Poisson at low rates, which is the opposite of what a C-parts catalogue
-- looks like and the opposite of what this demo says on screen. Doing it
-- properly costs one window function.
--
-- The pmf below is the same `lgamma` expression the reorder-point step uses, run
-- in the opposite direction: build the CDF over k, then take the first k whose
-- cumulative probability passes a keyed uniform. So the fixture is drawn from
-- exactly the model the fit inverts, and `anofox_bayes_uniform` being a pure
-- function of `(seed, key, draw)` makes it a function of this file rather than
-- of the run.
CREATE OR REPLACE TABLE issues AS
WITH weeks AS (SELECT range AS week FROM range(1, 105)),
grid AS (SELECT range AS k FROM range(0, 80)),
cell AS (
    SELECT p.part, w.week, p.weekly_rate AS mu, 1.4 AS phi,
           anofox_bayes_uniform(20260801, p.part, w.week) AS u
    FROM parts p CROSS JOIN weeks w
),
cdf AS (
    SELECT c.part, c.week, c.u, g.k,
           sum(exp(lgamma(g.k + c.phi) - lgamma(c.phi) - lgamma(g.k + 1)
                   + c.phi * ln(c.phi / (c.phi + c.mu))
                   + g.k   * ln(c.mu   / (c.phi + c.mu))))
             OVER (PARTITION BY c.part, c.week ORDER BY g.k) AS cum
    FROM cell c CROSS JOIN grid g
)
SELECT part, week, coalesce(min(k) FILTER (WHERE cum >= u), 0)::BIGINT AS units
FROM cdf
GROUP BY part, week;
"""


class SafetyStock(BayesDemo):
    name = "safety-stock"
    title = "Sicherheitsbestands-Agent — reorder points for C-parts"
    family = "hier_negbin (F1)"
    draws_table = DRAWS
    intro = (
        "[b]The decision at stake:[/b] a [b]reorder point[/b] and a "
        "[b]Sicherheitsbestand[/b] for every C-part in the catalogue, at a service "
        "level the buyer chooses. Set it too low and the line stops for a €0.42 "
        "washer; set it too high across 48 parts and the working capital is real "
        "money sitting in a bin.\n\n"
        "[b]Why this is not a forecast:[/b] a reorder point is a [i]quantile[/i] of "
        "demand over the lead time, not an average. Most of these parts move a "
        "handful of times a year, so the average is small and the [i]tail[/i] is "
        "what stocks you out — and a point forecast has no tail. The model learns "
        "how much the parts differ from each other and shrinks each thin part "
        "toward its Warengruppe by exactly that much.\n\n"
        "Press [b]r[/b] to run it. Then press [b]w[/b] to change the service level "
        "and watch the answer come back without re-fitting."
    )
    params = (
        Param(
            key="service_level",
            label="Service level",
            default=0.95,
            help="The share of lead times that must be covered. 0.95 is the usual "
                 "starting point; the € consequence of moving it is the last step.",
            minimum=0.50,
            maximum=0.999,
        ),
        Param(
            key="lead_time_weeks",
            label="Lead time (weeks)",
            default=3.0,
            help="How long a replenishment order takes to arrive. Demand over this "
                 "window is what the reorder point has to cover.",
            minimum=0.5,
            maximum=26.0,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        rows = con.sql(
            """
            SELECT substr(part, 1, position('-' IN part) - 1) AS tier,
                   count(DISTINCT part) AS parts,
                   round(avg(units), 2)  AS mean_weekly,
                   round(100.0 * avg(CASE WHEN units = 0 THEN 1 ELSE 0 END), 0) AS pct_zero
            FROM issues GROUP BY tier ORDER BY mean_weekly DESC
            """
        ).fetchall()
        total = con.sql("SELECT count(*), count(DISTINCT part) FROM issues").fetchone()
        weekly = [
            r[0]
            for r in con.sql(
                "SELECT sum(units) FROM issues GROUP BY week ORDER BY week"
            ).fetchall()
        ]
        lines = [
            f"[b]{total[1]} parts[/b] × up to 104 weeks = [b]{total[0]:,} issue records[/b]  "
            f"· catalogue total per week: {sparkline([float(v) for v in weekly])}",
        ]
        for tier, parts, mean_weekly, pct_zero in rows:
            lines.append(
                f"  {tier:<7} {parts:>2} parts · {mean_weekly:>5.2f}/week · "
                f"{pct_zero:>3.0f}% zero weeks  {bar(float(mean_weekly), 10.0, 18)}"
            )
        return "\n".join(lines)

    def build(self, params) -> list[Step]:
        sl = float(params["service_level"])
        lt = float(params["lead_time_weeks"])
        return [
            Step(
                title="Profile the catalogue",
                kind=Kind.PROFILE,
                why=(
                    "Before modelling anything, look at what is there. The share of "
                    "zero weeks is the number that decides whether this is a "
                    "forecasting problem at all: a part that moves in 8 % of weeks "
                    "has no meaningful 'average week', and a normal-distribution "
                    "safety stock built on one is wrong in a direction that causes "
                    "stockouts."
                ),
                sql="""
SELECT substr(part, 1, position('-' IN part) - 1)              AS tier,
       count(DISTINCT part)                                    AS parts,
       round(avg(units), 3)                                    AS mean_weekly,
       round(var_samp(units) / nullif(avg(units), 0), 2)       AS variance_ratio,
       round(100.0 * avg(CASE WHEN units = 0 THEN 1 ELSE 0 END), 1) AS pct_zero_weeks
FROM issues
GROUP BY tier
ORDER BY mean_weekly DESC;
                """,
                verdict=_dispersion_verdict,
            ),
            Step(
                title="Quality gate: enough history?",
                kind=Kind.GATE,
                why=(
                    "A part with almost no history cannot be estimated on its own — "
                    "but that is not a reason to exclude it, it is the reason the "
                    "model is hierarchical. This gate exists to count them, so the "
                    "Entscheidungsvorlage can say which recommendations lean on the "
                    "Warengruppe rather than on the part's own record."
                ),
                sql="""
SELECT count(*) FILTER (WHERE observed_weeks >= 26) = count(*) AS all_parts_have_history,
       count(*)                                                AS parts,
       count(*) FILTER (WHERE observed_weeks < 26)             AS thin_parts,
       min(observed_weeks)                                     AS fewest_weeks
FROM (SELECT part, count(*) AS observed_weeks FROM issues GROUP BY part);
                """,
                verdict=lambda rows: (
                    f"[yellow]{rows[0][2]} of {rows[0][1]} parts[/yellow] have under "
                    "26 weeks of history. They are kept in the fit and pooled toward "
                    "their group — that is the whole point — and they are flagged as "
                    "pooled in the final table."
                    if rows and rows[0][2]
                    else "[green]Every part has enough history to speak for itself.[/green]"
                ),
            ),
            Step(
                title="Fit — one call, one table of draws",
                kind=Kind.FIT,
                silent=True,
                why=(
                    "This is the entire modelling step. `anofox_bayes_fit` is a table "
                    "function: data in, posterior draws out, nothing kept in session "
                    "state. 4 chains × 2000 draws over 48 parts is ~380 000 rows, and "
                    "every question from here on is SQL over that table.\n\n"
                    "Note what is [i]not[/i] configured: no pooling strength, no "
                    "distribution choice, no sampler settings. `tau` — how much the "
                    "parts differ — is a parameter with a posterior, not a dial."
                ),
                sql=f"""
CREATE OR REPLACE TABLE {DRAWS} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT part, units FROM issues),
    'hier_negbin',
    {{'y': 'units',
     'group': 'part',
     'draws': 2000,
     'chains': 4,
     'warmup': 1000,
     'seed': 20260801}}
);
                """,
            ),
            Step(
                title="Is the fit safe to act on?",
                kind=Kind.DIAGNOSE,
                why=(
                    "The fit tells you whether to trust it, on the same table. "
                    "`__status__` is the gate an agent branches on; R-hat and ESS say "
                    "whether the sampler explored the posterior; a single divergent "
                    "transition grades the whole fit `degenerate`, because draws "
                    "around a divergence are not from the posterior.\n\n"
                    "Press [b]d[/b] for the per-parameter breakdown."
                ),
                sql=f"""
SELECT anofox_bayes_is_actionable(param, value)  AS safe_to_act_on,
       anofox_bayes_status_text(param, value)    AS status,
       anofox_bayes_family_text(param, value)    AS family,
       max(CASE WHEN param = '__n_groups__' THEN value END)         AS parts_fitted,
       max(CASE WHEN param = '__n_groups_unready__' THEN value END) AS parts_refused,
       sum(CASE WHEN param = '__divergent__' THEN value END)        AS divergences
FROM {DRAWS};
                """,
                verdict=lambda rows: (
                    "[green]DECISION[/green] — every part was estimated and the "
                    "sampler is clean. The numbers below are actionable."
                    if rows and rows[0][0]
                    else "[yellow]Not actionable as a whole — read the status.[/yellow]"
                ),
            ),
            Step(
                title="How much do the parts actually differ?",
                kind=Kind.DECIDE,
                why=(
                    "`tau` is the spread of part-level demand rates around the "
                    "catalogue level, on the log scale — the number that decides how "
                    "hard a thin part is pulled toward its neighbours. It is estimated, "
                    "with an interval of its own, and that uncertainty propagates into "
                    "every per-part answer below.\n\n"
                    "`phi` is the overdispersion. Large `phi` is the Poisson limit, so "
                    "read `1/phi`: the further from zero, the burstier the demand."
                ),
                sql=f"""
SELECT param,
       round(median(value), 3)                                AS median,
       round(anofox_bayes_credible_lower(value, 0.90), 3)     AS ci_lower,
       round(anofox_bayes_credible_upper(value, 0.90), 3)     AS ci_upper
FROM {DRAWS}
WHERE param IN ('tau', 'phi') AND draw >= 0
GROUP BY param
ORDER BY param;
                """,
                verdict=lambda rows: (
                    "[b]Both are found, not assumed.[/b] A `tau` near zero would mean "
                    "the catalogue is homogeneous and every part should get the same "
                    "answer; a large `phi` would mean the demand is Poisson after all. "
                    "Neither is what this catalogue says."
                ),
            ),
            Step(
                title=f"Lead-time demand → reorder point at {sl:.0%}",
                kind=Kind.DECIDE,
                why=(
                    f"The decision, in one query. Demand over a {lt:g}-week lead time "
                    "is negative binomial with the part's own rate scaled up, and its "
                    "probability mass function is closed form — so the reorder point "
                    "is a running sum over the draws table.\n\n"
                    "Averaging the mass function [i]across draws[/i] is what integrates "
                    "out the catalogue level, the part's own offset, the pooling scale "
                    "and the dispersion all at once. That is the step a plug-in "
                    "estimate at a point cannot do, and it is why a thin part's "
                    "reorder point comes out honest rather than merely confident."
                ),
                sql=f"""
CREATE OR REPLACE TABLE reorder AS
WITH rate AS (
    SELECT group_id AS part, chain, draw, value * {lt} AS lt_rate
    FROM {DRAWS} WHERE param = 'rate' AND draw >= 0
),
disp AS (
    SELECT chain, draw, value AS phi FROM {DRAWS} WHERE param = 'phi' AND draw >= 0
),
pmf AS (
    SELECT r.part, k.k,
           avg(exp(lgamma(k.k + d.phi) - lgamma(d.phi) - lgamma(k.k + 1)
                   + d.phi * ln(d.phi / (d.phi + r.lt_rate))
                   + k.k   * ln(r.lt_rate / (d.phi + r.lt_rate)))) AS p
    FROM rate r JOIN disp d USING (chain, draw)
    CROSS JOIN (SELECT range AS k FROM range(0, 260)) k
    GROUP BY r.part, k.k
),
cdf AS (
    SELECT part, k, sum(p) OVER (PARTITION BY part ORDER BY k) AS cum FROM pmf
)
SELECT c.part,
       min(c.k) FILTER (WHERE c.cum >= {sl})            AS reorder_point,
       round(max(m.mean_lt_demand), 2)                  AS mean_lt_demand,
       min(c.k) FILTER (WHERE c.cum >= {sl})
         - round(max(m.mean_lt_demand), 2)              AS safety_stock
FROM cdf c
JOIN (SELECT part, avg(lt_rate) AS mean_lt_demand FROM rate GROUP BY part) m
  ON m.part = c.part
GROUP BY c.part;

SELECT part, reorder_point, mean_lt_demand, safety_stock
FROM reorder ORDER BY reorder_point DESC LIMIT 12;
                """,
                verdict=lambda rows: (
                    "[b]Sicherheitsbestand = reorder point − expected demand.[/b] It is "
                    "the part of the stock that exists only to absorb the uncertainty, "
                    "and it is bigger, relatively, for the parts that move least — "
                    "which is the opposite of what a percentage-of-average rule gives "
                    "you."
                ),
            ),
            Step(
                title="The service level ↔ working capital trade-off",
                kind=Kind.DECIDE,
                why=(
                    "The question a Leiter Einkauf actually asks: what does the next "
                    "percentage point of service cost? Four service levels, priced at "
                    "each part's unit cost, from the same draws — no re-fit, and the "
                    "curve is monotone by construction rather than by luck."
                ),
                sql=f"""
WITH rate AS (
    SELECT group_id AS part, chain, draw, value * {lt} AS lt_rate
    FROM {DRAWS} WHERE param = 'rate' AND draw >= 0
),
disp AS (SELECT chain, draw, value AS phi FROM {DRAWS} WHERE param = 'phi' AND draw >= 0),
pmf AS (
    SELECT r.part, k.k,
           avg(exp(lgamma(k.k + d.phi) - lgamma(d.phi) - lgamma(k.k + 1)
                   + d.phi * ln(d.phi / (d.phi + r.lt_rate))
                   + k.k   * ln(r.lt_rate / (d.phi + r.lt_rate)))) AS p
    FROM rate r JOIN disp d USING (chain, draw)
    CROSS JOIN (SELECT range AS k FROM range(0, 260)) k
    GROUP BY r.part, k.k
),
cdf AS (SELECT part, k, sum(p) OVER (PARTITION BY part ORDER BY k) AS cum FROM pmf),
levels AS (SELECT unnest([0.90, 0.95, 0.98, 0.99]) AS service_level)
SELECT l.service_level,
       sum(rp.k)                                        AS total_units_on_hand,
       round(sum(rp.k * p.unit_cost), 2)                AS inventory_value_eur
FROM levels l
CROSS JOIN LATERAL (
    SELECT c.part, min(c.k) FILTER (WHERE c.cum >= l.service_level) AS k
    FROM cdf c GROUP BY c.part
) rp
JOIN parts p ON p.part = rp.part
GROUP BY l.service_level
ORDER BY l.service_level;
                """,
                chart=lambda rows: _tradeoff_chart(rows),
                verdict=lambda rows: _tradeoff_verdict(rows, sl),
            ),
            Step(
                title="Where the uncertainty actually is",
                kind=Kind.DECIDE,
                why=(
                    "Not the biggest parts — the [b]least certain[/b] ones. Ranked by "
                    "how wide each interval is [i]relative to the part's own rate[/i], "
                    "which is the comparison that matters when the rates differ by a "
                    "factor of forty.\n\n"
                    "The intermittent tier comes out on top, and that is the honest "
                    "answer for an item nobody has issued in months. A point forecast "
                    "gives those parts a number that looks exactly as authoritative as "
                    "the fast movers' — this is the information it discards."
                ),
                sql=f"""
SELECT substr(group_id, 1, position('-' IN group_id) - 1)  AS tier,
       group_id                                            AS part,
       round(median(value), 3)                             AS rate_median,
       round(anofox_bayes_credible_lower(value, 0.90), 3)  AS rate_lower,
       round(anofox_bayes_credible_upper(value, 0.90), 3)  AS rate_upper,
       round((anofox_bayes_credible_upper(value, 0.90)
              - anofox_bayes_credible_lower(value, 0.90))
             / nullif(median(value), 0), 2)                AS relative_width
FROM {DRAWS}
WHERE param = 'rate' AND draw >= 0
GROUP BY group_id
ORDER BY relative_width DESC
LIMIT 12;
                """,
                chart=lambda rows: _relative_interval_chart(rows),
                verdict=lambda rows: (
                    "[b]The ranking is the point.[/b] These are the parts whose "
                    "reorder points rest most on the Warengruppe rather than on their "
                    "own history — the ones a Materialdisponent should look at before "
                    "signing the list."
                ),
            ),
        ]

    def summary(self, con, results) -> str:
        try:
            row = con.sql(
                """
                SELECT count(*) AS parts,
                       sum(reorder_point) AS units,
                       round(sum(r.reorder_point * p.unit_cost), 2) AS eur,
                       round(sum(r.safety_stock * p.unit_cost), 2)  AS safety_eur
                FROM reorder r JOIN parts p USING (part)
                """
            ).fetchone()
        except duckdb.Error:
            return ""
        if row is None:
            return ""
        parts, units, eur, safety_eur = row
        return (
            f"[b]Entscheidungsvorlage[/b]\n\n"
            f"  {parts} parts · reorder points totalling [b]{units:,} units[/b] "
            f"= [b]€{eur:,.2f}[/b] of stock\n"
            f"  of which [b]€{safety_eur:,.2f}[/b] is Sicherheitsbestand — the part "
            f"that exists purely to absorb uncertainty\n\n"
            "[dim]Every number above carries an interval, and every one was computed "
            "from a single fit. Press [b]w[/b] to move the service level or the lead "
            "time and watch the whole table re-price without the model being touched."
            "[/dim]"
        )


def _dispersion_verdict(rows) -> str:
    """Read the dispersion off the rows rather than asserting it in prose.

    The claim "this demand is overdispersed" is the reason the whole demo uses a
    negative binomial rather than a Poisson, so it has to be *true of the table
    on screen*. An earlier version of this file stated it as a fixed sentence
    and was wrong for two of the four tiers.

    The variance ratio is also noisy at low rates: a tier averaging a quarter of
    a unit a week has a theoretical ratio of about 1.2 and 104 observations to
    estimate it from, so it can land below 1 by chance. Saying that plainly is
    better than picking a fixture where it never happens.
    """
    if not rows:
        return ""
    above = [(r[0], float(r[3])) for r in rows if r[3] is not None and float(r[3]) > 1.0]
    below = [(r[0], float(r[3])) for r in rows if r[3] is not None and float(r[3]) <= 1.0]
    lead = (
        "[b]Read the variance ratio.[/b] For a Poisson process it is exactly 1. "
        "Above 1 is [i]overdispersion[/i] — burstier than Poisson — and it is why "
        "this demo fits a negative binomial: a Poisson reorder point on "
        "overdispersed demand is too tight exactly where the stockouts happen."
    )
    if not below:
        return lead + f"\nAll {len(above)} tiers are above 1."
    names = ", ".join(f"{t} ({v:.2f})" for t, v in below)
    return (
        lead
        + f"\n{len(above)} of {len(rows)} tiers are above 1. {names} sits at or below "
        "it — which is sampling noise rather than a finding: a tier averaging a "
        "fraction of a unit per week has only ~104 observations to estimate a "
        "variance from. The model is not asked to decide this per tier; it "
        "estimates one dispersion for the catalogue and lets the data say how far "
        "from Poisson it is."
    )


def _tradeoff_chart(rows) -> str:
    if not rows:
        return ""
    values = [float(r[2]) for r in rows]
    hi = max(values)
    out = []
    for level, units, eur in rows:
        out.append(
            f"  {float(level):>5.0%}  {bar(float(eur), hi, 26)}  €{float(eur):>9,.2f}  "
            f"({int(units):,} units)"
        )
    return "\n".join(out)


def _tradeoff_verdict(rows, chosen: float) -> str:
    if len(rows) < 2:
        return ""
    by_level = {round(float(r[0]), 4): float(r[2]) for r in rows}
    if 0.95 in by_level and 0.99 in by_level:
        delta = by_level[0.99] - by_level[0.95]
        return (
            f"[b]The last four points of service cost €{delta:,.2f}.[/b] That is the "
            "trade-off made explicit — not an argument about whether 95 % is enough, "
            f"but a price for 99 %. Currently showing reorder points at "
            f"[b]{chosen:.0%}[/b]."
        )
    return ""


def _relative_interval_chart(rows) -> str:
    """Each interval drawn on *its own* part's scale, so widths are comparable.

    A shared axis would be the wrong chart here: these parts differ in rate by a
    factor of forty, so on one axis every slow part collapses into a dot at the
    left edge and the comparison the step is making becomes invisible. Dividing
    by the median puts them all on "multiples of my own rate", which is the
    quantity the ranking is by.
    """
    if not rows:
        return ""
    scaled = []
    for tier, part, median, lower, upper, _ in rows:
        m = float(median) or 1.0
        scaled.append((tier, part, float(lower) / m, 1.0, float(upper) / m))
    hi = max(r[4] for r in scaled)
    out = ["  [dim]90 % interval as a multiple of the part's own rate[/dim]"]
    for tier, part, a, m, b in scaled:
        out.append(
            f"  {part:<12} {interval_bar(a, m, b, 0.0, hi, 32)} "
            f"[dim]×{a:.2f} – ×{b:.2f}[/dim]"
        )
    return "\n".join(out)


DEMO = SafetyStock()


def run() -> int:
    return main(DEMO)
