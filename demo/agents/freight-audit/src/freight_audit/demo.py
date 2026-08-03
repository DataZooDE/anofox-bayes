"""Agent 07 — freight cost audit, on `conjugate_anomaly` (F7).

Carrier invoices deviate from the contracted rate card: wrong lane rates,
surcharge stacking, duplicate lines, fuel-index misapplication. Manual audit
samples a fraction and the rest leaks.

**The structure of this demo is the point, and it is deliberately not
"everything is Bayesian".** There are three evidence classes, in strict order:

1. **Exact.** Where the rate card covers a lane, the expected charge is
   arithmetic. A mismatch is a fact, not an inference, and it goes on the
   dispute list with zero false positives.
2. **Statistical.** Where the rate card does *not* cover a lane — and it never
   covers everything — there is nothing to compute against, so the question
   becomes "is this line unusual for its lane". That is `conjugate_anomaly`,
   served by the **exact** engine: a closed-form posterior per lane, in
   milliseconds.
3. **Pattern.** Duplicates and structural oddities. `anofox_tabular` does this
   properly with an isolation forest; without it the demo runs a plain-SQL
   duplicate check and says so on screen rather than pretending.

A buyer will not use a dispute list they do not trust, so every row carries
which of the three produced it.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, interval_bar

DRAWS = "lane_posterior"

FIXTURE = """
-- Six months of invoice lines across twelve lane/service cells, plus a rate card
-- that deliberately does not cover all of them. Both are deterministic: the
-- scatter comes from `anofox_bayes_std_normal`, a pure function of its arguments.
CREATE OR REPLACE TABLE lanes AS
SELECT * FROM (VALUES
    ('DE-HAM','FR-PAR','ROAD',  1.42, true),
    ('DE-HAM','NL-RTM','ROAD',  0.98, true),
    ('DE-MUC','IT-MIL','ROAD',  1.15, true),
    ('DE-MUC','AT-VIE','ROAD',  0.87, true),
    ('DE-FRA','ES-BCN','ROAD',  1.63, true),
    ('DE-FRA','PL-WAW','ROAD',  1.09, true),
    ('DE-HAM','UK-LON','SEA',   2.05, true),
    ('DE-BRE','SE-GOT','SEA',   1.78, true),
    -- Not on the rate card. A quarter of the spend, which is typical and is a
    -- finding in its own right.
    ('DE-STR','TR-IST','ROAD',  2.24, false),
    ('DE-DUS','NO-OSL','ROAD',  1.94, false),
    ('DE-HAM','US-NYC','SEA',   3.40, false),
    ('DE-FRA','CN-SHA','AIR',  11.20, false)
) AS t(origin, destination, service, contract_rate_per_kg, on_rate_card);

CREATE OR REPLACE TABLE invoice_lines AS
WITH n AS (SELECT range AS i FROM range(1, 61))
SELECT
    l.origin || '>' || l.destination || '/' || l.service       AS lane,
    l.origin, l.destination, l.service, l.on_rate_card,
    'INV-' || lpad((1000 + n.i)::VARCHAR, 5, '0')              AS invoice_no,
    round(120 + 880 * anofox_bayes_uniform(70707, l.origin || l.destination, n.i), 1) AS weight_kg,
    l.contract_rate_per_kg,
    -- **A correct invoice bills the contracted rate exactly.** The first version
    -- of this fixture put +/-8 % noise on every line, which made a third of the
    -- covered lines exceed contract by construction and turned the "zero false
    -- positives" claim into 159 findings of nothing. A rate card is a contract,
    -- not a forecast.
    --
    -- So: covered lanes bill the contract rate, except for two deliberate error
    -- modes that are the two most common findings in a real audit. Uncovered
    -- lanes have no contract to bill against, so their cost per kg genuinely
    -- varies -- which is what leaves the statistical layer real work to do.
    greatest(0.05, round(
        CASE WHEN l.on_rate_card
             THEN l.contract_rate_per_kg
                  -- Every 17th road line: a wrong tariff band, +35 %.
                  * CASE WHEN n.i % 17 = 0 AND l.service = 'ROAD' THEN 1.35 ELSE 1.0 END
                  -- Every 23rd line: the fuel index applied at last quarter's
                  -- value, +6 %. Small, systematic, and invisible to a sample audit.
                  * CASE WHEN n.i % 23 = 0 THEN 1.06 ELSE 1.0 END
             ELSE l.contract_rate_per_kg
                  * (1.0 + 0.09 * anofox_bayes_std_normal(70707, l.origin || l.service, n.i))
                  -- ...and two genuine outliers on the uncovered lanes, which is
                  -- what the posterior tail is there to find.
                  * CASE WHEN n.i % 29 = 0 THEN 1.55 ELSE 1.0 END
        END, 4
    )) AS billed_rate_per_kg
FROM lanes l CROSS JOIN n;

CREATE OR REPLACE TABLE invoices AS
SELECT *,
       round(weight_kg * billed_rate_per_kg, 2) AS billed_eur,
       CASE WHEN on_rate_card
            THEN round(weight_kg * contract_rate_per_kg, 2) END AS expected_eur
FROM invoice_lines;
"""


class FreightAudit(BayesDemo):
    name = "freight-audit"
    title = "Frachtkosten-Audit — a dispute list your buyer will actually use"
    family = "conjugate_anomaly (F7)"
    draws_table = DRAWS
    wants = ("anofox_tabular",)
    intro = (
        "[b]The decision at stake:[/b] which carrier invoice lines to dispute, "
        "ranked by € at stake. Industry-typical recovery is low single-digit "
        "percent of freight spend — immediate, measurable, and self-funding.\n\n"
        "[b]Why a model at all:[/b] where the rate card covers a lane, the expected "
        "charge is [i]arithmetic[/i] and a mismatch is a fact. But a rate card never "
        "covers everything. For the uncovered quarter of the spend there is nothing "
        "to compute against, so the question becomes 'is this line unusual for its "
        "lane' — and that needs a posterior per lane, which "
        "[b]conjugate_anomaly[/b] gives in closed form, in milliseconds.\n\n"
        "Every row on the final list carries its [b]evidence class[/b]: exact, "
        "statistical, or pattern. A buyer will not act on a list they cannot audit.\n\n"
        "Press [b]r[/b] to run it, [b]w[/b] to move the dispute threshold."
    )
    params = (
        Param(
            key="tail_threshold",
            label="Statistical flag threshold",
            default=0.98,
            help="A line is flagged when it sits above this quantile of its lane's "
                 "posterior predictive. Higher = fewer, stronger flags.",
            minimum=0.80,
            maximum=0.9999,
        ),
        Param(
            key="min_eur",
            label="Minimum € at stake",
            default=25.0,
            help="Drop findings below this. A dispute costs the buyer time, and a "
                 "list of €3 items trains them to ignore the list.",
            minimum=0.0,
            maximum=10000.0,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        head = con.sql(
            """
            SELECT round(sum(billed_eur), 2), count(*),
                   round(100.0 * sum(CASE WHEN on_rate_card THEN billed_eur ELSE 0 END)
                         / sum(billed_eur), 1)
            FROM invoices
            """
        ).fetchone()
        rows = con.sql(
            """
            SELECT service, count(*), round(sum(billed_eur), 2)
            FROM invoices GROUP BY service ORDER BY 3 DESC
            """
        ).fetchall()
        if head is None or not rows:
            return ""
        total, lines, covered = head
        hi = max(float(r[2]) for r in rows)
        out = [
            f"[b]{lines:,} invoice lines[/b] · [b]€{float(total):,.2f}[/b] of freight "
            f"spend · [b]{float(covered):.0f}%[/b] covered by a rate card"
        ]
        for service, n, eur in rows:
            out.append(
                f"  {service:<5} {n:>3} lines · €{float(eur):>10,.2f}  "
                f"{bar(float(eur), hi, 22)}"
            )
        return "\n".join(out)

    def build(self, params) -> list[Step]:
        q = float(params["tail_threshold"])
        min_eur = float(params["min_eur"])
        return [
            Step(
                title="Rate-card coverage map",
                kind=Kind.PROFILE,
                why=(
                    "The first deliverable, and it needs no model at all. 'Your "
                    "contracts do not cover a quarter of your spend' is a finding a "
                    "Frachteneinkauf can act on this week, and it is also the reason "
                    "the rest of this pipeline exists: the uncovered lanes are where "
                    "arithmetic runs out."
                ),
                sql="""
SELECT on_rate_card,
       count(*)                                             AS lines,
       count(DISTINCT lane)                                 AS lanes,
       round(sum(billed_eur), 2)                            AS spend_eur,
       round(100.0 * sum(billed_eur) / sum(sum(billed_eur)) OVER (), 1) AS pct_of_spend
FROM invoices
GROUP BY on_rate_card
ORDER BY on_rate_card DESC;
                """,
                verdict=_coverage_verdict,
            ),
            Step(
                title="Layer 1 — exact: recompute the covered lines",
                kind=Kind.DECIDE,
                why=(
                    "Where the rate card is complete, the expected charge is "
                    "weight × contracted rate. A difference is arithmetic, not "
                    "inference, so these findings have [b]no false positives[/b] and "
                    "need no threshold.\n\n"
                    "This layer runs first on purpose. It is the majority of the "
                    "recoverable money in a real audit, and putting a statistical "
                    "method in front of it would be using a model where a subtraction "
                    "will do."
                ),
                sql="""
CREATE OR REPLACE TABLE exact_findings AS
SELECT lane, invoice_no, weight_kg,
       contract_rate_per_kg, billed_rate_per_kg,
       expected_eur, billed_eur,
       round(billed_eur - expected_eur, 2) AS delta_eur
FROM invoices
WHERE on_rate_card
  AND billed_eur > expected_eur + 0.01;

SELECT lane, invoice_no, weight_kg, contract_rate_per_kg, billed_rate_per_kg, delta_eur
FROM exact_findings ORDER BY delta_eur DESC LIMIT 10;
                """,
                verdict=lambda rows: (
                    f"[green]{len(rows)} shown[/green] — each is an arithmetic mismatch "
                    "against a contracted rate. Evidence class: [b]exact[/b], and the "
                    "dispute letter writes itself."
                    if rows
                    else "[dim]No arithmetic mismatches on the covered lanes.[/dim]"
                ),
            ),
            Step(
                title="Layer 2 — fit a posterior per uncovered lane",
                kind=Kind.FIT,
                silent=True,
                why=(
                    "For the lanes with no contracted rate, the reference has to come "
                    "from the lane's own history. `conjugate_anomaly` gives a Normal "
                    "posterior for each lane's cost per kg in [b]closed form[/b] — the "
                    "`exact` engine, no sampler, no warmup, no divergences to worry "
                    "about.\n\n"
                    "Check the engine in the diagnostics ([b]d[/b]): `0` is exact. That "
                    "is a different warranty from a Laplace or NUTS fit, and the draws "
                    "table records which one you got so a reviewer never has to guess."
                ),
                sql=f"""
CREATE OR REPLACE TABLE {DRAWS} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT lane, billed_rate_per_kg FROM invoices WHERE NOT on_rate_card),
    'conjugate_anomaly',
    {{'value': 'billed_rate_per_kg',
     'group': 'lane',
     'draws': 4000,
     'seed': 70707}}
);
                """,
            ),
            Step(
                title="Is the fit safe to act on?",
                kind=Kind.DIAGNOSE,
                why=(
                    "A lane with one invoice has no estimable variance no matter how "
                    "long anything runs, and the family says so up front rather than "
                    "returning a confident number.\n\n"
                    "R-hat is `NULL` here and that is correct: a single chain cannot "
                    "disagree with itself, and the shipped gate passes a NULL "
                    "deliberately rather than failing every exact fit in the catalog."
                ),
                sql=f"""
SELECT anofox_bayes_is_actionable(param, value) AS safe_to_act_on,
       anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_family_text(param, value)   AS family,
       max(CASE WHEN param = '__engine__' THEN value END)   AS engine_code,
       max(CASE WHEN param = '__n_groups__' THEN value END) AS lanes_fitted
FROM {DRAWS};
                """,
                verdict=lambda rows: (
                    "[green]DECISION[/green] — engine `0` is the closed-form exact "
                    "posterior, so these intervals are the posterior rather than an "
                    "approximation to it."
                    if rows and rows[0][0]
                    else "[yellow]Read the status before using the flags below.[/yellow]"
                ),
            ),
            Step(
                title="What each uncovered lane normally costs",
                kind=Kind.DECIDE,
                why=(
                    "The reference the statistical layer scores against: a lane's "
                    "typical cost per kg, with an interval that says how well the lane "
                    "is pinned down. A lane with few shipments gets a wide interval and "
                    "therefore flags fewer lines — the correct behaviour, and the "
                    "reason this is a posterior rather than a mean and a standard "
                    "deviation."
                ),
                sql=f"""
SELECT group_id                                              AS lane,
       round(median(value), 4)                               AS typical_eur_per_kg,
       round(anofox_bayes_credible_lower(value, 0.90), 4)    AS ci_lower,
       round(anofox_bayes_credible_upper(value, 0.90), 4)    AS ci_upper
FROM {DRAWS}
WHERE param = 'mu' AND draw >= 0
GROUP BY group_id
ORDER BY typical_eur_per_kg DESC;
                """,
                chart=_lane_intervals,
            ),
            Step(
                title=f"Layer 2 — statistical: score each line at the {q:.1%} tail",
                kind=Kind.DECIDE,
                why=(
                    "The posterior predictive for one more shipment on a lane is its "
                    "level plus its own scatter, drawn once per posterior draw so the "
                    "parameter uncertainty propagates rather than being conditioned "
                    f"away. A line above the {q:.1%} point of that is flagged.\n\n"
                    "[b]This is where a threshold lives, and it is the only place.[/b] "
                    "The exact layer needed none. Press [b]w[/b] to move it and watch "
                    "the list change without the model being refitted."
                ),
                sql=f"""
CREATE OR REPLACE TABLE statistical_findings AS
WITH predictive AS (
    SELECT m.group_id AS lane,
           m.value + s.value
             * anofox_bayes_std_normal(70707, m.group_id, m.draw::BIGINT) AS rate_star
    FROM (SELECT group_id, draw, value FROM {DRAWS}
          WHERE param = 'mu' AND draw >= 0) m
    JOIN (SELECT group_id, draw, value FROM {DRAWS}
          WHERE param = 'sigma' AND draw >= 0) s
      USING (group_id, draw)
),
cutoff AS (
    SELECT lane,
           anofox_bayes_service_level_quantile(rate_star, {q}) AS max_normal_rate,
           median(rate_star)                                   AS typical_rate
    FROM predictive GROUP BY lane
)
SELECT i.lane, i.invoice_no, i.weight_kg,
       i.billed_rate_per_kg,
       round(c.typical_rate, 4)                                        AS typical_rate,
       round(i.weight_kg * (i.billed_rate_per_kg - c.typical_rate), 2) AS delta_eur
FROM invoices i JOIN cutoff c ON c.lane = i.lane
WHERE NOT i.on_rate_card
  AND i.billed_rate_per_kg > c.max_normal_rate;

SELECT lane, invoice_no, weight_kg, billed_rate_per_kg, typical_rate, delta_eur
FROM statistical_findings ORDER BY delta_eur DESC LIMIT 10;
                """,
                verdict=lambda rows: (
                    f"[yellow]{len(rows)} shown[/yellow] — flagged because they sit in "
                    "the tail of their own lane's history, not because they broke a "
                    "contract. Evidence class: [b]statistical[/b], and the dispute "
                    "letter has to say so."
                    if rows
                    else "[dim]Nothing in the tail at this threshold. Lower it with "
                         "[b]w[/b] to see the ranking appear.[/dim]"
                ),
            ),
            Step(
                title="Layer 3 — pattern: duplicates",
                kind=Kind.DECIDE,
                why=(
                    "The third class: lines that are individually plausible and "
                    "collectively wrong. A real deployment runs an isolation forest "
                    "over the line features with [b]anofox_tabular[/b]; this demo runs "
                    "a plain-SQL duplicate check when that extension is not built, and "
                    "the activity log at the bottom says which one you got.\n\n"
                    "Saying which is not pedantry — a pattern finding is the weakest of "
                    "the three evidence classes, and the dispute list ranks it last."
                ),
                sql="""
CREATE OR REPLACE TABLE pattern_findings AS
SELECT lane, weight_kg, billed_rate_per_kg,
       count(*)                                    AS occurrences,
       string_agg(invoice_no, ', ')                AS invoices,
       round(sum(billed_eur) - max(billed_eur), 2) AS delta_eur
FROM invoices
GROUP BY lane, weight_kg, billed_rate_per_kg
HAVING count(*) > 1;

SELECT lane, weight_kg, billed_rate_per_kg, occurrences, invoices, delta_eur
FROM pattern_findings ORDER BY delta_eur DESC LIMIT 10;
                """,
                verdict=lambda rows: (
                    f"[yellow]{len(rows)} duplicate group(s).[/yellow]"
                    if rows
                    else "[green]No exact duplicates in this period — which is itself "
                         "worth reporting.[/green]"
                ),
            ),
            Step(
                title=f"The dispute list (≥ €{min_eur:,.0f})",
                kind=Kind.DECIDE,
                why=(
                    "The deliverable. One list, ranked by € at stake, with the evidence "
                    "class on every row so the buyer knows which letters they can send "
                    "today and which need a conversation.\n\n"
                    f"Findings under €{min_eur:,.0f} are dropped — not because they are "
                    "wrong, but because a dispute costs the buyer time and a list "
                    "padded with €3 items trains them to stop reading it."
                ),
                sql=f"""
CREATE OR REPLACE TABLE dispute_list AS
SELECT 'exact' AS evidence, lane, invoice_no AS reference, delta_eur,
       'billed above the contracted rate' AS reason
FROM exact_findings
UNION ALL
SELECT 'statistical', lane, invoice_no, delta_eur,
       'above the {q:.1%} point of this lane''s own history'
FROM statistical_findings
UNION ALL
SELECT 'pattern', lane, invoices, delta_eur,
       occurrences || ' identical lines'
FROM pattern_findings;

SELECT evidence, lane, reference, delta_eur, reason
FROM dispute_list
WHERE delta_eur >= {min_eur}
ORDER BY CASE evidence WHEN 'exact' THEN 0 WHEN 'statistical' THEN 1 ELSE 2 END,
         delta_eur DESC
LIMIT 14;
                """,
                chart=_evidence_mix,
            ),
            Step(
                title="Recovery projection, by evidence class",
                kind=Kind.DECIDE,
                why=(
                    "What to tell management. The three classes do not recover at the "
                    "same rate — an arithmetic mismatch against a signed contract is "
                    "nearly always paid, a tail flag is a negotiation — so projecting "
                    "one blended number across all of them would overstate the "
                    "arithmetic and understate the effort."
                ),
                sql=f"""
SELECT evidence,
       count(*)                       AS findings,
       round(sum(delta_eur), 2)       AS at_stake_eur,
       round(sum(delta_eur) * CASE evidence
               WHEN 'exact'       THEN 0.90
               WHEN 'statistical' THEN 0.45
               ELSE 0.60 END, 2)      AS projected_recovery_eur
FROM dispute_list
WHERE delta_eur >= {min_eur}
GROUP BY evidence
ORDER BY at_stake_eur DESC;
                """,
                verdict=lambda rows: (
                    "[dim]The three recovery rates are illustrative planning "
                    "assumptions, not model output — the one place in this demo where a "
                    "number is asserted rather than computed. A real deployment "
                    "replaces them with the customer's own dispute history, which is "
                    "the P2 feedback loop in the agent brief.[/dim]"
                ),
            ),
        ]

    def summary(self, con, results) -> str:
        try:
            row = con.sql(
                """
                SELECT count(*), sum(delta_eur),
                       coalesce(sum(delta_eur) FILTER (WHERE evidence = 'exact'), 0)
                FROM dispute_list
                """
            ).fetchone()
            spend = con.sql("SELECT sum(billed_eur) FROM invoices").fetchone()
        except duckdb.Error:
            return ""
        if not row or not spend or row[0] is None:
            return ""
        n, total, exact = row
        total = float(total or 0.0)
        pct = 100.0 * total / float(spend[0] or 1.0)
        return (
            f"[b]Dispute list[/b]\n\n"
            f"  {n} findings · [b]€{total:,.2f}[/b] at stake = [b]{pct:.1f}%[/b] of "
            f"audited freight spend\n"
            f"  of which [b]€{float(exact):,.2f}[/b] is arithmetic against a signed "
            f"contract — zero false positives by construction\n\n"
            "[dim]The statistical layer used one closed-form fit; every threshold "
            "change since then re-read the same posterior. Press [b]w[/b] to move "
            "it.[/dim]"
        )


def _coverage_verdict(rows) -> str:
    uncovered = [r for r in rows if not r[0]]
    if not uncovered:
        return "[green]Every lane is on a rate card.[/green]"
    _, _lines, lanes, spend, pct = uncovered[0]
    return (
        f"[yellow]{float(pct):.0f}% of spend[/yellow] (€{float(spend):,.2f} across "
        f"{lanes} lanes) has no contracted rate to check against. That is a finding "
        "before any model runs — and it is exactly the spend the next layers exist for."
    )


def _lane_intervals(rows) -> str:
    if not rows:
        return ""
    lo = min(float(r[2]) for r in rows)
    hi = max(float(r[3]) for r in rows)
    out = ["  [dim]90 % credible interval for the lane's typical €/kg[/dim]"]
    for lane, med, a, b in rows:
        out.append(
            f"  {lane:<22} {interval_bar(float(a), float(med), float(b), lo, hi, 28)} "
            f"[dim]{float(a):.2f} – {float(b):.2f}[/dim]"
        )
    return "\n".join(out)


def _evidence_mix(rows) -> str:
    if not rows:
        return ""
    totals: dict[str, float] = {}
    for evidence, _lane, _ref, delta, _reason in rows:
        totals[evidence] = totals.get(evidence, 0.0) + float(delta)
    hi = max(totals.values()) or 1.0
    colour = {"exact": "green", "statistical": "yellow", "pattern": "cyan"}
    out = ["  [dim]€ at stake on this page, by evidence class[/dim]"]
    for evidence, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        c = colour.get(evidence, "white")
        out.append(f"  [{c}]{evidence:<12}[/{c}] {bar(value, hi, 24)} €{value:>9,.2f}")
    return "\n".join(out)


DEMO = FreightAudit()


def run() -> int:
    return main(DEMO)
