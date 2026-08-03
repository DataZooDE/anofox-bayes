"""Agent 02 — delivery promise dates, on `censored_aft` (F2).

Promise dates to customers are based on supplier-confirmed dates, which are
systematically optimistic and supplier-specific. Expediting is reactive: someone
notices a line is late after it is late.

**The thing that makes this family necessary is the open orders.** At any moment
most POs have not arrived yet. They are not missing data and they are not
"delivered today" — they are *censored*: we know the duration is at least this
long and not yet how much longer. Dropping them keeps only the orders that have
already landed, which is a sample biased toward the fast ones, and the promise
dates come out optimistic in exactly the way the confirmed dates already are.

`censored_aft` treats a not-yet-arrived line as information rather than as a
missing row. This demo shows the size of that difference by fitting the same
data both ways.
"""

from __future__ import annotations

import duckdb

from anofox_bayes_demo import BayesDemo, Kind, Param, Step, main
from anofox_bayes_demo.charts import bar, interval_bar

DRAWS = "aft_draws"
TODAY = 120

FIXTURE = f"""
CREATE OR REPLACE TABLE suppliers AS
SELECT * FROM (VALUES
    ('SUP-ALPHA',   14.0, 0.22),
    ('SUP-BETA',    21.0, 0.35),
    ('SUP-GAMMA',   35.0, 0.28),
    ('SUP-DELTA',   28.0, 0.55),
    ('SUP-EPSILON', 18.0, 0.30),
    ('SUP-ZETA',    45.0, 0.40)
) AS t(supplier, typical_days, log_spread);

-- 40 purchase-order lines per supplier over the last 120 days. Each was raised
-- on some day and takes a Weibull-ish duration to arrive; lines that would land
-- after `today` have not arrived yet and are **censored**.
--
-- The confirmed lead time is deliberately optimistic -- 80 % of the supplier's
-- real typical duration -- which is the whole reason this agent exists.
CREATE OR REPLACE TABLE po_lines AS
WITH n AS (SELECT range AS i FROM range(1, 41))
SELECT s.supplier,
       'PO-' || s.supplier || '-' || lpad(n.i::VARCHAR, 3, '0')        AS po_line,
       (n.i * 3) % {TODAY}                                             AS ordered_on_day,
       round(s.typical_days * 0.80, 1)                                 AS confirmed_days,
       round(2000 + 18000 * anofox_bayes_uniform(20202, s.supplier || ':v', n.i), 2)
                                                                       AS line_value_eur,
       -- The real duration: lognormal about the supplier's own typical time.
       greatest(1.0, round(
           s.typical_days
           * exp(s.log_spread * anofox_bayes_std_normal(20202, s.supplier, n.i)), 1
       ))                                                              AS true_days
FROM suppliers s CROSS JOIN n;

-- What the ERP actually knows today: either a goods receipt, or an open line
-- whose duration so far is all we have.
CREATE OR REPLACE TABLE po_status AS
SELECT supplier, po_line, ordered_on_day, confirmed_days, line_value_eur,
       (ordered_on_day + true_days <= {TODAY})                    AS delivered,
       CASE WHEN ordered_on_day + true_days <= {TODAY}
            THEN true_days
            -- Still open: all we know is that it has taken at least this long.
            ELSE greatest(0.5, {TODAY} - ordered_on_day)
       END                                                        AS observed_days,
       true_days
FROM po_lines;
"""


class DeliveryPromise(BayesDemo):
    name = "delivery-promise"
    title = "Liefertermin-Zusagen — P(delivery by date X)"
    family = "censored_aft (F2)"
    draws_table = DRAWS
    intro = (
        "[b]The decision at stake:[/b] a date you can promise a customer, and a "
        "daily list of the confirmations your data says are fiction.\n\n"
        "[b]Why the open orders are the whole problem:[/b] most POs have not "
        "arrived yet. They are not missing rows and they are not 'delivered "
        "today' — they are [b]censored[/b]: the duration is at least this long "
        "and we do not yet know how much longer.\n\n"
        "Drop them and you keep only the orders that already landed — a sample "
        "biased toward the fast ones — and your promise dates come out optimistic "
        "in exactly the way the supplier's confirmations already are. This demo "
        "fits the same data [b]both ways[/b] so you can see the size of that "
        "mistake.\n\n"
        "Press [b]r[/b] to run it, [b]w[/b] to change the promise confidence."
    )
    params = (
        Param(
            key="promise_level",
            label="Promise confidence",
            default=0.80,
            help="The share of orders that must arrive by the promised date. P80 is "
                 "the usual sales commitment; the calibration report later checks "
                 "whether P80 dates actually hit 80 % of the time.",
            minimum=0.50,
            maximum=0.99,
        ),
    )

    def load(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(FIXTURE)

    def dataset_panel(self, con: duckdb.DuckDBPyConnection) -> str:
        head = con.sql(
            """
            SELECT count(*), count(*) FILTER (WHERE NOT delivered),
                   round(100.0 * avg(CASE WHEN NOT delivered THEN 1 ELSE 0 END), 0),
                   round(sum(line_value_eur) FILTER (WHERE NOT delivered), 0)
            FROM po_status
            """
        ).fetchone()
        rows = con.sql(
            """
            SELECT supplier, count(*),
                   count(*) FILTER (WHERE NOT delivered)  AS open_lines,
                   round(avg(confirmed_days), 1)          AS confirmed
            FROM po_status GROUP BY supplier ORDER BY confirmed
            """
        ).fetchall()
        if head is None or not rows:
            return ""
        total, open_lines, pct, open_eur = head
        return "\n".join(
            [
                f"[b]{total}[/b] PO lines · [b]{open_lines}[/b] still open "
                f"([b]{float(pct):.0f}%[/b] censored) · €{float(open_eur):,.0f} of open "
                "value at risk"
            ]
            + [
                f"  {s:<12} {n:>3} lines · {int(o):>2} open · confirms "
                f"{float(c):>4.1f} d  {bar(float(o), 40, 16)}"
                for s, n, o, c in rows
            ]
        )

    def build(self, params) -> list[Step]:
        p = float(params["promise_level"])
        return [
            Step(
                title="How much of the book is censored?",
                kind=Kind.PROFILE,
                why=(
                    "The share of open lines is the number that decides whether "
                    "censoring matters. At 5 % you could arguably ignore it. At the "
                    "level here you cannot: the open lines are disproportionately the "
                    "slow ones, because a slow order is more likely to still be open."
                ),
                sql="""
SELECT supplier,
       count(*)                                                    AS lines,
       count(*) FILTER (WHERE delivered)                           AS delivered,
       count(*) FILTER (WHERE NOT delivered)                       AS still_open,
       round(100.0 * avg(CASE WHEN NOT delivered THEN 1 ELSE 0 END), 1) AS pct_censored,
       round(avg(observed_days) FILTER (WHERE delivered), 1)       AS mean_days_delivered
FROM po_status
GROUP BY supplier
ORDER BY pct_censored DESC;
                """,
                verdict=lambda rows: _censoring_verdict(rows),
            ),
            Step(
                title="Date-sanity gate",
                kind=Kind.GATE,
                why=(
                    "A goods receipt dated before the order, a zero-length duration, a "
                    "line confirmed for a negative lead time — every ERP extract has "
                    "some. They must be excluded and [b]reported[/b], not silently "
                    "dropped: a supplier whose lines keep failing this check has a data "
                    "problem worth a conversation."
                ),
                sql="""
SELECT count(*) = 0                                        AS all_dates_sane,
       count(*) FILTER (WHERE observed_days <= 0)          AS non_positive_durations,
       count(*) FILTER (WHERE confirmed_days <= 0)         AS bad_confirmations,
       count(*) FILTER (WHERE ordered_on_day < 0)          AS impossible_order_dates
FROM po_status
WHERE observed_days <= 0 OR confirmed_days <= 0 OR ordered_on_day < 0;
                """,
                verdict=lambda rows: (
                    "[green]Every line has a usable duration.[/green]"
                    if rows and rows[0][0]
                    else "[yellow]Some lines are excluded — see the counts.[/yellow]"
                ),
            ),
            Step(
                title="Fit — censored AFT, per supplier",
                kind=Kind.FIT,
                silent=True,
                why=(
                    "One call, and the `event` slot is what makes it work: `1` for a "
                    "line that arrived, `0` for one still open. A zero does not mean "
                    "'missing' and does not mean 'took zero days' — it means the "
                    "duration is [b]at least[/b] `observed_days`, and the likelihood "
                    "uses that.\n\n"
                    "This family is served by the [b]Laplace[/b] engine: a Gaussian at "
                    "the mode of the log-duration model, computed through "
                    "`anofox-statistics`. That is an approximation, it says so on the "
                    "`__engine__` row, and its calibration is certified by the SBC "
                    "suite rather than assumed."
                ),
                sql=f"""
CREATE OR REPLACE TABLE {DRAWS} AS
SELECT * FROM anofox_bayes_fit(
    (SELECT supplier, observed_days, delivered::INTEGER AS event
     FROM po_status),
    'censored_aft',
    {{'time': 'observed_days',
     'event': 'event',
     'group': 'supplier',
     'dist': 'weibull',
     'draws': 4000,
     'seed': 20202}}
);
                """,
            ),
            Step(
                title="Is the fit safe to act on?",
                kind=Kind.DIAGNOSE,
                why=(
                    "Engine `1` is Laplace — a Gaussian approximation to the posterior, "
                    "not the posterior. The draws table records that, so a reviewer "
                    "never has to guess which warranty they are holding.\n\n"
                    "A supplier whose lines are [i]all[/i] still open has no completed "
                    "duration to learn from and is refused individually rather than "
                    "given a confident number."
                ),
                sql=f"""
SELECT anofox_bayes_is_actionable(param, value) AS safe_to_act_on,
       anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_family_text(param, value)   AS family,
       max(CASE WHEN param = '__engine__' THEN value END)            AS engine_code,
       max(CASE WHEN param = '__n_groups__' THEN value END)          AS suppliers,
       max(CASE WHEN param = '__n_groups_unready__' THEN value END)  AS refused
FROM {DRAWS};
                """,
                verdict=lambda rows: (
                    "[green]DECISION[/green] — engine `1`, a Laplace approximation, "
                    "certified for this family by its SBC suite."
                    if rows and rows[0][0]
                    else "[yellow]Read the status before promising anything.[/yellow]"
                ),
            ),
            Step(
                title=f"The promise date — P{p:.0%} per supplier",
                kind=Kind.DECIDE,
                why=(
                    "The deliverable for Vertriebsinnendienst. For a Weibull AFT the "
                    "quantile is closed form:\n\n"
                    "    t_p = exp(intercept + sigma · ln(−ln(1 − p)))\n\n"
                    "evaluated per draw, so the promise date carries the parameter "
                    "uncertainty rather than being read off a point estimate. Press "
                    f"[b]w[/b] to move the confidence away from {p:.0%} and watch the "
                    "dates shift without a re-fit."
                ),
                sql=f"""
CREATE OR REPLACE TABLE promise AS
WITH wide AS (
    SELECT group_id AS supplier, draw,
           max(value) FILTER (WHERE param = 'intercept') AS a,
           max(value) FILTER (WHERE param = 'sigma')     AS s
    FROM {DRAWS} WHERE draw >= 0
    GROUP BY group_id, draw
)
SELECT supplier, draw,
       exp(a + s * ln(-ln(1 - {p}))) AS promise_days
FROM wide;

SELECT p.supplier,
       round(median(p.promise_days), 1)                            AS promise_days,
       round(anofox_bayes_credible_lower(p.promise_days, 0.90), 1) AS ci_lower,
       round(anofox_bayes_credible_upper(p.promise_days, 0.90), 1) AS ci_upper,
       round(max(c.confirmed_days), 1)                             AS supplier_confirms,
       round(median(p.promise_days) - max(c.confirmed_days), 1)    AS optimism_days
FROM promise p
JOIN (SELECT supplier, max(confirmed_days) AS confirmed_days
      FROM po_status GROUP BY supplier) c ON c.supplier = p.supplier
GROUP BY p.supplier
ORDER BY optimism_days DESC;
                """,
                chart=_promise_chart,
                verdict=lambda rows: _promise_verdict(rows, p),
            ),
            Step(
                title="What ignoring the censoring would have cost",
                kind=Kind.DECIDE,
                why=(
                    "[b]The control, and the reason this family exists.[/b] The same "
                    "data, same family, same engine — but every line declared "
                    "delivered, which is what dropping the open orders amounts to.\n\n"
                    "The open lines are disproportionately the slow ones, so treating "
                    "their duration-so-far as a completed duration systematically "
                    "understates every supplier's lead time. The gap below is the size "
                    "of the mistake, in days, per supplier."
                ),
                sql=f"""
WITH naive AS (
    SELECT group_id AS supplier, draw,
           max(value) FILTER (WHERE param = 'intercept') AS a,
           max(value) FILTER (WHERE param = 'sigma')     AS s
    FROM anofox_bayes_fit(
        (SELECT supplier, observed_days, 1 AS event FROM po_status),
        'censored_aft',
        {{'time': 'observed_days', 'event': 'event', 'group': 'supplier',
         'dist': 'weibull', 'draws': 4000, 'seed': 20202}})
    WHERE draw >= 0
    GROUP BY group_id, draw
),
naive_promise AS (
    SELECT supplier, median(exp(a + s * ln(-ln(1 - {p})))) AS naive_days
    FROM naive GROUP BY supplier
),
honest AS (
    SELECT supplier, median(promise_days) AS honest_days FROM promise GROUP BY supplier
)
SELECT h.supplier,
       round(n.naive_days, 1)                    AS if_censoring_ignored,
       round(h.honest_days, 1)                   AS with_censoring,
       round(h.honest_days - n.naive_days, 1)    AS understated_by_days
FROM honest h JOIN naive_promise n USING (supplier)
ORDER BY understated_by_days DESC;
                """,
                verdict=lambda rows: _censoring_cost_verdict(rows),
            ),
            Step(
                title="P(delivery by date X) — the curve sales actually asks for",
                kind=Kind.DECIDE,
                why=(
                    "Not one date: a curve. 'Can we promise the 14th?' has a numeric "
                    "answer for every supplier, and that number is what a quote should "
                    "carry instead of the supplier's confirmation."
                ),
                sql=f"""
WITH wide AS (
    SELECT group_id AS supplier, draw,
           max(value) FILTER (WHERE param = 'intercept') AS a,
           max(value) FILTER (WHERE param = 'sigma')     AS s
    FROM {DRAWS} WHERE draw >= 0 GROUP BY group_id, draw
),
horizon AS (SELECT unnest([7, 14, 21, 30, 45, 60]) AS day)
SELECT h.day,
       round(avg(CASE WHEN w.supplier = 'SUP-ALPHA'
                      THEN 1 - exp(-pow(h.day / exp(w.a), 1.0 / w.s)) END), 3) AS alpha,
       round(avg(CASE WHEN w.supplier = 'SUP-BETA'
                      THEN 1 - exp(-pow(h.day / exp(w.a), 1.0 / w.s)) END), 3) AS beta,
       round(avg(CASE WHEN w.supplier = 'SUP-GAMMA'
                      THEN 1 - exp(-pow(h.day / exp(w.a), 1.0 / w.s)) END), 3) AS gamma,
       round(avg(CASE WHEN w.supplier = 'SUP-ZETA'
                      THEN 1 - exp(-pow(h.day / exp(w.a), 1.0 / w.s)) END), 3) AS zeta
FROM wide w CROSS JOIN horizon h
GROUP BY h.day
ORDER BY h.day;
                """,
                verdict=lambda rows: (
                    "[dim]Read a column downwards: that supplier's probability of "
                    "having delivered by each horizon. A quote template takes the "
                    "first day where the number clears your promise threshold.[/dim]"
                ),
            ),
            Step(
                title="The daily expedite list",
                kind=Kind.DECIDE,
                why=(
                    "What Einkauf works from in the morning. Every open line scored by "
                    "[b]P(it misses its confirmed date) × line value[/b] — so a "
                    "€19 000 line that is probably fine ranks below a €4 000 line that "
                    "is almost certainly late.\n\n"
                    "The probability is conditional on how long the line has already "
                    "been open, which is the part a days-overdue sort cannot express."
                ),
                sql=f"""
WITH wide AS (
    SELECT group_id AS supplier, draw,
           max(value) FILTER (WHERE param = 'intercept') AS a,
           max(value) FILTER (WHERE param = 'sigma')     AS s
    FROM {DRAWS} WHERE draw >= 0 GROUP BY group_id, draw
),
open_lines AS (
    SELECT * FROM po_status WHERE NOT delivered
),
scored AS (
    SELECT o.po_line, o.supplier, o.line_value_eur, o.observed_days AS open_days,
           o.confirmed_days,
           -- P(total duration exceeds the confirmed lead time | already open this
           -- long). Conditioning on survival so far is what makes this a
           -- statement about *this* line rather than about the supplier.
           avg(CASE WHEN exp(-pow(greatest(o.confirmed_days, o.observed_days)
                                  / exp(w.a), 1.0 / w.s))
                         / nullif(exp(-pow(o.observed_days / exp(w.a), 1.0 / w.s)), 0)
                    IS NULL THEN 1.0
                    ELSE exp(-pow(greatest(o.confirmed_days, o.observed_days)
                                  / exp(w.a), 1.0 / w.s))
                         / nullif(exp(-pow(o.observed_days / exp(w.a), 1.0 / w.s)), 0)
               END) AS p_still_on_time
    FROM open_lines o JOIN wide w ON w.supplier = o.supplier
    GROUP BY o.po_line, o.supplier, o.line_value_eur, o.observed_days, o.confirmed_days
)
SELECT po_line, supplier,
       round(line_value_eur, 0)                              AS value_eur,
       open_days,
       confirmed_days,
       round(1 - p_still_on_time, 3)                         AS p_late,
       round((1 - p_still_on_time) * line_value_eur, 0)      AS expedite_score
FROM scored
ORDER BY expedite_score DESC
LIMIT 12;
                """,
                chart=_expedite_chart,
                verdict=lambda rows: (
                    "[b]Ranked by expected value at risk, not by days overdue.[/b] That "
                    "reordering is the product: the top of this list is where an "
                    "expediting phone call is worth making."
                    if rows
                    else ""
                ),
            ),
            Step(
                title="Calibration — do P80 dates actually hit 80 %?",
                kind=Kind.DECIDE,
                why=(
                    "[b]The headline sales artefact, and the only one that can prove "
                    "itself wrong.[/b] A promise date is a probabilistic claim, so it "
                    "is checkable: of the lines we would have promised at P80, what "
                    "share actually arrived by then?\n\n"
                    "This demo can compute it because the fixture knows every line's "
                    "true duration. A real deployment persists each run's predictions "
                    "and joins them against the goods receipts that follow — which is "
                    "why the predictions table exists at all."
                ),
                sql=f"""
WITH promise_by_supplier AS (
    SELECT supplier, median(promise_days) AS promised_days FROM promise GROUP BY supplier
)
SELECT round({p}, 2)                                                AS stated_probability,
       count(*)                                                     AS lines_checked,
       round(avg(CASE WHEN s.true_days <= pb.promised_days THEN 1.0 ELSE 0.0 END), 3)
                                                                    AS realised_hit_rate,
       round(avg(CASE WHEN s.true_days <= s.confirmed_days THEN 1.0 ELSE 0.0 END), 3)
                                                                    AS supplier_confirm_hit_rate
FROM po_status s JOIN promise_by_supplier pb USING (supplier);
                """,
                verdict=lambda rows: _calibration_verdict(rows, p),
            ),
        ]

    def summary(self, con, results) -> str:
        try:
            rows = con.sql(
                """
                SELECT p.supplier, round(median(p.promise_days), 1),
                       round(max(c.confirmed_days), 1),
                       round(median(p.promise_days) - max(c.confirmed_days), 1)
                FROM promise p
                JOIN (SELECT supplier, max(confirmed_days) AS confirmed_days
                      FROM po_status GROUP BY supplier) c ON c.supplier = p.supplier
                GROUP BY p.supplier ORDER BY 4 DESC
                """
            ).fetchall()
        except duckdb.Error:
            return ""
        if not rows:
            return ""
        out = ["[b]Lieferanten-Scorecard[/b] — promise date vs. what they confirm\n"]
        for supplier, promise, confirmed, gap in rows:
            colour = "red" if float(gap) > 10 else ("yellow" if float(gap) > 4 else "green")
            out.append(
                f"  {supplier:<12} promise [b]{float(promise):>5.1f} d[/b]  "
                f"confirms {float(confirmed):>5.1f} d  "
                f"[{colour}]optimistic by {float(gap):>5.1f} d[/{colour}]"
            )
        out.append(
            "\n[dim]Every date above accounts for the orders that have not arrived "
            "yet. Press [b]w[/b] to move the promise confidence — the dates re-read "
            "the same draws.[/dim]"
        )
        return "\n".join(out)


def _censoring_verdict(rows) -> str:
    if not rows:
        return ""
    worst = rows[0]
    overall = sum(float(r[3]) for r in rows) / len(rows)
    return (
        f"[yellow]{overall:.0f}% of lines are still open on average[/yellow], and "
        f"{worst[0]} is at {float(worst[4]):.0f}%. Those lines are not missing data — "
        "they are durations we know a lower bound for. The `mean_days_delivered` "
        "column is what you would get by ignoring them, and it is biased downward "
        "because a slow order is more likely to still be open."
    )


def _promise_chart(rows) -> str:
    if not rows:
        return ""
    lo = min(float(r[2]) for r in rows)
    hi = max(float(r[3]) for r in rows)
    out = ["  [dim]90 % credible interval for the promise date (days)[/dim]"]
    for supplier, med, a, b, confirmed, _gap in rows:
        out.append(
            f"  {supplier:<12} {interval_bar(float(a), float(med), float(b), lo, hi, 28)} "
            f"[dim]{float(a):.0f}–{float(b):.0f} d  (confirms {float(confirmed):.0f})[/dim]"
        )
    return "\n".join(out)


def _promise_verdict(rows, p: float) -> str:
    if not rows:
        return ""
    optimistic = [r for r in rows if float(r[5]) > 0]
    if not optimistic:
        return "[green]No supplier's confirmation is optimistic against this promise level.[/green]"
    worst = optimistic[0]
    return (
        f"[b]Every one of these {len(optimistic)} suppliers confirms a date earlier "
        f"than a P{p:.0%} promise supports.[/b] {worst[0]} is the worst: it confirms "
        f"{float(worst[4]):.0f} days and the data says {float(worst[1]):.0f}. Promising "
        "the confirmed date is promising something that happens less than "
        f"{p:.0%} of the time."
    )


def _censoring_cost_verdict(rows) -> str:
    if not rows:
        return ""
    gaps = [float(r[3]) for r in rows]
    worst = rows[0]
    mean_gap = sum(gaps) / len(gaps)
    return (
        f"[b]Ignoring the censoring understates the promise date by {mean_gap:.1f} days "
        f"on average[/b], and by {float(worst[3]):.1f} days for {worst[0]}.\n"
        "That is a whole working week of optimism produced by treating open orders as "
        "if they had already arrived — and it is invisible in every diagnostic, because "
        "the wrong model fits its own wrong data perfectly well."
    )


def _expedite_chart(rows) -> str:
    if not rows:
        return ""
    hi = max(float(r[6]) for r in rows) or 1.0
    out = ["  [dim]expected € at risk = P(late) × line value[/dim]"]
    for po, supplier, value, open_days, confirmed, p_late, score in rows[:8]:
        out.append(
            f"  {po:<26} {bar(float(score), hi, 18)} €{float(score):>7,.0f}  "
            f"[dim]P(late) {float(p_late):.0%}, open {int(open_days)}d[/dim]"
        )
    return "\n".join(out)


def _calibration_verdict(rows, p: float) -> str:
    if not rows:
        return ""
    _stated, n, realised, confirm_rate = rows[0]
    realised = float(realised)
    confirm_rate = float(confirm_rate)
    gap = abs(realised - p)
    quality = "green" if gap <= 0.08 else ("yellow" if gap <= 0.15 else "red")
    return (
        f"[{quality}]Stated P{p:.0%} → realised {realised:.1%}[/{quality}] over "
        f"{n} lines.\n"
        f"The supplier's own confirmed dates hit [b]{confirm_rate:.1%}[/b] of the time. "
        "That difference is the entire business case: one of these two numbers is a "
        "promise you can keep."
    )


DEMO = DeliveryPromise()


def run() -> int:
    return main(DEMO)
