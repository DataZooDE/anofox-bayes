"""Turning SQL results into something a terminal can show.

The two decoders at the bottom deliberately **wrap the extension's own macros**
rather than reimplementing them. `anofox_bayes_status_text`,
`anofox_bayes_rhat_gate` and the rest are part of the shipped API, and a demo
that computed the same verdicts in Python would be demonstrating Python. If a
macro's threshold changes, these screens change with it.
"""

from __future__ import annotations

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

import duckdb

#: Rows shown before a result is truncated. A demo screen is not a data browser.
MAX_ROWS = 14


def sql_panel(sql: str) -> Syntax:
    """The SQL, verbatim and highlighted.

    `word_wrap` rather than horizontal scrolling: a reader who has to scroll to
    see the end of a `WHERE` clause will not read it, and these demos exist to
    make the SQL read.
    """
    return Syntax(sql.strip(), "sql", theme="monokai", word_wrap=True, background_color="default")


def result_table(columns: list[str], rows: list[tuple], max_rows: int = MAX_ROWS) -> Table:
    """A Rich table of real result rows, truncated with an honest footer.

    The footer says how many rows were dropped. Silent truncation would let a
    demo claim a coverage it does not have — "every segment is listed" when the
    screen shows the first fourteen — which is exactly the failure mode these
    demos are supposed to be arguing against.
    """
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for name in columns:
        table.add_column(name, overflow="fold")
    for row in rows[:max_rows]:
        table.add_row(*(_cell(v) for v in row))
    if len(rows) > max_rows:
        table.add_row(*(["…"] * len(columns)))
        table.caption = f"{len(rows) - max_rows} more rows not shown"
    return table


def _cell(value: object) -> Text:
    """Render one value, with NULL made obvious.

    A NULL in these tables is never noise: it is how a parameter that could not
    be estimated travels, and reading it as a blank would lose the one thing the
    refusal machinery is trying to say.
    """
    if value is None:
        return Text("NULL", style="red")
    if isinstance(value, bool):
        return Text(str(value), style="green" if value else "yellow")
    if isinstance(value, float):
        return Text(_number(value))
    return Text(str(value))


def _number(value: float) -> str:
    """Fixed notation with thousands separators, never scientific.

    `f"{v:,.4g}"` was the first attempt and renders €14 620 as `1.462e+04`, which
    is unreadable in a euro column and actively misleading next to a value that
    happened to stay under the switchover. These tables mix money, counts,
    probabilities and rates, so the rule is: keep four significant figures where
    the number is small, and never leave fixed notation.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:,.4g}"
    if magnitude == 0:
        return "0"
    # Below one, four significant figures needs more decimals than `.4g` gives
    # once the exponent goes negative -- a probability of 0.00042 must not
    # become `0.0004`.
    return f"{value:.6f}".rstrip("0").rstrip(".")


def status_line(con: duckdb.DuckDBPyConnection, draws: str) -> str:
    """The fit's own verdict, read off the table with the shipped macros.

    Returns Rich markup, coloured by whether the fit is actionable. `PARTIAL` is
    its own colour rather than sharing with `REFUSE`: several of these demos
    turn on the difference between "none of this is usable" and "these three
    groups are not, and the rest are".
    """
    row = con.sql(
        f"""
        SELECT anofox_bayes_status_text(param, value)    AS status,
               anofox_bayes_is_actionable(param, value)  AS actionable,
               anofox_bayes_family_text(param, value)    AS family,
               max(CASE WHEN param = '__engine__' THEN value END)            AS engine,
               max(CASE WHEN param = '__n_groups__' THEN value END)          AS n_groups,
               max(CASE WHEN param = '__n_groups_unready__' THEN value END)  AS unready
        FROM {draws}
        """
    ).fetchone()
    if row is None:
        return "[red]no metadata rows on the draws table[/red]"
    status, actionable, family, engine, n_groups, unready = row
    engines = {0: "exact", 1: "laplace", 2: "nuts"}
    engine_name = engines.get(int(engine or 0), "?")

    unready = int(unready or 0)
    n_groups = int(n_groups or 0)
    if actionable:
        colour, verdict = "green", "DECISION"
    elif unready and unready < n_groups:
        colour, verdict = "yellow", "PARTIAL"
    else:
        colour, verdict = "red", "REFUSE"

    detail = f"{unready} of {n_groups} groups unready" if unready else f"{n_groups} groups, all ready"
    return (
        f"[b {colour}]{verdict}[/b {colour}]  ·  status [b]{status}[/b]  ·  "
        f"family [b]{family}[/b] via [b]{engine_name}[/b]  ·  {detail}"
    )


def diagnostics(con: duckdb.DuckDBPyConnection, draws: str) -> tuple[list[str], list[tuple]]:
    """R-hat, ESS and the gate verdict, per parameter.

    Every number here comes from an aggregate the extension ships
    (`anofox_bayes_rhat`, `_ess_bulk`, `_ess_tail`) and every pass/fail from a
    macro it ships (`_rhat_gate`, `_ess_gate`). The thresholds are the shipped
    defaults — 1.01 and 400 — stated here so the screen is self-explaining.

    R-hat is `NULL` under a single chain, and `anofox_bayes_rhat_gate` passes a
    `NULL` deliberately: one chain cannot disagree with itself, and failing it
    would refuse every `exact` fit in the catalog.
    """
    cursor = con.sql(
        rf"""
        SELECT group_id,
               param,
               anofox_bayes_rhat(value, chain, draw)     AS rhat,
               anofox_bayes_ess_bulk(value, chain, draw) AS ess_bulk,
               anofox_bayes_ess_tail(value, chain, draw) AS ess_tail,
               anofox_bayes_rhat_gate(value, chain, draw, 1.01)
                 AND anofox_bayes_ess_gate(value, chain, draw, 400)  AS passes
        FROM {draws}
        WHERE draw >= 0 AND param NOT LIKE '\_\_%' ESCAPE '\'
        GROUP BY group_id, param
        ORDER BY passes, ess_bulk
        """
    )
    return list(cursor.columns), cursor.fetchall()


def divergences(con: duckdb.DuckDBPyConnection, draws: str) -> int | None:
    """Total divergent transitions, or `None` if no sampler ran.

    The distinction matters and is easy to get wrong: an `exact` or `laplace` fit
    has no `__divergent__` rows at all, and reporting that as zero divergences
    would be claiming a clean bill of health from a test that was never run.
    """
    row = con.sql(
        f"SELECT sum(value) FROM {draws} WHERE param = '__divergent__'"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def unready_groups(con: duckdb.DuckDBPyConnection, draws: str) -> list[tuple]:
    """The groups the fit refused individually, with their verdicts.

    Empty for a family that fits one joint design and therefore cannot honestly
    single out a group — which is most of them. `hier_elasticity` is the one
    where this list is the deliverable.
    """
    return con.sql(
        f"""
        SELECT group_id, anofox_bayes_status_name(value) AS verdict
        FROM {draws} WHERE param = '__group_status__'
        ORDER BY group_id
        """
    ).fetchall()
