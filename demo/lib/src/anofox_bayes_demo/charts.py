"""Unicode charts. No plotting library, because a terminal is the medium.

`sparkline`, `bar` and `multi_sparkline` are carried over from
`anofox-evolve/demo/*/charts.py`, where they are byte-identical across all nine
pilots. The three below them are new, and they are new because a posterior is not
a number: the thing a reader has to *see* here is a width, not a height.
"""

from __future__ import annotations

_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_BLOCKS[int((v - lo) / span * (len(_BLOCKS) - 1))] for v in values)


def bar(value: float, max_value: float, width: int = 20, fill: str = "█", empty: str = "·") -> str:
    if max_value <= 0:
        filled = 0
    else:
        filled = max(0, min(width, round(value / max_value * width)))
    return fill * filled + empty * (width - filled)


_STATUS_COLOR = {"ok": "green", "warn": "yellow", "bad": "red", "muted": "dim"}


def multi_sparkline(points: list[tuple[float | None, str]]) -> str:
    """Like `sparkline`, but each point carries a status colour.

    A `None` value renders as a fixed marker rather than a bucketed bar: there is
    no number to bucket, and drawing a zero-height bar would read as "small"
    rather than "absent".
    """
    numeric = [v for v, _ in points if v is not None]
    if numeric:
        lo, hi = min(numeric), max(numeric)
        span = (hi - lo) or 1.0
    else:
        lo, span = 0.0, 1.0
    out = []
    for value, status in points:
        colour = _STATUS_COLOR.get(status, "dim")
        if value is None:
            out.append(f"[{colour}]✗[/{colour}]")
            continue
        out.append(f"[{colour}]{_BLOCKS[int((value - lo) / span * (len(_BLOCKS) - 1))]}[/{colour}]")
    return "".join(out)


def histogram(values: list[float], bins: int = 24, width: int | None = None) -> str:
    """A posterior's shape, as one line of blocks.

    Used where the *form* of a posterior matters — bimodality, a pile-up against
    a bound — which a median and an interval both hide. `hier_elasticity` on a
    product whose volume rises with price is the case this was written for: the
    interval says "near zero" and the histogram says "against the wall".
    """
    if not values:
        return ""
    bins = width or bins
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / span * bins))
        counts[idx] += 1
    peak = max(counts) or 1
    return "".join(_BLOCKS[round(c / peak * (len(_BLOCKS) - 1))] for c in counts)


def interval_bar(
    lower: float,
    median: float,
    upper: float,
    lo: float,
    hi: float,
    width: int = 30,
) -> str:
    """A credible interval positioned on a shared axis: `····├──●───┤·····`

    **The endpoints are `├`/`┤` and not `[`/`]`, and that is not cosmetic.**
    Textual and Rich both read `[...]` as markup, so a bar drawn with square
    brackets turns `[──●───]` into an unknown tag: the surrounding `[dim]…[/dim]`
    then fails to balance and the whole panel raises `MarkupError`. It crashed a
    worker mid-run and was invisible to every headless test, because headless
    prints through a plain console that never parses the markup.

    **The most important primitive in this package.** Every number these demos
    put in front of a decision-maker is an interval, and a table of three columns
    per row makes that fact easy to skim past. Drawn on a *shared* `[lo, hi]`
    axis, one glance says which segments overlap and which do not — which is the
    actual question a price round or a service-level review is asking.

    Degenerate inputs render as a dotted line rather than raising: a refused
    group's draws are NULL, and it still has a row in the table.
    """
    span = (hi - lo) or 1.0

    def pos(v: float) -> int:
        return max(0, min(width - 1, round((v - lo) / span * (width - 1))))

    if not all(map(_finite, (lower, median, upper))):
        return "·" * width

    a, m, b = pos(lower), pos(median), pos(upper)
    cells = ["·"] * width
    for i in range(a, b + 1):
        cells[i] = "─"
    cells[a] = "├"
    cells[b] = "┤"
    cells[m] = "●"
    return "".join(cells)


def fan(rows: list[tuple[float, float, float, float]], width: int | None = None) -> list[str]:
    """A quantile fan over time, as three stacked sparklines.

    `rows` is `(x, low, median, high)` in x order. Returned as three lines so a
    caller can label them; drawn on one shared scale so the three are
    comparable, which is the whole point of a fan and is what a per-line
    autoscale would destroy.
    """
    if not rows:
        return ["", "", ""]
    if width and len(rows) > width:
        # Even thinning rather than truncation: a cash path that showed only its
        # first 60 days would be a different chart, not a smaller one.
        stride = len(rows) / width
        rows = [rows[min(len(rows) - 1, int(i * stride))] for i in range(width)]
    flat = [v for _, lo, mid, hi in rows for v in (lo, mid, hi)]
    lo_all, hi_all = min(flat), max(flat)
    span = (hi_all - lo_all) or 1.0

    def line(idx: int) -> str:
        return "".join(
            _BLOCKS[int((r[idx] - lo_all) / span * (len(_BLOCKS) - 1))] for r in rows
        )

    return [line(3), line(2), line(1)]


def _finite(v: object) -> bool:
    return isinstance(v, (int, float)) and v == v and abs(float(v)) != float("inf")
