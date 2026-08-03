# Agent 01 — safety-stock

> Reorder points and Sicherheitsbestand for a C-parts catalogue, at a service level the buyer chooses.

**Family:** `hier_negbin (F1)` · **Run:** `uv run safety-stock` (add `--headless` to print it)

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

## Run

```sh
cd demo && uv sync
uv run safety-stock
uv run safety-stock --headless          # print the whole run instead
```

Requires only that the extension is built — `make release -j$(nproc)` from the
repository root. No API key, no network.

## Controls

| Key | Does |
|---|---|
| `r` | Run the pipeline |
| `↑` / `↓` | Step through it |
| `enter` | Re-run the selected step |
| `s` | Every statement this demo runs, in one scrollable page |
| `d` | Diagnostics — R̂, ESS, `__status__`, divergences, refused groups |
| `w` | **Ask a different question** — re-runs the decision steps only |
| `q` | Quit |

## The pipeline

| # | Kind | Step |
|---|---|---|
| 1 | `profile` | Profile the catalogue |
| 2 | `gate` | Quality gate: enough history? |
| 3 | `fit` | Fit — one call, one table of draws |
| 4 | `diagnose` | Is the fit safe to act on? |
| 5 | `decide` | How much do the parts actually differ? |
| 6 | `decide` | Lead-time demand → reorder point at 95% |
| 7 | `decide` | The service level ↔ working capital trade-off |
| 8 | `decide` | Where the uncertainty actually is |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **Service level** (default `0.95`) — The share of lead times that must be covered. 0.95 is the usual starting point; the € consequence of moving it is the last step.
- **Lead time (weeks)** (default `3.0`) — How long a replenishment order takes to arrive. Demand over this window is what the reorder point has to cover.

## Reading the screen

- **📌 The decision at stake** — the business problem, and what is about to happen.
- **📈 The data** — the fixture, summarised, with a chart.
- **🧭 Pipeline** — the steps, their kind, and how long each took.
- **🔬 Selected step** — why the step exists, its SQL verbatim, its real result rows,
  and a plain-language verdict.
- **🏁 The answer** — the decision, as a table someone could act on.
- **📜 Activity log** — every statement with its timing. This is where the
  "no re-fit" claim is checkable rather than asserted.

## What it proves

1. The fit is one SQL call and the posterior is an ordinary table — see the `fit`
   step's SQL.
2. Every question after it is SQL over that table, and the timings in the
   activity log show the difference.
3. The fit reports its own trustworthiness: `__status__`, `__engine__`, R̂ and ESS
   are read with the extension's own macros, not recomputed here. Press `d`.
4. The fixture is deterministic — built from `anofox_bayes_uniform` /
   `anofox_bayes_std_normal`, which are pure functions of `(seed, key, draw)`.
   `demo/tests/test_demos.py` asserts it.
5. The same family is exercised as a sqllogictest in `test/sql/f1_hier_negbin.test`, against
   assertions rather than against a screen.
