# Agent 07 — freight-audit

> Which carrier invoice lines to dispute, ranked by euro at stake, with the evidence class on every row.

**Family:** `conjugate_anomaly (F7)` · **Run:** `uv run freight-audit` (add `--headless` to print it)

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

## Run

```sh
cd demo && uv sync
uv run freight-audit
uv run freight-audit --headless          # print the whole run instead
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
| 1 | `profile` | Rate-card coverage map |
| 2 | `decide` | Layer 1 — exact: recompute the covered lines |
| 3 | `fit` | Layer 2 — fit a posterior per uncovered lane |
| 4 | `diagnose` | Is the fit safe to act on? |
| 5 | `decide` | What each uncovered lane normally costs |
| 6 | `decide` | Layer 2 — statistical: score each line at the 98.0% tail |
| 7 | `decide` | Layer 3 — pattern: duplicates |
| 8 | `decide` | The dispute list (≥ €25) |
| 9 | `decide` | Recovery projection, by evidence class |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **Statistical flag threshold** (default `0.98`) — A line is flagged when it sits above this quantile of its lane's posterior predictive. Higher = fewer, stronger flags.
- **Minimum € at stake** (default `25.0`) — Drop findings below this. A dispute costs the buyer time, and a list of €3 items trains them to ignore the list.

## Optional sibling extension

- **`anofox_tabular`** — used when a sibling checkout has it built, or when `ANOFOX_TABULAR_EXTENSION` points at it. Without it the demo takes a labelled plain-SQL path and says so in its activity log rather than implying the full method ran.

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
5. The same family is exercised as a sqllogictest in `test/sql/f7_freight_audit.test`, against
   assertions rather than against a screen.
