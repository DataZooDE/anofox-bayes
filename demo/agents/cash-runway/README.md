# Agent 04 — cash-runway

> The probability you cover payroll on a named date, and which receivable to chase to move it.

**Family:** `payment_delay (F4)` · **Run:** `uv run cash-runway` (add `--headless` to print it)

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

## Run

```sh
cd demo && uv sync
uv run cash-runway
uv run cash-runway --headless          # print the whole run instead
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
| 1 | `gate` | Reconciliation gate |
| 2 | `profile` | How does each segment actually pay? |
| 3 | `fit` | Fit — Gamma delays, pooled across segments |
| 4 | `diagnose` | Is the fit safe to act on? |
| 5 | `decide` | Each segment's payment habit, with its uncertainty |
| 6 | `decide` | The forward simulation — a cash path per draw |
| 7 | `decide` | P(covered on day 72) |
| 8 | `decide` | The 90-day fan, and where it dips |
| 9 | `decide` | The chase list — which invoice moves the number most |
| 10 | `decide` | The same ledger as a lognormal — how far apart is the tail? |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **Day to test cover on** (default `72.0`) — Days from today. Day 72 is the third payroll — the first date the opening balance and the receivables do not obviously cover, and therefore the only one where the probability is worth computing.
- **Minimum acceptable balance (€)** (default `0.0`) — The covenant floor, or zero for plain solvency. The probability reported is P(balance stays above this).

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
5. The same family is exercised as a sqllogictest in `test/sql/f4_cash_runway.test`, against
   assertions rather than against a screen.
