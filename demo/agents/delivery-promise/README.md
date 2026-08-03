# Agent 02 — delivery-promise

> A delivery date you can promise a customer, and a daily list of the confirmations your data says are fiction.

**Family:** `censored_aft (F2)` · **Run:** `uv run delivery-promise` (add `--headless` to print it)

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

## Run

```sh
cd demo && uv sync
uv run delivery-promise
uv run delivery-promise --headless          # print the whole run instead
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
| 1 | `profile` | How much of the book is censored? |
| 2 | `gate` | Date-sanity gate |
| 3 | `fit` | Fit — censored AFT, per supplier |
| 4 | `diagnose` | Is the fit safe to act on? |
| 5 | `decide` | The promise date — P80% per supplier |
| 6 | `decide` | What ignoring the censoring would have cost |
| 7 | `decide` | P(delivery by date X) — the curve sales actually asks for |
| 8 | `decide` | The daily expedite list |
| 9 | `decide` | Calibration — do P80 dates actually hit 80 %? |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **Promise confidence** (default `0.8`) — The share of orders that must arrive by the promised date. P80 is the usual sales commitment; the calibration report later checks whether P80 dates actually hit 80 % of the time.

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
5. The same family is exercised as a sqllogictest in `test/sql/f2_delivery_promise.test`, against
   assertions rather than against a screen.
