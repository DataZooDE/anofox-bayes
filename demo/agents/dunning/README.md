# Agent 05 — dunning

> Which debtors have quietly stopped paying, ranked by expected recoverable euro.

**Family:** `payer_alive (F5)` · **Run:** `uv run dunning` (add `--headless` to print it)

Dunning runs on days-overdue: everyone at 30 days gets letter 1. But a reliable
payer at 35 days needs nothing, while a silently-churning account at 20 days
needs a phone call today. Collections effort is misallocated and bad-debt
provisioning is reactive.

`payer_alive` is a BG/NBD buy-till-you-die model, reframed: the "transaction
process" is payments while the account is still behaving, and "dropout" is a
customer who has quietly stopped paying without ever saying so. `P(alive)` per
debtor is then a closed-form expression over four population parameters — which
means **daily rescoring is SQL over a weekly fit**, not a nightly retrain.

**One API characteristic worth watching.** F5 has no `group` slot. Its four
parameters describe one population, so a segmented portfolio needs one fit per
segment, and this demo loops rather than pretending otherwise. That is a real
property of the family and the pipeline shows it rather than hiding it behind a
helper.

**Where the data shape comes from.** The `(frequency, recency, age)` summary is
the standard BG/NBD input, and the fixture's shape follows the CDNOW dataset that
the BTYD literature and PyMC-Marketing both benchmark against. The rows are
generated here; the inference is real.

## Run

```sh
cd demo && uv sync
uv run dunning
uv run dunning --headless          # print the whole run instead
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
| 1 | `gate` | Event completeness gate |
| 2 | `profile` | The signal, before any model |
| 3 | `fit` | Fit — HANDEL |
| 4 | `setup` | Fit — INDUSTRIE |
| 5 | `setup` | Fit — OEFFENTLICH |
| 6 | `diagnose` | Are the three fits safe to act on? |
| 7 | `decide` | P(alive) — scored against every debtor, no re-fit |
| 8 | `decide` | Does P(alive) actually separate the churned accounts? |
| 9 | `decide` | The daily list — expected loss, tiered at P(alive) < 0.35 |
| 10 | `decide` | Why is this customer on the list? |
| 11 | `decide` | Provisioning table, per segment |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **P(alive) below which to call** (default `0.35`) — Accounts under this get a phone call rather than a letter. Lower means a shorter, higher-precision list.
- **Minimum exposure to act on (€)** (default `2500.0`) — Below this the collections effort costs more than it recovers.

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
5. The same family is exercised as a sqllogictest in `test/sql/f5_payer_alive.test`, against
   assertions rather than against a screen.
