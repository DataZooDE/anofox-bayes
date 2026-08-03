# anofox-bayes demos

Seven Textual TUIs, one per agent concept, each showing what a model family is
*for*. They are built for a viewer with a business background and no SQL — every
panel translates one thing — but the SQL is on screen underneath it, unedited,
because the point of these demos is that the mechanism is legible.

Nothing here is mocked. Each demo opens a real DuckDB, loads the `anofox_bayes`
extension built from this checkout, generates a deterministic fixture and runs
real `anofox_bayes_fit` calls. Every number on screen came out of that.

## 60-second quickstart

```sh
make release -j$(nproc)          # from the repository root, once
cd demo && uv sync
uv run safety-stock
```

**No API key and no network are needed.** Unlike the `anofox-evolve` demos these
are modelled on, nothing here calls an LLM — the whole computation is the
extension and SQL.

In the TUI: `r` runs the pipeline, `↑`/`↓` step through it, `s` shows every
statement at once, `d` opens the diagnostics, and **`w` asks a different
question** — which re-runs the decision steps only and is the thing worth
watching.

Add `--headless` to any of them to print the whole run — prose, SQL and results —
to stdout instead. That is the fastest way to review one, and it is what the test
suite drives.

## The seven

| Demo | The decision at stake | Family |
|---|---|---|
| [`safety-stock`](agents/safety-stock) | Reorder point and Sicherheitsbestand per C-part, at a chosen service level | `hier_negbin` (F1) |
| [`delivery-promise`](agents/delivery-promise) | A delivery date you can promise, and a daily expedite list | `censored_aft` (F2) |
| [`intervention`](agents/intervention) | Did the carrier switch work — and is there a defensible control group at all? | `pooled_gaussian` (F3) |
| [`cash-runway`](agents/cash-runway) | P(we cover payroll on day 72), and which invoice to chase | `payment_delay` (F4) |
| [`dunning`](agents/dunning) | Who has quietly stopped paying, ranked by expected loss | `payer_alive` (F5) |
| [`price-increase`](agents/price-increase) | What +5 % on the list price costs in volume, per segment, as a band | `hier_elasticity` (F6) |
| [`freight-audit`](agents/freight-audit) | Which carrier invoice lines to dispute, with the evidence class on each | `conjugate_anomaly` (F7) |

`varying_variance_gaussian` (F8) appears inside `cash-runway`, where the demo
notes that a ledger whose segments differ in *spread* rather than in *level*
belongs to it rather than to F4.

## What every demo is built to show

**One fit, then SQL.** Each pipeline has exactly one `anofox_bayes_fit` step.
Everything after it re-reads the same persisted draws table, and the activity log
timestamps every statement — so pressing `w` and watching the answer come back in
milliseconds after a fit that took seconds is the argument, not a sentence
claiming it. The test suite asserts this rather than trusting it.

**Refusal is a result.** Four of the seven contain a step that says no:
`intervention` refuses a donor pool with no parallel pre-trend, `price-increase`
names the segment whose prices never moved, `dunning` quarantines a segment with
too little churn to model, `cash-runway` refuses a ledger that does not
reconcile. Each is a deliverable a consultant would bill for, and each is reached
from a `__status__` or `__group_status__` row rather than from an exception.

**The engine is on the table.** `exact`, `laplace` and `nuts` look identical in
SQL and do not carry the same warranty. Every demo's diagnostics panel reads
`__engine__` and says which one ran.

## Where the data comes from

The fixtures are generated in SQL, deterministically, using
`anofox_bayes_uniform` and `anofox_bayes_std_normal` — pure functions of
`(seed, key, draw)`. No `setseed()`, no `random()`: the same rows on every
machine, and a test asserts it. Several fixtures draw from the family's own
likelihood by inverse CDF, so the data really is from the model the fit inverts.

The *shapes* follow published work, cited in each demo's module docstring:

- **safety-stock** — the fast/medium/slow/intermittent mix used by the
  `anofox-evolve` replenishment pilot, itself following the intermittent-demand
  literature PyMC Labs writes about under the TSB model.
- **price-increase** — PyMC Labs' *Hierarchical Pricing Elasticity Models* case
  study and Juan Camilo Orduz's write-up of it over Kaggle retail scanner data.
- **dunning** — the `(frequency, recency, age)` summary that BG/NBD reads, in the
  shape of the CDNOW dataset that the BTYD literature and PyMC-Marketing both
  benchmark against.

The rows are synthetic. The inference is real.

## Layout

```
demo/
├── lib/          anofox_bayes_demo — the shared Textual shell
│                   duck.py    extension discovery and loading
│                   steps.py   Step / Pipeline: a demo is an ordered list of SQL
│                   demo.py    what a demo has to provide
│                   app.py     the screen, the modals, headless mode
│                   charts.py  sparkline, bar, histogram, interval_bar, fan
│                   format.py  SQL highlighting and the diagnostics decoders
├── agents/       one uv workspace member per demo
└── tests/        every pipeline, run against the real extension
```

Each demo is a real package with its own entry point, so `uv run <name>` works
for all seven after a single `uv sync`.

## Tests

```sh
cd demo && uv run pytest
```

42 tests, and they assert the claims rather than the plumbing: every step
executes, there is exactly one fit, the questions are cheaper than fitting, the
family the header advertises is the family the draws table reports, every what-if
knob actually moves an answer, and every fixture is reproducible. They skip
loudly — not fail — when the extension has not been built.

## Optional sibling extensions

Two demos can use more than `anofox_bayes` when it is available:

- `intervention` would use **`anofox_solve`** for synthetic-control weights (a
  small QP). Without it, the difference-in-differences path runs, which needs no
  solver.
- `freight-audit` would use **`anofox_tabular`** for an isolation forest over
  invoice lines. Without it, a plain-SQL duplicate check runs.

Both are found automatically in a sibling checkout, or via
`ANOFOX_SOLVE_EXTENSION` / `ANOFOX_TABULAR_EXTENSION`. When one is missing the
demo says so in its activity log and takes the labelled fallback — it does not
quietly imply the full method ran.
