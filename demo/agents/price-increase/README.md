# Agent 06 — price-increase

> What a list-price move costs in volume and earns in margin, per segment, as a band.

**Family:** `hier_elasticity (F6)` · **Run:** `uv run price-increase` (add `--headless` to print it)

The annual price round is negotiated on anecdote: sales says customers will
leave, management needs margin. The transaction data contains the actual
elasticities per segment, and nobody computes them — let alone with honest
uncertainty.

Two things this demo exists to show, and both are things
`pooled_gaussian` + `random_slopes` cannot do:

**Every elasticity is negative, by construction.** Not "almost always" — the
family parameterises `b_g = -exp(...)`, so a positive draw is impossible rather
than improbable. On a thin segment an unconstrained Gaussian slope routinely
puts real mass above zero, and a price meeting handed an interval saying that
raising the price might sell *more* stops reading the interval.

**A segment whose prices never moved is named, not quietly pooled.** That is the
PARTIAL the Entscheidungsvorlage has to carry: *"keine Aussage möglich, die
Preise waren konstant"* arrives as a `__group_status__` row rather than as a
plausible-looking number. This demo deliberately includes such a segment.

**Where the data shape comes from.** The segment structure and the elasticity
range follow PyMC Labs' *Hierarchical Pricing Elasticity Models* case study and
Juan Camilo Orduz's write-up of it over Kaggle retail scanner data. The rows are
generated here, deterministically; the inference is real.

## Run

```sh
cd demo && uv sync
uv run price-increase
uv run price-increase --headless          # print the whole run instead
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
| 1 | `gate` | Identification gate: did prices actually move? |
| 2 | `profile` | What the raw data looks like |
| 3 | `fit` | Fit — elasticity per segment, sign-constrained |
| 4 | `diagnose` | The model's own verdict — and it is a PARTIAL |
| 5 | `decide` | Which segment was refused, by name |
| 6 | `decide` | The elasticities, with their bands |
| 7 | `decide` | The scenario table — list +5.0% |
| 8 | `decide` | The regret view |
| 9 | `decide` | Ask three moves at once |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **List price move (%)** (default `5.0`) — Applied to every segment. The scenario table shows what it costs in volume and what it earns in contribution margin.

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
5. The same family is exercised as a sqllogictest in `test/sql/f6_price_elasticity.test`, against
   assertions rather than against a screen.
