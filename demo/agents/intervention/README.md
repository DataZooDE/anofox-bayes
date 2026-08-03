# Agent 03 — intervention

> Did the logistics change work — and is there a defensible control group at all?

**Family:** `pooled_gaussian (F3)` · **Run:** `uv run intervention` (add `--headless` to print it)

Logistics interventions — a carrier switch, a depot consolidation, a WMS
rollout — get evaluated by before/after averages contaminated by seasonality,
mix shifts and trend. Millions get spent on rollouts justified by noise, and a
warehouse cannot run an A/B test.

**The most important step in this demo is the one that can say no.** Before any
effect is estimated, the pre-trend check asks whether the control units and the
treated one were moving together *before* the change. If they were not, there is
no counterfactual and no honest effect estimate — and saying so is the
deliverable, not a failure. This demo runs the gate twice: once on a panel where
identification holds, and once on a donor pool where it does not.

**Where `anofox_solve` would come in.** A synthetic-control weighting is a small
quadratic program (weights ≥ 0 summing to 1, minimising pre-period error). This
demo runs the difference-in-differences path, which needs no solver, and says on
screen when the QP path is unavailable rather than implying it ran.

## Run

```sh
cd demo && uv sync
uv run intervention
uv run intervention --headless          # print the whole run instead
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
| 1 | `profile` | Panel balance and missingness |
| 2 | `profile` | The naive answer, and why it is wrong |
| 3 | `gate` | Identification gate — do the controls track the treated unit? |
| 4 | `gate` | The same gate on a pool that fails it |
| 5 | `fit` | Fit — difference-in-differences with depot effects |
| 6 | `diagnose` | Is the fit safe to act on? |
| 7 | `decide` | The effect, with its credible interval |
| 8 | `decide` | Is it big enough to matter? (P(|effect| > €0.20)) |
| 9 | `decide` | Placebo test — the same estimate on a date nothing happened |
| 10 | `decide` | € per year, annualised |

Exactly one step calls `anofox_bayes_fit`. Every `decide` step after it re-reads
the same persisted draws table — which is why `w` returns in milliseconds, and
why the activity log shows the timings side by side.

## What you can change with `w`

- **Practically relevant effect (€/shipment)** (default `0.2`) — Below this the effect is real but not worth a rollout. The model reports P(|effect| > this), which is the number the steering committee actually needs.

## Optional sibling extension

- **`anofox_solve`** — used when a sibling checkout has it built, or when `ANOFOX_SOLVE_EXTENSION` points at it. Without it the demo takes a labelled plain-SQL path and says so in its activity log rather than implying the full method ran.

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
5. The same family is exercised as a sqllogictest in `test/sql/f3_intervention.test`, against
   assertions rather than against a screen.
