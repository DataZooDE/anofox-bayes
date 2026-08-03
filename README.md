<h1 align="center">Anofox Bayes</h1>
<p align="center"><strong>Bayesian inference for enterprise decision models, inside DuckDB</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSL%201.1-blue.svg" alt="License: BSL 1.1"></a>
  <a href="https://duckdb.org"><img src="https://img.shields.io/badge/DuckDB-1.4.5%20LTS%20%7C%201.5.5-green.svg" alt="DuckDB"></a>
  <img src="https://img.shields.io/badge/draws%20schema-v1-informational.svg" alt="Draws schema v1">
  <img src="https://img.shields.io/badge/status-v0.1%20early-orange.svg" alt="Status">
</p>

Fit a model with one SQL call, get back a table of posterior samples, and answer
every downstream question — intervals, probabilities, service levels, what-ifs — in
plain SQL. No Python, no sampler sidecar.

```sql
INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';   -- start duckdb with -unsigned
LOAD anofox_bayes;

CREATE TABLE sales(region VARCHAR, units DOUBLE);
INSERT INTO sales VALUES
  ('north',102),('north',98),('north',105),('north',99),('north',101),
  ('south',54), ('south',49),('south',52), ('south',51),('south',48);

SELECT group_id AS region,
       round(median(value), 1)                            AS typical_units,
       round(anofox_bayes_credible_lower(value, 0.95), 1) AS lo,
       round(anofox_bayes_credible_upper(value, 0.95), 1) AS hi
FROM anofox_bayes_fit(
       (SELECT region, units FROM sales),
       'conjugate_anomaly',
       {'value': 'units', 'group': 'region'})
WHERE param = 'mu' AND draw >= 0
GROUP BY region ORDER BY region;
```

```
┌─────────┬───────────────┬───────┬───────┐
│ region  │ typical_units │  lo   │  hi   │
├─────────┼───────────────┼───────┼───────┤
│ north   │         101.1 │  97.2 │ 104.3 │
│ south   │          50.8 │  47.6 │  54.0 │
└─────────┴───────────────┴───────┴───────┘
```

A 95 % credible interval per region, from ten rows of data and one query.

**→ [User Guide](docs/GUIDE.md)** — task-oriented: get your table into shape, find an
anomaly, measure an intervention, set a service level, check the fit.
**→ [Theory](docs/THEORY.md)** — what the numbers mean, written for people who have
never done Bayesian statistics.
**→ [Evaluating it?](#for-reviewers)** — models, priors and validation, directly.

---

## Why you might want this

Ordinary analytics gives you an estimate. Decisions need a **range**:

| The question you actually have | What it needs |
|---|---|
| "Stock enough to serve 95 % of weeks" | a quantile |
| "Was the promotion worth its €5/store cost?" | `P(effect > 5)` |
| "Is this carrier overbilling us?" | `P(rate > contracted rate)` |

None of those come out of a point estimate. All of them are one SQL aggregate away
once you have the posterior — which is what this returns.

## What it is not

**Not a general-purpose probabilistic programming language.** You choose from a
catalog of model families and tune documented options; you cannot write your own
likelihood. That restriction is what lets each family ship with fixed parameterisation
decisions, a validated config schema and its own calibration suite.

Today the catalog is eight families — enough for anomaly detection, intervention
measurement, censored durations, safety stock across a thin catalogue, payment-delay
tails, customer-churn scoring, price elasticity and per-segment service levels.
See [the roadmap](#roadmap).

## Install

`anofox-bayes` is BSL-licensed, so it is served from the DataZoo channel rather than
the DuckDB community repository. Binaries are published for DuckDB **v1.4.5 LTS** and
**v1.5.5**, on linux/macOS/Windows (amd64 + arm64) and WASM.

```sql
INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';
LOAD anofox_bayes;
```

The binaries are **unsigned**, so start DuckDB with `-unsigned` (CLI) or
`allow_unsigned_extensions` (client libraries).

### From source

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-bayes.git
cd anofox-bayes
export VCPKG_TOOLCHAIN_PATH=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

make release -j$(nproc)      # the first build compiles DuckDB: 30-60 minutes
./build/release/duckdb       # a shell with the extension already loaded
```

Needs a stable Rust toolchain, CMake 3.15+, and a C++ compiler. Full development
setup in [CONTRIBUTING.md](CONTRIBUTING.md).

## See it work

Seven Textual TUIs, one per model family, each stepping through a real pipeline
with the SQL on screen — reorder points, delivery promises, causal effects, cash
runway, dunning, a price round, a freight audit.

```sh
make release -j$(nproc)
cd demo && uv sync && uv run safety-stock
```

No API key and no network: the whole computation is this extension and SQL. Add
`--headless` to print a run instead of opening the TUI. See
[`demo/README.md`](demo/README.md).

## The catalog

| Family | Use it for | Parameters |
|---|---|---|
| `conjugate_anomaly` | A level or rate per group; anomaly detection | `mu`, `sigma` (Normal) or `lambda` (Poisson) |
| `pooled_gaussian` | Effect measurement; diff-in-diff, interrupted time series | `intercept`, `beta[…]`, `sigma`, per-group effects |
| `censored_aft` | Time until something happens, when some of it has not happened yet — delivery promises, time-to-pay | `intercept`, `beta[…]`, `sigma` (accelerated failure time) |
| `hier_negbin` | How much of this part will be wanted? Safety stock and reorder points across a catalogue where most items are thin | `intercept`, `tau`, `phi` (population); `u`, `rate` per group |
| `payment_delay` | When will this invoice actually be paid? Cash runway, covenant cover, the right tail of a payment habit | `intercept`, `tau`, `shape`/`sigma` (population); `u`, `mu` per segment |
| `payer_alive` | Has this customer churned? Collections, dunning, retention | `r`, `alpha`, `a`, `b` (population level; `P(alive)` per customer is SQL over them) |
| `hier_elasticity` | What does a price rise cost in volume? A per-segment elasticity that is negative by construction, on a count response | `intercept`, `elasticity`, `tau`, `tau_level`, `phi`/`shape`; `group_effect`, `group_elasticity` per segment |
| `varying_variance_gaussian` | A *spread* per group: service levels, buffers, "which segments are unpredictable" | `intercept`, `beta[…]`, `pool_scale`, `sigma_pop`, `sigma_spread`, plus `group_effect` and `sigma` per group |

Rule of thumb: **one number per group → `conjugate_anomaly`; a response explained by
predictors → `pooled_gaussian`; a duration where some cases have not finished →
`censored_aft`; a duration that has finished, and it is the tail you care about →
`payment_delay`; a repeat-purchase history and a churn question → `payer_alive`; a
price move and a volume response → `hier_elasticity`; a question about a group's
*spread* rather than its level → `varying_variance_gaussian`.**

`pooled_gaussian` also does **random slopes**: each group gets its own coefficient on a
predictor, pooled toward the population value rather than estimated in isolation. That
is one way to get a per-store price elasticity, and where the response is
well-populated and the sign is not in doubt it is the faster one. `hier_elasticity` is
the other: it constrains every segment's elasticity to be negative and puts a count
likelihood under it, which is what a thin segment in a price round needs.

Three engines: `exact` (closed-form, the default), `laplace` (a Gaussian at the mode)
and `nuts` (the No-U-Turn Sampler, via [`nuts-rs`](https://github.com/pymc-devs/nuts-rs)).
Switching between them changes no caller SQL.

The BRD's F1–F7 grid is complete. What remains a **decided non-goal** is an
elasticity family for a *Gaussian* response: random slopes above already are that, and
a second family id for the same mathematics would split `model_id`, the cache and the
calibration evidence to save a `log()`. `hier_elasticity` exists alongside it rather
than replacing it, because a sign constraint and a count likelihood are a different
model rather than the same one renamed. See [the roadmap](docs/ROADMAP.md) §3.4.

There is deliberately **no `predict` function** — posterior prediction is a join over
the draws table, which is both simpler and faster. See
[the Guide](docs/GUIDE.md#ask-a-what-if-without-re-fitting).

## How it works, in three properties

**Posteriors are tables.** `anofox_bayes_fit` is a pure table-in/table-out function.
It materialises nothing and keeps no session state; you persist the draws with
`CREATE TABLE draws AS SELECT * FROM anofox_bayes_fit(...)`. Everything downstream is
SQL over that table, and a second question costs no second fit.

**Refusal is a first-class outcome.** A fit that cannot answer says so, inside the
same table, on a `__status__` row. Parameters that cannot be estimated get `NULL`
draws rather than a plausible-looking number.

**Reproducible by construction.** `model_id` is a hash of the family, the config, the
data and the seed — so the same request always yields the same numbers, and an
auditor can re-run a recommendation and check it. Draws are byte-identical across
thread counts.

## Documentation

| | |
|---|---|
| **[User Guide](docs/GUIDE.md)** | Start here. Tasks, recipes, common mistakes. |
| **[Theory](docs/THEORY.md)** | What a posterior is, the models, the priors, the diagnostics. No prior statistics assumed. |
| [API Reference](docs/API_REFERENCE.md) | Every function, every config slot. |
| [Draws Contract](docs/DRAWS_CONTRACT.md) | The output schema, versioned. |
| [Scalability](docs/SCALABILITY.md) | Measured runtime and memory; known limits. |
| [Releasing](docs/RELEASING.md) | How self-distribution to `get.erpl.io` works, and how to cut a release. |
| [BRD](docs/BRD.md) · [HLD](docs/HLD.md) | Product and architecture rationale. |

Runnable examples for every feature live in [`test/sql/`](test/sql/) — they are the
test suite, so they cannot drift from the implementation.

## Roadmap

| | |
|---|---|
| **now** | Eight families — the BRD's F1–F7 grid complete, plus `varying_variance_gaussian`. All three engines: exact, Laplace, NUTS. Diagnostics, SBC and PyMC parity, prior-predictive checks, deterministic predictive draws |
| next | Streaming sufficient statistics; `anofox_scenario` catalog registration |
| later | Censored durations under a hierarchical delay model; a second predictive surface |

Breaking changes are expected at v0.x. Use the
[issues page](https://github.com/DataZooDE/anofox-bayes/issues) to report a bug or ask
for a family.

## For reviewers

Deciding whether to trust it? These are the direct links, so you do not have to read
linearly:

| Question | Where |
|---|---|
| What model is actually fitted? | [Theory §4](docs/THEORY.md#4-the-two-shipped-families) (equations) · [API §2](docs/API_REFERENCE.md) (slots) |
| What are the default priors, and why? | [Theory §3](docs/THEORY.md#3-priors-and-why-the-defaults-look-the-way-they-do) |
| Exact or approximate? | [Theory §5](docs/THEORY.md#5-engines) — and where the approximation is admissible, measured |
| How is it validated? | [Theory §8](docs/THEORY.md#8-how-we-know-it-is-right) · [`validation/`](validation/) for the PyMC suite |
| What does it refuse to answer? | [Theory §7](docs/THEORY.md#7-refusal) |
| Runtime and memory | [Scalability](docs/SCALABILITY.md) — measured, with known gaps |

The mathematics also lives beside the code, in the module documentation of
[`crates/anofox-bayes-core/src/catalog/`](crates/anofox-bayes-core/src/catalog/).

## Validation

Every family is checked three ways, because each catches what the others cannot:
closed-form unit tests, **simulation-based calibration** (does a 90 % interval contain
the truth 90 % of the time?), and **parity against a pinned PyMC reference** in CI.
[Theory §8](docs/THEORY.md#8-how-we-know-it-is-right) explains why all three are
needed — the PyMC suite has already caught a real over-confidence bug that calibration
structurally could not see.

## Telemetry

Anonymous usage counts, on by default, no data values ever transmitted. Opt out with:

```sql
SET anofox_telemetry_enabled = false;
```

Details in [TELEMETRY.md](TELEMETRY.md).

## License

[Business Source License 1.1](LICENSE). Production use is permitted; offering the
Licensed Work to third parties on a hosted or embedded basis is not. Converts to
MPL 2.0 five years after publication. Licensor: DataZoo GmbH.

Commercial licensing: [info@data-zoo.de](mailto:info@data-zoo.de).

## Citation

```bibtex
@software{anofox_bayes,
  title  = {anofox-bayes: Bayesian inference for enterprise decision models in DuckDB},
  author = {{DataZoo GmbH}},
  year   = {2026},
  url    = {https://github.com/DataZooDE/anofox-bayes}
}
```
