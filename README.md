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

> **To run that today you have to build from source** — there are no published
> binaries yet, and the first build compiles DuckDB, so budget 30–60 minutes.
> [Details below](#install).

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

Today the catalog is two families — enough for anomaly detection and intervention
measurement. See [the roadmap](#roadmap).

## Install

> **No binaries are published yet.** The release pipeline and the S3 channel are
> configured; publishing is blocked on one IAM trust-policy entry — see
> [docs/RELEASING.md](docs/RELEASING.md). Build from source until then.

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-bayes.git
cd anofox-bayes
export VCPKG_TOOLCHAIN_PATH=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

make release -j$(nproc)      # the first build compiles DuckDB: 30-60 minutes
./build/release/duckdb       # a shell with the extension already loaded
```

Needs a stable Rust toolchain, CMake 3.15+, and a C++ compiler. Full development
setup in [CONTRIBUTING.md](CONTRIBUTING.md).

Once published it will install from the DataZoo channel — `anofox-bayes` is
BSL-licensed, so it will not go to the DuckDB community repository:

```sql
INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';   -- not live yet
```

## The catalog

| Family | Use it for | Parameters |
|---|---|---|
| `conjugate_anomaly` | A level or rate per group; anomaly detection | `mu`, `sigma` (Normal) or `lambda` (Poisson) |
| `pooled_gaussian` | Effect measurement; diff-in-diff, interrupted time series | `intercept`, `beta[…]`, `sigma`, per-group effects |

Rule of thumb: **one number per group → `conjugate_anomaly`; a response explained by
predictors → `pooled_gaussian`.**

Planned, and *not* present today: hierarchical negative-binomial (F1), censored
survival (F2), payment delay (F4), payer-alive/BTYD (F5), elasticity (F6), and the
NUTS engine. See [the roadmap](#roadmap).

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
| **v0.1** (now) | `conjugate_anomaly`, `pooled_gaussian`, exact + Laplace engines, diagnostics, calibration suites |
| v0.2 | NUTS engine, hierarchical negative-binomial (F1), censored survival (F2) |
| v0.3 | Payment delay, BTYD, elasticity families; scenario integration |

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
