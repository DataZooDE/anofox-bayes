<h1 align="center">Anofox Bayes</h1>
<p align="center"><strong>Bayesian inference for enterprise decision models, inside DuckDB</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSL%201.1-blue.svg" alt="License: BSL 1.1"></a>
  <a href="https://duckdb.org"><img src="https://img.shields.io/badge/DuckDB-1.4.5%20LTS%20%7C%201.5.5-green.svg" alt="DuckDB"></a>
  <img src="https://img.shields.io/badge/draws%20schema-v1-informational.svg" alt="Draws schema v1">
  <img src="https://img.shields.io/badge/status-v0.1%20early-orange.svg" alt="Status">
</p>

> [!IMPORTANT]
> `anofox-bayes` is at **v0.1**. Two model families ship today (`conjugate_anomaly`
> and `pooled_gaussian`), both served by the exact conjugate engine. Everything
> marked *planned* below is not implemented and will not run. Breaking changes are
> expected; use the [issues page](https://github.com/DataZooDE/anofox-bayes/issues)
> to report bugs or request families.

Fit a cataloged Bayesian model with one table function, get a table of posterior
draws back, and answer every downstream question — intervals, decisions,
what-ifs, diagnostics — in plain SQL. No Python, no sampler sidecar, no second
artefact to keep in sync.

## What this is, and what it deliberately is not

`anofox-bayes` is **not a general-purpose probabilistic programming language**.
You cannot write a likelihood. The closed catalog *is* the product: because model
families are code rather than user input, each one can ship with fixed
parameterisation decisions, a validated config schema, a calibration suite and a
bounded correctness liability. Generality is the failure mode we are avoiding.

Three consequences shape the whole API:

**Posteriors are tables.** `anofox_bayes_fit` is a pure table-in / table-out
function. It materialises nothing, keeps no session state, and returns rows in the
[draws contract](docs/DRAWS_CONTRACT.md): `(model_id, group_id, chain, draw, param, value)`.
Persistence is your `CREATE TABLE draws AS SELECT * FROM anofox_bayes_fit(...)`.

**Refusal is a first-class outcome.** A fit that cannot answer the question says
so, inside the same table, on a `__status__` row. Unfittable groups get `NULL`
draws rather than a plausible-looking number. An agent gates on one comparison.

**The primary consumer is an agent, not a human.** Every quality gate is a query
a deterministic workflow node can enforce; every output is structured and
branchable. There is no notebook experience and none is planned.

## Installation

The extension is distributed for DuckDB **v1.4.5 LTS** and **v1.5.5**.

Because `anofox-bayes` is BSL-licensed it will **not** be published to the DuckDB
community repository. It is intended to ship from the DataZoo distribution channel,
like `erpl` and `anofox-statistics`:

```sql
-- Not live yet -- see below.
INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';
LOAD anofox_bayes;
```

> **No binaries are published yet.** The release pipeline is in place but has never
> run, and the channel needs a `DEPLOY_S3_BUCKET` variable plus an IAM OIDC trust-policy
> entry for this repository before it can. Until then, **build from source.**

### From source

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-bayes.git
cd anofox-bayes
export VCPKG_TOOLCHAIN_PATH=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake

make release -j$(nproc)      # or: GEN=ninja make release
                             # the first build compiles DuckDB: 30-60 minutes

# build/release/extension/anofox_bayes/anofox_bayes.duckdb_extension
# build/release/duckdb          (a shell with the extension linked in)
```

Requires a Rust toolchain (stable), CMake 3.15+, and a C++ compiler. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full development setup.

## Quick start — the freight audit

The question is not "which lane is expensive". `BRE-ANT` is legitimately
expensive. The question is **which lane stopped behaving like itself**.

```sql
LOAD anofox_bayes;

-- Three lanes billed monthly over three years. HAM-ROT and BRE-ANT are stable at
-- different contractual rates; DUS-MIL picks up an undeclared accessorial
-- surcharge halfway through the window.
CREATE TABLE invoices AS
SELECT 'HAM-ROT' AS lane, i AS period, 2.00 + ((i % 5) - 2) * 0.02 AS cost_per_kg
FROM generate_series(0, 35) AS t(i)
UNION ALL SELECT 'BRE-ANT', i, 3.00 + ((i % 7) - 3) * 0.03
FROM generate_series(0, 35) AS t(i)
UNION ALL SELECT 'DUS-MIL', i,
       2.00 + ((i % 5) - 2) * 0.02 + CASE WHEN i >= 18 THEN 1.20 ELSE 0.0 END
FROM generate_series(0, 35) AS t(i);

-- The fit. One call, one table of draws.
-- Note the first argument is a *subquery*, not `TABLE invoices`.
CREATE TABLE draws AS
SELECT * FROM anofox_bayes_fit(
    (SELECT lane, cost_per_kg FROM invoices),
    'conjugate_anomaly',
    {'value': 'cost_per_kg', 'group': 'lane', 'draws': 4000, 'seed': 42}
);
```

**Step 1 — the gate.** Status travels inside the draws table, so an agent gates
without a second query against state that may not survive the session.

```sql
SELECT value = 0 AS safe_to_act_on FROM draws WHERE param = '__status__';
```

```
┌────────────────┐
│ safe_to_act_on │
├────────────────┤
│ true           │
└────────────────┘
```

**Step 2 — the intervals.** Plain SQL quantiles over the draws.

```sql
SELECT group_id,
       round(quantile_cont(value, 0.025), 2) AS lo,
       round(quantile_cont(value, 0.500), 2) AS median,
       round(quantile_cont(value, 0.975), 2) AS hi
FROM draws WHERE param = 'mu' AND draw >= 0
GROUP BY group_id ORDER BY group_id;
```

```
┌──────────┬────────┬────────┬────────┐
│ group_id │   lo   │ median │   hi   │
├──────────┼────────┼────────┼────────┤
│ BRE-ANT  │   2.98 │    3.0 │   3.02 │
│ DUS-MIL  │   2.39 │    2.6 │    2.8 │
│ HAM-ROT  │   1.99 │    2.0 │   2.01 │
└──────────┴────────┴────────┴────────┘
```

**Step 3 — the decision.** `P(lane is billing above its contracted rate)` is an
average over the draws. No re-fit, no second artefact, no theory.

```sql
CREATE TABLE contract(lane VARCHAR, agreed_rate DOUBLE);
INSERT INTO contract VALUES ('HAM-ROT', 2.05), ('BRE-ANT', 3.05), ('DUS-MIL', 2.05);

SELECT d.group_id,
       round(avg(CASE WHEN d.value > c.agreed_rate THEN 1.0 ELSE 0.0 END), 3) AS p_overbilled
FROM draws d JOIN contract c ON c.lane = d.group_id
WHERE d.param = 'mu' AND d.draw >= 0
GROUP BY d.group_id ORDER BY p_overbilled DESC, d.group_id;
```

```
┌──────────┬──────────────┐
│ group_id │ p_overbilled │
├──────────┼──────────────┤
│ DUS-MIL  │          1.0 │
│ BRE-ANT  │          0.0 │
│ HAM-ROT  │          0.0 │
└──────────┴──────────────┘
```

Compared against a fleet-wide average, `BRE-ANT` would be flagged every month.
Compared against its own contract, only the lane that changed is flagged.

**Step 4 — the diagnostics gate.** Effective sample size per parameter, as an
aggregate. This is the shape a deterministic workflow node enforces before letting
a recommendation through.

```sql
SELECT count(*) AS failing_parameters FROM (
    SELECT group_id, param
    FROM draws WHERE draw >= 0
    GROUP BY group_id, param
    HAVING NOT anofox_bayes_ess_gate(value, chain, draw, 400)
);
```

```
┌────────────────────┐
│ failing_parameters │
├────────────────────┤
│                  0 │
└────────────────────┘
```

## Intervention evaluation — difference-in-differences

The other shipped family. Twelve stores over 24 months; six get a merchandising
change in month 12. The coefficient on the interaction term *is* the causal
effect, with its interval.

```sql
CREATE TABLE panel AS
SELECT 'S' || lpad(s::VARCHAR, 2, '0') AS store,
       m AS month,
       (m >= 12)::INTEGER AS post,
       ((s < 6) AND (m >= 12))::INTEGER AS treated_post,
       100.0 + 0.8 * m                              -- shared trend
            + 6.0 * sin(m * 2 * pi() / 12)          -- shared seasonality
            + 5.0 * s                               -- persistent store level
            + CASE WHEN s < 6 AND m >= 12 THEN 8.0 ELSE 0.0 END  -- the intervention
            + ((s * 7 + m * 3) % 5 - 2) * 0.4       -- idiosyncratic noise
       AS units
FROM generate_series(0, 11) AS a(s), generate_series(0, 23) AS b(m);

CREATE TABLE dd AS
SELECT * FROM anofox_bayes_fit(
    (SELECT store, units, post, treated_post, month FROM panel),
    'pooled_gaussian',
    {'y': 'units',
     'x': ['post', 'treated_post', 'month'],
     'group': 'store',
     'pool_scale': 20.0,
     'draws': 4000,
     'seed': 42}
);

SELECT round(quantile_cont(value, 0.025), 2) AS lo,
       round(quantile_cont(value, 0.500), 2) AS effect,
       round(quantile_cont(value, 0.975), 2) AS hi,
       round(avg(CASE WHEN value > 5.0 THEN 1.0 ELSE 0.0 END), 3) AS p_beats_rollout_cost
FROM dd WHERE param = 'beta[treated_post]' AND draw >= 0;
```

```
┌────────┬────────┬────────┬──────────────────────┐
│   lo   │ effect │   hi   │ p_beats_rollout_cost │
├────────┼────────┼────────┼──────────────────────┤
│   6.72 │   8.03 │   9.29 │                  1.0 │
└────────┴────────┴────────┴──────────────────────┘
```

The true incremental effect in the generated panel is `+8.0`. The last column is
the question a decision-maker actually asks — *is the effect big enough to beat
the rollout cost of 5 units per store-month?* — answered directly from the
posterior, with no p-value in sight.

Store-level intercepts are partially pooled, so a store with a short history
borrows strength from the rest instead of contributing noise. `treated` is
deliberately **not** a predictor: with one intercept per store, treatment status
is a function of the store and its main effect is not separately identified.

## Refusal, in full

A brand-new lane cannot support a variance estimate no matter how the model is
configured. The fit succeeds; the status refuses; the unfittable group's draws
are `NULL` rather than a number an agent might act on.

```sql
CREATE TABLE thin AS
SELECT 'HAM-ROT' AS lane, 2.00 + ((i % 5) - 2) * 0.02 AS cost_per_kg
FROM generate_series(0, 35) AS t(i)
UNION ALL SELECT 'NEW-LANE', 9.99;

CREATE TABLE thin_draws AS
SELECT * FROM anofox_bayes_fit(
    (SELECT lane, cost_per_kg FROM thin), 'conjugate_anomaly',
    {'value': 'cost_per_kg', 'group': 'lane', 'draws': 1000});

SELECT value AS status FROM thin_draws WHERE param = '__status__';        -- 2.0
SELECT group_id, count(*) AS draws, count(value) AS non_null
FROM thin_draws WHERE draw >= 0 GROUP BY group_id ORDER BY group_id;
```

```
┌──────────┬───────┬──────────┐
│ group_id │ draws │ non_null │
├──────────┼───────┼──────────┤
│ HAM-ROT  │  2000 │     2000 │
│ NEW-LANE │  2000 │        0 │
└──────────┴───────┴──────────┘
```

`__status__` is `0` converged, `1` degenerate, `2` insufficient_data, `3` failed.
The healthy lane in the same fit is untouched.

## Reproducibility

`model_id` is `BLAKE3(family, canonical_config, data_fingerprint, seed)`, so:

* same inputs → same id → byte-identical draws, and an auditor can re-run a
  recommendation and check it;
* refit detection is a comparison, not a registry lookup;
* the config is rendered key-sorted, so two callers who write the same options in
  a different order get the same model;
* the data fingerprint covers only the columns the model reads, over the rows it
  uses — adding an unrelated column invalidates nothing.

The default seed is `20260801`; pass `'seed'` to change it. A different seed is a
different model.

## The catalog

| Family | Status | Structure | Question it answers |
|---|---|---|---|
| `conjugate_anomaly` (F7) | **shipped** | Normal-Inverse-Gamma or Gamma-Poisson, closed form, per group | "Which group stopped behaving like itself?" |
| `pooled_gaussian` (F3) | **shipped** | Gaussian linear model, conjugate NIG posterior, optional partial pooling by group | "How much of this change was the intervention?" |
| F1 hierarchical count GLM | *planned (0.2)* | Poisson / Negative Binomial, partial pooling | safety stock for C-parts |
| F2 duration / survival | *planned (0.2)* | Lognormal / Gamma / Weibull AFT, right-censoring | delivery promises |
| F4 payment delay | *planned (0.3)* | Gamma / lognormal per segment, pooled | cash runway |
| F5 payer-alive (BTYD) | *planned (0.3)* | Pareto/NBD-style | collections |
| F6 elasticity GLM | *planned (0.3)* | log-log NB/Gamma, pooled by segment | pricing |

Asking for an unknown family fails immediately, and the error lists the catalog:

```
Invalid Input Error: anofox_bayes_fit: unknown model family 'foo'
  (catalog: pooled_gaussian, conjugate_anomaly)
```

## Engines

| Engine | Status | Notes |
|---|---|---|
| `exact` | **shipped**, and the default for both families | Both shipped families are conjugate, so the posterior is available in closed form. Approximating it would add error for nothing. Draws are independent, so `chains = 1`. |
| `laplace` | **shipped** | MAP + curvature on an unconstrained scale, for GLM-shaped posteriors. Available on `pooled_gaussian`, whose exact posterior is the reference it is checked against; `conjugate_anomaly` exposes no gradient and rejects it. |
| `nuts` | *planned (0.2)* | `nuts-rs` adapter, for hierarchical variance parameters. `{'engine': 'nuts'}` currently errors. |

Sampling defaults to **one chain**, so `anofox_bayes_rhat` returns `NULL` unless you
ask for more with `{'chains': 4}`. That default is deliberate rather than a
limitation: R̂ exists to catch a Markov chain that has not mixed, and both shipped
engines draw *independently*, so a second chain buys an R̂ of 1.0 that means nothing.
`NULL` rather than `1.00` matters — an agent gating on `rhat <= 1.01` must not be told
"converged" by a statistic that was never computed.

**Gate on `anofox_bayes_ess_bulk` and `anofox_bayes_ess_tail`.** Tail ESS is the
binding constraint: independent draws are worth roughly their own count for a
posterior mean but materially less for the 5 % and 95 % quantiles, and a service-level
or audit decision reads a quantile.

## Also planned, and not present today

Marked out explicitly so nobody writes SQL against them: the posterior- and
prior-predictive table function `anofox_bayes_predict`, the NUTS engine, families F1,
F2 and F4–F6, `anofox_scenario` catalog integration, and an async/job-style fit API.
See [docs/BRD.md](docs/BRD.md) §7 and [docs/HLD.md](docs/HLD.md) §9 for the phasing.

## Documentation

| Document | Contents |
|---|---|
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Every function, every config slot, every default |
| [docs/DRAWS_CONTRACT.md](docs/DRAWS_CONTRACT.md) | The output schema, reserved names, NULL semantics, `model_id` |
| [docs/BRD.md](docs/BRD.md) | Business requirements and roadmap |
| [docs/HLD.md](docs/HLD.md) | Architecture, engines, validation strategy |
| [TELEMETRY.md](TELEMETRY.md) | What is collected, and how to turn it off |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guidelines |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Attribution and third-party licenses |

## Multi-language support

Write the SQL once; it works from any language with DuckDB bindings — Python, R,
Julia, C++, Rust, Node.js, Go, Java. There is no serialization boundary and no
sidecar process: the fit runs inside the DuckDB you already have open.

## Telemetry

Anonymous, privacy-preserving usage telemetry is **on by default** and trivially
disabled:

```sql
SET anofox_telemetry_enabled = false;
```

```bash
export DATAZOO_DISABLE_TELEMETRY=1
```

No table names, column names, data values, config contents or SQL text ever leave
the machine. See [TELEMETRY.md](TELEMETRY.md) for the complete list.

## License

**Business Source License 1.1.** Licensor DataZoo GmbH; Change License MPL 2.0;
Change Date five years from first publication.

✅ Free for production use inside your business
✅ Free for development and research
❌ Cannot be offered to third parties on a hosted or embedded basis

See [LICENSE](LICENSE) for the full terms.

## Support

- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-bayes/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DataZooDE/anofox-bayes/discussions)
- **Email**: info@data-zoo.de

## Citation

```bibtex
@software{anofox_bayes,
  title  = {Anofox Bayes: Bayesian Inference for Enterprise Decision Models in DuckDB},
  author = {Joachim Rosskopf and DataZoo GmbH},
  year   = {2026},
  url    = {https://github.com/DataZooDE/anofox-bayes}
}
```

## Acknowledgments

Built on [DuckDB](https://duckdb.org), [faer](https://github.com/sarah-quinones/faer-rs),
[statrs](https://github.com/statrs-dev/statrs) and [BLAKE3](https://github.com/BLAKE3-team/BLAKE3).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

**Made with ❤️ by the Anofox Team at [DataZoo GmbH](https://data-zoo.de)**
