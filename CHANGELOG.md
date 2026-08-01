# Changelog

All notable changes to anofox-bayes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is semantic, and
the **draws schema** (`docs/DRAWS_CONTRACT.md`) is versioned separately so that draw
tables persisted by a customer stay readable across extension upgrades.

## [Unreleased]

### Added — initial release, HLD phase 0.1

**SQL surface**

- `anofox_bayes_fit(relation, family, config)` — a pure table in-out function
  returning posterior draws. It materialises nothing itself; persistence is the
  caller's `CREATE TABLE ... AS SELECT`.
- `anofox_bayes_rhat`, `anofox_bayes_ess_bulk`, `anofox_bayes_ess_tail` — convergence
  diagnostics as aggregates over `(value, chain, draw)`, for `GROUP BY param`.
- Decision macros: `anofox_bayes_credible_interval` / `_lower` / `_upper`,
  `anofox_bayes_prob_greater` / `_less`, `anofox_bayes_service_level_quantile`,
  `anofox_bayes_status_text`, `anofox_bayes_is_actionable`.
- `anofox_bayes_version()`, `anofox_bayes_draws_schema_version()`.

**Model catalog**

- **F7 `conjugate_anomaly`** — closed-form Normal-Inverse-Gamma and Gamma-Poisson
  posteriors per group, for anomaly questions answered as posterior tail
  probabilities. Serves the freight-audit agent.
- **F3 `pooled_gaussian`** — conjugate Gaussian linear model with optional partial
  pooling by group; the inference layer for difference-in-differences and interrupted
  time series. Serves the intervention-evaluation agent.

**Engines**

- `exact` — closed-form sampling for conjugate families; the default for both.
- `laplace` — MAP plus curvature on an unconstrained scale, available on
  `pooled_gaussian`. Checked against the exact posterior, which is the strongest
  correctness gate in the suite: two independent derivations of one distribution.

**Validation**

- Simulation-based calibration per family and per engine, with fixtures proving the
  harness rejects overconfident and biased posteriors before any real family runs.
- PyMC golden-run parity suite (`validation/`, uv + pytest), run in CI.
- 150 Rust unit tests, 164 sqllogictest assertions.

### Notes

- Sampling defaults to one chain, so `anofox_bayes_rhat` is `NULL` unless
  `{'chains': n}` is set. R̂ detects a Markov chain that has not mixed, and both 0.1
  engines draw independently — a second chain would buy an R̂ of 1.0 that means
  nothing. **The gate in 0.1 is ESS**, and tail ESS is its binding half.
- Draws schema version **1**.
- Licensed BSL 1.1 (Licensor DataZoo GmbH, Change License MPL 2.0).

### Not yet implemented

NUTS engine, families F1/F2/F4–F6, `anofox_bayes_predict` (posterior- and
prior-predictive), `anofox_scenario` catalog integration, async/job-style fit. See
[docs/BRD.md](docs/BRD.md) §7 and [docs/HLD.md](docs/HLD.md) §9 for phasing.
