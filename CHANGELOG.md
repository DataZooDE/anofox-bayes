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
  `anofox_bayes_status_text`, `anofox_bayes_is_actionable`,
  `anofox_bayes_family_text`.
- `anofox_bayes_version()`, `anofox_bayes_draws_schema_version()`.

**Draws contract**

- `__family__` — which catalog family produced the table, as its BRD F-number
  (`3` = `pooled_gaussian`, `7` = `conjugate_anomaly`). The `value` column is `DOUBLE`,
  so the name cannot travel; reusing the existing F-numbering rather than minting a
  second one keeps one identity per family. Append-only, like `FitStatus` and
  `EngineKind`.
- `__n_groups_unready__` — how many of `__n_groups__` failed their own readiness check.
  `__status__` remains the collapsed worst-case verdict; this says how much of the fit
  that verdict is about.
- Both are new `__`-prefixed metadata rows and therefore **not** breaking:
  `__schema_version__` stays at 1. A consumer that assumed the metadata block was
  exactly eight rows long was relying on something the contract already said not to.
- `__data_fingerprint__` was considered and **not** shipped: a hex digest has no
  lossless home in a `DOUBLE`, and a fingerprint that silently collides is worse than
  an absent one. See `docs/DRAWS_CONTRACT.md`.

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
- `nuts` — the No-U-Turn Sampler, as a thin adapter over a pinned
  [`nuts-rs`](https://github.com/pymc-devs/nuts-rs) `=0.18.3` (pymc-devs; the sampler
  behind nutpie). Available on `pooled_gaussian`. No MCMC kernel is written here — the
  whole seam is one file, so an upstream bump is a review of one file. It exists for
  the families that have no closed form and for which a Gaussian at the mode is not
  honest; certifying it now against a family whose answer is known is the point of
  landing it before them.
  - Checked against the exact conjugate posterior at a tolerance **derived from the
    Monte Carlo standard error** (`sd/sqrt(ESS)`, with ESS measured from the draws
    rather than assumed), not from a round number that happened to pass.
  - Its own SBC suite, plus a fixture — the same pipeline with every draw pulled 40 %
    toward its posterior mean — that the suite is required to reject.
  - Deterministic: same seed, byte-identical draws, whatever the chain count and
    whatever DuckDB's thread layout. Chains are seeded from `(seed, chain)` and run
    sequentially; `nuts-rs`'s `parallel` feature is switched off deliberately.
  - `chains` defaults to **4** under this engine, and `warmup` (default 1000) is a new
    common config slot whose draws are discarded.

**Validation**

- Simulation-based calibration per family and per engine, with fixtures proving the
  harness rejects overconfident and biased posteriors before any real family runs.
- PyMC golden-run parity suite (`validation/`, uv + pytest), run in CI.
- 153 Rust unit tests, 181 sqllogictest assertions, 32 PyMC parity tests.

### Fixed

- **F3's residual-scale posterior was over-confident.** The Normal-Inverse-Gamma
  shape `a0 + n/2` is correct only when the coefficient prior is proper and
  sigma-scaled; under the default flat prior the textbook shape is `a0 + (n - k)/2`
  for `k` freely estimated coefficients. Every F3 credible interval was too narrow by
  `sqrt((n - k)/n)` — ~2 % on typical designs, growing with `k/n`. Found by the PyMC
  parity suite, which is the only harness that could have: SBC must draw its truth
  from a proper prior, and so only ever exercised the case where the old formula was
  right.

### Notes

- Under `exact` and `laplace`, sampling defaults to one chain, so `anofox_bayes_rhat`
  is `NULL` unless `{'chains': n}` is set. R̂ detects a Markov chain that has not
  mixed, and those engines draw independently — a second chain would buy an R̂ of 1.0
  that means nothing. **Their gate is ESS**, and tail ESS is its binding half. Under
  `nuts` the default is four chains and R̂ is computed and load-bearing.
- **The reserved sampler statistics are now populated.** `__lp__`, `__divergent__`,
  `__energy__` and `__step_size__` appear on a `nuts` draws table, one row per kept
  draw. They remain absent for the other engines rather than being written as zeros.
  `__schema_version__` does **not** move: the contract has always required consumers
  to filter on the reserved names they know.
- **Any divergence makes a fit `degenerate`.** Not a warning and not a budget: the
  draws around a divergent trajectory are not from the posterior.
- One consequence of a diagonal mass matrix, measured rather than assumed:
  `pooled_gaussian` with a `group` column has an intercept/group-effect ridge that
  NUTS mixes slowly along. The diagnostics report it as a low ESS and an R̂ above 1.01
  and the fit is refused, which is the correct outcome — `exact` remains the default
  for this family for exactly this reason.
- Draws schema version **1**.
- Licensed BSL 1.1 (Licensor DataZoo GmbH, Change License MPL 2.0).

### Not yet implemented

Families F1/F2/F4–F6, `anofox_bayes_predict` (posterior- and
prior-predictive), `anofox_scenario` catalog integration, async/job-style fit. See
[docs/BRD.md](docs/BRD.md) §7 and [docs/HLD.md](docs/HLD.md) §9 for phasing.
