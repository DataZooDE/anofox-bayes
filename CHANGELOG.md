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
  (`3` = `pooled_gaussian`, `5` = `payer_alive`, `7` = `conjugate_anomaly`). The
  `value` column is `DOUBLE`,
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
- **F5 `payer_alive`** — BG/NBD buy-till-you-die over per-customer
  `(frequency, recency, age)` statistics, estimating four population parameters.
  Serves the collections agent. Its `P(alive)` is closed form and **expressible in
  SQL**, so a fitted model rescores any customer list with a join and no re-fit —
  which is why the family is BG/NBD and not Pareto/NBD, whose `P(alive)` needs a
  hypergeometric function no database has (roadmap §3.2, §5).

  Boundary solutions are a documented hazard of this likelihood: where no repeat
  buyer has been seen to go quiet, it has no interior maximum. The family therefore
  finds its own mode at compile time and applies four checks to it — admissible
  range, stationarity, positive-definite curvature, and a marginal narrow enough to
  be an interval — reporting `degenerate` with `NULL` draws rather than an interval
  derived from curvature that is not a posterior.

**Engines**

- `exact` — closed-form sampling for conjugate families; the default for both.
- `laplace` — MAP plus curvature on an unconstrained scale, available on
  `pooled_gaussian` and the only engine for `payer_alive`. On `pooled_gaussian` it is
  checked against the exact posterior, which is the strongest correctness gate in the
  suite: two independent derivations of one distribution.
- The Newton search gained two globalisations, both found by building F5 and both
  fixing behaviour that was wrong rather than merely absent:
  - a **trust region** on the step. A backtracking line search only shrinks a step
    that made things worse, so a step landing in a distant *local* optimum was
    accepted. Measured: from `a = b = 1`, F5's first step moved `ln a` by +14.6 onto a
    flat ridge whose density was 248 lower than the true mode.
  - a **relative stopping rule** on the improvement in log density. The absolute
    `1e-8` gradient tolerance is unreachable for a likelihood summed over thousands of
    rows, where rounding error alone exceeds it, so a good fit exhausted its iteration
    budget and reported a convergence failure. F5's stationarity check scales with the
    number of customers for the same reason.

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
