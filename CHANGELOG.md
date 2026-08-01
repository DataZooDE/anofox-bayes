# Changelog

All notable changes to anofox-bayes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is semantic, and
the **draws schema** (`docs/DRAWS_CONTRACT.md`) is versioned separately so that draw
tables persisted by a customer stay readable across extension upgrades.

## [Unreleased]

### Added — `censored_aft` (F2), bridged onto anofox-statistics

Roadmap gap 2, and the first time HLD §3's "reuse before rebuild" has actually been
exercised.

- **`censored_aft` (F2)** — accelerated failure time regression with right censoring,
  in Weibull, lognormal, log-logistic and exponential form, optionally per group.
  The delivery-promise / time-to-event family. Worked example in
  `test/sql/f2_delivery_promise.test`; reference in `docs/API_REFERENCE.md` §2.3.
- **The bridge** (`crates/anofox-bayes-core/src/bridge.rs`) — anofox-bayes now takes a
  Cargo git dependency on `anofox-stats-core`, **pinned to an exact revision**, and
  calls its censored AFT fit in-process. No SQL round trip and no serialised matrix,
  which is what lets a fit performed in another crate participate in `model_id`: the
  data fingerprint is computed over the rows this crate selected, and the bridge
  refuses outright if the upstream fit read a different number of them.
- **The full covariance matrix, not the diagonal.** `fit_aft` returns only slices of
  the diagonal (`std_errors`), but the pieces the matrix is made of are public, so the
  observed information is reassembled at the returned mode and inverted by
  `anofox-statistics`' own routine. The reassembly is *checked*, not trusted: its
  diagonal must reproduce the reported standard errors, which it does to the last bit,
  and a mismatch is an error rather than a number to publish.
  Sampling from the diagonal instead would have inflated a predictive interval by a
  factor of ~25 on a correlated design, with every diagnostic still green.
- **`GaussianApproximation` / `GaussianBlock`** — a new catalog capability for a model
  that arrives already fitted, consumed by the Laplace engine. It carries the full
  precision matrix by construction, so the next bridged likelihood cannot accidentally
  reintroduce a diagonal. Blocks are independent, so a per-group model is naturally
  block diagonal and a refused group simply contributes no block.
- **Refusal semantics are mapped, not inherited.** `anofox-stats-core`'s error
  vocabulary is translated into `FitStatus` with a test per row: every-row-censored →
  `degenerate`, too few rows → `insufficient_data`, rank-deficient → `degenerate`,
  non-convergence → `failed`; a non-positive duration or non-binary event indicator
  stays a *request error* naming the column, because it is a malformed request rather
  than weak evidence. An unmapped upstream variant reports itself instead of being
  graded.
- **SBC suites for the bridged likelihood**, because bridging is not an escape from the
  calibration bar. `f2_exponential_is_calibrated` is a complete certificate (χ² 13.0 /
  12.8 against a 37.7 threshold); the Weibull suite certifies the coefficients only and
  says so in its name, because the upstream fit admits no prior on the scale and an
  improper prior cannot be sampled from. **`sigma` is therefore not SBC-certified for
  the distributions that estimate it** — recorded in `docs/THEORY.md` §5 rather than
  left to be discovered, and closable by a one-field change upstream.
- **`__family__` gains code `2`**, decoded by `anofox_bayes_family_text`. Append-only;
  `__schema_version__` does not move.

**C++ changes required by this work, for the central build:** exactly one —
`anofox_bayes_family_text` in `src/macros/bayes_macros.cpp` gains
`WHEN 2 THEN 'censored_aft'`, which is included here. Nothing else on the C++ side
enumerates families: `anofox_bayes_fit` passes the family name through the FFI as a
string and the catalog resolves it, so a new family needs no binding, no type and no
registration. The Rust workspace, `cargo fmt` and `cargo clippy` are clean; the
sqllogictest suite (`test/sql/f2_delivery_promise.test`) has **not** been executed
here because it needs the full extension build — its fixture and every threshold in it
were instead reproduced against `anofox-bayes-core` directly and checked to hold with
margin.

Also recorded, in `docs/THEORY.md` §8: replacing the posterior precision with its own
diagonal leaves 219 of 220 unit tests **and all six SBC suites** green. A per-parameter
check cannot certify a covariance, so every family now needs one assertion on a linear
combination of its parameters.

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
  (`2` = `censored_aft`, `3` = `pooled_gaussian`, `7` = `conjugate_anomaly`). The `value` column is `DOUBLE`,
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
- `laplace` — MAP plus curvature on an unconstrained scale, available on **both**
  families. Neither needs it, since both are conjugate; it is there because checking
  it against the exact posterior is the strongest correctness gate in the suite, two
  independent derivations of one distribution. On `conjugate_anomaly` the size of the
  disagreement is itself closed form (`1 - sqrt((n-3)/n)` on the spread of `mu`, per
  group), and the measurement matches it to four digits.

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
