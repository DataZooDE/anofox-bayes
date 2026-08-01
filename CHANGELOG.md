# Changelog

All notable changes to anofox-bayes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is semantic, and
the **draws schema** (`docs/DRAWS_CONTRACT.md`) is versioned separately so that draw
tables persisted by a customer stay readable across extension upgrades.

## [Unreleased]

### Added — `pooled_gaussian` random slopes

Roadmap gap 4, first half. `docs/ROADMAP.md` §3.3 scoped it and the scope held: random
slopes at a *fixed* `pool_scale` are a design-matrix change and nothing else, so the
family keeps its closed-form warranty. A per-group `sigma` and a *learned* `pool_scale`
break conjugacy and are not here.

- **`random_slopes`** — a predictor or list of predictors, each of which gains one
  column per group under the same `N(0, sigma^2 * pool_scale^2)` prior as a group
  intercept. Emitted as `group_slope[<column>]`, one parameter name carried across
  `group_id` exactly as `group_effect` is, so `GROUP BY param` stays meaningful at
  hundreds of groups and an existing downstream query keeps working.
- **An entry must also appear in `x`.** A random slope is a deviation *from* a
  population slope; without one the group slopes would be shrunk toward zero, which
  asserts the predictor has no effect — the claim `docs/ROADMAP.md` criticises
  `beta_scale` for making. Refused with a message naming the column.
- **The ridge closed form still holds exactly** on a design containing group slopes
  (agreement to `1e-9` against an independently assembled and solved system), and
  `exact`, `laplace` and `nuts` still agree on such a design.
- **The design is rank deficient by construction and the prior is what fixes it.** A
  predictor's group-slope columns sum to its fixed column — the intercept-plus-group-
  dummies trap one level up. `X'X + P` is positive definite exactly when the
  *unpenalised* columns are independent; rank deficiency there is still refused.
- **Laplace understates `sigma` by more as the design widens.** Each random slope adds
  columns without adding observations, and the mode-to-mean ratio
  `sqrt((2(a_n-1))/(2 a_n + p))` falls with `p`. Measured 2.3 % at n = 240, p = 10.
  The engine-agreement test pins the predicted ratio rather than loosening a tolerance.
- New SBC suite `f3_random_slopes_are_calibrated_under_the_exact_engine` (χ² 18.5–26.7
  at 15 df against a 37.7 threshold) and a `test/sql/f3_price_elasticity.test` scenario.
- **Non-breaking.** A fit without `random_slopes` is unchanged, parameter for parameter
  and draw for draw; `__schema_version__` and `ALGORITHM_VERSION` do not move.

### Changed — `conjugate_anomaly` fits its groups in parallel

Roadmap gap 14, first of three. `docs/SCALABILITY.md` carries the measurements.

- **Group parallelism.** Sampling and diagnostics now run one rayon task per group and
  per parameter. Crate-side wall time at 5 000 groups × 1 000 draws falls from 1 904 ms
  to 229 ms (8x); the BRD BR-1 acceptance case falls from 4.4 s to 2.8 s end to end
  (1.6x — the fit is now a minority of the query, and `SCALABILITY.md` says where the
  rest goes). `pooled_gaussian` is untouched: a pooled fit is one joint system.
- **Diagnostics were the bottleneck, not sampling** — 1 422 ms of the 1 904 ms above.
  That was not what `SCALABILITY.md` predicted, and it is why the biggest single
  improvement here is in `diagnostics::diagnose` rather than in the sampler.
- **⚠ `conjugate_anomaly` draws have changed, and `ALGORITHM_VERSION` is now 3.** Each
  group draws from a stream keyed on the group's **own identity**
  (`BayesRng::for_group(seed, chain, key)`) rather than from a shared sequential
  stream. That is what makes the parallelism safe, and it is a strictly better
  property than the one it replaces: a group's draws no longer depend on its position
  in the input relation, so re-ordering the rows, batching a wide group set, or
  fitting a group alone all reproduce the same numbers. A fit re-run on this build
  gets a *different* `model_id` from the same request on the previous build, rather
  than silently serving old draws under a new identity.
- **Row partitioning is ~6x faster** (`DataView::group_rows`, 113 ms → 18 ms at
  520 000 rows): one hash lookup per row instead of two ordered `BTreeMap` probes.
  First-seen group order — which fixes the parameter order and therefore the emission
  order — is preserved exactly.
- **Per-group *fitting* is deliberately still serial.** Running it on rayon was tried
  and measured *slower*; the conjugate update is one pass over sufficient statistics
  and the per-task allocation costs more than the arithmetic.
- `validation/bench.py --threads` now varies both thread axes (`SET threads` and
  `RAYON_NUM_THREADS`) and asserts the draws digest is identical across all four
  configurations. `cargo run --release --example scale_profile` reports the crate-side
  phase breakdown the numbers above come from.

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
- `laplace` — MAP plus curvature on an unconstrained scale, available on **both**
  families. Neither needs it, since both are conjugate; it is there because checking
  it against the exact posterior is the strongest correctness gate in the suite, two
  independent derivations of one distribution. On `conjugate_anomaly` the size of the
  disagreement is itself closed form (`1 - sqrt((n-3)/n)` on the spread of `mu`, per
  group), and the measurement matches it to four digits.
- `laplace` — MAP plus curvature on an unconstrained scale, available on
  every family, and the only engine for `payer_alive` and `censored_aft`. On the
  conjugate families it is checked against the exact posterior, which is the strongest
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
