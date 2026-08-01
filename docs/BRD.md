# Business Requirements Document — anofox-bayes

| | |
|---|---|
| **Product** | anofox-bayes — Bayesian inference as a native DuckDB extension |
| **Status** | Draft v0.1 |
| **Owner** | Joachim Rosskopf (DataZoo GmbH) |
| **Date** | 2026-07-30 |
| **Related** | anofox-statistics, anofox-regression (Rust crate), anofox-forecast, anofox_scenario, anofox_solve, erpl, FDA go-to-market |

---

## 1. Problem Statement

Enterprise decision agents (demand/inventory, supplier lead times, causal impact evaluation, cash-flow risk, collections, pricing) require **calibrated uncertainty**, not point estimates. Today this capability lives exclusively in Python probabilistic-programming stacks (PyMC, Stan), which forces a Python runtime and sidecar service into every deployment — unacceptable for DataZoo's single-binary, on-premise delivery model at German manufacturing and enterprise SAP customers.

The cost of not solving it: every Forward Deployed Agent engagement carries a Python container, its dependency surface, and its ops burden onto customer infrastructure; rollouts take weeks instead of days; and the inference layer remains a commodity component instead of a DataZoo asset.

## 2. Product Vision

Bayesian inference for a **closed catalog of enterprise decision models**, executed inside DuckDB, with posteriors materialized as **queryable draw tables**. Fitting is a table function; diagnostics are SQL aggregates; what-if analysis is SQL over posterior draws, versioned in anofox_scenario branches. One static binary: agent runtime (adk-rust) + DuckDB + erpl + anofox extensions.

Explicit anti-vision: anofox-bayes is **not a general-purpose PPL**. Generality is the failure mode; the closed catalog is the product.

## 3. Goals

- **G1 — Enable the agent portfolio.** All seven identified logistics/finance agents can run their inference through anofox-bayes SQL calls (no Python at runtime).
- **G2 — Deployment collapse.** Inference adds zero deployment components: no sidecar, no container, no Python. Target: agent installation on customer infrastructure ≤ 1 day.
- **G3 — Provable calibration.** Every model family passes simulation-based calibration (SBC) in CI and matches PyMC golden runs within tolerance. This is both an engineering gate and a sales artifact.
- **G4 — Posterior-as-table differentiation.** What-if / counterfactual questions on fitted models answer in < 1 s as pure SQL, without re-fitting.
- **G5 — Strategic optionality.** Ships as a separate extension (not inside the community anofox-statistics), preserving BSL/commercial licensing options analogous to erpl.

## 4. Non-Goals

- **General model specification language** (arbitrary likelihoods, user-defined logp). Unbounded correctness liability; the catalog is the moat. → Won't have.
- **Gaussian processes, time-varying parameters, state-space models.** Not required by the current agent portfolio; anofox-forecast covers time-series point/interval forecasting. → Future consideration only.
- **Reimplementing the sampler.** NUTS comes from the existing `nuts-rs` crate (pymc-devs); we do not write MCMC kernels. → Won't have.
- **Frequentist regression features.** Mixed models, censored likelihoods, and explicit priors at the MAP/Laplace level belong in anofox-regression / anofox-statistics (see companion feature request). anofox-bayes consumes them. → Out of scope here.
- **A GUI / notebook experience.** The consumers are agents and SQL users.

## 5. Target Users & User Stories

**Persona A — FDA agent (machine consumer, primary).**
- As a decision agent, I want to fit a cataloged model via one table function so that inference is a deterministic, auditable workflow step.
- As a decision agent, I want R-hat/ESS/divergences as SQL aggregates so that my quality gate is a query a deterministic node can enforce.
- As a decision agent, I want to answer scenario questions from the draws table so that what-ifs need no re-fit.
- As a decision agent, I want a structured "insufficient signal" outcome so that I can refuse instead of fabricating a recommendation.

**Persona B — Data engineer / analyst at customer or DataZoo.**
- As a data engineer, I want to inspect model specs, draws, and diagnostics with plain SQL so that I can audit what the agent did.
- As an analyst, I want to branch a fitted model's scenario in anofox_scenario so that counterfactuals are versioned and diffable.

**Persona C — DataZoo (business).**
- As the vendor, I want the extension validated against the reference implementation so that CFO-facing agents (cash runway) are defensible.

## 6. Scope: Model Catalog v1

| Family | Likelihood / structure | Agents served |
|---|---|---|
| F1 Hierarchical count GLM | Poisson / Negative Binomial, partial pooling | #1 safety stock (C-parts), #4 |
| F2 Duration / survival | Lognormal, Gamma, Weibull AFT, right-censoring | #2 delivery promises |
| F3 Pooled Gaussian linear | Diff-in-diff, synthetic-control inference layer | #3 intervention evaluation |
| F4 Payment-delay | Gamma / lognormal per segment, pooled | #4 cash runway |
| F5 Payer-alive (BTYD) | Pareto/NBD-style | #5 collections |
| F6 Elasticity GLM | log-log NB/Gamma, pooled by segment | #6 pricing |
| F7 Conjugate anomaly | Normal/Gamma closed-form posteriors | #7 freight audit |

Each family ships with: declarative spec (typed config, prior slots with validated defaults), fixed parameterization decisions (e.g. non-centered hierarchies), analytic gradients, prior- and posterior-predictive support, SBC suite, PyMC golden-run parity test.

## 7. Requirements

### Must-Have (P0)
- **BR-1 Fit as table function.** `anofox_bayes_fit(family, TABLE, config)` materializes a draws table. *AC:* fitting F1 on 5k SKUs × 104 weeks completes on commodity hardware; result is a persistent DuckDB table.
- **BR-2 Draws table contract.** Stable schema `(model_id, group_id, chain, draw, param, value)` + model metadata table. *AC:* schema documented and versioned; quantile/interval queries are plain SQL.
- **BR-3 Diagnostics as aggregates.** `rhat`, `ess_bulk`, `ess_tail`, divergence count as aggregate functions. *AC:* a single SQL query yields pass/fail per parameter against configurable thresholds.
- **BR-4 Two inference paths, one interface.** Laplace approximation (MAP + curvature via anofox-regression) and full NUTS (`nuts-rs`), selected per family/config, invisible to the SQL surface. *AC:* switching engine does not change caller SQL.
- **BR-5 Refusal semantics.** Fit output includes machine-readable status (converged / degenerate / insufficient-data) — never silent bad numbers. *AC:* agents #1–#7 can branch on status without parsing text.
- **BR-6 Validation harness.** SBC per family + golden-run parity vs PyMC in CI. *AC:* CI blocks release on calibration regression; tolerance documented per family.
- **BR-7 Catalog v1 families F1–F3, F7.** Minimum set to ship agents #3, #7 (immediately) and #1, #2 (Phase 2 of GTM).

### Nice-to-Have (P1)
- **BR-8** Families F4–F6.
- **BR-9** anofox_scenario integration: model specs + draw tables registered in the branch-versioned catalog; scenario branch = counterfactual.
- **BR-10** Posterior-predictive table function reading draws tables.
- **BR-11** Prior-predictive check function (pre-fit sanity gate for agents).

### Future Considerations (P2)
- **BR-12** Additional families pulled by revenue (e.g. hierarchical elasticity with cross-effects).
- **BR-13** Streaming/incremental refit (warm starts from previous posterior).
- **BR-14** GPU/large-scale sampling backends.

## 8. Success Metrics

**Leading (≤ 90 days after v0.1):**
- Agents #3 and #7 running end-to-end on anofox-bayes at ≥ 1 pilot customer.
- 100 % of shipped families pass SBC in CI; parity vs PyMC within documented tolerance.
- Fit-to-first-decision time for agent #3 ≤ 1 h wall clock on customer data.

**Lagging (2–3 quarters):**
- ≥ 3 FDA engagements delivered without any Python runtime on customer infrastructure.
- Inference-related deployment effort ≤ 0.5 day per engagement (from current multi-day sidecar setup).
- anofox-bayes cited as differentiator in ≥ 1 closed deal (posterior-as-SQL / scenario branching story).

## 9. Dependencies

- **anofox-regression crate:** mixed-effects/pooling, censored likelihoods, explicit priors + Laplace (companion feature request; maintainer alignment with sipemu required).
- **nuts-rs** (pymc-devs): NUTS engine; pin and track upstream.
- **anofox_scenario:** catalog/branching integration (P1).
- **DuckDB extension API:** table + aggregate function registration; version pinning policy as for other anofox extensions.

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Numerical-correctness liability shifts from PyMC community to DataZoo | High | Closed catalog, analytic gradients, SBC + golden-run CI as release gate |
| Mixed-model numerics in the crate slip | Medium | Empirical-Bayes shrinkage as documented stepping stone (80 % of value) |
| Scope creep toward "general PPL" | High | Non-goals enforced in review; new families require a paying use case |
| nuts-rs API churn | Low–Med | Thin adapter layer; Laplace path is independent fallback |
| Long-running fits vs DuckDB execution model | Medium | Fit as table function with progress/cancellation; document runtime profile (see HLD) |

## 11. Timeline & Phasing

- **Phase 1 (v0.1):** F3 + F7, Laplace path, draws contract, diagnostics aggregates, SBC harness. Unblocks agents #3/#7.
- **Phase 2 (v0.2):** F1 + F2 (requires crate pooling + censoring), NUTS path for hierarchical variance parameters. Unblocks agents #1/#2.
- **Phase 3 (v0.3):** F4–F6, scenario-catalog integration, posterior-predictive tooling.
- No hard external deadline; sequencing is coupled to FDA pipeline (agent #3 as first fixed-price engagement).

## 12. Open Questions

- **Licensing (business):** BSL 1.1 like erpl, or dual-license from day one? Decide before first customer install.
  *Resolved 2026-08-01: BSL 1.1, Licensor DataZoo GmbH, Change License MPL 2.0, five-year Change Date — see `LICENSE`.*
- **Naming/packaging (business):** `anofox_bayes` vs. folding into a broader "decision" extension later.
- **Crate governance (engineering, sipemu):** where do priors/pooling/censoring land in anofox-regression's roadmap; release cadence coupling.
- **Runtime limits (engineering):** max acceptable fit duration inside a DuckDB query before we need a job-style API (non-blocking; prototype will answer).
