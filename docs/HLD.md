# High-Level Design — anofox-bayes

| | |
|---|---|
| **Component** | anofox-bayes DuckDB extension (Rust) |
| **Status** | Draft v0.1 |
| **Companion docs** | anofox-bayes BRD; anofox-statistics feature request (pooling / censoring / priors) |

---

## 1. Design Principles

1. **Closed catalog over general PPL.** Model families are code, not user input. Every family ships with fixed parameterization decisions, analytic gradients, and its own calibration suite.
2. **Posterior as data.** The unit of output is a DuckDB table of draws. Everything downstream — intervals, decisions, what-ifs, reports — is SQL.
3. **Reuse before rebuild.** Likelihood/IRLS machinery and MAP estimation come from `anofox-regression`; the NUTS kernel comes from `nuts-rs` (pymc-devs). anofox-bayes contributes the catalog, the compiler, the SQL surface, and the validation harness.
4. **Two engines, one contract.** Laplace and NUTS are interchangeable behind the same SQL interface and draws schema.
5. **Machine-first ergonomics.** The primary caller is an adk-rust agent workflow; every output is structured and branchable, every quality gate is a query.

## 2. Architecture Overview

```mermaid
flowchart TB
    subgraph agent["Agent plane (adk-rust, single binary)"]
        LLM[LLM agent<br/>problem framing, family selection]
        WF[Deterministic workflow nodes<br/>validation, gates, refusal path]
    end

    subgraph duck["DuckDB (embedded)"]
        SQL[SQL surface<br/>table fns, aggregates, macros]
        subgraph ext["anofox-bayes extension"]
            CAT[Model catalog<br/>F1..F7 specs]
            CMP[Spec compiler<br/>config -> logp + analytic grad]
            LAP[Engine A: Laplace<br/>MAP + curvature]
            NUTS[Engine B: NUTS<br/>nuts-rs adapter]
            DIAG[Diagnostics<br/>rhat, ess, divergences]
        end
        DR[(draws / metadata tables)]
        SCN[(anofox_scenario catalog)]
    end

    subgraph crate["anofox-regression crate"]
        IRLS[GLM likelihoods + IRLS/MAP]
        MIX[pooling / censoring / priors<br/>(feature request)]
    end

    ERP[(erpl: SAP / ERP data)] --> SQL
    LLM --> WF --> SQL
    SQL --> CMP
    CAT --> CMP
    CMP --> LAP & NUTS
    LAP & NUTS --> DR
    IRLS & MIX --> LAP
    CMP -. logp/grad .-> NUTS
    DIAG --> SQL
    DR --> SCN
```

## 3. Components

### 3.1 SQL Surface

| Function | Kind | Purpose |
|---|---|---|
| `anofox_bayes_fit(family, TABLE data, config JSON)` | table function | Fit; returns fit summary row(s); materializes draws + metadata tables |
| `anofox_bayes_draws(model_id)` | table function / view | Access draws `(model_id, group_id, chain, draw, param, value)` |
| `anofox_bayes_predict(model_id, TABLE newdata, kind)` | table function | Posterior- (or prior-) predictive samples for new rows |
| `rhat(value)`, `ess_bulk(value)`, `ess_tail(value)` | aggregates | Convergence diagnostics over draws, `GROUP BY param` |
| `anofox_bayes_status(model_id)` | scalar/table | Structured status: `converged \| degenerate \| insufficient_data \| failed` + reasons |
| Convenience macros | SQL macros | `credible_interval(...)`, `service_level_quantile(...)`, family-specific decision helpers |

Positional parameters, `anofox_bayes_*` prefix with short aliases — consistent with existing anofox extensions.

> **Implementation note (v0.1).** `anofox_bayes_fit` is realised as a *pure* table-in/table-out
> function: it returns the draw rows and materialises nothing itself. Persistence is the
> caller's `CREATE TABLE draws AS SELECT * FROM anofox_bayes_fit(...)`. This keeps the
> extension free of global mutable database state, makes every fit trivially testable in
> sqllogictest, and makes `model_id` a pure function of its inputs. Model-level status and
> metadata travel inside the same draws table on reserved `__`-prefixed parameter rows, so the
> single artefact the caller persists is self-describing. See `docs/DRAWS_CONTRACT.md`.

### 3.2 Model Catalog & Spec Compiler

- Each family = a Rust module implementing a `ModelFamily` trait: `validate(config, schema)`, `logp`, `grad`, `transform` (constrained↔unconstrained), `default_priors`, `predict`.
- Config is typed JSON validated against a per-family schema (prior slots, pooling structure, offsets/exposure, censoring column). Invalid config fails **before** any computation with machine-readable errors.
- Parameterization decisions are baked in (e.g. non-centered hierarchies, log-links, softplus for dispersion) — callers cannot select bad parameterizations.
- Gradients are **analytic**, hand-derived per family, unit-tested against finite differences. No general autodiff dependency.

### 3.3 Inference Engines

**Engine A — Laplace (default for non-hierarchical / large-data fits):**
1. MAP via penalized IRLS from `anofox-regression` (explicit priors = generalized penalties; Ridge is already the Gaussian-prior special case).
2. Curvature (observed information) at the mode → multivariate normal approximation on the unconstrained scale.
3. Sample N draws from the approximation, back-transform, write to draws table.
- Cost: seconds. Quality: excellent for GLM-shaped posteriors; flagged unsuitable per family where known to fail.

**Engine A′ — Exact (conjugate families).** Where the family is conjugate (F7, and F3 under
Gaussian priors) the posterior is available in closed form as a Normal-Inverse-Gamma, and
Laplace is exact for the coefficient block. These families sample from the exact posterior
instead, and the Laplace path is retained as an independent cross-check: the two engines must
agree to sampling error, which is the strongest correctness gate in the test suite.

**Engine B — NUTS (`nuts-rs` adapter):**
- The compiler exposes `logp + grad` through a thin adapter implementing the `nuts-rs` model interface (mirrors how nutpie serves PyMC).
- Multi-chain, parallel via rayon/tokio-blocking pool; draws streamed into DuckDB vectors chunk-wise (no full-run buffering).
- Default engine for hierarchical variance parameters, small-group settings, and any family whose SBC shows Laplace miscalibration.

**Engine selection:** per-family default + config override; recorded in model metadata. Same draws schema either way (Laplace writes `chain = 0`).

### 3.4 Storage Model

```
_anofox_bayes_models(model_id, family, config JSON, data_fingerprint,
                     engine, status, created_at, durations, seed, versions)
_anofox_bayes_draws(model_id, group_id, chain, draw, param, value)
_anofox_bayes_diagnostics(model_id, param, rhat, ess_bulk, ess_tail,
                          n_divergent, flags)
```

- Draws stored long-format for SQL ergonomics; DuckDB compression makes this cheap. Optional Parquet export for archival.
- `data_fingerprint` (hash of input relation) enables cache hits / refit detection.
- **anofox_scenario integration (P1):** model + draws registered as catalog objects; branching a scenario clones metadata by reference — counterfactual = branch + modified predict inputs, versioned and diffable.

> **Implementation note (v0.1).** These are *conventions for caller-owned tables*, not
> extension-managed state. `model_id` is a BLAKE3 digest of
> `(data_fingerprint, family, canonical_config, seed)`, so identical inputs always yield the
> same id and the same draws — refit detection is a comparison, not a registry lookup.

### 3.5 Diagnostics & Refusal Path

- Split-R̂, bulk/tail ESS, divergence counts per Betancourt-era standard practice; implemented as DuckDB aggregate functions so gates are queries.
- `status` is computed, never inferred by the caller: threshold config (defaults conservative) → `converged` only if all parameters pass.
- `insufficient_data` is a first-class outcome (e.g. hierarchical fit where group-level effects are prior-dominated) — this is the Decision-Lab-style calibrated refusal that agents surface as "signal too weak."

## 4. Agent Integration (adk-rust)

- Fits and gates are **deterministic workflow nodes** (SequentialAgent); the LLM agent selects family + config from the data profile but cannot bypass validation or diagnostics gates.
- LoopAgent implements criticism: fit → posterior-predictive check (SQL comparison vs holdout) → revise config or emit refusal.
- All calls are SQL through the embedded DuckDB handle — no IPC, no serialization boundary.
- Headless re-runs (monthly reorder points) are the same workflow minus the conversational layer.

## 5. Validation Strategy

1. **Unit:** analytic gradients vs finite differences; transforms round-trip.
2. **Golden-run parity:** per family, reference datasets fit in PyMC (pinned version, pinned seeds); posterior means/quantiles compared within documented tolerance. Run in CI.
3. **Simulation-based calibration (SBC):** per family, per engine; rank-uniformity checked; CI blocks release on regression. Laplace families additionally SBC-tested to certify where the approximation is admissible.
4. **Property tests:** degenerate inputs (constant columns, single-observation groups, all-censored) must yield `status != converged`, never NaN posteriors.
5. Validation reports published with releases (mirrors anofox-regression's R-validation practice) — doubles as sales collateral.

## 6. Performance Considerations

- Laplace path: dominated by IRLS; reuses crate's streaming sufficient-statistics fits where applicable.
- NUTS path: budget-per-fit config (max seconds / max draws); chunked draw streaming caps memory.
- Long fits inside a query: v0.1 accepts blocking table-function semantics with cancellation support; if field experience demands it, add a job-style API (`fit_async` + status polling) in v0.2 — decision explicitly deferred (BRD OQ-4).
- Group-parallelism: independent per-group fits (F7, per-segment F4/F6) parallelize trivially over DuckDB's execution; hierarchical fits are single jobs with internal chain parallelism.

## 7. Packaging, Licensing, Versioning

- Separate repo `DataZooDE/anofox-bayes`; **not** part of the community anofox-statistics extension (strategic/licensing separation per BRD G5). Licensed BSL 1.1 (Change License MPL 2.0).
- Depends on `anofox-regression` (crates.io, semver-pinned) and `nuts-rs` (pinned, adapter-isolated).
- Extension versioning and DuckDB-version matrix per existing anofox conventions; draws-schema changes are breaking and gated behind `model_id` metadata versioning.

## 8. Explicitly Rejected Alternatives

- **Python sidecar (PyMC service):** rejected for deployment weight on on-prem targets; retained only as internal reference for golden runs.
- **General autodiff in Rust:** rejected for v1 — closed catalog makes analytic gradients cheaper and audit-friendly.
- **Reimplementing NUTS:** rejected; `nuts-rs` is maintained by pymc-devs and battle-tested via nutpie.
- **Folding everything into anofox-statistics:** rejected; runtime profile (long fits, sampler dependency) and licensing optionality argue for a separate extension. Frequentist-usable math (pooling, censoring, priors/Laplace at crate level) still lands in anofox-regression/anofox-statistics — see companion feature request.

## 9. Phase Plan (engineering view)

| Phase | Deliverables |
|---|---|
| **0.1** | Extension skeleton, draws contract, F3 (pooled Gaussian linear) + F7 (conjugate anomaly), Laplace engine, diagnostics aggregates, SBC + golden-run CI |
| **0.2** | F1 (hier. NB) + F2 (censored durations) on crate's pooling/censoring; nuts-rs adapter; status/refusal hardening; scenario-catalog registration |
| **0.3** | F4–F6, posterior-/prior-predictive functions, decision macros, optional async-fit API |
