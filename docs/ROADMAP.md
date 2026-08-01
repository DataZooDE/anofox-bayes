# Roadmap

> What v0.1 does not do, in the order it is worth fixing. The ordering criterion is
> **decision agents unblocked per engineer-day**, not model interest. Where a number is
> a guess it says so.

v0.1 ships two families — `conjugate_anomaly` (F7) and `pooled_gaussian` (F3) — on two
engines, `exact` and `laplace`, with the draws contract, the diagnostics aggregates,
SBC and PyMC parity in CI. That covers agents 03 (intervention evaluation) and 07
(freight audit). Five of the seven planned agents have no path through this extension
today.

This document is the plan for the other five, plus the defects in what already ships.

## 1. The gaps, ranked

Ranked by agents unblocked per engineer-day. *Blocker* means the agent cannot run at
all; *degradation* means it runs with a caveat someone has to carry.

| # | Gap | Agents | Blocker / degradation | Est. days | Depends on |
|---:|---|---|---|---:|---|
| 1 | `random()` is not seeded by the fit | 01–07 | Degradation for all seven, including both shipped agents | 0.5 + 3 | — |
| 2 | Bridge `anofox-statistics` MAP+Laplace fits into the draws contract | 02 blocker; 01, 04, 06 degradation | Unblocks 02 outright; gives 01/04/06 an honest interim path | 6–9 | statistics exposing the vcov matrix |
| 3 | F5 payer-alive (BG/NBD) | 05 | **Shipped.** Agent 05 unblocked | 8–12 | — |
| 4 | Random slopes + learned pooling scale | 06 blocker; 03 degradation | Blocker for 06 — the interaction-column workaround left 10 of 12 intervals not covering | 12–18 | gap 6 for the learned scale |
| 5 | Per-group variance | 04 | Blocker for the tail question, which is why agent 04 exists | 5–8 with gap 4 | gap 4 |
| 6 | NUTS engine (`nuts-rs` adapter) | none directly | Dependency of gaps 4, 5, 7 | 8–12 | — |
| 7 | F1 hierarchical negative binomial | 01 | Blocker natively; degraded path via gap 2 | 12–20 | gap 6 |
| 8 | F4 payment delay, native | 04 | Degradation — `pooled_gaussian` on log-delay is already a lognormal model | 8–12 | gaps 4, 5 |
| 9 | F6 hierarchical elasticity, native | 06 | Degradation once gap 4 lands — elasticity is a log-log linear model | 6–10 | gap 4 |
| 10 | `conjugate_anomaly` has no `as_differentiable` | none | Correctness gate, not a feature | 2–3 | — |
| 11 | `Readiness::worst` downgrades the whole fit | 01, 07 | Degradation, sharply worse at thousands of groups | 3–5 | — |
| 12 | No prior-predictive check (BR-11) | 01–07 | Degradation; a pre-fit gate agents cannot run today | 4–6 | — |
| 13 | No `anofox-scenario` integration (BR-9) | 01–07 | Degradation; branching a draws table is the agent's job | 5–8 | scenario's catalog API |
| 14 | Group parallelism, streaming sufficient statistics, lazy draw emission | 01, 07 | Degradation, only past the measured ceilings | 8–14 | — |

Two ranking notes worth stating rather than leaving implicit.

**Gap 1 outranks everything despite unblocking no agent.** Every documented
posterior-predictive and forward-simulation recipe draws its noise from DuckDB's
`random()`, which the fit's `seed` does not touch. Measured on this repo: the same fit
returns `model_id b4557ce9268692d3` every time, while two predictive runs over its
draws returned **0.4224** and **0.5518**. So the fit is reproducible and everything
downstream of it is not, unless the caller happens to issue `setseed()` — which no
recipe in `GUIDE.md` does. The audit story in `DRAWS_CONTRACT.md` — "an auditor can
re-run a recommendation and check it" — is false for any agent that simulates forward,
which is all of them. Half a day stops the bleeding; about three days fixes it
properly. Nothing else here has that ratio.

**Gap 6 unblocks no agent on its own and is still near the top.** Gaps 4, 5 and 7 —
three agents between them — all bottom out in a hierarchical variance parameter whose
posterior is not conjugate and for which Laplace is known to be poor. Sequencing NUTS
late means sequencing three agents late.

## 2. Phases

### v0.2 — Trustworthy, and bridged

*Make what already ships defensible, and reach four agents without writing a new
family.* Gaps 1, 2, 10, 11 plus the quick wins. **14–20 engineer-days.**

The rationale is leverage. The bridge is the only item on this roadmap that touches
four agents for under ten days, because `anofox-statistics` has already paid for the
likelihoods: censored AFT (Weibull, lognormal, log-logistic, exponential), negative
binomial and Gamma GLMs, a mixed-effects GLM with random intercepts and per-group
BLUPs, and explicit Gaussian priors with `vcov = 'laplace'` throughout. What it lacks
is a posterior — it returns a point estimate and a standard error. What this extension
has and statistics does not is the draws contract, the diagnostics, the refusal path
and the calibration harness.

Writing F2 from scratch means re-deriving a censored AFT likelihood and gradient that
already exists, tested, two repositories over. Bridging writes the seam once and gets
four likelihood families through it.

`as_differentiable` on `conjugate_anomaly` unblocks no agent and is in v0.2 anyway: it
turns the Laplace engine into an independent check on a family whose posterior we know
in closed form. Per `AGENTS.md`, two independent derivations of one distribution is the
strongest correctness gate available, and it exists for F3 and not for F7.

### v0.3 — The hierarchies that have to be Bayesian

*The models where a MAP estimate and a standard error are genuinely not enough.*
Gaps 6, 3, 4, 5, 9. **30–45 engineer-days**, and the softest estimate on this page.

F5 goes first inside the phase despite being independent of NUTS: it is the cheapest
remaining *full* agent unblock, needs no new engine, and is the only item here that can
be built in parallel with the NUTS adapter without a merge conflict.

The hierarchical Gaussian family comes after NUTS because what agent 06 needs is
precisely what Laplace approximates badly — a variance parameter near zero, where a
Gaussian approximation on the unconstrained scale is least honest and where SBC will
say so.

### Later

Not scheduled, and not ranked against each other until a customer or an agent forces
the order.

- **F1 native** (gap 7), deferred on the expectation that the bridge over
  `glmm_fit_agg` with a `negbinomial` family covers agent 01 adequately. If SBC or
  field use shows the interval too tight where it matters — thin SKUs, which is most of
  them — this moves up sharply.
- **F4 native** (gap 8). A lognormal delay model is a Gaussian model on `log(delay)`,
  so `pooled_gaussian` already substitutes and the v0.3 family covers per-segment
  spread. A native Gamma-delay family buys the Gamma branch and little else.
- **Prior-predictive checks** (gap 12) and **`anofox-scenario` integration** (gap 13).
- **Scale work** (gap 14): rayon across groups for `conjugate_anomaly`, streaming
  sufficient statistics rather than buffering the input relation, lazy per-chunk draw
  emission. All three are recorded in `SCALABILITY.md`. None is hit by the workloads
  the shipped families target; the BR-1 acceptance case completes in ~4 s.

## 3. Decisions to take before building

Three questions where committing to the wrong answer costs weeks.

### 3.1 Is F2 worth building, given statistics already has censored AFT?

**Recommendation: no. Build the bridge, and treat it as a shipped feature rather than a
stopgap.**

`anofox_stats_aft_fit_agg` fits Weibull, lognormal, log-logistic and exponential AFT
with right censoring, accepts explicit Gaussian and Laplace priors, and defaults to
`vcov = 'laplace'` — the curvature of the log posterior at the mode. That is a MAP
estimate plus its observed information: a Laplace posterior in everything but the
output format. The missing piece is not mathematics, it is the last step of our own
Laplace engine — sample the multivariate normal at the mode and emit the draws.

One seam, four agents. No new family here approaches that ratio.

**Three costs, stated plainly.**

*The covariance matrix is not reachable from SQL.* `anofox-stats-core`'s
`glm_engine/laplace.rs:33` holds the full `pub vcov: Mat<f64>`, but the aggregates
return only `std_errors DOUBLE[]` — the diagonal. Verified: `aft_aggregate.cpp` has
zero occurrences of `vcov`. Sampling from the diagonal treats every coefficient as
independent, dropping the intercept/slope correlation that dominates a predictive
interval, and would produce intervals wrong in a way no diagnostic here would catch.
**The bridge is not viable until statistics exposes the matrix.** That is a cross-repo
change and the largest risk in the v0.2 plan.

Two ways out: a `vcov_matrix DOUBLE[][]` field on the returned struct, or — cleaner —
anofox-bayes takes a Cargo dependency on `anofox-stats-core` and calls the fit
in-process, with no SQL round trip and no serialisation of a `p × p` matrix. The second
is the only option that lets the bridge participate in `model_id`, since the data
fingerprint must be computed over the rows the fit actually read.

Worth noting: **anofox-bayes currently depends on neither `anofox-regression` nor
`anofox-stats-core`** — everything in `crates/anofox-bayes-core/` is hand-rolled. HLD §3
("reuse before rebuild") assumed otherwise. The bridge is the first place that
assumption gets tested, and this decision is really a decision about whether HLD §3 was
right.

*A bridged fit carries a weaker warranty and must say so.* A Laplace posterior is a
Gaussian approximation. The catalog promises every family is SBC-calibrated and
PyMC-checked; a bridged fit has neither unless we build them. The bridge is **not** an
escape from the `AGENTS.md` bar — each bridged likelihood still needs its SBC suite and
parity test, and any that fails SBC is documented as such rather than quietly shipped.
What the bridge saves is the likelihood, the gradient, the mode-finding and the
censoring logic: roughly half the work of a family, not all of it. The 6–9 day estimate
assumes the calibration work is still done.

*Refusal semantics must be mapped, not inherited.* `converged BOOLEAN` and statistics'
`NaN` conventions have to become `FitStatus` values meaning what the draws contract
says. Statistics' "every row censored → not identified" must arrive as `degenerate`.

**What would change this:** if exposing the vcov matrix proves expensive across the
repo boundary, a native F2 at 10–14 days becomes competitive — but it unblocks one
agent instead of four.

### 3.2 Is NUTS genuinely required for F5?

**Recommendation: no. Build F5 on MLE + Laplace, and record why.**

> **Settled, and the answer held.** F5 ships on MLE + Laplace. The SBC suite is the
> evidence: over 1 024 replications under proper log-normal priors, the rank
> histograms for all four parameters are uniform — chi-squared 13.0 (`alpha`), 13.3
> (`b`), 18.3 (`a`), 29.0 (`r`) against a 37.7 critical value at 15 degrees of
> freedom, with every slope below 0.04 — and they stay uniform down to 100 customers
> and up to 4 000. There is nothing for NUTS to fix here.
>
> Building it did surface two defects in the **Laplace engine**, both of which would
> have bitten any future family with a likelihood evaluated per observation, and
> neither of which the shipped families could expose because their gradients come from
> precomputed sufficient statistics. The Newton search now carries a trust region — a
> backtracking line search only shrinks a step that made things *worse*, so a leap into
> a distant local optimum was accepted; measured, F5's first step from the conventional
> start moved `ln a` by +14.6 onto a ridge 248 log-units below the true mode — and a
> relative stopping rule, because an absolute `1e-8` gradient tolerance is below the
> rounding error of a sum over a few thousand rows. Both are in `CHANGELOG.md`.

BG/NBD's likelihood is elementary — a handful of terms in `r, α, a, b` and the
per-customer sufficient statistics `(x, t_x, T)` — and `P(alive)` is closed form given
the parameters. Four population-level parameters with thousands of customers informing
them is the regime where a Gaussian approximation at the mode is at its best. The SBC
suite is the arbiter: if the ranks are uniform, NUTS on four parameters buys nothing at
a large multiple of the runtime.

**Pareto/NBD is a separate question and the answer is no.** Its likelihood needs the
Gaussian hypergeometric `₂F₁` — a special function to own and test — and, decisively,
`P(alive)` under Pareto/NBD is not expressible in SQL. That breaks the pure-SQL
rescoring that makes agent 05 cheap to operate: score today's dunning list from
yesterday's draws without refitting. See §5.

Residual risk: BG/NBD's likelihood surface is known to have flat ridges and boundary
solutions on some datasets, and a mode search landing on a boundary produces a
curvature that is not a posterior. The refusal path must catch that — a boundary
solution returns `degenerate`, not an interval.

> **Handled.** The family finds its own mode at compile time, before any engine runs,
> and applies four tests to the point: inside the admissible range, stationary (at a
> tolerance that scales with the number of customers), curvature that factors, and a
> marginal narrow enough to be an interval. The last of those is what catches the
> flat ridge, which the other three pass. Failing any is `degenerate` with `NULL`
> draws and a reason naming which. The fixture is a base in which no repeat buyer has
> ever gone quiet — a subscription book snapshotted at renewal — where the likelihood
> is maximised only as the dropout probability goes to zero.

### 3.3 Random slopes and per-group variance: extend `pooled_gaussian`, or a new family?

**Recommendation: a new family. Keep `pooled_gaussian` exact.**

`pooled_gaussian`'s warranty is that its posterior is closed form — no approximation,
no sampler, and the exact and Laplace engines cross-check each other. That warranty
survives exactly one of the three requested changes:

| Change | Still conjugate? |
|---|---|
| Random slopes at a **fixed** `pool_scale` | Yes — more columns with a Gaussian prior, nothing else |
| Per-group `sigma` | No — group variances with a shared hyperprior are not conjugate |
| **Learned** `pool_scale` | No — this is the hierarchical variance parameter NUTS is for |

So the split is not arbitrary. Random slopes at a fixed scale are a design-matrix change
and should land in `pooled_gaussian`: 3–5 days, no engine work, no new calibration
story, and it removes agent 06's interaction-column-plus-hand-tuned-`beta_scale` hack.
It does **not** fix the coverage problem — the scale is still an analyst dial — but it
makes the model say what agent 06 means, which is a precondition for diagnosing the
rest.

The other two belong in a new family served by Laplace and NUTS. Folding them into
`pooled_gaussian` would make one family bimodal: sometimes exact and instant, sometimes
sampled and slow, decided by a config slot several levels deep. A caller reading
`__engine__` off a draws table would find it varying by configuration under one family
name. The catalog's premise is that a family is a fixed set of parameterisation
decisions with one calibration story; a family with two warranties has neither.

The `beta_scale` complaint resolves inside the same decision: it shrinks every slope
toward zero, which is a claim about the world ("effects are small") nobody asked to
make. Shrinking group deviations toward a common mean is the random-slope structure.

**Naming is open.** `hierarchical_gaussian` is the obvious candidate and not obviously
right, since `pooled_gaussian` is also hierarchical. Decide before the config surface is
public — a family id feeds `model_id`, so renaming one invalidates every persisted fit.

## 4. Quick wins

Under a day each, independent of everything above, each fixing something currently
*wrong* rather than merely absent.

**`setseed()` in every recipe that uses `random()`** — ~0.5 d. Measured above: the same
draws table yields 0.4224 and 0.5518 on consecutive runs. `diagnostics.test` already
seeds, so the pattern exists in the repo and simply was not applied where it matters.
Add `setseed()` to every recipe, state the requirement in `GUIDE.md` and
`DRAWS_CONTRACT.md`, and add a sqllogictest that runs a predictive recipe twice and
asserts identical output.

That is a stopgap. The fix — about three days, in v0.2 — is a pure scalar
`anofox_bayes_std_normal(seed, row_id, draw)` hashing its arguments into a
deterministic standard normal. That makes the predictive reproducible without depending
on session-global state, which is the same reason this extension keeps no state of its
own: a recipe whose correctness depends on a `SET` the caller might not have issued is
a recipe that will silently be wrong somewhere.

**Record the family in the draws table** — **done.** `__family__` carries the catalog
F-number (`3` = `pooled_gaussian`, `7` = `conjugate_anomaly`), decoded in SQL by
`anofox_bayes_family_text`. The `value` column is `DOUBLE`, so the name could not
travel; reusing the BRD's existing F-numbering rather than inventing a second one keeps
a family to one identity. Append-only, like `FitStatus` and `EngineKind`, and
non-breaking — `__schema_version__` did not move.

**Record the data fingerprint** — **not shipped, and not on a half-day budget.**
`model_id` is a digest *over* the fingerprint, so recovering it from the table would
need the fingerprint itself, and the fingerprint is a hex digest with no honest home in
a `DOUBLE`: 53 bits of mantissa against a 64-bit digest means silent collisions, and a
fingerprint that silently collides is worse than an absent one, because its only use is
deciding whether a table describes a given relation. The lossless routes — a new
`VARCHAR` column, or overloading `group_id` on the metadata row — are both breaking, so
this now waits for the next schema version rather than shipping in a form that would
have to be un-promised. Recorded in `DRAWS_CONTRACT.md` alongside the workaround: with
`__family__`, `__engine__` and `__seed__` on the table, a caller holding the config can
re-derive `model_id` from a candidate relation and compare.

**Count the unready groups** — **done.** `__n_groups_unready__` reports how many groups
failed their own readiness check, beside the unchanged collapsed `__status__`.
`Readiness::worst` still returns the worst verdict and should — the doctrine that a fit
an agent must inspect is not 99.4 % trustworthy is right. What was missing was the
number telling the agent how much inspecting there is, and that is now on the table. It
remains a partial answer while gap 11 waits: `conjugate_anomaly` fits each group
independently and so counts exactly, while `pooled_gaussian` reaches one verdict over
one design and reports all of its groups when it refuses.

## 5. Non-goals

Recorded with reasons, because a non-goal without one gets relitigated every quarter.

**A general model specification language.** BRD non-goal, unchanged. The closed catalog
is what makes analytic gradients, per-family SBC and a bounded correctness liability
possible at all.

**Pareto/NBD.** See §3.2 — `₂F₁` is a special function to own, and `P(alive)` cannot be
expressed in SQL, which breaks the pure-SQL rescoring that makes agent 05 cheap to run.

**Writing an MCMC kernel.** `nuts-rs` is maintained by pymc-devs and battle-tested via
nutpie. We write the adapter and nothing below it.

**Gaussian processes, state-space models, time-varying parameters.** No agent needs
them; `anofox-forecast` covers time-series intervals.

**An `anofox_bayes_predict` table function.** Not a scheduling decision: DuckDB permits
at most one subquery parameter per table function, so a function taking both draws and
new rows cannot bind. The join recipe is the design; its executable specification is
`test/sql/posterior_predictive.test`.

**Autodiff.** The catalog is closed, so each gradient is written once and checked once
against finite differences away from the mode. A general mechanism would cost more to
audit than the derivatives it replaced.

**An async / job-style fit API.** BRD OQ-4, still deferred. v0.1's blocking table
function completes the BR-1 acceptance case in ~4 s. Build it when a customer hits a
timeout — and note that NUTS in v0.3 is the change most likely to produce that customer.

**GPU backends, warm-start refits, incremental updates.** No agent is constrained by fit
time today.

**Folding into `anofox-statistics`.** BRD G5: licensing optionality and the runtime
profile both argue for a separate extension. The §3.1 bridge is the opposite of folding
— it lets the two products stay separate while sharing mathematics that should only be
written once.

## 6. How the estimates were built

Every family estimate assumes the full `AGENTS.md` bar, which is the only bar applied
here so far. An estimate skipping it would be wrong by about half.

| Step | Typical days |
|---|---:|
| Family module: config validation, parameterisation, compile | 2–4 |
| Closed-form posterior test, or analytic gradient checked against finite differences **away from the mode** | 1–3 |
| SBC suite under a proper prior, in `sbc.rs::families` | 1–2 |
| A `test/sql/` scenario a customer would actually run | 1 |
| PyMC parity test in `validation/` | 1–2 |

So roughly seven days is the floor for even a simple family. F1 and F4 are higher
because a hierarchical count model has an SBC suite that is slow to run and awkward to
make pass.

The two ranges to distrust most: **F1 at 12–20 days**, where the spread is really
uncertainty about how the negative-binomial dispersion parameter behaves under NUTS;
and **the NUTS adapter at 8–12 days**, a guess about an integration nobody here has
done. Both should be re-estimated after a spike rather than planned against.
