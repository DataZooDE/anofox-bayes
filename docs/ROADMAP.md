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
| 2 | ~~Bridge `anofox-statistics` MAP+Laplace fits into the draws contract~~ **seam done, F2 shipped** | 02 unblocked; 01, 04, 06 still degraded | The seam and censored AFT (F2) have landed; the remaining likelihoods are small additions, costed in §3.1 | 6–9 | — |
| 3 | F5 payer-alive (BG/NBD) | 05 | Blocker | 8–12 | — |
| 2 | Bridge `anofox-statistics` MAP+Laplace fits into the draws contract | 02 blocker; 01, 04, 06 degradation | Unblocks 02 outright; gives 01/04/06 an honest interim path | 6–9 | statistics exposing the vcov matrix |
| 3 | F5 payer-alive (BG/NBD) | 05 | **Shipped.** Agent 05 unblocked | 8–12 | — |
| 4 | Random slopes + learned pooling scale | 06 blocker; 03 degradation | Blocker for 06 — the interaction-column workaround left 10 of 12 intervals not covering | 12–18 | gap 6 for the learned scale |
| 5 | Per-group variance | 04 | Blocker for the tail question, which is why agent 04 exists | 5–8 with gap 4 | gap 4 |
| 6 | NUTS engine (`nuts-rs` adapter) | none directly | Dependency of gaps 4, 5, 7 | **done** | — |
| 7 | F1 hierarchical negative binomial | 01 | Blocker natively; degraded path via gap 2 | 12–20 | gap 6 — **met** |
| 8 | F4 payment delay, native | 04 | Degradation — `pooled_gaussian` on log-delay is already a lognormal model | 8–12 | gaps 4, 5 |
| 9 | F6 hierarchical elasticity, native | 06 | Degradation once gap 4 lands — elasticity is a log-log linear model | 6–10 | gap 4 |
| 10 | ~~`conjugate_anomaly` has no `as_differentiable`~~ **done** | none | Correctness gate, not a feature | 2–3 | — |
| 11 | `Readiness::worst` downgrades the whole fit | 01, 07 | Degradation, sharply worse at thousands of groups | 3–5 | — |
| 12 | No prior-predictive check (BR-11) | 01–07 | Degradation; a pre-fit gate agents cannot run today | 4–6 | — |
| 13 | ~~No `anofox-scenario` integration (BR-9)~~ **closed, as documentation** | 01–07 | The two extensions already compose; see §5 | 0.5 | — |
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

**Gap 6 is done.** The `nuts` engine ships as a pinned `nuts-rs` adapter isolated in
`crates/anofox-bayes-core/src/engines/nuts.rs`, certified against `pooled_gaussian`'s
closed-form posterior at a Monte-Carlo-derived tolerance and by its own SBC suite. It
is the first engine to produce a Markov chain, so it is also the first to make R̂ and
`__divergent__ ` mean anything. The 8–12 day estimate below was a guess about an
integration nobody here had done; the integration itself was small, and what took the
time was the certification, which is the same shape as every other estimate on this
page. Two things the estimate did not anticipate, recorded because they are the parts a
future upgrade will trip over: `nuts-rs` rejects a starting point whose gradient is
exactly zero — which is what a family's `initial()` returns — and its per-draw
statistics are reachable only through a struct its crate root does not export.

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

**Done.** Two findings came out of it, both worth carrying forward.

*The gate is weaker than it looks against shape errors.* A missing log-Jacobian shifts
`alpha_n` by one, which is an `O(1/n)` perturbation — the same order as the Laplace
approximation error the comparison is measuring. At n = 400 it moves `sigma` by 0.25 %,
inside any tolerance loose enough not to flake. Verified by mutation: dropping the
Jacobian leaves the engine-agreement test green. What catches it is a separate test
that differences `logp` between two points and compares against the closed-form density
ratio. **Any future family's engine-agreement test needs that companion**, or the
Jacobian is untested.

*Laplace is poor on this family for exactly the groups it exists to look at.* Because
`conjugate_anomaly` fits each group independently, the relevant sample size is the
group's own. The spread of `mu` is too narrow by `1 - sqrt((n-3)/n)` — 0.4 % at n = 400,
**29 % at n = 6** — and `sigma` is worse (44 % on the spread at n = 6). Measured against
the closed form to four digits, so it is a known cost rather than a suspicion. `exact`
stays the default, and if the bridge in gap 2 ever routes a thin-group model through a
Laplace posterior, this is the number to remember.

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
- **Prior-predictive checks** (gap 12). (Gap 13, the `anofox-scenario` integration,
  turned out to need no code at all — see §5.)
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

*The covariance matrix is not reachable from SQL.* **Settled, and the resolution was
not quite either of the two options anticipated.**

The in-process route was taken: anofox-bayes now has a Cargo git dependency on
`anofox-stats-core`, pinned to an exact revision. That was necessary but not
sufficient, because `fit_aft` **discards the covariance in-process too** — it computes
a full `LaplaceInference` internally and then keeps only slices of its diagonal on
`AftInference` (`std_errors`, `z_values`, `ci_*`). The matrix is no more reachable from
the returned struct than it was from SQL.

It is reachable from the *primitives*, all of which are public: `AftDistribution`
exposes the log density, the log survival and their first two derivatives;
`glm_engine::laplace::inference` is public and returns the full `vcov`;
`PriorSpec::precision` gives the prior's contribution. So `bridge.rs` reassembles the
observed information at the mode `fit_aft` returned and hands it to statistics' own
inference routine. **The reassembly is cross-checked rather than trusted**: its
diagonal must reproduce the reported `std_errors`, and does so to the last bit — which
is also a standing guard against the pinned revision drifting.

Measured, on a design with the covariate away from zero: `corr(intercept, slope) =
−0.998`, and the predictive standard deviation of the linear predictor is **24.9×
larger** from the diagonal alone than from the full matrix. Confirmed by mutation that
this is invisible to everything else — see below.

*Still worth doing upstream:* adding `pub vcov: Mat<f64>` to `AftInference` (and to
`GlmInferenceResult`) would delete the reassembly and its cross-check entirely. It is a
one-field change and the natural companion to a prior slot on the AFT scale, which is
the other thing the bridge wants from `anofox-statistics` — see the SBC note below.

Worth noting: **anofox-bayes previously depended on neither `anofox-regression` nor
`anofox-stats-core`** — everything in `crates/anofox-bayes-core/` was hand-rolled.
HLD §3 ("reuse before rebuild") assumed otherwise, and this was the first test of it.
The verdict is in §3.1's postscript below.

*A bridged fit carries a weaker warranty and must say so.* **Done, with one gap
recorded rather than closed.** `__engine__` reads `laplace` on every bridged fit and
`as_exact` returns `None`, so asking for an exact posterior is an error rather than a
silent substitution. The SBC suites are built and pass:

| Suite | χ² (15 df, threshold 37.7) | Status |
|---|---|---|
| `exponential`, n = 200 | 13.0 / 12.8 | **Complete certificate** — every free parameter properly priored |
| `weibull`, n = 200 | 14.0 / 14.2 | Coefficients only; **`sigma` uncertified** |
| `exponential`, n = 25, heavily censored | 7.2 / 9.7 | Calibrated on a thin cohort too |

The gap: `anofox-stats-core`'s AFT accepts Gaussian priors on the **coefficients only**,
and SBC cannot draw a truth from an improper prior — so for the three distributions
that estimate `sigma`, `sigma` itself is not certified. `exponential` fixes it at 1,
which is what makes that suite complete and therefore a genuine certificate for the
seam. Recorded in `docs/THEORY.md` §5. Closing it needs a prior slot on the scale
upstream.

The thin-cohort result is worth carrying forward for the opposite reason to the note in
§2: it is *good*, where `conjugate_anomaly`'s Laplace is bad (29 % too narrow at n = 6).
A regression coefficient's posterior is near-Gaussian at modest n; a variance
parameter's is not. So the §2 warning about routing thin groups through Laplace applies
to variance parameters, not to bridged coefficients.

*Refusal semantics must be mapped, not inherited.* **Done, with a test per row.**
Every-row-censored → `degenerate`; too few rows → `insufficient_data`; rank-deficient
at the mode → `degenerate`; non-convergence → `failed`. A non-positive duration or a
non-binary event indicator stays a **request error naming the column**, since it is a
malformed request rather than weak evidence. An unmapped upstream variant reports
itself rather than being graded.

Note on `converged BOOLEAN`: in-process it does not exist. A non-converged AFT fit never
returns — `newton` turns it into `ConvergenceFailure` — so the boolean the SQL aggregate
publishes reaches this crate as an error. `Failed` is the right verdict: there is no
mode, so there is no posterior, not merely a poor one.

**The finding that generalises.** Replacing the posterior precision with its own
diagonal — the change that would follow from consuming `std_errors` instead of
reassembling the matrix — leaves **219 of 220 unit tests and all six SBC suites green**.
SBC ranks parameters one at a time, so it tests marginals, and marginals are exactly
what a diagonal preserves. Only a test written on a *function of several parameters at
once* catches it. **Every family needs at least one assertion on a linear combination
of its parameters**, not only on each parameter separately.

### 3.1a What the remaining likelihoods cost

The seam is `GaussianApproximation` / `GaussianBlock`: a family supplies a mode and a
full precision matrix per independent block, and the Laplace engine does the rest. The
matrix is carried by the seam rather than by each family, so a future bridge cannot
reintroduce a diagonal by accident.

| Likelihood | Upstream entry point | Extra work | Est. days |
|---|---|---|---:|
| Censored AFT (F2) | `fit_aft` | **done** | — |
| Negative binomial GLM | `fit_negbinomial` | Reassemble the IRLS information (`X'WX + P`, weights from the family) rather than the AFT Hessian; the dispersion parameter is estimated outside the IRLS loop and is **not** in the curvature, so it is either conditioned on or the family reports it without a posterior. That question is the whole cost. | 3–5 |
| Gamma GLM | `fit_gamma` | As above minus the dispersion question, which Gamma answers the same awkward way. Same design matrix machinery, so it lands with negbinomial or immediately after. | 1–2 |
| Mixed-effects GLM (`fit_glmm`) | `fit_glmm` | The information includes the random-effect block, so "one block per group" no longer holds — the whole fit is one correlated block of size `p + n_groups`, and the Cholesky is `O((p+G)^3)`. Needs a sparse or blocked path before it is usable at the group counts agent 01 has. Also the case where a Laplace approximation to a variance component is known to be poor, so SBC is likely to fail and that failure is the useful output. | 6–10 |

Two shared costs apply to each: an SBC suite (1–2 days) and a `test/sql/` scenario
(1 day), which the estimates above exclude. And the same upstream limitation binds
throughout — priors on coefficients only — so each bridged likelihood will certify its
coefficients and leave its dispersion or variance parameter uncertified until
`anofox-statistics` grows the slot.

**What would change this:** nothing now. The bridge is built and F2 is shipped; a
native F2 is no longer competitive at any price.

### 3.1b Postscript: was HLD §3 right?

"Reuse before rebuild" was **right, and for a more specific reason than it states.**

What the bridge actually reused was the *mathematics that is expensive to get right and
cheap to verify*: four censored likelihoods with their first and second derivatives, a
damped Newton search, and the prior-to-penalty translation. Re-deriving those would have
been most of a fortnight and every line of it would have needed its own finite-difference
check.

What it did **not** reuse, and could not, is anything shaped like an *interface*. The
returned struct threw away the one object the bridge needed; the refusal vocabulary had
to be re-mapped rather than adopted; the prior surface stops exactly where a Bayesian
consumer needs it to continue. Roughly a third of the work was assembling a matrix from
public primitives that already existed as a private local variable one crate over.

So the rule that survived contact is narrower than HLD §3's wording: **reuse the
mathematics, expect to re-derive the interface, and cross-check the seam against
something the upstream crate publishes.** The `std_errors` agreement check is what makes
this a dependency rather than a fork — it is a standing assertion that both crates still
mean the same thing by the same likelihood, and it will fail loudly if the pinned
revision is bumped to something that does not.

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

### Known gap: the counterfactual suite does not run in CI

`test/sql/scenario_counterfactual.test` is a specification rather than a running test.
`require anofox_scenario` *skips* rather than fails when the extension will not load,
anofox-scenario's binaries are unsigned, and `test/unittest` is a Catch binary with no
`-unsigned` flag. A skipped suite reports as a pass, so `make test_scenario` now exits
non-zero on a skip rather than letting it read as green.

Fixing it needs one of two things, neither in this repository: signed anofox-scenario
binaries, or a `test/unittest` that permits unsigned extensions. What has been verified
by hand meanwhile: every statement in the suite executes without error with both
extensions loaded in the shell, and the composition it depends on — fitting from an
attached scenario catalog, with `model_id` diverging because the data fingerprint
covers the branch's rows — is directly checked.

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

**An `anofox-scenario` integration surface (BR-9, gap 13).** Investigated and closed
without code, which was the surprise. `anofox-scenario` has no registration or
provider API to register against: its entire public surface is SQL — `scenario_create`,
`ATTACH … (TYPE scenario)`, `scenario_diff` — and its only `extern "C"` symbols are the
DuckDB extension entry points. Nor does it need one. This extension publishes exactly
one artefact, a table of draws, and that extension branches exactly one thing, a table;
they already meet in the catalog.

The binding constraint from the `anofox_bayes_predict` entry above is worse here, not
better: a `scenario_compare(draws, baseline, counterfactual)` needs *three* relations
where DuckDB allows one subquery parameter. A string-argument variant that opened the
tables by name would have to re-implement, inside a single-threaded table function, the
join the caller can already write — and would have to hard-code knowledge of scenario
catalogs to do it, coupling the release cycles of two separately licensed products for
no gain.

What actually needed writing was the *discipline*, because two of its three rules are
easy to get wrong and neither fails loudly: branch the assumptions rather than the
draws (editing a posterior is fabricating evidence), and key the simulation noise on
the row rather than on anything the branch changes. Get the second wrong and the
comparison still returns the right estimate with an interval several times too wide —
measured in `test/sql/scenario_counterfactual.test`, `+149 [+81, +218]` becomes
`+149 [−21, +318]`, which is the difference between a decision and a shrug. The
pattern is documented in `GUIDE.md` and its executable specification is that test.

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
