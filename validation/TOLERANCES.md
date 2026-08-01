# Tolerances

Every number in `tests/_support.py` is derived here. If you change one, change
this file in the same commit — a tolerance without a reason is a tolerance that
gets loosened until the suite is green.

## The unit

Both sides of the comparison are Monte Carlo, so the tolerances are expressed in
the only scale-free unit available: **the reference posterior's own standard
deviation**.

| statistic | delta measured as | why |
|---|---|---|
| `mean` | `abs(ext − ref) / sd_ref` | absolute units would need one tolerance per parameter; a coefficient in euros and a rate in claims-per-consignment share this one |
| `sd` | `abs(ext − ref) / sd_ref` | a relative tolerance is the natural one for a scale |
| `q05`, `q95` | `abs(ext − ref) / sd_ref` | interval endpoints live on the parameter's scale, so the same unit as `mean` applies |

## The floor: Monte Carlo error

Neither side is exact, so no tolerance can be tighter than the noise.

**Extension side.** It draws i.i.d. from a closed-form posterior, so with
`EXTENSION_DRAWS = 40_000`:

* `MCSE(mean) = sd / sqrt(40_000) = 0.005 sd`
* `MCSE(q95) ≈ sqrt(0.05·0.95 / 40_000) / f(q95) ≈ 0.011 sd` for a near-normal
  posterior (`f` is the density at the quantile, ≈ 0.103/sd there)

**Reference side.** NUTS, 4 chains × 5 000 draws after 2 000 tuning. On the F7
posteriors and the ungrouped F3 posterior, ESS is close to the raw 20 000 draws;
on the F3 panel the intercept/group-effect block is only ridge-identified, and
ESS lands nearer 5 000–10 000.

* `MCSE(mean) = sd / sqrt(ESS)` → 0.007 sd (F7) to 0.014 sd (F3 panel)
* `MCSE(sd) ≈ sd / sqrt(2·ESS)` → 0.5 % (F7) to 1.0 % (F3 panel)
* tail quantiles are the noisiest, roughly 2× the mean's MCSE

Adding the two sides in quadrature gives a floor of roughly **0.012 sd** on a
mean and **0.026 sd** on a 5 %/95 % endpoint for F7, about double that for the
F3 panel.

## The chosen values

```python
F7_TOL = Tolerance(mean=0.05, sd=0.05, quantile=0.09)   # conjugate_anomaly
F3_TOL = Tolerance(mean=0.07, sd=0.07, quantile=0.12)   # pooled_gaussian
```

### F7 `conjugate_anomaly` — mean 0.05 sd, sd 5 %, quantiles 0.09 sd

Roughly **3.5–4× the combined MCSE floor**. That factor is the smallest one that
does not put a deterministic-but-jittery statistic (a tail quantile of a skewed
`sigma` posterior) within one fluctuation of the limit, and it is small enough
that the negative controls trip by two orders of magnitude — see below.

`sd` gets a 5 % relative tolerance rather than an sd-unit one because `sd` is
itself the scale; 5 % is ~10× the 0.5 % MCSE on the reference's own sd estimate.

Measured worst case across the 9 F7 comparisons: **55 % of tolerance**
(`q95` of `sigma[BRE-ANT]`). Nothing else exceeds 50 %.

### F3 `pooled_gaussian` — mean 0.07 sd, sd 7 %, quantiles 0.12 sd

Looser than F7 for one structural reason and one substantive one.

**Structural.** The panel's intercept and its six store effects are identified
only by the `pool_scale` ridge; their posterior is a narrow ridge in a
seven-dimensional space, NUTS autocorrelation is materially higher, and the
reference MCSE roughly doubles against F7.

**Substantive, and worth understanding before touching these numbers.** The
extension's residual scale is systematically short by `sqrt((n − k)/n)`, where
`k` counts the coefficients under a flat prior (see README, "What this found").
At the sizes used here that is **1.9 %** (ungrouped, n=80, k=3) and **1.7 %**
(panel, n=120, k=4). Every coefficient inherits it, because
`beta | sigma² ~ N(b_n, sigma² A⁻¹)`.

So the 7 % `sd` tolerance is deliberately ~3.5× that systematic deficit. It
would be tempting to tighten it to 1 % and let the parity tests catch the bug —
but 1 % is *at* the panel's own MCSE on `sd`, so the suite would flake as often
as it would be right. Instead:

* the coefficient parity tests pass, with the deficit absorbed and disclosed;
* `test_residual_scale_deficit_is_exactly_the_flat_dimension_count` measures the
  deficit directly, to a bound of 0.003 — far tighter than any tolerance here,
  because `E[sigma]` is the statistic with the least reference-side noise;
* `test_coefficient_intervals_inherit_the_same_deficit` shows it propagates,
  with a looser 0.015 bound that respects the ~1 % MCSE on a NUTS `sd`;
* the two `sigma` parity tests are `xfail(strict=True)`, so the discrepancy is
  on the record and its repair is detected.

Measured worst case across the 15 F3 comparisons, excluding the two `xfail`ed
`sigma` ones: **66 % of tolerance** (`q95` of `beta[month]`).

## Families whose extension side is also a Markov chain

F7 and F3 draw i.i.d. from a closed form, so the extension's Monte Carlo error
is `sd/sqrt(draws)` and 40 000 draws makes it negligible by construction. **F8
and F1 are served by NUTS**, so that is no longer true: the extension side is
autocorrelated and its error is `sd/sqrt(ESS)`.

ESS is therefore **measured, not assumed**, on both sides, and the measurement
is itself asserted — `test_the_extension_side_is_well_enough_mixed_to_be_compared`
in each suite fails if the extension's ESS drops below the value the tolerance
was derived from. A tolerance derived from an ESS nobody re-checks is a
tolerance that silently becomes wrong the first time mixing regresses.

Both use `EXTENSION_NUTS_DRAWS = 4_000` × 4 chains after 1 000 warmup, against
the reference's 4 × 5 000 after 2 000. THEORY.md says 4 × 2 000 is what clears
this extension's own R̂ gate; clearing the gate is the floor for a *usable* fit,
not for one that is refereeing another implementation, so the budget is doubled.

### F8 `varying_variance_gaussian` — mean 0.09 sd, sd 6 %, quantiles 0.17 sd

Measured ESS (worst parameter of the seventeen, both `intercept` — the
ridge-identified block, as in F3):

| side | ESS (bulk) | MCSE(mean) |
|---|---:|---|
| extension, 4 × 4 000 | 2 275 | `1/sqrt(2275)` = 0.0210 sd |
| reference, 4 × 5 000 | 5 078 | `1/sqrt(5078)` = 0.0140 sd |

In quadrature: **0.0252 sd** on a mean. Then, as in the F7 derivation,
`MCSE(sd) = MCSE(mean)/sqrt(2)` = 0.0178, and a tail quantile is roughly twice a
mean's = 0.0505 sd.

At the same 3.5× the rest of this file uses, that gives 0.088 / 0.062 / 0.177,
rounded to **0.09 / 0.06 / 0.17**.

Measured worst case across the 17 comparisons: **61 % of tolerance**
(`q95` of `sigma_spread`, the parameter with the least data behind it — six
groups is six observations for a spread-of-spreads).

### F1 `hier_negbin` — mean 0.11 sd, sd 8 %, quantiles 0.23 sd

Same arithmetic, worse mixing. `tau` is the parameter the intercept/group-offset
ridge runs through and it sets the floor on the extension side; the reference is
also slower here than anywhere else in the suite.

| side | ESS (bulk) | MCSE(mean) |
|---|---:|---|
| extension, 4 × 4 000 | 1 734 | 0.0240 sd |
| reference, 4 × 5 000 | 3 026 | 0.0182 sd |

Quadrature: **0.0301 sd**; `sd` 0.0213; quantiles 0.0602 sd. At 3.5× that is
0.105 / 0.075 / 0.211, rounded to **0.11 / 0.08 / 0.23**.

Measured worst case across the 27 comparisons: **54 % of tolerance**
(`q95` of `tau`).

## Families served by an approximation, and why they need two tolerances each

F2 and F5 have **no closed form and no exact engine**. Their posterior *is* the
Laplace approximation — a Gaussian at the mode with the observed information as
its precision. THEORY.md puts it plainly for F2: "`laplace` is not a cross-check
on it — it is the fit."

A single comparison against NUTS would confound two different things:

1. is the **likelihood** right? (a bug)
2. is a Gaussian at the mode a good enough stand-in for the posterior? (a
   documented design decision)

One tolerance cannot answer both, and setting one loose enough for (2) destroys
its ability to answer (1) — the approximation error here is 10–100× the Monte
Carlo error, so a tolerance sized for it would hide almost any algebra bug.
So each family carries two.

### The algebra gate: `F2_EXACT_TOL`, `F5_EXACT_TOL` = 0.02 sd / 2 % / 0.04 sd

The reference is `_support.laplace_reference`: Nelder-Mead then BFGS on a log
posterior written out separately in NumPy, with the Hessian obtained by
differencing a **numerical** gradient. The extension uses damped Newton on an
**analytic** gradient and differences that. No shared code, no shared
derivation. Both sides then describe the same Gaussian, and the reference's
marginals are evaluated in closed form rather than sampled, so the reference
contributes no Monte Carlo error at all.

Floor, from the extension's 200 000 i.i.d. draws alone:

* `MCSE(mean) = 1/sqrt(200 000)` = **0.0022 sd**
* `MCSE(sd) = 1/sqrt(400 000)` = 0.0016
* `MCSE(q95) ≈ sqrt(0.05·0.95/200 000)/0.103` = 0.0047 sd

Measured discrepancy: **0.0033 sd** on the worst mean and 0.0089 sd on the worst
quantile — about 1.5–1.9× the pure Monte Carlo floor. The excess is not Monte
Carlo: it is the finite-difference Hessian's step size and the fact that two
different optimisers stop at two slightly different points.

The tolerance is set at **0.02 / 0.02 / 0.04**, roughly 6× the measured value and
9× the floor. That is deliberately more headroom than the 3.5× used elsewhere,
for a reason specific to this gate: the residual is dominated by a
finite-difference step size rather than by a random error, so it does not
average out and a tolerance sitting just above it would be re-tuned on the first
scipy upgrade rather than believed. It costs nothing in sensitivity — the
perturbations this gate exists to catch (a dropped censoring indicator, the
wrong `dist`, a changed prior) breach it by one to three orders of magnitude,
as the negative controls in both files demonstrate.

Measured worst case: **22 % of tolerance** (F2), **21 %** (F5).

### The approximation budget: `F2_TOL`, `F5_TOL`

These are **not** derived from Monte Carlo, and saying they were would be a
fiction — the MC floor is ~0.013 sd and the quantity being bounded is 5–15×
that. They are derived from the *measured* Laplace approximation error, at 2×,
which leaves room for it to drift with a scipy or PyMC bump without leaving room
for it to change character.

| | measured worst | tolerance | ratio |
|---|---|---|---|
| F2 mean | 0.062 sd (`beta[distance]`) | 0.12 | 1.9× |
| F2 sd | 2.4 % (`intercept`) | 0.05 | 2.1× |
| F2 quantile | 0.137 sd (`beta[distance]`) | 0.26 | 1.9× |
| F5 mean | 0.062 sd (`b`) | 0.15 | 2.4× |
| F5 sd | 4.9 % (`b`) | 0.11 | 2.2× |
| F5 quantile | 0.173 sd (`a`) | 0.38 | 2.2× |

**Only the coefficients are compared this way in F2.** `sigma` is not, for the
same reason F3's `sigma` is not: it carries a systematic discrepancy that a
tolerance either absorbs or flakes on, and the right treatment is to measure it
directly. See the next section.

## What this found

### F2's scale is over-confident, and it is the approximation, not the bridge

The extension's `log sigma` posterior sits **0.19 reference sd below** NUTS's and
is **2.4 % narrower**; on the natural scale `E[sigma]` is low by ~1.5 % with an
interval ~4 % too tight.

The valuable part is being able to say what it is *not*. The algebra gate above
shows the extension reproduces an independently computed mode and observed
information to 0.003 sd, and reproduces the intercept/slope correlation
(−0.9597) to four digits. So the bridge — the part THEORY.md flags as
load-bearing, where sampling from the diagonal instead of the full covariance
would be 25× wrong — is exact. What is left is the definition of Laplace: it
reports the **mode**, and a scale parameter's posterior is right-skewed, so its
mode sits below its mean and a Gaussian cannot reproduce its upper tail.

The direction is the one that matters. `sigma` sets the width of a transit-time
quantile, so a low `sigma` is a delivery promise tighter than the data supports —
over-confident, never under. That is the same direction as the F3 defect this
suite found earlier, arrived at for a completely different reason, and it is
worth stating together with THEORY.md §5's existing caveat that `sigma` is not
SBC-certified for the distributions that estimate it. **These are two
independent gaps in the same parameter**: SBC cannot certify it because the
upstream fit has no prior slot on the scale, and parity shows it carries the
bulk of the approximation error.

Pinned by `test_the_scale_carries_the_approximation_error` (offset in
[−0.30, −0.08] sd, sd ratio in [0.94, 1.0]) and
`test_the_coefficients_carry_far_less_of_it_than_the_scale`.

### F5's dropout parameters carry ~10× the approximation error of its purchase parameters

`r` and `alpha` agree with NUTS to under 0.01 sd. `a` and `b` agree to ~0.06 sd
with intervals ~4 % too narrow. This is the family's own documentation being
right — `a` and `b` are only weakly separately identified, so `(log a, log b)`
has a long curved ridge along fixed `a/(a+b)` that a Gaussian at the mode fits
poorly — and it is now asserted rather than remarked upon
(`test_the_purchase_process_is_approximated_far_better_than_the_dropout_process`).

Again the error is toward **narrow**, and again that is the direction that
matters: an over-confident dropout distribution understates how uncertain
`P(alive)` is for the customers in the middle of the book, who are exactly the
ones a dunning decision is marginal for.

**F5 was expected to be impractical to check this way and is not.** The
prediction was that a custom BG/NBD `logp` through `pm.Potential` would be too
awkward and that the Laplace-vs-NUTS gap would swamp everything. Neither held:
the likelihood transcribes to PyTensor in about fifteen lines, and at 600
customers the approximation is good enough that all four parameters pass a
parity tolerance of 2× the measured gap.

### F1 is graded `degenerate` on realistic thin-catalogue data

`hier_negbin` sets `max_divergent = 0` — one divergence in the kept draws fails
the fit — and does **not** override `nuts-rs`'s default acceptance target of
0.8. On a fixture with six-week thin parts, which is the shape THEORY.md says a
C-parts catalogue actually has ("mostly items with four or five observations"),
that combination produces 1–3 divergences in 16 000 draws and a `degenerate`
verdict.

Three measurements say this is about the sampler's step size and not about the
data being bad:

* it is **seed-dependent**: `seed = 7` gives 0 divergences on the same fixture
  where `20260801` gives 2 and `99` gives 3;
* it is **not monotone in sample size**: 6-week thin parts diverge, 10-week ones
  do not, 15-week ones diverge again at all three seeds tried;
* more warmup does not fix it — 1 000, 2 000, 4 000 and 8 000 all adapt to a
  step size near 0.465 and still divergence occasionally;
* **PyMC diverges on the same posterior**, 3–13 times in 20 000 draws at the same
  `target_accept = 0.8`, so this is a property of the geometry rather than of
  `nuts-rs`.

Meanwhile the posterior itself is right: on the six-week fixture all 27
parameters agreed with the PyMC reference inside tolerance, on a fit the
extension was simultaneously grading `degenerate`.

F8 asks the sampler for `target_accept = 0.95` and takes zero divergences on a
comparably hierarchical posterior. **F1 taking the default 0.8 looks like the
gap.** THEORY.md's advice for a `degenerate` verdict here — "usually means take
more draws" — is not what these measurements show; more draws is exactly what
does not help, because the divergence rate per draw is what is too high.

This is a finding about the extension, not about the suite, so the suite does
not work around it: `F1_THIN_WEEKS = 10` backs the fixture off to where the
verdict is `converged`, and the retreat is commented at the fixture and recorded
here. Comparing draws from a fit the extension itself says not to act on would
be measuring the right numbers under the wrong banner.

## Proof the tolerances gate something

A tolerance is only meaningful if a wrong answer breaches it. Four tests exist
solely to show that:

| test | perturbation | result |
|---|---|---|
| `test_wrong_prior_is_detected` | fit under `kappa0=8, alpha0=3, beta0=4`, compare to the reference-prior PyMC run | breach on `mu`, ~86× the tolerance |
| `test_perturbed_data_is_detected` | shift one lane's data by 0.5 units (≈2 posterior sd) | breach on that lane; the untouched lane in the same fit still passes |
| `test_wrong_pool_scale_is_detected` | refit with `pool_scale` 10× tighter | breach on the group effects |
| `test_scrambled_response_is_detected` | permute `y`, leaving every marginal of `y` intact | breach on the slopes |
| `test_ignoring_the_censoring_is_detected` (F2) | declare every shipment delivered; the `days` column is untouched | breach, reaching `log_sigma` |
| `test_a_wrong_distribution_is_detected` (F2) | fit the same data as lognormal against a Weibull reference | breach on the coordinates |
| `test_a_wrong_prior_is_detected` (F5) | refit under a different proper prior | breach against the original reference |
| `test_scrambling_recency_against_frequency_is_detected` (F5) | permute recency within repeat buyers, every marginal intact | breach |
| `test_a_base_in_which_nobody_has_lapsed_is_refused` (F5) | truncate every repeat payer's window at their last payment | `degenerate`, all draws NULL |
| `test_an_inflated_group_spread_is_detected` (F8) | triple one segment's scatter, leaving its level alone | breach on that segment's `sigma` |
| `test_the_group_labels_are_not_interchangeable` (F8) | rotate the segment labels | breach on the group effects |
| `test_a_flattened_catalogue_is_detected` (F1) | permute demand across parts, totals intact | breach on the per-part rates |

Two of those are worth singling out, because they check something no
per-parameter tolerance can. `test_the_full_covariance_is_used_and_not_just_its_diagonal`
(F2) compares the *correlation* of the draws against the observed information's,
because sampling from the diagonal would leave every marginal — and therefore
every `mean`/`sd`/`q05`/`q95` in this file — completely unchanged while making
the predictive spread 25× too wide. And the F8 control deliberately does **not**
assert that untouched segments stay inside tolerance, the way F7's equivalent
does: F8's groups share `sigma_pop` and `sigma_spread`, so perturbing one
legitimately moves the others. That is the partial pooling working, and
demanding otherwise would be demanding the model not be hierarchical.

`test_informative_prior_actually_moved_the_posterior` is the guard on the first
of these: it fails if the informative prior shifts the posterior by less than
3 sd, which would make the negative control easy to pass for the wrong reason.

## Running margins yourself

`pytest` prints a margin table at the end of every run — each comparison's delta
as a percentage of its tolerance. Use it before tightening or loosening
anything; the numbers quoted above came from it.
