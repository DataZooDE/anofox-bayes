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

## Proof the tolerances gate something

A tolerance is only meaningful if a wrong answer breaches it. Four tests exist
solely to show that:

| test | perturbation | result |
|---|---|---|
| `test_wrong_prior_is_detected` | fit under `kappa0=8, alpha0=3, beta0=4`, compare to the reference-prior PyMC run | breach on `mu`, ~86× the tolerance |
| `test_perturbed_data_is_detected` | shift one lane's data by 0.5 units (≈2 posterior sd) | breach on that lane; the untouched lane in the same fit still passes |
| `test_wrong_pool_scale_is_detected` | refit with `pool_scale` 10× tighter | breach on the group effects |
| `test_scrambled_response_is_detected` | permute `y`, leaving every marginal of `y` intact | breach on the slopes |

`test_informative_prior_actually_moved_the_posterior` is the guard on the first
of these: it fails if the informative prior shifts the posterior by less than
3 sd, which would make the negative control easy to pass for the wrong reason.

## Running margins yourself

`pytest` prints a margin table at the end of every run — each comparison's delta
as a percentage of its tolerance. Use it before tightening or loosening
anything; the numbers quoted above came from it.
