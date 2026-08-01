# PyMC golden-run parity validation

The extension computes its posteriors in closed form. That is fast and exactly
reproducible, and it is also the failure mode nobody notices: a closed form with
a wrong constant in it produces confident, stable, plausible numbers forever.

This suite is the check. For each family it takes one fixed-seed dataset, fits it
**twice** — once through `anofox_bayes_fit`, once with a PyMC model written
against the *same prior* — and compares the posterior mean, standard deviation,
5 % and 95 % quantiles. PyMC knows nothing about the extension's algebra; it
samples the model with NUTS. If the extension's derivation is right, the two
posteriors are the same distribution and their summaries agree up to Monte
Carlo error.

## Running it

```bash
cd validation
uv sync --frozen
uv run pytest -v
```

Roughly 50 seconds on a 4-core machine; most of it is NUTS.

The suite loads the extension from
`build/release/extension/anofox_bayes/anofox_bayes.duckdb_extension`. Build it
first with `make release` from the repository root, or point the suite elsewhere:

```bash
ANOFOX_BAYES_EXTENSION=/path/to/anofox_bayes.duckdb_extension uv run pytest -v
```

If the binary is missing the suite **skips** with that message rather than
failing — a missing build is an environment problem, not a correctness result.
CI treats a skip as a failure by asserting the binary exists before running.

Every run ends with a margin table: each comparison's discrepancy as a
percentage of its tolerance. Read it before changing any tolerance.

## What it proves

**F7 `conjugate_anomaly`** (`tests/test_parity_f7.py`)

* Normal likelihood under the **default reference prior**
  (`mu0=0, kappa0=0, alpha0=-1/2, beta0=0`), two groups, `mu` and `sigma`.
* Normal likelihood under an **explicit proper NIG prior**
  (`kappa0=8, alpha0=3, beta0=4`) — this is what proves the `prior` config slot
  reaches the posterior. The default-prior tests alone would still pass if the
  extension ignored `prior` entirely.
* Poisson likelihood with an `exposure` column under the default
  `Gamma(a0=1/2, rate b0=0)` prior, two carriers.

All four statistics agree, worst case 55 % of tolerance. **F7 is clean.**

**F3 `pooled_gaussian`** (`tests/test_parity_f3.py`)

* An ungrouped two-predictor regression under the default flat coefficient prior
  with `a0 = s0 = 0`.
* A six-store, twenty-period difference-in-differences panel with
  `pool_scale = 5`: intercept, three slopes (including `beta[treated_post]`, the
  causal estimate itself) and all six partially pooled store effects.

All four statistics agree **for every coefficient**, worst case 66 % of
tolerance. The residual scale `sigma` does not — see below.

### Reference models are not a formality

Expressing the extension's *improper* default priors in PyMC is the part of this
suite that is easiest to get quietly wrong, so each reference model in
`tests/_support.py` carries its derivation. The short version:

* `p(sigma) ∝ 1/sigma` is `pm.Flat` on `log(sigma)`, **not** `pm.HalfFlat` on
  `sigma` — `HalfFlat` gives `p(sigma) ∝ 1`, a different prior, and would shift
  the answer by exactly the kind of amount this suite is looking for.
* `Gamma(a0, rate 0)` is flat-on-`log(lambda)` (which implies `1/lambda`) times a
  `pm.Potential` of `a0·log(lambda)`.

## What this found

### A real discrepancy in F3's residual scale

`crates/anofox-bayes-core/src/catalog/f3_pooled_gaussian.rs` computes

```rust
let a_n = a0 + n as f64 / 2.0;
```

That is the Normal-Inverse-Gamma result, and it is correct **when the
coefficient prior is proper and scales with sigma** — the `(sigma²)^(-p/2)`
factor the prior contributes is integrated back out when `beta` is marginalised.

The default prior is not that. `beta_scale` defaults to infinity, and the
intercept is never penalised at any `beta_scale`, so those `k` coefficients
carry a genuinely flat prior which contributes no such factor. The correct shape
is then the textbook one:

```
sigma² | y ~ Inv-chi²(n − k, s²)  ==  InvGamma(a0 + (n − k)/2, s_n)
```

The consequence is that **every credible interval F3 reports is too narrow by
`sqrt((n − k)/n)`**, and the fit is over-confident, never under. Measured:

| design | n | k (flat coefficients) | predicted `sqrt((n−k)/n)` | measured `E[sigma]_ext / E[sigma]_pymc` |
|---|---|---|---|---|
| ungrouped, 2 predictors | 80 | 3 | 0.98107 | 0.98005 |
| DiD panel, 3 slopes, 6 stores | 120 | 4 | 0.98319 | 0.98309 |

The six store effects are *not* part of `k`: they carry a proper
`N(0, sigma²·pool_scale²)` prior and legitimately supply their own factors.

This is small at these sample sizes and grows with the number of predictors
relative to `n` — at `n = 30` with 8 predictors it is 15 %. For a product whose
output is a credible interval that a decision is gated on, a systematically
over-confident interval is the wrong direction to be wrong in.

It is recorded as two `xfail(strict=True)` tests
(`test_simple_residual_scale_parity`, `test_panel_residual_scale_parity`) plus
two measurement tests that pin the factor. **`xfail_strict` is on**, so the day
`a_n` is corrected those tests turn red and the markers — and the two
measurement tests — have to come off deliberately.

### Two documentation/implementation mismatches in F3

Neither is a numerical error, but both make the module doc comment describe a
different model from the one the code fits.

1. **The group-effect prior scales with sigma.** The doc says each group
   intercept gets `N(0, pool_scale²)`. The code forms `A = X'X + P` and draws
   `beta | sigma² ~ N(b_n, sigma² A⁻¹)`, which is `N(0, sigma²·pool_scale²)`.
   The reference model here encodes what the code does. The practical difference
   is that the amount of pooling depends on the residual scale: noisier data
   pools *less* under the implemented model, more under the documented one.

2. **`s_n` is documented in a generality the code does not implement.** The doc
   gives `s_n = s0 + (y'y + b0' P b0 − b_n' A b_n)/2`; the code computes
   `s0 + (y'y − b_n' X'y)/2`. The two coincide exactly at `b0 = 0`, and `b0` is
   not a configurable slot, so nothing is wrong today. It is worth noting only
   because adding a prior mean later would need the two dropped terms put back,
   and the doc reads as though they already are.

## Layout

```
validation/
  pyproject.toml        exact pins; see the comments for why each one is pinned
  uv.lock               committed, so `uv sync --frozen` is reproducible
  conftest.py           extension loading, skip behaviour, the margin report
  TOLERANCES.md         every tolerance and its derivation
  tests/
    _support.py         PyMC reference models, summary/comparison helpers, datasets
    test_parity_f7.py   conjugate_anomaly
    test_parity_f3.py   pooled_gaussian
```

`.github/workflows/validation.yml` builds the extension and runs this suite as
its own required job, separate from any platform matrix.

## Track record

This suite has already earned its cost. It found a real degrees-of-freedom error in
`f3_pooled_gaussian.rs`: the residual-scale posterior used the Normal-Inverse-Gamma
shape `a0 + n/2`, which is correct only when the coefficient prior is proper and
sigma-scaled. Under the default flat prior the textbook shape is `a0 + (n - k)/2`, so
every F3 credible interval came out too narrow by `sqrt((n - k)/n)` — over-confident,
never under, and growing with `k/n`.

Two things about *how* it was found are worth keeping in mind when adding families:

* **SBC could not have caught it.** Simulation-based calibration must draw the truth
  from the prior, so it only ever runs under a *proper* prior — precisely the case
  where the old formula was right. The two harnesses are not redundant.
* **The per-parameter tolerances could not have caught it either.** The error was ~2 %
  on these designs, comfortably inside the sd tolerance, and it is not possible to
  tighten that tolerance far enough without tripping on Monte Carlo noise. It was
  caught by a test that measured the *ratio* directly and compared it against a
  predicted constant.

The tests that found it are still here, inverted into regression guards.
