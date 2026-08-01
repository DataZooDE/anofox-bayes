# The statistics behind anofox-bayes

This is the **background track**. The [User Guide](GUIDE.md) shows you what to type;
this explains what the numbers mean and why they are computed the way they are.

You do not need to read it to use the extension. Read it when you want to know what
you are looking at, when a reviewer asks what model was fitted, or when you have to
defend a recommendation.

It is written in two layers. Each section starts with the plain-language version;
the parts headed **In detail** carry the mathematics and are safe to skip.

**Contents**

1. [Why a distribution instead of a number](#1-why-a-distribution-instead-of-a-number)
2. [What a draws table actually is](#2-what-a-draws-table-actually-is)
3. [Priors, and why the defaults look the way they do](#3-priors-and-why-the-defaults-look-the-way-they-do)
4. [The two shipped families](#4-the-two-shipped-families)
5. [Engines: exact, Laplace, and what an approximation costs](#5-engines)
6. [Diagnostics: is this fit safe to use?](#6-diagnostics)
7. [Refusal, and why it is a feature](#7-refusal)
8. [How we know it is right](#8-how-we-know-it-is-right)
9. [Further reading](#9-further-reading)

---

## 1. Why a distribution instead of a number

A conventional analysis gives you an estimate: *"this lane costs €2.45/kg."* It is
usually accompanied by a standard error, which most people ignore.

The problem is that decisions are not made about estimates. They are made about
consequences, and consequences depend on the *whole range* of plausible values:

- *"Stock enough that we can serve 95 % of weeks"* — that is a **quantile**, not a mean.
- *"Was the promotion worth its €5/store cost?"* — that is **P(effect > 5)**, a
  probability, not a point.
- *"Is this carrier overbilling us?"* — that is **P(true rate > contracted rate)**.

None of those can be computed from an estimate plus a standard error without extra
assumptions. All of them are trivial if you have the *distribution* of the unknown
quantity. That distribution is called the **posterior**, and producing it is all this
extension does. Everything else — intervals, probabilities, service levels — is
arithmetic on top.

The word *Bayesian* just means the posterior is a statement about the **unknown
quantity** ("there is a 95 % probability the rate is between 2.4 and 2.8"), rather
than about the **procedure** that produced it, which is what a frequentist confidence
interval is. That difference is why `P(effect > 5)` is a legitimate question here and
is not, strictly, a question you can ask of a confidence interval.

## 2. What a draws table actually is

A posterior is a distribution over, say, a lane's true cost level. Rather than hand
you a formula, the extension hands you **samples from it** — typically a few thousand
numbers, each an equally plausible value of the unknown.

That is what the `value` column of a draws table contains. If 4 000 draws of `mu` for
lane `HAM-ROT` are mostly between 1.99 and 2.01, then the lane's true cost level is
almost certainly in that range.

Once you have samples, every question is an aggregate:

| Question | SQL |
|---|---|
| Best single guess | `median(value)` |
| 95 % credible interval | `quantile_cont(value, 0.025)`, `…0.975` |
| Probability it exceeds a threshold | `avg(CASE WHEN value > t THEN 1.0 ELSE 0.0 END)` |
| Quantity covering 95 % of outcomes | `quantile_cont(value, 0.95)` |

This is the entire reason posteriors are shipped as tables rather than as summary
statistics. A summary answers the questions its author anticipated; a table of draws
answers questions nobody has asked yet, without re-fitting.

> **In detail.** The draws are i.i.d. samples from the exact posterior for both shipped
> families — not a Markov chain — so there is no burn-in, no thinning and no
> autocorrelation. Monte Carlo error on a posterior mean is `sd/√N`; on a tail quantile
> it is larger, which is why [§6](#6-diagnostics) treats tail precision separately.

## 3. Priors, and why the defaults look the way they do

A Bayesian model needs a **prior**: what you believed before seeing the data. The
posterior combines prior and data, weighted by how much information each carries.

Priors make people uneasy, usually because of a fear that the answer can be rigged.
The defaults here are chosen to make that impossible by construction.

**The defaults are *reference priors*** — the formal expression of "let the data
speak". Under them:

- the posterior mean of a group's level is the **sample mean**;
- the posterior interval for a coefficient is numerically the **same interval a
  frequentist would report**;
- there is no number you could have chosen differently to get a different answer.

So if you supply no priors, you get the textbook answer, interpreted as a statement
about the parameter instead of about the procedure.

**Why not a "weakly informative" default?** Because such a default is a claim about
*scale*. A prior of `N(0, 10)` is nearly flat for costs in euros and crushingly
restrictive for costs in cents. Any concrete default would quietly dominate the data
for every customer whose units differ from the author's. A reference prior is
scale-free, so it is the only default that is honest without knowing your data.

**When should you set one?** When you genuinely know something and the data is thin —
a new lane with four invoices, where the fleet-wide rate is a better starting point
than four numbers. Setting `prior` then pulls a thin group toward what you told it,
and the pull weakens automatically as data accumulates. That is *shrinkage*, and it is
the single most useful thing a prior does.

> **In detail.** The Normal family's reference prior is
> `p(mu, sigma²) ∝ 1/sigma²`, i.e. `kappa0 = 0, alpha0 = -1/2, beta0 = 0`, which yields
> `mu | y ~ t_{n-1}(ȳ, s²/n)`. The Poisson family uses the Jeffreys prior
> `Gamma(a0 = 1/2, rate b0 = 0)`. Both are improper — they do not integrate to one —
> which is fine for a posterior but means they cannot be *sampled from*; that has a
> consequence for validation, see [§8](#8-how-we-know-it-is-right).

## 4. The shipped families

Model families are **code, not user input**. You choose among them and tune documented
slots; you cannot write your own likelihood. That is a deliberate restriction: it is
what makes it possible to ship each family with fixed parameterisation choices, a
validated config schema and its own calibration suite.

### `conjugate_anomaly` — a level per group

*Use it for:* "is this lane / carrier / cost centre behaving differently from its
own history or its contract?"

It estimates a level for each group independently. Anomaly is then a question you ask
of the draws — `P(mu > contracted_rate)` — rather than a threshold baked into the
model. Two likelihoods:

- **Normal** (default) — for continuous measurements like cost per kg. Parameters
  `mu` (the level) and `sigma` (how much it varies).
- **Poisson** — for counts, optionally per unit of exposure. One parameter `lambda`,
  a *rate*: claims per thousand consignments rather than claims. Set `exposure` and
  the arithmetic is handled for you.

> **In detail.** Normal: a Normal-Inverse-Gamma prior on `(mu, sigma²)`, conjugate to
> the Gaussian likelihood, giving
> `sigma² | y ~ InvGamma(alpha_n, beta_n)` and `mu | sigma², y ~ N(mu_n, sigma²/kappa_n)` with
> `kappa_n = kappa0 + n`, `mu_n = (kappa0·mu0 + n·ȳ)/kappa_n`, `alpha_n = alpha0 + n/2`,
> `beta_n = beta0 + SS/2 + kappa0·n·(ȳ-mu0)²/(2·kappa_n)`.
> Poisson: `y ~ Poisson(lambda·exposure)` with `lambda ~ Gamma(a0, rate b0)`, posterior
> `Gamma(a0 + Σy, rate = b0 + Σexposure)`.

### `pooled_gaussian` — a linear model

*Use it for:* measuring the effect of something — a price change, a promotion, a
process change — while holding other factors constant. This is the family behind
difference-in-differences and interrupted time series.

You give it a response `y` and predictors `x`. Each predictor gets a coefficient, and
the coefficient's posterior *is* the answer: "the promotion added 8.0 units per
store-month, 95 % credible interval 7.7 to 8.3."

**Pooling.** With a `group` column it also fits one intercept per group, shrunk toward
the population level. A group with three observations borrows strength from the rest
instead of reporting noise; a group with three hundred is left alone. How hard the
shrinkage pulls is `pool_scale`, measured **in residual standard deviations** — so
noisier data pools more at the same setting, which is what you want.

> **In detail.** `y = Xβ + ε`, `ε ~ N(0, sigma²)`, with a Normal-Inverse-Gamma prior on
> `(β, sigma²)`. With `A = X'X + P` and `b_n = A⁻¹X'y`:
> `sigma² | y ~ InvGamma(a_n, s_n)`, `β | sigma², y ~ N(b_n, sigma²A⁻¹)`,
> `a_n = a0 + (n − k)/2`, `s_n = s0 + (y'y − b_n'X'y)/2`, where **`k` counts
> coefficients carrying a flat prior**. Coefficients with a proper, sigma-scaled prior
> act as prior observations and cost no degrees of freedom; flat ones consume one each.
> Getting this wrong makes every interval too narrow by `√((n−k)/n)` — it is exactly
> the bug the PyMC parity suite caught during development.
>
> `pool_scale` enters as a precision `1/pool_scale²` added to the diagonal of `A`,
> which is equivalent to the prior `β_group | sigma² ~ N(0, sigma²·pool_scale²)`. The
> intercept is never penalised: it lives on the scale of the response, where a prior
> centred at zero means something nobody intends, and shrinking it silently pushes
> every slope the other way to compensate.
>
> **Not estimated:** `pool_scale` is fixed by configuration. Estimating it makes the
> posterior non-conjugate and needs a general sampler; that arrives with the NUTS
> engine. Fixing it is the documented stepping stone, and the value used is recorded
> in the fit.

## 5. Engines

An **engine** turns a model into draws. The choice is invisible to your SQL — same
function, same output columns, same diagnostics — and is recorded in the draws table
on the `__engine__` row so a reviewer can see which one ran.

| Engine | What it does | Status |
|---|---|---|
| `exact` | Samples the closed-form posterior directly. No approximation. | default for the two conjugate families |
| `laplace` | Fits a Gaussian at the posterior's peak and samples that. | available everywhere; **the** engine for `censored_aft` |
| `nuts` | General-purpose sampler for models with no closed form. | planned (0.2) |

Where a closed form exists, `exact` is both faster and more accurate, so it is the
default. `laplace` exists because it generalises to families that have no closed form
— and, in the meantime, because it provides an **independent check**: both shipped
families are conjugate, so both engines describe the same posterior by different
routes, and they are tested to agree.

**When is the approximation good enough?** Measured, not asserted. On
`pooled_gaussian` the two engines agree on every coefficient to well under a percent
at n = 400. Where they differ is the tails at small samples: the exact marginal for a
coefficient is a Student-t, and Laplace returns its Gaussian limit, so it slightly
*understates* how wide a 99 % interval should be. The scale parameter carries a
separate O(1/n) discrepancy, measured falling from ~5 % at n = 20 to ~0.1 % at
n = 2 000.

On `conjugate_anomaly` the same comparison has a closed-form answer, and the
measurement matches it to four digits: the Laplace spread for `mu` is too narrow by
exactly `1 − √((n−3)/n)` **per group** — 0.4 % on a group of 400 observations and
**29 % on a group of 6**, with `sigma` worse still (44 % on the spread, 20 % on the
mean, at n = 6). Since this family fits every group independently, the relevant `n` is
the group's own observation count, not the size of the table. That is a strong reason
to leave `exact` in place here: an anomaly model earns its keep on exactly the thin
lanes where the approximation is worst, and too narrow is the direction that
manufactures both false alarms and unearned all-clears.

### `censored_aft` — a bridged Laplace posterior

`censored_aft` is the first family here with **no closed form at all**, so `laplace` is
not a cross-check on it — it is the fit. That matters for how much you should trust it,
and the honest accounting is:

The mode and the curvature are not computed here. They come from `anofox-statistics`,
which already owns a tested censored AFT likelihood, its analytic gradient and Hessian,
and a Newton search with damping. anofox-bayes calls that fit in-process, reassembles
the observed information at the mode it returns, and turns the pair into draws. A MAP
estimate together with its observed information *is* a Laplace posterior; the only step
that was missing was sampling the multivariate normal.

**The full covariance is used, and that is the whole ballgame.** The upstream fit
publishes only standard errors — the diagonal. Sampling from a diagonal treats every
coefficient as independent, and in a duration model with a covariate measured away from
zero the intercept and the slope are almost perfectly anti-correlated: measured on the
test fixture, `corr = −0.998`. Their errors then cancel in the linear predictor, so the
predictive standard deviation computed from the full matrix is about **25 times
smaller** than the one computed from the diagonal. Both answers are finite, both pass
every diagnostic, and only one is the posterior.

**What has been certified, and what has not.** SBC requires drawing the truth from the
prior the fit actually uses, and the upstream fit accepts priors on the coefficients
and on nothing else — the scale is estimated by maximum likelihood under a flat prior.
So:

| Suite | Result (χ², 15 df; threshold 37.7) |
|---|---|
| `exponential`, n = 200 — every free parameter properly priored | **13.0 / 12.8 — passes.** A complete certificate for the bridge |
| `weibull`, n = 200 — coefficients only | 14.0 / 14.2 — passes, but *conditional*: `sigma` is uncertified |
| `exponential`, n = 25, heavily censored | 7.2 / 9.7 — calibrated even on a thin cohort |

`sigma` **is not SBC-certified for the distributions that estimate it**, and cannot be
until `anofox-statistics` grows a prior slot on the scale. That is a one-field change
upstream, and until it lands the gap is stated here rather than left to be discovered.

Worth contrasting with `conjugate_anomaly`: the thin-cohort result is *good*, where
F7's is bad (29 % too narrow at n = 6). The difference is what is being approximated.
A regression coefficient's posterior is close to Gaussian at modest sample sizes; a
variance parameter's is not.

> **In detail.** Laplace works on an unconstrained scale — `sigma` is sampled as
> `log sigma`, so every draw is positive by construction; a Gaussian fitted directly to
> a scale parameter would put mass below zero. The mode is found by Newton iteration on
> an analytic gradient with a backtracking line search, and the covariance is the
> inverse of the negative Hessian there, obtained by differencing the *analytic
> gradient* rather than the log density twice (second differences of a scalar lose
> roughly two-thirds of the available precision).

## 6. Diagnostics

Two questions, and they are not the same question:

1. **Did the fit work?** (Is the sampler trustworthy?)
2. **Does the data support a conclusion?** (See [§7](#7-refusal).)

For (1) there are two statistics.

### Effective sample size — how much your draws are worth

If draws are correlated with each other, 4 000 of them carry less information than
4 000 independent ones. **ESS** is how many independent draws they are equivalent to.
Low ESS means your estimate is noisy for reasons that have nothing to do with your
data — take more draws.

It comes in two flavours, and the distinction matters commercially:

- **`ess_bulk`** governs the reliability of the **mean**.
- **`ess_tail`** governs the reliability of the **5 % and 95 % quantiles**.

A service-level or safety-stock decision reads a *quantile*. Gating only on bulk ESS
would certify precisely the number you are not entitled to use. **Gate on both** —
`anofox_bayes_ess_gate` does.

### R-hat — did independent chains agree?

R̂ compares variance *between* chains to variance *within* them. Above about 1.01, the
chains have not settled on the same answer and nothing else should be believed.

**In v0.1 this is usually `NULL`, and that is correct.** Both current engines draw
independently, so there is no Markov chain that could fail to mix and the statistic is
undefined. It returns `NULL` rather than a reassuring `1.00` because a diagnostic that
was never computed must not read as one that passed. It becomes load-bearing when NUTS
arrives.

> **In detail.** Both follow Vehtari et al. (2021): split chains, rank-normalised
> draws, and Geyer's initial monotone positive sequence for the autocorrelation
> truncation. Rank normalisation is what makes them defined for heavy-tailed
> posteriors, where a variance-ratio statistic is not. `ess_tail` is the smaller of the
> ESS of the indicator series `1[x ≤ q₀₅]` and `1[x ≤ q₉₅]`. The implementation
> mirrors Stan's reference, which is what makes the ArviZ parity comparison meaningful.

## 7. Refusal

Some questions cannot be answered from the data available, and the most valuable thing
a statistical tool can do is say so.

A lane with one invoice has no estimable variability. A regression that fits its data
perfectly has no residual variance. A design with two identical predictors has no
unique solution. In each case there is no amount of computation that produces a
trustworthy answer — but there is always *some* number a less careful tool would print.

Every fit therefore carries a `__status__`:

| Status | Meaning | Act on it? |
|---|---|---|
| `converged` | Fit succeeded, diagnostics passed | **yes** |
| `degenerate` | Ran, but the diagnostics failed | no — take more draws |
| `insufficient_data` | Ran, but the data cannot support a conclusion | no — this is a refusal |
| `failed` | Could not complete | no |

And a parameter that could not be estimated has **`NULL` draws**, not a plausible
number — so "no estimate" stays distinguishable from an estimate at every step. This
is why `anofox_bayes_prob_greater` returns `NULL` rather than `0.0` for a refused
parameter: a confident "definitely not" would be worse than no answer.

The practical consequence: `insufficient_data` is not an error you work around. It is
the model telling you that the recommendation you were about to make is not supported.

## 8. How we know it is right

Three independent checks, because each catches what the others cannot.

**Closed-form tests.** Every posterior is pinned against its textbook formula in a unit
test. Catches an algebra error.

**Simulation-based calibration (SBC).** Draw a parameter from the prior, simulate data
from it, fit, and record where the truth falls among the posterior draws. Repeated
enough times, that rank must be *uniform* — any deviation is a calibration error, and
its shape names the fault: a ∪-shaped histogram means the posterior is too narrow
(over-confident), ∩-shaped means too wide, a slope means biased.

This is the check that matters commercially, because a model can be *accurate* and
still ruin a service-level decision by reporting intervals that are too tight. Run per
family and per engine; the Laplace suite is what certifies where that approximation is
admissible. Fixtures prove the harness rejects a deliberately over-confident posterior,
so it is a gate rather than a formality.

**Parity against PyMC.** The same data fitted by a pinned PyMC reference, compared
within documented tolerances, in CI.

**Why all three.** SBC must draw the truth from the prior, so it can only test *proper*
priors — and the defaults here are reference priors, which are improper. The PyMC suite
covers exactly that gap: it found a real degrees-of-freedom error in `pooled_gaussian`
that SBC structurally could not see, and that the per-parameter tolerances were too
loose to catch. Neither harness is redundant.

**And a fourth, learned from the bridge: a per-parameter check cannot certify a
covariance.** SBC ranks each parameter on its own, so it tests *marginal* posteriors —
and the marginals are exactly what a covariance matrix's diagonal preserves. Replacing
`censored_aft`'s posterior precision with its own diagonal, which would throw away
every correlation between coefficients, leaves **219 of 220 unit tests and all six SBC
suites green**. The one test that fails is the one written on a *function of several
parameters at once* — the standard deviation of the linear predictor, checked against
the full-covariance answer and against the diagonal-only one, which differ by a factor
of 25.

So every family gets at least one assertion on a linear combination of its parameters,
not only on each parameter separately. Marginal checks cannot see a joint error, and a
joint error is what produces a confidently wrong interval.

## 9. Further reading

- Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, **Bayesian Data Analysis**, 3rd ed. —
  Ch. 2–3 for conjugate models, Ch. 14 for the linear model. The source for every
  posterior in [§4](#4-the-two-shipped-families).
- Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), *Rank-normalization, folding,
  and localization: An improved R̂ for assessing convergence of MCMC*, **Bayesian
  Analysis** 16(2). The definitions in [§6](#6-diagnostics).
- Talts, Betancourt, Simpson, Vehtari, Gelman (2018), *Validating Bayesian inference
  algorithms with simulation-based calibration*, arXiv:1804.06788. The method in
  [§8](#8-how-we-know-it-is-right).
- McElreath, **Statistical Rethinking**, 2nd ed. — the gentlest route in if §1–§3 went
  too fast.

---

Practical next steps: the [User Guide](GUIDE.md) for tasks, the
[API Reference](API_REFERENCE.md) for every config slot, and the
[Draws Contract](DRAWS_CONTRACT.md) for the output schema.
