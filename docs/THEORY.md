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
4. [The shipped families](#4-the-shipped-families)
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

### `hier_negbin` — how much of this part will be wanted?

*Use it for:* demand on a catalogue of thousands of items, most of which have almost
no history. Safety stock, reorder points, service levels.

The difficulty is not forecasting the busy items. It is that a C-parts catalogue is
mostly items with four or five observations, and there are two wrong ways to handle
them. Fit each item on its own and a part that happened to be issued twice last month
looks like a part that moves twice a month. Fit them all together and a bearing that
moves twenty a week gets the same rate as a seal that moves three.

**Partial pooling** is the third way. Each part gets its own rate, and each rate is
pulled toward what the catalogue does by an amount the data decides: hardly at all for
a part with forty weeks of history, most of the way for one with five. Measured on the
scenario in `test/sql/f1_hier_negbin.test`, the five-week parts move an average of
7.4 % away from their own sample mean and the forty-week parts 2.1 %.

**Overdispersion.** Spare-parts demand is burstier than a Poisson process: one order
for twelve, then nothing for a month. A Poisson model reads that as a steady rate and
reports an interval far too tight, which is a reorder point that stocks out. The
negative binomial adds a dispersion parameter `phi` that absorbs the burstiness —
`Var(y) = mu + mu²/phi`, so a large `phi` *is* the Poisson limit and the model can
report that no extra burstiness was found. `likelihood: 'poisson'` is available for
data that genuinely is Poisson, and is measurably worse where it is not: on
overdispersed data a 95 % reorder point set from the Poisson model achieved 87.4 %,
against 95.4 % from the negative binomial.

**The reorder point.** The posterior predictive for next period's demand is a mixture
of negative binomials, one per posterior draw, and its probability mass function is
closed form — so a service level is a sum over a draws table in plain SQL, with no
re-fit and no simulation. `test/sql/f1_hier_negbin.test` is that query.

> **In detail.** For observation `i` in group `j`, with optional exposure `E` and
> optional population-level covariates `x`:
>
> ```text
>   y_ij ~ NegBin(mu_ij, phi),   Var = mu + mu^2/phi
>   log mu_ij = intercept + x_ij'beta + tau * z_j + log E_ij
>   z_j ~ N(0, 1)
> ```
>
> The group effect is written `tau * z_j` with `z_j ~ N(0, 1)` — the **non-centred**
> parameterisation — and this is fixed by the family rather than offered as a choice.
> Writing it the other way round, with `u_j ~ N(0, tau^2)` as a coordinate, makes the
> prior's width depend on a parameter the sampler is also moving, and the resulting
> funnel is the classic reason a hierarchical model fails to mix. Measured on the same
> data, same sampler, same seed: **`tau` reaches an effective sample size of 634 with
> R̂ = 1.004 non-centred, against 196 and R̂ = 1.016 centred** — the centred version
> fails this extension's own convergence gate and the non-centred one passes.
>
> Parameters are reported as `intercept`, `tau`, `phi` at the population level and, per
> group, `u` (the group's offset, `tau * z_j`) and `rate` (`exp(intercept + u)`, the
> group's expected count per unit of exposure).
>
> **Priors.** The coefficients are flat by default, as everywhere else here. `tau`
> defaults to a **uniform prior on `tau` itself** rather than the scale-free `1/tau`:
> `1/tau` gives an improper posterior for a variance component, while uniform is proper
> for three or more groups (Gelman 2006) and is the standard reference choice. `phi`
> defaults to a **uniform prior on the overdispersion `1/phi`**, which is flat exactly
> where the Poisson limit is — so the default cannot push a fit toward finding
> burstiness that is not there. Both are declared on the natural scale while the
> sampler works on `log tau` and `log phi`, so the density carries `+ log tau` and
> `- log phi` **log-Jacobian** terms. Those terms are not decoration and are not
> visible to any engine-agreement test — both engines would explore the same wrong
> surface — so they are pinned directly against the closed form.
>
> **Engine: NUTS only.** There is no closed form, and unlike every other family here
> the Laplace approximation is not merely worse, it is inadmissible. A Laplace
> posterior is a Gaussian at the joint mode, and a non-centred hierarchy has no usable
> one: when every `z_j` is zero the likelihood does not depend on `tau` at all, so the
> density has a ridge along `{z = 0, tau -> infinity}` that the `+ log tau` Jacobian
> makes rise without bound. The ridge carries no posterior *mass* — the region where
> the likelihood is any good shrinks like `tau^-G` — which is why a sampler is
> untroubled by it and a mode search walks straight up it. Measured: under the default
> prior the mode search does not converge; under a proper half-normal(1) on `tau` it
> converges to a mean `tau` of 1.63 where the truth is 0.5, and grades itself
> `degenerate`. So `engine: 'laplace'` is refused with that explanation rather than
> served.

#### Why this family is native rather than bridged

`ROADMAP.md` deferred F1 on the expectation that a negative-binomial GLMM through the
`anofox-statistics` bridge would cover the safety-stock agent adequately. It does not,
and the reason is exactly the one the roadmap flagged: **the dispersion is estimated
outside the IRLS loop, so it is not in the curvature.** Three facts about the pinned
revision of `anofox-stats-core`, each asserted in
`catalog::f1_hier_negbin::bridge_comparison` rather than quoted:

1. `GlmmFamily::from_name("negbinomial")` returns `theta = 1.0`. The dispersion is an
   **input** to `fit_glmm`, and `GlmmResult` has no field that could carry a posterior
   for it.
2. `GlmmResult::var_group` — the pooling scale — is a Brent profile point estimate,
   with no standard error anywhere in the struct. A bridged posterior would have to
   condition on that too.
3. `fit_negbinomial` with `alpha: None`, the only data-driven dispersion upstream
   offers, **failed to converge on 20 of 20** simulated thin-SKU panels.

So a bridged F1 conditions on two point estimates, and the measured consequence is the
one that matters commercially. On 40 SKUs of four periods each, drawn from
`tau = 0.6`, `phi = 2.0`, the 90 % credible interval for a SKU's own demand rate
covers:

| | mean demand 3/period | mean demand 25/period |
|---|---:|---:|
| bridged, plug-in dispersion | **0.76** | **0.41** |
| bridged, true dispersion handed to it | 0.75 | 0.81 |
| **native `hier_negbin`** | **0.90** | — |

Nominal is 0.90. The middle row is the informative one: handing the bridge the true
dispersion for free moves 0.41 to 0.81, so the dispersion error does not merely widen
the interval, it *propagates into the pooling scale* — the fitted `tau` collapses from
0.44 to 0.17 against a true 0.6 — and produces an interval that is both wrong and
narrower. There is no diagnostic that says so, which is the whole problem.

The predictive interval, as opposed to the parameter interval, is less obviously
broken: integer support pads a discrete interval, so the bridge's achieved service
level lands between 0.949 and 0.964 against a nominal 0.95 — sometimes over, sometimes
under, depending on data it cannot see. That is not a defence of the bridged path. It
is the reason the parameter interval is the one to measure.

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

**Random slopes.** `random_slopes` extends that from the level to the *effect*: each
group gets its own coefficient on a named predictor, shrunk toward the population
coefficient by the same `pool_scale`. "This store's customers are twice as
price-sensitive as the average, and this newly-opened one probably is not as unusual
as its five weeks suggest" is a sentence the model can now make, and could not before.

The alternative people reach for — one interaction column per group — fits every group
independently and shrinks nothing, so the thin group reports whatever its handful of
rows happen to say. That is the workaround this replaces.

Two boundaries worth stating, because they are the reason this is a small change and
not a new family:

- The predictor must also be a fixed effect (`x`), so that a group's deviation is a
  deviation *from* a population slope. Shrinking group slopes toward **zero** would
  assert the predictor has no effect, which is a claim about the world nobody asked to
  make; shrinking them toward a common slope asserts only that groups are alike until
  shown otherwise. The request is refused rather than reinterpreted.
- `pool_scale` is still **fixed**, and a per-group `sigma` is still unavailable. Both
  break conjugacy and belong to a sampled family — see `ROADMAP.md` §3.3. Random slopes
  at a fixed scale do not: they are more columns with a Gaussian prior, and nothing
  else.

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
> A **random slope** on predictor `c` adds one column per group carrying `c`'s value on
> that group's rows and zero elsewhere, with the same `1/pool_scale²` on its diagonal.
> Those columns sum exactly to `c`'s fixed column, so `X` is rank deficient by
> construction — the same trap as an intercept plus a full set of group dummies. What
> resolves it is the prior: `A = X'X + P` is positive definite exactly when the
> *unpenalised* columns are linearly independent, and the group block is penalised.
> Rank deficiency among the unpenalised columns is still refused.
>
> One consequence to know about: each random slope adds a group's worth of columns to
> `p` without adding an observation, and the Laplace engine centres `sigma` on the joint
> mode `2·s_n/(2·a_n + p)` against the exact posterior mean `s_n/(a_n − 1)`. So Laplace
> understates `sigma` by `√((2(a_n−1))/(2·a_n + p))`, and random slopes make that gap
> wider. It is derived rather than tolerated — the engine-agreement test asserts the
> predicted ratio — and `exact` remains the default.
>
> **Not estimated:** `pool_scale` is fixed by configuration. Estimating it makes the
> posterior non-conjugate and needs a general sampler. The NUTS engine that such a
> family will run on now exists ([§5](#5-engines)); the family itself does not yet, so
> for now fixing the scale remains the documented stepping stone, and the value used is
> recorded in the fit.

### `payer_alive` — is this customer still a customer?

*Use it for:* collections, dunning, retention — anywhere the question is whether a
customer who has gone quiet has actually left.

The difficulty is that customers of this kind never announce their departure. There is
no cancellation event; there is only a purchase that has not happened yet, and a
purchase that will never happen, and the two look identical. What separates them is
*context*: three weeks of silence from someone who buys weekly means something very
different from three weeks of silence from someone who buys twice a year.

The model is **BG/NBD** — "buy till you die". Each customer has their own purchase
rate and their own propensity to drop out, both unobserved; what the model estimates is
the *distribution* of those two things across your base. Then any individual customer's
`P(alive)` follows from where their own history sits against that population.

Only three numbers per customer are needed: repeat purchases, when the last one was,
and how long you have been watching. Everything else in a transaction history is
irrelevant to this question, which is why a decade of data becomes one row.

Two consequences worth knowing:

- **A customer who has never repeated always scores 1.0.** Dropping out is something
  that happens *after* a purchase, so a one-purchase customer has had no opportunity
  to churn and the data says nothing either way. Their uncertainty is about how often
  they buy, not about whether they are still there.
- **`P(alive)` is a closed-form expression you can evaluate in SQL**, so a fitted model
  scores tomorrow's customer list — or a list you never fitted — with a join and no
  re-fit. This is the reason the family is BG/NBD rather than the older Pareto/NBD,
  which fits marginally better on some data and whose `P(alive)` needs a hypergeometric
  function that no database has.

> **In detail.** Transactions are Poisson(`lambda`) while alive with
> `lambda ~ Gamma(r, rate alpha)`; after each transaction the customer drops out with
> probability `p ~ Beta(a, b)`. Integrating both out gives, per customer,
> `ln L = lnΓ(r+x) − lnΓ(r) + r·ln(alpha) + lnΓ(b+x) + lnΓ(a+b) − lnΓ(b) − lnΓ(a+b+x)
> + ln[(alpha+T)^−(r+x) + 1{x>0}·a/(b+x−1)·(alpha+t_x)^−(r+x)]`, and
> `P(alive) = 1/(1 + 1{x>0}·a/(b+x−1)·((alpha+T)/(alpha+t_x))^(r+x))`. The two terms of
> that bracket are the two histories consistent with the data — still alive at `T`, or
> gone some time after `t_x` — and `P(alive)` is simply the first one's share.
>
> All four parameters are positive and fitted on the log scale; priors are declared
> there too, so the default (flat on `log`) is the scale-free `p(θ) ∝ 1/θ` and the mode
> is the maximum likelihood estimate.
>
> **Refusal.** BG/NBD's likelihood has boundary solutions. If no repeat buyer has been
> seen to go quiet — every `t_x` equal to its `T` — the likelihood keeps increasing as
> the dropout probability goes to zero and there is no interior maximum at all. The
> family finds its own mode before any engine runs and checks four things about it: that
> it is inside a sane range, that it is stationary, that its curvature is a covariance,
> and that the resulting marginals are narrow enough to be intervals. Failing any of
> them reports `degenerate` with `NULL` draws rather than a confident number derived
> from curvature that is not a posterior.

### `varying_variance_gaussian` — a spread per group, and the pooling decided by the data

*Use it for:* "how much buffer does **this** segment need?" — a service level, a
payment-delay reserve, a worst-case lead time. Any question whose answer is a *tail*
rather than an average.

`pooled_gaussian` estimates a level per group and one spread for the whole dataset. It
has to: one shared `sigma` is what makes its posterior closed form. The consequence is
that two segments with the same average must receive the same interval, whatever the
data says about how scattered each one is — and a decision about a tail reads exactly
that interval. Measured on the six-segment fixture in
`test/sql/f8_segment_spread.test`, two segments with an identical 30.0-day mean delay
need 95 % buffers a day and a half apart, and no fit of `pooled_gaussian` can produce
the difference: there is no row in its output where the difference could appear.

This family fixes that, and one other thing, at the price of the closed form:

| | `pooled_gaussian` | `varying_variance_gaussian` |
|---|---|---|
| Residual spread | one, shared | one **per group** |
| Pooling strength | `pool_scale`, an analyst setting | `pool_scale`, a **parameter with a posterior** |
| Posterior | closed form, `exact` engine | sampled, `nuts` engine |

The second row matters as much as the first. In `pooled_gaussian` how hard a thin group
is shrunk toward the population is a number somebody typed; here it is estimated from
how far apart the groups actually turn out to be, and its own uncertainty widens every
group effect. A panel whose segments sit on top of each other learns a small
`pool_scale` and pools hard; a panel whose segments are genuinely different learns a
large one and leaves them alone.

**Why a separate family rather than a mode of `pooled_gaussian`.** Either change alone
destroys conjugacy, so a combined family would be exact and instantaneous under some
configurations and sampled and slow under others, with `__engine__` varying by config
slot under one name. `pooled_gaussian`'s warranty is that its posterior is a formula
cross-checked by three engines; a family with two warranties has neither.

**On the name.** `hierarchical_gaussian` was the obvious candidate and was rejected:
`pooled_gaussian` is hierarchical too — it has group effects drawn toward a common
level — so the pair would give you nothing to choose on. `heteroscedastic_gaussian`
says the right thing but has two accepted spellings, and the family id is a string you
type and that feeds `model_id`. What is left is the plain description: the variance
**varies**, within a group and between them.

> **In detail.** The model, and the parameterisation, which is fixed and not
> selectable:
>
> ```text
>   y_i     ~ N(x_i'beta + eta_g(i),  sigma_g(i)^2)
>   eta_g   = tau * z_g,                   z_g ~ N(0, 1)
>   sigma_g = exp(mu_s + tau_s * w_g),     w_g ~ N(0, 1)
> ```
>
> reported as `pool_scale` = `tau`, `sigma_pop` = `exp(mu_s)`, `sigma_spread` =
> `tau_s`, plus `group_effect` and `sigma` per group.
>
> **Non-centred, from the start.** The textbook form writes `eta_g ~ N(0, tau^2)`
> directly, and it is unusable: where `tau` is small the admissible `eta` shrink with
> it, so the posterior is a funnel (Neal 2003) whose curvature changes by orders of
> magnitude and which no single step size can explore. Writing `eta_g = tau * z_g`
> makes `z` a priori standard normal and independent of `tau`. There is no `centred`
> option, because the premise of the closed catalog is that a caller cannot select a
> bad parameterisation.
>
> **The hyperpriors are declared on the natural scale, and this differs from
> `payer_alive` on purpose.** A flat prior on `log tau` is `p(tau) ∝ 1/tau`, which for
> a hierarchical variance leaves the *posterior* improper — the likelihood is bounded
> as `tau → 0` and `1/tau` is not integrable there. Flat on `tau` is proper for three
> or more groups, which is why fewer than three is a refusal.
>
> **The default `pool_scale` hyperprior is the response's own standard deviation**, and
> it is the one concrete prior default in this extension.
> [§3](#3-priors-and-why-the-defaults-look-the-way-they-do) rejects concrete defaults
> because they are claims about *units*; a scale taken from the data makes no such claim — double the observations and
> the prior doubles with them. It is not flat because flat was measured and rejected:
> under `p(tau) ∝ 1` the upper tail is long enough that the sampler diverges in it, 34
> times in 8 000 draws on the eight-group fixture, and every divergence is a refusal.
> The spread-of-spreads prior `sigma_spread` defaults to a half-Normal at **one log
> unit**, where a concrete number *is* admissible: a log unit is a factor of e, which
> means the same thing in euros and in kilograms.
>
> **This family asks the sampler for a finer step** than the others — an acceptance
> target of 0.95 against `nuts-rs`'s 0.8, the same dial as Stan's `adapt_delta` and
> raised for the same models. It costs leapfrog steps, not correctness. It is declared
> by the family and is not reachable from SQL.
>
> **Budget more draws than the default.** A hierarchical posterior mixes more slowly
> than a conjugate one: the unpenalised intercept and the group effects trade off along
> a ridge that a diagonal mass matrix cannot precondition, which costs effective sample
> size rather than correctness. Measured on an eight-group panel, 4 × 1 000 draws lands
> R̂ just above the 1.01 gate and 4 × 2 000 clears it. The diagnostics say so; a
> `degenerate` verdict here usually means "take more draws", not "the model is wrong".

## 5. Engines

An **engine** turns a model into draws. The choice is invisible to your SQL — same
function, same output columns, same diagnostics — and is recorded in the draws table
on the `__engine__` row so a reviewer can see which one ran.

| Engine | What it does | Status |
|---|---|---|
| `exact` | Samples the closed-form posterior directly. No approximation. | default for the two conjugate families |
| `laplace` | Fits a Gaussian at the posterior's peak and samples that. | available on every family; **the** engine for `censored_aft` |
| `nuts` | Explores the posterior itself with Hamiltonian dynamics. No closed form needed. | available wherever a gradient is; **the only** engine for `hier_negbin` |
| `exact` | Samples the closed-form posterior directly. No approximation. | default for `conjugate_anomaly` and `pooled_gaussian` |
| `laplace` | Fits a Gaussian at the posterior's peak and samples that. | available on `pooled_gaussian`; the only engine for `payer_alive` |
| `nuts` | General-purpose sampler for models with no closed form. | default for `varying_variance_gaussian`; available wherever a gradient is |

Where a closed form exists, `exact` is both faster and more accurate, so it is the
default. `laplace` and `nuts` exist because they generalise to families that have no
closed form — and, in the meantime, because they provide an **independent check**: on
the conjugate families every engine describes the same posterior by a different route,
and they are tested to agree.

**What NUTS is for.** A Gaussian approximation at the mode is excellent for a
GLM-shaped posterior with reasonable data and poor for a hierarchical variance
parameter with few groups, which is precisely the shape the next families need. NUTS
makes no such assumption: given enough draws it converges on the posterior whatever
shape it is. The price is roughly three orders of magnitude in runtime, so it is not a
default anywhere it is not needed.

Three things follow from NUTS producing a **Markov chain** rather than independent
draws, and they are visible from SQL:

- **`chains` defaults to 4** for this engine, where the other two default to 1. R̂
  compares chains, and one chain cannot support it.
- **R̂ stops being `NULL`.** See [§6](#6-diagnostics).
- **The sampler statistics appear.** `__lp__`, `__divergent__`, `__energy__` and
  `__step_size__` are on the draws table, one row per draw per chain — the first
  engine to populate rows the contract has always reserved.

**Divergences are a refusal, not a warning.** A divergent trajectory means the
integrator left the region it was exploring, so the draws around it are not from the
posterior. One is enough to make `__status__` say `degenerate`; there is no small
budget of acceptable divergences, because there is no small amount of "these numbers
are from somewhere else" that a decision can absorb. See [§7](#7-refusal).

**Is NUTS right?** Checked the same way everything else here is, and twice. It is
compared against `pooled_gaussian`'s **closed-form** posterior — a family with a right
answer, chosen for the certification precisely because it has one — with a tolerance
derived from the Monte Carlo standard error rather than picked to pass. And it has its
own SBC suite, plus a deliberately over-confident fixture that the suite is required to
reject, so the calibration gate is a gate rather than a formality.

**When is the Laplace approximation good enough?** Measured, not asserted. On
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

### Where the Laplace approximation is **not** admissible: `varying_variance_gaussian`

This is the family the roadmap predicted would break it, and the prediction held. The
SBC suite was run per engine, and the two engines disagree completely.

| | `nuts` | `laplace` |
|---|---|---|
| `pool_scale` | 13.3 | **3 942** |
| `sigma_spread` | 6.3 | **4 403** |
| `sigma_pop` | 19.8 | 181 |
| `intercept` | 16.0 | 246 |
| `group_effect` (5 groups) | 9.8 – 24.6 | 131 – 178 |
| `sigma` (5 groups) | 12.8 – 21.0 | 126 – 170 |

χ² at 15 degrees of freedom, 1 024 replications, five groups of ten observations; the
threshold is 37.7. Under NUTS every one of the fourteen parameters is calibrated. Under
Laplace **not one of them is**, the two learned scales are wrong by two orders of
magnitude, and the strong negative slope of their rank histograms (−0.75 and −0.79)
says the approximation places them far *above* the truth rather than merely too
narrowly around it. On top of that, the mode search does not converge at all on 3.1 %
of replications: for those there is no curvature, so there is no posterior to be
miscalibrated.

That is why `nuts` is this family's default, and why the number in this table is
recorded rather than the threshold being loosened until it went green. The result also
sharpens the note in [§8](#8-how-we-know-it-is-right): SBC per family is not enough,
because *per engine* is where this appeared.

Laplace remains reachable by explicit `engine` for anyone who wants a fast
approximation and has read this section. It is not certified, and the numbers above are
what "not certified" means here.


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
> **In detail (NUTS).** NUTS is not implemented here. The sampler is
> [`nuts-rs`](https://github.com/pymc-devs/nuts-rs), maintained by pymc-devs and the
> engine behind nutpie; this extension writes only the adapter — the same `logp` and
> analytic gradient the Laplace engine uses, on the same unconstrained scale, plus the
> warmup discard and the translation of the per-draw diagnostics into the draws
> contract. Adaptation is `nuts-rs`'s dual-averaging step size and diagonal mass
> matrix, 1000 warmup draws by default (`warmup`), discarded. Chains start
> overdispersed — drawn from the Laplace approximation at the mode, widened by a factor
> of two — so that R̂ has a real failure to detect rather than agreement built in by
> starting every chain in the same place. Chains run sequentially and each is seeded
> from `(seed, chain)`, so the draws are byte-identical regardless of how many chains
> were requested or how many threads are in flight.
>
> One consequence worth knowing: a diagonal mass matrix cannot precondition a posterior
> whose correlations are not axis-aligned. `pooled_gaussian` with a `group` column is
> such a posterior — the intercept and the group effects trade off along a ridge — and
> NUTS mixes slowly there, reporting a low ESS and an R̂ above 1.01 rather than pretending
> otherwise. The exact engine has no such difficulty on the same design, which is the
> reason it remains the default for this family.

> **In detail (Laplace).** Laplace works on an unconstrained scale — `sigma` is sampled as
> `log sigma`, so every draw is positive by construction; a Gaussian fitted directly to
> a scale parameter would put mass below zero. The mode is found by Newton iteration on
> an analytic gradient with a backtracking line search, and the covariance is the
> inverse of the negative Hessian there, obtained by differencing the *analytic
> gradient* rather than the log density twice (second differences of a scalar lose
> roughly two-thirds of the available precision).
>
> **Why `payer_alive` is served by Laplace and not by NUTS.** It has no closed form, so
> the question was open. The case for the approximation is that it has only four
> parameters and every customer in the base informs all four — the regime in which a
> Gaussian at the mode is at its best — and the case is settled by its SBC suite rather
> than by argument: measured over 1 024 replications at 800 customers, the rank
> histograms for all four parameters are uniform (chi-squared 13–29 against a 37.7
> threshold at 15 degrees of freedom, slopes below 0.04), and they stay uniform down to
> 100 customers. NUTS would cost a large multiple of the runtime to reproduce a
> posterior that is already calibrated.

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

**Under `exact` and `laplace` this is `NULL`, and that is correct.** Both draw
independently, so there is no Markov chain that could fail to mix and the statistic is
undefined. It returns `NULL` rather than a reassuring `1.00` because a diagnostic that
was never computed must not read as one that passed.

**Under `nuts` it is computed, and it is the gate.** That engine produces a genuine
Markov chain, defaults to four of them, and starts them at overdispersed points so
that chains which have not found the same answer say so. An R̂ above 1.01 makes the fit
`degenerate`.

### Divergences — did the sampler stay where it was integrating?

Only NUTS reports this, on the `__divergent__` rows. A divergence means the
leapfrog integrator lost energy conservation badly enough that the trajectory left the
region it was exploring — usually a posterior with curvature too sharp for the adapted
step size, which is the classic signature of a hierarchical model that wants
reparameterising. The draws around a divergence are not from the posterior, so **any**
divergence is a refusal rather than a warning.

`sum(__divergent__)` is `NULL` for the other engines, and deliberately so: a zero would
read as "the sampler explored cleanly" when the truth is that no sampler ran.

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
joint error is what produces a confidently wrong interval. `hier_negbin`'s SBC suite
carries that assertion *inside the calibration run*: alongside the three population
parameters it ranks `marginal_sd`, the standard deviation of one period's demand for a
randomly chosen group, which reads the level, the pooling scale and the dispersion at
once and is the number a reorder point is set from. Measured over 1024 replications:
chi-squared 8.1, 17.3, 21.3 and 14.3 at 15 degrees of freedom, against a 99.9 %
critical value of 37.7 — and 502, 630, 490 and 616 for the same pipeline with every
draw pulled 40 % toward its own posterior mean, so the gate is a gate on the joint
quantity too.

For `varying_variance_gaussian` that assertion is the posterior spread of a group's
**level**, `intercept + group_effect[g]` — the quantity every statement about that
group is made of. It has an external reference: where the pooling is weak, a group's
level is data-dominated and its posterior standard deviation is `sigma_g / sqrt(n_g)`.
Measured on a six-group panel, the joint answer is 0.169 against a reference of 0.162,
while adding the two *marginal* variances — which is what treating the parameters as
independent would give — returns 3.86, twenty-three times larger. Both halves are
asserted, because a test that only checked the first would pass without the correlation
being right.

## 9. Further reading

- Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, **Bayesian Data Analysis**, 3rd ed. —
  Ch. 2–3 for conjugate models, Ch. 14 for the linear model. The source for every
  posterior in [§4](#4-the-shipped-families).
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
