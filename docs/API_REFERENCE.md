# API Reference — anofox-bayes v0.1

> **New here?** Start with the [User Guide](GUIDE.md) for tasks and recipes, or
> [Theory](THEORY.md) for what the numbers mean. This page is the exhaustive
> reference — every function, every config slot.

The complete SQL surface. Everything documented here exists and runs in v0.1;
anything not listed here does not exist. Roadmap items are collected in
[§6 Not implemented](#6-not-implemented) so they cannot be mistaken for API.

Output rows follow the [draws contract](DRAWS_CONTRACT.md), which is the
authoritative description of the schema, the reserved `__`-prefixed parameter
names, and `NULL` semantics. This document does not duplicate it.

**Contents**

1. [`anofox_bayes_fit`](#1-anofox_bayes_fit)
2. [Model families](#2-model-families)
   - [`conjugate_anomaly` (F7)](#21-conjugate_anomaly-f7)
   - [`pooled_gaussian` (F3)](#22-pooled_gaussian-f3)
   - [`censored_aft` (F2)](#23-censored_aft-f2)
3. [Diagnostics aggregates](#3-diagnostics-aggregates)
4. [Scalar functions](#4-scalar-functions)
5. [Settings](#5-settings)
6. [Not implemented](#6-not-implemented)

---

## 1. `anofox_bayes_fit`

Table in / table out. Fits a cataloged model and returns posterior draws.

```sql
anofox_bayes_fit(data, family, config)
```

| Argument | Type | Description |
|---|---|---|
| `data` | subquery | **A parenthesised subquery**, e.g. `(SELECT lane, cost FROM invoices)`. Not `TABLE invoices` and not a table name string. |
| `family` | `VARCHAR` | A family id from [§2](#2-model-families). Constant, evaluated at bind time. |
| `config` | `STRUCT` (or any value renderable as JSON) | Family slots plus the common slots in [§1.2](#12-common-config-slots). |

Returns the six-column draws contract:

| Column | Type |
|---|---|
| `model_id` | `VARCHAR` |
| `group_id` | `VARCHAR` |
| `chain` | `INTEGER` |
| `draw` | `INTEGER` |
| `param` | `VARCHAR` |
| `value` | `DOUBLE` |

The function **materialises nothing**. Persist with `CREATE TABLE ... AS`:

```sql
CREATE TABLE draws AS
SELECT * FROM anofox_bayes_fit(
    (SELECT lane, cost_per_kg FROM invoices),
    'conjugate_anomaly',
    {'value': 'cost_per_kg', 'group': 'lane', 'draws': 4000, 'seed': 42}
);
```

### 1.1 The config argument

The config is a DuckDB `STRUCT` literal, rendered to JSON internally.

* **Strings** name columns: `{'value': 'cost_per_kg'}`.
* **Lists** are accepted where a slot takes several columns: `{'x': ['post', 'month']}`.
  A single-element slot also accepts a bare string: `{'x': 'month'}`.
* **Nested structs** carry prior hyperparameters: `{'prior': {'kappa0': 1.0}}`.
* **`NULL` means absent.** `{'group': NULL}` and omitting `group` are the same
  request, which is what a caller building a config conditionally expects.
* **Booleans and numbers** pass through as JSON. Where a slot is documented as a
  flag, `0`/`1` are the canonical values (`{'intercept': 0}`).

Unknown slots are rejected before any computation, with a suggestion:

```sql
SELECT * FROM anofox_bayes_fit(
    (SELECT lane, cost_per_kg FROM invoices), 'conjugate_anomaly',
    {'value': 'cost_per_kg', 'grup': 'lane'});
-- Invalid Input Error: anofox_bayes_fit: invalid config at 'grup': unknown option
--   (did you mean 'group'?); this family accepts: value, group, likelihood,
--   exposure, prior, draws, seed, engine, min_obs
```

A missing column is rejected the same way, so a typo never reaches the
mathematics:

```sql
-- Invalid Input Error: anofox_bayes_fit: column 'cost_per_kilo' not found in
--   input data (available: cost_per_kg)
```

An unknown family lists the catalog:

```sql
-- Invalid Input Error: anofox_bayes_fit: unknown model family 'foo'
--   (catalog: pooled_gaussian, conjugate_anomaly)
```

### 1.2 Common config slots

Understood by every family. Each family still declares them, so they pass
`reject_unknown`.

| Slot | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `draws` | integer | `1000` | 4 … 1 000 000 | Posterior draws per chain. 1000 clears the conventional 400 ESS gate for an independent sampler; use 4000 for tail probabilities and service levels. |
| `chains` | integer | `1`, or `4` under `nuts` | 1 … 64 | Independent chains. See the note below. |
| `warmup` | integer | `1000` | 1 … 1 000 000 | Adaptation draws for `nuts`, **discarded** — they never appear in the output. Ignored by `exact` and `laplace`, which have nothing to adapt. Raise it when a NUTS fit reports divergences or a low ESS. Part of `model_id`. |
| `seed` | integer | `20260801` | ≥ 0 (exact integer) | Feeds a ChaCha counter-based stream, so draws are byte-identical across platforms for a given seed. A different seed is a different `model_id`. |
| `engine` | string | family default | `exact`, `laplace`, `nuts` | See [§1.3](#13-engines). |
| `max_draw_megabytes` | integer | `2048` | 1 … 1 048 576 | Ceiling on the in-memory draw buffer, checked *before* allocating. A larger request is refused with a message naming the shape. Raise only if the memory genuinely exists — see [Scalability](SCALABILITY.md). |

**On `chains`.** The default depends on the engine, and has to. Under `exact` and
`laplace` it is 1: those engines draw *independently*, so there is no Markov chain that
could fail to mix and a second chain buys an R̂ of 1.0 that means nothing. Raise it
there only if you want the cross-check for its own sake; the gate is ESS. Under `nuts`
it is 4, because R̂ is the diagnostic that would reveal a chain which had not converged
and one chain cannot support it. Memory scales linearly with it.

### 1.3 `sample_from` — the prior-predictive check

| Value | Meaning |
|---|---|
| `'posterior'` (default) | Condition on the data. The ordinary fit. |
| `'prior'` | Draw from the prior alone, ignoring the data. |

A **prior predictive** is the pre-fit gate (BR-11): before spending anything on a
posterior, check that the model as configured considers the observed world possible.
A prior implying negative delivery times is a modelling error worth finding in the
first second rather than defending in the last.

```sql
-- Does the prior already rule out what we routinely observe?
SELECT anofox_bayes_credible_lower(value, 0.95) AS lo,
       anofox_bayes_credible_upper(value, 0.95) AS hi
FROM anofox_bayes_fit((SELECT depot, days FROM deliveries), 'conjugate_anomaly',
       {'value': 'days', 'group': 'depot', 'sample_from': 'prior',
        'prior': {'mu0': 6.0, 'kappa0': 1.0, 'alpha0': 4.0, 'beta0': 24.0}})
WHERE param = 'mu' AND draw >= 0;
```

Same function, same output contract, one slot changed — so an agent gates on it with
the SQL it already has. The result carries `__sample_from__ = 1` and a distinct
`model_id`.

**It requires a proper prior, and refuses without one.** The shipped defaults are
*reference* priors: scale-free, and improper for exactly that reason — they carry no
finite mass, so there is nothing to draw from. They make a perfectly good posterior
once data arrives. Set an explicit prior to run the check:

| Family | Slots that must be positive |
|---|---|
| `conjugate_anomaly`, Normal | `prior.kappa0`, `prior.alpha0`, `prior.beta0` |
| `conjugate_anomaly`, Poisson | `prior.a0`, `prior.b0` |
| `pooled_gaussian` | `prior.intercept_scale`, `prior.beta_scale`, `prior.a0`, `prior.s0` |

`pooled_gaussian`'s `prior.intercept_scale` exists for this and defaults to flat, which
is what shipped and what an ordinary fit still gets: an intercept prior centred at zero
says something nobody means. A joint prior with one flat coordinate is improper, so
without the slot the check would be permanently unavailable for that family.

Worked example: `test/sql/prior_predictive.test`.

### 1.4 Engines

| Value | Status |
|---|---|
| `exact` | **Available**, and the default for the two conjugate families. Samples the closed-form posterior directly — no approximation, so where it applies it is both faster and more accurate. Not available for `censored_aft`, whose posterior has no closed form; asking for it there is an error rather than a silent substitution. |
| `laplace` | **Available on every family, and the default for `censored_aft`.** Fits a Gaussian at the posterior mode on an unconstrained scale. On the conjugate families it serves as an independent check on the exact posterior rather than as the way to fit; on `censored_aft` it is the fit. |
| `nuts` | **Available wherever a family exposes a gradient.** The No-U-Turn Sampler, via a pinned [`nuts-rs`](https://github.com/pymc-devs/nuts-rs) (pymc-devs, the sampler behind nutpie). Needed for posteriors with no closed form; far slower than the other two, and not the default anywhere it is not needed. |

**What is different about a `nuts` table.** Same columns, same parameter names, same
diagnostics — but it is the only engine that produces a Markov chain, so three things
appear that the others leave absent:

- `chains` defaults to **4**, and `anofox_bayes_rhat` returns a number rather than `NULL`.
- The reserved sample-statistic rows are populated: `__lp__`, `__divergent__`,
  `__energy__` and `__step_size__`, one row per draw per chain. See
  [the draws contract](DRAWS_CONTRACT.md).
- `warmup` (default 1000) adaptation draws run before the kept ones and are discarded.

**Divergences fail the fit.** `sum(__divergent__) > 0` makes `__status__` `degenerate`.
There is no tolerance to raise: the draws around a divergent trajectory are not from
the posterior. If it happens, raise `warmup` first.

Determinism holds for `nuts` exactly as for the other engines: the same seed gives
byte-identical draws, whatever the chain count and whatever DuckDB's thread layout,
because chains are seeded from `(seed, chain)` and run sequentially.
| `exact` | **Available**, and the default for `conjugate_anomaly` and `pooled_gaussian`. Samples the closed-form conjugate posterior directly — no approximation, so where it applies it is both faster and more accurate. `payer_alive` has no closed form and rejects it: *"the exact engine cannot serve family 'payer_alive'"*. |
| `laplace` | **Available on `pooled_gaussian`, and the default and only engine for `payer_alive`.** Fits a Gaussian at the posterior mode on an unconstrained scale. `conjugate_anomaly` exposes no gradient and rejects it: *"the laplace engine cannot serve family 'conjugate_anomaly'"*. |
| `nuts` | **Available**, and the default and only engine for `hier_negbin`. Explores the posterior itself, so it needs no closed form and makes no Gaussian assumption. |

Switching engines changes no caller SQL: same function, same output columns, same
diagnostics. It does change `model_id`, because two posteriors carrying different
warranties must not share an identity.

**When is the Laplace approximation good enough?** Measured rather than asserted. On
`pooled_gaussian` both engines agree on every coefficient to well under a percent at
n = 400. Where they differ is the tails at small n — the exact marginal is a Student-t
and Laplace returns its Gaussian limit, so it slightly *understates* a 99 % interval —
and in the scale parameter, whose discrepancy falls from ~5 % at n = 20 to ~0.1 % at
n = 2000. On `conjugate_anomaly` the answer is closed form and less comfortable: the
Laplace spread for `mu` is too narrow by exactly `1 - sqrt((n-3)/n)`, where `n` is the
**group's own** observation count, since this family fits each group independently.
That is 0.4 % on a group of 400 and 29 % on a group of 6, with `sigma` worse still.
Prefer `exact` here — it is the default, it is faster, and a thin lane is precisely
what an anomaly model is looking at. Both engines have their own calibration suite. See
[Theory §5](THEORY.md#5-engines).

`__engine__` in the metadata rows is `0` exact, `1` laplace, `2` nuts. **Read it.** An
`exact` posterior and a `laplace` one look identical in SQL and do not carry the same
warranty: the first is the posterior, the second is a Gaussian approximation to it.
The family that ran is on the table too, as `__family__`: `1` for `hier_negbin`, `2`
for `censored_aft`, `3` for `pooled_gaussian`, `5` for `payer_alive`, `7` for
`conjugate_anomaly` and `8` for `varying_variance_gaussian` — the catalog F-numbers
where one applies — decoded by `anofox_bayes_family_text(param, value)`. See
[the draws contract](DRAWS_CONTRACT.md#__family__--which-model-was-fitted).

### 1.5 Null handling and grouping

Rows with a `NULL` in any column the model reads (the value/response, any
predictor, any exposure, the group key) are dropped before fitting. `__n_obs__`
reports the survivors. When no `group` slot is given, every parameter is emitted
under `group_id = '__global__'`.

---

## 2. Model families

### 2.1 `conjugate_anomaly` (F7)

> Closed-form Normal or Poisson posteriors per group, for anomaly and outlier
> questions answered as posterior tail probabilities.

Each group — a lane, a carrier, a cost centre — gets its own closed-form
posterior over the level it operates at. "Is this group anomalous?" is then a
question you ask of the draws in SQL, not a threshold baked into the model.

**Config slots**

| Slot | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `value` | column | **yes** | — | The observed quantity. Counts for the Poisson likelihood. |
| `group` | column | no | none (one global fit) | Groups the data; one independent posterior per distinct key. |
| `likelihood` | `'normal'` \| `'poisson'` | no | `'normal'` | |
| `exposure` | column | no | none | Poisson only. `y ~ Poisson(lambda * exposure)`. Rejected for the Normal likelihood with *"only applies to the Poisson likelihood"*. |
| `prior` | struct | no | reference prior | See below. |
| `min_obs` | integer | no | `2` (normal), `1` (poisson) | 1 … 1 000 000. Groups below this are reported `insufficient_data` with `NULL` draws. |
| `draws`, `seed`, `engine` | | | | [§1.2](#12-common-config-slots) |

**Parameters emitted**

| Likelihood | `param` values |
|---|---|
| `normal` | `mu`, `sigma` |
| `poisson` | `lambda` |

`sigma` rather than `sigma_squared`: a standard deviation is on the scale of the
data, so its credible interval is directly readable by whoever has to act on it.

**Model**

*Normal.* `y ~ N(mu, sigma^2)` with a Normal-Inverse-Gamma prior on `(mu, sigma^2)`.
For prior `(mu0, kappa0, alpha0, beta0)` and data with `n` observations, mean
`ybar` and centred sum of squares `SS`:

```
kappa_n = kappa0 + n
mu_n    = (kappa0*mu0 + n*ybar) / kappa_n
alpha_n = alpha0 + n/2
beta_n  = beta0 + SS/2 + kappa0*n*(ybar - mu0)^2 / (2*kappa_n)
```

*Poisson.* `y ~ Poisson(lambda * exposure)` with a `Gamma(a0, rate b0)` prior.
Posterior is `Gamma(a0 + sum(y), rate = b0 + sum(exposure))` — which is what makes
"cost per shipment" and "claims per thousand consignments" the same model with a
different exposure column.

**Prior slots** — `prior` accepts only the slots belonging to the chosen
likelihood; anything else is rejected.

| Likelihood | Slot | Default | Constraint |
|---|---|---|---|
| `normal` | `mu0` | `0.0` | any |
| `normal` | `kappa0` | `0.0` | ≥ 0 |
| `normal` | `alpha0` | `-0.5` | ≥ -1 |
| `normal` | `beta0` | `0.0` | ≥ 0 |
| `poisson` | `a0` | `0.5` | > 0 |
| `poisson` | `b0` | `0.0` | ≥ 0 |

The defaults **are the reference priors**, and that is deliberate: they are
scale-free. Any concrete "weakly informative" default would encode an assumption
about whether costs are measured in cents or in millions, and would quietly
dominate the data for a customer whose units differed from the author's. Under
the reference prior the Normal posterior for `mu` is exactly the Student-t centred
on the sample mean with `n-1` degrees of freedom — the textbook answer, and one an
auditor can check by hand.

`alpha0` is legitimately negative (the reference values in use are `-1`, `-0.5`
and `0`) but is bounded below at `-1`: a value beneath it is not a prior anyone
holds, it merely makes the posterior improper, which would then surface as
"insufficient data" — a confusing diagnosis for what is really a typo.

```sql
-- Invalid Input Error: anofox_bayes_fit: invalid config at 'prior.alpha0':
--   must be >= -1 (reference values are -1, -0.5 or 0), got -5
```

**Examples**

```sql
-- Level per lane, with a weakly informative prior centred on the contracted rate.
SELECT group_id, round(median(value), 3) AS mu
FROM anofox_bayes_fit(
    (SELECT lane, cost_per_kg FROM invoices), 'conjugate_anomaly',
    {'value': 'cost_per_kg', 'group': 'lane',
     'prior': {'mu0': 2.0, 'kappa0': 1.0, 'alpha0': 2.0, 'beta0': 1.0},
     'draws': 4000, 'seed': 1})
WHERE param = 'mu' AND draw >= 0
GROUP BY group_id ORDER BY group_id;
--  BRE-ANT | 2.97    <- shrunk toward mu0 = 2.0 by kappa0 = 1
--  DUS-MIL | 2.579
--  HAM-ROT | 1.999

-- Rate per carrier, with exposure: damage claims per thousand consignments.
CREATE TABLE claims AS
SELECT 'CARRIER-A' AS carrier, i AS period,
       (2 + (i % 3))::BIGINT AS claims, 1000.0 AS consignments
FROM generate_series(0, 23) AS t(i)
UNION ALL
SELECT 'CARRIER-B', i, (11 + (i % 4))::BIGINT, 1000.0
FROM generate_series(0, 23) AS t(i);

SELECT group_id, round(median(value) * 1000, 1) AS claims_per_1000
FROM anofox_bayes_fit(
    (SELECT carrier, claims, consignments FROM claims), 'conjugate_anomaly',
    {'value': 'claims', 'group': 'carrier', 'likelihood': 'poisson',
     'exposure': 'consignments', 'draws': 4000, 'seed': 7})
WHERE param = 'lambda' AND draw >= 0
GROUP BY group_id ORDER BY group_id;
--  CARRIER-A |  3.0
--  CARRIER-B | 12.5
```

**Refusal conditions**

* fewer than `min_obs` observations in a group → `__status__ = 2`
  (`insufficient_data`), that group's draws `NULL`;
* all observations in a group identical, so the variance is not estimable →
  refusal rather than an infinitely confident answer;
* an improper posterior for the configured prior (`alpha_n <= 0`, `beta_n <= 0`
  or `kappa_n <= 0`).

Refusal is per group and worst-wins for the model: a fit covering 500 lanes of
which three are unidentifiable is not 99.4 % trustworthy, it is a fit an agent
must look at before acting on any of it.

---

### 2.2 `pooled_gaussian` (F3)

> Gaussian linear model with a conjugate Normal-Inverse-Gamma posterior and
> optional partial pooling by group; the inference layer for
> difference-in-differences and interrupted time series.

**Config slots**

| Slot | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `y` | column | **yes** | — | The response. |
| `x` | column or list of columns | no | `[]` | Predictors. A bare string is accepted for a single predictor. |
| `intercept` | flag | no | `1` | `0` drops the intercept. |
| `group` | column | no | none | Adds one intercept per group, partially pooled. |
| `random_slopes` | column or list of columns | no | `[]` | Each named predictor also gets a per-group slope deviation, partially pooled. Requires `group`, and every entry must appear in `x`. |
| `pool_scale` | double | no | `1.0` | > 0. Standard deviation of the `N(0, sigma^2 * pool_scale^2)` prior on each group deviation — intercepts and random slopes alike. |
| `prior` | struct | no | flat | See below. |
| `draws`, `seed`, `engine` | | | | [§1.2](#12-common-config-slots) |

At least one of `x`, `intercept` or `group` must be present:

```sql
-- Invalid Input Error: anofox_bayes_fit: invalid config at 'x': a model with no
--   predictors, no intercept and no groups has nothing to estimate
```

**Parameters emitted**

| `param` | `group_id` | Present when |
|---|---|---|
| `intercept` | `__global__` | `intercept` is not `0` |
| `beta[<column>]` | `__global__` | one per entry of `x`, named after the column |
| `group_effect` | the group key | `group` is set; one row-set per group |
| `group_slope[<column>]` | the group key | one per entry of `random_slopes`, named after the column; one row-set per group |
| `sigma` | `__global__` | always |

Note the shape: `group_effect` is a *single parameter name* carried across many
`group_id` values, not one name per group. Filter with
`WHERE param = 'group_effect' AND group_id = 'S03'`. `group_slope[price]` has the
same shape, one name per predictor carried across every group key — so
`GROUP BY param` stays a meaningful diagnostics query at hundreds of groups.

A group's *total* response to a predictor is `beta[price] + group_slope[price]`, and
the two are correlated in the posterior, so they must be added **within a draw**:

```sql
SELECT d.group_id AS store, median(b.value + d.value) AS elasticity
FROM el AS d
JOIN el AS b ON b.chain = d.chain AND b.draw = d.draw
            AND b.param = 'beta[log_price]'
WHERE d.param = 'group_slope[log_price]' AND d.draw >= 0
GROUP BY d.group_id;
```

**Model**

```
y = X beta + eps,   eps ~ N(0, sigma^2)

A     = X'X + P                (P is the prior precision, zero for a flat prior)
b_n   = A^-1 (X'y + P b0)
a_n   = a0 + n/2
s_n   = s0 + (y'y + b0' P b0 - b_n' A b_n) / 2

sigma^2 | y        ~ InvGamma(a_n, s_n)
beta | sigma^2, y  ~ N(b_n, sigma^2 A^-1)
```

Under a flat prior `b_n` is the ordinary least-squares estimate and `s_n` is half
the residual sum of squares, so the marginal posterior for each coefficient is the
Student-t whose interval a frequentist would report — with the difference that
here it is a statement about the coefficient rather than about a procedure, and it
can be pushed through `P(effect > threshold)` in SQL without further theory.

**Prior slots**

| Slot | Default | Constraint | Meaning |
|---|---|---|---|
| `beta_scale` | `∞` (flat) | > 0 | Standard deviation of a zero-centred Normal prior on each slope. The intercept is never penalised. |
| `a0` | `0.0` | any | Inverse-Gamma shape for `sigma^2`. |
| `s0` | `0.0` | ≥ 0 | Inverse-Gamma scale for `sigma^2`. |

`beta_scale` is infinite by default because any finite default would be a scale
assumption about someone else's data. A finite `beta_scale` makes the posterior
mean the ridge estimate with that penalty.

**Pooling.** `pool_scale` is *fixed by configuration, not estimated.* Estimating
it means a hierarchical variance parameter whose posterior is not conjugate and
needs a sampler rather than a formula. The `nuts` engine now exists; the family that
would use it this way does not yet. Fixing it is the documented stepping stone, and small
groups are therefore shrunk toward the population intercept while large ones are
not.

**Random slopes.** `random_slopes` lets each group have its own coefficient on a
predictor, shrunk toward the population coefficient by the same `pool_scale`. This is
a design-matrix change and nothing else — more columns, more entries on the diagonal
prior precision — so the posterior stays closed form and all three engines still
agree on it.

Three things to be precise about, because they are three different claims:

* `prior.beta_scale` shrinks the **population** slope `beta[price]` toward zero. That
  says the effect is small, which is a claim about the world.
* `pool_scale` shrinks each group's `group_slope[price]` toward the population slope.
  That says only that groups are alike until the data says otherwise, which is what
  an analyst asking for random slopes means.
* The predictor **must also appear in `x`**, so the deviation is a deviation *from* a
  population slope rather than from zero. Without it the group slopes would be shrunk
  toward "this predictor has no effect", and the request is refused rather than
  silently reinterpreted.

One caveat that `pool_scale` cannot express: an intercept deviation is in units of the
response and a slope deviation is in response *per unit of the predictor*, and a
single `pool_scale` governs both. Centring and scaling the predictor is what makes the
two comparable, and is worth doing before reading the shrinkage.

The group-slope columns for a predictor sum exactly to that predictor's fixed column,
so the design is rank deficient in the same way an intercept plus a full set of group
dummies is. What identifies the split is the prior: the group block carries positive
precision and the fixed block does not. A design whose *unpenalised* columns are
linearly dependent is still refused.

```sql
-- Per-store price elasticity, with a thin store borrowing strength.
SELECT * FROM anofox_bayes_fit(
    (SELECT store, log_price, log_units FROM weekly),
    'pooled_gaussian',
    {'y': 'log_units', 'x': ['log_price'], 'group': 'store',
     'random_slopes': ['log_price'], 'pool_scale': 1.5,
     'draws': 2000, 'seed': 42});
```

Its executable specification is `test/sql/f3_price_elasticity.test`.

**Example**

```sql
CREATE TABLE dd AS
SELECT * FROM anofox_bayes_fit(
    (SELECT store, units, post, treated_post, month FROM panel),
    'pooled_gaussian',
    {'y': 'units',
     'x': ['post', 'treated_post', 'month'],
     'group': 'store',
     'pool_scale': 20.0,
     'draws': 4000,
     'seed': 42});

SELECT round(quantile_cont(value, 0.025), 2) AS lo,
       round(quantile_cont(value, 0.500), 2) AS effect,
       round(quantile_cont(value, 0.975), 2) AS hi,
       round(avg(CASE WHEN value > 5.0 THEN 1.0 ELSE 0.0 END), 3) AS p_beats_cost
FROM dd WHERE param = 'beta[treated_post]' AND draw >= 0;
--  6.72 | 8.03 | 9.29 | 1.0
```

**Refusal conditions**

* A rank-deficient design — a constant column, a duplicated predictor, an
  intercept plus a full set of group indicators, or a treatment indicator that is
  collinear with the group effects — raises an error rather than returning one of
  infinitely many answers:

  ```sql
  -- Invalid Input Error: anofox_bayes_fit: singular or rank-deficient design matrix
  ```

* A perfectly explained response leaves no residual variance to estimate, so the
  fit is `__status__ = 1` (`degenerate`) with `NULL` draws rather than infinitely
  confident.

---

### 2.3 `censored_aft` (F2)

> Accelerated failure time regression with right censoring (Weibull, lognormal,
> log-logistic, exponential); the inference layer for delivery-promise and
> time-to-event questions.

The model behind "how long until this happens, when some of them have not happened
yet":

```text
log T = x'beta + sigma * W
```

A shipment that has been delivered contributes its density; one still in transit
contributes its **survival** — the probability that it takes at least as long as we
have watched it for. Dropping the ones still in transit is the obvious thing to do and
it is wrong in a known direction: the ones still moving are the slow ones, so
discarding them makes every lane look faster than it is.

**This family is bridged, and carries a weaker warranty.** The likelihood, its
gradient and the mode search come from
[`anofox-statistics`](https://github.com/DataZooDE/anofox-statistics), called
in-process; anofox-bayes contributes the posterior, the draws contract, the refusal
path and the calibration. The posterior is a **Gaussian approximation at the mode**,
not a closed form — `__engine__` reads `laplace` and there is no `exact` alternative.
See [Theory §5](THEORY.md#5-engines) for what that costs and
[§8](THEORY.md#8-how-we-know-it-is-right) for what has been certified.

**Config slots**

| Slot | Type | Default | Meaning |
|---|---|---|---|
| `time` | string | *required* | Duration column. Strictly positive: the elapsed time to the event, or to the moment observation stopped. A non-positive value is a request error naming the column, not weak evidence. |
| `event` | string | *required* | `1` where the event was observed, `0` where the row is still open. Any other value is rejected. |
| `x` | string or list | `[]` | Predictors. Their coefficients are on the **log-time** scale, so `exp(beta)` is a multiplicative effect on duration. |
| `intercept` | 0/1 | `1` | Fit a duration level. |
| `group` | string | — | Fit one **independent** model per group. There is no pooling across groups; a thin group borrows no strength from a thick one. |
| `dist` | string | `weibull` | `weibull`, `lognormal`, `loglogistic`, `exponential`. `exponential` holds `sigma` at 1 rather than estimating it. |
| `prior.beta_scale` | number | `∞` | `N(0, beta_scale)` on each predictor's coefficient. The intercept is never penalised — shrinking a duration *level* toward zero would claim everything happens instantly. |

**Parameters reported** — per group: `intercept`, one `beta[<column>]` per predictor,
and `sigma`. `sigma` is reported for every distribution, including `exponential` where
it was not estimated, so a downstream query need not branch on `dist`.

**Turning the posterior into a promise.** For a Weibull AFT the `p`-quantile of the
duration is closed form, so a posterior *over the promise itself* is arithmetic over
the draws table — no re-fit and no extension code:

```sql
-- The 95th-percentile transit time for an 800 km haul, one draw per posterior draw.
WITH wide AS (
    SELECT group_id AS lane, draw,
           max(value) FILTER (WHERE param = 'intercept')            AS a,
           max(value) FILTER (WHERE param = 'beta[distance_100km]') AS b,
           max(value) FILTER (WHERE param = 'sigma')                AS s
    FROM draws WHERE draw >= 0
    GROUP BY group_id, draw
)
SELECT lane,
       quantile_cont(exp(a + b * 8.0 + s * ln(-ln(1 - 0.95))), 0.9) AS promise_days
FROM wide GROUP BY lane;
```

The inner `ln(-ln(1 - p))` is the quantile of the standard extreme-value error and is
specific to `weibull`; `loglogistic` uses `ln(p / (1 - p))` and `lognormal` needs a
probit. The outer `quantile_cont(..., 0.9)` is the part that matters commercially:
publish a conservative quantile of the posterior, not its median, so the promise
survives the model having been optimistic.

Because every draw carries the level, the distance effect and the spread **jointly**,
that promise inherits the correlation between them. It has to: in a duration model
with a covariate measured away from zero, the intercept and the slope are strongly
anti-correlated, and an interval built from their variances alone can be an order of
magnitude too wide.

**Refusal.** Statuses are mapped from the upstream fit, not inherited from it:

| Situation | `__status__` |
|---|---|
| Every row in a group still open | `degenerate` — plenty of rows, no information about *when* |
| Fewer usable rows than parameters | `insufficient_data` |
| Rank-deficient design at the mode | `degenerate` — the curvature there is not a posterior |
| Mode search did not converge | `failed` — there is no mode, so there is no posterior |
| Non-positive duration, non-binary event | **not a status** — a request error naming the column |

Refusal is per group and worst-wins for the model, with `__n_groups_unready__` saying
how many groups the verdict is about. A refused group's draws are `NULL`, so it still
appears in the table under its own name.

**Worked example:** `test/sql/f2_delivery_promise.test`.
### 2.5 `hier_negbin` (F1)

> Hierarchical count GLM — Poisson or negative binomial with a partially pooled
> per-group level, non-centred — for per-SKU demand and the reorder quantile a
> safety-stock decision reads off it.

One row per group per period. The response is a whole count.

**Config slots**

| Slot | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `y` | column | **yes** | — | The count. Non-negative whole numbers; a fraction is a request error, not a status. |
| `group` | key column | **yes** | — | The SKU, part or item. A hierarchical model needs groups, so this is not optional. |
| `x` | column or list | no | none | Population-level covariates — a promotion flag, a seasonal index. One coefficient each, shared across groups. |
| `exposure` | column | no | none | Time or volume each row was observed over, entering as a `log` offset with coefficient one. Must be `> 0`. With it, `rate` is a count *per unit of exposure*. |
| `likelihood` | `negbinomial` \| `poisson` | no | `negbinomial` | Poisson is the no-extra-burstiness special case. Prefer the default unless you know the data is Poisson: on overdispersed data a Poisson reorder point under-delivers measurably. |
| `min_groups` | integer ≥ 2 | no | `3` | Below this many groups the fit is `insufficient_data`: a pooling scale estimated from fewer describes the sample rather than the catalogue. |
| `prior.intercept.mean` / `.sd` | number / number > 0 | no | `0` / ∞ | Normal prior on the population log rate. Absent `sd` is flat — the scale-free default. |
| `prior.beta.scale` | number > 0 | no | ∞ | Normal prior sd shared by every `x` coefficient. |
| `prior.tau.scale` | number > 0 | no | ∞ | Half-normal scale for the pooling scale. Absent means the reference prior: **uniform on `tau`**, which is proper for three or more groups. |
| `prior.phi.log_mean` / `.log_sd` | number / number > 0 | no | — | Lognormal prior on the dispersion. `log_sd` is what switches the prior on; supplying `log_mean` alone is a config error, because a prior mean with no spread is not a prior. Absent means the reference prior: **uniform on the overdispersion `1/phi`**, which is flat exactly at the Poisson limit. |

**Parameters emitted**

| Parameter | `group_id` | Meaning |
|---|---|---|
| `intercept` | `__global__` | Population log rate. |
| `beta[<column>]` | `__global__` | One per `x` column. |
| `tau` | `__global__` | Pooling scale: the standard deviation of group log rates about the population level. Small `tau` means the catalogue is homogeneous and thin items are shrunk hard. |
| `phi` | `__global__` | Negative binomial dispersion. `Var(y) = mu + mu²/phi`, so **large `phi` is the Poisson limit**; read `1/phi` to talk about overdispersion. Absent under `likelihood: 'poisson'`. |
| `u` | per group | That group's offset from the population level, on the log scale. |
| `rate` | per group | `exp(intercept + u)` — the group's expected count per unit of exposure, with any covariates at zero. This is the number a reorder point is built from. |

**Engine.** `nuts` only, and both alternatives refuse rather than approximate.
`exact` has no closed form to sample. `laplace` is **inadmissible**, not merely worse:
a Laplace posterior is a Gaussian at the joint mode, and a non-centred hierarchy has
none — when every group offset is zero the likelihood does not depend on `tau` at all.
Asking for it errors with *"'hier_negbin' is served by NUTS only…"*. Because NUTS is
the engine, `chains` defaults to `4`, R̂ is a real statistic rather than `NULL`, the
four sampler-statistic rows appear on the draws table, and a single divergence makes
the fit `degenerate`.

**The reorder point, in SQL.** The posterior predictive for the next period is a
mixture of negative binomials — one per posterior draw — and its probability mass
function is closed form, so a service level is a sum over the draws table with no
re-fit and no simulation:

```sql
SELECT part, min(k) FILTER (WHERE cdf >= 0.95) AS reorder_point
FROM (
  SELECT part, k, sum(pmf) OVER (PARTITION BY part ORDER BY k) AS cdf
  FROM (
    SELECT r.group_id AS part, k.k,
           avg(exp(lgamma(k.k + p.value) - lgamma(p.value) - lgamma(k.k + 1)
                   + p.value * ln(p.value / (p.value + r.value))
                   + k.k * ln(r.value / (p.value + r.value)))) AS pmf
    FROM (SELECT group_id, chain, draw, value FROM draws WHERE param = 'rate' AND draw >= 0) r
    JOIN (SELECT chain, draw, value FROM draws WHERE param = 'phi' AND draw >= 0) p
      USING (chain, draw)
    CROSS JOIN (SELECT range AS k FROM range(0, 201)) k
    GROUP BY r.group_id, k.k
  )
) GROUP BY part;
```

Averaging over the draws integrates out the level, the group's own offset, the pooling
scale and the dispersion at once, which is exactly what a plug-in estimate at a point
cannot do and why a thin item's interval comes out honest.

**Validation and refusal**

| Situation | Outcome |
|---|---|
| Fewer than `min_groups` groups | `insufficient_data`, draws still emitted |
| Every count zero | `degenerate`, every draw `NULL` — the rate is identified only in the limit |
| R̂, ESS or a divergence fails | `degenerate` |
| Fractional or negative `y`, non-positive `exposure` | **not a status** — a request error naming the column |

**Worked example:** `test/sql/f1_hier_negbin.test`.

### 2.3 `payer_alive` (F5)

> BG/NBD buy-till-you-die model over per-customer (frequency, recency, age)
> statistics, whose closed-form `P(alive)` rescores a customer base in SQL without
> re-fitting.

One row per customer. All three statistics are measured **from that customer's first
transaction**, in one consistent time unit.

**Config slots**

| Slot | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `frequency` | column | **yes** | — | Repeat transactions *after* the first. A customer with one transaction has `0`. |
| `recency` | column | **yes** | — | Time from the first transaction to the last. `0` when `frequency = 0`. |
| `age` | column | **yes** | — | Time from the first transaction to the end of the observation window — normally *today*, not the last transaction. |
| `min_customers` | integer ≥ 1 | no | `50` | Below this many customers the fit is reported `insufficient_data`. Four population parameters estimated from fewer describe the sample rather than the base. |
| `prior.r.log_mean` | number | no | `0` | Prior mean of `ln r`. |
| `prior.r.log_sd` | number > 0 | no | ∞ | Prior sd of `ln r`. Absent means flat on the log scale — the scale-free default. |
| `prior.alpha.*`, `prior.a.*`, `prior.b.*` | | no | as above | Same two slots for each of the other three parameters. |

There is **no `group` slot**. The four parameters are population level by
construction; to fit segments separately, call the function once per segment.

**Parameters emitted** — four, all at `group_id = '__global__'`:

| Parameter | Meaning |
|---|---|
| `r`, `alpha` | Shape and rate of the `Gamma` spread of per-customer transaction rates. The population mean rate is `r/alpha`. |
| `a`, `b` | Shape parameters of the `Beta` spread of per-customer dropout probability. The population mean dropout per transaction is `a/(a+b)`. |

`a` and `b` are only weakly identified individually — the data speaks clearly about
where the Beta sits and faintly about how wide it is — so read `a/(a+b)` per draw
rather than either alone.

**Scoring customers.** `P(alive)` is closed form and is evaluated in SQL against the
draws, with no re-fit and against any customer list:

```sql
1.0 / (1.0 + CASE WHEN frequency = 0 THEN 0.0 ELSE (a / (b + frequency - 1)) * pow((alpha + age) / (alpha + recency), r + frequency) END)
```

Full recipe in [the guide](GUIDE.md#tell-which-customers-have-quietly-stopped-buying);
worked end to end in `test/sql/f5_payer_alive.test`.

**Validation and refusal**

* `frequency` must be a non-negative whole number; `age` must be `> 0`; `recency`
  must lie in `[0, age]`, and must be `> 0` whenever `frequency > 0`. Each violation
  is a config error naming the slot and the offending row.
* Fewer usable rows than parameters is `insufficient data: N usable rows for 4
  parameters`.
* A base with **no repeat transactions at all** is `__status__ = 1` (`degenerate`):
  the likelihood does not contain `a` or `b` when every frequency is zero.
* A base in which **no repeat buyer has been seen to go quiet** — every `recency`
  equal to its `age` — is also `degenerate`, with `NULL` draws. The likelihood then
  has no interior maximum, and a curvature computed where the search stopped is not a
  posterior. The reason names which of the four mode checks failed. Fix `age` first
  (it is usually the last transaction date rather than today); if the shape is real,
  set a proper `prior`.

---

### 2.4 `varying_variance_gaussian` (no F-number)

> Gaussian linear model with a residual scale per group and a learned pooling scale,
> non-centred; the family for questions about a group's tail rather than its level.

Use it when the decision reads a **spread**: a service level, a payment-delay buffer, a
worst-case lead time. [`pooled_gaussian`](#22-pooled_gaussian-f3) has one residual scale
for the whole design, so two groups with the same mean necessarily get the same
predictive interval — that is structural and no setting changes it. Here each group has
its own `sigma`, drawn from a shared hyperprior so a thin group borrows its spread from
the rest rather than reporting noise.

The second difference is `pool_scale`. In `pooled_gaussian` it is a number you supply;
here it is a **parameter with a posterior**, estimated from how far apart the groups
turn out to be, and its uncertainty widens every group effect. That is why it appears
under `prior` here and at the top level there: what you set is the *scale of its
hyperprior*, not the pooling itself. Writing it at the top level is an error naming the
slot.

`__family__` is `8` rather than an F-number: this family is the hierarchical substrate
the BRD's F4 and F6 will be built on, not either of them.

**Config slots**

| Slot | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `y` | column | **yes** | — | The response. |
| `group` | column | **yes** | — | One level and one scale per group. Required: a family about per-group variance has nothing to say without groups. |
| `x` | column or list of columns | no | `[]` | Predictors, with population-level coefficients. |
| `intercept` | flag | no | `1` | `0` drops the intercept; at least one of `x` or `intercept` must remain. |
| `prior` | struct | no | see below | The hyperpriors. |
| `draws`, `chains`, `warmup`, `seed`, `engine` | | | | [§1.2](#12-common-config-slots) |

**`prior` slots**

| Slot | Default | Meaning |
|---|---|---|
| `pool_scale` | **the response's own standard deviation** | > 0. Half-Normal scale for `pool_scale`, the spread of group levels. |
| `sigma_spread` | `1.0` | > 0. Half-Normal scale for `sigma_spread`, the spread of `log sigma` across groups. One log unit is a factor of e. |
| `sigma_log_mean` | `0.0` | Mean of the Normal prior on `log(sigma_pop)`. |
| `sigma_log_sd` | flat | > 0. Its standard deviation; flat by default. |
| `beta_scale` | flat | > 0. `N(0, beta_scale^2)` on each predictor coefficient. |
| `intercept_scale` | flat | > 0. Flat by default, for the reason `pooled_gaussian` gives. |

`pool_scale` is the one prior in this extension with a concrete default, and it is
concrete only in form: it is taken from the data, so it rescales with your units and
asserts nothing about them. Flat was measured and rejected — its upper tail makes the
sampler diverge, and a divergence is a refusal. See
[Theory §4](THEORY.md#varying_variance_gaussian--a-spread-per-group-and-the-pooling-decided-by-the-data).

**Parameters emitted**

| `param` | `group_id` | Meaning |
|---|---|---|
| `intercept` | `__global__` | Population level, unless `intercept: 0` |
| `beta[<column>]` | `__global__` | One per entry of `x` |
| `pool_scale` | `__global__` | Learned spread of the group levels |
| `sigma_pop` | `__global__` | Population-level residual scale |
| `sigma_spread` | `__global__` | Spread of `log sigma` across groups |
| `group_effect` | the group key | That group's deviation from `intercept` |
| `sigma` | the group key | **That group's own residual scale** |

A group's level is `intercept + group_effect`, and its predictive for one more
observation is that plus `sigma * z`. Both parts matter: the two are strongly
anti-correlated, so combining them draw by draw is not the same as combining their
intervals.

**Engines**

`nuts` is the default and is the only engine certified here. `laplace` is reachable and
is **not certified**: its SBC suite fails on every parameter, by two orders of magnitude
on the learned scales, and its mode search does not converge at all on about 3 % of
fits. The numbers are in [Theory §5](THEORY.md#5-engines). `exact` is refused — there is
no closed form.

**Budget**

Expect to set `draws` above the 1000 default. A hierarchical posterior mixes more
slowly than a conjugate one; measured on an eight-group panel, 4 × 1000 draws leaves R̂
just above the 1.01 gate and 4 × 2000 clears it. A `degenerate` verdict here usually
means "take more draws".

**Refusals**

* Fewer than **three** groups is `insufficient_data`: `pool_scale` is a parameter, and
  fewer than three groups cannot identify it. The message names `pooled_gaussian`,
  which takes the pooling as a setting instead.
* Fewer than two groups with two or more observations each is `insufficient_data`:
  there is nothing for `sigma_spread` to be estimated from.
* Every observation identical is `degenerate`: the residual scale is zero and
  `log sigma` has no mode.
* `sample_from: 'prior'` is an error — the prior has no closed-form draw here.

---

## 3. Diagnostics aggregates

All three take the same three arguments and are used with `GROUP BY param` (and
`GROUP BY group_id` when the model has groups). They return `DOUBLE`, or `NULL`
when the statistic does not exist for the input.

```sql
anofox_bayes_rhat(value, chain, draw)      -> DOUBLE
anofox_bayes_ess_bulk(value, chain, draw)  -> DOUBLE
anofox_bayes_ess_tail(value, chain, draw)  -> DOUBLE
```

| Argument | Type | Notes |
|---|---|---|
| `value` | `DOUBLE` | The draw. |
| `chain` | `BIGINT` | `INTEGER` upcasts implicitly, so the draws contract's own columns bind without a cast. |
| `draw` | `BIGINT` | Order of arrival is irrelevant — the sequence is reconstructed from `(chain, draw)`. |

| Function | Statistic | `NULL` when |
|---|---|---|
| `anofox_bayes_rhat` | Rank-normalised split-R̂ (Vehtari et al. 2021). Above `1.01` indicates chains that have not mixed. | fewer than 2 chains, fewer than 4 draws, chains of unequal length, non-finite draws, or every draw identical |
| `anofox_bayes_ess_bulk` | Bulk effective sample size: how many independent draws the posterior *mean* is worth. | the draws cannot be assessed |
| `anofox_bayes_ess_tail` | Tail ESS: how many independent draws the 5 % and 95 % *quantiles* are worth. Gate service-level and safety-stock decisions on this rather than on bulk ESS. | the draws cannot be assessed |

> **`anofox_bayes_rhat` is `NULL` unless you ask for more than one chain.** Sampling
> defaults to `chains = 1`, and split-R̂ is not defined for a single chain. Set
> `{'chains': 4}` to make it available. The default is deliberate: R̂ exists to catch
> a Markov chain that has not mixed, and both shipped engines draw *independently*,
> so a second chain buys an R̂ of 1.0 that means nothing. `NULL` rather than `1.0`
> matters for the same reason — an agent gating on `rhat <= 1.01` must not be told
> "converged" by a statistic that was never computed. **Gate on ESS**; R̂ becomes
> load-bearing under the `nuts` engine, which defaults to four chains for this reason.

Filter to real parameters before diagnosing — sample statistics and metadata rows
are not posterior draws:

```sql
SELECT group_id, param,
       anofox_bayes_ess_bulk(value, chain, draw) AS ess_bulk,
       anofox_bayes_ess_tail(value, chain, draw) AS ess_tail
FROM draws
WHERE draw >= 0 AND param NOT LIKE '\_\_%' ESCAPE '\'
GROUP BY group_id, param;
```

The gate as a workflow node enforces it — one query, one number:

```sql
SELECT count(*) AS failing_parameters FROM (
    SELECT group_id, param
    FROM draws WHERE draw >= 0
    GROUP BY group_id, param
    HAVING NOT anofox_bayes_ess_gate(value, chain, draw, 400)
);
```

Because `NULL` fails a `<` comparison rather than passing it, a parameter whose
ESS could not be computed is *not* silently counted as passing — but it is also
not counted as failing. If you need "uncomputable is a failure", test explicitly
with `IS NULL`.

---

## 4. Scalar functions

```sql
anofox_bayes_version()               -> VARCHAR   -- e.g. '0.1.0'
anofox_bayes_draws_schema_version()  -> INTEGER   -- e.g. 1
```

`anofox_bayes_version` is the extension/crate version. `anofox_bayes_draws_schema_version`
is the version of the [draws contract](DRAWS_CONTRACT.md), and matches the
`__schema_version__` row written into every fit. It moves only for a breaking
change to column meaning or reserved-name semantics.

```sql
SELECT anofox_bayes_version() AS version,
       anofox_bayes_draws_schema_version() AS draws_schema;
--  0.1.0 | 1
```

### 4.1 Keyed randomness

```sql
anofox_bayes_uniform(seed BIGINT, key VARCHAR, draw BIGINT)    -> DOUBLE  -- in (0, 1)
anofox_bayes_std_normal(seed BIGINT, key VARCHAR, draw BIGINT) -> DOUBLE  -- N(0, 1)
```

Noise for a predictive step. **Use these rather than DuckDB's `random()`**, which is
seeded per session rather than by the fit and therefore makes any recipe built on it
irreproducible — see [the Guide](GUIDE.md#ask-a-what-if-without-re-fitting).

Both are *pure functions of their three arguments*: the value is a function of where
it sits in the stream, not of how many values were produced before it. So the same
query returns the same numbers under any thread count, any row order, and any
re-execution — properties pinned in `test/sql/keyed_random.test`.

| Argument | Meaning |
|---|---|
| `seed` | Any value you choose. Record it beside `model_id` and the run regenerates. |
| `key` | What is being simulated — a SKU, a lane, a row id. Cast to `VARCHAR`. |
| `draw` | The draw index, so each posterior draw carries its own noise. |

`anofox_bayes_std_normal` is exactly the standard normal quantile of
`anofox_bayes_uniform` at the same coordinates. That identity is contractual: it lets
you build any other distribution by applying its own quantile function to
`anofox_bayes_uniform` and inheriting every property above.

```sql
-- An exponential with rate 2, on the same reproducible stream.
SELECT -ln(anofox_bayes_uniform(42, sku, draw)) / 2.0 FROM ...;
```

The uniform is open at both ends, so `ln(u)` and `ln(1 - u)` are always finite —
DuckDB raises on `ln(0)`, so a closed endpoint would abort the query rather than
quietly produce an infinity.

Two consequences worth using deliberately:

* **Paired scenarios share their noise.** A baseline and a what-if evaluated at the
  same `(seed, key, draw)` see the same shock, so their difference is the effect
  rather than the effect plus sampling jitter.
* **Vary the key across rows.** Passing one key for every row gives every row the
  identical shock, which shows up as an implausibly smooth forecast band.

---

## 5. Settings

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `anofox_telemetry_enabled` | `BOOLEAN` | `true` | Set `false` to disable anonymous usage telemetry for the session. |
| `anofox_telemetry_key` | `VARCHAR` | DataZoo PostHog key | Redirect telemetry to your own PostHog project. |

```sql
SET anofox_telemetry_enabled = false;
```

See [TELEMETRY.md](../TELEMETRY.md) for what is and is not collected.

---

## Posterior prediction

There is **no `predict` function**, and there will not be one taking both the draws
and the new rows: DuckDB permits a table function at most one subquery parameter, so
that signature cannot bind.

For a linear model the posterior predictive is a join. The recipe,
including how to add observation noise and how to check the join matched every
predictor, is in
**[the Guide](GUIDE.md#ask-a-what-if-without-re-fitting)**; a worked end-to-end
version is `test/sql/posterior_predictive.test`.

## Decision macros

SQL macros over a draws table. Each is a short expression you could write by hand;
shipping them makes the idiom consistent and discoverable, and a macro is transparent
in `duckdb_functions()` — which matters when the output goes into an audit trail.

All are **aggregates over `value`**, so they compose with `GROUP BY group_id, param`
exactly as the underlying quantile functions do. Filter to real draws first
(`WHERE draw >= 0`).

| Macro | Returns | Notes |
|---|---|---|
| `anofox_bayes_status_name(value)` | VARCHAR | Decodes one `FitStatus` value. Scalar, unlike `anofox_bayes_status_text`, which aggregates a whole table to find the model-level `__status__` row — use this one on the per-group `__group_status__` rows |
| `anofox_bayes_credible_lower(value, level)` | DOUBLE | Lower end of an equal-tailed interval |
| `anofox_bayes_credible_upper(value, level)` | DOUBLE | Upper end |
| `anofox_bayes_credible_interval(value, level)` | STRUCT(lower, median, upper) | The interval as one column |
| `anofox_bayes_prob_greater(value, threshold)` | DOUBLE | `P(param > threshold)` |
| `anofox_bayes_prob_less(value, threshold)` | DOUBLE | `P(param < threshold)` |
| `anofox_bayes_service_level_quantile(value, level)` | DOUBLE | The quantity covering demand in `level` of futures |
| `anofox_bayes_ess_gate(value, chain, draw, min_ess)` | BOOLEAN | Whether a parameter clears its ESS gate. **NULL-safe: an undefined ESS is a failure.** Use this rather than comparing ESS by hand — `HAVING ess_bulk < 400` fails *open*, because `NULL < 400` is `NULL` and the parameter most in need of flagging slips through |
| `anofox_bayes_rhat_gate(value, chain, draw, max_rhat)` | BOOLEAN | Whether R̂ clears its gate. Coalesces to **true**, not false: an absent R̂ is legitimate under an independent sampler, where there is no chain that could have failed to mix |
| `anofox_bayes_status_text(param, value)` | VARCHAR | `converged` / `degenerate` / `insufficient_data` / `failed` |
| `anofox_bayes_is_actionable(param, value)` | BOOLEAN | True only for `converged` |
| `anofox_bayes_family_text(param, value)` | VARCHAR | `pooled_gaussian` / `conjugate_anomaly`, decoded from the `__family__` row; `unknown` for a code this build does not know |

`anofox_bayes_prob_greater` is the one that earns its place: *"is the effect bigger
than the rollout cost?"* is a mean of an indicator over the draws, with no
distributional theory in between.

```sql
-- Is the intervention worth 5 units per store-month?
SELECT anofox_bayes_prob_greater(value, 5.0) AS p_worth_it,
       anofox_bayes_credible_interval(value, 0.95) AS ci
FROM dd WHERE param = 'beta[treated_post]' AND draw >= 0;
```

`anofox_bayes_service_level_quantile` is a posterior quantile, **not** mean-plus-k-sigma.
The two coincide only when the posterior happens to be symmetric, and the whole reason
to carry a posterior is that it often is not.

```sql
-- The gate, in one row.
SELECT anofox_bayes_status_text(param, value)  AS status,
       anofox_bayes_is_actionable(param, value) AS safe_to_act_on
FROM draws;
```

## 6. Not implemented

Listed so they are not mistaken for API. None of the following exist in v0.1;
calling them is a syntax or binder error.

| Planned surface | Phase | Notes |
|---|---|---|
| A prior-predictive check function | 0.3 | BRD BR-11 |
| `anofox_bayes_draws(model_id)` / `anofox_bayes_status(model_id)` | — | Superseded in v0.1: draws are the caller's own table, and status travels on a `__status__` row inside it. |
| `anofox_scenario` catalog registration / branch-versioned counterfactuals | 0.2–0.3 | BRD BR-9 |
| Async / job-style fit (`fit_async` + polling) | deferred | HLD §6; v0.1 accepts blocking table-function semantics |
| `anofox_bayes_predict(draws, newdata, kind)` | **never** | DuckDB allows a table function at most one subquery parameter, so this signature cannot bind. Posterior prediction is a join today — see [the Guide](GUIDE.md#ask-a-what-if-without-re-fitting). A *different* predictive surface may appear in 0.3; this one will not. |
| Families F1, F2, F4, F5, F6 | 0.2–0.3 | See [BRD §6](BRD.md) |


A two-argument `anofox_bayes_fit(data, family)` form is **not** planned. DuckDB
refuses to bind a function that has a `TABLE` parameter and multiple overloads,
and it would not be much of a convenience anyway: every family requires at least a
value column, so a fit with no config is never a valid request.
