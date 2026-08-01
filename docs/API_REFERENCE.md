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
| `chains` | integer | `1` | 1 … 64 | Independent chains. See the note below before raising it. |
| `seed` | integer | `20260801` | ≥ 0 (exact integer) | Feeds a ChaCha counter-based stream, so draws are byte-identical across platforms for a given seed. A different seed is a different `model_id`. |
| `engine` | string | family default | `exact`, `laplace`, `nuts` | See [§1.3](#13-engines). |
| `max_draw_megabytes` | integer | `2048` | 1 … 1 048 576 | Ceiling on the in-memory draw buffer, checked *before* allocating. A larger request is refused with a message naming the shape. Raise only if the memory genuinely exists — see [Scalability](SCALABILITY.md). |

**On `chains`.** It defaults to 1, and that is not a limitation to work around. R̂
exists to detect a Markov chain that has not mixed, and both v0.1 engines draw
*independently* — so a second chain buys an R̂ of 1.0 that means nothing. Raise it only
if you want the cross-check for its own sake; the gate in v0.1 is ESS. It becomes
load-bearing when the NUTS engine lands. Memory scales linearly with it.

### 1.3 Engines

| Value | Status |
|---|---|
| `exact` | **Available**, and the default for both families. Samples the closed-form conjugate posterior directly — no approximation, so where it applies it is both faster and more accurate. |
| `laplace` | **Available on both families.** Fits a Gaussian at the posterior mode on an unconstrained scale. Neither family needs it — both are conjugate — so on both it serves as an independent check on the exact posterior rather than as the way to fit. |
| `nuts` | **Not available.** Errors with *"the NUTS engine arrives in 0.2. Until then use 'exact' … or 'laplace' …"*. |

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

`__engine__` in the metadata rows is `0` exact, `1` laplace, `2` nuts.

### 1.4 Null handling and grouping

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
| `pool_scale` | double | no | `1.0` | > 0. Standard deviation of the `N(0, sigma^2 * pool_scale^2)` prior on each group intercept. |
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
| `sigma` | `__global__` | always |

Note the shape: `group_effect` is a *single parameter name* carried across many
`group_id` values, not one name per group. Filter with
`WHERE param = 'group_effect' AND group_id = 'S03'`.

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
needs the NUTS engine (0.2). Fixing it is the documented stepping stone, and small
groups are therefore shrunk toward the population intercept while large ones are
not.

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
> load-bearing when the NUTS engine lands in 0.2.

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
| `nuts` engine | 0.2 | Config value is recognised and rejected with an explanatory error |
| Families F1, F2, F4, F5, F6 | 0.2–0.3 | See [BRD §6](BRD.md) |


A two-argument `anofox_bayes_fit(data, family)` form is **not** planned. DuckDB
refuses to bind a function that has a `TABLE` parameter and multiple overloads,
and it would not be much of a convenience anyway: every family requires at least a
value column, so a fit with no config is never a valid request.
