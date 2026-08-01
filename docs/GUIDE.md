# User Guide

How to get answers out of `anofox-bayes`. Task-oriented: each section is a question
you might actually have.

If you want to know *what the numbers mean* rather than what to type, read
[the theory track](THEORY.md) — it is written to be readable without a statistics
background.

**Contents**

- [The five-minute version](#the-five-minute-version)
- [The words you need](#the-words-you-need)
- [Reading the output table](#reading-the-output-table)
- [Choosing a family](#choosing-a-family)
- [How do I…](#how-do-i)
  - […find which group is behaving abnormally?](#find-which-group-is-behaving-abnormally)
  - […measure the effect of something I changed?](#measure-the-effect-of-something-i-changed)
  - […get a service-level quantity?](#get-a-service-level-quantity)
  - […ask a what-if without re-fitting?](#ask-a-what-if-without-re-fitting)
  - […check my fit is trustworthy?](#check-my-fit-is-trustworthy)
  - […handle a refusal?](#handle-a-refusal)
  - […work with counts instead of measurements?](#work-with-counts-instead-of-measurements)
  - […use what I already know about a thin group?](#use-what-i-already-know-about-a-thin-group)
- [Choosing `draws`](#choosing-draws)
- [Common mistakes](#common-mistakes)

---

## The five-minute version

Ten rows, one query, an answer you can read.

```sql
LOAD anofox_bayes;

CREATE TABLE sales(region VARCHAR, units DOUBLE);
INSERT INTO sales VALUES
  ('north',102),('north',98),('north',105),('north',99),('north',101),
  ('south',54), ('south',49),('south',52), ('south',51),('south',48);

SELECT group_id AS region,
       round(median(value), 1)                              AS typical_units,
       round(anofox_bayes_credible_lower(value, 0.95), 1)   AS lo,
       round(anofox_bayes_credible_upper(value, 0.95), 1)   AS hi
FROM anofox_bayes_fit(
       (SELECT region, units FROM sales),   -- a subquery, not `TABLE sales`
       'conjugate_anomaly',                 -- which model
       {'value': 'units', 'group': 'region'})
WHERE param = 'mu' AND draw >= 0
GROUP BY region ORDER BY region;
```

```
┌─────────┬───────────────┬───────┬───────┐
│ region  │ typical_units │  lo   │  hi   │
├─────────┼───────────────┼───────┼───────┤
│ north   │         101.1 │  97.2 │ 104.3 │
│ south   │          50.8 │  47.6 │  54.0 │
└─────────┴───────────────┴───────┴───────┘
```

You have just fitted a Bayesian model per region and read a 95 % credible interval
off it. (If those words are new, the [next section](#the-words-you-need) defines
them — you do not need any statistics background for this guide.)

Three things to notice, because they explain the rest:

1. **The first argument is a subquery.** `TABLE sales` is a parser error.
2. **The function returned a table of samples**, not a summary. `median` and the
   interval macros are ordinary SQL aggregates over those samples — which is why you
   can ask it anything later without re-fitting.
3. **`WHERE param = 'mu' AND draw >= 0`.** The output holds several parameters plus
   some bookkeeping rows. Next section.

## The words you need

Enough to read the rest of this page. [Theory](THEORY.md) explains each properly.

| Term | In one line |
|---|---|
| **posterior** | The range of values the data says are plausible for an unknown quantity, and how plausible each is. The thing this extension computes. |
| **draw** | One sample from that range. You get thousands; together they *are* the posterior. Each row of the output is one draw of one parameter. |
| **credible interval** | "There is a 95 % probability the true value lies between these two numbers." Just two quantiles of the draws. |
| **prior** | What you believed before seeing the data. Leave it out and you get the "let the data speak" default. [More](THEORY.md#3-priors-and-why-the-defaults-look-the-way-they-do) |
| **ESS** (effective sample size) | How many *independent* draws yours are worth. Too low means your answer is noisy for sampling reasons, not data reasons — take more draws. |
| **conjugate** | A model whose posterior has a closed-form solution, so it can be computed exactly instead of approximated. Both shipped families are conjugate. |
| **parameter** | One unknown the model estimates, named in the `param` column. `mu` is a group's level, `sigma` its variability, `lambda` a rate, `beta[x]` the effect of predictor `x`. |
| **exposure** | The denominator of a rate — shipments, consignments, store-weeks. Lets you compare units of different sizes. |
| **R-hat** | A convergence check that compares independent chains. In v0.1 it is `NULL` and that is correct: both engines draw independently, so there is nothing to check. Gate on ESS instead. |

Two more you will meet in error messages: a fit is **degenerate** when the draws are
untrustworthy (take more), and reports **insufficient_data** when the data itself
cannot answer the question ([what to do](#handle-a-refusal)).

## Reading the output table

Every fit returns the same six columns:

```
model_id  group_id  chain  draw  param  value
```

One row per (parameter, draw). With 2 regions × 2 parameters × 1000 draws you get
4 000 rows plus a handful of metadata rows.

| To get | Filter |
|---|---|
| The draws of one parameter | `WHERE param = 'mu' AND draw >= 0` |
| All real draws, no bookkeeping | `WHERE draw >= 0` |
| The fit's status | `WHERE param = '__status__'` (or use the status macros, which find the row themselves) |

`draw >= 0` is the important habit: rows with `draw = -1` are model metadata
(status, seed, row counts), not samples. Full schema in the
[Draws Contract](DRAWS_CONTRACT.md).

**Persist it.** The draws table is the artefact — write it once, query it many times:

```sql
CREATE TABLE draws AS SELECT * FROM anofox_bayes_fit(...);
```

## Choosing a family

| Your question | Family |
|---|---|
| "Is this group behaving unusually?" | `conjugate_anomaly` |
| "What level does each group operate at?" | `conjugate_anomaly` |
| "How many claims per thousand shipments?" | `conjugate_anomaly`, `likelihood: 'poisson'` |
| "Did the change I made have an effect?" | `pooled_gaussian` |
| "What's the effect of price on volume, controlling for season?" | `pooled_gaussian` |

Rule of thumb: **one number per group → `conjugate_anomaly`. A response explained by
predictors → `pooled_gaussian`.**

Every config slot for both is in the [API Reference](API_REFERENCE.md); the models
themselves are described in [Theory §4](THEORY.md#4-the-two-shipped-families).

## How do I…

> The recipes below use table names like `invoices`, `panel` and `contract` as
> stand-ins for **your** data — they are patterns to adapt, not scripts to paste. The
> only fully self-contained examples are [the five-minute version](#the-five-minute-version)
> above and the files in [`test/sql/`](../test/sql/), which are the test suite and
> therefore always current.

### …get my table into the shape it expects?

Two columns is the minimum: something to measure, and optionally something to group
by. Most real preparation is one `SELECT`.

```sql
-- Typical: derive the measure, pick the grain, drop rows that cannot contribute.
CREATE VIEW lane_costs AS
SELECT carrier || '/' || origin || '-' || destination AS lane,   -- the grain
       freight_charge / NULLIF(weight_kg, 0)          AS cost_per_kg,
       invoice_date
FROM invoices
WHERE invoice_date >= DATE '2025-01-01'      -- filter *before* fitting: every row of
  AND freight_charge > 0                     -- the subquery is buffered in memory
  AND weight_kg > 0;
```

Three decisions to make deliberately:

**The grain.** `group` defines what gets its own estimate. Finer grain means more
groups, each with less data — and a group with one row will be
[refused](#handle-a-refusal). Coarsen until every group has at least a handful of
observations.

**NULLs.** A row is dropped if *any* column the model reads is NULL, so you do not
need to pre-clean — but you should know how many rows that removes. `__n_obs__` in the
output tells you how many survived.

**Rates, not counts.** If groups differ in size, use the Poisson likelihood with an
`exposure` column rather than comparing raw counts — see
[below](#work-with-counts-instead-of-measurements).

### …find which group is behaving abnormally?

Not "which is expensive" — that flags whichever group is legitimately expensive.
Compare each group against **its own** contract or history.

```sql
CREATE TABLE draws AS
SELECT * FROM anofox_bayes_fit(
    (SELECT lane, cost_per_kg FROM invoices),
    'conjugate_anomaly',
    {'value': 'cost_per_kg', 'group': 'lane', 'draws': 4000, 'seed': 42});

-- P(true level exceeds the contracted rate), per lane.
SELECT d.group_id AS lane,
       round(anofox_bayes_prob_greater(d.value, c.agreed_rate), 3) AS p_overbilled
FROM draws d
JOIN contract c ON c.lane = d.group_id
WHERE d.param = 'mu' AND d.draw >= 0
GROUP BY d.group_id
ORDER BY p_overbilled DESC;
```

A probability near 1 means the evidence is strong; near 0.6 means "maybe, keep
watching". That gradation is the point — a threshold test would give you neither.

`sigma` is a second, independent signal: a lane whose *variability* jumped has become
unpredictable even if its average has not.

### …measure the effect of something I changed?

Put an indicator for "treated **and** after" in the model. Its coefficient is the
effect.

```sql
CREATE TABLE dd AS
SELECT * FROM anofox_bayes_fit(
    (SELECT store, units, post, treated_post, month FROM panel),
    'pooled_gaussian',
    {'y': 'units', 'x': ['post', 'treated_post', 'month'],
     'group': 'store', 'pool_scale': 20.0, 'draws': 4000, 'seed': 42});

SELECT round(median(value), 2)                            AS effect,
       anofox_bayes_credible_interval(value, 0.95)        AS ci,
       round(anofox_bayes_prob_greater(value, 5.0), 3)    AS p_beats_cost
FROM dd WHERE param = 'beta[treated_post]' AND draw >= 0;
```

`p_beats_cost` is the number a decision actually needs: not "is the effect
significant" but "is it bigger than what it cost us".

> **Two traps.** Include a **control group** that did not receive the change — without
> one the model cannot separate your effect from the underlying trend. And do *not*
> add a `treated` main effect when you have per-unit `group` intercepts: treatment
> status is then a function of the unit and the two are not separately identifiable.

### …get a service-level quantity?

A service level is a **quantile of the posterior**, not mean-plus-k-sigma. The two
agree only if the distribution is symmetric, and the whole reason to carry a posterior
is that it often is not.

```sql
SELECT group_id,
       round(anofox_bayes_service_level_quantile(value, 0.95), 1) AS covers_95pct
FROM draws WHERE param = 'mu' AND draw >= 0
GROUP BY group_id;
```

For the level of a group, that is the query above. For the *next observation* — which
is what stock has to cover — you also need the observation noise; see
[posterior prediction](#ask-a-what-if-without-re-fitting).

### …ask a what-if without re-fitting?

The draws are the fitted model. A counterfactual is a different input table joined
against the same draws — no re-fit, and the answer is itself a distribution.

Put the new rows in **long format**, one row per (observation, predictor), with
`param` matching the names the fit emitted:

```sql
-- The linear predictor, per new row and per draw.
CREATE TABLE mu_pred AS
SELECT n.row_id, d.draw, sum(d.value * n.x) AS mu
FROM draws d JOIN newdata_long n USING (param)
WHERE d.draw >= 0
GROUP BY n.row_id, d.draw;
```

**Check the join matched every predictor** — a missing partner silently drops a term
and gives you a confident, wrong forecast:

```sql
SELECT count(DISTINCT param) FROM draws d JOIN newdata_long n USING (param);
```

For the **next observation** rather than its average, add each draw's own noise. This
interval is wider, and confusing the two is the most common way a forecast interval
ends up too tight:

```sql
CREATE TABLE y_pred AS
SELECT m.row_id, m.draw,
       m.mu + s.value * sqrt(-2 * ln(random())) * cos(2 * pi() * random()) AS y
FROM mu_pred m
JOIN (SELECT draw, value FROM draws WHERE param = 'sigma' AND draw >= 0) s USING (draw);
```

A worked end-to-end version, including a counterfactual, is in
`test/sql/posterior_predictive.test`.

### …check my fit is trustworthy?

Two checks. First, did the model refuse?

```sql
-- No WHERE clause: these two macros scan for the __status__ row themselves, which is
-- why they take `param` as well as `value`.
SELECT anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_is_actionable(param, value) AS safe_to_act_on
FROM draws;
```

Second, are there enough effective draws per parameter?

```sql
SELECT count(*) AS parameters_failing_the_gate
FROM (SELECT group_id, param FROM draws WHERE draw >= 0
      GROUP BY group_id, param
      HAVING NOT anofox_bayes_ess_gate(value, chain, draw, 400));
```

**Use `anofox_bayes_ess_gate`, not a hand-written comparison.** Writing
`HAVING ess_bulk < 400` looks equivalent but fails *open*: ESS is `NULL` where it is
undefined, `NULL < 400` is `NULL` rather than true, and the parameter most in need of
flagging is exactly the one that slips through.

If the gate fails, raise `draws`. See [Theory §6](THEORY.md#6-diagnostics) for what
the statistics mean.

### …handle a refusal?

`insufficient_data` is not an error to work around — it is the model telling you the
recommendation is not supported. Find which group caused it:

```sql
SELECT group_id, param, count(*) FILTER (WHERE value IS NULL) AS null_draws
FROM draws WHERE draw >= 0
GROUP BY group_id, param
HAVING count(*) FILTER (WHERE value IS NULL) > 0;
```

Typical causes and what to do:

| Cause | Remedy |
|---|---|
| A group with one observation | Exclude it, or merge it into a coarser grouping |
| Every observation identical | Nothing to estimate — the data has no variation |
| A perfectly-fitting regression | Too few rows for the predictors; drop one or add data |
| Thin but not hopeless | Supply a `prior` — [see below](#use-what-i-already-know-about-a-thin-group) |

Healthy groups in the same fit are unaffected: only the refused group's draws are
`NULL`.

### …work with counts instead of measurements?

Use the Poisson likelihood, and give it an `exposure` column so the answer is a
**rate** rather than a count:

```sql
SELECT group_id AS carrier,
       round(median(value) * 1000, 2) AS claims_per_1000_consignments
FROM anofox_bayes_fit(
       (SELECT carrier, claims, consignments FROM shipments),
       'conjugate_anomaly',
       {'value': 'claims', 'group': 'carrier',
        'likelihood': 'poisson', 'exposure': 'consignments', 'draws': 4000})
WHERE param = 'lambda' AND draw >= 0
GROUP BY group_id;
```

Without `exposure`, a carrier with twice the volume looks twice as bad. With it,
carriers of different sizes are directly comparable. `value` must be non-negative
whole numbers.

### …use what I already know about a thin group?

Supply a prior. It pulls thin groups toward what you told it, and the pull weakens by
itself as data accumulates.

```sql
-- The fleet rate is about 2.00 with a spread of ~0.30; treat that belief as worth
-- about five observations.
{'value': 'cost_per_kg', 'group': 'lane',
 'prior': {'mu0': 2.00, 'kappa0': 5.0, 'alpha0': 3.0, 'beta0': 0.27}}
```

`kappa0` is the honest dial: it is *how many observations your prior belief is worth*.
A lane with 3 invoices will be dominated by it; a lane with 300 will barely notice.

Leave `prior` out and you get the reference prior, under which the posterior mean is
the sample mean and the interval matches the textbook frequentist one — see
[Theory §3](THEORY.md#3-priors-and-why-the-defaults-look-the-way-they-do).

## Choosing `draws`

`draws` controls Monte-Carlo precision only. It does **not** make your data better
and it cannot rescue a thin group.

| Setting | When |
|---|---|
| `1000` (default) | Fine for medians and 95 % intervals |
| `4000` | Tail probabilities, service levels, anything read at 1 % or 99 % |
| more | Rarely useful; raises memory linearly |

The cost is memory, not just time: roughly `16 × groups × draws` bytes for the Normal
family. A request too large to hold is refused with a message naming the shape rather
than exhausting memory. See [Scalability](SCALABILITY.md).

## Common mistakes

| Symptom | Cause |
|---|---|
| `Parser Error: syntax error at or near "TABLE"` | Pass a subquery: `(SELECT a, b FROM t)`, not `TABLE t` |
| Metadata rows in your aggregates | Add `WHERE draw >= 0` |
| `anofox_bayes_rhat` is always `NULL` | Expected — both engines draw independently, so it is undefined. Gate on ESS. [Why](THEORY.md#6-diagnostics) |
| A gate that never flags anything | `HAVING ess < 400` fails open on `NULL`; use `anofox_bayes_ess_gate` |
| `invalid config at 'grup'` | A typo — the message names the slot and suggests the intended one |
| `singular or rank-deficient design matrix` | Two predictors carry the same information (e.g. a constant column beside an intercept) |
| Effect estimate looks far too large | A before/after comparison with no control group absorbs the underlying trend |

---

**Next:** [Theory](THEORY.md) for what the numbers mean · [API Reference](API_REFERENCE.md)
for every slot · [Draws Contract](DRAWS_CONTRACT.md) for the output schema ·
[Scalability](SCALABILITY.md) for runtime and memory.
