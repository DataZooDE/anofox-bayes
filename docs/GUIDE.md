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
  - […keep several named what-ifs side by side?](#keep-several-named-what-ifs-side-by-side)
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
INSTALL 'anofox_bayes' FROM 'http://get.erpl.io';   -- start duckdb with -unsigned
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
| **R-hat** | A convergence check that compares independent chains. Under the default `exact` engine (and under `laplace`) it is `NULL` and that is correct: those engines draw independently, so there is nothing to check — gate on ESS instead. Under `engine: 'nuts'` it is computed, defaults to four chains, and must be at or below 1.01. |

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
| "How much of this spare part will be wanted next week, and how much should I stock?" | `hier_negbin` |
| "Most of my catalogue has four weeks of history — can I forecast it at all?" | `hier_negbin` |
| "Is this customer still a customer, or have they quietly gone?" | `payer_alive` |
| "Which accounts on my dunning list are worth chasing?" | `payer_alive` |
| "When will this invoice actually be paid — and will we be covered on the 28th?" | `payment_delay` |
| "What does +5 % on the list price cost me in volume, per segment?" | `hier_elasticity` |
| "How much buffer does *this* segment need to cover 95 % of cases?" | `varying_variance_gaussian` |
| "Which segments are unpredictable, as opposed to merely worse?" | `varying_variance_gaussian` |

Rule of thumb: **one number per group → `conjugate_anomaly`. A response explained by
predictors → `pooled_gaussian`. A duration that has finished, where the tail is the
decision → `payment_delay`. A repeat-purchase history and a churn question →
`payer_alive`. A price move and a volume response → `hier_elasticity`. A question about
a group's *spread* rather than its level → `varying_variance_gaussian`.**

Two pairs are easy to confuse. **`payment_delay` vs `censored_aft`:** if some of the
items have not happened yet — open POs, unpaid invoices — you need `censored_aft`, which
models the not-yet-happened as information rather than as a missing row. `payment_delay`
is for durations that have all completed, and it refuses a non-positive one rather than
guessing. **`hier_elasticity` vs `pooled_gaussian` + `random_slopes`:** both give you a
per-segment elasticity. Use `pooled_gaussian` when the response is well-populated and
you can take its log; use `hier_elasticity` when the volume is a count that is sometimes
zero, or when a thin segment's interval keeps straddling zero and handing you a
"raising the price might sell more" reading you know is an artefact of the width.

The last row is the one people get wrong. `pooled_gaussian` will happily fit a panel by
segment and give you an interval per segment — but it has a single residual scale for
the whole design, so those intervals differ only by how much *data* each segment has,
never by how variable each segment actually is. Two segments with the same mean get the
same interval. If your decision reads a tail, that is the wrong model no matter how the
numbers look.

Every config slot is in the [API Reference](API_REFERENCE.md); the models themselves
are described in [Theory §4](THEORY.md#4-the-shipped-families).

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

### …let the effect differ by store, region or segment?

Add the predictor to `random_slopes`. Each group then gets its own coefficient on it,
shrunk toward the population coefficient — so a group with forty observations is
believed and a group with five borrows strength.

```sql
CREATE TABLE el AS
SELECT * FROM anofox_bayes_fit(
    (SELECT store, log_price, log_units FROM weekly),
    'pooled_gaussian',
    {'y': 'log_units', 'x': ['log_price'], 'group': 'store',
     'random_slopes': ['log_price'], 'pool_scale': 1.5,
     'draws': 2000, 'seed': 42});

-- A store's own elasticity is the population slope plus its deviation, added
-- *within a draw* -- the two are correlated, so two medians would be wrong.
SELECT d.group_id AS store,
       round(median(b.value + d.value), 2)                       AS elasticity,
       anofox_bayes_credible_interval(b.value + d.value, 0.95)   AS ci
FROM el AS d
JOIN el AS b ON b.chain = d.chain AND b.draw = d.draw
            AND b.param = 'beta[log_price]'
WHERE d.param = 'group_slope[log_price]' AND d.draw >= 0
GROUP BY d.group_id;
```

> **What to know before you turn the dial.** The predictor must also be in `x` — a
> random slope is a deviation *from* a population slope, and without one the model
> would be shrinking every group toward "this predictor does nothing". `pool_scale` is
> in **residual standard deviations** and governs the deviations, not the population
> slope; `prior.beta_scale` is the one that shrinks the population slope toward zero,
> which is a different and much stronger claim. And because one `pool_scale` governs
> both intercept and slope deviations — which live on different scales — centre and
> scale the predictor before reading the shrinkage.
>
> `pool_scale` is *not estimated*. It is your assumption about how alike the groups
> are, and the intervals move with it.

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
       m.mu + s.value * anofox_bayes_std_normal(2026, m.row_id::VARCHAR, m.draw) AS y
FROM mu_pred m
JOIN (SELECT draw, value FROM draws WHERE param = 'sigma' AND draw >= 0) s USING (draw);
```

**Use `anofox_bayes_std_normal`, not `random()`.** This is the one place where the
obvious SQL is wrong. `random()` is seeded per *session* by `setseed()` and is not
covered by the fit's `seed`, so a recipe built on it makes the fit reproducible and the
forecast not — measured on this repo, the same draws table gave `P(units > 105)` of
0.142 and 0.132 on consecutive runs. `anofox_bayes_std_normal(seed, key, draw)` is a
pure function of its three arguments, so the same query returns the same numbers
whatever the thread count, the row order, or what the session did earlier.

Read the arguments as *coordinates in a fixed random stream*:

| Argument | What to pass |
|---|---|
| `seed` | any `BIGINT` you choose; record it beside `model_id` and the run regenerates |
| `key` | what is being simulated — a SKU, a lane, a row id. Cast to `VARCHAR` |
| `draw` | the draw index, so each posterior draw gets its own noise |

Vary all three. Reusing one key across rows gives every row the *same* shock, which
looks like a fit with suspiciously smooth forecasts.

`anofox_bayes_uniform(seed, key, draw)` is the same stream before the normal quantile
is applied — use it to build any other distribution, e.g. `-ln(u) / rate` for an
exponential. `anofox_bayes_std_normal` is exactly the normal quantile of
`anofox_bayes_uniform` at the same coordinates.

One consequence worth exploiting: **a scenario and its baseline evaluated at the same
coordinates share their noise**, so their difference is the effect rather than the
effect plus sampling jitter. That is what makes a paired what-if comparison stable
enough to act on.

A worked end-to-end version, including a counterfactual, is in
`test/sql/posterior_predictive.test`; the properties themselves are pinned in
`test/sql/keyed_random.test`.

### …keep several named what-ifs side by side?

Once there are three or four what-ifs and someone has to sign off on one of them, the
`CREATE TABLE counterfactual AS …` above stops being enough: there is no record of
what each branch changed, nothing stops a branch drifting, and nothing marks the
version that was approved.

That is a data-versioning problem, not a Bayesian one, and
[`anofox-scenario`](https://github.com/DataZooDE/anofox-scenario) already solves it —
it gives any DuckDB table Git-like branches, storing only the rows you actually
changed. There is nothing to install on the `anofox-bayes` side and no API to learn:
this extension publishes one artefact (a table of draws) and that extension branches
one thing (a table), so the two compose through the catalog.

Three rules make the combination work.

**1. Branch the assumptions, never the draws.** A counterfactual changes what you
plan to *do*, not what the data *said*. The posterior is evidence; editing it is
fabricating evidence. So the table you branch is the plan — prices, capacities,
promotion flags — and the draws stay exactly as fitted, which is also why N branches
cost N joins and no refits.

**2. Write the predictive step as views over the plan.** `anofox-scenario` rebinds
*unqualified* table references inside a view to the scenario's own tables, so a whole
predictive chain re-evaluates against a branch for free:

```sql
CREATE TABLE plan (week INTEGER PRIMARY KEY, price DOUBLE, promo INTEGER);

-- every reference below is unqualified: that is what makes it re-bindable
CREATE VIEW plan_long AS
SELECT week AS row_id, 'intercept' AS param, 1.0 AS x FROM plan
UNION ALL SELECT week, 'beta[price]', price FROM plan
UNION ALL SELECT week, 'beta[promo]', promo::DOUBLE FROM plan;

CREATE VIEW mu_pred AS
SELECT n.row_id, d.draw, sum(d.value * n.x) AS mu
FROM draws d JOIN plan_long n USING (param)
WHERE d.draw >= 0 GROUP BY n.row_id, d.draw;

CREATE VIEW units_pred AS
SELECT m.row_id, m.draw,
       m.mu + s.value * anofox_bayes_std_normal(4242, m.row_id::VARCHAR, m.draw) AS units
FROM mu_pred m
JOIN (SELECT draw, value FROM draws WHERE param = 'sigma' AND draw >= 0) s USING (draw);
```

```sql
LOAD anofox_scenario;

CALL scenario_create('discount', 'list price 20.00 -> 18.00');
ATTACH 'discount' AS sc_discount (TYPE scenario);
UPDATE sc_discount.plan SET price = 18.0;      -- the base plan is not written to
```

`sc_discount.units_pred` now runs the same chain against the branch's plan. Two
consequences worth checking once: the branch needs a `PRIMARY KEY` (or
`key_columns :=`) on the plan table before it can be `UPDATE`d, and
`SELECT * FROM scenario_diff('discount', 'plan')` is the changelog of what the branch
assumed — the audit trail the hand-rolled version does not have.

**3. Key the noise on the row, never on anything a branch changes.** This is the rule
that decides whether the comparison is usable. Because `anofox_bayes_std_normal` is a
pure function of `(seed, key, draw)`, every branch that keys on the *row id* sees the
same simulated shock, and the shock cancels out of the difference:

```sql
SELECT median(c.profit - b.profit)              AS effect,
       quantile_cont(c.profit - b.profit, 0.025) AS lo,
       quantile_cont(c.profit - b.profit, 0.975) AS hi
FROM profit b JOIN sc_discount.profit c USING (draw);   -- paired on the draw
```

Key it on the price, or let each side draw its own noise, and the difference picks up
the variance of two independent simulated futures instead. Measured in
`test/sql/scenario_counterfactual.test` on exactly one such comparison, the estimate
is unchanged and the verdict is not:

| Comparison | Effect | 95% interval | Verdict |
|---|---:|---|---|
| Paired — same `(seed, row_id, draw)` on both sides | +149 | +81 … +218 | act |
| Unpaired — each side its own noise key | +149 | −21 … +318 | cannot distinguish from zero |

Same model, same branch, same answer, and only one of them is a decision. That file
is the full worked example: one fit, three branches (including a branch off a branch),
the ranking read off the posterior difference, and the winning branch frozen as the
record of what was approved.

### …check my fit is trustworthy?

Two checks. First, did the model refuse?

```sql
-- No WHERE clause: these two macros scan for the __status__ row themselves, which is
-- why they take `param` as well as `value`.
SELECT anofox_bayes_status_text(param, value)   AS status,
       anofox_bayes_is_actionable(param, value) AS safe_to_act_on
FROM draws;
```

If the answer is `insufficient_data`, the next question is *how much* of the fit is
the problem. The status is the worst verdict across every group — one unfittable lane
downgrades all five thousand, deliberately — so the count is what tells you whether
this is a handful of thin groups or the whole dataset:

```sql
SELECT max(value) FILTER (WHERE param = '__n_groups_unready__') AS unready,
       max(value) FILTER (WHERE param = '__n_groups__')         AS groups
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

### …tell which customers have quietly stopped buying?

This is the `payer_alive` family, and it answers a question none of the others can:
your customers never tell you they have left. There is no cancellation event — there
is only a payment that did not arrive, and a payment that has not arrived *yet* looks
exactly the same.

**One row per customer, three numbers.** Whatever your transaction history looks like,
it collapses to this:

```sql
CREATE VIEW payers AS
SELECT customer_id,
       count(*) - 1                                              AS frequency,  -- repeats after the first
       date_diff('day', min(paid_on), max(paid_on))              AS recency,    -- first purchase to last
       date_diff('day', min(paid_on), DATE '2026-08-01')         AS age         -- first purchase to today
FROM payments
GROUP BY customer_id;
```

Three things to get right, because each of them is a way to get a confident wrong
answer:

- **All three are measured from the customer's *first* purchase**, not from a calendar
  date. `age` is how long you have been watching that customer, so a customer acquired
  last month has a small `age` and their silence means much less.
- **`frequency` counts *repeat* purchases**, so a customer who bought once has
  `frequency = 0` and `recency = 0`. Such a customer always scores `P(alive) = 1`,
  which is the model being careful rather than optimistic — churn can only be observed
  *after* a repeat purchase, so there is no evidence either way.
- **Use one unit throughout** (days, or weeks — it does not matter which, but `recency`
  and `age` must agree).

**The fit is over the population, not over customers.** Four numbers come back — `r`,
`alpha`, `a`, `b` — describing how fast your base buys and how readily it churns.
Individual customers are scored afterwards.

```sql
CREATE TABLE draws AS
SELECT * FROM anofox_bayes_fit(
    (SELECT frequency, recency, age FROM payers),
    'payer_alive',
    {'frequency': 'frequency', 'recency': 'recency', 'age': 'age',
     'draws': 4000, 'seed': 42});
```

**Scoring is pure SQL, and needs no re-fit.** Reshape the draws once...

```sql
CREATE TABLE population AS
SELECT draw,
       max(value) FILTER (WHERE param = 'r')     AS r,
       max(value) FILTER (WHERE param = 'alpha') AS alpha,
       max(value) FILTER (WHERE param = 'a')     AS a,
       max(value) FILTER (WHERE param = 'b')     AS b
FROM draws WHERE draw >= 0 GROUP BY draw;
```

...then join any customer list against it. **This is the expression**, and it is the
whole reason the family is BG/NBD rather than the better-known Pareto/NBD, whose
equivalent cannot be written in SQL at all:

```sql
SELECT p.customer_id,
       avg(1.0 / (1.0 + CASE WHEN frequency = 0 THEN 0.0 ELSE (a / (b + frequency - 1)) * pow((alpha + age) / (alpha + recency), r + frequency) END)) AS p_alive
FROM payers p CROSS JOIN population d
GROUP BY p.customer_id
ORDER BY p_alive;
```

`avg` over the draws makes `p_alive` a posterior mean; swap it for
`quantile_cont(..., 0.05)` if you want the pessimistic end, or keep the per-draw values
and you have the full distribution of each customer's `P(alive)`.

The customer list does **not** have to be the one you fitted. Yesterday's draws score
today's arrivals, a segment you did not fit, or the same base re-cut a different way —
the draws mention no customer, so they join against anything with those three columns.
That is what makes this cheap enough to run daily.

> **What the score is worth.** It is not "days since last payment" with extra steps. A
> customer who paid 24 times in a year and has been quiet for 28 weeks scores *lower*
> than one who paid five times and has been quiet for a year, because 28 weeks of
> silence from someone buying fortnightly is overwhelming, and a year of silence from
> an occasional buyer is not. A recency rule cannot express that; this is the reason
> to fit a model at all.

**If it refuses.** `payer_alive` returns `degenerate` when nobody in the base has ever
been seen to stop — when every repeat buyer's last purchase sits at the very end of
their observation window. That happens most often because `age` was taken as the last
payment date rather than as today, and it is a genuine refusal: with no observed
silence anywhere, the data contains nothing about how often customers churn, and any
interval would be invented. Fix the `age` column first. If the shape is real —
a base snapshotted at renewal, say — supply a proper prior:

```sql
{'frequency': 'frequency', 'recency': 'recency', 'age': 'age',
 'prior': {'r':     {'log_mean': 0.0, 'log_sd': 0.7},
           'alpha': {'log_mean': 2.5, 'log_sd': 1.0},
           'a':     {'log_mean': 0.0, 'log_sd': 0.7},
           'b':     {'log_mean': 0.7, 'log_sd': 1.0}}}
```

Priors here are set **on the log scale** (`log_mean` is the log of a typical value,
`log_sd` how many multiplicative factors of doubt around it), because all four
parameters are positive and none has a natural unit. Leave them out and each is flat
on that scale, which is the scale-free default and makes the answer the maximum
likelihood estimate.

A worked end-to-end version, including the refusal, is in
[`test/sql/f5_payer_alive.test`](../test/sql/f5_payer_alive.test).

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
| `anofox_bayes_rhat` is `NULL` | Expected under `exact` and `laplace` — they draw independently, so it is undefined. Gate on ESS. It is computed under `engine: 'nuts'`. [Why](THEORY.md#6-diagnostics) |
| A `nuts` fit is `degenerate` with divergences | The posterior has curvature the adapted step size cannot follow, so those draws are not from it. Raise `warmup`; the fit must not be acted on until it is clean. [Why](THEORY.md#7-refusal) |
| A 5 000-group fit reports `insufficient_data` and you cannot tell which groups | Query `__group_status__` — one row per refused group, with its key in `group_id` |
| A gate that never flags anything | `HAVING ess < 400` fails open on `NULL`; use `anofox_bayes_ess_gate` |
| `invalid config at 'grup'` | A typo — the message names the slot and suggests the intended one |
| `singular or rank-deficient design matrix` | Two predictors carry the same information (e.g. a constant column beside an intercept) |
| Effect estimate looks far too large | A before/after comparison with no control group absorbs the underlying trend |
| `hier_negbin` says `degenerate` and every draw is `NULL` | Every count in the table is zero. There is no demand rate to estimate for a part nobody has ever issued |
| `hier_negbin` rejects `engine: 'laplace'` | It is not a limitation to work around: a Gaussian at the joint mode of a non-centred hierarchy is not a posterior. Drop the slot and let it use NUTS ([Theory §4](THEORY.md)) |
| `payer_alive` says `degenerate` and every draw is `NULL` | No repeat buyer in the base has ever gone quiet — usually `age` was taken as the last payment date instead of today. [What to do](#tell-which-customers-have-quietly-stopped-buying) |
| Every `payer_alive` customer scores `P(alive) = 1` | `frequency` counted *all* purchases instead of repeats, so nobody has had an opportunity to churn |
| A forecast changes between runs of the same fit | The recipe uses `random()`, which the fit's `seed` does not cover. Use `anofox_bayes_std_normal(seed, key, draw)` instead and record the seed alongside `model_id` |
| Every simulated row moves together; the band is implausibly smooth | The same `key` was passed for every row, so they all got the same shock. Key on the thing being simulated |
| A predictive interval barely wider than the interval for the mean | The observation noise was never added — see [the what-if recipe](#ask-a-what-if-without-re-fitting) |

---

**Next:** [Theory](THEORY.md) for what the numbers mean · [API Reference](API_REFERENCE.md)
for every slot · [Draws Contract](DRAWS_CONTRACT.md) for the output schema ·
[Scalability](SCALABILITY.md) for runtime and memory.
