# The draws contract

> This is the output schema, for consumers that need every detail. To *use* the
> output, the [User Guide](GUIDE.md) is shorter; for what the numbers mean, see
> [Theory §2](THEORY.md#2-what-a-draws-table-actually-is).

**Schema version 1.** Reported by `anofox_bayes_draws_schema_version()` and written
into every draws table as the `__schema_version__` row.

Everything `anofox-bayes` produces is a table of posterior draws in one long format.
Intervals, decisions, what-ifs and diagnostics are then SQL over that table — there is
no second artefact to keep in sync, and no extension state that has to survive the
session.

```
model_id  VARCHAR   deterministic id of this fit
group_id  VARCHAR   the group a parameter belongs to, or '__global__'
chain     INTEGER   chain index, 0-based; -1 on model metadata rows
draw      INTEGER   draw index within the chain, 0-based; -1 on model metadata rows
param     VARCHAR   parameter name
value     DOUBLE    the drawn value; NULL where the model could not estimate it
```

## Three kinds of row

Rows are distinguished by the `param` name and by whether `draw` is negative.

### 1. Model parameters — `draw >= 0`, ordinary `param`

The posterior itself. `group_id` carries the hierarchy level: a group key for
group-specific parameters, `'__global__'` for population-level ones.

```sql
SELECT group_id, quantile_cont(value, 0.5) AS median
FROM draws WHERE param = 'mu' AND draw >= 0
GROUP BY group_id;
```

### 2. Sampler statistics — `draw >= 0`, `__`-prefixed `param`

Per-draw diagnostics, following ArviZ's `sample_stats` convention.

| `param` | Meaning |
|---|---|
| `__lp__` | log posterior density at this draw, up to a constant |
| `__divergent__` | 1.0 if the trajectory diverged, 0.0 otherwise |
| `__energy__` | Hamiltonian energy |
| `__step_size__` | adapted step size |

**These rows are absent when no sampler produced them.** The exact and Laplace engines
draw independently, so they emit none. `sum(value) FILTER (WHERE param = '__divergent__')`
returning 0 over an exact fit means *no rows*, not *no divergences* — check
`count(*)` if the distinction matters to you. Emitting a reassuring zero for a sampler
that never ran would be the more dangerous default.

### 3. Model metadata — `chain = -1, draw = -1`

Emitted exactly once per fit, with `group_id = '__global__'`.

| `param` | Meaning |
|---|---|
| `__schema_version__` | version of this contract |
| `__status__` | `0` converged, `1` degenerate, `2` insufficient_data, `3` failed |
| `__engine__` | `0` exact, `1` laplace, `2` nuts |
| `__seed__` | the seed the fit used |
| `__n_obs__` | observations that survived null filtering |
| `__n_groups__` | distinct groups fitted |
| `__n_chains__`, `__n_draws__` | sampling shape |

Carrying status *inside* the draws table is deliberate. An agent that persists one
table has persisted the fit, its provenance and its refusal status together; there is
no second table to lose, and no session-scoped extension state to depend on.

```sql
-- The gate, in full.
SELECT value = 0 AS safe_to_act_on FROM draws WHERE param = '__status__';
```

## Reserved names

The `__` prefix belongs to this contract. A model parameter may never begin with it —
`ParamName` rejects one at construction, so a column literally named `__lp__` cannot
overwrite a sample statistic and silently corrupt every diagnostic computed from the
table.

Filter user parameters with:

```sql
WHERE param NOT LIKE '\_\_%' ESCAPE '\'
```

## NULL means "not estimable"

A `value` of NULL is not a missing measurement; it is the model declining to produce
one. A lane with a single invoice has no estimable variance, a perfectly-fitted
regression has no residual variance, and no amount of sampling changes either. A
number in those slots would be indistinguishable from an estimate, and telling the two
apart is the whole purpose of the refusal path.

## `model_id` is a pure function of the request

```
model_id = BLAKE3(family, canonical_config, data_fingerprint,
                  resolved_engine, algorithm_version, seed)[:16]
```

Fields are length-prefixed before hashing, so `("ab", "c")` and `("a", "bc")` cannot
collide. The config is rendered key-sorted, so two callers who write the same options
in a different order get the same model. The data fingerprint covers only the columns
the model actually reads, restricted to the rows it actually uses — so adding an
unrelated column to the input relation does not invalidate anything.

Two of the inputs are not part of the caller's request, and both are deliberate:

* **The *resolved* engine, not the configured one.** A caller who omits `engine` gets
  the family's default. If that default later changes, the same config would produce a
  posterior with a different warranty under what would otherwise be the same id.
* **`algorithm_version`**, bumped whenever a change makes identical inputs produce
  different draws. A corrected posterior is exactly that case: inputs unchanged, output
  deliberately different. Without it a cache would serve the old, wrong answer for the
  new, correct request — which has already happened once during development.

Consequences worth relying on:

* **Reproducibility.** Same inputs, same id, same numbers. An auditor can re-run a
  recommendation and check it.
* **Cache detection is a comparison,** not a registry lookup.
* **Refit detection is free.** A changed id means the question changed.

## Ordering

Rows are emitted metadata-first, then draws in `(chain, draw, param)` order, with each
draw's sample statistics immediately after its parameters. This order is part of the
contract so that `LIMIT`ed queries are testable — but do not *depend* on it in
production SQL: DuckDB is free to reorder a result set, and every query above is
order-independent by construction.

## Compatibility

`__schema_version__` moves only for a **breaking** change. The rules, so that a
consumer written today keeps working:

**Consumers must ignore reserved rows they do not recognise.** New `__`-prefixed
metadata and sampler-statistic rows will be added — the NUTS engine brings several —
and their arrival is *not* breaking. Filter on the names you know rather than
assuming a fixed set.

| Change | Breaking? |
|---|---|
| A new `__`-prefixed metadata row | no |
| A new sampler statistic (`__energy__`, …) | no |
| A new model parameter for an existing family | no |
| Row **order** within a draw | no — treat it as unspecified |
| The meaning of an existing reserved row | **yes** |
| A column's type or meaning | **yes** |
| `FitStatus` / `EngineKind` numbering | **yes**, and it is append-only for that reason |

Sampler statistics are uniform across draws within a fit — a fit reports the same
*set* of statistics for every draw, even where the values differ — so the number of
rows per draw is constant and can be relied on within one table. It is not constant
*between* tables written by different engines or versions.
