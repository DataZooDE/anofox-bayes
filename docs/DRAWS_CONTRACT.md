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

**The `nuts` engine emits all four**, one row per kept draw per chain. Warmup draws are
not kept, so no statistic row describes one. `__divergent__` is the one to gate on:
a fit with any divergence has `__status__ = degenerate`, because the draws around a
divergent trajectory are not from the posterior.

```sql
-- Did this fit's sampler explore cleanly? NULL means no sampler ran.
SELECT sum(value) FILTER (WHERE param = '__divergent__') AS divergences
FROM draws;
```

### 3. Model metadata — `chain = -1, draw = -1`

Emitted exactly once per fit, with `group_id = '__global__'`.

| `param` | Meaning |
|---|---|
| `__schema_version__` | version of this contract |
| `__sample_from__` | `0` posterior, `1` prior — see below |
| `__family__` | which catalog family produced the table — see below |
| `__status__` | `0` converged, `1` degenerate, `2` insufficient_data, `3` failed |
| `__engine__` | `0` exact, `1` laplace, `2` nuts |
| `__seed__` | the seed the fit used |
| `__n_obs__` | observations that survived null filtering |
| `__n_groups__` | distinct groups fitted |
| `__n_groups_unready__` | how many of those groups the family refused — see below |
| `__n_chains__`, `__n_draws__` | sampling shape |

**`__sample_from__` distinguishes a fit from a pre-fit check.** A prior-predictive
table (`sample_from: 'prior'`) has the identical schema to a posterior one and means
something entirely different — it is what the model assumed *before* seeing any data.
Acting on it in the belief that it is the posterior is acting on no evidence at all,
so the distinction travels with the table rather than living in whoever wrote it.
`sample_from` is part of the canonical config, so the two also carry different
`model_id`s and a cache cannot serve one for the other.

Carrying status *inside* the draws table is deliberate. An agent that persists one
table has persisted the fit, its provenance and its refusal status together; there is
no second table to lose, and no session-scoped extension state to depend on.

```sql
-- The gate, in full.
SELECT value = 0 AS safe_to_act_on FROM draws WHERE param = '__status__';
```

### `__family__` — which model was fitted

| code | family | BRD |
|---:|---|---|
| 2 | `censored_aft` | F2 |
| 3 | `pooled_gaussian` | F3 |
| 5 | `payer_alive` | F5 |
| 7 | `conjugate_anomaly` | F7 |
| 8 | `varying_variance_gaussian` | — (outside the F1–F7 grid) |

`value` is `DOUBLE`, so the family cannot travel under its name; it travels as its
**catalog F-number**, the numbering already fixed in [BRD §6](BRD.md) and used
throughout [the API reference](API_REFERENCE.md) and [the HLD](HLD.md). Reusing that
numbering rather than inventing a registration-ordered one means a family has a single
identity: the gaps are the families this catalog does not ship yet, not accidents. A
family outside the BRD's F1–F7 grid takes the next unused code.

Like `FitStatus` and `EngineKind`, the numbering is **append-only** — these values sit
in tables customers have already persisted, so renumbering one would change what a
table written last quarter says it contains.

Decode it in SQL with the shipped macro, which needs no join and no knowledge of the
table:

```sql
SELECT anofox_bayes_family_text(param, value) AS family FROM draws;
-- conjugate_anomaly
```

### `__n_groups_unready__` — how much of the fit to inspect

The number of groups whose own readiness verdict was not `ready`, out of
`__n_groups__`. `__status__` is the *worst* verdict across every group and stays that
way: a fit covering 500 lanes of which three are unidentifiable is not 99.4 %
trustworthy, it is a fit an agent must look at before acting on any of it. What the
collapse loses is the **scale** of that inspection — three lanes and four hundred
lanes both arrive as `insufficient_data` — and that is what this row restores.

```sql
SELECT
    max(value) FILTER (WHERE param = '__n_groups_unready__') AS unready,
    max(value) FILTER (WHERE param = '__n_groups__')         AS groups
FROM draws;
```

Two things it is not:

* **It is not a diagnostics count.** It counts the groups the *family* refused from
  their sufficient statistics alone — a lane with one invoice, a group whose
  observations are all identical. R-hat and ESS are computed per parameter, not per
  group, so a fit downgraded to `degenerate` by diagnostics can report zero unready
  groups. `__status__` remains the only gate.
* **It is not always exact.** A family that reaches one verdict over one design
  containing every group — `pooled_gaussian` — cannot single out a subset of its
  groups, and reports all of them when it refuses. `conjugate_anomaly`, which fits
  each group independently, reports the exact count. Over-counting is the safe
  direction: it sends an agent to look at more than it must, never at less.

### 3b. Per-group readiness — `chain = -1, draw = -1`, `group_id` names the group

One `__group_status__` row per group the family **refused**, carrying that group's key
in `group_id` and its `FitStatus` in `value`. This is the only reserved row that is not
`__global__`, and the only one whose count varies with the data.

```sql
-- Which lanes must I quarantine, and why?
SELECT group_id AS lane, anofox_bayes_status_name(value) AS verdict
FROM draws WHERE param = '__group_status__';
```

Emitted only for refused groups. A healthy fit emits none, rather than a row per group
saying "fine" — for a 5 000-lane fit that would double the metadata to say nothing.

**Why this exists alongside `__status__`.** The model-level status is the collapsed
worst case and stays that way: a fit covering 5 000 lanes of which three are
unidentifiable is not 99.4 % trustworthy, it is a fit an agent must look at before
acting on any of it. But "must look at" needs somewhere to look. `__n_groups_unready__`
says how many; these rows say which, so an agent can quarantine three lanes instead of
the whole table.

**A family that fits one joint design emits none of these even when it refuses.**
`pooled_gaussian` solves a single system, so a rank deficiency implicates every group
and there is no honest subset to name; `__n_groups_unready__` reports the full count.
Absence of these rows is therefore *not* evidence that every group is fine — read
`__status__` for that.

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

**One thing `model_id` does not cover.** It fingerprints the *fit*, not what you do
downstream of it. A posterior prediction or forward simulation adds noise of its own,
and that noise carries its own seed.

Use `anofox_bayes_std_normal(seed, key, draw)` — or `anofox_bayes_uniform` — for it,
and record that seed beside `model_id`. The pair is then sufficient to regenerate a
recommendation exactly, which is what the reproducibility claim above depends on.

Do **not** use DuckDB's `random()`. It is seeded per session by `setseed()`, not by the
fit, so a recipe built on it is reproducible only if the caller remembers a `SET` —
and the audit trail records nothing about whether they did.

Consequences worth relying on:

* **Reproducibility.** Same inputs, same id, same numbers. An auditor can re-run a
  recommendation and check it.
* **Cache detection is a comparison,** not a registry lookup.
* **Refit detection is free.** A changed id means the question changed.

### The data fingerprint is not in the table, and cannot be

`model_id` is a digest *over* the data fingerprint, so the fingerprint cannot be
recovered from it. A `__data_fingerprint__` row would let a consumer check a draws
table against a source relation without re-deriving the id — and it is **not
shipped**, because there is nowhere honest to put it.

The fingerprint is a hex digest. The `value` column is `DOUBLE`, and a `DOUBLE` has 53
bits of mantissa: any encoding of a 64-bit digest into it silently drops bits, and a
fingerprint that collides silently is strictly worse than one that is absent. A
consumer would use it for exactly one thing — deciding that a table does or does not
describe a given relation — which is the decision a lossy encoding gets wrong while
looking right.

The two lossless routes are both breaking changes: a new `VARCHAR` column (`model_id`
is one, but adding a column changes the schema every consumer binds to), or
overloading `group_id` on the metadata row, which is contractually `__global__`.
Either moves `__schema_version__`, and neither is worth doing on its own.

So this stays open until the contract breaks for another reason, and the fingerprint
travels with the next schema version rather than in a form that would have to be
un-promised later. In the meantime, `__family__`, `__engine__` and `__seed__` are on
the table, and a caller who also has the config can re-derive `model_id` from a
candidate relation and compare — which answers the same question without a new
encoding.

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
metadata and sampler-statistic rows will be added, and their arrival is *not* breaking.
Filter on the names you know rather than assuming a fixed set. The `nuts` engine is the
worked example: it began populating all four sampler statistics without moving
`__schema_version__`, because a consumer that filters on the rows it knows sees exactly
what it saw before.

| Change | Breaking? |
|---|---|
| A new `__`-prefixed metadata row | no |
| A new sampler statistic (`__energy__`, …) | no |
| A new model parameter for an existing family | no |
| Row **order** within a draw | no — treat it as unspecified |
| The meaning of an existing reserved row | **yes** |
| A column's type or meaning | **yes** |
| `FitStatus` / `EngineKind` / `FamilyCode` numbering | **yes**, and it is append-only for that reason |

`__family__` and `__n_groups_unready__` arrived after schema version 1 was published
and did **not** move it: they are new `__`-prefixed metadata rows, the first line of
this table. A consumer that filters on the names it knows is unaffected; one that
assumed the metadata block was eight rows long was relying on something this document
already said not to rely on.

Sampler statistics are uniform across draws within a fit — a fit reports the same
*set* of statistics for every draw, even where the values differ — so the number of
rows per draw is constant and can be relied on within one table. It is not constant
*between* tables written by different engines or versions.
