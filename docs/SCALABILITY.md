# Runtime profile and scalability

> Practical usage is in the [User Guide](GUIDE.md); this page is the measured
> runtime profile and its limits.

Measured, not estimated. Every number here comes from `validation/bench.py` and
`cargo run --release --example scale_profile` on one machine (32 cores, Linux
x86_64, DuckDB `SET threads=8`, `conjugate_anomaly` with a Normal likelihood and
one group column); reproduce with the commands at the bottom.

**Read this before pointing a fit at production-scale data.** The headline is that
`conjugate_anomaly` now fits its groups in parallel and holds the whole posterior in
memory. The parallelism is worth about **1.6x** end to end, not the 6x the crate-side
speedup would suggest, and the section on where the rest of the time goes says why.

## What was measured

`before` is the same binary with `RAYON_NUM_THREADS=1`, which reproduces the v0.1
figures to within noise; `after` is the default pool. Best of five runs each — the
noise on a shared machine is one-sided, so the fastest run is the closest estimate of
the cost of the work rather than of the work plus whatever else was scheduled.

| groups | rows | draws | before | after | speedup | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 400 | 1 000 | 0.12 s | 0.13 s | — | 50 MB |
| 1 000 | 104 000 | 1 000 | 0.89 s | 0.57 s | 1.57x | 186 MB |
| 5 000 | 520 000 | 1 000 | 4.38 s | 2.76 s | 1.59x | 828 MB |
| 5 000 | 520 000 | 4 000 | 16.65 s | 8.98 s | 1.85x | 2 963 MB |
| 20 000 | 2 080 000 | 1 000 | 17.36 s | 10.85 s | 1.60x | 3 229 MB |

The 5 000-group row is BRD BR-1's acceptance case (5 000 SKUs × 104 weeks). It now
completes in under three seconds. **Peak memory is unchanged** — this work bought
time, not bytes.

At a hundred groups there is nothing to parallelise and the pool costs a millisecond
or two. That is the right shape: the overhead is flat and the benefit grows.

## Threading: two pools, and only one of them matters

| DuckDB threads (`SET threads`) | wall (2 000 groups) |
|---:|---:|
| 1 | 1.11 s |
| 2 | 1.11 s |
| 4 | 1.10 s |
| 8 | 1.08 s |
| 16 | 1.10 s |

Still flat, and still for the original reason: `BayesFitGlobalState::MaxThreads()`
returns 1 and the operator buffers its whole input before fitting. A posterior is a
function of *every* observation, so the fit genuinely cannot start before the last row
arrives.

| fit threads (`RAYON_NUM_THREADS`) | wall (2 000 groups) |
|---:|---:|
| 1 | 1.74 s |
| 2 | 1.40 s |
| 4 | 1.49 s |
| 8 | 1.25 s |
| 16 | 1.17 s |
| default (= cores) | 1.16 s |

This is the pool that does the work. `conjugate_anomaly` fits each group
independently, so sampling and diagnostics run one rayon task per group and per
parameter respectively. `pooled_gaussian` is untouched and cannot be treated this
way: a pooled fit is one joint system, which is the whole point of pooling.

**Determinism holds across both thread counts.** `SET threads` ∈ {1, 8} and
`RAYON_NUM_THREADS` ∈ {1, 16} all produce a byte-identical draws table (verified by
MD5 of the full result, `validation/bench.py --threads`). That is not an accident of
scheduling, it is constructed:

* Each group draws from `BayesRng::for_group(seed, chain, key)` — a stream keyed by
  the group's **own identity**, not by its position in a shared sequential stream.
  So a group's numbers do not depend on which task ran first, on how the rows were
  ordered, or on whether the group was fitted alone or beside twenty thousand others.
  `a_groups_draws_do_not_depend_on_the_order_the_groups_arrived_in` and
  `a_group_gets_the_same_draws_whether_it_is_fitted_alone_or_in_company` pin it, and
  both fail against the old shared-stream implementation.
* Parallel results are written back at indices fixed by the parameter list, and
  `diagnose` collects over an indexed parallel iterator, which preserves order.

Because the per-group streams are new, **the numbers themselves changed**:
`ALGORITHM_VERSION` moved to 3, so a fit re-run on this build gets a different
`model_id` from the same request on v0.1 rather than silently serving old draws under
a new identity.

## Where the time actually goes

Crate-side phases, from `cargo run --release --example scale_profile`, at 5 000
groups × 104 periods × 1 000 draws:

| phase | before | after |
|---|---:|---:|
| compile (partition, fingerprint, per-group conjugate updates) | 144 ms | 47 ms |
| sample (1 000 draws × 10 000 parameters) | 256 ms | 33 ms |
| render (10 M long-format rows) | 82 ms | 82 ms |
| diagnostics (R̂, bulk ESS, tail ESS per parameter) | **1 422 ms** | 67 ms |
| **total** | **1 904 ms** | **229 ms** |

and at 20 000 groups: 7 605 ms → 946 ms, an 8x crate-side improvement.

Three findings, none of which were what the previous version of this page predicted:

**Diagnostics were the bottleneck, not sampling.** R̂ and the two ESS estimators cost
three quarters of the crate's time at 5 000 groups and, at 20 000, more than
everything else together. They are also the easiest thing here to parallelise safely:
each parameter is read on its own and nothing is shared. 21x at 5 000 groups, 19x at
20 000.

**Partitioning the rows cost more than fitting them.** Of a 144 ms compile, 113 ms was
`group_rows` probing a `BTreeMap` twice per row and 27 ms was the fingerprint hash;
the per-group conjugate updates were about 7 ms. One hash lookup per row instead took
`group_rows` to 18 ms. **The per-group fits are deliberately still serial** — running
them on rayon was tried and made compile *slower*, 199 ms against 150 ms, because
allocating each group's value vector on a worker costs more than the arithmetic it
feeds.

**The fit is now a minority of the query.** At 5 000 groups the 2.76 s breaks down as
0.15 s generating the input relation, ~1.2 s in the operator and the FFI marshalling
10 M rows across the boundary, 1.26 s for DuckDB to materialise the output table — and
0.23 s of actual inference. That is why 8x in the crate is 1.6x in SQL, and it is
where any further work has to aim.

## Memory model — where the bytes go

Unchanged by this work. Three copies of the data exist at peak:

1. DuckDB's own materialisation of the input relation,
2. the operator's column-major copy in `BayesFitGlobalState` (needed because DuckDB
   recycles chunk buffers between calls, so the FFI cannot borrow them),
3. the draw buffer in the Rust handle: `chains × draws × params × 8` bytes, held
   whole until the last row is emitted.

For the Normal likelihood, `params = 2 × groups`. So

```
draw bytes  ≈  16 × groups × draws × chains
```

5 000 groups × 4 000 draws ≈ 320 MB of draws, plus the output DuckDB table itself,
which is the larger share — 2 963 MB peak for that fit, so the draw buffer is about
11 % of it. That ratio is why lazy draw emission is not the win it sounds like; see
"Known gaps".

Group-parallel sampling adds one bounded staging buffer, not a second posterior: a
group produces its draws contiguously and they are transposed into the chain-major
output a 4 MiB slab at a time.

**An oversized request is refused, not attempted.** `max_draw_megabytes` (default
2048) is checked with overflow-safe arithmetic *before* allocating:

```
invalid config at 'draws': this fit would need 15258 MB of draws
(200000 parameters x 1 chain(s) x 10000 draws), above the 2048 MB limit.
Reduce `draws`, fit fewer groups at a time, or raise `max_draw_megabytes`
if the memory is genuinely available
```

Before this guard, that request aborted the DuckDB process — Rust's allocator aborts
on failure, so an over-ambitious query took the whole session with it. Raise the slot
only when the memory genuinely exists.

## Practical guidance

- **Default `draws = 1000` is the right starting point.** It clears the ESS gate for
  an independent sampler; more draws buy precision you probably cannot use.
- **Batch very wide group sets.** 20 000 groups in one call still costs ~3 GB. Fitting
  in chunks of a few thousand and unioning the draws tables costs the same total time
  with a fraction of the peak, and each chunk gets its own `model_id`. Since the
  per-group streams are keyed on the group, **a group's draws are identical whichever
  batch it lands in** — batching no longer perturbs the numbers.
- **Filter before fitting.** Only the columns named in the config are read, but every
  row of the subquery is buffered. `(SELECT lane, cost FROM invoices WHERE year = 2026)`
  is materially cheaper than passing the whole table.
- **The output table is usually bigger than the fit.** Persist with
  `CREATE TABLE ... AS SELECT`, then query the draws; don't re-fit to ask a second
  question. That is the entire point of posterior-as-a-table.
- **Leave `RAYON_NUM_THREADS` alone** unless the DuckDB process shares the machine
  with something that must not be starved. The default is the number of cores, and
  the operator is single-threaded from DuckDB's point of view, so the pool is not
  competing with a partitioned scan for the same cores.

## `pooled_gaussian` scales in *groups*, not rows

The table above measures `conjugate_anomaly`, which fits each group independently.
`pooled_gaussian` is a different shape and a much harder limit: it solves one dense
joint system, so cost grows with the **number of coefficients**, and the number of
coefficients grows with the number of groups.

| groups | rows | wall |
|---:|---:|---:|
| 50 | 1 000 | 0.15 s |
| 200 | 4 000 | 0.26 s |
| 400 | 8 000 | 1.39 s |
| 800 | 16 000 | 9.99 s |

Roughly **8× per doubling** — consistent with the `O(n·p²)` accumulation of `X'X` plus
an `O(p³)` solve. Extrapolated, 10 000 groups is a 16 GB design matrix and hours of
arithmetic.

That is by design: pooling means one joint system, and one joint system is what makes
a thin group borrow strength from the rest. It is the right model for **tens to
hundreds** of groups — segments, plants, depots, stores. It is the wrong model for
tens of thousands of customers or SKUs. Group parallelism does not apply and was not
attempted.

A request that would not finish is **refused before allocating**, with a message
naming the shape and pointing at `conjugate_anomaly`, which fits groups independently
and scales to tens of thousands. `max_design_megabytes` (default 512) is the dial.

## Known gaps

Recorded rather than hidden.

1. **The FFI row boundary is the new bottleneck, and it is single-threaded.** At
   5 000 groups the crate spends 0.23 s and the query takes 2.76 s; roughly 1.2 s of
   the difference is `BayesFitFinalize` pulling 10 M rows through
   `anofox_bayes_ffi_fit_rows` one DuckDB vector at a time, re-interning every
   `model_id`, `group_id` and `param` string per row. Dictionary vectors for the three
   string columns, or a wider hand-off, would attack the largest remaining share.
   This is C++ work in `src/table_functions/bayes_fit.cpp`.

2. **The input relation is still fully buffered.** A relation larger than RAM will not
   stream, and the Rust side cannot fix it. The buffering is imposed by
   `BayesFitGlobalState`: `BayesFitInOut` appends every chunk's values into
   `numeric_values` / `key_values`, and `RunFit` hands the finished columns to
   `anofox_bayes_ffi_data_new` in one call. By the time the core sees anything it is
   already a whole materialised relation, and `DataView` is defined as a borrow of
   one. Both shipped families need only per-group sufficient statistics — `n`, `Σy` and
   `Σy²` for the Normal, `n`, `Σy` and total exposure for the Poisson — so `conjugate_anomaly` *could* accumulate
   in constant memory, but only behind a different seam: a streaming FFI
   (`ffi_accumulate(chunk)` per `BayesFitInOut` call, `ffi_finish()` at finalize), a
   `SufficientStatistics` trait beside `ModelFamily::compile`, and a fallback for
   families that genuinely need the rows. Two further constraints make it more than a
   refactor: `data_fingerprint` hashes every value in row order and feeds `model_id`,
   and group order is first-seen order, so a streaming accumulator must reproduce both
   exactly or every cached `model_id` moves. Scope it as C++ plus core, not core
   alone. **It remains the single biggest available win for memory**, and it is not
   available for time.

3. **The posterior is still materialised whole before the first row is emitted, and
   this is worth less than it looks.** Each draw is an independent sample, so draws
   could in principle be generated per output chunk — but the buffer being saved is
   320 MB of a 2 963 MB peak at 5 000 groups × 4 000 draws, about 11 %, because
   DuckDB's own output table dominates. It is also blocked by diagnostics: R̂ and ESS
   are autocorrelation statistics over a parameter's whole chain, so the draws have to
   exist somewhere before the fit can be graded, and grading happens before the first
   row is emitted. Generating them twice — once to diagnose, once to emit — trades 11 %
   of peak memory for roughly a doubling of sampling cost.

   A design that would have unlocked it was measured and rejected: keying the stream
   on `(seed, chain, group, draw)` rather than `(seed, chain, group)` makes every draw
   a pure function of its coordinates, so any draw could be regenerated on demand. It
   costs a BLAKE3 hash and a ChaCha20 initialisation per draw per group: 1.10 s for
   5 M of them against 0.12 s for the sampling they would wrap, a 9x increase in the
   CPU the sampling phase burns. Not worth it for an 11 % memory saving.

4. **Per-group *fitting* is serial on purpose.** Measured; see "Where the time
   actually goes". Revisit if a family arrives whose per-group fit is iterative rather
   than a single pass over sufficient statistics — a bridged MAP+Laplace fit would be.

## Reproducing

```bash
make release                              # then, from the repo root:
python3 validation/bench.py               # scaling table
python3 validation/bench.py --threads     # both thread axes and the determinism digests
cargo run --release --example scale_profile -- 5000 104 1000   # crate-side phases
```
