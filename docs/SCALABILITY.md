# Runtime profile and scalability

> Practical usage is in the [User Guide](GUIDE.md); this page is the measured
> runtime profile and its limits.

Measured, not estimated. Every number here comes from `validation/bench.py` on one
machine (8 threads, Linux x86_64, `conjugate_anomaly` with a Normal likelihood and one
group column); reproduce with the command at the bottom.

**Read this before pointing a fit at production-scale data.** The headline is that
v0.1 is single-threaded and holds the whole posterior in memory. That is fine for
every workload the shipped families are meant for, and it is not fine for everything.

## What was measured

| groups | rows | draws | wall | peak RSS |
|---:|---:|---:|---:|---:|
| 100 | 10 400 | 1 000 | 0.17 s | 48 MB |
| 1 000 | 104 000 | 1 000 | 0.91 s | 183 MB |
| 5 000 | 520 000 | 1 000 | 4.4 s | 824 MB |
| 5 000 | 520 000 | 4 000 | 16.1 s | 2 952 MB |
| 20 000 | 2 080 000 | 1 000 | 17.6 s | 3 218 MB |

The 5 000-group row is BRD BR-1's acceptance case (5 000 SKUs × 104 weeks). It
completes in seconds.

## Threading: the fit is single-threaded, deliberately for now

| threads | wall (2 000 groups) |
|---:|---:|
| 1 | 1.76 s |
| 2 | 1.73 s |
| 4 | 1.73 s |
| 8 | 1.73 s |
| 16 | 1.74 s |

Flat, because `BayesFitGlobalState::MaxThreads()` returns 1 and the operator buffers
its whole input before fitting. A posterior is a function of *every* observation, so
the fit genuinely cannot start before the last row arrives; what is *not* yet
exploited is that `conjugate_anomaly` fits each group **independently**, which is
embarrassingly parallel. HLD §6 says so, and it is not implemented. See "Known gaps".

**Determinism holds across thread counts.** `SET threads=1` and `SET threads=8`
produce a byte-identical draws table (verified by MD5 of the full result). That is a
property worth keeping: the accumulation is order-independent by construction, because
row order never reaches the mathematics.

## Memory model — where the bytes go

Three copies of the data exist at peak, which is the honest reason for the numbers
above:

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
which is the larger share.

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
- **Batch very wide group sets.** 20 000 groups in one call costs ~3 GB. Fitting in
  chunks of a few thousand and unioning the draws tables costs the same total time
  with a fraction of the peak, and each chunk gets its own `model_id`.
- **Filter before fitting.** Only the columns named in the config are read, but every
  row of the subquery is buffered. `(SELECT lane, cost FROM invoices WHERE year = 2026)`
  is materially cheaper than passing the whole table.
- **The output table is usually bigger than the fit.** Persist with
  `CREATE TABLE ... AS SELECT`, then query the draws; don't re-fit to ask a second
  question. That is the entire point of posterior-as-a-table.

## Known gaps

Recorded rather than hidden. None of these are hit by the workloads the shipped
families target, and all three are worth doing before a customer pushes past them.

1. **No group parallelism.** `conjugate_anomaly` fits each group independently and
   could use rayon across groups; the linear-algebra path in `pooled_gaussian` cannot,
   because a pooled fit is one joint system. HLD §6 anticipated this.
2. **The input relation is fully buffered.** A relation larger than RAM will not
   stream. Both shipped families need only per-group sufficient statistics
   (`n`, `Σy`, `Σy²`), so `conjugate_anomaly` could accumulate in constant memory
   rather than retaining rows. That is the single biggest available win.
3. **The posterior is materialised whole before the first row is emitted.** Draws
   could be generated lazily per output chunk, since each is an independent sample.

## Reproducing

```bash
make release                       # then, from the repo root:
python3 validation/bench.py        # scaling table
python3 validation/bench.py --threads   # thread scaling and the determinism digest
```
