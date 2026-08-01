# Agent instructions — anofox-bayes

Read this before changing anything. The standards below are not style preferences;
each one exists because its absence produced a wrong number that nothing downstream
would have caught.

## Build and test

```bash
export VCPKG_TOOLCHAIN_PATH=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
export GEN=ninja

cargo test --workspace         # fast loop -- run this constantly
make lint                      # clippy with -D warnings, exactly as CI runs it
make release                   # builds DuckDB the first time (30-60 min)
make test                      # sqllogictest suite
make test_sbc                  # calibration suites (slow, ~30 s, release mode)
cargo fmt --all && clang-format -i src/*.cpp src/*/*.cpp src/include/*.hpp
```

`make format` is **extension-ci-tools'** C++ target. The Rust equivalents here are
`make format_rust` / `make format_rust_fix` — overriding `format` would silently drop
the clang-format pass.

**Verify the duckdb submodule pin before and after every build.**
`git -C duckdb describe --tags` must read `v1.5.5`. An `M duckdb` in `git status`
means stop and re-pin; a silent downgrade costs two full rebuilds.

## Where things live

| | |
|---|---|
| `crates/anofox-bayes-core` | All mathematics. Knows nothing about DuckDB or FFI. |
| `crates/anofox-bayes-ffi` | `unsafe` and memory ownership only. No arithmetic. |
| `src/` | C++ SQL surface. No mathematics. |
| `test/sql/` | sqllogictest. One file per realistic scenario. |
| `validation/` | PyMC golden-run parity (uv + pytest). |

If you find yourself writing arithmetic in `src/` or in the FFI crate, it belongs in
the core, where it can be tested without `unsafe` and without a database.

## Non-negotiables

**Refusal over plausible numbers.** A model that cannot answer must say so. A group
with one observation, a perfectly-fitted regression, a rank-deficient design: each
returns a status that refuses, or a typed error — never a converged status beside a
number an agent would act on. Unestimable parameters draw `NaN`, which the SQL layer
renders as `NULL`; a number there would be indistinguishable from an estimate.

**Never fabricate a diagnostic.** R̂ over one chain is `None`, not `1.0`. An exact
sampler emits no `__divergent__` row rather than `0.0`. A statistic that was never
computed must not read as a passing one.

**Priors are scale-free by default.** Every family defaults to its reference prior.
Any concrete "weakly informative" default encodes an assumption about whether the
customer measures costs in cents or in millions.

**Determinism is a test, not an aspiration.** Same seed, same bytes. Chain seeds are
derived by hashing `(seed, chain)`, never by XOR — `seed ^ chain` makes `(4,1)` and
`(5,0)` the same stream, so two "independent" chains would be one chain and R̂ would
report a perfect 1.0 for exactly the wrong reason.

**Config errors name their slot.** `invalid config at 'prior.alpha0': must be >= -1`
is repairable by an agent; `invalid configuration` is not. Unknown slots are
rejected with a did-you-mean, because a misspelled `seeed` that silently takes the
default produces a fit that is correct, reproducible, and not the one that was asked
for.

## Adding a family

1. Implement `ModelFamily` in `crates/anofox-bayes-core/src/catalog/`, register it in
   `catalog::all()`.
2. Pin the posterior against its **closed form** in a unit test. "Looks about right"
   is not a test.
3. Implement `LogPosterior` if the family is to be served by a gradient engine, and
   check the gradient against finite differences **away from the mode** as well as at
   it — at the mode the gradient is zero, where a sign error is invisible.
4. Add an SBC suite in `sbc.rs::families`. It must run under a *proper* prior, since
   SBC draws the truth from the prior.
5. Add a `test/sql/` file exercising a realistic scenario end to end, not a synthetic
   one. The existing files model a freight audit and a difference-in-differences
   panel; write the query a customer would actually run.
6. Add a PyMC parity test in `validation/`. **What that means depends on the
   engine**, and getting it wrong makes the suite vacuous rather than red:

   | engine | reference | why |
   |---|---|---|
   | `exact` | NUTS on the same prior | the extension side is i.i.d., so only the reference carries autocorrelation |
   | `nuts` | NUTS, **with ESS measured on both sides** | the floor is `sd/sqrt(ESS)`, never `sd/sqrt(draws)`; assert the measured ESS so the derivation cannot rot |
   | `laplace` | **two** references: an independently computed mode + observed information, *and* NUTS | one comparison cannot separate "the likelihood is wrong" from "a Gaussian at the mode is not the posterior" |

   Every tolerance is derived in `validation/TOLERANCES.md`, in units of the
   reference posterior's own sd. A tolerance without a derivation is one that
   gets loosened until it is green.

   Where the two implementations genuinely cannot agree — a prior PyMC cannot
   express, an approximation that is the shipped answer — **document the
   discrepancy and pin it with a direct measurement**, the way F3's residual
   scale and F2's are. A documented, understood discrepancy is a result; a
   silently loosened tolerance is not.

**SBC and parity are not redundant, and neither subsumes the other.** SBC
certifies calibration *under the model's own prior*, so it only ever runs where
that prior is proper — which is exactly where F3's degrees-of-freedom bug was
not present. Parity checks the posterior against an independent implementation,
so it sees improper defaults, log-Jacobian terms and likelihood constants that
SBC structurally cannot. Both have now caught things the other could not.

## Adding an engine

Implement `Engine`. Return `false` from `supports()` rather than approximating
something you cannot serve exactly — an agent that asked for an exact posterior and
quietly received an approximation reports unearned confidence. Where a family is
conjugate, add a test that the new engine agrees with `ExactEngine`: two independent
derivations of one distribution is the strongest check available.

## Changing the draws contract

`docs/DRAWS_CONTRACT.md` is a versioned wire contract that customers persist. Adding
a reserved metadata row is compatible; changing what an existing one means is not, and
requires bumping `DRAWS_SCHEMA_VERSION`. `FitStatus` and `ErrorCode` numbering is
append-only — renumbering would turn a refusal into an approval at a customer site.

## Issue tracking

This project uses **bd** (beads). Run `bd onboard` to get started.

```bash
bd ready                             # available work
bd show <id>                         # details
bd update <id> --status in_progress  # claim
bd close <id>                        # complete
bd sync                              # sync with git
```

## Landing the plane

Work is not complete until `git push` succeeds.

1. File issues for anything left over.
2. Run the quality gates CI runs: `cargo test --workspace`, `make lint`, `make test`, `cargo fmt --all -- --check`.
3. Update issue status.
4. `git pull --rebase && bd sync && git push`, then confirm `git status` says up to date.
5. Hand off with context for the next session.
