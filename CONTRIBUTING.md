# Contributing to Anofox Bayes

Thank you for your interest in contributing. This document covers the development
setup, the test loop, and the standards a change has to meet before it is merged.

Read this first, because it is the thing that most surprises people: **`anofox-bayes`
is a closed catalog, not a probabilistic programming language.** New model
families are welcome; a mechanism for users to express their own models is not. A
new family needs a real, named use case behind it. That restriction is what makes
analytic gradients, per-family calibration and a bounded correctness liability
possible at all, and it is defended in review. See [docs/BRD.md](docs/BRD.md) §4
for the full non-goals list.

## Getting started

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-bayes.git
cd anofox-bayes
```

### Prerequisites

- Rust toolchain (stable, via rustup), with `rustfmt` and `clippy`
- CMake 3.15+ and Make or Ninja
- A C++17 compiler (GCC 9+ or Clang 10+)
- OpenSSL development libraries (for the telemetry transport)
- Python 3.12 and [uv](https://github.com/astral-sh/uv), only if you want to run
  the PyMC parity suite

**Ubuntu/Debian**

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libssl-dev
```

**Manjaro/Arch**

```bash
sudo pacman -S base-devel cmake ninja openssl
```

**Fedora/RHEL**

```bash
sudo dnf install gcc-c++ cmake ninja-build openssl-devel
```

**macOS**

```bash
brew install cmake ninja openssl
```

Windows is supported through MSVC/vcpkg and WSL. The `windows_amd64_mingw` and
`windows_amd64_rtools` targets are excluded from CI: rtools42's MinGW lacks the
`libbcrypt.a` that Rust's `getrandom` 0.3+ needs.

### Build

```bash
make release -j$(nproc)      # or: GEN=ninja make release
```

This builds the Rust workspace, links `anofox_bayes_ffi` into the extension, and
produces both a loadable extension and a `build/release/duckdb` shell with the
extension statically linked in.

## The repository, in one pass

```
crates/anofox-bayes-core/   the mathematics: catalog, engines, diagnostics, draws
  src/catalog/              one module per model family
  src/engines/              one module per inference engine
  src/diagnostics/          R-hat, ESS
crates/anofox-bayes-ffi/    the C ABI the extension calls across
src/                        the DuckDB surface: table fn, aggregates, scalars
test/sql/                   sqllogictest suites, written as agent scenarios
validation/                 PyMC golden-run parity (Python, uv + pytest)
docs/                       BRD, HLD, draws contract, API reference
```

The trait split in `crates/anofox-bayes-core/src/catalog/mod.rs` is load-bearing:
`ModelFamily` turns a validated config plus data into a `CompiledModel`, and knows
nothing about how the posterior will be explored. **Adding an engine touches no
family. Adding a family touches no engine.** A change that breaks that separation
will be sent back.

## The test loop

```bash
make test_rust          # cargo test --workspace -- the loop you run while coding
make release && make test   # the sqllogictest suites in test/sql/
make format_rust        # cargo fmt --all -- --check
make format_rust_fix    # cargo fmt --all
make test_sbc           # simulation-based calibration; slow, #[ignore]d by default
```

Note the naming: the Rust targets are `format_rust` / `format_rust_fix` rather
than `format` / `format-fix`, because `extension-ci-tools` already defines the
latter for the C++ side and overriding them silently drops the `clang-format`
pass. Run both.

CI runs `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features
-- -D warnings` and `cargo test --workspace` **first**, and gates the entire
cross-platform build matrix on them. A formatting slip therefore fails in two
minutes rather than twenty.

### PyMC parity suite

Golden-run parity against the reference implementation lives in `validation/`.
It is a separate Python project so that the extension itself never grows a Python
dependency.

```bash
cd validation
uv sync
uv run pytest            # skips loudly if build/release/... is missing
```

Build the extension first — the suite loads the locally built binary, not an
installed one.

## Standards

### Every claim in a doc must be runnable

`README.md` and `docs/API_REFERENCE.md` document only what exists. Roadmap items
are marked *planned* and collected in their own sections so nobody writes SQL
against them. If you add an example, run it against `./build/release/duckdb`
before you commit, and paste the output you actually got.

### Refusal over plausible numbers

This is the core design commitment of the project and the most common reason a
change is rejected.

- A statistic that could not be computed is `NULL`, never a reassuring default.
  R-hat over one chain returns `NULL`, not `1.0`, because an agent gating on
  `rhat <= 1.01` must not be told "converged" by something that was never
  measured.
- A group the model cannot fit gets `NULL` draws and an `insufficient_data`
  status, not a number derived mostly from the prior.
- A model-level status is worst-wins across groups. A fit covering 500 lanes of
  which three are unidentifiable is not 99.4 % trustworthy.
- Validation happens **before** any arithmetic. A typo'd column name, an unknown
  config slot or an inadmissible prior must fail at compile time with a precise,
  machine-readable error, not surface later as a strange posterior.

### Priors must be scale-free by default

Any concrete "weakly informative" default encodes an assumption about whether the
customer measures costs in cents or in millions, and will quietly dominate the
data for anyone whose units differ from yours. Defaults are reference priors. If
you think a family needs an informative default, argue it in the PR with the
units it assumes.

### Determinism is a contract, not a nicety

`model_id = BLAKE3(family, canonical_config, data_fingerprint, seed)`. Anything
that makes the same request produce different draws — an unseeded RNG, a
thread-count-dependent reduction order, an unstable config rendering — is a bug,
even if the numbers are statistically indistinguishable. Fields are
length-prefixed before hashing and the config is rendered key-sorted; keep it
that way.

## Adding a model family

1. **Bring the use case.** A family needs a named agent or customer question it
   serves, in the PR description. See [docs/BRD.md](docs/BRD.md) §6 for the
   catalog and the phase it belongs to.
2. **New module under `crates/anofox-bayes-core/src/catalog/`,** implementing
   `ModelFamily`. Declare a `SLOTS` const and call `cfg.reject_unknown(SLOTS)`
   first thing in `compile`.
3. **Write the module doc comment as the spec.** Every existing family opens with
   the likelihood, the prior, the closed-form posterior update in `text` fences,
   and the reasoning behind the default priors. This is the documentation an
   auditor reads; it is not optional.
4. **Fixed parameterisation decisions.** Non-centered hierarchies, log links,
   softplus for dispersion. Callers must not be able to select a bad
   parameterisation.
5. **Structural refusal via `Readiness`,** reached from the sufficient statistics
   before any sampling. Some inadequacies need no draws to detect.
6. **Tests, all of them:**
   - inline `#[cfg(test)] mod tests` covering the posterior against a hand-checked
     or closed-form reference, and against the degenerate inputs (constant column,
     single-observation group, rank-deficient design, perfectly-fitted response);
   - analytic gradients against finite differences, if the family has them;
   - a `test/sql/` suite written as the *agent scenario* it serves — the existing
     ones read as a freight audit and an intervention evaluation, not as unit
     tests, and that is deliberate;
   - an SBC case, and a PyMC parity case in `validation/`.
7. **Document it** in `docs/API_REFERENCE.md` §2 with every slot, its type,
   default and constraint, and move it from *planned* to *shipped* in the
   `README.md` catalog table.

## Adding an engine

Implement `Engine` in `crates/anofox-bayes-core/src/engines/`, register it in
`resolve`, and add the config value to `EngineKind::parse`. Do not touch any
family. An engine that cannot serve a model must say so through `supports`, so
the mismatch is a config error rather than a runtime surprise.

Engine choice must be invisible to caller SQL: the draws contract, the
diagnostics and the gate query are identical whichever engine ran. `__engine__`
records which one it was.

## Changing the draws contract

`docs/DRAWS_CONTRACT.md` is versioned and reported by
`anofox_bayes_draws_schema_version()`. Adding a new reserved metadata row is not
breaking; changing what an existing one means is. A breaking change needs a
version bump, an update to the contract document, and a look at every test that
asserts on `__schema_version__` — including the Python parity suite, which pins
it deliberately so the bump cannot pass unnoticed.

## Submitting changes

```bash
git checkout -b feature/your-feature-name
# ... make changes, run make format_rust_fix && make test_rust && make test
git commit -m "Add X: what and why"
git push origin feature/your-feature-name
```

Open a pull request against `main` with:

- **Title**: a clear, concise description of the change
- **Description**: what, why, and how — plus the use case, for a new family
- **Tests**: all suites passing; new tests for new behaviour
- **Docs**: `docs/API_REFERENCE.md` and `README.md` updated in the same PR, never
  a follow-up
- **Breaking changes**: marked explicitly, especially anything touching the draws
  contract

## Reporting issues

**Bugs** — include the DuckDB version, extension version
(`SELECT anofox_bayes_version()`), operating system, a minimal reproducible
example including the `config` struct, and expected vs actual behaviour. If the
fit produced a status you did not expect, include the metadata rows:

```sql
SELECT param, value FROM draws WHERE draw < 0 ORDER BY param;
```

**Feature requests** — include the use case, the decision it feeds, proposed
SQL, and the alternatives you considered. For a new model family, say which agent
or business question it serves.

## Code of conduct

Be respectful and inclusive. Welcome newcomers. Focus on constructive feedback.
Assume good intentions.

## License

By contributing, you agree that your contributions will be licensed under the
project's Business Source License 1.1. See [LICENSE](LICENSE).

## Questions?

- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-bayes/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DataZooDE/anofox-bayes/discussions)
- **Email**: info@data-zoo.de
