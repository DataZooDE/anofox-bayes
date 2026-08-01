# THIRD PARTY NOTICES AND LICENSES

The Anofox Bayes Extension for DuckDB incorporates material from the projects
listed below. The original copyright notices and licenses under which DataZoo GmbH
received such material are set forth below.

The extension itself is licensed under the Business Source License 1.1; see
[LICENSE](LICENSE). Nothing here modifies that license — this file records the
terms under which the third-party components are used.

---

## DuckDB

   <https://github.com/duckdb/duckdb>

   The analytical database this extension is built for and linked against, together
   with `duckdb/extension-ci-tools` (build and distribution tooling, included as a
   submodule).

   The MIT License (MIT)

   Copyright 2018-2025 Stichting DuckDB Foundation

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

---

## faer

   <https://github.com/sarah-quinones/faer-rs>

   Dense linear algebra for Rust.

   The MIT License (MIT)

   Copyright (c) 2022 sarah

   Components used:
   - Cholesky factorisation and triangular solves for the pooled Gaussian
     posterior precision (`crates/anofox-bayes-core/src/linalg.rs`)
   - Matrix storage for the design matrix and posterior covariance
     (`crates/anofox-bayes-core/src/catalog/f3_pooled_gaussian.rs`)

   Default features — which pull in `rayon` — are disabled: DuckDB owns
   parallelism, and a nested thread pool inside a table function is a liability.

   faer transitively includes `gemm`, `nano-gemm`, `pulp`, `dyn-stack`, `reborrow`
   and related crates by the same author and under the same MIT terms.

---

## statrs

   <https://github.com/statrs-dev/statrs>

   Statistical computation library for Rust.

   The MIT License (MIT)

   Copyright (c) 2016 Michael Ma

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   Components used:
   - Log-gamma and related special functions
   - Normal quantile function, used for the rank-normalisation step of split-R̂
     and the effective-sample-size diagnostics
     (`crates/anofox-bayes-core/src/diagnostics/`)

---

## nalgebra

   <https://github.com/dimforge/nalgebra>

   Linear algebra library for Rust, pulled in transitively by `statrs`.

   Licensed under the Apache License, Version 2.0.

   Copyright (c) 2013 Sébastien Crozet and contributors.

   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

---

## rand, rand_chacha, rand_distr

   <https://github.com/rust-random/rand>

   Random number generation for Rust.

   Dual-licensed under Apache 2.0 and MIT licenses.

   Copyright (c) 2018 Developers of the Rand project;
   Copyright (c) 2014 The Rust Project Developers.

   Components used:
   - `rand_chacha`'s ChaCha20 counter-based stream cipher as the seeded RNG. This
     is the reproducibility guarantee of the extension: a given seed yields
     byte-identical draws on every platform, which the SQL test suite and the
     `model_id` contract both depend on (`crates/anofox-bayes-core/src/rng.rs`)
   - `rand_distr`'s Gamma and standard-Normal samplers, which the exact conjugate
     engine composes into the Normal-Inverse-Gamma and Gamma-Poisson posteriors
     (`crates/anofox-bayes-core/src/rng.rs`)

---

## BLAKE3

   <https://github.com/BLAKE3-team/BLAKE3>

   Cryptographic hash function.

   Licensed under CC0 1.0, Apache License 2.0, or Apache License 2.0 with the
   LLVM exception, at your option.

   Copyright (c) 2019 Jack O'Connor and Samuel Neves.

   Components used:
   - Deterministic `model_id` derivation and input-relation data fingerprints
     (`crates/anofox-bayes-core/src/fit.rs`, `data.rs`). Fields are
     length-prefixed before hashing so that distinct inputs cannot collide.

---

## serde_json

   <https://github.com/serde-rs/json>

   JSON serialization and deserialization for Rust.

   Dual-licensed under Apache 2.0 and MIT licenses.

   Copyright (c) 2014 Erick Tryzelaar, David Tolnay and contributors.

   Components used:
   - Parsing and canonicalising the `config` argument as it crosses the SQL/FFI
     boundary (`crates/anofox-bayes-core/src/config.rs`). The key-sorted `BTreeMap`
     representation is what makes a canonical config string — and therefore a
     stable `model_id` — possible.

   Transitively includes `serde`, `serde_derive` and `itoa` under the same terms.

---

## thiserror

   <https://github.com/dtolnay/thiserror>

   Derive macro for the standard library's `std::error::Error` trait.

   Dual-licensed under Apache 2.0 and MIT licenses.

   Copyright (c) David Tolnay

---

## libc

   <https://github.com/rust-lang/libc>

   Raw FFI bindings to platform libraries.

   Dual-licensed under Apache 2.0 and MIT licenses.

   Copyright (c) The Rust Project Developers

---

## Corrosion

   <https://github.com/corrosion-rs/corrosion>

   CMake integration for Rust crates, used to build and link the
   `anofox_bayes_ffi` static archive into the extension.

   The MIT License (MIT)

   Copyright (c) 2018 Andrew Gaspar

---

## posthog-telemetry

   <https://github.com/DataZooDE/posthog-telemetry>

   Shared DataZoo telemetry library, included as a submodule. Copyright (c)
   DataZoo GmbH. See the submodule's own LICENSE for terms, and
   [TELEMETRY.md](TELEMETRY.md) for what it sends and how to disable it.

---

## Algorithm references

The following are **not** dependencies; no code from them is included. They are
the published references the implementations follow, recorded so that the
provenance of each statistic is auditable.

### Rank-normalised split-R̂ and bulk / tail ESS

   Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
   *Rank-normalization, folding, and localization: An improved R̂ for assessing
   convergence of MCMC.* Bayesian Analysis, 16(2), 667–718.

   Implemented in `crates/anofox-bayes-core/src/diagnostics/`, cross-checked
   against the ArviZ reference implementation
   (<https://github.com/arviz-devs/arviz>, Apache License 2.0).

### Sample-statistics naming

   The `__lp__`, `__divergent__`, `__energy__` and `__step_size__` reserved
   parameter names follow ArviZ's `sample_stats` convention, so that draws tables
   exported from this extension are recognisable to existing Bayesian tooling.

### Conjugate posteriors

   Normal-Inverse-Gamma and Gamma-Poisson updates as given in Gelman et al.,
   *Bayesian Data Analysis* (3rd ed.), and Murphy, K. P. (2007), *Conjugate
   Bayesian analysis of the Gaussian distribution.* The reference priors used as
   defaults are the standard Jeffreys / uniform-on-log-variance choices.

### PyMC / Stan

   <https://github.com/pymc-devs/pymc> · <https://github.com/stan-dev/stan>

   Used as external reference implementations for golden-run parity testing only.
   No PyMC or Stan code is included in or linked by this extension. The planned
   NUTS engine (0.2) will depend on `nuts-rs` (pymc-devs, MIT/Apache-2.0); that
   dependency is not present today and this file will be updated when it lands.
