//! anofox-bayes-core: Bayesian inference for a closed catalog of enterprise decision
//! models.
//!
//! This crate holds all of the mathematics and none of the plumbing. It knows nothing
//! about DuckDB, about FFI, or about SQL; it takes columns of `f64` plus a validated
//! configuration and produces posterior draws in the long format described in
//! `docs/DRAWS_CONTRACT.md`.
//!
//! The shape of the crate follows the one constraint that matters for a product whose
//! liability is numerical correctness: **families never know about engines, and engines
//! never know about families.**
//!
//! ```text
//!   FamilyConfig ──validate──▶ compiled model (dyn CompiledModel)
//!                                       │
//!                        ┌──────────────┼──────────────┐
//!                        ▼              ▼              ▼
//!                  ExactEngine    LaplaceEngine    NutsEngine (0.2)
//!                        └──────────────┼──────────────┘
//!                                       ▼
//!                                   Posterior ──▶ DrawSink ──▶ long-format rows
//! ```
//!
//! A family exposes a log posterior and its analytic gradient; if it happens to be
//! conjugate it additionally implements [`engines::ExactPosterior`] and gets closed-form
//! sampling for free. Adding an engine touches no family; adding a family touches no
//! engine. That is the whole design.

pub mod catalog;
pub mod config;
pub mod data;
pub mod diagnostics;
pub mod draws;
pub mod engines;
pub mod errors;
pub mod fit;
pub mod rng;
pub mod types;

pub use errors::{BayesError, BayesResult};
pub use types::*;

/// Version of this crate, as reported through the SQL surface.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_version_is_reported() {
        assert_eq!(VERSION, "0.1.0");
    }
}
