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

/// The one place the release version is written. A macro rather than a `const`
/// because the FFI needs it as a *literal* to build a NUL-terminated string at
/// compile time (`concat!` takes literals, not constants), and one source of
/// truth beats two literals that can drift.
macro_rules! release_version {
    () => {
        "2026.08.10"
    };
}

pub mod bridge;
pub mod catalog;
pub mod config;
pub mod data;
pub mod diagnostics;
pub mod draws;
pub mod engines;
pub mod errors;
pub mod fit;
pub mod keyed_rng;
pub mod linalg;
pub mod parallel;
pub mod rng;
pub mod sbc;
pub mod types;

pub use errors::{BayesError, BayesResult};
pub use types::*;

/// Release version, as reported through the SQL surface. CalVer, `YYYY.MM.DD`,
/// matching the git tag (`v2026.08.10`) and the `<ext_version>` segment of the
/// published S3 path.
///
/// NOT `env!("CARGO_PKG_VERSION")`: cargo parses the manifest version as semver
/// and rejects `2026.08.10` outright — *invalid leading zero in minor version
/// number*. Only the zero-stripped `2026.8.10` would be accepted, which is a
/// different string from the tag and so defeats the point. The manifest keeps a
/// semver number for crate metadata; this constant is the release identity.
///
/// Bump this in the same commit that moves the CHANGELOG, before tagging. See
/// `docs/RELEASING.md`.
pub const VERSION: &str = release_version!();

/// NUL-terminated form of [`VERSION`] for the C FFI, which must hand out a
/// pointer that is valid for the life of the process. Built from the same macro
/// so the two cannot drift.
pub const VERSION_C: &str = concat!(release_version!(), "\0");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reported_version_is_calver() {
        // Shape, not a literal: pinning the exact date here would mean every
        // release edits this test, and a test that must change with the thing it
        // checks stops being evidence. `scripts/check_version.sh` is what proves
        // the number agrees with the C++ fallback, the banner and the git tag.
        let parts: Vec<&str> = VERSION.split('.').collect();
        assert_eq!(parts.len(), 3, "expected YYYY.MM.DD, got {VERSION}");
        assert_eq!(parts[0].len(), 4, "expected a 4-digit year in {VERSION}");
        assert_eq!(
            parts[1].len(),
            2,
            "expected a zero-padded month in {VERSION}"
        );
        assert_eq!(parts[2].len(), 2, "expected a zero-padded day in {VERSION}");
        assert!(
            parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())),
            "expected digits only, got {VERSION}"
        );
    }

    #[test]
    fn the_c_string_form_matches_and_is_nul_terminated() {
        // The FFI hands out VERSION_C's pointer; if the two ever drifted, callers
        // would read a different version than the SQL surface reports.
        assert_eq!(VERSION_C, format!("{VERSION}\0"));
        assert!(VERSION_C.ends_with('\0'));
    }
}
