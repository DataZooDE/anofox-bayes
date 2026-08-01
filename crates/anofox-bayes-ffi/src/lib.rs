//! C FFI boundary for anofox-bayes.
//!
//! This crate is deliberately thin: it owns memory-safety obligations and nothing
//! else. Every function here is a translation of a `anofox_bayes_core` call into
//! pointers and status codes, with no arithmetic of its own — anything worth testing
//! lives in the core crate, where it can be tested without `unsafe`.
//!
//! Conventions, all of which the C++ side relies on:
//!
//! * Strings returned by value are `'static` and NUL-terminated; the caller must not
//!   free them.
//! * Fallible calls return `bool` and write a code into an out-parameter; `true`
//!   means the out-parameters are populated.

use anofox_bayes_core::{diagnostics, DRAWS_SCHEMA_VERSION};
use std::os::raw::c_char;

/// Version of the extension's Rust core, NUL-terminated and statically allocated.
///
/// Doubles as the smoke test for the entire build chain: if this call returns the
/// right string through SQL, then corrosion built the crate, the static archive
/// linked into the loadable extension, and the C++ layer found the symbol.
#[no_mangle]
pub extern "C" fn anofox_bayes_ffi_version() -> *const c_char {
    // Built from a compile-time constant, so the NUL byte is guaranteed present and
    // the pointer outlives every caller.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Version of the long-format draws schema (`docs/DRAWS_CONTRACT.md`).
///
/// Separate from the extension version on purpose: draw tables persisted by a
/// customer must stay readable across extension upgrades, so this number only moves
/// when the contract itself breaks.
#[no_mangle]
pub extern "C" fn anofox_bayes_ffi_draws_schema_version() -> i32 {
    DRAWS_SCHEMA_VERSION
}

/// Which diagnostic [`anofox_bayes_ffi_diagnostic`] should compute.
///
/// A single entry point with a selector rather than three near-identical `unsafe`
/// functions: the pointer validation is the only part with a memory-safety
/// obligation, and writing it once means it can only be got wrong once.
pub const DIAGNOSTIC_RHAT: i32 = 0;
pub const DIAGNOSTIC_ESS_BULK: i32 = 1;
pub const DIAGNOSTIC_ESS_TAIL: i32 = 2;

/// Compute one convergence diagnostic from unordered `(value, chain, draw)` triples.
///
/// `out_defined` distinguishes "the statistic is 1.0" from "the statistic does not
/// exist here" — a single chain has no R̂, and the C++ layer turns an undefined result
/// into SQL `NULL` rather than into a number an agent might gate on.
///
/// # Safety
/// `values`, `chains` and `draws` must each point to at least `n` readable elements,
/// and `out_value` / `out_defined` must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_diagnostic(
    kind: i32,
    values: *const f64,
    chains: *const i32,
    draws: *const i32,
    n: usize,
    out_value: *mut f64,
    out_defined: *mut bool,
) -> bool {
    if out_value.is_null() || out_defined.is_null() {
        return false;
    }
    *out_value = f64::NAN;
    *out_defined = false;

    if n == 0 {
        // Not an error: an empty group simply has no diagnostic.
        return true;
    }
    if values.is_null() || chains.is_null() || draws.is_null() {
        return false;
    }

    let values = std::slice::from_raw_parts(values, n);
    let chains = std::slice::from_raw_parts(chains, n);
    let draws = std::slice::from_raw_parts(draws, n);

    let ordered = diagnostics::chains_from_rows(values, chains, draws);
    if ordered.is_empty() {
        return true;
    }

    match kind {
        DIAGNOSTIC_RHAT => {
            if let Some(r) = diagnostics::rhat(&ordered) {
                *out_value = r;
                *out_defined = true;
            }
        }
        DIAGNOSTIC_ESS_BULK | DIAGNOSTIC_ESS_TAIL => {
            let ess = if kind == DIAGNOSTIC_ESS_BULK {
                diagnostics::ess_bulk(&ordered)
            } else {
                diagnostics::ess_tail(&ordered)
            };
            // Zero is this estimator's "not assessable" signal, not a measurement.
            if ess > 0.0 {
                *out_value = ess;
                *out_defined = true;
            }
        }
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use anofox_bayes_core::VERSION;
    use std::ffi::CStr;

    #[test]
    fn the_version_pointer_is_a_valid_nul_terminated_string() {
        let ptr = anofox_bayes_ffi_version();
        assert!(!ptr.is_null());
        // Safe: the pointer comes from a `concat!` literal with an explicit NUL.
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(s, VERSION);
    }

    #[test]
    fn the_ffi_reports_the_cores_draws_schema_version() {
        assert_eq!(
            anofox_bayes_ffi_draws_schema_version(),
            DRAWS_SCHEMA_VERSION
        );
    }
}
