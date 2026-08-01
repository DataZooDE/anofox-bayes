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

mod fit;
pub use fit::*;

/// Run `body`, converting a panic into `fallback` instead of letting it escape.
///
/// **Every `extern "C"` function in this crate must go through this.** A Rust panic
/// that unwinds across an FFI boundary is undefined behaviour — not "probably fine",
/// not "aborts cleanly", but UB — and the boundary here is a C++ DuckDB process
/// belonging to a customer. The mathematics is panic-free by intent, but "by intent"
/// is not a guarantee: a slice index or an arithmetic overflow anywhere in the call
/// graph would do it, and the call graph is the entire core crate.
///
/// The panic message is printed to stderr before being swallowed, so a bug still
/// leaves evidence rather than silently becoming a null return.
///
/// `AssertUnwindSafe` is warranted because every one of these functions either owns
/// its state exclusively for the call or works through raw pointers whose invariants
/// the caller already guarantees; there is no shared Rust-side state that a partial
/// unwind could leave inconsistent.
pub(crate) fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            eprintln!(
                "anofox_bayes: a panic was caught at the FFI boundary and converted \
                 into a failure. This is a bug -- please report it."
            );
            fallback
        }
    }
}

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

    let computed = guard(None, || {
        let ordered = diagnostics::chains_from_rows(values, chains, draws);
        if ordered.is_empty() {
            return Some(None);
        }
        match kind {
            DIAGNOSTIC_RHAT => Some(diagnostics::rhat(&ordered)),
            DIAGNOSTIC_ESS_BULK => Some(defined_if_positive(diagnostics::ess_bulk(&ordered))),
            DIAGNOSTIC_ESS_TAIL => Some(defined_if_positive(diagnostics::ess_tail(&ordered))),
            _ => None,
        }
    });

    match computed {
        // Unknown kind, or a panic: a misuse the caller must hear about.
        None => false,
        Some(None) => true,
        Some(Some(value)) => {
            *out_value = value;
            *out_defined = true;
            true
        }
    }
}

/// Zero is the ESS estimator's "not assessable" signal, not a measurement.
fn defined_if_positive(ess: f64) -> Option<f64> {
    (ess > 0.0).then_some(ess)
}

// --- Keyed randomness for the predictive step -------------------------------
//
// Pure functions of their arguments, so unlike the fit entry points there is no
// handle to own and nothing to free. They still go through `guard`: the core call
// reaches `statrs`'s inverse CDF, and a panic crossing this boundary would be UB
// regardless of how unlikely it is.

/// A draw from `Uniform(0, 1)`, open at both ends.
///
/// `key`/`key_len` is arbitrary caller-supplied bytes and may be empty; `key` may be
/// null only when `key_len` is zero.
///
/// # Safety
/// `key` must point to at least `key_len` readable bytes, or be null with
/// `key_len == 0`.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_uniform(
    seed: i64,
    key: *const u8,
    key_len: usize,
    draw: i64,
) -> f64 {
    guard(f64::NAN, || {
        let bytes = key_slice(key, key_len);
        anofox_bayes_core::keyed_rng::uniform(seed, bytes, draw)
    })
}

/// A draw from `N(0, 1)`. See [`anofox_bayes_ffi_uniform`] for the key contract.
///
/// # Safety
/// Same as [`anofox_bayes_ffi_uniform`].
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_std_normal(
    seed: i64,
    key: *const u8,
    key_len: usize,
    draw: i64,
) -> f64 {
    guard(f64::NAN, || {
        let bytes = key_slice(key, key_len);
        anofox_bayes_core::keyed_rng::std_normal(seed, bytes, draw)
    })
}

/// Borrow caller-owned key bytes.
///
/// A null pointer with a zero length is the natural spelling of an empty DuckDB
/// string and must not reach `slice::from_raw_parts`, which requires a non-null
/// aligned pointer even for an empty slice.
unsafe fn key_slice<'a>(key: *const u8, key_len: usize) -> &'a [u8] {
    if key.is_null() || key_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(key, key_len)
    }
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

    /// The guard is the only thing standing between a bug in the core and undefined
    /// behaviour in a customer's DuckDB process, so it is worth proving it fires
    /// rather than assuming it does. Panic output is silenced for the duration so a
    /// passing test run does not print an alarming backtrace.
    #[test]
    fn the_ffi_guard_converts_a_panic_into_its_fallback() {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = guard(-1i32, || panic!("simulated bug in the core"));
        std::panic::set_hook(previous);

        assert_eq!(caught, -1, "a panic must become the fallback, not unwind");
        // ...and the happy path is untouched.
        assert_eq!(guard(-1i32, || 7), 7);
    }

    /// The C++ layer passes `string_t::GetData()` straight through, and DuckDB is
    /// entitled to hand back a null pointer for an empty string. `from_raw_parts`
    /// requires non-null even at length zero, so this path is a genuine UB obligation
    /// rather than a defensive nicety.
    #[test]
    fn an_empty_key_may_arrive_as_a_null_pointer() {
        let from_null = unsafe { anofox_bayes_ffi_std_normal(1, std::ptr::null(), 0, 0) };
        let from_empty = unsafe { anofox_bayes_ffi_std_normal(1, b"".as_ptr(), 0, 0) };
        assert!(from_null.is_finite());
        assert_eq!(from_null, from_empty);
        assert_eq!(
            from_null,
            anofox_bayes_core::keyed_rng::std_normal(1, b"", 0)
        );
    }

    /// A non-empty key must actually be read, not silently treated as empty — the
    /// failure mode where every group shares one random stream.
    #[test]
    fn the_key_bytes_reach_the_core_unchanged() {
        let key = b"lane-7";
        let through_ffi = unsafe { anofox_bayes_ffi_uniform(9, key.as_ptr(), key.len(), 4) };
        assert_eq!(
            through_ffi,
            anofox_bayes_core::keyed_rng::uniform(9, key, 4)
        );
        assert_ne!(
            through_ffi,
            unsafe { anofox_bayes_ffi_uniform(9, std::ptr::null(), 0, 4) },
            "a populated key must not behave like an empty one"
        );
    }

    #[test]
    fn the_ffi_reports_the_cores_draws_schema_version() {
        assert_eq!(
            anofox_bayes_ffi_draws_schema_version(),
            DRAWS_SCHEMA_VERSION
        );
    }
}
