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

use anofox_bayes_core::{DRAWS_SCHEMA_VERSION, VERSION};
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

#[cfg(test)]
mod tests {
    use super::*;
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
