//! FFI for fitting: build a relation, fit it, stream the draws back.
//!
//! Three opaque handles, because the alternative — passing a fully described relation
//! through one call — would need a variadic C signature that neither side could keep
//! type-safe.
//!
//! ```text
//!   data_new ─▶ add_numeric / add_key ─▶ fit ─▶ rows(offset, max) ─▶ free
//! ```
//!
//! **Validity masks are `u8`, not `bool`.** C++ `bool` has no guaranteed size or
//! representation, and Rust `bool` is UB for any bit pattern other than 0 or 1 — so a
//! `vector<char>` reinterpreted as `bool*` is undefined behaviour that happens to work
//! wherever `sizeof(bool) == 1`. Passing an explicit byte type and testing `!= 0`
//! makes the contract real rather than an ABI accident.
//!
//! **Ownership.** Every pointer the caller hands in is borrowed for the duration of
//! the call only; the builder copies. Every pointer handed back points into the
//! handle and is valid until that handle is freed. Strings are returned as
//! `(ptr, len)` rather than NUL-terminated, because Rust strings are not NUL
//! terminated and copying them just to add a byte would double the cost of the
//! largest output this extension produces.

use std::ffi::c_char;

use anofox_bayes_core::config::Config;
use anofox_bayes_core::data::{DataView, KeyColumn, NumericColumn};
use anofox_bayes_core::errors::ErrorCode;
use anofox_bayes_core::fit::{fit as core_fit, Fit};
use anofox_bayes_core::BayesError;

/// A borrowed string slice, as `(pointer, length)`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BayesStr {
    pub ptr: *const c_char,
    pub len: usize,
}

impl BayesStr {
    fn from(s: &str) -> Self {
        Self {
            ptr: s.as_ptr() as *const c_char,
            len: s.len(),
        }
    }

    fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// # Safety
    /// Must point to `len` bytes of valid UTF-8 that outlive the call.
    unsafe fn as_str(&self) -> Option<&str> {
        if self.ptr.is_null() {
            return None;
        }
        std::str::from_utf8(std::slice::from_raw_parts(self.ptr as *const u8, self.len)).ok()
    }
}

/// An error, written into a caller-provided buffer.
///
/// The message is truncated rather than allocated: an error crossing this boundary is
/// about to become a DuckDB exception string, and a fixed buffer removes any question
/// of who frees what on a path that is already the unhappy one.
#[repr(C)]
pub struct BayesFfiError {
    pub code: i32,
    pub message: [c_char; 512],
}

impl BayesFfiError {
    fn set(&mut self, err: &BayesError) {
        self.code = err.code() as i32;
        let text = err.to_string();
        let bytes = text.as_bytes();
        let n = bytes.len().min(self.message.len() - 1);
        for (i, b) in bytes[..n].iter().enumerate() {
            self.message[i] = *b as c_char;
        }
        self.message[n] = 0;
    }

    fn clear(&mut self) {
        self.code = ErrorCode::Success as i32;
        self.message[0] = 0;
    }
}

/// Owned columns for one input relation.
///
/// Owned rather than borrowed because the C++ layer assembles the relation across
/// many DuckDB chunks whose buffers are recycled between calls; borrowing them would
/// leave dangling pointers by the time the fit runs.
pub struct BayesData {
    n_rows: usize,
    numeric: Vec<(String, Vec<f64>, Vec<bool>)>,
    keys: Vec<(String, Vec<String>, Vec<bool>)>,
}

/// A completed fit plus the strings its rows point into.
pub struct BayesFit {
    fit: Fit,
    /// Pre-joined so [`anofox_bayes_ffi_fit_status`] can hand back a borrowed slice
    /// rather than allocate a string the caller would then have to free.
    reasons_joined: String,
}

/// Create an empty relation with `n_rows` rows.
#[no_mangle]
pub extern "C" fn anofox_bayes_ffi_data_new(n_rows: usize) -> *mut BayesData {
    Box::into_raw(Box::new(BayesData {
        n_rows,
        numeric: Vec::new(),
        keys: Vec::new(),
    }))
}

/// # Safety
/// `data` must come from [`anofox_bayes_ffi_data_new`] and not yet be freed.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_data_free(data: *mut BayesData) {
    if !data.is_null() {
        drop(Box::from_raw(data));
    }
}

/// Add a numeric column. `values` and `valid` must each have `n_rows` elements.
///
/// # Safety
/// All pointers must be valid for the length the relation was created with.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_data_add_numeric(
    data: *mut BayesData,
    name: BayesStr,
    values: *const f64,
    valid: *const u8,
    len: usize,
) -> bool {
    let Some(data) = data.as_mut() else {
        return false;
    };
    let Some(name) = name.as_str() else {
        return false;
    };
    if len != data.n_rows || (len > 0 && (values.is_null() || valid.is_null())) {
        return false;
    }
    let values = if len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(values, len).to_vec()
    };
    let valid = if len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(valid, len)
            .iter()
            .map(|b| *b != 0)
            .collect()
    };
    data.numeric.push((name.to_string(), values, valid));
    true
}

/// Add a key (grouping) column.
///
/// # Safety
/// `values` must point to `len` [`BayesStr`] entries, each valid UTF-8 for the
/// duration of this call.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_data_add_key(
    data: *mut BayesData,
    name: BayesStr,
    values: *const BayesStr,
    valid: *const u8,
    len: usize,
) -> bool {
    let Some(data) = data.as_mut() else {
        return false;
    };
    let Some(name) = name.as_str() else {
        return false;
    };
    if len != data.n_rows || (len > 0 && (values.is_null() || valid.is_null())) {
        return false;
    }
    let mut owned = Vec::with_capacity(len);
    if len > 0 {
        for s in std::slice::from_raw_parts(values, len) {
            owned.push(s.as_str().unwrap_or("").to_string());
        }
    }
    let valid = if len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(valid, len)
            .iter()
            .map(|b| *b != 0)
            .collect()
    };
    data.keys.push((name.to_string(), owned, valid));
    true
}

/// Fit a cataloged model. Returns null on failure, with `out_error` populated.
///
/// # Safety
/// `data` must be a live handle; `out_error` must be writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit(
    family: BayesStr,
    config_json: BayesStr,
    data: *const BayesData,
    out_error: *mut BayesFfiError,
) -> *mut BayesFit {
    let Some(err) = out_error.as_mut() else {
        return std::ptr::null_mut();
    };
    err.clear();

    let (Some(data), Some(family), Some(config_json)) =
        (data.as_ref(), family.as_str(), config_json.as_str())
    else {
        err.set(&BayesError::Internal("null argument to fit".into()));
        return std::ptr::null_mut();
    };

    let result = (|| {
        let cfg = Config::parse(config_json)?;

        let mut view = DataView::new(data.n_rows);
        for (name, values, valid) in &data.numeric {
            view.add_numeric(name.clone(), NumericColumn { values, valid })?;
        }
        // Key columns are stored as owned `String`s; the view needs `&str` slices,
        // which must outlive it -- hence materialising them here rather than inside
        // the loop, where they would be dropped at the end of each iteration.
        let key_refs: Vec<Vec<&str>> = data
            .keys
            .iter()
            .map(|(_, values, _)| values.iter().map(String::as_str).collect())
            .collect();
        for (i, (name, _, valid)) in data.keys.iter().enumerate() {
            view.add_key(
                name.clone(),
                KeyColumn {
                    values: &key_refs[i],
                    valid,
                },
            )?;
        }

        core_fit(family, &cfg, &view)
    })();

    match result {
        Ok(fit) => {
            let reasons_joined = fit.reasons.join("\n");
            Box::into_raw(Box::new(BayesFit {
                fit,
                reasons_joined,
            }))
        }
        Err(e) => {
            err.set(&e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `fit` must come from [`anofox_bayes_ffi_fit`] and not yet be freed.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_free(fit: *mut BayesFit) {
    if !fit.is_null() {
        drop(Box::from_raw(fit));
    }
}

/// Total rows the fit renders to in the long draws format.
///
/// # Safety
/// `fit` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_row_count(fit: *const BayesFit) -> usize {
    match fit.as_ref() {
        None => 0,
        Some(f) => f.fit.posterior.n_rows(),
    }
}

/// Copy up to `max` rows starting at `offset` into the caller's column buffers.
///
/// Returns the number of rows written. The returned string pointers borrow from the
/// fit handle and stay valid until it is freed.
///
/// # Safety
/// Every output pointer must be writable for `max` elements.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_rows(
    fit: *const BayesFit,
    offset: usize,
    max: usize,
    out_model_id: *mut BayesStr,
    out_group_id: *mut BayesStr,
    out_chain: *mut i32,
    out_draw: *mut i32,
    out_param: *mut BayesStr,
    out_value: *mut f64,
) -> usize {
    let Some(fit) = fit.as_ref() else {
        return 0;
    };
    if out_model_id.is_null()
        || out_group_id.is_null()
        || out_chain.is_null()
        || out_draw.is_null()
        || out_param.is_null()
        || out_value.is_null()
    {
        return 0;
    }

    let mut written = 0;
    while written < max {
        let Some(row) = fit.fit.posterior.row_at(offset + written) else {
            break;
        };
        *out_model_id.add(written) = BayesStr::from(row.model_id);
        *out_group_id.add(written) = BayesStr::from(row.group_id);
        *out_chain.add(written) = row.chain;
        *out_draw.add(written) = row.draw;
        *out_param.add(written) = BayesStr::from(row.param);
        *out_value.add(written) = row.value;
        written += 1;
    }
    written
}

/// The fit's status code, and its reasons joined by newlines.
///
/// Exposed separately from the draws so that a caller can gate without scanning the
/// table, even though the same status also travels inside it.
///
/// # Safety
/// `fit` must be live; `out_reasons` must be writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_status(
    fit: *const BayesFit,
    out_reasons: *mut BayesStr,
) -> i32 {
    let Some(f) = fit.as_ref() else {
        if let Some(r) = out_reasons.as_mut() {
            *r = BayesStr::empty();
        }
        return -1;
    };
    if let Some(r) = out_reasons.as_mut() {
        *r = BayesStr::from(&f.reasons_joined);
    }
    f.fit.posterior.meta.status as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_buf() -> BayesFfiError {
        BayesFfiError {
            code: 0,
            message: [0; 512],
        }
    }

    fn s(text: &str) -> BayesStr {
        BayesStr::from(text)
    }

    /// The full round trip a table function performs, exercised without DuckDB.
    #[test]
    fn a_relation_built_through_the_ffi_fits_and_streams_its_draws_back() {
        unsafe {
            let data = anofox_bayes_ffi_data_new(6);
            let values = [2.0, 2.1, 1.9, 2.05, 1.95, 2.02];
            let valid: [u8; 6] = [1; 6];
            assert!(anofox_bayes_ffi_data_add_numeric(
                data,
                s("cost"),
                values.as_ptr(),
                valid.as_ptr(),
                6
            ));

            let mut err = err_buf();
            let fit = anofox_bayes_ffi_fit(
                s("conjugate_anomaly"),
                s(r#"{"value": "cost", "draws": 1000, "seed": 4}"#),
                data,
                &mut err,
            );
            assert!(!fit.is_null(), "fit failed with code {}", err.code);
            assert_eq!(err.code, ErrorCode::Success as i32);

            let total = anofox_bayes_ffi_fit_row_count(fit);
            // 8 metadata rows + 1000 draws x 2 parameters.
            assert_eq!(total, 8 + 1000 * 2);

            // Read in chunks, as the C++ layer does.
            let cap = 512;
            let mut model_id = vec![BayesStr::empty(); cap];
            let mut group_id = vec![BayesStr::empty(); cap];
            let mut chain = vec![0i32; cap];
            let mut draw = vec![0i32; cap];
            let mut param = vec![BayesStr::empty(); cap];
            let mut value = vec![0.0f64; cap];

            let mut seen = 0usize;
            let mut mu_sum = 0.0;
            let mut mu_n = 0usize;
            loop {
                let n = anofox_bayes_ffi_fit_rows(
                    fit,
                    seen,
                    cap,
                    model_id.as_mut_ptr(),
                    group_id.as_mut_ptr(),
                    chain.as_mut_ptr(),
                    draw.as_mut_ptr(),
                    param.as_mut_ptr(),
                    value.as_mut_ptr(),
                );
                if n == 0 {
                    break;
                }
                for i in 0..n {
                    assert!(!model_id[i].ptr.is_null());
                    if param[i].as_str() == Some("mu") {
                        mu_sum += value[i];
                        mu_n += 1;
                    }
                }
                seen += n;
            }
            assert_eq!(seen, total);
            assert_eq!(mu_n, 1000);
            assert!((mu_sum / mu_n as f64 - 2.003).abs() < 0.05);

            let mut reasons = BayesStr::empty();
            assert_eq!(anofox_bayes_ffi_fit_status(fit, &mut reasons), 0);

            anofox_bayes_ffi_fit_free(fit);
            anofox_bayes_ffi_data_free(data);
        }
    }

    #[test]
    fn a_bad_config_returns_null_with_a_readable_message() {
        unsafe {
            let data = anofox_bayes_ffi_data_new(2);
            let values = [1.0, 2.0];
            let valid: [u8; 2] = [1; 2];
            anofox_bayes_ffi_data_add_numeric(data, s("cost"), values.as_ptr(), valid.as_ptr(), 2);

            let mut err = err_buf();
            let fit = anofox_bayes_ffi_fit(
                s("conjugate_anomaly"),
                s(r#"{"value": "nope"}"#),
                data,
                &mut err,
            );
            assert!(fit.is_null());
            assert_eq!(err.code, ErrorCode::MissingColumn as i32);

            let msg: Vec<u8> = err
                .message
                .iter()
                .take_while(|c| **c != 0)
                .map(|c| *c as u8)
                .collect();
            let msg = String::from_utf8(msg).unwrap();
            assert!(msg.contains("nope"), "{msg}");
            assert!(msg.contains("cost"), "{msg}");

            anofox_bayes_ffi_data_free(data);
        }
    }

    #[test]
    fn a_column_whose_length_disagrees_with_the_relation_is_rejected() {
        unsafe {
            let data = anofox_bayes_ffi_data_new(3);
            let values = [1.0, 2.0];
            let valid: [u8; 2] = [1; 2];
            assert!(!anofox_bayes_ffi_data_add_numeric(
                data,
                s("cost"),
                values.as_ptr(),
                valid.as_ptr(),
                2
            ));
            anofox_bayes_ffi_data_free(data);
        }
    }

    #[test]
    fn null_handles_are_survivable_rather_than_fatal() {
        unsafe {
            assert_eq!(anofox_bayes_ffi_fit_row_count(std::ptr::null()), 0);
            anofox_bayes_ffi_fit_free(std::ptr::null_mut());
            anofox_bayes_ffi_data_free(std::ptr::null_mut());
            let mut reasons = BayesStr::empty();
            assert_eq!(
                anofox_bayes_ffi_fit_status(std::ptr::null(), &mut reasons),
                -1
            );
        }
    }

    /// A grouped fit through the FFI, with the group keys arriving as borrowed
    /// strings the builder must copy.
    #[test]
    fn key_columns_are_copied_so_the_callers_buffers_may_be_recycled() {
        unsafe {
            let data = anofox_bayes_ffi_data_new(6);
            let values = [1.0, 1.1, 0.9, 5.0, 5.1, 4.9];
            let valid: [u8; 6] = [1; 6];
            anofox_bayes_ffi_data_add_numeric(data, s("cost"), values.as_ptr(), valid.as_ptr(), 6);

            {
                // These strings go out of scope before the fit runs.
                let owned: Vec<String> = ["A", "A", "A", "B", "B", "B"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let slices: Vec<BayesStr> =
                    owned.iter().map(|s| BayesStr::from(s.as_str())).collect();
                assert!(anofox_bayes_ffi_data_add_key(
                    data,
                    s("lane"),
                    slices.as_ptr(),
                    valid.as_ptr(),
                    6
                ));
            }

            let mut err = err_buf();
            let fit = anofox_bayes_ffi_fit(
                s("conjugate_anomaly"),
                s(r#"{"value": "cost", "group": "lane", "draws": 500}"#),
                data,
                &mut err,
            );
            assert!(!fit.is_null(), "code {}", err.code);
            // 2 lanes x (mu, sigma) x 500 draws, plus metadata.
            assert_eq!(anofox_bayes_ffi_fit_row_count(fit), 8 + 500 * 4);

            anofox_bayes_ffi_fit_free(fit);
            anofox_bayes_ffi_data_free(data);
        }
    }
}
