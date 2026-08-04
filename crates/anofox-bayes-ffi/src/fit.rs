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
use anofox_bayes_core::draws::RunKind;
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
    threads: u32,
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

    let result = crate::guard(
        Err(BayesError::Internal(
            "the fit panicked; this is a bug, not a data problem".to_string(),
        )),
        || {
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

            // The whole fit runs inside the caller's thread budget, not just the
            // parts that happen to use rayon today: compile, sample and diagnostics
            // are all parallel sites or may become ones, and a budget applied at only
            // some of them is a budget a reader has to audit rather than trust.
            anofox_bayes_core::parallel::with_thread_budget(threads as usize, || {
                core_fit(family, &cfg, &view)
            })
        },
    );

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

    crate::guard(0, || {
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
    })
}

/// A block of structurally identical rows, described rather than materialised.
///
/// See `Posterior::run_at`. `kind` is 0 for a run of parameter rows and 1 for a single
/// row with no exploitable structure; a caller reads the latter through
/// `anofox_bayes_ffi_fit_rows` at that index.
///
/// For a parameter run, `values` points **into the fit's own buffer** and stays valid
/// as long as the fit does. That is the whole point: the `value` column of the block is
/// already contiguous and in emission order, so it can be copied out in one move
/// instead of a row at a time.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BayesRun {
    pub kind: i32,
    pub chain: i32,
    pub draw: i32,
    pub start: usize,
    pub len: usize,
    pub first_param: usize,
    pub values: *const f64,
    /// Whether any value in the block is NaN, so the common case can skip the scan
    /// that turns NaN into SQL NULL.
    pub has_nan: bool,
}

/// Describe the run of rows beginning at `offset`, clamped to `max` rows.
///
/// Returns false when `offset` is past the end.
///
/// # Safety
/// `fit` must come from `anofox_bayes_ffi_fit`, and `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_run(
    fit: *const BayesFit,
    offset: usize,
    max: usize,
    out: *mut BayesRun,
) -> bool {
    let Some(fit) = fit.as_ref() else {
        return false;
    };
    if out.is_null() || max == 0 {
        return false;
    }
    crate::guard(false, || {
        let Some(run) = fit.fit.posterior.run_at(offset) else {
            return false;
        };
        let len = run.len.min(max);
        let values = &run.values[..len.min(run.values.len())];
        *out = BayesRun {
            kind: match run.kind {
                RunKind::Params => 0,
                RunKind::Single => 1,
            },
            chain: run.chain,
            draw: run.draw,
            start: run.start,
            len,
            first_param: run.first_param,
            values: if values.is_empty() {
                std::ptr::null()
            } else {
                values.as_ptr()
            },
            has_nan: values.iter().any(|v| v.is_nan()),
        };
        true
    })
}

/// The parameter names and group ids, in parameter order.
///
/// A run of parameter rows names its parameters by index, so a caller can build the
/// dictionary once per fit instead of re-deriving a name per row.
///
/// # Safety
/// `fit` must come from `anofox_bayes_ffi_fit`; `out_group_id` and `out_param` must
/// each have room for `anofox_bayes_ffi_fit_param_count` entries.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_param_names(
    fit: *const BayesFit,
    out_group_id: *mut BayesStr,
    out_param: *mut BayesStr,
) -> usize {
    let Some(fit) = fit.as_ref() else {
        return 0;
    };
    if out_group_id.is_null() || out_param.is_null() {
        return 0;
    }
    crate::guard(0, || {
        let params = &fit.fit.posterior.params;
        for (i, p) in params.iter().enumerate() {
            *out_group_id.add(i) = BayesStr::from(p.group_id.as_str());
            *out_param.add(i) = BayesStr::from(p.name.as_str());
        }
        params.len()
    })
}

/// How many parameters the fit reports, which is the length of the run dictionary.
///
/// # Safety
/// `fit` must come from `anofox_bayes_ffi_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_bayes_ffi_fit_param_count(fit: *const BayesFit) -> usize {
    match fit.as_ref() {
        None => 0,
        Some(f) => f.fit.posterior.params.len(),
    }
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
    use anofox_bayes_core::draws::META_ROWS;

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
    /// **The run path must produce the same table as the row path, through the ABI.**
    ///
    /// `run_at` is pinned against `row_at` in the core. This pins the C surface on top
    /// of it: the same fit, walked by runs and reassembled from the dictionary, must
    /// give byte-identical values and the same names in the same order. A caller that
    /// trusted the runs and got a different table would be silently wrong in a way no
    /// SQL assertion downstream would localise.
    #[test]
    fn the_run_surface_reassembles_the_same_table_as_the_row_surface() {
        unsafe {
            let data = anofox_bayes_ffi_data_new(8);
            let values = [2.0, 2.1, 1.9, 2.05, 1.95, 2.02, 2.2, 1.85];
            let valid: [u8; 8] = [1; 8];
            assert!(anofox_bayes_ffi_data_add_numeric(
                data,
                s("cost"),
                values.as_ptr(),
                valid.as_ptr(),
                8
            ));
            let mut err = err_buf();
            let fit = anofox_bayes_ffi_fit(
                s("conjugate_anomaly"),
                s(r#"{"value": "cost", "draws": 200, "seed": 11}"#),
                data,
                1,
                &mut err,
            );
            assert!(!fit.is_null(), "fit failed with code {}", err.code);

            let n_params = anofox_bayes_ffi_fit_param_count(fit);
            let mut dict_group = vec![BayesStr::empty(); n_params];
            let mut dict_param = vec![BayesStr::empty(); n_params];
            assert_eq!(
                anofox_bayes_ffi_fit_param_names(
                    fit,
                    dict_group.as_mut_ptr(),
                    dict_param.as_mut_ptr()
                ),
                n_params
            );
            let as_str = |b: BayesStr| -> String {
                if b.ptr.is_null() {
                    String::new()
                } else {
                    std::str::from_utf8(std::slice::from_raw_parts(b.ptr as *const u8, b.len))
                        .unwrap()
                        .to_string()
                }
            };

            let total = anofox_bayes_ffi_fit_row_count(fit);
            let cap = 64usize;
            let mut offset = 0usize;
            while offset < total {
                let mut run = BayesRun {
                    kind: -1,
                    chain: 0,
                    draw: 0,
                    start: 0,
                    len: 0,
                    first_param: 0,
                    values: std::ptr::null(),
                    has_nan: false,
                };
                assert!(anofox_bayes_ffi_fit_run(fit, offset, cap, &mut run));
                assert!(run.len > 0 && run.len <= cap);
                assert_eq!(run.start, offset);

                // The same rows, read the old way.
                let n = run.len;
                let mut model_id = vec![BayesStr::empty(); n];
                let mut group_id = vec![BayesStr::empty(); n];
                let mut chain = vec![0i32; n];
                let mut draw = vec![0i32; n];
                let mut param = vec![BayesStr::empty(); n];
                let mut value = vec![0.0f64; n];
                assert_eq!(
                    anofox_bayes_ffi_fit_rows(
                        fit,
                        offset,
                        n,
                        model_id.as_mut_ptr(),
                        group_id.as_mut_ptr(),
                        chain.as_mut_ptr(),
                        draw.as_mut_ptr(),
                        param.as_mut_ptr(),
                        value.as_mut_ptr(),
                    ),
                    n
                );

                if run.kind == 0 {
                    let vals = std::slice::from_raw_parts(run.values, run.len);
                    let mut any_nan = false;
                    for i in 0..n {
                        assert_eq!(run.chain, chain[i], "chain constant across a run");
                        assert_eq!(run.draw, draw[i], "draw constant across a run");
                        assert_eq!(
                            vals[i].to_bits(),
                            value[i].to_bits(),
                            "the run's contiguous values are the row values"
                        );
                        any_nan |= vals[i].is_nan();
                        let slot = run.first_param + i;
                        assert_eq!(as_str(dict_group[slot]), as_str(group_id[i]));
                        assert_eq!(as_str(dict_param[slot]), as_str(param[i]));
                    }
                    assert_eq!(run.has_nan, any_nan, "has_nan must describe the block");
                }

                offset += run.len;
            }
            assert_eq!(offset, total, "the run walk covers the table exactly");
            let mut past = BayesRun {
                kind: -1,
                chain: 0,
                draw: 0,
                start: 0,
                len: 0,
                first_param: 0,
                values: std::ptr::null(),
                has_nan: false,
            };
            assert!(!anofox_bayes_ffi_fit_run(fit, total, cap, &mut past));

            anofox_bayes_ffi_fit_free(fit);
        }
    }

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
                // A budget of 1: a unit test should not silently take the machine.
                1,
                &mut err,
            );
            assert!(!fit.is_null(), "fit failed with code {}", err.code);
            assert_eq!(err.code, ErrorCode::Success as i32);

            let total = anofox_bayes_ffi_fit_row_count(fit);
            // The metadata block + 1000 draws x 2 parameters. Taken from `META_ROWS`
            // rather than written out: the block grows as new provenance rows are
            // added, and that growth is explicitly not a contract break.
            assert_eq!(total, META_ROWS.len() + 1000 * 2);

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
                1,
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
                1,
                &mut err,
            );
            assert!(!fit.is_null(), "code {}", err.code);
            // 2 lanes x (mu, sigma) x 500 draws, plus the metadata block.
            assert_eq!(
                anofox_bayes_ffi_fit_row_count(fit),
                META_ROWS.len() + 500 * 4
            );

            anofox_bayes_ffi_fit_free(fit);
            anofox_bayes_ffi_data_free(data);
        }
    }
}
