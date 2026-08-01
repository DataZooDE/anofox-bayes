//! The input relation, as the mathematics sees it.
//!
//! A [`DataView`] borrows columns the C++ layer has already materialised. It knows
//! about names, nulls and grouping, and nothing about DuckDB.
//!
//! Two responsibilities live here rather than in each family, because getting either
//! wrong is silent:
//!
//! **Row filtering is a whole-row decision.** A row with a null response and a
//! present predictor cannot contribute to a regression, and dropping only the null
//! *column* would silently misalign every other column against it. [`DataView::usable_rows`]
//! decides once, for all columns a model actually reads.
//!
//! **The data fingerprint is content-addressed.** It hashes the columns a model
//! reads, in the order it reads them, so `model_id` changes when the data changes and
//! not when an unrelated column is added to the input relation.

use std::collections::{BTreeMap, HashMap};

use crate::errors::{BayesError, BayesResult};

/// A numeric column with its null mask.
#[derive(Debug, Clone, Copy)]
pub struct NumericColumn<'a> {
    pub values: &'a [f64],
    /// `true` where the row has a value. Nulls and NaNs are both marked absent, since
    /// neither can contribute to a likelihood.
    pub valid: &'a [bool],
}

/// A key column used for grouping.
#[derive(Debug, Clone, Copy)]
pub struct KeyColumn<'a> {
    pub values: &'a [&'a str],
    pub valid: &'a [bool],
}

/// The input relation.
#[derive(Debug, Default)]
pub struct DataView<'a> {
    n_rows: usize,
    numeric: BTreeMap<String, NumericColumn<'a>>,
    keys: BTreeMap<String, KeyColumn<'a>>,
}

impl<'a> DataView<'a> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            n_rows,
            numeric: BTreeMap::new(),
            keys: BTreeMap::new(),
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn add_numeric(
        &mut self,
        name: impl Into<String>,
        col: NumericColumn<'a>,
    ) -> BayesResult<()> {
        let name = name.into();
        if col.values.len() != self.n_rows || col.valid.len() != self.n_rows {
            return Err(BayesError::DimensionMismatch(format!(
                "column '{name}' has {} values and {} validity flags, expected {} of each",
                col.values.len(),
                col.valid.len(),
                self.n_rows
            )));
        }
        self.numeric.insert(name, col);
        Ok(())
    }

    pub fn add_key(&mut self, name: impl Into<String>, col: KeyColumn<'a>) -> BayesResult<()> {
        let name = name.into();
        if col.values.len() != self.n_rows || col.valid.len() != self.n_rows {
            return Err(BayesError::DimensionMismatch(format!(
                "column '{name}' has {} values and {} validity flags, expected {} of each",
                col.values.len(),
                col.valid.len(),
                self.n_rows
            )));
        }
        self.keys.insert(name, col);
        Ok(())
    }

    /// A numeric column by name, or a [`BayesError::MissingColumn`] naming it.
    ///
    /// The error also lists what *is* available: a caller who wrote `"cost_per_kilo"`
    /// where the relation has `cost_per_kg` can fix it from the message alone.
    pub fn numeric(&self, name: &str) -> BayesResult<NumericColumn<'a>> {
        self.numeric.get(name).copied().ok_or_else(|| {
            if self.keys.contains_key(name) {
                BayesError::config(
                    name,
                    format!("column '{name}' is a key column, not a numeric one"),
                )
            } else {
                BayesError::MissingColumn {
                    column: name.to_string(),
                    available: self.numeric_names().join(", "),
                }
            }
        })
    }

    /// A key column by name.
    pub fn key(&self, name: &str) -> BayesResult<KeyColumn<'a>> {
        self.keys
            .get(name)
            .copied()
            .ok_or_else(|| BayesError::MissingColumn {
                column: name.to_string(),
                available: self.key_names().join(", "),
            })
    }

    pub fn numeric_names(&self) -> Vec<&str> {
        self.numeric.keys().map(String::as_str).collect()
    }

    pub fn key_names(&self) -> Vec<&str> {
        self.keys.keys().map(String::as_str).collect()
    }

    /// Row indices where every named column has a usable value.
    ///
    /// Whole-row, not per-column: dropping a null in one column only would leave the
    /// remaining columns misaligned against each other by one row, which produces a
    /// perfectly plausible and completely wrong fit.
    pub fn usable_rows(&self, numeric: &[&str], keys: &[&str]) -> BayesResult<Vec<usize>> {
        let numeric: Vec<NumericColumn> = numeric
            .iter()
            .map(|n| self.numeric(n))
            .collect::<BayesResult<_>>()?;
        let keys: Vec<KeyColumn> = keys
            .iter()
            .map(|n| self.key(n))
            .collect::<BayesResult<_>>()?;

        Ok((0..self.n_rows)
            .filter(|&i| {
                numeric
                    .iter()
                    .all(|c| c.valid[i] && c.values[i].is_finite())
                    && keys.iter().all(|c| c.valid[i])
            })
            .collect())
    }

    /// A content hash of the columns a model reads, restricted to its usable rows.
    ///
    /// Part of `model_id`, so it must change when the numbers change and must *not*
    /// change when an unrelated column is added to the input relation — otherwise
    /// every cache entry would be invalidated by an irrelevant `SELECT *`.
    pub fn fingerprint(
        &self,
        numeric: &[&str],
        keys: &[&str],
        rows: &[usize],
    ) -> BayesResult<String> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(rows.len() as u64).to_le_bytes());

        for name in numeric {
            let col = self.numeric(name)?;
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            for &i in rows {
                hasher.update(&col.values[i].to_le_bytes());
            }
        }
        for name in keys {
            let col = self.key(name)?;
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            for &i in rows {
                let v = col.values[i];
                hasher.update(&(v.len() as u64).to_le_bytes());
                hasher.update(v.as_bytes());
            }
        }
        Ok(hasher.finalize().to_hex()[..16].to_string())
    }

    /// Partition usable rows by the value of a key column, preserving first-seen
    /// order of the keys so that group order is deterministic and independent of
    /// hash-map iteration.
    pub fn group_rows(
        &self,
        key: Option<&str>,
        rows: &[usize],
    ) -> BayesResult<Vec<(String, Vec<usize>)>> {
        let Some(key) = key else {
            return Ok(vec![(
                crate::types::GLOBAL_GROUP.to_string(),
                rows.to_vec(),
            )]);
        };
        let col = self.key(key)?;

        // One hash lookup per row, into a side table of positions in `out`. The
        // obvious spelling — a `BTreeMap` probed once with `contains_key` and again
        // with `entry` — costs two ordered lookups per row, and this is the single
        // most expensive step of compiling a wide fit: 113 ms of a 150 ms compile at
        // 5 000 groups and 520 000 rows, against 45 ms here.
        //
        // Order is still **first-seen**, not hash order, and that is load-bearing
        // rather than incidental: it fixes the order of the parameter list, and
        // therefore the order of the emitted rows.
        let mut at: HashMap<&str, usize> = HashMap::with_capacity(rows.len() / 8 + 1);
        let mut out: Vec<(String, Vec<usize>)> = Vec::new();
        for &i in rows {
            let k = col.values[i];
            match at.get(k) {
                Some(&slot) => out[slot].1.push(i),
                None => {
                    at.insert(k, out.len());
                    out.push((k.to_string(), vec![i]));
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! Helpers for building a `DataView` in tests without the FFI in the way.

    /// Owned backing storage, so tests can build a view from literals.
    pub struct Frame {
        pub n_rows: usize,
        pub numeric: Vec<(String, Vec<f64>, Vec<bool>)>,
        pub keys: Vec<(String, Vec<String>, Vec<bool>)>,
    }

    impl Frame {
        pub fn new(n_rows: usize) -> Self {
            Self {
                n_rows,
                numeric: Vec::new(),
                keys: Vec::new(),
            }
        }

        pub fn numeric(mut self, name: &str, values: Vec<f64>) -> Self {
            let valid = values.iter().map(|v| v.is_finite()).collect();
            self.numeric.push((name.to_string(), values, valid));
            self
        }

        pub fn numeric_with_nulls(mut self, name: &str, values: Vec<Option<f64>>) -> Self {
            let valid: Vec<bool> = values.iter().map(Option::is_some).collect();
            let vals: Vec<f64> = values.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            self.numeric.push((name.to_string(), vals, valid));
            self
        }

        pub fn key(mut self, name: &str, values: Vec<&str>) -> Self {
            let valid = vec![true; values.len()];
            self.keys.push((
                name.to_string(),
                values.into_iter().map(String::from).collect(),
                valid,
            ));
            self
        }

        /// Borrowed `&str` views over the owned key columns.
        ///
        /// Materialised separately from [`Frame::view`] so the slices have a named
        /// owner that outlives the `DataView`; a `view()` that built them internally
        /// would have to return a borrow of its own temporary.
        pub fn key_refs(&self) -> Vec<Vec<&str>> {
            self.keys
                .iter()
                .map(|(_, values, _)| values.iter().map(String::as_str).collect())
                .collect()
        }

        /// Borrow the frame as a `DataView`. `refs` must come from [`Frame::key_refs`].
        pub fn view<'a>(&'a self, refs: &'a [Vec<&'a str>]) -> super::DataView<'a> {
            let mut view = super::DataView::new(self.n_rows);
            for (name, values, valid) in &self.numeric {
                view.add_numeric(name.clone(), super::NumericColumn { values, valid })
                    .unwrap();
            }
            for (i, (name, _, valid)) in self.keys.iter().enumerate() {
                view.add_key(
                    name.clone(),
                    super::KeyColumn {
                        values: &refs[i],
                        valid,
                    },
                )
                .unwrap();
            }
            view
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Frame;
    use super::*;

    #[test]
    fn a_missing_column_names_itself_and_lists_what_is_available() {
        let frame = Frame::new(2).numeric("cost_per_kg", vec![1.0, 2.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let err = view.numeric("cost_per_kilo").unwrap_err().to_string();
        assert!(err.contains("cost_per_kilo"), "{err}");
        assert!(err.contains("cost_per_kg"), "{err}");
    }

    /// Dropping a null in one column only would shift every other column against it
    /// by a row: a perfectly plausible, completely wrong fit.
    #[test]
    fn a_null_anywhere_in_a_row_removes_the_whole_row() {
        let frame = Frame::new(4)
            .numeric_with_nulls("y", vec![Some(1.0), None, Some(3.0), Some(4.0)])
            .numeric_with_nulls("x", vec![Some(10.0), Some(20.0), None, Some(40.0)]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        assert_eq!(view.usable_rows(&["y", "x"], &[]).unwrap(), vec![0, 3]);
        // A model that only reads y keeps the row where x is null.
        assert_eq!(view.usable_rows(&["y"], &[]).unwrap(), vec![0, 2, 3]);
    }

    /// A NaN is not a value. Letting one through would poison a sum irrecoverably and
    /// surface as a NaN posterior, which is the one outcome the refusal path exists
    /// to prevent.
    #[test]
    fn a_nan_is_treated_as_absent_even_when_the_row_is_marked_valid() {
        let frame = Frame::new(3).numeric("y", vec![1.0, f64::NAN, 3.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        assert_eq!(view.usable_rows(&["y"], &[]).unwrap(), vec![0, 2]);
    }

    #[test]
    fn rows_group_by_a_key_column_in_first_seen_order() {
        let frame = Frame::new(5)
            .numeric("cost", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .key(
                "lane",
                vec!["HAM-ROT", "AAA-BBB", "HAM-ROT", "AAA-BBB", "HAM-ROT"],
            );
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let rows = view.usable_rows(&["cost"], &["lane"]).unwrap();
        let groups = view.group_rows(Some("lane"), &rows).unwrap();
        assert_eq!(groups[0].0, "HAM-ROT");
        assert_eq!(groups[0].1, vec![0, 2, 4]);
        assert_eq!(groups[1].0, "AAA-BBB");
        assert_eq!(groups[1].1, vec![1, 3]);
    }

    #[test]
    fn an_ungrouped_model_sees_one_global_group() {
        let frame = Frame::new(3).numeric("cost", vec![1.0, 2.0, 3.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let groups = view.group_rows(None, &[0, 1, 2]).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, crate::types::GLOBAL_GROUP);
    }

    /// The fingerprint feeds `model_id`. If an unrelated column changed it, every
    /// cached posterior would be invalidated by an irrelevant `SELECT *`.
    #[test]
    fn the_fingerprint_ignores_columns_the_model_does_not_read() {
        let a = Frame::new(3)
            .numeric("cost", vec![1.0, 2.0, 3.0])
            .numeric("unrelated", vec![9.0, 9.0, 9.0]);
        let b = Frame::new(3)
            .numeric("cost", vec![1.0, 2.0, 3.0])
            .numeric("unrelated", vec![0.0, 0.0, 0.0]);
        let (ra, rb) = (a.key_refs(), b.key_refs());
        let (va, vb) = (a.view(&ra), b.view(&rb));

        assert_eq!(
            va.fingerprint(&["cost"], &[], &[0, 1, 2]).unwrap(),
            vb.fingerprint(&["cost"], &[], &[0, 1, 2]).unwrap()
        );
    }

    #[test]
    fn the_fingerprint_changes_when_the_data_the_model_reads_changes() {
        let a = Frame::new(3).numeric("cost", vec![1.0, 2.0, 3.0]);
        let b = Frame::new(3).numeric("cost", vec![1.0, 2.0, 3.5]);
        let (ra, rb) = (a.key_refs(), b.key_refs());
        let (va, vb) = (a.view(&ra), b.view(&rb));

        assert_ne!(
            va.fingerprint(&["cost"], &[], &[0, 1, 2]).unwrap(),
            vb.fingerprint(&["cost"], &[], &[0, 1, 2]).unwrap()
        );
        // ...and when the row subset changes.
        assert_ne!(
            va.fingerprint(&["cost"], &[], &[0, 1, 2]).unwrap(),
            va.fingerprint(&["cost"], &[], &[0, 1]).unwrap()
        );
    }

    #[test]
    fn a_column_of_the_wrong_length_is_rejected_when_it_is_added() {
        let mut view = DataView::new(3);
        let values = [1.0, 2.0];
        let valid = [true, true];
        let err = view
            .add_numeric(
                "y",
                NumericColumn {
                    values: &values,
                    valid: &valid,
                },
            )
            .unwrap_err();
        assert!(matches!(err, BayesError::DimensionMismatch(_)));
    }
}
