//! Errors that cross the FFI boundary as machine-readable codes.
//!
//! The primary caller of this extension is an agent, not a human, so an error is a
//! branch point in a workflow rather than a message in a log. Two consequences shape
//! this type:
//!
//! 1. **Configuration errors carry the offending slot.** An agent that gets
//!    `Config { slot: "priors.beta.scale", .. }` can repair its own request; one that
//!    gets `"invalid configuration"` can only give up.
//! 2. **Errors are not the refusal path.** "The data is too weak to support a
//!    conclusion" is a *successful* fit with [`FitStatus::InsufficientData`], not an
//!    error — see [`crate::types::FitStatus`]. `BayesError` means the request could not
//!    be executed at all.

use thiserror::Error;

pub type BayesResult<T> = Result<T, BayesError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum BayesError {
    /// The requested family is not in the catalog. The catalog is closed by design
    /// (BRD §4), so this is a permanent, not a transient, failure.
    #[error("unknown model family '{name}' (catalog: {catalog})")]
    UnknownFamily { name: String, catalog: String },

    /// A configuration slot failed validation. `slot` is a dotted path into the
    /// config object so the caller can repair exactly one field.
    #[error("invalid config at '{slot}': {reason}")]
    Config { slot: String, reason: String },

    /// A column named in the config is absent from the input relation.
    ///
    /// `available` lists what the relation does offer, so a caller who wrote
    /// `cost_per_kilo` for a `cost_per_kg` column can repair the request from the
    /// message alone.
    #[error("column '{column}' not found in input data (available: {available})")]
    MissingColumn { column: String, available: String },

    /// Column lengths disagree — a bug in the caller, not in the data.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Not enough usable rows to identify the model at all. Distinct from
    /// [`crate::types::FitStatus::InsufficientData`], which means the fit ran but the
    /// posterior is prior-dominated.
    #[error("insufficient data: {rows} usable rows for {params} parameters")]
    InsufficientData { rows: usize, params: usize },

    /// The design matrix has no full-rank solution and no policy resolved it.
    #[error("singular or rank-deficient design matrix")]
    SingularMatrix,

    /// A decomposition that requires positive definiteness was handed something else.
    /// In practice this means the curvature at the mode is not a valid covariance.
    #[error("matrix is not positive definite: {0}")]
    NotPositiveDefinite(String),

    /// An iterative fit ran out of iterations.
    #[error("failed to converge after {iterations} iterations (tolerance {tolerance})")]
    ConvergenceFailure { iterations: u32, tolerance: f64 },

    /// A parameter name collided with the reserved `__`-prefixed namespace used by
    /// the draws contract for sample statistics and model metadata.
    #[error("parameter name '{0}' is reserved (the '__' prefix belongs to the draws contract)")]
    ReservedParamName(String),

    /// Anything that should be impossible. Carries context because a bug report with
    /// no context costs a day.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Stable numeric codes for the FFI boundary.
///
/// These are part of the public contract: the C++ layer and, through it, agent
/// workflows branch on them. Codes are append-only — never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorCode {
    Success = 0,
    UnknownFamily = 1,
    Config = 2,
    MissingColumn = 3,
    DimensionMismatch = 4,
    InsufficientData = 5,
    SingularMatrix = 6,
    NotPositiveDefinite = 7,
    ConvergenceFailure = 8,
    ReservedParamName = 9,
    Internal = 99,
}

impl BayesError {
    pub fn code(&self) -> ErrorCode {
        match self {
            BayesError::UnknownFamily { .. } => ErrorCode::UnknownFamily,
            BayesError::Config { .. } => ErrorCode::Config,
            BayesError::MissingColumn { .. } => ErrorCode::MissingColumn,
            BayesError::DimensionMismatch(_) => ErrorCode::DimensionMismatch,
            BayesError::InsufficientData { .. } => ErrorCode::InsufficientData,
            BayesError::SingularMatrix => ErrorCode::SingularMatrix,
            BayesError::NotPositiveDefinite(_) => ErrorCode::NotPositiveDefinite,
            BayesError::ConvergenceFailure { .. } => ErrorCode::ConvergenceFailure,
            BayesError::ReservedParamName(_) => ErrorCode::ReservedParamName,
            BayesError::Internal(_) => ErrorCode::Internal,
        }
    }

    /// Convenience constructor for the common `Config` case.
    pub fn config(slot: impl Into<String>, reason: impl Into<String>) -> Self {
        BayesError::Config {
            slot: slot.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_errors_name_the_slot_that_failed() {
        let err = BayesError::config("priors.beta.scale", "must be > 0, got -1");
        assert_eq!(
            err.to_string(),
            "invalid config at 'priors.beta.scale': must be > 0, got -1"
        );
        assert_eq!(err.code(), ErrorCode::Config);
    }

    /// Codes are a wire contract. If this test needs editing, the C++ layer and any
    /// deployed agent workflow need editing too.
    #[test]
    fn error_codes_are_stable() {
        assert_eq!(ErrorCode::Success as i32, 0);
        assert_eq!(ErrorCode::UnknownFamily as i32, 1);
        assert_eq!(ErrorCode::Config as i32, 2);
        assert_eq!(ErrorCode::MissingColumn as i32, 3);
        assert_eq!(ErrorCode::DimensionMismatch as i32, 4);
        assert_eq!(ErrorCode::InsufficientData as i32, 5);
        assert_eq!(ErrorCode::SingularMatrix as i32, 6);
        assert_eq!(ErrorCode::NotPositiveDefinite as i32, 7);
        assert_eq!(ErrorCode::ConvergenceFailure as i32, 8);
        assert_eq!(ErrorCode::ReservedParamName as i32, 9);
        assert_eq!(ErrorCode::Internal as i32, 99);
    }
}
