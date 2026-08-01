//! Shared vocabulary: the draws contract, fit status, engine identity.
//!
//! Everything in this module is part of a versioned wire contract. The SQL surface,
//! the FFI layer and agent workflows all read these values, so numeric encodings are
//! append-only and changes to [`DRAWS_SCHEMA_VERSION`] are breaking.

use crate::errors::{BayesError, BayesResult};

/// Version of the long-format draws schema documented in `docs/DRAWS_CONTRACT.md`.
///
/// Emitted into every draws table as the `__schema_version__` row so that a table
/// written by an older extension can still be read and interpreted correctly. Bump
/// only for a breaking change to column meaning or reserved-name semantics.
pub const DRAWS_SCHEMA_VERSION: i32 = 1;

/// Prefix reserved for sample statistics and model metadata in the `param` column.
///
/// User-facing model parameters may never start with this; see
/// [`validate_param_name`]. Keeping the two namespaces disjoint is what lets a single
/// draws table be self-describing without a second metadata table.
pub const RESERVED_PREFIX: &str = "__";

/// `group_id` used for parameters that are not group-specific (population level).
pub const GLOBAL_GROUP: &str = "__global__";

/// Outcome of a fit, as seen by an agent's quality gate.
///
/// This is the machine-readable refusal path of BRD BR-5. A fit that cannot support a
/// conclusion returns [`FitStatus::InsufficientData`] *successfully* — the agent then
/// refuses rather than reporting a fabricated number. Silent bad numbers are the one
/// outcome this type exists to make impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FitStatus {
    /// All parameters passed their diagnostics thresholds. Safe to act on.
    Converged = 0,
    /// The fit ran but diagnostics failed (R-hat, ESS, or divergences). The draws
    /// exist and can be inspected, but must not drive a decision.
    Degenerate = 1,
    /// The fit ran and is numerically fine, but the posterior is dominated by the
    /// prior: the data carries too little signal to answer the question.
    InsufficientData = 2,
    /// The fit could not be completed. Draws are absent.
    Failed = 3,
}

impl FitStatus {
    /// The only status an agent may act on without further qualification.
    pub fn is_actionable(&self) -> bool {
        matches!(self, FitStatus::Converged)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FitStatus::Converged => "converged",
            FitStatus::Degenerate => "degenerate",
            FitStatus::InsufficientData => "insufficient_data",
            FitStatus::Failed => "failed",
        }
    }

    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(FitStatus::Converged),
            1 => Some(FitStatus::Degenerate),
            2 => Some(FitStatus::InsufficientData),
            3 => Some(FitStatus::Failed),
            _ => None,
        }
    }
}

/// Which inference engine produced a posterior.
///
/// Recorded in the draws table so that a downstream consumer can tell an exact
/// conjugate posterior from a Gaussian approximation to one — the numbers look
/// identical in SQL but do not carry the same warranty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum EngineKind {
    /// Closed-form conjugate posterior. Exact up to Monte-Carlo error in the draws.
    #[default]
    Exact = 0,
    /// MAP + curvature: a multivariate normal approximation on the unconstrained
    /// scale. Cheap and excellent for GLM-shaped posteriors; certified per family by
    /// the SBC suite.
    Laplace = 1,
    /// No-U-Turn Sampler via `nuts-rs`. Phase 0.2.
    Nuts = 2,
}

impl EngineKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineKind::Exact => "exact",
            EngineKind::Laplace => "laplace",
            EngineKind::Nuts => "nuts",
        }
    }

    /// Parse an engine override from a config value. Unknown names are a config
    /// error rather than a silent fallback: an agent that asked for NUTS and quietly
    /// got Laplace would report unearned confidence.
    pub fn parse(name: &str) -> BayesResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "exact" | "conjugate" => Ok(EngineKind::Exact),
            "laplace" => Ok(EngineKind::Laplace),
            "nuts" => Ok(EngineKind::Nuts),
            other => Err(BayesError::config(
                "engine",
                format!("unknown engine '{other}' (expected exact, laplace or nuts)"),
            )),
        }
    }
}

/// Reject parameter names that would collide with the reserved namespace.
///
/// Called wherever a family derives parameter names from user data — a column
/// literally named `__lp__` would otherwise overwrite the log-density sample
/// statistic and silently corrupt every diagnostic computed from the table.
pub fn validate_param_name(name: &str) -> BayesResult<()> {
    if name.starts_with(RESERVED_PREFIX) {
        return Err(BayesError::ReservedParamName(name.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_converged_is_actionable() {
        assert!(FitStatus::Converged.is_actionable());
        assert!(!FitStatus::Degenerate.is_actionable());
        assert!(!FitStatus::InsufficientData.is_actionable());
        assert!(!FitStatus::Failed.is_actionable());
    }

    /// Status codes travel into agent workflows as integers. Renumbering them would
    /// silently turn a refusal into an approval at a customer site.
    #[test]
    fn status_codes_are_stable() {
        assert_eq!(FitStatus::Converged as i32, 0);
        assert_eq!(FitStatus::Degenerate as i32, 1);
        assert_eq!(FitStatus::InsufficientData as i32, 2);
        assert_eq!(FitStatus::Failed as i32, 3);
        for code in 0..4 {
            assert_eq!(FitStatus::from_code(code).map(|s| s as i32), Some(code));
        }
        assert_eq!(FitStatus::from_code(4), None);
    }

    #[test]
    fn engine_names_round_trip() {
        for engine in [EngineKind::Exact, EngineKind::Laplace, EngineKind::Nuts] {
            assert_eq!(EngineKind::parse(engine.as_str()).unwrap(), engine);
        }
        assert_eq!(EngineKind::parse("CONJUGATE").unwrap(), EngineKind::Exact);
    }

    #[test]
    fn an_unknown_engine_is_a_config_error_not_a_silent_default() {
        let err = EngineKind::parse("hmc").unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "engine"));
    }

    #[test]
    fn reserved_parameter_names_are_rejected() {
        assert!(validate_param_name("beta[price]").is_ok());
        assert!(validate_param_name("sigma").is_ok());
        let err = validate_param_name("__lp__").unwrap_err();
        assert!(matches!(err, BayesError::ReservedParamName(_)));
    }
}
