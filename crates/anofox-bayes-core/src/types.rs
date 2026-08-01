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

/// Which catalog family produced a posterior.
///
/// Recorded in the draws table as the `__family__` row so that an auditor holding
/// only the persisted table can say *what model was fitted*. Without it the table
/// reports how it was fitted (`__engine__`), with what budget and to what data, but
/// not which likelihood produced the numbers.
///
/// **Why a number and not the family's name.** The `value` column of the draws
/// contract is `DOUBLE`, and the only `VARCHAR` columns — `model_id` and `group_id` —
/// already carry meanings that a metadata row may not overload. Adding a column would
/// be a breaking change to the contract; a numeric code is not.
///
/// **Why *these* numbers.** They are the catalog's F-numbers, already fixed in
/// `docs/BRD.md` §6 and used throughout `docs/API_REFERENCE.md` and `docs/HLD.md`:
/// `pooled_gaussian` is F3 and `conjugate_anomaly` is F7. Inventing a second,
/// registration-ordered numbering would have created two identities for one family
/// and a table mapping between them to keep in step. The gaps are the families this
/// catalog does not ship yet, which is information rather than an accident.
///
/// Numbering is **append-only**, exactly as for [`FitStatus`] and [`EngineKind`]:
/// these values travel into persisted customer tables, so a renumbering would change
/// the meaning of a table already written. A family outside the BRD's F1–F7 planning
/// grid takes the next unused code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum FamilyCode {
    /// F3 — pooled Gaussian linear model.
    PooledGaussian = 3,
    /// F7 — conjugate anomaly (Normal / Poisson closed forms).
    ConjugateAnomaly = 7,
}

impl FamilyCode {
    /// The family's SQL identifier. Equal to `ModelFamily::id()` by construction, and
    /// a catalog test enforces it.
    pub fn as_str(&self) -> &'static str {
        match self {
            FamilyCode::PooledGaussian => "pooled_gaussian",
            FamilyCode::ConjugateAnomaly => "conjugate_anomaly",
        }
    }

    /// Decode a `__family__` value read back off a draws table.
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            3 => Some(FamilyCode::PooledGaussian),
            7 => Some(FamilyCode::ConjugateAnomaly),
            _ => None,
        }
    }
}

/// Which inference engine produced a posterior.
///
/// Recorded in the draws table so that a downstream consumer can tell an exact
/// conjugate posterior from a Gaussian approximation to one — the numbers look
/// Which distribution a fit draws from.
///
/// A **prior predictive** check (BR-11) is the pre-fit gate: draw parameters from the
/// prior alone, simulate data, and look at whether the implied data is physically
/// plausible before spending anything on the posterior. A prior that puts most of its
/// mass on negative delivery times is a modelling error worth finding in the first
/// second rather than the last.
///
/// It is a config slot rather than a separate function because the output is the same
/// draws contract, and because being part of the canonical config means `model_id`
/// covers it for free -- a prior-predictive table and a posterior table over the same
/// data cannot collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum SampleFrom {
    #[default]
    Posterior = 0,
    Prior = 1,
}

impl SampleFrom {
    pub fn as_str(&self) -> &'static str {
        match self {
            SampleFrom::Posterior => "posterior",
            SampleFrom::Prior => "prior",
        }
    }

    pub fn parse(name: &str) -> BayesResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "posterior" => Ok(SampleFrom::Posterior),
            "prior" => Ok(SampleFrom::Prior),
            other => Err(BayesError::config(
                "sample_from",
                format!("unknown: '{other}'; expected 'posterior' or 'prior'"),
            )),
        }
    }
}

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

/// Reject a group key that would collide with the contract's own sentinels.
///
/// `GLOBAL_GROUP` marks population-level parameters, and every metadata row carries
/// it. A dataset with a literal group named `__global__` would put that group's
/// parameters in the same bucket, so `WHERE group_id = '__global__'` — the documented
/// way to select population-level parameters — would silently return one customer
/// segment's estimates mixed in with them.
///
/// Rejecting is right rather than escaping: `__global__` is not a plausible business
/// key, so an error costs nobody anything, while a mangled key would show up in a
/// report under a name the analyst does not recognise.
pub fn validate_group_key(key: &str) -> BayesResult<()> {
    if key.starts_with(RESERVED_PREFIX) {
        return Err(BayesError::config(
            "group",
            format!(
                "group key '{key}' begins with the reserved '{RESERVED_PREFIX}' prefix, \
                 which the draws contract uses for its own rows"
            ),
        ));
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

    /// Family codes travel into persisted customer tables. Renumbering one would
    /// change what a table written last quarter says it contains.
    #[test]
    fn family_codes_are_stable_and_round_trip() {
        assert_eq!(FamilyCode::PooledGaussian as i32, 3);
        assert_eq!(FamilyCode::ConjugateAnomaly as i32, 7);
        for code in [FamilyCode::PooledGaussian, FamilyCode::ConjugateAnomaly] {
            assert_eq!(FamilyCode::from_code(code as i32), Some(code));
        }
        // The gaps are families the catalog does not ship, not aliases for one it
        // does -- decoding one must fail rather than pick a neighbour.
        assert_eq!(FamilyCode::from_code(1), None);
        assert_eq!(FamilyCode::from_code(4), None);
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
    fn a_group_key_may_not_occupy_the_reserved_namespace() {
        assert!(validate_group_key("HAM-ROT").is_ok());
        assert!(validate_group_key("").is_ok());
        // A literal `__global__` group would land its parameters in the same bucket
        // as the population-level ones, so the documented filter would quietly
        // return the wrong rows.
        assert!(validate_group_key(GLOBAL_GROUP).is_err());
        assert!(validate_group_key("__anything__").is_err());
    }

    #[test]
    fn reserved_parameter_names_are_rejected() {
        assert!(validate_param_name("beta[price]").is_ok());
        assert!(validate_param_name("sigma").is_ok());
        let err = validate_param_name("__lp__").unwrap_err();
        assert!(matches!(err, BayesError::ReservedParamName(_)));
    }
}
