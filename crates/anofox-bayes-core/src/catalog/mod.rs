//! The closed model catalog.
//!
//! Model families are **code, not user input** (HLD §1). A family owns its
//! parameterisation, its priors and its validation; a caller chooses among families
//! and tunes their documented slots, and cannot express a model the catalog does not
//! contain. That restriction is the product: it is what makes analytic gradients,
//! per-family calibration and a bounded correctness liability possible at all.
//!
//! The trait split is what keeps families and engines from knowing about each other:
//!
//! * [`ModelFamily`] turns a validated config plus data into a [`CompiledModel`].
//!   It knows nothing about how the posterior will be explored.
//! * [`CompiledModel`] is everything an engine needs: parameter names, a readiness
//!   verdict, and — for conjugate families — closed-form sampling via
//!   [`ExactPosterior`].
//!
//! Adding an engine touches no family. Adding a family touches no engine.

pub mod f3_pooled_gaussian;
pub mod f7_conjugate;

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::rng::BayesRng;
use crate::types::{EngineKind, FitStatus};

/// A family's structural verdict on its data, reached before any sampling.
///
/// This is the honest half of the refusal path: some inadequacies are visible from
/// the sufficient statistics alone and need no draws to detect. A lane with one
/// invoice has no estimable variance no matter how long a sampler runs on it, and
/// saying so up front is cheaper and clearer than sampling and then explaining.
#[derive(Debug, Clone, PartialEq)]
pub struct Readiness {
    pub status: FitStatus,
    /// Human-readable reasons, empty when the model is ready. Machine consumers
    /// branch on `status`; these are for the analyst reading the audit trail.
    pub reasons: Vec<String>,
}

impl Readiness {
    pub fn ready() -> Self {
        Self {
            status: FitStatus::Converged,
            reasons: Vec::new(),
        }
    }

    /// The data is usable but too weak to answer the question: the posterior would be
    /// dominated by the prior. A first-class outcome, not an error.
    pub fn insufficient(reason: impl Into<String>) -> Self {
        Self {
            status: FitStatus::InsufficientData,
            reasons: vec![reason.into()],
        }
    }

    pub fn degenerate(reason: impl Into<String>) -> Self {
        Self {
            status: FitStatus::Degenerate,
            reasons: vec![reason.into()],
        }
    }

    /// Combine per-group verdicts into one model-level verdict, keeping the worst.
    ///
    /// Worst-wins rather than majority: a fit covering 500 lanes of which three are
    /// unidentifiable is not 99.4 % trustworthy, it is a fit an agent must look at
    /// before acting on any of it.
    pub fn worst(verdicts: impl IntoIterator<Item = Readiness>) -> Self {
        let mut status = FitStatus::Converged;
        let mut reasons = Vec::new();
        for v in verdicts {
            if (v.status as i32) > (status as i32) {
                status = v.status;
            }
            reasons.extend(v.reasons);
        }
        Self { status, reasons }
    }
}

/// A model family in the catalog.
///
/// `Debug` is a supertrait rather than an afterthought: a fit that goes wrong at a
/// customer site is diagnosed from a log, and a trait object that prints as nothing
/// costs a day.
pub trait ModelFamily: Send + Sync + std::fmt::Debug {
    /// Stable identifier used in SQL and recorded in `model_id`.
    fn id(&self) -> &'static str;

    /// One-line description for the catalog listing.
    fn description(&self) -> &'static str;

    /// Engine used unless the caller overrides it.
    fn default_engine(&self) -> EngineKind;

    /// Config slots this family understands. Anything else is rejected.
    fn config_slots(&self) -> &'static [&'static str];

    /// Validate the configuration against the data and compile a fittable model.
    ///
    /// Every check that can be made without arithmetic on the observations happens
    /// here, so that by the time an engine runs there is nothing left to validate.
    fn compile<'a>(
        &self,
        cfg: &Config,
        data: &'a DataView<'a>,
    ) -> BayesResult<Box<dyn CompiledModel + 'a>>;
}

/// A family instantiated against a particular dataset and configuration.
pub trait CompiledModel: std::fmt::Debug {
    /// Parameter identities, in the order [`ExactPosterior::sample_into`] writes them.
    fn param_names(&self) -> &[ParamName];

    /// Observations that survived null filtering.
    fn n_obs(&self) -> usize;

    /// Distinct groups fitted.
    fn n_groups(&self) -> usize;

    /// Hash of the data this model reads. Feeds `model_id`.
    fn data_fingerprint(&self) -> &str;

    /// The structural verdict described on [`Readiness`].
    fn readiness(&self) -> Readiness;

    /// Closed-form sampling, when the family is conjugate.
    ///
    /// Returning `None` is how a family declines the exact engine; the engine then
    /// reports that it cannot serve this model rather than silently substituting an
    /// approximation.
    fn as_exact(&self) -> Option<&dyn ExactPosterior> {
        None
    }
}

/// A model whose posterior is available in closed form.
pub trait ExactPosterior {
    /// Draw one sample into `out`, which has one slot per parameter name.
    fn sample_into(&self, rng: &mut BayesRng, out: &mut [f64]) -> BayesResult<()>;
}

/// Look up a family by its SQL identifier.
///
/// An unknown name is a permanent error listing the catalog, not a fallback: the
/// catalog is closed, and a caller who asked for a family that does not exist must
/// not be silently served one that does.
pub fn lookup(id: &str) -> BayesResult<&'static dyn ModelFamily> {
    all()
        .iter()
        .copied()
        .find(|f| f.id().eq_ignore_ascii_case(id))
        .ok_or_else(|| BayesError::UnknownFamily {
            name: id.to_string(),
            catalog: all().iter().map(|f| f.id()).collect::<Vec<_>>().join(", "),
        })
}

/// Every family in the catalog.
pub fn all() -> &'static [&'static dyn ModelFamily] {
    const FAMILIES: &[&dyn ModelFamily] = &[
        &f3_pooled_gaussian::PooledGaussian,
        &f7_conjugate::ConjugateAnomaly,
    ];
    FAMILIES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_are_found_by_their_identifier_case_insensitively() {
        assert_eq!(
            lookup("conjugate_anomaly").unwrap().id(),
            "conjugate_anomaly"
        );
        assert_eq!(
            lookup("CONJUGATE_ANOMALY").unwrap().id(),
            "conjugate_anomaly"
        );
    }

    /// The catalog is closed by design. A caller who names a family that does not
    /// exist must be told so, and told what does exist, rather than quietly served a
    /// different model.
    #[test]
    fn an_unknown_family_lists_the_catalog_rather_than_falling_back() {
        let err = lookup("gaussian_process").unwrap_err();
        assert!(matches!(err, BayesError::UnknownFamily { .. }));
        let msg = err.to_string();
        assert!(msg.contains("gaussian_process"), "{msg}");
        assert!(msg.contains("conjugate_anomaly"), "{msg}");
    }

    #[test]
    fn every_catalog_entry_has_a_distinct_identifier() {
        let ids: Vec<&str> = all().iter().map(|f| f.id()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate family id in {ids:?}");
    }

    /// Worst-wins, not majority: a fit covering 500 lanes of which three are
    /// unidentifiable is not 99.4 % trustworthy.
    #[test]
    fn a_single_bad_group_downgrades_the_whole_fit() {
        let verdict = Readiness::worst(vec![
            Readiness::ready(),
            Readiness::ready(),
            Readiness::insufficient("lane AAA-BBB has 1 observation"),
        ]);
        assert_eq!(verdict.status, FitStatus::InsufficientData);
        assert_eq!(verdict.reasons.len(), 1);

        let verdict = Readiness::worst(vec![
            Readiness::insufficient("thin"),
            Readiness::degenerate("zero variance"),
        ]);
        // Failed > InsufficientData > Degenerate > Converged by code order, so the
        // more severe of these two wins.
        assert_eq!(verdict.status, FitStatus::InsufficientData);
        assert_eq!(verdict.reasons.len(), 2);
    }

    #[test]
    fn an_all_ready_fit_carries_no_reasons() {
        let verdict = Readiness::worst(vec![Readiness::ready(), Readiness::ready()]);
        assert_eq!(verdict.status, FitStatus::Converged);
        assert!(verdict.reasons.is_empty());
        assert!(verdict.status.is_actionable());
    }
}
