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

pub mod f1_hier_negbin;
pub mod f2_censored_aft;
pub mod f3_pooled_gaussian;
pub mod f4_payment_delay;
pub mod f5_btyd;
pub mod f6_hier_elasticity;
pub mod f7_conjugate;
pub mod f8_varying_variance;

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::rng::BayesRng;
use crate::types::{EngineKind, FamilyCode, FitStatus, SampleFrom};

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

    /// The fit could not be completed at all: there is no mode, so there is no
    /// posterior to approximate. Distinct from [`Readiness::degenerate`], where a
    /// point was found and the curvature there turned out not to be a covariance.
    ///
    /// Draws for the affected parameters are NULL-shaped rather than absent, so that
    /// the shape of a draws table does not depend on whether the fit succeeded — a
    /// consumer joins on the same columns either way and reads `__status__` to find
    /// out.
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            status: FitStatus::Failed,
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

    /// Numeric identity written into the draws table as `__family__`.
    ///
    /// Separate from [`ModelFamily::id`] only because the `value` column is `DOUBLE`;
    /// the two must name the same family, which
    /// `a_family_code_and_a_family_id_name_the_same_family` enforces.
    fn code(&self) -> FamilyCode;

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

    /// Parameters this model *reports* but does not *estimate*.
    ///
    /// A structurally-fixed parameter is one whose value the model holds constant by
    /// construction — `censored_aft`'s scale under `dist: 'exponential'`, which the
    /// distribution defines as 1. It is still published, so every variant of a family
    /// produces the same parameter set and a downstream query need not branch, but its
    /// posterior is a point and its diagnostics are meaningless: `rhat` is undefined
    /// and `ess` is zero.
    ///
    /// The gate in [`crate::fit::grade`] skips these and only these. It cannot instead
    /// notice that the draws never move, and the distinction is the whole point: a
    /// genuinely stuck sampler produces identical draws too, and `ess` returning zero
    /// for it is deliberate — it is what stops a stuck chain sailing through an
    /// `ess >= 400` gate. Inferring "fixed" from the values would hand that stuck chain
    /// the same exemption. So the family declares it, and a family that declares
    /// nothing gets no exemption.
    ///
    /// Names are matched against [`crate::draws::ParamName::name`], across every group.
    fn fixed_params(&self) -> &[&'static str] {
        &[]
    }

    /// How many of [`CompiledModel::n_groups`] groups did **not** reach
    /// `FitStatus::Converged` on their own.
    ///
    /// [`Readiness::worst`] collapses per-group verdicts into one and continues to do
    /// so: a fit covering 500 lanes of which three are unidentifiable is not 99.4 %
    /// trustworthy, it is a fit an agent must look at. What the collapse throws away
    /// is the *scale* of that inspection — three lanes and four hundred lanes reach
    /// SQL as the same `insufficient_data`. This is the missing number, reported
    /// beside the verdict rather than in place of it.
    ///
    /// The default is the honest answer for a family whose verdict is reached over
    /// the whole design rather than per group: such a family cannot single out a
    /// subset of its groups, so a downgrade implicates all of them. Over-counting is
    /// also the safe direction — it sends an agent to look at more than it must,
    /// never at less.
    fn n_groups_unready(&self) -> usize {
        if self.readiness().status.is_actionable() {
            0
        } else {
            self.n_groups()
        }
    }

    /// Which groups were refused, and with what verdict.
    ///
    /// `n_groups_unready` says how many; this says which, so an agent holding a
    /// 5 000-lane fit can quarantine the three bad lanes rather than the whole table.
    /// The model-level status stays the collapsed worst case — that doctrine is
    /// deliberate and unchanged — and this is the detail it collapses.
    ///
    /// Defaults to empty rather than to "all of them". A family that fits one joint
    /// design cannot honestly single out a group, and naming every group as unready
    /// would be a list an agent could act on wrongly; `n_groups_unready` already tells
    /// it the refusal implicates everything.
    fn unready_groups(&self) -> Vec<(String, crate::types::FitStatus)> {
        Vec::new()
    }

    /// Closed-form sampling, when the family is conjugate.
    ///
    /// Returning `None` is how a family declines the exact engine; the engine then
    /// reports that it cannot serve this model rather than silently substituting an
    /// approximation.
    fn as_exact(&self) -> Option<&dyn ExactPosterior> {
        None
    }

    /// Gradient-based access, for the Laplace and NUTS engines.
    fn as_differentiable(&self) -> Option<&dyn LogPosterior> {
        None
    }

    /// A Gaussian approximation the family already holds, for a fit performed
    /// elsewhere.
    ///
    /// Returning `Some` says: the mode and the curvature at it were computed by
    /// something that is not this crate's Newton search — today, `anofox-stats-core`
    /// through [`crate::bridge`] — so there is nothing left for the Laplace engine to
    /// find, only draws to generate. See [`GaussianApproximation`].
    fn as_gaussian(&self) -> Option<&dyn GaussianApproximation> {
        None
    }
}

/// One independent Gaussian block of a posterior.
///
/// A block is a set of parameters fitted jointly and independently of every other
/// block — one per group for a per-group model, exactly one for a pooled one. Blocks
/// exist rather than one big matrix because a per-group fit is genuinely block
/// diagonal, and because a group that could not be fitted is simply absent from the
/// list instead of contributing a row of zeros that would have to be special-cased in
/// the factorisation.
#[derive(Debug, Clone)]
pub struct GaussianBlock {
    /// The mode, on the unconstrained scale.
    pub mode: Vec<f64>,
    /// The observed information at the mode — the **full** symmetric positive-definite
    /// matrix, never its diagonal.
    ///
    /// The precision rather than the covariance because that is what a Cholesky
    /// factor of it is directly useful for: `theta = mode + L^-T z` needs a factor of
    /// the precision, so forming the covariance and factoring *that* would cost an
    /// inversion and lose conditioning for nothing.
    ///
    /// **Why this is a matrix and not a vector.** Its off-diagonal is the correlation
    /// between coefficients, and in a regression with a covariate measured away from
    /// zero that correlation dominates the width of any predictive interval. A
    /// diagonal here would produce draws that are individually plausible, pass every
    /// diagnostic in this crate, and imply intervals wrong by an order of magnitude.
    pub precision: faer::Mat<f64>,
    /// Which entries of [`CompiledModel::param_names`] this block writes, in the order
    /// [`GaussianApproximation::constrain`] fills them.
    pub params: Vec<usize>,
}

/// A model that arrives already fitted: a mode and the full curvature at it.
///
/// This is the seam onto a fit performed outside this crate. A MAP estimate plus its
/// observed information *is* a Laplace posterior, so a family that can produce those
/// two objects needs nothing else from an engine except the last step — sample the
/// multivariate normal and back-transform.
///
/// It is deliberately separate from [`LogPosterior`]. A family exposing a log density
/// and a gradient is asking the engine to *find* the mode; a family implementing this
/// trait already knows it, and the engine must not go looking for a different one.
/// Keeping the two apart is also what keeps the warranty honest: a bridged posterior
/// is `EngineKind::Laplace` — a Gaussian approximation — regardless of who computed
/// the curvature, and it carries the same obligation to an SBC suite.
pub trait GaussianApproximation {
    /// The independent blocks of the posterior. Parameters covered by no block are
    /// reported as NULL, which is how a refused group travels.
    fn blocks(&self) -> &[GaussianBlock];

    /// Map one block's unconstrained draw onto the parameters it reports.
    ///
    /// `theta` has `blocks()[block].mode.len()` entries; `out` has
    /// `blocks()[block].params.len()`. The two need not be the same length — a family
    /// may report a parameter it does not sample over, or sample a coordinate it does
    /// not report.
    fn constrain(&self, block: usize, theta: &[f64], out: &mut [f64]);
}

/// A model whose posterior is available in closed form.
pub trait ExactPosterior {
    /// Draw one sample into `out`, which has one slot per parameter name.
    fn sample_into(&self, rng: &mut BayesRng, out: &mut [f64]) -> BayesResult<()>;

    /// Draw one sample from the **prior**, for a prior-predictive check (BR-11).
    ///
    /// Defaults to a refusal rather than to the posterior. A family that has not
    /// implemented this must say so: silently returning posterior draws under a
    /// `sample_from: 'prior'` request would make the pre-fit gate agree with the data
    /// it is supposed to be checked against, which is the one thing it must never do.
    fn sample_prior_into(&self, _rng: &mut BayesRng, _out: &mut [f64]) -> BayesResult<()> {
        Err(BayesError::config(
            "sample_from",
            "this family cannot draw from its prior",
        ))
    }

    /// Fill one whole chain: `out` is `n_draws` blocks of `param_names().len()`
    /// values, in the layout [`Posterior`](crate::draws::Posterior) expects.
    ///
    /// The default runs the chain sequentially from a single stream derived from
    /// `(seed, chain)`, which is exactly what the engine did before this method
    /// existed — a family that does not override it is unaffected.
    ///
    /// A family whose groups are independent overrides it to sample them in
    /// parallel. The override is a family's responsibility rather than the engine's
    /// because only the family knows which slots of `out` belong to which group, and
    /// therefore which randomness may be split without changing the answer. Any
    /// override owes the same guarantee the default gives for free: the numbers must
    /// not depend on the thread count or on the order the groups are visited in.
    fn sample_chain_into(
        &self,
        seed: u64,
        chain: u32,
        n_draws: usize,
        sample_from: SampleFrom,
        out: &mut [f64],
    ) -> BayesResult<()> {
        if n_draws == 0 || out.is_empty() {
            return Ok(());
        }
        if !out.len().is_multiple_of(n_draws) {
            return Err(BayesError::DimensionMismatch(format!(
                "{} slots do not divide into {n_draws} draws",
                out.len()
            )));
        }
        let n_params = out.len() / n_draws;
        let mut rng = BayesRng::for_chain(seed, chain);
        for draw in 0..n_draws {
            let slots = &mut out[draw * n_params..(draw + 1) * n_params];
            match sample_from {
                SampleFrom::Posterior => self.sample_into(&mut rng, slots)?,
                SampleFrom::Prior => self.sample_prior_into(&mut rng, slots)?,
            }
        }
        Ok(())
    }
}

/// A model whose log posterior and its gradient are available on an unconstrained
/// scale.
///
/// This is what the gradient-based engines consume: the Laplace engine needs it to
/// find the mode and the curvature there, and the NUTS adapter (0.2) will need
/// nothing else. Working on an unconstrained scale is what makes a positive parameter
/// like `sigma` tractable — a Gaussian approximation to a quantity that must stay
/// positive is wrong near zero, while a Gaussian approximation to its logarithm is
/// not.
///
/// Gradients are **analytic and hand-derived**, per family, and every implementation
/// is unit-tested against finite differences. There is no autodiff dependency: the
/// catalog is closed, so the derivatives are written once and checked once, and the
/// check is cheaper and more auditable than a general mechanism would be.
///
/// # `Sync`, and why it is a supertrait
///
/// The NUTS engine runs one chain per rayon task and every task reads this same
/// posterior, so it has to be shareable across threads. That is not a new constraint
/// discovered late: a log density is a *pure function* of its coordinates — every
/// implementation in this catalog is a struct of data read immutably and none of them
/// has ever had interior mutability — so the bound records a property the families
/// already had rather than imposing one on them.
///
/// It is also the property that makes parallel chains admissible at all. A posterior
/// that mutated per evaluation would make a chain's draws depend on what other chains
/// were doing, and `nuts.rs`'s reproducibility contract would be unenforceable rather
/// than merely broken.
pub trait LogPosterior: Sync {
    /// Number of unconstrained coordinates.
    fn dim(&self) -> usize;

    /// Log posterior density at `theta`, up to an additive constant, including the
    /// log-Jacobian of the constraining transform.
    fn logp(&self, theta: &[f64]) -> f64;

    /// Gradient of [`LogPosterior::logp`] at `theta`, written into `out`.
    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()>;

    /// A starting point for the mode search. Families return something already close,
    /// since they generally know their own answer.
    fn initial(&self) -> Vec<f64>;

    /// The Hamiltonian acceptance rate a Markov sampler should adapt its step size to
    /// for **this** posterior.
    ///
    /// A family declares it; a caller cannot, which is the same rule as for every other
    /// parameterisation decision (HLD §3.2). The default is `nuts-rs`'s own 0.8, which
    /// is right for the GLM-shaped posteriors the conjugate families produce, and every
    /// family that does not override this is bit-for-bit unaffected.
    ///
    /// It exists because a hierarchical posterior needs a smaller step than a Gaussian
    /// one for the same acceptance behaviour: the curvature varies sharply along the
    /// variance components, and a step tuned to the bulk overshoots in the tail and is
    /// reported as a divergence. Stan's `adapt_delta` is the same dial and is raised for
    /// the same models and the same reason. Raising it costs leapfrog steps, not
    /// correctness -- the target distribution is untouched.
    fn target_accept(&self) -> f64 {
        0.8
    }

    /// Map unconstrained coordinates to the parameters the draws table reports.
    ///
    /// `out` has one slot per [`CompiledModel::param_names`] entry, which need not
    /// match [`LogPosterior::dim`] — a family may report a parameter it does not
    /// sample over.
    fn constrain(&self, theta: &[f64], out: &mut [f64]);
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
        &f1_hier_negbin::HierNegbin,
        &f2_censored_aft::CensoredAft,
        &f3_pooled_gaussian::PooledGaussian,
        &f4_payment_delay::PaymentDelay,
        &f5_btyd::PayerAlive,
        &f6_hier_elasticity::HierElasticity,
        &f7_conjugate::ConjugateAnomaly,
        &f8_varying_variance::VaryingVarianceGaussian,
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

    /// A family has one identity wearing two hats — a name for SQL and a number for
    /// the `DOUBLE` value column of the draws table. If the two ever disagreed, a
    /// persisted table would name a different model than the one that wrote it.
    #[test]
    fn a_family_code_and_a_family_id_name_the_same_family() {
        for family in all() {
            assert_eq!(
                family.code().as_str(),
                family.id(),
                "family '{}' carries code {:?}, which names '{}'",
                family.id(),
                family.code(),
                family.code().as_str()
            );
            assert_eq!(
                FamilyCode::from_code(family.code() as i32),
                Some(family.code()),
                "the code of '{}' does not decode",
                family.id()
            );
        }
    }

    /// Two families sharing a code would make a persisted draws table ambiguous about
    /// which model produced it, which is the one question `__family__` exists to
    /// answer.
    #[test]
    fn every_catalog_entry_has_a_distinct_family_code() {
        let mut codes: Vec<i32> = all().iter().map(|f| f.code() as i32).collect();
        let n = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), n, "duplicate family code in the catalog");
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
