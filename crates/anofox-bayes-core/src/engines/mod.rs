//! Inference engines.
//!
//! An engine turns a [`CompiledModel`](crate::catalog::CompiledModel) into draws. It
//! knows nothing about which family produced the model, and a family knows nothing
//! about which engine will consume it — that separation is what lets a family gain a
//! sampler, or an engine gain a family, without either being edited.
//!
//! | Engine | Posterior | Chains | Sample statistics |
//! |---|---|---|---|
//! | [`ExactEngine`] | closed form, conjugate families only | 1 | none |
//! | [`LaplaceEngine`] | Gaussian approximation at the mode | 1 | none |
//! | [`NutsEngine`] | asymptotically exact, any differentiable family | many | `__lp__`, `__divergent__`, `__energy__`, `__step_size__` |
//!
//! Engines that draw independently write `chain = 0` and emit no Hamiltonian sample
//! statistics; R̂ is then undefined for their output, which is correct — there is no
//! convergence to assess when every draw is already independent. NUTS is the first
//! engine here that produces a genuine Markov chain, so it is the first for which R̂
//! and the divergence count mean anything at all.

pub mod exact;
pub mod laplace;
pub mod nuts;

pub use exact::ExactEngine;
pub use laplace::LaplaceEngine;
pub use nuts::NutsEngine;

use crate::catalog::CompiledModel;
use crate::draws::SampleStats;
use crate::errors::BayesResult;
use crate::types::EngineKind;

/// Adaptation draws a Markov sampler takes before the draws it keeps.
///
/// Stan's and PyMC's default, and `nuts-rs`'s own, and there is no reason to differ:
/// it is long enough to fit a diagonal mass matrix and a step size on the models this
/// catalog contains, and short enough that a per-group fit stays interactive. It is a
/// config slot (`warmup`) rather than a constant because a badly conditioned posterior
/// legitimately needs more, and because a caller who has to raise it has learned
/// something about their model that they should be allowed to act on.
pub const DEFAULT_WARMUP: usize = 1000;

/// How much sampling to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleOptions {
    pub n_chains: usize,
    pub n_draws: usize,
    /// Adaptation draws, discarded before the output. Ignored by engines that draw
    /// independently — there is nothing to adapt when every draw is already exact.
    pub n_warmup: usize,
    pub seed: u64,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            n_chains: 1,
            // 1000 draws give a bulk ESS comfortably above the 400 gate for an
            // independent sampler, without making a per-group fit expensive.
            n_draws: 1000,
            n_warmup: DEFAULT_WARMUP,
            seed: crate::config::DEFAULT_SEED,
        }
    }
}

/// Raw sampler output, before it becomes a [`crate::draws::Posterior`].
#[derive(Debug)]
pub struct Sample {
    /// Chain-major, draw-major, parameter-minor — the layout `Posterior` expects.
    pub values: Vec<f64>,
    /// Empty when the engine reports no per-draw statistics.
    pub stats: Vec<SampleStats>,
}

pub trait Engine: std::fmt::Debug {
    fn kind(&self) -> EngineKind;

    /// Whether this engine can serve this model at all.
    ///
    /// Checked before sampling so that an unsupported combination is a clear error
    /// rather than a quiet substitution. An agent that asked for an exact posterior
    /// and received an approximation would report unearned confidence.
    fn supports(&self, model: &dyn CompiledModel) -> bool;

    fn sample(&self, model: &dyn CompiledModel, opts: &SampleOptions) -> BayesResult<Sample>;
}

/// Pick the engine for a fit: the family's default unless the caller overrode it.
pub fn resolve(kind: EngineKind) -> BayesResult<Box<dyn Engine>> {
    match kind {
        EngineKind::Exact => Ok(Box::new(ExactEngine)),
        EngineKind::Laplace => Ok(Box::new(LaplaceEngine)),
        EngineKind::Nuts => Ok(Box::new(NutsEngine)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exact_engine_is_available_and_reports_its_kind() {
        let engine = resolve(EngineKind::Exact).unwrap();
        assert_eq!(engine.kind(), EngineKind::Exact);
    }

    #[test]
    fn the_laplace_engine_is_available_and_reports_its_kind() {
        assert_eq!(
            resolve(EngineKind::Laplace).unwrap().kind(),
            EngineKind::Laplace
        );
    }

    #[test]
    fn the_nuts_engine_is_available_and_reports_its_kind() {
        assert_eq!(resolve(EngineKind::Nuts).unwrap().kind(), EngineKind::Nuts);
    }

    /// Every engine in the catalog resolves. The refusal path for an engine that
    /// cannot serve a *family* still exists and is exercised by
    /// `fit::tests::an_engine_that_cannot_serve_the_family_is_an_error`; what is gone
    /// is the refusal for an engine that had not been written yet.
    #[test]
    fn every_engine_kind_resolves_to_an_engine_that_reports_the_same_kind() {
        for kind in [EngineKind::Exact, EngineKind::Laplace, EngineKind::Nuts] {
            assert_eq!(resolve(kind).unwrap().kind(), kind);
        }
    }

    #[test]
    fn the_default_sampling_budget_clears_the_ess_gate_for_an_independent_sampler() {
        let opts = SampleOptions::default();
        assert!(opts.n_draws as f64 > crate::diagnostics::Thresholds::default().min_ess_bulk);
        assert_eq!(opts.n_chains, 1);
    }
}
