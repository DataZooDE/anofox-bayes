//! Inference engines.
//!
//! An engine turns a [`CompiledModel`](crate::catalog::CompiledModel) into draws. It
//! knows nothing about which family produced the model, and a family knows nothing
//! about which engine will consume it — that separation is what lets a family gain a
//! sampler, or an engine gain a family, without either being edited.
//!
//! | Engine | Posterior | Chains |
//! |---|---|---|
//! | [`ExactEngine`] | closed form, conjugate families only | 1 |
//! | Laplace (0.1, next) | Gaussian approximation at the mode | 1 |
//! | NUTS (0.2) | asymptotically exact, any family | many |
//!
//! Engines that draw independently write `chain = 0` and emit no Hamiltonian sample
//! statistics; R̂ is then undefined for their output, which is correct — there is no
//! convergence to assess when every draw is already independent.

pub mod exact;
pub mod laplace;

pub use exact::ExactEngine;
pub use laplace::LaplaceEngine;

use crate::catalog::CompiledModel;
use crate::draws::SampleStats;
use crate::errors::{BayesError, BayesResult};
use crate::types::EngineKind;

/// How much sampling to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleOptions {
    pub n_chains: usize,
    pub n_draws: usize,
    pub seed: u64,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            n_chains: 1,
            // 1000 draws give a bulk ESS comfortably above the 400 gate for an
            // independent sampler, without making a per-group fit expensive.
            n_draws: 1000,
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
        EngineKind::Nuts => Err(BayesError::config(
            "engine",
            "the NUTS engine arrives in 0.2. Until then use 'exact' (closed-form, the \
             default for both families) or 'laplace' (pooled_gaussian only)",
        )),
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

    /// An engine that is not built yet must say so. Falling back to a different
    /// engine would give the caller numbers with a different warranty than the ones
    /// they asked for, and nothing downstream could tell.
    #[test]
    fn an_unavailable_engine_is_an_error_rather_than_a_substitution() {
        let err = resolve(EngineKind::Nuts).unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "engine"));
    }

    #[test]
    fn the_default_sampling_budget_clears_the_ess_gate_for_an_independent_sampler() {
        let opts = SampleOptions::default();
        assert!(opts.n_draws as f64 > crate::diagnostics::Thresholds::default().min_ess_bulk);
        assert_eq!(opts.n_chains, 1);
    }
}
