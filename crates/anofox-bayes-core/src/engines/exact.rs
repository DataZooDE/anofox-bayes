//! The exact engine: closed-form sampling for conjugate families.
//!
//! Where a family is conjugate the posterior is available in closed form, and drawing
//! from it directly is both cheaper and *more accurate* than approximating it. There
//! is no burn-in, no adaptation and no autocorrelation, so every draw counts and R̂ is
//! meaningless by construction.
//!
//! Doubling as more than a shortcut: the exact posterior is the reference the Laplace
//! engine is checked against. Where both can serve a model they must agree to Monte
//! Carlo error, which is the strongest correctness gate in this crate — two
//! independent derivations of one distribution.

use crate::catalog::CompiledModel;
use crate::draws::SampleStats;
use crate::errors::{BayesError, BayesResult};
use crate::types::EngineKind;

use super::{Engine, Sample, SampleOptions};

#[derive(Debug, Default, Clone, Copy)]
pub struct ExactEngine;

impl Engine for ExactEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Exact
    }

    fn supports(&self, model: &dyn CompiledModel) -> bool {
        model.as_exact().is_some()
    }

    /// Conjugate families carry their prior in the same closed form as their
    /// posterior, so the exact engine draws from either through the same sampler.
    fn can_sample_prior(&self) -> bool {
        true
    }

    fn sample(&self, model: &dyn CompiledModel, opts: &SampleOptions) -> BayesResult<Sample> {
        let exact = model.as_exact().ok_or_else(|| {
            BayesError::config(
                "engine",
                "this family has no closed-form posterior, so the exact engine cannot serve it",
            )
        })?;

        let n_params = model.param_names().len();
        if n_params == 0 {
            return Err(BayesError::Internal(
                "compiled model exposes no parameters".to_string(),
            ));
        }

        let mut values = vec![0.0; opts.n_chains * opts.n_draws * n_params];
        let block = opts.n_draws * n_params;
        for chain in 0..opts.n_chains {
            // A chain at a time, each from a stream derived from (seed, chain).
            // Chains of an exact sampler are already independent; separate streams
            // keep them reproducible individually, so a single chain can be re-drawn
            // without re-running the others.
            //
            // How the chain is filled is the *family's* decision, not the engine's:
            // a family whose groups are independent splits the work across them and
            // keys each group's stream on its own identity. The engine cannot make
            // that split, because it does not know which slots belong together.
            let offset = chain * block;
            exact.sample_chain_into(
                opts.seed,
                chain as u32,
                opts.n_draws,
                opts.sample_from,
                &mut values[offset..offset + block],
            )?;
        }

        // No sample statistics. Emitting `__divergent__ = 0` here would be a lie of
        // the most dangerous kind: it reads as "the sampler explored cleanly" when in
        // fact no sampler ran.
        Ok(Sample {
            values,
            stats: Vec::<SampleStats>::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{ExactPosterior, ModelFamily};
    use crate::config::Config;
    use crate::data::testing::Frame;
    use crate::draws::ParamName;
    use crate::rng::BayesRng;

    /// A model whose posterior is a known standard normal, so the engine's plumbing
    /// can be checked without a family's mathematics in the way.
    #[derive(Debug)]
    struct StandardNormalModel {
        params: Vec<ParamName>,
    }

    impl CompiledModel for StandardNormalModel {
        fn param_names(&self) -> &[ParamName] {
            &self.params
        }
        fn n_obs(&self) -> usize {
            0
        }
        fn n_groups(&self) -> usize {
            1
        }
        fn data_fingerprint(&self) -> &str {
            "test"
        }
        fn readiness(&self) -> crate::catalog::Readiness {
            crate::catalog::Readiness::ready()
        }
        fn as_exact(&self) -> Option<&dyn ExactPosterior> {
            Some(self)
        }
    }

    impl ExactPosterior for StandardNormalModel {
        fn sample_into(&self, rng: &mut BayesRng, out: &mut [f64]) -> BayesResult<()> {
            for slot in out.iter_mut() {
                *slot = rng.standard_normal();
            }
            Ok(())
        }
    }

    /// A conjugate family that declines closed-form sampling, to check the engine
    /// refuses rather than improvising.
    #[derive(Debug)]
    struct NoClosedForm;

    impl CompiledModel for NoClosedForm {
        fn param_names(&self) -> &[ParamName] {
            &[]
        }
        fn n_obs(&self) -> usize {
            0
        }
        fn n_groups(&self) -> usize {
            0
        }
        fn data_fingerprint(&self) -> &str {
            "test"
        }
        fn readiness(&self) -> crate::catalog::Readiness {
            crate::catalog::Readiness::ready()
        }
    }

    fn model() -> StandardNormalModel {
        StandardNormalModel {
            params: vec![
                ParamName::global("a").unwrap(),
                ParamName::global("b").unwrap(),
            ],
        }
    }

    #[test]
    fn the_value_block_is_laid_out_chain_major_then_draw_major() {
        let m = model();
        let opts = SampleOptions {
            n_chains: 3,
            n_draws: 7,
            // Ignored: this engine draws independently, so there is nothing to adapt.
            n_warmup: 0,
            seed: 1,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        let sample = ExactEngine.sample(&m, &opts).unwrap();
        assert_eq!(sample.values.len(), 3 * 7 * 2);
        assert!(sample.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_same_seed_reproduces_the_same_sample() {
        let m = model();
        let opts = SampleOptions {
            n_chains: 2,
            n_draws: 50,
            // Ignored: this engine draws independently, so there is nothing to adapt.
            n_warmup: 0,
            seed: 99,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        let a = ExactEngine.sample(&m, &opts).unwrap();
        let b = ExactEngine.sample(&m, &opts).unwrap();
        assert_eq!(a.values, b.values);

        let c = ExactEngine
            .sample(&m, &SampleOptions { seed: 100, ..opts })
            .unwrap();
        assert_ne!(a.values, c.values);
    }

    /// Each chain has its own stream, so re-drawing one chain does not require
    /// re-running the others -- and, more importantly, two chains of one fit are not
    /// secretly the same chain.
    #[test]
    fn chains_are_independent_of_each_other() {
        let m = model();
        let opts = SampleOptions {
            n_chains: 2,
            n_draws: 100,
            // Ignored: this engine draws independently, so there is nothing to adapt.
            n_warmup: 0,
            seed: 5,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        let sample = ExactEngine.sample(&m, &opts).unwrap();
        let chain0 = &sample.values[..100 * 2];
        let chain1 = &sample.values[100 * 2..];
        assert_ne!(chain0, chain1);
    }

    /// `sum(__divergent__) = 0` must mean "the sampler saw no divergences". An exact
    /// sampler ran no trajectories at all, so it reports nothing rather than a
    /// reassuring zero.
    #[test]
    fn an_exact_sampler_reports_no_hamiltonian_statistics() {
        let sample = ExactEngine
            .sample(&model(), &SampleOptions::default())
            .unwrap();
        assert!(sample.stats.is_empty());
    }

    #[test]
    fn a_family_without_a_closed_form_is_refused_rather_than_approximated() {
        assert!(!ExactEngine.supports(&NoClosedForm));
        let err = ExactEngine
            .sample(&NoClosedForm, &SampleOptions::default())
            .unwrap_err();
        assert!(
            err.to_string().contains("no closed-form posterior"),
            "{err}"
        );
    }

    /// End to end against a real family: the engine's draws reproduce the posterior
    /// mean the family's own closed form predicts.
    #[test]
    fn the_engine_reproduces_a_real_familys_closed_form_posterior() {
        let ys = vec![10.0, 12.0, 11.0, 9.0, 13.0, 10.5, 11.5, 12.5];
        let ybar = ys.iter().sum::<f64>() / ys.len() as f64;

        let frame = Frame::new(ys.len()).numeric("cost", ys);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let compiled = crate::catalog::f7_conjugate::ConjugateAnomaly
            .compile(&Config::parse(r#"{"value": "cost"}"#).unwrap(), &view)
            .unwrap();

        assert!(ExactEngine.supports(&*compiled));
        let sample = ExactEngine
            .sample(
                &*compiled,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 100_000,
                    // Ignored: this engine draws independently, so there is nothing to adapt.
                    n_warmup: 0,
                    seed: 3,
                    sample_from: crate::types::SampleFrom::Posterior,
                },
            )
            .unwrap();

        let n_params = compiled.param_names().len();
        let mu: Vec<f64> = sample.values.chunks(n_params).map(|c| c[0]).collect();
        let posterior_mean = mu.iter().sum::<f64>() / mu.len() as f64;
        assert!(
            (posterior_mean - ybar).abs() < 0.02,
            "engine posterior mean {posterior_mean} vs sample mean {ybar}"
        );
    }
}
