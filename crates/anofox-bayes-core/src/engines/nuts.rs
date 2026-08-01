//! The NUTS engine: the No-U-Turn Sampler, as a thin adapter over `nuts-rs`.
//!
//! Where the exact engine samples a formula and the Laplace engine samples a Gaussian
//! fitted to the mode, this one explores the log posterior itself with Hamiltonian
//! dynamics. It is slower than both by orders of magnitude and it is the only engine
//! here that is asymptotically exact for a posterior with no closed form — which is
//! the whole reason it exists. `ROADMAP.md` gap 6 sequences it ahead of three model
//! families because each of them bottoms out in a hierarchical variance parameter
//! whose posterior is not Gaussian on any scale, and Laplace is known to be poor
//! exactly there.
//!
//! ## Everything below this file is somebody else's code, on purpose
//!
//! `ROADMAP.md` §5 records "writing an MCMC kernel" as a non-goal. `nuts-rs` is
//! maintained by pymc-devs and is the sampler behind nutpie, so the multinomial
//! trajectory sampling, the U-turn criterion, the dual-averaging step size and the
//! diagonal mass matrix adaptation are all theirs and none of it is reimplemented
//! here. This module is the seam and nothing else: it maps
//! [`LogPosterior`] onto `nuts_rs::CpuLogpFunc`, drives one
//! `nuts_rs::Chain` per chain, discards warmup, and translates the per-draw
//! diagnostics into [`crate::draws::SampleStats`]. Keeping all of that in one file is
//! what makes an upstream version bump a review of one file.
//!
//! ## Determinism
//!
//! Contractual, and structural rather than hopeful.
//!
//! Each chain is seeded from [`BayesRng::for_chain`] — the same BLAKE3 derivation the
//! other engines use — and `nuts-rs` seeds its internal ChaCha8 from the bytes we
//! feed it. Chains run **sequentially**, one `nuts_rs::Chain` at a time, and this
//! engine starts no threads. So the draws of chain *c* depend on `(seed, c)` and on
//! nothing else: not on how many chains were asked for, not on which order they ran
//! in, and not on how many threads DuckDB happens to have in flight. That is why
//! `nuts-rs`'s `parallel` feature is switched off in `Cargo.toml` rather than merely
//! unused.
//!
//! Chains being sequential is also the reason there is no `rayon` here. The workspace
//! already refuses one — DuckDB owns parallelism and a nested thread pool inside a
//! table function is a liability — and the natural unit of parallelism for this
//! extension is groups and fits, not the four chains of one small fit.
//!
//! ## What the sampler statistics mean
//!
//! NUTS is the first engine to populate the reserved statistic rows the draws contract
//! has always held open. All four come from the sampler rather than being recomputed:
//! `__lp__` and `__energy__` are the potential and the total Hamiltonian energy at the
//! accepted point, `__step_size__` is the adapted step size, and `__divergent__` flags
//! a trajectory that left the level set it was integrating along. Divergences are
//! counted over kept draws only — a divergence during warmup is adaptation working,
//! not a defect — and one of them is enough for [`crate::fit::fit`] to refuse the fit.

use std::collections::HashMap;
use std::convert::Infallible;

use nuts_rs::{
    rand::TryRng, Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError,
    Settings,
};

use crate::catalog::{CompiledModel, LogPosterior};
use crate::draws::SampleStats;
use crate::errors::{BayesError, BayesResult};
use crate::linalg::{cholesky, sample_mvn};
use crate::rng::BayesRng;
use crate::types::EngineKind;

use super::{Engine, Sample, SampleOptions};

/// How far the chains are spread around the mode, in posterior standard deviations.
///
/// R̂ only detects a failure to mix if the chains had somewhere to fail to mix *from*.
/// Chains all started at the mode agree with each other by construction and would make
/// the statistic a formality. Two posterior standard deviations is the classical
/// "overdispersed relative to the target" prescription: far enough apart that a sampler
/// stuck in one region is visible, close enough that warmup is not spent travelling.
const OVERDISPERSION: f64 = 2.0;

/// Attempts at a usable starting point before giving up.
///
/// `nuts-rs` rejects an initial point whose gradient is exactly zero — which the mode
/// is, and which is exactly what a family's `initial()` returns. A jittered point has
/// probability zero of landing there again, so one retry would do; a handful costs
/// nothing and turns a pathological coincidence into a slower fit rather than a failed
/// one.
const MAX_START_ATTEMPTS: usize = 8;

#[derive(Debug, Default, Clone, Copy)]
pub struct NutsEngine;

impl Engine for NutsEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Nuts
    }

    fn supports(&self, model: &dyn CompiledModel) -> bool {
        model.as_differentiable().is_some()
    }

    fn sample(&self, model: &dyn CompiledModel, opts: &SampleOptions) -> BayesResult<Sample> {
        let target = model.as_differentiable().ok_or_else(|| {
            BayesError::config(
                "engine",
                "this family does not expose a differentiable log posterior, \
                 so the NUTS engine cannot serve it",
            )
        })?;

        let n_params = model.param_names().len();
        let dim = target.dim();
        let mut values = vec![0.0; opts.n_chains * opts.n_draws * n_params];
        let mut stats = Vec::with_capacity(opts.n_chains * opts.n_draws);

        for chain in 0..opts.n_chains {
            // One stream per chain, derived exactly as every other engine derives it.
            let mut rng = BayesRng::for_chain(opts.seed, chain as u32);
            let mut settings = DiagNutsSettings {
                num_tune: opts.n_warmup as u64,
                num_draws: opts.n_draws as u64,
                num_chains: 1,
                ..Default::default()
            };
            // `nuts-rs` uses this only in its own multi-chain driver, which this engine
            // does not use. It is set anyway so that a reader of the settings is not
            // left wondering which seed is in force.
            settings.seed = opts.seed;

            let mut sampler = None;
            let mut last_error = None;
            for _ in 0..MAX_START_ATTEMPTS {
                let start = overdispersed_start(target, &mut rng)?;
                let math = CpuMath::new(Target { target, dim });
                let mut candidate =
                    settings.new_chain(chain as u64, math, &mut RngBridge(&mut rng));
                match candidate.set_position(&start) {
                    Ok(()) => {
                        sampler = Some(candidate);
                        break;
                    }
                    // The one documented reason this fails is a zero or non-finite
                    // gradient at the starting point. Re-jittering is the fix; the
                    // error is kept so that a genuinely unusable posterior reports
                    // what the sampler actually said rather than an attempt count.
                    Err(e) => last_error = Some(e.to_string()),
                }
            }
            let mut sampler = sampler.ok_or_else(|| {
                BayesError::Internal(format!(
                    "NUTS could not find a usable starting point for chain {chain} after \
                     {MAX_START_ATTEMPTS} attempts: {}",
                    last_error.unwrap_or_else(|| "no reason reported".into())
                ))
            })?;

            // Adaptation. These draws are from a sampler that is still changing, so
            // they are not draws from the posterior and never reach the output.
            for _ in 0..opts.n_warmup {
                sampler
                    .draw()
                    .map_err(|e| BayesError::Internal(format!("NUTS warmup failed: {e}")))?;
            }

            for draw in 0..opts.n_draws {
                let (position, _expanded, sampler_stats, progress) = sampler
                    .expanded_draw()
                    .map_err(|e| BayesError::Internal(format!("NUTS draw failed: {e}")))?;

                let offset = (chain * opts.n_draws + draw) * n_params;
                target.constrain(&position, &mut values[offset..offset + n_params]);
                stats.push(SampleStats {
                    lp: Some(sampler_stats.point.logp),
                    divergent: Some(if progress.diverging { 1.0 } else { 0.0 }),
                    energy: Some(sampler_stats.point.energy),
                    step_size: Some(progress.step_size),
                });
            }
        }

        Ok(Sample { values, stats })
    }
}

/// A starting point drawn from the Laplace approximation at the family's own initial
/// guess, widened by [`OVERDISPERSION`].
///
/// Using the curvature rather than a fixed jitter is what makes the spread mean the
/// same thing for every parameter. A coefficient measured in millions and a log-scale
/// parameter of order one need wildly different absolute jitters to be equally
/// dispersed, and only the posterior itself knows the ratio. Where the curvature is
/// unusable — a flat direction, an initial point far from any mode — the fallback is a
/// jitter scaled to the coordinate's own magnitude, which is crude but never worse than
/// starting every chain in the same place.
fn overdispersed_start(target: &dyn LogPosterior, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
    let centre = target.initial();
    let dim = target.dim();
    if centre.len() != dim {
        return Err(BayesError::Internal(format!(
            "family's initial point has {} coordinates, expected {dim}",
            centre.len()
        )));
    }

    let mut out = vec![0.0; dim];
    let curvature = super::laplace::negative_hessian(target, &centre)
        .and_then(|h| cholesky(&h))
        .ok();
    match curvature {
        Some(chol) => sample_mvn(&chol, &centre, OVERDISPERSION, rng, &mut out)?,
        None => {
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = centre[j] + 0.1 * centre[j].abs().max(1.0) * rng.standard_normal();
            }
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(BayesError::Internal(
            "the starting point for a NUTS chain is not finite".to_string(),
        ));
    }
    Ok(out)
}

/// `nuts-rs`'s view of a [`LogPosterior`].
///
/// The whole adapter, in one impl. `nuts-rs` wants a single call that returns the
/// density and writes the gradient; this crate's families expose the two separately,
/// both analytic and both checked against finite differences in their own module.
struct Target<'a> {
    target: &'a dyn LogPosterior,
    dim: usize,
}

impl std::fmt::Debug for Target<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Target").field("dim", &self.dim).finish()
    }
}

/// Failure of the gradient, which for a catalog family means a dimension mismatch and
/// therefore a bug rather than a numerical excursion.
///
/// Deliberately **not** recoverable: a recoverable error tells `nuts-rs` to treat the
/// step as a divergence and carry on, which would turn a wiring bug into a fit that
/// merely looks badly behaved. A non-finite *density* is a different matter and is not
/// an error here — it is returned as the infinity it is, and `nuts-rs` rejects the step
/// and reports the divergence, which is the honest description of what happened.
#[derive(Debug, thiserror::Error)]
#[error("the log posterior gradient could not be evaluated: {0}")]
struct GradientFailure(String);

impl LogpError for GradientFailure {
    fn is_recoverable(&self) -> bool {
        false
    }
}

impl HasDims for Target<'_> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [("unconstrained_parameter".to_string(), self.dim as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for Target<'_> {
    type LogpError = GradientFailure;
    type ExpandedVector = Vec<f64>;
    /// Unused: this engine runs `nuts-rs`'s Euclidean adaptation, not its normalising
    /// flow variant, so there are no flow parameters to carry.
    type FlowParameters = ();

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, GradientFailure> {
        self.target
            .grad(position, gradient)
            .map_err(|e| GradientFailure(e.to_string()))?;
        Ok(self.target.logp(position))
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        // The expanded vector is `nuts-rs`'s hook for deriving reported quantities from
        // the unconstrained draw. This engine constrains the position itself, into the
        // caller's buffer, so there is nothing to add here.
        Ok(position.to_vec())
    }
}

/// Drives `nuts-rs` from a [`BayesRng`].
///
/// `nuts-rs` depends on a different major version of `rand` than this crate does, so
/// the two `Rng` traits are unrelated types and one of them has to be bridged. Bridging
/// here rather than seeding `nuts-rs` from a fresh RNG of its own is what keeps the
/// chain-seed derivation — and the guarantee that neighbouring seeds give unrelated
/// streams — in `rng.rs`, where it is tested.
struct RngBridge<'a>(&'a mut BayesRng);

impl TryRng for RngBridge<'_> {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.0.next_u32())
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.0.next_u64())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        self.0.fill_bytes(dst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{f3_pooled_gaussian::PooledGaussian, ModelFamily};
    use crate::config::Config;
    use crate::data::testing::Frame;
    use crate::diagnostics::ess_bulk;
    use crate::draws::SampleStats;
    use crate::engines::ExactEngine;

    fn frame(n: usize) -> Frame {
        let x1: Vec<f64> = (0..n).map(|i| i as f64 / 5.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 4.0 + 1.5 * x1[i] - 0.75 * x2[i] + ((i % 11) as f64 - 5.0) * 0.3)
            .collect();
        Frame::new(n)
            .numeric("y", y)
            .numeric("x1", x1)
            .numeric("x2", x2)
    }

    macro_rules! with_model {
        ($n:expr, $cfg:expr, $model:ident, $body:block) => {{
            let f = frame($n);
            let refs = f.key_refs();
            let view = f.view(&refs);
            let $model = PooledGaussian
                .compile(&Config::parse($cfg).unwrap(), &view)
                .unwrap();
            $body
        }};
    }

    /// Mean and standard deviation of parameter `j`, and the chains of it, from a flat
    /// chain-major value block.
    fn column(values: &[f64], p: usize, j: usize) -> Vec<f64> {
        values.chunks(p).map(|c| c[j]).collect()
    }

    fn mean_sd(col: &[f64]) -> (f64, f64) {
        let m = col.iter().sum::<f64>() / col.len() as f64;
        let sd = (col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (col.len() - 1) as f64).sqrt();
        (m, sd)
    }

    /// **The headline correctness gate for this engine.**
    ///
    /// `pooled_gaussian` has a closed-form posterior, so there is a right answer to
    /// compare against rather than a second approximation. A sampler that agrees with
    /// it to Monte Carlo error is a sampler whose adapter — the transform, the
    /// gradient hand-off, the warmup discard, the chain layout — is wired correctly;
    /// almost any mistake in any of those shifts a mean or a spread.
    ///
    /// **Where the tolerance comes from.** Not from a round number that happens to
    /// pass. The difference between two Monte Carlo estimates of the same quantity is
    /// itself random, with standard error
    ///
    /// ```text
    ///   MCSE(mean_nuts - mean_exact) = sd * sqrt(1/ESS_nuts + 1/N_exact)
    ///   MCSE(sd_nuts   - sd_exact)   = sd * sqrt(1/(2 ESS_nuts) + 1/(2 N_exact))
    /// ```
    ///
    /// the first being the standard error of a mean over `ESS` effectively independent
    /// draws, and the second the asymptotic standard error of a sample standard
    /// deviation, `sd/sqrt(2n)`, which is exact in the Gaussian limit this posterior
    /// sits in at n = 400. `ESS_nuts` is *measured* from the draws with the crate's own
    /// rank-normalised estimator rather than assumed equal to the draw count, because
    /// NUTS draws are autocorrelated and pretending otherwise would understate the
    /// tolerance and make the test flaky for a correct sampler.
    ///
    /// The band is five standard errors. Under the null — a correct sampler — a
    /// five-sigma excursion has probability about 6e-7 per comparison, so across the
    /// six comparisons here the suite flakes about once in 300 000 runs, while a bias
    /// large enough to matter for a decision is many standard errors wide and cannot
    /// hide inside it.
    #[test]
    fn the_nuts_engine_agrees_with_the_exact_conjugate_posterior() {
        with_model!(400, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let n_exact = 200_000;
            let exact = ExactEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 1,
                        n_draws: n_exact,
                        n_warmup: 0,
                        seed: 17,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            let nuts_opts = SampleOptions {
                n_chains: 4,
                n_draws: 2000,
                n_warmup: 1000,
                seed: 17,
                sample_from: crate::types::SampleFrom::Posterior,
            };
            let nuts = NutsEngine.sample(&*model, &nuts_opts).unwrap();

            let p = model.param_names().len();
            for j in 0..p {
                let name = &model.param_names()[j].name;
                let ecol = column(&exact.values, p, j);
                let ncol = column(&nuts.values, p, j);
                let (em, esd) = mean_sd(&ecol);
                let (nm, nsd) = mean_sd(&ncol);

                let chains: Vec<Vec<f64>> = (0..nuts_opts.n_chains)
                    .map(|c| ncol[c * nuts_opts.n_draws..(c + 1) * nuts_opts.n_draws].to_vec())
                    .collect();
                let ess = ess_bulk(&chains);
                assert!(ess > 100.0, "{name}: implausibly low ESS {ess}");

                let mcse_mean = esd * (1.0 / ess + 1.0 / n_exact as f64).sqrt();
                let mcse_sd = esd * (0.5 / ess + 0.5 / n_exact as f64).sqrt();
                println!(
                    "{name}: exact ({em:.6}, {esd:.6}) nuts ({nm:.6}, {nsd:.6}) \
                     ess {ess:.0} mcse_mean {mcse_mean:.2e} mcse_sd {mcse_sd:.2e}"
                );

                assert!(
                    (nm - em).abs() < 5.0 * mcse_mean,
                    "{name}: mean {nm} vs exact {em}, gap {} exceeds 5 MCSE = {}",
                    (nm - em).abs(),
                    5.0 * mcse_mean
                );
                assert!(
                    (nsd - esd).abs() < 5.0 * mcse_sd,
                    "{name}: sd {nsd} vs exact {esd}, gap {} exceeds 5 MCSE = {}",
                    (nsd - esd).abs(),
                    5.0 * mcse_sd
                );
            }
        });
    }

    /// A per-store price panel with a genuinely per-store elasticity — the shape a
    /// random-slopes fit exists for.
    fn store_frame(n_per_store: usize, slopes: &[f64]) -> (Frame, Vec<String>) {
        let n = n_per_store * slopes.len();
        let (mut y, mut price, mut store) = (Vec::new(), Vec::new(), Vec::new());
        for (s, &slope) in slopes.iter().enumerate() {
            for i in 0..n_per_store {
                let p = ((i % 7) as f64) - 3.0;
                price.push(p);
                y.push(20.0 + 3.0 * s as f64 + slope * p + ((i % 5) as f64 - 2.0) * 0.4);
                store.push(format!("S{s}"));
            }
        }
        let keys: Vec<String> = store.clone();
        let frame = Frame::new(n).numeric("units", y).numeric("price", price);
        (frame, keys)
    }

    /// **The warranty, on a random-slopes design: all three engines still agree.**
    ///
    /// `pooled_gaussian` promises a closed-form posterior cross-checked by independent
    /// derivations. Random slopes are a design-matrix change, so that promise has to
    /// survive them unchanged — and this is the test that says whether it did, across
    /// `exact` (the formula), `laplace` (a Gaussian fit to the same log density) and
    /// `nuts` (a Markov chain over it).
    ///
    /// The tolerances are not the same for the two comparisons and should not be.
    /// `laplace` differs from `exact` by a genuine `O(1/n)` approximation error on
    /// `sigma`, which is a property of the method rather than of the code; `nuts` is
    /// asymptotically exact, so it is held to five Monte Carlo standard errors with the
    /// effective sample size *measured* from its own draws.
    ///
    /// **Recorded finding, because it is the reason the numbers below look the way they
    /// do.** `pooled_gaussian` with a `group` column already mixes badly under NUTS:
    /// the unpenalised intercept and the pooled group effects form a ridge that is not
    /// axis-aligned, and a diagonal mass matrix cannot precondition it. Random slopes
    /// add a second such ridge per predictor. The test therefore *prints* the effective
    /// sample size per parameter and gates on a floor low enough to be honest about
    /// what the sampler achieves rather than on one that would flake — the fit's own
    /// diagnostics, not this test, are what refuse a chain that explored too little.
    #[test]
    fn all_three_engines_agree_on_a_random_slopes_design() {
        let (frame, keys) = store_frame(60, &[-1.5, -0.6, -2.4, -1.2]);
        let key_ref: Vec<&str> = keys.iter().map(String::as_str).collect();
        let frame = frame.key("store", key_ref);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = PooledGaussian
            .compile(
                &Config::parse(
                    r#"{"y": "units", "x": "price", "group": "store",
                        "random_slopes": "price", "pool_scale": 0.5}"#,
                )
                .unwrap(),
                &view,
            )
            .unwrap();
        let p = model.param_names().len();
        assert_eq!(
            p, 11,
            "intercept + slope + 4 effects + 4 group slopes + sigma"
        );

        let n_indep = 200_000;
        let indep_opts = |seed| SampleOptions {
            n_chains: 1,
            n_draws: n_indep,
            n_warmup: 0,
            seed,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        let exact = ExactEngine.sample(&*model, &indep_opts(21)).unwrap();
        let laplace = crate::engines::LaplaceEngine
            .sample(&*model, &indep_opts(22))
            .unwrap();
        let nuts_opts = SampleOptions {
            n_chains: 4,
            n_draws: 2000,
            n_warmup: 1000,
            seed: 23,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        let nuts = NutsEngine.sample(&*model, &nuts_opts).unwrap();

        for j in 0..p {
            let name = &model.param_names()[j].name;
            let group = &model.param_names()[j].group_id;
            let ecol = column(&exact.values, p, j);
            let lcol = column(&laplace.values, p, j);
            let ncol = column(&nuts.values, p, j);
            let (em, esd) = mean_sd(&ecol);
            let (lm, lsd) = mean_sd(&lcol);
            let (nm, nsd) = mean_sd(&ncol);

            let chains: Vec<Vec<f64>> = (0..nuts_opts.n_chains)
                .map(|c| ncol[c * nuts_opts.n_draws..(c + 1) * nuts_opts.n_draws].to_vec())
                .collect();
            let ess = ess_bulk(&chains);
            println!(
                "{name}/{group}: exact ({em:.5}, {esd:.5}) laplace ({lm:.5}, {lsd:.5}) \
                 nuts ({nm:.5}, {nsd:.5}) ess {ess:.0}"
            );

            // --- laplace vs exact.
            if name == "sigma" {
                // **A derived offset, pinned rather than tolerated.** `sigma` is the
                // one coordinate whose posterior is not Gaussian on the unconstrained
                // scale, so Laplace centres it on the joint mode
                // `sigma^2 = 2 s_n / (2 a_n + p)` while the exact posterior mean of
                // `sigma^2` is `s_n / (a_n - 1)`. The ratio is a function of the
                // *shape* alone and so is computable here from the request:
                // `a_n = (n - n_flat)/2` with two flat coefficients (the intercept and
                // the population slope; every group column is penalised by
                // `pool_scale`), against `p` coefficients in the design.
                //
                // This is the number random slopes move. Each one adds a group's worth
                // of columns to `p` without adding an observation, so the Laplace
                // engine's understatement of `sigma` grows with the width of the
                // design. Asserting the predicted ratio rather than a round tolerance
                // is what turns that from a mystery into a documented cost.
                let n_obs = model.n_obs() as f64;
                let a_n = (n_obs - 2.0) / 2.0;
                let predicted = ((2.0 * (a_n - 1.0)) / (2.0 * a_n + p as f64)).sqrt();
                let ratio = lm / em;
                println!("  sigma laplace/exact {ratio:.5}, mode/mean predicts {predicted:.5}");
                assert!(
                    (ratio - predicted).abs() < 0.005,
                    "sigma: laplace/exact {ratio} vs the predicted mode-to-mean ratio \
                     {predicted}; the Laplace error on this family's variance parameter \
                     is a known O(1/n) offset and this is where its size is pinned"
                );
            } else {
                // The coefficients' conditional posterior is exactly Gaussian, so the
                // two engines agree far inside this band.
                assert!(
                    (lm - em).abs() < 0.02 * esd,
                    "{name}/{group}: laplace mean {lm} vs exact {em}"
                );
            }
            assert!(
                (lsd - esd).abs() < 0.05 * esd,
                "{name}/{group}: laplace sd {lsd} vs exact {esd}"
            );

            // --- nuts vs exact, at five measured Monte Carlo standard errors.
            assert!(ess > 100.0, "{name}/{group}: implausibly low ESS {ess}");
            let mcse_mean = esd * (1.0 / ess + 1.0 / n_indep as f64).sqrt();
            let mcse_sd = esd * (0.5 / ess + 0.5 / n_indep as f64).sqrt();
            assert!(
                (nm - em).abs() < 5.0 * mcse_mean,
                "{name}/{group}: nuts mean {nm} vs exact {em}, gap {} exceeds {}",
                (nm - em).abs(),
                5.0 * mcse_mean
            );
            assert!(
                (nsd - esd).abs() < 5.0 * mcse_sd,
                "{name}/{group}: nuts sd {nsd} vs exact {esd}, gap {} exceeds {}",
                (nsd - esd).abs(),
                5.0 * mcse_sd
            );
        }
    }

    /// Warmup draws are adaptation, not posterior samples: the step size and mass
    /// matrix are still moving, so they are draws from a sequence of different
    /// samplers. Emitting them would bias every quantile a caller reads, and nothing
    /// downstream could tell.
    ///
    /// Checked by counting rather than by inspection, because that is the property the
    /// contract makes: exactly `chains * draws` rows, whatever the warmup budget.
    #[test]
    fn warmup_draws_do_not_reach_the_output() {
        with_model!(60, r#"{"y": "y", "x": "x1"}"#, model, {
            let p = model.param_names().len();
            for n_warmup in [50, 300, 1000] {
                let sample = NutsEngine
                    .sample(
                        &*model,
                        &SampleOptions {
                            n_chains: 2,
                            n_draws: 200,
                            n_warmup,
                            seed: 5,
                            sample_from: crate::types::SampleFrom::Posterior,
                        },
                    )
                    .unwrap();
                assert_eq!(sample.values.len(), 2 * 200 * p);
                assert_eq!(sample.stats.len(), 2 * 200);
            }
        });
    }

    /// Determinism is contractual, not aspirational: `DRAWS_CONTRACT.md` promises an
    /// auditor can re-run a customer's fit and get the same recommendation.
    #[test]
    fn the_same_seed_reproduces_the_same_nuts_draws_bit_for_bit() {
        with_model!(50, r#"{"y": "y", "x": "x1"}"#, model, {
            let opts = SampleOptions {
                n_chains: 3,
                n_draws: 150,
                n_warmup: 200,
                seed: 23,
                sample_from: crate::types::SampleFrom::Posterior,
            };
            let a = NutsEngine.sample(&*model, &opts).unwrap();
            let b = NutsEngine.sample(&*model, &opts).unwrap();
            assert_eq!(
                a.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(a.stats, b.stats);

            // ...and a different seed does not, which is what makes the equality above
            // evidence of reproduction rather than of a constant.
            let c = NutsEngine
                .sample(&*model, &SampleOptions { seed: 24, ..opts })
                .unwrap();
            assert_ne!(a.values, c.values);
        });
    }

    /// **The determinism requirement in its strongest form.** Same seed, same draws,
    /// byte-identical, regardless of how many threads are running.
    ///
    /// This engine drives one `nuts_rs::Chain` at a time from a `BayesRng::for_chain`
    /// stream, and takes no thread pool of its own, so the claim is structural rather
    /// than hopeful. The test still runs it, because "structural" is exactly the kind
    /// of claim that stops being true when someone later adds a parallel iterator.
    ///
    /// Each thread rebuilds its own frame and model: a `CompiledModel` borrows its
    /// data and is not `Send`, and that is the honest shape of the check anyway — a
    /// concurrent DuckDB pipeline fits per thread, it does not share one fit.
    #[test]
    fn nuts_draws_are_identical_however_many_threads_are_running() {
        fn draw_bits() -> Vec<u64> {
            let f = frame(80);
            let refs = f.key_refs();
            let view = f.view(&refs);
            let model = PooledGaussian
                .compile(
                    &Config::parse(r#"{"y": "y", "x": ["x1", "x2"]}"#).unwrap(),
                    &view,
                )
                .unwrap();
            let sample = NutsEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 2,
                        n_draws: 120,
                        n_warmup: 250,
                        seed: 4242,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            sample.values.iter().map(|v| v.to_bits()).collect()
        }

        let reference = draw_bits();
        for n_threads in [1usize, 2, 8] {
            let handles: Vec<_> = (0..n_threads)
                .map(|_| std::thread::spawn(draw_bits))
                .collect();
            for h in handles {
                assert_eq!(
                    h.join().unwrap(),
                    reference,
                    "the draws changed with {n_threads} threads in flight"
                );
            }
        }
    }

    /// NUTS is the first engine to populate the sample-statistic half of the draws
    /// contract. All four must be present on every kept draw — the contract requires a
    /// uniform statistic shape, and `Posterior::new` rejects a ragged one.
    #[test]
    fn every_kept_draw_carries_the_four_sampler_statistics() {
        with_model!(60, r#"{"y": "y", "x": "x1"}"#, model, {
            let sample = NutsEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 2,
                        n_draws: 100,
                        n_warmup: 200,
                        seed: 8,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            assert_eq!(sample.stats.len(), 200);
            for s in &sample.stats {
                assert!(!s.is_empty());
                let lp = s.lp.expect("lp");
                let energy = s.energy.expect("energy");
                let step = s.step_size.expect("step size");
                let div = s.divergent.expect("divergence flag");
                assert!(lp.is_finite(), "lp {lp}");
                assert!(energy.is_finite(), "energy {energy}");
                assert!(step.is_finite() && step > 0.0, "step size {step}");
                assert!(div == 0.0 || div == 1.0, "divergence flag {div}");
            }

            // The energy must *move*. It is the potential plus a kinetic term drawn
            // fresh at the start of every trajectory, so a constant energy series would
            // mean the momentum was never resampled — the failure mode that makes
            // E-BFMI, the diagnostic `__energy__` exists to support, meaningless. A
            // constant would also be exactly what a wiring mistake produces if the same
            // statistic were copied to every draw.
            let energies: Vec<f64> = sample.stats.iter().map(|s| s.energy.unwrap()).collect();
            let mean = energies.iter().sum::<f64>() / energies.len() as f64;
            let var = energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>()
                / (energies.len() - 1) as f64;
            assert!(var > 0.0, "the energy is constant across every draw");

            // Note what is *not* asserted: `energy >= -lp`. `nuts-rs` reports the
            // energy of the transformed point, which carries the log-determinant of
            // the mass-matrix transform, so the two are not on a common scale and the
            // inequality does not hold. Both are still the sampler's own numbers,
            // reported rather than recomputed.
        });
    }

    /// A well-conditioned Gaussian posterior is the easiest thing NUTS ever has to do.
    /// If this fit reported a divergence, the adapter — not the sampler — would be
    /// wrong, and the divergence path in `fit::grade` would be firing on noise.
    #[test]
    fn a_clean_gaussian_posterior_produces_no_divergences() {
        with_model!(200, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let sample = NutsEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 4,
                        n_draws: 500,
                        n_warmup: 500,
                        seed: 99,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            let divergences: f64 = sample.stats.iter().filter_map(|s| s.divergent).sum();
            assert_eq!(divergences, 0.0, "{divergences} divergences on a Gaussian");
        });
    }

    /// Chains that all start at the mode agree with each other by construction, which
    /// would make R̂ a formality rather than a diagnostic. Overdispersed starting
    /// points are what give it something to detect.
    ///
    /// `nuts-rs` also rejects an initial point whose gradient is exactly zero — which
    /// the mode is — so this is load-bearing twice over.
    #[test]
    fn chains_start_from_different_overdispersed_points() {
        with_model!(60, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let target = model.as_differentiable().unwrap();
            let mode = target.initial();
            let starts: Vec<Vec<f64>> = (0..4)
                .map(|c| {
                    let mut rng = crate::rng::BayesRng::for_chain(31, c);
                    super::overdispersed_start(target, &mut rng).unwrap()
                })
                .collect();

            for c in 0..4 {
                assert_ne!(starts[c], mode, "chain {c} started exactly at the mode");
                let mut grad = vec![0.0; target.dim()];
                target.grad(&starts[c], &mut grad).unwrap();
                assert!(
                    grad.iter().all(|g| g.is_finite()) && grad.iter().any(|g| *g != 0.0),
                    "chain {c} started where the gradient is zero or not finite"
                );
                for d in (c + 1)..4 {
                    assert_ne!(starts[c], starts[d], "chains {c} and {d} share a start");
                }
            }
        });
    }

    /// `sigma` is sampled as `log sigma`, so every draw is positive by construction.
    #[test]
    fn a_positive_parameter_stays_positive_because_it_is_sampled_on_the_log_scale() {
        with_model!(30, r#"{"y": "y", "x": "x1"}"#, model, {
            let sample = NutsEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 2,
                        n_draws: 500,
                        n_warmup: 400,
                        seed: 29,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            let p = model.param_names().len();
            assert_eq!(model.param_names()[p - 1].name, "sigma");
            assert!(sample.values.chunks(p).all(|c| c[p - 1] > 0.0));
        });
    }

    #[test]
    fn a_family_without_a_gradient_is_refused_rather_than_approximated() {
        /// A model that exposes no differentiable path, which is the only thing this
        /// engine needs and the only thing it can refuse for.
        ///
        /// A stub rather than a real family: every family in the catalog now exposes a
        /// gradient, so there is nothing left to point at. The refusal path still has
        /// to work -- the next family added might not -- and a test that can only be
        /// written against a stub is exactly the case where a stub is right.
        #[derive(Debug)]
        struct NoGradient;

        impl crate::catalog::CompiledModel for NoGradient {
            fn param_names(&self) -> &[crate::draws::ParamName] {
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

        assert!(!NutsEngine.supports(&NoGradient));
        let err = NutsEngine
            .sample(&NoGradient, &SampleOptions::default())
            .unwrap_err();
        assert!(err.to_string().contains("differentiable"), "{err}");
    }

    /// NUTS on a family whose posterior is known in closed form.
    ///
    /// `conjugate_anomaly` gained a differentiable path (roadmap gap 10) after this
    /// engine was written, so all three engines now describe the same distribution by
    /// three different routes. That is the strongest correctness gate available here:
    /// a sampler is being checked against an answer, not against another sampler.
    #[test]
    fn nuts_agrees_with_the_closed_form_conjugate_posterior() {
        let values: Vec<f64> = (0..200)
            .map(|i| 10.0 + ((i % 13) as f64 - 6.0) * 0.5)
            .collect();
        let f = Frame::new(values.len()).numeric("cost", values);
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = crate::catalog::f7_conjugate::ConjugateAnomaly
            .compile(&Config::parse(r#"{"value": "cost"}"#).unwrap(), &view)
            .unwrap();

        assert!(
            NutsEngine.supports(&*model),
            "conjugate_anomaly exposes a gradient since gap 10"
        );

        let opts = SampleOptions {
            n_chains: 4,
            n_draws: 2000,
            n_warmup: 1000,
            seed: 21,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        let sampled = NutsEngine.sample(&*model, &opts).unwrap();
        let exact = crate::engines::exact::ExactEngine
            .sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 200_000,
                    ..opts
                },
            )
            .unwrap();

        // Column-major over (chain, draw): mu is slot 0, sigma slot 1 of each draw.
        let width = model.param_names().len();
        for slot in 0..width {
            let take =
                |v: &[f64]| -> Vec<f64> { v.iter().skip(slot).step_by(width).copied().collect() };
            let (a, b) = (take(&sampled.values), take(&exact.values));
            let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let sd = |v: &[f64]| {
                let mu = m(v);
                (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
            };
            let name = &model.param_names()[slot].name;
            // A loose band on purpose: the point is that NUTS lands on the right
            // distribution, and the MCSE-derived bound for this engine already has its
            // own dedicated test on F3.
            assert!(
                (m(&a) - m(&b)).abs() < 0.02 * sd(&b).max(1e-9) + 1e-3,
                "{name} mean: nuts {} vs exact {}",
                m(&a),
                m(&b)
            );
            assert!(
                (sd(&a) - sd(&b)).abs() < 0.08 * sd(&b),
                "{name} sd: nuts {} vs exact {}",
                sd(&a),
                sd(&b)
            );
        }
    }

    /// **The layout and the determinism claim, checked as one property.**
    ///
    /// Chain *c* is seeded from `(seed, c)` and from nothing else, so its block of
    /// draws must be byte-identical whether the caller asked for three chains or four.
    /// That single equality pins both halves of what this engine promises: the values
    /// really are laid out chain-major in the order `Posterior` expects (a draw-major
    /// write would scatter chain 0 across the buffer and fail immediately), and no
    /// chain's numbers depend on how many other chains were run beside it.
    ///
    /// It is also the property that would break first if someone parallelised the
    /// chain loop with a shared RNG, which is precisely the change most likely to be
    /// attempted here.
    #[test]
    fn a_chains_draws_depend_on_its_index_and_not_on_how_many_chains_were_asked_for() {
        with_model!(60, r#"{"y": "y", "x": "x1"}"#, model, {
            let p = model.param_names().len();
            let n_draws = 100;
            let run = |n_chains: usize| {
                NutsEngine
                    .sample(
                        &*model,
                        &SampleOptions {
                            n_chains,
                            n_draws,
                            n_warmup: 300,
                            seed: 12,
                            sample_from: crate::types::SampleFrom::Posterior,
                        },
                    )
                    .unwrap()
            };
            let three = run(3);
            let four = run(4);

            for c in 0..3 {
                let block = c * n_draws * p..(c + 1) * n_draws * p;
                assert_eq!(
                    three.values[block.clone()]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    four.values[block]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "chain {c} changed when a fourth chain was added"
                );
                let stat_block = c * n_draws..(c + 1) * n_draws;
                assert_eq!(three.stats[stat_block.clone()], four.stats[stat_block]);
            }
            // ...and the chains are not copies of each other, which is what makes the
            // equality above evidence rather than a tautology.
            assert_ne!(
                three.values[..n_draws * p],
                three.values[n_draws * p..2 * n_draws * p]
            );
        });
    }

    /// The step size is adapted per chain, and after warmup `nuts-rs` jitters it around
    /// the adapted value on every draw — so it is deliberately *not* constant, and a
    /// test asserting that it were would have been asserting a property of an older
    /// upstream. What must hold is that it stays positive and finite, and that chains
    /// adapt independently rather than to one shared number.
    #[test]
    fn each_chain_adapts_its_own_step_size() {
        with_model!(60, r#"{"y": "y", "x": "x1"}"#, model, {
            let (n_chains, n_draws) = (3, 100);
            let sample = NutsEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains,
                        n_draws,
                        n_warmup: 300,
                        seed: 12,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            let step: Vec<f64> = sample.stats.iter().map(|s| s.step_size.unwrap()).collect();
            assert!(step.iter().all(|s| s.is_finite() && *s > 0.0));

            let means: Vec<f64> = (0..n_chains)
                .map(|c| step[c * n_draws..(c + 1) * n_draws].iter().sum::<f64>() / n_draws as f64)
                .collect();
            let mut unique: Vec<u64> = means.iter().map(|m| m.to_bits()).collect();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                unique.len(),
                n_chains,
                "chains adapted to identical step sizes {means:?}, \
                 which would mean they are not independent"
            );
        });
    }

    /// Guards the assumption the whole engine rests on: a `SampleStats` value carrying
    /// all four fields is what the draws contract calls a full statistic row.
    #[test]
    fn a_full_statistic_row_is_not_empty() {
        let s = SampleStats {
            lp: Some(-1.0),
            divergent: Some(0.0),
            energy: Some(2.0),
            step_size: Some(0.3),
        };
        assert!(!s.is_empty());
    }
}
