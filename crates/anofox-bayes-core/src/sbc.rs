//! Simulation-based calibration.
//!
//! Every other test in this crate checks that a posterior matches a formula. SBC
//! checks something stronger and much harder to fake: that the posterior is
//! *calibrated* — that a 90 % credible interval contains the truth 90 % of the time.
//!
//! The construction (Talts et al. 2018) is a consequence of one identity. If
//! `theta ~ prior` and `y ~ p(y | theta)` and `theta_1..theta_L ~ p(theta | y)`, then
//! `theta` and the posterior draws are exchangeable, so the **rank** of `theta` among
//! the `L` draws is uniform on `0..L`. Any deviation from uniformity is a calibration
//! error, and the shape of the deviation names the fault:
//!
//! | Rank histogram | Diagnosis |
//! |---|---|
//! | ∪-shaped | posterior too narrow — overconfident |
//! | ∩-shaped | posterior too wide — underconfident |
//! | sloped | posterior biased |
//!
//! This is the property that matters commercially. A model that is merely *accurate*
//! still ruins a service-level decision if its intervals are too tight, because the
//! decision reads the interval, not the point estimate. And unlike a comparison
//! against a reference implementation, SBC needs no reference: it tests the fit
//! against the generative model it claims to invert.
//!
//! Runs are slow by construction — hundreds of complete fits per family per engine —
//! so the suites are `#[ignore]`d and run explicitly via `make test_sbc` or in CI.

use crate::errors::BayesResult;
use crate::rng::BayesRng;

/// One SBC run's outcome for one parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct RankHistogram {
    pub param: String,
    /// `ranks[r]` is how many replications gave rank `r`.
    pub counts: Vec<u32>,
    pub n_replications: u32,
}

impl RankHistogram {
    pub fn new(param: impl Into<String>, n_bins: usize) -> Self {
        Self {
            param: param.into(),
            counts: vec![0; n_bins],
            n_replications: 0,
        }
    }

    pub fn record(&mut self, rank: usize) {
        let bin = rank.min(self.counts.len() - 1);
        self.counts[bin] += 1;
        self.n_replications += 1;
    }

    /// Pearson chi-squared statistic against the uniform expectation.
    pub fn chi_squared(&self) -> f64 {
        let bins = self.counts.len() as f64;
        let expected = self.n_replications as f64 / bins;
        if expected <= 0.0 {
            return f64::NAN;
        }
        self.counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum()
    }

    /// Degrees of freedom for [`RankHistogram::chi_squared`].
    pub fn degrees_of_freedom(&self) -> usize {
        self.counts.len().saturating_sub(1)
    }

    /// Whether the histogram is uniform enough to pass.
    ///
    /// The threshold is a chi-squared upper-tail critical value, supplied by the
    /// caller so that the same harness can be run at different strictness. Compared
    /// against the statistic directly rather than against a p-value: computing the
    /// p-value would need the incomplete gamma function for no gain, since the
    /// decision is a single comparison either way.
    pub fn passes(&self, critical_value: f64) -> bool {
        let stat = self.chi_squared();
        stat.is_finite() && stat <= critical_value
    }

    /// Signed measure of asymmetry, positive when high ranks dominate.
    ///
    /// Distinguishes a *biased* posterior from a merely miscalibrated one: bias
    /// slopes the histogram, while a width error bows it. Reported alongside the
    /// chi-squared so a failure says which of the two it is.
    pub fn slope(&self) -> f64 {
        let bins = self.counts.len();
        if bins < 2 || self.n_replications == 0 {
            return 0.0;
        }
        let mid = (bins - 1) as f64 / 2.0;
        let weighted: f64 = self
            .counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as f64 - mid) * c as f64)
            .sum();
        weighted / (self.n_replications as f64 * mid)
    }
}

/// A generative model the harness can round-trip: draw a parameter from the prior,
/// simulate data from it, fit, and rank.
///
/// The trait is what keeps the harness family-agnostic. It knows how to test
/// calibration; it knows nothing about what is being calibrated.
pub trait SbcModel {
    /// Names of the parameters whose ranks are tracked.
    fn param_names(&self) -> Vec<String>;

    /// Draw one parameter vector from the prior.
    fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>>;

    /// Fit data simulated from `truth`, returning `n_draws` posterior draws per
    /// parameter as `draws[param][draw]`.
    ///
    /// Simulation and fitting are one method rather than two because the simulated
    /// dataset never needs to leave: keeping it internal means the harness cannot
    /// accidentally fit a dataset generated from a different truth, which is the one
    /// bug that would make SBC pass vacuously.
    fn simulate_and_fit(
        &self,
        truth: &[f64],
        rng: &mut BayesRng,
        n_draws: usize,
    ) -> BayesResult<Vec<Vec<f64>>>;
}

/// Run `n_replications` of the SBC loop.
///
/// `n_draws` becomes the number of rank bins: with `L` posterior draws the rank of
/// the truth falls in `0..=L`, so there are `L + 1` possible values. Choosing `L`
/// so that `(L + 1)` divides the replication count evenly keeps the uniform
/// expectation exact and avoids a spurious chi-squared contribution from rounding.
pub fn run_sbc(
    model: &dyn SbcModel,
    n_replications: u32,
    n_draws: usize,
    seed: u64,
) -> BayesResult<Vec<RankHistogram>> {
    let names = model.param_names();
    let mut histograms: Vec<RankHistogram> = names
        .iter()
        .map(|n| RankHistogram::new(n.clone(), n_draws + 1))
        .collect();

    for replication in 0..n_replications {
        // A fresh stream per replication, so a failure at replication 137 can be
        // reproduced without re-running the first 136.
        let mut rng = BayesRng::for_chain(seed, replication);
        let truth = model.draw_prior(&mut rng)?;
        let draws = model.simulate_and_fit(&truth, &mut rng, n_draws)?;

        for (p, hist) in histograms.iter_mut().enumerate() {
            // The rank is how many posterior draws fall below the truth.
            let rank = draws[p].iter().filter(|d| **d < truth[p]).count();
            hist.record(rank);
        }
    }
    Ok(histograms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that is calibrated by construction: the "posterior" is the prior, and
    /// the truth is drawn from the same prior. Ranks must be uniform.
    struct PerfectlyCalibrated;

    impl SbcModel for PerfectlyCalibrated {
        fn param_names(&self) -> Vec<String> {
            vec!["theta".into()]
        }
        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            Ok(vec![rng.standard_normal()])
        }
        fn simulate_and_fit(
            &self,
            _truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            Ok(vec![(0..n_draws).map(|_| rng.standard_normal()).collect()])
        }
    }

    /// An overconfident model: the posterior is half as wide as it should be. The
    /// truth then falls outside it too often, so extreme ranks are over-represented
    /// and the histogram is U-shaped.
    struct Overconfident;

    impl SbcModel for Overconfident {
        fn param_names(&self) -> Vec<String> {
            vec!["theta".into()]
        }
        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            Ok(vec![rng.standard_normal()])
        }
        fn simulate_and_fit(
            &self,
            _truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            Ok(vec![(0..n_draws)
                .map(|_| 0.5 * rng.standard_normal())
                .collect()])
        }
    }

    /// A biased model: the posterior is shifted. The histogram slopes.
    struct Biased;

    impl SbcModel for Biased {
        fn param_names(&self) -> Vec<String> {
            vec!["theta".into()]
        }
        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            Ok(vec![rng.standard_normal()])
        }
        fn simulate_and_fit(
            &self,
            _truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            Ok(vec![(0..n_draws)
                .map(|_| rng.standard_normal() + 1.0)
                .collect()])
        }
    }

    // Chi-squared 99.9% critical values, so a passing suite almost never flakes.
    const CRITICAL_15_DF: f64 = 37.7;

    #[test]
    fn a_calibrated_model_produces_uniform_ranks() {
        let hists = run_sbc(&PerfectlyCalibrated, 1600, 15, 1).unwrap();
        let h = &hists[0];
        assert_eq!(h.n_replications, 1600);
        assert_eq!(h.counts.len(), 16);
        assert!(
            h.passes(CRITICAL_15_DF),
            "chi-squared {} should be below {CRITICAL_15_DF}; counts {:?}",
            h.chi_squared(),
            h.counts
        );
        assert!(h.slope().abs() < 0.1, "slope {}", h.slope());
    }

    /// **The test that proves the harness can fail.** A suite that only ever passes
    /// is not a gate. An overconfident posterior -- the failure mode that matters
    /// commercially, because it produces intervals that are too tight and decisions
    /// that are too bold -- must be rejected.
    #[test]
    fn an_overconfident_posterior_is_rejected() {
        let hists = run_sbc(&Overconfident, 1600, 15, 2).unwrap();
        let h = &hists[0];
        assert!(
            !h.passes(CRITICAL_15_DF),
            "chi-squared {} should exceed {CRITICAL_15_DF}; counts {:?}",
            h.chi_squared(),
            h.counts
        );
        // U-shaped, not sloped: the extremes carry the mass, and the two ends
        // roughly balance, so the slope stays small.
        let bins = h.counts.len();
        let extremes = h.counts[0] + h.counts[bins - 1];
        let middle = h.counts[bins / 2 - 1] + h.counts[bins / 2];
        assert!(extremes > 2 * middle, "counts {:?}", h.counts);
        assert!(
            h.slope().abs() < 0.2,
            "an overconfident fit should bow, not slope"
        );
    }

    /// A biased posterior slopes rather than bowing, and the two diagnoses must be
    /// distinguishable -- otherwise a failure report cannot say what to fix.
    #[test]
    fn a_biased_posterior_is_rejected_and_slopes_rather_than_bowing() {
        let hists = run_sbc(&Biased, 1600, 15, 3).unwrap();
        let h = &hists[0];
        assert!(!h.passes(CRITICAL_15_DF), "chi-squared {}", h.chi_squared());
        // The posterior sits above the truth, so the truth ranks low.
        assert!(
            h.slope() < -0.3,
            "slope {} should be strongly negative",
            h.slope()
        );
    }

    #[test]
    fn the_rank_of_the_truth_is_how_many_draws_fall_below_it() {
        let mut h = RankHistogram::new("theta", 4);
        h.record(0);
        h.record(3);
        h.record(3);
        assert_eq!(h.counts, vec![1, 0, 0, 2]);
        assert_eq!(h.n_replications, 3);
        // A rank beyond the last bin is clamped rather than panicking; it can only
        // arise from an off-by-one and is caught by the uniformity test anyway.
        h.record(99);
        assert_eq!(h.counts[3], 3);
    }

    #[test]
    fn a_perfectly_uniform_histogram_has_a_chi_squared_of_zero() {
        let mut h = RankHistogram::new("theta", 4);
        for _ in 0..25 {
            for r in 0..4 {
                h.record(r);
            }
        }
        assert!(h.chi_squared().abs() < 1e-12);
        assert_eq!(h.degrees_of_freedom(), 3);
        assert!(h.slope().abs() < 1e-12);
    }
}

/// Calibration suites for the shipped families.
///
/// These are `#[ignore]`d: each runs hundreds of complete fits and takes minutes.
/// Run them with `make test_sbc`, and in CI as a release gate.
///
/// **A note on which priors can be certified.** SBC requires drawing the truth from
/// the prior, so it can only certify a model under a *proper* one. Both families
/// default to reference priors, which are improper and cannot be sampled from — so
/// these suites run under explicit, proper priors. That is not a loophole: the
/// posterior machinery being exercised (the conjugate update, the sampler, the
/// transform) is identical either way, and it is the machinery that calibration
/// tests. What the reference-prior default changes is the amount of prior
/// information, not the correctness of the update.
#[cfg(test)]
mod families {
    use super::*;
    use crate::catalog::{
        f3_pooled_gaussian::PooledGaussian, f7_conjugate::ConjugateAnomaly, ModelFamily,
    };
    use crate::config::Config;
    use crate::data::testing::Frame;
    use crate::engines::{Engine, ExactEngine, LaplaceEngine, NutsEngine, SampleOptions};
    use crate::types::EngineKind;

    /// Draw from `InvGamma(shape, rate)`.
    fn inv_gamma(rng: &mut BayesRng, shape: f64, rate: f64) -> BayesResult<f64> {
        Ok(1.0 / rng.gamma(shape, rate)?)
    }

    /// F7 with a Normal likelihood and a proper Normal-Inverse-Gamma prior.
    struct F7Normal {
        n_obs: usize,
        mu0: f64,
        kappa0: f64,
        alpha0: f64,
        beta0: f64,
    }

    impl SbcModel for F7Normal {
        fn param_names(&self) -> Vec<String> {
            vec!["mu".into(), "sigma".into()]
        }

        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            let sigma_sq = inv_gamma(rng, self.alpha0, self.beta0)?;
            let mu = self.mu0 + (sigma_sq / self.kappa0).sqrt() * rng.standard_normal();
            Ok(vec![mu, sigma_sq.sqrt()])
        }

        fn simulate_and_fit(
            &self,
            truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            let (mu, sigma) = (truth[0], truth[1]);
            let y: Vec<f64> = (0..self.n_obs)
                .map(|_| mu + sigma * rng.standard_normal())
                .collect();

            let frame = Frame::new(self.n_obs).numeric("y", y);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let cfg = format!(
                r#"{{"value": "y", "prior": {{"mu0": {}, "kappa0": {}, "alpha0": {}, "beta0": {}}}}}"#,
                self.mu0, self.kappa0, self.alpha0, self.beta0
            );
            let model = ConjugateAnomaly.compile(&Config::parse(&cfg).unwrap(), &view)?;

            let sample = ExactEngine.sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws,
                    // Ignored: this engine draws independently, so there is nothing to adapt.
                    n_warmup: 0,
                    // A stream distinct from the simulation's, so the posterior draws
                    // cannot correlate with the noise that produced the data.
                    seed: rng.uniform().to_bits(),
                },
            )?;
            let p = model.param_names().len();
            Ok((0..2)
                .map(|j| sample.values.chunks(p).map(|c| c[j]).collect())
                .collect())
        }
    }

    /// F3 with a proper Gaussian prior on the slopes and no intercept.
    ///
    /// The intercept is deliberately absent: this family never penalises it, so its
    /// prior is flat and improper and there is nothing to draw a truth from. Slopes
    /// and `sigma` carry proper priors and are the parameters a causal-impact
    /// decision actually reads.
    struct F3Slopes {
        n_obs: usize,
        beta_scale: f64,
        a0: f64,
        s0: f64,
        engine: EngineKind,
        /// Keep every `thin`-th draw. `1` for the engines that draw independently.
        ///
        /// SBC's guarantee is that the rank of the truth among `L` **exchangeable**
        /// posterior draws is uniform. NUTS draws are a Markov chain and consecutive
        /// draws are correlated, which biases the rank toward the middle of the
        /// histogram and would make a perfectly correct sampler look under-confident.
        /// Talts et al. §4 prescribe thinning for exactly this reason, and it is a
        /// property of the *sampler*, not of the model, so it lives beside `engine`.
        thin: usize,
    }

    impl SbcModel for F3Slopes {
        fn param_names(&self) -> Vec<String> {
            vec!["beta[x1]".into(), "beta[x2]".into(), "sigma".into()]
        }

        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            // The conjugate form: beta | sigma^2 ~ N(0, sigma^2 * beta_scale^2),
            // which is exactly what adding 1/beta_scale^2 to the normal equations
            // implies.
            let sigma_sq = inv_gamma(rng, self.a0, self.s0)?;
            let sd = (sigma_sq).sqrt() * self.beta_scale;
            Ok(vec![
                sd * rng.standard_normal(),
                sd * rng.standard_normal(),
                sigma_sq.sqrt(),
            ])
        }

        fn simulate_and_fit(
            &self,
            truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            let n = self.n_obs;
            let x1: Vec<f64> = (0..n).map(|_| rng.standard_normal()).collect();
            let x2: Vec<f64> = (0..n).map(|_| rng.standard_normal()).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| truth[0] * x1[i] + truth[1] * x2[i] + truth[2] * rng.standard_normal())
                .collect();

            let frame = Frame::new(n)
                .numeric("y", y)
                .numeric("x1", x1)
                .numeric("x2", x2);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let cfg = format!(
                r#"{{"y": "y", "x": ["x1", "x2"], "intercept": 0,
                     "prior": {{"beta_scale": {}, "a0": {}, "s0": {}}}}}"#,
                self.beta_scale, self.a0, self.s0
            );
            let model = PooledGaussian.compile(&Config::parse(&cfg).unwrap(), &view)?;

            let opts = SampleOptions {
                n_chains: 1,
                // Thinning is done by *drawing* more and keeping every `thin`-th, so
                // the kept draws are `thin` sampler steps apart.
                n_draws: n_draws * self.thin,
                // The shipped default, not a reduced one: certifying a warmup budget
                // nobody uses would certify nothing. Ignored by the other engines.
                n_warmup: crate::engines::DEFAULT_WARMUP,
                seed: rng.uniform().to_bits(),
            };
            let sample = match self.engine {
                EngineKind::Laplace => LaplaceEngine.sample(&*model, &opts)?,
                EngineKind::Nuts => NutsEngine.sample(&*model, &opts)?,
                EngineKind::Exact => ExactEngine.sample(&*model, &opts)?,
            };
            let p = model.param_names().len();
            Ok((0..3)
                .map(|j| {
                    sample
                        .values
                        .chunks(p)
                        .skip(self.thin - 1)
                        .step_by(self.thin)
                        .map(|c| c[j])
                        .collect()
                })
                .collect())
        }
    }

    /// A real fit, narrowed on purpose.
    ///
    /// **The fixture that proves the gate can fail.** A calibration suite that only
    /// ever passes is not a gate, and the ones above would still pass if `run_sbc`
    /// quietly ranked nothing. The unit tests at the top of this module make that
    /// argument with a synthetic posterior; this one makes it through the *whole* NUTS
    /// pipeline — the same simulation, the same compile, the same sampler, the same
    /// ranking — with one deliberate defect injected at the last possible moment.
    ///
    /// The defect is over-confidence: every draw is pulled toward the posterior mean,
    /// so the intervals are too tight. That is the failure mode that matters
    /// commercially — a service-level decision reads the interval, and one that is 40 %
    /// too narrow under-covers silently — and it is the one an accuracy test cannot
    /// see, because the mean is untouched.
    struct Narrowed<M: SbcModel> {
        inner: M,
        /// Multiplier on each parameter's spread about its own posterior mean.
        factor: f64,
    }

    impl<M: SbcModel> SbcModel for Narrowed<M> {
        fn param_names(&self) -> Vec<String> {
            self.inner.param_names()
        }

        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            self.inner.draw_prior(rng)
        }

        fn simulate_and_fit(
            &self,
            truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            let mut draws = self.inner.simulate_and_fit(truth, rng, n_draws)?;
            for column in draws.iter_mut() {
                let mean = column.iter().sum::<f64>() / column.len() as f64;
                for v in column.iter_mut() {
                    *v = mean + self.factor * (*v - mean);
                }
            }
            Ok(draws)
        }
    }

    // Chi-squared 99.9% critical value at 15 degrees of freedom. Deliberately lax:
    // a calibration suite that flakes gets disabled, and a disabled gate protects
    // nobody.
    const CRITICAL_15_DF: f64 = 37.7;
    const REPLICATIONS: u32 = 1024;
    const BINS: usize = 15;

    fn assert_calibrated(hists: &[RankHistogram], label: &str) {
        for h in hists {
            // Printed so a run leaves evidence it did the work. A calibration gate
            // that silently passes because it never executed is worse than no gate.
            println!(
                "{label}/{}: chi2 {:.1} (df {}), slope {:+.3}, n {}",
                h.param,
                h.chi_squared(),
                h.degrees_of_freedom(),
                h.slope(),
                h.n_replications
            );
            assert!(
                h.n_replications == REPLICATIONS,
                "{label}/{}: only {} replications ran",
                h.param,
                h.n_replications
            );
            assert!(
                h.passes(CRITICAL_15_DF),
                "{label}/{}: chi-squared {:.1} exceeds {CRITICAL_15_DF} (slope {:.3}); counts {:?}",
                h.param,
                h.chi_squared(),
                h.slope(),
                h.counts
            );
        }
    }

    #[test]
    #[ignore = "slow: hundreds of complete fits"]
    fn f7_normal_is_calibrated_under_the_exact_engine() {
        let model = F7Normal {
            n_obs: 20,
            mu0: 0.0,
            kappa0: 1.0,
            alpha0: 3.0,
            beta0: 3.0,
        };
        let hists = run_sbc(&model, REPLICATIONS, BINS, 101).unwrap();
        assert_calibrated(&hists, "f7_normal/exact");
    }

    #[test]
    #[ignore = "slow: hundreds of complete fits"]
    fn f3_is_calibrated_under_the_exact_engine() {
        let model = F3Slopes {
            n_obs: 30,
            beta_scale: 1.0,
            a0: 3.0,
            s0: 3.0,
            engine: EngineKind::Exact,
            thin: 1,
        };
        let hists = run_sbc(&model, REPLICATIONS, BINS, 102).unwrap();
        assert_calibrated(&hists, "f3/exact");
    }

    /// The Laplace path certified independently. This is the run that says *where*
    /// the approximation may be used: if it were miscalibrated for this family, the
    /// rank histogram would bow and the family's default engine would have to change.
    #[test]
    #[ignore = "slow: hundreds of complete fits"]
    fn f3_is_calibrated_under_the_laplace_engine() {
        let model = F3Slopes {
            // More data than the exact suite: the Gaussian approximation is
            // asymptotic, and certifying it at n = 30 would be certifying it where
            // nobody should be using it anyway.
            n_obs: 200,
            beta_scale: 1.0,
            a0: 3.0,
            s0: 3.0,
            engine: EngineKind::Laplace,
            thin: 1,
        };
        let hists = run_sbc(&model, REPLICATIONS, BINS, 103).unwrap();
        assert_calibrated(&hists, "f3/laplace");
    }

    /// The NUTS path certified independently, because SBC is per family **per engine**.
    ///
    /// The closed-form and Laplace suites say nothing about this one: the adapter has
    /// its own transform hand-off, its own warmup discard and its own chain layout, and
    /// a mistake in any of them produces draws that are individually plausible and
    /// collectively miscalibrated. This is also the suite that will still be here when
    /// the engine is serving a family with no closed form to check it against, which is
    /// the reason gap 6 exists at all.
    ///
    /// Run at `n_obs = 30`, matching the exact suite rather than the Laplace one: NUTS
    /// makes no asymptotic promise, so there is no reason to certify it only where the
    /// posterior happens to be nearly Gaussian.
    fn f3_under_nuts() -> F3Slopes {
        F3Slopes {
            n_obs: 30,
            beta_scale: 1.0,
            a0: 3.0,
            s0: 3.0,
            engine: EngineKind::Nuts,
            // Every fifth draw. See `F3Slopes::thin`: consecutive NUTS draws are
            // correlated, and SBC ranks assume exchangeability.
            thin: 5,
        }
    }

    #[test]
    #[ignore = "slow: hundreds of complete NUTS fits"]
    fn f3_is_calibrated_under_the_nuts_engine() {
        let hists = run_sbc(&f3_under_nuts(), REPLICATIONS, BINS, 104).unwrap();
        assert_calibrated(&hists, "f3/nuts");
    }

    /// **The suite above, proved to be a gate.**
    ///
    /// The same NUTS pipeline, with every draw pulled 40 % of the way toward its own
    /// posterior mean before ranking. Nothing else changes — same priors, same
    /// simulated data, same sampler, same seed discipline — so a failure here can only
    /// be the injected over-confidence, and a *pass* here would mean
    /// `f3_is_calibrated_under_the_nuts_engine` certifies nothing.
    ///
    /// Asserted as ∪-shaped rather than merely failing: a narrow posterior leaves the
    /// truth outside it too often, so the extreme ranks carry the mass while the slope
    /// stays near zero. Distinguishing that from a *biased* posterior is what makes an
    /// SBC failure report actionable rather than just red.
    #[test]
    #[ignore = "slow: hundreds of complete NUTS fits"]
    fn a_deliberately_overconfident_nuts_posterior_is_rejected() {
        let model = Narrowed {
            inner: f3_under_nuts(),
            factor: 0.6,
        };
        let hists = run_sbc(&model, REPLICATIONS, BINS, 105).unwrap();
        for h in &hists {
            println!(
                "f3/nuts/narrowed/{}: chi2 {:.1}, slope {:+.3}, counts {:?}",
                h.param,
                h.chi_squared(),
                h.slope(),
                h.counts
            );
            assert!(
                !h.passes(CRITICAL_15_DF),
                "{}: a posterior 40% too narrow passed the calibration gate \
                 (chi-squared {:.1}); the gate is not a gate",
                h.param,
                h.chi_squared()
            );
            let bins = h.counts.len();
            assert!(
                h.counts[0] + h.counts[bins - 1] > h.counts[bins / 2 - 1] + h.counts[bins / 2],
                "{}: an over-confident posterior should bow upward at the extremes; counts {:?}",
                h.param,
                h.counts
            );
            assert!(
                h.slope().abs() < 0.2,
                "{}: an over-confident fit should bow, not slope ({:.3})",
                h.param,
                h.slope()
            );
        }
    }
}
