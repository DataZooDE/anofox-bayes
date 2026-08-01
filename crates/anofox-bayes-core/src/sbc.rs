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
        f3_pooled_gaussian::PooledGaussian, f5_btyd::PayerAlive, f7_conjugate::ConjugateAnomaly,
        ModelFamily,
    };
    use crate::config::Config;
    use crate::data::testing::Frame;
    use crate::engines::{Engine, ExactEngine, LaplaceEngine, SampleOptions};
    use crate::errors::BayesError;
    use crate::types::{EngineKind, FitStatus};

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
                n_draws,
                seed: rng.uniform().to_bits(),
            };
            let sample = match self.engine {
                EngineKind::Laplace => LaplaceEngine.sample(&*model, &opts)?,
                _ => ExactEngine.sample(&*model, &opts)?,
            };
            let p = model.param_names().len();
            Ok((0..3)
                .map(|j| sample.values.chunks(p).map(|c| c[j]).collect())
                .collect())
        }
    }

    /// F5 (BG/NBD) under proper log-normal priors on all four population parameters,
    /// served by the Laplace engine.
    ///
    /// **This suite is the arbiter of the roadmap's deferred question** — whether F5
    /// genuinely needs NUTS, or whether a Gaussian approximation at the mode is
    /// adequate for four population parameters informed by a whole customer base. If
    /// these ranks come out uniform, Laplace is certified for this family and NUTS
    /// buys nothing at a large multiple of the runtime. If they bow, the family's
    /// default engine has to change, and that is a finding rather than a test to
    /// loosen.
    ///
    /// The priors are the ones a caller with a thin base would set, and they are
    /// proper because SBC draws the truth from the prior — the same constraint, and
    /// the same remedy, as for the other two families. A proper prior is also what
    /// keeps the boundary solutions of `f5_btyd` out of the loop: a flat prior on
    /// `ln a` is exactly what lets `a` run to zero, so a suite under the default
    /// prior would be measuring the refusal path rather than the posterior.
    struct F5PayerAlive {
        n_customers: usize,
        horizon: f64,
        /// `(log_mean, log_sd)` for `r`, `alpha`, `a`, `b` in that order.
        prior: [(f64, f64); 4],
    }

    impl F5PayerAlive {
        fn config(&self) -> String {
            let names = ["r", "alpha", "a", "b"];
            let slots: Vec<String> = (0..4)
                .map(|j| {
                    format!(
                        r#""{}": {{"log_mean": {}, "log_sd": {}}}"#,
                        names[j], self.prior[j].0, self.prior[j].1
                    )
                })
                .collect();
            format!(
                r#"{{"frequency": "x", "recency": "t_x", "age": "T", "min_customers": 1,
                     "prior": {{{}}}}}"#,
                slots.join(", ")
            )
        }
    }

    impl SbcModel for F5PayerAlive {
        fn param_names(&self) -> Vec<String> {
            vec!["r".into(), "alpha".into(), "a".into(), "b".into()]
        }

        fn draw_prior(&self, rng: &mut BayesRng) -> BayesResult<Vec<f64>> {
            Ok(self
                .prior
                .iter()
                .map(|(m, s)| (m + s * rng.standard_normal()).exp())
                .collect())
        }

        fn simulate_and_fit(
            &self,
            truth: &[f64],
            rng: &mut BayesRng,
            n_draws: usize,
        ) -> BayesResult<Vec<Vec<f64>>> {
            let base = crate::catalog::f5_btyd::testing::simulate(
                rng,
                self.n_customers,
                truth[0],
                truth[1],
                truth[2],
                truth[3],
                self.horizon,
            )?;
            let frame = base.frame();
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let model = PayerAlive.compile(&Config::parse(&self.config()).unwrap(), &view)?;

            // A replication the family refused carries no ranks, and quietly dropping
            // it would bias the histogram toward whatever the surviving replications
            // have in common. Failing loudly is the only honest handling: under a
            // proper prior a refusal means the *prior* admits data the model cannot
            // fit, which is a finding about the suite's own design.
            if model.readiness().status != FitStatus::Converged {
                return Err(BayesError::Internal(format!(
                    "SBC replication refused by the family ({:?}); truth {truth:?}",
                    model.readiness().reasons
                )));
            }

            let sample = LaplaceEngine.sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws,
                    seed: rng.uniform().to_bits(),
                },
            )?;
            let p = model.param_names().len();
            Ok((0..4)
                .map(|j| sample.values.chunks(p).map(|c| c[j]).collect())
                .collect())
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
        };
        let hists = run_sbc(&model, REPLICATIONS, BINS, 103).unwrap();
        assert_calibrated(&hists, "f3/laplace");
    }

    /// The run that decides whether `payer_alive` may ship on Laplace at all.
    ///
    /// 800 customers is a small base by the standards of the agent this family serves
    /// — a collections book is tens of thousands — and certifying at the small end is
    /// the point: if the approximation holds where the data is thinnest, it holds
    /// above that too.
    #[test]
    #[ignore = "slow: hundreds of complete fits"]
    fn f5_is_calibrated_under_the_laplace_engine() {
        let model = F5PayerAlive {
            n_customers: 800,
            horizon: 52.0,
            prior: [(0.0, 0.4), (2.5, 0.4), (0.0, 0.4), (1.0, 0.4)],
        };
        let hists = run_sbc(&model, REPLICATIONS, BINS, 105).unwrap();
        assert_calibrated(&hists, "f5/laplace");
    }
}
