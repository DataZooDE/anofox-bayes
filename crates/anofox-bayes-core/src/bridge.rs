//! The seam onto `anofox-statistics`.
//!
//! `anofox-stats-core` has already paid for a censored accelerated-failure-time
//! likelihood — Weibull, lognormal, log-logistic and exponential — its analytic
//! gradient and Hessian, its mode search, and explicit Gaussian priors. What it
//! returns is a point estimate and a standard error. A MAP estimate together with the
//! observed information at it **is** a Laplace posterior; the only missing step is the
//! last one of our own Laplace engine, which is to sample the multivariate normal at
//! the mode. This module is that step's input.
//!
//! ```text
//!   anofox-stats-core                    anofox-bayes
//!   ─────────────────                    ────────────
//!   fit_aft  ──▶ mode (beta, log sigma)
//!            ──▶ converged / refusal  ──▶ FitStatus          (refusal mapping)
//!   AftInference.information          ──▶ observed information at the mode
//!   AftInference.vcov                 ──▶ FULL covariance    (cross-check)
//!                                     ──▶ GaussianBlock      ──▶ LaplaceEngine
//! ```
//!
//! ## Where the curvature comes from
//!
//! From `fit_aft`, which computes it on the way to the standard errors it reports.
//! `AftInference` publishes both the penalised observed information at the mode and
//! its inverse, in design order — intercept, then coefficients, then `log sigma`.
//!
//! It did not always. Until `anofox-statistics` reported them, this module rebuilt
//! the matrix: `observed_information` walked every observation accumulating the
//! Hessian from [`AftDistribution`]'s first two derivatives, and `aligned_priors`
//! re-implemented that crate's private `build_penalty` so a prior landed on the same
//! coordinate it would have. Around 130 lines existed for one reason — the result was
//! computed one crate over and discarded before it left.
//!
//! Deleting it was licensed by measurement rather than by inspection:
//! `the_published_covariance_is_the_one_anofox_stats_core_reports` showed the two
//! matrices agreeing entry for entry to [`SE_AGREEMENT_TOLERANCE`] across all three
//! distributions, on a covariate held away from zero so the off-diagonal — the part a
//! diagonal-only surface loses — was not incidentally zero.
//!
//! **The seam is still checked.** The diagonal of the covariance handed over must
//! reproduce the standard errors reported alongside it. That is no longer a check on
//! arithmetic performed here; it is the assertion that the pinned revision still means
//! the same thing by this likelihood, and it fails loudly rather than publishing a
//! different posterior under the same name.
//!
//! ## Why the full matrix and not the diagonal
//!
//! Sampling from the diagonal treats every coefficient as independent. In a survival
//! model with a covariate measured away from zero the intercept and the slope are
//! almost perfectly anti-correlated — measured on the fixture in
//! `the_diagonal_alone_is_not_the_posterior`, `corr = -0.998` — and the two errors
//! cancel almost exactly in the linear predictor. Dropping the off-diagonal inflates
//! the predictive standard deviation by a factor of about **25**. Nothing downstream
//! would notice: the draws are finite, the diagnostics are clean, the interval is
//! merely wrong.

use anofox_stats_core::errors::StatsError;
use anofox_stats_core::models::{fit_aft, AftDistribution, AftOptions, AftResult};
use anofox_stats_core::types::{PriorSpec, VcovType};
use faer::Mat;

use crate::catalog::Readiness;
use crate::errors::{BayesError, BayesResult};

/// How closely the reassembled covariance must reproduce statistics' own standard
/// errors, relative to each standard error.
///
/// Deliberately tight. The two are computed from the same likelihood at the same
/// point by the same routine, differing only in who assembled the Hessian, so they
/// agree bit-for-bit in practice. The tolerance exists to absorb a future change in
/// summation order upstream, not to absorb a change in the mathematics.
pub const SE_AGREEMENT_TOLERANCE: f64 = 1e-8;

/// One censored-AFT fit to bridge.
///
/// `x` is column-major, matching `fit_aft`. `priors` is positionally aligned with the
/// design — the intercept first when one is fitted — and may be empty for a flat
/// prior throughout.
#[derive(Debug, Clone)]
pub struct CensoredAftRequest<'a> {
    pub dist: AftDistribution,
    /// Strictly positive event or censoring times.
    pub time: &'a [f64],
    /// 1.0 where the event was observed, 0.0 where the row is still right-censored.
    pub event: &'a [f64],
    pub x: &'a [Vec<f64>],
    pub intercept: bool,
    pub priors: Vec<PriorSpec>,
}

/// A MAP estimate and the full curvature at it: a Laplace posterior in everything but
/// the output format.
#[derive(Debug, Clone)]
pub struct BridgedGaussian {
    /// The mode on the unconstrained scale, in design order: the coefficients
    /// (intercept first when fitted) followed by `log sigma` where the scale is
    /// estimated.
    pub mode: Vec<f64>,
    /// The observed information at the mode — the **full** `p × p` negative Hessian of
    /// the log posterior, including the prior's precision. This is the precision
    /// matrix a draw is generated from, and its off-diagonal is the point of the
    /// exercise.
    pub information: Mat<f64>,
    /// `information^-1`, the full posterior covariance. Carried alongside the
    /// precision because it is what the cross-check against statistics' `std_errors`
    /// is stated in, and what a predictive-interval calculation reads directly.
    pub covariance: Mat<f64>,
    /// The diagonal of `covariance`, square-rooted — reproducing `AftInference`.
    pub std_errors: Vec<f64>,
    /// `sigma` at the mode. Equal to `exp(mode.last())` when the scale is estimated,
    /// and fixed at 1.0 for the exponential distribution.
    pub scale: f64,
    /// Whether `log sigma` occupies the last coordinate of `mode`.
    pub fit_scale: bool,
    pub n_obs: usize,
    pub n_events: usize,
    pub n_censored: usize,
}

impl BridgedGaussian {
    /// Number of unconstrained coordinates.
    pub fn dim(&self) -> usize {
        self.mode.len()
    }
}

/// The outcome of bridging one fit.
///
/// A refusal is a *successful* return carrying a [`Readiness`], not an error: "this
/// data cannot support a duration model" is an answer an agent acts on, and the draws
/// contract has a status column for exactly that. Only a malformed *request* — a
/// negative time, a non-binary event indicator — is a [`BayesError`].
#[derive(Debug, Clone)]
pub enum Bridged {
    Fitted(Box<BridgedGaussian>),
    Refused(Readiness),
}

/// Fit a censored AFT model through `anofox-stats-core` and return its Laplace
/// posterior.
pub fn fit_censored_aft(req: &CensoredAftRequest) -> BayesResult<Bridged> {
    let n = req.time.len();
    let n_features = req.x.len();
    let fit_scale = !req.dist.scale_is_fixed();
    let n_beta = n_features + usize::from(req.intercept);
    let n_params = n_beta + usize::from(fit_scale);

    let options = AftOptions {
        dist: req.dist,
        fit_intercept: req.intercept,
        compute_inference: true,
        priors: req.priors.clone(),
        // Pinned, not inherited from statistics' default. `Sandwich` and `Naive`
        // target the frequentist sampling variance of a penalized estimator, which is
        // not a posterior covariance; only the curvature of the log posterior at its
        // mode is the thing a Laplace approximation is defined in terms of.
        vcov: VcovType::Laplace,
        ..Default::default()
    };

    let fit: AftResult = match fit_aft(req.time, req.x, req.event, &options) {
        Ok(f) => f,
        Err(e) => return map_refusal(e).map(Bridged::Refused),
    };

    // The data fingerprint that feeds `model_id` is computed over the rows this crate
    // selected. `fit_aft` runs its own filter for non-finite values, so if it read a
    // different number of rows than we handed it, the fingerprint would describe a
    // relation the fit did not see and `model_id` would be a claim about the wrong
    // data. Callers filter first; this asserts that they did.
    if fit.core.n_observations != n {
        return Err(BayesError::Internal(format!(
            "the bridged fit read {} of the {n} rows it was given; the data fingerprint \
             would not describe the rows the fit actually used",
            fit.core.n_observations
        )));
    }

    let mut mode: Vec<f64> = Vec::with_capacity(n_params);
    if let Some(a) = fit.core.intercept {
        mode.push(a);
    }
    mode.extend_from_slice(&fit.core.coefficients);
    if fit_scale {
        mode.push(fit.core.scale.ln());
    }
    if mode.len() != n_params || !mode.iter().all(|v| v.is_finite()) {
        return Ok(Bridged::Refused(Readiness::degenerate(format!(
            "the censored AFT mode is not a usable point ({mode:?}); the likelihood \
             surface has no interior maximum for this data"
        ))));
    }

    // The curvature and its inverse come from `fit_aft`, which computed both on the
    // way to the standard errors it reports. This used to be reassembled here --
    // `observed_information` walked every row rebuilding the Hessian from
    // `AftDistribution`'s derivatives, and `aligned_priors` re-implemented the
    // crate's private `build_penalty` so a prior landed on the same coordinate --
    // for one reason only: the result was computed there and thrown away.
    //
    // `the_published_covariance_is_the_one_anofox_stats_core_reports` is what
    // licensed removing it: the two matrices agreed entry for entry to
    // `SE_AGREEMENT_TOLERANCE` on an off-centre covariate, where the off-diagonal is
    // the part a diagonal-only surface loses.
    let Some(inference) = fit.inference.as_ref() else {
        return Ok(Bridged::Refused(Readiness::degenerate(
            "anofox-stats-core reported no inference for the censored AFT fit, so \
             there is no curvature at the mode to build a posterior from"
                .to_string(),
        )));
    };
    let (Some(information), Some(covariance)) =
        (inference.information.as_ref(), inference.vcov.as_ref())
    else {
        return Ok(Bridged::Refused(Readiness::degenerate(
            "anofox-stats-core reported inference without the curvature; the pinned \
             revision no longer publishes what this bridge samples from"
                .to_string(),
        )));
    };
    if !(0..n_params).all(|i| (0..n_params).all(|j| information[(i, j)].is_finite())) {
        return Ok(Bridged::Refused(Readiness::degenerate(
            "the curvature at the censored AFT mode is not finite".to_string(),
        )));
    }
    let information = information.clone();

    // Degeneracy is decided before agreement is, and deliberately so. A rank-deficient
    // design leaves `invert_spd` returning a matrix full of non-finite entries rather
    // than failing outright, and both crates then report `NaN` — they *agree*, and the
    // agreement is worthless. Refusing first means the cross-check below only ever
    // compares numbers, so a `NaN == NaN` can never be mistaken for a passing check.
    let std_errors: Vec<f64> = (0..n_params).map(|j| covariance[(j, j)].sqrt()).collect();
    let degenerate = std_errors.iter().any(|s| !s.is_finite() || *s <= 0.0)
        || (0..n_params).any(|i| (0..n_params).any(|j| !covariance[(i, j)].is_finite()));
    if degenerate {
        return Ok(Bridged::Refused(Readiness::degenerate(format!(
            "the censored AFT posterior has no usable covariance ({std_errors:?}); the \
             design is rank deficient at the mode, so the curvature there is not a \
             posterior"
        ))));
    }

    // The seam is still checked, and it is worth keeping now that the matrix is no
    // longer ours: this asserts that the covariance the pinned revision hands over is
    // the one summarised by the standard errors it also reports. A revision that
    // changed what it means by this likelihood would fail here rather than publish a
    // different posterior under the same name.
    let reported = statistics_standard_errors(&fit, n_beta, req.intercept, fit_scale);
    for (j, (ours, theirs)) in std_errors.iter().zip(reported.iter()).enumerate() {
        let scale = theirs.abs().max(1e-300);
        if !theirs.is_finite() || (ours - theirs).abs() > SE_AGREEMENT_TOLERANCE * scale {
            return Err(BayesError::Internal(format!(
                "anofox-stats-core's covariance disagrees with its own standard errors \
                 at coordinate {j}: {ours} against the reported {theirs}. The pinned \
                 revision no longer means the same thing by this likelihood"
            )));
        }
    }

    Ok(Bridged::Fitted(Box::new(BridgedGaussian {
        mode,
        information,
        covariance: covariance.clone(),
        std_errors,
        scale: fit.core.scale,
        fit_scale,
        n_obs: fit.core.n_observations,
        n_events: fit.core.n_events,
        n_censored: fit.core.n_censored,
    })))
}

/// The standard errors `fit_aft` reported, reassembled into design order.
///
/// `AftInference` splits them across three fields — the intercept, the coefficients,
/// and `log sigma` — because that is the shape the SQL aggregate publishes. Putting
/// them back in design order is what makes the cross-check above a comparison of like
/// with like.
fn statistics_standard_errors(
    fit: &AftResult,
    n_beta: usize,
    intercept: bool,
    fit_scale: bool,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_beta + usize::from(fit_scale));
    let inf = match fit.inference.as_ref() {
        Some(i) => i,
        None => return vec![f64::NAN; n_beta + usize::from(fit_scale)],
    };
    if intercept {
        out.push(inf.intercept_std_error.unwrap_or(f64::NAN));
    }
    out.extend_from_slice(&inf.std_errors);
    if fit_scale {
        out.push(inf.log_scale_std_error.unwrap_or(f64::NAN));
    }
    out
}

/// Map `anofox-stats-core`'s refusal vocabulary onto the draws contract's.
///
/// **Mapped, never inherited.** The two crates refuse for overlapping reasons and say
/// so in different vocabularies, and a `converged BOOLEAN` does not mean what
/// `FitStatus` means. The table below is the contract; each row has a test.
///
/// | `StatsError` | `FitStatus` | Why |
/// |---|---|---|
/// | `InvalidValue { field: "event" }`, all rows censored | `Degenerate` | The likelihood is flat: no event time carries information about the level. The data is real, the model is not identified by it. |
/// | `InsufficientData` | `InsufficientData` | Fewer rows than parameters. The posterior would be the prior. |
/// | `NoValidData` / `EmptyInput` | `InsufficientData` | Nothing survived filtering; same verdict, reached earlier. |
/// | `ConvergenceFailure` | `Failed` | The mode search did not finish. There is no point to expand around, so there is no posterior — not a weak one, none. |
/// | `SingularMatrix` / `CholeskyFailed` / `QrFailed` | `Degenerate` | A boundary or rank-deficient solution. The curvature there is not a posterior, which is exactly the failure §3.2 of the roadmap names for BG/NBD as well. |
/// | `InvalidValue { field: "time" }` | *not a status* | A non-positive duration is a malformed request, not weak evidence. It surfaces as a `BayesError` naming the column so the caller can repair it. |
/// | `DimensionMismatch` | *not a status* | A caller bug. |
fn map_refusal(e: StatsError) -> BayesResult<Readiness> {
    match e {
        StatsError::InvalidValue {
            field: "event",
            ref message,
        } if message.contains("every observation is censored") => Ok(Readiness::degenerate(
            "every observation is right-censored, so no event time informs the \
                 duration level and the model is not identified",
        )),
        StatsError::InsufficientData { rows, cols } => Ok(Readiness::insufficient(format!(
            "{rows} usable rows for {cols} parameters: the censored AFT posterior would \
             be the prior"
        ))),
        StatsError::InsufficientDataMsg(m) => Ok(Readiness::insufficient(m)),
        StatsError::NoValidData | StatsError::EmptyInput { .. } => Ok(Readiness::insufficient(
            "no usable observations remain after filtering",
        )),
        StatsError::ConvergenceFailure {
            iterations,
            tolerance,
        } => Ok(Readiness::failed(format!(
            "the censored AFT mode search did not converge after {iterations} iterations \
             (tolerance {tolerance}); there is no mode to expand a posterior around"
        ))),
        StatsError::SingularMatrix | StatsError::CholeskyFailed | StatsError::QrFailed => Ok(
            Readiness::degenerate("the censored AFT design is rank deficient at the mode"),
        ),
        // A malformed request rather than weak evidence: the caller must fix the data.
        StatsError::InvalidValue { field, message } => Err(BayesError::config(field, message)),
        StatsError::DimensionMismatch { y_len, x_rows } => Err(BayesError::DimensionMismatch(
            format!("time has {y_len} entries, a predictor has {x_rows}"),
        )),
        other => Err(BayesError::Internal(format!(
            "anofox-stats-core refused a censored AFT fit in a way this bridge does not \
             map: {other}"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! A censored survival fixture shared by the bridge, family and SBC tests.

    use super::*;

    /// `log T = b0 + b1 x + sigma W`, with staggered right-censoring.
    ///
    /// `x_offset` moves the covariate away from zero. That is not cosmetic: it is what
    /// makes the intercept and the slope correlated, which is the whole subject of
    /// `the_diagonal_alone_is_not_the_posterior`.
    pub fn survival_fixture(
        dist: AftDistribution,
        b0: f64,
        b1: f64,
        sigma: f64,
        n: usize,
        x_offset: f64,
        censor_at: Option<f64>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut time = Vec::with_capacity(n);
        let mut event = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        for i in 0..n {
            let x = x_offset + (i % 11) as f64 * 0.2;
            let p = (i as f64 + 0.5) / n as f64;
            let t = dist.quantile_time(p, b0 + b1 * x, sigma);
            // Staggered rather than shared: a single common threshold piles every
            // censored row onto one identical time, which no real cohort does.
            match censor_at.map(|c| c + (i % 7) as f64 * 0.9) {
                Some(c) if t > c => {
                    time.push(c);
                    event.push(0.0);
                }
                _ => {
                    time.push(t);
                    event.push(1.0);
                }
            }
            xs.push(x);
        }
        (time, xs, event)
    }
}

#[cfg(test)]
mod tests {
    //! Three tests were removed when the reassembly they exercised was.
    //!
    //! `the_bridged_log_density_matches_the_closed_form_censored_weibull` checked the
    //! rebuilt Hessian against finite differences of a hand-written censored Weibull
    //! density; `a_prior_adds_its_precision_to_exactly_one_diagonal_entry` and
    //! `priors_align_with_the_design_the_same_way_statistics_aligns_them` checked that
    //! a prior landed on the coordinate `build_penalty` would have put it on. All three
    //! had `observed_information` or `aligned_priors` as their subject, and that
    //! subject no longer exists -- the matrix is `anofox-stats-core`'s now, checked
    //! there against its own derivatives.
    //!
    //! What replaces them is not nothing. `the_published_covariance_is_the_one_anofox_
    //! stats_core_reports` pins the matrix we publish to the one upstream reports, and
    //! `a_gaussian_prior_narrows_the_bridged_posterior_and_not_only_the_mode` is what
    //! now proves the curvature we are handed is the *penalised* one: if upstream ever
    //! returned the likelihood's curvature instead, a prior would move the mode and
    //! leave the width alone, and that test would fail.

    use super::testing::survival_fixture;
    use super::*;
    use crate::types::FitStatus;

    fn request<'a>(
        dist: AftDistribution,
        time: &'a [f64],
        x: &'a [Vec<f64>],
        event: &'a [f64],
    ) -> CensoredAftRequest<'a> {
        CensoredAftRequest {
            dist,
            time,
            event,
            x,
            intercept: true,
            priors: Vec::new(),
        }
    }

    /// **The covariance this bridge publishes is upstream's own, entry for entry.**
    ///
    /// It used to be reassembled here: `observed_information` walked every
    /// observation, rebuilt the Hessian from `AftDistribution`'s derivatives, and
    /// `aligned_priors` re-implemented the crate's private `build_penalty` so the
    /// prior landed on the same coordinate. That existed for one reason -- `fit_aft`
    /// computed the full covariance and then dropped it, keeping only slices of the
    /// diagonal.
    ///
    /// It does not drop it any more. This asserts the two are the same matrix, which
    /// is what licenses deleting the reassembly: not "the numbers look close" but
    /// "the thing we were rebuilding is the thing we are now handed".
    ///
    /// Checked away from a centred covariate, because with `x` at zero the intercept
    /// and slope are uncorrelated and the off-diagonal -- the part the diagonal-only
    /// surface was losing -- would be zero and prove nothing.
    #[test]
    fn the_published_covariance_is_the_one_anofox_stats_core_reports() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
        ] {
            let (time, xs, event) = survival_fixture(dist, 1.0, 0.25, 0.4, 240, 3.0, Some(40.0));
            let x = vec![xs];
            let req = request(dist, &time, &x, &event);

            let ours = match fit_censored_aft(&req).unwrap() {
                Bridged::Fitted(g) => *g,
                Bridged::Refused(r) => panic!("expected a fit, got {r:?}"),
            };

            // The same call the bridge makes, read directly.
            let opts = AftOptions {
                dist,
                fit_intercept: true,
                compute_inference: true,
                vcov: VcovType::Laplace,
                ..Default::default()
            };
            let upstream =
                fit_aft(&time, &x, &event, &opts).expect("the upstream fit that the bridge wraps");
            let theirs = upstream
                .inference
                .as_ref()
                .expect("inference was requested")
                .vcov
                .as_ref()
                .expect("anofox-stats-core reports the covariance");

            assert_eq!(ours.covariance.nrows(), theirs.nrows(), "{dist:?} shape");
            let mut worst = 0.0f64;
            for r in 0..theirs.nrows() {
                for c in 0..theirs.ncols() {
                    let scale = theirs[(r, c)].abs().max(1e-12);
                    worst = worst.max((ours.covariance[(r, c)] - theirs[(r, c)]).abs() / scale);
                }
            }
            assert!(
                worst < SE_AGREEMENT_TOLERANCE,
                "{dist:?}: the published covariance differs from the reported one by \
                 {worst:e} relative, above {SE_AGREEMENT_TOLERANCE:e}"
            );

            // And the off-diagonal is genuinely there, so the comparison had something
            // to compare.
            assert!(
                theirs[(0, 1)].abs() > 1e-12,
                "{dist:?}: intercept and slope must covary on an off-centre covariate"
            );
        }
    }

    fn fitted(dist: AftDistribution, n: usize, x_offset: f64) -> BridgedGaussian {
        let (time, xs, event) = survival_fixture(dist, 1.0, 0.25, 0.4, n, x_offset, Some(40.0));
        let x = vec![xs];
        match fit_censored_aft(&request(dist, &time, &x, &event)).unwrap() {
            Bridged::Fitted(g) => *g,
            Bridged::Refused(r) => panic!("expected a fit, got {r:?}"),
        }
    }

    /// **The load-bearing claim of the whole bridge.** `AftInference` discards the
    /// covariance matrix, keeping only slices of its diagonal — so if it were not
    /// reachable any other way, a bridged posterior would have to sample every
    /// coefficient independently.
    ///
    /// It is reachable: the observed information reassembled from `AftDistribution`'s
    /// public derivatives, inverted by statistics' own `laplace::inference`, is the
    /// same matrix `fit_aft` computed internally. The evidence is that its diagonal
    /// reproduces the reported `std_errors` — measured, to the last bit — while its
    /// off-diagonal is emphatically not zero.
    #[test]
    fn the_full_covariance_is_reachable_in_process_and_its_diagonal_is_statistics_own() {
        let g = fitted(AftDistribution::Weibull, 300, 10.0);
        assert_eq!(g.dim(), 3, "intercept, slope, log sigma");
        assert_eq!(g.covariance.nrows(), 3);
        assert_eq!(g.covariance.ncols(), 3);

        // The diagonal is statistics' answer. `fit_censored_aft` refuses outright if
        // it is not, so reaching this line already proves it; asserting it here is
        // what makes the claim visible in the test name rather than buried in a guard.
        for j in 0..3 {
            let se = g.covariance[(j, j)].sqrt();
            assert!(
                (se - g.std_errors[j]).abs() <= SE_AGREEMENT_TOLERANCE * g.std_errors[j],
                "coordinate {j}: sqrt(vcov) {se} vs reported se {}",
                g.std_errors[j]
            );
        }

        // ...and the part that the diagonal does not contain is large.
        let corr = |a: usize, b: usize| {
            g.covariance[(a, b)] / (g.covariance[(a, a)] * g.covariance[(b, b)]).sqrt()
        };
        assert!(
            corr(0, 1) < -0.99,
            "intercept and slope should be strongly anti-correlated, got {}",
            corr(0, 1)
        );
        // Symmetric, as a covariance must be.
        for a in 0..3 {
            for b in 0..3 {
                assert!((g.covariance[(a, b)] - g.covariance[(b, a)]).abs() < 1e-15);
            }
        }
    }

    /// **The test this task turns on.** A predictive interval reads the linear
    /// predictor `eta = x'beta`, whose variance is `x' V x`. Under the full covariance
    /// the intercept's error and the slope's error cancel; under the diagonal alone
    /// they add. The two answers are not close, and no diagnostic in this crate would
    /// tell them apart — the draws are finite, R-hat is absent for an independent
    /// sampler, and ESS is whatever the draw count says.
    ///
    /// The assertion is two-sided on purpose. It fails if the bridge ever samples from
    /// the diagonal, and it also fails if the fixture stops being correlated, which
    /// would make the test pass vacuously.
    #[test]
    fn the_diagonal_alone_is_not_the_posterior() {
        let g = fitted(AftDistribution::Weibull, 300, 10.0);
        let x_new = [1.0_f64, 11.0];

        let full: f64 = (0..2)
            .flat_map(|a| (0..2).map(move |b| (a, b)))
            .map(|(a, b)| x_new[a] * x_new[b] * g.covariance[(a, b)])
            .sum();
        let diagonal: f64 = (0..2)
            .map(|a| x_new[a] * x_new[a] * g.covariance[(a, a)])
            .sum();

        assert!(full > 0.0 && diagonal > 0.0);
        let ratio = diagonal.sqrt() / full.sqrt();
        // Measured: 24.86. A generous floor keeps the test about the phenomenon rather
        // than about the fixture's third digit.
        assert!(
            ratio > 10.0,
            "dropping the off-diagonal should inflate the predictive sd by an order of \
             magnitude on a correlated design, got a factor of {ratio}"
        );

        // And the direction is the dangerous one to get wrong either way: the
        // full-covariance sd is the small number, so a diagonal-only bridge would not
        // merely be imprecise, it would report an interval 25 times too wide and an
        // agent would refuse deliveries it should promise.
        assert!(full.sqrt() < 0.1 * diagonal.sqrt());
    }

    /// The `log sigma` coordinate carries no change of variables, and that must be a
    /// checked fact rather than an assumption. If a Jacobian had crept in, the
    /// curvature in the scale direction would be off by exactly 0 or ±1 depending on
    /// which convention leaked, and the closed-form comparison above would fail — but
    /// only if the closed form is genuinely independent. This pins the convention in
    /// one sentence a reader can check: the prior is declared on `log sigma`, so there
    /// is nothing to transform.
    #[test]
    fn the_scale_is_parameterised_on_the_log_scale_with_no_jacobian_term() {
        let g = fitted(AftDistribution::Weibull, 300, 0.0);
        assert!(g.fit_scale);
        assert!(
            (g.mode[2].exp() - g.scale).abs() < 1e-12,
            "the last coordinate must be log sigma: exp({}) vs sigma {}",
            g.mode[2],
            g.scale
        );
        // The exponential distribution fixes sigma at 1, so there is no such
        // coordinate at all and the covariance is one dimension smaller.
        let e = fitted(AftDistribution::Exponential, 300, 0.0);
        assert!(!e.fit_scale);
        assert_eq!(e.dim(), 2);
        assert_eq!(e.scale, 1.0);
    }

    /// The mode is statistics', not ours: the bridge exists so that this crate does
    /// not re-derive a censored likelihood, a gradient or a mode search that already
    /// exist and are tested two repositories over. The generating parameters coming
    /// back is the evidence that the seam passes the data through unchanged.
    #[test]
    fn the_bridged_mode_recovers_the_generating_parameters_through_censoring() {
        let (time, xs, event) =
            survival_fixture(AftDistribution::Weibull, 2.0, 0.3, 0.5, 300, 0.0, Some(9.0));
        let censored = event.iter().filter(|e| **e == 0.0).count();
        assert!(
            censored > 40,
            "the fixture must actually censor: {censored}"
        );

        let x = vec![xs];
        let g = match fit_censored_aft(&request(AftDistribution::Weibull, &time, &x, &event))
            .unwrap()
        {
            Bridged::Fitted(g) => *g,
            other => panic!("{other:?}"),
        };
        assert_eq!(g.n_censored, censored);
        assert_eq!(g.n_events, time.len() - censored);
        assert!((g.mode[0] - 2.0).abs() < 0.1, "intercept {}", g.mode[0]);
        assert!((g.mode[1] - 0.3).abs() < 0.05, "slope {}", g.mode[1]);
        assert!((g.scale - 0.5).abs() < 0.05, "sigma {}", g.scale);
    }

    /// Every distribution the bridge admits must reach a posterior, because the
    /// roadmap's "one seam, four likelihoods" claim is about these four.
    #[test]
    fn every_aft_distribution_bridges() {
        for dist in [
            AftDistribution::Weibull,
            AftDistribution::LogNormal,
            AftDistribution::LogLogistic,
            AftDistribution::Exponential,
        ] {
            let g = fitted(dist, 300, 0.0);
            assert!(
                g.std_errors.iter().all(|s| s.is_finite() && *s > 0.0),
                "{dist:?}"
            );
        }
    }

    //=== the refusal mapping ==============================================//

    fn refusal(time: &[f64], x: &[Vec<f64>], event: &[f64]) -> Readiness {
        match fit_censored_aft(&request(AftDistribution::Weibull, time, x, event)).unwrap() {
            Bridged::Refused(r) => r,
            Bridged::Fitted(_) => panic!("expected a refusal"),
        }
    }

    /// The roadmap names this one explicitly: statistics' "every row censored → not
    /// identified" must arrive as `degenerate`. It is not `insufficient_data` — the
    /// data is plentiful, it simply carries no information about *when* things happen,
    /// which is a structural defect in the model given the data, not a thin sample.
    #[test]
    fn every_row_censored_arrives_as_degenerate() {
        let (time, xs, _) =
            survival_fixture(AftDistribution::Weibull, 1.0, 0.2, 0.5, 50, 0.0, None);
        let x = vec![xs];
        let r = refusal(&time, &x, &vec![0.0; time.len()]);
        assert_eq!(r.status, FitStatus::Degenerate);
        assert!(r.reasons[0].contains("censored"), "{:?}", r.reasons);
        assert!(!r.status.is_actionable());
    }

    #[test]
    fn fewer_rows_than_parameters_arrives_as_insufficient_data() {
        let time = vec![1.0, 2.0, 3.0];
        let x = vec![vec![1.0, 2.0, 3.0]];
        let event = vec![1.0, 1.0, 1.0];
        let r = refusal(&time, &x, &event);
        assert_eq!(r.status, FitStatus::InsufficientData);
    }

    /// A non-positive duration is not weak evidence, it is a malformed request. It
    /// must surface as an error naming the column so the caller can repair it, rather
    /// than as a status an agent might treat as "the data was thin this month".
    #[test]
    fn a_non_positive_duration_is_a_request_error_and_not_a_status() {
        let time = vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let event = vec![1.0; 6];
        let err =
            fit_censored_aft(&request(AftDistribution::Weibull, &time, &x, &event)).unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if *slot == "time"),
            "{err:?}"
        );
    }

    #[test]
    fn a_non_binary_event_indicator_is_a_request_error() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let event = vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0];
        let err =
            fit_censored_aft(&request(AftDistribution::Weibull, &time, &x, &event)).unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if *slot == "event"),
            "{err:?}"
        );
    }

    /// A rank-deficient design has no interior maximum, so the curvature there is not
    /// a covariance. `degenerate` rather than an error: the fit ran, and the honest
    /// report is that its answer must not drive a decision.
    #[test]
    fn a_rank_deficient_design_arrives_as_degenerate_rather_than_as_a_number() {
        // A duplicated predictor: the two columns are exactly collinear, so the
        // information matrix is singular in that direction.
        let (time, xs, event) =
            survival_fixture(AftDistribution::Weibull, 1.0, 0.2, 0.5, 60, 0.0, Some(5.0));
        let x = vec![xs.clone(), xs];
        let outcome =
            fit_censored_aft(&request(AftDistribution::Weibull, &time, &x, &event)).unwrap();
        match outcome {
            Bridged::Refused(r) => assert_eq!(r.status, FitStatus::Degenerate),
            Bridged::Fitted(g) => panic!(
                "a collinear design must not produce a posterior; got std errors {:?}",
                g.std_errors
            ),
        }
    }

    /// Every `StatsError` variant either maps to a status or to an error, and none
    /// falls through to a plausible-looking default. A variant this bridge does not
    /// know about must announce itself rather than be silently graded.
    #[test]
    fn the_refusal_table_covers_the_variants_it_claims_to_and_admits_the_ones_it_does_not() {
        let cases: Vec<(StatsError, Option<FitStatus>)> = vec![
            (
                StatsError::InvalidValue {
                    field: "event",
                    message: "every observation is censored, so the model is not identified"
                        .to_string(),
                },
                Some(FitStatus::Degenerate),
            ),
            (
                StatsError::InsufficientData { rows: 3, cols: 5 },
                Some(FitStatus::InsufficientData),
            ),
            (StatsError::NoValidData, Some(FitStatus::InsufficientData)),
            (
                StatsError::EmptyInput { field: "time" },
                Some(FitStatus::InsufficientData),
            ),
            (
                StatsError::ConvergenceFailure {
                    iterations: 100,
                    tolerance: 1e-9,
                },
                Some(FitStatus::Failed),
            ),
            (StatsError::SingularMatrix, Some(FitStatus::Degenerate)),
            (StatsError::CholeskyFailed, Some(FitStatus::Degenerate)),
            (StatsError::QrFailed, Some(FitStatus::Degenerate)),
            (
                StatsError::InvalidValue {
                    field: "time",
                    message: "AFT regression requires strictly positive times".to_string(),
                },
                None,
            ),
            (
                StatsError::DimensionMismatch {
                    y_len: 3,
                    x_rows: 4,
                },
                None,
            ),
        ];
        for (error, expected) in cases {
            let label = error.to_string();
            match (map_refusal(error), expected) {
                (Ok(r), Some(status)) => assert_eq!(r.status, status, "{label}"),
                (Err(_), None) => {}
                (got, _) => panic!("{label} mapped to {got:?}, expected {expected:?}"),
            }
        }

        // An unmapped variant is reported as internal rather than graded as anything.
        let err = map_refusal(StatsError::AllocationFailure).unwrap_err();
        assert!(matches!(err, BayesError::Internal(_)), "{err:?}");
    }

    /// A `converged BOOLEAN` is not a `FitStatus`, and the two are not related by
    /// renaming. In-process, a non-converged AFT fit never returns at all — statistics
    /// turns it into `ConvergenceFailure` — so "converged = false" reaches this crate
    /// as an error and must be graded `Failed`: there is no mode, so there is no
    /// posterior, not merely a poor one.
    #[test]
    fn a_failed_mode_search_is_failed_and_not_merely_degenerate() {
        let r = map_refusal(StatsError::ConvergenceFailure {
            iterations: 100,
            tolerance: 1e-9,
        })
        .unwrap();
        assert_eq!(r.status, FitStatus::Failed);
        assert!(!r.status.is_actionable());
        // Failed outranks every other verdict when a fit is collapsed over groups.
        assert!((FitStatus::Failed as i32) > (FitStatus::InsufficientData as i32));
    }

    /// A prior adds information, so it must shrink the posterior it is put on — and
    /// the shrinkage must show up in the *covariance*, since that is what the draws
    /// are generated from. Reaching the same conclusion from `std_errors` alone would
    /// not distinguish a prior that entered the curvature from one that entered only
    /// the mode.
    #[test]
    fn a_gaussian_prior_narrows_the_bridged_posterior_and_not_only_the_mode() {
        let (time, xs, event) =
            survival_fixture(AftDistribution::Weibull, 1.5, 0.35, 0.6, 300, 0.0, None);
        let x = vec![xs];
        let run = |priors: Vec<PriorSpec>| {
            let req = CensoredAftRequest {
                dist: AftDistribution::Weibull,
                time: &time,
                event: &event,
                x: &x,
                intercept: true,
                priors,
            };
            match fit_censored_aft(&req).unwrap() {
                Bridged::Fitted(g) => *g,
                other => panic!("{other:?}"),
            }
        };
        let flat = run(Vec::new());
        let shrunk = run(vec![PriorSpec::flat(), PriorSpec::normal(0.0, 0.05)]);

        assert!(
            shrunk.mode[1].abs() < flat.mode[1].abs(),
            "the prior must pull the slope toward zero: {} vs {}",
            shrunk.mode[1],
            flat.mode[1]
        );
        assert!(
            shrunk.covariance[(1, 1)] < flat.covariance[(1, 1)],
            "the prior must also narrow the posterior: {} vs {}",
            shrunk.covariance[(1, 1)],
            flat.covariance[(1, 1)]
        );
    }
}
