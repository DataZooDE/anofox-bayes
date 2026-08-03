//! The Laplace engine: MAP plus curvature.
//!
//! Fit a Gaussian to the log posterior at its mode, on the unconstrained scale, and
//! draw from that:
//!
//! ```text
//!   theta_hat = argmax logp(theta)          (Newton, on the analytic gradient)
//!   H         = -d2 logp / dtheta2 |theta_hat
//!   theta     ~ N(theta_hat, H^-1)
//! ```
//!
//! Cost is a handful of gradient evaluations rather than thousands, and the draws are
//! independent, so a Laplace fit clears the ESS gate at any budget the caller asks
//! for. The price is that the approximation is only as good as the posterior is
//! Gaussian — which for GLM-shaped posteriors with reasonable data is very good, and
//! for a hierarchical variance parameter with few groups is not good at all.
//!
//! **This is why the engine exists alongside the exact one rather than instead of it.**
//! Where a family is conjugate, both engines can serve the same model, and the exact
//! posterior is the reference the approximation is measured against. That comparison
//! is the strongest correctness check in this crate: two independent derivations of
//! one distribution, one of them closed-form, agreeing to Monte Carlo error. It also
//! measures *where* Laplace is admissible — the tests below pin the specific way it
//! errs, understating tail width on a Student-t marginal at small sample sizes.
//!
//! The Hessian is obtained by differencing the **analytic** gradient rather than by
//! differencing the log density twice. Second differences of a scalar lose roughly
//! two-thirds of the available precision; first differences of an exact gradient lose
//! half of it, which is the difference between a usable curvature and a matrix that
//! will not factor.

use faer::Mat;

use crate::catalog::{CompiledModel, GaussianApproximation, LogPosterior};
use crate::draws::SampleStats;
use crate::errors::{BayesError, BayesResult};
use crate::linalg::{cholesky, sample_mvn, solve_with};
use crate::rng::BayesRng;
use crate::types::EngineKind;

use super::{Engine, Sample, SampleOptions};

#[derive(Debug, Default, Clone, Copy)]
pub struct LaplaceEngine;

/// Newton iterations before giving up. The objective is smooth and the starting point
/// comes from the family, so a well-posed problem converges in a handful; needing
/// more means the surface is not the shape this engine assumes.
const MAX_NEWTON: u32 = 100;
const GRAD_TOLERANCE: f64 = 1e-8;

/// Relative improvement in the log density below which the search has finished.
///
/// [`GRAD_TOLERANCE`] alone is not a sufficient stopping rule, and the reason is
/// arithmetic rather than modelling. The gradient of a log posterior is a **sum over
/// observations**, so its rounding error grows with the sample: `pooled_gaussian`
/// reaches an absolute `1e-8` because its gradient is assembled from precomputed
/// sufficient statistics, while a family that walks its observations — `payer_alive`
/// evaluates a log-gamma per customer — carries a noise floor nearer `1e-9 * n`. Above
/// a few thousand rows that floor sits *above* `1e-8`, and a search that has genuinely
/// arrived would spend its whole iteration budget taking steps of no consequence and
/// then report `ConvergenceFailure` on a perfectly good fit.
///
/// So the search also stops when a step it accepted moved the log density by less than
/// this fraction of its own magnitude: no further improvement is available, whatever
/// the gradient's last few digits say. The two rules are complementary — the gradient
/// test proves stationarity, this one proves exhaustion — and a family that needs to
/// know *which* one fired checks the gradient itself afterwards, as
/// [`crate::catalog::f5_btyd`] does.
const IMPROVEMENT_TOLERANCE: f64 = 1e-12;

/// How far a single Newton step may move a coordinate, relative to that coordinate's
/// own magnitude.
///
/// A trust region, and the reason for it is that a backtracking line search is not by
/// itself enough: it only ever *shrinks* a step that made things worse, and a step
/// that lands somewhere higher is accepted however far away it is. Where the surface
/// is nearly flat the Newton step is proportionally enormous, so an unguarded first
/// iteration can leap across the parameter space into a *local* optimum, improve the
/// density on arrival, and never come back.
///
/// That is not hypothetical. Measured on `payer_alive`: from the conventional
/// `a = b = 1` start, the first step moved `ln a` by +14.6 — a factor of two million —
/// onto the flat ridge where the dropout probability is effectively zero, and the
/// search then sat there. The log posterior at that point was **248 lower** than at
/// the true mode, which the same search reaches without difficulty when started
/// nearby. The family had no way to tell that from a genuine boundary solution, so a
/// perfectly ordinary customer base came back as `degenerate`.
///
/// Capping at twice a coordinate's own size costs a handful of extra iterations on a
/// well-behaved surface and cannot cost anything on one already at its mode — the step
/// is then zero. `pooled_gaussian` starts its search *at* the analytic mode, so this
/// never binds there.
///
/// The regression test lives with the family that exposed it —
/// `f5_btyd::tests::a_larger_base_does_not_become_harder_to_fit_than_a_smaller_one`.
/// A synthetic one-dimensional fixture was tried and abandoned: on every two-optimum
/// curve of that shape the line search alone recovers, and the failure needs the flat
/// four-dimensional ridge of a real BG/NBD surface to appear at all. A fixture that
/// passes with the guard removed would have been worse than none.
const MAX_NEWTON_STEP: f64 = 2.0;

impl Engine for LaplaceEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Laplace
    }

    fn supports(&self, model: &dyn CompiledModel) -> bool {
        model.as_gaussian().is_some() || model.as_differentiable().is_some()
    }

    fn sample(&self, model: &dyn CompiledModel, opts: &SampleOptions) -> BayesResult<Sample> {
        // A model that already holds its own mode and curvature is served directly.
        // Checked *first*, not as a fallback: a family that arrives fitted has an
        // answer, and re-running a Newton search on a differentiable view of the same
        // posterior would at best waste the work and at worst find a different point
        // and publish it under the same `model_id`.
        if let Some(g) = model.as_gaussian() {
            return sample_gaussian(model, g, opts);
        }

        let target = model.as_differentiable().ok_or_else(|| {
            BayesError::config(
                "engine",
                "this family does not expose a differentiable log posterior, \
                 so the Laplace engine cannot serve it",
            )
        })?;

        let mode = find_mode(target)?;
        let hessian = negative_hessian(target, &mode)?;
        // The curvature at a maximum is positive definite. If it is not, the point
        // found is not a maximum -- a saddle, or a flat direction from an
        // unidentified parameter -- and a covariance derived from it would be
        // meaningless rather than merely imprecise.
        let chol = cholesky(&hessian).map_err(|e| {
            BayesError::NotPositiveDefinite(format!(
                "the curvature at the mode is not a valid covariance ({e}); \
                 the posterior is flat in at least one direction"
            ))
        })?;

        let n_params = model.param_names().len();
        let dim = target.dim();
        let mut values = vec![0.0; opts.n_chains * opts.n_draws * n_params];
        let mut unconstrained = vec![0.0; dim];

        for chain in 0..opts.n_chains {
            let mut rng = BayesRng::for_chain(opts.seed, chain as u32);
            for draw in 0..opts.n_draws {
                sample_mvn(&chol, &mode, 1.0, &mut rng, &mut unconstrained)?;
                let offset = (chain * opts.n_draws + draw) * n_params;
                target.constrain(&unconstrained, &mut values[offset..offset + n_params]);
            }
        }

        // Independent draws, no trajectories: nothing to report, and reporting
        // `__divergent__ = 0` would read as "the sampler explored cleanly".
        Ok(Sample {
            values,
            stats: Vec::<SampleStats>::new(),
        })
    }
}

/// Draw from a Gaussian approximation the family already holds.
///
/// This is the last step of the Laplace recipe and nothing else: the mode and the
/// curvature came from wherever the family got them — for a bridged fit, from
/// `anofox-stats-core` — and all that remains is to sample and back-transform.
///
/// **The factor is taken of the full precision matrix.** `cholesky` is handed
/// `block.precision` whole, so `sample_mvn` produces draws with the posterior's
/// correlation structure intact. There is no code path in which a diagonal is
/// extracted, and `the_draws_carry_the_posterior_correlation_and_not_only_its_variances`
/// fails if one is ever introduced.
///
/// Parameters covered by no block stay `NaN`, which reaches SQL as NULL: that is how a
/// group whose fit was refused travels, beside the groups that succeeded, without
/// making the shape of the draws table depend on the outcome.
fn sample_gaussian(
    model: &dyn CompiledModel,
    approx: &dyn GaussianApproximation,
    opts: &SampleOptions,
) -> BayesResult<Sample> {
    let blocks = approx.blocks();
    let n_params = model.param_names().len();

    // Factor once, outside the draw loop: the factorisation is `O(p^3)` and the
    // sampling `O(p^2)`, so refactoring per draw would dominate the cost of the fit.
    let factors: Vec<faer::Mat<f64>> = blocks
        .iter()
        .enumerate()
        .map(|(b, block)| {
            cholesky(&block.precision).map_err(|e| {
                BayesError::NotPositiveDefinite(format!(
                    "block {b} of the bridged posterior has a curvature that is not a \
                     valid precision ({e}); the fit should have been refused before \
                     reaching the engine"
                ))
            })
        })
        .collect::<BayesResult<_>>()?;

    let widest = blocks.iter().map(|b| b.mode.len()).max().unwrap_or(0);
    let widest_out = blocks.iter().map(|b| b.params.len()).max().unwrap_or(0);
    let mut unconstrained = vec![0.0; widest];
    let mut constrained = vec![0.0; widest_out];

    let mut values = vec![f64::NAN; opts.n_chains * opts.n_draws * n_params];
    for chain in 0..opts.n_chains {
        let mut rng = BayesRng::for_chain(opts.seed, chain as u32);
        for draw in 0..opts.n_draws {
            let offset = (chain * opts.n_draws + draw) * n_params;
            for (b, block) in blocks.iter().enumerate() {
                let dim = block.mode.len();
                sample_mvn(
                    &factors[b],
                    &block.mode,
                    1.0,
                    &mut rng,
                    &mut unconstrained[..dim],
                )?;
                let out = &mut constrained[..block.params.len()];
                approx.constrain(b, &unconstrained[..dim], out);
                for (k, &slot) in block.params.iter().enumerate() {
                    if slot >= n_params {
                        return Err(BayesError::Internal(format!(
                            "block {b} writes parameter slot {slot}, but the model reports \
                             only {n_params} parameters"
                        )));
                    }
                    values[offset + slot] = out[k];
                }
            }
        }
    }

    Ok(Sample {
        values,
        stats: Vec::<SampleStats>::new(),
    })
}

/// Newton's method on the analytic gradient, with a backtracking line search.
///
/// The line search is what makes this robust rather than merely fast: a full Newton
/// step from a poor starting point can overshoot into a region where the density is
/// lower, and an unguarded iteration then oscillates or diverges. Halving the step
/// until the density actually improves costs a few extra evaluations and removes the
/// failure mode entirely.
///
/// Visible to the crate, not only to this engine, because a family whose likelihood
/// has boundary solutions has to run this search itself at compile time and *inspect
/// the answer* before an engine acts on it — see
/// [`crate::catalog::f5_btyd`]. Two copies of a Newton loop would be two things to
/// keep in step, and the one that mattered would be the one nobody looked at.
pub(crate) fn find_mode(target: &dyn LogPosterior) -> BayesResult<Vec<f64>> {
    let (theta, converged) = find_mode_best_effort(target)?;
    if converged {
        Ok(theta)
    } else {
        Err(BayesError::ConvergenceFailure {
            iterations: MAX_NEWTON,
            tolerance: GRAD_TOLERANCE,
        })
    }
}

/// The same search, returning the best point reached and whether it settled.
///
/// [`find_mode`] discards the point when the budget runs out, which is right for an
/// engine: a search that did not settle has nothing an engine should sample around.
/// It is wrong for a family that must *classify* its own failure. `payer_alive`
/// applies four tests to the point it gets — is it inside the admissible range, is it
/// stationary, does its curvature factor, is the implied marginal an interval at all —
/// and every one of those is more informative about what went wrong than "ran out of
/// iterations". Handing back the point lets the family say which, and lets a slow but
/// perfectly sound convergence on a large dataset be recognised as one.
pub(crate) fn find_mode_best_effort(target: &dyn LogPosterior) -> BayesResult<(Vec<f64>, bool)> {
    let dim = target.dim();
    let mut theta = target.initial();
    if theta.len() != dim {
        return Err(BayesError::Internal(format!(
            "family's initial point has {} coordinates, expected {dim}",
            theta.len()
        )));
    }
    let mut grad = vec![0.0; dim];

    for _ in 0..MAX_NEWTON {
        target.grad(&theta, &mut grad)?;
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err(BayesError::Internal(
                "the log-posterior gradient is not finite at the current point".to_string(),
            ));
        }
        if norm < GRAD_TOLERANCE {
            return Ok((theta, true));
        }

        let hessian = negative_hessian(target, &theta)?;
        // Fall back to a gradient step where the curvature is unusable: far from the
        // mode a Newton step is not defined, but uphill still is.
        let step = match cholesky(&hessian).and_then(|l| solve_with(&l, &grad)) {
            Ok(newton) => newton,
            Err(_) => grad.clone(),
        };

        // Trust region: shrink the whole step until no coordinate moves further than
        // `MAX_NEWTON_STEP` times its own magnitude. Applied before the line search,
        // which can only shrink a step that made things worse and would happily accept
        // a leap into a distant local optimum that happened to be higher.
        let overreach = (0..dim)
            .map(|j| step[j].abs() / theta[j].abs().max(1.0))
            .fold(0.0f64, f64::max);

        let before = target.logp(&theta);
        let mut scale = if overreach > MAX_NEWTON_STEP {
            MAX_NEWTON_STEP / overreach
        } else {
            1.0
        };
        let mut accepted = false;
        for _ in 0..40 {
            let candidate: Vec<f64> = (0..dim).map(|j| theta[j] + scale * step[j]).collect();
            let after = target.logp(&candidate);
            if after.is_finite() && after >= before {
                let gained = after - before;
                theta = candidate;
                accepted = true;
                // Arrived: the best step available buys nothing. See
                // `IMPROVEMENT_TOLERANCE` -- without this the search burns its whole
                // budget on steps of no consequence and then calls a good fit a
                // convergence failure.
                if gained <= IMPROVEMENT_TOLERANCE * before.abs().max(1.0) {
                    return Ok((theta, true));
                }
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            // No improving step exists in this direction: already at the mode to
            // within floating-point resolution.
            return Ok((theta, true));
        }
    }

    Ok((theta, false))
}

/// `-d2 logp / dtheta2`, by central differences of the analytic gradient.
///
/// The step is scaled to each coordinate's magnitude, so a coefficient measured in
/// millions and a log-scale parameter of order one both get a sensible perturbation.
/// The result is symmetrised, because differencing produces a matrix that is only
/// symmetric up to rounding and Cholesky assumes exactly symmetric input.
///
/// `pub(crate)` because two callers outside this module reuse it: the NUTS engine, to
/// choose starting points overdispersed *in the posterior's own units* (see
/// [`super::nuts::overdispersed_start`]), and `f5_btyd`, to test the curvature at its
/// mode for the boundary solutions BG/NBD is prone to.
pub(crate) fn negative_hessian(target: &dyn LogPosterior, theta: &[f64]) -> BayesResult<Mat<f64>> {
    let dim = target.dim();
    let mut h: Mat<f64> = Mat::zeros(dim, dim);
    let mut plus = vec![0.0; dim];
    let mut minus = vec![0.0; dim];
    let mut point = theta.to_vec();

    for j in 0..dim {
        // Cube root of machine epsilon is the standard step for a central difference
        // of a first derivative: it balances truncation error against cancellation.
        let step = 6.06e-6 * theta[j].abs().max(1.0);
        point[j] = theta[j] + step;
        target.grad(&point, &mut plus)?;
        point[j] = theta[j] - step;
        target.grad(&point, &mut minus)?;
        point[j] = theta[j];

        for i in 0..dim {
            h[(i, j)] = -(plus[i] - minus[i]) / (2.0 * step);
        }
    }

    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = avg;
            h[(j, i)] = avg;
        }
        if !h[(i, i)].is_finite() {
            return Err(BayesError::NotPositiveDefinite(format!(
                "curvature at coordinate {i} is not finite"
            )));
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{f3_pooled_gaussian::PooledGaussian, ModelFamily};
    use crate::config::Config;
    use crate::data::testing::Frame;
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

    /// The single most valuable test in the crate. A hand-derived gradient that is
    /// subtly wrong still finds *a* mode and still produces plausible-looking draws;
    /// nothing downstream would notice. Finite differences notice.
    #[test]
    fn the_analytic_gradient_matches_finite_differences() {
        with_model!(40, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let target = model.as_differentiable().unwrap();
            let dim = target.dim();

            // Checked away from the mode as well as at it: at the mode the gradient
            // is zero, where a sign error or a missing term is invisible.
            for offset in [0.0, 0.4, -0.9] {
                let theta: Vec<f64> = target
                    .initial()
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v + offset * (1.0 + j as f64 * 0.15))
                    .collect();

                let mut analytic = vec![0.0; dim];
                target.grad(&theta, &mut analytic).unwrap();

                for j in 0..dim {
                    let step = 1e-6 * theta[j].abs().max(1.0);
                    let mut up = theta.clone();
                    let mut down = theta.clone();
                    up[j] += step;
                    down[j] -= step;
                    let numeric = (target.logp(&up) - target.logp(&down)) / (2.0 * step);
                    let tol = 1e-4 * numeric.abs().max(1.0);
                    assert!(
                        (analytic[j] - numeric).abs() < tol,
                        "offset {offset}, coordinate {j}: analytic {} vs numeric {numeric}",
                        analytic[j]
                    );
                }
            }
        });
    }

    /// The mode the Newton search finds must be the mode the algebra predicts:
    /// `beta = b_n` and `sigma^2 = 2 s_n / (2 a_n + p)`.
    #[test]
    fn the_mode_search_lands_where_the_gradient_vanishes() {
        with_model!(40, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let target = model.as_differentiable().unwrap();
            let mode = find_mode(target).unwrap();

            let mut grad = vec![0.0; target.dim()];
            target.grad(&mode, &mut grad).unwrap();
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            assert!(norm < 1e-6, "gradient norm at the mode is {norm}");

            // And it is a maximum, not merely a stationary point.
            for j in 0..target.dim() {
                for delta in [1e-3, -1e-3] {
                    let mut nudged = mode.clone();
                    nudged[j] += delta;
                    assert!(
                        target.logp(&nudged) <= target.logp(&mode) + 1e-12,
                        "coordinate {j} is not at a maximum"
                    );
                }
            }
        });
    }

    /// **The headline correctness gate.** Two independent derivations of one
    /// posterior -- a closed-form conjugate sampler and a Gaussian fit to the log
    /// density -- must agree. A mistake in either shows up here, because a mistake
    /// that happened to affect both identically is not a plausible failure.
    #[test]
    fn the_laplace_engine_agrees_with_the_exact_conjugate_posterior() {
        with_model!(400, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let opts = SampleOptions {
                n_chains: 1,
                n_draws: 200_000,
                // Ignored: this engine draws independently, so there is nothing to adapt.
                n_warmup: 0,
                seed: 17,
                sample_from: crate::types::SampleFrom::Posterior,
            };
            let exact = ExactEngine.sample(&*model, &opts).unwrap();
            let laplace = LaplaceEngine.sample(&*model, &opts).unwrap();

            let p = model.param_names().len();
            let summarise = |values: &[f64], j: usize| {
                let col: Vec<f64> = values.chunks(p).map(|c| c[j]).collect();
                let m = col.iter().sum::<f64>() / col.len() as f64;
                let sd = (col.iter().map(|v| (v - m).powi(2)).sum::<f64>()
                    / (col.len() - 1) as f64)
                    .sqrt();
                (m, sd)
            };

            for j in 0..p {
                let (em, esd) = summarise(&exact.values, j);
                let (lm, lsd) = summarise(&laplace.values, j);
                let name = &model.param_names()[j].name;

                // Means agree to 1% of the parameter's own scale. The coefficients
                // agree far more tightly than that -- their conditional posterior is
                // exactly Gaussian -- while `sigma` carries a genuine O(1/n)
                // discrepancy, because Laplace draws it as the exponential of a
                // Gaussian and the exact marginal is not lognormal. That the
                // discrepancy is second-order rather than a mistake is pinned by
                // `the_laplace_error_in_sigma_shrinks_as_the_sample_grows`.
                let scale = esd.max(em.abs() * 0.01).max(1e-6);
                assert!(
                    (em - lm).abs() < 0.01 * em.abs().max(esd),
                    "{name}: exact mean {em} vs laplace {lm} (sd {esd}, scale {scale})"
                );
                // Standard deviations agree to 2%: with 400 observations the
                // Student-t marginal is indistinguishable from its Gaussian limit.
                assert!(
                    (esd - lsd).abs() < 0.02 * esd,
                    "{name}: exact sd {esd} vs laplace {lsd}"
                );
            }
        });
    }

    /// The `sigma` discrepancy above is a second-order effect of the transform, not a
    /// mistake: Laplace draws `sigma` as `exp` of a Gaussian, so it is lognormal,
    /// while the exact marginal is a scaled inverse chi. The two agree to `O(1/n)`.
    ///
    /// A bug would not behave this way -- it would leave a discrepancy that stayed
    /// put, or grew. Watching it shrink as the sample grows is what distinguishes
    /// "a known approximation error" from "wrong", and it is the evidence behind
    /// the claim that Laplace is admissible for this family at realistic sample
    /// sizes.
    #[test]
    fn the_laplace_error_in_sigma_shrinks_as_the_sample_grows() {
        let relative_error_at = |n: usize| {
            let f = frame(n);
            let refs = f.key_refs();
            let view = f.view(&refs);
            let model = PooledGaussian
                .compile(
                    &Config::parse(r#"{"y": "y", "x": ["x1", "x2"]}"#).unwrap(),
                    &view,
                )
                .unwrap();
            let opts = SampleOptions {
                n_chains: 1,
                n_draws: 200_000,
                // Ignored: this engine draws independently, so there is nothing to adapt.
                n_warmup: 0,
                seed: 31,
                sample_from: crate::types::SampleFrom::Posterior,
            };
            let p = model.param_names().len();
            let mean_of = |values: &[f64]| {
                let col: Vec<f64> = values.chunks(p).map(|c| c[p - 1]).collect();
                col.iter().sum::<f64>() / col.len() as f64
            };
            let e = mean_of(&ExactEngine.sample(&*model, &opts).unwrap().values);
            let l = mean_of(&LaplaceEngine.sample(&*model, &opts).unwrap().values);
            (e - l).abs() / e
        };

        let small = relative_error_at(20);
        let large = relative_error_at(2000);

        // Observed: ~5e-2 at n=20 falling to ~1.1e-3 at n=2000, which is close to
        // 2/n and comfortably above the Monte Carlo noise floor of ~7e-5 for the
        // 200k draws used here -- so this is a measured bias with the predicted
        // order, not sampling noise.
        assert!(
            small / large > 10.0,
            "the sigma error should fall roughly like 1/n: {small} at n=20 vs {large} at n=2000"
        );
        // A tenth of a percent on a scale parameter changes no decision anyone makes
        // with it. This bound is what "Laplace is admissible for this family at
        // realistic sample sizes" concretely means.
        assert!(
            large < 3e-3,
            "at n=2000 the error should be negligible, got {large}"
        );
    }

    /// Where the approximation errs, and by how much. With few observations the exact
    /// marginal for a coefficient is a Student-t with heavy tails, while Laplace
    /// returns its Gaussian limit -- so Laplace *understates* the spread. Pinning the
    /// direction and rough size of that error is what certifies where the engine may
    /// be used, which is the whole purpose of running both.
    #[test]
    fn laplace_understates_the_tails_on_small_samples() {
        with_model!(12, r#"{"y": "y", "x": ["x1", "x2"]}"#, model, {
            let opts = SampleOptions {
                n_chains: 1,
                n_draws: 200_000,
                // Ignored: this engine draws independently, so there is nothing to adapt.
                n_warmup: 0,
                seed: 19,
                sample_from: crate::types::SampleFrom::Posterior,
            };
            let exact = ExactEngine.sample(&*model, &opts).unwrap();
            let laplace = LaplaceEngine.sample(&*model, &opts).unwrap();
            let p = model.param_names().len();

            let spread = |values: &[f64], j: usize| {
                let mut col: Vec<f64> = values.chunks(p).map(|c| c[j]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                // 99% interval width: the tails, where the two differ.
                col[(col.len() as f64 * 0.995) as usize] - col[(col.len() as f64 * 0.005) as usize]
            };

            let exact_width = spread(&exact.values, 1);
            let laplace_width = spread(&laplace.values, 1);
            assert!(
                laplace_width < exact_width,
                "laplace 99% width {laplace_width} should be narrower than exact {exact_width}"
            );
            // ...but not absurdly so, even at n = 12.
            assert!(
                laplace_width > 0.5 * exact_width,
                "laplace width {laplace_width} vs exact {exact_width}"
            );
        });
    }

    #[test]
    fn the_same_seed_reproduces_the_same_laplace_draws() {
        with_model!(50, r#"{"y": "y", "x": "x1"}"#, model, {
            let opts = SampleOptions {
                n_chains: 2,
                n_draws: 100,
                // Ignored: this engine draws independently, so there is nothing to adapt.
                n_warmup: 0,
                seed: 23,
                sample_from: crate::types::SampleFrom::Posterior,
            };
            let a = LaplaceEngine.sample(&*model, &opts).unwrap();
            let b = LaplaceEngine.sample(&*model, &opts).unwrap();
            assert_eq!(a.values, b.values);
            assert!(a.stats.is_empty());
        });
    }

    /// `sigma` is sampled on the log scale, so every draw is positive by
    /// construction. A Gaussian approximation applied directly to `sigma` would put
    /// mass below zero, which is the reason for the unconstrained parameterisation.
    #[test]
    fn a_positive_parameter_stays_positive_because_it_is_sampled_on_the_log_scale() {
        with_model!(30, r#"{"y": "y", "x": "x1"}"#, model, {
            let sample = LaplaceEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 1,
                        n_draws: 20_000,
                        // Ignored: this engine draws independently, so there is nothing to adapt.
                        n_warmup: 0,
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

    //=== conjugate_anomaly ================================================//
    //
    // F7's posterior is closed form too, so the same two-independent-derivations gate
    // applies to it. What differs from F3 is the arithmetic of *how closely* they may
    // be expected to agree, which is worked out per parameter below rather than
    // assumed.

    /// Mean and sample standard deviation of parameter `j` in a draws buffer.
    fn summarise(values: &[f64], p: usize, j: usize) -> (f64, f64) {
        let col: Vec<f64> = values.chunks(p).map(|c| c[j]).collect();
        let m = col.iter().sum::<f64>() / col.len() as f64;
        let sd = (col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (col.len() - 1) as f64).sqrt();
        (m, sd)
    }

    /// Compile F7 over `n` synthetic observations and return both engines' summaries
    /// per parameter, as `(exact_mean, exact_sd, laplace_mean, laplace_sd)`.
    fn f7_engines(cfg: &str, f: &Frame, seed: u64) -> Vec<(f64, f64, f64, f64)> {
        let refs = f.key_refs();
        let view = f.view(refs.as_slice());
        let model = crate::catalog::f7_conjugate::ConjugateAnomaly
            .compile(&Config::parse(cfg).unwrap(), &view)
            .unwrap();
        let opts = SampleOptions {
            n_chains: 1,
            n_draws: 200_000,
            seed,
            sample_from: crate::types::SampleFrom::Posterior,
            n_warmup: 0,
        };
        let exact = ExactEngine.sample(&*model, &opts).unwrap();
        let laplace = LaplaceEngine.sample(&*model, &opts).unwrap();
        let p = model.param_names().len();
        (0..p)
            .map(|j| {
                let (em, esd) = summarise(&exact.values, p, j);
                let (lm, lsd) = summarise(&laplace.values, p, j);
                (em, esd, lm, lsd)
            })
            .collect()
    }

    fn normal_frame(n: usize) -> Frame {
        // Deterministic, and genuinely spread out: a sinusoid at an irrational-ish
        // step never repeats over the sample sizes used here.
        Frame::new(n).numeric(
            "cost",
            (0..n)
                .map(|i| 12.0 + (i as f64 * 0.7).sin() * 1.5 + (i as f64 * 0.13).cos())
                .collect(),
        )
    }

    /// **The correctness gate this family did not have.** `conjugate_anomaly`'s
    /// posterior is closed form, so the Laplace engine serving it makes the two engines
    /// two independent derivations of one distribution -- one of them exact.
    ///
    /// **Tolerance and why.** Under the reference prior `alpha_n = (n-1)/2` and
    /// `kappa_n = n`, and the Laplace covariance is the inverse curvature at the mode.
    /// Both are closed form, so how far apart the engines may legitimately be is
    /// derived rather than guessed:
    ///
    /// * `mu` -- both engines centre on `mu_n` exactly, so the means differ only by
    ///   Monte Carlo error. The spreads differ by a factor that is available in closed
    ///   form: the exact marginal is Student-t with variance
    ///   `beta_n / (kappa_n (alpha_n - 1))`, the Laplace one is Gaussian with variance
    ///   `2 beta_n / (kappa_n (2 alpha_n + 1))`, so
    ///   `sd_laplace / sd_exact = sqrt((n-3)/n)`, i.e. **0.38 % low at n = 400**
    ///   against 0.51 % measured, the gap being Monte Carlo noise. Where the effect is
    ///   large enough to swamp the noise the prediction is exact: 0.29289 predicted
    ///   against 0.29288 measured at n = 6.
    /// * `sigma` -- Laplace draws it as `exp` of a Gaussian in `u = log sigma` with
    ///   `Var(u) = 1/(2 (2 alpha_n + 1))`, while the exact marginal is a scaled inverse
    ///   chi. Working the two through: the mean is **0.25 % low** and the sd **0.70 %
    ///   low** at n = 400 (measured 0.24 % and 0.68 %).
    ///
    /// Monte Carlo error over 200 000 draws is ~0.22 % of an sd on a mean and ~0.16 %
    /// of an sd on an sd, from each engine. **1 % on means and 2 % on spreads** is
    /// therefore about three times the largest genuine discrepancy and six times the
    /// noise floor. Both engines are seeded, so there is no flake to guard against;
    /// the headroom exists so that the bound tracks the approximation rather than a
    /// particular sample. These are the same numbers used for `pooled_gaussian` above,
    /// which is not a coincidence -- both families are Normal-Inverse-Gamma.
    ///
    /// **What this gate does not catch, and why the family's own tests exist.** A
    /// missing log-Jacobian shifts `alpha_n` by one, which is an `O(1/n)` change --
    /// the same order as the approximation error being measured here, and at n = 400
    /// it moves both the mean and the sd of `sigma` by only 0.25 %. Verified by
    /// mutation: dropping the Jacobian leaves *this* test green and fails
    /// `f7_conjugate::tests::the_normal_log_density_matches_the_closed_form_posterior_up_to_a_constant`.
    /// Two-engine agreement is a strong gate against sign errors and mis-assembled
    /// hyperparameters, and a weak one against anything that only perturbs a shape
    /// parameter. It is not a substitute for checking the density itself.
    #[test]
    fn the_laplace_engine_agrees_with_the_exact_conjugate_posterior_for_the_normal_likelihood() {
        let names = ["mu", "sigma"];
        for (j, (em, esd, lm, lsd)) in f7_engines(r#"{"value": "cost"}"#, &normal_frame(400), 37)
            .into_iter()
            .enumerate()
        {
            let name = names[j];
            assert!(
                (em - lm).abs() < 0.01 * em.abs().max(esd),
                "{name}: exact mean {em} vs laplace {lm} (exact sd {esd})"
            );
            assert!(
                (esd - lsd).abs() < 0.02 * esd,
                "{name}: exact sd {esd} vs laplace {lsd}"
            );
        }
    }

    /// The Poisson half. `lambda ~ Gamma(a_n, rate b_n)` exactly; Laplace fits a
    /// Gaussian to `v = log lambda`, where the curvature at the mode is exactly `a_n`,
    /// so it draws `lambda` lognormal with `mean = (a_n/b_n) exp(1/(2 a_n))` against an
    /// exact mean of `a_n/b_n`. The relative error is `1/(2 a_n)` on the mean and
    /// `~3/(4 a_n)` on the spread. With `a_n = 0.5 + sum(y) ~ 2000` here that is
    /// **0.025 % and 0.04 %**, both *below* the ~0.16 % Monte Carlo floor at this draw
    /// count -- so for a well-observed rate the two engines are not merely close, they
    /// are indistinguishable, and the 1 % / 2 % bounds are measuring nothing but the
    /// absence of a mistake. Where the approximation does bite is small counts:
    /// `1/(2 a_n)` is 4.76 % at `a_n = 10.5`, and a four-observation group measures
    /// 4.82 %.
    #[test]
    fn the_laplace_engine_agrees_with_the_exact_conjugate_posterior_for_the_poisson_likelihood() {
        let n = 400;
        let f = Frame::new(n)
            .numeric("claims", (0..n).map(|i| ((i % 9) + 1) as f64).collect())
            .numeric(
                "consignments",
                (0..n).map(|i| 5.0 + (i % 4) as f64).collect(),
            );
        let summaries = f7_engines(
            r#"{"value": "claims", "likelihood": "poisson", "exposure": "consignments"}"#,
            &f,
            41,
        );
        assert_eq!(summaries.len(), 1);
        let (em, esd, lm, lsd) = summaries[0];
        assert!(
            (em - lm).abs() < 0.01 * em.abs().max(esd),
            "lambda: exact mean {em} vs laplace {lm} (exact sd {esd})"
        );
        assert!(
            (esd - lsd).abs() < 0.02 * esd,
            "lambda: exact sd {esd} vs laplace {lsd}"
        );
    }

    /// **Where the approximation is not admissible, measured.** The agreement above is
    /// a statement about a well-observed group, and `conjugate_anomaly` exists to look
    /// at groups that are *not* well observed. `1 - sqrt((n-3)/n)` is 0.4 % at n = 400
    /// and **29 % at n = 6** -- measured 0.2929 against a predicted 0.29289. A thin
    /// lane's Laplace interval for `mu` is nearly a third too narrow, and `sigma` is
    /// worse: 44 % on the sd and 20 % on the mean at n = 6. Too narrow is the direction
    /// that turns "this lane is behaving oddly" into a false alarm and "this lane is
    /// fine" into an unearned all-clear.
    ///
    /// This is a finding, not a defect: the closed form says the error must be this
    /// size, and the measurement agrees to four digits. It is why `exact` remains the
    /// family's default, and why this test pins the error rather than tolerating it
    /// silently.
    #[test]
    fn the_laplace_spread_on_a_thin_group_is_far_too_narrow_and_the_error_shrinks_with_the_group() {
        let error_at = |n: usize| {
            let s = f7_engines(r#"{"value": "cost"}"#, &normal_frame(n), 43);
            let (_, esd, _, lsd) = s[0]; // mu
            (esd - lsd) / esd
        };

        // Predicted 1 - sqrt((n-3)/n): 0.35 at n = 6, 0.0038 at n = 400.
        let thin = error_at(6);
        let thick = error_at(400);
        assert!(
            thin > 0.2,
            "at n = 6 the Laplace spread for mu should be visibly too narrow, got {thin}"
        );
        assert!(
            thick < 0.02,
            "at n = 400 the error should be negligible, got {thick}"
        );
        assert!(
            thin / thick > 10.0,
            "the error should shrink sharply with the group: {thin} at n = 6 vs {thick} at n = 400"
        );
    }

    /// A group whose posterior does not exist must not take the rest of the fit down
    /// with it. Its coordinates are excluded from the unconstrained vector entirely, so
    /// the mode search and the Cholesky never see a `NaN` -- and the group comes back
    /// NULL-shaped, the same answer the exact engine gives.
    #[test]
    fn an_unfittable_group_draws_null_under_laplace_without_poisoning_the_others() {
        let f = Frame::new(7)
            .numeric("cost", vec![1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 42.0])
            .key("lane", vec!["A", "A", "A", "A", "A", "A", "SOLO"]);
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = crate::catalog::f7_conjugate::ConjugateAnomaly
            .compile(
                &Config::parse(r#"{"value": "cost", "group": "lane"}"#).unwrap(),
                &view,
            )
            .unwrap();

        let sample = LaplaceEngine
            .sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 1000,
                    seed: 47,
                    sample_from: crate::types::SampleFrom::Posterior,
                    n_warmup: 0,
                },
            )
            .unwrap();
        let p = model.param_names().len();
        assert_eq!(p, 4); // (mu, sigma) for A and for SOLO
        for row in sample.values.chunks(p) {
            assert!(row[0].is_finite() && row[1] > 0.0, "lane A: {row:?}");
            assert!(row[2].is_nan() && row[3].is_nan(), "SOLO: {row:?}");
        }
    }

    /// Every family exposing a gradient is *mechanically* servable by the Laplace
    /// engine, and every *conjugate* family is additionally served by the exact one —
    /// which is what gives it the two-engine cross-check, the strongest correctness
    /// gate in this crate.
    ///
    /// The list of families that decline `exact` is spelled out rather than derived,
    /// because declining it costs a warranty. `censored_aft` is on that list because
    /// its posterior genuinely has no closed form; a family added there without that
    /// justification fails review, and a family added to the catalog and to neither
    /// list fails here.
    ///
    /// **`hier_negbin` is the one family that declines `laplace` too**, and it does so
    /// in `compile` rather than through `Engine::supports`, because the reason is a
    /// property of its geometry rather than of the trait it implements: it exposes a
    /// gradient because NUTS needs one, and a Gaussian at the joint mode of a
    /// non-centred hierarchy is still not a posterior. See
    /// `catalog::f1_hier_negbin::build`.
    #[test]
    fn every_family_is_served_by_laplace_and_every_conjugate_one_also_by_exact() {
        let n = 40;
        let (days, distance, delivered) = crate::bridge::testing::survival_fixture(
            anofox_stats_core::models::AftDistribution::Weibull,
            1.0,
            0.25,
            0.4,
            n,
            0.0,
            Some(20.0),
        );
        let f = Frame::new(n)
            .numeric("y", (0..n).map(|i| 1.0 + i as f64 * 0.5).collect())
            .numeric(
                "cost",
                (0..n).map(|i| 1.0 + ((i % 7) as f64) * 0.3).collect(),
            )
            .numeric("x1", (0..n).map(|i| i as f64).collect())
            .numeric("days", days)
            .numeric("distance", distance)
            .numeric("delivered", delivered)
            .numeric("freq", (0..n).map(|i| (i % 5) as f64).collect())
            .numeric("rec", (0..n).map(|i| ((i % 5) as f64) * 6.0).collect())
            .numeric("age", (0..n).map(|_| 40.0).collect())
            .numeric("units", (0..n).map(|i| (i % 6) as f64).collect())
            // Strictly positive, as `payment_delay` requires, and not constant, so the
            // family does not decline the fixture as degenerate before the engine
            // questions are reached.
            .numeric(
                "delay",
                (0..n).map(|i| 10.0 + ((i * 7) % 23) as f64).collect(),
            )
            // A log-price column that varies *within* each segment, which is what
            // identifies an elasticity and therefore what keeps `hier_elasticity` from
            // declining the fixture before the engine questions are reached.
            .numeric(
                "logp",
                (0..n).map(|i| ((i / 4) % 5) as f64 * 0.1 - 0.2).collect(),
            )
            .key(
                "segment",
                (0..n).map(|i| ["A", "B", "C", "D"][i % 4]).collect(),
            )
            .key("sku", (0..n).map(|i| ["A", "B", "C", "D"][i % 4]).collect());
        let refs = f.key_refs();
        let view = f.view(&refs);

        // Families whose posterior has no closed form, and why.
        const NO_EXACT_POSTERIOR: &[&str] = &[
            // A censored AFT likelihood is not conjugate to anything; the family is a
            // bridge onto a MAP fit and is a Gaussian approximation by construction.
            "censored_aft",
            // BG/NBD's four population parameters have no conjugate prior. The SBC
            // suite is what certifies the Gaussian approximation here -- see
            // `sbc::families::f5_is_calibrated_under_the_laplace_engine`.
            "payer_alive",
            // Group variances under a shared hyperprior, and a learned pooling scale.
            // Either one alone destroys conjugacy, which is why this is a separate
            // family from `pooled_gaussian` (ROADMAP.md 3.3). Being *servable* by
            // Laplace is not the same as being *served* by it: the SBC suite measures
            // that approximation as inadmissible here, and the family's default engine
            // is `nuts`. See `sbc::families::f8_under_laplace_is_measured_and_is_not_certified`.
            "varying_variance_gaussian",
            // A hierarchical negative binomial has no conjugate prior either, and
            // unlike the one above it has no *usable* Gaussian approximation at all:
            // a non-centred hierarchy has no joint mode, because at z = 0 the
            // likelihood does not depend on tau and the log-Jacobian rises without
            // bound along a ridge carrying no mass. It is sampled, and `laplace` on it
            // is refused rather than served badly.
            "hier_negbin",
            // The same geometry as `hier_negbin`, one likelihood over: a Gamma or
            // lognormal duration under a non-centred hierarchy. No conjugate prior, and
            // no usable joint mode either.
            "payment_delay",
            // Two non-centred hierarchies rather than one, and a `-exp` transform on the
            // elasticity that is not conjugate to anything either.
            "hier_elasticity",
        ];

        // Refuses `laplace` in `compile`, for a reason `Engine::supports` cannot see.
        const NO_LAPLACE_POSTERIOR: &[&str] = &["hier_negbin", "payment_delay", "hier_elasticity"];

        let configs = [
            ("pooled_gaussian", r#"{"y": "y", "x": "x1"}"#),
            ("hier_negbin", r#"{"y": "units", "group": "sku"}"#),
            ("conjugate_anomaly", r#"{"value": "cost"}"#),
            (
                "censored_aft",
                r#"{"time": "days", "event": "delivered", "x": "distance"}"#,
            ),
            ("payment_delay", r#"{"y": "delay", "group": "segment"}"#),
            (
                "payer_alive",
                r#"{"frequency": "freq", "recency": "rec", "age": "age", "min_customers": 1}"#,
            ),
            (
                "varying_variance_gaussian",
                r#"{"y": "y", "x": "x1", "group": "segment"}"#,
            ),
            (
                "hier_elasticity",
                r#"{"y": "units", "price": "logp", "group": "segment"}"#,
            ),
        ];
        assert_eq!(
            configs.len(),
            crate::catalog::all().len(),
            "a family was added to the catalog without being exercised here"
        );

        for (family, cfg) in configs {
            let model = crate::catalog::lookup(family)
                .unwrap()
                .compile(&Config::parse(cfg).unwrap(), &view)
                .unwrap();
            assert!(
                LaplaceEngine.supports(&*model),
                "{family} must be servable by the Laplace engine"
            );
            // ...and where a family nevertheless refuses it, the refusal is precise
            // and arrives from the family rather than from the engine.
            if NO_LAPLACE_POSTERIOR.contains(&family) {
                let mut with_laplace: serde_json::Value = serde_json::from_str(cfg).unwrap();
                with_laplace["engine"] = "laplace".into();
                let e = crate::catalog::lookup(family)
                    .unwrap()
                    .compile(&Config::parse(&with_laplace.to_string()).unwrap(), &view)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("NUTS only"), "{family}: {e}");
            }
            assert_eq!(
                crate::engines::ExactEngine.supports(&*model),
                !NO_EXACT_POSTERIOR.contains(&family),
                "{family} disagrees with its declared conjugacy"
            );
        }
    }
}
