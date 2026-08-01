//! F3 — pooled Gaussian linear model.
//!
//! The inference layer under intervention evaluation: difference-in-differences,
//! interrupted time series, and the regression half of a synthetic control. It serves
//! the causal-impact agent, whose question is never "what is the average" but "how
//! much of this change was the intervention, and how sure are we".
//!
//! ```text
//!   y = X beta + eps,   eps ~ N(0, sigma^2)
//! ```
//!
//! with a Normal-Inverse-Gamma prior on `(beta, sigma^2)`. The posterior is available
//! in closed form, so no approximation and no sampler are involved. The prior mean
//! `b0` is fixed at zero — it is not a config slot, because a non-zero prior mean on
//! a coefficient is a claim almost nobody can justify — which is why the `b0' P b0`
//! term of the general formula does not appear below:
//!
//! ```text
//!   A     = X'X + P                (P is the prior precision, zero for a flat prior)
//!   b_n   = A^-1 (X'y + P b0)
//!   a_n   = a0 + (n - k)/2         (k = coefficients carrying a flat prior)
//!   s_n   = s0 + (y'y - b_n' X'y) / 2
//!
//!   sigma^2 | y  ~ InvGamma(a_n, s_n)
//!   beta | sigma^2, y ~ N(b_n, sigma^2 A^-1)
//! ```
//!
//! Under a flat prior `b_n` is the ordinary least-squares estimate and `s_n` is half
//! the residual sum of squares, so the marginal posterior for each coefficient is the
//! Student-t whose interval a frequentist would report — with the difference that here
//! it is a statement about the coefficient rather than about a procedure, and it can
//! be pushed through `P(effect > threshold)` in SQL without further theory.
//!
//! **Pooling.** An optional `group` column adds one intercept per group, each with an
//! independent `N(0, sigma^2 * pool_scale^2)` prior. The scaling by `sigma^2` is what
//! makes the prior conjugate — it is the `V0` of the Normal-Inverse-Gamma — and it
//! means `pool_scale` is measured *in residual standard deviations*, not in the units
//! of the response. A noisier dataset therefore pools more at the same `pool_scale`,
//! which is the behaviour you want: the noisier the data, the less a single group's
//! deviation should be believed. Small groups are therefore shrunk toward
//! the population intercept and large ones are not, which is the partial pooling that
//! makes a thin segment borrow strength instead of reporting noise. The pooling scale
//! is *fixed by configuration, not estimated*: estimating it means a hierarchical
//! variance parameter, whose posterior is not conjugate and needs the NUTS engine
//! (0.2). Fixing it is the documented stepping stone, and `pool_scale` is reported in
//! the fit so nobody mistakes it for something the data chose.

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::linalg::{cholesky, sample_mvn, solve_with};
use crate::rng::BayesRng;
use crate::types::{EngineKind, GLOBAL_GROUP};

use super::{CompiledModel, ExactPosterior, ModelFamily, Readiness};
use faer::Mat;

#[derive(Debug)]
pub struct PooledGaussian;

const SLOTS: &[&str] = &[
    "y",
    "x",
    "intercept",
    "group",
    "pool_scale",
    "prior",
    "draws",
    "chains",
    "seed",
    "engine",
];

impl ModelFamily for PooledGaussian {
    fn id(&self) -> &'static str {
        "pooled_gaussian"
    }

    fn description(&self) -> &'static str {
        "Gaussian linear model with a conjugate Normal-Inverse-Gamma posterior and \
         optional partial pooling by group; the inference layer for difference-in-\
         differences and interrupted time series."
    }

    fn default_engine(&self) -> EngineKind {
        EngineKind::Exact
    }

    fn config_slots(&self) -> &'static [&'static str] {
        SLOTS
    }

    fn compile<'a>(
        &self,
        cfg: &Config,
        data: &'a DataView<'a>,
    ) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        cfg.reject_unknown(SLOTS)?;

        let y_name = cfg.require_str("y")?.to_string();
        let x_names = cfg.str_list("x")?;
        let intercept = cfg.f64_or("intercept", 1.0)? != 0.0;
        let group = cfg.opt_str("group")?.map(str::to_string);
        let pool_scale = cfg.positive_f64_or("pool_scale", 1.0)?;

        let prior = cfg.nested("prior")?;
        prior.reject_unknown(&["beta_scale", "a0", "s0"])?;
        // Infinite by default: a flat prior on the coefficients, which makes the
        // posterior mean the least-squares estimate. Any finite default would be a
        // scale assumption about someone else's data.
        let beta_scale = prior.f64_or("beta_scale", f64::INFINITY)?;
        if beta_scale <= 0.0 {
            return Err(BayesError::config("prior.beta_scale", "must be > 0"));
        }
        let a0 = prior.f64_or("a0", 0.0)?;
        let s0 = prior.non_negative_f64_or("s0", 0.0)?;

        if x_names.is_empty() && !intercept && group.is_none() {
            return Err(BayesError::config(
                "x",
                "a model with no predictors, no intercept and no groups has nothing to estimate",
            ));
        }

        // --- Resolve columns and filter nulls, before any arithmetic. ---
        let mut numeric_cols: Vec<&str> = vec![y_name.as_str()];
        numeric_cols.extend(x_names.iter().map(String::as_str));
        let key_cols: Vec<&str> = group.iter().map(String::as_str).collect();

        let rows = data.usable_rows(&numeric_cols, &key_cols)?;
        let fingerprint = data.fingerprint(&numeric_cols, &key_cols, &rows)?;

        let y_col = data.numeric(&y_name)?;
        let x_cols: Vec<_> = x_names
            .iter()
            .map(|n| data.numeric(n))
            .collect::<BayesResult<_>>()?;
        let groups = data.group_rows(group.as_deref(), &rows)?;
        let group_keys: Vec<String> = if group.is_some() {
            groups.iter().map(|(k, _)| k.clone()).collect()
        } else {
            Vec::new()
        };

        // --- Assemble the design matrix. ---
        //
        // Column order is fixed and is what `param_names` mirrors: intercept, then
        // predictors in the caller's order, then one column per group. Keeping the
        // two in lockstep here means no later stage has to re-derive it.
        let n = rows.len();
        let mut params: Vec<ParamName> = Vec::new();
        if intercept {
            params.push(ParamName::global("intercept")?);
        }
        for name in &x_names {
            params.push(ParamName::global(format!("beta[{name}]"))?);
        }
        for key in &group_keys {
            params.push(ParamName::grouped(key.clone(), "group_effect")?);
        }
        // The design has one column per coefficient; `sigma` is a parameter of the
        // posterior but not of the linear predictor, so it is appended after the
        // count is taken and occupies the last output slot.
        let p = params.len();
        params.push(ParamName::global("sigma")?);

        if n <= p {
            return Err(BayesError::InsufficientData { rows: n, params: p });
        }

        let mut x: Mat<f64> = Mat::zeros(n, p);
        let mut y = vec![0.0; n];
        // Row index of each observation within the design, so group columns can be set.
        let mut group_of_row: Vec<usize> = vec![0; n];
        if group.is_some() {
            let mut pos = 0usize;
            for (g, (_, idx)) in groups.iter().enumerate() {
                for _ in idx {
                    group_of_row[pos] = g;
                    pos += 1;
                }
            }
        }
        // Walk rows in group order so `group_of_row` lines up with the design.
        let ordered: Vec<usize> = if group.is_some() {
            groups
                .iter()
                .flat_map(|(_, idx)| idx.iter().copied())
                .collect()
        } else {
            rows.clone()
        };

        for (r, &row) in ordered.iter().enumerate() {
            y[r] = y_col.values[row];
            let mut c = 0;
            if intercept {
                x[(r, c)] = 1.0;
                c += 1;
            }
            for col in &x_cols {
                x[(r, c)] = col.values[row];
                c += 1;
            }
            if !group_keys.is_empty() {
                x[(r, c + group_of_row[r])] = 1.0;
            }
        }

        // --- Prior precision. ---
        //
        // Diagonal: `1/beta_scale^2` on the fixed effects (zero for the flat default)
        // and `1/pool_scale^2` on the group effects, which is what makes the pooling
        // partial rather than none.
        let slope_precision = if beta_scale.is_finite() {
            1.0 / (beta_scale * beta_scale)
        } else {
            0.0
        };
        let group_precision = 1.0 / (pool_scale * pool_scale);
        let n_fixed = p - group_keys.len();
        let first_slope = usize::from(intercept);
        let precision: Vec<f64> = (0..p)
            .map(|j| {
                if j < first_slope {
                    // The intercept is never penalised. `beta_scale` is a statement
                    // about how large an *effect* is plausibly, and an effect prior
                    // centred at zero is a sensible default; the intercept lives on
                    // the scale of the response, where a prior centred at zero says
                    // something nobody means -- and shrinking it silently pushes
                    // every slope the other way to compensate.
                    0.0
                } else if j < n_fixed {
                    slope_precision
                } else {
                    group_precision
                }
            })
            .collect();

        // --- Normal equations. ---
        let mut a: Mat<f64> = Mat::zeros(p, p);
        let mut xty = vec![0.0; p];
        for j in 0..p {
            for k in 0..p {
                let mut s = 0.0;
                for r in 0..n {
                    s += x[(r, j)] * x[(r, k)];
                }
                a[(j, k)] = s;
            }
            a[(j, j)] += precision[j];
            xty[j] = (0..n).map(|r| x[(r, j)] * y[r]).sum();
        }

        // A singular system here means the design is rank deficient -- a constant
        // column, a duplicated predictor, or an intercept plus a full set of group
        // dummies. It is reported rather than pseudo-inverted, because a pseudo-inverse
        // picks one of infinitely many answers and reports it with a straight face.
        let chol = cholesky(&a).map_err(|_| BayesError::SingularMatrix)?;
        let b_n = solve_with(&chol, &xty)?;

        let yty: f64 = y.iter().map(|v| v * v).sum();
        let bab: f64 = (0..p).map(|j| b_n[j] * xty[j]).sum();
        // Each coefficient carrying a *flat* prior costs one degree of freedom.
        //
        // Under the proper conjugate NIG prior, `beta | sigma^2 ~ N(b0, sigma^2 V0)`
        // is scaled by sigma^2, so those coefficients act as prior observations and
        // the shape is simply `a0 + n/2`. A flat coefficient is not sigma^2-scaled:
        // estimating it consumes an observation's worth of information about the
        // residual scale, and the textbook result is
        // `sigma^2 | y ~ Inv-chi^2(n - k, s^2)` with `k` the number of freely
        // estimated coefficients.
        //
        // Using `a0 + n/2` regardless makes the posterior for sigma too tight by
        // sqrt((n - k)/n) -- an overconfident interval, which is the direction that
        // matters: it is the one that produces service levels that quietly under-cover.
        // Found by the PyMC parity suite; SBC could not see it, because SBC must draw
        // the truth from a proper prior and the bug only appears under a flat one.
        let n_flat = precision.iter().filter(|p| **p == 0.0).count();
        let a_n = a0 + (n as f64 - n_flat as f64) / 2.0;
        // s_n = s0 + (y'y - b_n' X'y) / 2, which is s0 + RSS/2 for a flat prior. The
        // small negative values that rounding can produce are clamped: a residual sum
        // of squares is not negative, and letting one through would make the scale
        // parameter of the variance posterior negative and the draw NaN.
        let s_n = s0 + ((yty - bab) / 2.0).max(0.0);

        let readiness = if a_n <= 0.0 || s_n <= 0.0 {
            Readiness::degenerate(format!(
                "the response is perfectly explained by the design ({n} observations, {p} parameters), \
                 so the residual variance is not estimable"
            ))
        } else {
            Readiness::ready()
        };

        Ok(Box::new(CompiledPooledGaussian {
            params,
            chol,
            b_n,
            a_n,
            s_n,
            n_obs: n,
            n_groups: group_keys.len().max(1),
            fingerprint,
            readiness,
        }))
    }
}

#[derive(Debug)]
struct CompiledPooledGaussian {
    params: Vec<ParamName>,
    /// Cholesky factor of the posterior *precision* `A`. Shared between the solve
    /// that produced `b_n` and the sampler, so no inversion happens anywhere.
    chol: Mat<f64>,
    b_n: Vec<f64>,
    a_n: f64,
    s_n: f64,
    n_obs: usize,
    n_groups: usize,
    fingerprint: String,
    readiness: Readiness,
}

impl CompiledModel for CompiledPooledGaussian {
    fn param_names(&self) -> &[ParamName] {
        &self.params
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_groups(&self) -> usize {
        self.n_groups
    }
    fn data_fingerprint(&self) -> &str {
        &self.fingerprint
    }
    fn readiness(&self) -> Readiness {
        self.readiness.clone()
    }
    fn as_exact(&self) -> Option<&dyn ExactPosterior> {
        Some(self)
    }
    fn as_differentiable(&self) -> Option<&dyn super::LogPosterior> {
        Some(self)
    }
}

/// The same posterior, written as a differentiable log density on an unconstrained
/// scale, so the gradient-based engines can consume it.
///
/// Coordinates are `(beta, u)` with `sigma = exp(u)`. Substituting `sigma^2 = e^{2u}`
/// into the Normal-Inverse-Gamma posterior and adding the log-Jacobian `log 2 + 2u`
/// of that transform collapses to
///
/// ```text
///   log p(beta, u) = c*u - e^{-2u} * (s_n + Q/2) + const,
///   c = -(2 a_n + p),   Q = (beta - b_n)' A (beta - b_n)
/// ```
///
/// which is exact, not an approximation — the Laplace engine's error is entirely in
/// the Gaussian fit to this surface, not in the surface itself. That separation is
/// what makes "does Laplace agree with the exact posterior" a meaningful question.
impl super::LogPosterior for CompiledPooledGaussian {
    fn dim(&self) -> usize {
        // One coordinate per coefficient, plus log sigma.
        self.b_n.len() + 1
    }

    fn logp(&self, theta: &[f64]) -> f64 {
        let p = self.b_n.len();
        let u = theta[p];
        let quad = self.quadratic_form(&theta[..p]);
        let c = -(2.0 * self.a_n + p as f64);
        c * u - (-2.0 * u).exp() * (self.s_n + quad / 2.0)
    }

    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        let p = self.b_n.len();
        if theta.len() != p + 1 || out.len() != p + 1 {
            return Err(BayesError::DimensionMismatch(format!(
                "expected {} coordinates, got theta {} and out {}",
                p + 1,
                theta.len(),
                out.len()
            )));
        }
        let u = theta[p];
        let inv_var = (-2.0 * u).exp();

        // d/dbeta = -e^{-2u} A (beta - b_n)
        let diff: Vec<f64> = (0..p).map(|j| theta[j] - self.b_n[j]).collect();
        let a_diff = self.precision_times(&diff);
        for j in 0..p {
            out[j] = -inv_var * a_diff[j];
        }

        // d/du = c + 2 e^{-2u} (s_n + Q/2)
        let quad: f64 = (0..p).map(|j| diff[j] * a_diff[j]).sum();
        let c = -(2.0 * self.a_n + p as f64);
        out[p] = c + 2.0 * inv_var * (self.s_n + quad / 2.0);
        Ok(())
    }

    fn initial(&self) -> Vec<f64> {
        let mut theta = self.b_n.clone();
        // The joint mode of `u`, obtained by setting the gradient above to zero:
        // sigma^2 = 2 s_n / (2 a_n + p).
        let p = self.b_n.len() as f64;
        let sigma_sq = (2.0 * self.s_n / (2.0 * self.a_n + p)).max(f64::MIN_POSITIVE);
        theta.push(0.5 * sigma_sq.ln());
        theta
    }

    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        let p = self.b_n.len();
        out[..p].copy_from_slice(&theta[..p]);
        out[p] = theta[p].exp();
    }
}

impl CompiledPooledGaussian {
    /// `A x`, where `A = L L'` is the posterior precision.
    fn precision_times(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        // L' x first, then L (that product).
        let mut t = vec![0.0; n];
        for i in 0..n {
            t[i] = (i..n).map(|k| self.chol[(k, i)] * x[k]).sum();
        }
        let mut out = vec![0.0; n];
        for i in 0..n {
            out[i] = (0..=i).map(|k| self.chol[(i, k)] * t[k]).sum();
        }
        out
    }

    fn quadratic_form(&self, beta: &[f64]) -> f64 {
        let diff: Vec<f64> = (0..beta.len()).map(|j| beta[j] - self.b_n[j]).collect();
        let a_diff = self.precision_times(&diff);
        (0..beta.len()).map(|j| diff[j] * a_diff[j]).sum()
    }
}

impl ExactPosterior for CompiledPooledGaussian {
    fn sample_into(&self, rng: &mut BayesRng, out: &mut [f64]) -> BayesResult<()> {
        let p = self.params.len();
        // `out` covers every parameter: the coefficients first, then sigma last.
        if out.len() != p {
            return Err(BayesError::DimensionMismatch(format!(
                "expected {p} slots, got {}",
                out.len()
            )));
        }
        if !(self.a_n > 0.0 && self.s_n > 0.0) {
            out.fill(f64::NAN);
            return Ok(());
        }
        // sigma^2 ~ InvGamma(a_n, s_n) = 1 / Gamma(a_n, rate s_n).
        let sigma_sq = 1.0 / rng.gamma(self.a_n, self.s_n)?;
        sample_mvn(
            &self.chol,
            &self.b_n,
            sigma_sq.sqrt(),
            rng,
            &mut out[..p - 1],
        )?;
        out[p - 1] = sigma_sq.sqrt();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::testing::Frame;
    use crate::types::FitStatus;

    fn compile<'a>(cfg: &str, data: &'a DataView<'a>) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        PooledGaussian.compile(&Config::parse(cfg).unwrap(), data)
    }

    fn draw(model: &dyn CompiledModel, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let exact = model.as_exact().unwrap();
        let p = model.param_names().len();
        let mut rng = BayesRng::for_chain(seed, 0);
        let mut cols = vec![Vec::with_capacity(n); p];
        let mut buf = vec![0.0; p];
        for _ in 0..n {
            exact.sample_into(&mut rng, &mut buf).unwrap();
            for (j, v) in buf.iter().enumerate() {
                cols[j].push(*v);
            }
        }
        cols
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn index_of(model: &dyn CompiledModel, name: &str) -> usize {
        model
            .param_names()
            .iter()
            .position(|p| p.name == name)
            .unwrap_or_else(|| panic!("no parameter named {name}"))
    }

    /// Under a flat prior the posterior mean is exactly the least-squares estimate.
    /// Pinned against a dataset generated from known coefficients with no noise, where
    /// the answer is not merely close but exact.
    #[test]
    fn a_flat_prior_recovers_the_least_squares_coefficients() {
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| i as f64 / 4.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) - 2.0).collect();
        // y = 3 + 2*x1 - 0.5*x2, plus a small deterministic wobble so the residual
        // variance is estimable.
        let y: Vec<f64> = (0..n)
            .map(|i| 3.0 + 2.0 * x1[i] - 0.5 * x2[i] + ((i % 3) as f64 - 1.0) * 0.01)
            .collect();

        let frame = Frame::new(n)
            .numeric("y", y)
            .numeric("x1", x1)
            .numeric("x2", x2);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": ["x1", "x2"]}"#, &view).unwrap();

        let cols = draw(&*model, 100_000, 5);
        assert!((mean(&cols[index_of(&*model, "intercept")]) - 3.0).abs() < 0.01);
        assert!((mean(&cols[index_of(&*model, "beta[x1]")]) - 2.0).abs() < 0.005);
        assert!((mean(&cols[index_of(&*model, "beta[x2]")]) + 0.5).abs() < 0.005);
    }

    /// A Gaussian prior on the coefficients is ridge regression, whose solution is
    /// `(X'X + lambda I)^-1 X'y`. Checking against that closed form pins the prior
    /// precision assembly independently of any sampling.
    #[test]
    fn a_gaussian_prior_reproduces_the_ridge_closed_form() {
        let n = 30;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64) / 3.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 2.0 * x1[i] + ((i % 4) as f64 - 1.5) * 0.2)
            .collect();

        let frame = Frame::new(n)
            .numeric("y", y.clone())
            .numeric("x1", x1.clone());
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let scale = 0.5;
        let model = compile(
            &format!(r#"{{"y": "y", "x": "x1", "prior": {{"beta_scale": {scale}}}}}"#),
            &view,
        )
        .unwrap();

        // Closed form, assembled here by hand.
        let lambda = 1.0 / (scale * scale);
        let (s_x, s_xx): (f64, f64) = (x1.iter().sum(), x1.iter().map(|v| v * v).sum());
        let (s_y, s_xy): (f64, f64) = (y.iter().sum(), x1.iter().zip(&y).map(|(a, b)| a * b).sum());
        // [[n, s_x], [s_x, s_xx + l]] [b0, b1]' = [s_y, s_xy]' -- the penalty sits on
        // the slope only, matching the family's treatment of the intercept.
        let (a11, a12, a22) = (n as f64, s_x, s_xx + lambda);
        let det = a11 * a22 - a12 * a12;
        let b0 = (a22 * s_y - a12 * s_xy) / det;
        let b1 = (a11 * s_xy - a12 * s_y) / det;

        let cols = draw(&*model, 200_000, 6);
        let got0 = mean(&cols[index_of(&*model, "intercept")]);
        let got1 = mean(&cols[index_of(&*model, "beta[x1]")]);
        assert!((got0 - b0).abs() < 0.01, "intercept {got0} vs ridge {b0}");
        assert!((got1 - b1).abs() < 0.005, "slope {got1} vs ridge {b1}");

        // ...and the prior really is doing something: the ridge estimate is shrunk
        // relative to the flat-prior one.
        let flat = compile(r#"{"y": "y", "x": "x1"}"#, &view).unwrap();
        let flat_slope = mean(&draw(&*flat, 50_000, 6)[index_of(&*flat, "beta[x1]")]);
        assert!(
            got1 < flat_slope,
            "ridge slope {got1} should shrink below {flat_slope}"
        );
    }

    /// Partial pooling: a group with few observations is pulled toward zero (the
    /// population level) far more than a group with many. This is the whole reason to
    /// pool -- a thin segment borrows strength instead of reporting noise.
    #[test]
    fn small_groups_shrink_further_than_large_ones() {
        // Both groups genuinely sit 1.0 above the intercept, but SMALL has 3
        // observations and LARGE has 60.
        let mut y = Vec::new();
        let mut g = Vec::new();
        for i in 0..3 {
            y.push(6.0 + ((i % 3) as f64 - 1.0) * 0.1);
            g.push("SMALL");
        }
        for i in 0..60 {
            y.push(6.0 + ((i % 5) as f64 - 2.0) * 0.1);
            g.push("LARGE");
        }
        for i in 0..60 {
            y.push(5.0 + ((i % 5) as f64 - 2.0) * 0.1);
            g.push("BASE");
        }
        let n = y.len();
        let frame = Frame::new(n).numeric("y", y).key("segment", g);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let model = compile(
            r#"{"y": "y", "group": "segment", "pool_scale": 0.5, "intercept": 0}"#,
            &view,
        )
        .unwrap();
        let cols = draw(&*model, 50_000, 8);

        let small = mean(&cols[0]);
        let large = mean(&cols[1]);
        // Both estimate the same underlying level, but the thin group is pulled
        // further toward the prior mean of zero.
        assert!(
            small < large,
            "small {small} should shrink below large {large}"
        );
        assert!(large > 5.5, "large group {large} should stay near its data");
    }

    /// A design with a duplicated predictor has no unique solution. Reporting that is
    /// the point: a pseudo-inverse would pick one of infinitely many answers.
    #[test]
    fn a_rank_deficient_design_refuses_rather_than_inventing_a_solution() {
        let n = 20;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let dup: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();

        let frame = Frame::new(n)
            .numeric("y", y)
            .numeric("x", x)
            .numeric("dup", dup);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"y": "y", "x": ["x", "dup"]}"#, &view).unwrap_err();
        assert!(matches!(err, BayesError::SingularMatrix), "{err}");
    }

    /// A constant predictor alongside an intercept is the same problem wearing a
    /// different hat, and is the single most common way a real dataset is rank
    /// deficient.
    #[test]
    fn a_constant_predictor_alongside_an_intercept_refuses() {
        let n = 20;
        let frame = Frame::new(n)
            .numeric("y", (0..n).map(|i| i as f64).collect())
            .numeric("flat", vec![7.0; n]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        assert!(matches!(
            compile(r#"{"y": "y", "x": "flat"}"#, &view).unwrap_err(),
            BayesError::SingularMatrix
        ));
    }

    #[test]
    fn fewer_observations_than_parameters_is_reported_before_any_solve() {
        let frame = Frame::new(2)
            .numeric("y", vec![1.0, 2.0])
            .numeric("a", vec![1.0, 2.0])
            .numeric("b", vec![3.0, 1.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        assert!(matches!(
            compile(r#"{"y": "y", "x": ["a", "b"]}"#, &view).unwrap_err(),
            BayesError::InsufficientData { .. }
        ));
    }

    /// A design that fits the response exactly leaves no residual variance to
    /// estimate. The draws are NULL-shaped rather than a spurious zero uncertainty,
    /// which would read as total confidence.
    #[test]
    fn a_perfect_fit_is_degenerate_rather_than_infinitely_confident() {
        let n = 10;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 + 3.0 * v).collect();
        let frame = Frame::new(n).numeric("y", y).numeric("x", x);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let model = compile(r#"{"y": "y", "x": "x"}"#, &view).unwrap();
        assert_eq!(model.readiness().status, FitStatus::Degenerate);
        let cols = draw(&*model, 20, 9);
        assert!(cols[0].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn a_model_with_nothing_to_estimate_is_a_config_error() {
        let frame = Frame::new(4).numeric("y", vec![1.0, 2.0, 3.0, 4.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"y": "y", "intercept": 0}"#, &view).unwrap_err();
        assert!(err.to_string().contains("nothing to estimate"), "{err}");
    }

    /// The residual scale must match the textbook `sigma^2 | y ~ Inv-chi^2(n - k, s^2)`
    /// under a flat prior, whose mean is `RSS / (n - k - 2)`. Using `n` in place of
    /// `n - k` makes the posterior too tight by `sqrt((n - k)/n)`, which is an
    /// overconfident interval -- the direction that produces service levels that
    /// quietly under-cover.
    #[test]
    fn the_residual_scale_matches_the_textbook_inverse_chi_squared() {
        let n = 24usize;
        let x1: Vec<f64> = (0..n).map(|i| i as f64 / 3.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) - 2.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 + 1.5 * x1[i] - 0.4 * x2[i] + ((i % 7) as f64 - 3.0) * 0.5)
            .collect();

        let frame = Frame::new(n)
            .numeric("y", y.clone())
            .numeric("x1", x1.clone())
            .numeric("x2", x2.clone());
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": ["x1", "x2"]}"#, &view).unwrap();

        let k = 3usize; // intercept + two slopes, all flat
        let cols = draw(&*model, 200_000, 41);
        let var: Vec<f64> = cols[index_of(&*model, "sigma")].iter().map(|s| s * s).collect();

        // The degrees of freedom are what set the *shape*, so the scale-free
        // signature to check is the coefficient of variation: for InvGamma(a, b) the
        // mean is b/(a-1) and the sd is b/((a-1) sqrt(a-2)), so CV = 1/sqrt(a-2) --
        // independent of `b`, and therefore of the residual sum of squares, which is
        // exactly what makes it a clean test of `a_n` alone.
        let m = var.iter().sum::<f64>() / var.len() as f64;
        let sd = (var.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (var.len() - 1) as f64).sqrt();
        let a_n = (n - k) as f64 / 2.0;
        let expected_cv = 1.0 / (a_n - 2.0).sqrt();
        let cv = sd / m;
        assert!(
            (cv - expected_cv).abs() < 0.03 * expected_cv,
            "coefficient of variation {cv} vs Inv-chi^2(n-k) prediction {expected_cv}"
        );

        // And the wrong shape -- a_n = n/2, ignoring the flat coefficients -- would be
        // visibly tighter. Pinning the distance from it is what stops the test from
        // passing under the bug it was written to catch.
        let wrong_cv = 1.0 / (n as f64 / 2.0 - 2.0).sqrt();
        assert!(
            (cv - wrong_cv).abs() > 0.03 * wrong_cv,
            "cv {cv} must be distinguishable from the a_n = n/2 answer {wrong_cv}"
        );
    }

    /// A *proper* coefficient prior is sigma^2-scaled, so it costs no degrees of
    /// freedom and the shape stays `a0 + n/2`. The correction must apply only where
    /// the prior is flat.
    #[test]
    fn a_proper_coefficient_prior_costs_no_degrees_of_freedom() {
        let n = 24usize;
        let frame = Frame::new(n)
            .numeric("y", (0..n).map(|i| 2.0 + ((i % 7) as f64 - 3.0) * 0.5).collect())
            .numeric("x1", (0..n).map(|i| i as f64 / 3.0).collect());
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        // intercept flat (always), slope proper -> one flat coefficient.
        let model = compile(
            r#"{"y": "y", "x": "x1", "prior": {"beta_scale": 1.0}}"#,
            &view,
        )
        .unwrap();
        let cols = draw(&*model, 200_000, 43);
        let var: Vec<f64> = cols[index_of(&*model, "sigma")].iter().map(|s| s * s).collect();
        let m = var.iter().sum::<f64>() / var.len() as f64;
        let sd = (var.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (var.len() - 1) as f64).sqrt();

        let a_n = (n as f64 - 1.0) / 2.0; // only the intercept is flat
        let expected_cv = 1.0 / (a_n - 2.0).sqrt();
        assert!(
            (sd / m - expected_cv).abs() < 0.03 * expected_cv,
            "cv {} vs expected {expected_cv}",
            sd / m
        );
    }

    #[test]
    fn sigma_is_the_last_parameter_and_is_positive() {
        let n = 30;
        let frame = Frame::new(n)
            .numeric("y", (0..n).map(|i| 5.0 + ((i % 7) as f64 - 3.0)).collect())
            .numeric("x", (0..n).map(|i| i as f64).collect());
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x"}"#, &view).unwrap();

        let names: Vec<&str> = model
            .param_names()
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["intercept", "beta[x]", "sigma"]);
        let cols = draw(&*model, 2000, 10);
        assert!(cols[2].iter().all(|v| *v > 0.0));
    }
}
