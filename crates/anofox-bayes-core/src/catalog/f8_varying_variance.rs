//! `varying_variance_gaussian` — a Gaussian linear model whose **variance components
//! are estimated rather than set**.
//!
//! ```text
//!   y_i ~ N( x_i'beta + eta_{g(i)},  sigma_{g(i)}^2 )
//!
//!   eta_g   = tau * z_g,                    z_g ~ N(0, 1)     (group level)
//!   sigma_g = exp(mu_s + tau_s * w_g),      w_g ~ N(0, 1)     (group spread)
//! ```
//!
//! Two things separate it from [`pooled_gaussian`](super::f3_pooled_gaussian), and
//! both are the reason it exists:
//!
//! * **`sigma` is per group.** `pooled_gaussian` has one residual scale for the whole
//!   design, so two groups with the same mean necessarily get the same predictive
//!   interval. A decision about a *tail* — a service level, a worst-case payment
//!   delay, a segment that is merely noisier rather than worse — is then unanswerable
//!   by construction, whatever the data says.
//! * **`pool_scale` is learned.** In `pooled_gaussian` it is an analyst dial, which
//!   means the amount of shrinkage is an assumption rather than a finding. Here `tau`
//!   is a parameter with a posterior, and its own uncertainty propagates into every
//!   group effect.
//!
//! Both changes destroy conjugacy, which is why this is a separate family rather than
//! a mode of `pooled_gaussian` (`ROADMAP.md` §3.3). `pooled_gaussian`'s warranty is a
//! closed-form posterior cross-checked by three engines; a family that is sometimes
//! exact and sometimes sampled, decided by a config slot, has no warranty at all.
//!
//! ## On the name
//!
//! `hierarchical_gaussian` was the obvious candidate and was rejected: `pooled_gaussian`
//! is hierarchical too, in structure if not in inference, so the pair would give a
//! caller nothing to choose on. `heteroscedastic_gaussian` says the right thing but has
//! two accepted spellings in the literature (`-sc-` and `-sk-`), and a family id is a
//! string a caller types and that feeds `model_id`. What is left is the plain
//! description of the discriminating capability: the variance **varies** — within a
//! group it is that group's own, and between groups it is estimated rather than fixed.
//!
//! ## Parameterisation, which is baked in
//!
//! **Non-centred, from the start and not as an option.** The centred form
//! `eta_g ~ N(0, tau^2)` puts the group effects and their own scale in a funnel
//! (Neal 2003): where `tau` is small the admissible `eta` shrink with it, so the
//! posterior has curvature that changes by orders of magnitude along `log tau` and no
//! single step size works anywhere. Writing `eta_g = tau * z_g` makes `z` a priori
//! standard normal and independent of `tau`, which is the geometry a diagonal mass
//! matrix can actually precondition. The measured warning that motivated this is
//! recorded in `docs/THEORY.md` §5: `pooled_gaussian` under NUTS already mixes badly
//! on a *fixed*-scale hierarchy, and a learned scale is the textbook worse case.
//!
//! The BRD's premise is that callers cannot select a bad parameterisation, so there is
//! no `centred` slot to get wrong.
//!
//! **Sampling is on the log scale for every positive quantity** — `log tau`,
//! `log sigma_pop`, `log tau_s` — so a draw cannot be negative and a Gaussian
//! approximation is fitted to something that is not bounded at zero.
//!
//! **The priors on `tau` and `tau_s` are declared on the natural scale, not the log
//! scale**, and that is a deliberate difference from `payer_alive`. A flat prior on
//! `log tau` is `p(tau) ∝ 1/tau`, which for a hierarchical variance parameter gives an
//! **improper posterior** — the classic failure, since the likelihood is bounded as
//! `tau → 0` and `1/tau` is not integrable there. Flat on `tau` itself is proper for
//! three or more groups (Gelman 2006), so that is the default, and the log-Jacobian
//! `+ log tau` appears explicitly in the density. `the_log_jacobian_of_log_tau_is_present`
//! pins it: a missing Jacobian is an `O(1/G)` perturbation that hides inside any
//! engine-agreement tolerance (`ROADMAP.md` §2), so it is tested directly.
//!
//! ## Engine
//!
//! NUTS by default. Laplace is available and is *not* certified for the variance
//! components: see `sbc.rs::families` and `docs/THEORY.md`. A Gaussian fitted at the
//! mode on the unconstrained scale is least honest exactly where a variance parameter
//! is near zero, which is the regime a thin group puts it in.

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::linalg::{cholesky, solve_with};
use crate::types::{EngineKind, FamilyCode};

use super::{CompiledModel, LogPosterior, ModelFamily, Readiness};
use faer::Mat;

#[derive(Debug)]
pub struct VaryingVarianceGaussian;

const SLOTS: &[&str] = &[
    "y",
    "x",
    "intercept",
    "group",
    "prior",
    "draws",
    "chains",
    "warmup",
    "max_draw_megabytes",
    "seed",
    "engine",
    "sample_from",
];

const PRIOR_SLOTS: &[&str] = &[
    "beta_scale",
    "intercept_scale",
    "pool_scale",
    "sigma_spread",
    "sigma_log_mean",
    "sigma_log_sd",
];

/// Groups below which the between-group variance is not identified.
///
/// With two groups there is one difference, and one number cannot separate "the groups
/// differ" from "this group is noisy": the flat prior on `tau` then leaves the
/// posterior improper in the same way `p(tau) ∝ 1/tau` does. Three is the smallest
/// count for which Gelman (2006) reports the flat prior proper, and it is a refusal
/// rather than a config slot because raising it would not make the number estimable.
const MIN_GROUPS: usize = 3;

impl ModelFamily for VaryingVarianceGaussian {
    fn id(&self) -> &'static str {
        "varying_variance_gaussian"
    }

    fn code(&self) -> FamilyCode {
        FamilyCode::VaryingVarianceGaussian
    }

    fn description(&self) -> &'static str {
        "Gaussian linear model with a residual scale per group and a learned pooling \
         scale, non-centred; the family for questions about a group's tail rather than \
         its level."
    }

    fn default_engine(&self) -> EngineKind {
        // Not a preference. The variance components are exactly where a Gaussian
        // approximation at the mode is least honest, and the SBC suite says so per
        // engine rather than leaving it to argument.
        EngineKind::Nuts
    }

    fn config_slots(&self) -> &'static [&'static str] {
        SLOTS
    }

    fn compile<'a>(
        &self,
        cfg: &Config,
        data: &'a DataView<'a>,
    ) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        Ok(Box::new(build(cfg, data)?))
    }
}

/// Independent Normal priors on the unconstrained fixed effects, plus the two
/// hyperpriors that make this family hierarchical.
///
/// `pool_scale` and `sigma_spread` are half-Normal **on the natural scale**; an
/// infinite scale is the flat default. See the module header for why flatness is
/// declared there rather than on the log scale.
#[derive(Debug, Clone, Copy)]
struct Prior {
    intercept_scale: f64,
    beta_scale: f64,
    /// Half-Normal sd for `tau`, the between-group sd of the group effects.
    ///
    /// **This one has a default, and the other families' argument for flatness does
    /// not carry here.** `docs/THEORY.md` §3 rejects concrete defaults because they are
    /// claims about *scale*, and a claim about scale is wrong for every customer whose
    /// units differ from the author's. The default here is not a number: it is the
    /// response's own standard deviation, which rescales with the data and therefore
    /// asserts nothing about units. What it does assert is that the spread *between*
    /// groups is not many times the spread of the response as a whole, which is true
    /// of every dataset for which this family is the right model.
    ///
    /// The reason it is not flat is measured rather than aesthetic. Under a flat prior
    /// on `tau` the posterior is proper (Gelman 2006) but its upper tail is long, and
    /// the sampler diverges there: on the eight-group fixture, 34 divergences in 8 000
    /// draws, concentrated at `pool_scale` averaging 5.7 against a bulk of 2.3. Every
    /// divergence is a refusal under `Thresholds::max_divergent = 0`, so a flat default
    /// would have made the family report `degenerate` on clean simulated data. See
    /// `the_default_pooling_hyperprior_is_the_responses_own_scale`.
    ///
    /// A caller who sets it explicitly gets exactly what they set, and the SBC suite
    /// must, because a data-derived prior scale cannot be drawn from.
    pool_scale: f64,
    /// Half-Normal sd for `tau_s`, the sd of `log sigma` across groups.
    sigma_spread: f64,
    /// Normal prior on `log sigma_pop`, the population-level log residual scale.
    sigma_log_mean: f64,
    sigma_log_sd: f64,
}

impl Prior {
    /// Default sd of `log sigma` across groups, in **log units**.
    ///
    /// A concrete number is admissible here where it would not be on the response's
    /// own scale: one log unit is a factor of `e`, which means the same thing in
    /// euros, cents and kilograms. One is deliberately loose -- it puts a 95 % prior
    /// interval of roughly a factor of 50 between the quietest and noisiest group --
    /// while still excluding the tail that has no finite curvature.
    const DEFAULT_SIGMA_SPREAD: f64 = 1.0;

    /// `response_sd` is the standard deviation of `y`, used as the default scale of
    /// the `pool_scale` hyperprior. See [`Prior::pool_scale`].
    fn parse(cfg: &Config, response_sd: f64) -> BayesResult<Self> {
        cfg.reject_unknown(PRIOR_SLOTS)?;
        let prior = Prior {
            intercept_scale: cfg.positive_f64_or("intercept_scale", f64::INFINITY)?,
            beta_scale: cfg.positive_f64_or("beta_scale", f64::INFINITY)?,
            pool_scale: cfg.positive_f64_or("pool_scale", response_sd)?,
            sigma_spread: cfg.positive_f64_or("sigma_spread", Self::DEFAULT_SIGMA_SPREAD)?,
            sigma_log_mean: cfg.f64_or("sigma_log_mean", 0.0)?,
            sigma_log_sd: cfg.positive_f64_or("sigma_log_sd", f64::INFINITY)?,
        };
        Ok(prior)
    }
}

/// A half-Normal(0, `scale`) log density at `v > 0`, dropping the constant. An
/// infinite scale is flat and contributes nothing.
fn half_normal_logp(v: f64, scale: f64) -> f64 {
    if scale.is_finite() {
        -0.5 * (v * v) / (scale * scale)
    } else {
        0.0
    }
}

/// Its derivative with respect to `log v`, which is the coordinate actually sampled.
fn half_normal_dlog(v: f64, scale: f64) -> f64 {
    if scale.is_finite() {
        -(v * v) / (scale * scale)
    } else {
        0.0
    }
}

fn normal_logp(v: f64, mean: f64, sd: f64) -> f64 {
    if sd.is_finite() {
        let z = (v - mean) / sd;
        -0.5 * z * z
    } else {
        0.0
    }
}

fn normal_dlog(v: f64, mean: f64, sd: f64) -> f64 {
    if sd.is_finite() {
        -(v - mean) / (sd * sd)
    } else {
        0.0
    }
}

fn build(cfg: &Config, data: &DataView) -> BayesResult<CompiledVaryingVariance> {
    cfg.reject_unknown(SLOTS)?;

    let y_name = cfg.require_str("y")?.to_string();
    let x_names = cfg.str_list("x")?;
    let intercept = cfg.f64_or("intercept", 1.0)? != 0.0;
    // Required, unlike `pooled_gaussian`'s optional slot: a family whose subject is
    // per-group variance has nothing to say about a dataset with no groups, and
    // silently degenerating to one shared sigma would be `pooled_gaussian` wearing a
    // name that promises otherwise.
    let group_name = cfg.require_str("group")?.to_string();

    if cfg.opt_str("sample_from")? == Some("prior") {
        return Err(BayesError::config(
            "sample_from",
            "this family has no closed-form prior draw; a prior-predictive check needs \
             the exact engine, which no non-conjugate family offers",
        ));
    }

    let mut numeric_cols: Vec<&str> = vec![y_name.as_str()];
    numeric_cols.extend(x_names.iter().map(String::as_str));
    let key_cols = [group_name.as_str()];

    let rows = data.usable_rows(&numeric_cols, &key_cols)?;
    let fingerprint = data.fingerprint(&numeric_cols, &key_cols, &rows)?;

    let y_col = data.numeric(&y_name)?;
    let x_cols: Vec<_> = x_names
        .iter()
        .map(|n| data.numeric(n))
        .collect::<BayesResult<_>>()?;
    let groups = data.group_rows(Some(group_name.as_str()), &rows)?;
    for (key, _) in &groups {
        crate::types::validate_group_key(key)?;
    }

    let n = rows.len();
    let n_groups = groups.len();

    // The response's own spread, which is the default scale of the `pool_scale`
    // hyperprior. Computed over the usable rows only, so it describes the data the
    // model will actually see.
    let response_sd = {
        let vals: Vec<f64> = rows.iter().map(|&i| y_col.values[i]).collect();
        let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>()
            / (vals.len().saturating_sub(1)).max(1) as f64;
        // A constant response has no spread; the readiness check below refuses it, and
        // until then a positive placeholder keeps the prior a distribution.
        var.sqrt().max(f64::MIN_POSITIVE)
    };
    let prior = Prior::parse(&cfg.nested("prior")?, response_sd)?;
    let n_fixed = usize::from(intercept) + x_names.len();
    if n_fixed == 0 {
        return Err(BayesError::config(
            "x",
            "a model with no intercept and no predictors has no linear part; the group \
             effects are deviations from something, and there is nothing to deviate from",
        ));
    }

    // Coefficients, three hyperparameters, and two parameters per group.
    let n_params = n_fixed + 3 + 2 * n_groups;
    if n <= n_fixed + n_groups {
        return Err(BayesError::InsufficientData {
            rows: n,
            params: n_fixed + n_groups,
        });
    }

    // --- Rows, laid out group-major so a group's observations are contiguous. ---
    let mut y = Vec::with_capacity(n);
    let mut x: Mat<f64> = Mat::zeros(n, n_fixed);
    let mut offsets = Vec::with_capacity(n_groups + 1);
    offsets.push(0usize);
    let mut r = 0usize;
    for (_, idx) in &groups {
        for &row in idx {
            y.push(y_col.values[row]);
            let mut c = 0;
            if intercept {
                x[(r, c)] = 1.0;
                c += 1;
            }
            for col in &x_cols {
                x[(r, c)] = col.values[row];
                c += 1;
            }
            r += 1;
        }
        offsets.push(r);
    }

    // --- Parameter identities, in the order `constrain` writes them. ---
    let mut params: Vec<ParamName> = Vec::with_capacity(n_params);
    if intercept {
        params.push(ParamName::global("intercept")?);
    }
    for name in &x_names {
        params.push(ParamName::global(format!("beta[{name}]"))?);
    }
    params.push(ParamName::global("pool_scale")?);
    params.push(ParamName::global("sigma_pop")?);
    params.push(ParamName::global("sigma_spread")?);
    for (key, _) in &groups {
        params.push(ParamName::grouped(key.clone(), "group_effect")?);
    }
    for (key, _) in &groups {
        params.push(ParamName::grouped(key.clone(), "sigma")?);
    }
    debug_assert_eq!(params.len(), n_params);

    let readiness = assess(&y, &offsets, n_groups, &group_name);

    let mut model = CompiledVaryingVariance {
        params,
        y,
        x,
        offsets,
        n_fixed,
        n_groups,
        intercept,
        prior,
        fingerprint,
        readiness,
        start: Vec::new(),
    };
    model.start = model.starting_point();
    Ok(model)
}

/// The refusals visible from the sufficient statistics alone.
fn assess(y: &[f64], offsets: &[usize], n_groups: usize, group_name: &str) -> Readiness {
    if n_groups < MIN_GROUPS {
        return Readiness::insufficient(format!(
            "'{group_name}' has {n_groups} group(s); the between-group scale `pool_scale` \
             is a parameter here rather than a setting, and fewer than {MIN_GROUPS} groups \
             cannot identify it. Use 'pooled_gaussian' with a `pool_scale` you choose, or \
             coarsen the grouping less"
        ));
    }
    // A group of one observation contributes nothing about its own spread, and that is
    // fine -- borrowing it from the rest is the whole point of the hyperprior. What is
    // not fine is *every* group being like that, because then the spread of spreads has
    // nothing to estimate itself from.
    let informative = (0..n_groups)
        .filter(|&g| offsets[g + 1] - offsets[g] >= 2)
        .count();
    if informative < 2 {
        return Readiness::insufficient(format!(
            "only {informative} group(s) have two or more observations, so the spread of \
             the per-group variances is not identified: `sigma_spread` would be reporting \
             its prior"
        ));
    }
    // Every observation identical means zero residual variance under any assignment of
    // the group effects, and `log sigma` runs to minus infinity.
    let first = y[0];
    if y.iter().all(|v| *v == first) {
        return Readiness::degenerate(
            "every observation is identical, so the residual scale is zero and \
             `log sigma` has no mode"
                .to_string(),
        );
    }
    Readiness::ready()
}

/// The compiled model.
///
/// Rows are stored group-major and the design densely, because every gradient
/// evaluation walks all of them: NUTS asks for thousands, so the layout is chosen for
/// the inner loop rather than for the one-off compile.
#[derive(Debug)]
pub(crate) struct CompiledVaryingVariance {
    params: Vec<ParamName>,
    y: Vec<f64>,
    x: Mat<f64>,
    /// `offsets[g]..offsets[g + 1]` are group `g`'s rows.
    offsets: Vec<usize>,
    n_fixed: usize,
    n_groups: usize,
    /// Whether column 0 of the design is the intercept. It alone is unpenalised by
    /// default, so the prior loop has to know where it is.
    intercept: bool,
    prior: Prior,
    fingerprint: String,
    readiness: Readiness,
    start: Vec<f64>,
}

impl CompiledVaryingVariance {
    /// Offsets of the unconstrained coordinate blocks.
    ///
    /// Written once, here, rather than open-coded in `logp`, `grad` and `constrain`,
    /// because three copies of an index arithmetic is three chances to disagree and the
    /// disagreement would be a silently wrong posterior rather than a crash.
    fn layout(&self) -> Layout {
        let z = self.n_fixed;
        let lt = z + self.n_groups;
        Layout {
            fixed: 0,
            z,
            lt,
            lsm: lt + 1,
            w: lt + 2,
            lts: lt + 2 + self.n_groups,
        }
    }

    /// A starting point good enough for a Newton search and for overdispersing NUTS
    /// chains around.
    ///
    /// The fixed effects come from an ordinary least-squares fit ignoring the groups,
    /// which is cheap and is the right answer when the group effects are small. Where
    /// that solve fails — a rank-deficient design — the fallback is the response mean
    /// on the intercept and zero elsewhere, which is finite and therefore usable, which
    /// is all `initial()` has to be.
    fn starting_point(&self) -> Vec<f64> {
        let n = self.y.len();
        let p = self.n_fixed;
        let mut a: Mat<f64> = Mat::zeros(p, p);
        let mut xty = vec![0.0; p];
        for j in 0..p {
            for k in 0..p {
                a[(j, k)] = (0..n).map(|r| self.x[(r, j)] * self.x[(r, k)]).sum();
            }
            xty[j] = (0..n).map(|r| self.x[(r, j)] * self.y[r]).sum();
        }
        let mean_y = self.y.iter().sum::<f64>() / n as f64;
        let beta = cholesky(&a)
            .and_then(|l| solve_with(&l, &xty))
            .ok()
            .filter(|b| b.iter().all(|v| v.is_finite()))
            .unwrap_or_else(|| {
                let mut b = vec![0.0; p];
                b[0] = mean_y;
                b
            });

        // Residual scale under that fit, and the spread of the group means around it:
        // rough, but on the right order of magnitude, which is what a log-scale start
        // needs to be.
        let fitted = |r: usize| (0..p).map(|j| self.x[(r, j)] * beta[j]).sum::<f64>();
        let rss: f64 = (0..n).map(|r| (self.y[r] - fitted(r)).powi(2)).sum();
        let sigma = (rss / (n.saturating_sub(p).max(1)) as f64).sqrt().max(1e-6);

        let mut group_means = Vec::with_capacity(self.n_groups);
        for g in 0..self.n_groups {
            let (lo, hi) = (self.offsets[g], self.offsets[g + 1]);
            let m: f64 = (lo..hi).map(|r| self.y[r] - fitted(r)).sum::<f64>() / (hi - lo) as f64;
            group_means.push(m);
        }
        let gbar = group_means.iter().sum::<f64>() / self.n_groups as f64;
        let tau = (group_means.iter().map(|m| (m - gbar).powi(2)).sum::<f64>()
            / self.n_groups as f64)
            .sqrt()
            .max(1e-3 * sigma);

        let mut theta = vec![0.0; self.dim_of()];
        let l = self.layout();
        theta[l.fixed..l.fixed + p].copy_from_slice(&beta);
        theta[l.lt] = tau.ln();
        theta[l.lsm] = sigma.ln();
        // Half a log unit of spread across groups: neither a claim that the variances
        // are equal (which zero would be, and which is a boundary a search cannot
        // leave cleanly) nor a large one.
        theta[l.lts] = 0.5f64.ln();
        theta
    }

    fn has_intercept(&self) -> bool {
        self.intercept
    }

    /// Whether the compile-time verdict was that there is no posterior here.
    ///
    /// **Without this the family breaks its own refusal contract.** `assess`
    /// already detects a constant response and returns `Readiness::degenerate`
    /// — but a verdict only decides the `__status__` row, and the engine still
    /// runs. On a likelihood with no interior maximum NUTS cannot find a usable
    /// starting point at all, and the fit came back as
    /// `internal error: NUTS could not find a usable starting point` instead of
    /// as the `degenerate` row with NULL draws the caller is promised. An agent
    /// branching on `__status__` never saw it, because there was no table.
    ///
    /// The fix is the one `hier_negbin` documents and `payment_delay` and
    /// `hier_elasticity` inherit: expose a trivially-explorable standard normal
    /// whose `constrain` yields NaN, so the refusal travels as data.
    ///
    /// `InsufficientData` is deliberately not one of these — that verdict says
    /// the data is weak, not that the surface is unusable, so the draws are real
    /// and it is the status that refuses.
    fn refuses(&self) -> bool {
        matches!(
            self.readiness.status,
            crate::types::FitStatus::Degenerate | crate::types::FitStatus::Failed
        )
    }

    fn dim_of(&self) -> usize {
        self.n_fixed + 2 * self.n_groups + 3
    }
}

/// Where each block of unconstrained coordinates begins.
#[derive(Debug, Clone, Copy)]
struct Layout {
    fixed: usize,
    z: usize,
    lt: usize,
    lsm: usize,
    w: usize,
    lts: usize,
}

impl CompiledModel for CompiledVaryingVariance {
    fn param_names(&self) -> &[ParamName] {
        &self.params
    }
    fn n_obs(&self) -> usize {
        self.y.len()
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
    fn as_differentiable(&self) -> Option<&dyn LogPosterior> {
        Some(self)
    }
}

impl LogPosterior for CompiledVaryingVariance {
    fn dim(&self) -> usize {
        self.dim_of()
    }

    /// ```text
    ///   log p = sum_g [ -n_g S_g - exp(-2 S_g) SS_g / 2 ]      (likelihood)
    ///         - (z'z + w'w)/2                                  (non-centred priors)
    ///         - tau^2/(2 T^2)     + log tau                    (half-Normal + Jacobian)
    ///         - tau_s^2/(2 S^2)   + log tau_s                  (half-Normal + Jacobian)
    ///         - ((mu_s - m)/s)^2/2 - (fixed-effect priors)
    /// ```
    ///
    /// with `S_g = mu_s + tau_s w_g` the group's log scale and
    /// `SS_g = sum_{i in g} (y_i - x_i'beta - tau z_g)^2`. Normalising constants are
    /// dropped throughout, uniformly, which is what the trait permits and what makes
    /// the closed-form tests compare *differences*.
    fn logp(&self, theta: &[f64]) -> f64 {
        if self.refuses() {
            // See `refuses`: a trivially-explorable surface, so the refusal
            // reaches SQL as a `degenerate` row with NULL draws instead of as an
            // engine failure.
            return -0.5 * theta.iter().map(|v| v * v).sum::<f64>();
        }
        let l = self.layout();
        let p = self.n_fixed;
        let tau = theta[l.lt].exp();
        let tau_s = theta[l.lts].exp();
        let mu_s = theta[l.lsm];

        let mut acc = 0.0;
        for g in 0..self.n_groups {
            let s = mu_s + tau_s * theta[l.w + g];
            let inv_var = (-2.0 * s).exp();
            let shift = tau * theta[l.z + g];
            let (lo, hi) = (self.offsets[g], self.offsets[g + 1]);
            let mut ss = 0.0;
            for r in lo..hi {
                let fitted: f64 = (0..p).map(|j| self.x[(r, j)] * theta[l.fixed + j]).sum();
                let resid = self.y[r] - fitted - shift;
                ss += resid * resid;
            }
            acc -= (hi - lo) as f64 * s + 0.5 * inv_var * ss;
            acc -= 0.5 * (theta[l.z + g] * theta[l.z + g] + theta[l.w + g] * theta[l.w + g]);
        }

        // The two hyperpriors, each on the natural scale, each with the log-Jacobian
        // of the log transform that is actually sampled.
        acc += half_normal_logp(tau, self.prior.pool_scale) + theta[l.lt];
        acc += half_normal_logp(tau_s, self.prior.sigma_spread) + theta[l.lts];
        acc += normal_logp(mu_s, self.prior.sigma_log_mean, self.prior.sigma_log_sd);

        // Fixed effects. The intercept is not penalised by default, for the reason
        // `pooled_gaussian` gives: a prior centred at zero on the scale of the response
        // says something nobody means.
        let mut j = 0;
        if self.has_intercept() {
            acc += normal_logp(theta[l.fixed], 0.0, self.prior.intercept_scale);
            j = 1;
        }
        while j < p {
            acc += normal_logp(theta[l.fixed + j], 0.0, self.prior.beta_scale);
            j += 1;
        }
        acc
    }

    /// The analytic gradient. Every term is one line because the residual and the
    /// group's precision are shared between all of them:
    ///
    /// ```text
    ///   d/dbeta_j = sum_i v_g r_i x_ij
    ///   d/dz_g    = tau v_g R_g - z_g                    R_g = sum_{i in g} r_i
    ///   d/dlog tau= tau sum_g z_g v_g R_g - tau^2/T^2 + 1
    ///   d/dS_g    = -n_g + v_g SS_g
    ///   d/dmu_s   = sum_g d/dS_g   - (mu_s - m)/s^2
    ///   d/dw_g    = tau_s d/dS_g   - w_g
    ///   d/dlog tau_s = tau_s sum_g w_g d/dS_g - tau_s^2/S^2 + 1
    /// ```
    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        let dim = self.dim_of();
        if theta.len() != dim || out.len() != dim {
            return Err(BayesError::DimensionMismatch(format!(
                "expected {dim} coordinates, got theta {} and out {}",
                theta.len(),
                out.len()
            )));
        }
        if self.refuses() {
            for (slot, v) in out.iter_mut().zip(theta) {
                *slot = -v;
            }
            return Ok(());
        }
        let l = self.layout();
        let p = self.n_fixed;
        let tau = theta[l.lt].exp();
        let tau_s = theta[l.lts].exp();
        let mu_s = theta[l.lsm];

        out.fill(0.0);
        let mut d_lt = 0.0;
        let mut d_lsm = 0.0;
        let mut d_lts = 0.0;

        for g in 0..self.n_groups {
            let s = mu_s + tau_s * theta[l.w + g];
            let inv_var = (-2.0 * s).exp();
            let shift = tau * theta[l.z + g];
            let (lo, hi) = (self.offsets[g], self.offsets[g + 1]);

            let mut sum_resid = 0.0;
            let mut ss = 0.0;
            for r in lo..hi {
                let fitted: f64 = (0..p).map(|j| self.x[(r, j)] * theta[l.fixed + j]).sum();
                let resid = self.y[r] - fitted - shift;
                sum_resid += resid;
                ss += resid * resid;
                for j in 0..p {
                    out[l.fixed + j] += inv_var * resid * self.x[(r, j)];
                }
            }

            // Group effect: through `eta_g = tau * z_g`, so `tau` picks up `z_g`'s
            // share of the same derivative. This is where the non-centring is visible
            // in the arithmetic -- `tau` and `z` share one residual term rather than
            // `z` being conditionally normal given `tau`.
            out[l.z + g] += tau * inv_var * sum_resid - theta[l.z + g];
            d_lt += tau * theta[l.z + g] * inv_var * sum_resid;

            // Log scale: `d/dS_g` of `-n_g S_g - exp(-2 S_g) SS_g / 2`.
            let d_s = -((hi - lo) as f64) + inv_var * ss;
            d_lsm += d_s;
            out[l.w + g] += tau_s * d_s - theta[l.w + g];
            d_lts += tau_s * theta[l.w + g] * d_s;
        }

        // Hyperpriors, and the `+1` each Jacobian contributes.
        out[l.lt] = d_lt + half_normal_dlog(tau, self.prior.pool_scale) + 1.0;
        out[l.lts] = d_lts + half_normal_dlog(tau_s, self.prior.sigma_spread) + 1.0;
        out[l.lsm] = d_lsm + normal_dlog(mu_s, self.prior.sigma_log_mean, self.prior.sigma_log_sd);

        let mut j = 0;
        if self.has_intercept() {
            out[l.fixed] += normal_dlog(theta[l.fixed], 0.0, self.prior.intercept_scale);
            j = 1;
        }
        while j < p {
            out[l.fixed + j] += normal_dlog(theta[l.fixed + j], 0.0, self.prior.beta_scale);
            j += 1;
        }
        Ok(())
    }

    fn initial(&self) -> Vec<f64> {
        if self.refuses() {
            return vec![0.0; self.dim_of()];
        }
        self.start.clone()
    }

    /// Raised from the engine's 0.8, because this posterior's curvature varies sharply
    /// along `log tau` and a step tuned to the bulk overshoots in the upper tail. See
    /// `LogPosterior::target_accept`; measured effect is in
    /// `the_sampler_clears_a_well_posed_panel_without_a_single_divergence`.
    fn target_accept(&self) -> f64 {
        0.95
    }

    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        if self.refuses() {
            out.fill(f64::NAN);
            return;
        }
        let l = self.layout();
        let p = self.n_fixed;
        out[..p].copy_from_slice(&theta[l.fixed..l.fixed + p]);
        let tau = theta[l.lt].exp();
        let tau_s = theta[l.lts].exp();
        let mu_s = theta[l.lsm];
        out[p] = tau;
        out[p + 1] = mu_s.exp();
        out[p + 2] = tau_s;
        for g in 0..self.n_groups {
            out[p + 3 + g] = tau * theta[l.z + g];
            out[p + 3 + self.n_groups + g] = (mu_s + tau_s * theta[l.w + g]).exp();
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::data::testing::Frame;
    use crate::engines::Engine;
    use crate::rng::BayesRng;

    pub(crate) fn compile<'a>(
        cfg: &str,
        data: &'a DataView<'a>,
    ) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        VaryingVarianceGaussian.compile(&Config::parse(cfg).unwrap(), data)
    }

    /// A panel with a known level per group, a known slope, and a **deliberately
    /// unequal** spread per group -- which is the whole subject of this family.
    pub(crate) fn panel(
        seed: u64,
        levels: &[f64],
        sigmas: &[f64],
        n_per_group: usize,
        slope: f64,
    ) -> Frame {
        let mut rng = BayesRng::for_chain(seed, 0);
        let (mut y, mut x, mut g) = (Vec::new(), Vec::new(), Vec::new());
        let keys: Vec<String> = (0..levels.len()).map(|i| format!("G{i:02}")).collect();
        for (i, (&level, &sigma)) in levels.iter().zip(sigmas).enumerate() {
            for j in 0..n_per_group {
                let xv = ((j % 7) as f64 - 3.0) / 3.0;
                y.push(level + slope * xv + sigma * rng.standard_normal());
                x.push(xv);
                g.push(keys[i].clone());
            }
        }
        let n = y.len();
        Frame::new(n)
            .numeric("y", y)
            .numeric("x", x)
            .key("segment", g.iter().map(String::as_str).collect())
    }

    #[test]
    fn the_family_is_in_the_catalog_under_its_own_code() {
        let family = crate::catalog::lookup("varying_variance_gaussian").unwrap();
        assert_eq!(family.code(), FamilyCode::VaryingVarianceGaussian);
        assert_eq!(family.default_engine(), EngineKind::Nuts);
    }

    /// Parameter identities and their order, which `constrain` writes into and which
    /// every downstream consumer joins on.
    #[test]
    fn the_parameters_are_the_coefficients_three_hyperparameters_and_two_per_group() {
        let frame = panel(1, &[10.0, 12.0, 11.0], &[1.0, 1.0, 1.0], 20, 0.5);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();

        let names: Vec<String> = model
            .param_names()
            .iter()
            .map(|p| format!("{}/{}", p.group_id, p.name))
            .collect();
        assert_eq!(
            names,
            vec![
                "__global__/intercept",
                "__global__/beta[x]",
                "__global__/pool_scale",
                "__global__/sigma_pop",
                "__global__/sigma_spread",
                "G00/group_effect",
                "G01/group_effect",
                "G02/group_effect",
                "G00/sigma",
                "G01/sigma",
                "G02/sigma",
            ]
        );
        // 2 fixed + 3 hyper + 2 x 3 groups
        assert_eq!(model.param_names().len(), 11);
        // 2 fixed + 3 groups of z + log tau + log sigma_pop + 3 groups of w + log tau_s
        assert_eq!(model.as_differentiable().unwrap().dim(), 11);
    }

    /// A dataset with no groups has nothing this family can say. Degenerating silently
    /// to one shared sigma would be `pooled_gaussian` under a name that promises the
    /// opposite.
    #[test]
    fn the_group_column_is_required_rather_than_optional() {
        let frame = panel(2, &[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], 10, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"y": "y", "x": "x"}"#, &view).unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "group"),
            "{err}"
        );
    }

    /// `pool_scale` is a *parameter* here, so two groups cannot identify it. Refusing
    /// and naming the family that takes it as a setting is worth more than a number.
    #[test]
    fn two_groups_cannot_identify_a_learned_pooling_scale() {
        let frame = panel(3, &[1.0, 2.0], &[1.0, 1.0], 30, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();
        assert_eq!(
            model.readiness().status,
            crate::types::FitStatus::InsufficientData
        );
        assert!(
            model.readiness().reasons[0].contains("pooled_gaussian"),
            "{:?}",
            model.readiness().reasons
        );
    }

    //=== The log density, checked against its closed form directly ==============//
    //
    // Engine agreement does *not* catch a missing log-Jacobian: it is an O(1/n)
    // perturbation that hides inside any tolerance loose enough not to flake, and
    // `ROADMAP.md` §2 records it being proved so by mutation on `conjugate_anomaly`.
    // This family has three transforms, so all three are tested here, on the density
    // itself, against arithmetic written out by hand.

    /// A small fixed dataset, written out rather than generated, so the reference
    /// below is arithmetic a reader can check.
    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<usize>, Frame) {
        let y = vec![
            10.2, 9.7, 10.9, 11.4, 12.6, 11.9, 13.4, 12.1, 8.8, 9.4, 9.1, 8.2,
        ];
        let x = vec![
            -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0,
        ];
        let g = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let keys = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"];
        let frame = Frame::new(12)
            .numeric("y", y.clone())
            .numeric("x", x.clone())
            .key("segment", keys.to_vec());
        (y, x, g, frame)
    }

    /// The model's log density, written out from the specification in the module
    /// header with no reference to the implementation. Up to an additive constant.
    #[allow(clippy::too_many_arguments)]
    fn reference_logp(
        y: &[f64],
        x: &[f64],
        g: &[usize],
        n_groups: usize,
        theta: &[f64],
        prior: &Prior,
    ) -> f64 {
        let (a, b) = (theta[0], theta[1]);
        let z = &theta[2..2 + n_groups];
        let lt = theta[2 + n_groups];
        let lsm = theta[3 + n_groups];
        let w = &theta[4 + n_groups..4 + 2 * n_groups];
        let lts = theta[4 + 2 * n_groups];
        let (tau, tau_s) = (lt.exp(), lts.exp());

        let mut acc = 0.0;
        for i in 0..y.len() {
            let s = lsm + tau_s * w[g[i]];
            let mu = a + b * x[i] + tau * z[g[i]];
            acc += -s - (y[i] - mu).powi(2) / (2.0 * (2.0 * s).exp());
        }
        for j in 0..n_groups {
            acc += -0.5 * z[j] * z[j];
            acc += -0.5 * w[j] * w[j];
        }
        // Half-Normal on the natural scale, plus the log-Jacobian of log tau.
        if prior.pool_scale.is_finite() {
            acc += -0.5 * tau * tau / (prior.pool_scale * prior.pool_scale);
        }
        acc += lt;
        if prior.sigma_spread.is_finite() {
            acc += -0.5 * tau_s * tau_s / (prior.sigma_spread * prior.sigma_spread);
        }
        acc += lts;
        if prior.sigma_log_sd.is_finite() {
            acc += -0.5 * ((lsm - prior.sigma_log_mean) / prior.sigma_log_sd).powi(2);
        }
        if prior.intercept_scale.is_finite() {
            acc += -0.5 * (a / prior.intercept_scale).powi(2);
        }
        if prior.beta_scale.is_finite() {
            acc += -0.5 * (b / prior.beta_scale).powi(2);
        }
        acc
    }

    fn points(n_groups: usize) -> Vec<Vec<f64>> {
        let dim = 2 + 2 * n_groups + 3;
        (0..6)
            .map(|k| {
                (0..dim)
                    .map(|j| {
                        let s = ((j * 7 + k * 13) % 11) as f64 / 11.0 - 0.5;
                        // Keep the log scales in a range where the density is not
                        // numerically degenerate, and the rest genuinely spread out.
                        if j == 2 + n_groups || j == 3 + n_groups || j == 4 + 2 * n_groups {
                            s
                        } else {
                            2.0 * s + (k as f64) * 0.3
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Differences of the log density between points, against the closed form. A
    /// difference rather than an absolute value because the density is only defined up
    /// to a constant, and pinning the constant would pin the implementation rather
    /// than the mathematics.
    #[test]
    fn the_log_density_matches_its_closed_form() {
        let (y, x, g, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = r#"{"y": "y", "x": "x", "group": "segment",
                      "prior": {"pool_scale": 3.0, "sigma_spread": 1.5,
                                "sigma_log_mean": 0.2, "sigma_log_sd": 2.0,
                                "beta_scale": 4.0, "intercept_scale": 20.0}}"#;
        let model = compile(cfg, &view).unwrap();
        let target = model.as_differentiable().unwrap();
        let prior = Prior {
            intercept_scale: 20.0,
            beta_scale: 4.0,
            pool_scale: 3.0,
            sigma_spread: 1.5,
            sigma_log_mean: 0.2,
            sigma_log_sd: 2.0,
        };

        let pts = points(3);
        for i in 1..pts.len() {
            let got = target.logp(&pts[i]) - target.logp(&pts[0]);
            let want = reference_logp(&y, &x, &g, 3, &pts[i], &prior)
                - reference_logp(&y, &x, &g, 3, &pts[0], &prior);
            assert!(
                (got - want).abs() < 1e-9 * want.abs().max(1.0),
                "point {i}: logp difference {got} vs closed form {want}"
            );
        }
    }

    /// The same, under the **default** priors, which are not flat: the pooling scale
    /// carries a half-Normal at the response's own standard deviation and the spread of
    /// log-sigmas a half-Normal at one log unit. Pinning the defaults here means a
    /// change to either is a test failure rather than a silent change of model.
    #[test]
    fn the_log_density_matches_its_closed_form_under_the_default_priors() {
        let (y, x, g, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();
        let target = model.as_differentiable().unwrap();

        let m = y.iter().sum::<f64>() / y.len() as f64;
        let response_sd =
            (y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (y.len() - 1) as f64).sqrt();
        let prior = Prior {
            intercept_scale: f64::INFINITY,
            beta_scale: f64::INFINITY,
            pool_scale: response_sd,
            sigma_spread: 1.0,
            sigma_log_mean: 0.0,
            sigma_log_sd: f64::INFINITY,
        };

        let pts = points(3);
        for i in 1..pts.len() {
            let got = target.logp(&pts[i]) - target.logp(&pts[0]);
            let want = reference_logp(&y, &x, &g, 3, &pts[i], &prior)
                - reference_logp(&y, &x, &g, 3, &pts[0], &prior);
            assert!(
                (got - want).abs() < 1e-9 * want.abs().max(1.0),
                "point {i}: logp difference {got} vs closed form {want}"
            );
        }
    }

    /// **The default pooling hyperprior is the response's own scale, and that is a
    /// decision rather than an accident.**
    ///
    /// `docs/THEORY.md` §3 rejects concrete prior defaults because they are claims about
    /// units. A scale taken *from the data* makes no such claim: double every
    /// observation and the prior doubles with it. This test is what that promise is
    /// worth — the same data in different units gives the same density difference, once
    /// the coordinates that live on the response's scale are moved with it.
    #[test]
    fn the_default_pooling_hyperprior_is_the_responses_own_scale() {
        let (y, x, _, _) = fixture();
        let keys = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"];
        let k = 100.0f64;

        let mut diffs = Vec::new();
        for factor in [1.0, k] {
            let frame = Frame::new(12)
                .numeric("y", y.iter().map(|v| v * factor).collect())
                .numeric("x", x.clone())
                .key("segment", keys.to_vec());
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();
            let t = model.as_differentiable().unwrap();
            // The coordinates that live on the response's scale: the two fixed effects,
            // `log tau` and `log sigma_pop`. `z`, `w` and `log tau_s` are already
            // dimensionless.
            let shift = |theta: &[f64]| -> Vec<f64> {
                let mut v = theta.to_vec();
                v[0] *= factor;
                v[1] *= factor;
                v[2 + 3] += factor.ln();
                v[3 + 3] += factor.ln();
                v
            };
            let pts = points(3);
            diffs.push(t.logp(&shift(&pts[1])) - t.logp(&shift(&pts[0])));
        }
        assert!(
            (diffs[0] - diffs[1]).abs() < 1e-9 * diffs[0].abs().max(1.0),
            "the default prior is not scale-free: rescaling the response by {k} moved \
             the log-density difference from {} to {}",
            diffs[0],
            diffs[1]
        );
    }

    /// **The Jacobian of `log tau`, isolated.**
    ///
    /// At `z = 0` the group effects are zero whatever `tau` is, so `tau` has left the
    /// likelihood entirely and only the hyperprior and the log-Jacobian remain. With an
    /// explicit half-Normal scale `T` the whole dependence is arithmetic:
    /// `log p(b) - log p(a) = (b - a) - (e^{2b} - e^{2a}) / (2 T^2)`.
    ///
    /// The first term *is* the Jacobian, and the second assertion pins that dropping it
    /// would be visible here. That matters because it is not visible anywhere else: a
    /// missing Jacobian is an `O(1/G)` perturbation that hides inside any
    /// engine-agreement tolerance, proved by mutation on `conjugate_anomaly`
    /// (`ROADMAP.md` §2). The same test also distinguishes a prior declared on the
    /// natural scale from one declared on the log scale — the latter would be quadratic
    /// in `b` rather than in `e^b`, and would leave the posterior improper at zero.
    #[test]
    fn the_log_jacobian_of_log_tau_is_present() {
        let (_, _, _, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let t = 2.0;
        let model = compile(
            &format!(
                r#"{{"y": "y", "x": "x", "group": "segment", "prior": {{"pool_scale": {t}}}}}"#
            ),
            &view,
        )
        .unwrap();
        let target = model.as_differentiable().unwrap();

        let mut theta = vec![0.0; target.dim()];
        theta[0] = 10.0; // intercept, so the residuals are not absurd
        theta[3 + 3] = 0.5; // log sigma_pop
        for (a, b) in [(-2.0, 1.0), (0.5, 1.5), (-5.0, -4.0)] {
            let mut lo = theta.clone();
            lo[2 + 3] = a;
            let mut hi = theta.clone();
            hi[2 + 3] = b;
            let moved = target.logp(&hi) - target.logp(&lo);
            let prior_only = -((2.0 * b).exp() - (2.0 * a).exp()) / (2.0 * t * t);
            let want = (b - a) + prior_only;
            assert!(
                (moved - want).abs() < 1e-9,
                "log tau {a} -> {b} moved by {moved}, expected {want}"
            );
            assert!(
                (moved - prior_only).abs() > 1e-6,
                "the Jacobian term is indistinguishable from zero over {a} -> {b}, \
                 so this test could not detect it going missing"
            );
        }
    }

    /// The same isolation for `log tau_s`: at `w = 0` the per-group scales do not depend
    /// on it, so only its hyperprior and its Jacobian are left.
    #[test]
    fn the_log_jacobian_of_log_sigma_spread_is_present() {
        let (_, _, _, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let sc = 1.5;
        let model = compile(
            &format!(
                r#"{{"y": "y", "x": "x", "group": "segment", "prior": {{"sigma_spread": {sc}}}}}"#
            ),
            &view,
        )
        .unwrap();
        let target = model.as_differentiable().unwrap();

        let mut theta = vec![0.0; target.dim()];
        theta[0] = 10.0;
        theta[3 + 3] = 0.5;
        let lts = target.dim() - 1;
        for (a, b) in [(-3.0, 0.5), (1.0, 2.0)] {
            let mut lo = theta.clone();
            lo[lts] = a;
            let mut hi = theta.clone();
            hi[lts] = b;
            let moved = target.logp(&hi) - target.logp(&lo);
            let prior_only = -((2.0 * b).exp() - (2.0 * a).exp()) / (2.0 * sc * sc);
            let want = (b - a) + prior_only;
            assert!(
                (moved - want).abs() < 1e-9,
                "log tau_s {a} -> {b} moved by {moved}, expected {want}"
            );
            assert!(
                (moved - prior_only).abs() > 1e-6,
                "the Jacobian term is indistinguishable from zero over {a} -> {b}"
            );
        }
    }

    //=== The gradient ==========================================================//

    /// **Away from the mode, which is the only place the check has content.**
    ///
    /// At a mode every component of the analytic gradient is zero and so is every
    /// finite difference, so the comparison passes for any implementation whatsoever.
    /// The points below are drawn well away from the family's own starting point and
    /// include strongly negative log scales, which is where the transforms' Jacobian
    /// terms carry a large share of the derivative.
    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let frame = panel(11, &[10.0, 12.5, 9.0, 11.0], &[0.6, 1.8, 1.1, 0.4], 12, 0.8);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(
            r#"{"y": "y", "x": "x", "group": "segment",
                "prior": {"pool_scale": 2.5, "sigma_spread": 1.0,
                          "sigma_log_mean": 0.0, "sigma_log_sd": 1.5,
                          "beta_scale": 3.0, "intercept_scale": 30.0}}"#,
            &view,
        )
        .unwrap();
        let target = model.as_differentiable().unwrap();
        let dim = target.dim();

        let start = target.initial();
        let mut grad_at_start = vec![0.0; dim];
        target.grad(&start, &mut grad_at_start).unwrap();

        for k in 0..4 {
            // Deliberately far from `initial()`: a displacement of order one on every
            // coordinate, different in each direction.
            let theta: Vec<f64> = (0..dim)
                .map(|j| start[j] + (((j * 5 + k * 3) % 7) as f64 - 3.0) * 0.35)
                .collect();

            // ...and the check is worth nothing if this point happens to be a mode.
            let mut analytic = vec![0.0; dim];
            target.grad(&theta, &mut analytic).unwrap();
            let size = analytic.iter().map(|g| g.abs()).fold(0.0, f64::max);
            assert!(
                size > 1.0,
                "point {k} is too near a mode: |grad|max = {size}"
            );

            for j in 0..dim {
                // Central differences at a step scaled to the coordinate, which is the
                // usual cube-root-of-epsilon compromise between truncation and
                // cancellation error.
                let h = 1e-5 * theta[j].abs().max(1.0);
                let mut up = theta.clone();
                up[j] += h;
                let mut down = theta.clone();
                down[j] -= h;
                let fd = (target.logp(&up) - target.logp(&down)) / (2.0 * h);
                let tol = 1e-4 * fd.abs().max(analytic[j].abs()).max(1.0);
                assert!(
                    (analytic[j] - fd).abs() < tol,
                    "point {k}, coordinate {j}: analytic {} vs finite difference {fd}",
                    analytic[j]
                );
            }
        }
    }

    //=== Behaviour: what the family is for =====================================//

    fn run(cfg: &str, frame: &Frame) -> crate::fit::Fit {
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        crate::fit::fit(
            "varying_variance_gaussian",
            &Config::parse(cfg).unwrap(),
            &view,
        )
        .unwrap()
    }

    /// Every kept draw of one parameter, across every chain.
    fn col(fit: &crate::fit::Fit, group: &str, name: &str) -> Vec<f64> {
        let j = fit
            .posterior
            .params
            .iter()
            .position(|p| p.group_id == group && p.name == name)
            .unwrap_or_else(|| panic!("no parameter {group}/{name}"));
        (0..fit.posterior.n_chains)
            .flat_map(|c| fit.posterior.chain_values(c, j).collect::<Vec<_>>())
            .collect()
    }

    fn quantile(xs: &[f64], q: f64) -> f64 {
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn sd(xs: &[f64]) -> f64 {
        let m = mean(xs);
        (xs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
    }

    /// **Parameter recovery.** Simulate from known per-group sigmas and a known spread
    /// of group levels, fit, and require the posterior to cover the truth.
    ///
    /// The truth for `pool_scale` is stated carefully: the levels below are *fixed*
    /// constants, not themselves drawn from `N(0, tau^2)`, so the quantity the data can
    /// identify is the realised spread of those eight levels rather than the parameter
    /// of a distribution they were never drawn from. Asserting coverage of the realised
    /// spread is the honest claim; asserting coverage of some other number would be a
    /// test of a model nobody fitted.
    #[test]
    fn a_simulated_panel_recovers_its_per_group_sigmas_and_its_pooling_scale() {
        let levels = [10.0, 13.0, 8.5, 11.5, 9.0, 12.5];
        let sigmas = [0.4, 1.8, 0.9, 0.6, 1.4, 0.7];
        let frame = panel(2026, &levels, &sigmas, 20, 0.8);
        let fit = run(
            r#"{"y": "y", "x": "x", "group": "segment",
                "draws": 1000, "chains": 4, "warmup": 800, "seed": 7}"#,
            &frame,
        );

        // The slope is a nuisance here but is the easiest thing to get wrong, so it is
        // checked first: if the linear part were mis-assembled every variance below
        // would absorb the error and still look plausible.
        let beta = col(&fit, "__global__", "beta[x]");
        assert!(
            quantile(&beta, 0.025) < 0.8 && 0.8 < quantile(&beta, 0.975),
            "beta[x] 95% interval [{}, {}] misses 0.8",
            quantile(&beta, 0.025),
            quantile(&beta, 0.975)
        );

        for (g, &truth) in sigmas.iter().enumerate() {
            let s = col(&fit, &format!("G{g:02}"), "sigma");
            let (lo, hi) = (quantile(&s, 0.025), quantile(&s, 0.975));
            println!("sigma[G{g:02}]: truth {truth}, 95% [{lo:.3}, {hi:.3}]");
            assert!(
                lo < truth && truth < hi,
                "sigma[G{g:02}] 95% interval [{lo}, {hi}] misses its truth {truth}"
            );
        }

        // The realised spread of the eight levels, which is what `pool_scale` is the
        // scale of.
        let gbar = levels.iter().sum::<f64>() / levels.len() as f64;
        let realised = (levels.iter().map(|l| (l - gbar).powi(2)).sum::<f64>()
            / (levels.len() - 1) as f64)
            .sqrt();
        let tau = col(&fit, "__global__", "pool_scale");
        let (lo, hi) = (quantile(&tau, 0.025), quantile(&tau, 0.975));
        println!("pool_scale: realised spread {realised:.3}, 95% [{lo:.3}, {hi:.3}]");
        assert!(
            lo < realised && realised < hi,
            "pool_scale 95% interval [{lo}, {hi}] misses the realised level spread {realised}"
        );

        assert_eq!(
            fit.posterior.meta.status,
            crate::types::FitStatus::Converged,
            "{:?}",
            fit.reasons
        );
        // A hierarchical posterior is where divergences come from, and every one of
        // them is a refusal under the shipped thresholds. Zero is the bar.
        assert_eq!(fit.posterior.n_divergent(), Some(0));
    }

    /// The learned scale is genuinely learned: two panels differing only in how far
    /// apart their group levels sit must produce different `pool_scale` posteriors.
    /// In `pooled_gaussian` this number is a config slot and cannot move at all.
    #[test]
    fn the_pooling_scale_follows_the_data_rather_than_a_setting() {
        let sigmas = [1.0; 6];
        let tight = panel(31, &[10.0, 10.2, 9.8, 10.1, 9.9, 10.3], &sigmas, 15, 0.0);
        let wide = panel(31, &[6.0, 14.0, 8.0, 12.0, 5.0, 15.0], &sigmas, 15, 0.0);
        let cfg = r#"{"y": "y", "x": "x", "group": "segment",
                      "draws": 300, "chains": 2, "warmup": 400, "seed": 11}"#;

        let a = mean(&col(&run(cfg, &tight), "__global__", "pool_scale"));
        let b = mean(&col(&run(cfg, &wide), "__global__", "pool_scale"));
        println!("pool_scale: tight panel {a:.3}, wide panel {b:.3}");
        assert!(
            b > 3.0 * a,
            "a panel whose levels are far apart should learn a much larger pooling \
             scale: tight {a}, wide {b}"
        );
    }

    /// **The question agent 04 actually asks, and the reason this family exists.**
    ///
    /// Two segments with the *same* mean and genuinely different spread. A decision
    /// about a tail — how late might this segment pay, how much stock covers 95 % of
    /// weeks — reads the predictive interval, and the two must differ.
    ///
    /// `pooled_gaussian` is fitted to the same data as the control. It has one residual
    /// scale for the whole design, so its two segments get predictive intervals of the
    /// same width whatever the data says: the failure is structural, not a matter of
    /// tuning, which is why the fix had to be a new family.
    #[test]
    fn a_noisier_segment_gets_a_wider_predictive_interval_than_a_quiet_one() {
        // Same level, four-fold difference in spread. The other segments exist so the
        // hyperprior has something to learn the spread of spreads from.
        let levels = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let sigmas = [0.5, 2.0, 0.8, 1.2, 0.6, 1.5];
        let frame = panel(77, &levels, &sigmas, 30, 0.0);
        let fit = run(
            r#"{"y": "y", "x": "x", "group": "segment",
                "draws": 500, "chains": 2, "warmup": 500, "seed": 13}"#,
            &frame,
        );

        // The posterior predictive for one more observation in group g:
        //   y* = intercept + group_effect_g + sigma_g * N(0, 1)
        // drawn once per posterior draw, which is what propagates the parameter
        // uncertainty rather than conditioning on a point estimate.
        let width = |g: usize| {
            let key = format!("G{g:02}");
            let a = col(&fit, "__global__", "intercept");
            let e = col(&fit, &key, "group_effect");
            let s = col(&fit, &key, "sigma");
            let mut rng = BayesRng::for_chain(999, g as u32);
            let pred: Vec<f64> = (0..a.len())
                .map(|d| a[d] + e[d] + s[d] * rng.standard_normal())
                .collect();
            quantile(&pred, 0.95) - quantile(&pred, 0.05)
        };

        let (quiet, noisy) = (width(0), width(1));
        println!("predictive 90% width: quiet {quiet:.3}, noisy {noisy:.3}");
        // The truth is a factor of four. Anything above two is unambiguously a
        // different interval rather than sampling noise.
        assert!(
            noisy > 2.0 * quiet,
            "the noisier segment's predictive interval ({noisy}) should be far wider \
             than the quiet one's ({quiet})"
        );

        // ...and the control: `pooled_gaussian`, which cannot say this.
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let pooled = crate::fit::fit(
            "pooled_gaussian",
            &Config::parse(
                r#"{"y": "y", "x": "x", "group": "segment", "pool_scale": 5.0,
                    "draws": 2000, "seed": 13}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();

        let pooled_width = |g: usize| {
            let key = format!("G{g:02}");
            let j = |grp: &str, name: &str| {
                pooled
                    .posterior
                    .params
                    .iter()
                    .position(|p| p.group_id == grp && p.name == name)
                    .unwrap()
            };
            let (ji, je, js) = (
                j("__global__", "intercept"),
                j(&key, "group_effect"),
                j("__global__", "sigma"),
            );
            let mut rng = BayesRng::for_chain(999, g as u32);
            let pred: Vec<f64> = (0..pooled.posterior.n_draws)
                .map(|d| {
                    pooled.posterior.value(0, d, ji)
                        + pooled.posterior.value(0, d, je)
                        + pooled.posterior.value(0, d, js) * rng.standard_normal()
                })
                .collect();
            quantile(&pred, 0.95) - quantile(&pred, 0.05)
        };
        let (pq, pn) = (pooled_width(0), pooled_width(1));
        println!("pooled_gaussian predictive 90% width: quiet {pq:.3}, noisy {pn:.3}");
        assert!(
            (pn / pq - 1.0).abs() < 0.15,
            "pooled_gaussian gave the two segments materially different widths \
             ({pq} vs {pn}); it has one sigma, so this test's premise is wrong"
        );
    }

    /// **The assertion on a function of several parameters at once.**
    ///
    /// SBC ranks one parameter at a time, so it tests *marginals*, and a marginal is
    /// exactly what a wrong correlation preserves: `ROADMAP.md` §3.1 records a diagonal
    /// posterior precision leaving all six SBC suites green while a predictive spread
    /// was wrong by 25x. So every family owes one check on a joint quantity.
    ///
    /// The joint quantity here is a group's **level**, `intercept + group_effect[g]`,
    /// which is what any statement about that group is actually made of. Its posterior
    /// has an external reference: for a group with `n_g` observations of its own and
    /// weak pooling, the level is data-dominated and its posterior sd is
    /// `sigma_g / sqrt(n_g)`. The two parts are strongly anti-correlated — the
    /// unpenalised intercept and the group effects trade off along a ridge — so adding
    /// their marginal variances instead gives a number several times too large. Both
    /// halves are asserted: the joint answer must match the reference, *and* it must be
    /// far from the independence answer, or the test would pass without the
    /// correlation being right.
    #[test]
    fn the_posterior_of_a_groups_level_is_a_joint_quantity_not_a_sum_of_marginals() {
        // Levels far apart, so the learned `tau` is large and the pooling is weak,
        // which is what makes `sigma_g / sqrt(n_g)` the right reference.
        let levels = [4.0, 16.0, 8.0, 20.0, 12.0, 24.0];
        let sigmas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let n_per_group = 40;
        let frame = panel(5150, &levels, &sigmas, n_per_group, 0.0);
        let fit = run(
            r#"{"y": "y", "x": "x", "group": "segment",
                "draws": 500, "chains": 2, "warmup": 500, "seed": 19}"#,
            &frame,
        );

        let a = col(&fit, "__global__", "intercept");
        let e = col(&fit, "G02", "group_effect");
        let s = col(&fit, "G02", "sigma");

        let level: Vec<f64> = (0..a.len()).map(|d| a[d] + e[d]).collect();
        let joint = sd(&level);
        let independent = (sd(&a).powi(2) + sd(&e).powi(2)).sqrt();
        let reference = mean(&s) / (n_per_group as f64).sqrt();
        println!(
            "level sd: joint {joint:.4}, marginals-added {independent:.4}, \
             reference sigma/sqrt(n) {reference:.4}"
        );

        assert!(
            (joint - reference).abs() < 0.25 * reference,
            "the joint posterior sd of the group's level ({joint}) should be near \
             sigma/sqrt(n) = {reference}"
        );
        // The correlation is not a rounding detail: dropping it inflates the answer
        // by a factor a decision would notice.
        assert!(
            independent > 2.0 * joint,
            "adding the marginal variances gave {independent} against the joint {joint}; \
             if these agree the parameters are uncorrelated and this test checks nothing"
        );
    }

    //=== Engines and refusals ==================================================//

    /// There is no closed form here, so the exact engine must decline rather than
    /// substitute something. An agent that asked for an exact posterior and silently
    /// received an approximation would report unearned confidence.
    #[test]
    fn the_exact_engine_declines_this_family_rather_than_approximating_it() {
        let frame = panel(21, &[1.0, 2.0, 3.0, 4.0], &[1.0, 0.5, 2.0, 1.5], 10, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();
        assert!(model.as_exact().is_none());
        assert!(!crate::engines::ExactEngine.supports(&*model));
        // ...while both gradient-based engines can serve it.
        assert!(crate::engines::NutsEngine.supports(&*model));
        assert!(crate::engines::LaplaceEngine.supports(&*model));
    }

    /// **Why the family raises the sampler's step-size target.**
    ///
    /// The default 0.8 is right for a GLM-shaped posterior and produces divergences
    /// here: measured on the eight-group fixture under a flat pooling hyperprior, 34 in
    /// 8 000 draws, concentrated where `pool_scale` is in its upper tail. Every one of
    /// them is a refusal, so a family that left the default in place would report
    /// `degenerate` on clean simulated data.
    ///
    /// The value is declared by the family and cannot be reached from SQL, which is the
    /// same rule as for the non-centring: a caller cannot select a bad parameterisation.
    #[test]
    fn the_family_asks_the_sampler_for_a_finer_step_than_a_glm_shaped_posterior_needs() {
        let frame = panel(22, &[1.0, 2.0, 3.0], &[1.0, 0.5, 2.0], 10, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();
        assert_eq!(model.as_differentiable().unwrap().target_accept(), 0.95);

        // ...and the conjugate families are untouched by the seam that carries it.
        let f = Frame::new(20)
            .numeric("y", (0..20).map(|i| (i % 5) as f64).collect())
            .numeric("x", (0..20).map(|i| i as f64).collect());
        let r = f.key_refs();
        let v = f.view(&r);
        let pooled = crate::catalog::f3_pooled_gaussian::PooledGaussian
            .compile(&Config::parse(r#"{"y": "y", "x": "x"}"#).unwrap(), &v)
            .unwrap();
        assert_eq!(pooled.as_differentiable().unwrap().target_accept(), 0.8);
    }

    /// A prior-predictive check needs a distribution to draw from in closed form, and
    /// this family has none. Refusing at compile time with a sentence that says why is
    /// better than a fit that quietly returns the posterior under a row claiming it is
    /// the prior.
    #[test]
    fn a_prior_predictive_check_is_refused_because_there_is_no_closed_form_prior_draw() {
        let frame = panel(23, &[1.0, 2.0, 3.0], &[1.0, 0.5, 2.0], 10, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"y": "y", "x": "x", "group": "segment", "sample_from": "prior"}"#,
            &view,
        )
        .unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "sample_from"),
            "{err}"
        );
    }

    /// A misspelled prior slot must name itself, with its full path. `pool_scale` is
    /// the one most likely to be written at the top level, because that is where
    /// `pooled_gaussian` has it -- and there it is a setting, while here it is the scale
    /// of a hyperprior.
    #[test]
    fn a_pooling_scale_written_where_pooled_gaussian_puts_it_is_rejected() {
        let frame = panel(24, &[1.0, 2.0, 3.0], &[1.0, 0.5, 2.0], 10, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"y": "y", "x": "x", "group": "segment", "pool_scale": 2.0}"#,
            &view,
        )
        .unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "pool_scale"),
            "{err}"
        );
        assert!(err.to_string().contains("unknown option"), "{err}");
    }

    /// A panel whose groups are all singletons cannot say how variances differ, because
    /// no group has a variance of its own to differ with.
    #[test]
    fn a_panel_of_singleton_groups_cannot_identify_the_spread_of_variances() {
        let frame = panel(25, &[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0; 5], 1, 0.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        // n = 5 rows against 5 group effects plus an intercept: refused before the
        // spread question is even reached.
        assert!(matches!(
            compile(r#"{"y": "y", "group": "segment"}"#, &view).unwrap_err(),
            BayesError::InsufficientData { .. }
        ));
    }

    /// **A degenerate dataset must reach SQL as a `degenerate` row, not as an error.**
    ///
    /// Found by running the demo suite against the shipped build: on a constant
    /// response `assess` correctly returns `Readiness::degenerate`, but the
    /// verdict only decides the `__status__` row — the engine still ran, NUTS
    /// could not find a usable starting point on a likelihood with no interior
    /// maximum, and the whole call failed with
    /// `internal error: NUTS could not find a usable starting point for chain 0`.
    ///
    /// That is the one thing the refusal contract exists to prevent. An agent
    /// branching on `__status__` never saw the refusal, because there was no
    /// table to branch on. `hier_negbin` had solved this before this family was
    /// written; the fix is its `refuses()` short-circuit, and this test is what
    /// keeps it.
    #[test]
    fn a_constant_response_is_refused_through_the_status_row_rather_than_an_error() {
        let frame = Frame::new(12)
            .numeric("y", vec![30.0; 12])
            .numeric("x", (0..12).map(|i| (i % 4) as f64).collect())
            .key(
                "segment",
                vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            );
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        // The whole fit completes -- this is the assertion, and it is what used
        // to raise.
        let fit = crate::fit::fit(
            "varying_variance_gaussian",
            &Config::parse(
                r#"{"y": "y", "x": "x", "group": "segment",
                    "draws": 200, "chains": 2, "warmup": 200, "seed": 1}"#,
            )
            .unwrap(),
            &view,
        )
        .expect("a degenerate dataset must return a refused fit, not an error");

        assert_eq!(
            fit.posterior.meta.status,
            crate::types::FitStatus::Degenerate
        );
        // ...and every parameter it could not estimate is NULL rather than a
        // plausible-looking number.
        for c in 0..fit.posterior.n_chains {
            for j in 0..fit.posterior.params.len() {
                for v in fit.posterior.chain_values(c, j) {
                    assert!(
                        v.is_nan(),
                        "parameter {} of a degenerate fit came back as {v}",
                        fit.posterior.params[j].name
                    );
                }
            }
        }
    }

    /// The constrained draw is what reaches SQL, so every positive quantity must be
    /// positive by construction rather than by luck.
    #[test]
    fn the_constrained_draw_is_positive_wherever_the_parameter_is_a_scale() {
        let frame = panel(4, &[5.0, 6.0, 4.0, 5.5], &[0.5, 2.0, 1.0, 0.7], 15, 0.3);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "y", "x": "x", "group": "segment"}"#, &view).unwrap();
        let target = model.as_differentiable().unwrap();

        let mut theta = target.initial();
        // Somewhere well away from the start, including strongly negative log scales.
        for (j, v) in theta.iter_mut().enumerate() {
            *v += ((j % 5) as f64 - 2.0) * 0.9;
        }
        let mut out = vec![0.0; model.param_names().len()];
        target.constrain(&theta, &mut out);

        let p = 2usize; // intercept + slope
        assert!(out[p] > 0.0, "pool_scale {}", out[p]);
        assert!(out[p + 1] > 0.0, "sigma_pop {}", out[p + 1]);
        assert!(out[p + 2] > 0.0, "sigma_spread {}", out[p + 2]);
        for g in 0..4 {
            assert!(
                out[p + 3 + 4 + g] > 0.0,
                "sigma[{g}] {}",
                out[p + 3 + 4 + g]
            );
        }
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
