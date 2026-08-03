//! F4 — hierarchical positive-duration GLM, Gamma or lognormal, with partial pooling.
//!
//! The inference layer under the cash-runway agent. Its question is "when will this
//! invoice actually be paid", asked of a handful of customer segments whose payment
//! behaviour differs and whose thin segments would otherwise be estimated from a
//! dozen invoices each. The CFO's question is not the mean — it is
//! `P(cash >= obligation on date X)`, so the whole right tail of the delay
//! distribution is the decision.
//!
//! ```text
//!   log mu_i = intercept + x_i'beta + tau * z_{g(i)},    z_g ~ N(0, 1)
//!
//!   dist = gamma:      delay_i ~ Gamma(shape, shape / mu_i)         E[delay] = mu_i
//!   dist = lognormal:  log delay_i ~ N(log mu_i - sigma^2 / 2, sigma^2)
//! ```
//!
//! ## Why this is not `pooled_gaussian` on `log(delay)`
//!
//! `ROADMAP.md` §"F4 native" is right that a lognormal delay model *is* a Gaussian
//! model on `log(delay)`, and this family agrees with it: `dist: 'lognormal'` is
//! exactly that model, offered here so the two branches can be compared without
//! rewriting the query. What `pooled_gaussian` cannot do is the **Gamma** branch, and
//! that branch is the reason the family exists.
//!
//! A lognormal and a Gamma with the same mean and the same coefficient of variation
//! disagree about the far right tail by a factor that grows without bound, and the far
//! right tail is the entire subject: a covenant test asks about the 95th percentile of
//! a cash position, not its centre. Which of the two describes a given ledger is an
//! empirical question, and a family that offers only one answers it by assumption.
//!
//! The second difference is smaller but real. On `log(delay)` a Gaussian model's
//! residual scale is constant in log units, so the absolute spread of a segment that
//! pays late is forced to be proportionally the same as one that pays on time. Under
//! the Gamma branch the variance scales as `mu^2 / shape`, which is the same
//! proportionality — but the *skew* does not: a Gamma is skewed on the natural scale
//! and symmetric-in-logs is not, and a chase list ranks on the natural scale.
//!
//! ## What this family deliberately does not do
//!
//! **A residual scale per group.** That is
//! [`varying_variance_gaussian`](super::f8_varying_variance)'s subject, and adding a
//! second answer to it here would give a caller two families to choose between on a
//! question neither of them would then own. `dispersion` is pooled: one `shape` (or
//! one `sigma`) for the whole design, partially pooled *levels* on top of it. A ledger
//! whose segments differ in spread rather than in level is F8's, and the family
//! description says so.
//!
//! **Censoring.** An unpaid open item is a right-censored observation and this family
//! does not model one; it is fitted on cleared items alone.
//! [`censored_aft`](super::f2_censored_aft) is the family for a censored duration, and
//! silently treating "not paid yet" as "paid today" would bias every delay downward by
//! exactly the amount a cash forecast most needs to be right about. The refusal is a
//! validation error on a non-positive or missing delay, not a status.
//!
//! ## Parameterisation, which is baked in
//!
//! **Non-centred, from the start and not as an option**, for the reason
//! `varying_variance_gaussian`'s header sets out at length: `eta_g ~ N(0, tau^2)` puts
//! the group effects and their own scale in Neal's funnel, and `eta_g = tau * z_g`
//! does not. The BRD's premise is that callers cannot select a bad parameterisation,
//! so there is no `centred` slot to get wrong.
//!
//! **`tau` carries a proper default.** `docs/THEORY.md` §3 rejects concrete prior
//! defaults because they are claims about units — but `tau` here is the between-segment
//! spread of a **log** mean, so it is dimensionless: one log unit is a factor of `e`,
//! which means the same thing in euros, days and kilograms. The default half-Normal(1)
//! is the same admissibility argument `varying_variance_gaussian` makes for its
//! `DEFAULT_SIGMA_SPREAD`, and the same measured motivation: under a flat prior on
//! `tau` the posterior is proper (Gelman 2006) but its upper tail is long and the
//! sampler diverges there, and every divergence is a refusal under
//! `Thresholds::max_divergent = 0`.
//!
//! ## Engine
//!
//! NUTS only. A Laplace posterior is a Gaussian at the joint mode and a non-centred
//! hierarchy has no usable one — the same geometry `hier_negbin` refuses on, with the
//! same measured evidence behind it. An explicit `engine: 'laplace'` is a config error
//! rather than a silently worse answer.

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::types::{EngineKind, FamilyCode};

use super::{CompiledModel, LogPosterior, ModelFamily, Readiness};
use statrs::function::gamma::{digamma, ln_gamma};

/// The family singleton registered in the catalog.
#[derive(Debug)]
pub struct PaymentDelay;

const SLOTS: &[&str] = &[
    "y",
    "group",
    "x",
    "dist",
    "min_groups",
    "prior",
    "draws",
    "chains",
    "warmup",
    "max_draw_megabytes",
    "seed",
    "engine",
    "sample_from",
];

/// Log-scale coordinates outside `+/- LOG_BOX` are refused rather than explored.
const LOG_BOX: f64 = 30.0;

/// Largest magnitude of the linear predictor the arithmetic will evaluate.
///
/// Both signs matter here, unlike in `hier_negbin`. A large positive `eta` overflows
/// `exp`; a large negative one makes `y / mu` overflow instead, and the Gamma branch
/// divides by the mean. `e^60` is about `1e26` days, and `e^-60` is a mean delay of
/// `1e-26` days — neither is a ledger, both are a search that has run away.
const ETA_MAX: f64 = 60.0;

/// Which positive distribution the delays are drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dist {
    /// `Gamma(shape, shape / mu)`, mean `mu`, variance `mu^2 / shape`.
    Gamma,
    /// `log delay ~ N(log mu - sigma^2/2, sigma^2)`, mean `mu`.
    ///
    /// The `- sigma^2/2` is what makes `mu` the mean rather than the median, so the
    /// two branches parameterise the same quantity and a caller can switch `dist`
    /// without the coefficients changing meaning.
    Lognormal,
}

impl Dist {
    fn parse(name: &str) -> BayesResult<Self> {
        match name {
            "gamma" => Ok(Dist::Gamma),
            "lognormal" => Ok(Dist::Lognormal),
            other => Err(BayesError::config("dist", format!("unknown: '{other}'"))),
        }
    }

    /// The name the dispersion parameter is reported under.
    ///
    /// Different names because they are different quantities pointing opposite ways:
    /// a large `shape` is a *tight* Gamma, a large `sigma` is a *wide* lognormal.
    /// Reporting both as `dispersion` would invite a query that reads one as the
    /// other.
    fn dispersion_name(&self) -> &'static str {
        match self {
            Dist::Gamma => "shape",
            Dist::Lognormal => "sigma",
        }
    }
}

impl ModelFamily for PaymentDelay {
    fn id(&self) -> &'static str {
        "payment_delay"
    }

    fn code(&self) -> FamilyCode {
        FamilyCode::PaymentDelay
    }

    fn description(&self) -> &'static str {
        "Hierarchical positive-duration GLM -- Gamma or lognormal delays with a \
         partially pooled per-group log-mean, non-centred -- for payment behaviour \
         per customer segment and the cash-cover probability a liquidity decision \
         reads off its right tail."
    }

    fn default_engine(&self) -> EngineKind {
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

/// Prior on the population coefficients, the pooling scale and the dispersion.
#[derive(Debug, Clone)]
pub(crate) struct Prior {
    intercept_mean: f64,
    intercept_sd: f64,
    beta_scale: f64,
    /// Half-Normal scale for `tau`, the between-group sd of the log means.
    ///
    /// Defaults to [`Prior::DEFAULT_TAU_SCALE`] rather than to flatness. See the
    /// module header: `tau` is dimensionless, so a concrete default asserts nothing
    /// about the caller's units, and flatness here is measurably worse than merely
    /// uninformative.
    tau_scale: f64,
    /// Normal `(mean, sd)` on **`log` of the dispersion**, which is the coordinate
    /// actually sampled. Declared on the log scale, so there is no Jacobian to add:
    /// a lognormal prior on `shape` *is* a normal prior on `log shape`.
    dispersion_log: (f64, f64),
}

impl Prior {
    /// Half-Normal scale on `tau`, in log units.
    ///
    /// One log unit is a factor of `e`. A 95 % prior interval of roughly `[1/7, 7]`
    /// on the ratio between the slowest- and fastest-paying segment's mean delay is
    /// loose enough to be wrong about no real ledger and tight enough to exclude the
    /// upper tail the sampler diverges in.
    const DEFAULT_TAU_SCALE: f64 = 1.0;

    fn parse(cfg: &Config) -> BayesResult<Self> {
        cfg.reject_unknown(&["intercept", "beta", "tau", "dispersion"])?;

        let intercept = cfg.nested("intercept")?;
        intercept.reject_unknown(&["mean", "sd"])?;
        let intercept_mean = intercept.f64_or("mean", 0.0)?;
        let intercept_sd = intercept.positive_f64_or("sd", f64::INFINITY)?;

        let beta = cfg.nested("beta")?;
        beta.reject_unknown(&["scale"])?;
        let beta_scale = beta.positive_f64_or("scale", f64::INFINITY)?;

        let tau = cfg.nested("tau")?;
        tau.reject_unknown(&["scale"])?;
        let tau_scale = tau.positive_f64_or("scale", Self::DEFAULT_TAU_SCALE)?;

        let dispersion = cfg.nested("dispersion")?;
        dispersion.reject_unknown(&["log_mean", "log_sd"])?;
        let dispersion_log = (
            dispersion.f64_or("log_mean", 0.0)?,
            dispersion.positive_f64_or("log_sd", f64::INFINITY)?,
        );

        Ok(Prior {
            intercept_mean,
            intercept_sd,
            beta_scale,
            tau_scale,
            dispersion_log,
        })
    }

    /// Every coordinate carries a proper prior, so the joint is samplable and the
    /// family can be certified by SBC.
    pub(crate) fn is_proper(&self) -> bool {
        self.intercept_sd.is_finite()
            && self.beta_scale.is_finite()
            && self.tau_scale.is_finite()
            && self.dispersion_log.1.is_finite()
    }
}

#[derive(Debug)]
pub(crate) struct CompiledPaymentDelay {
    params: Vec<ParamName>,
    /// Delay, one entry per usable row. Strictly positive by validation.
    y: Vec<f64>,
    /// `log(y)`, formed once because both branches need it on every evaluation.
    log_y: Vec<f64>,
    /// Design, `p` columns of `n` values each. Empty when there are no covariates.
    x: Vec<Vec<f64>>,
    /// Dense group index per row.
    group_of: Vec<usize>,
    group_keys: Vec<String>,
    dist: Dist,
    prior: Prior,
    structural: Option<Readiness>,
    fingerprint: String,
    start: Vec<f64>,
}

/// Coordinate layout of the unconstrained vector.
///
/// ```text
///   [0]                  intercept
///   [1 .. 1+p]           beta, in the caller's column order
///   [1+p]                log tau
///   [2+p]                log shape   (gamma) / log sigma (lognormal)
///   [3+p .. 3+p+G]       z, the non-centred group offsets
/// ```
impl CompiledPaymentDelay {
    fn n_beta(&self) -> usize {
        self.x.len()
    }
    fn i_log_tau(&self) -> usize {
        1 + self.n_beta()
    }
    fn i_log_disp(&self) -> usize {
        2 + self.n_beta()
    }
    fn i_z0(&self) -> usize {
        3 + self.n_beta()
    }
    pub(crate) fn n_groups_inner(&self) -> usize {
        self.group_keys.len()
    }
    fn dim_inner(&self) -> usize {
        self.i_z0() + self.n_groups_inner()
    }

    /// Whether the compile-time verdict was that there is no posterior here.
    ///
    /// `InsufficientData` is deliberately not one of these, for the reason
    /// `hier_negbin` gives: that verdict says the data is weak, not that the surface
    /// is unusable.
    fn refuses(&self) -> bool {
        self.structural.as_ref().is_some_and(|r| {
            matches!(
                r.status,
                crate::types::FitStatus::Degenerate | crate::types::FitStatus::Failed
            )
        })
    }

    /// Linear predictor — the **log mean** delay — for row `i` at `theta`.
    fn eta(&self, theta: &[f64], i: usize) -> f64 {
        let tau = theta[self.i_log_tau()].exp();
        let z = theta[self.i_z0() + self.group_of[i]];
        let mut eta = theta[0] + tau * z;
        for (k, col) in self.x.iter().enumerate() {
            eta += theta[1 + k] * col[i];
        }
        eta
    }

    /// The log posterior, and optionally its gradient.
    ///
    /// One function rather than two because every derivative below is a reweighting
    /// of the same per-row quantities the density already computed.
    ///
    /// ```text
    ///   gamma:      log f = k log k - lnG(k) + (k-1) log y - k eta - k y e^-eta
    ///               d/deta   = k (y/mu - 1)
    ///               d/dlog k = k [log k + 1 - digamma(k) + log y - eta - y/mu]
    ///
    ///   lognormal:  r     = log y - eta + s^2/2
    ///               log f = -log s - r^2 / (2 s^2)
    ///               d/deta   = r / s^2
    ///               d/dlog s = -1 - r + r^2 / s^2
    /// ```
    ///
    /// Normalising constants that do not depend on any coordinate are dropped
    /// uniformly, which is what the trait permits and what makes the closed-form tests
    /// compare *differences*.
    pub(crate) fn logp_and_grad(&self, theta: &[f64], mut grad: Option<&mut [f64]>) -> f64 {
        let dim = self.dim_inner();
        if let Some(g) = grad.as_deref_mut() {
            g[..dim].fill(0.0);
        }

        let log_tau = theta[self.i_log_tau()];
        let log_disp = theta[self.i_log_disp()];
        if !log_tau.is_finite()
            || log_tau.abs() > LOG_BOX
            || !log_disp.is_finite()
            || log_disp.abs() > LOG_BOX
        {
            return f64::NEG_INFINITY;
        }
        let tau = log_tau.exp();
        let disp = log_disp.exp();
        if theta[..self.i_z0()].iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }

        let mut total = 0.0;
        // Accumulated derivative with respect to `log disp`, which every row
        // contributes to and which is formed once rather than in a second pass.
        let mut d_log_disp = 0.0;

        // The parts of the Gamma density that do not vary by row, hoisted out of the
        // loop: `n (k log k - lnGamma(k))`.
        if self.dist == Dist::Gamma {
            let n = self.y.len() as f64;
            total += n * (disp * log_disp - ln_gamma(disp));
            if grad.is_some() {
                // d/dk of [k log k - lnGamma(k)] = log k + 1 - digamma(k); chain rule
                // to `log k` multiplies by k.
                d_log_disp += n * disp * (log_disp + 1.0 - digamma(disp));
            }
        } else {
            // `-n log sigma`; d/d(log sigma) = -n.
            let n = self.y.len() as f64;
            total -= n * log_disp;
            if grad.is_some() {
                d_log_disp -= n;
            }
        }

        for i in 0..self.y.len() {
            let eta = self.eta(theta, i);
            if !eta.is_finite() || eta.abs() > ETA_MAX {
                return f64::NEG_INFINITY;
            }
            let d_eta;
            match self.dist {
                Dist::Gamma => {
                    // y / mu, formed as exp(log y - eta) so a small mu cannot overflow
                    // before the subtraction has a chance to cancel.
                    let ratio = (self.log_y[i] - eta).exp();
                    total += (disp - 1.0) * self.log_y[i] - disp * eta - disp * ratio;
                    d_eta = disp * (ratio - 1.0);
                    if grad.is_some() {
                        d_log_disp += disp * (self.log_y[i] - eta - ratio);
                    }
                }
                Dist::Lognormal => {
                    let r = self.log_y[i] - eta + 0.5 * disp * disp;
                    let inv_var = 1.0 / (disp * disp);
                    total -= 0.5 * r * r * inv_var;
                    d_eta = r * inv_var;
                    if grad.is_some() {
                        d_log_disp += -r + r * r * inv_var;
                    }
                }
            }
            if let Some(g) = grad.as_deref_mut() {
                g[0] += d_eta;
                for (k, col) in self.x.iter().enumerate() {
                    g[1 + k] += col[i] * d_eta;
                }
                let j = self.group_of[i];
                let z = theta[self.i_z0() + j];
                g[self.i_z0() + j] += tau * d_eta;
                g[self.i_log_tau()] += tau * z * d_eta;
            }
        }
        if !total.is_finite() {
            return f64::NEG_INFINITY;
        }

        // The non-centred hierarchy: `z ~ N(0, 1)`, and the pooling scale lives in the
        // linear predictor rather than in this density.
        for j in 0..self.n_groups_inner() {
            let z = theta[self.i_z0() + j];
            total -= 0.5 * z * z;
            if let Some(g) = grad.as_deref_mut() {
                g[self.i_z0() + j] -= z;
            }
        }

        // --- Priors. ---
        if self.prior.intercept_sd.is_finite() {
            let s = self.prior.intercept_sd;
            let d = theta[0] - self.prior.intercept_mean;
            total -= 0.5 * d * d / (s * s);
            if let Some(g) = grad.as_deref_mut() {
                g[0] -= d / (s * s);
            }
        }
        if self.prior.beta_scale.is_finite() {
            let s = self.prior.beta_scale;
            for k in 0..self.n_beta() {
                let b = theta[1 + k];
                total -= 0.5 * b * b / (s * s);
                if let Some(g) = grad.as_deref_mut() {
                    g[1 + k] -= b / (s * s);
                }
            }
        }
        // `tau`: the prior is declared on `tau`, the coordinate is `log tau`, so a
        // `+ log tau` Jacobian appears whether or not a scale was given. Omitting it is
        // invisible to every engine-agreement test and shows up only against the closed
        // form -- see `the_log_jacobian_of_log_tau_is_present`.
        total += log_tau;
        if let Some(g) = grad.as_deref_mut() {
            g[self.i_log_tau()] += 1.0;
        }
        if self.prior.tau_scale.is_finite() {
            let s = self.prior.tau_scale;
            total -= 0.5 * tau * tau / (s * s);
            if let Some(g) = grad.as_deref_mut() {
                g[self.i_log_tau()] -= tau * tau / (s * s);
            }
        }
        // The dispersion prior is declared on the log scale, which *is* the sampled
        // coordinate, so there is no Jacobian here. That asymmetry with `tau` is
        // deliberate and is the same one `hier_negbin` draws for `phi`.
        if self.prior.dispersion_log.1.is_finite() {
            let (m, s) = self.prior.dispersion_log;
            let d = log_disp - m;
            total -= 0.5 * d * d / (s * s);
            if grad.is_some() {
                d_log_disp -= d / (s * s);
            }
        }
        if let Some(g) = grad {
            g[self.i_log_disp()] += d_log_disp;
        }

        total
    }
}

impl CompiledModel for CompiledPaymentDelay {
    fn param_names(&self) -> &[ParamName] {
        &self.params
    }

    fn n_obs(&self) -> usize {
        self.y.len()
    }

    fn n_groups(&self) -> usize {
        self.group_keys.len()
    }

    fn data_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn readiness(&self) -> Readiness {
        self.structural.clone().unwrap_or_else(Readiness::ready)
    }

    fn as_differentiable(&self) -> Option<&dyn LogPosterior> {
        Some(self)
    }
}

impl LogPosterior for CompiledPaymentDelay {
    fn dim(&self) -> usize {
        self.dim_inner()
    }

    fn logp(&self, theta: &[f64]) -> f64 {
        if self.refuses() {
            // The surface a refusing model exposes: a standard normal, trivially
            // explorable, whose draws `constrain` turns into NaN. Same construction and
            // same reason as `hier_negbin`.
            return -0.5 * theta.iter().map(|v| v * v).sum::<f64>();
        }
        self.logp_and_grad(theta, None)
    }

    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        if theta.len() != self.dim_inner() || out.len() != self.dim_inner() {
            return Err(BayesError::DimensionMismatch(format!(
                "expected {} coordinates, got theta {} and out {}",
                self.dim_inner(),
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
        self.logp_and_grad(theta, Some(out));
        Ok(())
    }

    fn initial(&self) -> Vec<f64> {
        if self.refuses() {
            return vec![0.0; self.dim_inner()];
        }
        self.start.clone()
    }

    /// A finer step than the engine default of 0.8, for the reason `hier_negbin` and
    /// `varying_variance_gaussian` both give: a hierarchical posterior's curvature
    /// varies sharply along `tau`, a step tuned to the bulk overshoots in the tail, and
    /// every overshoot is reported as a divergence. This family inherits
    /// `Thresholds::max_divergent = 0`, so one divergence grades the fit `degenerate`
    /// — the two together would make a correct posterior get refused.
    ///
    /// It costs leapfrog steps, not correctness: the target distribution is untouched.
    fn target_accept(&self) -> f64 {
        0.95
    }

    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        if self.refuses() {
            out.fill(f64::NAN);
            return;
        }
        let tau = theta[self.i_log_tau()].exp();
        let mut at = 0;
        out[at] = theta[0];
        at += 1;
        for k in 0..self.n_beta() {
            out[at] = theta[1 + k];
            at += 1;
        }
        out[at] = tau;
        at += 1;
        out[at] = theta[self.i_log_disp()].exp();
        at += 1;
        for j in 0..self.n_groups_inner() {
            let u = tau * theta[self.i_z0() + j];
            out[at] = u;
            // The group's own mean delay with every covariate at zero, which is the
            // number a segment-level statement is made of and saves every caller the
            // same exponentiation.
            out[at + 1] = (theta[0] + u).exp();
            at += 2;
        }
    }
}

pub(crate) fn build(cfg: &Config, data: &DataView) -> BayesResult<CompiledPaymentDelay> {
    cfg.reject_unknown(SLOTS)?;

    let y_name = cfg.require_str("y")?.to_string();
    let group_name = cfg.require_str("group")?.to_string();
    let x_names = cfg.str_list("x")?;
    let dist = Dist::parse(&cfg.one_of("dist", &["gamma", "lognormal"], "gamma")?)?;
    let min_groups = cfg.usize_in("min_groups", 3, 2, 1_000_000_000)?;

    // The one engine question a family is allowed to answer, because it is a statement
    // about *this* geometry and nothing else knows it. The argument is `hier_negbin`'s
    // verbatim, because the geometry is: under the non-centred parameterisation the
    // likelihood does not depend on `tau` at all when every `z` is zero, so the density
    // has a ridge along `{z = 0, tau -> infinity}` that the `+ log tau` Jacobian makes
    // rise without bound. The ridge carries no posterior mass, which is why a sampler is
    // untroubled by it and a mode search walks straight up it.
    if let Some("laplace") = cfg
        .opt_str("engine")?
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        return Err(BayesError::config(
            "engine",
            "'payment_delay' is served by NUTS only. A Laplace posterior is a Gaussian \
             at the joint mode, and a non-centred hierarchy has no usable one: the \
             likelihood does not depend on the pooling scale when every group offset is \
             zero, so the mode search climbs a ridge that carries no posterior mass. \
             Drop the 'engine' slot to use the default",
        ));
    }
    if cfg.opt_str("sample_from")? == Some("prior") {
        return Err(BayesError::config(
            "sample_from",
            "this family has no closed-form prior draw; a prior-predictive check needs \
             the exact engine, which no non-conjugate family offers",
        ));
    }
    let prior = Prior::parse(&cfg.nested("prior")?)?;

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

    let groups = data.group_rows(Some(&group_name), &rows)?;
    for (key, _) in &groups {
        crate::types::validate_group_key(key)?;
    }
    let group_keys: Vec<String> = groups.iter().map(|(k, _)| k.clone()).collect();

    // Row order follows the groups, so a group's observations are contiguous.
    let mut y = Vec::with_capacity(rows.len());
    let mut log_y = Vec::with_capacity(rows.len());
    let mut group_of = Vec::with_capacity(rows.len());
    let mut x: Vec<Vec<f64>> = vec![Vec::with_capacity(rows.len()); x_cols.len()];
    for (j, (_, members)) in groups.iter().enumerate() {
        for &i in members {
            let v = y_col.values[i];
            if !v.is_finite() || v <= 0.0 {
                return Err(BayesError::config(
                    "y",
                    format!(
                        "must be a strictly positive duration; row {i} is {v}. A delay \
                         measured from the *due* date is negative whenever an invoice is \
                         paid early, and this family has no support there: measure the \
                         delay from the invoice date, or model log(delay) with \
                         'pooled_gaussian' where a negative value is admissible. An \
                         open item that has not been paid at all is right-censored \
                         rather than zero, and belongs to 'censored_aft'"
                    ),
                ));
            }
            y.push(v);
            log_y.push(v.ln());
            group_of.push(j);
            for (k, col) in x_cols.iter().enumerate() {
                x[k].push(col.values[i]);
            }
        }
    }

    let n = y.len();
    let n_groups = group_keys.len();
    let p = x_cols.len();
    // intercept + beta + log tau + log dispersion.
    let n_fixed = 3 + p;
    if n <= n_fixed {
        return Err(BayesError::InsufficientData {
            rows: n,
            params: n_fixed,
        });
    }

    let mut params: Vec<ParamName> = Vec::with_capacity(n_fixed + 2 * n_groups);
    params.push(ParamName::global("intercept")?);
    for name in &x_names {
        params.push(ParamName::global(format!("beta[{name}]"))?);
    }
    params.push(ParamName::global("tau")?);
    params.push(ParamName::global(dist.dispersion_name())?);
    for key in &group_keys {
        params.push(ParamName::grouped(key.clone(), "u")?);
        params.push(ParamName::grouped(key.clone(), "mu")?);
    }

    // --- The verdicts reachable from the sufficient statistics alone. ---
    let first = y[0];
    let structural = if n_groups < min_groups {
        Some(Readiness::insufficient(format!(
            "{n_groups} group(s) is below the min_groups threshold of {min_groups}: a \
             pooling scale estimated from this few describes the sample rather than the \
             population, and every per-segment interval inherits that"
        )))
    } else if y.iter().all(|v| *v == first) {
        Some(Readiness::degenerate(format!(
            "every one of the {n} delays is exactly {first}, so the dispersion is \
             identified only in the limit of a point mass and there is no interior \
             maximum to put a posterior around. A ledger whose invoices all clear on \
             the same day needs no model{}",
            if prior.is_proper() {
                ""
            } else {
                "; set a proper `prior.dispersion` if a prior-only answer is wanted"
            }
        )))
    } else {
        None
    };

    let start = starting_point(&log_y, &group_of, n_groups, p, dist);

    Ok(CompiledPaymentDelay {
        params,
        y,
        log_y,
        x,
        group_of,
        group_keys,
        dist,
        prior,
        structural,
        fingerprint,
        start,
    })
}

/// A starting point already scaled to the data.
///
/// The coordinate that matters is the intercept: it is a log mean delay in the
/// caller's units, so starting it at zero when invoices clear in forty days costs the
/// sampler its whole warmup climbing out of a numerically flat region. `z` starts from
/// each group's own observed deviation divided by the starting `tau`, which is that
/// group's sample level expressed in the non-centred coordinate.
fn starting_point(
    log_y: &[f64],
    group_of: &[usize],
    n_groups: usize,
    p: usize,
    dist: Dist,
) -> Vec<f64> {
    let n = log_y.len();
    let mut sum = vec![0.0; n_groups];
    let mut count: Vec<f64> = vec![0.0; n_groups];
    for i in 0..n {
        sum[group_of[i]] += log_y[i];
        count[group_of[i]] += 1.0;
    }
    let group_mean: Vec<f64> = (0..n_groups).map(|j| sum[j] / count[j].max(1.0)).collect();
    let b0 = group_mean.iter().sum::<f64>() / n_groups as f64;
    let between = group_mean.iter().map(|m| (m - b0).powi(2)).sum::<f64>() / n_groups.max(2) as f64;
    // The observed spread of group means is between-group variance *plus* sampling
    // noise, so it overstates tau. Half of it is a start, not an estimate.
    let tau = (0.5 * between).sqrt().clamp(0.02, 5.0);

    // Within-group variance of the log delays, which both branches read: for a Gamma
    // with shape k the variance of `log y` is `trigamma(k) ~ 1/k`, and for a lognormal
    // it is `sigma^2` exactly.
    let within = (0..n)
        .map(|i| (log_y[i] - group_mean[group_of[i]]).powi(2))
        .sum::<f64>()
        / (n.saturating_sub(n_groups)).max(1) as f64;
    let within = within.max(1e-6);
    let log_disp = match dist {
        Dist::Gamma => (1.0 / within).clamp(1e-2, 1e4).ln(),
        Dist::Lognormal => within.sqrt().clamp(1e-3, 1e3).ln(),
    };

    let mut theta = vec![0.0; 3 + p + n_groups];
    theta[0] = b0.clamp(-ETA_MAX / 2.0, ETA_MAX / 2.0);
    theta[1 + p] = tau.ln();
    theta[2 + p] = log_disp;
    for j in 0..n_groups {
        theta[3 + p + j] = ((group_mean[j] - b0) / tau).clamp(-5.0, 5.0);
    }
    theta
}

/// The real surface, exposed regardless of the compile-time verdict.
///
/// Exists for one test, and that test is the most valuable one in the module: without
/// it the finite-difference check would pass on any dataset the model refused, which is
/// exactly the dataset a wrong gradient produces.
#[cfg(test)]
pub(crate) struct TrueSurface<'a>(pub(crate) &'a CompiledPaymentDelay);

#[cfg(test)]
impl LogPosterior for TrueSurface<'_> {
    fn dim(&self) -> usize {
        self.0.dim_inner()
    }
    fn logp(&self, theta: &[f64]) -> f64 {
        self.0.logp_and_grad(theta, None)
    }
    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        self.0.logp_and_grad(theta, Some(out));
        Ok(())
    }
    fn initial(&self) -> Vec<f64> {
        self.0.start.clone()
    }
    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        self.0.constrain(theta, out);
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! The generative process, run forwards.
    //!
    //! Shared with `sbc.rs` rather than duplicated, because SBC's whole warranty is
    //! that the simulator and the likelihood describe the same model. Two copies would
    //! eventually describe two models, and the suite would certify the wrong one.

    use super::Dist;
    use crate::data::testing::Frame;
    use crate::errors::BayesResult;
    use crate::rng::BayesRng;

    /// A simulated ledger in the columns the family reads.
    pub(crate) struct Ledger {
        pub delay: Vec<f64>,
        pub segment: Vec<String>,
        /// The mean delay each segment was actually generated from.
        pub true_mu: Vec<f64>,
    }

    impl Ledger {
        pub(crate) fn frame(&self) -> Frame {
            Frame::new(self.delay.len())
                .numeric("delay_days", self.delay.clone())
                .key("segment", self.segment.iter().map(String::as_str).collect())
        }
    }

    /// One draw from the family's own likelihood, at mean `mu`.
    ///
    /// The Gamma branch is written in the crate's rate parameterisation, so
    /// `Gamma(k, k/mu)` has mean `mu`; the lognormal branch carries the `- s^2/2`
    /// correction that makes `mu` the mean rather than the median. Getting either
    /// convention wrong would make the simulator describe a different model from the
    /// one the likelihood inverts, and SBC would then certify the wrong thing.
    pub(crate) fn draw_delay(
        rng: &mut BayesRng,
        mu: f64,
        dist: Dist,
        disp: f64,
    ) -> BayesResult<f64> {
        match dist {
            Dist::Gamma => rng.gamma(disp, disp / mu),
            Dist::Lognormal => {
                Ok((mu.ln() - 0.5 * disp * disp + disp * rng.standard_normal()).exp())
            }
        }
    }

    /// Simulate `n_groups` segments of `n_per` cleared invoices each, from the family's
    /// own model.
    pub(crate) fn simulate(
        rng: &mut BayesRng,
        n_groups: usize,
        n_per: usize,
        intercept: f64,
        tau: f64,
        dist: Dist,
        disp: f64,
    ) -> BayesResult<Ledger> {
        let (mut delay, mut segment, mut true_mu) = (vec![], vec![], vec![]);
        for j in 0..n_groups {
            let mu = (intercept + tau * rng.standard_normal()).exp();
            true_mu.push(mu);
            for _ in 0..n_per {
                delay.push(draw_delay(rng, mu, dist, disp)?);
                segment.push(format!("SEG-{j:03}"));
            }
        }
        Ok(Ledger {
            delay,
            segment,
            true_mu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::testing::Frame;
    use crate::engines::Engine;
    use crate::rng::BayesRng;

    fn compile<'a>(cfg: &str, data: &'a DataView<'a>) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        PaymentDelay.compile(&Config::parse(cfg).unwrap(), data)
    }

    /// A ledger with known per-segment mean delays, generated from the family's own
    /// model so that recovery is a statement about the fit rather than about the
    /// fixture.
    fn ledger(seed: u64, n_groups: usize, n_per: usize, dist: Dist, disp: f64) -> Frame {
        let mut rng = BayesRng::for_chain(seed, 0);
        testing::simulate(&mut rng, n_groups, n_per, 3.4, 0.35, dist, disp)
            .unwrap()
            .frame()
    }

    #[test]
    fn the_family_is_in_the_catalog_under_its_own_code() {
        let family = crate::catalog::lookup("payment_delay").unwrap();
        assert_eq!(family.code(), FamilyCode::PaymentDelay);
        assert_eq!(family.default_engine(), EngineKind::Nuts);
        assert_eq!(family.code() as i32, 4);
    }

    /// Parameter identities and their order, which `constrain` writes into and which
    /// every downstream consumer joins on.
    #[test]
    fn the_parameters_are_the_coefficients_two_scales_and_two_per_group() {
        let frame = ledger(1, 3, 12, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap();

        let names: Vec<String> = model
            .param_names()
            .iter()
            .map(|p| format!("{}/{}", p.group_id, p.name))
            .collect();
        assert_eq!(
            names,
            vec![
                "__global__/intercept",
                "__global__/tau",
                "__global__/shape",
                "SEG-000/u",
                "SEG-000/mu",
                "SEG-001/u",
                "SEG-001/mu",
                "SEG-002/u",
                "SEG-002/mu",
            ]
        );
        // intercept + log tau + log shape + one z per group.
        assert_eq!(model.as_differentiable().unwrap().dim(), 3 + 3);
    }

    /// The lognormal branch reports `sigma`, not `shape`. They point opposite ways --
    /// a large shape is a tight Gamma, a large sigma a wide lognormal -- so a shared
    /// name would invite a query that reads one as the other.
    #[test]
    fn the_dispersion_is_named_after_the_distribution_it_belongs_to() {
        let frame = ledger(2, 3, 12, Dist::Lognormal, 0.5);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(
            r#"{"y": "delay_days", "group": "segment", "dist": "lognormal"}"#,
            &view,
        )
        .unwrap();
        assert!(model
            .param_names()
            .iter()
            .any(|p| p.name == "sigma" && p.group_id == crate::types::GLOBAL_GROUP));
        assert!(model.param_names().iter().all(|p| p.name != "shape"));
    }

    //=== The log density, checked against its closed form directly ==============//

    /// A small fixed dataset, written out rather than generated, so the reference below
    /// is arithmetic a reader can check.
    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<usize>, Frame) {
        let y = vec![
            32.0, 41.0, 28.0, 55.0, 61.0, 47.0, 72.0, 58.0, 18.0, 22.0, 15.0, 26.0,
        ];
        let x = vec![
            -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0,
        ];
        let g = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let keys = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"];
        let frame = Frame::new(12)
            .numeric("delay_days", y.clone())
            .numeric("amount_band", x.clone())
            .key("segment", keys.to_vec());
        (y, x, g, frame)
    }

    /// The model's log density, written out from the specification in the module header
    /// with no reference to the implementation. Up to an additive constant.
    fn reference_logp(
        y: &[f64],
        x: &[f64],
        g: &[usize],
        n_groups: usize,
        theta: &[f64],
        dist: Dist,
        prior: &Prior,
    ) -> f64 {
        let (a, b) = (theta[0], theta[1]);
        let (log_tau, log_disp) = (theta[2], theta[3]);
        let z = &theta[4..4 + n_groups];
        let (tau, disp) = (log_tau.exp(), log_disp.exp());

        let mut acc = 0.0;
        for i in 0..y.len() {
            let eta = a + b * x[i] + tau * z[g[i]];
            let mu = eta.exp();
            match dist {
                Dist::Gamma => {
                    acc += disp * disp.ln() - ln_gamma(disp) + (disp - 1.0) * y[i].ln()
                        - disp * eta
                        - disp * y[i] / mu;
                }
                Dist::Lognormal => {
                    let r = y[i].ln() - eta + 0.5 * disp * disp;
                    acc += -log_disp - 0.5 * r * r / (disp * disp);
                }
            }
        }
        for zj in z.iter().take(n_groups) {
            acc -= 0.5 * zj * zj;
        }
        // Half-Normal on the natural scale, plus the log-Jacobian of `log tau`.
        if prior.tau_scale.is_finite() {
            acc -= 0.5 * tau * tau / (prior.tau_scale * prior.tau_scale);
        }
        acc += log_tau;
        // Normal on the log scale, so no Jacobian.
        if prior.dispersion_log.1.is_finite() {
            let (m, s) = prior.dispersion_log;
            acc -= 0.5 * ((log_disp - m) / s).powi(2);
        }
        if prior.intercept_sd.is_finite() {
            acc -= 0.5 * ((a - prior.intercept_mean) / prior.intercept_sd).powi(2);
        }
        if prior.beta_scale.is_finite() {
            acc -= 0.5 * (b / prior.beta_scale).powi(2);
        }
        acc
    }

    fn points(n_groups: usize) -> Vec<Vec<f64>> {
        let dim = 4 + n_groups;
        (0..6)
            .map(|k| {
                (0..dim)
                    .map(|j| {
                        let s = ((j * 7 + k * 13) % 11) as f64 / 11.0 - 0.5;
                        match j {
                            // Keep the intercept near a real log delay, and the two log
                            // scales in a range where the density is not numerically
                            // degenerate.
                            0 => 3.5 + s,
                            2 | 3 => s + 0.7,
                            _ => 1.5 * s,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn check_closed_form(dist: Dist, dist_slot: &str) {
        let (y, x, g, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = format!(
            r#"{{"y": "delay_days", "x": "amount_band", "group": "segment",
                 "dist": "{dist_slot}",
                 "prior": {{"intercept": {{"mean": 3.0, "sd": 2.0}},
                            "beta": {{"scale": 1.5}},
                            "tau": {{"scale": 0.8}},
                            "dispersion": {{"log_mean": 0.5, "log_sd": 1.2}}}}}}"#
        );
        let model = compile(&cfg, &view).unwrap();
        let target = model.as_differentiable().unwrap();
        let prior = Prior {
            intercept_mean: 3.0,
            intercept_sd: 2.0,
            beta_scale: 1.5,
            tau_scale: 0.8,
            dispersion_log: (0.5, 1.2),
        };

        let pts = points(3);
        for i in 1..pts.len() {
            let got = target.logp(&pts[i]) - target.logp(&pts[0]);
            let want = reference_logp(&y, &x, &g, 3, &pts[i], dist, &prior)
                - reference_logp(&y, &x, &g, 3, &pts[0], dist, &prior);
            assert!(
                (got - want).abs() < 1e-9 * want.abs().max(1.0),
                "{dist_slot} point {i}: logp difference {got} vs closed form {want}"
            );
        }
    }

    /// Differences of the log density between points, against the closed form. A
    /// difference rather than an absolute value because the density is only defined up
    /// to a constant, and pinning the constant would pin the implementation rather than
    /// the mathematics.
    #[test]
    fn the_gamma_log_density_matches_its_closed_form() {
        check_closed_form(Dist::Gamma, "gamma");
    }

    #[test]
    fn the_lognormal_log_density_matches_its_closed_form() {
        check_closed_form(Dist::Lognormal, "lognormal");
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
    /// (`ROADMAP.md` §2).
    #[test]
    fn the_log_jacobian_of_log_tau_is_present() {
        let (_, _, _, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let t = 0.9;
        let model = compile(
            &format!(
                r#"{{"y": "delay_days", "x": "amount_band", "group": "segment",
                     "prior": {{"tau": {{"scale": {t}}}}}}}"#
            ),
            &view,
        )
        .unwrap();
        let target = model.as_differentiable().unwrap();

        let mut theta = vec![0.0; target.dim()];
        theta[0] = 3.6; // intercept, so the residuals are not absurd
        theta[3] = 1.0; // log dispersion
        for (a, b) in [(-2.0, 0.4), (-0.5, 0.5), (-5.0, -4.0)] {
            let mut lo = theta.clone();
            lo[2] = a;
            let mut hi = theta.clone();
            hi[2] = b;
            let moved = target.logp(&hi) - target.logp(&lo);
            let prior_only = -((2.0 * b).exp() - (2.0 * a).exp()) / (2.0 * t * t);
            let want = (b - a) + prior_only;
            assert!(
                (moved - want).abs() < 1e-9,
                "log tau {a} -> {b} moved by {moved}, expected {want}"
            );
            assert!(
                (moved - prior_only).abs() > 1e-6,
                "the Jacobian term is indistinguishable from zero over {a} -> {b}, so \
                 this test could not detect it going missing"
            );
        }
    }

    /// **The dispersion prior carries no Jacobian, and that is deliberate.**
    ///
    /// It is declared on `log shape`, which *is* the sampled coordinate, so there is
    /// nothing to transform. Adding a Jacobian here would be as wrong as omitting the
    /// one on `tau`, and in the opposite direction. The check is the same construction:
    /// with the likelihood held fixed, moving `log shape` under a flat prior must move
    /// the density by the likelihood's own amount and nothing else.
    #[test]
    fn the_dispersion_prior_is_declared_on_the_coordinate_it_is_sampled_on() {
        let (_, _, _, frame) = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let (m, s) = (0.4, 1.1);
        let with = compile(
            &format!(
                r#"{{"y": "delay_days", "group": "segment",
                     "prior": {{"dispersion": {{"log_mean": {m}, "log_sd": {s}}}}}}}"#
            ),
            &view,
        )
        .unwrap();
        let without = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap();

        let dim = with.as_differentiable().unwrap().dim();
        let mut theta = vec![0.0; dim];
        theta[0] = 3.6;
        for v in [-0.5, 0.8, 1.7] {
            theta[2] = v;
            let a = with.as_differentiable().unwrap().logp(&theta);
            let b = without.as_differentiable().unwrap().logp(&theta);
            // The only difference between the two is the Normal prior on `log disp`,
            // evaluated at the same point. No Jacobian, in either.
            let want = -0.5 * ((theta[2] - m) / s).powi(2);
            assert!(
                (a - b - want).abs() < 1e-9,
                "log disp {v}: prior contributed {} rather than {want}",
                a - b
            );
        }
    }

    //=== The gradient ==========================================================//

    fn finite_difference_check(dist_slot: &str, seed: u64) {
        let frame = ledger(
            seed,
            5,
            14,
            if dist_slot == "gamma" {
                Dist::Gamma
            } else {
                Dist::Lognormal
            },
            if dist_slot == "gamma" { 5.0 } else { 0.4 },
        );
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(
            &format!(
                r#"{{"y": "delay_days", "group": "segment", "dist": "{dist_slot}",
                     "prior": {{"intercept": {{"mean": 3.0, "sd": 3.0}},
                                "tau": {{"scale": 0.8}},
                                "dispersion": {{"log_mean": 1.0, "log_sd": 1.0}}}}}}"#
            ),
            &view,
        )
        .unwrap();
        let target = model.as_differentiable().unwrap();
        let dim = target.dim();
        let start = target.initial();

        for k in 0..4 {
            // Deliberately far from `initial()`: a displacement of order one on every
            // coordinate, different in each direction. At a mode every component of the
            // analytic gradient is zero and so is every finite difference, so the
            // comparison would pass for any implementation whatsoever.
            let theta: Vec<f64> = (0..dim)
                .map(|j| start[j] + (((j * 5 + k * 3) % 7) as f64 - 3.0) * 0.25)
                .collect();

            let mut analytic = vec![0.0; dim];
            target.grad(&theta, &mut analytic).unwrap();
            let size = analytic.iter().map(|g| g.abs()).fold(0.0, f64::max);
            assert!(
                size > 1.0,
                "{dist_slot} point {k} is too near a mode: |grad|max = {size}"
            );

            for j in 0..dim {
                let h = 1e-6 * theta[j].abs().max(1.0);
                let mut up = theta.clone();
                up[j] += h;
                let mut down = theta.clone();
                down[j] -= h;
                let fd = (target.logp(&up) - target.logp(&down)) / (2.0 * h);
                let tol = 1e-4 * fd.abs().max(analytic[j].abs()).max(1.0);
                assert!(
                    (analytic[j] - fd).abs() < tol,
                    "{dist_slot} point {k}, coordinate {j}: analytic {} vs finite \
                     difference {fd}",
                    analytic[j]
                );
            }
        }
    }

    #[test]
    fn the_gamma_analytic_gradient_matches_finite_differences() {
        finite_difference_check("gamma", 11);
    }

    #[test]
    fn the_lognormal_analytic_gradient_matches_finite_differences() {
        finite_difference_check("lognormal", 12);
    }

    /// **The gradient on a dataset the family refuses.**
    ///
    /// Without this the finite-difference check above would pass on any dataset the
    /// model declined, which is exactly the dataset a wrong gradient produces: a
    /// refusing model exposes a standard normal, and a standard normal's gradient is
    /// trivially right.
    #[test]
    fn the_gradient_is_checked_on_the_real_surface_of_a_refused_fit() {
        // Every delay identical: degenerate, so `logp` serves the standard normal.
        let frame = Frame::new(12).numeric("delay_days", vec![30.0; 12]).key(
            "segment",
            vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        );
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"y": "delay_days", "group": "segment"}"#).unwrap();
        let model = build(&cfg, &view).unwrap();
        assert_eq!(
            model.readiness().status,
            crate::types::FitStatus::Degenerate
        );

        // The advertised surface is the standard normal...
        let mut out = vec![0.0; model.dim_inner()];
        let theta: Vec<f64> = (0..model.dim_inner())
            .map(|j| 0.3 * j as f64 - 0.4)
            .collect();
        model.grad(&theta, &mut out).unwrap();
        for (j, v) in out.iter().enumerate() {
            assert!((v + theta[j]).abs() < 1e-12);
        }

        // ...and the real one underneath it still has a correct gradient.
        let real = TrueSurface(&model);
        let mut analytic = vec![0.0; real.dim()];
        real.grad(&theta, &mut analytic).unwrap();
        for j in 0..real.dim() {
            let h = 1e-6 * theta[j].abs().max(1.0);
            let mut up = theta.clone();
            up[j] += h;
            let mut down = theta.clone();
            down[j] -= h;
            let fd = (real.logp(&up) - real.logp(&down)) / (2.0 * h);
            let tol = 1e-4 * fd.abs().max(analytic[j].abs()).max(1.0);
            assert!(
                (analytic[j] - fd).abs() < tol,
                "coordinate {j}: analytic {} vs finite difference {fd}",
                analytic[j]
            );
        }
    }

    //=== Behaviour: what the family is for =====================================//

    fn run(cfg: &str, frame: &Frame) -> crate::fit::Fit {
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        crate::fit::fit("payment_delay", &Config::parse(cfg).unwrap(), &view).unwrap()
    }

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

    /// **Parameter recovery.** Simulate from a known intercept, pooling scale and
    /// dispersion, fit, and require the posterior to cover the truth.
    ///
    /// **On the draw budget.** `tau` is estimated from as many numbers as there are
    /// segments, so it is the coordinate that mixes slowest and the one that decides
    /// this budget: at twelve segments and 4 x 1 000 draws its R-hat lands at 1.0125
    /// against a gate of 1.01, and the fit is graded `degenerate` for a posterior that
    /// is correct. The answer is more draws, not a looser gate — the gate is what makes
    /// the interval mean anything.
    #[test]
    fn a_simulated_ledger_recovers_the_delays_it_was_generated_from() {
        let (intercept, tau, shape) = (3.4f64, 0.35f64, 6.0f64);
        let mut rng = BayesRng::for_chain(2026, 0);
        let sim = testing::simulate(&mut rng, 12, 40, intercept, tau, Dist::Gamma, shape).unwrap();
        let frame = sim.frame();
        let fit = run(
            r#"{"y": "delay_days", "group": "segment",
                "draws": 2000, "chains": 4, "warmup": 2000, "seed": 7}"#,
            &frame,
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

        let b0 = col(&fit, "__global__", "intercept");
        let (lo, hi) = (quantile(&b0, 0.025), quantile(&b0, 0.975));
        println!("intercept: truth {intercept}, 95% [{lo:.3}, {hi:.3}]");
        assert!(
            lo < intercept && intercept < hi,
            "intercept 95% interval [{lo}, {hi}] misses {intercept}"
        );

        let k = col(&fit, "__global__", "shape");
        let (lo, hi) = (quantile(&k, 0.025), quantile(&k, 0.975));
        println!("shape: truth {shape}, 95% [{lo:.3}, {hi:.3}]");
        assert!(
            lo < shape && shape < hi,
            "shape 95% interval [{lo}, {hi}] misses {shape}"
        );

        // Each segment's own mean delay, which is the number a cash forecast reads.
        let mut covered = 0;
        for (j, &truth) in sim.true_mu.iter().enumerate() {
            let mu = col(&fit, &format!("SEG-{j:03}"), "mu");
            let (lo, hi) = (quantile(&mu, 0.025), quantile(&mu, 0.975));
            println!("mu[SEG-{j:03}]: truth {truth:.2}, 95% [{lo:.2}, {hi:.2}]");
            if lo < truth && truth < hi {
                covered += 1;
            }
        }
        // Twelve nominal-95% intervals; requiring all twelve would flake on a fixture
        // that is genuinely random, and requiring far fewer would not be a test.
        assert!(
            covered >= 10,
            "only {covered} of 12 segment means were covered by their 95% intervals"
        );
    }

    /// **The question agent 04 actually asks, and the reason the Gamma branch exists.**
    ///
    /// A cash-cover test reads the right tail. Fitted to the *same* right-skewed
    /// ledger, a Gamma and a lognormal agree closely about the mean and disagree about
    /// the 95th percentile — which is the number the covenant is written against. A
    /// family offering only one of the two would answer that question by assumption.
    #[test]
    fn the_two_branches_agree_about_the_centre_and_differ_in_the_tail() {
        let mut rng = BayesRng::for_chain(4004, 0);
        // Generated from the Gamma branch, so the lognormal is the misspecified one
        // here and its tail is the one that has to move.
        let frame = testing::simulate(&mut rng, 6, 30, 3.4, 0.3, Dist::Gamma, 2.0)
            .unwrap()
            .frame();
        let cfg = |d: &str| {
            format!(
                r#"{{"y": "delay_days", "group": "segment", "dist": "{d}",
                     "draws": 600, "chains": 2, "warmup": 800, "seed": 31}}"#
            )
        };
        let gamma = run(&cfg("gamma"), &frame);
        let lognormal = run(&cfg("lognormal"), &frame);

        // The posterior mean delay of one segment, under each branch.
        let centre = |fit: &crate::fit::Fit| mean(&col(fit, "SEG-000", "mu"));
        let (cg, cl) = (centre(&gamma), centre(&lognormal));
        println!("segment mean delay: gamma {cg:.2}, lognormal {cl:.2}");
        assert!(
            (cg / cl - 1.0).abs() < 0.20,
            "the two branches should agree about the centre: gamma {cg}, lognormal {cl}"
        );

        // The 95th percentile of one more invoice from that segment, which is what a
        // covenant date is tested against. Drawn once per posterior draw, so the
        // parameter uncertainty propagates rather than being conditioned away.
        let tail = |fit: &crate::fit::Fit, dist: Dist, disp_name: &str| {
            let mu = col(fit, "SEG-000", "mu");
            let d = col(fit, "__global__", disp_name);
            let mut rng = BayesRng::for_chain(555, 0);
            let pred: Vec<f64> = (0..mu.len())
                .map(|i| testing::draw_delay(&mut rng, mu[i], dist, d[i]).unwrap())
                .collect();
            quantile(&pred, 0.95)
        };
        let (tg, tl) = (
            tail(&gamma, Dist::Gamma, "shape"),
            tail(&lognormal, Dist::Lognormal, "sigma"),
        );
        println!("95th percentile delay: gamma {tg:.1}, lognormal {tl:.1}");
        assert!(
            (tl / tg - 1.0).abs() > 0.15,
            "the two branches gave the same tail ({tg} vs {tl}); if these agree, \
             offering both buys nothing and this family's premise is wrong"
        );
    }

    /// **The assertion on a function of several parameters at once.**
    ///
    /// SBC ranks one parameter at a time, so it tests *marginals*, and a marginal is
    /// exactly what a wrong correlation preserves: `ROADMAP.md` §3.1 records a diagonal
    /// posterior precision leaving all six SBC suites green while a predictive spread
    /// was wrong by 25x. So every family owes one check on a joint quantity.
    ///
    /// The joint quantity here is a segment's **mean delay** in the reported `mu`,
    /// which is `exp(intercept + u)` and therefore reads two coordinates at once. Its
    /// posterior has an external reference: for a segment with `n_g` invoices of its own
    /// and weak pooling, the log mean is data-dominated and its posterior sd is
    /// `sd(log delay) / sqrt(n_g)`. The intercept and the group offsets trade off along
    /// a ridge, so adding their marginal variances instead gives a number several times
    /// too large.
    #[test]
    fn the_posterior_of_a_segments_mean_is_a_joint_quantity_not_a_sum_of_marginals() {
        // A wide spread of segment levels, so the learned `tau` is large, the pooling
        // is weak, and `sd/sqrt(n)` is the right reference.
        let mut rng = BayesRng::for_chain(5150, 0);
        let n_per = 50;
        let sim = testing::simulate(&mut rng, 6, n_per, 3.4, 0.8, Dist::Gamma, 8.0).unwrap();
        let frame = sim.frame();
        let fit = run(
            r#"{"y": "delay_days", "group": "segment",
                "draws": 600, "chains": 2, "warmup": 800, "seed": 19,
                "prior": {"tau": {"scale": 3.0}}}"#,
            &frame,
        );

        let a = col(&fit, "__global__", "intercept");
        let u = col(&fit, "SEG-002", "u");
        let sd = |xs: &[f64]| {
            let m = mean(xs);
            (xs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
        };

        let log_level: Vec<f64> = (0..a.len()).map(|d| a[d] + u[d]).collect();
        let joint = sd(&log_level);
        let independent = (sd(&a).powi(2) + sd(&u).powi(2)).sqrt();
        // `Var(log y)` for a Gamma of shape k is `trigamma(k)`, which at k = 8 is close
        // to `1/k + 1/(2 k^2)`.
        let k = 8.0f64;
        let reference = (1.0 / k + 0.5 / (k * k)).sqrt() / (n_per as f64).sqrt();
        println!(
            "log level sd: joint {joint:.4}, marginals-added {independent:.4}, \
             reference sd/sqrt(n) {reference:.4}"
        );

        assert!(
            (joint - reference).abs() < 0.35 * reference,
            "the joint posterior sd of the segment's log level ({joint}) should be near \
             sd/sqrt(n) = {reference}"
        );
        assert!(
            independent > 1.8 * joint,
            "adding the marginal variances gave {independent} against the joint \
             {joint}; if these agree the parameters are uncorrelated and this test \
             checks nothing"
        );
    }

    //=== Engines and refusals ==================================================//

    /// There is no closed form here, so the exact engine must decline rather than
    /// substitute something.
    #[test]
    fn the_exact_engine_declines_this_family_rather_than_approximating_it() {
        let frame = ledger(21, 4, 10, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap();
        assert!(model.as_exact().is_none());
        assert!(!crate::engines::ExactEngine.supports(&*model));
        assert!(crate::engines::NutsEngine.supports(&*model));
    }

    /// The Laplace engine is refused at *compile* time with a reason, rather than left
    /// to produce a Gaussian at a mode that is not there. See the argument in `build`.
    #[test]
    fn an_explicit_laplace_request_is_refused_with_its_reason() {
        let frame = ledger(22, 4, 10, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"y": "delay_days", "group": "segment", "engine": "laplace"}"#,
            &view,
        )
        .unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "engine"),
            "{err}"
        );
        assert!(err.to_string().contains("NUTS only"), "{err}");
    }

    /// A prior-predictive check needs a distribution to draw from in closed form, and
    /// this family has none.
    #[test]
    fn a_prior_predictive_check_is_refused_because_there_is_no_closed_form_prior_draw() {
        let frame = ledger(23, 4, 10, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"y": "delay_days", "group": "segment", "sample_from": "prior"}"#,
            &view,
        )
        .unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "sample_from"),
            "{err}"
        );
    }

    /// **The clock, named in the error.**
    ///
    /// A delay measured from the due date is negative whenever an invoice is paid
    /// early, and this family has no support there. Refusing with a sentence that says
    /// which clock to use is worth more than a number computed from the rows that
    /// happened to be positive — that subset is exactly the late payers, and a cash
    /// forecast fitted to it would be biased in the direction that matters most.
    #[test]
    fn a_delay_measured_from_the_due_date_is_refused_with_the_clock_named() {
        let frame = Frame::new(9)
            .numeric(
                "delay_days",
                vec![3.0, -2.0, 8.0, 4.0, 1.0, 9.0, 2.0, 7.0, 5.0],
            )
            .key("segment", vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "y"),
            "{err}"
        );
        let msg = err.to_string();
        assert!(msg.contains("invoice date"), "{msg}");
        assert!(msg.contains("censored_aft"), "{msg}");
    }

    /// A zero delay is refused for the same reason and by the same check: on the log
    /// scale it is minus infinity, and a same-day payment is a rounding of the clock
    /// rather than an observation of zero duration.
    #[test]
    fn a_zero_delay_is_refused_rather_than_taken_as_a_duration() {
        let frame = Frame::new(9)
            .numeric(
                "delay_days",
                vec![3.0, 0.0, 8.0, 4.0, 1.0, 9.0, 2.0, 7.0, 5.0],
            )
            .key("segment", vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        assert!(matches!(
            compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap_err(),
            BayesError::Config { ref slot, .. } if slot == "y"
        ));
    }

    /// Two segments cannot identify a pooling scale, and saying so is worth more than
    /// a number that describes the sample rather than the population.
    #[test]
    fn too_few_segments_cannot_identify_a_pooling_scale() {
        let frame = ledger(24, 2, 20, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap();
        assert_eq!(
            model.readiness().status,
            crate::types::FitStatus::InsufficientData
        );
        assert!(
            model.readiness().reasons[0].contains("min_groups"),
            "{:?}",
            model.readiness().reasons
        );
    }

    /// A ledger whose invoices all clear on the same day has no dispersion to estimate.
    /// The verdict is `degenerate` rather than an error, because it is a fact about the
    /// data that an agent reads off `__status__` alongside everything else.
    #[test]
    fn a_ledger_with_no_variation_at_all_is_degenerate_rather_than_fitted() {
        let frame = Frame::new(12).numeric("delay_days", vec![30.0; 12]).key(
            "segment",
            vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        );
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap();
        assert_eq!(
            model.readiness().status,
            crate::types::FitStatus::Degenerate
        );
        // ...and the draws it produces are NULL-shaped rather than plausible numbers.
        let mut out = vec![0.0; model.param_names().len()];
        model.as_differentiable().unwrap().constrain(
            &vec![0.3; model.as_differentiable().unwrap().dim()],
            &mut out,
        );
        assert!(out.iter().all(|v| v.is_nan()));
    }

    /// The group column is required. A family whose subject is *per-segment* payment
    /// behaviour has nothing to say about a ledger with no segments, and silently
    /// pooling everything would be a different model wearing this one's name.
    #[test]
    fn the_group_column_is_required_rather_than_optional() {
        let frame = ledger(25, 4, 10, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"y": "delay_days"}"#, &view).unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "group"),
            "{err}"
        );
    }

    /// A misspelled slot must name itself and suggest the nearest real one.
    #[test]
    fn an_unknown_slot_is_rejected_with_a_suggestion() {
        let frame = ledger(26, 4, 10, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"y": "delay_days", "group": "segment", "distribution": "gamma"}"#,
            &view,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown option"), "{msg}");
        assert!(msg.contains("dist"), "{msg}");
    }

    /// The constrained draw is what reaches SQL, so every positive quantity must be
    /// positive by construction rather than by luck.
    #[test]
    fn the_constrained_draw_is_positive_wherever_the_parameter_is_a_scale() {
        let frame = ledger(4, 4, 15, Dist::Gamma, 4.0);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"y": "delay_days", "group": "segment"}"#, &view).unwrap();
        let target = model.as_differentiable().unwrap();

        let mut theta = target.initial();
        for (j, v) in theta.iter_mut().enumerate() {
            *v += ((j % 5) as f64 - 2.0) * 0.9;
        }
        let mut out = vec![0.0; model.param_names().len()];
        target.constrain(&theta, &mut out);

        // intercept, tau, shape, then (u, mu) per group.
        assert!(out[1] > 0.0, "tau {}", out[1]);
        assert!(out[2] > 0.0, "shape {}", out[2]);
        for g in 0..4 {
            assert!(out[3 + 2 * g + 1] > 0.0, "mu[{g}] {}", out[3 + 2 * g + 1]);
        }
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
