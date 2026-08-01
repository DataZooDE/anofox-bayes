//! F1 — hierarchical count GLM, Poisson or negative binomial, with partial pooling.
//!
//! The inference layer under the C-parts safety-stock agent. Its question is "how much
//! of this part will be wanted next period, and how sure are we", asked of thousands of
//! SKUs of which most have a handful of observations. A point forecast answers half of
//! it; the reorder point is a **quantile**, so the interval *is* the decision.

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::types::{EngineKind, FamilyCode};

use super::{CompiledModel, LogPosterior, ModelFamily, Readiness};
use statrs::function::gamma::{digamma, ln_gamma};

/// The family singleton registered in the catalog.
#[derive(Debug)]
pub struct HierNegbin;

const SLOTS: &[&str] = &[
    "y",
    "group",
    "x",
    "exposure",
    "likelihood",
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

/// Largest linear predictor the arithmetic will evaluate.
///
/// `e^60` is about `1e26`. No count model anyone holds has a mean there; a linear
/// predictor that large is a search that has run away, and evaluating it would put
/// `exp` into a region where the density is an infinity and every number after it a
/// NaN.
const ETA_MAX: f64 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Likelihood {
    Poisson,
    NegBinomial,
}

impl Likelihood {
    fn parse(name: &str) -> BayesResult<Self> {
        match name {
            "poisson" => Ok(Likelihood::Poisson),
            "negbinomial" => Ok(Likelihood::NegBinomial),
            other => Err(BayesError::config(
                "likelihood",
                format!("unknown: '{other}'"),
            )),
        }
    }
}

impl ModelFamily for HierNegbin {
    fn id(&self) -> &'static str {
        "hier_negbin"
    }

    fn code(&self) -> FamilyCode {
        FamilyCode::HierNegbin
    }

    fn description(&self) -> &'static str {
        "Hierarchical count GLM -- Poisson or negative binomial with a partially \
         pooled per-group level, non-centred -- for per-SKU demand and the reorder \
         quantile a safety-stock decision reads off it."
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
    /// Half-normal scale for `tau`, or infinite for the default uniform prior.
    tau_scale: f64,
    /// Lognormal hyperparameters for `phi`, or `None` for the default uniform-on-`1/phi`.
    phi_lognormal: Option<(f64, f64)>,
}

impl Prior {
    fn parse(cfg: &Config) -> BayesResult<Self> {
        cfg.reject_unknown(&["intercept", "beta", "tau", "phi"])?;

        let intercept = cfg.nested("intercept")?;
        intercept.reject_unknown(&["mean", "sd"])?;
        let intercept_mean = intercept.f64_or("mean", 0.0)?;
        let intercept_sd = intercept.positive_f64_or("sd", f64::INFINITY)?;

        let beta = cfg.nested("beta")?;
        beta.reject_unknown(&["scale"])?;
        let beta_scale = beta.positive_f64_or("scale", f64::INFINITY)?;

        let tau = cfg.nested("tau")?;
        tau.reject_unknown(&["scale"])?;
        let tau_scale = tau.positive_f64_or("scale", f64::INFINITY)?;

        let phi = cfg.nested("phi")?;
        phi.reject_unknown(&["log_mean", "log_sd"])?;
        // Present-or-absent rather than a sentinel default: `log_sd` is what switches
        // `phi` from its reference prior to a lognormal one, and a caller who wrote
        // only `log_mean` has asked for something this family cannot serve.
        let phi_lognormal = if phi.keys().contains(&"log_sd") {
            Some((
                phi.f64_or("log_mean", 0.0)?,
                phi.positive_f64_or("log_sd", 1.0)?,
            ))
        } else if phi.keys().contains(&"log_mean") {
            return Err(BayesError::config(
                "prior.phi.log_sd",
                "is required alongside 'log_mean': a prior mean with no spread is not a prior",
            ));
        } else {
            None
        };

        Ok(Prior {
            intercept_mean,
            intercept_sd,
            beta_scale,
            tau_scale,
            phi_lognormal,
        })
    }

    /// Every coordinate carries a proper prior, so the joint is samplable and the
    /// family can be certified by SBC.
    pub(crate) fn is_proper(&self) -> bool {
        self.intercept_sd.is_finite() && self.beta_scale.is_finite() && self.tau_scale.is_finite()
    }
}

#[derive(Debug)]
pub(crate) struct CompiledHierNegbin {
    params: Vec<ParamName>,
    /// Response, one entry per usable row.
    y: Vec<f64>,
    /// Design, `p` columns of `n` values each. Empty when there are no covariates.
    x: Vec<Vec<f64>>,
    /// `log(exposure)` per row, all zero when no exposure column was named.
    log_exposure: Vec<f64>,
    /// Dense group index per row.
    group_of: Vec<usize>,
    group_keys: Vec<String>,
    likelihood: Likelihood,
    prior: Prior,
    structural: Option<Readiness>,
    fingerprint: String,
    /// Starting point for the mode search / the sampler.
    start: Vec<f64>,
}

/// Coordinate layout of the unconstrained vector.
///
/// ```text
///   [0]                  intercept
///   [1 .. 1+p]           beta, in the caller's column order
///   [1+p]                log tau
///   [2+p]                log phi        (negative binomial only)
///   [k0 .. k0+G]         z, the non-centred group offsets
/// ```
impl CompiledHierNegbin {
    fn n_beta(&self) -> usize {
        self.x.len()
    }
    fn i_log_tau(&self) -> usize {
        1 + self.n_beta()
    }
    fn i_log_phi(&self) -> Option<usize> {
        match self.likelihood {
            Likelihood::NegBinomial => Some(2 + self.n_beta()),
            Likelihood::Poisson => None,
        }
    }
    fn i_z0(&self) -> usize {
        match self.likelihood {
            Likelihood::NegBinomial => 3 + self.n_beta(),
            Likelihood::Poisson => 2 + self.n_beta(),
        }
    }
    pub(crate) fn n_groups_inner(&self) -> usize {
        self.group_keys.len()
    }

    /// Whether the compile-time verdict was that there is no posterior here.
    ///
    /// `InsufficientData` is deliberately not one of these: that verdict says the
    /// data is weak, not that the surface is unusable, so the draws are real and the
    /// status is what refuses. `Degenerate` and `Failed` say the opposite -- there is
    /// no interior maximum, and a number taken from one would be indistinguishable
    /// from an estimate.
    fn refuses(&self) -> bool {
        self.structural.as_ref().is_some_and(|r| {
            matches!(
                r.status,
                crate::types::FitStatus::Degenerate | crate::types::FitStatus::Failed
            )
        })
    }
    fn dim_inner(&self) -> usize {
        self.i_z0() + self.n_groups_inner()
    }

    /// Linear predictor for row `i` at `theta`.
    fn eta(&self, theta: &[f64], i: usize) -> f64 {
        let tau = theta[self.i_log_tau()].exp();
        let z = theta[self.i_z0() + self.group_of[i]];
        let mut eta = theta[0] + tau * z + self.log_exposure[i];
        for (k, col) in self.x.iter().enumerate() {
            eta += theta[1 + k] * col[i];
        }
        eta
    }

    /// The log posterior, and optionally its gradient.
    ///
    /// One function rather than two because every derivative below is a reweighting
    /// of the same per-row quantities the density already computed; deriving them
    /// twice would double the cost of a NUTS step for nothing.
    pub(crate) fn logp_and_grad(&self, theta: &[f64], mut grad: Option<&mut [f64]>) -> f64 {
        let dim = self.dim_inner();
        if let Some(g) = grad.as_deref_mut() {
            g[..dim].fill(0.0);
        }

        let log_tau = theta[self.i_log_tau()];
        if !log_tau.is_finite() || log_tau.abs() > LOG_BOX {
            return f64::NEG_INFINITY;
        }
        let tau = log_tau.exp();
        let (phi, log_phi) = match self.i_log_phi() {
            Some(i) => {
                let lp = theta[i];
                if !lp.is_finite() || lp.abs() > LOG_BOX {
                    return f64::NEG_INFINITY;
                }
                (lp.exp(), lp)
            }
            None => (f64::INFINITY, 0.0),
        };
        if theta[..self.i_z0()].iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }

        let mut total = 0.0;
        // Accumulated d(log likelihood)/d(eta_i), needed by four different
        // coordinates, so it is formed once per row.
        let mut d_phi = 0.0;
        for i in 0..self.y.len() {
            let eta = self.eta(theta, i);
            if !eta.is_finite() || eta > ETA_MAX {
                return f64::NEG_INFINITY;
            }
            let mu = eta.exp();
            let y = self.y[i];
            let d_eta;
            match self.likelihood {
                Likelihood::Poisson => {
                    total += y * eta - mu - ln_gamma(y + 1.0);
                    d_eta = y - mu;
                }
                Likelihood::NegBinomial => {
                    let s = phi + mu;
                    total += ln_gamma(y + phi) - ln_gamma(phi) - ln_gamma(y + 1.0) + phi * log_phi
                        - (y + phi) * s.ln()
                        + y * eta;
                    d_eta = y - (y + phi) * mu / s;
                    if grad.is_some() {
                        d_phi += digamma(y + phi) - digamma(phi) + log_phi + 1.0
                            - s.ln()
                            - (y + phi) / s;
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

        // The non-centred hierarchy: z ~ N(0, 1), and the pooling scale lives in the
        // linear predictor rather than in this density. That is the whole point of the
        // parameterisation -- the prior on z does not depend on tau, so the funnel
        // that a centred version has at small tau is not there to fall into.
        for j in 0..self.n_groups_inner() {
            let z = theta[self.i_z0() + j];
            total -= 0.5 * z * z;
            if let Some(g) = grad.as_deref_mut() {
                g[self.i_z0() + j] -= z;
            }
        }

        // --- Priors, all declared on the natural scale, with their Jacobians. ---
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
        // `tau`: the density is declared on `tau`, the coordinate is `log tau`, so a
        // `+ log tau` Jacobian appears whichever prior is in force. Omitting it is
        // invisible to every engine-agreement test -- both engines would explore the
        // same wrong surface -- and shows up only against the closed form.
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
        if let Some(i_phi) = self.i_log_phi() {
            match self.prior.phi_lognormal {
                // Declared on `log phi` directly, so no Jacobian: a lognormal prior on
                // `phi` *is* a normal prior on `log phi`.
                Some((m, s)) => {
                    let d = log_phi - m;
                    total -= 0.5 * d * d / (s * s);
                    if let Some(g) = grad.as_deref_mut() {
                        g[i_phi] -= d / (s * s);
                    }
                }
                // The default: flat on the overdispersion `alpha = 1/phi`. In the
                // `log phi` coordinate that is `-log phi`.
                None => {
                    total -= log_phi;
                    if let Some(g) = grad.as_deref_mut() {
                        g[i_phi] -= 1.0;
                    }
                }
            }
            if let Some(g) = &mut grad {
                g[i_phi] += phi * d_phi;
            }
        }

        total
    }
}

impl CompiledModel for CompiledHierNegbin {
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

impl LogPosterior for CompiledHierNegbin {
    fn dim(&self) -> usize {
        self.dim_inner()
    }

    fn logp(&self, theta: &[f64]) -> f64 {
        if self.refuses() {
            // The surface a refusing model exposes: a standard normal, trivially
            // explorable, whose draws `constrain` turns into NaN. Refusing through
            // the *status* is worth more to an agent than refusing through an error,
            // because a `degenerate` fit is a row in a table it already reads -- but
            // letting a sampler loose on a likelihood with no interior maximum would
            // burn a warmup budget walking to the edge of the box.
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
        if let Some(i) = self.i_log_phi() {
            out[at] = theta[i].exp();
            at += 1;
        }
        for j in 0..self.n_groups_inner() {
            let u = tau * theta[self.i_z0() + j];
            out[at] = u;
            out[at + 1] = (theta[0] + u).exp();
            at += 2;
        }
    }
}

pub(crate) fn build(cfg: &Config, data: &DataView) -> BayesResult<CompiledHierNegbin> {
    cfg.reject_unknown(SLOTS)?;

    let y_name = cfg.require_str("y")?.to_string();
    let group_name = cfg.require_str("group")?.to_string();
    let x_names = cfg.str_list("x")?;
    let exposure_name = cfg.opt_str("exposure")?.map(str::to_string);
    let likelihood = Likelihood::parse(&cfg.one_of(
        "likelihood",
        &["negbinomial", "poisson"],
        "negbinomial",
    )?)?;
    let min_groups = cfg.usize_in("min_groups", 3, 2, 1_000_000_000)?;

    // The one engine question a family is allowed to answer, because it is a
    // statement about *this* geometry and nothing else knows it.
    //
    // A Laplace posterior is a Gaussian at the joint mode. This family has no usable
    // joint mode: under the non-centred parameterisation the likelihood does not
    // depend on `tau` at all when every `z` is zero, so the density has a ridge along
    // `{z = 0, tau -> infinity}` that the `+ log tau` Jacobian makes rise without
    // bound. The ridge carries no posterior *mass* -- the region where the likelihood
    // is any good shrinks like `tau^-G` -- which is why a sampler is untroubled by it
    // and a mode search walks straight up it. Measured: with the default prior the
    // search does not converge at all; with a proper half-normal(1) on `tau` it
    // converges to a mean `tau` of 1.63 where the truth is 0.5, and grades itself
    // `degenerate`. Serving that as a posterior would be an approximation nobody
    // asked for, so it is refused instead.
    if let Some("laplace") = cfg
        .opt_str("engine")?
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        return Err(BayesError::config(
            "engine",
            "'hier_negbin' is served by NUTS only. A Laplace posterior is a Gaussian at \
             the joint mode, and a non-centred hierarchy has no usable one: the \
             likelihood does not depend on the pooling scale when every group offset is \
             zero, so the mode search climbs a ridge that carries no posterior mass. \
             Drop the 'engine' slot to use the default",
        ));
    }
    let prior = Prior::parse(&cfg.nested("prior")?)?;

    let mut numeric_cols: Vec<&str> = vec![y_name.as_str()];
    numeric_cols.extend(x_names.iter().map(String::as_str));
    if let Some(e) = &exposure_name {
        numeric_cols.push(e.as_str());
    }
    let key_cols = [group_name.as_str()];

    let rows = data.usable_rows(&numeric_cols, &key_cols)?;
    let fingerprint = data.fingerprint(&numeric_cols, &key_cols, &rows)?;

    let y_col = data.numeric(&y_name)?;
    let x_cols: Vec<_> = x_names
        .iter()
        .map(|n| data.numeric(n))
        .collect::<BayesResult<_>>()?;
    let exposure_col = exposure_name
        .as_ref()
        .map(|n| data.numeric(n))
        .transpose()?;

    let groups = data.group_rows(Some(&group_name), &rows)?;
    for (key, _) in &groups {
        crate::types::validate_group_key(key)?;
    }
    let group_keys: Vec<String> = groups.iter().map(|(k, _)| k.clone()).collect();

    // Row order follows the groups, so a group's observations are contiguous. Nothing
    // in the arithmetic requires it; it makes the per-row loop's memory access
    // predictable and the fixtures readable.
    let mut y = Vec::with_capacity(rows.len());
    let mut group_of = Vec::with_capacity(rows.len());
    let mut log_exposure = Vec::with_capacity(rows.len());
    let mut x: Vec<Vec<f64>> = vec![Vec::with_capacity(rows.len()); x_cols.len()];
    for (j, (_, members)) in groups.iter().enumerate() {
        for &i in members {
            let v = y_col.values[i];
            if v < 0.0 || v.fract() != 0.0 {
                return Err(BayesError::config(
                    "y",
                    format!("must be a non-negative whole count; row {i} is {v}"),
                ));
            }
            y.push(v);
            group_of.push(j);
            match exposure_col {
                Some(c) => {
                    let e = c.values[i];
                    if !e.is_finite() || e <= 0.0 {
                        return Err(BayesError::config(
                            "exposure",
                            format!("must be > 0: a row observed over no exposure carries no information; row {i} is {e}"),
                        ));
                    }
                    log_exposure.push(e.ln());
                }
                None => log_exposure.push(0.0),
            }
            for (k, col) in x_cols.iter().enumerate() {
                x[k].push(col.values[i]);
            }
        }
    }

    let n = y.len();
    let n_groups = group_keys.len();
    let p = x_cols.len();
    let n_fixed = 1 + p + 1 + usize::from(likelihood == Likelihood::NegBinomial);
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
    if likelihood == Likelihood::NegBinomial {
        params.push(ParamName::global("phi")?);
    }
    for key in &group_keys {
        params.push(ParamName::grouped(key.clone(), "u")?);
        params.push(ParamName::grouped(key.clone(), "rate")?);
    }

    // --- The verdicts reachable from the sufficient statistics alone. ---
    let total: f64 = y.iter().sum();
    let structural = if n_groups < min_groups {
        Some(Readiness::insufficient(format!(
            "{n_groups} group(s) is below the min_groups threshold of {min_groups}: a \
             pooling scale estimated from this few describes the sample rather than the \
             population, and every per-group interval inherits that"
        )))
    } else if total <= 0.0 {
        Some(Readiness::degenerate(format!(
            "every one of the {n} observations is zero, so the level is identified only \
             in the limit of a zero rate and there is no interior maximum to put a \
             posterior around.{}",
            if prior.is_proper() {
                " The prior is already proper, so it is the data that says nothing: a \
                 part nobody has ever issued has no demand rate to estimate"
            } else {
                " Set a proper `prior.intercept` if a catalogue-wide rate is a defensible \
                 answer for a part nobody has ever issued"
            }
        )))
    } else {
        None
    };

    let start = starting_point(&y, &group_of, &log_exposure, n_groups, p, likelihood);

    Ok(CompiledHierNegbin {
        params,
        y,
        x,
        log_exposure,
        group_of,
        group_keys,
        likelihood,
        prior,
        structural,
        fingerprint,
        start,
    })
}

/// A starting point already scaled to the data.
///
/// The coordinate that matters is the intercept: it is a log rate in the caller's
/// units, so starting it at zero when the SKUs move a thousand units a week costs the
/// sampler its whole warmup climbing out of a region where the density is numerically
/// flat. `z` starts from each group's own observed deviation, divided by the starting
/// `tau`, which is the non-centred coordinate of the group's sample level.
fn starting_point(
    y: &[f64],
    group_of: &[usize],
    log_exposure: &[f64],
    n_groups: usize,
    p: usize,
    likelihood: Likelihood,
) -> Vec<f64> {
    let n = y.len();
    let mut sum = vec![0.0; n_groups];
    let mut expo = vec![0.0; n_groups];
    for i in 0..n {
        sum[group_of[i]] += y[i];
        expo[group_of[i]] += log_exposure[i].exp();
    }
    // Half a count, so a group that saw nothing still has a finite log rate.
    let log_rate: Vec<f64> = (0..n_groups)
        .map(|j| ((sum[j] + 0.5) / expo[j].max(1e-12)).ln())
        .collect();
    let b0 = log_rate.iter().sum::<f64>() / n_groups as f64;
    let var = log_rate.iter().map(|r| (r - b0).powi(2)).sum::<f64>() / n_groups.max(2) as f64;
    // The observed spread of group log-rates is between-group variance *plus* sampling
    // noise, so it overstates tau. Half of it is a start, not an estimate.
    let tau = (0.5 * var).sqrt().clamp(0.05, 5.0);

    let n_fixed = 1 + p + 1 + usize::from(likelihood == Likelihood::NegBinomial);
    let mut theta = vec![0.0; n_fixed + n_groups];
    theta[0] = b0.clamp(-ETA_MAX / 2.0, ETA_MAX / 2.0);
    theta[1 + p] = tau.ln();
    if likelihood == Likelihood::NegBinomial {
        theta[2 + p] = 0.0;
    }
    for j in 0..n_groups {
        theta[n_fixed + j] = ((log_rate[j] - b0) / tau).clamp(-5.0, 5.0);
    }
    theta
}

/// The real surface, exposed regardless of the compile-time verdict.
///
/// Exists for one test, and that test is the most valuable one in the module: without
/// it the finite-difference check would pass on any dataset the model refused, which
/// is exactly the dataset a wrong gradient produces.
#[cfg(test)]
pub(crate) struct TrueSurface<'a>(pub(crate) &'a CompiledHierNegbin);

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

    use crate::data::testing::Frame;
    use crate::errors::BayesResult;
    use crate::rng::BayesRng;

    /// A simulated SKU panel in the columns the family reads.
    pub(crate) struct Panel {
        pub y: Vec<f64>,
        pub sku: Vec<String>,
        /// One held-out next observation per SKU -- the quantity a reorder point is
        /// a promise about.
        pub holdout: Vec<f64>,
        /// The rate each SKU was actually generated from.
        pub true_rate: Vec<f64>,
        pub n_groups: usize,
    }

    impl Panel {
        pub(crate) fn frame(&self) -> Frame {
            Frame::new(self.y.len())
                .numeric("units", self.y.clone())
                .key("sku", self.sku.iter().map(String::as_str).collect())
        }
    }

    /// One draw from `NegBin(mean mu, dispersion phi)`, as the Poisson-Gamma mixture
    /// the likelihood is derived from. `phi = None` is the Poisson limit.
    pub(crate) fn draw_count(rng: &mut BayesRng, mu: f64, phi: Option<f64>) -> BayesResult<f64> {
        let lambda = match phi {
            Some(phi) => rng.gamma(phi, phi / mu)?,
            None => mu,
        };
        rng.poisson(lambda)
    }

    /// Simulate `n_groups` SKUs of `n_per` periods each from the family's own model.
    pub(crate) fn simulate(
        rng: &mut BayesRng,
        n_groups: usize,
        n_per: usize,
        intercept: f64,
        tau: f64,
        phi: Option<f64>,
    ) -> BayesResult<Panel> {
        let (mut y, mut sku, mut holdout, mut true_rate) = (vec![], vec![], vec![], vec![]);
        for j in 0..n_groups {
            let rate = (intercept + tau * rng.standard_normal()).exp();
            true_rate.push(rate);
            for _ in 0..n_per {
                y.push(draw_count(rng, rate, phi)?);
                sku.push(format!("SKU-{j:04}"));
            }
            holdout.push(draw_count(rng, rate, phi)?);
        }
        Ok(Panel {
            y,
            sku,
            holdout,
            true_rate,
            n_groups,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::testing::Frame;

    /// A small panel: three SKUs, four periods each, with one covariate and an
    /// exposure column, so every term of the density is exercised.
    fn fixture() -> Frame {
        Frame::new(12)
            .numeric(
                "units",
                vec![2.0, 0.0, 5.0, 1.0, 9.0, 12.0, 7.0, 11.0, 0.0, 1.0, 0.0, 2.0],
            )
            .numeric(
                "promo",
                vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            )
            .numeric(
                "weeks",
                vec![1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            )
            .key(
                "sku",
                vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            )
    }

    fn compile(cfg: &str) -> CompiledHierNegbin {
        let frame = fixture();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        build(&Config::parse(cfg).unwrap(), &view).unwrap()
    }

    const FULL: &str = r#"{"y": "units", "group": "sku", "x": ["promo"], "exposure": "weeks"}"#;

    /// The density, against the formula written out independently.
    ///
    /// Engine agreement cannot do this job: a missing log-Jacobian is a term both
    /// engines would explore identically, so both would agree on the same wrong
    /// surface. Only a direct comparison against the closed form sees it.
    #[test]
    fn the_log_density_matches_its_closed_form() {
        let m = compile(FULL);
        let theta: Vec<f64> = vec![
            0.7,  // intercept
            -0.3, // beta[promo]
            -0.5, // log tau
            0.4,  // log phi
            0.8, -1.1, 0.25, // z
        ];
        assert_eq!(m.dim(), theta.len());

        let (b0, beta, tau, phi) = (theta[0], theta[1], theta[2].exp(), theta[3].exp());
        let y: [f64; 12] = [2.0, 0.0, 5.0, 1.0, 9.0, 12.0, 7.0, 11.0, 0.0, 1.0, 0.0, 2.0];
        let promo: [f64; 12] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let weeks: [f64; 12] = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        let z: [f64; 3] = [0.8, -1.1, 0.25];

        let mut expected = 0.0;
        for i in 0..12 {
            let eta = b0 + beta * promo[i] + tau * z[i / 4] + weeks[i].ln();
            let mu = eta.exp();
            expected += ln_gamma(y[i] + phi) - ln_gamma(phi) - ln_gamma(y[i] + 1.0)
                + phi * phi.ln()
                - (y[i] + phi) * (phi + mu).ln()
                + y[i] * eta;
        }
        // z ~ N(0, 1), which is the whole of the hierarchy's prior under the
        // non-centred parameterisation.
        for zj in z {
            expected += -0.5 * zj * zj;
        }
        // The two Jacobians. `tau` carries a uniform prior on the natural scale, so
        // moving to `log tau` multiplies the density by `tau`; `phi` carries a
        // uniform prior on `1/phi`, so moving to `log phi` multiplies it by `1/phi`.
        expected += tau.ln();
        expected -= phi.ln();

        let got = m.logp(&theta);
        assert!(
            (got - expected).abs() < 1e-9,
            "logp {got} against closed form {expected}"
        );
    }

    /// The same, for the Poisson branch, where there is no `phi` and therefore one
    /// fewer coordinate and one fewer Jacobian.
    #[test]
    fn the_poisson_log_density_matches_its_closed_form() {
        let m = compile(
            r#"{"y": "units", "group": "sku", "likelihood": "poisson", "exposure": "weeks"}"#,
        );
        let theta: Vec<f64> = vec![0.9, -0.4, 0.3, -0.7, 1.2];
        assert_eq!(m.dim(), theta.len());
        let (b0, tau) = (theta[0], theta[1].exp());
        let y: [f64; 12] = [2.0, 0.0, 5.0, 1.0, 9.0, 12.0, 7.0, 11.0, 0.0, 1.0, 0.0, 2.0];
        let weeks: [f64; 12] = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        let z: [f64; 3] = [theta[2], theta[3], theta[4]];
        let mut expected = 0.0;
        for i in 0..12 {
            let eta = b0 + tau * z[i / 4] + weeks[i].ln();
            expected += y[i] * eta - eta.exp() - ln_gamma(y[i] + 1.0);
        }
        for zj in z {
            expected += -0.5 * zj * zj;
        }
        expected += tau.ln();
        assert!((m.logp(&theta) - expected).abs() < 1e-9);
    }

    /// The Jacobian is not decoration: dropping it changes the density by exactly
    /// `log(tau2 / tau1)`, which is `O(1)` in the parameter and `O(1/n)` against a
    /// likelihood, so it hides inside any tolerance an engine comparison could use.
    #[test]
    fn the_tau_jacobian_is_present_and_has_the_right_size() {
        let m = compile(FULL);
        let mut a: Vec<f64> = vec![0.7, -0.3, -0.5, 0.4, 0.0, 0.0, 0.0];
        let mut b = a.clone();
        // Move only `log tau`, and put `z` at zero so the likelihood does not move
        // with it: what is left is the prior plus its Jacobian.
        a[2] = -2.0;
        b[2] = 1.0;
        assert!(
            (m.logp(&b) - m.logp(&a) - (1.0 - -2.0)).abs() < 1e-12,
            "the difference must be exactly log(tau_b) - log(tau_a)"
        );
    }

    /// **Away from the mode.** At the mode the gradient is zero, so a sign error is
    /// invisible; this is the check that sees one.
    #[test]
    fn analytic_gradient_matches_finite_differences() {
        for cfg in [
            FULL,
            r#"{"y": "units", "group": "sku"}"#,
            r#"{"y": "units", "group": "sku", "likelihood": "poisson", "x": ["promo"]}"#,
            r#"{"y": "units", "group": "sku", "exposure": "weeks",
                 "prior": {"intercept": {"mean": 1.0, "sd": 2.0}, "tau": {"scale": 1.5},
                           "phi": {"log_mean": 0.5, "log_sd": 0.8}}}"#,
            r#"{"y": "units", "group": "sku", "x": ["promo"],
                 "prior": {"beta": {"scale": 0.5}}}"#,
        ] {
            let compiled = compile(cfg);
            let m = TrueSurface(&compiled);
            let dim = m.dim();
            // Deliberately off-mode points: a mode search is never run here.
            for offset in [0.0_f64, 0.7, -1.3] {
                let theta: Vec<f64> = (0..dim)
                    .map(|j| 0.3 + offset + 0.37 * ((j as f64) * 1.7).sin())
                    .collect();
                let mut analytic = vec![0.0; dim];
                m.grad(&theta, &mut analytic).unwrap();
                // Guard against a vacuous pass: at a genuine mode every entry would
                // be zero and any implementation would agree.
                assert!(
                    analytic.iter().map(|g| g.abs()).fold(0.0, f64::max) > 1e-3,
                    "{cfg}: the test point is at a stationary point, so it proves nothing"
                );
                for j in 0..dim {
                    let h = 1e-6 * theta[j].abs().max(1.0);
                    let mut up = theta.clone();
                    let mut down = theta.clone();
                    up[j] += h;
                    down[j] -= h;
                    let fd = (m.logp(&up) - m.logp(&down)) / (2.0 * h);
                    let tol = 1e-4 * fd.abs().max(1.0);
                    assert!(
                        (analytic[j] - fd).abs() < tol,
                        "{cfg} offset {offset} coord {j}: analytic {} vs finite difference {fd}",
                        analytic[j]
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod inference_tests {
    use super::testing::{simulate, Panel};
    use super::*;
    use crate::draws::Posterior;
    use crate::fit::fit;
    use crate::rng::BayesRng;
    use crate::types::FitStatus;

    /// Fit a simulated panel through the same entry point the SQL surface calls.
    fn fit_panel(panel: &Panel, cfg: &str) -> crate::fit::Fit {
        let frame = panel.frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        fit("hier_negbin", &Config::parse(cfg).unwrap(), &view).unwrap()
    }

    /// All draws of one global parameter.
    fn global(p: &Posterior, name: &str) -> Vec<f64> {
        let idx = p
            .params
            .iter()
            .position(|q| q.name == name && q.group_id == crate::types::GLOBAL_GROUP)
            .unwrap_or_else(|| panic!("no global parameter '{name}'"));
        (0..p.n_chains)
            .flat_map(|c| p.chain_values(c, idx).collect::<Vec<_>>())
            .collect()
    }

    /// All draws of one group's parameter.
    fn grouped(p: &Posterior, group: &str, name: &str) -> Vec<f64> {
        let idx = p
            .params
            .iter()
            .position(|q| q.name == name && q.group_id == group)
            .unwrap_or_else(|| panic!("no parameter '{name}' for group '{group}'"));
        (0..p.n_chains)
            .flat_map(|c| p.chain_values(c, idx).collect::<Vec<_>>())
            .collect()
    }

    fn quantile(sorted: &[f64], q: f64) -> f64 {
        sorted[((q * sorted.len() as f64) as usize).min(sorted.len() - 1)]
    }

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    fn median(v: &[f64]) -> f64 {
        quantile(&sorted(v.to_vec()), 0.5)
    }

    /// **The recovery test.** Slow because it is a real fit at the shipped defaults;
    /// run in release by `cargo test --release -- --ignored`.
    #[test]
    #[ignore]
    fn a_fit_recovers_the_process_it_was_given() {
        let mut rng = BayesRng::for_chain(11, 0);
        let panel = simulate(&mut rng, 30, 8, 2.0, 0.7, Some(3.0)).unwrap();
        let f = fit_panel(
            &panel,
            r#"{"y": "units", "group": "sku", "draws": 2000, "warmup": 1000,
                "chains": 4, "seed": 4}"#,
        );
        assert_eq!(
            f.posterior.meta.status,
            FitStatus::Converged,
            "{:?}",
            f.reasons
        );

        let intercept = median(&global(&f.posterior, "intercept"));
        let tau = median(&global(&f.posterior, "tau"));
        let phi = median(&global(&f.posterior, "phi"));
        assert!(
            (intercept - 2.0).abs() < 0.35,
            "intercept {intercept} against a true 2.0"
        );
        assert!((0.3..1.4).contains(&tau), "tau {tau} against a true 0.7");
        assert!((1.0..12.0).contains(&phi), "phi {phi} against a true 3.0");

        // The per-group rate is the number the agent reads, so it is checked per
        // group and not only on average.
        let mut covered = 0;
        for j in 0..panel.n_groups {
            let r = sorted(grouped(&f.posterior, &format!("SKU-{j:04}"), "rate"));
            if panel.true_rate[j] >= quantile(&r, 0.05) && panel.true_rate[j] <= quantile(&r, 0.95)
            {
                covered += 1;
            }
        }
        assert!(
            covered >= 24,
            "only {covered} of 30 per-SKU 90% intervals covered the truth"
        );
    }
}

#[cfg(test)]
mod parameterisation_tests {
    //! Why the non-centred parameterisation is not a preference.
    //!
    //! The BRD's premise is that a caller cannot select a bad parameterisation, which
    //! means this family has to pick one and be right. The comparison below is the
    //! evidence that it did: the *same* model, the *same* data, the *same* sampler,
    //! written the two ways, with the mixing measured rather than asserted.

    use super::testing::simulate;
    use super::*;
    use crate::diagnostics::{ess_bulk, rhat};
    use crate::engines::{Engine, NutsEngine, SampleOptions};
    use crate::rng::BayesRng;
    use crate::types::SampleFrom;

    /// The same likelihood and the same priors, written **centred**: the group level
    /// `u_j` is a coordinate and its prior is `N(0, tau^2)`, so the prior's width
    /// depends on a parameter the sampler is also moving.
    #[derive(Debug)]
    struct Centred<'a> {
        inner: &'a CompiledHierNegbin,
        params: Vec<ParamName>,
    }

    impl Centred<'_> {
        fn n_groups(&self) -> usize {
            self.inner.n_groups_inner()
        }
    }

    impl CompiledModel for Centred<'_> {
        fn param_names(&self) -> &[ParamName] {
            &self.params
        }
        fn n_obs(&self) -> usize {
            self.inner.y.len()
        }
        fn n_groups(&self) -> usize {
            self.inner.n_groups_inner()
        }
        fn data_fingerprint(&self) -> &str {
            self.inner.data_fingerprint()
        }
        fn readiness(&self) -> Readiness {
            Readiness::ready()
        }
        fn as_differentiable(&self) -> Option<&dyn LogPosterior> {
            Some(self)
        }
    }

    impl LogPosterior for Centred<'_> {
        fn dim(&self) -> usize {
            3 + self.n_groups()
        }

        fn logp(&self, theta: &[f64]) -> f64 {
            let (b0, log_tau, log_phi) = (theta[0], theta[1], theta[2]);
            if [b0, log_tau, log_phi]
                .iter()
                .any(|v| !v.is_finite() || v.abs() > LOG_BOX)
            {
                return f64::NEG_INFINITY;
            }
            let (tau, phi) = (log_tau.exp(), log_phi.exp());
            let mut total = 0.0;
            for i in 0..self.inner.y.len() {
                let eta = b0 + theta[3 + self.inner.group_of[i]];
                if !eta.is_finite() || eta > ETA_MAX {
                    return f64::NEG_INFINITY;
                }
                let (y, mu) = (self.inner.y[i], eta.exp());
                total += ln_gamma(y + phi) - ln_gamma(phi) - ln_gamma(y + 1.0) + phi * log_phi
                    - (y + phi) * (phi + mu).ln()
                    + y * eta;
            }
            for j in 0..self.n_groups() {
                let u = theta[3 + j];
                total += -log_tau - 0.5 * u * u / (tau * tau);
            }
            total + log_tau - log_phi
        }

        fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
            // Finite differences: this surface exists only to be sampled against, and
            // a second hand-derived gradient would be a second thing to get wrong.
            for j in 0..self.dim() {
                let h = 1e-6 * theta[j].abs().max(1.0);
                let (mut up, mut down) = (theta.to_vec(), theta.to_vec());
                up[j] += h;
                down[j] -= h;
                out[j] = (self.logp(&up) - self.logp(&down)) / (2.0 * h);
            }
            Ok(())
        }

        fn initial(&self) -> Vec<f64> {
            let start = self.inner.initial();
            let tau = start[self.inner.i_log_tau()].exp();
            let mut theta = vec![start[0], tau.ln(), 0.0];
            for j in 0..self.n_groups() {
                theta.push(tau * start[self.inner.i_z0() + j]);
            }
            theta
        }

        fn constrain(&self, theta: &[f64], out: &mut [f64]) {
            out[0] = theta[0];
            out[1] = theta[1].exp();
            out[2] = theta[2].exp();
            out[3..3 + self.n_groups()].copy_from_slice(&theta[3..3 + self.n_groups()]);
        }
    }

    fn mixing(model: &dyn CompiledModel, draws: usize) -> Vec<(String, f64, Option<f64>)> {
        let opts = SampleOptions {
            n_chains: 4,
            n_draws: draws,
            n_warmup: 1000,
            seed: 91,
            sample_from: SampleFrom::Posterior,
        };
        let sample = NutsEngine.sample(model, &opts).unwrap();
        let p = model.param_names().len();
        model
            .param_names()
            .iter()
            .enumerate()
            .take(3)
            .map(|(j, name)| {
                let chains: Vec<Vec<f64>> = (0..4)
                    .map(|c| {
                        (0..draws)
                            .map(|d| sample.values[((c * draws) + d) * p + j])
                            .collect()
                    })
                    .collect();
                (name.name.clone(), ess_bulk(&chains), rhat(&chains))
            })
            .collect()
    }

    /// **Non-centred beats centred on the same data, and the margin is the reason the
    /// choice is not a config slot.**
    ///
    /// The centred surface has the funnel: the prior on `u` is `N(0, tau^2)`, so the
    /// region the sampler must reach at small `tau` is a narrow neck that no single
    /// step size fits. The non-centred surface has `z ~ N(0, 1)` regardless of `tau`,
    /// so the neck is not there.
    #[test]
    #[ignore]
    fn the_non_centred_parameterisation_mixes_and_the_centred_one_does_not() {
        let mut rng = BayesRng::for_chain(3, 0);
        // Thin groups and a small pooling scale: the regime agent 01 is in, and the
        // regime the funnel is worst in.
        let panel = simulate(&mut rng, 40, 3, 1.5, 0.3, Some(4.0)).unwrap();
        let frame = panel.frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = build(
            &Config::parse(r#"{"y": "units", "group": "sku"}"#).unwrap(),
            &view,
        )
        .unwrap();

        let mut params = vec![
            ParamName::global("intercept").unwrap(),
            ParamName::global("tau").unwrap(),
            ParamName::global("phi").unwrap(),
        ];
        for j in 0..model.n_groups_inner() {
            params.push(ParamName::grouped(format!("SKU-{j:04}"), "u").unwrap());
        }
        let centred = Centred {
            inner: &model,
            params,
        };

        let nc = mixing(&model, 1000);
        let c = mixing(&centred, 1000);
        println!("non-centred: {nc:?}");
        println!("centred    : {c:?}");

        // `tau` is the coordinate the funnel bites: comparing on it is comparing on
        // the thing the parameterisation is chosen for.
        let nc_tau = nc.iter().find(|(n, ..)| n == "tau").unwrap();
        let c_tau = c.iter().find(|(n, ..)| n == "tau").unwrap();
        assert!(
            nc_tau.1 > 2.0 * c_tau.1,
            "non-centred ESS for tau {} should be far above centred {}",
            nc_tau.1,
            c_tau.1
        );
        assert!(
            nc_tau.2.unwrap() < 1.01,
            "non-centred R-hat for tau is {:?}",
            nc_tau.2
        );
    }
}

#[cfg(test)]
mod coverage_tests {
    //! **The tests this family exists to pass.**
    //!
    //! A safety-stock decision reads a *quantile*, not a mean. If the interval a
    //! reorder point is taken from is too tight, the part stocks out more often than
    //! the service level printed on the report — and nothing downstream can tell,
    //! because the point estimate is fine and every diagnostic is green. So coverage
    //! is measured against the generative truth, on the thin SKUs where it is hardest
    //! and where most of a C-parts catalogue lives.

    use super::testing::{draw_count, simulate};
    use super::*;
    use crate::draws::Posterior;
    use crate::fit::fit;
    use crate::rng::BayesRng;

    /// The demand a customer actually has: 40 SKUs, four periods each.
    const GROUPS: usize = 40;
    const PERIODS: usize = 4;
    const INTERCEPT: f64 = 1.0986; // ln 3 -- three units a period
    const TAU: f64 = 0.6;
    const PHI: f64 = 2.0;

    fn draws_of(p: &Posterior, group: &str, name: &str) -> Vec<f64> {
        let idx = p
            .params
            .iter()
            .position(|q| q.name == name && q.group_id == group)
            .unwrap();
        (0..p.n_chains)
            .flat_map(|c| p.chain_values(c, idx).collect::<Vec<_>>())
            .collect()
    }

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    fn q(sorted: &[f64], p: f64) -> f64 {
        sorted[((p * sorted.len() as f64) as usize).min(sorted.len() - 1)]
    }

    #[derive(Default, Debug)]
    struct Tally {
        two_sided: usize,
        service: usize,
        rate_ci: usize,
        n: usize,
        width: f64,
    }

    impl Tally {
        fn rate(&self, k: usize) -> f64 {
            k as f64 / self.n.max(1) as f64
        }
    }

    /// Fit `reps` simulated panels and tally what the agent would have read off each.
    fn measure(
        reps: u32,
        likelihood: &str,
        truth_phi: Option<f64>,
        tau: f64,
        seed_base: u64,
    ) -> Tally {
        let mut t = Tally::default();
        for rep in 0..reps {
            let mut rng = BayesRng::for_chain(seed_base, rep);
            let panel = simulate(&mut rng, GROUPS, PERIODS, INTERCEPT, tau, truth_phi).unwrap();
            let frame = panel.frame();
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let cfg = format!(
                r#"{{"y": "units", "group": "sku", "likelihood": "{likelihood}",
                     "draws": 1000, "warmup": 1000, "chains": 4, "seed": {}}}"#,
                1000 + rep
            );
            let f = fit("hier_negbin", &Config::parse(&cfg).unwrap(), &view).unwrap();
            let phi_draws = (likelihood == "negbinomial")
                .then(|| draws_of(&f.posterior, crate::types::GLOBAL_GROUP, "phi"));

            let mut prng = BayesRng::for_chain(seed_base ^ 0xabc, rep);
            for j in 0..GROUPS {
                let key = format!("SKU-{j:04}");
                let rates = draws_of(&f.posterior, &key, "rate");
                let pred: Vec<f64> = rates
                    .iter()
                    .enumerate()
                    .map(|(d, &r)| {
                        draw_count(&mut prng, r, phi_draws.as_ref().map(|p| p[d])).unwrap()
                    })
                    .collect();
                let pred = sorted(pred);
                let rates = sorted(rates);
                let (lo, hi) = (q(&pred, 0.05), q(&pred, 0.95));
                let holdout = panel.holdout[j];
                t.n += 1;
                t.width += hi - lo;
                if holdout >= lo && holdout <= hi {
                    t.two_sided += 1;
                }
                // The reorder point: stock to the 95th percentile of next period's
                // demand and the promise is a 95 % service level.
                if holdout <= q(&pred, 0.95) {
                    t.service += 1;
                }
                if panel.true_rate[j] >= q(&rates, 0.05) && panel.true_rate[j] <= q(&rates, 0.95) {
                    t.rate_ci += 1;
                }
            }
        }
        t
    }

    /// **The headline.** Nominal coverage on thin SKUs, from the native family.
    ///
    /// Too tight is the failure that matters commercially, so the assertions are
    /// one-sided in spirit: a lower bound on every coverage, and a loose upper bound
    /// only to catch an interval so wide it says nothing.
    #[test]
    #[ignore]
    fn thin_sku_intervals_have_their_nominal_coverage() {
        let t = measure(25, "negbinomial", Some(PHI), TAU, 5150);
        println!(
            "NATIVE thin-SKU over {} intervals: two-sided90 {:.4}  service95 {:.4}  \
             rate-CI90 {:.4}  mean width {:.2}",
            t.n,
            t.rate(t.two_sided),
            t.rate(t.service),
            t.rate(t.rate_ci),
            t.width / t.n as f64
        );
        // The parameter interval, where discreteness cannot flatter the number: this
        // is the one the bridge gets wrong (0.76 at this demand level, 0.42 at a
        // higher one -- see `the_bridge_cannot_cover_a_thin_sku`).
        assert!(
            (0.86..=0.96).contains(&t.rate(t.rate_ci)),
            "90% credible interval for a SKU's own rate covered {:.4}",
            t.rate(t.rate_ci)
        );
        // The promise a reorder point makes. Integer support means a discrete
        // interval covers a little more than nominal; under is the direction that
        // costs a stock-out.
        assert!(
            t.rate(t.service) >= 0.93,
            "a 95% reorder point achieved only {:.4}",
            t.rate(t.service)
        );
        assert!(
            t.rate(t.two_sided) >= 0.88,
            "the 90% predictive interval covered {:.4}",
            t.rate(t.two_sided)
        );
    }

    /// **Overdispersion is detected, and is not invented.**
    ///
    /// Both directions matter. A negative binomial that cannot see real overdispersion
    /// reports a reorder point that stocks out; one that invents overdispersion where
    /// there is none reports one that ties up working capital in a warehouse.
    #[test]
    #[ignore]
    fn overdispersion_is_detected_on_overdispersed_data() {
        let nb = measure(20, "negbinomial", Some(PHI), TAU, 777);
        let po = measure(20, "poisson", Some(PHI), TAU, 777);
        println!(
            "overdispersed data: NB service95 {:.4} width {:.2} | Poisson service95 {:.4} width {:.2}",
            nb.rate(nb.service),
            nb.width / nb.n as f64,
            po.rate(po.service),
            po.width / po.n as f64
        );
        assert!(
            nb.rate(nb.service) >= 0.93,
            "the negative binomial achieved only {:.4}",
            nb.rate(nb.service)
        );
        assert!(
            po.rate(po.service) < nb.rate(nb.service) - 0.03,
            "the Poisson likelihood should under-deliver on overdispersed data: \
             Poisson {:.4} against negative binomial {:.4}",
            po.rate(po.service),
            nb.rate(nb.service)
        );
    }

    /// The other direction: on data that really is Poisson, the negative binomial
    /// must not manufacture a dispersion parameter and pad the interval with it.
    #[test]
    #[ignore]
    fn no_spurious_overdispersion_on_poisson_data() {
        let nb = measure(20, "negbinomial", None, TAU, 424);
        let po = measure(20, "poisson", None, TAU, 424);
        let ratio = (nb.width / nb.n as f64) / (po.width / po.n as f64);
        println!(
            "Poisson data: NB service95 {:.4} width {:.2} | Poisson service95 {:.4} width {:.2} \
             (width ratio {ratio:.3})",
            nb.rate(nb.service),
            nb.width / nb.n as f64,
            po.rate(po.service),
            po.width / po.n as f64
        );
        assert!(
            ratio < 1.15,
            "the negative binomial widened the interval by {ratio:.3} on data with no \
             overdispersion in it"
        );
        assert!(
            nb.rate(nb.service) >= 0.93 && po.rate(po.service) >= 0.93,
            "NB {:.4}, Poisson {:.4}",
            nb.rate(nb.service),
            po.rate(po.service)
        );
    }

    /// The posterior for the dispersion, which is the object the bridge cannot
    /// produce at all: on Poisson data it must concentrate near no overdispersion.
    #[test]
    #[ignore]
    fn the_dispersion_posterior_points_at_the_poisson_limit_when_the_data_is_poisson() {
        let mut rng = BayesRng::for_chain(31, 0);
        let panel = simulate(&mut rng, 60, 8, INTERCEPT, TAU, None).unwrap();
        let frame = panel.frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "sku", "draws": 2000, "warmup": 1000,
                     "chains": 4, "seed": 8}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        // `alpha = 1/phi` is the overdispersion; Poisson is `alpha = 0`.
        let alpha = sorted(
            draws_of(&f.posterior, crate::types::GLOBAL_GROUP, "phi")
                .iter()
                .map(|p| 1.0 / p)
                .collect(),
        );
        println!(
            "alpha on Poisson data: median {:.4}, 95th percentile {:.4}",
            q(&alpha, 0.5),
            q(&alpha, 0.95)
        );
        assert!(
            q(&alpha, 0.05) < 0.05,
            "the posterior for the overdispersion excludes the Poisson limit: 5th \
             percentile {:.4}",
            q(&alpha, 0.05)
        );

        // ...and on genuinely overdispersed data it moves, decisively.
        let mut rng = BayesRng::for_chain(31, 1);
        let panel = simulate(&mut rng, 60, 8, INTERCEPT, TAU, Some(1.5)).unwrap();
        let frame = panel.frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "sku", "draws": 2000, "warmup": 1000,
                     "chains": 4, "seed": 8}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        let alpha = sorted(
            draws_of(&f.posterior, crate::types::GLOBAL_GROUP, "phi")
                .iter()
                .map(|p| 1.0 / p)
                .collect(),
        );
        println!(
            "alpha on overdispersed data: median {:.4}, 5th percentile {:.4}",
            q(&alpha, 0.5),
            q(&alpha, 0.05)
        );
        assert!(
            q(&alpha, 0.05) > 0.15,
            "the posterior for the overdispersion should exclude the Poisson limit \
             here: 5th percentile {:.4}",
            q(&alpha, 0.05)
        );
    }
}

#[cfg(test)]
mod bridge_comparison {
    //! **Why this family is native and not bridged.**
    //!
    //! `ROADMAP.md` deferred F1 on the expectation that a negative-binomial GLMM
    //! through [`crate::bridge`] would cover agent 01 adequately, with one caveat: the
    //! dispersion is estimated outside the IRLS loop, so it is not in the curvature.
    //! This module is the measurement that settled it. It builds the bridged path at
    //! its **best**, on the same generative process the native family is measured on
    //! in `coverage_tests`, and reports what the two produce.
    //!
    //! Three facts about `anofox-stats-core` at the pinned revision, each asserted
    //! below rather than quoted:
    //!
    //! 1. `GlmmFamily::from_name("negbinomial")` returns `theta = 1.0`. The dispersion
    //!    is an **input** to `fit_glmm`, fixed at a constant with no reference to the
    //!    data, and `GlmmResult` has no field that could carry a posterior for it.
    //! 2. `GlmmResult::var_group` — the pooling scale — is a Brent profile point
    //!    estimate. There is no standard error for it anywhere in the struct, so a
    //!    bridged posterior must condition on it too.
    //! 3. `fit_negbinomial` with `alpha: None` — the only data-driven dispersion
    //!    upstream offers — **does not converge on this data at all**.
    //!
    //! So a bridged F1 would have to condition on two point estimates, and the
    //! measured consequence is that the interval a reorder point is read from is far
    //! too tight exactly where the catalogue is thin. `docs/THEORY.md` records the
    //! numbers.

    use super::testing::{draw_count, simulate};
    use crate::rng::BayesRng;
    use anofox_stats_core::models::{fit_glmm, fit_negbinomial, GlmmFamily, GlmmOptions};
    use anofox_stats_core::types::NegBinomialOptions;

    /// Upstream's own moment update for the dispersion, lifted so that the plug-in
    /// below is upstream's estimator and not one of our choosing.
    fn theta_moments(y: &[f64], mu: &[f64]) -> f64 {
        let num: f64 = y
            .iter()
            .zip(mu)
            .map(|(&yi, &mui)| (yi - mui).powi(2) - mui)
            .sum();
        let den: f64 = mu.iter().map(|&m| m * m).sum();
        if den <= 0.0 || num <= 0.0 {
            return 1e6;
        }
        (den / num).clamp(1e-6, 1e6)
    }

    /// The best plug-in dispersion the bridge can construct: upstream's own update,
    /// damped and clamped so the alternation terminates where upstream's does not.
    fn plugin_theta(y: &[f64], x: &[Vec<f64>]) -> Option<f64> {
        let mut theta = 1.0_f64;
        for _ in 0..40 {
            let fit = fit_negbinomial(
                y,
                x,
                &NegBinomialOptions {
                    alpha: Some(theta),
                    max_iterations: 200,
                    // 1e-8 and 1e-4 both hit a limit cycle on count panels with many
                    // small counts; 1e-3 converges in four iterations.
                    tolerance: 1e-3,
                    ..Default::default()
                },
            )
            .ok()?;
            let b0 = fit.core.intercept.unwrap_or(0.0);
            let mu: Vec<f64> = (0..y.len())
                .map(|i| {
                    (b0 + fit
                        .core
                        .coefficients
                        .iter()
                        .zip(x)
                        .map(|(b, col)| b * col[i])
                        .sum::<f64>())
                    .exp()
                })
                .collect();
            let next = theta_moments(y, &mu).clamp(0.05, 1e4);
            let damped = (0.5 * theta.ln() + 0.5 * next.ln()).exp();
            if (damped - theta).abs() / theta < 1e-4 {
                return Some(damped);
            }
            theta = damped;
        }
        Some(theta)
    }

    /// A panel in the shape `fit_glmm` wants: dense group indices and at least one
    /// covariate column, which it requires.
    struct UpstreamPanel {
        y: Vec<f64>,
        x: Vec<Vec<f64>>,
        group: Vec<i32>,
        holdout: Vec<f64>,
        true_rate: Vec<f64>,
    }

    fn upstream_panel(
        rng: &mut BayesRng,
        g: usize,
        n_per: usize,
        intercept: f64,
        tau: f64,
        phi: f64,
    ) -> UpstreamPanel {
        let panel = simulate(rng, g, n_per, intercept, tau, Some(phi)).unwrap();
        let group: Vec<i32> = (0..g)
            .flat_map(|j| std::iter::repeat_n(j as i32, n_per))
            .collect();
        // A covariate with a true coefficient of zero. `fit_glmm` rejects an empty
        // design, and this probe is about the level, not about a slope.
        let x: Vec<f64> = (0..g * n_per)
            .map(|i| (i % n_per) as f64 - (n_per as f64 - 1.0) / 2.0)
            .collect();
        UpstreamPanel {
            y: panel.y,
            x: vec![x],
            group,
            holdout: panel.holdout,
            true_rate: panel.true_rate,
        }
    }

    #[test]
    #[ignore]
    fn the_bridge_cannot_cover_a_thin_sku() {
        // Fact 1: the dispersion is a constant, not an estimate.
        assert_eq!(
            GlmmFamily::from_name("negbinomial"),
            Some(GlmmFamily::NegativeBinomial { theta: 1.0 })
        );

        // Fact 3: upstream's own dispersion loop does not run on this data.
        let mut refused = 0;
        for rep in 0..20 {
            let mut rng = BayesRng::for_chain(7, rep);
            let p = upstream_panel(&mut rng, 40, 4, 1.0986, 0.6, 2.0);
            if fit_negbinomial(
                &p.y,
                &p.x,
                &NegBinomialOptions {
                    max_iterations: 500,
                    tolerance: 1e-6,
                    ..Default::default()
                },
            )
            .is_err()
            {
                refused += 1;
            }
        }
        println!("fit_negbinomial(alpha = None) refused {refused} of 20 thin-SKU panels");
        assert_eq!(
            refused, 20,
            "if upstream's dispersion loop now converges this measurement is stale"
        );

        // The coverage the bridge would deliver, at two demand levels, with the
        // dispersion both plugged in and handed to it for free.
        for (mean_demand, oracle) in [(3.0, false), (3.0, true), (25.0, false), (25.0, true)] {
            let intercept = f64::ln(mean_demand);
            let (mut rate_ci, mut service, mut n) = (0usize, 0usize, 0usize);
            let (mut taus, mut thetas) = (vec![], vec![]);
            for rep in 0..40 {
                let mut rng = BayesRng::for_chain(1234, rep);
                let p = upstream_panel(&mut rng, 40, 4, intercept, 0.6, 2.0);
                let (holdout, true_rate) = (&p.holdout, &p.true_rate);
                let Some(theta_hat) = (if oracle {
                    Some(2.0)
                } else {
                    plugin_theta(&p.y, &p.x)
                }) else {
                    continue;
                };
                thetas.push(theta_hat);
                let Ok(r) = fit_glmm(
                    &p.y,
                    &p.x,
                    &p.group,
                    &GlmmOptions {
                        family: GlmmFamily::NegativeBinomial { theta: theta_hat },
                        compute_inference: true,
                        ..Default::default()
                    },
                ) else {
                    continue;
                };
                // Fact 2: the pooling scale is a point, and this is where it is read.
                taus.push(r.var_group.sqrt());
                let b0 = r.intercept.unwrap();
                let b0_se = r.intercept_std_error.unwrap_or(0.0);

                let mut prng = BayesRng::for_chain(99, rep);
                for j in 0..40 {
                    let re = &r.ranef[j];
                    let mut rates = Vec::with_capacity(4000);
                    let mut pred = Vec::with_capacity(4000);
                    for _ in 0..4000 {
                        let rate = (b0
                            + b0_se * prng.standard_normal()
                            + re.value
                            + re.se * prng.standard_normal())
                        .exp();
                        rates.push(rate);
                        pred.push(draw_count(&mut prng, rate, Some(theta_hat)).unwrap());
                    }
                    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    pred.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    n += 1;
                    if true_rate[j] >= rates[200] && true_rate[j] <= rates[3800] {
                        rate_ci += 1;
                    }
                    if holdout[j] <= pred[3800] {
                        service += 1;
                    }
                }
            }
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
            println!(
                "BRIDGE mean-demand {mean_demand:>5} oracle-dispersion {oracle:<5}: \
                 rate-CI90 {:.4}  service95 {:.4}  (tau_hat {:.3} against a true 0.600, \
                 theta_hat {:.3} against a true 2.000) over {n} intervals",
                rate_ci as f64 / n.max(1) as f64,
                service as f64 / n.max(1) as f64,
                mean(&taus),
                mean(&thetas),
            );
            // The claim: the interval on a SKU's own rate is too tight. Nominal is
            // 0.90; the native family measures 0.90 on the same process.
            assert!(
                (rate_ci as f64 / n.max(1) as f64) < 0.86,
                "the bridged 90% interval covered {:.4}, which would make this \
                 comparison stale",
                rate_ci as f64 / n.max(1) as f64
            );
        }
    }
}

#[cfg(test)]
mod surface_tests {
    //! The fast half: identity, configuration, refusal and determinism. Everything
    //! here runs in the default `cargo test` loop.

    use super::testing::simulate;
    use super::*;
    use crate::data::testing::Frame;
    use crate::fit::fit;
    use crate::rng::BayesRng;
    use crate::types::{FitStatus, GLOBAL_GROUP};

    /// Every draw in a posterior, flattened.
    fn all_values(p: &crate::draws::Posterior) -> Vec<f64> {
        (0..p.n_chains)
            .flat_map(|c| {
                (0..p.n_params()).flat_map(move |j| p.chain_values(c, j).collect::<Vec<_>>())
            })
            .collect()
    }

    fn panel_frame(groups: usize, per: usize, seed: u64) -> (Vec<f64>, Frame) {
        let mut rng = BayesRng::for_chain(seed, 0);
        let panel = simulate(&mut rng, groups, per, 1.6, 0.5, Some(3.0)).unwrap();
        (panel.true_rate.clone(), panel.frame())
    }

    fn err(frame: &Frame, cfg: &str) -> String {
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        build(&Config::parse(cfg).unwrap(), &view)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn the_family_is_in_the_catalog_under_its_own_name_and_code() {
        let f = super::super::lookup("hier_negbin").unwrap();
        assert_eq!(f.id(), "hier_negbin");
        assert_eq!(f.code(), FamilyCode::HierNegbin);
        assert_eq!(f.code() as i32, 1);
        assert_eq!(f.default_engine(), EngineKind::Nuts);
    }

    #[test]
    fn the_parameter_list_is_the_order_the_draws_are_written_in() {
        let (_, frame) = panel_frame(3, 4, 1);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let m = build(
            &Config::parse(r#"{"y": "units", "group": "sku"}"#).unwrap(),
            &view,
        )
        .unwrap();
        let names: Vec<String> = m
            .param_names()
            .iter()
            .map(|p| format!("{}/{}", p.group_id, p.name))
            .collect();
        assert_eq!(
            names,
            vec![
                "__global__/intercept",
                "__global__/tau",
                "__global__/phi",
                "SKU-0000/u",
                "SKU-0000/rate",
                "SKU-0001/u",
                "SKU-0001/rate",
                "SKU-0002/u",
                "SKU-0002/rate",
            ]
        );
    }

    /// A Poisson fit reports no dispersion at all rather than a dispersion that
    /// happens to be large: the parameter does not exist in that model, and emitting
    /// a placeholder for it would invite an agent to read one.
    #[test]
    fn the_poisson_likelihood_reports_no_dispersion() {
        let (_, frame) = panel_frame(3, 4, 1);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let m = build(
            &Config::parse(r#"{"y": "units", "group": "sku", "likelihood": "poisson"}"#).unwrap(),
            &view,
        )
        .unwrap();
        assert!(!m.param_names().iter().any(|p| p.name == "phi"));
        assert_eq!(m.dim(), 2 + 3);
    }

    /// **A function of several parameters at once, checked draw by draw.**
    ///
    /// `rate` is not a coordinate: it is `exp(intercept + u)`, so it carries the joint
    /// behaviour of two parameters that a marginal check cannot see. Pinning the
    /// identity here is what lets the SQL surface treat `rate` as the SKU's demand
    /// rate rather than as a number that happens to be near one.
    #[test]
    fn rate_is_exactly_the_intercept_and_the_group_effect() {
        let (_, frame) = panel_frame(4, 5, 2);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "sku", "draws": 200, "warmup": 200,
                     "chains": 1, "seed": 3}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        let p = &f.posterior;
        let idx = |group: &str, name: &str| {
            p.params
                .iter()
                .position(|q| q.name == name && q.group_id == group)
                .unwrap()
        };
        let (i_int, i_u, i_rate) = (
            idx(GLOBAL_GROUP, "intercept"),
            idx("SKU-0002", "u"),
            idx("SKU-0002", "rate"),
        );
        for d in 0..200 {
            let expected = (p.value(0, d, i_int) + p.value(0, d, i_u)).exp();
            assert!(
                (p.value(0, d, i_rate) - expected).abs() < 1e-12,
                "draw {d}: rate {} against exp(intercept + u) {expected}",
                p.value(0, d, i_rate)
            );
        }
    }

    #[test]
    fn the_same_seed_produces_the_same_bytes() {
        let (_, frame) = panel_frame(4, 4, 5);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(
            r#"{"y": "units", "group": "sku", "draws": 60, "warmup": 60, "chains": 2, "seed": 7}"#,
        )
        .unwrap();
        let a = fit("hier_negbin", &cfg, &view).unwrap();
        let b = fit("hier_negbin", &cfg, &view).unwrap();
        assert_eq!(a.posterior.meta.model_id, b.posterior.meta.model_id);
        assert_eq!(all_values(&a.posterior), all_values(&b.posterior));
    }

    /// A short NUTS fit, in the default loop, to keep the whole path compiled and
    /// exercised. Convergence is not asserted here — that is
    /// `a_fit_recovers_the_process_it_was_given`, which runs in release.
    #[test]
    fn a_short_nuts_fit_produces_finite_draws_and_sampler_statistics() {
        let (_, frame) = panel_frame(6, 5, 9);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "sku", "draws": 120, "warmup": 200,
                     "chains": 2, "seed": 2}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        assert!(all_values(&f.posterior).iter().all(|v| v.is_finite()));
        assert!(f.posterior.n_divergent().is_some());
        assert_eq!(f.posterior.meta.n_groups, 6);
        assert_eq!(f.posterior.meta.n_obs, 30);
        assert_eq!(f.posterior.meta.family, FamilyCode::HierNegbin);
    }

    /// **Neither approximate engine is served, and both say so.**
    ///
    /// A family that quietly accepted an engine it cannot be represented by would
    /// report unearned confidence, which is the one thing the refusal path exists to
    /// prevent. See `build` for why the Laplace approximation is inadmissible here.
    #[test]
    fn the_approximate_engines_decline_rather_than_approximate() {
        let (_, frame) = panel_frame(5, 6, 4);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let e = fit(
            "hier_negbin",
            &Config::parse(r#"{"y": "units", "group": "sku", "engine": "laplace"}"#).unwrap(),
            &view,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("NUTS only"), "{e}");
        assert!(e.contains("pooling scale"), "{e}");

        let e = fit(
            "hier_negbin",
            &Config::parse(r#"{"y": "units", "group": "sku", "engine": "exact"}"#).unwrap(),
            &view,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("hier_negbin"), "{e}");
    }

    #[test]
    fn too_few_groups_refuses_rather_than_reporting_a_pooling_scale() {
        let (_, frame) = panel_frame(2, 8, 6);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "sku", "draws": 100, "warmup": 100,
                     "chains": 1, "seed": 1}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        assert_eq!(f.posterior.meta.status, FitStatus::InsufficientData);
        assert!(
            f.reasons.iter().any(|r| r.contains("min_groups")),
            "{:?}",
            f.reasons
        );
        assert_eq!(f.posterior.meta.n_groups_unready, 2);
        // The draws still exist: the verdict is that the data is weak, not that the
        // arithmetic broke, and an analyst may want to look at them.
        assert!(all_values(&f.posterior).iter().all(|v| v.is_finite()));
    }

    /// Every count zero: the level is identified only in the limit, so the fit
    /// refuses and every draw is NaN, which the SQL layer renders as NULL.
    #[test]
    fn a_panel_of_nothing_but_zeros_refuses_and_draws_null() {
        let frame = Frame::new(9)
            .numeric("units", vec![0.0; 9])
            .key("sku", vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "sku", "draws": 50, "warmup": 50,
                     "chains": 1, "seed": 1}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        assert_eq!(f.posterior.meta.status, FitStatus::Degenerate);
        assert!(all_values(&f.posterior).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn configuration_errors_name_their_slot() {
        let (_, frame) = panel_frame(3, 4, 1);

        assert!(err(&frame, r#"{"group": "sku"}"#).contains("y"));
        assert!(err(&frame, r#"{"y": "units"}"#).contains("group"));

        let e = err(&frame, r#"{"y": "units", "grp": "sku"}"#);
        assert!(e.contains("did you mean 'group'"), "{e}");

        let e = err(
            &frame,
            r#"{"y": "units", "group": "sku", "likelihood": "gaussian"}"#,
        );
        assert!(e.contains("likelihood") && e.contains("negbinomial"), "{e}");

        let e = err(
            &frame,
            r#"{"y": "units", "group": "sku", "prior": {"tau": {"sd": 1.0}}}"#,
        );
        assert!(e.contains("prior.tau.sd"), "{e}");

        // A prior mean with no spread is not a prior, and defaulting the spread would
        // silently invent one.
        let e = err(
            &frame,
            r#"{"y": "units", "group": "sku", "prior": {"phi": {"log_mean": 1.0}}}"#,
        );
        assert!(e.contains("prior.phi.log_sd"), "{e}");

        let e = err(
            &frame,
            r#"{"y": "units", "group": "sku", "prior": {"tau": {"scale": -1.0}}}"#,
        );
        assert!(e.contains("prior.tau.scale"), "{e}");
    }

    #[test]
    fn a_response_that_is_not_a_count_is_rejected_before_any_arithmetic() {
        let frame = Frame::new(6)
            .numeric("units", vec![1.0, 2.5, 3.0, 1.0, 0.0, 2.0])
            .key("sku", vec!["A", "A", "B", "B", "C", "C"]);
        let e = err(&frame, r#"{"y": "units", "group": "sku"}"#);
        assert!(e.contains("whole count"), "{e}");

        let frame = Frame::new(6)
            .numeric("units", vec![1.0, -2.0, 3.0, 1.0, 0.0, 2.0])
            .key("sku", vec!["A", "A", "B", "B", "C", "C"]);
        let e = err(&frame, r#"{"y": "units", "group": "sku"}"#);
        assert!(e.contains("non-negative"), "{e}");
    }

    #[test]
    fn an_exposure_of_zero_is_rejected_rather_than_producing_an_infinite_offset() {
        let frame = Frame::new(6)
            .numeric("units", vec![1.0, 2.0, 3.0, 1.0, 0.0, 2.0])
            .numeric("weeks", vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            .key("sku", vec!["A", "A", "B", "B", "C", "C"]);
        let e = err(
            &frame,
            r#"{"y": "units", "group": "sku", "exposure": "weeks"}"#,
        );
        assert!(e.contains("exposure"), "{e}");
    }

    /// The exposure is an **offset**, with coefficient one, not a covariate whose
    /// slope is estimated. Doubling every exposure and subtracting `ln 2` from the
    /// intercept must therefore leave the density exactly unchanged: the two moves
    /// cancel inside the linear predictor, which is the whole of what the likelihood
    /// reads.
    #[test]
    fn the_exposure_enters_as_an_offset() {
        let logp_at = |exposure: f64, intercept: f64| {
            let (_, base) = panel_frame(4, 5, 12);
            let n = base.n_rows;
            let frame = base.numeric("weeks", vec![exposure; n]);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let m = build(
                &Config::parse(r#"{"y": "units", "group": "sku", "exposure": "weeks"}"#).unwrap(),
                &view,
            )
            .unwrap();
            m.logp(&[intercept, -0.4, 0.2, 0.3, -0.6, 0.1, 0.8])
        };
        let a = logp_at(1.0, 0.5);
        let b = logp_at(2.0, 0.5 - f64::ln(2.0));
        assert!((a - b).abs() < 1e-9, "{a} against {b}");
        // ...and it is not merely being ignored.
        assert!((a - logp_at(2.0, 0.5)).abs() > 1.0);
    }
}

#[cfg(test)]
mod sql_fixture_check {
    //! The numbers `test/sql/f1_hier_negbin.test` pins, computed here.
    //!
    //! `make test` builds DuckDB, which takes the better part of an hour, so the SQL
    //! file's expected values are derived from the same core the extension links
    //! rather than by running the query and copying what came out. If this test and
    //! that file ever disagree, one of them has been edited without the other.

    use super::*;
    use crate::data::testing::Frame;
    use crate::fit::fit;

    /// The literal table in the `.test` file, as (part, units).
    fn issues() -> (Vec<f64>, Vec<String>) {
        let raw = include_str!("../../../../test/sql/f1_hier_negbin.test");
        let mut y = Vec::new();
        let mut part = Vec::new();
        for line in raw.lines() {
            let line = line.trim();
            if !line.starts_with("('") {
                continue;
            }
            for tuple in line.trim_end_matches(&[',', ';'][..]).split("),") {
                let t = tuple.trim().trim_start_matches('(').trim_end_matches(')');
                let mut it = t.split(',');
                let p = it.next().unwrap().trim().trim_matches('\'').to_string();
                let _week = it.next().unwrap();
                let units: f64 = it.next().unwrap().trim().parse().unwrap();
                part.push(p);
                y.push(units);
            }
        }
        (y, part)
    }

    #[test]
    #[ignore = "slow: the full four-chain fit the SQL scenario runs"]
    fn the_sql_scenario_values() {
        let (y, part) = issues();
        assert_eq!(y.len(), 280, "the fixture in the .test file has moved");
        let frame = Frame::new(y.len())
            .numeric("units", y.clone())
            .key("part", part.iter().map(String::as_str).collect());
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "part", "draws": 1000, "chains": 4, "seed": 42}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        let p = &f.posterior;
        println!("status {:?} rows {}", p.meta.status, p.n_rows());
        println!("reasons {:?}", f.reasons);
        println!("n_params {} n_groups {}", p.n_params(), p.meta.n_groups);

        let col = |group: &str, name: &str| -> Vec<f64> {
            let idx = p
                .params
                .iter()
                .position(|q| q.name == name && q.group_id == group)
                .unwrap();
            (0..p.n_chains)
                .flat_map(|c| p.chain_values(c, idx).collect::<Vec<_>>())
                .collect()
        };
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // DuckDB's median interpolates between the two middle order statistics.
            let n = v.len();
            if n.is_multiple_of(2) {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            } else {
                v[n / 2]
            }
        };
        for name in ["intercept", "tau", "phi"] {
            println!(
                "{name}: median {:.4}",
                med(col(crate::types::GLOBAL_GROUP, name))
            );
        }
        let phi = col(crate::types::GLOBAL_GROUP, "phi");

        let mut keys: Vec<String> = part.clone();
        keys.dedup();
        for key in &keys {
            let rate = col(key, "rate");
            let n_obs = part.iter().filter(|k| *k == key).count();
            let sample_mean: f64 = y
                .iter()
                .zip(&part)
                .filter(|(_, k)| *k == key)
                .map(|(v, _)| v)
                .sum::<f64>()
                / n_obs as f64;
            // The posterior predictive as a mixture of negative binomials, summed the
            // way the SQL does it.
            let mut cdf = 0.0;
            let mut reorder = [0i64; 3];
            let levels = [0.90, 0.95, 0.99];
            let mut done = [false; 3];
            let mut mass = 0.0;
            for k in 0..=200i64 {
                let kf = k as f64;
                let pmf: f64 = rate
                    .iter()
                    .zip(&phi)
                    .map(|(&r, &ph)| {
                        (ln_gamma(kf + ph) - ln_gamma(ph) - ln_gamma(kf + 1.0)
                            + ph * (ph / (ph + r)).ln()
                            + kf * (r / (ph + r)).ln())
                        .exp()
                    })
                    .sum::<f64>()
                    / rate.len() as f64;
                cdf += pmf;
                mass = cdf;
                for (i, lvl) in levels.iter().enumerate() {
                    if !done[i] && cdf >= *lvl {
                        reorder[i] = k;
                        done[i] = true;
                    }
                }
            }
            println!(
                "{key}\tweeks {n_obs}\tsample_mean {sample_mean:.4}\tposterior_rate {:.4}\t\
                 reorder 90/95/99 {reorder:?}\tmass {mass:.6}",
                med(rate)
            );
        }
    }

    /// The two refusals the SQL scenario pins, at the same budgets it uses.
    #[test]
    #[ignore = "slow: two more complete fits"]
    fn the_sql_scenario_refusals() {
        let (y, part) = issues();

        let keep: Vec<usize> = (0..y.len())
            .filter(|&i| part[i] == "BRG-100" || part[i] == "BRG-101")
            .collect();
        let two = Frame::new(keep.len())
            .numeric("units", keep.iter().map(|&i| y[i]).collect())
            .key("part", keep.iter().map(|&i| part[i].as_str()).collect());
        let refs = two.key_refs();
        let view = two.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "part", "draws": 200, "chains": 2, "seed": 1}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        println!("two parts: {:?}", f.posterior.meta.status);
        assert_eq!(
            f.posterior.meta.status,
            crate::types::FitStatus::InsufficientData
        );

        // Every count zero, five weeks each: the `never_issued` table.
        let mut zero_part = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        for p in &part {
            if !seen.contains(&p.as_str()) {
                seen.push(p);
            }
        }
        for p in &seen {
            for _ in 0..5 {
                zero_part.push(*p);
            }
        }
        let zeros = Frame::new(zero_part.len())
            .numeric("units", vec![0.0; zero_part.len()])
            .key("part", zero_part.clone());
        let refs = zeros.key_refs();
        let view = zeros.view(&refs);
        let f = fit(
            "hier_negbin",
            &Config::parse(
                r#"{"y": "units", "group": "part", "draws": 200, "chains": 2, "seed": 1}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        println!("all zeros: {:?}", f.posterior.meta.status);
        assert_eq!(f.posterior.meta.status, crate::types::FitStatus::Degenerate);
    }
}
