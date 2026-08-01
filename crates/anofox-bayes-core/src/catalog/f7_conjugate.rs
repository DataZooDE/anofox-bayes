//! F7 — conjugate anomaly detection.
//!
//! The simplest family in the catalog and the one that serves the freight-audit
//! agent. Each group (a lane, a carrier, a cost centre) gets its own closed-form
//! posterior over the level it operates at; "is this group anomalous?" is then a
//! question the agent asks of the draws in SQL, not a threshold baked into the model.
//!
//! Two likelihoods, both conjugate:
//!
//! **Normal.** `y ~ N(mu, sigma^2)` with a Normal-Inverse-Gamma prior on
//! `(mu, sigma^2)`. Posterior, for prior `(mu0, kappa0, alpha0, beta0)` and data with
//! `n` observations, mean `ybar` and centred sum of squares `SS`:
//!
//! ```text
//!   kappa_n = kappa0 + n
//!   mu_n    = (kappa0*mu0 + n*ybar) / kappa_n
//!   alpha_n = alpha0 + n/2
//!   beta_n  = beta0 + SS/2 + kappa0*n*(ybar - mu0)^2 / (2*kappa_n)
//! ```
//!
//! **Poisson.** `y ~ Poisson(lambda * exposure)` with a `Gamma(a0, rate b0)` prior on
//! the rate. Posterior is `Gamma(a0 + sum(y), rate = b0 + sum(exposure))`, which is
//! what makes "cost per shipment" and "claims per thousand consignments" the same
//! model with a different exposure column.
//!
//! **The default priors are the reference priors** — `kappa0 = 0, alpha0 = -1/2,
//! beta0 = 0` for the Normal, `a0 = 1/2, b0 = 0` for the Poisson. They are chosen
//! because they are *scale-free*: any concrete "weakly informative" default would
//! encode an assumption about whether costs are measured in cents or in millions, and
//! would quietly dominate the data for any customer whose units differed from the
//! author's. Under the reference prior the Normal posterior for `mu` is exactly the
//! Student-t centred on the sample mean with `n-1` degrees of freedom, which is both
//! the textbook answer and the one an auditor can check by hand.

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::rng::BayesRng;
use crate::types::{EngineKind, FamilyCode};

use super::{CompiledModel, ExactPosterior, LogPosterior, ModelFamily, Readiness};

/// The family singleton registered in the catalog.
#[derive(Debug)]
pub struct ConjugateAnomaly;

const SLOTS: &[&str] = &[
    "value",
    "group",
    "likelihood",
    "exposure",
    "prior",
    "draws",
    "chains",
    "max_draw_megabytes",
    "seed",
    "engine",
    "min_obs",
];

impl ModelFamily for ConjugateAnomaly {
    fn id(&self) -> &'static str {
        "conjugate_anomaly"
    }

    fn code(&self) -> FamilyCode {
        FamilyCode::ConjugateAnomaly
    }

    fn description(&self) -> &'static str {
        "Closed-form Normal or Poisson posteriors per group, for anomaly and \
         outlier questions answered as posterior tail probabilities."
    }

    fn default_engine(&self) -> EngineKind {
        // Conjugate: the posterior is available exactly, so approximating it would
        // add error for nothing.
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

        let value = cfg.require_str("value")?.to_string();
        let group = cfg.opt_str("group")?.map(str::to_string);
        let likelihood =
            Likelihood::parse(&cfg.one_of("likelihood", &["normal", "poisson"], "normal")?)?;
        let exposure = cfg.opt_str("exposure")?.map(str::to_string);
        let min_obs = cfg.usize_in("min_obs", likelihood.default_min_obs(), 1, 1_000_000)?;

        if exposure.is_some() && likelihood == Likelihood::Normal {
            return Err(BayesError::config(
                "exposure",
                "only applies to the Poisson likelihood; a Normal model has no exposure term",
            ));
        }

        let prior = Prior::parse(&cfg.nested("prior")?, likelihood)?;

        // Column resolution and null filtering happen before any arithmetic, so a
        // typo'd column name never reaches the mathematics.
        let mut numeric_cols = vec![value.as_str()];
        if let Some(e) = exposure.as_deref() {
            numeric_cols.push(e);
        }
        let key_cols: Vec<&str> = group.iter().map(String::as_str).collect();

        let rows = data.usable_rows(&numeric_cols, &key_cols)?;
        let fingerprint = data.fingerprint(&numeric_cols, &key_cols, &rows)?;
        let groups = data.group_rows(group.as_deref(), &rows)?;

        let values = data.numeric(&value)?;
        let exposures = exposure.as_deref().map(|e| data.numeric(e)).transpose()?;

        let mut posteriors = Vec::with_capacity(groups.len());
        let mut params = Vec::with_capacity(groups.len() * likelihood.param_names().len());

        // Only user-supplied keys are policed. An ungrouped fit uses GLOBAL_GROUP as
        // its single synthetic key, which is reserved by construction.
        if group.is_some() {
            for (key, _) in &groups {
                crate::types::validate_group_key(key)?;
            }
        }
        for (key, idx) in &groups {
            let ys: Vec<f64> = idx.iter().map(|&i| values.values[i]).collect();
            let es: Option<Vec<f64>> = exposures
                .as_ref()
                .map(|c| idx.iter().map(|&i| c.values[i]).collect());

            posteriors.push(GroupPosterior::fit(
                key.clone(),
                likelihood,
                &ys,
                es.as_deref(),
                &prior,
                min_obs,
            )?);
            for name in likelihood.param_names() {
                params.push(ParamName::grouped(key.clone(), *name)?);
            }
        }

        Ok(Box::new(CompiledConjugate::new(
            likelihood,
            posteriors,
            params,
            rows.len(),
            fingerprint,
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Likelihood {
    Normal,
    Poisson,
}

impl Likelihood {
    fn parse(s: &str) -> BayesResult<Self> {
        match s {
            "normal" => Ok(Likelihood::Normal),
            "poisson" => Ok(Likelihood::Poisson),
            other => Err(BayesError::config(
                "likelihood",
                format!("unknown: {other}"),
            )),
        }
    }

    fn param_names(&self) -> &'static [&'static str] {
        match self {
            // `sigma` rather than `sigma_squared`: a standard deviation is on the
            // scale of the data, so a credible interval for it is directly readable
            // by whoever has to act on it.
            Likelihood::Normal => &["mu", "sigma"],
            Likelihood::Poisson => &["lambda"],
        }
    }

    /// Below this many observations a group is reported as insufficient.
    ///
    /// Two for the Normal, because a single observation carries no information about
    /// spread at all and the reference posterior for `sigma` is improper. One for the
    /// Poisson, where a single count with a known exposure is genuinely informative
    /// about a rate.
    fn default_min_obs(&self) -> usize {
        match self {
            Likelihood::Normal => 2,
            Likelihood::Poisson => 1,
        }
    }
}

/// Prior hyperparameters, already validated.
#[derive(Debug, Clone, Copy)]
struct Prior {
    // Normal-Inverse-Gamma.
    mu0: f64,
    kappa0: f64,
    alpha0: f64,
    beta0: f64,
    // Gamma (rate parameterisation).
    a0: f64,
    b0: f64,
}

impl Prior {
    fn parse(cfg: &Config, likelihood: Likelihood) -> BayesResult<Self> {
        match likelihood {
            Likelihood::Normal => {
                cfg.reject_unknown(&["mu0", "kappa0", "alpha0", "beta0"])?;
                Ok(Prior {
                    mu0: cfg.f64_or("mu0", 0.0)?,
                    // Zero is the reference prior: no prior observations about the mean.
                    kappa0: cfg.non_negative_f64_or("kappa0", 0.0)?,
                    // Legitimately negative: the reference priors in use are -1
                    // (uniform on log sigma^2), -1/2 (Jeffreys, the default) and 0.
                    // Bounded below at -1 all the same, because a value beneath it is
                    // not a prior anyone holds -- it would simply subtract enough
                    // shape to make the posterior improper, and would then surface as
                    // "insufficient data", which is a confusing diagnosis for what is
                    // really a typo. Propriety is still re-checked per group, since a
                    // small group can fail it even for an admissible alpha0.
                    alpha0: {
                        let a = cfg.f64_or("alpha0", -0.5)?;
                        if a < -1.0 {
                            return Err(BayesError::config(
                                "prior.alpha0",
                                format!(
                                    "must be >= -1 (reference values are -1, -0.5 or 0), got {a}"
                                ),
                            ));
                        }
                        a
                    },
                    beta0: cfg.non_negative_f64_or("beta0", 0.0)?,
                    a0: 0.0,
                    b0: 0.0,
                })
            }
            Likelihood::Poisson => {
                cfg.reject_unknown(&["a0", "b0"])?;
                Ok(Prior {
                    mu0: 0.0,
                    kappa0: 0.0,
                    alpha0: 0.0,
                    beta0: 0.0,
                    a0: cfg.positive_f64_or("a0", 0.5)?,
                    b0: cfg.non_negative_f64_or("b0", 0.0)?,
                })
            }
        }
    }
}

/// The closed-form posterior for one group.
/// The closed-form posterior for one group.
///
/// The group's key is not stored: it already travels on every `ParamName` this family
/// emits, and a second copy would be a second thing to keep in step.
#[derive(Debug, Clone)]
struct GroupPosterior {
    kind: PosteriorKind,
    readiness: Readiness,
}

#[derive(Debug, Clone, Copy)]
enum PosteriorKind {
    /// Normal-Inverse-Gamma: `sigma^2 ~ InvGamma(alpha_n, beta_n)`,
    /// `mu | sigma^2 ~ N(mu_n, sigma^2 / kappa_n)`.
    Nig {
        mu_n: f64,
        kappa_n: f64,
        alpha_n: f64,
        beta_n: f64,
    },
    /// `lambda ~ Gamma(a_n, rate b_n)`.
    Gamma { a_n: f64, b_n: f64 },
}

impl GroupPosterior {
    fn fit(
        key: String,
        likelihood: Likelihood,
        ys: &[f64],
        exposure: Option<&[f64]>,
        prior: &Prior,
        min_obs: usize,
    ) -> BayesResult<Self> {
        // Dispatch on the likelihood the caller asked for, never on whether some
        // prior slot happens to be set. An inferred likelihood is a likelihood that
        // changes when an unrelated default moves.
        match likelihood {
            Likelihood::Normal => Self::fit_normal(key, ys, prior, min_obs),
            Likelihood::Poisson => Self::fit_poisson(key, ys, exposure, prior, min_obs),
        }
    }

    fn fit_normal(key: String, ys: &[f64], prior: &Prior, min_obs: usize) -> BayesResult<Self> {
        let n = ys.len();
        let kappa_n = prior.kappa0 + n as f64;
        let alpha_n = prior.alpha0 + n as f64 / 2.0;

        if n == 0 {
            return Ok(Self::unfittable(format!("group '{key}': no observations")));
        }

        let ybar = ys.iter().sum::<f64>() / n as f64;
        let ss: f64 = ys.iter().map(|y| (y - ybar).powi(2)).sum();
        let mu_n = (prior.kappa0 * prior.mu0 + n as f64 * ybar) / kappa_n;
        let beta_n = prior.beta0
            + ss / 2.0
            + prior.kappa0 * n as f64 * (ybar - prior.mu0).powi(2) / (2.0 * kappa_n);

        // An improper posterior cannot be sampled from. This is the structural half
        // of the refusal path: no amount of sampling would fix it.
        if alpha_n <= 0.0 || beta_n <= 0.0 || kappa_n <= 0.0 {
            let reason = if beta_n <= 0.0 && ss == 0.0 && n >= 2 {
                format!("group '{key}': all {n} observations are identical, so the variance is not estimable")
            } else {
                format!(
                    "group '{key}': {n} observations are too few to identify a mean and a variance"
                )
            };
            return Ok(Self::unfittable(reason));
        }

        let readiness = if n < min_obs {
            Readiness::insufficient(format!(
                "group '{key}': {n} observations is below the min_obs threshold of {min_obs}"
            ))
        } else {
            Readiness::ready()
        };

        Ok(Self {
            kind: PosteriorKind::Nig {
                mu_n,
                kappa_n,
                alpha_n,
                beta_n,
            },
            readiness,
        })
    }

    fn fit_poisson(
        key: String,
        ys: &[f64],
        exposure: Option<&[f64]>,
        prior: &Prior,
        min_obs: usize,
    ) -> BayesResult<Self> {
        let n = ys.len();
        for (i, y) in ys.iter().enumerate() {
            if *y < 0.0 || y.fract() != 0.0 {
                return Err(BayesError::config(
                    "value",
                    format!("a Poisson likelihood needs non-negative whole counts; row {i} of group '{key}' is {y}"),
                ));
            }
        }
        let total_exposure = match exposure {
            None => n as f64,
            Some(es) => {
                for (i, e) in es.iter().enumerate() {
                    if *e < 0.0 {
                        return Err(BayesError::config(
                            "exposure",
                            format!("must be non-negative; row {i} of group '{key}' is {e}"),
                        ));
                    }
                }
                es.iter().sum()
            }
        };

        let a_n = prior.a0 + ys.iter().sum::<f64>();
        let b_n = prior.b0 + total_exposure;

        if a_n <= 0.0 || b_n <= 0.0 {
            return Ok(Self::unfittable(format!(
                "group '{key}': total exposure is zero, so no rate is identifiable"
            )));
        }

        let readiness = if n < min_obs {
            Readiness::insufficient(format!(
                "group '{key}': {n} observations is below the min_obs threshold of {min_obs}"
            ))
        } else {
            Readiness::ready()
        };

        Ok(Self {
            kind: PosteriorKind::Gamma { a_n, b_n },
            readiness,
        })
    }

    /// A group whose posterior does not exist.
    ///
    /// Its draws come out as NaN, which the SQL layer renders as NULL. That is the
    /// only honest answer: a number here would be indistinguishable from an estimate,
    /// and the whole point of the refusal path is that an agent can tell the two
    /// apart.
    fn unfittable(reason: String) -> Self {
        Self {
            kind: PosteriorKind::Nig {
                mu_n: f64::NAN,
                kappa_n: f64::NAN,
                alpha_n: f64::NAN,
                beta_n: f64::NAN,
            },
            readiness: Readiness::insufficient(reason),
        }
    }

    /// Whether this group's posterior exists as a distribution at all.
    ///
    /// The single source of truth for that question: the exact sampler uses it to
    /// decide whether to emit NaN, and the differentiable path uses it to decide
    /// whether the group gets coordinates. Two separate answers would eventually
    /// disagree, and the disagreement would be a group that draws numbers under one
    /// engine and NULL under the other.
    fn is_proper(&self) -> bool {
        match self.kind {
            PosteriorKind::Nig {
                mu_n,
                kappa_n,
                alpha_n,
                beta_n,
            } => mu_n.is_finite() && kappa_n > 0.0 && alpha_n > 0.0 && beta_n > 0.0,
            PosteriorKind::Gamma { a_n, b_n } => a_n > 0.0 && b_n > 0.0,
        }
    }

    fn sample_into(&self, rng: &mut BayesRng, out: &mut [f64]) -> BayesResult<()> {
        if !self.is_proper() {
            out.fill(f64::NAN);
            return Ok(());
        }
        match self.kind {
            PosteriorKind::Nig {
                mu_n,
                kappa_n,
                alpha_n,
                beta_n,
            } => {
                // sigma^2 ~ InvGamma(alpha_n, beta_n) is 1 / Gamma(alpha_n, rate beta_n).
                let precision = rng.gamma(alpha_n, beta_n)?;
                let sigma_sq = 1.0 / precision;
                out[1] = sigma_sq.sqrt();
                out[0] = mu_n + (sigma_sq / kappa_n).sqrt() * rng.standard_normal();
                Ok(())
            }
            PosteriorKind::Gamma { a_n, b_n } => {
                out[0] = rng.gamma(a_n, b_n)?;
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
struct CompiledConjugate {
    likelihood: Likelihood,
    posteriors: Vec<GroupPosterior>,
    params: Vec<ParamName>,
    n_obs: usize,
    fingerprint: String,
    /// Where each group's coordinates start in the unconstrained vector, or `None` for
    /// a group whose posterior does not exist.
    ///
    /// An unfittable group must not occupy a coordinate. Its log density would be
    /// `NaN`, which propagates through the mode search and the Hessian and takes every
    /// *other* group down with it — one single-observation lane would turn a 5 000-lane
    /// fit into a Cholesky failure. Excluding it instead gives the same answer the
    /// exact engine gives: NULL for that group, real numbers for the rest.
    theta_offsets: Vec<Option<usize>>,
    dim: usize,
}

impl CompiledConjugate {
    fn new(
        likelihood: Likelihood,
        posteriors: Vec<GroupPosterior>,
        params: Vec<ParamName>,
        n_obs: usize,
        fingerprint: String,
    ) -> Self {
        let width = likelihood.param_names().len();
        let mut dim = 0usize;
        let theta_offsets = posteriors
            .iter()
            .map(|p| {
                if p.is_proper() {
                    let at = dim;
                    dim += width;
                    Some(at)
                } else {
                    None
                }
            })
            .collect();
        Self {
            likelihood,
            posteriors,
            params,
            n_obs,
            fingerprint,
            theta_offsets,
            dim,
        }
    }

    /// Iterate the groups that carry coordinates, paired with their offset.
    fn coordinated(&self) -> impl Iterator<Item = (&GroupPosterior, usize)> {
        self.posteriors
            .iter()
            .zip(&self.theta_offsets)
            .filter_map(|(p, off)| off.map(|o| (p, o)))
    }
}

impl CompiledModel for CompiledConjugate {
    fn param_names(&self) -> &[ParamName] {
        &self.params
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn n_groups(&self) -> usize {
        self.posteriors.len()
    }

    fn data_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn readiness(&self) -> Readiness {
        if self.posteriors.is_empty() {
            return Readiness::insufficient("no groups survived null filtering");
        }
        Readiness::worst(self.posteriors.iter().map(|p| p.readiness.clone()))
    }

    /// This family fits each group independently, so unlike the pooled families it
    /// knows exactly which groups are the problem and can count them.
    ///
    /// An empty catalog of groups reports zero rather than one: `readiness` calls
    /// that case insufficient, but there is no group it can be blamed on, and
    /// `__n_groups__` is zero too. A count larger than the population it counts would
    /// be worse than no count.
    fn n_groups_unready(&self) -> usize {
        self.posteriors
            .iter()
            .filter(|p| !p.readiness.status.is_actionable())
            .count()
    }

    fn as_exact(&self) -> Option<&dyn ExactPosterior> {
        Some(self)
    }

    fn as_differentiable(&self) -> Option<&dyn LogPosterior> {
        Some(self)
    }
}

/// The same posterior, written as a differentiable log density on an unconstrained
/// scale, so the gradient-based engines can consume it.
///
/// This family is conjugate, so nothing here is *needed* to fit it — the exact engine
/// already samples the closed form. It exists because it makes the two engines two
/// independent derivations of one distribution, which is the strongest correctness
/// gate this repository has. Per `AGENTS.md`, that check already existed for
/// `pooled_gaussian` and not for this family.
///
/// **Groups are independent**, so the joint log density is the sum of per-group terms
/// and the Hessian is block diagonal. The Laplace engine nevertheless assembles it
/// densely, at `O(dim^2)` memory and `O(dim^3)` arithmetic: this path is a correctness
/// gate and a small-model convenience, not the way to fit ten thousand lanes. The
/// exact engine remains the family's default for exactly that reason.
///
/// **Normal.** Coordinates `(mu, u)` with `sigma = e^u`. Substituting `sigma^2 = e^{2u}`
/// into the Normal-Inverse-Gamma posterior and adding the log-Jacobian
/// `log|d sigma^2 / du| = log 2 + 2u` collapses to
///
/// ```text
///   log p(mu, u) = -(2 alpha_n + 1) u - e^{-2u} (beta_n + kappa_n (mu - mu_n)^2 / 2)
/// ```
///
/// up to a constant. The `+2u` is the whole Jacobian: drop it and the coefficient of
/// `u` becomes `-(2 alpha_n + 3)`, which is a *different, tighter* posterior for sigma
/// that a gradient-versus-finite-difference test would happily certify, because both
/// sides difference the same wrong density.
///
/// **Poisson.** Coordinate `v` with `lambda = e^v`. The Gamma density
/// `(a_n - 1) log lambda - b_n lambda` plus the log-Jacobian `log|d lambda / dv| = v`
/// gives
///
/// ```text
///   log p(v) = a_n v - b_n e^v
/// ```
///
/// whose mode is `lambda = a_n / b_n`, the posterior *mean* rather than the posterior
/// mode of the Gamma — which is the visible signature that the Jacobian is present.
impl LogPosterior for CompiledConjugate {
    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(&self, theta: &[f64]) -> f64 {
        let mut total = 0.0;
        for (posterior, off) in self.coordinated() {
            total += match posterior.kind {
                PosteriorKind::Nig {
                    mu_n,
                    kappa_n,
                    alpha_n,
                    beta_n,
                } => {
                    let (mu, u) = (theta[off], theta[off + 1]);
                    let d = mu - mu_n;
                    -(2.0 * alpha_n + 1.0) * u - (-2.0 * u).exp() * (beta_n + kappa_n * d * d / 2.0)
                }
                PosteriorKind::Gamma { a_n, b_n } => {
                    let v = theta[off];
                    a_n * v - b_n * v.exp()
                }
            };
        }
        total
    }

    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        if theta.len() != self.dim || out.len() != self.dim {
            return Err(BayesError::DimensionMismatch(format!(
                "expected {} coordinates, got theta {} and out {}",
                self.dim,
                theta.len(),
                out.len()
            )));
        }
        for (posterior, off) in self.coordinated() {
            match posterior.kind {
                PosteriorKind::Nig {
                    mu_n,
                    kappa_n,
                    alpha_n,
                    beta_n,
                } => {
                    let (mu, u) = (theta[off], theta[off + 1]);
                    let d = mu - mu_n;
                    let inv_var = (-2.0 * u).exp();
                    out[off] = -inv_var * kappa_n * d;
                    out[off + 1] =
                        -(2.0 * alpha_n + 1.0) + 2.0 * inv_var * (beta_n + kappa_n * d * d / 2.0);
                }
                PosteriorKind::Gamma { a_n, b_n } => {
                    out[off] = a_n - b_n * theta[off].exp();
                }
            }
        }
        Ok(())
    }

    fn initial(&self) -> Vec<f64> {
        let mut theta = vec![0.0; self.dim];
        for (posterior, off) in self.coordinated() {
            match posterior.kind {
                PosteriorKind::Nig {
                    mu_n,
                    alpha_n,
                    beta_n,
                    ..
                } => {
                    theta[off] = mu_n;
                    // The joint mode of `u`, from setting the gradient above to zero.
                    let sigma_sq = (2.0 * beta_n / (2.0 * alpha_n + 1.0)).max(f64::MIN_POSITIVE);
                    theta[off + 1] = 0.5 * sigma_sq.ln();
                }
                PosteriorKind::Gamma { a_n, b_n } => theta[off] = (a_n / b_n).ln(),
            }
        }
        theta
    }

    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        let width = self.likelihood.param_names().len();
        for (i, off) in self.theta_offsets.iter().enumerate() {
            let slot = &mut out[i * width..(i + 1) * width];
            match off {
                // Same answer the exact engine gives for a group whose posterior does
                // not exist: NaN, which the SQL layer renders as NULL.
                None => slot.fill(f64::NAN),
                Some(off) => match self.likelihood {
                    Likelihood::Normal => {
                        slot[0] = theta[*off];
                        slot[1] = theta[*off + 1].exp();
                    }
                    Likelihood::Poisson => slot[0] = theta[*off].exp(),
                },
            }
        }
    }
}

impl ExactPosterior for CompiledConjugate {
    fn sample_into(&self, rng: &mut BayesRng, out: &mut [f64]) -> BayesResult<()> {
        let width = self.likelihood.param_names().len();
        if out.len() != self.posteriors.len() * width {
            return Err(BayesError::DimensionMismatch(format!(
                "expected {} slots, got {}",
                self.posteriors.len() * width,
                out.len()
            )));
        }
        for (i, posterior) in self.posteriors.iter().enumerate() {
            posterior.sample_into(rng, &mut out[i * width..(i + 1) * width])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::testing::Frame;
    use crate::types::FitStatus;

    fn compile<'a>(cfg: &str, data: &'a DataView<'a>) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        ConjugateAnomaly.compile(&Config::parse(cfg).unwrap(), data)
    }

    /// Draw `n` samples of every parameter, returning them per parameter index.
    fn draw(model: &dyn CompiledModel, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let exact = model.as_exact().expect("family is conjugate");
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

    fn sd(xs: &[f64]) -> f64 {
        let m = mean(xs);
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
    }

    //=== Normal ===========================================================//

    /// Under the reference prior the posterior for `mu` is exactly
    /// `t_{n-1}(ybar, s^2/n)`. Its mean is `ybar` and its standard deviation is
    /// `s/sqrt(n) * sqrt((n-1)/(n-3))`. This is the textbook answer, and it is what
    /// an auditor would compute by hand.
    #[test]
    fn the_reference_posterior_for_the_mean_is_the_textbook_student_t() {
        let ys: Vec<f64> = vec![
            10.2, 9.8, 10.5, 10.1, 9.9, 10.3, 10.0, 9.7, 10.4, 10.6, 9.6, 10.15,
        ];
        let n = ys.len() as f64;
        let ybar = ys.iter().sum::<f64>() / n;
        let s2 = ys.iter().map(|y| (y - ybar).powi(2)).sum::<f64>() / (n - 1.0);

        let frame = Frame::new(ys.len()).numeric("cost", ys);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();

        let cols = draw(&*model, 200_000, 7);
        let mu = &cols[0];

        assert!(
            (mean(mu) - ybar).abs() < 0.005,
            "posterior mean {} vs ybar {ybar}",
            mean(mu)
        );
        let expected_sd = (s2 / n).sqrt() * ((n - 1.0) / (n - 3.0)).sqrt();
        assert!(
            (sd(mu) - expected_sd).abs() < 0.01 * expected_sd.max(1.0) + 0.005,
            "posterior sd {} vs Student-t sd {expected_sd}",
            sd(mu)
        );
    }

    /// `sigma^2 | y ~ InvGamma((n-1)/2, SS/2)` under the reference prior, whose mean
    /// is `SS / (n - 3)`.
    #[test]
    fn the_reference_posterior_for_the_variance_matches_the_closed_form() {
        let ys: Vec<f64> = (0..30).map(|i| 5.0 + (i as f64 % 7.0) - 3.0).collect();
        let n = ys.len() as f64;
        let ybar = ys.iter().sum::<f64>() / n;
        let ss: f64 = ys.iter().map(|y| (y - ybar).powi(2)).sum();

        let frame = Frame::new(ys.len()).numeric("cost", ys);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();

        let cols = draw(&*model, 200_000, 11);
        let var: Vec<f64> = cols[1].iter().map(|s| s * s).collect();
        let expected = ss / (n - 3.0); // mean of InvGamma(a, b) is b / (a - 1)
        assert!(
            (mean(&var) - expected).abs() < 0.02 * expected,
            "posterior mean variance {} vs closed form {expected}",
            mean(&var)
        );
    }

    /// An informative prior pulls the posterior toward `mu0`, and pulls harder the
    /// less data there is. This is the shrinkage that makes a thin lane borrow
    /// strength rather than report noise.
    #[test]
    fn an_informative_prior_shrinks_the_posterior_toward_it_and_more_so_on_thin_data() {
        let posterior_mean_with = |n: usize| {
            let ys: Vec<f64> = (0..n)
                .map(|i| 10.0 + ((i % 3) as f64 - 1.0) * 0.1)
                .collect();
            let frame = Frame::new(n).numeric("cost", ys);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let model = compile(
                r#"{"value": "cost", "prior": {"mu0": 0.0, "kappa0": 20.0, "alpha0": 2.0, "beta0": 2.0}}"#,
                &view,
            )
            .unwrap();
            mean(&draw(&*model, 20_000, 3)[0])
        };

        let thin = posterior_mean_with(5);
        let thick = posterior_mean_with(200);

        // Both are pulled below the sample mean of ~10 toward mu0 = 0...
        assert!(thin < 10.0 && thick < 10.0);
        // ...and the thin one is pulled much further.
        assert!(
            thin < thick,
            "thin {thin} should shrink further than thick {thick}"
        );
        // With 200 observations against kappa0 = 20 the prior is nearly overwhelmed.
        assert!(thick > 8.5, "thick {thick} should be close to the data");
    }

    //=== Poisson ==========================================================//

    /// `lambda | y ~ Gamma(a0 + sum y, rate = b0 + sum exposure)`, whose mean is
    /// `(a0 + sum y) / (b0 + sum e)` -- the exposure-weighted rate.
    #[test]
    fn the_poisson_posterior_matches_the_gamma_closed_form() {
        let counts = vec![3.0, 7.0, 5.0, 4.0, 6.0, 8.0, 2.0, 5.0];
        let exposure = vec![10.0, 20.0, 15.0, 12.0, 18.0, 25.0, 8.0, 14.0];
        let total_y: f64 = counts.iter().sum();
        let total_e: f64 = exposure.iter().sum();

        let frame = Frame::new(8)
            .numeric("claims", counts)
            .numeric("consignments", exposure);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(
            r#"{"value": "claims", "likelihood": "poisson", "exposure": "consignments"}"#,
            &view,
        )
        .unwrap();

        let cols = draw(&*model, 200_000, 13);
        let expected_mean = (0.5 + total_y) / total_e;
        let expected_sd = ((0.5 + total_y).sqrt()) / total_e;
        assert!(
            (mean(&cols[0]) - expected_mean).abs() < 0.01 * expected_mean,
            "posterior mean {} vs closed form {expected_mean}",
            mean(&cols[0])
        );
        assert!(
            (sd(&cols[0]) - expected_sd).abs() < 0.02 * expected_sd,
            "posterior sd {} vs closed form {expected_sd}",
            sd(&cols[0])
        );
    }

    /// Without an exposure column every observation counts as one unit of exposure,
    /// so the posterior rate is the mean count.
    #[test]
    fn poisson_without_exposure_estimates_the_mean_count() {
        let counts = vec![4.0, 6.0, 5.0, 5.0, 4.0, 6.0];
        let frame = Frame::new(6).numeric("claims", counts.clone());
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "claims", "likelihood": "poisson"}"#, &view).unwrap();

        let cols = draw(&*model, 100_000, 17);
        let expected = (0.5 + counts.iter().sum::<f64>()) / counts.len() as f64;
        assert!((mean(&cols[0]) - expected).abs() < 0.01 * expected);
    }

    #[test]
    fn a_poisson_likelihood_rejects_data_that_is_not_counts() {
        let frame = Frame::new(3).numeric("claims", vec![1.0, 2.5, 3.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"value": "claims", "likelihood": "poisson"}"#, &view).unwrap_err();
        assert!(err.to_string().contains("whole counts"), "{err}");

        let frame = Frame::new(3).numeric("claims", vec![1.0, -2.0, 3.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        assert!(compile(r#"{"value": "claims", "likelihood": "poisson"}"#, &view).is_err());
    }

    //=== Grouping =========================================================//

    #[test]
    fn each_group_gets_its_own_posterior_named_after_it() {
        let frame = Frame::new(8)
            .numeric("cost", vec![1.0, 1.1, 0.9, 1.05, 5.0, 5.2, 4.8, 5.1])
            .key("lane", vec!["A", "A", "A", "A", "B", "B", "B", "B"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost", "group": "lane"}"#, &view).unwrap();

        assert_eq!(model.n_groups(), 2);
        assert_eq!(model.n_obs(), 8);
        let names = model.param_names();
        assert_eq!(names.len(), 4);
        assert_eq!(
            (names[0].group_id.as_str(), names[0].name.as_str()),
            ("A", "mu")
        );
        assert_eq!(
            (names[1].group_id.as_str(), names[1].name.as_str()),
            ("A", "sigma")
        );
        assert_eq!(
            (names[2].group_id.as_str(), names[2].name.as_str()),
            ("B", "mu")
        );

        let cols = draw(&*model, 20_000, 19);
        assert!(
            (mean(&cols[0]) - 1.0125).abs() < 0.05,
            "lane A mu {}",
            mean(&cols[0])
        );
        assert!(
            (mean(&cols[2]) - 5.025).abs() < 0.1,
            "lane B mu {}",
            mean(&cols[2])
        );
    }

    //=== Refusal ==========================================================//

    /// A lane with one invoice has no estimable spread no matter how long a sampler
    /// runs on it. Saying so before sampling is both cheaper and clearer.
    #[test]
    fn a_group_with_a_single_observation_refuses_rather_than_inventing_a_variance() {
        let frame = Frame::new(5)
            .numeric("cost", vec![1.0, 1.1, 0.9, 1.05, 42.0])
            .key("lane", vec!["A", "A", "A", "A", "SOLO"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost", "group": "lane"}"#, &view).unwrap();

        let verdict = model.readiness();
        assert_eq!(verdict.status, FitStatus::InsufficientData);
        assert!(
            verdict.reasons.iter().any(|r| r.contains("SOLO")),
            "{:?}",
            verdict.reasons
        );

        // The unfittable group's draws are NULL-shaped, never a plausible number...
        let cols = draw(&*model, 100, 23);
        assert!(
            cols[2].iter().all(|v| v.is_nan()),
            "SOLO mu must not be a number"
        );
        // ...while the healthy group is unaffected.
        assert!(cols[0].iter().all(|v| v.is_finite()));
    }

    /// All-identical observations carry no information about spread either, and the
    /// reference posterior for the variance is a point mass at zero.
    #[test]
    fn a_group_whose_observations_are_all_identical_refuses() {
        let frame = Frame::new(6).numeric("cost", vec![7.0; 6]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();

        let verdict = model.readiness();
        assert_ne!(verdict.status, FitStatus::Converged);
        assert!(
            verdict.reasons[0].contains("identical"),
            "{:?}",
            verdict.reasons
        );
    }

    #[test]
    fn a_group_below_the_min_obs_threshold_is_reported_as_insufficient() {
        let frame = Frame::new(4)
            .numeric("cost", vec![1.0, 1.1, 5.0, 5.2])
            .key("lane", vec!["A", "A", "THIN", "THIN"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        // With the default threshold of 2, both lanes are fine.
        let model = compile(r#"{"value": "cost", "group": "lane"}"#, &view).unwrap();
        assert_eq!(model.readiness().status, FitStatus::Converged);

        // Raising the bar to 3 makes both insufficient -- and the draws stay real
        // numbers, because the posterior exists; it is the *evidence* that is thin.
        let model = compile(r#"{"value": "cost", "group": "lane", "min_obs": 3}"#, &view).unwrap();
        assert_eq!(model.readiness().status, FitStatus::InsufficientData);
        let cols = draw(&*model, 50, 29);
        assert!(cols[0].iter().all(|v| v.is_finite()));
    }

    /// The count this family reports beside the collapsed verdict. Because it fits
    /// each group independently it knows exactly which groups are thin, so the number
    /// is exact rather than the conservative whole-model default.
    #[test]
    fn the_unready_groups_are_counted_exactly_while_the_verdict_stays_collapsed() {
        // Four lanes: two with four invoices, two with two.
        let frame = Frame::new(12)
            .numeric(
                "cost",
                vec![
                    1.0, 1.1, 0.9, 1.05, // A
                    2.0, 2.1, 1.9, 2.05, // B
                    5.0, 5.2, // THIN-1
                    7.0, 7.3, // THIN-2
                ],
            )
            .key(
                "lane",
                vec![
                    "A", "A", "A", "A", "B", "B", "B", "B", "THIN-1", "THIN-1", "THIN-2", "THIN-2",
                ],
            );
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let model = compile(r#"{"value": "cost", "group": "lane", "min_obs": 3}"#, &view).unwrap();
        assert_eq!(model.n_groups(), 4);
        assert_eq!(model.n_groups_unready(), 2);
        // The doctrine is unchanged: two good lanes do not make the fit safe.
        assert_eq!(model.readiness().status, FitStatus::InsufficientData);

        // Lowering the bar makes every lane ready, and the count follows.
        let model = compile(r#"{"value": "cost", "group": "lane", "min_obs": 2}"#, &view).unwrap();
        assert_eq!(model.n_groups_unready(), 0);
        assert_eq!(model.readiness().status, FitStatus::Converged);
    }

    /// A group whose posterior does not exist at all is unready too — the count is
    /// over readiness verdicts, not over one particular reason for failing them.
    #[test]
    fn a_group_with_no_estimable_posterior_counts_as_unready() {
        let frame = Frame::new(5)
            .numeric("cost", vec![1.0, 1.1, 0.9, 1.05, 42.0])
            .key("lane", vec!["A", "A", "A", "A", "SOLO"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost", "group": "lane"}"#, &view).unwrap();
        assert_eq!(model.n_groups(), 2);
        assert_eq!(model.n_groups_unready(), 1);
    }

    #[test]
    fn an_empty_relation_refuses_rather_than_returning_an_empty_success() {
        let frame = Frame::new(0).numeric("cost", vec![]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();
        assert_ne!(model.readiness().status, FitStatus::Converged);
    }

    //=== Config validation ================================================//

    #[test]
    fn the_value_column_is_required_and_must_exist() {
        let frame = Frame::new(2).numeric("cost", vec![1.0, 2.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        assert!(compile(r#"{}"#, &view).is_err());
        let err = compile(r#"{"value": "cst"}"#, &view).unwrap_err();
        assert!(matches!(err, BayesError::MissingColumn { .. }));
        assert!(err.to_string().contains("cost"), "{err}");
    }

    #[test]
    fn exposure_is_rejected_for_a_likelihood_that_has_no_exposure_term() {
        let frame = Frame::new(2)
            .numeric("cost", vec![1.0, 2.0])
            .numeric("n", vec![1.0, 1.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"value": "cost", "exposure": "n"}"#, &view).unwrap_err();
        assert!(
            err.to_string().contains("only applies to the Poisson"),
            "{err}"
        );
    }

    #[test]
    fn an_unknown_config_slot_is_rejected_before_any_arithmetic() {
        let frame = Frame::new(2).numeric("cost", vec![1.0, 2.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"value": "cost", "grup": "lane"}"#, &view).unwrap_err();
        assert!(err.to_string().contains("did you mean 'group'"), "{err}");
    }

    /// A negative shape is a real modelling choice, not a mistake: the reference
    /// priors this family defaults to are negative. Only values beneath every prior
    /// anyone actually holds are rejected -- and rejected as a config error, so they
    /// do not masquerade as thin data.
    #[test]
    fn admissible_reference_priors_are_accepted_and_absurd_ones_are_a_config_error() {
        let frame = Frame::new(8).numeric("cost", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        for alpha0 in ["-1.0", "-0.5", "0.0", "2.0"] {
            let cfg = format!(r#"{{"value": "cost", "prior": {{"alpha0": {alpha0}}}}}"#);
            assert!(
                compile(&cfg, &view).is_ok(),
                "alpha0 = {alpha0} should be admissible"
            );
        }

        let err = compile(r#"{"value": "cost", "prior": {"alpha0": -5.0}}"#, &view).unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "prior.alpha0"));
        assert!(err.to_string().contains("reference values"), "{err}");
    }

    #[test]
    fn an_unknown_prior_slot_is_rejected_with_its_full_path() {
        let frame = Frame::new(2).numeric("cost", vec![1.0, 2.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(r#"{"value": "cost", "prior": {"alpha": 1.0}}"#, &view).unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "prior.alpha"));
    }

    //=== The differentiable path ==========================================//

    /// The Normal-Inverse-Gamma hyperparameters under the reference prior
    /// (`kappa0 = 0, alpha0 = -1/2, beta0 = 0`), assembled here from the textbook
    /// formulae so the tests below do not inherit the family's own algebra.
    fn nig_by_hand(ys: &[f64]) -> (f64, f64, f64, f64) {
        let n = ys.len() as f64;
        let ybar = ys.iter().sum::<f64>() / n;
        let ss: f64 = ys.iter().map(|y| (y - ybar).powi(2)).sum();
        // (mu_n, kappa_n, alpha_n, beta_n)
        (ybar, n, -0.5 + n / 2.0, ss / 2.0)
    }

    /// Central-difference the log density along every coordinate and compare with the
    /// analytic gradient, at points deliberately displaced from the mode.
    fn check_gradient(target: &dyn crate::catalog::LogPosterior, theta: &[f64], at_the_mode: bool) {
        let dim = target.dim();
        let mut analytic = vec![0.0; dim];
        target.grad(theta, &mut analytic).unwrap();

        // A gradient test evaluated at the mode is vacuous: both sides are zero, and a
        // sign error or a dropped term is invisible. Assert that the point chosen is
        // genuinely somewhere the gradient has something to say.
        let norm = analytic.iter().map(|g| g * g).sum::<f64>().sqrt();
        if !at_the_mode {
            assert!(
                norm > 1e-3,
                "this point is at the mode (gradient norm {norm}), so the check is vacuous"
            );
        }

        for j in 0..dim {
            let step = 1e-6 * theta[j].abs().max(1.0);
            let mut up = theta.to_vec();
            let mut down = theta.to_vec();
            up[j] += step;
            down[j] -= step;
            let numeric = (target.logp(&up) - target.logp(&down)) / (2.0 * step);
            let tol = 1e-4 * numeric.abs().max(1.0);
            assert!(
                (analytic[j] - numeric).abs() < tol,
                "coordinate {j}: analytic {} vs numeric {numeric}",
                analytic[j]
            );
        }
    }

    /// A hand-derived gradient that is subtly wrong still finds *a* mode and still
    /// produces plausible-looking draws; nothing downstream would notice. Finite
    /// differences notice.
    #[test]
    fn the_normal_analytic_gradient_matches_finite_differences_away_from_the_mode() {
        let ys: Vec<f64> = (0..40)
            .map(|i| 12.0 + (i as f64 * 0.7).sin() * 1.5)
            .collect();
        let frame = Frame::new(ys.len()).numeric("cost", ys);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();

        let target = model
            .as_differentiable()
            .expect("conjugate_anomaly must expose a differentiable log posterior");
        assert_eq!(target.dim(), 2, "one group, coordinates (mu, log sigma)");

        for offset in [0.0, 0.6, -1.1] {
            let theta: Vec<f64> = target
                .initial()
                .iter()
                .enumerate()
                .map(|(j, v)| v + offset * (1.0 + j as f64 * 0.4))
                .collect();
            check_gradient(target, &theta, offset == 0.0);
        }
    }

    #[test]
    fn the_poisson_analytic_gradient_matches_finite_differences_away_from_the_mode() {
        let counts: Vec<f64> = (0..40).map(|i| ((i % 9) + 1) as f64).collect();
        let exposure: Vec<f64> = (0..40).map(|i| 5.0 + (i % 4) as f64).collect();
        let frame = Frame::new(40)
            .numeric("claims", counts)
            .numeric("consignments", exposure);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(
            r#"{"value": "claims", "likelihood": "poisson", "exposure": "consignments"}"#,
            &view,
        )
        .unwrap();

        let target = model
            .as_differentiable()
            .expect("conjugate_anomaly must expose a differentiable log posterior");
        assert_eq!(target.dim(), 1, "one group, coordinate log lambda");

        for offset in [0.0, 0.8, -1.4] {
            let theta: Vec<f64> = target.initial().iter().map(|v| v + offset).collect();
            check_gradient(target, &theta, offset == 0.0);
        }
    }

    /// **The test that catches a missing Jacobian.** A gradient-versus-finite-difference
    /// check cannot: it differences the same `logp`, so a dropped `+2u` term cancels on
    /// both sides and the check still passes. Here the reference density is written
    /// independently — the textbook Normal-Inverse-Gamma posterior evaluated at
    /// `sigma^2 = e^{2u}`, plus the log of `d sigma^2 / du = 2 e^{2u}` — so omitting the
    /// Jacobian shifts every difference by exactly `2 (u_a - u_b)`.
    #[test]
    fn the_normal_log_density_matches_the_closed_form_posterior_up_to_a_constant() {
        let ys: Vec<f64> = (0..25)
            .map(|i| 4.0 + (i as f64 * 1.3).cos() * 0.8)
            .collect();
        let (mu_n, kappa_n, alpha_n, beta_n) = nig_by_hand(&ys);

        let frame = Frame::new(ys.len()).numeric("cost", ys);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();
        let target = model.as_differentiable().unwrap();

        // log p(mu, sigma^2) + log |d sigma^2 / du|, written from the textbook form.
        let reference = |mu: f64, u: f64| {
            let s2 = (2.0 * u).exp();
            // sigma^2 ~ InvGamma(alpha_n, beta_n)
            let inv_gamma = -(alpha_n + 1.0) * s2.ln() - beta_n / s2;
            // mu | sigma^2 ~ N(mu_n, sigma^2 / kappa_n)
            let normal = -0.5 * (s2 / kappa_n).ln() - kappa_n * (mu - mu_n).powi(2) / (2.0 * s2);
            // sigma^2 = e^{2u}, so the Jacobian is 2 e^{2u}.
            inv_gamma + normal + 2.0f64.ln() + 2.0 * u
        };

        let u0 = target.initial()[1];
        let points = [
            [mu_n, u0],
            [mu_n + 0.35, u0],
            [mu_n - 0.6, u0 + 0.8],
            [mu_n + 1.2, u0 - 0.55],
        ];
        for a in &points {
            for b in &points {
                let got = target.logp(a) - target.logp(b);
                let want = reference(a[0], a[1]) - reference(b[0], b[1]);
                assert!(
                    (got - want).abs() < 1e-9 * want.abs().max(1.0),
                    "logp difference {got} vs closed form {want} between {a:?} and {b:?}"
                );
                // ...and the check has teeth: without the Jacobian the same difference
                // would be off by 2 (u_a - u_b), which this tolerance would reject.
                let jacobian_gap = 2.0 * (a[1] - b[1]);
                assert!(
                    jacobian_gap == 0.0 || jacobian_gap.abs() > 1e-9 * want.abs().max(1.0),
                    "a dropped Jacobian would be invisible between {a:?} and {b:?}"
                );
            }
        }
    }

    /// The Poisson half of the same check. `lambda = e^v` contributes a log-Jacobian of
    /// `v`, which is what turns the `a_n - 1` exponent of the Gamma density into `a_n`.
    /// Dropping it biases every posterior for `lambda` downward by one shape unit --
    /// negligible on a busy lane, and badly wrong on a thin one, which is precisely the
    /// group an anomaly model exists to look at.
    #[test]
    fn the_poisson_log_density_matches_the_closed_form_posterior_up_to_a_constant() {
        let counts: Vec<f64> = (0..18).map(|i| ((i % 6) + 2) as f64).collect();
        let a_n = 0.5 + counts.iter().sum::<f64>();
        let b_n = counts.len() as f64;

        let frame = Frame::new(counts.len()).numeric("claims", counts);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "claims", "likelihood": "poisson"}"#, &view).unwrap();
        let target = model.as_differentiable().unwrap();

        // Gamma(a_n, rate b_n) evaluated at lambda = e^v, plus log |d lambda / dv| = v.
        let reference = |v: f64| {
            let lambda = v.exp();
            (a_n - 1.0) * lambda.ln() - b_n * lambda + v
        };

        let v0 = target.initial()[0];
        for a in [v0, v0 + 0.5, v0 - 1.2, v0 + 2.0] {
            for b in [v0, v0 + 0.5, v0 - 1.2, v0 + 2.0] {
                let got = target.logp(&[a]) - target.logp(&[b]);
                let want = reference(a) - reference(b);
                assert!(
                    (got - want).abs() < 1e-9 * want.abs().max(1.0),
                    "logp difference {got} vs closed form {want} between {a} and {b}"
                );
            }
        }
    }

    /// The unconstrained coordinates must map back to the parameters the draws table
    /// reports, in the order `param_names` promises -- and a group whose posterior does
    /// not exist must come back NULL-shaped rather than occupying a coordinate the mode
    /// search would then wander along.
    #[test]
    fn the_differentiable_path_skips_unfittable_groups_and_reports_them_as_null() {
        let frame = Frame::new(6)
            .numeric("cost", vec![1.0, 1.2, 0.9, 1.1, 5.0, 42.0])
            .key("lane", vec!["A", "A", "A", "A", "SOLO", "OTHER"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost", "group": "lane"}"#, &view).unwrap();

        let target = model.as_differentiable().unwrap();
        // Three groups, two of them single-observation and therefore unfittable, so
        // only lane A contributes coordinates.
        assert_eq!(model.n_groups(), 3);
        assert_eq!(target.dim(), 2);

        let mut out = vec![0.0; model.param_names().len()];
        target.constrain(&target.initial(), &mut out);
        assert_eq!(out.len(), 6);
        assert!(out[0].is_finite() && out[1] > 0.0, "lane A: {out:?}");
        assert!(out[2..].iter().all(|v| v.is_nan()), "{out:?}");
    }

    //=== Determinism ======================================================//

    #[test]
    fn the_same_seed_reproduces_the_same_draws() {
        let frame = Frame::new(6).numeric("cost", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(r#"{"value": "cost"}"#, &view).unwrap();

        let a = draw(&*model, 500, 31);
        let b = draw(&*model, 500, 31);
        assert_eq!(a, b);
        assert_ne!(a, draw(&*model, 500, 32));
    }

    #[test]
    fn the_fingerprint_tracks_the_data_the_model_reads() {
        let make = |values: Vec<f64>| {
            let frame = Frame::new(values.len()).numeric("cost", values);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let model = compile(r#"{"value": "cost"}"#, &view).unwrap();
            let fp = model.data_fingerprint().to_string();
            fp
        };
        assert_eq!(make(vec![1.0, 2.0, 3.0]), make(vec![1.0, 2.0, 3.0]));
        assert_ne!(make(vec![1.0, 2.0, 3.0]), make(vec![1.0, 2.0, 4.0]));
    }
}
