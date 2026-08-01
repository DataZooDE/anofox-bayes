//! Convergence diagnostics: split-R̂, bulk and tail ESS, divergence counts.
//!
//! These exist so that an agent's quality gate is a SQL query rather than a judgement
//! call. Every function here is exposed as a DuckDB aggregate over
//! `GROUP BY param`, which turns "is this fit safe to act on?" into something a
//! deterministic workflow node can enforce and an auditor can re-run.
//!
//! The definitions follow Vehtari et al. (2021), *Rank-normalization, folding, and
//! localization: An improved R̂ for assessing convergence of MCMC* — the same
//! reference ArviZ implements, which is what makes the golden-run parity suite
//! meaningful. In particular:
//!
//! * R̂ is **split** R̂: each chain is halved before the between/within comparison, so
//!   a single chain that drifts steadily is caught even though its two halves would
//!   individually look stationary.
//! * R̂ is computed on **rank-normalised** draws, which makes it well-defined for
//!   heavy-tailed posteriors where the variance-based statistic is not.
//! * ESS comes in two flavours because they answer different questions: `ess_bulk`
//!   governs whether the posterior *mean* is trustworthy, `ess_tail` whether the 5 %
//!   and 95 % *quantiles* are. A safety-stock decision reads a tail quantile, so
//!   gating only on bulk ESS would certify exactly the wrong number.

mod ess;
mod rhat;

pub use ess::{ess_bulk, ess_tail};
pub use rhat::rhat;

use crate::draws::Posterior;
use statrs::distribution::{ContinuousCDF, Normal};

/// Replace values by their normal scores: rank, then map through the inverse normal
/// CDF with the Blom offset `(r - 3/8) / (n + 1/4)`.
///
/// Both R̂ and ESS are variance-based statistics, and a posterior without a finite
/// variance — several catalog families have heavy tails — makes them undefined. Rank
/// normalisation replaces the draws with a series that is guaranteed Gaussian-shaped
/// while preserving the ordering, and therefore preserving exactly the mixing
/// information the diagnostics are trying to measure.
///
/// Ties receive their average rank, which matters more than it looks: the tail-ESS
/// indicator series is all zeros and ones, so without tie handling every value would
/// map to one of two ranks and the statistic would be meaningless.
fn normal_scores(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .expect("callers filter non-finite values before ranking")
    });

    // Average ranks within tied runs (1-based).
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j {
            ranks[order[k]] = avg;
        }
        i = j + 1;
    }

    let normal = Normal::new(0.0, 1.0).expect("standard normal is always constructible");
    ranks
        .into_iter()
        .map(|r| {
            let p = (r - 0.375) / (n as f64 + 0.25);
            normal.inverse_cdf(p.clamp(1e-12, 1.0 - 1e-12))
        })
        .collect()
}

/// Default thresholds for turning diagnostics into a [`crate::types::FitStatus`].
///
/// Conservative by design: an agent that acts on a bad posterior costs a customer
/// real money, while one that refuses too often costs an analyst an afternoon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Vehtari et al. recommend 1.01; the older 1.1 is known to pass visibly
    /// unconverged chains.
    pub max_rhat: f64,
    /// 400 is the point at which the ESS estimate itself becomes reliable.
    pub min_ess_bulk: f64,
    pub min_ess_tail: f64,
    /// Any divergence indicates biased exploration, so the default is zero
    /// tolerance rather than a small budget.
    pub max_divergent: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            max_rhat: 1.01,
            min_ess_bulk: 400.0,
            min_ess_tail: 400.0,
            max_divergent: 0.0,
        }
    }
}

/// Diagnostics for one parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDiagnostics {
    pub group_id: String,
    pub param: String,
    /// `None` when the engine produced a single chain: R̂ compares chains, and a
    /// fabricated 1.0 would read as "converged".
    pub rhat: Option<f64>,
    pub ess_bulk: f64,
    pub ess_tail: f64,
}

impl ParamDiagnostics {
    /// Whether this parameter clears the gate.
    ///
    /// A missing R̂ does not fail the check — the exact and Laplace engines draw
    /// independently, so there is no between-chain variance to assess and no
    /// convergence to fail. The ESS checks still apply and still bite.
    pub fn passes(&self, t: &Thresholds) -> bool {
        self.rhat
            .map(|r| r.is_finite() && r <= t.max_rhat)
            .unwrap_or(true)
            && self.ess_bulk >= t.min_ess_bulk
            && self.ess_tail >= t.min_ess_tail
    }
}

/// Compute diagnostics for every parameter of a posterior.
pub fn diagnose(post: &Posterior) -> Vec<ParamDiagnostics> {
    (0..post.n_params())
        .map(|p| {
            let chains: Vec<Vec<f64>> = (0..post.n_chains)
                .map(|c| post.chain_values(c, p).collect())
                .collect();
            let name = &post.params[p];
            ParamDiagnostics {
                group_id: name.group_id.clone(),
                param: name.name.clone(),
                rhat: rhat(&chains),
                ess_bulk: ess_bulk(&chains),
                ess_tail: ess_tail(&chains),
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod testing {
    //! Chain fixtures shared by the R̂ and ESS tests.

    use crate::rng::BayesRng;

    /// `n` independent `N(mean, 1)` draws.
    pub fn iid_chain(seed: u64, n: usize, mean: f64) -> Vec<f64> {
        let mut rng = BayesRng::for_chain(seed, 0);
        (0..n).map(|_| mean + rng.standard_normal()).collect()
    }

    /// An AR(1) chain with autocorrelation `rho` and unit stationary variance.
    ///
    /// The stationary distribution is `N(mean, 1)` regardless of `rho`, so an
    /// estimator that ignores autocorrelation sees the same marginal as
    /// [`iid_chain`] — which is precisely the failure ESS exists to catch.
    pub fn ar1_chain(seed: u64, n: usize, rho: f64, mean: f64) -> Vec<f64> {
        let mut rng = BayesRng::for_chain(seed, 0);
        let scale = (1.0 - rho * rho).sqrt();
        let mut x = rng.standard_normal();
        let mut out = Vec::with_capacity(n);
        // Burn in so the chain starts from its stationary distribution.
        for _ in 0..500 {
            x = rho * x + scale * rng.standard_normal();
        }
        for _ in 0..n {
            x = rho * x + scale * rng.standard_normal();
            out.push(mean + x);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn a_parameter_with_no_rhat_can_still_fail_on_ess() {
        let t = Thresholds::default();
        let independent = ParamDiagnostics {
            group_id: "__global__".into(),
            param: "mu".into(),
            rhat: None,
            ess_bulk: 1800.0,
            ess_tail: 1500.0,
        };
        assert!(independent.passes(&t));

        let starved = ParamDiagnostics {
            ess_bulk: 120.0,
            ..independent.clone()
        };
        assert!(!starved.passes(&t));
    }

    /// A safety-stock decision reads the 95th percentile, not the mean. A fit whose
    /// bulk is fine but whose tail is starved must not pass the gate.
    #[test]
    fn a_starved_tail_fails_the_gate_even_when_the_bulk_is_healthy() {
        let t = Thresholds::default();
        let d = ParamDiagnostics {
            group_id: "__global__".into(),
            param: "sigma".into(),
            rhat: Some(1.001),
            ess_bulk: 4000.0,
            ess_tail: 90.0,
        };
        assert!(!d.passes(&t));
    }

    #[test]
    fn a_non_finite_rhat_never_passes() {
        let t = Thresholds::default();
        let d = ParamDiagnostics {
            group_id: "__global__".into(),
            param: "mu".into(),
            rhat: Some(f64::NAN),
            ess_bulk: 4000.0,
            ess_tail: 4000.0,
        };
        assert!(!d.passes(&t));
    }

    #[test]
    fn diagnose_reports_one_row_per_parameter_carrying_its_group() {
        use crate::draws::{ModelMeta, ParamName, Posterior};
        use crate::types::{EngineKind, FitStatus};

        let params = vec![
            ParamName::global("mu").unwrap(),
            ParamName::grouped("LANE-7", "sigma").unwrap(),
        ];
        let n_draws = 500;
        let mu = iid_chain(11, n_draws, 0.0);
        let sigma = iid_chain(12, n_draws, 3.0);
        let mut values = Vec::with_capacity(n_draws * 2);
        for d in 0..n_draws {
            values.push(mu[d]);
            values.push(sigma[d]);
        }
        let post = Posterior::new(
            ModelMeta {
                model_id: "m".into(),
                family: "f".into(),
                engine: EngineKind::Exact,
                status: FitStatus::Converged,
                seed: 1,
                n_obs: 10,
                n_groups: 1,
            },
            params,
            1,
            n_draws,
            values,
            Vec::new(),
        )
        .unwrap();

        let diags = diagnose(&post);
        assert_eq!(diags.len(), 2);
        assert_eq!(diags[0].param, "mu");
        assert_eq!(diags[0].group_id, "__global__");
        assert_eq!(diags[1].param, "sigma");
        assert_eq!(diags[1].group_id, "LANE-7");
        // Single chain: no between-chain comparison exists.
        assert!(diags[0].rhat.is_none());
        assert!(diags[0].ess_bulk > 400.0);
    }
}
