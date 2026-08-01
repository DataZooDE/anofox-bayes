//! Effective sample size, bulk and tail (Vehtari et al. 2021).
//!
//! `n` correlated draws carry less information than `n` independent ones. ESS is how
//! many independent draws they are worth:
//!
//! ```text
//!   ESS = m*n / tau,     tau = 1 + 2 * sum_t rho_t
//! ```
//!
//! where `rho_t` is the autocorrelation at lag `t`, estimated across all chains. The
//! infinite sum is truncated by **Geyer's initial monotone positive sequence**: the
//! adjacent-pair sums `rho_{2t} + rho_{2t+1}` are provably positive and decreasing for
//! a reversible chain, so the estimator keeps pairs while they stay positive and then
//! enforces monotonicity. Truncating at a fixed lag instead would accumulate the noise
//! in the high-lag estimates and inflate ESS — exactly the direction that makes an
//! under-sampled fit look adequate.
//!
//! The implementation follows Stan's reference `compute_effective_sample_size`
//! step for step, including the split-chain preprocessing and the `1/log10(N)` floor
//! on `tau`. That fidelity is what makes the PyMC/ArviZ golden-run parity suite a
//! real check rather than a coincidence.
//!
//! **Bulk vs. tail.** [`ess_bulk`] measures the reliability of the posterior *mean*;
//! [`ess_tail`] measures the reliability of the 5 % and 95 % *quantiles*, by computing
//! ESS of the indicator series `1[x <= q]` at each. They come apart in practice, and
//! a service-level or safety-stock decision reads a tail quantile — gating only on
//! bulk ESS would certify precisely the number that is not yet reliable.

use super::normal_scores;

/// ESS of the rank-normalised, split draws: how far the posterior *mean* can be
/// trusted.
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    let Some(split) = split_chains(chains) else {
        return 0.0;
    };
    let pooled: Vec<f64> = split.iter().flatten().copied().collect();
    if pooled.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let n = split[0].len();
    let scores = normal_scores(&pooled);
    let normalised: Vec<Vec<f64>> = (0..split.len())
        .map(|c| scores[c * n..(c + 1) * n].to_vec())
        .collect();
    ess_of(&normalised)
}

/// The smaller of the ESS at the 5 % and 95 % quantiles: how far the posterior
/// *tails* can be trusted.
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    if split_chains(chains).is_none() {
        return 0.0;
    }
    let mut pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    if pooled.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    pooled.sort_by(|a, b| a.partial_cmp(b).expect("finiteness checked above"));

    // ESS of the indicator series `1[x <= q]`. The series is 0/1, and the rank
    // normalisation inside `ess_bulk` is what turns it into something the
    // autocorrelation estimator can work with.
    let at = |q: f64| -> f64 {
        let indicators: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.iter().map(|&v| if v <= q { 1.0 } else { 0.0 }).collect())
            .collect();
        ess_bulk(&indicators)
    };

    at(quantile_sorted(&pooled, 0.05)).min(at(quantile_sorted(&pooled, 0.95)))
}

/// Halve every chain, as Stan does before computing R̂ and ESS.
///
/// Returns `None` when the input cannot be assessed: no chains, ragged chains, or
/// halves too short for a lag-2 autocorrelation.
fn split_chains(chains: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let m = chains.len();
    if m == 0 {
        return None;
    }
    let n = chains[0].len();
    if chains.iter().any(|c| c.len() != n) {
        return None;
    }
    let half = n / 2;
    if half < 4 {
        return None;
    }
    let mut out = Vec::with_capacity(m * 2);
    for c in chains {
        out.push(c[..half].to_vec());
        out.push(c[half..2 * half].to_vec());
    }
    Some(out)
}

/// Linear-interpolated quantile of an already-sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// The autocorrelation-based ESS estimator, on chains that are already split and
/// rank-normalised.
///
/// Kept separate from the preprocessing so it can be tested directly against the
/// closed-form AR(1) answer, where `tau = (1 + rho) / (1 - rho)` is known exactly.
fn ess_of(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m == 0 {
        return 0.0;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return 0.0;
    }
    let total = (m * n) as f64;

    let means: Vec<f64> = chains
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();

    // Biased autocovariance (divisor n), matching Stan/ArviZ. Computed lazily by
    // lag: Geyer's rule truncates after a few dozen lags for any chain that mixes,
    // so the quadratic worst case is never reached in practice.
    let acov = |lag: usize| -> f64 {
        (0..m)
            .map(|c| {
                let x = &chains[c];
                let mu = means[c];
                let mut s = 0.0;
                for i in 0..(n - lag) {
                    s += (x[i] - mu) * (x[i + lag] - mu);
                }
                s / n as f64
            })
            .sum::<f64>()
            / m as f64
    };

    let chain_vars: Vec<f64> = (0..m)
        .map(|c| {
            let mu = means[c];
            chains[c].iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (n - 1) as f64
        })
        .collect();
    let w = chain_vars.iter().sum::<f64>() / m as f64;

    // A parameter that never moves has no autocorrelation to discount. Reporting the
    // raw count here (rather than 0) keeps the ESS gate focused on under-sampling;
    // genuine degeneracy is caught by the status path instead.
    if w <= 0.0 || !w.is_finite() {
        return total;
    }

    let var_plus = if m > 1 {
        let grand = means.iter().sum::<f64>() / m as f64;
        let b =
            n as f64 * means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1) as f64;
        ((n - 1) as f64 * w + b) / n as f64
    } else {
        w
    };
    if var_plus <= 0.0 || !var_plus.is_finite() {
        return total;
    }

    let rho = |lag: usize| -> f64 { 1.0 - (w - acov(lag)) / var_plus };

    // Geyer's initial positive sequence: walk forward in adjacent pairs, stopping as
    // soon as a pair sum goes non-positive.
    let mut rho_hat = vec![1.0, rho(1)];
    let mut t = 1usize;
    while t + 2 < n - 2 {
        let even = rho(t + 1);
        let odd = rho(t + 2);
        if even + odd <= 0.0 {
            break;
        }
        rho_hat.push(even);
        rho_hat.push(odd);
        t += 2;
    }
    // Stan keeps a trailing positive even term beyond the truncation point.
    let trailing = if t + 1 < n { rho(t + 1).max(0.0) } else { 0.0 };

    // Initial monotone sequence: a pair sum may not exceed its predecessor.
    let mut i = 3;
    while i + 1 < rho_hat.len() {
        let prev = rho_hat[i - 2] + rho_hat[i - 1];
        if rho_hat[i] + rho_hat[i + 1] > prev {
            rho_hat[i] = prev / 2.0;
            rho_hat[i + 1] = rho_hat[i];
        }
        i += 2;
    }

    let mut tau = -1.0 + 2.0 * rho_hat.iter().sum::<f64>() + trailing;
    // Stan's floor: without it, a strongly antithetic chain yields a tau at or below
    // zero and hence an infinite or negative ESS.
    tau = tau.max(1.0 / total.log10().max(1.0));

    total / tau
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    #[test]
    fn independent_draws_have_an_ess_near_the_draw_count() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| iid_chain(400 + c, 1000, 0.0)).collect();
        let ess = ess_bulk(&chains);
        // 4000 draws; sampling noise in the autocorrelation estimate moves this by
        // a few percent in either direction.
        assert!(ess > 3200.0 && ess < 4800.0, "ess {ess} should be ~4000");
    }

    /// The whole reason ESS exists. An AR(1) chain has the *same* stationary
    /// distribution as an independent one, so the histogram, the mean and the
    /// variance all look fine; only the autocorrelation reveals that 1000 draws are
    /// worth far fewer.
    #[test]
    fn autocorrelated_draws_have_a_much_smaller_ess_than_their_draw_count() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| ar1_chain(500 + c, 1000, 0.9, 0.0)).collect();
        let ess = ess_bulk(&chains);
        // Theoretical tau = (1+rho)/(1-rho) = 19, so ESS ~ 4000/19 ~ 210.
        assert!(ess < 600.0, "ess {ess} should be far below 4000");
        assert!(ess > 60.0, "ess {ess} should not collapse to nothing");

        // And the marginal really is indistinguishable from the iid case, which is
        // what makes ESS rather than a histogram the right diagnostic.
        let flat: Vec<f64> = chains.iter().flatten().copied().collect();
        let mean = flat.iter().sum::<f64>() / flat.len() as f64;
        let var = flat.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (flat.len() - 1) as f64;
        assert!(mean.abs() < 0.25, "mean {mean}");
        assert!((var - 1.0).abs() < 0.25, "variance {var}");
    }

    /// The estimator is pinned against the closed-form AR(1) answer
    /// `tau = (1 + rho) / (1 - rho)`, so a regression in the Geyer truncation shows
    /// up as a number rather than as a vague "seems lower".
    #[test]
    fn ess_approximates_the_closed_form_ar1_answer() {
        for rho in [0.5, 0.8] {
            let chains: Vec<Vec<f64>> =
                (0..4).map(|c| ar1_chain(900 + c, 8000, rho, 0.0)).collect();
            let ess = ess_bulk(&chains);
            let tau = (1.0 + rho) / (1.0 - rho);
            let expected = 4.0 * 8000.0 / tau;
            let ratio = ess / expected;
            assert!(
                (0.7..1.4).contains(&ratio),
                "rho={rho}: ess {ess} vs closed-form {expected} (ratio {ratio})"
            );
        }
    }

    #[test]
    fn ess_falls_monotonically_as_autocorrelation_rises() {
        let ess_at = |rho: f64| {
            let chains: Vec<Vec<f64>> =
                (0..4).map(|c| ar1_chain(600 + c, 2000, rho, 0.0)).collect();
            ess_bulk(&chains)
        };
        let (low, mid, high) = (ess_at(0.0), ess_at(0.7), ess_at(0.95));
        assert!(
            low > mid,
            "ess {low} at rho=0 should exceed {mid} at rho=0.7"
        );
        assert!(
            mid > high,
            "ess {mid} at rho=0.7 should exceed {high} at rho=0.95"
        );
    }

    /// A chain that mixes well in the body but sticks in the tail: the mean is
    /// reliable long before the 95th percentile is. A gate on bulk ESS alone would
    /// approve the safety-stock quantile this fit cannot yet support.
    #[test]
    fn tail_ess_detects_tail_only_stickiness_that_bulk_ess_misses() {
        use crate::rng::BayesRng;
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                let mut rng = BayesRng::for_chain(700 + c, 0);
                let mut out: Vec<f64> = Vec::with_capacity(4000);
                while out.len() < 4000 {
                    let v = rng.standard_normal();
                    if v > 1.5 {
                        // Excursions into the tail are sticky: the chain stays there
                        // for a long run before returning.
                        for _ in 0..40 {
                            if out.len() == 4000 {
                                break;
                            }
                            out.push(v);
                        }
                    } else {
                        out.push(v);
                    }
                }
                out
            })
            .collect();

        let bulk = ess_bulk(&chains);
        let tail = ess_tail(&chains);
        assert!(
            tail < bulk * 0.7,
            "tail ess {tail} should be well below bulk ess {bulk}"
        );
    }

    #[test]
    fn ess_is_zero_for_input_it_cannot_assess() {
        assert_eq!(ess_bulk(&[]), 0.0);
        assert_eq!(ess_bulk(&[vec![1.0, 2.0]]), 0.0);
        assert_eq!(ess_bulk(&[vec![1.0; 10], vec![1.0; 9]]), 0.0);
        assert_eq!(ess_tail(&[]), 0.0);

        let mut bad = iid_chain(1, 100, 0.0);
        bad[3] = f64::INFINITY;
        assert_eq!(ess_bulk(&[bad]), 0.0);
    }

    #[test]
    fn ess_is_always_positive_and_finite_for_assessable_input() {
        let chains: Vec<Vec<f64>> = (0..2).map(|c| iid_chain(800 + c, 500, 0.0)).collect();
        let ess = ess_bulk(&chains);
        assert!(ess > 0.0 && ess.is_finite(), "ess {ess}");
    }

    /// An antithetic chain has negative lag-1 autocorrelation, so the naive `tau`
    /// goes below zero and the ESS would come out negative or infinite. Stan's
    /// floor is what keeps the number usable.
    #[test]
    fn an_antithetic_chain_yields_a_large_but_finite_ess() {
        let chains: Vec<Vec<f64>> = (0..2)
            .map(|c| {
                iid_chain(1000 + c, 2000, 0.0)
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| if i % 2 == 0 { v } else { -v })
                    .collect()
            })
            .collect();
        let ess = ess_bulk(&chains);
        assert!(ess.is_finite() && ess > 0.0, "ess {ess}");
    }
}
