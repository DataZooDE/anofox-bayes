//! Rank-normalised split-R̂ (Vehtari et al. 2021).
//!
//! R̂ compares the variance *between* chains to the variance *within* them. If the
//! chains have found the same distribution the two agree and R̂ → 1; if they are
//! still exploring different regions the between-chain variance dominates and R̂ rises.
//!
//! Two refinements over the textbook statistic, both of which the ArviZ reference
//! implements and both of which matter here:
//!
//! **Splitting.** Each chain is cut in half and the halves are treated as separate
//! chains. Without this, a chain drifting slowly in one direction — the classic
//! failure of a hierarchical variance parameter — compares only against other chains
//! drifting the same way, and reports a comfortable 1.00.
//!
//! **Rank normalisation.** Draws are replaced by their normal-scores before the
//! variance comparison. The plain statistic is a ratio of variances and is undefined
//! for a posterior without finite variance; several families in the catalog have
//! heavy tails, so this is not a theoretical concern.

use super::normal_scores;

/// Split-R̂ over `chains`, or `None` when the statistic is not defined.
///
/// `None` is returned when there is only one chain, or when chains are too short to
/// split (fewer than 4 draws), or when every draw is identical. Returning `None`
/// rather than 1.0 is deliberate: an agent gating on `rhat <= 1.01` must not be told
/// "converged" by a statistic that was never computed.
pub fn rhat(chains: &[Vec<f64>]) -> Option<f64> {
    if chains.len() < 2 {
        return None;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return None;
    }

    // Rank-normalise across the pooled draws, then split.
    let pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    if pooled.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scores = normal_scores(&pooled);

    let half = n / 2;
    let mut split: Vec<&[f64]> = Vec::with_capacity(chains.len() * 2);
    for (c, _) in chains.iter().enumerate() {
        let start = c * n;
        split.push(&scores[start..start + half]);
        split.push(&scores[start + half..start + 2 * half]);
    }

    rhat_of_splits(&split, half)
}

/// The variance-ratio statistic over already-split, already-normalised segments.
fn rhat_of_splits(split: &[&[f64]], n: usize) -> Option<f64> {
    let m = split.len();
    if m < 2 || n < 2 {
        return None;
    }

    let means: Vec<f64> = split
        .iter()
        .map(|s| s.iter().sum::<f64>() / n as f64)
        .collect();
    let vars: Vec<f64> = split
        .iter()
        .zip(&means)
        .map(|(s, &mu)| s.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (n - 1) as f64)
        .collect();

    let grand_mean = means.iter().sum::<f64>() / m as f64;
    // Between-chain variance, scaled to the per-draw scale.
    let b = n as f64
        * means
            .iter()
            .map(|mu| (mu - grand_mean).powi(2))
            .sum::<f64>()
        / (m - 1) as f64;
    let w = vars.iter().sum::<f64>() / m as f64;

    if w <= 0.0 || !w.is_finite() || !b.is_finite() {
        // Every draw identical: the posterior is a point mass, not a converged
        // exploration of anything. No R-hat is defined.
        return None;
    }

    // var_plus is the marginal posterior variance estimate combining both sources.
    let var_plus = ((n - 1) as f64 * w + b) / n as f64;
    Some((var_plus / w).sqrt())
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    #[test]
    fn chains_from_the_same_distribution_give_rhat_near_one() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| iid_chain(100 + c, 2000, 0.0)).collect();
        let r = rhat(&chains).unwrap();
        assert!(r < 1.01, "rhat {r} should be ~1 for well-mixed chains");
        // R-hat is not bounded below by 1: `var_plus` is a weighted average of the
        // within- and between-chain variances, so a run where the chain means happen
        // to agree more closely than sampling noise predicts lands just under 1.
        // Only the upper bound is a convergence claim.
        assert!(r > 0.95, "rhat {r} should still be close to 1");
    }

    /// The headline failure: two chains stuck in different places. A gate at 1.01
    /// must reject this.
    #[test]
    fn chains_exploring_different_regions_give_a_large_rhat() {
        let chains = vec![iid_chain(1, 2000, 0.0), iid_chain(2, 2000, 5.0)];
        let r = rhat(&chains).unwrap();
        assert!(r > 1.5, "rhat {r} should be large for separated chains");
    }

    /// Splitting is what catches this. Both chains drift in the same direction, so
    /// an unsplit R-hat compares two similar trajectories and reports ~1.0; the
    /// split statistic compares each chain's first half against its second and sees
    /// the drift immediately.
    #[test]
    fn split_rhat_catches_within_chain_drift_that_unsplit_rhat_would_miss() {
        let n = 2000;
        let drift: Vec<Vec<f64>> = (0..2)
            .map(|c| {
                iid_chain(200 + c, n, 0.0)
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| v + 6.0 * (i as f64) / (n as f64))
                    .collect()
            })
            .collect();

        let split = rhat(&drift).unwrap();
        assert!(
            split > 1.1,
            "split rhat {split} should flag a drifting chain"
        );

        // Sanity check that the drift really is invisible between chains: the two
        // chains' means agree closely, which is what an unsplit statistic compares.
        let mean = |c: &Vec<f64>| c.iter().sum::<f64>() / c.len() as f64;
        assert!((mean(&drift[0]) - mean(&drift[1])).abs() < 0.2);
    }

    /// Rank normalisation is what makes R-hat defined at all here: the plain
    /// variance-ratio statistic on Cauchy draws is a ratio of two quantities that do
    /// not converge.
    #[test]
    fn rhat_is_finite_for_a_heavy_tailed_posterior() {
        use crate::rng::BayesRng;
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                let mut rng = BayesRng::for_chain(300 + c, 0);
                // Ratio of two standard normals is standard Cauchy: no mean, no
                // variance.
                (0..2000)
                    .map(|_| rng.standard_normal() / rng.standard_normal())
                    .collect()
            })
            .collect();
        let r = rhat(&chains).unwrap();
        assert!(r.is_finite(), "rhat must be finite for heavy tails");
        assert!(
            r < 1.05,
            "rhat {r} should still be ~1 for well-mixed chains"
        );
    }

    /// The exact and Laplace engines write a single chain. A fabricated 1.0 would
    /// tell an agent "converged" about a statistic that was never computed.
    #[test]
    fn a_single_chain_yields_no_rhat_rather_than_a_reassuring_one() {
        assert_eq!(rhat(&[iid_chain(1, 2000, 0.0)]), None);
        assert_eq!(rhat(&[]), None);
    }

    #[test]
    fn a_point_mass_yields_no_rhat() {
        let chains = vec![vec![2.0; 100], vec![2.0; 100]];
        assert_eq!(rhat(&chains), None);
    }

    #[test]
    fn chains_that_are_too_short_or_ragged_yield_no_rhat() {
        assert_eq!(rhat(&[vec![1.0, 2.0], vec![1.0, 2.0]]), None);
        assert_eq!(rhat(&[vec![1.0; 10], vec![1.0; 9]]), None);
    }

    #[test]
    fn non_finite_draws_yield_no_rhat() {
        let mut bad = iid_chain(1, 100, 0.0);
        bad[7] = f64::NAN;
        assert_eq!(rhat(&[bad, iid_chain(2, 100, 0.0)]), None);
    }
}
