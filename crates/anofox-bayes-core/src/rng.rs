//! Deterministic, reproducible randomness.
//!
//! Every draw this crate produces must be byte-identical for a given
//! `(seed, chain)` pair on every platform and in every DuckDB thread layout. That is
//! not a nicety: the SQL test suite asserts on posterior quantiles, the golden-run
//! parity suite compares against pinned PyMC output, and an auditor at a customer
//! site has to be able to re-run a fit and get the same recommendation.
//!
//! Two decisions follow.
//!
//! **ChaCha20, not the thread RNG.** A counter-based stream cipher reproduces exactly
//! from a seed with no dependence on hardware entropy, thread scheduling, or the
//! order in which chains happen to be scheduled.
//!
//! **Chain seeds are derived by hashing, not by XOR.** The obvious `seed ^ chain`
//! collides embarrassingly often — seed 4 chain 1 and seed 5 chain 0 produce the
//! identical stream, which would make two "independent" chains identical and drive
//! R-hat to a perfect 1.0 for exactly the wrong reason. Hashing the pair removes the
//! structure; [`chains_are_independent_streams`] pins it.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};

use crate::errors::{BayesError, BayesResult};

/// The random stream for one chain of one fit.
pub struct BayesRng {
    inner: ChaCha20Rng,
}

impl BayesRng {
    /// Derive the stream for `chain` of a fit seeded with `seed`.
    ///
    /// The derivation is a BLAKE3 hash of the pair, so neighbouring seeds and
    /// neighbouring chain indices produce unrelated streams.
    pub fn for_chain(seed: u64, chain: u32) -> Self {
        let mut input = [0u8; 12];
        input[..8].copy_from_slice(&seed.to_le_bytes());
        input[8..].copy_from_slice(&chain.to_le_bytes());
        let digest = blake3::hash(&input);
        Self {
            inner: ChaCha20Rng::from_seed(*digest.as_bytes()),
        }
    }

    /// Derive the stream for one **group** of one chain of a fit.
    ///
    /// The reason this exists rather than `for_chain` being reused: a family that
    /// fits its groups independently samples them in parallel, and a single shared
    /// stream would tie a group's numbers to the *order* its task happened to run in.
    /// Keying on the group's own identity instead makes the draws a function of the
    /// data and the seed alone — same numbers on one thread and on thirty-two, same
    /// numbers whichever order the rows arrived in, and same numbers whether the group
    /// was fitted alone or alongside twenty thousand others.
    ///
    /// The key is length-prefixed before the chain index for the same reason
    /// `derive_model_id` length-prefixes its fields: without it, group `"a"` on chain
    /// 0x62 and group `"ab"` on chain 0 would hash the same bytes.
    pub fn for_group(seed: u64, chain: u32, group: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        hasher.update(&(group.len() as u64).to_le_bytes());
        hasher.update(group.as_bytes());
        hasher.update(&chain.to_le_bytes());
        Self {
            inner: ChaCha20Rng::from_seed(*hasher.finalize().as_bytes()),
        }
    }

    /// A draw from `N(0, 1)`.
    pub fn standard_normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.inner)
    }

    /// Fill `out` with independent `N(0, 1)` draws.
    pub fn fill_standard_normal(&mut self, out: &mut [f64]) {
        for slot in out.iter_mut() {
            *slot = self.standard_normal();
        }
    }

    /// A draw from `Gamma(shape, rate)`, in the *rate* (inverse-scale)
    /// parameterisation used throughout this crate.
    ///
    /// The rate convention is the one the conjugate updates are written in — a
    /// Gamma posterior for a precision accumulates `rate += sum of squares` — so
    /// converting once here is cheaper than converting at every call site and
    /// getting it wrong somewhere.
    pub fn gamma(&mut self, shape: f64, rate: f64) -> BayesResult<f64> {
        if !(shape.is_finite() && shape > 0.0) {
            return Err(BayesError::config(
                "gamma.shape",
                format!("must be finite and > 0, got {shape}"),
            ));
        }
        if !(rate.is_finite() && rate > 0.0) {
            return Err(BayesError::config(
                "gamma.rate",
                format!("must be finite and > 0, got {rate}"),
            ));
        }
        let dist = Gamma::new(shape, 1.0 / rate)
            .map_err(|e| BayesError::Internal(format!("gamma({shape}, rate={rate}): {e}")))?;
        Ok(dist.sample(&mut self.inner))
    }

    /// A draw from `ChiSquared(df)` expressed through [`BayesRng::gamma`].
    ///
    /// Used by the conjugate samplers, where `sigma^2 | data` is drawn as a scaled
    /// inverse chi-squared.
    pub fn chi_squared(&mut self, df: f64) -> BayesResult<f64> {
        self.gamma(df / 2.0, 0.5)
    }

    /// A draw from the uniform distribution on `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.inner.gen::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_and_chain_produce_identical_streams() {
        let mut a = BayesRng::for_chain(42, 0);
        let mut b = BayesRng::for_chain(42, 0);
        for _ in 0..1000 {
            assert_eq!(a.standard_normal().to_bits(), b.standard_normal().to_bits());
        }
    }

    /// The XOR trap: `seed ^ chain` makes (4, 1) and (5, 0) the same stream. Two
    /// chains that are secretly one chain drive R-hat to 1.0 and would certify a
    /// broken fit as converged.
    #[test]
    fn chains_are_independent_streams() {
        let mut c0 = BayesRng::for_chain(4, 1);
        let mut c1 = BayesRng::for_chain(5, 0);
        let a: Vec<u64> = (0..64).map(|_| c0.standard_normal().to_bits()).collect();
        let b: Vec<u64> = (0..64).map(|_| c1.standard_normal().to_bits()).collect();
        assert_ne!(a, b);

        // And the ordinary case: chains of one fit differ from each other.
        let mut x = BayesRng::for_chain(7, 0);
        let mut y = BayesRng::for_chain(7, 1);
        let xs: Vec<u64> = (0..64).map(|_| x.standard_normal().to_bits()).collect();
        let ys: Vec<u64> = (0..64).map(|_| y.standard_normal().to_bits()).collect();
        assert_ne!(xs, ys);
    }

    /// The same trap as `chains_are_independent_streams`, one level down: two groups
    /// sharing a stream would make two SKUs' posteriors identical, and a portfolio
    /// total built on them would have the wrong variance.
    #[test]
    fn groups_are_independent_streams() {
        let bits = |seed, chain, group| {
            let mut r = BayesRng::for_group(seed, chain, group);
            (0..64)
                .map(|_| r.standard_normal().to_bits())
                .collect::<Vec<_>>()
        };
        // Reproducible...
        assert_eq!(bits(7, 0, "SKU-1"), bits(7, 0, "SKU-1"));
        // ...and every coordinate matters on its own.
        assert_ne!(bits(7, 0, "SKU-1"), bits(7, 0, "SKU-2"));
        assert_ne!(bits(7, 0, "SKU-1"), bits(7, 1, "SKU-1"));
        assert_ne!(bits(7, 0, "SKU-1"), bits(8, 0, "SKU-1"));
        // Neighbouring keys must not share structure the way `seed ^ chain` does.
        assert_ne!(bits(7, 0, "SKU-0000001"), bits(7, 0, "SKU-0000002"));
    }

    /// Length-prefixing, checked rather than asserted in a comment: without it the
    /// key and the chain index would run together and `("a", 0x62..)` could be made
    /// to collide with `("ab", 0)`.
    #[test]
    fn a_group_key_cannot_bleed_into_the_chain_index() {
        let bits = |chain, group| {
            let mut r = BayesRng::for_group(0, chain, group);
            (0..8)
                .map(|_| r.standard_normal().to_bits())
                .collect::<Vec<_>>()
        };
        assert_ne!(bits(0, "ab"), bits(0x62, "a"));
    }

    #[test]
    fn standard_normal_draws_have_the_right_moments() {
        let mut rng = BayesRng::for_chain(1, 0);
        let n = 200_000;
        let draws: Vec<f64> = (0..n).map(|_| rng.standard_normal()).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        // MC standard error of the mean is 1/sqrt(n) ~ 0.0022; 5 SE is comfortable.
        assert!(mean.abs() < 0.012, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    /// Gamma is written in the rate parameterisation: mean = shape / rate,
    /// variance = shape / rate^2. Getting this backwards silently inflates or
    /// deflates every posterior variance drawn through it.
    #[test]
    fn gamma_uses_the_rate_parameterisation() {
        let mut rng = BayesRng::for_chain(2, 0);
        let (shape, rate) = (9.0, 3.0);
        let n = 200_000;
        let draws: Vec<f64> = (0..n).map(|_| rng.gamma(shape, rate).unwrap()).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!((mean - shape / rate).abs() < 0.02, "mean {mean} vs 3.0");
        assert!(
            (var - shape / (rate * rate)).abs() < 0.02,
            "variance {var} vs 1.0"
        );
    }

    #[test]
    fn chi_squared_has_mean_df_and_variance_two_df() {
        let mut rng = BayesRng::for_chain(3, 0);
        let df = 7.0;
        let n = 200_000;
        let draws: Vec<f64> = (0..n).map(|_| rng.chi_squared(df).unwrap()).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!((mean - df).abs() < 0.05, "mean {mean} vs {df}");
        assert!(
            (var - 2.0 * df).abs() < 0.35,
            "variance {var} vs {}",
            2.0 * df
        );
    }

    #[test]
    fn degenerate_gamma_parameters_are_rejected_rather_than_producing_nan() {
        let mut rng = BayesRng::for_chain(4, 0);
        assert!(rng.gamma(0.0, 1.0).is_err());
        assert!(rng.gamma(-1.0, 1.0).is_err());
        assert!(rng.gamma(1.0, 0.0).is_err());
        assert!(rng.gamma(f64::NAN, 1.0).is_err());
        assert!(rng.gamma(1.0, f64::INFINITY).is_err());
    }
}
