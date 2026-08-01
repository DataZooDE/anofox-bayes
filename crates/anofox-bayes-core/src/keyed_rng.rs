//! Stateless, keyed randomness for the predictive step.
//!
//! [`BayesRng`](crate::rng::BayesRng) makes a *fit* reproducible. This module makes
//! everything the caller does **downstream** of a fit reproducible, which is a
//! different problem with a different answer.
//!
//! The recipes in `GUIDE.md` all have the same shape: join a draws table to some rows,
//! add noise, aggregate. The noise came from DuckDB's `random()`, and `random()` is
//! seeded per session by `setseed()` — not by the fit. So the fit was reproducible and
//! the recommendation built on it was not. Measured on this repo before this module
//! existed: one fit, `model_id b4557ce9268692d3` every time, and two runs of the same
//! predictive query over its draws returning 0.4224 and 0.5518.
//!
//! `setseed()` patches that, and it is the wrong shape of fix. It makes correctness
//! depend on session-global state the caller might not have set, in an extension whose
//! entire design premise is that it keeps no state of its own. A recipe that is only
//! right when someone remembered a `SET` is a recipe that will silently be wrong
//! somewhere.
//!
//! # The design
//!
//! A **counter-based** generator: the value is a pure function of its coordinates in
//! the random stream, with no state carried between calls.
//!
//! ```text
//! uniform(seed, key, draw) = BLAKE3(seed ‖ key ‖ draw) -> (0, 1)
//! ```
//!
//! Three properties follow, and all three are what the fix actually needs:
//!
//! * **Order independence.** DuckDB may evaluate rows in any order, on any number of
//!   threads, and may re-evaluate a row after a spill. A stateful RNG gives a different
//!   answer under each; this gives the same one, because a row's noise is a function of
//!   the row and not of how many rows preceded it.
//! * **Composability.** Two queries over the same draws that touch the same
//!   `(key, draw)` see the same noise, so a scenario and its baseline share their
//!   randomness and their *difference* is the effect rather than the effect plus
//!   sampling jitter. That is what makes a paired what-if comparison meaningful.
//! * **Auditability.** Recording the seed beside `model_id` is enough to regenerate a
//!   recommendation exactly.
//!
//! # Inverse-CDF, not Box-Muller
//!
//! [`std_normal`] is [`uniform`] pushed through the standard normal quantile function,
//! so `std_normal(s, k, d) == qnorm(uniform(s, k, d))` exactly — pinned by
//! [`tests::the_normal_is_exactly_the_quantile_of_the_uniform`].
//!
//! That identity is the point rather than an implementation detail. It makes [`uniform`]
//! the single primitive: a caller needing a lognormal, an exponential, a Poisson or
//! anything else applies its own quantile function to the same stream and inherits
//! every property above. Box-Muller would consume two uniforms per normal and give the
//! caller no such handle, and its `ln(u1)` is `-inf` when `u1` is zero — a footgun this
//! module avoids by construction rather than by remembering to clamp.

use statrs::distribution::{ContinuousCDF, Normal};

/// Field separator for the hash input.
///
/// The three coordinates are length-prefixed before hashing for the same reason
/// `model_id` length-prefixes its fields: without it `("ab", 1)` and `("a", 0x62...)`
/// could be made to collide by choosing key material that absorbs the boundary.
fn digest(seed: i64, key: &[u8], draw: i64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(&(key.len() as u64).to_le_bytes());
    hasher.update(key);
    hasher.update(&draw.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// A draw from `Uniform(0, 1)`, open at both ends.
///
/// Open at zero specifically: a caller writing `ln(u)` for an exponential, or
/// `-2 * ln(u)` for a Box-Muller normal of their own, gets `-inf` on a zero and a NULL
/// or a NaN somewhere downstream. The construction below cannot produce one.
pub fn uniform(seed: i64, key: &[u8], draw: i64) -> f64 {
    let bytes = digest(seed, key, draw);
    unit_interval(u64::from_le_bytes(
        bytes[..8].try_into().expect("32-byte digest"),
    ))
}

/// Map 64 random bits onto the **open** interval `(0, 1)`.
///
/// Split out from [`uniform`] so the openness can be tested at the boundaries rather
/// than sampled for. Sampling cannot find this bug: the half-open `[0, 1)` spelling
/// differs only on `raw >> 12 == 0`, which a random search reaches once in 2^52 draws
/// and a test would therefore always pass against.
fn unit_interval(raw: u64) -> f64 {
    // 52 bits, not the full 53-bit mantissa, and the missing bit is deliberate.
    //
    // With 53 bits the top value is 2^53 - 0.5, which is *not* representable — in
    // [2^52, 2^53) the spacing is 1.0, so it rounds to 2^53 and the function returns
    // exactly 1.0, reintroducing the closed endpoint this exists to avoid. At 52 bits
    // the spacing below 2^52 is 0.5, the half step survives exactly, and the extremes
    // are 2^-53 and 1 - 2^-53. One bit of entropy is not a currency this needs.
    ((raw >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
}

/// A draw from `N(0, 1)`.
///
/// Equal to the standard normal quantile of [`uniform`] at the same coordinates.
pub fn std_normal(seed: i64, key: &[u8], draw: i64) -> f64 {
    standard_normal_quantile(uniform(seed, key, draw))
}

/// The standard normal quantile function, `qnorm`.
///
/// Exposed because it is half of the contract [`std_normal`] advertises, and because a
/// caller building a lognormal wants the same function applied to the same uniform.
pub fn standard_normal_quantile(u: f64) -> f64 {
    Normal::standard().inverse_cdf(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniforms for `draw = 0..n` under one seed and key.
    fn uniform_stream(seed: i64, key: &str, n: i64) -> Vec<f64> {
        (0..n).map(|d| uniform(seed, key.as_bytes(), d)).collect()
    }

    fn normal_stream(seed: i64, key: &str, n: i64) -> Vec<f64> {
        (0..n)
            .map(|d| std_normal(seed, key.as_bytes(), d))
            .collect()
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let mut num = 0.0;
        let (mut da, mut db) = (0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            num += (x - ma) * (y - mb);
            da += (x - ma).powi(2);
            db += (y - mb).powi(2);
        }
        num / (da.sqrt() * db.sqrt())
    }

    /// Kolmogorov-Smirnov distance between a sample and a reference CDF.
    ///
    /// A real distributional test. Mean and variance alone are a weak gate — a
    /// generator emitting only ±1 has mean 0 and variance 1 and is not remotely
    /// normal.
    fn ks_distance(sample: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len() as f64;
        sorted
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let f = cdf(x);
                let below = i as f64 / n;
                let above = (i as f64 + 1.0) / n;
                (f - below).abs().max((above - f).abs())
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn the_same_coordinates_always_give_the_same_value() {
        assert_eq!(std_normal(42, b"sku-7", 3), std_normal(42, b"sku-7", 3));
        assert_eq!(uniform(42, b"sku-7", 3), uniform(42, b"sku-7", 3));
    }

    #[test]
    fn each_coordinate_changes_the_value_independently() {
        let base = std_normal(42, b"sku-7", 3);
        assert_ne!(base, std_normal(43, b"sku-7", 3), "seed must matter");
        assert_ne!(base, std_normal(42, b"sku-8", 3), "key must matter");
        assert_ne!(base, std_normal(42, b"sku-7", 4), "draw must matter");
    }

    /// The whole reason this module exists: no state, so no evaluation order.
    #[test]
    fn the_value_does_not_depend_on_what_was_evaluated_before_it() {
        let forwards = normal_stream(7, "lane-a", 500);
        let backwards: Vec<f64> = (0..500)
            .rev()
            .map(|d| std_normal(7, b"lane-a", d))
            .collect();
        let mut backwards = backwards;
        backwards.reverse();
        assert_eq!(forwards, backwards);
    }

    /// `ln(0)` is the failure this guarantees against.
    ///
    /// Tested at the boundaries of the bit-to-float map rather than by sampling
    /// `uniform`, because the bug lives on exactly one of 2^53 inputs and no feasible
    /// random search reaches it — a sampled version of this test passes against the
    /// broken implementation.
    #[test]
    fn the_unit_interval_map_is_open_at_both_ends() {
        for raw in [0u64, 1, 4095, 4096, u64::MAX - 1, u64::MAX] {
            let u = unit_interval(raw);
            assert!(u > 0.0, "unit_interval({raw}) reached zero");
            assert!(u < 1.0, "unit_interval({raw}) reached one");
            assert!(u.ln().is_finite(), "ln({u}) is not finite");
            assert!((1.0 - u).ln().is_finite(), "ln(1 - {u}) is not finite");
        }
    }

    #[test]
    fn the_unit_interval_map_is_monotone_and_lossless() {
        // Distinct 53-bit inputs must give distinct outputs; a shift or scale that
        // rounds two apart into one would bias the stream invisibly.
        assert!(unit_interval(0) < unit_interval(4096));
        assert!(unit_interval(4096) < unit_interval(8192));
        assert_ne!(unit_interval(0), unit_interval(4096));
    }

    #[test]
    fn the_uniform_stays_inside_the_open_interval() {
        for seed in 0..2_000i64 {
            let u = uniform(seed, b"", seed);
            assert!(u > 0.0 && u < 1.0, "uniform outside (0,1): {u}");
        }
    }

    #[test]
    fn the_uniform_is_uniform() {
        let sample = uniform_stream(11, "lane-a", 20_000);
        let d = ks_distance(&sample, |x| x);
        // 1.63/sqrt(n) is the 1% critical value of the KS statistic; at n = 20 000
        // that is 0.0115. A generator with visible structure blows past it.
        assert!(d < 0.0115, "KS distance from Uniform(0,1) too large: {d}");
    }

    #[test]
    fn the_normal_is_normal() {
        let sample = normal_stream(11, "lane-a", 20_000);
        let normal = Normal::standard();
        let d = ks_distance(&sample, |x| normal.cdf(x));
        assert!(d < 0.0115, "KS distance from N(0,1) too large: {d}");
    }

    #[test]
    fn the_normal_is_exactly_the_quantile_of_the_uniform() {
        // Documented as an identity callers may rely on to build other distributions.
        for d in 0..1_000 {
            let u = uniform(3, b"sku-1", d);
            assert_eq!(std_normal(3, b"sku-1", d), standard_normal_quantile(u));
        }
    }

    /// Successive draws must be independent, not merely marginally correct.
    #[test]
    fn successive_draws_are_uncorrelated() {
        let s = normal_stream(5, "lane-a", 20_000);
        let r = pearson(&s[..s.len() - 1], &s[1..]);
        assert!(r.abs() < 0.02, "lag-1 autocorrelation too large: {r}");
    }

    /// The bug this catches: a hash that mixes badly leaves neighbouring keys sharing
    /// structure, so two SKUs fitted side by side would get correlated noise and a
    /// portfolio-level total would have the wrong variance.
    #[test]
    fn neighbouring_keys_produce_uncorrelated_streams() {
        let a = normal_stream(5, "sku-0000001", 20_000);
        let b = normal_stream(5, "sku-0000002", 20_000);
        let r = pearson(&a, &b);
        assert!(r.abs() < 0.02, "neighbouring keys correlated: {r}");
    }

    /// Same, for neighbouring seeds — the failure mode `BayesRng` documents for
    /// `seed ^ chain`, checked here for this generator too.
    #[test]
    fn neighbouring_seeds_produce_uncorrelated_streams() {
        let a = normal_stream(1000, "lane-a", 20_000);
        let b = normal_stream(1001, "lane-a", 20_000);
        assert!(pearson(&a, &b).abs() < 0.02);
    }

    #[test]
    fn the_normal_reaches_both_tails() {
        // An inverse-CDF generator truncating its uniform would quietly lose the tails,
        // and a service level read off a truncated posterior predictive is wrong in the
        // exact place it is being asked about.
        let s = normal_stream(9, "lane-a", 100_000);
        assert!(
            s.iter().cloned().fold(f64::MIN, f64::max) > 3.5,
            "no upper tail"
        );
        assert!(
            s.iter().cloned().fold(f64::MAX, f64::min) < -3.5,
            "no lower tail"
        );
    }

    #[test]
    fn an_empty_key_is_usable() {
        // The natural spelling when the noise is per-draw rather than per-row.
        assert!(std_normal(1, b"", 0).is_finite());
        assert_ne!(std_normal(1, b"", 0), std_normal(1, b"", 1));
    }

    /// Length-prefixing, checked rather than asserted in a comment.
    #[test]
    fn key_boundaries_cannot_be_confused() {
        assert_ne!(uniform(0, b"ab", 0), uniform(0, b"a", 0));
        assert_ne!(uniform(0, b"abc", 1), uniform(0, b"ab", 1));
    }
}
