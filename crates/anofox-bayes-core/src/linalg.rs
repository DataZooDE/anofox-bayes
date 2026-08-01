//! The small amount of linear algebra the conjugate linear model needs.
//!
//! Two operations, both on symmetric positive-definite matrices:
//!
//! * [`cholesky`] — the factor `L` with `L L' = A`, used both to solve the normal
//!   equations and to sample a multivariate normal.
//! * [`solve_with`] / [`sample_mvn`] — the two things that factor is then used for.
//!
//! Sharing the factor between the solve and the sampler is not only an optimisation.
//! Drawing `beta ~ N(b_n, sigma^2 V_n)` needs a factor of `V_n = (X'X + P)^-1`, and the
//! factor of the *precision* is exactly what solving the normal equations already
//! produced: `beta = b_n + sigma * L^-T z` for `z ~ N(0, I)`. Forming `V_n` explicitly
//! and factoring it again would cost an inversion, and inverting a matrix to then
//! factor it is the classic way to lose the conditioning you started with.

use faer::Mat;

use crate::errors::{BayesError, BayesResult};
use crate::rng::BayesRng;

/// Lower-triangular Cholesky factor `L` of a symmetric positive-definite `a`,
/// returned as a dense lower triangle.
///
/// Fails rather than returning a partial factor when `a` is not positive definite —
/// which, for a normal-equations matrix, means the design is rank deficient. That is
/// a modelling problem the caller must be told about, not one to paper over with a
/// pseudo-inverse that would silently pick one of infinitely many answers.
pub fn cholesky(a: &Mat<f64>) -> BayesResult<Mat<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(BayesError::DimensionMismatch(format!(
            "cholesky needs a square matrix, got {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }
    let mut l: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                // Relative, not absolute. Exact collinearity -- a duplicated
                // predictor, or a constant column beside an intercept -- leaves a
                // pivot that is mathematically zero but lands a few ulps above it in
                // floating point. An absolute `sum > 0` check waves those through and
                // produces a wildly ill-conditioned factor that then reports absurd
                // coefficients with narrow intervals. Scaling the threshold by the
                // original diagonal makes the test independent of the units the data
                // happens to be measured in.
                let tolerance = 1e-12 * a[(i, i)].abs().max(1.0);
                if !(sum > tolerance) || !sum.is_finite() {
                    return Err(BayesError::NotPositiveDefinite(format!(
                        "leading minor {} is {sum}, at or below the rank tolerance {tolerance}",
                        i + 1
                    )));
                }
                l[(i, j)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }
    Ok(l)
}

/// Solve `A x = b` given the Cholesky factor `L` of `A`, by forward then back
/// substitution.
pub fn solve_with(l: &Mat<f64>, b: &[f64]) -> BayesResult<Vec<f64>> {
    let n = l.nrows();
    if b.len() != n {
        return Err(BayesError::DimensionMismatch(format!(
            "right-hand side has {} entries, factor is {n}x{n}",
            b.len()
        )));
    }
    // Forward: L y = b.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[(i, k)] * y[k];
        }
        y[i] = sum / l[(i, i)];
    }
    // Back: L' x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[(k, i)] * x[k];
        }
        x[i] = sum / l[(i, i)];
    }
    Ok(x)
}

/// Draw from `N(mean, scale^2 * A^-1)` given the Cholesky factor `L` of the
/// *precision* `A`.
///
/// The draw is `mean + scale * L^-T z`, which needs only a back substitution — no
/// inversion, and no second factorisation of the covariance.
pub fn sample_mvn(
    l: &Mat<f64>,
    mean: &[f64],
    scale: f64,
    rng: &mut BayesRng,
    out: &mut [f64],
) -> BayesResult<()> {
    let n = l.nrows();
    if mean.len() != n || out.len() != n {
        return Err(BayesError::DimensionMismatch(format!(
            "mean has {} and output {} entries, factor is {n}x{n}",
            mean.len(),
            out.len()
        )));
    }
    // z ~ N(0, I), then solve L' w = z in place.
    for slot in out.iter_mut() {
        *slot = rng.standard_normal();
    }
    for i in (0..n).rev() {
        let mut sum = out[i];
        for k in (i + 1)..n {
            sum -= l[(k, i)] * out[k];
        }
        out[i] = sum / l[(i, i)];
    }
    for i in 0..n {
        out[i] = mean[i] + scale * out[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize) -> Mat<f64> {
        // A'A + n*I is symmetric positive definite for any A.
        let a = Mat::from_fn(n, n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
        let mut m: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[(k, i)] * a[(k, j)];
                }
                m[(i, j)] = s + if i == j { n as f64 } else { 0.0 };
            }
        }
        m
    }

    #[test]
    fn the_cholesky_factor_multiplies_back_to_the_original_matrix() {
        let a = spd(5);
        let l = cholesky(&a).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                let mut s = 0.0;
                for k in 0..5 {
                    s += l[(i, k)] * l[(j, k)];
                }
                assert!(
                    (s - a[(i, j)]).abs() < 1e-9,
                    "({i},{j}): {s} vs {}",
                    a[(i, j)]
                );
            }
            // The factor really is lower triangular.
            for j in (i + 1)..5 {
                assert_eq!(l[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn solving_recovers_the_vector_the_system_was_built_from() {
        let a = spd(4);
        let x_true = [1.5, -2.0, 0.25, 3.0];
        let b: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|j| a[(i, j)] * x_true[j]).sum())
            .collect();

        let l = cholesky(&a).unwrap();
        let x = solve_with(&l, &b).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-9,
                "{}: {} vs {}",
                i,
                x[i],
                x_true[i]
            );
        }
    }

    /// A rank-deficient design produces a singular normal-equations matrix. Failing
    /// here is the point: a pseudo-inverse would silently pick one of infinitely many
    /// answers and report it with a straight face.
    #[test]
    fn a_rank_deficient_matrix_fails_rather_than_being_pseudo_inverted() {
        // Second column is twice the first, so X'X is singular.
        let x = Mat::from_fn(4, 2, |i, j| {
            if j == 0 {
                (i + 1) as f64
            } else {
                2.0 * (i + 1) as f64
            }
        });
        let mut xtx: Mat<f64> = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                xtx[(i, j)] = (0..4).map(|k| x[(k, i)] * x[(k, j)]).sum();
            }
        }
        let err = cholesky(&xtx).unwrap_err();
        assert!(matches!(err, BayesError::NotPositiveDefinite(_)), "{err}");
    }

    #[test]
    fn a_matrix_containing_a_nan_fails_rather_than_propagating_it() {
        let mut a = spd(3);
        a[(1, 1)] = f64::NAN;
        assert!(cholesky(&a).is_err());
    }

    /// The sampler's empirical covariance must reproduce `scale^2 * A^-1`. This is
    /// the check that the back-substitution really inverts the factor rather than
    /// applying it.
    #[test]
    fn multivariate_draws_have_the_covariance_the_precision_implies() {
        let a = spd(3);
        let l = cholesky(&a).unwrap();
        let mean = [1.0, -2.0, 0.5];
        let scale = 2.0;

        let n = 200_000;
        let mut rng = BayesRng::for_chain(11, 0);
        let mut draws = vec![[0.0f64; 3]; n];
        let mut buf = [0.0; 3];
        for slot in draws.iter_mut() {
            sample_mvn(&l, &mean, scale, &mut rng, &mut buf).unwrap();
            *slot = buf;
        }

        // Empirical mean matches.
        for j in 0..3 {
            let m = draws.iter().map(|d| d[j]).sum::<f64>() / n as f64;
            assert!((m - mean[j]).abs() < 0.05, "mean {j}: {m} vs {}", mean[j]);
        }

        // Empirical covariance matches scale^2 * A^-1, computed independently by
        // solving A c_j = e_j one column at a time.
        for j in 0..3 {
            let mut e = vec![0.0; 3];
            e[j] = 1.0;
            let col = solve_with(&l, &e).unwrap();
            for i in 0..3 {
                let mi = draws.iter().map(|d| d[i]).sum::<f64>() / n as f64;
                let mj = draws.iter().map(|d| d[j]).sum::<f64>() / n as f64;
                let cov =
                    draws.iter().map(|d| (d[i] - mi) * (d[j] - mj)).sum::<f64>() / (n - 1) as f64;
                let expected = scale * scale * col[i];
                assert!(
                    (cov - expected).abs() < 0.05 * expected.abs().max(0.5),
                    "cov({i},{j}) = {cov} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn dimension_mismatches_are_typed_errors_rather_than_panics() {
        let l = cholesky(&spd(3)).unwrap();
        assert!(solve_with(&l, &[1.0, 2.0]).is_err());
        let mut out = [0.0; 2];
        let mut rng = BayesRng::for_chain(1, 0);
        assert!(sample_mvn(&l, &[1.0, 2.0, 3.0], 1.0, &mut rng, &mut out).is_err());
        assert!(cholesky(&Mat::from_fn(2, 3, |_, _| 1.0)).is_err());
    }
}
