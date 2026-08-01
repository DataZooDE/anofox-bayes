"""F2 `censored_aft` vs PyMC.

**This family is checked twice, against two different references, because one
comparison cannot answer both of the questions it raises.**

`censored_aft` has no closed form and its only engine is Laplace: a Gaussian
fitted at the posterior mode with the observed information as its precision.
THEORY.md is explicit that "`laplace` is not a cross-check on it -- it is the
fit". So a single comparison against NUTS would confound two entirely different
things: whether the likelihood is right, and whether a Gaussian at the mode is a
good enough stand-in for the posterior. Those need separating, because the first
is a bug and the second is a documented design decision.

1. **The algebra gate** (`F2_EXACT_TOL`, ~0.02 sd). The extension's draws
   against a mode and an observed information matrix computed in
   `_support.laplace_reference` -- Nelder-Mead then BFGS on a log posterior
   written out separately in NumPy, with the Hessian obtained by differencing a
   *numerical* gradient. The extension uses damped Newton on an *analytic*
   gradient and differences that. Two implementations sharing no code. If both
   describe the same Gaussian, the likelihood, the censoring, the flat priors
   and the absence of a `log sigma` Jacobian are all confirmed at once. This is
   the test that would catch a wrong constant.

2. **The approximation budget** (`F2_TOL`). The same draws against NUTS on the
   true posterior. This cannot be tight and is not meant to be: it measures how
   far the Laplace approximation is from the thing it approximates.

The measured answer to (1) is that the bridge is **exact** -- see
`test_the_laplace_posterior_is_the_one_the_likelihood_implies`, which agrees to
0.003 sd and reproduces the intercept/slope correlation to four digits. The
measured answer to (2) is that `sigma` is systematically low and narrow; that is
recorded and bounded in `test_the_scale_carries_the_approximation_error` rather
than absorbed into a loosened tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_LAPLACE_DRAWS,
    EXTENSION_SEED,
    F2_EXACT_TOL,
    F2_LANES,
    F2_TOL,
    assert_parity,
    extension_summary,
    f2_dataset,
    f2_neg_log_posterior,
    f2_panel_dataset,
    f2_start,
    fit,
    fit_metadata,
    gaussian_summaries,
    laplace_reference,
    log_scale_summary,
    parity_deltas,
    parity_failures,
    pymc_f2_weibull,
    reference_summaries,
)

X_NAME = "distance_100km"

# The extension's unconstrained coordinates, in order.
COORDINATES = ["intercept", f"beta[{X_NAME}]", "log_sigma"]

CONFIG = (
    "{'time': 'days', 'event': 'delivered', 'x': 'distance_100km', 'dist': 'weibull', "
    f"'draws': {EXTENSION_LAPLACE_DRAWS}, 'seed': {EXTENSION_SEED}}}"
)
COLUMNS = "distance_100km, days, delivered"


@pytest.fixture(scope="session")
def f2_data():
    return f2_dataset()


@pytest.fixture(scope="session")
def f2_extension(con, f2_data):
    return fit(con, f2_data, COLUMNS, "censored_aft", CONFIG, "f2_obs")


@pytest.fixture(scope="session")
def f2_metadata(con, f2_data):
    return fit_metadata(con, f2_data, COLUMNS, "censored_aft", CONFIG, "f2_meta")


@pytest.fixture(scope="session")
def f2_laplace(f2_data):
    """The independent mode and observed information, as marginal summaries."""
    neg_log_posterior, _ = f2_neg_log_posterior(
        f2_data["days"], f2_data["delivered"], f2_data[[X_NAME]].to_numpy()
    )
    mode, cov = laplace_reference(neg_log_posterior, f2_start(f2_data["days"]))
    return gaussian_summaries(mode, cov, COORDINATES), mode, cov


@pytest.fixture(scope="session")
def f2_nuts(f2_data):
    return reference_summaries(
        pymc_f2_weibull(
            f2_data["days"], f2_data["delivered"], f2_data[[X_NAME]].to_numpy(), [X_NAME]
        ),
        ["intercept", f"beta[{X_NAME}]", "sigma", "log_sigma"],
    )


def _extension_on_coordinates(draws, group_id="__global__"):
    """The extension's draws on the unconstrained scale the Laplace fit lives on.

    `sigma` is reported on the natural scale, having been sampled as
    `log sigma`, so taking the log back is what puts it on the coordinate the
    approximation is exactly Gaussian on. Comparing natural-scale `sigma` to a
    Gaussian would be comparing a lognormal to a normal and failing for a reason
    that has nothing to do with the extension.
    """
    return {
        "intercept": extension_summary(draws, "intercept", group_id=group_id),
        f"beta[{X_NAME}]": extension_summary(draws, f"beta[{X_NAME}]", group_id=group_id),
        "log_sigma": log_scale_summary(draws, "sigma", group_id=group_id),
    }


def test_the_fit_converged_and_was_served_by_laplace(f2_metadata):
    """`__engine__ = 1` is `laplace`, and it is the only engine this family has.

    `exact` is refused outright. A fit that arrived by some other route would
    make every comparison below describe a different estimator.
    """
    assert f2_metadata["__status__"] == 0.0, "the F2 fit did not report `converged`"
    assert f2_metadata["__engine__"] == 1.0, "the F2 fit was not served by the Laplace engine"
    assert f2_metadata["__n_obs__"] == 180.0


def test_the_fixture_actually_exercises_censoring(f2_data):
    """A book with nothing still in transit tests the uncensored branch twice.

    The censored branch contributes `log S(z)` and nothing else -- no `- log t`,
    no `- log sigma` -- so a fixture without it would leave the entire censoring
    path unvalidated while looking exactly as green.
    """
    censored = int((f2_data["delivered"] == 0).sum())
    assert 30 <= censored <= 90, (
        f"{censored} of {len(f2_data)} shipments are censored; the fixture is meant to "
        "exercise the censored branch substantially"
    )


# --------------------------------------------------------------------------
# 1. The algebra gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("coordinate", COORDINATES)
def test_the_laplace_posterior_is_the_one_the_likelihood_implies(
    f2_extension, f2_laplace, coordinate
):
    """The extension's draws are `N(mode, (-H)^-1)` for an independently found mode.

    Nothing about this comparison involves an approximation: both sides are
    describing the same Gaussian, so the only irreducible error is the
    extension's Monte Carlo on 200_000 i.i.d. draws (0.002 sd on a mean) plus
    the finite-difference Hessian. A breach here is an arithmetic error in the
    likelihood, the censoring, or the prior -- not a modelling judgement.
    """
    summaries, _, _ = f2_laplace
    assert_parity(
        f"{coordinate} (vs Laplace)",
        _extension_on_coordinates(f2_extension)[coordinate],
        summaries[coordinate],
        F2_EXACT_TOL,
    )


def test_the_full_covariance_is_used_and_not_just_its_diagonal(f2_extension, f2_laplace):
    """The off-diagonal is the whole ballgame, and it is the one that can vanish.

    The upstream fit publishes only standard errors. Sampling from a diagonal
    would treat the intercept and the slope as independent when they are almost
    perfectly anti-correlated in a duration model with a covariate measured away
    from zero -- their errors cancel in the linear predictor, and THEORY.md
    measures the predictive sd computed from the diagonal as ~25x the one from
    the full matrix. Both answers are finite, both pass every diagnostic, and
    only one is the posterior. A `sd`-only comparison cannot tell them apart:
    the marginals are identical either way.
    """
    _, _, cov = f2_laplace
    sd = np.sqrt(np.diag(cov))
    expected = cov[0, 1] / (sd[0] * sd[1])

    intercept = f2_extension[f2_extension["param"] == "intercept"]["value"].to_numpy()
    slope = f2_extension[f2_extension["param"] == f"beta[{X_NAME}]"]["value"].to_numpy()
    observed = float(np.corrcoef(intercept, slope)[0, 1])

    assert expected < -0.9, (
        f"the fixture's intercept/slope correlation is only {expected:.3f}; it is not "
        "strong enough for this test to distinguish a full covariance from a diagonal"
    )
    # 0.005 is ~2x the Monte Carlo error on a correlation this strong at
    # 200_000 draws, which is (1 - rho^2)/sqrt(N) ~ 0.0002, plus room for the
    # finite-difference Hessian.
    assert abs(observed - expected) < 0.005, (
        f"draws correlation {observed:.4f} against the observed information's "
        f"{expected:.4f}. A diagonal would give ~0."
    )


# --------------------------------------------------------------------------
# 2. The approximation budget
# --------------------------------------------------------------------------


@pytest.mark.parametrize("coordinate", ["intercept", f"beta[{X_NAME}]"])
def test_coefficient_parity(f2_extension, f2_nuts, coordinate):
    """The coefficients against NUTS on the true posterior.

    These pass at a tolerance derived from the *measured* Laplace bias rather
    than from Monte Carlo -- see TOLERANCES.md. A regression coefficient's
    posterior is close to Gaussian at n = 180, which is exactly why the same
    tolerance cannot be asked of the scale.
    """
    assert_parity(
        coordinate,
        _extension_on_coordinates(f2_extension)[coordinate],
        f2_nuts[coordinate],
        F2_TOL,
    )


def test_the_scale_carries_the_approximation_error(f2_extension, f2_nuts, f2_laplace):
    """`sigma` is systematically low and narrow, and this pins how much.

    Measured on this fixture, the extension's `log sigma` posterior sits about
    0.19 reference sd below NUTS's and is about 2.4 % narrower; on the natural
    scale that is `E[sigma]` low by ~1.5 % with an interval ~4 % too tight.

    **This is not a bridge error.** The algebra gate above shows the extension
    reproduces the mode and the observed information of the same log posterior
    to 0.003 sd. What is left is the definition of Laplace: it reports the
    *mode*, and the mode of a scale parameter's posterior is below its mean
    because that posterior is right-skewed. So the offset is the skew, and the
    narrowness is the Gaussian's missing upper tail.

    It is recorded rather than absorbed because the direction matters. `sigma`
    is what sets the width of a transit-time quantile, so a low `sigma` is a
    delivery promise that is tighter than the data supports -- over-confident,
    never under, the same direction as the F3 defect this suite found earlier
    and for a completely different reason.

    The bounds are loose enough not to flake on Monte Carlo (the reference's own
    MCSE on a mean here is ~0.012 sd) and far tighter than `F2_TOL`, so a change
    in the quality of the approximation is caught even though the parity
    tolerance could not see it.
    """
    extension = log_scale_summary(f2_extension, "sigma")
    reference = f2_nuts["log_sigma"]

    offset = (extension.mean - reference.mean) / reference.sd
    ratio = extension.sd / reference.sd

    assert -0.30 < offset < -0.08, (
        f"E[log sigma] sits {offset:+.4f} reference sd from NUTS, outside the "
        "[-0.30, -0.08] band this fixture has been measured at. Laplace reports the "
        "mode of a right-skewed posterior, so the offset must be negative; a move "
        "toward zero means the engine changed, and a move past -0.30 means the "
        "approximation degraded."
    )
    assert 0.94 < ratio < 1.0, (
        f"sd(log sigma) ratio extension/NUTS = {ratio:.4f}. A Gaussian at the mode "
        "cannot be wider than the skewed posterior it approximates, so a ratio above "
        "1 would mean something other than Laplace produced these draws."
    )


def test_the_coefficients_carry_far_less_of_it_than_the_scale(f2_extension, f2_nuts):
    """The approximation is good where the posterior is near-Gaussian, and this says so.

    THEORY.md's claim is that a regression coefficient's posterior is close to
    Gaussian at modest sample sizes while a variance parameter's is not, and
    that this is why the thin-cohort AFT result is good where F7's is bad. That
    is a testable statement about *this* fit, not a general remark, so it is
    tested: the scale's approximation error must be several times the
    coefficients'.
    """
    on_coordinates = _extension_on_coordinates(f2_extension)
    coefficient_error = max(
        abs(on_coordinates[c].mean - f2_nuts[c].mean) / f2_nuts[c].sd
        for c in ("intercept", f"beta[{X_NAME}]")
    )
    scale_error = abs(
        log_scale_summary(f2_extension, "sigma").mean - f2_nuts["log_sigma"].mean
    ) / f2_nuts["log_sigma"].sd

    assert scale_error > 2.0 * coefficient_error, (
        f"the scale's Laplace error is {scale_error:.4f} sd and the worst coefficient's "
        f"{coefficient_error:.4f} sd. If those have converged, the claim that the "
        "approximation is good precisely where the posterior is near-Gaussian no "
        "longer describes this family."
    )


# --------------------------------------------------------------------------
# Grouped: one wholly independent fit per lane
# --------------------------------------------------------------------------
#
# Checked against the algebra gate only. A group is an independent fit of the
# same likelihood, so NUTS would re-measure the same approximation error three
# times over; what is worth testing is that each lane's fit sees its own rows
# and only its own rows, and the exact reference tests that far more sharply.


PANEL_CONFIG = (
    "{'time': 'days', 'event': 'delivered', 'x': 'distance_100km', 'group': 'lane', "
    f"'dist': 'weibull', 'draws': {EXTENSION_LAPLACE_DRAWS}, 'seed': {EXTENSION_SEED}}}"
)
PANEL_COLUMNS = "lane, distance_100km, days, delivered"


@pytest.fixture(scope="session")
def f2_panel_data():
    return f2_panel_dataset()


@pytest.fixture(scope="session")
def f2_panel_extension(con, f2_panel_data):
    return fit(con, f2_panel_data, PANEL_COLUMNS, "censored_aft", PANEL_CONFIG, "f2_panel")


@pytest.fixture(scope="session")
def f2_panel_laplace(f2_panel_data):
    """One independent Laplace fit per lane, from that lane's rows alone."""
    out = {}
    for lane in F2_LANES:
        rows = f2_panel_data[f2_panel_data["lane"] == lane]
        neg_log_posterior, _ = f2_neg_log_posterior(
            rows["days"], rows["delivered"], rows[[X_NAME]].to_numpy()
        )
        mode, cov = laplace_reference(neg_log_posterior, f2_start(rows["days"]))
        out[lane] = gaussian_summaries(mode, cov, COORDINATES)
    return out


@pytest.mark.parametrize("lane", F2_LANES)
@pytest.mark.parametrize("coordinate", COORDINATES)
def test_each_lane_is_its_own_fit(f2_panel_extension, f2_panel_laplace, lane, coordinate):
    """Every lane matches a reference built from that lane's rows and no others.

    `censored_aft` does no pooling: a thin lane borrows no strength from a thick
    one. That makes this a sharp test of the row subsetting -- a fit that leaked
    even a few of another lane's shipments into the design would move the mode
    well outside a 0.02 sd tolerance, and the lanes are built with different
    levels and different spreads so that such a leak has somewhere to show.
    """
    assert_parity(
        f"{coordinate}[{lane}] (vs Laplace)",
        _extension_on_coordinates(f2_panel_extension, group_id=lane)[coordinate],
        f2_panel_laplace[lane][coordinate],
        F2_EXACT_TOL,
    )


def test_the_lanes_are_actually_different(f2_panel_laplace):
    """If the three lanes had the same posterior, the previous test proves nothing."""
    intercepts = [f2_panel_laplace[lane]["intercept"].mean for lane in F2_LANES]
    spread = max(intercepts) - min(intercepts)
    typical_sd = np.mean([f2_panel_laplace[lane]["intercept"].sd for lane in F2_LANES])
    assert spread > 3.0 * typical_sd, (
        f"the lanes' intercepts span {spread:.3f} against a typical posterior sd of "
        f"{typical_sd:.3f}; they are not distinct enough for a cross-lane leak to show"
    )


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------


def test_ignoring_the_censoring_is_detected(con, f2_data, f2_laplace):
    """Declare every shipment delivered, which is what dropping the indicator does.

    This is the specific mistake the family exists to prevent: the shipments
    still moving are the slow ones, so treating a censoring time as a delivery
    time biases every lane fast. The values in the `days` column are completely
    unchanged -- only the event indicator moves -- so nothing about the marginal
    distribution of the durations gives it away.
    """
    naive = f2_data.copy()
    naive["delivered"] = 1
    draws = fit(con, naive, COLUMNS, "censored_aft", CONFIG, "f2_naive")

    summaries, _, _ = f2_laplace
    breached = {
        c: parity_deltas(_extension_on_coordinates(draws)[c], summaries[c])
        for c in COORDINATES
        if parity_failures(_extension_on_coordinates(draws)[c], summaries[c], F2_EXACT_TOL)
    }
    assert breached, (
        "treating every censored shipment as delivered left every coordinate inside "
        "tolerance; the suite is not sensitive to the censoring the family exists for"
    )
    assert "log_sigma" in breached, (
        "the bias did not reach the scale, which is the parameter a delivery promise's "
        f"width is read from; breached: {sorted(breached)}"
    )


def test_a_wrong_distribution_is_detected(con, f2_data, f2_laplace):
    """Fit the same data as lognormal and check the comparison objects.

    `dist` selects the kernel, and a mis-plumbed `dist` would be invisible to
    any test that only asks whether the numbers are plausible: all four
    distributions produce a positive scale, a sensible intercept and a
    monotone survival curve.
    """
    lognormal_config = CONFIG.replace("'weibull'", "'lognormal'")
    draws = fit(con, f2_data, COLUMNS, "censored_aft", lognormal_config, "f2_lognormal")

    summaries, _, _ = f2_laplace
    breached = [
        c
        for c in COORDINATES
        if parity_failures(_extension_on_coordinates(draws)[c], summaries[c], F2_EXACT_TOL)
    ]
    assert breached, (
        "fitting a lognormal AFT against a Weibull reference produced no breach; the "
        "`dist` slot is not reaching the likelihood as far as this suite can tell"
    )
