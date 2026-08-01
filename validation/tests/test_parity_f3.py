"""F3 `pooled_gaussian` vs PyMC.

Same idea as the F7 suite: one dataset, two fits, compare the posterior
summaries. F3 is the harder comparison because the posterior is multivariate
and, in the panel case, only ridge-identified in the intercept/group-effect
block.

**Read this before reading the tests.** The extension's coefficient posteriors
agree with PyMC on all four summary statistics. Its *residual scale* posterior
does not: `sigma` is systematically tight by a factor of `sqrt((n - k)/n)`,
where `k` is the number of coefficients carrying a flat prior. This is a real
discrepancy in the extension, not tolerance noise --
`test_residual_scale_carries_no_degrees_of_freedom_deficit` pins the
factor to three decimal places on two different designs. The `sigma` parity
tests were `xfail(strict=True)` while the bug was live; the fix turned them red,
which is how the marker came off. They now guard against it returning, and will
fail loudly the day it is fixed, forcing the markers off.

The same factor narrows every coefficient interval by ~2%, which is under the
tolerance the panel's NUTS Monte Carlo error forces on us. TOLERANCES.md says
so explicitly; it is measured, not ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_DRAWS,
    EXTENSION_SEED,
    F3_POOL_SCALE,
    F3_TOL,
    assert_parity,
    extension_summary,
    f3_panel_dataset,
    f3_simple_dataset,
    fit,
    parity_deltas,
    parity_failures,
    pymc_f3,
    reference_summaries,
)

SIMPLE_X = ["x1", "x2"]
PANEL_X = ["post", "treated_post", "month"]

# Repeated on every marker so a CI log line is self-explanatory without opening
# this file.
# --------------------------------------------------------------------------
# Ungrouped regression
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def simple_data():
    return f3_simple_dataset()


@pytest.fixture(scope="session")
def simple_extension(con, simple_data):
    return fit(
        con,
        simple_data,
        "y, x1, x2",
        "pooled_gaussian",
        "{'y': 'y', 'x': ['x1', 'x2'], "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f3_simple_obs",
    )


@pytest.fixture(scope="session")
def simple_reference(simple_data):
    return reference_summaries(
        pymc_f3(
            simple_data["y"].to_numpy(),
            simple_data[SIMPLE_X].to_numpy(),
            SIMPLE_X,
        ),
        ["intercept", "beta", "sigma"],
    )


SIMPLE_COEFFICIENTS = ["intercept", "beta[x1]", "beta[x2]"]


@pytest.mark.parametrize("param", SIMPLE_COEFFICIENTS)
def test_simple_coefficient_parity(simple_extension, simple_reference, param):
    """Mean, sd, 5% and 95% of every coefficient, against NUTS."""
    assert_parity(
        param,
        extension_summary(simple_extension, param),
        simple_reference[param],
        F3_TOL,
    )


def test_simple_residual_scale_parity(simple_extension, simple_reference):
    assert_parity(
        "sigma",
        extension_summary(simple_extension, "sigma"),
        simple_reference["sigma"],
        F3_TOL,
    )


# --------------------------------------------------------------------------
# Grouped panel (difference-in-differences)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def panel_data():
    return f3_panel_dataset()


@pytest.fixture(scope="session")
def panel_extension(con, panel_data):
    return fit(
        con,
        panel_data,
        "store, units, month, post, treated_post",
        "pooled_gaussian",
        "{'y': 'units', 'x': ['post', 'treated_post', 'month'], 'group': 'store', "
        f"'pool_scale': {F3_POOL_SCALE}, "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f3_panel_obs",
    )


def _panel_design(panel_data):
    stores = sorted(panel_data["store"].unique())
    index = np.array([stores.index(s) for s in panel_data["store"]])
    return stores, index


@pytest.fixture(scope="session")
def panel_reference(panel_data):
    stores, index = _panel_design(panel_data)
    return reference_summaries(
        pymc_f3(
            panel_data["units"].to_numpy(),
            panel_data[PANEL_X].to_numpy(),
            PANEL_X,
            group_index=index,
            group_names=stores,
            pool_scale=F3_POOL_SCALE,
        ),
        ["intercept", "beta", "group_effect", "sigma"],
    )


def _panel_coefficients():
    stores, _ = _panel_design(f3_panel_dataset())
    params = [("intercept", "__global__", "intercept")]
    params += [(f"beta[{x}]", "__global__", f"beta[{x}]") for x in PANEL_X]
    params += [("group_effect", s, f"group_effect[{s}]") for s in stores]
    return params


PANEL_COEFFICIENTS = _panel_coefficients()


@pytest.mark.parametrize("ext_param,group_id,ref_label", PANEL_COEFFICIENTS)
def test_panel_coefficient_parity(
    panel_extension, panel_reference, ext_param, group_id, ref_label
):
    """Including `beta[treated_post]` -- the diff-in-diff causal estimate itself,
    and every partially pooled store effect."""
    assert_parity(
        f"{ext_param}[{group_id}]",
        extension_summary(panel_extension, ext_param, group_id=group_id),
        panel_reference[ref_label],
        F3_TOL,
    )


def test_panel_residual_scale_parity(panel_extension, panel_reference):
    assert_parity(
        "sigma",
        extension_summary(panel_extension, "sigma"),
        panel_reference["sigma"],
        F3_TOL,
    )


# --------------------------------------------------------------------------
# Characterising the discrepancy
# --------------------------------------------------------------------------

# (case, n observations, k coefficients carrying a flat prior)
#
# simple: intercept + x1 + x2.
# panel:  intercept + post + treated_post + month. The six group effects carry a
#         proper N(0, sigma^2 * pool_scale^2) prior, so they legitimately supply
#         their own (sigma^2)^(-1/2) factors and are NOT part of k.
DOF_CASES = [("simple", 80, 3), ("panel", 120, 4)]


@pytest.mark.parametrize("case,n,k_flat", DOF_CASES)
def test_residual_scale_carries_no_degrees_of_freedom_deficit(
    request, case, n, k_flat
):
    """Regression guard: the residual scale must carry no degrees-of-freedom deficit.

    This suite originally *found* a real bug here. `f3_pooled_gaussian.rs` used
    `a_n = a0 + n/2` -- the Normal-Inverse-Gamma result, correct only when the
    coefficient prior is proper and sigma-scaled. Under the default flat prior
    the textbook shape is `a0 + (n - k)/2`, so every credible interval came out
    too narrow by exactly `sqrt((n - k)/n)`: measured 0.98005 against a predicted
    0.98107 on the simple design, and 0.98309 against 0.98319 on the panel.

    Over-confident, never under -- the direction that quietly under-covers a
    service level. ~2% at these sizes, but it scales with `k/n`: at n=30 with 8
    predictors it is 15%.

    Fixed in `f3_pooled_gaussian.rs`. This test now asserts the ratio is 1, and
    would catch the deficit returning. `E[sigma]` is used rather than
    `sd(sigma)` because it is the statistic with the smallest Monte Carlo error
    on the reference side.
    """
    extension = request.getfixturevalue(f"{case}_extension")
    reference = request.getfixturevalue(f"{case}_reference")

    observed = extension_summary(extension, "sigma").mean / reference["sigma"].mean
    deficit = np.sqrt((n - k_flat) / n)

    assert abs(observed - 1.0) < 0.006, (
        f"[{case}] E[sigma] ratio extension/pymc = {observed:.5f}, expected 1. "
        f"If it has drifted toward sqrt(({n} - {k_flat})/{n}) = {deficit:.5f}, the "
        "degrees-of-freedom correction in f3_pooled_gaussian.rs has regressed."
    )


@pytest.mark.parametrize("case,n,k_flat", DOF_CASES)
def test_coefficient_intervals_carry_no_scale_deficit(request, case, n, k_flat):
    """The same guard, one level down: no coefficient inherits a scale deficit.

    `beta | sigma^2 ~ N(b_n, sigma^2 A^-1)`, so a residual scale short by
    `sqrt((n - k)/n)` hands the identical shortfall to every coefficient's sd.
    That is how the original bug stayed invisible to the per-parameter sd
    tolerance: ~2% is under it (see TOLERANCES.md for why the tolerance cannot be
    tightened enough to catch it). Measuring the ratio directly is what puts it
    on the record rather than letting it be absorbed.

    The bound is looser than the `sigma` test above because a posterior sd
    estimated from NUTS carries roughly sd/sqrt(2*ESS) of Monte Carlo error --
    about 1% on the ridge-identified panel block.
    """
    extension = request.getfixturevalue(f"{case}_extension")
    reference = request.getfixturevalue(f"{case}_reference")
    deficit = np.sqrt((n - k_flat) / n)

    coefficients = (
        [(p, "__global__", p) for p in SIMPLE_COEFFICIENTS]
        if case == "simple"
        else PANEL_COEFFICIENTS
    )
    ratios = {
        ref_label: extension_summary(extension, ext_param, group_id=gid).sd
        / reference[ref_label].sd
        for ext_param, gid, ref_label in coefficients
    }
    mean_ratio = float(np.mean(list(ratios.values())))

    assert abs(mean_ratio - 1.0) < 0.015, (
        f"[{case}] mean coefficient sd ratio extension/pymc = {mean_ratio:.5f}, "
        f"expected 1. If it has drifted toward sqrt(({n} - {k_flat})/{n}) = "
        f"{deficit:.5f}, the degrees-of-freedom correction has regressed. "
        "Per-coefficient: " + ", ".join(f"{k}={v:.4f}" for k, v in ratios.items())
    )


# --------------------------------------------------------------------------
# Negative controls: the harness must be able to fail
# --------------------------------------------------------------------------


def test_wrong_pool_scale_is_detected(con, panel_data, panel_reference):
    """Refit with a pooling scale 10x tighter and check the comparison objects.

    `pool_scale` is the one knob in F3 that a caller sets and cannot verify by
    eye. If a mis-plumbed `pool_scale` slipped past this suite, the partial
    pooling -- the whole reason F3 exists -- would be unvalidated.
    """
    stores, _ = _panel_design(panel_data)
    draws = fit(
        con,
        panel_data,
        "store, units, month, post, treated_post",
        "pooled_gaussian",
        "{'y': 'units', 'x': ['post', 'treated_post', 'month'], 'group': 'store', "
        f"'pool_scale': {F3_POOL_SCALE / 10.0}, "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f3_panel_shrunk",
    )

    breached = [
        s
        for s in stores
        if parity_failures(
            extension_summary(draws, "group_effect", group_id=s),
            panel_reference[f"group_effect[{s}]"],
            F3_TOL,
        )
    ]
    assert breached, (
        "shrinking pool_scale by 10x produced no tolerance breach on any group "
        "effect; the F3 tolerances cannot detect a mis-plumbed pooling scale"
    )


def test_scrambled_response_is_detected(con, simple_data, simple_reference):
    """A scrambled response must not still look like the same posterior.

    The cheapest way for a parity suite to be vacuously green is for the two
    sides to be summarising something that barely depends on the data. Permuting
    `y` destroys the regression relationship while leaving every marginal of `y`
    -- its mean, sd and quantiles -- exactly intact, so only a comparison that
    genuinely tracks the *fit* can notice.
    """
    scrambled = simple_data.copy()
    scrambled["y"] = scrambled["y"].to_numpy()[
        np.random.default_rng(9).permutation(len(scrambled))
    ]
    draws = fit(
        con,
        scrambled,
        "y, x1, x2",
        "pooled_gaussian",
        "{'y': 'y', 'x': ['x1', 'x2'], "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f3_scrambled",
    )
    breached = {
        p: parity_deltas(extension_summary(draws, p), simple_reference[p])
        for p in ("beta[x1]", "beta[x2]")
        if parity_failures(extension_summary(draws, p), simple_reference[p], F3_TOL)
    }
    assert breached, (
        "permuting the response left every slope inside tolerance; the comparison "
        "is not actually sensitive to the fit"
    )

    with pytest.raises(AssertionError, match="disagrees with the PyMC reference"):
        for p in breached:
            assert_parity(
                p, extension_summary(draws, p), simple_reference[p], F3_TOL, record=False
            )
