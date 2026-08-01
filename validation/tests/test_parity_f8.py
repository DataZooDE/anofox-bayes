"""F8 `varying_variance_gaussian` vs PyMC.

The first family in this suite where **both sides are Markov chains**. F7 and F3
draw i.i.d. from a closed form, so only the reference carried autocorrelation;
here the extension runs `nuts-rs` and PyMC runs its own NUTS, and neither side's
Monte Carlo error is `sd/sqrt(draws)`. Every tolerance below is derived from ESS
*measured on both sides* -- see TOLERANCES.md -- and
`test_the_extension_side_is_well_enough_mixed_to_be_compared` asserts the
measured ESS is actually what the derivation assumed, so a future change that
degrades mixing invalidates the tolerance loudly rather than quietly.

What the reference has to get right (all three are in `pymc_f8`'s docstring):

* `pool_scale` and `sigma_spread` carry half-Normal priors on their **natural**
  scale while the sampler works on the log, so the density carries `+ log tau`
  and `+ log tau_s`. PyMC's automatic log transform supplies exactly those.
* `sigma_pop` is `exp(mu_s)` where `mu_s` is the sampled coordinate under a flat
  prior, so there is **no** Jacobian for it. `pm.HalfFlat("sigma_pop")` would
  invent one.
* `pool_scale`'s prior scale is the response's own sd, recomputed from the data
  rather than hard-coded.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_NUTS_CHAINS,
    EXTENSION_NUTS_DRAWS,
    EXTENSION_NUTS_WARMUP,
    EXTENSION_SEED,
    F8_SEGMENTS,
    F8_TOL,
    assert_parity,
    extension_chained_summary,
    f8_dataset,
    f8_response_sd,
    fit_chained,
    fit_metadata,
    first_seen,
    parity_deltas,
    parity_failures,
    pymc_f8,
    reference_summaries,
)

X_NAMES = ["x1"]

# The ESS the tolerances in TOLERANCES.md were derived from. Asserted rather
# than assumed: `MCSE = sd/sqrt(ESS)` is only a floor if ESS is what you think.
MIN_EXTENSION_ESS = 1_800

CONFIG = (
    "{'y': 'delay_days', 'x': ['x1'], 'group': 'segment', "
    f"'draws': {EXTENSION_NUTS_DRAWS}, 'chains': {EXTENSION_NUTS_CHAINS}, "
    f"'warmup': {EXTENSION_NUTS_WARMUP}, 'seed': {EXTENSION_SEED}}}"
)

GLOBALS = ["intercept", "beta[x1]", "pool_scale", "sigma_pop", "sigma_spread"]


@pytest.fixture(scope="session")
def f8_data():
    return f8_dataset()


@pytest.fixture(scope="session")
def f8_extension(con, f8_data):
    return fit_chained(
        con, f8_data, "segment, delay_days, x1", "varying_variance_gaussian", CONFIG, "f8_obs"
    )


@pytest.fixture(scope="session")
def f8_reference(f8_data):
    return reference_summaries(
        pymc_f8(f8_data, "delay_days", "segment", X_NAMES),
        ["intercept", "beta[x1]", "pool_scale", "sigma_pop", "sigma_spread", "group_effect", "sigma"],
    )


@pytest.fixture(scope="session")
def f8_metadata(con, f8_data):
    return fit_metadata(
        con, f8_data, "segment, delay_days, x1", "varying_variance_gaussian", CONFIG, "f8_meta"
    )


def test_the_fit_converged_and_was_served_by_nuts(f8_metadata):
    """Nothing below means anything if the fit refused or a different engine ran.

    `__engine__ = 2` is `nuts`. This family reaches Laplace by explicit config
    and THEORY.md §5 records that Laplace is *not admissible* here -- under SBC
    not one of the fourteen parameters is calibrated -- so a silent fallback
    would be the single most damaging thing that could happen to this fit, and
    the parity assertions would report it as a tolerance problem.
    """
    assert f8_metadata["__status__"] == 0.0, "the F8 fit did not report `converged`"
    assert f8_metadata["__engine__"] == 2.0, "the F8 fit was not served by the NUTS engine"
    assert f8_metadata["__n_groups_unready__"] == 0.0


def _comparisons():
    """(extension param, extension group_id, reference label) for every parameter."""
    pairs = [(p, "__global__", p) for p in GLOBALS]
    pairs += [("group_effect", g, f"group_effect[{g}]") for g in F8_SEGMENTS]
    pairs += [("sigma", g, f"sigma[{g}]") for g in F8_SEGMENTS]
    return pairs


COMPARISONS = _comparisons()


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ext_param,group_id,ref_label", COMPARISONS)
def test_parameter_parity(f8_extension, f8_reference, ext_param, group_id, ref_label):
    """Mean, sd, 5% and 95% of all seventeen parameters, against an independent NUTS.

    Includes the two things this family exists to produce and `pooled_gaussian`
    structurally cannot: a `sigma` per group, and a `pool_scale` with a
    posterior rather than a value somebody typed.
    """
    assert_parity(
        f"{ext_param}[{group_id}]" if group_id != "__global__" else ext_param,
        extension_chained_summary(f8_extension, ext_param, group_id=group_id).summary,
        f8_reference[ref_label],
        F8_TOL,
    )


def test_the_extension_side_is_well_enough_mixed_to_be_compared(f8_extension):
    """The tolerances assume an ESS; this asserts the fit actually delivers it.

    Unlike the reference's convergence gate in `reference_summaries`, this is not
    a statement that the extension converged -- the `__status__` row is what says
    that. It is a statement that the *tolerance derivation* still holds. A change
    that halves ESS would double the extension's Monte Carlo error and silently
    consume the headroom every assertion above depends on.
    """
    measured = {
        f"{p}[{g}]": extension_chained_summary(f8_extension, p, group_id=g)
        for p, g, _ in COMPARISONS
    }
    thin = {k: v.ess_bulk for k, v in measured.items() if v.ess_bulk < MIN_EXTENSION_ESS}
    assert not thin, (
        f"extension-side ESS fell below the {MIN_EXTENSION_ESS} the F8 tolerances in "
        f"TOLERANCES.md are derived from: {thin}. Either mixing regressed or the "
        "draw budget needs raising; loosening the tolerance instead would hide it."
    )
    bad_rhat = {k: v.r_hat for k, v in measured.items() if v.r_hat > 1.01}
    assert not bad_rhat, f"extension-side R-hat above 1.01: {bad_rhat}"


def test_the_pooling_scale_is_learned_rather_than_assumed(f8_extension, f8_reference, f8_data):
    """`pool_scale` has a posterior, and it is not just its prior echoed back.

    This is the row that distinguishes F8 from F3, so a fit in which
    `pool_scale`'s posterior sat on top of its prior would agree with a PyMC
    reference that made the same mistake -- both would be sampling the prior.
    The check is that the data moved it: the posterior is narrower than the
    half-Normal prior whose scale is the response's own sd.
    """
    prior_scale = f8_response_sd(f8_data["delay_days"])
    # A half-Normal(s) has sd = s * sqrt(1 - 2/pi).
    prior_sd = prior_scale * np.sqrt(1.0 - 2.0 / np.pi)
    posterior = extension_chained_summary(f8_extension, "pool_scale").summary

    assert posterior.sd < 0.75 * prior_sd, (
        f"pool_scale posterior sd {posterior.sd:.4f} is not appreciably narrower than "
        f"its prior's {prior_sd:.4f}; the data is not identifying the pooling scale "
        "and this fixture cannot test that it is learned"
    )
    assert abs(posterior.mean - f8_reference["pool_scale"].mean) < F8_TOL.mean * f8_reference["pool_scale"].sd


def test_the_groups_really_do_have_different_spreads(f8_extension):
    """The fixture has to exercise the per-group scale, or the suite proves nothing.

    If every segment came out with the same `sigma`, this family's whole reason
    for existing would be untested and the parity above would be a slower way of
    testing `pooled_gaussian`.
    """
    means = {g: extension_chained_summary(f8_extension, "sigma", group_id=g).summary.mean for g in F8_SEGMENTS}
    spread = max(means.values()) / min(means.values())
    assert spread > 2.0, (
        f"widest/narrowest segment sigma is only {spread:.2f}x ({means}); the fixture "
        "does not exercise the varying variance"
    )


# --------------------------------------------------------------------------
# Negative control: the harness must be able to fail
# --------------------------------------------------------------------------


def test_an_inflated_group_spread_is_detected(con, f8_data, f8_reference):
    """Triple one segment's scatter about its own mean and nothing else.

    The segment's *level* is left alone, so a comparison that only tracked group
    means would not notice. Only the per-group `sigma` -- the parameter this
    family exists to report -- moves, which is exactly the sensitivity the
    parity assertions above claim to have.
    """
    perturbed = f8_data.copy()
    target = F8_SEGMENTS[0]
    mask = perturbed["segment"] == target
    level = perturbed.loc[mask, "delay_days"].mean()
    perturbed.loc[mask, "delay_days"] = level + 3.0 * (perturbed.loc[mask, "delay_days"] - level)

    draws = fit_chained(
        con, perturbed, "segment, delay_days, x1", "varying_variance_gaussian", CONFIG, "f8_inflated"
    )
    breached = parity_failures(
        extension_chained_summary(draws, "sigma", group_id=target).summary,
        f8_reference[f"sigma[{target}]"],
        F8_TOL,
    )
    assert breached, (
        f"tripling {target}'s scatter produced no tolerance breach on its sigma; the F8 "
        "tolerances cannot detect a wrong per-group scale"
    )

    # NOT asserted: that the other segments stay inside tolerance. F7's
    # equivalent control can assert that, because F7 fits each group
    # independently. Here the groups share `sigma_pop` and `sigma_spread`, so
    # widening one segment genuinely moves every other segment's posterior --
    # that is the partial pooling working, and demanding otherwise would be
    # demanding the model not be hierarchical. What can be asserted, and is,
    # is that the perturbed segment moves furthest.
    untouched = F8_SEGMENTS[1]
    moved = {
        g: max(
            parity_deltas(
                extension_chained_summary(draws, "sigma", group_id=g).summary,
                f8_reference[f"sigma[{g}]"],
            ).values()
        )
        for g in (target, untouched)
    }
    assert moved[target] > 2.0 * moved[untouched], (
        f"tripling {target}'s scatter moved it by {moved[target]:.3f} and the untouched "
        f"{untouched} by {moved[untouched]:.3f}; the comparison is not attributing the "
        "change to the group that actually changed"
    )


def test_the_group_labels_are_not_interchangeable(con, f8_data, f8_reference):
    """Rotate the segment labels, leaving the multiset of observations intact.

    Every global parameter -- `pool_scale`, `sigma_pop`, `sigma_spread`, the
    intercept -- is unchanged by a relabelling, so nothing at the population
    level can catch it. Only a per-group comparison that is actually keyed on the
    group can, and this asserts it is.
    """
    rotated = f8_data.copy()
    order = first_seen(f8_data["segment"])
    rotation = {g: order[(i + 1) % len(order)] for i, g in enumerate(order)}
    rotated["segment"] = [rotation[g] for g in rotated["segment"]]

    draws = fit_chained(
        con, rotated, "segment, delay_days, x1", "varying_variance_gaussian", CONFIG, "f8_rotated"
    )
    breached = [
        g
        for g in F8_SEGMENTS
        if parity_failures(
            extension_chained_summary(draws, "group_effect", group_id=g).summary,
            f8_reference[f"group_effect[{g}]"],
            F8_TOL,
        )
    ]
    assert breached, (
        "rotating the segment labels left every group effect inside tolerance; the "
        "comparison is not keyed on the group it claims to be"
    )
