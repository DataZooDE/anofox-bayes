"""F4 `payment_delay` vs PyMC.

Both sides are Markov chains, as in F1 and F8, so the tolerances are derived
from ESS measured on each side rather than from the draw count.

**What only this harness can see.** Two things, and they are the reason this file
exists rather than being covered by the SBC suite next door:

* **The mean parameterisation.** `pm.Gamma(alpha=k, beta=k/mu)` has mean `mu`,
  and the extension writes the same thing as `-k*eta - k*y*e^{-eta}` plus
  `k log k - lnGamma(k)`. A family that had quietly parameterised the *median*,
  or that had dropped the `- sigma^2/2` correction on the lognormal branch,
  would still produce a well-behaved posterior that SBC would certify as
  calibrated -- because SBC draws its truth from whatever the code implements.
  Only an independent implementation of the *same stated model* can see it.
* **The `+ log tau` Jacobian, and the absence of one on the dispersion.** The
  extension declares a half-Normal on `tau` itself and samples `log tau`, so the
  density carries `+ log tau`; it declares the dispersion prior on `log shape`,
  which *is* the sampled coordinate, so that one carries nothing. Getting either
  wrong is invisible to every engine-agreement test -- both engines would explore
  the same wrong surface -- and invisible to SBC for the reason above.

`ROADMAP.md` §2 records a missing Jacobian being proved undetectable by mutation
on `conjugate_anomaly`, which is why both signs are asserted here directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_NUTS_CHAINS,
    EXTENSION_NUTS_DRAWS,
    EXTENSION_NUTS_WARMUP,
    EXTENSION_SEED,
    F4_SEGMENTS,
    F4_TOL,
    F4_TRUTH,
    assert_parity,
    extension_chained_summary,
    f4_dataset,
    first_seen,
    fit_chained,
    fit_metadata,
    parity_deltas,
    parity_failures,
    pymc_f4,
    reference_summaries,
)

# `tau` mixes worst; this is the floor the F4 tolerances in TOLERANCES.md assume.
MIN_EXTENSION_ESS = 1_000

# **Double the shared `EXTENSION_NUTS_WARMUP`, and measured rather than guessed.**
#
# At 1 000 warmup this fixture draws exactly **one** divergence out of 16 000.
# `payment_delay` inherits `max_divergent = 0`, so the fit is graded `degenerate`
# and there is nothing to compare -- while every other diagnostic is comfortably
# healthy (worst `ess_bulk` 1 719, worst R-hat 1.005). It is a step size that has
# not finished adapting, not a posterior that is wrong.
#
# Measured across the three budgets, same seed and same data:
#
#     warmup 1000 -> 1 divergence, `degenerate`, worst ESS 1 719
#     warmup 2000 -> 0 divergences, `converged`,  worst ESS 2 020
#     warmup 3000 -> 0 divergences, `converged`,  worst ESS 1 839
#
# So the extra warmup buys the *verdict*, and the draw budget is already ample.
# Raising it here rather than raising the shared constant keeps F1 and F8
# bit-for-bit unaffected: their fixtures clear the gate at 1 000 and re-running
# them at a different budget would change numbers the tolerances were derived
# from.
EXTENSION_F4_WARMUP = 2 * EXTENSION_NUTS_WARMUP

CONFIG = (
    "{'y': 'delay_days', 'group': 'segment', "
    f"'draws': {EXTENSION_NUTS_DRAWS}, 'chains': {EXTENSION_NUTS_CHAINS}, "
    f"'warmup': {EXTENSION_F4_WARMUP}, 'seed': {EXTENSION_SEED}}}"
)

GLOBALS = ["intercept", "tau", "shape"]


@pytest.fixture(scope="session")
def f4_data():
    return f4_dataset()


@pytest.fixture(scope="session")
def f4_segments(f4_data):
    return first_seen(f4_data["segment"])


@pytest.fixture(scope="session")
def f4_extension(con, f4_data):
    return fit_chained(
        con, f4_data, "segment, delay_days", "payment_delay", CONFIG, "f4_obs"
    )


@pytest.fixture(scope="session")
def f4_metadata(con, f4_data):
    return fit_metadata(
        con, f4_data, "segment, delay_days", "payment_delay", CONFIG, "f4_meta"
    )


@pytest.fixture(scope="session")
def f4_reference(f4_data):
    return reference_summaries(
        pymc_f4(f4_data, "delay_days", "segment"),
        ["intercept", "tau", "shape", "u", "mu"],
    )


def _comparisons():
    segments = first_seen(f4_dataset()["segment"])
    pairs = [(p, "__global__", p) for p in GLOBALS]
    pairs += [("u", s, f"u[{s}]") for s in segments]
    pairs += [("mu", s, f"mu[{s}]") for s in segments]
    return pairs


COMPARISONS = _comparisons()


def test_the_fit_converged_and_was_served_by_nuts(f4_metadata):
    """`payment_delay` has exactly one admissible engine, and this asserts it ran.

    `laplace` is refused outright rather than served badly: a non-centred
    hierarchy has no usable joint mode, because when every `z_j` is zero the
    likelihood does not depend on `tau` at all and the `+ log tau` Jacobian makes
    the density rise without bound along that ridge. `__engine__ = 2` is `nuts`.
    """
    assert f4_metadata["__status__"] == 0.0, "the F4 fit did not report `converged`"
    assert f4_metadata["__engine__"] == 2.0, "the F4 fit was not served by the NUTS engine"
    assert f4_metadata["__n_groups__"] == float(len(F4_SEGMENTS))
    assert f4_metadata["__family__"] == 4.0


@pytest.mark.parametrize("ext_param,group_id,ref_label", COMPARISONS)
def test_parameter_parity(f4_extension, f4_reference, ext_param, group_id, ref_label):
    """Mean, sd, 5% and 95% of all 15 parameters, against an independent NUTS.

    `mu` is the one a cash forecast reads -- the segment's own mean delay -- so it
    is checked per segment rather than only at the population level.
    """
    assert_parity(
        f"{ext_param}[{group_id}]" if group_id != "__global__" else ext_param,
        extension_chained_summary(f4_extension, ext_param, group_id=group_id).summary,
        f4_reference[ref_label],
        F4_TOL,
    )


def test_the_extension_side_is_well_enough_mixed_to_be_compared(f4_extension):
    """The tolerances assume an ESS; this asserts the fit delivers it."""
    measured = {
        f"{p}[{g}]": extension_chained_summary(f4_extension, p, group_id=g)
        for p, g, _ in COMPARISONS
    }
    thin = {k: v.ess_bulk for k, v in measured.items() if v.ess_bulk < MIN_EXTENSION_ESS}
    assert not thin, (
        f"extension-side ESS fell below the {MIN_EXTENSION_ESS} the F4 tolerances in "
        f"TOLERANCES.md are derived from: {thin}. Raise the draw budget rather than "
        "the tolerance; the tolerance is what the ESS pays for."
    )
    bad_rhat = {k: v.r_hat for k, v in measured.items() if v.r_hat > 1.01}
    assert not bad_rhat, f"extension-side R-hat above 1.01: {bad_rhat}"


# --------------------------------------------------------------------------
# The two prior declarations, which are the part nothing else can see
# --------------------------------------------------------------------------


def test_the_tau_jacobian_is_present_and_the_dispersion_one_is_not(
    f4_extension, f4_reference
):
    """`tau` and `shape` agree with a reference that spells the asymmetry out.

    This is the same assertion `test_parameter_parity` already makes for these
    two. It exists separately because the *reason* is not visible from there, and
    because it is the assertion most likely to be weakened by accident: a future
    maintainer reaching for `pm.HalfNormal` on `tau` would get PyMC's own
    transform Jacobian on top of the explicit one, and reaching for
    `pm.LogNormal` on `shape` would add one that should not be there at all.

    The sign of each error is what it would move:

    * dropping `+ log tau` reweights toward small `tau`, over-pooling every thin
      segment toward the ledger mean and understating how differently the slow
      payers behave;
    * *adding* a Jacobian to the dispersion prior reweights toward large `shape`,
      i.e. toward a tighter distribution, and a 95 % cash buffer computed from it
      is too small in the direction that causes an overdraft.

    Both are directional, so the check is that the deltas are small rather than
    merely bounded.
    """
    for name in ("tau", "shape"):
        deltas = parity_deltas(
            extension_chained_summary(f4_extension, name).summary, f4_reference[name]
        )
        assert deltas["mean"] < F4_TOL.mean, (
            f"{name} disagrees with a reference that declares the `+ log tau` "
            f"Jacobian explicitly and the dispersion prior without one: mean delta "
            f"{deltas['mean']:.4f} sd. Check both declarations on both sides before "
            "touching the tolerance."
        )


def test_the_skew_is_found_rather_than_assumed(f4_extension):
    """`shape` must stay far from the Gaussian limit on data that is skewed.

    The fixture is drawn at `shape = 6`, i.e. genuinely right-skewed. A fit
    returning a large `shape` would be reporting something close to symmetric,
    the parity comparison would still pass against a reference making the same
    mistake, and the resulting cash buffers would be too tight in the tail --
    which is the only part of the distribution the decision reads.
    """
    shape = extension_chained_summary(f4_extension, "shape").summary
    assert shape.q95 < 30.0, (
        f"shape's 95% point is {shape.q95:.1f}; the fit is reporting something close "
        "to a Gaussian on data drawn at shape = 6"
    )
    # The truth is inside a 90% interval. Not a calibration claim -- one draw of
    # one dataset cannot make one -- but a fixture whose truth sat outside would
    # be a poor choice for measuring agreement.
    assert shape.q05 < F4_TRUTH["shape"] < shape.q95


def test_mu_is_the_mean_rather_than_the_median(f4_extension, f4_data, f4_segments):
    """The parameterisation, checked against the data rather than the reference.

    For a Gamma of shape 6 the mean sits about 8 % above the median, so a family
    that had parameterised the median would produce a `mu` systematically low by
    roughly that much on every segment -- a bias the symmetric parity tolerance
    above would partly absorb and which no amount of agreement with a reference
    making the *same* choice would reveal.

    Comparing against each segment's own observed mean is the independent check:
    with 40 invoices per segment the sample mean has a standard error near 6 % of
    the mean, so agreement to within 10 % is a real constraint while a median
    parameterisation would sit outside it in a consistent direction.
    """
    observed = f4_data.groupby("segment")["delay_days"].mean()
    ratios = []
    for segment in f4_segments:
        posterior = extension_chained_summary(
            f4_extension, "mu", group_id=segment
        ).summary.mean
        ratios.append(posterior / float(observed[segment]))
    ratios = np.array(ratios)
    assert np.all(np.abs(ratios - 1.0) < 0.10), (
        f"posterior `mu` against each segment's observed mean delay: {ratios}. "
        "A consistent shortfall near 8% would mean the family is reporting the "
        "median rather than the mean."
    )
    # ...and the deviation is not one-sided, which a median parameterisation
    # would make it.
    assert 0.97 < float(ratios.mean()) < 1.03, (
        f"the mean ratio is {ratios.mean():.4f}; a one-sided deviation is a "
        "parameterisation error rather than sampling noise"
    )


# --------------------------------------------------------------------------
# Negative control
# --------------------------------------------------------------------------


def test_a_shuffled_ledger_is_detected(con, f4_data, f4_reference, f4_segments):
    """Shuffle delays across segments, leaving the ledger total alone.

    This is the perturbation `payment_delay` exists to distinguish from the
    truth: a ledger in which every segment pays the same way. The population
    parameters barely move -- the grand mean is unchanged by construction -- so
    only `tau` and the per-segment means can notice, which is exactly the claim
    the per-group parity assertions make. If none of them breaches, those
    assertions are not testing the per-segment fit that is the whole model.
    """
    rng = np.random.default_rng(4242)
    shuffled = f4_data.copy()
    shuffled["delay_days"] = rng.permutation(f4_data["delay_days"].to_numpy())

    draws = fit_chained(
        con, shuffled, "segment, delay_days", "payment_delay", CONFIG, "f4_shuffled"
    )
    breached = [
        s
        for s in f4_segments
        if parity_failures(
            extension_chained_summary(draws, "mu", group_id=s).summary,
            f4_reference[f"mu[{s}]"],
            F4_TOL,
        )
    ]
    assert breached, (
        "shuffling delays across segments left every segment's mean inside tolerance; "
        "the F4 comparison is not sensitive to the per-segment fit that is the whole "
        "model"
    )
