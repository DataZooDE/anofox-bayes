"""The tolerances are sized for the number of comparisons, not for one.

Every other test here asks whether one parameter agrees with PyMC. This one asks
whether the *suite* can be trusted to be quiet when nothing is wrong -- which is a
different question, and the one that was being got wrong.

`TOLERANCES.md` derived each limit as roughly 3.5x a Monte-Carlo standard error.
That is the right size for a single comparison and the wrong size for a suite: the
run compares 126 parameters on four statistics each, and a 3.5-sigma limit applied
504 times is expected to trip 0.23 times per run, so **one run in five fails on
nothing at all**. Measured, not modelled: the failures land on `tau` first, because
it is the hierarchical scale and has the lowest ESS in every family that has one.

So the limits are sized family-wise instead. `PARITY_SIGMA` is the per-comparison
sigma that keeps the probability of *any* spurious failure across the whole run at
1 %, and the tolerances are that multiple rather than 3.5.
"""

import math
from statistics import NormalDist

import pytest
from _support import (
    ALL_TOLERANCES,
    PARITY_COMPARISON_BUDGET,
    PARITY_FAMILY_WISE_ALPHA,
    PARITY_SIGMA,
    PER_COMPARISON_SIGMA_BEFORE,
    TOLERANCE_BASELINE,
)


def test_the_sigma_is_the_one_the_comparison_count_implies():
    """`PARITY_SIGMA` is derived, not chosen. This is the derivation."""
    expected = NormalDist().inv_cdf(
        1 - PARITY_FAMILY_WISE_ALPHA / (2 * PARITY_COMPARISON_BUDGET)
    )
    assert math.isclose(PARITY_SIGMA, expected, rel_tol=0.01), (
        f"PARITY_SIGMA is {PARITY_SIGMA}, but {PARITY_COMPARISON_BUDGET} comparisons "
        f"at a {PARITY_FAMILY_WISE_ALPHA:.0%} family-wise budget require "
        f"{expected:.2f}. Change one and the other has to move."
    )


def test_the_old_sigma_really_was_too_small_for_this_many_comparisons():
    """The reason for the change, kept as an assertion rather than a paragraph.

    If the suite ever shrinks enough that 3.5 sigma is adequate again, this fails
    and the widening should be reconsidered rather than inherited.
    """
    p = 2 * (1 - NormalDist().cdf(PER_COMPARISON_SIGMA_BEFORE))
    any_failure = 1 - (1 - p) ** PARITY_COMPARISON_BUDGET
    assert any_failure > 0.05, (
        f"at {PER_COMPARISON_SIGMA_BEFORE} sigma over {PARITY_COMPARISON_BUDGET} "
        f"comparisons the spurious-failure rate is {any_failure:.1%}, which no longer "
        "justifies a family-wise correction"
    )


@pytest.mark.parametrize("label", sorted(TOLERANCE_BASELINE))
def test_every_monte_carlo_tolerance_carries_the_family_wise_factor(label):
    """Each MC-derived limit is at least its old value scaled by the new sigma.

    Expressed against the recorded baseline rather than against an MCSE re-derived
    here: the baselines *are* the per-comparison derivation in TOLERANCES.md, and
    restating that arithmetic in a test would only give it a second place to drift.
    """
    tol = next(t for t in ALL_TOLERANCES if t.label == label)
    base = TOLERANCE_BASELINE[label]
    factor = PARITY_SIGMA / PER_COMPARISON_SIGMA_BEFORE
    for stat in ("mean", "sd", "quantile"):
        want = base[stat] * factor
        got = getattr(tol, stat)
        assert got >= want - 1e-9, (
            f"{label}.{stat} is {got:.4f}; sized for {PARITY_COMPARISON_BUDGET} "
            f"comparisons it must be at least {want:.4f} "
            f"({base[stat]:.4f} x {factor:.2f})"
        )


def test_the_comparison_budget_still_covers_what_the_suite_compares(parity_comparison_count):
    """Adding a family must force the sigma to be re-derived, not silently inherited."""
    assert parity_comparison_count <= PARITY_COMPARISON_BUDGET, (
        f"the suite now makes {parity_comparison_count} comparisons against a budget of "
        f"{PARITY_COMPARISON_BUDGET}; re-derive PARITY_SIGMA before raising the budget"
    )
