"""F6 `hier_elasticity` vs PyMC.

Both sides are Markov chains, so the tolerances are derived from ESS measured on
each side. F6 is the loosest set in this suite and deliberately: it carries two
pooling scales rather than one, so there are two ridges, and the per-segment
elasticities are a `-exp` transform of coordinates that are themselves poorly
conditioned.

**What only this harness can see.** The `-exp` transform. The family samples
`psi = log |elasticity|` and declares its prior *there*, so a lognormal prior on
the magnitude is a normal prior on `psi` and there is **no Jacobian**. A
reference reaching for a bounded or negative-support distribution would have
PyMC supply one, and the comparison would drift; a family that had added one
itself would be exploring a different posterior that SBC would nonetheless
certify as calibrated, because SBC draws its truth from whatever prior the code
implements. The same argument applies in the opposite direction to the two
pooling scales, which are declared on the natural scale and sampled on the log,
so each *does* carry `+ log tau`. Two families of error, opposite signs, and
this file is the only place both are visible.

`ROADMAP.md` §3.4 has the argument for why this family exists at all alongside
`pooled_gaussian` + `random_slopes` -- a sign constraint and a count likelihood,
neither of which that model can express. `test_every_segment_elasticity_is_negative`
below is the executable half of the first of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_NUTS_CHAINS,
    EXTENSION_NUTS_DRAWS,
    EXTENSION_NUTS_WARMUP,
    EXTENSION_SEED,
    F6_SEGMENTS,
    F6_TOL,
    F6_TRUTH,
    assert_parity,
    extension_chained_summary,
    f6_dataset,
    first_seen,
    fit_chained,
    fit_metadata,
    parity_deltas,
    parity_failures,
    pymc_f6,
    reference_summaries,
)

# `tau` mixes worst; this is the floor the F6 tolerances in TOLERANCES.md assume.
MIN_EXTENSION_ESS = 1_000

# Double the shared `EXTENSION_NUTS_WARMUP`, for the reason set out at length in
# `test_parity_f4.py`: at 1 000 the step size has not finished adapting, a single
# divergence out of 16 000 grades the fit `degenerate` under `max_divergent = 0`,
# and there is then nothing to compare. This family has *two* pooling scales
# rather than one, so it has more adaptation to do, not less.
EXTENSION_F6_WARMUP = 2 * EXTENSION_NUTS_WARMUP

CONFIG = (
    "{'y': 'units', 'price': 'log_price', 'group': 'segment', "
    f"'draws': {EXTENSION_NUTS_DRAWS}, 'chains': {EXTENSION_NUTS_CHAINS}, "
    f"'warmup': {EXTENSION_F6_WARMUP}, 'seed': {EXTENSION_SEED}}}"
)

COLUMNS = "segment, log_price, units"
GLOBALS = ["intercept", "elasticity", "tau", "tau_level", "phi"]


@pytest.fixture(scope="session")
def f6_data():
    return f6_dataset()


@pytest.fixture(scope="session")
def f6_segments(f6_data):
    return first_seen(f6_data["segment"])


@pytest.fixture(scope="session")
def f6_extension(con, f6_data):
    return fit_chained(con, f6_data, COLUMNS, "hier_elasticity", CONFIG, "f6_obs")


@pytest.fixture(scope="session")
def f6_metadata(con, f6_data):
    return fit_metadata(con, f6_data, COLUMNS, "hier_elasticity", CONFIG, "f6_meta")


@pytest.fixture(scope="session")
def f6_reference(f6_data):
    return reference_summaries(
        pymc_f6(f6_data, "units", "log_price", "segment"),
        [
            "intercept",
            "elasticity",
            "tau",
            "tau_level",
            "phi",
            "group_effect",
            "group_elasticity",
        ],
    )


def _comparisons():
    segments = first_seen(f6_dataset()["segment"])
    pairs = [(p, "__global__", p) for p in GLOBALS]
    pairs += [("group_effect", s, f"group_effect[{s}]") for s in segments]
    pairs += [("group_elasticity", s, f"group_elasticity[{s}]") for s in segments]
    return pairs


COMPARISONS = _comparisons()


def test_the_fit_converged_and_was_served_by_nuts(f6_metadata):
    """One admissible engine, and every segment identified.

    `__n_groups_unready__` is zero because the parity fixture gives every segment
    a price ladder. The per-segment identification refusal -- a segment whose
    prices never moved -- is a per-group verdict reached before any arithmetic,
    and it is exercised in `test/sql/f6_price_elasticity.test` and the Rust
    suite. Putting it here would mean comparing a segment whose posterior is its
    prior, which measures the prior rather than the likelihood.
    """
    assert f6_metadata["__status__"] == 0.0, "the F6 fit did not report `converged`"
    assert f6_metadata["__engine__"] == 2.0, "the F6 fit was not served by the NUTS engine"
    assert f6_metadata["__n_groups__"] == float(len(F6_SEGMENTS))
    assert f6_metadata["__n_groups_unready__"] == 0.0
    assert f6_metadata["__family__"] == 6.0


@pytest.mark.parametrize("ext_param,group_id,ref_label", COMPARISONS)
def test_parameter_parity(f6_extension, f6_reference, ext_param, group_id, ref_label):
    """Mean, sd, 5% and 95% of all 21 parameters, against an independent NUTS.

    `group_elasticity` is the one a price round reads, and it is the one whose
    interval a recommendation band is built from -- so it is checked per segment.
    """
    assert_parity(
        f"{ext_param}[{group_id}]" if group_id != "__global__" else ext_param,
        extension_chained_summary(f6_extension, ext_param, group_id=group_id).summary,
        f6_reference[ref_label],
        F6_TOL,
    )


def test_the_extension_side_is_well_enough_mixed_to_be_compared(f6_extension):
    """The tolerances assume an ESS; this asserts the fit delivers it."""
    measured = {
        f"{p}[{g}]": extension_chained_summary(f6_extension, p, group_id=g)
        for p, g, _ in COMPARISONS
    }
    thin = {k: v.ess_bulk for k, v in measured.items() if v.ess_bulk < MIN_EXTENSION_ESS}
    assert not thin, (
        f"extension-side ESS fell below the {MIN_EXTENSION_ESS} the F6 tolerances in "
        f"TOLERANCES.md are derived from: {thin}. Raise the draw budget rather than "
        "the tolerance; the tolerance is what the ESS pays for."
    )
    bad_rhat = {k: v.r_hat for k, v in measured.items() if v.r_hat > 1.01}
    assert not bad_rhat, f"extension-side R-hat above 1.01: {bad_rhat}"


# --------------------------------------------------------------------------
# The transforms, which are the part nothing else can see
# --------------------------------------------------------------------------


def test_the_elasticity_transform_carries_no_jacobian_and_the_scales_do(
    f6_extension, f6_reference
):
    """The asymmetry between `psi` and the two pooling scales, asserted directly.

    This repeats what `test_parameter_parity` already covers for these three
    parameters. It exists separately because the *reason* is not visible from
    there, and because it is the assertion most likely to be weakened by
    accident.

    * `elasticity` is `-exp(psi)`, and the prior is declared on `psi` — which is
      the sampled coordinate — so there is no Jacobian. Adding one would reweight
      toward small magnitudes, understating what a price rise costs, in a
      direction a price round would act on.
    * `tau` and `tau_level` are declared on the *natural* scale and sampled on
      the log, so each carries `+ log tau`. Dropping one reweights toward small
      pooling, over-shrinking every thin segment toward the population and
      making the per-segment recommendation bands narrower than they should be.

    Opposite signs, so a single symmetric tolerance on the whole set would
    absorb one while catching the other; each is checked on its own.
    """
    for name in ("elasticity", "tau", "tau_level"):
        deltas = parity_deltas(
            extension_chained_summary(f6_extension, name).summary, f6_reference[name]
        )
        assert deltas["mean"] < F6_TOL.mean, (
            f"{name} disagrees with a reference that declares the elasticity prior on "
            f"`psi` without a Jacobian and both pooling scales on the natural scale "
            f"with one: mean delta {deltas['mean']:.4f} sd. Check all three "
            "declarations on both sides before touching the tolerance."
        )


def test_every_segment_elasticity_is_negative(f6_extension, f6_segments):
    """The sign constraint, which is half the reason this family exists.

    Not "almost always" -- never. `b_g = -exp(psi + tau z_g)` makes it a property
    of the parameterisation, so a 5 % point at or above zero would be a bug
    rather than a tail event. The population coefficient is checked the same way.

    This is the assertion `pooled_gaussian` with `random_slopes` cannot make. On
    a segment this thin its Gaussian slope routinely puts real mass above zero,
    and a price meeting handed an interval saying that raising the price might
    sell *more* stops reading the interval.
    """
    population = extension_chained_summary(f6_extension, "elasticity").summary
    assert population.q95 < 0.0, (
        f"the population elasticity's 95% point is {population.q95:.4f}; the -exp "
        "transform is supposed to make a non-negative value impossible"
    )
    for segment in f6_segments:
        s = extension_chained_summary(
            f6_extension, "group_elasticity", group_id=segment
        ).summary
        assert s.q95 < 0.0, (
            f"{segment}'s elasticity has a 95% point of {s.q95:.4f}, which is not "
            "negative"
        )
    # ...and the constraint is not achieved by collapsing everything onto zero.
    # A family that satisfied the bound by reporting elasticities of -1e-6 would
    # pass every assertion above and be useless.
    assert population.mean < -0.3, (
        f"the population elasticity's mean is {population.mean:.4f}; the fixture was "
        f"drawn at {-np.exp(F6_TRUTH['psi']):.2f}, so a value near zero means the "
        "bound is being satisfied by collapse rather than by fit"
    )


def test_the_segments_are_told_apart_rather_than_pooled_into_one(
    f6_extension, f6_segments
):
    """`tau` must be doing real work, or the per-segment parity tests are hollow.

    The fixture is drawn with segment elasticities genuinely spread about the
    population value. If the fit pooled them all onto one number, every
    `group_elasticity` comparison above would be re-testing the population
    coefficient under eight names, and the family's central claim -- a *band per
    segment* -- would be untested.
    """
    medians = np.array(
        [
            extension_chained_summary(
                f6_extension, "group_elasticity", group_id=s
            ).summary.mean
            for s in f6_segments
        ]
    )
    spread = float(medians.max() - medians.min())
    assert spread > 0.15, (
        f"the eight segment elasticities span only {spread:.4f}; they have been "
        "pooled onto a single value and the per-segment assertions are testing the "
        "population coefficient eight times"
    )
    tau = extension_chained_summary(f6_extension, "tau").summary
    assert tau.q05 > 0.02, (
        f"tau's 5% point is {tau.q05:.4f}, i.e. indistinguishable from complete "
        "pooling"
    )


# --------------------------------------------------------------------------
# Negative control
# --------------------------------------------------------------------------


def test_a_price_ladder_shuffled_within_each_segment_is_detected(
    con, f6_data, f6_reference, f6_segments
):
    """Permute the price column *within* each segment, leaving volumes alone.

    This is the sharpest perturbation available here, and it is deliberately not
    the obvious one. Shuffling volumes across segments would move the levels,
    which `group_effect` would catch for reasons that have nothing to do with
    elasticity. Permuting price within a segment leaves every segment's mean
    volume, mean price and price *range* exactly as they were -- so the level
    parameters and the identification check see no change at all -- and destroys
    only the association between the two, which is the elasticity itself.

    If no segment's elasticity breaches tolerance under that, the per-segment
    comparisons are not testing the thing the family is for.
    """
    rng = np.random.default_rng(6262)
    shuffled = f6_data.copy()
    for segment in f6_segments:
        mask = shuffled["segment"] == segment
        shuffled.loc[mask, "log_price"] = rng.permutation(
            shuffled.loc[mask, "log_price"].to_numpy()
        )

    draws = fit_chained(con, shuffled, COLUMNS, "hier_elasticity", CONFIG, "f6_shuffled")
    breached = [
        s
        for s in f6_segments
        if parity_failures(
            extension_chained_summary(draws, "group_elasticity", group_id=s).summary,
            f6_reference[f"group_elasticity[{s}]"],
            F6_TOL,
        )
    ]
    assert breached, (
        "permuting price within every segment left every segment's elasticity inside "
        "tolerance; the F6 comparison is not sensitive to the price-volume association "
        "that is the whole model"
    )
