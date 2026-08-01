"""F1 `hier_negbin` vs PyMC.

Both sides are Markov chains, as in F8, so the tolerances are again derived from
ESS measured on each side rather than from the draw count. F1 is the harder of
the two: `tau` is the worst-mixing parameter anywhere in this suite, because the
unpenalised intercept and the twelve group offsets trade off along a ridge a
diagonal mass matrix cannot precondition.

**The two hyperpriors are the whole test.** Everything else in this family --
the negative binomial log-density, the non-centred `tau * z_j` -- is standard
and would be caught by almost any check. The priors are not:

* `tau` is uniform on **`tau` itself**. The scale-free `1/tau` that every other
  positive parameter in this extension defaults to gives an *improper posterior*
  for a variance component; uniform is proper for three or more groups
  (Gelman 2006). The sampler works on `log tau`, so the density carries
  `+ log tau`.
* `phi` is uniform on the **overdispersion `1/phi`**, which is flat exactly at
  the Poisson limit, so the default cannot push a fit toward finding burstiness
  that is not there. On `log phi` that is `- log phi`.

THEORY.md says of those two Jacobians: "Those terms are not decoration and are
not visible to any engine-agreement test -- both engines would explore the same
wrong surface." They are not visible to SBC either, which draws the truth from
whatever prior the code implements and would certify a wrong one as calibrated.
An independent implementation of the prior is the only harness that can see
them, and `test_the_two_hyperprior_jacobians_are_both_present` below turns that
from an implicit property of the reference model into an explicit assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_NUTS_CHAINS,
    EXTENSION_NUTS_DRAWS,
    EXTENSION_NUTS_WARMUP,
    EXTENSION_SEED,
    F1_TOL,
    F1_THIN_WEEKS,
    F1_TRUTH,
    assert_parity,
    extension_chained_summary,
    f1_dataset,
    fit_chained,
    fit_metadata,
    first_seen,
    parity_deltas,
    parity_failures,
    pymc_f1,
    reference_summaries,
)

# `tau` mixes worst; this is the floor the F1 tolerances in TOLERANCES.md assume.
MIN_EXTENSION_ESS = 1_000

CONFIG = (
    "{'y': 'units', 'group': 'part', "
    f"'draws': {EXTENSION_NUTS_DRAWS}, 'chains': {EXTENSION_NUTS_CHAINS}, "
    f"'warmup': {EXTENSION_NUTS_WARMUP}, 'seed': {EXTENSION_SEED}}}"
)

GLOBALS = ["intercept", "tau", "phi"]


@pytest.fixture(scope="session")
def f1_data():
    return f1_dataset()


@pytest.fixture(scope="session")
def f1_parts(f1_data):
    return first_seen(f1_data["part"])


@pytest.fixture(scope="session")
def f1_extension(con, f1_data):
    return fit_chained(con, f1_data, "part, units", "hier_negbin", CONFIG, "f1_obs")


@pytest.fixture(scope="session")
def f1_metadata(con, f1_data):
    return fit_metadata(con, f1_data, "part, units", "hier_negbin", CONFIG, "f1_meta")


@pytest.fixture(scope="session")
def f1_reference(f1_data):
    return reference_summaries(
        pymc_f1(f1_data, "units", "part"), ["intercept", "tau", "phi", "u", "rate"]
    )


def _comparisons():
    parts = first_seen(f1_dataset()["part"])
    pairs = [(p, "__global__", p) for p in GLOBALS]
    pairs += [("u", p, f"u[{p}]") for p in parts]
    pairs += [("rate", p, f"rate[{p}]") for p in parts]
    return pairs


COMPARISONS = _comparisons()


def test_the_fit_converged_and_was_served_by_nuts(f1_metadata):
    """`hier_negbin` has exactly one admissible engine, and this asserts it ran.

    THEORY.md refuses `laplace` for this family outright rather than serving it
    badly: a non-centred hierarchy has no usable joint mode, because when every
    `z_j` is zero the likelihood does not depend on `tau` at all and the
    `+ log tau` Jacobian makes the density rise without bound along that ridge.
    A mode search walks straight up it. `__engine__ = 2` is `nuts`.
    """
    assert f1_metadata["__status__"] == 0.0, "the F1 fit did not report `converged`"
    assert f1_metadata["__engine__"] == 2.0, "the F1 fit was not served by the NUTS engine"
    assert f1_metadata["__n_groups__"] == 12.0


@pytest.mark.parametrize("ext_param,group_id,ref_label", COMPARISONS)
def test_parameter_parity(f1_extension, f1_reference, ext_param, group_id, ref_label):
    """Mean, sd, 5% and 95% of all 27 parameters, against an independent NUTS.

    `rate` is the one a safety-stock agent reads, and it is the one whose
    interval a reorder point is set from -- so it is checked per part, on the
    thin parts as well as the thick ones.
    """
    assert_parity(
        f"{ext_param}[{group_id}]" if group_id != "__global__" else ext_param,
        extension_chained_summary(f1_extension, ext_param, group_id=group_id).summary,
        f1_reference[ref_label],
        F1_TOL,
    )


def test_the_extension_side_is_well_enough_mixed_to_be_compared(f1_extension):
    """The tolerances assume an ESS; this asserts the fit delivers it.

    `tau` is the binding constraint and always will be: it is the parameter the
    ridge runs through.
    """
    measured = {
        f"{p}[{g}]": extension_chained_summary(f1_extension, p, group_id=g)
        for p, g, _ in COMPARISONS
    }
    thin = {k: v.ess_bulk for k, v in measured.items() if v.ess_bulk < MIN_EXTENSION_ESS}
    assert not thin, (
        f"extension-side ESS fell below the {MIN_EXTENSION_ESS} the F1 tolerances in "
        f"TOLERANCES.md are derived from: {thin}. Raise the draw budget rather than "
        "the tolerance; the tolerance is what the ESS pays for."
    )
    bad_rhat = {k: v.r_hat for k, v in measured.items() if v.r_hat > 1.01}
    assert not bad_rhat, f"extension-side R-hat above 1.01: {bad_rhat}"


# --------------------------------------------------------------------------
# The hyperpriors, which are the part nothing else can see
# --------------------------------------------------------------------------


def test_the_two_hyperprior_jacobians_are_both_present(f1_extension, f1_reference):
    """`tau` and `phi` agree with a reference that spells both Jacobians out.

    This is the same assertion `test_parameter_parity` already makes for these
    two parameters. It exists separately because the *reason* it matters is not
    visible from there, and because it is the assertion most likely to be
    weakened by accident: a future maintainer reaching for `pm.HalfFlat` or a
    `pm.Uniform` with PyMC's own transform would silently change the reference's
    prior, and the only symptom would be this comparison drifting.

    The sign of each term is what a wrong one would move:

    * dropping `+ log tau` reweights toward small `tau`, over-pooling every thin
      part toward the catalogue mean;
    * dropping `- log phi` reweights toward large `phi`, i.e. toward the Poisson
      limit, reporting less burstiness than the data shows and a reorder point
      that stocks out.

    Both are directional errors that a symmetric tolerance would absorb, so the
    check is that the deltas are small, not merely that they are bounded.
    """
    for name in ("tau", "phi"):
        deltas = parity_deltas(
            extension_chained_summary(f1_extension, name).summary, f1_reference[name]
        )
        assert deltas["mean"] < F1_TOL.mean, (
            f"{name} disagrees with a reference that carries both hyperprior "
            f"Jacobians explicitly: mean delta {deltas['mean']:.4f} sd. "
            "Check the `+ log tau` / `- log phi` terms on both sides before "
            "touching the tolerance."
        )


def test_overdispersion_is_found_rather_than_assumed(f1_extension, f1_reference):
    """`phi` must be far from the Poisson limit on data that is overdispersed.

    The fixture is drawn at `phi = 2.5`, i.e. genuinely bursty. If the fit
    returned a large `phi` it would be reporting the Poisson limit, the parity
    comparison would still pass against a reference making the same mistake, and
    the resulting reorder points would be set from intervals that are far too
    tight -- the failure THEORY.md measures at 87.4 % achieved service against a
    95 % nominal.
    """
    phi = extension_chained_summary(f1_extension, "phi").summary
    assert phi.q95 < 20.0, (
        f"phi's 95% point is {phi.q95:.1f}; the fit is reporting something close to the "
        "Poisson limit on data drawn at phi = 2.5"
    )
    # The truth is inside a 90% interval. Not a calibration claim -- one draw of
    # one dataset cannot make one -- but a fixture whose truth sat outside would
    # be a poor choice for measuring agreement.
    assert phi.q05 < F1_TRUTH["phi"] < phi.q95


def test_the_thin_parts_are_pooled_and_the_thick_ones_are_not(f1_extension, f1_data, f1_parts):
    """Partial pooling is the reason this family exists, so the fixture must show it.

    A part with six observations should be pulled toward the catalogue
    substantially harder than one with thirty. If both were pooled the same
    amount, `tau` would effectively not be participating and the per-group
    parity above would be testing a much weaker model than the one shipped.
    """
    counts = f1_data.groupby("part")["units"].agg(["mean", "count"])
    intercept = extension_chained_summary(f1_extension, "intercept").summary.mean

    def retention(cohort_weeks: int) -> float:
        """Least-squares slope of the fitted offset on the unpooled one, through 0.

        A per-part offset is measured against the catalogue level on the log
        scale, which is where the pooling is linear. The slope is 1 when a part
        keeps its own estimate entirely and 0 when it is pulled all the way to
        the catalogue, so *smaller means more pooled*.

        A slope rather than a per-part ratio because a part whose own mean
        happens to sit on the catalogue level has an unpooled offset of nearly
        zero, and a ratio against it is a division by noise -- the first version
        of this test reported a shrinkage of -249 for exactly that reason.
        """
        raw, fitted = [], []
        for part in f1_parts:
            if counts.loc[part, "count"] != cohort_weeks:
                continue
            # +0.5 is the same continuity correction the family's own starting
            # point uses, and it keeps a zero-demand week from taking a log of 0.
            raw.append(np.log(float(counts.loc[part, "mean"]) + 0.5) - intercept)
            fitted.append(extension_chained_summary(f1_extension, "u", group_id=part).summary.mean)
        raw, fitted = np.array(raw), np.array(fitted)
        return float(raw @ fitted / (raw @ raw))

    thin, thick = retention(F1_THIN_WEEKS), retention(30)
    assert thin < thick, (
        f"thin parts ({F1_THIN_WEEKS} weeks) retained {thin:.3f} of their own estimate "
        f"and thick parts (30 weeks) {thick:.3f}; a thin part must be pulled toward the "
        "catalogue *harder* than a thick one, and partial pooling is the reason this "
        "family exists"
    )


# --------------------------------------------------------------------------
# Negative control
# --------------------------------------------------------------------------


def test_a_flattened_catalogue_is_detected(con, f1_data, f1_reference, f1_parts):
    """Replace every part's demand with the catalogue's, leaving the totals alone.

    This is the perturbation `hier_negbin` exists to distinguish from the truth:
    a catalogue in which every part moves at the same rate. The population
    parameters barely move -- the grand mean is unchanged by construction -- so
    only `tau` and the per-part offsets can notice, which is exactly the claim
    the per-group parity assertions make.
    """
    rng = np.random.default_rng(4242)
    flattened = f1_data.copy()
    pooled = f1_data["units"].to_numpy()
    flattened["units"] = rng.permutation(pooled)

    draws = fit_chained(con, flattened, "part, units", "hier_negbin", CONFIG, "f1_flat")
    breached = [
        p
        for p in f1_parts
        if parity_failures(
            extension_chained_summary(draws, "rate", group_id=p).summary,
            f1_reference[f"rate[{p}]"],
            F1_TOL,
        )
    ]
    assert breached, (
        "scrambling demand across parts left every part's rate inside tolerance; the F1 "
        "comparison is not sensitive to the per-part fit that is the whole model"
    )
