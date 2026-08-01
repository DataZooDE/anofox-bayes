"""F7 `conjugate_anomaly` vs PyMC.

The extension draws i.i.d. from a closed-form Normal-Inverse-Gamma or Gamma
posterior. PyMC runs NUTS against the *same* prior and the *same* data. If the
extension's algebra is right the two posteriors are the same distribution, and
their mean, sd, 5% and 95% quantiles agree up to Monte Carlo error.

Priors under test (see crates/anofox-bayes-core/src/catalog/f7_conjugate.rs):

* Normal, default: NIG with mu0 = 0, kappa0 = 0, alpha0 = -1/2, beta0 = 0 --
  the improper reference prior.
* Normal, explicit: a proper NIG, to prove the prior slots are actually wired
  into the posterior rather than being decoration.
* Poisson, default: Gamma(a0 = 1/2, rate b0 = 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_DRAWS,
    EXTENSION_SEED,
    F7_TOL,
    assert_parity,
    extension_summary,
    f7_normal_dataset,
    f7_poisson_dataset,
    fit,
    parity_failures,
    pymc_f7_normal,
    pymc_f7_poisson,
    reference_summaries,
)

NORMAL_LANES = ["HAM-ROT", "BRE-ANT"]
POISSON_CARRIERS = ["CARRIER-A", "CARRIER-B"]


# --------------------------------------------------------------------------
# Fixtures: one PyMC run per (model, group), reused across assertions
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def normal_data():
    return f7_normal_dataset()


@pytest.fixture(scope="session")
def poisson_data():
    return f7_poisson_dataset()


@pytest.fixture(scope="session")
def normal_reference(normal_data):
    """Reference posteriors under the *default* (reference) prior, per lane."""
    out = {}
    for lane in NORMAL_LANES:
        y = normal_data.loc[normal_data["lane"] == lane, "cost"].to_numpy()
        out[lane] = reference_summaries(pymc_f7_normal(y), ["mu", "sigma"])
    return out


@pytest.fixture(scope="session")
def normal_extension(con, normal_data):
    return fit(
        con,
        normal_data,
        "lane, cost",
        "conjugate_anomaly",
        "{'value': 'cost', 'group': 'lane', "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f7_normal_obs",
    )


@pytest.fixture(scope="session")
def poisson_reference(poisson_data):
    out = {}
    for carrier in POISSON_CARRIERS:
        rows = poisson_data[poisson_data["carrier"] == carrier]
        out[carrier] = reference_summaries(
            pymc_f7_poisson(
                rows["claims"].to_numpy(), rows["consignments"].to_numpy()
            ),
            ["lambda"],
        )
    return out


@pytest.fixture(scope="session")
def poisson_extension(con, poisson_data):
    return fit(
        con,
        poisson_data,
        "carrier, claims, consignments",
        "conjugate_anomaly",
        "{'value': 'claims', 'group': 'carrier', 'likelihood': 'poisson', "
        "'exposure': 'consignments', "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f7_poisson_obs",
    )


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lane", NORMAL_LANES)
@pytest.mark.parametrize("param", ["mu", "sigma"])
def test_normal_default_prior_parity(normal_extension, normal_reference, lane, param):
    """Every summary of every parameter of every group, against NUTS."""
    assert_parity(
        f"{param}[{lane}]",
        extension_summary(normal_extension, param, group_id=lane),
        normal_reference[lane][param],
        F7_TOL,
    )


@pytest.mark.parametrize("carrier", POISSON_CARRIERS)
def test_poisson_default_prior_parity(poisson_extension, poisson_reference, carrier):
    assert_parity(
        f"lambda[{carrier}]",
        extension_summary(poisson_extension, "lambda", group_id=carrier),
        poisson_reference[carrier]["lambda"],
        F7_TOL,
    )


# An informative prior chosen to actually move the posterior: kappa0 = 8 is
# worth 8 pseudo-observations against n = 60 real ones, and mu0 = 0 is far from
# the data, so the posterior mean shifts by roughly 12% of the sample mean. A
# prior that changed nothing would make this test vacuous.
INFORMATIVE_PRIOR = dict(mu0=0.0, kappa0=8.0, alpha0=3.0, beta0=4.0)


@pytest.fixture(scope="session")
def informative_reference(normal_data):
    y = normal_data.loc[normal_data["lane"] == "HAM-ROT", "cost"].to_numpy()
    return reference_summaries(pymc_f7_normal(y, **INFORMATIVE_PRIOR), ["mu", "sigma"])


@pytest.fixture(scope="session")
def informative_extension(con, normal_data):
    p = INFORMATIVE_PRIOR
    return fit(
        con,
        normal_data,
        "lane, cost",
        "conjugate_anomaly",
        "{'value': 'cost', 'group': 'lane', "
        f"'prior': {{'mu0': {p['mu0']}, 'kappa0': {p['kappa0']}, "
        f"'alpha0': {p['alpha0']}, 'beta0': {p['beta0']}}}, "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f7_normal_obs",
    )


@pytest.mark.parametrize("param", ["mu", "sigma"])
def test_normal_informative_prior_parity(informative_extension, informative_reference, param):
    """A proper NIG prior must reach PyMC's answer, not just the default one.

    This is what proves the `prior` config slot is plumbed through: the default
    tests above would still pass if the extension ignored `prior` entirely.
    """
    assert_parity(
        f"{param}[HAM-ROT] (informative prior)",
        extension_summary(informative_extension, param, group_id="HAM-ROT"),
        informative_reference[param],
        F7_TOL,
    )


def test_informative_prior_actually_moved_the_posterior(
    informative_extension, normal_extension
):
    """Guard for the guard: if the prior did nothing, the test above is empty."""
    default = extension_summary(normal_extension, "mu", group_id="HAM-ROT")
    informative = extension_summary(informative_extension, "mu", group_id="HAM-ROT")
    shift = abs(default.mean - informative.mean) / default.sd
    assert shift > 3.0, (
        "the informative prior barely moved the posterior "
        f"({shift:.2f} sd), so test_normal_informative_prior_parity proves little"
    )


# --------------------------------------------------------------------------
# Negative control: the harness must be able to fail
# --------------------------------------------------------------------------


def test_wrong_prior_is_detected(informative_extension, normal_reference):
    """Compare an informative-prior fit against the *reference*-prior PyMC run.

    These are genuinely different posteriors, so a parity harness that is doing
    its job must reject them. If this test ever passes silently, every green
    test above is meaningless -- the comparison would be too loose to detect a
    wrong prior, which is the most likely way for this family to be wrong.
    """
    actual = extension_summary(informative_extension, "mu", group_id="HAM-ROT")
    mismatched_reference = normal_reference["HAM-ROT"]["mu"]

    failures = parity_failures(actual, mismatched_reference, F7_TOL)
    assert failures, (
        "a deliberately mismatched prior produced no tolerance breach; the "
        "tolerances in TOLERANCES.md are too loose to gate anything"
    )

    with pytest.raises(AssertionError, match="disagrees with the PyMC reference"):
        assert_parity("mu[HAM-ROT]", actual, mismatched_reference, F7_TOL, record=False)


def test_perturbed_data_is_detected(con, normal_data, normal_reference):
    """The same control from the data side rather than the prior side.

    A shift of 0.5 units against a posterior sd of ~0.26 is a 2-sd move: small
    enough that a numerically sloppy comparison might miss it, large enough that
    a correct one cannot.
    """
    perturbed = normal_data.copy()
    perturbed["cost"] = perturbed["cost"] + np.where(
        perturbed["lane"] == "HAM-ROT", 0.5, 0.0
    )
    draws = fit(
        con,
        perturbed,
        "lane, cost",
        "conjugate_anomaly",
        "{'value': 'cost', 'group': 'lane', "
        f"'draws': {EXTENSION_DRAWS}, 'seed': {EXTENSION_SEED}}}",
        "f7_perturbed_obs",
    )
    actual = extension_summary(draws, "mu", group_id="HAM-ROT")
    assert parity_failures(actual, normal_reference["HAM-ROT"]["mu"], F7_TOL), (
        "a 0.5-unit shift in the data went undetected by the parity comparison"
    )

    # ...and the untouched lane in the same fit still agrees, which shows the
    # detection is localised rather than the comparison simply being noisy.
    assert_parity(
        "mu[BRE-ANT] (unperturbed lane)",
        extension_summary(draws, "mu", group_id="BRE-ANT"),
        normal_reference["BRE-ANT"]["mu"],
        F7_TOL,
    )
