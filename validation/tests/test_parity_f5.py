"""F5 `payer_alive` vs PyMC.

BG/NBD has no closed form and no PyMC distribution, so the reference is a custom
`logp` through `pm.Potential` -- the likelihood written out a second time from
Fader-Hardie-Lee rather than shared with the implementation under test. That is
the whole value of the comparison: this is the hardest likelihood in the
catalogue, four `lnGamma` blocks and a two-branch log-sum-exp, and it is the one
place where a transcription error would be least likely to look wrong.

Like F2, this family is served by Laplace, so it is checked against **two**
references for the two separable questions -- see `test_parity_f2.py` for why
one number cannot answer both:

1. `F5_EXACT_TOL`, against an independently located mode and observed
   information. This is the algebra gate, and it is where a wrong `lnGamma`
   argument or a dropped branch would show.
2. `F5_TOL`, against NUTS on the true posterior. The approximation budget.

**Both sides run under the proper prior, not the default.** The default is flat
on the log scale, which makes the target improper except for the family's hard
`|log theta| <= 30` box -- and a NUTS reference on a target that is only proper
because of a box constraint is not a reference, it is a random walk with a
fence. The proper prior is also the configuration F5's SBC suite certifies, so
the two harnesses are looking at the same model.

The measured answer: the algebra is **exact** (0.003 sd), and the Laplace
approximation is very good for `r` and `alpha` and noticeably worse for `a` and
`b` -- which is what the family's own documentation predicts, because `a` and
`b` are only weakly separately identified. That prediction is turned into an
assertion below rather than left as a remark.
"""

from __future__ import annotations

import numpy as np
import pytest

from _support import (
    EXTENSION_LAPLACE_DRAWS,
    EXTENSION_SEED,
    F5_EXACT_TOL,
    F5_PARAMS,
    F5_TOL,
    F5_TRUTH,
    assert_parity,
    extension_summary,
    f5_dataset,
    f5_neg_log_posterior,
    f5_prior_config,
    f5_start,
    fit,
    fit_metadata,
    gaussian_summaries,
    laplace_reference,
    log_scale_summary,
    parity_deltas,
    parity_failures,
    pymc_f5_bgnbd,
    reference_summaries,
)

LOG_PARAMS = [f"log_{name}" for name in F5_PARAMS]

CONFIG = (
    "{'frequency': 'frequency', 'recency': 'recency', 'age': 'age', "
    f"'draws': {EXTENSION_LAPLACE_DRAWS}, 'seed': {EXTENSION_SEED}, "
    f"'prior': {f5_prior_config()}}}"
)
COLUMNS = "frequency, recency, age"


@pytest.fixture(scope="session")
def f5_data():
    return f5_dataset()


@pytest.fixture(scope="session")
def f5_extension(con, f5_data):
    return fit(con, f5_data, COLUMNS, "payer_alive", CONFIG, "f5_payers")


@pytest.fixture(scope="session")
def f5_metadata(con, f5_data):
    return fit_metadata(con, f5_data, COLUMNS, "payer_alive", CONFIG, "f5_meta")


@pytest.fixture(scope="session")
def f5_laplace(f5_data):
    mode, cov = laplace_reference(f5_neg_log_posterior(f5_data), f5_start(f5_data))
    return gaussian_summaries(mode, cov, LOG_PARAMS), mode, cov


@pytest.fixture(scope="session")
def f5_nuts(f5_data):
    return reference_summaries(pymc_f5_bgnbd(f5_data), F5_PARAMS + LOG_PARAMS)


def test_the_fit_converged_and_was_served_by_laplace(f5_metadata):
    """`__engine__ = 1` is `laplace`.

    THEORY.md settles the choice by SBC rather than by argument: over 1_024
    replications at 800 customers all four rank histograms are uniform, and stay
    uniform down to 100. NUTS would cost a large multiple of the runtime to
    reproduce a posterior that is already calibrated. This asserts the engine
    that was certified is the one that ran.
    """
    assert f5_metadata["__status__"] == 0.0, "the F5 fit did not report `converged`"
    assert f5_metadata["__engine__"] == 1.0, "the F5 fit was not served by the Laplace engine"
    assert f5_metadata["__n_obs__"] == 600.0


def test_the_fixture_contains_all_three_kinds_of_customer(f5_data):
    """A base that is all one kind leaves most of the likelihood unexercised.

    The `x = 0` branch is genuinely different code: the dead term is absent and
    the whole Beta block cancels, so `a` and `b` drop out of that customer's
    contribution entirely. A fixture without never-repeaters would never
    evaluate it, and one without customers who have gone quiet would hit the
    boundary the family refuses.
    """
    never = int((f5_data["frequency"] == 0).sum())
    assert never > 50, f"only {never} customers never repeated; the x = 0 branch is thin"
    repeat = f5_data[f5_data["frequency"] > 0]
    quiet = int((repeat["age"] - repeat["recency"] > 10.0).sum())
    assert quiet > 50, (
        f"only {quiet} repeat customers have been silent for more than 10 periods; the "
        "dropout process is what makes this model identifiable"
    )


# --------------------------------------------------------------------------
# 1. The algebra gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", F5_PARAMS)
def test_the_laplace_posterior_is_the_one_the_likelihood_implies(f5_extension, f5_laplace, name):
    """Each parameter's draws against an independently located mode and curvature.

    Compared on the log scale, which is where the draws are exactly Gaussian --
    the extension samples `(log r, log alpha, log a, log b)` and exponentiates,
    so the natural-scale marginals are lognormal by construction and comparing
    them to a normal would fail for a reason that says nothing about BG/NBD.

    This is the assertion that pins the likelihood itself: the seven `lnGamma`
    terms, the `r ln(alpha)`, the two-branch bracket and its `a/(b + x - 1)`
    weight, and the fact that the log-scale prior contributes no Jacobian.
    """
    summaries, _, _ = f5_laplace
    assert_parity(
        f"log_{name} (vs Laplace)",
        log_scale_summary(f5_extension, name),
        summaries[f"log_{name}"],
        F5_EXACT_TOL,
    )


def test_the_located_mode_is_stationary_and_interior(f5_laplace, f5_data):
    """The reference found a real interior mode, not a boundary or a second basin.

    BG/NBD has a much worse local optimum reachable from `a = b = 1`, and the
    extension guards against it with a trust region -- THEORY's note records a
    first unguarded Newton step moving `log a` by +14.6 to a point where the log
    posterior was 248 lower. If the reference had landed there instead, the
    algebra gate above would fail while the extension was right, so this asserts
    the reference is where it should be before its verdict is trusted.
    """
    _, mode, cov = f5_laplace
    assert np.all(np.abs(mode) < 25.0), (
        f"the reference mode {mode} is outside the admissible range the family "
        "requires its own mode to sit inside"
    )
    assert np.all(np.linalg.eigvalsh(cov) > 0.0), (
        "the reference covariance is not positive definite, so the located point is "
        "not a maximum and cannot referee anything"
    )
    # The truth is recovered. Not a calibration claim, but a wrong basin would
    # not sit near the parameters the fixture was generated from.
    for j, name in enumerate(F5_PARAMS):
        assert abs(np.exp(mode[j]) - F5_TRUTH[name]) < 0.5 * F5_TRUTH[name], (
            f"{name} recovered as {np.exp(mode[j]):.3f} against a true {F5_TRUTH[name]}"
        )


# --------------------------------------------------------------------------
# 2. The approximation budget
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", F5_PARAMS)
def test_parameter_parity(f5_extension, f5_nuts, name):
    """The reported (natural-scale) parameters against NUTS on the true posterior.

    Natural scale here, not log: this is what the draws table actually contains
    and what a collections agent's SQL would read, so it is the comparison that
    describes the shipped answer.
    """
    assert_parity(name, extension_summary(f5_extension, name), f5_nuts[name], F5_TOL)


def test_the_purchase_process_is_approximated_far_better_than_the_dropout_process(
    f5_extension, f5_nuts
):
    """`r` and `alpha` agree with NUTS an order of magnitude better than `a` and `b`.

    This is the family's own documented prediction turned into an assertion. The
    data speaks clearly about where the Beta sits -- the dropout *rate* -- and
    only faintly about how wide it is, so `(log a, log b)` has a long curved
    ridge along fixed `a/(a+b)` that a Gaussian at the mode fits poorly, while
    `r` and `alpha` are well identified and near-quadratic.

    Worth knowing before reading an `a` or `b` interval: measured here, `r` and
    `alpha` agree to under 0.01 reference sd, and `a` and `b` to ~0.06 sd with
    intervals ~4 % too narrow.
    """
    error = {
        name: abs(extension_summary(f5_extension, name).mean - f5_nuts[name].mean)
        / f5_nuts[name].sd
        for name in F5_PARAMS
    }
    purchase = max(error["r"], error["alpha"])
    dropout = max(error["a"], error["b"])
    assert purchase < 0.02, (
        f"the well-identified purchase parameters disagree by {purchase:.4f} sd; Laplace "
        "should be nearly exact for these, so this is more likely an algebra change "
        f"than an approximation one. All: {error}"
    )
    assert dropout > 2.0 * purchase, (
        f"the dropout parameters' error ({dropout:.4f} sd) is no longer distinguishable "
        f"from the purchase parameters' ({purchase:.4f} sd); the documented shape of "
        "this posterior has changed"
    )


def test_the_dropout_intervals_are_narrow_rather_than_wide(f5_extension, f5_nuts):
    """Where Laplace is wrong on this family, it is over-confident.

    A Gaussian at the mode cannot reproduce the upper tail of a right-skewed
    ridge, so `a` and `b` come out tight. That is the direction that matters for
    a collections agent: an over-confident dropout distribution understates how
    uncertain `P(alive)` is for the customers in the middle of the book, who are
    precisely the ones a dunning decision is marginal for.
    """
    ratios = {
        name: extension_summary(f5_extension, name).sd / f5_nuts[name].sd for name in ("a", "b")
    }
    for name, ratio in ratios.items():
        assert 0.90 < ratio < 1.0, (
            f"sd({name}) ratio extension/NUTS = {ratio:.4f}, outside the [0.90, 1.0] band "
            "measured for this fixture. Above 1 would mean Laplace is no longer the "
            "narrow side, which would contradict how the approximation works."
        )


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------


def test_a_base_in_which_nobody_has_lapsed_is_refused(con, f5_data):
    """Truncate every repeat payer's window at their last payment.

    This is the shape a snapshot taken at the renewal date produces, and it is
    the boundary case BG/NBD cannot answer: with no repeat buyer ever seen to go
    quiet, the likelihood keeps increasing as the dropout probability goes to
    zero and there is no interior maximum at all. The right answer is a refusal
    with NULL draws, not a confident number derived from curvature that is not a
    posterior -- and a parity suite that only ever fits well-behaved data would
    never find out which one it gets.
    """
    never_lapsed = f5_data.copy()
    repeat = never_lapsed["frequency"] > 0
    never_lapsed.loc[repeat, "age"] = never_lapsed.loc[repeat, "recency"]

    # Under the *default* (flat) prior, which is where the boundary bites.
    default_config = (
        "{'frequency': 'frequency', 'recency': 'recency', 'age': 'age', 'draws': 1000}"
    )
    metadata = fit_metadata(
        con, never_lapsed, COLUMNS, "payer_alive", default_config, "f5_never_lapsed"
    )
    assert metadata["__status__"] == 1.0, (
        "a base in which nobody has ever been seen to stop was not refused; the family "
        "reported a status other than `degenerate`"
    )

    draws = fit(con, never_lapsed, COLUMNS, "payer_alive", default_config, "f5_never_lapsed_d")
    for name in F5_PARAMS:
        values = draws[draws["param"] == name]["value"].to_numpy()
        assert np.isnan(values).all(), (
            f"{name} carries real draws on a refused fit; the draws contract reserves "
            "NULL for 'not estimable', and a number there is indistinguishable from an "
            "estimate"
        )


def test_a_wrong_prior_is_detected(con, f5_data, f5_laplace):
    """Refit under a different proper prior and check the comparison objects.

    The `prior` slot is the one input to this family that a caller cannot verify
    by eye, and every test above uses the same prior on both sides -- so if the
    slot were ignored entirely, they would all still pass. This is the test that
    says it is not.
    """
    shifted = ("{'r': {'log_mean': 1.5, 'log_sd': 0.3}, 'alpha': {'log_mean': 3.5, 'log_sd': 0.3}, "
               "'a': {'log_mean': 1.5, 'log_sd': 0.3}, 'b': {'log_mean': 2.0, 'log_sd': 0.3}}")
    config = (
        "{'frequency': 'frequency', 'recency': 'recency', 'age': 'age', "
        f"'draws': {EXTENSION_LAPLACE_DRAWS}, 'seed': {EXTENSION_SEED}, 'prior': {shifted}}}"
    )
    draws = fit(con, f5_data, COLUMNS, "payer_alive", config, "f5_wrong_prior")

    summaries, _, _ = f5_laplace
    breached = [
        name
        for name in F5_PARAMS
        if parity_failures(
            log_scale_summary(draws, name), summaries[f"log_{name}"], F5_EXACT_TOL
        )
    ]
    assert breached, (
        "a materially different prior produced no breach against the original "
        "reference; the `prior` slot is not reaching the posterior"
    )


def test_scrambling_recency_against_frequency_is_detected(con, f5_data, f5_laplace):
    """Permute recency within the repeat buyers, leaving every marginal intact.

    Recency only means something *relative to* a customer's own frequency and
    observation window -- three weeks of silence from a weekly buyer is not
    three weeks of silence from a twice-a-year buyer. Permuting it destroys that
    pairing while leaving the distribution of recencies, of frequencies and of
    ages exactly as they were, so only a fit that genuinely uses the joint
    structure can notice.
    """
    scrambled = f5_data.copy()
    repeat = scrambled["frequency"] > 0
    rng = np.random.default_rng(31337)
    values = scrambled.loc[repeat, "recency"].to_numpy()
    scrambled.loc[repeat, "recency"] = rng.permutation(values)
    # Permuting can put a recency past its own customer's age, which the family
    # rejects as a config error rather than a status; clip so the test exercises
    # the fit rather than the validator.
    scrambled["recency"] = np.minimum(scrambled["recency"], scrambled["age"])

    draws = fit(con, scrambled, COLUMNS, "payer_alive", CONFIG, "f5_scrambled")
    summaries, _, _ = f5_laplace
    breached = {
        name: parity_deltas(log_scale_summary(draws, name), summaries[f"log_{name}"])
        for name in F5_PARAMS
        if parity_failures(
            log_scale_summary(draws, name), summaries[f"log_{name}"], F5_EXACT_TOL
        )
    }
    assert breached, (
        "scrambling recency against frequency left every parameter inside tolerance; "
        "the comparison is not sensitive to the pairing the model is built on"
    )
