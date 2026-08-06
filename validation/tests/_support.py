"""Shared machinery for the PyMC parity tests.

Everything here exists to make one comparison honest: the *same* dataset, fitted
by the extension's closed-form sampler and by a PyMC model that encodes the
*same* prior, must produce the same posterior summaries.

Two things make that comparison non-trivial and are handled here explicitly.

1. **The priors the extension defaults to are improper reference priors.** PyMC
   cannot express them with an off-the-shelf distribution, so they are built out
   of `pm.Flat` on the right transformed scale (plus a `pm.Potential` where a
   power of the parameter is needed). Getting this wrong is the single easiest
   way to make a parity suite vacuous, so every reference model below carries a
   derivation in its docstring.

2. **Both sides are Monte Carlo.** The extension draws i.i.d. from a closed-form
   posterior; PyMC runs NUTS. Neither is exact, so the tolerances are expressed
   in units of the reference posterior's own standard deviation, which is the
   only scale-free unit available. See TOLERANCES.md.
"""

from __future__ import annotations

import dataclasses
import os

# PyTensor's default backend is compiled C; in a container without a toolchain
# it falls back noisily. Leave the user's choice alone if they set one.
os.environ.setdefault("PYTENSOR_FLAGS", "")

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import optimize
from scipy.special import gammaln

# --------------------------------------------------------------------------
# Sampling budgets
# --------------------------------------------------------------------------

# The extension samples i.i.d. from a closed form, so its Monte Carlo error is
# sd/sqrt(N). 40_000 draws puts MCSE(mean) at 0.005 sd, an order of magnitude
# under every tolerance below, which keeps the extension side from being what
# the test is measuring.
EXTENSION_DRAWS = 40_000
EXTENSION_SEED = 20260801

# The two NUTS-served families (F1 `hier_negbin`, F8 `varying_variance_gaussian`)
# do *not* draw i.i.d. -- the extension side is a Markov chain too, so its Monte
# Carlo error is sd/sqrt(ESS) with ESS measured, not sd/sqrt(draws). On the
# hierarchical scale parameters ESS runs an order of magnitude below the draw
# count, so the budget is set here rather than inherited from EXTENSION_DRAWS.
# 4 x 4_000 rather than THEORY.md's "4 x 2000 clears the R-hat gate": clearing
# the gate is the floor for a usable fit, not the floor for refereeing one.
EXTENSION_NUTS_DRAWS = 4_000
EXTENSION_NUTS_CHAINS = 4
EXTENSION_NUTS_WARMUP = 1_000

# The Laplace-served families (F2 `censored_aft`, F5 `payer_alive`) draw i.i.d.
# from the fitted Gaussian, so the same sd/sqrt(N) logic as EXTENSION_DRAWS
# applies. They are compared twice -- once against an independently computed
# mode and observed information, once against NUTS -- and the first of those is
# tight enough that the extension side must not be the limiting term.
EXTENSION_LAPLACE_DRAWS = 200_000

# NUTS is the noisier side. 4 chains x 5_000 draws after 2_000 tuning gives
# ESS in the 10k-20k range on these (easy, low-dimensional) posteriors, so
# MCSE(mean) lands near 0.01 sd.
PYMC_CHAINS = 4
PYMC_DRAWS = 5_000
PYMC_TUNE = 2_000
PYMC_SEED = 20260801

# Below these the *reference* is not trustworthy and a "pass" would mean
# nothing. Gate on the reference before comparing anything to it.
MIN_ESS = 1_000
MAX_RHAT = 1.01


def _sample_kwargs(**overrides):
    kwargs = dict(
        draws=PYMC_DRAWS,
        tune=PYMC_TUNE,
        chains=PYMC_CHAINS,
        cores=min(PYMC_CHAINS, os.cpu_count() or 1),
        random_seed=PYMC_SEED,
        progressbar=False,
    )
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------


# The suite compares 126 parameters on four statistics each. A limit sized for one
# comparison is the wrong size for 504 of them: at the 3.5 sigma TOLERANCES.md
# derived, the expected number of spurious failures is 0.23 per run, so **one run in
# five fails on nothing**. That is not a hypothetical -- it is what sent `tau` over
# the line on an unrelated PR, on two families at once, while the same commit passed
# on re-run.
#
# So the sigma is chosen family-wise: the probability that *any* of the 504
# comparisons trips when both implementations are correct is held at 1 %.
#
# `tau` is always the first to go, and for a reason worth keeping in view: it is the
# hierarchical scale, its ESS is the lowest in any family that has one, and MCSE on
# an sd goes as 1/sqrt(2*ESS).
PARITY_FAMILY_WISE_ALPHA = 0.01
PARITY_COMPARISON_BUDGET = 504
PER_COMPARISON_SIGMA_BEFORE = 3.5
PARITY_SIGMA = 4.27

# The per-comparison derivation TOLERANCES.md sets out, kept verbatim so the
# family-wise factor is visibly a *scaling* of it rather than a new set of numbers
# picked to make the suite quiet. `test_tolerance_sizing.py` asserts the relation.
TOLERANCE_BASELINE = {
    "F7 conjugate_anomaly": {"mean": 0.05, "sd": 0.05, "quantile": 0.09},
    "F3 pooled_gaussian": {"mean": 0.07, "sd": 0.07, "quantile": 0.12},
    "F8 varying_variance_gaussian": {"mean": 0.09, "sd": 0.06, "quantile": 0.17},
    "F1 hier_negbin": {"mean": 0.11, "sd": 0.08, "quantile": 0.23},
    "F4 payment_delay": {"mean": 0.10, "sd": 0.08, "quantile": 0.20},
    "F6 hier_elasticity": {"mean": 0.13, "sd": 0.10, "quantile": 0.26},
    "F2 censored_aft (vs Laplace)": {"mean": 0.02, "sd": 0.02, "quantile": 0.04},
    "F5 payer_alive (vs Laplace)": {"mean": 0.02, "sd": 0.02, "quantile": 0.04},
}


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """A tolerance triple, in units documented in TOLERANCES.md.

    `mean` and `quantile` are multiples of the reference posterior sd; `sd` is a
    relative tolerance. Expressing location tolerances in sd units rather than in
    absolute units is what lets one number cover a coefficient measured in euros
    and a rate measured in claims-per-consignment.
    """

    mean: float
    sd: float
    quantile: float
    label: str = ""


# F7 is closed-form on both sides and the PyMC posteriors are one- or
# two-dimensional and near-Gaussian, so NUTS mixes essentially perfectly
# (ESS ~= the raw draw count). Combined MCSE is ~0.012 sd on a mean and ~0.026
# sd on a 5%/95% endpoint; the numbers below are roughly 3.5x that. See
# TOLERANCES.md for the derivation and the measured margins.
F7_TOL = Tolerance(mean=0.061, sd=0.061, quantile=0.110, label="F7 conjugate_anomaly")

# F3's grouped posterior has a ridge-identified intercept/group-effect block, so
# NUTS autocorrelation is materially higher and the reference MCSE roughly
# doubles. The sd tolerance is deliberately ~3x the systematic ~2% interval
# deficit documented in test_parity_f3.py -- that deficit is measured directly
# rather than caught here, because a tolerance tight enough to catch it would
# flake on the panel's own Monte Carlo error.
F3_TOL = Tolerance(mean=0.086, sd=0.086, quantile=0.147, label="F3 pooled_gaussian")

# --- The NUTS-served families. Both sides are Markov chains, so the floor is
# sd/sqrt(ESS) on each side with ESS *measured* (see TOLERANCES.md for the
# arithmetic and the measured ESS these were derived from).
F8_TOL = Tolerance(mean=0.110, sd=0.074, quantile=0.208, label="F8 varying_variance_gaussian")
F1_TOL = Tolerance(mean=0.135, sd=0.098, quantile=0.281, label="F1 hier_negbin")

# F4 sits between F8 and F1. Its `tau` runs through the same intercept/offset
# ridge, but with only six segments there are fewer coordinates in it, and the
# Gamma likelihood is better conditioned than the negative binomial's -- the
# dispersion enters the log-density through `lnGamma` rather than through a
# `(y + phi) log(phi + mu)` that couples it to every mean.
F4_TOL = Tolerance(mean=0.122, sd=0.098, quantile=0.244, label="F4 payment_delay")

# F6 is the loosest of the NUTS-served set, and deliberately. It carries *two*
# pooling scales rather than one, so there are two ridges rather than one, and
# the `-exp` transform on the elasticity means the per-segment coefficients are
# a nonlinear function of coordinates that are themselves poorly conditioned.
# Measured ESS on `tau` is the binding constraint; see TOLERANCES.md.
F6_TOL = Tolerance(mean=0.159, sd=0.122, quantile=0.318, label="F6 hier_elasticity")

# --- The Laplace-served families. These carry *two* tolerances each, because
# there are two different questions and one number cannot answer both.
#
# `*_EXACT_TOL` is the algebra gate: the extension's draws against a mode and an
# observed information matrix computed independently in this file. Both sides
# describe the same Gaussian, so the only irreducible error is Monte Carlo on
# the extension's 200_000 i.i.d. draws plus the finite-difference Hessian, and
# the tolerance is correspondingly tight. This is the test that would catch a
# wrong constant in the likelihood.
#
# `*_TOL` is the approximation budget: the same draws against NUTS. Laplace is
# an approximation, so this comparison can never be tight, and pretending
# otherwise would mean loosening the algebra gate to accommodate it. It is
# derived from the *measured* approximation error rather than from Monte Carlo.
F2_EXACT_TOL = Tolerance(mean=0.025, sd=0.025, quantile=0.049, label="F2 censored_aft (vs Laplace)")
F2_TOL = Tolerance(mean=0.12, sd=0.05, quantile=0.26, label="F2 censored_aft")
F5_EXACT_TOL = Tolerance(mean=0.025, sd=0.025, quantile=0.049, label="F5 payer_alive (vs Laplace)")
F5_TOL = Tolerance(mean=0.15, sd=0.11, quantile=0.38, label="F5 payer_alive")

# Every tolerance the suite defines, so conftest's margin report does not have to
# be edited each time a family is added -- a report that silently omits a family
# is worse than no report.
ALL_TOLERANCES = (
    F7_TOL,
    F3_TOL,
    F8_TOL,
    F1_TOL,
    F4_TOL,
    F6_TOL,
    F2_TOL,
    F2_EXACT_TOL,
    F5_TOL,
    F5_EXACT_TOL,
)


# --------------------------------------------------------------------------
# Posterior summaries
# --------------------------------------------------------------------------

CI_PROB = 0.90  # -> 5% and 95% endpoints, which is what the deliverable asks for


@dataclasses.dataclass(frozen=True)
class Summary:
    mean: float
    sd: float
    q05: float
    q95: float

    def __str__(self) -> str:
        return (
            f"mean={self.mean:+.6g} sd={self.sd:.6g} "
            f"q05={self.q05:+.6g} q95={self.q95:+.6g}"
        )


def extension_summary(draws: pd.DataFrame, param: str, group_id: str = "__global__") -> Summary:
    """Summarise one parameter of an `anofox_bayes_fit` result.

    `draw >= 0` filtering is the draws contract's way of separating posterior
    draws from metadata and sampler statistics; see docs/DRAWS_CONTRACT.md.
    """
    sel = draws[(draws["param"] == param) & (draws["group_id"] == group_id)]
    values = sel["value"].to_numpy(dtype=float)
    if values.size == 0:
        raise AssertionError(
            f"no draws for param={param!r} group_id={group_id!r}; "
            f"available: {sorted(set(zip(draws['group_id'], draws['param'])))}"
        )
    if np.isnan(values).any():
        raise AssertionError(
            f"param={param!r} group_id={group_id!r} contains NULL draws, which the "
            "draws contract reserves for 'not estimable' -- the fit refused"
        )
    return Summary(
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)),
        q05=float(np.quantile(values, 0.05)),
        q95=float(np.quantile(values, 0.95)),
    )


@dataclasses.dataclass(frozen=True)
class ChainedSummary:
    """A `Summary` plus the diagnostics that say what its Monte Carlo error is.

    Only meaningful for a family the extension serves with NUTS. For the exact
    and Laplace engines the draws are i.i.d. and ESS is the draw count by
    construction, so asking for it would be inviting a false precision.
    """

    summary: Summary
    ess_bulk: float
    ess_tail: float
    r_hat: float


def extension_chained_summary(
    draws: pd.DataFrame, param: str, group_id: str = "__global__"
) -> ChainedSummary:
    """Summarise one parameter of a NUTS-served fit, with its own ESS and R-hat.

    The extension is the *subject* of this suite, not a trusted reference, so its
    diagnostics are not used to decide whether it converged -- that is what the
    `__status__` row is for. They are used to size the tolerance: a delta cannot
    be attributed to the extension's algebra if it is inside the extension's own
    Monte Carlo error, and that error is sd/sqrt(ESS), never sd/sqrt(draws).
    """
    sel = draws[(draws["param"] == param) & (draws["group_id"] == group_id)]
    if sel.empty:
        raise AssertionError(
            f"no draws for param={param!r} group_id={group_id!r}; "
            f"available: {sorted(set(zip(draws['group_id'], draws['param'])))}"
        )
    sel = sel.sort_values(["chain", "draw"])
    values = sel["value"].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise AssertionError(
            f"param={param!r} group_id={group_id!r} contains NULL draws, which the "
            "draws contract reserves for 'not estimable' -- the fit refused"
        )
    n_chains = int(sel["chain"].nunique())
    per_chain = values.reshape(n_chains, -1)
    idata = az.from_dict({"posterior": {"v": per_chain}})
    return ChainedSummary(
        summary=Summary(
            mean=float(values.mean()),
            sd=float(values.std(ddof=1)),
            q05=float(np.quantile(values, 0.05)),
            q95=float(np.quantile(values, 0.95)),
        ),
        ess_bulk=float(az.ess(idata, var_names=["v"], method="bulk").v.values),
        ess_tail=float(az.ess(idata, var_names=["v"], method="tail").v.values),
        r_hat=float(az.rhat(idata, var_names=["v"]).v.values),
    )


# --------------------------------------------------------------------------
# An independent Laplace reference
# --------------------------------------------------------------------------


def laplace_reference(neg_log_posterior, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locate a mode and its observed information, independently of everything.

    The extension's Laplace engine finds the mode by damped Newton on an
    *analytic* gradient and inverts a Hessian obtained by differencing that
    gradient. This function does neither: it minimises with Nelder-Mead followed
    by BFGS on a log posterior written out separately in NumPy, and differences
    a *numerical* gradient. Two implementations that share no code and no
    derivation is the point -- an error copied into the reference would make the
    comparison vacuous.

    Returns `(mode, covariance)` on whatever scale `neg_log_posterior` is written
    in, which for both Laplace families here is the unconstrained (log) scale.
    """
    coarse = optimize.minimize(
        neg_log_posterior,
        np.asarray(start, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1e-13, "fatol": 1e-13, "maxiter": 200_000, "maxfev": 200_000},
    )
    fine = optimize.minimize(neg_log_posterior, coarse.x, method="BFGS", options={"gtol": 1e-10})
    mode = fine.x
    dim = mode.size

    def numerical_gradient(theta: np.ndarray, rel: float = 1e-6) -> np.ndarray:
        out = np.zeros(dim)
        for j in range(dim):
            step = np.zeros(dim)
            step[j] = rel * max(abs(theta[j]), 1.0)
            out[j] = (neg_log_posterior(theta + step) - neg_log_posterior(theta - step)) / (
                2.0 * step[j]
            )
        return out

    hessian = np.zeros((dim, dim))
    for j in range(dim):
        step = np.zeros(dim)
        step[j] = 1e-5 * max(abs(mode[j]), 1.0)
        hessian[:, j] = (
            numerical_gradient(mode + step) - numerical_gradient(mode - step)
        ) / (2.0 * step[j])
    # The observed information is symmetric; differencing is not, so the
    # asymmetry is pure numerical error and averaging halves it.
    hessian = 0.5 * (hessian + hessian.T)
    return mode, np.linalg.inv(hessian)


def gaussian_summaries(mode: np.ndarray, cov: np.ndarray, names: list[str]) -> dict[str, Summary]:
    """The exact marginal summaries of `N(mode, cov)`, coordinate by coordinate.

    Closed form rather than sampled: this is the reference the extension's
    Laplace draws are checked against, and a sampled reference would put Monte
    Carlo error on both sides of the tightest comparison in the suite.
    """
    sd = np.sqrt(np.diag(cov))
    # 5% and 95% points of a standard normal, to the precision of the constant.
    z95 = 1.6448536269514722
    return {
        name: Summary(
            mean=float(mode[j]),
            sd=float(sd[j]),
            q05=float(mode[j] - z95 * sd[j]),
            q95=float(mode[j] + z95 * sd[j]),
        )
        for j, name in enumerate(names)
    }


def log_scale_summary(draws: pd.DataFrame, param: str, group_id: str = "__global__") -> Summary:
    """Summarise `log(param)` from a set of extension draws.

    Both Laplace families report every positive parameter on the natural scale,
    having sampled it on the log scale. Taking the log back puts the comparison
    on the coordinate the approximation is actually Gaussian on, which is where
    a discrepancy means something about the algebra rather than about the
    lognormal shape the exponential imposes.
    """
    sel = draws[(draws["param"] == param) & (draws["group_id"] == group_id)]
    values = sel["value"].to_numpy(dtype=float)
    if values.size == 0:
        raise AssertionError(f"no draws for param={param!r} group_id={group_id!r}")
    if np.isnan(values).any():
        raise AssertionError(f"param={param!r} group_id={group_id!r} contains NULL draws")
    values = np.log(values)
    return Summary(
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)),
        q05=float(np.quantile(values, 0.05)),
        q95=float(np.quantile(values, 0.95)),
    )


def reference_summaries(idata, var_names: list[str]) -> dict[str, Summary]:
    """ArviZ summaries of a PyMC posterior, keyed by the label ArviZ assigns.

    Also asserts the reference itself converged. A parity test whose reference
    is garbage passes for the wrong reason, which is worse than failing.
    """
    table = az.summary(
        idata,
        var_names=var_names,
        kind="all",
        ci_prob=CI_PROB,
        ci_kind="eti",
        round_to="none",
    )
    lo_col = next(c for c in table.columns if c.endswith("_lb"))
    hi_col = next(c for c in table.columns if c.endswith("_ub"))

    import os as _os
    if _os.environ.get("ANOFOX_DUMP_REFERENCE_ESS"):
        print("\n--- PyMC reference diagnostics ---")
        print(table[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]].to_string())
    bad = table[(table["ess_bulk"] < MIN_ESS) | (table["ess_tail"] < MIN_ESS) | (table["r_hat"] > MAX_RHAT)]
    if len(bad):
        raise AssertionError(
            "the PyMC reference did not converge, so it cannot referee anything:\n"
            f"{bad[['ess_bulk', 'ess_tail', 'r_hat']]}"
        )

    return {
        str(label): Summary(
            mean=float(row["mean"]),
            sd=float(row["sd"]),
            q05=float(row[lo_col]),
            q95=float(row[hi_col]),
        )
        for label, row in table.iterrows()
    }


# --------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------


def parity_deltas(actual: Summary, reference: Summary) -> dict[str, float]:
    """Scale-free discrepancies between an extension fit and its reference.

    `mean`, `q05` and `q95` are in units of the reference sd; `sd` is relative.
    """
    scale = reference.sd
    return {
        "mean": abs(actual.mean - reference.mean) / scale,
        "sd": abs(actual.sd - reference.sd) / reference.sd,
        "q05": abs(actual.q05 - reference.q05) / scale,
        "q95": abs(actual.q95 - reference.q95) / scale,
    }


def parity_failures(actual: Summary, reference: Summary, tol: Tolerance) -> dict[str, float]:
    """The subset of `parity_deltas` that breaches `tol`. Empty means agreement."""
    limits = {"mean": tol.mean, "sd": tol.sd, "q05": tol.quantile, "q95": tol.quantile}
    deltas = parity_deltas(actual, reference)
    return {k: v for k, v in deltas.items() if v > limits[k]}


# Every comparison the suite makes, so the terminal summary can report how much
# headroom each tolerance actually had. A suite whose margins are unknown is one
# nobody can tighten with confidence.
MARGINS: list[tuple[str, str, dict[str, float]]] = []


def assert_parity(
    name: str,
    actual: Summary,
    reference: Summary,
    tol: Tolerance,
    stats=None,
    record: bool = True,
) -> None:
    """Fail unless every requested statistic agrees within `tol`.

    `stats` restricts the comparison to a subset of the four statistics.
    `record=False` keeps a deliberately mismatched comparison -- the negative
    controls -- out of the margin report, where it would swamp the real margins.
    """
    if record:
        MARGINS.append((tol.label, name, parity_deltas(actual, reference)))

    failures = parity_failures(actual, reference, tol)
    if stats is not None:
        failures = {k: v for k, v in failures.items() if k in stats}
    if failures:
        limits = {"mean": tol.mean, "sd": tol.sd, "q05": tol.quantile, "q95": tol.quantile}
        detail = ", ".join(f"{k}: {v:.4f} > {limits[k]:.4f}" for k, v in sorted(failures.items()))
        raise AssertionError(
            f"[{tol.label}] {name} disagrees with the PyMC reference: {detail}\n"
            f"  extension: {actual}\n"
            f"  reference: {reference}\n"
            "  (mean/q05/q95 deltas are in units of the reference sd; sd is relative)"
        )


# --------------------------------------------------------------------------
# Fitting through the extension
# --------------------------------------------------------------------------


def fit(con, frame: pd.DataFrame, columns: str, family: str, config: str, table: str) -> pd.DataFrame:
    """Register `frame` and run one `anofox_bayes_fit`, returning posterior draws.

    The table function takes a *subquery*, not a `TABLE` reference, which is why
    the call is spelled this way.
    """
    con.register(table, frame)
    return con.sql(
        f"""
        SELECT group_id, param, value
        FROM anofox_bayes_fit((SELECT {columns} FROM {table}), '{family}', {config})
        WHERE draw >= 0
        """
    ).df()


def fit_chained(
    con, frame: pd.DataFrame, columns: str, family: str, config: str, table: str
) -> pd.DataFrame:
    """`fit`, but keeping `chain` and `draw`.

    A NUTS-served family's draws are autocorrelated, so its Monte Carlo error can
    only be estimated with the chain structure intact. `fit` discards it, which
    is the right default for the exact and Laplace engines where the draws are
    i.i.d. and the columns would only invite a meaningless ESS.
    """
    con.register(table, frame)
    return con.sql(
        f"""
        SELECT group_id, chain, draw, param, value
        FROM anofox_bayes_fit((SELECT {columns} FROM {table}), '{family}', {config})
        WHERE draw >= 0
        """
    ).df()


def fit_metadata(con, frame: pd.DataFrame, columns: str, family: str, config: str, table: str) -> dict:
    """The reserved `__...__` metadata rows of a fit, keyed by name.

    Used to assert the fit was `converged` and served by the engine the family
    documents, before any of its numbers are compared to anything.
    """
    con.register(table, frame)
    rows = con.sql(
        f"""
        SELECT param, value
        FROM anofox_bayes_fit((SELECT {columns} FROM {table}), '{family}', {config})
        WHERE draw < 0
        """
    ).df()
    return dict(zip(rows["param"], rows["value"]))


def fit_status(con, frame: pd.DataFrame, columns: str, family: str, config: str, table: str) -> float:
    con.register(table, frame)
    (status,) = con.sql(
        f"""
        SELECT value
        FROM anofox_bayes_fit((SELECT {columns} FROM {table}), '{family}', {config})
        WHERE param = '__status__'
        """
    ).fetchone()
    return float(status)


# --------------------------------------------------------------------------
# Datasets (fixed seed, generated once, shared by both sides)
# --------------------------------------------------------------------------


def f7_normal_dataset() -> pd.DataFrame:
    """Two lanes with different levels and different dispersions."""
    rng = np.random.default_rng(20260801)
    return pd.DataFrame(
        {
            "lane": np.repeat(["HAM-ROT", "BRE-ANT"], 60),
            "cost": np.concatenate(
                [rng.normal(10.0, 2.0, 60), rng.normal(14.0, 3.0, 60)]
            ),
        }
    )


def f7_poisson_dataset() -> pd.DataFrame:
    """Damage claims against a known consignment exposure, two carriers."""
    rng = np.random.default_rng(1312)
    expo = np.full(24, 1000.0)
    a = rng.poisson(0.004 * expo)
    b = rng.poisson(0.012 * expo)
    return pd.DataFrame(
        {
            "carrier": np.repeat(["CARRIER-A", "CARRIER-B"], 24),
            "claims": np.concatenate([a, b]).astype("int64"),
            "consignments": np.concatenate([expo, expo]),
        }
    )


def f3_simple_dataset() -> pd.DataFrame:
    """An ungrouped two-predictor regression: the cleanest F3 configuration."""
    rng = np.random.default_rng(770077)
    n = 80
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = 3.0 + 1.5 * x1 - 0.7 * x2 + rng.normal(0.0, 0.8, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


F3_PANEL_GROUPS = 6
F3_PANEL_PERIODS = 20
F3_POOL_SCALE = 5.0


def f3_panel_dataset() -> pd.DataFrame:
    """A difference-in-differences panel: the shape F3 exists for."""
    rng = np.random.default_rng(4242)
    gid = np.repeat(np.arange(F3_PANEL_GROUPS), F3_PANEL_PERIODS)
    t = np.tile(np.arange(F3_PANEL_PERIODS), F3_PANEL_GROUPS).astype(float)
    post = (t >= 10).astype(float)
    treated_post = ((gid < 3) & (t >= 10)).astype(float)
    store_level = rng.normal(0.0, 3.0, F3_PANEL_GROUPS)
    y = (
        50.0
        + 0.5 * t
        + 4.0 * treated_post
        + store_level[gid]
        + rng.normal(0.0, 1.0, len(t))
    )
    return pd.DataFrame(
        {
            "store": [f"S{g:02d}" for g in gid],
            "units": y,
            "month": t,
            "post": post,
            "treated_post": treated_post,
        }
    )


F2_LANES = ["HAM-ROT", "BRE-ANT", "DUS-MIL"]


def f2_dataset(n: int = 180) -> pd.DataFrame:
    """A right-censored shipment book, drawn from the Weibull AFT the family fits.

    `log t = b0 + b1 * distance + sigma * W` with `W` standard extreme-value
    (Gumbel minimum), which is exactly the kernel `f2_censored_aft.rs` uses.
    Each shipment gets its own reporting horizon, because a real book is
    censored by when each shipment was dispatched rather than by one shared
    cut-off, and a single shared horizon would make the censoring pattern a
    function of the duration alone.
    """
    rng = np.random.default_rng(20220)
    b0, b1, sigma = 0.9, 0.25, 0.35
    distance = 1.0 + rng.uniform(0.0, 2.0, n)
    u = rng.uniform(0.0, 1.0, n)
    transit = np.exp(b0 + b1 * distance + sigma * np.log(-np.log(1.0 - u)))
    horizon = 3.2 + rng.uniform(0.0, 2.5, n)
    return pd.DataFrame(
        {
            "distance_100km": distance,
            "days": np.minimum(transit, horizon),
            "delivered": (transit <= horizon).astype("int64"),
        }
    )


def f2_panel_dataset(per_lane: int = 60) -> pd.DataFrame:
    """Three lanes with different levels and different spreads.

    `censored_aft` fits each group as a wholly independent model -- no pooling,
    no shared scale -- so this exercises the row-subsetting and the per-group
    parameter block, which is the part a single-group fixture cannot reach.
    """
    rng = np.random.default_rng(2222)
    frames = []
    for lane, b0, b1, sigma, horizon in [
        ("HAM-ROT", 0.90, 0.25, 0.35, 4.0),
        ("BRE-ANT", 1.30, 0.25, 0.35, 5.5),
        ("DUS-MIL", 1.55, 0.30, 0.45, 7.0),
    ]:
        distance = 1.0 + rng.uniform(0.0, 2.0, per_lane)
        u = rng.uniform(0.0, 1.0, per_lane)
        transit = np.exp(b0 + b1 * distance + sigma * np.log(-np.log(1.0 - u)))
        cut = horizon + rng.uniform(0.0, 2.0, per_lane)
        frames.append(
            pd.DataFrame(
                {
                    "lane": lane,
                    "distance_100km": distance,
                    "days": np.minimum(transit, cut),
                    "delivered": (transit <= cut).astype("int64"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# The BG/NBD population the F5 fixture is drawn from, and the proper prior both
# sides are given. See `pymc_f5_bgnbd` for why the default (flat) prior is not
# the one under test.
F5_TRUTH = {"r": 1.2, "alpha": 10.0, "a": 1.5, "b": 2.0}
F5_PRIOR = {
    "r": (0.0, 0.7),
    "alpha": (2.5, 1.0),
    "a": (0.0, 0.7),
    "b": (0.7, 1.0),
}
F5_PARAMS = ["r", "alpha", "a", "b"]


def f5_prior_config() -> str:
    """`F5_PRIOR` spelled as the extension's `prior` config slot."""
    entries = ", ".join(
        f"'{name}': {{'log_mean': {m}, 'log_sd': {s}}}" for name, (m, s) in F5_PRIOR.items()
    )
    return "{" + entries + "}"


def f5_dataset(n: int = 600) -> pd.DataFrame:
    """A customer base simulated from BG/NBD itself, one row per payer.

    Transactions are Poisson(`lambda`) while alive; after each one the customer
    drops out with probability `p`. `lambda ~ Gamma(r, rate alpha)` and
    `p ~ Beta(a, b)`, so the base really does come from the model the family
    fits. Observation windows differ per account, which is what makes recency
    informative: three weeks of silence means something different on a 20-week
    account than on a 78-week one.
    """
    rng = np.random.default_rng(50505)
    lam = rng.gamma(F5_TRUTH["r"], 1.0 / F5_TRUTH["alpha"], n)
    p = rng.beta(F5_TRUTH["a"], F5_TRUTH["b"], n)
    age = rng.uniform(20.0, 78.0, n)
    frequency = np.zeros(n, dtype=int)
    recency = np.zeros(n)
    for i in range(n):
        clock, count, last = 0.0, 0, 0.0
        while True:
            clock += rng.exponential(1.0 / lam[i])
            if clock > age[i]:
                break
            count += 1
            last = clock
            if rng.random() < p[i]:
                break
        frequency[i], recency[i] = count, last
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "frequency": frequency.astype("int64"),
            # Rounded the way a real extract would be, and the way the
            # test/sql fixture is, so the two cannot silently diverge in
            # precision.
            "recency": np.round(recency, 4),
            "age": np.round(age, 4),
        }
    )


F8_SEGMENTS = ["SEG0", "SEG1", "SEG2", "SEG3", "SEG4", "SEG5"]


def f8_dataset(per_group: int = 30) -> pd.DataFrame:
    """Six segments with genuinely different spreads, drawn from the F8 model.

    `sigma_g = exp(mu_s + tau_s * w_g)` with `tau_s = 0.6`, so the widest
    segment scatters about three times as much as the tightest. A fixture whose
    groups shared a spread would leave `sigma_spread` estimating its prior, and
    the parity test would be checking nothing that `pooled_gaussian` does not
    already do.
    """
    rng = np.random.default_rng(80808)
    tau, mu_s, tau_s = 2.0, np.log(1.5), 0.6
    z = rng.normal(0.0, 1.0, len(F8_SEGMENTS))
    w = rng.normal(0.0, 1.0, len(F8_SEGMENTS))
    group_effect = tau * z
    sigma = np.exp(mu_s + tau_s * w)
    rows = []
    for g, name in enumerate(F8_SEGMENTS):
        x = rng.normal(0.0, 1.0, per_group)
        y = 20.0 + 1.3 * x + group_effect[g] + rng.normal(0.0, sigma[g], per_group)
        rows.extend(zip([name] * per_group, y, x))
    return pd.DataFrame(rows, columns=["segment", "delay_days", "x1"])


F1_TRUTH = {"intercept": float(np.log(6.0)), "tau": 0.7, "phi": 2.5}


# Ten weeks, not the five or six a C-parts catalogue is really full of. That is
# a deliberate retreat and it is documented in TOLERANCES.md: at six weeks this
# fixture draws one to three divergences out of 16_000, `hier_negbin` sets
# `max_divergent = 0`, and the fit is therefore graded `degenerate` -- a verdict
# the *posterior* does not deserve, since all 27 parameters agree with an
# independent implementation. Comparing draws from a fit the extension itself
# tells you not to act on would be measuring the right numbers under the wrong
# banner, so the fixture backs off to where the verdict is `converged`. The
# effect is real, seed-dependent and not monotone in sample size; see
# TOLERANCES.md, "What this found".
F1_THIN_WEEKS = 10


def f1_dataset() -> pd.DataFrame:
    """A spare-parts demand history: six thick parts and six thin ones.

    The thin parts are the point. A C-parts catalogue is mostly items with a
    handful of observations, and it is exactly there that the pooling has to
    work and that a per-group posterior is most sensitive to how `tau` is
    handled. Drawn from the model at `tau = 0.7`, `phi = 2.5`.
    """
    rng = np.random.default_rng(11011)
    parts = [(f"BRG-{i:03d}", 30) for i in range(6)] + [
        (f"SEA-{i:03d}", F1_THIN_WEEKS) for i in range(6)
    ]
    z = rng.normal(0.0, 1.0, len(parts))
    rows = []
    for j, (name, weeks) in enumerate(parts):
        mu = np.exp(F1_TRUTH["intercept"] + F1_TRUTH["tau"] * z[j])
        # numpy parameterises by success probability; phi/(phi+mu) is the one
        # that gives Var = mu + mu^2/phi, the same as the extension's `phi`.
        units = rng.negative_binomial(F1_TRUTH["phi"], F1_TRUTH["phi"] / (F1_TRUTH["phi"] + mu), weeks)
        rows.extend((name, w + 1, int(units[w])) for w in range(weeks))
    return pd.DataFrame(rows, columns=["part", "week", "units"])


F4_SEGMENTS = ["RETAIL", "WHOLESALE", "PUBLIC", "EXPORT", "OEM", "KEY_ACCOUNT"]

F4_TRUTH = {"intercept": float(np.log(30.0)), "tau": 0.35, "shape": 6.0}


def f4_dataset() -> pd.DataFrame:
    """A cleared-invoice ledger: six customer segments, drawn from the model.

    Forty invoices per segment, which is what a mid-sized ledger gives a segment
    in a quarter. The delay is measured from the *invoice* date, so it is
    strictly positive -- the family refuses a due-date clock, and it refuses it
    as a request error rather than a status, so a fixture that violated it would
    not reach a comparison at all.

    Drawn at `shape = 6`: genuinely right-skewed, which is the regime the Gamma
    branch exists for. A near-Gaussian shape would make the two `dist` branches
    agree and the family's premise untestable.
    """
    rng = np.random.default_rng(40404)
    z = rng.normal(0.0, 1.0, len(F4_SEGMENTS))
    rows = []
    for g, name in enumerate(F4_SEGMENTS):
        mu = np.exp(F4_TRUTH["intercept"] + F4_TRUTH["tau"] * z[g])
        # numpy parameterises Gamma by (shape, scale); scale = mu/shape gives
        # mean mu and variance mu^2/shape, which is the extension's convention.
        delay = rng.gamma(F4_TRUTH["shape"], mu / F4_TRUTH["shape"], 40)
        rows.extend((name, float(d)) for d in delay)
    return pd.DataFrame(rows, columns=["segment", "delay_days"])


F6_TRUTH = {
    "intercept": 5.0,
    # psi = log |population elasticity|; exp(-0.11) is about 0.9.
    "psi": float(np.log(0.9)),
    "tau": 0.30,
    "tau_level": 0.60,
    "phi": 20.0,
}

F6_SEGMENTS = [
    "COMMODITY",
    "MIDMARKET",
    "PREMIUM",
    "OEM",
    "SPARE_PARTS",
    "EXPORT",
    "PROJECT",
    "AFTERMARKET",
]


def f6_dataset() -> pd.DataFrame:
    """A billing panel: eight segments, eighteen months, drawn from the model.

    Every segment's log price walks a deterministic ladder about its own mean.
    **Within**-segment price variation is what identifies an elasticity, and a
    fixture whose prices moved only *between* segments would be measuring
    something else entirely -- so the ladder is the same in every segment and
    only the response differs.

    No segment here has a flat price column. The identification refusal is a
    per-group verdict reached before any arithmetic and is exercised in
    `test/sql/f6_price_elasticity.test` and the Rust suite; putting it in the
    parity fixture would mean comparing a segment whose posterior is its prior,
    which measures the prior rather than the likelihood.
    """
    rng = np.random.default_rng(60606)
    z = rng.normal(0.0, 1.0, len(F6_SEGMENTS))
    v = rng.normal(0.0, 1.0, len(F6_SEGMENTS))
    months = 18
    ladder = 0.6 * (np.arange(months) / (months - 1) - 0.5)
    rows = []
    for g, name in enumerate(F6_SEGMENTS):
        b = -np.exp(F6_TRUTH["psi"] + F6_TRUTH["tau"] * z[g])
        level = F6_TRUTH["tau_level"] * v[g]
        mu = np.exp(F6_TRUTH["intercept"] + level + b * ladder)
        units = rng.negative_binomial(
            F6_TRUTH["phi"], F6_TRUTH["phi"] / (F6_TRUTH["phi"] + mu)
        )
        rows.extend(
            (name, float(ladder[t]), int(units[t])) for t in range(months)
        )
    return pd.DataFrame(rows, columns=["segment", "log_price", "units"])


def first_seen(values) -> list[str]:
    """Group keys in first-seen order, which is the order the extension uses.

    Sorting instead would line up for these fixtures and silently mismatch on
    one whose keys are not already alphabetical.
    """
    return list(dict.fromkeys(values))


# --------------------------------------------------------------------------
# PyMC reference models
# --------------------------------------------------------------------------


def pymc_f7_normal(y: np.ndarray, mu0=0.0, kappa0=0.0, alpha0=-0.5, beta0=0.0):
    """The F7 Normal reference, for an arbitrary Normal-Inverse-Gamma prior.

    The extension's prior is NIG:

        sigma^2 ~ InvGamma(alpha0, beta0)
        mu | sigma^2 ~ N(mu0, sigma^2 / kappa0)

    **Defaults (kappa0 = 0, alpha0 = -1/2, beta0 = 0)** are the improper
    reference prior. Taking the kappa0 -> 0 limit of the NIG density and dropping
    the constants leaves

        p(mu, sigma^2)  proportional to  (sigma^2)^(-1/2) * (sigma^2)^(-(alpha0+1))
                        =  (sigma^2)^(-1)                       [at alpha0 = -1/2]

    i.e. Jeffreys. In terms of sigma that is p(sigma) ~ 1/sigma, which is a flat
    prior on log(sigma) *with* PyMC's automatic Jacobian -- hence the explicit
    `log_sigma = pm.Flat(...)` rather than `pm.HalfFlat("sigma")`. `HalfFlat`
    would give p(sigma) ~ 1, a different (and wrong) prior.

    The resulting posterior is the textbook one the extension computes in closed
    form: sigma^2 ~ InvGamma((n-1)/2, SS/2), mu | sigma^2 ~ N(ybar, sigma^2/n).
    """
    y = np.asarray(y, dtype=float)
    proper_mu = kappa0 > 0.0
    proper_sigma = alpha0 > 0.0 and beta0 > 0.0

    with pm.Model() as model:
        if proper_sigma:
            sigma_sq = pm.InverseGamma("sigma_sq", alpha=alpha0, beta=beta0)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma_sq))
        else:
            if not (alpha0 == -0.5 and beta0 == 0.0):
                # A general improper (alpha0, beta0) is expressible, but nothing
                # in this suite needs it and a silently-wrong prior is the exact
                # failure mode this module is written to prevent.
                raise NotImplementedError(
                    f"improper NIG with alpha0={alpha0}, beta0={beta0} is not encoded here"
                )
            log_sigma = pm.Flat("log_sigma", initval=float(np.log(y.std(ddof=1))))
            sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

        if proper_mu:
            mu = pm.Normal("mu", mu=mu0, sigma=sigma / np.sqrt(kappa0))
        else:
            mu = pm.Flat("mu", initval=float(y.mean()))

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(**_sample_kwargs())
    return idata


def pymc_f7_poisson(counts: np.ndarray, exposure: np.ndarray, a0=0.5, b0=0.0):
    """The F7 Poisson reference: y_i ~ Poisson(lambda * exposure_i).

    The extension's prior is Gamma(a0, rate b0), which at the default
    b0 = 0 is improper: p(lambda) ~ lambda^(a0 - 1).

    A flat prior on theta = log(lambda) implies p(lambda) ~ 1/lambda (the
    Jacobian). Multiplying by lambda^a0 -- i.e. adding `a0 * theta` to the log
    density via `pm.Potential` -- gives exactly lambda^(a0-1). The rate term
    `-b0 * lambda` is added when b0 > 0.

    Posterior: Gamma(a0 + sum(y), rate = b0 + sum(exposure)).
    """
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    with pm.Model() as model:
        log_lam = pm.Flat(
            "log_lambda", initval=float(np.log(counts.sum() / exposure.sum()))
        )
        lam = pm.Deterministic("lambda", pt.exp(log_lam))
        pm.Potential("gamma_prior", a0 * log_lam - b0 * lam)
        pm.Poisson("obs", mu=lam * exposure, observed=counts.astype(int))
        idata = pm.sample(**_sample_kwargs())
    return idata


def pymc_f3(
    y: np.ndarray,
    X: np.ndarray,
    x_names: list[str],
    group_index: np.ndarray | None = None,
    group_names: list[str] | None = None,
    pool_scale: float = 1.0,
    a0: float = 0.0,
    s0: float = 0.0,
):
    """The F3 reference: y = intercept + X beta + group_effect[g] + eps.

    Prior, matching the extension's documented defaults and its implemented
    pooling semantics:

    * `intercept` and every `beta` are **flat** (`beta_scale` defaults to
      infinity, and the intercept is never penalised at any `beta_scale`).
    * `sigma^2 ~ InvGamma(a0, s0)`; at the defaults a0 = s0 = 0 this is
      p(sigma^2) ~ (sigma^2)^(-1), i.e. p(sigma) ~ 1/sigma, i.e. flat on
      log(sigma) -- same construction as `pymc_f7_normal`.
    * `group_effect[g] ~ N(0, sigma^2 * pool_scale^2)`.

    That last line is worth reading twice. The module doc for
    `f3_pooled_gaussian.rs` describes the group prior as `N(0, pool_scale^2)`,
    but the implementation forms `A = X'X + P` and draws
    `beta | sigma^2 ~ N(b_n, sigma^2 A^-1)`, which is the Normal-Inverse-Gamma
    convention in which the coefficient prior scale is *proportional to sigma*.
    The reference below encodes what the code does, not what the comment says,
    because the point of a parity suite is to pin the implemented model. The
    docs discrepancy is recorded in README.md.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    grouped = group_index is not None

    coords = {"pred": list(x_names)}
    if grouped:
        coords["group"] = list(group_names)

    with pm.Model(coords=coords) as model:
        log_sigma = pm.Flat("log_sigma", initval=float(np.log(y.std(ddof=1))))
        sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
        if a0 != 0.0 or s0 != 0.0:
            # p(sigma^2) ~ (sigma^2)^-(a0+1) exp(-s0/sigma^2); flat-on-log-sigma
            # already supplies the a0 = s0 = 0 case, so only the excess is added.
            sigma_sq = sigma**2
            pm.Potential("nig_scale_prior", -a0 * pt.log(sigma_sq) - s0 / sigma_sq)

        intercept = pm.Flat("intercept", initval=float(y.mean()))
        mu = intercept
        if len(x_names):
            beta = pm.Flat("beta", dims="pred", initval=np.zeros(len(x_names)))
            mu = mu + pt.dot(X, beta)
        if grouped:
            group_effect = pm.Normal(
                "group_effect", 0.0, sigma * pool_scale, dims="group"
            )
            mu = mu + group_effect[group_index]

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        # The intercept/group-effect block is only ridge-identified, so the
        # posterior geometry is narrow in one direction; a higher target_accept
        # is what keeps ESS above the MIN_ESS gate.
        idata = pm.sample(**_sample_kwargs(target_accept=0.95))
    return idata


# --------------------------------------------------------------------------
# F2 `censored_aft`
# --------------------------------------------------------------------------
#
# Both the PyMC model and the NumPy log posterior below encode the *same* four
# lines of `f2_censored_aft.rs`, which for a log-time AFT with
# `z = (log t - x'beta) / sigma` are
#
#     uncensored (event = 1):  log f_W(z) - log sigma - log t
#     censored   (event = 0):  log S_W(z)
#
# Three details are load-bearing and each is the kind of thing that makes a
# parity suite quietly vacuous:
#
# * The `- log t` term is the log-time-to-time Jacobian. Writing the lognormal
#   case as `Normal(log t | eta, sigma)` drops it, and the posterior for `sigma`
#   moves. It is kept here.
# * `sigma` carries **no prior and no log-Jacobian**. It is estimated by maximum
#   likelihood under a flat prior on `log sigma`, and `log sigma` is the
#   coordinate the extension samples, so no change of variables happens
#   anywhere. `pm.Flat("log_sigma")` plus a `Deterministic` is therefore right
#   and `pm.HalfFlat("sigma")` is wrong -- PyMC would add `+ log sigma`.
# * The coefficient prior defaults to flat, including on the intercept, which is
#   never penalised at any `beta_scale`.


def _f2_extreme_value_terms(log_t, event, design, beta, sigma, log_sigma, ops):
    """The Weibull/exponential kernel: `f(z) = exp(z - e^z)`, `S(z) = exp(-e^z)`."""
    z = (log_t - ops["dot"](design, beta)) / sigma
    log_density = z - ops["exp"](z) - log_sigma - log_t
    log_survival = -ops["exp"](z)
    return event * log_density + (1.0 - event) * log_survival


def pymc_f2_weibull(days, event, x, x_names: list[str]):
    """The F2 Weibull AFT reference, under the default flat priors.

    The posterior this samples is the exact target the extension's Laplace
    engine approximates -- it is *not* the Laplace approximation itself. That is
    deliberate: comparing the two measures how good the approximation is, which
    for a family whose only engine is Laplace is a thing worth measuring. The
    algebra is checked separately and far more tightly by
    `f2_neg_log_posterior` + `laplace_reference`.
    """
    log_t = np.log(np.asarray(days, dtype=float))
    event = np.asarray(event, dtype=float)
    design = np.column_stack([np.ones(len(log_t)), np.asarray(x, dtype=float).reshape(len(log_t), -1)])

    with pm.Model(coords={"pred": list(x_names)}):
        beta = pm.Flat("beta_vec", shape=design.shape[1], initval=np.r_[log_t.mean(), np.zeros(design.shape[1] - 1)])
        log_sigma = pm.Flat("log_sigma", initval=float(np.log(log_t.std(ddof=1))))
        sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
        pm.Deterministic("intercept", beta[0])
        for j, name in enumerate(x_names):
            pm.Deterministic(f"beta[{name}]", beta[j + 1])
        pm.Potential(
            "aft",
            pt.sum(
                _f2_extreme_value_terms(
                    log_t, event, design, beta, sigma, log_sigma,
                    {"dot": pt.dot, "exp": pt.exp},
                )
            ),
        )
        # The intercept and the slope of a duration model with a covariate
        # measured away from zero are almost perfectly anti-correlated
        # (corr = -0.96 on this fixture), so the geometry is a narrow ridge and
        # the default acceptance target leaves ESS lower than the reference
        # needs to referee anything.
        idata = pm.sample(**_sample_kwargs(target_accept=0.9))
    return idata


def f2_neg_log_posterior(days, event, x):
    """The same target in plain NumPy, for the independent Laplace reference.

    Written out separately rather than reusing the PyMC graph on purpose: the
    point of `laplace_reference` is that it shares no code with either the
    extension or PyMC. Coordinates are `(intercept, beta..., log sigma)`, which
    is the extension's unconstrained ordering.
    """
    log_t = np.log(np.asarray(days, dtype=float))
    event = np.asarray(event, dtype=float)
    design = np.column_stack([np.ones(len(log_t)), np.asarray(x, dtype=float).reshape(len(log_t), -1)])

    def neg_log_posterior(theta: np.ndarray) -> float:
        beta, log_sigma = theta[:-1], theta[-1]
        value = np.sum(
            _f2_extreme_value_terms(
                log_t, event, design, beta, np.exp(log_sigma), log_sigma,
                {"dot": np.dot, "exp": np.exp},
            )
        )
        return 1e300 if not np.isfinite(value) else -float(value)

    return neg_log_posterior, design.shape[1]


def f2_start(days) -> np.ndarray:
    log_t = np.log(np.asarray(days, dtype=float))
    return np.array([log_t.mean(), 0.0, np.log(log_t.std(ddof=1))])


# --------------------------------------------------------------------------
# F5 `payer_alive`
# --------------------------------------------------------------------------
#
# BG/NBD's likelihood, integrated over both latent processes, from
# `f5_btyd.rs::log_likelihood`:
#
#   ln L = lnG(r+x) - lnG(r) + r ln(alpha) + lnG(b+x) + lnG(a+b) - lnG(b)
#          - lnG(a+b+x) + ln[ (alpha+T)^-(r+x) + 1{x>0} a/(b+x-1) (alpha+t_x)^-(r+x) ]
#
# The bracket's two terms are the two histories consistent with the data: still
# alive at T, or gone some time after t_x. At x = 0 the second term is absent
# *and* the whole Beta block cancels, so `a` and `b` genuinely drop out of a
# never-repeated customer's contribution -- which is why such a customer scores
# P(alive) = 1.0 rather than something the prior invented.
#
# The extension declares its priors on the log scale and samples there, so the
# default (flat on log) is `p(theta) ~ 1/theta` and **no log-Jacobian appears
# anywhere**. `pm.Normal` on `log_r` etc. is therefore the exact match for a
# configured `{'log_mean': m, 'log_sd': s}` prior.
#
# The tests use the *proper* prior rather than the default. Under the default
# the target is improper except for the hard `|log theta| <= 30` box the family
# imposes, and a NUTS reference on a target that is only proper because of a box
# constraint is not a reference -- it is a random walk with a fence. The proper
# prior is also the configuration F5's SBC suite certifies, so the two harnesses
# are looking at the same model.


def _f5_bracket(r, alpha, a, b, x, t_x, T, ops):
    """`ln[ alive + 1{x>0} dead ]`, by log-sum-exp rather than by exponentiating."""
    rx = r + x
    alive = -rx * ops["log"](alpha + T)
    repeat = x > 0
    # `b + x - 1` is >= b > 0 wherever the dead branch is used; the clamp keeps
    # the *unused* entries from evaluating log of a non-positive number, which
    # would poison the gradient with a NaN even though the branch is masked.
    safe = np.where(repeat, x - 1.0, 0.0)
    dead = ops["log"](a) - ops["log"](b + safe) - rx * ops["log"](alpha + t_x)
    return ops["where"](repeat, ops["logaddexp"](alive, dead), alive)


def _f5_log_likelihood(r, alpha, a, b, x, t_x, T, ops):
    rx = r + x
    lg = ops["lgamma"]
    return ops["sum"](
        lg(rx)
        - lg(r)
        + r * ops["log"](alpha)
        + lg(b + x)
        + lg(a + b)
        - lg(b)
        - lg(a + b + x)
        + _f5_bracket(r, alpha, a, b, x, t_x, T, ops)
    )


_F5_NUMPY_OPS = {
    "log": np.log,
    "lgamma": gammaln,
    "sum": np.sum,
    "where": np.where,
    "logaddexp": np.logaddexp,
}
_F5_PYTENSOR_OPS = {
    "log": pt.log,
    "lgamma": pt.gammaln,
    "sum": pt.sum,
    "where": pt.switch,
    "logaddexp": pt.logaddexp,
}


def pymc_f5_bgnbd(frame: pd.DataFrame, prior=None):
    """The F5 BG/NBD reference: a custom logp through `pm.Potential`.

    There is no PyMC distribution for BG/NBD, and there should not be one here
    either -- the whole value of the comparison is that the likelihood is
    written out a second time from the paper rather than shared with the
    implementation under test.
    """
    prior = prior or F5_PRIOR
    x = frame["frequency"].to_numpy(dtype=float)
    t_x = frame["recency"].to_numpy(dtype=float)
    T = frame["age"].to_numpy(dtype=float)

    with pm.Model():
        logs = {
            name: pm.Normal(f"log_{name}", mu=prior[name][0], sigma=prior[name][1])
            for name in F5_PARAMS
        }
        natural = {name: pm.Deterministic(name, pt.exp(v)) for name, v in logs.items()}
        pm.Potential(
            "bgnbd",
            _f5_log_likelihood(
                natural["r"], natural["alpha"], natural["a"], natural["b"],
                x, t_x, T, _F5_PYTENSOR_OPS,
            ),
        )
        # `a` and `b` are only weakly separately identified -- the data says
        # where the Beta sits, and much less about how wide it is -- so the
        # posterior has a long curved ridge that the default step size explores
        # badly.
        idata = pm.sample(**_sample_kwargs(target_accept=0.9))
    return idata


def f5_neg_log_posterior(frame: pd.DataFrame, prior=None):
    """The same target in NumPy, on `(log r, log alpha, log a, log b)`."""
    prior = prior or F5_PRIOR
    x = frame["frequency"].to_numpy(dtype=float)
    t_x = frame["recency"].to_numpy(dtype=float)
    T = frame["age"].to_numpy(dtype=float)
    means = np.array([prior[name][0] for name in F5_PARAMS])
    sds = np.array([prior[name][1] for name in F5_PARAMS])

    def neg_log_posterior(theta: np.ndarray) -> float:
        # The family's own support: |log theta| <= 30 is a hard rejection, so
        # the reference has to have it too or the two targets differ.
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 30.0):
            return 1e300
        r, alpha, a, b = np.exp(theta)
        value = _f5_log_likelihood(r, alpha, a, b, x, t_x, T, _F5_NUMPY_OPS)
        value -= 0.5 * np.sum(((theta - means) / sds) ** 2)
        return 1e300 if not np.isfinite(value) else -float(value)

    return neg_log_posterior


def f5_start(frame: pd.DataFrame) -> np.ndarray:
    """The family's own starting point, so a second local optimum is not found.

    BG/NBD has a much worse local optimum reachable from `a = b = 1`; the
    extension guards against it with a trust region. Starting the reference
    where the extension starts keeps the comparison about the target rather than
    about optimiser luck -- and the stationarity of the located mode is asserted
    separately, so a wrong basin would still show up.
    """
    x = frame["frequency"].to_numpy(dtype=float)
    T = frame["age"].to_numpy(dtype=float)
    return np.array([0.0, float(np.log(T.mean() / max(x.mean(), 0.5))), 0.0, 0.0])


# --------------------------------------------------------------------------
# F8 `varying_variance_gaussian`
# --------------------------------------------------------------------------


def f8_response_sd(y) -> float:
    """The default `pool_scale` hyperprior scale: the response's own sd, ddof=1.

    This is the one concrete prior default in the extension, and it is concrete
    only in the sense that it is read off the data -- double the observations
    and it doubles with them, so it makes no claim about units. It has to be
    recomputed here from the same rows rather than hard-coded, or the reference
    encodes a different prior from the fit.
    """
    return float(np.std(np.asarray(y, dtype=float), ddof=1))


def pymc_f8(frame: pd.DataFrame, y_col: str, group_col: str, x_names: list[str]):
    """The F8 reference: per-group level and per-group spread, both non-centred.

        y_i     ~ N(x_i'beta + eta_g,  sigma_g^2)
        eta_g   = tau * z_g,                    z_g ~ N(0, 1)
        sigma_g = exp(mu_s + tau_s * w_g),      w_g ~ N(0, 1)

    Two mappings here are easy to get wrong and are worth reading carefully.

    * `tau` (`pool_scale`) and `tau_s` (`sigma_spread`) carry half-Normal priors
      declared **on their natural scale** while the sampler works on the log,
      so the extension's density carries `+ log tau` and `+ log tau_s`
      Jacobians. `pm.HalfNormal` is log-transformed by PyMC and contributes
      exactly those, so the two agree.
    * `mu_s` carries a **flat prior and no Jacobian**, because `mu_s` *is* the
      sampled coordinate -- `sigma_pop = exp(mu_s)` is only a reported
      transform. `pm.Flat("mu_s")` is therefore right and
      `pm.HalfFlat("sigma_pop")` is wrong; the latter would add `+ log sigma_pop`
      and tilt the whole spread block.

    The `1/tau` prior a scale-free default would suggest is not available: for a
    hierarchical variance it leaves the posterior improper, because the
    likelihood is bounded as `tau -> 0` and `1/tau` is not integrable there.
    """
    y = frame[y_col].to_numpy(dtype=float)
    groups = first_seen(frame[group_col])
    group_index = np.array([groups.index(g) for g in frame[group_col]])
    design = frame[x_names].to_numpy(dtype=float) if x_names else None

    with pm.Model(coords={"group": groups, "pred": list(x_names)}):
        intercept = pm.Flat("intercept", initval=float(y.mean()))
        mu = intercept
        if x_names:
            beta = pm.Flat("beta_vec", shape=len(x_names), initval=np.zeros(len(x_names)))
            for j, name in enumerate(x_names):
                pm.Deterministic(f"beta[{name}]", beta[j])
            mu = mu + pt.dot(design, beta)

        pool_scale = pm.HalfNormal("pool_scale", sigma=f8_response_sd(y))
        sigma_spread = pm.HalfNormal("sigma_spread", sigma=1.0)
        mu_s = pm.Flat("mu_s", initval=float(np.log(y.std(ddof=1))))
        pm.Deterministic("sigma_pop", pt.exp(mu_s))

        z = pm.Normal("z", 0.0, 1.0, dims="group")
        w = pm.Normal("w", 0.0, 1.0, dims="group")
        group_effect = pm.Deterministic("group_effect", pool_scale * z, dims="group")
        sigma = pm.Deterministic("sigma", pt.exp(mu_s + sigma_spread * w), dims="group")

        pm.Normal("obs", mu=mu + group_effect[group_index], sigma=sigma[group_index], observed=y)
        # 0.95 is what the family itself asks nuts-rs for, declared in the
        # Rust and not reachable from SQL. Matching it keeps the reference from
        # being the noisier side for a reason that has nothing to do with the
        # model.
        idata = pm.sample(**_sample_kwargs(target_accept=0.95))
    return idata


# --------------------------------------------------------------------------
# F4 `payment_delay`
# --------------------------------------------------------------------------


def pymc_f4(frame: pd.DataFrame, y_col: str, group_col: str):
    """The F4 reference: non-centred hierarchical Gamma duration model.

        delay_ij ~ Gamma(shape, shape / mu_ij),   E = mu, Var = mu^2/shape
        log mu_ij = intercept + tau * z_j
        z_j ~ N(0, 1)

    `pm.Gamma(alpha=, beta=)` is a rate parameterisation, so `beta = shape/mu`
    is what makes `mu` the mean -- the same convention `f4_payment_delay.rs`
    writes as `-k eta - k y e^{-eta}` plus `k log k - lnGamma(k)`.

    **The prior on `tau` is the part nothing else can see.** The extension
    declares a half-Normal on `tau` *itself*, not on `log tau`, and samples the
    log -- so the density carries `+ log tau`. A half-Normal declared on the log
    coordinate instead would be a different model, and one whose posterior is
    improper at zero in the way `p(tau) ~ 1/tau` is. Writing it as `pm.Flat` plus
    an explicit `Potential` rather than as `pm.HalfNormal` is deliberate: PyMC
    would supply its own transform Jacobian and there would then be two, one of
    them uninvited.

    The dispersion prior is flat on `log shape`, which is the extension's default
    and *is* the sampled coordinate, so there is no Jacobian to add. That
    asymmetry with `tau` is the thing
    `test_the_tau_jacobian_is_present_and_the_dispersion_one_is_not` asserts.
    """
    y = frame[y_col].to_numpy(dtype=float)
    groups = first_seen(frame[group_col])
    group_index = np.array([groups.index(g) for g in frame[group_col]])
    # The family's own default: half-Normal(1) on tau, in log units.
    tau_scale = 1.0

    with pm.Model(coords={"group": groups}):
        intercept = pm.Flat("intercept", initval=float(np.log(y.mean())))
        log_tau = pm.Flat("log_tau", initval=-1.0)
        tau = pm.Deterministic("tau", pt.exp(log_tau))
        pm.Potential(
            "tau_half_normal_on_the_natural_scale_plus_jacobian",
            -0.5 * tau**2 / tau_scale**2 + log_tau,
        )
        log_shape = pm.Flat("log_shape", initval=1.5)
        shape = pm.Deterministic("shape", pt.exp(log_shape))

        z = pm.Normal("z", 0.0, 1.0, dims="group")
        u = pm.Deterministic("u", tau * z, dims="group")
        # `mu` is the segment's own mean delay with covariates at zero, which is
        # exactly what the extension reports under that name.
        pm.Deterministic("mu", pt.exp(intercept + u), dims="group")

        mean = pt.exp(intercept + u[group_index])
        pm.Gamma("obs", alpha=shape, beta=shape / mean, observed=y)
        # The family declares target_accept = 0.95, so the reference matches it.
        idata = pm.sample(**_sample_kwargs(target_accept=0.95))
    return idata


# --------------------------------------------------------------------------
# F6 `hier_elasticity`
# --------------------------------------------------------------------------


def pymc_f6(frame: pd.DataFrame, y_col: str, price_col: str, group_col: str):
    """The F6 reference: sign-constrained hierarchical elasticity.

        units_ij ~ NegBin(mu_ij, phi),   Var = mu + mu^2/phi
        log mu_ij = intercept + eta_j + b_j * logprice_ij
        eta_j = tau_level * v_j,        v_j ~ N(0, 1)
        b_j   = -exp(psi + tau * z_j),  z_j ~ N(0, 1)

    **The `-exp` is the whole family and is where a Jacobian error would live.**
    It is written here as an explicit transform of a `pm.Normal` on `psi` rather
    than as a constrained variable, because the extension samples `psi` directly
    and declares its prior there: a lognormal prior on the elasticity's magnitude
    *is* a normal prior on the log of it, and there is no Jacobian. A reference
    that reached for a `pm.Bound` or a negative-support distribution would
    silently add one, and the only symptom would be this comparison drifting.

    The two pooling scales are half-Normals declared on the natural scale with
    their `+ log tau` Jacobians written out, same as F4's -- and this family has
    two of them, which is two chances to lose one.
    """
    y = frame[y_col].to_numpy(dtype=float)
    price = frame[price_col].to_numpy(dtype=float)
    groups = first_seen(frame[group_col])
    group_index = np.array([groups.index(g) for g in frame[group_col]])
    # The family's own defaults.
    elasticity_prior = (0.0, 1.0)
    tau_scale, tau_level_scale = 0.5, 2.0

    with pm.Model(coords={"group": groups}):
        intercept = pm.Flat("intercept", initval=float(np.log(y.mean() + 0.5)))
        psi = pm.Normal("psi", elasticity_prior[0], elasticity_prior[1], initval=0.0)
        pm.Deterministic("elasticity", -pt.exp(psi))

        log_tau = pm.Flat("log_tau", initval=float(np.log(0.3)))
        tau = pm.Deterministic("tau", pt.exp(log_tau))
        pm.Potential(
            "tau_half_normal_plus_jacobian",
            -0.5 * tau**2 / tau_scale**2 + log_tau,
        )
        log_tau_level = pm.Flat("log_tau_level", initval=float(np.log(0.6)))
        tau_level = pm.Deterministic("tau_level", pt.exp(log_tau_level))
        pm.Potential(
            "tau_level_half_normal_plus_jacobian",
            -0.5 * tau_level**2 / tau_level_scale**2 + log_tau_level,
        )
        log_phi = pm.Flat("log_phi", initval=float(np.log(20.0)))
        phi = pm.Deterministic("phi", pt.exp(log_phi))

        z = pm.Normal("z", 0.0, 1.0, dims="group")
        v = pm.Normal("v", 0.0, 1.0, dims="group")
        b = pm.Deterministic("group_elasticity", -pt.exp(psi + tau * z), dims="group")
        eta = pm.Deterministic("group_effect", tau_level * v, dims="group")

        mean = pt.exp(intercept + eta[group_index] + b[group_index] * price)
        pm.NegativeBinomial("obs", mu=mean, alpha=phi, observed=y)
        idata = pm.sample(**_sample_kwargs(target_accept=0.95))
    return idata


# --------------------------------------------------------------------------
# F1 `hier_negbin`
# --------------------------------------------------------------------------


def pymc_f1(frame: pd.DataFrame, y_col: str, group_col: str):
    """The F1 reference: non-centred hierarchical negative binomial.

        y_ij ~ NegBin(mu_ij, phi),   Var = mu + mu^2/phi
        log mu_ij = intercept + tau * z_j
        z_j ~ N(0, 1)

    `pm.NegativeBinomial(mu=, alpha=)` is the same parameterisation as the
    extension's `phi`: expanding `f1_hier_negbin.rs`'s

        lnG(y+phi) - lnG(phi) - lnG(y+1) + phi ln phi - (y+phi) ln(phi+mu) + y eta

    gives PyMC's log-density term for term, normalising constants included.

    **The two priors are the subtle part**, and neither is expressible as a PyMC
    distribution without saying what it is on the sampled coordinate:

    * `tau` is uniform on **`tau` itself** -- not the scale-free `1/tau`, which
      gives an improper posterior for a variance component; uniform is proper
      for three or more groups (Gelman 2006). The sampler works on `log tau`, so
      the density carries `+ log tau`.
    * `phi` is uniform on the **overdispersion `1/phi`**, which is flat exactly
      at the Poisson limit, so the default cannot push a fit toward finding
      burstiness that is not there. On `log phi` that is `- log phi`.

    Those two terms are the whole of the hyperprior, and neither is visible to
    an engine-agreement test -- both engines would explore the same wrong
    surface -- which is why they are written out explicitly here rather than
    inherited from a PyMC distribution that happens to look similar.
    """
    y = frame[y_col].to_numpy(dtype=float)
    groups = first_seen(frame[group_col])
    group_index = np.array([groups.index(g) for g in frame[group_col]])

    with pm.Model(coords={"group": groups}):
        intercept = pm.Flat("intercept", initval=float(np.log(y.mean() + 0.5)))
        # pm.Flat on the log coordinate plus an explicit Potential, rather than a
        # positive-constrained variable: PyMC would supply its own transform
        # Jacobian and there would then be two, one of them uninvited.
        log_tau = pm.Flat("log_tau", initval=-0.7)
        tau = pm.Deterministic("tau", pt.exp(log_tau))
        pm.Potential("tau_uniform_on_the_natural_scale", pt.log(tau))
        log_phi = pm.Flat("log_phi", initval=0.0)
        phi = pm.Deterministic("phi", pt.exp(log_phi))
        pm.Potential("phi_uniform_on_the_overdispersion", -pt.log(phi))

        z = pm.Normal("z", 0.0, 1.0, dims="group")
        u = pm.Deterministic("u", tau * z, dims="group")
        # `rate` is the group's own expected count per unit exposure, so it is
        # the intercept and the group offset only -- covariates and exposure are
        # deliberately not in it.
        pm.Deterministic("rate", pt.exp(intercept + u), dims="group")

        pm.NegativeBinomial("obs", mu=pt.exp(intercept + u[group_index]), alpha=phi, observed=y)
        # The family does not override nuts-rs's 0.8, so the reference does not
        # override PyMC's either.
        idata = pm.sample(**_sample_kwargs(target_accept=0.8))
    return idata
