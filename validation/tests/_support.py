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

# --------------------------------------------------------------------------
# Sampling budgets
# --------------------------------------------------------------------------

# The extension samples i.i.d. from a closed form, so its Monte Carlo error is
# sd/sqrt(N). 40_000 draws puts MCSE(mean) at 0.005 sd, an order of magnitude
# under every tolerance below, which keeps the extension side from being what
# the test is measuring.
EXTENSION_DRAWS = 40_000
EXTENSION_SEED = 20260801

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
F7_TOL = Tolerance(mean=0.05, sd=0.05, quantile=0.09, label="F7 conjugate_anomaly")

# F3's grouped posterior has a ridge-identified intercept/group-effect block, so
# NUTS autocorrelation is materially higher and the reference MCSE roughly
# doubles. The sd tolerance is deliberately ~3x the systematic ~2% interval
# deficit documented in test_parity_f3.py -- that deficit is measured directly
# rather than caught here, because a tolerance tight enough to catch it would
# flake on the panel's own Monte Carlo error.
F3_TOL = Tolerance(mean=0.07, sd=0.07, quantile=0.12, label="F3 pooled_gaussian")


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
