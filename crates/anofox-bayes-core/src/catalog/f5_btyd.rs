//! F5 — payer-alive / BTYD, as the **BG/NBD** model.
//!
//! The inference layer under the collections agent. Its question is not "how much does
//! this customer owe" but "is this customer still transacting at all", because a
//! dunning letter sent to a customer who has silently churned is a cost with no
//! recovery attached, and a dunning letter *not* sent to one who is merely late is a
//! receivable written off for nothing.
//!
//! Each customer arrives as three sufficient statistics over one observation window:
//!
//! | | |
//! |---|---|
//! | `x` | repeat transactions after the first one |
//! | `t_x` | recency — when the last of those happened, measured from the first |
//! | `T` | how long the customer has been observed, from the first transaction |
//!
//! Those three are all the likelihood reads, which is what makes this family cheap:
//! ten years of transaction history collapses to one row per customer.
//!
//! ## The model
//!
//! ```text
//!   while alive, transactions ~ Poisson(lambda)      lambda ~ Gamma(r, rate alpha)
//!   after each transaction, drop out with prob. p    p      ~ Beta(a, b)
//! ```
//!
//! `lambda` and `p` are *per customer* and are integrated out analytically, so the
//! four parameters this family reports — `r`, `alpha`, `a`, `b` — are population
//! level. That is the whole design: a fit describes the customer base, and any
//! individual customer is scored against it afterwards, in SQL, without re-fitting.
//!
//! Integrating out `lambda` and `p` (Fader, Hardie & Lee 2005) gives, per customer:
//!
//! ```text
//!   ln L = lnG(r+x) - lnG(r) + r ln(alpha)
//!        + lnG(b+x) + lnG(a+b) - lnG(b) - lnG(a+b+x)
//!        + ln[ (alpha+T)^-(r+x)  +  1{x>0} * a/(b+x-1) * (alpha+t_x)^-(r+x) ]
//! ```
//!
//! The two terms of the bracket are the two ways the data could have arisen: the
//! customer is still alive at `T` (first term), or dropped out at some point after
//! `t_x` (second). Their relative weight *is* the answer to the agent's question.
//!
//! ## P(alive), and why it has to be a SQL expression
//!
//! ```text
//!   P(alive | x, t_x, T) = 1 / (1 + 1{x>0} * a/(b+x-1) * ((alpha+T)/(alpha+t_x))^(r+x))
//! ```
//!
//! Elementary arithmetic on four numbers and three per-customer statistics — no
//! special functions, no integral. That is not an aesthetic point. Agent 05 scores
//! *today's* dunning list from *yesterday's* draws: one join between a draws table and
//! a customer table, no fit, sub-second, and every customer gets a full posterior for
//! `P(alive)` rather than a point estimate. [`P_ALIVE_SQL`] is that expression, kept in
//! one place so the Rust and the SQL cannot drift apart, and
//! `p_alive_matches_the_documented_sql_expression` pins that they have not.
//!
//! **This is why the family is BG/NBD and not Pareto/NBD** (roadmap §3.2, §5).
//! Pareto/NBD's likelihood needs the Gaussian hypergeometric `2F1` — a special
//! function to own and test — and, decisively, its `P(alive)` is not expressible in
//! SQL. Choosing it would trade a modest gain in realism for the operating premise of
//! the agent that asked for the family.
//!
//! ## Parameterisation and priors
//!
//! All four parameters are positive, so the model is written on the **log scale**:
//! `theta = (ln r, ln alpha, ln a, ln b)`. Everything a gradient engine sees lives
//! there, which is what keeps a Gaussian approximation honest — a Gaussian fitted
//! directly to `a` would put mass below zero exactly where the interesting data is.
//!
//! Priors are declared **on that log scale**, as independent Normals, so the default
//! `log_sd = infinity` is a flat prior on `ln r` — equivalently `p(r) ∝ 1/r`, the
//! scale-free reference prior for a positive parameter. Under the defaults the mode
//! of the log posterior is therefore exactly the **maximum likelihood estimate**, and
//! no Jacobian term appears: declaring the prior on the log scale is what makes the
//! transform free rather than something to remember to add.
//!
//! That default is improper and so cannot be sampled from, which is why the SBC suite
//! runs under explicit finite `log_sd`s — the same argument, and the same remedy, as
//! for the other two families.
//!
//! ## Engine: Laplace, and the boundary that would break it
//!
//! Four population parameters informed by thousands of customers is the regime a
//! Gaussian approximation at the mode is best at, so this family is served by
//! [`crate::engines::LaplaceEngine`] and the roadmap's deferred question — "is NUTS
//! genuinely required for F5?" — is answered by the SBC suite in `sbc.rs`, not by
//! assertion.
//!
//! The known hazard is that BG/NBD's likelihood has flat ridges and boundary solutions
//! on some datasets. The canonical one: if no repeat buyer's last transaction is
//! followed by silence — every `t_x` equal to its `T` — then the likelihood is
//! maximised only in the limit `p -> 0`, which is `a -> 0` (or `b -> infinity`), and
//! the search runs away. **A curvature computed at a runaway point is not a
//! posterior**, and reporting an interval from it would be the exact failure this
//! extension exists to prevent. So the mode is found *here*, at compile time, and
//! checked before any engine sees it; a mode that failed the check reports
//! `degenerate` and draws `NaN` — a refusal, not an interval.

use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::linalg::cholesky;
use crate::types::{EngineKind, FamilyCode};

use super::{CompiledModel, LogPosterior, ModelFamily, Readiness};
use statrs::function::gamma::{digamma, ln_gamma};

/// The closed-form `P(alive)` of this family, as a DuckDB expression.
///
/// The identifiers are placeholders for whatever the caller's columns are called:
/// `r`, `alpha`, `a`, `b` come from the draws table, and `frequency`, `recency`,
/// `age` are the same three per-customer statistics the fit consumed.
///
/// Held here rather than only in the documentation because it is a second
/// implementation of [`p_alive`], and two implementations of one formula drift unless
/// something is watching. `docs/GUIDE.md`, `docs/API_REFERENCE.md` and
/// `test/sql/f5_payer_alive.test` all contain this exact string, and a test asserts it.
pub const P_ALIVE_SQL: &str = "1.0 / (1.0 + CASE WHEN frequency = 0 THEN 0.0 ELSE (a / (b + frequency - 1)) * pow((alpha + age) / (alpha + recency), r + frequency) END)";

/// Posterior probability that a customer is still active, given the population
/// parameters and that customer's three statistics.
///
/// Closed form, and deliberately written as the same arithmetic as [`P_ALIVE_SQL`]
/// rather than as `exp(alive_term - log_sum_exp)`; the likelihood computes it the
/// second way, and `p_alive_agrees_with_the_likelihoods_own_decomposition` checks the
/// two against each other.
pub fn p_alive(r: f64, alpha: f64, a: f64, b: f64, x: f64, t_x: f64, t: f64) -> f64 {
    if x <= 0.0 {
        // A customer who has never repeated offers no evidence of dropping out: the
        // model's dropout opportunity arrives only *after* a transaction.
        return 1.0;
    }
    let odds_dead = (a / (b + x - 1.0)) * ((alpha + t) / (alpha + t_x)).powf(r + x);
    1.0 / (1.0 + odds_dead)
}

/// The family singleton registered in the catalog.
#[derive(Debug)]
pub struct PayerAlive;

const SLOTS: &[&str] = &[
    "frequency",
    "recency",
    "age",
    "prior",
    "min_customers",
    "draws",
    "chains",
    "max_draw_megabytes",
    "seed",
    "engine",
];

/// Coordinate order of the unconstrained parameter vector, and of `param_names`.
const PARAMS: [&str; 4] = ["r", "alpha", "a", "b"];

/// Log-scale coordinates outside `+/- LOG_BOX` are refused rather than explored.
///
/// `e^30` is about `1e13`. A transaction rate, a time scale or a Beta shape outside
/// `[1e-13, 1e13]` is not a parameter anyone holds; it is a search that has run away.
/// Bounding the domain is what keeps `lnG` of an astronomical argument — and the NaNs
/// that follow — out of the arithmetic entirely.
const LOG_BOX: f64 = 30.0;

/// How close to [`LOG_BOX`] a mode may sit before it is called a boundary solution.
const LOG_BOUNDARY: f64 = 25.0;

/// Per-customer share of the gradient norm at which a point is still a mode.
///
/// Looser than the Newton search's own `1e-8` tolerance on purpose: the question here
/// is not "did the search polish the last digit" but "is this a stationary point at
/// all". See `IMPROVEMENT_TOLERANCE` in the Laplace engine for why the tighter test
/// cannot be the one that decides, and
/// [`CompiledPayerAlive::stationarity_tolerance`] for why this one scales with the
/// size of the base.
const MODE_GRAD_TOLERANCE_PER_CUSTOMER: f64 = 1e-5;

/// The widest marginal posterior, on the log scale, that still counts as an estimate.
///
/// This is the check that catches the *flat ridge*, which the other two do not. The
/// canonical BG/NBD ridge sends `a` and `b` to infinity together with `a/(a+b)` fixed:
/// as the Beta concentrates, the dropout probability becomes a point mass, and every
/// point along the ridge fits equally well. The search stops somewhere on it at a
/// perfectly respectable gradient, well inside the box, with a curvature that still
/// factors — so nothing so far has objected — and the resulting marginal for `a` spans
/// many orders of magnitude.
///
/// A posterior standard deviation of 3 on the log scale is a 95 % interval a factor of
/// `e^11.8` — about 10^5 — wide. No decision reads a number known that poorly, and no
/// well-identified fit comes close: a healthy `payer_alive` fit on a few thousand
/// customers lands nearer 0.05. Refusing between the two is not a close call.
const MAX_LOG_SD: f64 = 3.0;

impl ModelFamily for PayerAlive {
    fn id(&self) -> &'static str {
        "payer_alive"
    }

    fn code(&self) -> FamilyCode {
        FamilyCode::PayerAlive
    }

    fn description(&self) -> &'static str {
        "BG/NBD buy-till-you-die model over per-customer (frequency, recency, age) \
         statistics, whose closed-form P(alive) rescores a customer base in SQL \
         without re-fitting."
    }

    fn default_engine(&self) -> EngineKind {
        // No closed-form posterior exists, and four parameters informed by thousands
        // of customers is where a Gaussian approximation at the mode is at its best.
        // Certified, not assumed: see the SBC suite in `sbc.rs`.
        EngineKind::Laplace
    }

    fn config_slots(&self) -> &'static [&'static str] {
        SLOTS
    }

    fn compile<'a>(
        &self,
        cfg: &Config,
        data: &'a DataView<'a>,
    ) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        Ok(Box::new(build(cfg, data)?))
    }
}

/// The whole of `compile`, returning the concrete type.
///
/// Split out so that tests can reach the model's *true* log density even when the
/// compiled model has decided to refuse — see [`Fitted::Boundary`], which swaps the
/// surface for a placeholder. A finite-difference check run through the trait object
/// would then be measuring the placeholder, and would pass no matter how wrong the
/// real gradient was.
fn build(cfg: &Config, data: &DataView) -> BayesResult<CompiledPayerAlive> {
    cfg.reject_unknown(SLOTS)?;

    let frequency = cfg.require_str("frequency")?.to_string();
    let recency = cfg.require_str("recency")?.to_string();
    let age = cfg.require_str("age")?.to_string();
    let min_customers = cfg.usize_in("min_customers", 50, 1, 1_000_000_000)?;
    let prior = Prior::parse(&cfg.nested("prior")?)?;

    let numeric_cols = [frequency.as_str(), recency.as_str(), age.as_str()];
    let rows = data.usable_rows(&numeric_cols, &[])?;
    let fingerprint = data.fingerprint(&numeric_cols, &[], &rows)?;

    let x_col = data.numeric(&frequency)?;
    let tx_col = data.numeric(&recency)?;
    let t_col = data.numeric(&age)?;

    let mut obs = Vec::with_capacity(rows.len());
    for &i in &rows {
        let (x, t_x, t) = (x_col.values[i], tx_col.values[i], t_col.values[i]);
        // Validated here rather than defended against inside the likelihood: a
        // negative count or a recency past the end of the observation window is a
        // data-preparation mistake, and a likelihood that silently absorbed one
        // would return a number nobody could tell was wrong.
        if x < 0.0 || x.fract() != 0.0 {
            return Err(BayesError::config(
                "frequency",
                format!(
                    "must be a non-negative whole count of repeat transactions; row {i} is {x}"
                ),
            ));
        }
        if t <= 0.0 {
            return Err(BayesError::config(
                    "age",
                    format!("must be > 0: a customer observed for no time carries no information; row {i} is {t}"),
                ));
        }
        if t_x < 0.0 || t_x > t {
            return Err(BayesError::config(
                "recency",
                format!("must lie in [0, age]; row {i} has recency {t_x} against age {t}"),
            ));
        }
        if x > 0.0 && t_x <= 0.0 {
            return Err(BayesError::config(
                "recency",
                format!("a customer with {x} repeat transactions cannot have recency 0; row {i}"),
            ));
        }
        obs.push(Customer { x, t_x, t });
    }

    let n = obs.len();
    if n <= PARAMS.len() {
        return Err(BayesError::InsufficientData {
            rows: n,
            params: PARAMS.len(),
        });
    }

    let params: Vec<ParamName> = PARAMS
        .iter()
        .map(|p| ParamName::global(*p))
        .collect::<BayesResult<_>>()?;

    // The one refusal visible from the sufficient statistics alone. With every
    // frequency at zero the Beta terms of the likelihood cancel exactly, so `a`
    // and `b` do not appear in it at all -- there is nothing for a search to find,
    // and saying that costs one pass over the data.
    let structural = (obs.iter().all(|c| c.x <= 0.0)).then(|| {
        format!(
            "none of the {n} customers has a repeat transaction, so the dropout \
             process is not identified: the likelihood does not depend on `a` or `b` \
             at all when every frequency is zero"
        )
    });

    let start = starting_point(&model_stats(&obs));
    let mut model = CompiledPayerAlive {
        params,
        obs,
        prior,
        // Provisional. `LogPosterior::initial` reads this field, so the search
        // below starts here; its verdict then replaces it.
        fitted: Fitted::Mode(start),
        structural,
        n_obs: n,
        fingerprint,
        min_customers,
    };
    model.fitted = if model.structural.is_some() {
        // No search: the flat directions are already known, and running one would
        // pick an arbitrary point along them. `readiness` reports the structural
        // sentence rather than this cause; what the cause is for here is the
        // placeholder surface and the `NaN` draws that come with it.
        Fitted::Boundary(Boundary::CurvatureIsNotAPosterior)
    } else {
        model.locate_mode()
    };

    Ok(model)
}

/// Prior on the *unconstrained* (log) scale: independent Normals, one per coordinate.
///
/// An infinite `sd` is a flat prior on the log scale, which is `p(param) ∝ 1/param` on
/// the natural scale — scale-free, and the default for every coordinate. It is
/// improper, so it contributes nothing to the density and nothing to the gradient;
/// the branch below is what expresses that rather than a very large finite number,
/// which would be a scale assumption wearing a disguise.
#[derive(Debug, Clone, Copy)]
struct Prior {
    log_mean: [f64; 4],
    log_sd: [f64; 4],
}

impl Prior {
    fn parse(cfg: &Config) -> BayesResult<Self> {
        cfg.reject_unknown(&PARAMS)?;
        let mut log_mean = [0.0; 4];
        let mut log_sd = [f64::INFINITY; 4];
        for (j, name) in PARAMS.iter().enumerate() {
            let nested = cfg.nested(name)?;
            nested.reject_unknown(&["log_mean", "log_sd"])?;
            log_mean[j] = nested.f64_or("log_mean", 0.0)?;
            // Absent reads as infinite -- `f64_or` returns an absent slot's default
            // without the finiteness check that a *supplied* value must pass, which is
            // exactly the asymmetry wanted here: the flat default is expressible and
            // an infinity written by hand is not.
            log_sd[j] = nested.positive_f64_or("log_sd", f64::INFINITY)?;
        }
        Ok(Prior { log_mean, log_sd })
    }

    fn logp(&self, theta: &[f64]) -> f64 {
        (0..4)
            .filter(|&j| self.log_sd[j].is_finite())
            .map(|j| {
                let z = (theta[j] - self.log_mean[j]) / self.log_sd[j];
                -0.5 * z * z
            })
            .sum()
    }

    fn add_grad(&self, theta: &[f64], out: &mut [f64]) {
        for j in 0..4 {
            if self.log_sd[j].is_finite() {
                out[j] -= (theta[j] - self.log_mean[j]) / (self.log_sd[j] * self.log_sd[j]);
            }
        }
    }

    /// Whether every coordinate carries a proper prior. Reported so a caller can tell
    /// a fit that could be SBC-certified from one that could not.
    fn is_proper(&self) -> bool {
        self.log_sd.iter().all(|s| s.is_finite())
    }
}

/// One customer's sufficient statistics.
#[derive(Debug, Clone, Copy)]
struct Customer {
    x: f64,
    t_x: f64,
    t: f64,
}

/// The log of the likelihood's two-branch bracket, and the weight of its second
/// branch.
///
/// Returns `(log_sum, w)` where `log_sum` is
/// `ln[(alpha+T)^-(r+x) + 1{x>0} a/(b+x-1) (alpha+t_x)^-(r+x)]` and `w` is the share
/// of that sum contributed by the "dropped out" term.
///
/// `w` is doing two jobs at once, and that is the point. It is the factor every
/// derivative in the gradient needs — each one is a `w`-weighted average of the two
/// branches, which is why the gradient is four short lines for a likelihood this
/// long. And it is exactly `1 - P(alive)`, which is what
/// `p_alive_agrees_with_the_likelihoods_own_decomposition` uses to check the score an
/// agent acts on against the quantity the fit actually maximised.
///
/// Computed by log-sum-exp rather than by exponentiating and adding. With a heavy
/// buyer over a long window, `(r+x) ln(alpha+T)` reaches several hundred, so both
/// terms underflow to zero and their ratio becomes `0/0` — a NaN where the answer is
/// a perfectly ordinary probability.
fn branch_weight(r: f64, alpha: f64, a: f64, b: f64, c: &Customer) -> (f64, f64) {
    let rx = r + c.x;
    let alive = -rx * (alpha + c.t).ln();
    if c.x <= 0.0 {
        // No transaction has yet given the customer an opportunity to drop out, so
        // the second branch does not exist.
        return (alive, 0.0);
    }
    let dead = a.ln() - (b + c.x - 1.0).ln() - rx * (alpha + c.t_x).ln();
    let m = alive.max(dead);
    let (ea, ed) = ((alive - m).exp(), (dead - m).exp());
    let s = ea + ed;
    (m + s.ln(), ed / s)
}

/// What the compile-time mode search concluded.
#[derive(Debug, Clone)]
enum Fitted {
    /// An interior stationary point with positive-definite curvature: a posterior.
    Mode(Vec<f64>),
    /// The search left the admissible region, failed to reach a stationary point, or
    /// reached one whose curvature does not factor. Any of the three means the
    /// Gaussian the engine would fit is not an approximation to a posterior.
    ///
    /// The model still exposes a differentiable surface, because refusing through the
    /// *status* is worth more to an agent than refusing through an error: a
    /// `degenerate` fit is a row in a table it already reads. So the surface becomes a
    /// standard normal — trivially mode-findable, trivially factorable — and
    /// [`LogPosterior::constrain`] writes `NaN` for every parameter, so not one number
    /// derived from it can escape as an estimate. The draws are `NULL`, exactly as for
    /// an unfittable group of `conjugate_anomaly`.
    Boundary(Boundary),
}

/// Which of the mode check's four tests refused, carried into the fit's `reasons`.
///
/// Named individually rather than collapsed into one sentence because they call for
/// different remedies, and an agent that is told only "degenerate" has to guess. The
/// distinction cost nothing to keep and was earned during development: a scale bug in
/// the stationarity test made *large* customer bases refuse while small ones passed,
/// and an undifferentiated message would have sent the search in the wrong direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    /// The Newton search never finished: it ran out of iterations, or walked somewhere
    /// the gradient is not a number.
    SearchDidNotFinish,
    /// The search left the range of parameters anyone holds — see [`LOG_BOX`].
    LeftTheAdmissibleRange,
    /// It stopped somewhere the gradient is still materially non-zero, so it stopped
    /// because it could go no further, not because it had arrived.
    NotStationary,
    /// The curvature at the point found is not positive definite: a saddle, or an
    /// exactly flat direction.
    CurvatureIsNotAPosterior,
    /// The curvature factors, but implies a marginal so wide it locates nothing — the
    /// nearly-flat ridge. See [`MAX_LOG_SD`].
    FlatRidge,
}

impl Boundary {
    fn explain(&self) -> &'static str {
        match self {
            Boundary::SearchDidNotFinish => {
                "the search for the most likely parameters never settled"
            }
            Boundary::LeftTheAdmissibleRange => {
                "the search ran off toward zero or infinity instead of settling"
            }
            Boundary::NotStationary => {
                "the search stopped where the likelihood is still climbing, so there is no \
                 interior maximum to put a posterior around"
            }
            Boundary::CurvatureIsNotAPosterior => {
                "the curvature at the best-fitting parameters is not a covariance: the \
                 likelihood is exactly flat in at least one direction"
            }
            Boundary::FlatRidge => {
                "the likelihood is nearly flat along a ridge, so the parameters are only \
                 identified as a combination and not individually"
            }
        }
    }
}

#[derive(Debug)]
struct CompiledPayerAlive {
    params: Vec<ParamName>,
    obs: Vec<Customer>,
    prior: Prior,
    fitted: Fitted,
    /// A refusal reached from the sufficient statistics alone, before any search.
    structural: Option<String>,
    n_obs: usize,
    fingerprint: String,
    min_customers: usize,
}

/// Crude moments used only to place the search's starting point.
struct Moments {
    mean_x: f64,
    mean_t: f64,
}

fn model_stats(obs: &[Customer]) -> Moments {
    if obs.is_empty() {
        return Moments {
            mean_x: 1.0,
            mean_t: 1.0,
        };
    }
    let n = obs.len() as f64;
    Moments {
        mean_x: obs.iter().map(|c| c.x).sum::<f64>() / n,
        mean_t: obs.iter().map(|c| c.t).sum::<f64>() / n,
    }
}

/// A starting point that is already scaled to the data.
///
/// `r = a = b = 1` is the conventional start, and `alpha` is the one coordinate where
/// it matters: `alpha` is measured in the caller's time units, so starting it at 1
/// when the window is 10 000 seconds costs the search a dozen iterations climbing out
/// of a region where the likelihood is numerically flat. `r/alpha` is the population
/// mean transaction rate, so `alpha = mean(T)/mean(x)` at `r = 1` starts the search
/// where the average customer's observed rate already is.
fn starting_point(m: &Moments) -> Vec<f64> {
    let alpha = (m.mean_t / m.mean_x.max(0.5)).clamp(1e-6, 1e6);
    vec![0.0, alpha.ln(), 0.0, 0.0]
}

impl CompiledPayerAlive {
    /// The largest gradient norm at which this fit's mode is still called stationary.
    ///
    /// Proportional to the number of customers, and that is not a fudge factor. The
    /// log likelihood is a **sum**, so its gradient is a sum of `n` per-customer terms
    /// and both its magnitude and its rounding error scale with `n`. A fixed absolute
    /// threshold therefore means something different at 100 customers than at 100 000,
    /// and in the direction that matters least conveniently: measured during
    /// development, a fixed `1e-3` passed every base of a few hundred and refused a
    /// perfectly ordinary base of four thousand. A model that refuses *more* as the
    /// evidence grows is exactly backwards, and would have shipped as an intermittent
    /// mystery at the largest customers.
    ///
    /// `1e-5` per customer is far below anything a genuine runaway produces — the
    /// boundary fixtures fail this by orders of magnitude, or fail the ridge test
    /// instead — and comfortably above the arithmetic's own noise floor.
    fn stationarity_tolerance(&self) -> f64 {
        MODE_GRAD_TOLERANCE_PER_CUSTOMER * (self.n_obs as f64).max(1.0)
    }

    /// Find the mode and decide whether it is one.
    fn locate_mode(&self) -> Fitted {
        // The point the search reached, whether or not it settled there. Whether it
        // settled is not the question this family asks -- the four tests below judge
        // the *point*, and each of them says something more useful about a refusal
        // than an iteration count would.
        let Ok((mode, _)) = crate::engines::laplace::find_mode_best_effort(self) else {
            // The gradient stopped being a number somewhere along the way.
            return Fitted::Boundary(Boundary::SearchDidNotFinish);
        };
        if mode
            .iter()
            .any(|v| !v.is_finite() || v.abs() > LOG_BOUNDARY)
        {
            return Fitted::Boundary(Boundary::LeftTheAdmissibleRange);
        }
        let mut grad = vec![0.0; 4];
        self.true_grad(&mode, &mut grad);
        if grad.iter().any(|g| !g.is_finite()) {
            return Fitted::Boundary(Boundary::SearchDidNotFinish);
        }
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm >= self.stationarity_tolerance() {
            return Fitted::Boundary(Boundary::NotStationary);
        }
        // The curvature is the posterior's precision. If it does not factor, the point
        // is a saddle or sits on an exactly flat ridge; if it factors but implies an
        // absurdly wide marginal, it sits on a nearly flat one. Both describe nothing.
        let Ok(hessian) = crate::engines::laplace::negative_hessian(self, &mode) else {
            return Fitted::Boundary(Boundary::CurvatureIsNotAPosterior);
        };
        let Ok(factor) = cholesky(&hessian) else {
            return Fitted::Boundary(Boundary::CurvatureIsNotAPosterior);
        };
        for j in 0..4 {
            let mut unit = vec![0.0; 4];
            unit[j] = 1.0;
            // Column j of the inverse precision, whose j-th entry is the marginal
            // posterior variance of coordinate j.
            let Ok(column) = crate::linalg::solve_with(&factor, &unit) else {
                return Fitted::Boundary(Boundary::CurvatureIsNotAPosterior);
            };
            if !column[j].is_finite() || column[j] < 0.0 || column[j].sqrt() > MAX_LOG_SD {
                return Fitted::Boundary(Boundary::FlatRidge);
            }
        }
        Fitted::Mode(mode)
    }

    /// Log likelihood and its gradient with respect to the **natural** parameters.
    fn log_likelihood(
        &self,
        r: f64,
        alpha: f64,
        a: f64,
        b: f64,
        grad: Option<&mut [f64; 4]>,
    ) -> f64 {
        let mut total = 0.0;
        let mut g = [0.0f64; 4];
        let want_grad = grad.is_some();

        for c in &self.obs {
            let (x, t_x, t) = (c.x, c.t_x, c.t);
            let rx = r + x;

            let (log_sum, w) = branch_weight(r, alpha, a, b, c);

            total +=
                ln_gamma(rx) - ln_gamma(r) + r * alpha.ln() + ln_gamma(b + x) + ln_gamma(a + b)
                    - ln_gamma(b)
                    - ln_gamma(a + b + x)
                    + log_sum;

            if want_grad {
                let ln_at = (alpha + t).ln();
                let ln_atx = if x > 0.0 { (alpha + t_x).ln() } else { 0.0 };
                // d/dr
                g[0] += digamma(rx) - digamma(r) + alpha.ln() - (1.0 - w) * ln_at - w * ln_atx;
                // d/dalpha
                g[1] += r / alpha - (1.0 - w) * rx / (alpha + t) - w * rx / (alpha + t_x);
                // d/da
                g[2] += digamma(a + b) - digamma(a + b + x) + w / a;
                // d/db
                g[3] += digamma(b + x) + digamma(a + b)
                    - digamma(b)
                    - digamma(a + b + x)
                    - if x > 0.0 { w / (b + x - 1.0) } else { 0.0 };
            }
        }

        if let Some(out) = grad {
            *out = g;
        }
        total
    }
}

impl CompiledModel for CompiledPayerAlive {
    fn param_names(&self) -> &[ParamName] {
        &self.params
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// One: the four parameters describe the whole customer base.
    ///
    /// A per-customer `P(alive)` is not a per-group *parameter* — it is arithmetic on
    /// these four, done in SQL after the fit. Reporting one group per customer would
    /// make a 200 000-customer fit emit 800 000 parameters to say the same thing four
    /// numbers already say.
    fn n_groups(&self) -> usize {
        1
    }

    fn data_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn readiness(&self) -> Readiness {
        if let Some(reason) = &self.structural {
            return Readiness::degenerate(reason.clone());
        }
        if let Fitted::Boundary(cause) = self.fitted {
            return Readiness::degenerate(format!(
                "no interior maximum for these {} customers: {}. A curvature computed \
                 there is not a posterior, so this fit reports no interval. The usual \
                 cause is that too few repeat buyers have gone quiet -- a recency equal \
                 to its age means the customer has never been seen to stop -- which \
                 drives the dropout probability toward zero and leaves `a` and `b` \
                 unidentified.{}",
                self.n_obs,
                cause.explain(),
                if self.prior.is_proper() {
                    " Extend the observation window past the last transactions: the \
                     prior is already proper, so it is the data that carries no \
                     information about churning, and a tighter prior would only be \
                     answering the question itself"
                } else {
                    " Extend the observation window past the last transactions, or set \
                     a proper `prior`"
                }
            ));
        }
        if self.n_obs < self.min_customers {
            return Readiness::insufficient(format!(
                "{} customers is below the min_customers threshold of {}: four population \
                 parameters estimated from this few describe the sample rather than the base",
                self.n_obs, self.min_customers
            ));
        }
        Readiness::ready()
    }

    fn as_differentiable(&self) -> Option<&dyn LogPosterior> {
        Some(self)
    }
}

impl CompiledPayerAlive {
    /// The real log posterior, whatever the compile-time verdict was.
    ///
    /// Kept separate from [`LogPosterior::logp`] so that the placeholder surface of
    /// [`Fitted::Boundary`] is one visible branch rather than something woven through
    /// the arithmetic, and so that the finite-difference test can measure *this*
    /// unconditionally.
    fn true_logp(&self, theta: &[f64]) -> f64 {
        if theta.iter().any(|v| !v.is_finite() || v.abs() > LOG_BOX) {
            // Outside the admissible box the density is declared to be zero, which
            // stops the Newton line search at the box edge instead of letting it walk
            // into the region where `lnG` of an astronomical argument returns an
            // infinity and every subsequent number is a NaN.
            return f64::NEG_INFINITY;
        }
        let ll = self.log_likelihood(
            theta[0].exp(),
            theta[1].exp(),
            theta[2].exp(),
            theta[3].exp(),
            None,
        );
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        ll + self.prior.logp(theta)
    }

    /// The real gradient. See [`CompiledPayerAlive::true_logp`].
    fn true_grad(&self, theta: &[f64], out: &mut [f64]) {
        let natural = [
            theta[0].exp(),
            theta[1].exp(),
            theta[2].exp(),
            theta[3].exp(),
        ];
        let mut g = [0.0f64; 4];
        self.log_likelihood(natural[0], natural[1], natural[2], natural[3], Some(&mut g));
        // Chain rule onto the log scale: d/d(ln p) = p * d/dp. The prior is declared
        // on the log scale already, so it needs no such factor -- and that is also why
        // no log-Jacobian term appears anywhere.
        for j in 0..4 {
            out[j] = g[j] * natural[j];
        }
        self.prior.add_grad(theta, out);
    }
}

impl LogPosterior for CompiledPayerAlive {
    fn dim(&self) -> usize {
        4
    }

    fn logp(&self, theta: &[f64]) -> f64 {
        match self.fitted {
            Fitted::Boundary(_) => -0.5 * theta.iter().map(|v| v * v).sum::<f64>(),
            Fitted::Mode(_) => self.true_logp(theta),
        }
    }

    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        if theta.len() != 4 || out.len() != 4 {
            return Err(BayesError::DimensionMismatch(format!(
                "expected 4 coordinates, got theta {} and out {}",
                theta.len(),
                out.len()
            )));
        }
        match self.fitted {
            Fitted::Boundary(_) => {
                for j in 0..4 {
                    out[j] = -theta[j];
                }
            }
            Fitted::Mode(_) => self.true_grad(theta, out),
        }
        Ok(())
    }

    fn initial(&self) -> Vec<f64> {
        match &self.fitted {
            Fitted::Mode(theta) => theta.clone(),
            Fitted::Boundary(_) => vec![0.0; 4],
        }
    }

    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        if let Fitted::Boundary(_) = self.fitted {
            out.fill(f64::NAN);
            return;
        }
        for j in 0..4 {
            out[j] = theta[j].exp();
        }
    }
}

/// The real surface, exposed as a [`LogPosterior`] regardless of the verdict.
///
/// Exists for one test, and that test is the most valuable one in the module, so the
/// wrapper earns its keep: without it the finite-difference check would pass on any
/// dataset the model refused, which is exactly the dataset a wrong gradient produces.
#[cfg(test)]
struct TrueSurface<'a>(&'a CompiledPayerAlive);

#[cfg(test)]
impl LogPosterior for TrueSurface<'_> {
    fn dim(&self) -> usize {
        4
    }
    fn logp(&self, theta: &[f64]) -> f64 {
        self.0.true_logp(theta)
    }
    fn grad(&self, theta: &[f64], out: &mut [f64]) -> BayesResult<()> {
        self.0.true_grad(theta, out);
        Ok(())
    }
    fn initial(&self) -> Vec<f64> {
        starting_point(&model_stats(&self.0.obs))
    }
    fn constrain(&self, theta: &[f64], out: &mut [f64]) {
        for j in 0..4 {
            out[j] = theta[j].exp();
        }
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! The generative process, run forwards.
    //!
    //! Shared with `sbc.rs` rather than duplicated, because SBC's whole warranty is
    //! that the simulator and the likelihood describe the same model. Two copies would
    //! eventually describe two models, and the suite would certify the wrong one — the
    //! one bug that makes a calibration test pass vacuously.

    use crate::data::testing::Frame;
    use crate::errors::BayesResult;
    use crate::rng::BayesRng;

    /// A simulated customer base in the three columns the family reads.
    pub(crate) struct Base {
        pub x: Vec<f64>,
        pub t_x: Vec<f64>,
        pub t: Vec<f64>,
    }

    impl Base {
        pub(crate) fn frame(&self) -> Frame {
            Frame::new(self.x.len())
                .numeric("x", self.x.clone())
                .numeric("t_x", self.t_x.clone())
                .numeric("T", self.t.clone())
        }

        /// Share of customers whose last transaction was not at the very end of their
        /// window — the customers that carry the information about dropping out.
        pub(crate) fn lapsed_share(&self) -> f64 {
            let repeaters: Vec<usize> = (0..self.x.len()).filter(|&i| self.x[i] > 0.0).collect();
            if repeaters.is_empty() {
                return 0.0;
            }
            repeaters
                .iter()
                .filter(|&&i| self.t_x[i] < self.t[i] * 0.999)
                .count() as f64
                / repeaters.len() as f64
        }
    }

    /// Draw `n` customers from BG/NBD with the given population parameters.
    ///
    /// Observation windows deliberately vary: a real base is acquired over time, and a
    /// single common `T` removes most of what identifies `alpha` against `r`.
    pub(crate) fn simulate(
        rng: &mut BayesRng,
        n: usize,
        r: f64,
        alpha: f64,
        a: f64,
        b: f64,
        horizon: f64,
    ) -> BayesResult<Base> {
        let mut base = Base {
            x: Vec::with_capacity(n),
            t_x: Vec::with_capacity(n),
            t: Vec::with_capacity(n),
        };
        for _ in 0..n {
            let lambda = rng.gamma(r, alpha)?;
            // Beta(a, b) as the ratio of two Gammas -- the standard construction, and
            // the only one available from the crate's RNG surface.
            let g1 = rng.gamma(a, 1.0)?;
            let g2 = rng.gamma(b, 1.0)?;
            let p = g1 / (g1 + g2);

            let window = horizon * (0.25 + 0.75 * rng.uniform());
            let (mut clock, mut x, mut t_x) = (0.0f64, 0.0f64, 0.0f64);
            loop {
                // Inter-transaction time is Exponential(lambda) while the customer is
                // alive.
                let u = rng.uniform().max(f64::MIN_POSITIVE);
                clock -= u.ln() / lambda;
                if clock > window {
                    break;
                }
                x += 1.0;
                t_x = clock;
                // The "BG" in BG/NBD: the coin is flipped *after* each transaction.
                if rng.uniform() < p {
                    break;
                }
            }
            base.x.push(x);
            base.t_x.push(t_x);
            base.t.push(window);
        }
        Ok(base)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::simulate;
    use super::*;
    use crate::data::testing::Frame;
    use crate::rng::BayesRng;
    use crate::types::FitStatus;

    fn compile<'a>(cfg: &str, data: &'a DataView<'a>) -> BayesResult<Box<dyn CompiledModel + 'a>> {
        PayerAlive.compile(&Config::parse(cfg).unwrap(), data)
    }

    pub(super) const CFG: &str = r#"{"frequency": "x", "recency": "t_x", "age": "T"}"#;

    /// The same model under a proper prior, which is what SBC certifies and what a
    /// caller with a thin base should use.
    pub(super) const PROPER_PRIOR_CFG: &str = r#"{"frequency": "x", "recency": "t_x", "age": "T",
         "prior": {"r": {"log_mean": 0.0, "log_sd": 0.7},
                   "alpha": {"log_mean": 2.5, "log_sd": 1.0},
                   "a": {"log_mean": 0.0, "log_sd": 0.7},
                   "b": {"log_mean": 0.7, "log_sd": 1.0}}}"#;

    /// Population parameters used by every fixture below.
    ///
    /// `r = 0.9, alpha = 12` puts the average customer near one purchase per 13 time
    /// units; `a = 1.1, b = 3.0` puts the mean dropout probability per transaction
    /// near 0.27. Both are in the range Fader et al. report for real retail bases,
    /// which matters: a family tested only where it is comfortable is not tested.
    const TRUTH: (f64, f64, f64, f64) = (0.9, 12.0, 1.1, 3.0);

    /// A customer base drawn from the generative model itself.
    ///
    /// Hand-written archetypes were tried first and rejected: with only three or four
    /// distinct repeat patterns the *spread* of the dropout probability is not
    /// identified, `a` and `b` run off to infinity together at a fixed ratio, and the
    /// fit is legitimately degenerate. That is a true statement about such data and a
    /// useless fixture, so the fixtures simulate.
    pub(super) fn base(n: usize, seed: u64) -> Frame {
        let mut rng = BayesRng::for_chain(seed, 0);
        let (r, alpha, a, b) = TRUTH;
        simulate(&mut rng, n, r, alpha, a, b, 52.0).unwrap().frame()
    }

    /// **The single most valuable test in this module.** A hand-derived gradient that
    /// is subtly wrong still finds *a* mode and still produces plausible-looking
    /// draws; nothing downstream would notice. Finite differences notice.
    ///
    /// Two things make it a real test rather than a ritual.
    ///
    /// It is evaluated **away from the mode** as well as at it. At the mode the
    /// gradient is zero, so a missing term or a sign error is invisible there.
    ///
    /// And it measures [`TrueSurface`] rather than the compiled model, so a wrong
    /// gradient cannot hide behind the refusal path. That is not hypothetical: a
    /// gradient that is not the gradient of anything has an asymmetric Jacobian, the
    /// compile-time mode check calls the fit degenerate, and the trait object then
    /// hands back a placeholder standard normal — whose gradient matches finite
    /// differences perfectly. Measured this way, each of the eight terms below is
    /// individually load-bearing.
    #[test]
    fn the_analytic_gradient_matches_finite_differences() {
        for cfg in [CFG, PROPER_PRIOR_CFG] {
            let frame = base(1500, 11);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let model = build(&Config::parse(cfg).unwrap(), &view).unwrap();
            let target = TrueSurface(&model);

            // Deliberately scattered: every offset moves every coordinate by a
            // different amount, so no two coordinates can swap their derivatives
            // unnoticed.
            for offset in [0.0, 0.35, -0.8, 1.4] {
                let theta: Vec<f64> = target
                    .initial()
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v + offset * (1.0 + j as f64 * 0.21))
                    .collect();
                assert!(
                    target.logp(&theta).is_finite(),
                    "offset {offset} left the admissible region"
                );

                let mut analytic = vec![0.0; 4];
                target.grad(&theta, &mut analytic).unwrap();

                for j in 0..4 {
                    let step = 1e-6 * theta[j].abs().max(1.0);
                    let mut up = theta.clone();
                    let mut down = theta.clone();
                    up[j] += step;
                    down[j] -= step;
                    let numeric = (target.logp(&up) - target.logp(&down)) / (2.0 * step);
                    let tol = 1e-4 * numeric.abs().max(1.0);
                    assert!(
                        (analytic[j] - numeric).abs() < tol,
                        "offset {offset}, coordinate {j} ({}): analytic {} vs numeric {numeric}",
                        PARAMS[j],
                        analytic[j]
                    );
                }
            }
        }
    }

    //=== P(alive) =========================================================//

    /// The closed form, against values computed independently from the formula in the
    /// module header. These same five cases are evaluated by the SQL expression in
    /// `test/sql/f5_payer_alive.test`, which is what ties the two implementations to
    /// one set of numbers rather than to each other's word.
    ///
    /// If you change these, change them there too — deliberately, because the whole
    /// point is that they cannot drift silently.
    /// `(r, alpha, a, b)`, `(x, t_x, T)`, and the expected `P(alive)`.
    type PAliveCase = ([f64; 4], [f64; 3], f64);

    #[test]
    fn p_alive_matches_its_closed_form() {
        let cases: [PAliveCase; 5] = [
            // Never repeated: no dropout opportunity has arisen, so certainly alive.
            ([0.9, 12.0, 1.1, 3.0], [0.0, 0.0, 40.0], 1.0),
            // Bought right up to the end of the window: almost certainly alive.
            ([0.9, 12.0, 1.1, 3.0], [6.0, 39.0, 40.0], 0.8641443110059065),
            // Bought a lot early, then silence for most of a long window: dead.
            (
                [0.9, 12.0, 1.1, 3.0],
                [6.0, 8.0, 40.0],
                0.009864516991833232,
            ),
            // One repeat, long ago. The evidence is much weaker with x = 1, which is
            // the behaviour that stops the model condemning a light buyer.
            ([0.9, 12.0, 1.1, 3.0], [1.0, 5.0, 40.0], 0.2458340720809729),
            // A shorter window: the same recency means much less.
            ([0.9, 12.0, 1.1, 3.0], [6.0, 8.0, 10.0], 0.7902596327648121),
        ];
        for ([r, alpha, a, b], [x, t_x, t], expected) in cases {
            let got = p_alive(r, alpha, a, b, x, t_x, t);
            assert!(
                (got - expected).abs() < 1e-12,
                "p_alive({r}, {alpha}, {a}, {b}, {x}, {t_x}, {t}) = {got}, expected {expected}"
            );
        }
    }

    /// **Two derivations of one number.** `p_alive` multiplies out the odds directly;
    /// the likelihood arrives at the same quantity as the log-sum-exp weight of its
    /// two branches, because `1 - w` *is* `P(alive)`. They are written differently and
    /// used for different purposes, and they must agree — a mismatch would mean the
    /// number an agent acts on is not the number the fit was maximising.
    #[test]
    fn p_alive_agrees_with_the_likelihoods_own_decomposition() {
        for &(r, alpha, a, b) in &[
            (0.9, 12.0, 1.1, 3.0),
            (2.4, 0.7, 0.3, 9.0),
            (0.15, 130.0, 4.0, 0.6),
        ] {
            for &(x, t_x, t) in &[
                (0.0, 0.0, 40.0),
                (1.0, 5.0, 40.0),
                (6.0, 8.0, 40.0),
                (6.0, 39.5, 40.0),
                (23.0, 51.0, 52.0),
            ] {
                let c = Customer { x, t_x, t };
                let (_, w) = branch_weight(r, alpha, a, b, &c);
                let direct = p_alive(r, alpha, a, b, x, t_x, t);
                assert!(
                    (direct - (1.0 - w)).abs() < 1e-12,
                    "direct {direct} vs likelihood branch {} at ({x}, {t_x}, {t})",
                    1.0 - w
                );
            }
        }
    }

    /// The SQL expression is a second implementation of [`p_alive`], living in a file
    /// no Rust test can execute. What *can* be checked mechanically is that the one
    /// string this crate publishes is the string the guide and the SQL suite actually
    /// contain — so an edit to any of the three that forgets the others fails here
    /// rather than at a customer.
    ///
    /// The numbers themselves are tied together by
    /// `p_alive_matches_its_closed_form`, whose five cases the SQL file evaluates and
    /// compares against the same constants.
    #[test]
    fn p_alive_matches_the_documented_sql_expression() {
        for (name, text) in [
            ("docs/GUIDE.md", include_str!("../../../../docs/GUIDE.md")),
            (
                "docs/API_REFERENCE.md",
                include_str!("../../../../docs/API_REFERENCE.md"),
            ),
            (
                "test/sql/f5_payer_alive.test",
                include_str!("../../../../test/sql/f5_payer_alive.test"),
            ),
        ] {
            assert!(
                text.contains(P_ALIVE_SQL),
                "{name} no longer contains the published P(alive) expression:\n{P_ALIVE_SQL}"
            );
        }
    }

    /// The behaviour the collections agent reads: among customers of the same
    /// frequency, the one who has gone quiet is the one the model doubts. Monotone in
    /// recency, which is the property that makes the score usable as a ranking.
    #[test]
    fn p_alive_falls_monotonically_as_a_customer_goes_quiet() {
        let (r, alpha, a, b) = TRUTH;
        let scores: Vec<f64> = [2.0, 10.0, 20.0, 30.0, 39.0]
            .iter()
            .map(|&t_x| p_alive(r, alpha, a, b, 8.0, t_x, 40.0))
            .collect();
        for pair in scores.windows(2) {
            assert!(pair[0] < pair[1], "not monotone in recency: {scores:?}");
        }
        assert!(
            scores[0] < 0.001,
            "a long-silent heavy buyer: {}",
            scores[0]
        );
        assert!(scores[4] > 0.85, "a buyer still active: {}", scores[4]);
    }

    /// More transactions before the silence means stronger evidence of dropping out:
    /// the model has seen more coin flips survive, so a long gap after many purchases
    /// is more damning than the same gap after one.
    #[test]
    fn heavier_buyers_are_condemned_faster_by_the_same_silence() {
        let (r, alpha, a, b) = TRUTH;
        let light = p_alive(r, alpha, a, b, 1.0, 8.0, 40.0);
        let heavy = p_alive(r, alpha, a, b, 12.0, 8.0, 40.0);
        assert!(
            heavy < light,
            "heavy {heavy} should rank below light {light}"
        );
    }

    /// A simulated base of the size a real collections agent works with must fit
    /// cleanly. Without this the refusal path could swallow every dataset and the
    /// suite would still be green.
    #[test]
    fn a_simulated_customer_base_reaches_an_interior_mode() {
        let frame = base(1500, 11);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(CFG, &view).unwrap();
        assert_eq!(
            model.readiness().status,
            FitStatus::Converged,
            "{:?}",
            model.readiness().reasons
        );
        assert_eq!(model.n_obs(), 1500);
        assert_eq!(model.n_groups(), 1);
        let names: Vec<&str> = model
            .param_names()
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["r", "alpha", "a", "b"]);
        // Population parameters, so every one of them is global rather than attached
        // to a customer -- that is what lets one draws table rescore any customer.
        assert!(model
            .param_names()
            .iter()
            .all(|p| p.group_id == crate::types::GLOBAL_GROUP));
    }

    //=== Recovery =========================================================//

    /// Every parameter's draws, keyed by name.
    type Draws = std::collections::BTreeMap<String, Vec<f64>>;

    /// Draws of every parameter from a completed fit, run end to end through the same
    /// path the SQL surface uses.
    fn posterior(cfg: &str, frame: &Frame) -> Draws {
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let fit = crate::fit::fit("payer_alive", &Config::parse(cfg).unwrap(), &view).unwrap();
        assert_eq!(
            fit.posterior.meta.status,
            FitStatus::Converged,
            "{:?}",
            fit.reasons
        );
        let p = fit.posterior.n_params();
        (0..p)
            .map(|j| {
                let name = fit.posterior.params[j].name.clone();
                let col: Vec<f64> = fit
                    .posterior
                    .rows()
                    .filter(|row| row.draw >= 0 && row.param == name)
                    .map(|row| row.value)
                    .collect();
                (name, col)
            })
            .collect()
    }

    fn quantile(xs: &[f64], q: f64) -> f64 {
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    /// **Parameter recovery.** Simulate from known population parameters, fit, and
    /// require the posterior to cover the truth. This is the check that the likelihood,
    /// the gradient, the mode search and the curvature agree on the same model — a
    /// mistake in any one of them moves the interval off the truth.
    ///
    /// `r` and `alpha` are checked individually. The dropout process is checked through
    /// **`a/(a+b)`, the mean dropout probability per transaction**, and that is a
    /// statement about BG/NBD rather than a convenience: `a` and `b` are only weakly
    /// separately identified — the data speaks clearly about where the Beta sits and
    /// only faintly about how wide it is — so an interval for `a` alone is honestly
    /// wide even when the model has learned everything there is to learn. The quantity
    /// a collections decision reads is the mean, and that is what is pinned.
    #[test]
    fn a_simulated_base_recovers_the_population_parameters_it_was_drawn_from() {
        let (r, alpha, a, b) = TRUTH;
        let frame = base(6000, 2026);
        let draws = posterior(
            r#"{"frequency": "x", "recency": "t_x", "age": "T", "draws": 4000, "seed": 5}"#,
            &frame,
        );

        for (name, truth) in [("r", r), ("alpha", alpha)] {
            let col = &draws[name];
            let (lo, hi) = (quantile(col, 0.025), quantile(col, 0.975));
            assert!(
                lo <= truth && truth <= hi,
                "{name}: 95% interval [{lo}, {hi}] does not cover {truth}"
            );
        }

        // The mean dropout probability, computed per draw -- exactly how an analyst
        // would derive it in SQL from the same table.
        let mean_p: Vec<f64> = draws["a"]
            .iter()
            .zip(&draws["b"])
            .map(|(a, b)| a / (a + b))
            .collect();
        let truth_p = a / (a + b);
        let (lo, hi) = (quantile(&mean_p, 0.025), quantile(&mean_p, 0.975));
        assert!(
            lo <= truth_p && truth_p <= hi,
            "a/(a+b): 95% interval [{lo}, {hi}] does not cover {truth_p}"
        );
        // ...and the interval is informative rather than trivially wide: an interval
        // covering everything covers the truth for free.
        assert!(
            hi - lo < 0.2,
            "the dropout interval [{lo}, {hi}] says nothing"
        );
    }

    /// **A model must not refuse more as the evidence grows.** Both of these truths
    /// were found by the SBC suite, each on a base the first implementation refused
    /// while happily fitting smaller ones drawn from the same prior — the worst shape
    /// a defect can take, because it appears only at the largest customers.
    ///
    /// Two separate causes, both scale effects of a sum over observations. An absolute
    /// gradient threshold for stationarity means something different at 100 customers
    /// than at 100 000, so it now scales with the base. And the Newton search's fixed
    /// iteration budget is reached on a large, slowly converging surface, so the family
    /// now judges the *point* the search reached rather than whether the search
    /// finished. Neither is a tolerance loosened to make a test pass: each replaced a
    /// quantity that was wrong to hold fixed.
    #[test]
    fn a_larger_base_does_not_become_harder_to_fit_than_a_smaller_one() {
        for (n, truth, seed) in [
            (
                4000usize,
                (1.2866078, 11.1291939, 0.3478822, 3.3143089),
                91u64,
            ),
            (8000, (2.5946126, 11.9349800, 0.6401639, 3.2077485), 92),
        ] {
            let mut rng = BayesRng::for_chain(seed, 0);
            let sim = simulate(&mut rng, n, truth.0, truth.1, truth.2, truth.3, 52.0).unwrap();
            let frame = sim.frame();
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let model = compile(CFG, &view).unwrap();
            assert_eq!(
                model.readiness().status,
                FitStatus::Converged,
                "n = {n}: {:?}",
                model.readiness().reasons
            );
        }
    }

    //=== Refusal ==========================================================//

    /// **The boundary solution the roadmap called out as the residual risk of choosing
    /// Laplace.** If no repeat buyer has gone quiet — every recency sitting at the end
    /// of its own observation window — then nothing in the data has ever looked like
    /// dropping out. The likelihood is then maximised only in the limit `p -> 0`, the
    /// mode search runs away toward `a -> 0`, and the curvature at wherever it stopped
    /// is not a posterior.
    ///
    /// Reporting an interval from that curvature is the precise failure this extension
    /// exists to prevent, so the fit refuses: `degenerate`, with `NULL` draws, and a
    /// reason naming the cause.
    #[test]
    fn a_base_where_no_repeat_buyer_ever_went_quiet_is_refused_as_degenerate() {
        // A real shape, not a contrivance: a subscription base snapshotted at the
        // renewal date, so every active account's last payment is "just now" and the
        // window ends there. Nobody in the table has ever been seen to stop.
        let mut base = super::testing::simulate(
            &mut BayesRng::for_chain(77, 0),
            800,
            TRUTH.0,
            TRUTH.1,
            TRUTH.2,
            TRUTH.3,
            52.0,
        )
        .unwrap();
        for i in 0..base.x.len() {
            if base.x[i] > 0.0 {
                base.t[i] = base.t_x[i];
            }
        }
        assert_eq!(base.lapsed_share(), 0.0, "the fixture must have no lapsers");

        let frame = base.frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let fit = crate::fit::fit("payer_alive", &Config::parse(CFG).unwrap(), &view).unwrap();

        assert_eq!(fit.posterior.meta.status, FitStatus::Degenerate);
        assert!(!fit.posterior.meta.status.is_actionable());
        assert!(
            fit.reasons
                .iter()
                .any(|r| r.contains("no interior maximum")),
            "{:?}",
            fit.reasons
        );
        // Not an interval: every draw is NULL-shaped, so no number derived from this
        // fit can be mistaken for an estimate.
        assert!(
            fit.posterior
                .rows()
                .filter(|row| row.draw >= 0)
                .all(|row| row.value.is_nan()),
            "a refused fit must not emit numbers"
        );
        // ...and the same data with a proper prior is fittable again, because the
        // prior supplies the information the data does not. That is the documented way
        // out, and it has to actually work.
        let rescued = crate::fit::fit(
            "payer_alive",
            &Config::parse(PROPER_PRIOR_CFG).unwrap(),
            &view,
        )
        .unwrap();
        assert_eq!(rescued.posterior.meta.status, FitStatus::Converged);
    }

    /// The other end of the same problem, and cheap enough to catch without a search:
    /// with no repeat transactions anywhere, the likelihood does not contain `a` or `b`
    /// at all. Both terms of the Beta cancel exactly when `x = 0`.
    #[test]
    fn a_base_with_no_repeat_purchases_cannot_identify_the_dropout_process() {
        let n = 200;
        let frame = Frame::new(n)
            .numeric("x", vec![0.0; n])
            .numeric("t_x", vec![0.0; n])
            .numeric("T", (0..n).map(|i| 20.0 + (i % 30) as f64).collect());
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let model = compile(CFG, &view).unwrap();

        let verdict = model.readiness();
        assert_eq!(verdict.status, FitStatus::Degenerate);
        assert!(
            verdict.reasons[0].contains("repeat transaction"),
            "{:?}",
            verdict.reasons
        );
    }

    /// A base too thin to say anything about a population is a refusal, not an error:
    /// the arithmetic works, the answer describes the sample. The draws stay real
    /// numbers -- it is the evidence that is thin, not the posterior that is absent --
    /// which is the same distinction `conjugate_anomaly` draws at `min_obs`.
    #[test]
    fn a_base_below_the_min_customers_threshold_is_reported_as_insufficient() {
        let frame = base(1500, 11);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let model = compile(CFG, &view).unwrap();
        assert_eq!(model.readiness().status, FitStatus::Converged);

        let model = compile(
            r#"{"frequency": "x", "recency": "t_x", "age": "T", "min_customers": 5000}"#,
            &view,
        )
        .unwrap();
        assert_eq!(model.readiness().status, FitStatus::InsufficientData);
        assert!(model.readiness().reasons[0].contains("min_customers"));
    }

    /// Fewer usable rows than parameters is a hard error rather than a status: there is
    /// no posterior to report a verdict about.
    #[test]
    fn fewer_customers_than_parameters_is_reported_before_any_search() {
        let frame = Frame::new(3)
            .numeric("x", vec![1.0, 2.0, 0.0])
            .numeric("t_x", vec![3.0, 4.0, 0.0])
            .numeric("T", vec![10.0, 10.0, 10.0]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        assert!(matches!(
            compile(CFG, &view).unwrap_err(),
            BayesError::InsufficientData { .. }
        ));
    }

    //=== Config and data validation =======================================//

    #[test]
    fn the_three_statistics_are_required_and_must_exist() {
        let frame = base(60, 3);
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        assert!(compile(r#"{}"#, &view).is_err());
        assert!(compile(r#"{"frequency": "x"}"#, &view).is_err());
        let err = compile(
            r#"{"frequency": "x", "recency": "t_x", "age": "tenure"}"#,
            &view,
        )
        .unwrap_err();
        assert!(matches!(err, BayesError::MissingColumn { .. }));
        assert!(err.to_string().contains("T"), "{err}");
    }

    #[test]
    fn an_unknown_slot_is_rejected_with_a_suggestion() {
        let frame = base(60, 3);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"frequency": "x", "recency": "t_x", "age": "T", "recncy": "t_x"}"#,
            &view,
        )
        .unwrap_err();
        assert!(err.to_string().contains("did you mean 'recency'"), "{err}");
    }

    #[test]
    fn an_unknown_prior_slot_is_rejected_with_its_full_path() {
        let frame = base(60, 3);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = compile(
            r#"{"frequency": "x", "recency": "t_x", "age": "T", "prior": {"r": {"sd": 1.0}}}"#,
            &view,
        )
        .unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "prior.r.sd"));
    }

    /// Data that cannot have come from the process is a mistake in preparation, and
    /// naming it is worth far more than a likelihood that quietly absorbs it.
    #[test]
    fn statistics_that_could_not_have_arisen_are_rejected_and_say_which_row() {
        let bad = |x: Vec<f64>, t_x: Vec<f64>, t: Vec<f64>, slot: &str| {
            let n = x.len();
            let frame = Frame::new(n)
                .numeric("x", x)
                .numeric("t_x", t_x)
                .numeric("T", t);
            let refs = frame.key_refs();
            let view = frame.view(&refs);
            let err = PayerAlive
                .compile(&Config::parse(CFG).unwrap(), &view)
                .unwrap_err();
            assert!(
                matches!(&err, BayesError::Config { slot: got, .. } if got == slot),
                "expected a {slot} error, got {err}"
            );
        };
        let ok_t = vec![10.0; 6];
        // A fractional transaction count.
        bad(
            vec![1.0, 2.5, 3.0, 1.0, 0.0, 2.0],
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
            ok_t.clone(),
            "frequency",
        );
        // A negative count.
        bad(
            vec![1.0, -2.0, 3.0, 1.0, 0.0, 2.0],
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 5.0],
            ok_t.clone(),
            "frequency",
        );
        // Recency past the end of the observation window.
        bad(
            vec![1.0; 6],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 99.0],
            ok_t.clone(),
            "recency",
        );
        // A repeat buyer whose last transaction is at time zero.
        bad(
            vec![1.0; 6],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
            ok_t.clone(),
            "recency",
        );
        // An observation window of no length.
        bad(
            vec![1.0; 6],
            vec![1.0; 6],
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 0.0],
            "age",
        );
    }

    #[test]
    fn the_same_seed_reproduces_the_same_draws_and_a_different_one_does_not() {
        let frame = base(400, 19);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let run = |seed: u64| {
            crate::fit::fit(
                "payer_alive",
                &Config::parse(&format!(
                    r#"{{"frequency": "x", "recency": "t_x", "age": "T", "draws": 500, "seed": {seed}}}"#
                ))
                .unwrap(),
                &view,
            )
            .unwrap()
        };
        let (a, b) = (run(3), run(3));
        assert_eq!(a.posterior.meta.model_id, b.posterior.meta.model_id);
        assert_eq!(
            a.posterior.rows().collect::<Vec<_>>(),
            b.posterior.rows().collect::<Vec<_>>()
        );
        assert_ne!(a.posterior.meta.model_id, run(4).posterior.meta.model_id);
    }

    /// This family has no closed form, so the exact engine must decline rather than
    /// substitute something. An agent that asked for an exact posterior and quietly
    /// received an approximation would report unearned confidence.
    #[test]
    fn the_exact_engine_declines_this_family_rather_than_approximating_it() {
        let frame = base(400, 19);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let err = crate::fit::fit(
            "payer_alive",
            &Config::parse(
                r#"{"frequency": "x", "recency": "t_x", "age": "T", "engine": "exact"}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "engine"),
            "{err}"
        );
        assert!(err.to_string().contains("payer_alive"), "{err}");
    }
}
