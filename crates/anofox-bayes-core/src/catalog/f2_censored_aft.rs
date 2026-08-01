//! F2 — censored accelerated failure time, **bridged** onto `anofox-statistics`.
//!
//! The duration model behind a delivery promise: how long until the thing happens,
//! when some of the things have not happened yet.
//!
//! ```text
//!   log T = x'beta + sigma * W
//! ```
//!
//! for a fixed standard error distribution `W` — extreme value (`weibull`), normal
//! (`lognormal`), logistic (`loglogistic`), or extreme value with `sigma` held at 1
//! (`exponential`). A row whose event was observed contributes its density; a row
//! still open at the end of the window contributes its survival, which is the entire
//! point. Fitting a duration model on observed times alone attenuates every covariate
//! effect and compresses the spread, because a large unobserved duration has been
//! replaced by the smaller time at which we stopped looking.
//!
//! ## This family is a bridge, and says so
//!
//! Nothing in this module derives a likelihood, a gradient or a mode. All four of
//! those already exist, tested, in `anofox-stats-core`, and this family calls them
//! in-process through [`crate::bridge`]. What this crate contributes is what
//! `anofox-statistics` does not have: the draws contract, the diagnostics, the refusal
//! path, `model_id`, and the calibration harness.
//!
//! Three consequences are deliberate and load-bearing.
//!
//! **The posterior is a Gaussian approximation, and the warranty is weaker than F3's
//! or F7's.** F3 and F7 are conjugate: their posteriors are exact and two independent
//! engines cross-check each other. F2 has one engine and no closed form. Its
//! `__engine__` row reads `laplace` and that is not a detail — a caller reading a
//! draws table can tell an exact posterior from an approximate one, and must.
//!
//! **The approximation is on the unconstrained scale.** `sigma` is positive, so it is
//! sampled as `log sigma` and exponentiated. That is also the coordinate the upstream
//! likelihood is already written in, and the prior — when one is given — is declared
//! on the coefficients only, so no change of variables happens anywhere and there is
//! no log-Jacobian term. Checked directly against the closed-form density in
//! `bridge::tests::the_bridged_log_density_matches_the_closed_form_censored_weibull`,
//! not inferred from agreement between engines.
//!
//! **Each group is an independent fit.** With a `group` column the family fits one AFT
//! per group — there is no pooling, and a thin group borrows no strength from a thick
//! one. That is the honest description of what the bridged likelihood does; partial
//! pooling across groups is a hierarchical variance parameter and belongs to the NUTS
//! phase, not here.
//!
//! ## Calibration
//!
//! Bridging is not an escape from the `AGENTS.md` bar. The SBC suite for this family
//! lives in `sbc.rs::families` and its results — including where it does *not* pass —
//! are recorded in `docs/THEORY.md`. A likelihood that fails SBC is documented as
//! failing rather than quietly shipped.

use anofox_stats_core::models::AftDistribution;
use anofox_stats_core::types::PriorSpec;

use crate::bridge::{fit_censored_aft, Bridged, BridgedGaussian, CensoredAftRequest};
use crate::config::Config;
use crate::data::DataView;
use crate::draws::ParamName;
use crate::errors::{BayesError, BayesResult};
use crate::types::{EngineKind, FamilyCode};

use super::{CompiledModel, GaussianApproximation, GaussianBlock, ModelFamily, Readiness};

#[derive(Debug)]
pub struct CensoredAft;

const SLOTS: &[&str] = &[
    "time",
    "event",
    "x",
    "intercept",
    "group",
    "dist",
    "prior",
    "draws",
    "chains",
    "max_draw_megabytes",
    "seed",
    "engine",
    "sample_from",
];

const DISTRIBUTIONS: &[&str] = &["weibull", "lognormal", "loglogistic", "exponential"];

impl ModelFamily for CensoredAft {
    fn id(&self) -> &'static str {
        "censored_aft"
    }

    fn code(&self) -> FamilyCode {
        FamilyCode::CensoredAft
    }

    fn description(&self) -> &'static str {
        "Accelerated failure time regression with right censoring (Weibull, lognormal, \
         log-logistic, exponential), bridged onto anofox-statistics' likelihood and \
         served as a Laplace posterior; the inference layer for delivery-promise and \
         time-to-event questions."
    }

    /// Laplace, and there is no alternative: the censored AFT posterior has no closed
    /// form, so `as_exact` is `None` and asking for `engine = 'exact'` is an error
    /// rather than a silent substitution.
    fn default_engine(&self) -> EngineKind {
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
        cfg.reject_unknown(SLOTS)?;

        let time_name = cfg.require_str("time")?.to_string();
        let event_name = cfg.require_str("event")?.to_string();
        let x_names = cfg.str_list("x")?;
        let intercept = cfg.f64_or("intercept", 1.0)? != 0.0;
        let group = cfg.opt_str("group")?.map(str::to_string);
        let dist_name = cfg.one_of("dist", DISTRIBUTIONS, "weibull")?;
        let dist = AftDistribution::from_name(&dist_name).ok_or_else(|| {
            BayesError::config("dist", format!("unknown AFT distribution '{dist_name}'"))
        })?;

        let prior = cfg.nested("prior")?;
        prior.reject_unknown(&["beta_scale"])?;
        // Infinite by default: a flat prior on the coefficients. Any finite default
        // would be a scale assumption about somebody else's durations.
        let beta_scale = prior.f64_or("beta_scale", f64::INFINITY)?;
        if beta_scale <= 0.0 {
            return Err(BayesError::config("prior.beta_scale", "must be > 0"));
        }

        if x_names.is_empty() && !intercept {
            return Err(BayesError::config(
                "x",
                "a duration model with no predictors and no intercept has nothing to \
                 estimate",
            ));
        }
        if x_names.iter().any(|n| *n == time_name || *n == event_name) {
            return Err(BayesError::config(
                "x",
                "the duration and the event indicator may not also be predictors",
            ));
        }

        // --- Resolve columns and filter nulls, before any arithmetic. ---
        //
        // Whole-row filtering happens here and nowhere else. `fit_aft` runs a filter of
        // its own, and the bridge refuses outright if that filter removes anything,
        // because the data fingerprint below is computed over *these* rows and
        // `model_id` would otherwise be a claim about rows the fit never read.
        let mut numeric_cols: Vec<&str> = vec![time_name.as_str(), event_name.as_str()];
        numeric_cols.extend(x_names.iter().map(String::as_str));
        let key_cols: Vec<&str> = group.iter().map(String::as_str).collect();

        let rows = data.usable_rows(&numeric_cols, &key_cols)?;
        let fingerprint = data.fingerprint(&numeric_cols, &key_cols, &rows)?;

        let time_col = data.numeric(&time_name)?;
        let event_col = data.numeric(&event_name)?;
        let x_cols: Vec<_> = x_names
            .iter()
            .map(|n| data.numeric(n))
            .collect::<BayesResult<_>>()?;

        let groups = data.group_rows(group.as_deref(), &rows)?;
        if group.is_some() {
            for (key, _) in &groups {
                crate::types::validate_group_key(key)?;
            }
        }

        // The prior list is positionally aligned with the design: a flat entry for the
        // intercept (this family never penalises it — shrinking a duration *level*
        // toward zero would be a claim that everything happens instantly) followed by
        // one entry per predictor.
        let priors: Vec<PriorSpec> = if beta_scale.is_finite() {
            let mut p = Vec::with_capacity(x_names.len() + usize::from(intercept));
            if intercept {
                p.push(PriorSpec::flat());
            }
            p.extend(std::iter::repeat_n(
                PriorSpec::normal(0.0, beta_scale),
                x_names.len(),
            ));
            p
        } else {
            Vec::new()
        };

        // --- One independent fit per group. ---
        let mut params: Vec<ParamName> = Vec::new();
        let mut blocks: Vec<GaussianBlock> = Vec::new();
        let mut metas: Vec<BlockMeta> = Vec::new();
        let mut verdicts: Vec<Readiness> = Vec::new();
        let mut n_obs = 0usize;

        for (key, group_rows) in &groups {
            let time: Vec<f64> = group_rows.iter().map(|&i| time_col.values[i]).collect();
            let event: Vec<f64> = group_rows.iter().map(|&i| event_col.values[i]).collect();
            let x: Vec<Vec<f64>> = x_cols
                .iter()
                .map(|c| group_rows.iter().map(|&i| c.values[i]).collect())
                .collect();
            n_obs += time.len();

            // Parameter slots are allocated for every group, fitted or not: a refused
            // group reports NULL draws under its own name rather than vanishing from
            // the table, so an agent iterating over lanes sees all of them.
            let first_slot = params.len();
            if intercept {
                params.push(ParamName::grouped(key.clone(), "intercept")?);
            }
            for name in &x_names {
                params.push(ParamName::grouped(key.clone(), format!("beta[{name}]"))?);
            }
            let n_coef = params.len() - first_slot;
            params.push(ParamName::grouped(key.clone(), "sigma")?);
            let slots: Vec<usize> = (first_slot..params.len()).collect();

            let request = CensoredAftRequest {
                dist,
                time: &time,
                event: &event,
                x: &x,
                intercept,
                priors: priors.clone(),
            };
            match fit_censored_aft(&request)? {
                Bridged::Fitted(g) => {
                    let g: BridgedGaussian = *g;
                    if g.dim() != n_coef + usize::from(g.fit_scale) {
                        return Err(BayesError::Internal(format!(
                            "the bridged fit for group '{key}' returned {} coordinates for \
                             {n_coef} coefficients",
                            g.dim()
                        )));
                    }
                    metas.push(BlockMeta {
                        n_coef,
                        fit_scale: g.fit_scale,
                        fixed_scale: g.scale,
                    });
                    blocks.push(GaussianBlock {
                        mode: g.mode,
                        precision: g.information,
                        params: slots,
                    });
                    verdicts.push(Readiness::ready());
                }
                Bridged::Refused(mut r) => {
                    // Name the group in the reason. A model-level status of
                    // `degenerate` over four hundred lanes is only actionable if it
                    // says which lane.
                    for reason in r.reasons.iter_mut() {
                        *reason = format!("group '{key}': {reason}");
                    }
                    verdicts.push(r);
                }
            }
        }

        let readiness = Readiness::worst(verdicts.iter().cloned());
        let n_groups_unready = verdicts
            .iter()
            .filter(|v| !v.status.is_actionable())
            .count();

        Ok(Box::new(CompiledCensoredAft {
            params,
            blocks,
            metas,
            readiness,
            n_groups_unready,
            n_obs,
            n_groups: groups.len(),
            fingerprint,
        }))
    }
}

/// How one block's unconstrained coordinates map onto the parameters it reports.
#[derive(Debug, Clone)]
struct BlockMeta {
    /// Coefficients, which pass through unchanged.
    n_coef: usize,
    /// Whether a `log sigma` coordinate follows them.
    fit_scale: bool,
    /// The value of `sigma` when it is not estimated (the exponential case, where it
    /// is held at 1). Reported as a draw so that every distribution produces the same
    /// parameter set and a downstream query need not branch on `dist`.
    fixed_scale: f64,
}

#[derive(Debug)]
struct CompiledCensoredAft {
    params: Vec<ParamName>,
    blocks: Vec<GaussianBlock>,
    metas: Vec<BlockMeta>,
    readiness: Readiness,
    n_groups_unready: usize,
    n_obs: usize,
    n_groups: usize,
    fingerprint: String,
}

impl CompiledModel for CompiledCensoredAft {
    fn param_names(&self) -> &[ParamName] {
        &self.params
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn n_groups(&self) -> usize {
        self.n_groups
    }

    fn data_fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn readiness(&self) -> Readiness {
        self.readiness.clone()
    }

    /// Counted exactly rather than defaulted: this family fits each group
    /// independently, so it knows precisely which groups it refused.
    fn n_groups_unready(&self) -> usize {
        self.n_groups_unready
    }

    fn as_gaussian(&self) -> Option<&dyn GaussianApproximation> {
        Some(self)
    }
}

impl GaussianApproximation for CompiledCensoredAft {
    fn blocks(&self) -> &[GaussianBlock] {
        &self.blocks
    }

    /// Coefficients pass through; `sigma` is the exponential of the last coordinate.
    ///
    /// The exponential is the whole of the constraining transform, and it is why the
    /// approximation is fitted on `log sigma` in the first place: a Gaussian on
    /// `sigma` puts mass below zero, which is not a scale.
    fn constrain(&self, block: usize, theta: &[f64], out: &mut [f64]) {
        let meta = &self.metas[block];
        out[..meta.n_coef].copy_from_slice(&theta[..meta.n_coef]);
        out[meta.n_coef] = if meta.fit_scale {
            theta[meta.n_coef].exp()
        } else {
            meta.fixed_scale
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::testing::survival_fixture;
    use crate::data::testing::Frame;
    use crate::engines::{Engine, LaplaceEngine, SampleOptions};
    use crate::types::FitStatus;

    /// A pooled delivery-time fixture: one lane, a covariate measured well away from
    /// zero (distance in hundreds of kilometres), staggered right censoring.
    fn frame(n: usize, x_offset: f64, censor_at: Option<f64>) -> Frame {
        let (time, xs, event) = survival_fixture(
            AftDistribution::Weibull,
            1.0,
            0.25,
            0.4,
            n,
            x_offset,
            censor_at,
        );
        Frame::new(n)
            .numeric("days", time)
            .numeric("distance", xs)
            .numeric("delivered", event)
    }

    /// Compile against an already-borrowed view. The view is built by the caller
    /// because a `CompiledModel` borrows it for its lifetime, so a helper that made
    /// one internally would be returning a borrow of its own temporary.
    fn compile<'a>(view: &'a DataView<'a>, cfg: &str) -> Box<dyn CompiledModel + 'a> {
        CensoredAft
            .compile(&Config::parse(cfg).unwrap(), view)
            .unwrap()
    }

    const CFG: &str = r#"{"time": "days", "event": "delivered", "x": "distance"}"#;

    #[test]
    fn the_family_reports_one_parameter_per_coefficient_plus_the_scale() {
        let f = frame(200, 10.0, Some(40.0));
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, CFG);
        let names: Vec<&str> = model
            .param_names()
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, vec!["intercept", "beta[distance]", "sigma"]);
        assert_eq!(model.n_obs(), 200);
        assert_eq!(model.n_groups(), 1);
        assert_eq!(model.readiness().status, FitStatus::Converged);
        assert_eq!(model.n_groups_unready(), 0);
    }

    /// **The test that makes or breaks this task, stated on the draws themselves.**
    ///
    /// Everything above the engine could be correct and the posterior still be wrong
    /// if the draws were generated coefficient by coefficient. So the check is made
    /// where a customer would feel it: the standard deviation of the *linear
    /// predictor* `eta = intercept + beta * distance` computed from the draws, against
    /// the two candidate answers computed in closed form from the covariance —
    /// `x' V x` using the whole matrix, and `sum x_j^2 V_jj` using only its diagonal.
    ///
    /// The draws must match the first and must not match the second. On this fixture
    /// the two differ by a factor of about 25, so there is no tolerance at which both
    /// could pass.
    ///
    /// **Verified by mutation, and the result is the reason this test exists.**
    /// Replacing `GaussianBlock::precision` with its own diagonal — the change that
    /// would follow from taking `std_errors` off the SQL aggregate instead of
    /// reassembling the matrix — leaves **219 of the 220 tests in this crate green**,
    /// including the diagnostics, the round trip through the draws contract and
    /// `model_id` reproducibility. This one fails.
    ///
    /// **All six SBC suites also stay green under the mutation.** That is not a defect
    /// in the calibration harness, it is what SBC measures: ranks are computed per
    /// parameter, so they test the *marginal* posterior of each coefficient — and the
    /// marginals are exactly what a diagonal preserves. What a diagonal destroys is the
    /// joint, which no marginal rank can see. So the strongest gate in this crate does
    /// not cover the one error the bridge was most likely to make, and a test written
    /// on a *function of several parameters at once* is the only thing that does.
    ///
    /// The general lesson, worth carrying to the next bridged likelihood: **a
    /// per-parameter check cannot certify a covariance.** Whatever else a new family
    /// gets, it needs one assertion on a linear combination of its parameters.
    #[test]
    fn the_draws_carry_the_posterior_correlation_and_not_only_its_variances() {
        let f = frame(300, 10.0, Some(40.0));
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, CFG);

        let sample = LaplaceEngine
            .sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 100_000,
                    seed: 5,
                    sample_from: crate::types::SampleFrom::Posterior,
                },
            )
            .unwrap();

        let p = model.param_names().len();
        let distance = 11.0_f64;
        let eta: Vec<f64> = sample
            .values
            .chunks(p)
            .map(|c| c[0] + c[1] * distance)
            .collect();
        let mean = eta.iter().sum::<f64>() / eta.len() as f64;
        let observed =
            (eta.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (eta.len() - 1) as f64).sqrt();

        // The two candidate answers, from the covariance the bridge computed.
        let g = {
            let (time, xs, event) = survival_fixture(
                AftDistribution::Weibull,
                1.0,
                0.25,
                0.4,
                300,
                10.0,
                Some(40.0),
            );
            let x = vec![xs];
            match fit_censored_aft(&CensoredAftRequest {
                dist: AftDistribution::Weibull,
                time: &time,
                event: &event,
                x: &x,
                intercept: true,
                priors: Vec::new(),
            })
            .unwrap()
            {
                Bridged::Fitted(g) => *g,
                other => panic!("{other:?}"),
            }
        };
        let xv = [1.0_f64, distance];
        let full: f64 = (0..2)
            .flat_map(|a| (0..2).map(move |b| (a, b)))
            .map(|(a, b)| xv[a] * xv[b] * g.covariance[(a, b)])
            .sum::<f64>()
            .sqrt();
        let diagonal: f64 = (0..2)
            .map(|a| xv[a] * xv[a] * g.covariance[(a, a)])
            .sum::<f64>()
            .sqrt();

        assert!(
            diagonal / full > 10.0,
            "the fixture must be correlated for this test to mean anything: {diagonal} vs {full}"
        );
        // 100k independent draws give ~0.2 % Monte Carlo error on a standard
        // deviation, so 2 % is ten times the noise floor and a twentieth of the gap
        // being discriminated.
        assert!(
            (observed - full).abs() < 0.02 * full,
            "the draws' predictive sd {observed} must match the full-covariance answer \
             {full}, not the diagonal-only {diagonal}"
        );
        assert!(
            (observed - diagonal).abs() > 0.5 * diagonal,
            "the draws' predictive sd {observed} must be nowhere near the diagonal-only \
             answer {diagonal}"
        );
    }

    /// `sigma` is drawn as the exponential of a Gaussian coordinate, so every draw is
    /// positive by construction. A Gaussian fitted to `sigma` directly would put mass
    /// below zero, which is not a scale.
    #[test]
    fn the_scale_stays_positive_because_it_is_sampled_on_the_log_scale() {
        let f = frame(200, 0.0, Some(40.0));
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, CFG);
        let sample = LaplaceEngine
            .sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 20_000,
                    seed: 9,
                    sample_from: crate::types::SampleFrom::Posterior,
                },
            )
            .unwrap();
        let p = model.param_names().len();
        assert_eq!(model.param_names()[p - 1].name, "sigma");
        assert!(sample.values.chunks(p).all(|c| c[p - 1] > 0.0));
    }

    /// The exponential distribution holds `sigma` at 1 and does not estimate it, so
    /// the block is one coordinate narrower — but the family still reports a `sigma`
    /// parameter, so a downstream query does not have to branch on `dist`.
    #[test]
    fn the_exponential_distribution_reports_a_scale_it_did_not_estimate() {
        let f = frame(200, 0.0, Some(40.0));
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(
            &view,
            r#"{"time": "days", "event": "delivered", "x": "distance", "dist": "exponential"}"#,
        );
        let g = model.as_gaussian().unwrap();
        assert_eq!(g.blocks()[0].mode.len(), 2, "no log sigma coordinate");
        assert_eq!(g.blocks()[0].params.len(), 3, "sigma is still reported");

        let sample = LaplaceEngine
            .sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 500,
                    seed: 3,
                    sample_from: crate::types::SampleFrom::Posterior,
                },
            )
            .unwrap();
        let p = model.param_names().len();
        assert!(sample.values.chunks(p).all(|c| c[p - 1] == 1.0));
    }

    /// Ignoring censoring attenuates the covariate effect and compresses the scale.
    /// This is the bias the family exists to avoid, measured through the draws rather
    /// than asserted in a comment.
    #[test]
    fn honouring_the_event_indicator_changes_the_answer() {
        let (time, xs, event) =
            survival_fixture(AftDistribution::Weibull, 2.0, 0.3, 0.5, 300, 0.0, Some(9.0));
        let censored = event.iter().filter(|e| **e == 0.0).count();
        assert!(censored > 40, "the fixture must censor: {censored}");

        let f = Frame::new(300)
            .numeric("days", time)
            .numeric("distance", xs)
            .numeric("delivered", event)
            .numeric("pretend_all_delivered", vec![1.0; 300]);
        let refs = f.key_refs();
        let view = f.view(&refs);

        let slope_of = |event_col: &str| {
            let cfg = format!(r#"{{"time": "days", "event": "{event_col}", "x": "distance"}}"#);
            let model = compile(&view, &cfg);
            model.as_gaussian().unwrap().blocks()[0].mode[1]
        };
        let honest = slope_of("delivered");
        let naive = slope_of("pretend_all_delivered");

        assert!((honest - 0.3).abs() < 0.05, "honest slope {honest}");
        assert!(
            naive < 0.5 * honest,
            "ignoring censoring must attenuate the slope: {naive} vs {honest}"
        );
    }

    //=== grouping and refusal ============================================//

    fn grouped_frame() -> Frame {
        // Two healthy lanes and one where every shipment is still in transit.
        let mut days = Vec::new();
        let mut distance = Vec::new();
        let mut delivered = Vec::new();
        let mut lane = Vec::new();
        for (key, b0) in [("HAM-ROT", 1.0), ("BRE-ANT", 1.4)] {
            let (t, x, e) = survival_fixture(
                AftDistribution::Weibull,
                b0,
                0.25,
                0.4,
                120,
                0.0,
                Some(20.0),
            );
            days.extend(t);
            distance.extend(x);
            delivered.extend(e);
            lane.extend(std::iter::repeat_n(key, 120));
        }
        let (t, x, _) = survival_fixture(AftDistribution::Weibull, 1.0, 0.25, 0.4, 60, 0.0, None);
        days.extend(t);
        distance.extend(x);
        delivered.extend(std::iter::repeat_n(0.0, 60));
        lane.extend(std::iter::repeat_n("OPEN-ONLY", 60));

        Frame::new(300)
            .numeric("days", days)
            .numeric("distance", distance)
            .numeric("delivered", delivered)
            .key("lane", lane)
    }

    const GROUPED_CFG: &str =
        r#"{"time": "days", "event": "delivered", "x": "distance", "group": "lane"}"#;

    /// A lane where nothing has been delivered yet cannot support a duration model,
    /// and the roadmap names the verdict: `degenerate`, not `insufficient_data`. There
    /// is no shortage of rows — there is no information in them about when things
    /// happen.
    #[test]
    fn a_lane_with_no_delivered_shipment_is_degenerate_and_named() {
        let f = grouped_frame();
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, GROUPED_CFG);

        assert_eq!(model.n_groups(), 3);
        assert_eq!(model.n_groups_unready(), 1);
        let r = model.readiness();
        assert_eq!(r.status, FitStatus::Degenerate);
        assert!(
            r.reasons.iter().any(|s| s.contains("OPEN-ONLY")),
            "the refusing lane must be named: {:?}",
            r.reasons
        );
    }

    /// A refused group must not take the rest of the fit down with it, and must come
    /// back NULL-shaped rather than absent — the same answer `conjugate_anomaly` gives
    /// for an unfittable lane. An agent iterating over lanes then sees all of them.
    #[test]
    fn a_refused_lane_draws_null_without_poisoning_the_lanes_that_fitted() {
        let f = grouped_frame();
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, GROUPED_CFG);

        let sample = LaplaceEngine
            .sample(
                &*model,
                &SampleOptions {
                    n_chains: 1,
                    n_draws: 500,
                    seed: 13,
                    sample_from: crate::types::SampleFrom::Posterior,
                },
            )
            .unwrap();
        let p = model.param_names().len();
        assert_eq!(p, 9, "3 lanes x (intercept, beta, sigma)");

        let slot = |group: &str, name: &str| {
            model
                .param_names()
                .iter()
                .position(|q| q.group_id == group && q.name == name)
                .unwrap()
        };
        for row in sample.values.chunks(p) {
            for lane in ["HAM-ROT", "BRE-ANT"] {
                assert!(row[slot(lane, "intercept")].is_finite(), "{lane}");
                assert!(row[slot(lane, "sigma")] > 0.0, "{lane}");
            }
            assert!(row[slot("OPEN-ONLY", "intercept")].is_nan());
            assert!(row[slot("OPEN-ONLY", "beta[distance]")].is_nan());
            assert!(row[slot("OPEN-ONLY", "sigma")].is_nan());
        }
    }

    /// Groups are independent fits, so their draws must be independent too. If the
    /// blocks shared a random stream in a way that correlated them, a downstream
    /// comparison of two lanes would understate its own uncertainty.
    #[test]
    fn separate_lanes_are_fitted_independently() {
        let f = grouped_frame();
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, GROUPED_CFG);
        let g = model.as_gaussian().unwrap();
        assert_eq!(g.blocks().len(), 2, "the refused lane contributes no block");
        // Each block covers exactly its own lane's parameters, and the two are
        // disjoint.
        let a: std::collections::BTreeSet<usize> = g.blocks()[0].params.iter().copied().collect();
        let b: std::collections::BTreeSet<usize> = g.blocks()[1].params.iter().copied().collect();
        assert!(a.is_disjoint(&b));
        assert_eq!(a.len(), 3);
    }

    //=== configuration ===================================================//

    #[test]
    fn the_exact_engine_is_refused_rather_than_approximated() {
        let f = frame(200, 0.0, Some(40.0));
        let refs = f.key_refs();
        let view = f.view(&refs);
        let model = compile(&view, CFG);
        assert!(model.as_exact().is_none());
        assert!(!crate::engines::ExactEngine.supports(&*model));
        assert!(LaplaceEngine.supports(&*model));
    }

    #[test]
    fn a_misspelled_slot_names_the_typo() {
        let f = frame(50, 0.0, None);
        let refs = f.key_refs();
        let view = f.view(&refs);
        let err = CensoredAft
            .compile(
                &Config::parse(r#"{"time": "days", "event": "delivered", "dst": "weibull"}"#)
                    .unwrap(),
                &view,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("dst"), "{err}");
        assert!(err.contains("did you mean 'dist'"), "{err}");
    }

    #[test]
    fn an_unknown_distribution_is_an_error_rather_than_a_fallback() {
        let f = frame(50, 0.0, None);
        let refs = f.key_refs();
        let view = f.view(&refs);
        let err = CensoredAft
            .compile(
                &Config::parse(r#"{"time": "days", "event": "delivered", "dist": "gompertz"}"#)
                    .unwrap(),
                &view,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected one of weibull"), "{err}");
    }

    #[test]
    fn every_admitted_distribution_compiles_and_samples() {
        let f = frame(200, 0.0, Some(40.0));
        let refs = f.key_refs();
        let view = f.view(&refs);
        for dist in DISTRIBUTIONS {
            let cfg = format!(
                r#"{{"time": "days", "event": "delivered", "x": "distance", "dist": "{dist}"}}"#
            );
            let model = compile(&view, &cfg);
            assert_eq!(model.readiness().status, FitStatus::Converged, "{dist}");
            let sample = LaplaceEngine
                .sample(
                    &*model,
                    &SampleOptions {
                        n_chains: 1,
                        n_draws: 200,
                        seed: 21,
                        sample_from: crate::types::SampleFrom::Posterior,
                    },
                )
                .unwrap();
            assert!(sample.values.iter().all(|v| v.is_finite()), "{dist}");
        }
    }

    /// A prior narrows the posterior it is put on; the intercept is never penalised,
    /// because shrinking a duration *level* toward zero is a claim that everything
    /// happens instantly.
    #[test]
    fn a_prior_shrinks_the_slope_and_leaves_the_level_alone() {
        let f = frame(200, 0.0, None);
        let refs = f.key_refs();
        let view = f.view(&refs);
        let flat = compile(&view, CFG);
        let shrunk = compile(
            &view,
            r#"{"time": "days", "event": "delivered", "x": "distance",
                "prior": {"beta_scale": 0.02}}"#,
        );
        let mode = |m: &dyn CompiledModel| m.as_gaussian().unwrap().blocks()[0].mode.clone();
        let (a, b) = (mode(&*flat), mode(&*shrunk));
        assert!(
            b[1].abs() < 0.5 * a[1].abs(),
            "the prior must shrink the slope: {} vs {}",
            b[1],
            a[1]
        );
        // A prior adds information, so the slope's own curvature must rise sharply --
        // by roughly the prior precision 1/0.02^2 = 2500. Not asserted exactly: the
        // mode moves too, so the likelihood's contribution is evaluated at a different
        // point and the difference is only approximately the prior's. The exact claim
        // -- that the precision the prior contributes lands on the slope's diagonal
        // entry and nowhere else -- is made at a *fixed* point in
        // `bridge::tests::a_prior_adds_its_precision_to_exactly_one_diagonal_entry`,
        // where nothing else is free to move.
        let pa = &flat.as_gaussian().unwrap().blocks()[0].precision;
        let pb = &shrunk.as_gaussian().unwrap().blocks()[0].precision;
        assert!(
            pb[(1, 1)] > pa[(1, 1)] + 1000.0,
            "the prior must add its precision to the slope: {} vs {}",
            pb[(1, 1)],
            pa[(1, 1)]
        );
    }

    #[test]
    fn a_model_with_nothing_to_estimate_is_rejected() {
        let f = frame(50, 0.0, None);
        let refs = f.key_refs();
        let view = f.view(&refs);
        let err = CensoredAft.compile(
            &Config::parse(r#"{"time": "days", "event": "delivered", "intercept": 0}"#).unwrap(),
            &view,
        );
        assert!(err.is_err());
    }

    /// Nulls are filtered whole-row before the fingerprint is taken, and the bridge
    /// refuses if `fit_aft` then drops anything further — so `model_id` always
    /// describes exactly the rows the fit read.
    #[test]
    fn a_null_anywhere_in_a_row_removes_the_whole_row_before_the_fingerprint() {
        let (time, xs, event) = survival_fixture(
            AftDistribution::Weibull,
            1.0,
            0.25,
            0.4,
            60,
            0.0,
            Some(20.0),
        );
        let mut with_null: Vec<Option<f64>> = xs.iter().map(|v| Some(*v)).collect();
        with_null[7] = None;

        let clean = Frame::new(60)
            .numeric("days", time.clone())
            .numeric("distance", xs)
            .numeric("delivered", event.clone());
        let holed = Frame::new(60)
            .numeric("days", time)
            .numeric_with_nulls("distance", with_null)
            .numeric("delivered", event);

        let (rc, rh) = (clean.key_refs(), holed.key_refs());
        let (vc, vh) = (clean.view(&rc), holed.view(&rh));
        let mc = compile(&vc, CFG);
        let mh = compile(&vh, CFG);
        assert_eq!(mc.n_obs(), 60);
        assert_eq!(mh.n_obs(), 59);
        assert_ne!(mc.data_fingerprint(), mh.data_fingerprint());
    }
}
