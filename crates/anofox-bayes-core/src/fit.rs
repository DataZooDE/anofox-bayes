//! Fitting: the one entry point the SQL surface calls.
//!
//! ```text
//!   family + config + data
//!        │
//!        ├─ lookup      the catalog is closed; an unknown family stops here
//!        ├─ compile     config validated, columns resolved, nulls filtered
//!        ├─ resolve     engine chosen (family default, or caller override)
//!        ├─ sample      draws produced
//!        └─ grade       structural readiness + diagnostics -> FitStatus
//! ```
//!
//! Everything before `sample` is cheap and total: a request that is going to fail
//! fails before any arithmetic runs, which is what makes a bad config a fast, precise
//! error rather than a slow, vague one.
//!
//! The grading step is where the two halves of the refusal path meet.
//! [`Readiness`](crate::catalog::Readiness) is what the family could tell from the
//! sufficient statistics alone — a lane with one invoice, a group whose observations
//! are all identical. The diagnostics are what only the draws can reveal — chains
//! that did not mix, an effective sample size too small to trust a tail quantile.
//! Either can veto, and the worse verdict wins, because a fit is only as trustworthy
//! as its weakest parameter.

use crate::catalog::{self, Readiness};
use crate::config::Config;
use crate::data::DataView;
use crate::diagnostics::{self, ParamDiagnostics, Thresholds};
use crate::draws::{derive_model_id, ModelMeta, Posterior};
use crate::engines::{self, SampleOptions};
use crate::errors::BayesResult;
use crate::types::{EngineKind, FitStatus};

/// A completed fit.
#[derive(Debug)]
pub struct Fit {
    pub posterior: Posterior,
    pub diagnostics: Vec<ParamDiagnostics>,
    /// Why the status is what it is. Empty for a clean fit.
    pub reasons: Vec<String>,
}

/// Fit a cataloged model to a relation.
pub fn fit(family_id: &str, cfg: &Config, data: &DataView) -> BayesResult<Fit> {
    let family = catalog::lookup(family_id)?;

    // Sampling budget and engine choice are common to every family, so they are
    // parsed here rather than repeated in each one. Families still declare the slots
    // in `config_slots` so that `reject_unknown` accepts them.
    let n_draws = cfg.usize_in("draws", SampleOptions::default().n_draws, 4, 1_000_000)?;
    // Defaults to one chain, and that is not an oversight. R-hat compares chains to
    // detect a Markov chain that has not mixed; both engines in 0.1 draw
    // *independently*, so there is nothing to fail to mix and a second chain would
    // buy an R-hat of 1.0 that means nothing. The slot exists because a caller may
    // want the cross-check anyway, and because NUTS in 0.2 will default it to 4.
    // Until then, the gate is ESS -- see docs/API_REFERENCE.md.
    let n_chains = cfg.usize_in("chains", 1, 1, 64)?;
    let seed = cfg.seed()?;
    let engine_kind = match cfg.opt_str("engine")? {
        Some(name) => EngineKind::parse(name)?,
        None => family.default_engine(),
    };

    let model = family.compile(cfg, data)?;
    let engine = engines::resolve(engine_kind)?;
    if !engine.supports(&*model) {
        return Err(crate::BayesError::config(
            "engine",
            format!(
                "the {} engine cannot serve family '{}'",
                engine_kind.as_str(),
                family.id()
            ),
        ));
    }

    let opts = SampleOptions {
        n_chains,
        n_draws,
        seed,
    };
    let sample = engine.sample(&*model, &opts)?;

    let readiness = model.readiness();
    let meta = ModelMeta {
        model_id: derive_model_id(
            family.id(),
            &cfg.canonical(),
            model.data_fingerprint(),
            seed,
        ),
        family: family.id().to_string(),
        engine: engine_kind,
        // Provisional: replaced below once the draws have been graded.
        status: readiness.status,
        seed,
        n_obs: model.n_obs(),
        n_groups: model.n_groups(),
    };

    let posterior = Posterior::new(
        meta,
        model.param_names().to_vec(),
        opts.n_chains,
        opts.n_draws,
        sample.values,
        sample.stats,
    )?;

    Ok(grade(posterior, readiness, &Thresholds::default()))
}

/// Combine the structural verdict with the sampled diagnostics into a final status.
fn grade(mut posterior: Posterior, readiness: Readiness, thresholds: &Thresholds) -> Fit {
    let diags = diagnostics::diagnose(&posterior);
    let mut reasons = readiness.reasons;

    let failing: Vec<&ParamDiagnostics> = diags.iter().filter(|d| !d.passes(thresholds)).collect();
    let sampling_status = if failing.is_empty() {
        FitStatus::Converged
    } else {
        for d in &failing {
            reasons.push(format!(
                "parameter '{}' of group '{}' failed diagnostics (rhat {}, ess_bulk {:.0}, ess_tail {:.0})",
                d.param,
                d.group_id,
                d.rhat.map(|r| format!("{r:.4}")).unwrap_or_else(|| "n/a".into()),
                d.ess_bulk,
                d.ess_tail
            ));
        }
        FitStatus::Degenerate
    };

    // Worse wins. A structurally insufficient fit whose draws happen to mix well is
    // still insufficient, and a structurally fine fit whose draws did not mix is
    // still not safe to act on.
    let status = if (readiness.status as i32) >= (sampling_status as i32) {
        readiness.status
    } else {
        sampling_status
    };
    posterior.meta.status = status;

    Fit {
        posterior,
        diagnostics: diags,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::testing::Frame;
    use crate::draws::{META_INDEX, META_STATUS};

    fn freight_frame() -> Frame {
        // Two clean lanes and one where costs jumped.
        let mut costs = Vec::new();
        let mut lanes = Vec::new();
        for i in 0..40 {
            costs.push(2.00 + ((i % 5) as f64 - 2.0) * 0.02);
            lanes.push("HAM-ROT");
        }
        for i in 0..40 {
            costs.push(3.00 + ((i % 7) as f64 - 3.0) * 0.03);
            lanes.push("BRE-ANT");
        }
        for i in 0..40 {
            // An accessorial surcharge that appears halfway through the window --
            // the shape a freight audit is actually looking for.
            costs.push(2.00 + ((i % 5) as f64 - 2.0) * 0.02 + if i >= 20 { 1.20 } else { 0.0 });
            lanes.push("DUS-MIL");
        }
        Frame::new(120).numeric("cost", costs).key("lane", lanes)
    }

    #[test]
    fn a_healthy_fit_converges_and_carries_no_reasons() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 2000}"#).unwrap();

        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(fit.posterior.meta.status, FitStatus::Converged);
        assert!(fit.reasons.is_empty(), "{:?}", fit.reasons);
        assert_eq!(fit.posterior.meta.n_obs, 120);
        assert_eq!(fit.posterior.meta.n_groups, 3);
        assert_eq!(fit.posterior.n_draws, 2000);
        // 3 lanes x (mu, sigma)
        assert_eq!(fit.posterior.n_params(), 6);
        assert_eq!(fit.diagnostics.len(), 6);
    }

    #[test]
    fn the_status_reaches_sql_inside_the_draws_table() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 2000}"#).unwrap();
        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();

        let status_row = fit
            .posterior
            .rows()
            .find(|r| r.param == META_STATUS)
            .expect("status must be emitted");
        assert_eq!(status_row.chain, META_INDEX);
        assert_eq!(status_row.value, FitStatus::Converged as i32 as f64);
    }

    /// The realistic shape of the freight-audit question: which lane's cost level
    /// sits above the fleet baseline, and with what posterior confidence? Answered
    /// from the draws, not from a threshold inside the model.
    #[test]
    fn a_shifted_lane_is_detectable_from_the_draws_and_a_clean_lane_is_not() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 4000, "seed": 7}"#)
            .unwrap();
        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();

        // P(mu > 2.30) per lane -- the tail probability an agent would compute in SQL.
        let exceed = |group: &str| {
            let idx = fit
                .posterior
                .params
                .iter()
                .position(|p| p.group_id == group && p.name == "mu")
                .unwrap();
            let draws: Vec<f64> = fit.posterior.chain_values(0, idx).collect();
            draws.iter().filter(|v| **v > 2.30).count() as f64 / draws.len() as f64
        };

        // The lane whose costs jumped mid-window sits above the threshold...
        assert!(
            exceed("DUS-MIL") > 0.99,
            "shifted lane P = {}",
            exceed("DUS-MIL")
        );
        // ...the stable cheap lane does not...
        assert!(
            exceed("HAM-ROT") < 0.01,
            "clean lane P = {}",
            exceed("HAM-ROT")
        );
        // ...and the stable expensive lane is above it for a legitimate reason, which
        // is exactly why the model reports a level per lane rather than one baseline.
        assert!(exceed("BRE-ANT") > 0.99);
    }

    #[test]
    fn a_structurally_insufficient_group_downgrades_the_whole_fit() {
        let frame = Frame::new(5)
            .numeric("cost", vec![1.0, 1.1, 0.9, 1.05, 42.0])
            .key("lane", vec!["A", "A", "A", "A", "SOLO"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane"}"#).unwrap();

        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(fit.posterior.meta.status, FitStatus::InsufficientData);
        assert!(!fit.posterior.meta.status.is_actionable());
        assert!(
            fit.reasons.iter().any(|r| r.contains("SOLO")),
            "{:?}",
            fit.reasons
        );
    }

    /// Tail ESS is the binding constraint, not bulk. Independent draws are worth
    /// roughly their own count for the posterior *mean*, but materially less for the
    /// 5 % and 95 % *quantiles* -- and a service-level, safety-stock or audit decision
    /// reads a quantile. A gate on bulk ESS alone would approve budgets that cannot
    /// support the number actually being used.
    #[test]
    fn the_tail_is_the_binding_constraint_on_the_sampling_budget() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let tail_deficit = |draws: usize| {
            let cfg = Config::parse(&format!(
                r#"{{"value": "cost", "group": "lane", "draws": {draws}}}"#
            ))
            .unwrap();
            let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
            let d = &fit.diagnostics[0];
            (d.ess_bulk, d.ess_tail)
        };

        // At a modest budget the tail is worth appreciably less than the bulk...
        let (bulk, tail) = tail_deficit(300);
        assert!(tail < bulk, "tail {tail} should trail bulk {bulk}");

        // ...and at that budget the gate rejects the fit, even though the bulk alone
        // would have passed it.
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 300}"#).unwrap();
        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(fit.posterior.meta.status, FitStatus::Degenerate);
        assert!(
            fit.reasons.iter().any(|r| r.contains("ess_tail")),
            "{:?}",
            fit.reasons
        );
    }

    /// Same inputs, same id and same numbers -- so a cache hit is a comparison, and
    /// an auditor can reproduce a recommendation from the inputs alone.
    #[test]
    fn an_identical_request_reproduces_the_model_id_and_the_draws() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "seed": 11}"#).unwrap();

        let a = fit("conjugate_anomaly", &cfg, &view).unwrap();
        let b = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(a.posterior.meta.model_id, b.posterior.meta.model_id);
        let (ra, rb): (Vec<_>, Vec<_>) =
            (a.posterior.rows().collect(), b.posterior.rows().collect());
        assert_eq!(ra, rb);
    }

    #[test]
    fn a_different_seed_changes_the_model_id_and_the_draws() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let a = fit(
            "conjugate_anomaly",
            &Config::parse(r#"{"value": "cost", "group": "lane", "seed": 1}"#).unwrap(),
            &view,
        )
        .unwrap();
        let b = fit(
            "conjugate_anomaly",
            &Config::parse(r#"{"value": "cost", "group": "lane", "seed": 2}"#).unwrap(),
            &view,
        )
        .unwrap();
        assert_ne!(a.posterior.meta.model_id, b.posterior.meta.model_id);
    }

    /// R-hat is structurally absent under one chain, which is the default. Asking
    /// for more chains makes it available and it must come out near 1 -- these
    /// engines draw independently, so the chains genuinely are exchangeable. A value
    /// far from 1 here would mean the per-chain streams were not independent, which
    /// is exactly the seed-derivation bug `chains_are_independent_streams` guards
    /// against from the other side.
    #[test]
    fn asking_for_more_chains_makes_rhat_available_and_it_comes_out_near_one() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let single = fit(
            "conjugate_anomaly",
            &Config::parse(r#"{"value": "cost", "group": "lane", "draws": 2000}"#).unwrap(),
            &view,
        )
        .unwrap();
        assert!(
            single.diagnostics.iter().all(|d| d.rhat.is_none()),
            "one chain cannot support an R-hat"
        );

        let multi = fit(
            "conjugate_anomaly",
            &Config::parse(r#"{"value": "cost", "group": "lane", "draws": 2000, "chains": 4}"#)
                .unwrap(),
            &view,
        )
        .unwrap();
        assert_eq!(multi.posterior.n_chains, 4);
        for d in &multi.diagnostics {
            let r = d.rhat.expect("four chains support an R-hat");
            assert!(r < 1.01, "{}/{}: rhat {r}", d.group_id, d.param);
        }
        assert_eq!(multi.posterior.meta.status, FitStatus::Converged);
    }

    #[test]
    fn an_unknown_family_fails_before_any_work_is_done() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost"}"#).unwrap();
        assert!(fit("gaussian_process", &cfg, &view).is_err());
    }

    #[test]
    fn an_engine_that_cannot_serve_the_family_is_an_error() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "engine": "nuts"}"#).unwrap();
        let err = fit("conjugate_anomaly", &cfg, &view).unwrap_err();
        assert!(matches!(err, crate::BayesError::Config { ref slot, .. } if slot == "engine"));
    }

    #[test]
    fn the_draw_count_is_bounded_and_the_bound_is_reported() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "draws": 2}"#).unwrap();
        let err = fit("conjugate_anomaly", &cfg, &view)
            .unwrap_err()
            .to_string();
        assert!(err.contains("between 4 and"), "{err}");
    }

    /// Too few draws to support a tail quantile: the fit runs, the numbers are real,
    /// and the status says do not act on them.
    #[test]
    fn a_sampling_budget_too_small_for_the_diagnostics_is_reported_as_degenerate() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 40}"#).unwrap();

        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(fit.posterior.meta.status, FitStatus::Degenerate);
        assert!(
            fit.reasons.iter().any(|r| r.contains("ess_bulk")),
            "{:?}",
            fit.reasons
        );
        // The draws themselves are perfectly finite -- it is the evidence they carry
        // that is too thin, and the status is the only thing that says so.
        assert!(fit
            .posterior
            .rows()
            .filter(|r| r.draw >= 0)
            .all(|r| r.value.is_finite()));
    }
}
