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
use crate::errors::{BayesError, BayesResult};
use crate::types::{EngineKind, FitStatus};

/// Default ceiling on the in-memory draw buffer, in megabytes.
///
/// Two gigabytes is chosen to be comfortably above any fit the shipped families
/// realistically need — 5000 groups x 2 parameters x 4000 draws is ~320 MB — while
/// staying well below the point where a fit would evict a customer's other work. It
/// is a config slot, not a constant, because "realistically" is a claim about our
/// customers rather than a law.
pub const DEFAULT_MAX_DRAW_MEGABYTES: usize = 2048;

/// Reject a request whose draw buffer would exceed the budget, before allocating it.
fn check_output_budget(
    n_params: usize,
    n_chains: usize,
    n_draws: usize,
    max_megabytes: usize,
) -> BayesResult<()> {
    // Checked throughout: on a 32-bit target this product overflows long before it
    // exhausts memory, and a wrapped length would allocate a buffer far too small
    // and then write past it.
    let cells = n_chains
        .checked_mul(n_draws)
        .and_then(|c| c.checked_mul(n_params));
    let bytes = cells.and_then(|c| c.checked_mul(std::mem::size_of::<f64>()));

    let budget = max_megabytes.saturating_mul(1024 * 1024);
    let requested = match bytes {
        Some(b) if b <= budget => return Ok(()),
        Some(b) => b,
        None => usize::MAX,
    };

    Err(BayesError::config(
        "draws",
        format!(
            "this fit would need {} MB of draws ({n_params} parameters x {n_chains} chain(s) \
             x {n_draws} draws), above the {max_megabytes} MB limit. Reduce `draws`, fit fewer \
             groups at a time, or raise `max_draw_megabytes` if the memory is genuinely available",
            requested / (1024 * 1024)
        ),
    ))
}

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

    // Output size is budgeted *before* the model is compiled into draws. The draw
    // buffer is `chains * draws * params` f64s, and `params` grows with the number of
    // groups in the data, which no config slot bounds. Left unchecked, a request like
    // 100k groups x 10k draws asks for 16 GB and Rust's allocator aborts the process
    // -- taking the customer's whole DuckDB session with it, for what is really just a
    // request that was too big. A refusal that names the shape is recoverable; an
    // abort is not.
    let max_megabytes =
        cfg.usize_in("max_draw_megabytes", DEFAULT_MAX_DRAW_MEGABYTES, 1, 1 << 20)?;

    let model = family.compile(cfg, data)?;
    check_output_budget(model.param_names().len(), n_chains, n_draws, max_megabytes)?;
    let engine = engines::resolve(engine_kind)?;
    if !engine.supports(&*model) {
        return Err(BayesError::config(
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
            engine_kind,
            seed,
        ),
        family: family.code(),
        engine: engine_kind,
        // Provisional: replaced below once the draws have been graded.
        status: readiness.status,
        seed,
        n_obs: model.n_obs(),
        n_groups: model.n_groups(),
        // Structural only, and deliberately so: this counts the groups the *family*
        // refused from their sufficient statistics. Diagnostics are computed per
        // parameter rather than per group and can downgrade the fit below without
        // implicating any particular group, so folding them in would produce a count
        // that does not correspond to anything an agent can go and look at.
        n_groups_unready: model.n_groups_unready(),
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

    /// An auditor holding only the persisted table must be able to say which model
    /// was fitted. The value column is DOUBLE, so the family travels as its catalog
    /// F-number rather than its name.
    #[test]
    fn the_family_that_produced_the_table_travels_with_the_draws() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let code_of = |family: &str, cfg: &str| {
            fit(family, &Config::parse(cfg).unwrap(), &view)
                .unwrap()
                .posterior
                .rows()
                .find(|r| r.param == "__family__")
                .expect("the family must be emitted")
                .value
        };
        assert_eq!(
            code_of(
                "conjugate_anomaly",
                r#"{"value": "cost", "group": "lane", "draws": 500}"#
            ),
            7.0
        );
        assert_eq!(
            code_of("pooled_gaussian", r#"{"y": "cost", "draws": 500}"#),
            3.0
        );
    }

    /// `Readiness::worst` collapses per-group verdicts on purpose -- a fit an agent
    /// must inspect is not 99.4 % trustworthy. What the collapse destroys is the
    /// *scale* of the inspection, and that is what this row restores: the status
    /// still says `insufficient_data` for the whole fit, and the count says how many
    /// of the groups are actually the problem.
    #[test]
    fn the_number_of_unready_groups_survives_the_collapse_into_one_status() {
        let mut costs = Vec::new();
        let mut lanes = Vec::new();
        for lane in ["HAM-ROT", "BRE-ANT", "DUS-MIL"] {
            for i in 0..20 {
                costs.push(2.0 + ((i % 5) as f64 - 2.0) * 0.02);
                lanes.push(lane);
            }
        }
        // Two lanes with two invoices each: fittable, but below any sane threshold.
        for lane in ["THIN-1", "THIN-2"] {
            costs.extend([1.9, 2.1]);
            lanes.extend([lane, lane]);
        }
        let frame = Frame::new(64).numeric("cost", costs).key("lane", lanes);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg =
            Config::parse(r#"{"value": "cost", "group": "lane", "min_obs": 5, "draws": 2000}"#)
                .unwrap();
        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();

        let row = |name: &str| {
            fit.posterior
                .rows()
                .find(|r| r.param == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .value
        };
        assert_eq!(row("__n_groups__"), 5.0);
        assert_eq!(row("__n_groups_unready__"), 2.0);
        // The collapsed verdict is unchanged: three good lanes do not make the fit
        // safe to act on.
        assert_eq!(row("__status__"), FitStatus::InsufficientData as i32 as f64);
    }

    /// The count is only useful if a clean fit reports zero rather than nothing.
    #[test]
    fn a_healthy_fit_reports_no_unready_groups_rather_than_omitting_the_row() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 2000}"#).unwrap();
        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
        let row = fit
            .posterior
            .rows()
            .find(|r| r.param == "__n_groups_unready__")
            .expect("the count must be emitted even when it is zero");
        assert_eq!(row.value, 0.0);
        assert_eq!(fit.posterior.meta.status, FitStatus::Converged);
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

    /// A request too large to hold must be refused, not attempted. Rust's allocator
    /// aborts the process on failure, which would take the customer's whole DuckDB
    /// session down for what is only an over-ambitious query.
    #[test]
    fn an_oversized_request_is_refused_before_anything_is_allocated() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        // 3 lanes x 2 params x 1_000_000 draws x 8 bytes = 48 MB, under a 1 MB cap.
        let cfg = Config::parse(
            r#"{"value": "cost", "group": "lane", "draws": 1000000, "max_draw_megabytes": 1}"#,
        )
        .unwrap();
        let err = fit("conjugate_anomaly", &cfg, &view)
            .unwrap_err()
            .to_string();
        assert!(err.contains("MB of draws"), "{err}");
        assert!(err.contains("6 parameters"), "{err}");
        assert!(err.contains("max_draw_megabytes"), "{err}");

        // ...and a fit inside the budget is unaffected.
        let cfg = Config::parse(
            r#"{"value": "cost", "group": "lane", "draws": 2000, "max_draw_megabytes": 1}"#,
        )
        .unwrap();
        assert!(fit("conjugate_anomaly", &cfg, &view).is_ok());
    }

    #[test]
    fn the_output_budget_uses_checked_arithmetic() {
        // On any target, this product overflows usize. A wrapped length would
        // allocate a far-too-small buffer and then be written past.
        let err = check_output_budget(usize::MAX / 2, 64, 1_000_000, 2048).unwrap_err();
        assert!(err.to_string().contains("MB of draws"));
    }

    /// Two posteriors with different warranties must not share an identity, even when
    /// the caller never named an engine and the two configs are byte-identical.
    #[test]
    fn the_engine_that_actually_ran_changes_the_model_id() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let id_of = |cfg: &str| {
            fit("pooled_gaussian", &Config::parse(cfg).unwrap(), &view)
                .unwrap()
                .posterior
                .meta
                .model_id
        };
        assert_ne!(
            id_of(r#"{"y": "cost", "draws": 500, "engine": "exact"}"#),
            id_of(r#"{"y": "cost", "draws": 500, "engine": "laplace"}"#)
        );
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
