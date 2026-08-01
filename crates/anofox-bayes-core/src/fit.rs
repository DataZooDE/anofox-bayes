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
use crate::types::{EngineKind, FitStatus, SampleFrom};

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
    let seed = cfg.seed()?;
    // Posterior unless asked otherwise. Parsed here rather than per family because
    // the slot means the same thing everywhere; families still declare it so that
    // `reject_unknown` accepts it.
    let sample_from = match cfg.opt_str("sample_from")? {
        Some(name) => SampleFrom::parse(name)?,
        None => SampleFrom::Posterior,
    };
    let engine_kind = match cfg.opt_str("engine")? {
        Some(name) => EngineKind::parse(name)?,
        None => family.default_engine(),
    };

    // The chain default depends on the engine, and has to.
    //
    // R-hat compares chains to detect a Markov chain that has not mixed. The exact and
    // Laplace engines draw *independently*, so there is nothing to fail to mix and a
    // second chain would buy an R-hat of 1.0 that means nothing; their gate is ESS.
    // NUTS produces a genuine Markov chain, and one chain of it cannot support the
    // single diagnostic that would reveal it had not converged -- so a one-chain NUTS
    // default would ship the fit whose failure mode R-hat exists to catch with R-hat
    // switched off. Four is the Stan and PyMC default and for the same reason: enough
    // chains for the between-chain variance to be estimated at all.
    let default_chains = match engine_kind {
        EngineKind::Nuts => 4,
        EngineKind::Exact | EngineKind::Laplace => 1,
    };
    let n_chains = cfg.usize_in("chains", default_chains, 1, 64)?;
    // Adaptation draws, discarded before the output. Bounded below by 1 rather than 0:
    // an unadapted NUTS run with a default step size is not a sampler whose output
    // means anything, and letting a caller ask for one would produce draws carrying
    // the same `converged` warranty as an adapted fit.
    let n_warmup = cfg.usize_in("warmup", engines::DEFAULT_WARMUP, 1, 1_000_000)?;

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

    if sample_from == SampleFrom::Prior && !engine.can_sample_prior() {
        return Err(BayesError::config(
            "sample_from",
            format!(
                "the {} engine cannot draw from a prior; a prior-predictive check \
                 needs the exact engine, which is available for conjugate families",
                engine_kind.as_str()
            ),
        ));
    }

    let opts = SampleOptions {
        n_chains,
        n_draws,
        n_warmup,
        seed,
        sample_from,
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
        sample_from,
    };

    let posterior = Posterior::with_unready(
        meta,
        model.param_names().to_vec(),
        opts.n_chains,
        opts.n_draws,
        sample.values,
        sample.stats,
        model.unready_groups(),
    )?;

    Ok(grade(posterior, readiness, &Thresholds::default()))
}

/// Combine the structural verdict with the sampled diagnostics into a final status.
fn grade(mut posterior: Posterior, readiness: Readiness, thresholds: &Thresholds) -> Fit {
    let diags = diagnostics::diagnose(&posterior);
    let mut reasons = readiness.reasons;

    let failing: Vec<&ParamDiagnostics> = diags.iter().filter(|d| !d.passes(thresholds)).collect();
    let mut sampling_status = FitStatus::Converged;
    if !failing.is_empty() {
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
        sampling_status = FitStatus::Degenerate;
    }

    // Divergences, which are a property of the *fit* rather than of any one parameter
    // and so cannot live in `ParamDiagnostics::passes`.
    //
    // A divergent trajectory is the sampler reporting that it left the region it was
    // integrating over -- the draws that follow it are not from the posterior, they
    // are from wherever the integrator ended up. THEORY §7 is unambiguous about what
    // that means here: there is a number, and it must not drive a decision. So a
    // single divergence downgrades the fit, in line with `Thresholds::max_divergent`
    // defaulting to zero rather than to a small budget, and by the same worst-wins
    // doctrine that `Readiness::worst` applies to structural verdicts.
    //
    // `None` -- an engine that reported no divergence statistic at all -- is not a
    // pass. It is silence, and it leaves this branch untouched, exactly as the
    // omitted `__divergent__` row leaves a SQL consumer's `sum()` NULL.
    if let Some(divergent) = posterior.n_divergent() {
        if divergent as f64 > thresholds.max_divergent {
            reasons.push(format!(
                "the sampler reported {divergent} divergent transition(s) out of {} kept draws \
                 (tolerance {}); the draws after a divergence are not from the posterior. \
                 Raise `warmup`, or reparameterise",
                posterior.n_chains * posterior.n_draws,
                thresholds.max_divergent
            ));
            sampling_status = FitStatus::Degenerate;
        }
    }

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

    /// R̂ stops being decorative the moment a Markov chain is involved. NUTS therefore
    /// defaults to four chains, where the exact and Laplace engines default to one:
    /// a single NUTS chain cannot support the one diagnostic that would reveal it had
    /// not converged.
    #[test]
    fn a_nuts_fit_defaults_to_four_chains_and_produces_a_real_rhat() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"y": "cost", "engine": "nuts", "draws": 1000}"#).unwrap();
        let fit = fit("pooled_gaussian", &cfg, &view).unwrap();

        assert_eq!(fit.posterior.n_chains, 4);
        assert_eq!(
            fit.posterior
                .rows()
                .find(|r| r.param == "__n_chains__")
                .unwrap()
                .value,
            4.0
        );
        for d in &fit.diagnostics {
            let r = d.rhat.expect("four NUTS chains support an R-hat");
            assert!(r < 1.01, "{}/{}: rhat {r}", d.group_id, d.param);
        }
        assert_eq!(fit.posterior.meta.status, FitStatus::Converged);
        assert!(fit.reasons.is_empty(), "{:?}", fit.reasons);
    }

    /// The four reserved sample statistics reach SQL, which no engine had ever made
    /// them do before. `__divergent__` summing to zero here is the honest "the sampler
    /// explored cleanly" that the contract reserves it for.
    #[test]
    fn a_nuts_fit_puts_the_sampler_statistics_on_the_draws_table() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        // Intercept and residual scale only. The grouped design is deliberately not
        // used here: see
        // `a_posterior_nuts_explores_poorly_is_refused_rather_than_shipped`, which is
        // about that geometry rather than about the statistic rows.
        let cfg = Config::parse(
            r#"{"y": "cost", "engine": "nuts", "draws": 500, "chains": 2, "warmup": 500}"#,
        )
        .unwrap();
        let fit = fit("pooled_gaussian", &cfg, &view).unwrap();

        for stat in [
            crate::draws::PARAM_LP,
            crate::draws::PARAM_DIVERGENT,
            crate::draws::PARAM_ENERGY,
            crate::draws::PARAM_STEP_SIZE,
        ] {
            let rows: Vec<f64> = fit
                .posterior
                .rows()
                .filter(|r| r.param == stat)
                .map(|r| r.value)
                .collect();
            assert_eq!(rows.len(), 2 * 500, "{stat} must appear once per kept draw");
            assert!(
                rows.iter().all(|v| v.is_finite()),
                "{stat} carries non-finite values"
            );
        }
        assert_eq!(fit.posterior.n_divergent(), Some(0));
        assert_eq!(fit.posterior.meta.status, FitStatus::Converged);
    }

    /// **The diagnostics earning their keep on a real posterior.**
    ///
    /// `pooled_gaussian` with a `group` column puts an unpenalised intercept beside
    /// per-group effects that carry a `pool_scale`-wide prior. Only their *sum* is
    /// sharply identified by the data, so the posterior is a long, thin ridge — and a
    /// diagonal mass matrix, which is what this engine adapts, cannot precondition a
    /// ridge that is not axis-aligned. NUTS therefore mixes slowly here, and at a
    /// modest budget it says so: R̂ above 1.01 and a bulk ESS in the low hundreds.
    ///
    /// The point of the test is that this arrives as a **refusal** rather than as
    /// draws. The exact engine samples the same ridge perfectly well because it
    /// factorises the joint; a caller who switches engine and keeps the budget gets
    /// numbers that look the same in SQL and are worth much less, and `__status__` is
    /// the only thing standing between that and a decision. It is also the first time
    /// in this crate that R̂ is a load-bearing diagnostic rather than a structural
    /// `NULL`.
    #[test]
    fn a_posterior_nuts_explores_poorly_is_refused_rather_than_shipped() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(
            r#"{"y": "cost", "group": "lane", "engine": "nuts", "draws": 500, "chains": 2, "warmup": 500}"#,
        )
        .unwrap();
        let sampled = fit("pooled_gaussian", &cfg, &view).unwrap();

        assert_eq!(sampled.posterior.meta.status, FitStatus::Degenerate);
        assert!(!sampled.posterior.meta.status.is_actionable());
        assert!(
            sampled.reasons.iter().any(|r| r.contains("ess_bulk")),
            "{:?}",
            sampled.reasons
        );
        // Not a divergence problem: the trajectories are fine, they are just slow.
        assert_eq!(sampled.posterior.n_divergent(), Some(0));
        assert!(sampled.reasons.iter().all(|r| !r.contains("divergent")));

        // The same design under the exact engine is fine, which is what makes this a
        // statement about the sampler rather than about the data.
        let exact = fit(
            "pooled_gaussian",
            &Config::parse(r#"{"y": "cost", "group": "lane", "draws": 1000}"#).unwrap(),
            &view,
        )
        .unwrap();
        assert_eq!(exact.posterior.meta.status, FitStatus::Converged);
    }

    /// **The refusal doctrine applied to divergences.** THEORY §7: a fit whose numbers
    /// are not from the posterior must not drive a decision, and the only thing that
    /// says so is the status. Graded here on a synthetic posterior rather than by
    /// hunting for a model that diverges, so the rule is pinned exactly and cannot
    /// stop being tested when the sampler improves.
    #[test]
    fn a_single_divergence_downgrades_the_fit_and_says_why() {
        let params = vec![crate::draws::ParamName::global("mu").unwrap()];
        let meta = crate::draws::ModelMeta {
            model_id: "d".into(),
            family: crate::types::FamilyCode::PooledGaussian,
            engine: EngineKind::Nuts,
            status: FitStatus::Converged,
            seed: 1,
            n_obs: 100,
            n_groups: 1,
            n_groups_unready: 0,
            sample_from: crate::types::SampleFrom::Posterior,
        };
        // 2 chains x 1000 draws of an independent standard normal: every ESS and R-hat
        // gate passes comfortably, so the only thing that can fail is the divergence.
        let mut rng = crate::rng::BayesRng::for_chain(77, 0);
        let values: Vec<f64> = (0..2000).map(|_| rng.standard_normal()).collect();

        let graded = |n_divergent: usize| {
            let stats: Vec<crate::draws::SampleStats> = (0..2000)
                .map(|i| crate::draws::SampleStats {
                    lp: Some(-1.0),
                    divergent: Some(if i < n_divergent { 1.0 } else { 0.0 }),
                    energy: Some(2.0),
                    step_size: Some(0.5),
                })
                .collect();
            let post = Posterior::new(meta.clone(), params.clone(), 2, 1000, values.clone(), stats)
                .unwrap();
            grade(post, Readiness::ready(), &Thresholds::default())
        };

        let clean = graded(0);
        assert_eq!(clean.posterior.meta.status, FitStatus::Converged);
        assert!(clean.reasons.is_empty(), "{:?}", clean.reasons);

        // One divergence in two thousand draws is 0.05 % and is still a refusal: the
        // default tolerance is zero, not a small budget.
        let dirty = graded(1);
        assert_eq!(dirty.posterior.meta.status, FitStatus::Degenerate);
        assert!(!dirty.posterior.meta.status.is_actionable());
        assert!(
            dirty.reasons.iter().any(|r| r.contains("divergent")),
            "{:?}",
            dirty.reasons
        );
    }

    /// An engine that reports no divergence statistic must not be graded as if it had
    /// reported none — silence is not a pass.
    #[test]
    fn an_engine_that_reports_no_divergences_is_not_credited_with_zero() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 2000}"#).unwrap();
        let fit = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(fit.posterior.n_divergent(), None);
        assert_eq!(fit.posterior.meta.status, FitStatus::Converged);
        assert!(fit.reasons.iter().all(|r| !r.contains("divergent")));
    }

    /// Warmup is a documented slot with a documented default, and asking for none is a
    /// config error rather than an unadapted sampler wearing a `converged` badge.
    #[test]
    fn the_warmup_budget_is_configurable_and_bounded_below() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);

        let err = fit(
            "pooled_gaussian",
            &Config::parse(r#"{"y": "cost", "group": "lane", "engine": "nuts", "warmup": 0}"#)
                .unwrap(),
            &view,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("warmup"), "{err}");

        // ...and a legitimate budget is honoured without leaking into the output.
        let fit = fit(
            "pooled_gaussian",
            &Config::parse(
                r#"{"y": "cost", "group": "lane", "engine": "nuts", "warmup": 300, "draws": 400, "chains": 2}"#,
            )
            .unwrap(),
            &view,
        )
        .unwrap();
        assert_eq!(fit.posterior.n_draws, 400);
        assert_eq!(fit.posterior.n_chains, 2);
    }

    /// Warmup is part of the request, so two fits differing only in it are different
    /// questions and must not share an identity. `warmup` reaches `model_id` through
    /// the canonical config string, like every other slot.
    #[test]
    fn the_warmup_budget_is_part_of_the_model_identity() {
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
            id_of(
                r#"{"y": "cost", "group": "lane", "engine": "nuts", "draws": 400, "warmup": 200}"#
            ),
            id_of(
                r#"{"y": "cost", "group": "lane", "engine": "nuts", "draws": 400, "warmup": 300}"#
            )
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
        // `exact` on a family with no closed form. This used to be spelled `nuts` on
        // `conjugate_anomaly`, which stopped being a refusal when that family gained a
        // differentiable path (roadmap gap 10) -- every engine now serves it. The
        // property under test is unchanged: an engine that cannot serve a family says
        // so, rather than quietly substituting one that can.
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "engine": "exact"}"#).unwrap();
        let err = fit("censored_aft", &cfg, &view).unwrap_err();
        assert!(
            matches!(err, crate::BayesError::Config { .. }),
            "expected a config error, got {err}"
        );
    }

    /// The composition the change above records: `conjugate_anomaly` is now served by
    /// all three engines, and asking for any of them yields a fit rather than a
    /// refusal. Three independent routes to one posterior that is known in closed
    /// form.
    #[test]
    fn every_engine_serves_the_conjugate_family_since_it_gained_a_gradient() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        for engine in ["exact", "laplace", "nuts"] {
            let cfg = Config::parse(&format!(
                r#"{{"value": "cost", "group": "lane", "draws": 200, "engine": "{engine}"}}"#
            ))
            .unwrap();
            assert!(
                fit("conjugate_anomaly", &cfg, &view).is_ok(),
                "the {engine} engine should serve conjugate_anomaly"
            );
        }
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
    /// A prior-predictive request that the engine cannot honour must be refused.
    ///
    /// The dangerous case, and the reason this is checked centrally rather than per
    /// family: only the exact engine knows how to draw from a prior. A Laplace or
    /// NUTS fit given `sample_from: 'prior'` would otherwise run normally and return
    /// the *posterior*, correctly shaped, correctly graded, with a `__sample_from__`
    /// row claiming it is the prior. A pre-fit gate that silently agrees with the
    /// data it is meant to be checked against is worse than no gate at all.
    #[test]
    fn a_prior_predictive_is_refused_by_an_engine_that_cannot_draw_from_a_prior() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(
            r#"{"value": "cost", "group": "lane", "engine": "laplace",
                "sample_from": "prior",
                "prior": {"mu0": 2.0, "kappa0": 2.0, "alpha0": 3.0, "beta0": 2.0}}"#,
        )
        .unwrap();

        let err = fit("conjugate_anomaly", &cfg, &view).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("laplace") && msg.contains("prior"),
            "the refusal must name the engine and the request: {msg}"
        );
    }

    /// ...and the same request on the exact engine goes through, so the test above is
    /// about the engine rather than about the slot being rejected everywhere.
    #[test]
    fn the_exact_engine_serves_the_same_prior_predictive_request() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(
            r#"{"value": "cost", "group": "lane", "sample_from": "prior",
                "prior": {"mu0": 2.0, "kappa0": 2.0, "alpha0": 3.0, "beta0": 2.0}}"#,
        )
        .unwrap();

        let f = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(f.posterior.meta.sample_from, SampleFrom::Prior);
    }
    /// Roadmap gap 11: an agent must be able to find the bad lanes, not just count
    /// them.
    ///
    /// `Readiness::worst` collapses per-group verdicts into one and still does — a fit
    /// with three unidentifiable lanes out of 5 000 is not 99.4 % trustworthy. But a
    /// collapsed verdict plus a count sends an agent looking through 5 000 groups for
    /// three. These rows name them.
    #[test]
    fn the_refused_groups_are_named_and_not_merely_counted() {
        let frame = Frame::new(9)
            .numeric("cost", vec![1.0, 1.1, 0.9, 5.0, 5.2, 4.8, 42.0, 7.0, 7.0])
            .key(
                "lane",
                vec!["A", "A", "A", "B", "B", "B", "SOLO", "FLAT", "FLAT"],
            );
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 100}"#).unwrap();
        let f = fit("conjugate_anomaly", &cfg, &view).unwrap();

        let named: Vec<(&str, f64)> = f
            .posterior
            .rows()
            .filter(|r| r.param == crate::draws::META_GROUP_STATUS)
            .map(|r| (r.group_id, r.value))
            .collect();

        let keys: Vec<&str> = named.iter().map(|(g, _)| *g).collect();
        assert!(
            keys.contains(&"SOLO"),
            "a one-observation lane must be named: {keys:?}"
        );
        assert!(
            keys.contains(&"FLAT"),
            "a zero-variance lane must be named: {keys:?}"
        );
        assert!(
            !keys.contains(&"A"),
            "a healthy lane must not be named: {keys:?}"
        );
        assert!(
            !keys.contains(&"B"),
            "a healthy lane must not be named: {keys:?}"
        );

        // The count and the names must agree -- two ways of saying the same thing that
        // could drift apart is worse than one.
        assert_eq!(named.len(), f.posterior.meta.n_groups_unready);

        // Every named status is a refusal, never `converged`.
        for (g, v) in &named {
            assert_ne!(*v, 0.0, "group {g} was named but reports converged");
        }

        // ...and the model-level verdict is still the collapsed worst case.
        assert_ne!(f.posterior.meta.status, FitStatus::Converged);
    }

    /// A clean fit names nothing, rather than emitting a row per group saying "fine".
    #[test]
    fn a_healthy_fit_names_no_groups() {
        let frame = freight_frame();
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 100}"#).unwrap();
        let f = fit("conjugate_anomaly", &cfg, &view).unwrap();
        assert_eq!(
            f.posterior
                .rows()
                .filter(|r| r.param == crate::draws::META_GROUP_STATUS)
                .count(),
            0
        );
    }

    /// The rows must not disturb the draws they precede: `row_at` is O(1) random
    /// access that subtracts the header length, so a variable number of header rows is
    /// exactly where an off-by-one would land -- and it would silently misattribute
    /// every parameter value.
    #[test]
    fn naming_refused_groups_does_not_shift_the_draw_rows() {
        let frame = Frame::new(7)
            .numeric("cost", vec![1.0, 1.1, 0.9, 5.0, 5.2, 4.8, 42.0])
            .key("lane", vec!["A", "A", "A", "B", "B", "B", "SOLO"]);
        let refs = frame.key_refs();
        let view = frame.view(&refs);
        let cfg = Config::parse(r#"{"value": "cost", "group": "lane", "draws": 50}"#).unwrap();
        let f = fit("conjugate_anomaly", &cfg, &view).unwrap();

        // Streaming and random access must agree row for row. Compared bitwise
        // because a refused group's draws are NaN, and NaN != NaN would make this pass
        // or fail for reasons that have nothing to do with the indexing under test.
        let streamed: Vec<_> = f.posterior.rows().collect();
        assert_eq!(streamed.len(), f.posterior.n_rows());
        for (i, want) in streamed.iter().enumerate() {
            let got = f.posterior.row_at(i).unwrap();
            assert_eq!(
                (
                    got.group_id,
                    got.chain,
                    got.draw,
                    got.param,
                    got.value.to_bits()
                ),
                (
                    want.group_id,
                    want.chain,
                    want.draw,
                    want.param,
                    want.value.to_bits()
                ),
                "row {i}"
            );
        }

        // Lane A's mu draws are finite and its own; SOLO's are NULL-shaped.
        let mu_a: Vec<f64> = streamed
            .iter()
            .filter(|r| r.param == "mu" && r.group_id == "A" && r.draw >= 0)
            .map(|r| r.value)
            .collect();
        assert_eq!(mu_a.len(), 50);
        assert!(mu_a.iter().all(|v| v.is_finite() && (0.0..3.0).contains(v)));
    }
}
