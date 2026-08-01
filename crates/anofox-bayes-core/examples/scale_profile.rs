//! Where the wall time of a `conjugate_anomaly` fit actually goes.
//!
//! `validation/bench.py` measures the whole SQL statement; this measures the three
//! phases inside the crate — compile (per-group sufficient statistics), sample
//! (draws), and row rendering — so an optimisation can be aimed at the phase that
//! costs something rather than at the one that is easiest to change.
//!
//! ```bash
//! cargo run --release --example scale_profile -- 5000 104 1000
//! ```

use std::time::Instant;

use anofox_bayes_core::catalog;
use anofox_bayes_core::config::Config;
use anofox_bayes_core::data::{DataView, KeyColumn, NumericColumn};
use anofox_bayes_core::engines::{self, SampleOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let groups: usize = args.get(1).map_or(5000, |s| s.parse().unwrap());
    let periods: usize = args.get(2).map_or(104, |s| s.parse().unwrap());
    let draws: usize = args.get(3).map_or(1000, |s| s.parse().unwrap());

    let n = groups * periods;
    let mut values = Vec::with_capacity(n);
    let mut keys: Vec<String> = Vec::with_capacity(n);
    for g in 1..=groups {
        for p in 1..=periods {
            values.push(100.0 + g as f64 * 0.1 + (((g * 7 + p * 3) % 11) as f64 - 5.0) * 0.5);
            keys.push(format!("G{g}"));
        }
    }
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    let valid = vec![true; n];

    let mut view = DataView::new(n);
    view.add_numeric(
        "v",
        NumericColumn {
            values: &values,
            valid: &valid,
        },
    )
    .unwrap();
    view.add_key(
        "grp",
        KeyColumn {
            values: &key_refs,
            valid: &valid,
        },
    )
    .unwrap();

    // Compile, broken down. The three steps below are the whole of it apart from the
    // per-group conjugate updates, so what they do not account for is what the fits
    // themselves cost -- which is how the decision to leave the fits serial was made.
    let t = Instant::now();
    let rows = view.usable_rows(&["v"], &["grp"]).unwrap();
    let usable_rows = t.elapsed();
    let t = Instant::now();
    view.fingerprint(&["v"], &["grp"], &rows).unwrap();
    let fingerprint = t.elapsed();
    let t = Instant::now();
    view.group_rows(Some("grp"), &rows).unwrap();
    let group_rows = t.elapsed();

    let cfg = Config::parse(r#"{"value":"v","group":"grp","seed":1}"#).unwrap();
    let family = catalog::lookup("conjugate_anomaly").unwrap();

    let t = Instant::now();
    let model = family.compile(&cfg, &view).unwrap();
    let compile = t.elapsed();

    let opts = SampleOptions {
        n_chains: 1,
        n_draws: draws,
        seed: 1,
        sample_from: anofox_bayes_core::types::SampleFrom::Posterior,
    };
    let engine = engines::resolve(anofox_bayes_core::types::EngineKind::Exact).unwrap();
    let t = Instant::now();
    let sample = engine.sample(&*model, &opts).unwrap();
    let sampling = t.elapsed();

    let params = model.param_names().to_vec();
    let n_params = params.len();
    let posterior = anofox_bayes_core::draws::Posterior::new(
        anofox_bayes_core::draws::ModelMeta {
            model_id: "profile".to_string(),
            family: family.code(),
            engine: anofox_bayes_core::types::EngineKind::Exact,
            status: anofox_bayes_core::types::FitStatus::Converged,
            seed: 1,
            n_obs: model.n_obs(),
            n_groups: model.n_groups(),
            n_groups_unready: 0,
            sample_from: anofox_bayes_core::types::SampleFrom::Posterior,
        },
        params,
        1,
        draws,
        sample.values,
        sample.stats,
    )
    .unwrap();

    let t = Instant::now();
    let mut acc = 0.0f64;
    for row in posterior.rows() {
        acc += row.value;
    }
    let render = t.elapsed();

    let t = Instant::now();
    let diags = anofox_bayes_core::diagnostics::diagnose(&posterior);
    let diagnostics = t.elapsed();

    println!(
        "groups={groups} rows={n} draws={draws} params={n_params}\n  \
         compile     {compile:>10.3?}   (usable_rows {usable_rows:.3?}, \
         fingerprint {fingerprint:.3?}, group_rows {group_rows:.3?})\n  \
         sample      {sampling:>10.3?}\n  \
         render      {render:>10.3?}\n  \
         diagnostics {diagnostics:>10.3?}\n  \
         (checksum {acc:.3}, {} diagnostics)",
        diags.len()
    );
}
