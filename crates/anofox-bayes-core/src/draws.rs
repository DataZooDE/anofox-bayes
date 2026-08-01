//! The draws contract: how a posterior becomes rows of SQL.
//!
//! A fitted model leaves this crate as a [`Posterior`] — a compact column-major block
//! of numbers — and reaches SQL as long-format rows
//! `(model_id, group_id, chain, draw, param, value)`. [`DrawRows`] is the streaming
//! translation between the two.
//!
//! Two things about the contract are worth stating plainly, because they are the
//! reason it looks the way it does.
//!
//! **The draws table is self-describing.** Alongside the model parameters, the same
//! table carries per-draw sample statistics (`__lp__`, `__divergent__`, …) and
//! model-level metadata (`__status__`, `__n_obs__`, …) under a reserved `__` prefix.
//! An agent that persists one table has persisted the fit, its provenance, and its
//! refusal status together; there is no second table to lose or forget to join.
//!
//! **`model_id` is a pure function of the inputs.** It is a BLAKE3 digest of the data
//! fingerprint, the family, the canonical configuration and the seed. Identical
//! inputs produce an identical id and identical numbers, so cache-hit detection is a
//! comparison rather than a registry, and an auditor can reproduce a customer's
//! recommendation from the inputs alone.

use crate::errors::BayesResult;
use crate::types::{
    validate_param_name, EngineKind, FitStatus, DRAWS_SCHEMA_VERSION, GLOBAL_GROUP,
};

// Reserved parameter names. Sample statistics are per `(chain, draw)`; metadata rows
// are emitted once per model with `chain = -1, draw = -1`.
pub const PARAM_LP: &str = "__lp__";
pub const PARAM_DIVERGENT: &str = "__divergent__";
pub const PARAM_ENERGY: &str = "__energy__";
pub const PARAM_STEP_SIZE: &str = "__step_size__";

pub const META_STATUS: &str = "__status__";
pub const META_ENGINE: &str = "__engine__";
pub const META_SEED: &str = "__seed__";
pub const META_N_OBS: &str = "__n_obs__";
pub const META_N_GROUPS: &str = "__n_groups__";
pub const META_N_CHAINS: &str = "__n_chains__";
pub const META_N_DRAWS: &str = "__n_draws__";
pub const META_SCHEMA_VERSION: &str = "__schema_version__";

/// Sentinel chain/draw index for model-level metadata rows.
///
/// Negative so that `WHERE draw >= 0` cleanly selects real draws, and so that a
/// consumer that forgets to filter gets an obviously-wrong index rather than a
/// plausible one.
pub const META_INDEX: i32 = -1;

/// A parameter's identity: which group it belongs to, and what it is called.
///
/// Group-level parameters carry the group's key (an SKU, a lane, a segment);
/// population-level parameters carry [`GLOBAL_GROUP`]. Keeping the group in the
/// identity rather than baking it into the name is what makes
/// `GROUP BY param` a meaningful diagnostics query across thousands of groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamName {
    pub group_id: String,
    pub name: String,
}

impl ParamName {
    /// A population-level parameter, shared across all groups.
    pub fn global(name: impl Into<String>) -> BayesResult<Self> {
        let name = name.into();
        validate_param_name(&name)?;
        Ok(Self {
            group_id: GLOBAL_GROUP.to_string(),
            name,
        })
    }

    /// A parameter belonging to one group of a hierarchical or per-segment model.
    pub fn grouped(group_id: impl Into<String>, name: impl Into<String>) -> BayesResult<Self> {
        let name = name.into();
        validate_param_name(&name)?;
        Ok(Self {
            group_id: group_id.into(),
            name,
        })
    }
}

/// Per-draw sampler diagnostics, in ArviZ's `sample_stats` spirit.
///
/// The exact and Laplace engines produce independent draws and leave the
/// Hamiltonian-specific fields `None`; NUTS fills them in. Emitting the field as
/// absent rather than as a misleading zero is deliberate — `sum(__divergent__) = 0`
/// must mean "the sampler reported no divergences", not "no sampler ran".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SampleStats {
    /// Log density of the posterior at this draw, up to a constant.
    pub lp: Option<f64>,
    /// 1.0 if the trajectory diverged, 0.0 otherwise.
    pub divergent: Option<f64>,
    pub energy: Option<f64>,
    pub step_size: Option<f64>,
}

impl SampleStats {
    pub fn is_empty(&self) -> bool {
        self.lp.is_none()
            && self.divergent.is_none()
            && self.energy.is_none()
            && self.step_size.is_none()
    }
}

/// Provenance recorded on every fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMeta {
    pub model_id: String,
    pub family: String,
    pub engine: EngineKind,
    pub status: FitStatus,
    pub seed: u64,
    pub n_obs: usize,
    pub n_groups: usize,
}

/// A fitted posterior, before it becomes rows.
///
/// `values` is chain-major and draw-major within a chain:
/// `values[(chain * n_draws + draw) * n_params + p]`. That layout is the one the
/// samplers write in (a chain fills its own contiguous block) and the one the
/// diagnostics read in (a parameter's chains are a strided gather), so no
/// transposition happens anywhere.
#[derive(Debug, Clone)]
pub struct Posterior {
    pub meta: ModelMeta,
    pub params: Vec<ParamName>,
    pub n_chains: usize,
    pub n_draws: usize,
    values: Vec<f64>,
    /// Per `(chain, draw)`, in the same order as `values`. Empty when no engine
    /// reported statistics.
    stats: Vec<SampleStats>,
}

impl Posterior {
    /// Build a posterior from a flat value block.
    ///
    /// Returns [`crate::BayesError::DimensionMismatch`] rather than panicking: an
    /// engine that miscounts its own output is a bug, and a bug that surfaces as a
    /// typed error at the boundary is one an agent can report instead of a crash
    /// that takes the customer's query down.
    pub fn new(
        meta: ModelMeta,
        params: Vec<ParamName>,
        n_chains: usize,
        n_draws: usize,
        values: Vec<f64>,
        stats: Vec<SampleStats>,
    ) -> BayesResult<Self> {
        let expected = n_chains * n_draws * params.len();
        if values.len() != expected {
            return Err(crate::BayesError::DimensionMismatch(format!(
                "expected {expected} values for {n_chains} chains x {n_draws} draws x {} params, got {}",
                params.len(),
                values.len()
            )));
        }
        if !stats.is_empty() && stats.len() != n_chains * n_draws {
            return Err(crate::BayesError::DimensionMismatch(format!(
                "expected {} sample-stat entries, got {}",
                n_chains * n_draws,
                stats.len()
            )));
        }
        Ok(Self {
            meta,
            params,
            n_chains,
            n_draws,
            values,
            stats,
        })
    }

    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// The value of parameter `param` in `chain` at `draw`.
    pub fn value(&self, chain: usize, draw: usize, param: usize) -> f64 {
        self.values[(chain * self.n_draws + draw) * self.params.len() + param]
    }

    /// All draws of one parameter in one chain, in draw order.
    ///
    /// This is the shape the diagnostics want, and the reason for the chain-major
    /// layout: a chain's draws for one parameter are a single strided walk.
    pub fn chain_values(&self, chain: usize, param: usize) -> impl Iterator<Item = f64> + '_ {
        let n_params = self.params.len();
        let base = chain * self.n_draws * n_params + param;
        (0..self.n_draws).map(move |d| self.values[base + d * n_params])
    }

    /// Stream the posterior as long-format rows.
    pub fn rows(&self) -> DrawRows<'_> {
        DrawRows::new(self)
    }
}

/// One row of the draws contract.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawRow<'a> {
    pub model_id: &'a str,
    pub group_id: &'a str,
    pub chain: i32,
    pub draw: i32,
    pub param: &'a str,
    pub value: f64,
}

/// Streaming long-format view of a [`Posterior`].
///
/// An iterator rather than a `Vec<DrawRow>` because a hierarchical fit with thousands
/// of groups produces tens of millions of rows, and the C++ layer consumes them one
/// DuckDB vector at a time (HLD §3.3: no full-run buffering).
///
/// Emission order is fixed and is part of the contract, so that SQL tests can assert
/// on `LIMIT`ed output: metadata rows first, then draws in `(chain, draw, param)`
/// order with each draw's sample statistics immediately after its parameters.
pub struct DrawRows<'a> {
    post: &'a Posterior,
    meta_rows: Vec<(&'static str, f64)>,
    meta_cursor: usize,
    chain: usize,
    draw: usize,
    param: usize,
    stat: usize,
}

impl<'a> DrawRows<'a> {
    fn new(post: &'a Posterior) -> Self {
        let m = &post.meta;
        let meta_rows = vec![
            (META_SCHEMA_VERSION, DRAWS_SCHEMA_VERSION as f64),
            (META_STATUS, m.status as i32 as f64),
            (META_ENGINE, m.engine as i32 as f64),
            (META_SEED, m.seed as f64),
            (META_N_OBS, m.n_obs as f64),
            (META_N_GROUPS, m.n_groups as f64),
            (META_N_CHAINS, post.n_chains as f64),
            (META_N_DRAWS, post.n_draws as f64),
        ];
        Self {
            post,
            meta_rows,
            meta_cursor: 0,
            chain: 0,
            draw: 0,
            param: 0,
            stat: 0,
        }
    }

    /// The sample statistic at position `stat` of the current draw, if present.
    fn next_stat(&mut self) -> Option<(&'static str, f64)> {
        let idx = self.chain * self.post.n_draws + self.draw;
        let stats = self.post.stats.get(idx)?;
        // Fixed probe order so the emitted row order is deterministic.
        let probes: [(&'static str, Option<f64>); 4] = [
            (PARAM_LP, stats.lp),
            (PARAM_DIVERGENT, stats.divergent),
            (PARAM_ENERGY, stats.energy),
            (PARAM_STEP_SIZE, stats.step_size),
        ];
        while self.stat < probes.len() {
            let (name, value) = probes[self.stat];
            self.stat += 1;
            if let Some(v) = value {
                return Some((name, v));
            }
        }
        None
    }
}

impl<'a> Iterator for DrawRows<'a> {
    type Item = DrawRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // 1. Model-level metadata, once.
        if self.meta_cursor < self.meta_rows.len() {
            let (param, value) = self.meta_rows[self.meta_cursor];
            self.meta_cursor += 1;
            return Some(DrawRow {
                model_id: &self.post.meta.model_id,
                group_id: GLOBAL_GROUP,
                chain: META_INDEX,
                draw: META_INDEX,
                param,
                value,
            });
        }

        loop {
            if self.chain >= self.post.n_chains {
                return None;
            }

            // 2. This draw's parameters.
            if self.param < self.post.n_params() {
                let p = &self.post.params[self.param];
                let value = self.post.value(self.chain, self.draw, self.param);
                self.param += 1;
                return Some(DrawRow {
                    model_id: &self.post.meta.model_id,
                    group_id: &p.group_id,
                    chain: self.chain as i32,
                    draw: self.draw as i32,
                    param: &p.name,
                    value,
                });
            }

            // 3. This draw's sample statistics.
            if let Some((param, value)) = self.next_stat() {
                return Some(DrawRow {
                    model_id: &self.post.meta.model_id,
                    group_id: GLOBAL_GROUP,
                    chain: self.chain as i32,
                    draw: self.draw as i32,
                    param,
                    value,
                });
            }

            // 4. Advance.
            self.param = 0;
            self.stat = 0;
            self.draw += 1;
            if self.draw >= self.post.n_draws {
                self.draw = 0;
                self.chain += 1;
            }
        }
    }
}

/// Derive the deterministic `model_id` for a fit.
///
/// Hashing `(family, config, fingerprint, seed)` rather than assigning a counter or a
/// UUID is what makes a fit reproducible and cacheable: the same question asked twice
/// gets the same id, and a different question can never collide with it.
pub fn derive_model_id(
    family: &str,
    canonical_config: &str,
    data_fingerprint: &str,
    seed: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    // Length-prefix each field so that ("ab", "c") and ("a", "bc") differ.
    for field in [family, canonical_config, data_fingerprint] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.update(&seed.to_le_bytes());
    hasher.finalize().to_hex()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ModelMeta {
        ModelMeta {
            model_id: "abc123".to_string(),
            family: "conjugate_anomaly".to_string(),
            engine: EngineKind::Exact,
            status: FitStatus::Converged,
            seed: 42,
            n_obs: 120,
            n_groups: 3,
        }
    }

    /// 2 chains x 3 draws x 2 params, values chosen so that every cell is uniquely
    /// identifiable as `100*chain + 10*draw + param`.
    fn posterior() -> Posterior {
        let params = vec![
            ParamName::global("mu").unwrap(),
            ParamName::grouped("LANE-7", "sigma").unwrap(),
        ];
        let mut values = Vec::new();
        for chain in 0..2 {
            for draw in 0..3 {
                for param in 0..2 {
                    values.push((100 * chain + 10 * draw + param) as f64);
                }
            }
        }
        Posterior::new(meta(), params, 2, 3, values, Vec::new()).unwrap()
    }

    #[test]
    fn the_value_layout_is_chain_major_then_draw_major() {
        let p = posterior();
        assert_eq!(p.value(0, 0, 0), 0.0);
        assert_eq!(p.value(0, 2, 1), 21.0);
        assert_eq!(p.value(1, 0, 0), 100.0);
        assert_eq!(p.value(1, 2, 1), 121.0);
    }

    #[test]
    fn a_chains_draws_for_one_parameter_are_read_in_draw_order() {
        let p = posterior();
        assert_eq!(
            p.chain_values(0, 0).collect::<Vec<_>>(),
            vec![0.0, 10.0, 20.0]
        );
        assert_eq!(
            p.chain_values(1, 1).collect::<Vec<_>>(),
            vec![101.0, 111.0, 121.0]
        );
    }

    #[test]
    fn a_miscounted_value_block_is_a_typed_error_not_a_panic() {
        let params = vec![ParamName::global("mu").unwrap()];
        let err = Posterior::new(meta(), params, 2, 3, vec![1.0, 2.0], Vec::new()).unwrap_err();
        assert!(matches!(err, crate::BayesError::DimensionMismatch(_)));
    }

    #[test]
    fn every_value_round_trips_through_the_long_format() {
        let p = posterior();
        let rows: Vec<_> = p.rows().collect();

        let draw_rows: Vec<_> = rows.iter().filter(|r| r.draw >= 0).collect();
        assert_eq!(draw_rows.len(), 2 * 3 * 2);

        for chain in 0..2i32 {
            for draw in 0..3i32 {
                for (idx, param) in p.params.iter().enumerate() {
                    let row = draw_rows
                        .iter()
                        .find(|r| r.chain == chain && r.draw == draw && r.param == param.name)
                        .unwrap_or_else(|| panic!("missing {}/{chain}/{draw}", param.name));
                    assert_eq!(row.value, p.value(chain as usize, draw as usize, idx));
                    assert_eq!(row.group_id, param.group_id);
                    assert_eq!(row.model_id, "abc123");
                }
            }
        }
    }

    /// The whole point of the reserved namespace: an agent that persists one table
    /// has persisted the refusal status too, and can gate on it without a join.
    #[test]
    fn model_metadata_travels_in_the_same_table_as_the_draws() {
        let p = posterior();
        let rows: Vec<_> = p.rows().collect();
        let meta_rows: Vec<_> = rows.iter().filter(|r| r.draw == META_INDEX).collect();

        let find = |name: &str| {
            meta_rows
                .iter()
                .find(|r| r.param == name)
                .unwrap_or_else(|| panic!("missing metadata row {name}"))
                .value
        };
        assert_eq!(find(META_SCHEMA_VERSION), DRAWS_SCHEMA_VERSION as f64);
        assert_eq!(find(META_STATUS), FitStatus::Converged as i32 as f64);
        assert_eq!(find(META_ENGINE), EngineKind::Exact as i32 as f64);
        assert_eq!(find(META_SEED), 42.0);
        assert_eq!(find(META_N_OBS), 120.0);
        assert_eq!(find(META_N_GROUPS), 3.0);
        assert_eq!(find(META_N_CHAINS), 2.0);
        assert_eq!(find(META_N_DRAWS), 3.0);

        // Metadata is emitted exactly once, not once per chain.
        assert_eq!(meta_rows.len(), 8);
        for row in &meta_rows {
            assert_eq!(row.chain, META_INDEX);
        }
    }

    /// `sum(__divergent__) = 0` must mean "the sampler saw no divergences", never
    /// "no sampler ran". Engines that produce independent draws emit no statistic at
    /// all rather than a reassuring zero.
    #[test]
    fn absent_sample_statistics_emit_no_rows_rather_than_misleading_zeros() {
        let p = posterior();
        assert!(p.rows().all(|r| r.param != PARAM_DIVERGENT));
    }

    #[test]
    fn present_sample_statistics_are_emitted_per_draw() {
        let base = posterior();
        let stats: Vec<SampleStats> = (0..6)
            .map(|i| SampleStats {
                lp: Some(-(i as f64)),
                divergent: Some(if i == 4 { 1.0 } else { 0.0 }),
                ..Default::default()
            })
            .collect();
        let p = Posterior::new(
            base.meta.clone(),
            base.params.clone(),
            2,
            3,
            (0..12).map(|v| v as f64).collect(),
            stats,
        )
        .unwrap();

        let rows: Vec<_> = p.rows().collect();
        let divergences: f64 = rows
            .iter()
            .filter(|r| r.param == PARAM_DIVERGENT)
            .map(|r| r.value)
            .sum();
        assert_eq!(divergences, 1.0);
        assert_eq!(rows.iter().filter(|r| r.param == PARAM_LP).count(), 6);
        // Energy and step size were absent, so they contribute nothing.
        assert!(rows.iter().all(|r| r.param != PARAM_ENERGY));
    }

    #[test]
    fn parameter_names_may_not_invade_the_reserved_namespace() {
        assert!(ParamName::global("__lp__").is_err());
        assert!(ParamName::grouped("LANE-7", "__status__").is_err());
        assert!(ParamName::grouped("__global__", "beta").is_ok());
    }

    #[test]
    fn the_same_question_asked_twice_gets_the_same_model_id() {
        let a = derive_model_id("pooled_gaussian", "{\"y\":\"units\"}", "fp-1", 42);
        let b = derive_model_id("pooled_gaussian", "{\"y\":\"units\"}", "fp-1", 42);
        assert_eq!(a, b);
    }

    #[test]
    fn a_different_question_gets_a_different_model_id() {
        let base = derive_model_id("pooled_gaussian", "{\"y\":\"units\"}", "fp-1", 42);
        assert_ne!(
            base,
            derive_model_id("conjugate_anomaly", "{\"y\":\"units\"}", "fp-1", 42)
        );
        assert_ne!(
            base,
            derive_model_id("pooled_gaussian", "{\"y\":\"cost\"}", "fp-1", 42)
        );
        assert_ne!(
            base,
            derive_model_id("pooled_gaussian", "{\"y\":\"units\"}", "fp-2", 42)
        );
        assert_ne!(
            base,
            derive_model_id("pooled_gaussian", "{\"y\":\"units\"}", "fp-1", 43)
        );
    }

    /// Concatenating fields without length prefixes would make ("ab", "c") and
    /// ("a", "bc") the same model, silently serving one customer's cached posterior
    /// in answer to another's question.
    #[test]
    fn model_id_fields_cannot_bleed_into_each_other() {
        assert_ne!(
            derive_model_id("ab", "c", "fp", 0),
            derive_model_id("a", "bc", "fp", 0)
        );
    }
}
