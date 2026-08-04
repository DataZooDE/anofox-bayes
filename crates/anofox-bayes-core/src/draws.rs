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
    validate_param_name, EngineKind, FamilyCode, FitStatus, SampleFrom, DRAWS_SCHEMA_VERSION,
    GLOBAL_GROUP,
};

// Reserved parameter names. Sample statistics are per `(chain, draw)`; metadata rows
// are emitted once per model with `chain = -1, draw = -1`.
pub const PARAM_LP: &str = "__lp__";
pub const PARAM_DIVERGENT: &str = "__divergent__";
pub const PARAM_ENERGY: &str = "__energy__";
pub const PARAM_STEP_SIZE: &str = "__step_size__";

pub const META_STATUS: &str = "__status__";
pub const META_ENGINE: &str = "__engine__";
pub const META_FAMILY: &str = "__family__";
pub const META_SEED: &str = "__seed__";
pub const META_N_OBS: &str = "__n_obs__";
pub const META_N_GROUPS: &str = "__n_groups__";
pub const META_N_GROUPS_UNREADY: &str = "__n_groups_unready__";
pub const META_N_CHAINS: &str = "__n_chains__";
pub const META_N_DRAWS: &str = "__n_draws__";
pub const META_SCHEMA_VERSION: &str = "__schema_version__";
pub const META_SAMPLE_FROM: &str = "__sample_from__";
/// Per-group readiness. Unlike every other reserved metadata row this one is emitted
/// once *per unready group* and carries that group's key in `group_id`.
pub const META_GROUP_STATUS: &str = "__group_status__";

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
    /// Which catalog family produced this posterior. Held as the code rather than as
    /// the name so that the value written to `__family__` and the name used in
    /// `model_id` cannot drift apart: `FamilyCode::as_str` is the only mapping.
    pub family: FamilyCode,
    pub engine: EngineKind,
    pub status: FitStatus,
    pub seed: u64,
    pub n_obs: usize,
    pub n_groups: usize,
    /// How many of `n_groups` failed their own readiness check. See
    /// [`crate::catalog::CompiledModel::n_groups_unready`] — the model-level
    /// `status` is still the collapsed worst-case verdict, and this says how much of
    /// the fit that verdict is about.
    pub n_groups_unready: usize,
    /// Whether these draws are the posterior or a prior-predictive check.
    ///
    /// Emitted because a persisted prior-predictive table is otherwise
    /// indistinguishable from a posterior one, and acting on the prior in the belief
    /// it is the posterior is acting on no evidence at all.
    pub sample_from: SampleFrom,
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
    /// Groups the family refused, and why, in the order the family reported them.
    /// Empty for a fit with nothing to quarantine.
    unready: Vec<(String, FitStatus)>,
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
        Self::with_unready(meta, params, n_chains, n_draws, values, stats, Vec::new())
    }

    /// As [`Posterior::new`], naming the groups the family refused.
    ///
    /// `Readiness::worst` collapses per-group verdicts into one and still should: a fit
    /// covering 5 000 lanes of which three are unidentifiable is not 99.4 % trustworthy,
    /// it is a fit an agent must look at. But "look at" needs somewhere to look, and
    /// `__n_groups_unready__` only says how many. These rows say *which*, so an agent
    /// holding a 5 000-lane table can quarantine three lanes instead of the fit.
    pub fn with_unready(
        meta: ModelMeta,
        params: Vec<ParamName>,
        n_chains: usize,
        n_draws: usize,
        values: Vec<f64>,
        stats: Vec<SampleStats>,
        unready: Vec<(String, FitStatus)>,
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
        // Every draw must report the same *set* of statistics, even where the values
        // differ. A sampler that reported energy on some draws and not others would
        // make the long format ragged, and a consumer counting rows per draw would
        // silently misalign. It would also be meaningless: a statistic is a property
        // of the sampler, not of an individual draw.
        if let Some(first) = stats.first() {
            let shape = |s: &SampleStats| {
                (
                    s.lp.is_some(),
                    s.divergent.is_some(),
                    s.energy.is_some(),
                    s.step_size.is_some(),
                )
            };
            let expected = shape(first);
            if stats.iter().any(|s| shape(s) != expected) {
                return Err(crate::BayesError::Internal(
                    "sampler reported a different set of statistics on different draws".to_string(),
                ));
            }
        }
        Ok(Self {
            meta,
            params,
            n_chains,
            n_draws,
            values,
            stats,
            unready,
        })
    }

    /// Rows that precede the draws: model metadata, then one row per unready group.
    fn header_rows(&self) -> usize {
        META_ROWS.len() + self.unready.len()
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

    /// The per-draw statistics of one chain, in draw order.
    ///
    /// The chain-major layout that makes [`Self::chain_values`] a strided walk makes
    /// this a contiguous slice. Diagnostics that need to tell one chain from another
    /// -- which one diverged, which one adapted to a larger step -- cannot use the
    /// flat `stats` view, because a total says nothing about whether the trouble was
    /// spread evenly or belonged to a single chain.
    pub fn stats_of_chain(&self, chain: usize) -> &[SampleStats] {
        if self.stats.is_empty() {
            return &[];
        }
        let base = chain * self.n_draws;
        &self.stats[base..base + self.n_draws]
    }

    /// How many kept draws the sampler reported as divergent.
    ///
    /// `None` — not `Some(0)` — when no engine reported the statistic, for the same
    /// reason the row is omitted rather than written as zero: "the sampler saw no
    /// divergences" and "no sampler ran" are different claims, and only one of them
    /// licenses a decision. [`crate::fit::fit`] reads this to grade the fit, so the
    /// distinction is load-bearing rather than cosmetic.
    pub fn n_divergent(&self) -> Option<usize> {
        // The statistic shape is uniform across draws (`Posterior::new` enforces it),
        // so the first draw decides whether the sampler measured divergences at all.
        self.stats.first()?.divergent?;
        Some(
            self.stats
                .iter()
                .filter(|s| s.divergent.is_some_and(|d| d != 0.0))
                .count(),
        )
    }

    /// Statistics rows emitted per draw. Uniform across draws by construction.
    fn stats_per_draw(&self) -> usize {
        match self.stats.first() {
            None => 0,
            Some(s) => {
                usize::from(s.lp.is_some())
                    + usize::from(s.divergent.is_some())
                    + usize::from(s.energy.is_some())
                    + usize::from(s.step_size.is_some())
            }
        }
    }

    /// Total rows this posterior renders to in the long format.
    pub fn n_rows(&self) -> usize {
        self.header_rows()
            + self.n_chains * self.n_draws * (self.n_params() + self.stats_per_draw())
    }

    /// A maximal block of rows that share their structure, starting at `index`.
    ///
    /// The draws table is emitted row by row across the FFI, and a row is not cheap to
    /// produce one at a time: `row_at` pays two divisions, two modulos and a three-way
    /// branch each. But the table is far more regular than that walk admits, and this
    /// exposes the regularity so a consumer can copy blocks instead.
    ///
    /// Inside one draw the parameter rows are **contiguous in `values`** -- the layout
    /// is `values[(chain*n_draws + draw)*n_params + param]` and the rows are emitted in
    /// exactly that order -- so their whole `value` column is a slice rather than a
    /// loop. Their `(group_id, param)` pairs are a function of the parameter index
    /// alone and so repeat identically for every draw, and `chain` and `draw` are
    /// constant across the block. A caller can turn those three facts into a memcpy, a
    /// dictionary and a constant.
    ///
    /// Header and statistic rows carry no such structure and are reported as runs of
    /// one, so a caller falls back to [`Self::row_at`] without a special case. They are
    /// also few: the metadata block is fixed-size and statistics are at most four rows
    /// per draw, against an `n_params` that is thousands once groups are involved.
    pub fn run_at(&self, index: usize) -> Option<Run<'_>> {
        if index >= self.n_rows() {
            return None;
        }
        let single = |chain: i32, draw: i32, first_param: usize| Run {
            kind: RunKind::Single,
            chain,
            draw,
            start: index,
            len: 1,
            first_param,
            values: &[],
        };

        let per_draw = self.n_params() + self.stats_per_draw();
        if index < self.header_rows() || per_draw == 0 {
            return Some(single(META_INDEX, META_INDEX, 0));
        }

        let offset = index - self.header_rows();
        let flat_draw = offset / per_draw;
        let slot = offset % per_draw;
        let chain = (flat_draw / self.n_draws) as i32;
        let draw = (flat_draw % self.n_draws) as i32;
        if slot >= self.n_params() {
            return Some(single(chain, draw, slot));
        }

        // The remainder of this draw's parameter rows: the block worth having.
        let base = flat_draw * self.n_params();
        Some(Run {
            kind: RunKind::Params,
            chain,
            draw,
            start: index,
            len: self.n_params() - slot,
            first_param: slot,
            values: &self.values[base + slot..base + self.n_params()],
        })
    }

    /// The row at a linear index, in O(1).
    ///
    /// Random access rather than a stateful iterator because the C++ layer emits
    /// draws a DuckDB vector at a time and may be resumed from any offset. It is O(1)
    /// precisely because the statistics shape is uniform: the number of rows a draw
    /// occupies is the same for every draw, so a row index divides cleanly into
    /// (chain, draw, slot) with no scanning.
    pub fn row_at(&self, index: usize) -> Option<DrawRow<'_>> {
        let model_id = &self.meta.model_id;
        if index < META_ROWS.len() {
            let (param, value) = self.meta_row(index);
            return Some(DrawRow {
                model_id,
                group_id: GLOBAL_GROUP,
                chain: META_INDEX,
                draw: META_INDEX,
                param,
                value,
            });
        }
        if index < self.header_rows() {
            let (group_id, status) = &self.unready[index - META_ROWS.len()];
            return Some(DrawRow {
                model_id,
                group_id,
                chain: META_INDEX,
                draw: META_INDEX,
                param: META_GROUP_STATUS,
                value: *status as i32 as f64,
            });
        }

        let per_draw = self.n_params() + self.stats_per_draw();
        if per_draw == 0 {
            return None;
        }
        let offset = index - self.header_rows();
        let flat_draw = offset / per_draw;
        let slot = offset % per_draw;
        if flat_draw >= self.n_chains * self.n_draws {
            return None;
        }
        let chain = flat_draw / self.n_draws;
        let draw = flat_draw % self.n_draws;

        if slot < self.n_params() {
            let p = &self.params[slot];
            return Some(DrawRow {
                model_id,
                group_id: &p.group_id,
                chain: chain as i32,
                draw: draw as i32,
                param: &p.name,
                value: self.value(chain, draw, slot),
            });
        }

        let (param, value) = self.stat_row(flat_draw, slot - self.n_params())?;
        Some(DrawRow {
            model_id,
            group_id: GLOBAL_GROUP,
            chain: chain as i32,
            draw: draw as i32,
            param,
            value,
        })
    }

    fn meta_row(&self, i: usize) -> (&'static str, f64) {
        let m = &self.meta;
        let value = match META_ROWS[i] {
            META_SCHEMA_VERSION => DRAWS_SCHEMA_VERSION as f64,
            META_STATUS => m.status as i32 as f64,
            META_ENGINE => m.engine as i32 as f64,
            META_FAMILY => m.family as i32 as f64,
            META_SAMPLE_FROM => m.sample_from as i32 as f64,
            META_SEED => m.seed as f64,
            META_N_OBS => m.n_obs as f64,
            META_N_GROUPS => m.n_groups as f64,
            META_N_GROUPS_UNREADY => m.n_groups_unready as f64,
            META_N_CHAINS => self.n_chains as f64,
            META_N_DRAWS => self.n_draws as f64,
            // Unreachable as long as `META_ROWS` and this match are edited together,
            // which `every_metadata_row_reports_its_own_quantity` enforces. NaN
            // rather than a panic — this runs inside a customer's DuckDB process —
            // and NaN reaches SQL as NULL, which reads as "no value" rather than as
            // some neighbouring row's number. The previous catch-all arm returned
            // `n_draws`, so a name added to `META_ROWS` alone would have been
            // published with a plausible and entirely wrong value.
            _ => f64::NAN,
        };
        (META_ROWS[i], value)
    }

    /// The `nth` present statistic of a draw, in a fixed probe order.
    fn stat_row(&self, flat_draw: usize, nth: usize) -> Option<(&'static str, f64)> {
        let s = self.stats.get(flat_draw)?;
        [
            (PARAM_LP, s.lp),
            (PARAM_DIVERGENT, s.divergent),
            (PARAM_ENERGY, s.energy),
            (PARAM_STEP_SIZE, s.step_size),
        ]
        .into_iter()
        .filter_map(|(name, v)| v.map(|v| (name, v)))
        .nth(nth)
    }

    /// Stream the posterior as long-format rows.
    pub fn rows(&self) -> impl Iterator<Item = DrawRow<'_>> + '_ {
        (0..self.n_rows()).filter_map(|i| self.row_at(i))
    }
}

/// Model-level metadata rows, in emission order. Part of the draws contract.
///
/// Its **length** is load-bearing: [`Posterior::row_at`] treats every index below it
/// as a metadata row and subtracts it to reach the draws, and [`Posterior::n_rows`]
/// adds it. Adding a name here therefore shifts every draw row by one and is only
/// correct when [`Posterior::meta_row`] learns to produce that name's value — adding
/// the name alone would emit it with no meaning attached.
///
/// Adding a row is nevertheless **not** a breaking change under the contract's
/// compatibility rules: consumers are required to filter on the reserved names they
/// know rather than assume a fixed set, so `__schema_version__` does not move.
pub const META_ROWS: &[&str] = &[
    META_SCHEMA_VERSION,
    META_FAMILY,
    META_SAMPLE_FROM,
    META_STATUS,
    META_ENGINE,
    META_SEED,
    META_N_OBS,
    META_N_GROUPS,
    META_N_GROUPS_UNREADY,
    META_N_CHAINS,
    META_N_DRAWS,
];

/// What kind of rows a [`Run`] covers, and therefore what a caller may assume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    /// Parameter rows inside one draw: `values` is their `value` column verbatim,
    /// `chain` and `draw` are constant, and the parameter at offset `i` is
    /// `params[first_param + i]`.
    Params,
    /// One row with no exploitable structure -- metadata, a group status, or a
    /// sampler statistic. Read it with [`Posterior::row_at`].
    Single,
}

/// A block of rows sharing their structure. See [`Posterior::run_at`].
#[derive(Debug, Clone, Copy)]
pub struct Run<'a> {
    pub kind: RunKind,
    pub chain: i32,
    pub draw: i32,
    /// Row index of the first row in the block.
    pub start: usize,
    /// How many rows the block covers. Never zero.
    pub len: usize,
    /// Index into `params` of the first row, when `kind` is [`RunKind::Params`].
    pub first_param: usize,
    /// The `value` column of the block, when `kind` is [`RunKind::Params`].
    pub values: &'a [f64],
}

impl Run<'_> {
    /// The row at `offset` within the block, for tests and for callers that would
    /// rather not special-case the kinds.
    pub fn row<'p>(&self, offset: usize, posterior: &'p Posterior) -> DrawRow<'p> {
        posterior
            .row_at(self.start + offset)
            .expect("a run only covers live rows")
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

/// Bumped whenever a change makes identical inputs produce different draws.
///
/// This is not the extension version and not the draws-schema version. It answers one
/// question: *would this build reproduce the numbers an older build wrote?* A bug fix
/// in a posterior is exactly the case — the inputs are unchanged and the output is
/// deliberately different, so without this the two fits would share a `model_id` and
/// a cache would serve the old, wrong answer for the new, correct request.
///
/// That has already happened once: correcting F3's residual-scale degrees of freedom
/// changed every F3 posterior while leaving the inputs alone.
///
/// | version | change |
/// |---|---|
/// | 1 | initial |
/// | 2 | F3 residual scale corrected to `a0 + (n - k)/2` |
/// | 3 | F7 draws a stream per group rather than one shared sequential stream |
/// | 4 | F1's step target raised to 0.95, so a correct fit is no longer refused |
pub const ALGORITHM_VERSION: u32 = 4;

/// Derive the deterministic `model_id` for a fit.
///
/// Hashing the request rather than assigning a counter or a UUID is what makes a fit
/// reproducible and cacheable: the same question asked twice gets the same id, and a
/// different question can never collide with it.
///
/// The **resolved** engine is part of the digest, not the configured one. A caller who
/// omits `engine` gets the family's default, and if that default later changes the
/// same config would produce a different posterior — with a different warranty — under
/// what would otherwise be the same id.
pub fn derive_model_id(
    family: &str,
    canonical_config: &str,
    data_fingerprint: &str,
    engine: EngineKind,
    seed: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    // Length-prefix each field so that ("ab", "c") and ("a", "bc") differ.
    for field in [family, canonical_config, data_fingerprint] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.update(&(engine as i32).to_le_bytes());
    hasher.update(&ALGORITHM_VERSION.to_le_bytes());
    hasher.update(&seed.to_le_bytes());
    hasher.finalize().to_hex()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ModelMeta {
        ModelMeta {
            model_id: "abc123".to_string(),
            family: FamilyCode::ConjugateAnomaly,
            engine: EngineKind::Exact,
            status: FitStatus::Converged,
            seed: 42,
            n_obs: 120,
            n_groups: 3,
            n_groups_unready: 1,
            sample_from: crate::types::SampleFrom::Posterior,
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

    /// Build a posterior of an arbitrary shape, including the awkward ones.
    fn shaped(
        n_chains: usize,
        n_draws: usize,
        n_params: usize,
        unready: usize,
        stats: bool,
    ) -> Posterior {
        let mut params = Vec::new();
        for p in 0..n_params {
            params.push(if p % 3 == 0 {
                ParamName::global(format!("g{p}")).unwrap()
            } else {
                ParamName::grouped(format!("K-{}", p % 7), format!("p{p}")).unwrap()
            });
        }
        let mut values = Vec::new();
        for c in 0..n_chains {
            for d in 0..n_draws {
                for p in 0..n_params {
                    // NaN in one cell: the refusal path travels as a value, and the
                    // run walk must carry it as faithfully as the row walk.
                    values.push(if c == 0 && d == 0 && p == n_params / 2 {
                        f64::NAN
                    } else {
                        (1000 * c + 10 * d + p) as f64
                    });
                }
            }
        }
        let stat_rows = if stats {
            (0..n_chains * n_draws)
                .map(|i| SampleStats {
                    lp: Some(-(i as f64)),
                    divergent: Some(if i % 5 == 0 { 1.0 } else { 0.0 }),
                    energy: Some(i as f64 * 0.5),
                    step_size: Some(0.1),
                })
                .collect()
        } else {
            Vec::new()
        };
        let unready_groups: Vec<(String, FitStatus)> = (0..unready)
            .map(|i| (format!("U-{i}"), FitStatus::InsufficientData))
            .collect();
        Posterior::with_unready(
            meta(),
            params,
            n_chains,
            n_draws,
            values,
            stat_rows,
            unready_groups,
        )
        .unwrap()
    }

    /// **The invariant the run-based emission must not break.**
    ///
    /// `run_at` exists to let the FFI hand DuckDB whole contiguous blocks instead of
    /// one row at a time. It is only allowed to do that if walking by runs visits
    /// exactly the same rows, in exactly the same order, carrying exactly the same
    /// values as walking by `row_at` -- which is what the draws contract promises and
    /// what every SQL consumer joins on.
    ///
    /// Checked across the shapes that break naive chunking: more parameters than a
    /// vector holds, more draws than parameters, a single draw, refused groups in the
    /// header, and with and without sampler statistics.
    #[test]
    fn walking_by_runs_visits_exactly_the_rows_walking_by_index_does() {
        let shapes = [
            (2, 3, 2, 0, false),
            (2, 3, 2, 1, true),
            (1, 1, 1, 0, false),
            (4, 2, 5000, 2, true),
            (1, 500, 3, 0, true),
            (3, 7, 11, 3, false),
        ];
        for (n_chains, n_draws, n_params, unready, stats) in shapes {
            let p = shaped(n_chains, n_draws, n_params, unready, stats);
            let total = p.n_rows();
            let mut index = 0usize;
            let mut seen = 0usize;
            while index < total {
                let run = p.run_at(index).expect("a run at every live index");
                assert!(
                    run.len > 0,
                    "a zero-length run would not terminate the walk"
                );
                for offset in 0..run.len {
                    let expected = p.row_at(index + offset).expect("row inside the run");
                    let got = run.row(offset, &p);
                    assert_eq!(got.model_id, expected.model_id);
                    assert_eq!(
                        got.group_id,
                        expected.group_id,
                        "shape {n_chains}x{n_draws}x{n_params} row {}",
                        index + offset
                    );
                    assert_eq!(got.chain, expected.chain);
                    assert_eq!(got.draw, expected.draw);
                    assert_eq!(got.param, expected.param);
                    assert_eq!(
                        got.value.to_bits(),
                        expected.value.to_bits(),
                        "value at row {} of shape {n_chains}x{n_draws}x{n_params}",
                        index + offset
                    );
                }
                index += run.len;
                seen += run.len;
            }
            assert_eq!(
                seen, total,
                "the run walk must cover every row exactly once"
            );
            assert!(p.run_at(total).is_none(), "no run past the end");
        }
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
        assert_eq!(
            find(META_FAMILY),
            FamilyCode::ConjugateAnomaly as i32 as f64
        );
        assert_eq!(find(META_STATUS), FitStatus::Converged as i32 as f64);
        assert_eq!(find(META_ENGINE), EngineKind::Exact as i32 as f64);
        assert_eq!(find(META_SEED), 42.0);
        assert_eq!(find(META_N_OBS), 120.0);
        assert_eq!(find(META_N_GROUPS), 3.0);
        assert_eq!(find(META_N_GROUPS_UNREADY), 1.0);
        assert_eq!(find(META_N_CHAINS), 2.0);
        assert_eq!(find(META_N_DRAWS), 3.0);

        // Metadata is emitted exactly once, not once per chain. The count is checked
        // rather than merely the presence of each name: `META_ROWS.len()` is the
        // offset `row_at` subtracts to reach the draws, so a name added to the list
        // without a value behind it would shift every draw row by one.
        assert_eq!(meta_rows.len(), META_ROWS.len());
        assert_eq!(meta_rows.len(), 11);
        for row in &meta_rows {
            assert_eq!(row.chain, META_INDEX);
        }
    }

    /// Every name in `META_ROWS` must have a value behind it in `meta_row`.
    ///
    /// The two are edited together by hand, and the old catch-all arm made a
    /// half-finished edit invisible: a new name fell through to `n_draws` and was
    /// published as a plausible, entirely wrong number. Distinct sentinel values here
    /// make any such crossing detectable, and an unmapped name now shows up as NaN.
    #[test]
    fn every_metadata_row_reports_its_own_quantity() {
        let mut m = meta();
        m.status = FitStatus::Failed; // 3
        m.engine = EngineKind::Nuts; // 2
        m.seed = 101;
        m.n_obs = 102;
        m.n_groups = 103;
        m.n_groups_unready = 104;
        m.sample_from = SampleFrom::Prior; // 1, distinct from every count below
        let params = vec![ParamName::global("mu").unwrap()];
        // 5 chains x 7 draws, so neither is confusable with the other or with a count.
        let p = Posterior::new(m, params, 5, 7, vec![0.0; 35], Vec::new()).unwrap();

        let expected: Vec<(&str, f64)> = vec![
            (META_SCHEMA_VERSION, DRAWS_SCHEMA_VERSION as f64),
            (META_FAMILY, 7.0),
            (META_SAMPLE_FROM, 1.0),
            (META_STATUS, 3.0),
            (META_ENGINE, 2.0),
            (META_SEED, 101.0),
            (META_N_OBS, 102.0),
            (META_N_GROUPS, 103.0),
            (META_N_GROUPS_UNREADY, 104.0),
            (META_N_CHAINS, 5.0),
            (META_N_DRAWS, 7.0),
        ];
        // Every name in the contract is covered by this table, so a name added to
        // `META_ROWS` without being added here fails the assertion below rather than
        // slipping through untested.
        assert_eq!(
            expected.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            META_ROWS.to_vec()
        );
        for (i, (name, value)) in expected.iter().enumerate() {
            let row = p.row_at(i).expect("metadata row");
            assert_eq!(row.param, *name);
            assert_eq!(row.value, *value, "{name}");
        }
        // ...and the first draw row starts immediately after them, which is the
        // invariant `META_ROWS.len()` buys `row_at`.
        let first_draw = p.row_at(META_ROWS.len()).expect("first draw row");
        assert_eq!((first_draw.chain, first_draw.draw), (0, 0));
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

    /// The count a fit is graded on. An engine that reported nothing must return
    /// `None`, so that "clean" and "unmeasured" cannot be confused by the grader.
    #[test]
    fn the_divergence_count_distinguishes_clean_from_unmeasured() {
        let base = posterior();
        assert_eq!(base.n_divergent(), None);

        let with_stats = |divergent: &[f64]| {
            let stats: Vec<SampleStats> = divergent
                .iter()
                .map(|d| SampleStats {
                    lp: Some(-1.0),
                    divergent: Some(*d),
                    ..Default::default()
                })
                .collect();
            Posterior::new(
                base.meta.clone(),
                base.params.clone(),
                2,
                3,
                (0..12).map(|v| v as f64).collect(),
                stats,
            )
            .unwrap()
        };
        assert_eq!(with_stats(&[0.0; 6]).n_divergent(), Some(0));
        assert_eq!(
            with_stats(&[0.0, 1.0, 0.0, 0.0, 1.0, 1.0]).n_divergent(),
            Some(3)
        );

        // A sampler that reports `lp` but not `divergent` has not measured
        // divergences, and must not be read as having found none.
        let lp_only: Vec<SampleStats> = (0..6)
            .map(|_| SampleStats {
                lp: Some(-1.0),
                ..Default::default()
            })
            .collect();
        let p = Posterior::new(
            base.meta.clone(),
            base.params.clone(),
            2,
            3,
            (0..12).map(|v| v as f64).collect(),
            lp_only,
        )
        .unwrap();
        assert_eq!(p.n_divergent(), None);
    }

    #[test]
    fn parameter_names_may_not_invade_the_reserved_namespace() {
        assert!(ParamName::global("__lp__").is_err());
        assert!(ParamName::grouped("LANE-7", "__status__").is_err());
        assert!(ParamName::grouped("__global__", "beta").is_ok());
    }

    #[test]
    fn the_same_question_asked_twice_gets_the_same_model_id() {
        let a = derive_model_id(
            "pooled_gaussian",
            "{\"y\":\"units\"}",
            "fp-1",
            EngineKind::Exact,
            42,
        );
        let b = derive_model_id(
            "pooled_gaussian",
            "{\"y\":\"units\"}",
            "fp-1",
            EngineKind::Exact,
            42,
        );
        assert_eq!(a, b);
    }

    /// A caller who omits `engine` gets the family default. If that default changes,
    /// the same config yields a posterior with a different warranty -- so the resolved
    /// engine, not the configured one, has to be in the digest.
    #[test]
    fn the_resolved_engine_is_part_of_the_identity() {
        let with = |e| derive_model_id("pooled_gaussian", "{}", "fp-1", e, 42);
        assert_ne!(with(EngineKind::Exact), with(EngineKind::Laplace));
        assert_ne!(with(EngineKind::Laplace), with(EngineKind::Nuts));
    }

    #[test]
    fn a_different_question_gets_a_different_model_id() {
        let base = derive_model_id(
            "pooled_gaussian",
            "{\"y\":\"units\"}",
            "fp-1",
            EngineKind::Exact,
            42,
        );
        assert_ne!(
            base,
            derive_model_id(
                "conjugate_anomaly",
                "{\"y\":\"units\"}",
                "fp-1",
                EngineKind::Exact,
                42
            )
        );
        assert_ne!(
            base,
            derive_model_id(
                "pooled_gaussian",
                "{\"y\":\"cost\"}",
                "fp-1",
                EngineKind::Exact,
                42
            )
        );
        assert_ne!(
            base,
            derive_model_id(
                "pooled_gaussian",
                "{\"y\":\"units\"}",
                "fp-2",
                EngineKind::Exact,
                42
            )
        );
        assert_ne!(
            base,
            derive_model_id(
                "pooled_gaussian",
                "{\"y\":\"units\"}",
                "fp-1",
                EngineKind::Exact,
                43
            )
        );
    }

    /// Concatenating fields without length prefixes would make ("ab", "c") and
    /// ("a", "bc") the same model, silently serving one customer's cached posterior
    /// in answer to another's question.
    #[test]
    fn model_id_fields_cannot_bleed_into_each_other() {
        let e = EngineKind::Exact;
        assert_ne!(
            derive_model_id("ab", "c", "fp", e, 0),
            derive_model_id("a", "bc", "fp", e, 0)
        );
    }
}
