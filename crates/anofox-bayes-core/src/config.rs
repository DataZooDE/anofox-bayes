//! Typed, validated model configuration.
//!
//! Configuration is the one place a caller can reach into the mathematics, so it is
//! the one place that has to be pedantic. Three rules, each of which exists because
//! the alternative produces a plausible number from a wrong request:
//!
//! **Every error names its slot.** `invalid config at 'prior.alpha0': must be > 0,
//! got -1` is something an agent can repair. `invalid configuration` is something it
//! can only give up on.
//!
//! **Unknown slots are rejected, not ignored.** A misspelled `"seeed": 7` that
//! silently falls back to the default seed produces a fit that is correct,
//! reproducible, and not the one that was asked for. [`Config::reject_unknown`] turns
//! that into an error naming the typo.
//!
//! **Validation happens before any computation.** By the time an engine runs, every
//! value it reads has already been checked, so no engine contains a "what if this is
//! negative" branch.

use serde_json::Value;

use crate::errors::{BayesError, BayesResult};

/// A validated view over a JSON configuration object.
///
/// `prefix` is the dotted path of this object within the original config, so that a
/// nested accessor can report `prior.alpha0` rather than a bare `alpha0`.
#[derive(Debug, Clone)]
pub struct Config {
    root: Value,
    prefix: String,
}

impl Config {
    /// Parse a JSON object. Anything that is not an object is rejected outright.
    pub fn parse(json: &str) -> BayesResult<Self> {
        let root: Value = serde_json::from_str(json)
            .map_err(|e| BayesError::config("", format!("not valid JSON: {e}")))?;
        // serde_json resolves a repeated key by keeping the last one, silently. That
        // is exactly the ambiguity this module exists to refuse: `{"value":"nope",
        // "value":"cost"}` would fit `cost` and never mention that something else was
        // asked for first.
        reject_duplicate_keys(json)?;
        if !root.is_object() {
            return Err(BayesError::config(
                "",
                format!("expected a JSON object, got {}", type_name(&root)),
            ));
        }
        Ok(Self {
            root,
            prefix: String::new(),
        })
    }

    /// An empty configuration — every slot takes its default.
    pub fn empty() -> Self {
        Self {
            root: Value::Object(Default::default()),
            prefix: String::new(),
        }
    }

    fn slot(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.root.get(name).filter(|v| !v.is_null())
    }

    /// The keys actually present, for error messages and diagnostics.
    pub fn keys(&self) -> Vec<&str> {
        self.root
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Reject any slot the family does not understand.
    ///
    /// The error suggests the closest known slot, because the overwhelmingly common
    /// cause is a typo and an agent repairing its own request benefits from the hint.
    pub fn reject_unknown(&self, allowed: &[&str]) -> BayesResult<()> {
        for key in self.keys() {
            if allowed.contains(&key) {
                continue;
            }
            let hint = closest(key, allowed)
                .map(|s| format!(" (did you mean '{s}'?)"))
                .unwrap_or_default();
            return Err(BayesError::config(
                self.slot(key),
                format!(
                    "unknown option{hint}; this family accepts: {}",
                    allowed.join(", ")
                ),
            ));
        }
        Ok(())
    }

    /// A nested object. Absent nests read as empty, so every slot inside them takes
    /// its default.
    pub fn nested(&self, name: &str) -> BayesResult<Config> {
        match self.get(name) {
            None => Ok(Config {
                root: Value::Object(Default::default()),
                prefix: self.slot(name),
            }),
            Some(v) if v.is_object() => Ok(Config {
                root: v.clone(),
                prefix: self.slot(name),
            }),
            Some(v) => Err(BayesError::config(
                self.slot(name),
                format!("expected an object, got {}", type_name(v)),
            )),
        }
    }

    /// A required string.
    pub fn require_str(&self, name: &str) -> BayesResult<&str> {
        match self.get(name) {
            None => Err(BayesError::config(self.slot(name), "is required")),
            Some(Value::String(s)) if !s.is_empty() => Ok(s),
            Some(Value::String(_)) => Err(BayesError::config(self.slot(name), "must not be empty")),
            Some(v) => Err(BayesError::config(
                self.slot(name),
                format!("expected a string, got {}", type_name(v)),
            )),
        }
    }

    /// An optional string.
    pub fn opt_str(&self, name: &str) -> BayesResult<Option<&str>> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::String(s)) if !s.is_empty() => Ok(Some(s)),
            Some(Value::String(_)) => Err(BayesError::config(self.slot(name), "must not be empty")),
            Some(v) => Err(BayesError::config(
                self.slot(name),
                format!("expected a string, got {}", type_name(v)),
            )),
        }
    }

    /// A string drawn from a fixed set of choices.
    ///
    /// Unknown choices are an error rather than a fallback: a caller who asked for a
    /// Poisson likelihood and quietly received a Gaussian one gets numbers that are
    /// wrong in a way nothing downstream can detect.
    pub fn one_of(
        &self,
        name: &str,
        choices: &[&str],
        default: &'static str,
    ) -> BayesResult<String> {
        let value = match self.opt_str(name)? {
            None => return Ok(default.to_string()),
            Some(s) => s,
        };
        if choices.iter().any(|c| c.eq_ignore_ascii_case(value)) {
            return Ok(value.to_ascii_lowercase());
        }
        Err(BayesError::config(
            self.slot(name),
            format!("expected one of {}, got '{value}'", choices.join(", ")),
        ))
    }

    /// A list of strings. A bare string is accepted as a one-element list, since
    /// `{"x": "price"}` is what a caller writes for a single predictor.
    pub fn str_list(&self, name: &str) -> BayesResult<Vec<String>> {
        match self.get(name) {
            None => Ok(Vec::new()),
            Some(Value::String(s)) => Ok(vec![s.clone()]),
            Some(Value::Array(items)) => items
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    Value::String(s) if !s.is_empty() => Ok(s.clone()),
                    Value::String(_) => Err(BayesError::config(
                        format!("{}[{i}]", self.slot(name)),
                        "must not be empty",
                    )),
                    other => Err(BayesError::config(
                        format!("{}[{i}]", self.slot(name)),
                        format!("expected a string, got {}", type_name(other)),
                    )),
                })
                .collect(),
            Some(v) => Err(BayesError::config(
                self.slot(name),
                format!(
                    "expected a string or array of strings, got {}",
                    type_name(v)
                ),
            )),
        }
    }

    /// A finite number, defaulting when absent.
    pub fn f64_or(&self, name: &str, default: f64) -> BayesResult<f64> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => {
                let n = v.as_f64().ok_or_else(|| {
                    BayesError::config(
                        self.slot(name),
                        format!("expected a number, got {}", type_name(v)),
                    )
                })?;
                if !n.is_finite() {
                    return Err(BayesError::config(self.slot(name), "must be finite"));
                }
                Ok(n)
            }
        }
    }

    /// A strictly positive number — the shape of most prior hyperparameters.
    pub fn positive_f64_or(&self, name: &str, default: f64) -> BayesResult<f64> {
        let n = self.f64_or(name, default)?;
        if n <= 0.0 {
            return Err(BayesError::config(
                self.slot(name),
                format!("must be > 0, got {n}"),
            ));
        }
        Ok(n)
    }

    /// A non-negative number. Distinct from [`Config::positive_f64_or`] because zero
    /// is meaningful for the reference priors, where it means "no prior information".
    pub fn non_negative_f64_or(&self, name: &str, default: f64) -> BayesResult<f64> {
        let n = self.f64_or(name, default)?;
        if n < 0.0 {
            return Err(BayesError::config(
                self.slot(name),
                format!("must be >= 0, got {n}"),
            ));
        }
        Ok(n)
    }

    /// A count within an inclusive range.
    pub fn usize_in(
        &self,
        name: &str,
        default: usize,
        min: usize,
        max: usize,
    ) -> BayesResult<usize> {
        let n = self.f64_or(name, default as f64)?;
        if n.fract() != 0.0 {
            return Err(BayesError::config(
                self.slot(name),
                format!("must be a whole number, got {n}"),
            ));
        }
        let n = n as i64;
        if n < min as i64 || n > max as i64 {
            return Err(BayesError::config(
                self.slot(name),
                format!("must be between {min} and {max}, got {n}"),
            ));
        }
        Ok(n as usize)
    }

    /// The random seed. Defaults to a fixed value rather than to entropy: an
    /// unseeded fit would be irreproducible, and an auditor who cannot reproduce a
    /// recommendation cannot check it.
    pub fn seed(&self) -> BayesResult<u64> {
        match self.get("seed") {
            None => Ok(DEFAULT_SEED),
            // Read as an integer, not through `f64`. Above 2^53 a double cannot
            // represent every u64, so a seed round-tripped through one would silently
            // become a *different* seed -- and a fit that cannot be reproduced from
            // the seed it reports is not auditable.
            Some(v) if v.is_u64() => Ok(v.as_u64().expect("checked is_u64")),
            // One message for both rejections: a caller who passed -1 and a caller who
            // passed 1.5 made the same class of mistake and should read the same
            // sentence.
            Some(v) => Err(BayesError::config(
                "seed",
                format!("must be a non-negative whole number, got {v}"),
            )),
        }
    }

    /// A canonical, key-sorted rendering of this configuration.
    ///
    /// Feeds the `model_id` digest, so two callers who write the same options in a
    /// different order get the same model. `serde_json`'s default map is a
    /// `BTreeMap`, which is what makes the ordering canonical for free.
    pub fn canonical(&self) -> String {
        self.root.to_string()
    }
}

/// Reject any object in the document that names the same key twice.
///
/// `serde_json` cannot report this — its map keeps the last value and discards the
/// rest — so the raw text is re-scanned with a streaming visitor that sees keys in
/// order. The visitor accepts every JSON value type and recurses through objects and
/// arrays; it exists only for its side effect of noticing a repeat.
fn reject_duplicate_keys(json: &str) -> BayesResult<()> {
    use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
    use std::collections::BTreeSet;
    use std::fmt;

    struct Walk;

    impl<'de> DeserializeSeed<'de> for Walk {
        type Value = ();
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            d.deserialize_any(Walk)
        }
    }

    impl<'de> Visitor<'de> for Walk {
        type Value = ();

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("any JSON value")
        }

        // Scalars carry nothing to check; they only have to be accepted.
        fn visit_bool<E>(self, _: bool) -> Result<(), E> {
            Ok(())
        }
        fn visit_i64<E>(self, _: i64) -> Result<(), E> {
            Ok(())
        }
        fn visit_u64<E>(self, _: u64) -> Result<(), E> {
            Ok(())
        }
        fn visit_f64<E>(self, _: f64) -> Result<(), E> {
            Ok(())
        }
        fn visit_str<E>(self, _: &str) -> Result<(), E> {
            Ok(())
        }
        fn visit_unit<E>(self) -> Result<(), E> {
            Ok(())
        }
        fn visit_none<E>(self) -> Result<(), E> {
            Ok(())
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            let mut seen: BTreeSet<String> = BTreeSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if !seen.insert(key.clone()) {
                    return Err(serde::de::Error::custom(key));
                }
                map.next_value_seed(Walk)?;
            }
            Ok(())
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
            while seq.next_element_seed(Walk)?.is_some() {}
            Ok(())
        }
    }

    let mut de = serde_json::Deserializer::from_str(json);
    Walk.deserialize(&mut de).map_err(|e| {
        // The offending key is carried as the custom error message; strip serde's
        // positional suffix so the slot name comes back clean.
        let slot = e.to_string();
        let slot = slot
            .split(" at line")
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        BayesError::config(
            slot,
            "given more than once; a repeated option is ambiguous, so it is rejected \
             rather than resolved by position",
        )
    })
}

/// The seed used when a caller does not supply one.
pub const DEFAULT_SEED: u64 = 20260801;

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// The closest allowed key by edit distance, if one is close enough to be a likely
/// typo rather than a different word entirely.
fn closest<'a>(key: &str, allowed: &[&'a str]) -> Option<&'a str> {
    allowed
        .iter()
        .map(|candidate| (*candidate, edit_distance(key, candidate)))
        .filter(|(candidate, d)| *d <= 2.max(candidate.len() / 3))
        .min_by_key(|(_, d)| *d)
        .map(|(candidate, _)| candidate)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(json: &str) -> Config {
        Config::parse(json).unwrap()
    }

    #[test]
    fn required_strings_are_required() {
        let c = cfg(r#"{"value": "cost"}"#);
        assert_eq!(c.require_str("value").unwrap(), "cost");

        let err = c.require_str("group").unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "group"));
    }

    /// The headline reason this module is pedantic: a misspelled slot that silently
    /// takes its default produces a fit that is correct, reproducible, and not the
    /// one that was asked for.
    #[test]
    fn a_misspelled_slot_is_an_error_that_names_the_typo() {
        let c = cfg(r#"{"value": "cost", "seeed": 7}"#);
        let err = c.reject_unknown(&["value", "group", "seed"]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("seeed"), "{msg}");
        assert!(msg.contains("did you mean 'seed'"), "{msg}");
    }

    #[test]
    fn a_slot_that_is_not_a_near_miss_gets_no_misleading_suggestion() {
        let c = cfg(r#"{"quantum_flux": 1}"#);
        let msg = c
            .reject_unknown(&["value", "group"])
            .unwrap_err()
            .to_string();
        assert!(!msg.contains("did you mean"), "{msg}");
        assert!(msg.contains("value, group"), "{msg}");
    }

    #[test]
    fn nested_slots_report_their_full_path() {
        let c = cfg(r#"{"prior": {"alpha0": -1}}"#);
        let err = c
            .nested("prior")
            .unwrap()
            .positive_f64_or("alpha0", 1.0)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid config at 'prior.alpha0': must be > 0, got -1"
        );
    }

    #[test]
    fn an_absent_nest_lets_every_slot_take_its_default() {
        let c = cfg(r#"{"value": "cost"}"#);
        let prior = c.nested("prior").unwrap();
        assert_eq!(prior.positive_f64_or("alpha0", 2.5).unwrap(), 2.5);
    }

    /// A caller who asked for a Poisson likelihood and quietly received a Gaussian
    /// one gets numbers that are wrong in a way nothing downstream can detect.
    #[test]
    fn an_unknown_choice_is_an_error_rather_than_a_fallback() {
        let c = cfg(r#"{"likelihood": "student_t"}"#);
        let err = c
            .one_of("likelihood", &["normal", "poisson"], "normal")
            .unwrap_err();
        assert!(err.to_string().contains("expected one of normal, poisson"));

        // ...while a known choice is accepted case-insensitively.
        let c = cfg(r#"{"likelihood": "POISSON"}"#);
        assert_eq!(
            c.one_of("likelihood", &["normal", "poisson"], "normal")
                .unwrap(),
            "poisson"
        );
    }

    #[test]
    fn a_single_predictor_may_be_written_without_brackets() {
        assert_eq!(
            cfg(r#"{"x": "price"}"#).str_list("x").unwrap(),
            vec!["price"]
        );
        assert_eq!(
            cfg(r#"{"x": ["price", "promo"]}"#).str_list("x").unwrap(),
            vec!["price", "promo"]
        );
        assert!(cfg(r#"{}"#).str_list("x").unwrap().is_empty());
    }

    #[test]
    fn a_predictor_list_with_a_non_string_names_the_offending_position() {
        let err = cfg(r#"{"x": ["price", 7]}"#).str_list("x").unwrap_err();
        assert!(matches!(err, BayesError::Config { ref slot, .. } if slot == "x[1]"));
    }

    #[test]
    fn numeric_bounds_are_enforced_with_the_offending_value_in_the_message() {
        let c = cfg(r#"{"draws": 0}"#);
        let err = c.usize_in("draws", 1000, 1, 1_000_000).unwrap_err();
        assert!(err.to_string().contains("between 1 and 1000000, got 0"));

        let c = cfg(r#"{"draws": 2.5}"#);
        assert!(c
            .usize_in("draws", 1000, 1, 1_000_000)
            .unwrap_err()
            .to_string()
            .contains("whole number"));
    }

    #[test]
    fn zero_is_allowed_where_it_means_no_prior_information() {
        let c = cfg(r#"{"kappa0": 0}"#);
        assert_eq!(c.non_negative_f64_or("kappa0", 1.0).unwrap(), 0.0);
        assert!(c.positive_f64_or("kappa0", 1.0).is_err());
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        // JSON has no literal for infinity, so the reachable case is a string that
        // parses as a number nowhere.
        let err = cfg(r#"{"kappa0": "1e400"}"#)
            .f64_or("kappa0", 1.0)
            .unwrap_err();
        assert!(err.to_string().contains("expected a number"));
    }

    /// Without a default seed a fit would be irreproducible, and an auditor who
    /// cannot reproduce a recommendation cannot check it.
    #[test]
    fn the_seed_defaults_rather_than_drawing_from_entropy() {
        assert_eq!(cfg(r#"{}"#).seed().unwrap(), DEFAULT_SEED);
        assert_eq!(cfg(r#"{"seed": 42}"#).seed().unwrap(), 42);
        assert!(cfg(r#"{"seed": -1}"#).seed().is_err());
        assert!(cfg(r#"{"seed": 1.5}"#).seed().is_err());

        // Above 2^53 an f64 cannot represent every integer. Read through one, this
        // seed would come back as 9007199254740994 -- a different fit from the one
        // the caller asked for, reported under the seed they did ask for.
        assert_eq!(
            cfg(r#"{"seed": 9007199254740993}"#).seed().unwrap(),
            9007199254740993
        );
        assert_eq!(
            cfg(r#"{"seed": 18446744073709551615}"#).seed().unwrap(),
            u64::MAX
        );
    }

    /// Two callers who write the same options in a different order must get the same
    /// model_id, or the cache would miss and an audit would show two models for one
    /// question.
    #[test]
    fn the_canonical_rendering_is_independent_of_key_order() {
        let a = cfg(r#"{"value": "cost", "draws": 100}"#).canonical();
        let b = cfg(r#"{"draws": 100, "value": "cost"}"#).canonical();
        assert_eq!(a, b);
    }

    /// `serde_json` keeps the last value for a repeated key, silently. A config that
    /// names `value` twice is an ambiguous request, and fitting whichever one came
    /// last is the definition of a plausible answer to the wrong question.
    #[test]
    fn a_repeated_key_is_rejected_rather_than_resolved_by_position() {
        let err = Config::parse(r#"{"value": "nope", "value": "cost"}"#).unwrap_err();
        assert!(
            matches!(err, BayesError::Config { ref slot, .. } if slot == "value"),
            "{err}"
        );
        assert!(err.to_string().contains("more than once"), "{err}");

        // Nested objects are checked too.
        assert!(Config::parse(r#"{"prior": {"a0": 1, "a0": 2}}"#).is_err());
        // ...and a document with no repeats is untouched.
        assert!(Config::parse(r#"{"value": "cost", "prior": {"a0": 1, "b0": 2}}"#).is_ok());
    }

    #[test]
    fn input_that_is_not_a_json_object_is_rejected() {
        assert!(Config::parse("not json").is_err());
        assert!(Config::parse("[1, 2, 3]").is_err());
        assert!(Config::parse("42").is_err());
        assert!(Config::parse("{}").is_ok());
    }

    /// An explicit JSON `null` is the same as an absent slot, because that is what a
    /// SQL `NULL` in a STRUCT literal becomes on the way in.
    #[test]
    fn an_explicit_null_reads_as_absent() {
        let c = cfg(r#"{"group": null}"#);
        assert_eq!(c.opt_str("group").unwrap(), None);
        assert_eq!(c.f64_or("group", 3.0).unwrap(), 3.0);
    }
}
