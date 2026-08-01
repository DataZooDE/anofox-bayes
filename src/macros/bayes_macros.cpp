#include "duckdb.hpp"
#include "duckdb/catalog/default/default_functions.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"

#include "../include/anofox_bayes_extension.hpp"

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Decision helpers over a draws table
//===--------------------------------------------------------------------===//
//
// These are SQL macros, not C++ functions, and deliberately so. Every one of them is
// a short expression a caller could write by hand; shipping them as macros makes the
// idiom discoverable and consistent without putting anything between the caller and
// the draws. A macro is also transparent -- `SELECT macro_definition FROM
// duckdb_functions()` shows exactly what it does, which matters when the output is
// going into an audit trail.
//
// All of them are aggregates over `value`, so they compose with `GROUP BY group_id`
// exactly the way the raw quantile functions do.

const DefaultMacro ANOFOX_BAYES_MACROS[] = {
    // --- Intervals --------------------------------------------------------
    {DEFAULT_SCHEMA,
     "anofox_bayes_credible_lower",
     {"value", "level", nullptr},
     {{nullptr, nullptr}},
     "quantile_cont(value, (1.0 - level) / 2.0)"},
    {DEFAULT_SCHEMA,
     "anofox_bayes_credible_upper",
     {"value", "level", nullptr},
     {{nullptr, nullptr}},
     "quantile_cont(value, 1.0 - (1.0 - level) / 2.0)"},
    // The interval as one struct, for callers who want a single column.
    {DEFAULT_SCHEMA,
     "anofox_bayes_credible_interval",
     {"value", "level", nullptr},
     {{nullptr, nullptr}},
     "{'lower': quantile_cont(value, (1.0 - level) / 2.0),"
     " 'median': quantile_cont(value, 0.5),"
     " 'upper': quantile_cont(value, 1.0 - (1.0 - level) / 2.0)}"},

    // --- Decisions --------------------------------------------------------
    //
    // The posterior probability that a parameter exceeds a threshold. This is the
    // question a decision-maker actually asks -- "is the effect bigger than the
    // rollout cost?" -- and on a draws table it is a mean of an indicator, with no
    // distributional theory in between.
    //
    // NULL propagates rather than counting as "not greater". A parameter the model
    // declined to estimate has NULL draws, and the naive
    // `CASE WHEN value > t THEN 1 ELSE 0 END` scores every one of them as a miss --
    // so `P(effect > 0)` on a refused parameter comes back `0.0`, a confident
    // "definitely not", when the honest answer is "no estimate". `avg` already skips
    // NULLs, so mapping an unestimable draw to NULL makes the whole aggregate NULL
    // exactly when every draw is unestimable, which is the behaviour the draws
    // contract promises.
    {DEFAULT_SCHEMA,
     "anofox_bayes_prob_greater",
     {"value", "threshold", nullptr},
     {{nullptr, nullptr}},
     "avg(CASE WHEN value IS NULL THEN NULL WHEN value > threshold THEN 1.0 ELSE 0.0 END)"},
    {DEFAULT_SCHEMA,
     "anofox_bayes_prob_less",
     {"value", "threshold", nullptr},
     {{nullptr, nullptr}},
     "avg(CASE WHEN value IS NULL THEN NULL WHEN value < threshold THEN 1.0 ELSE 0.0 END)"},

    // The quantile a service-level target implies. A 95% service level means the
    // quantity that covers demand in 95% of futures, which is the 95th percentile of
    // the posterior -- not the mean plus some multiple of a standard deviation, which
    // is only equivalent if the posterior happens to be symmetric.
    {DEFAULT_SCHEMA,
     "anofox_bayes_service_level_quantile",
     {"value", "service_level", nullptr},
     {{nullptr, nullptr}},
     "quantile_cont(value, service_level)"},

    // --- Gating -----------------------------------------------------------
    //
    // Whether a parameter clears its effective-sample-size gate, NULL-safe by
    // construction.
    //
    // This exists because the obvious hand-written form is wrong in a way that fails
    // open. `HAVING ess_bulk < 400` looks like it flags under-sampled parameters, but
    // ESS is NULL where it is not defined -- a parameter that never moved, or too few
    // draws to assess -- and `NULL < 400` is NULL, not true. The parameter most in
    // need of flagging is exactly the one that slips through. Wrapping the comparison
    // in `coalesce(..., false)` makes an absent diagnostic a failure, which is the
    // only safe reading.
    {DEFAULT_SCHEMA,
     "anofox_bayes_ess_gate",
     {"value", "chain", "draw", "min_ess", nullptr},
     {{nullptr, nullptr}},
     "coalesce(anofox_bayes_ess_bulk(value, chain, draw) >= min_ess"
     "     AND anofox_bayes_ess_tail(value, chain, draw) >= min_ess, false)"},
    // The R-hat half of the same idea. A NULL R-hat is *not* a failure -- both 0.1
    // engines draw independently, so the statistic is legitimately undefined -- which
    // is why this defaults to true rather than false, and why it is a separate macro
    // rather than being folded into the ESS gate.
    {DEFAULT_SCHEMA,
     "anofox_bayes_rhat_gate",
     {"value", "chain", "draw", "max_rhat", nullptr},
     {{nullptr, nullptr}},
     "coalesce(anofox_bayes_rhat(value, chain, draw) <= max_rhat, true)"},

    //
    // Decodes the reserved __status__ row. Aggregate rather than scalar so it can be
    // applied straight to a draws table without a WHERE clause.
    {DEFAULT_SCHEMA,
     "anofox_bayes_status_text",
     {"param", "value", nullptr},
     {{nullptr, nullptr}},
     "CASE max(CASE WHEN param = '__status__' THEN value END)"
     "  WHEN 0 THEN 'converged'"
     "  WHEN 1 THEN 'degenerate'"
     "  WHEN 2 THEN 'insufficient_data'"
     "  WHEN 3 THEN 'failed'"
     "  ELSE 'unknown' END"},
    // True only for a fit an agent may act on without further qualification.
    {DEFAULT_SCHEMA,
     "anofox_bayes_is_actionable",
     {"param", "value", nullptr},
     {{nullptr, nullptr}},
     "coalesce(max(CASE WHEN param = '__status__' THEN value END) = 0, false)"},

    // Decodes the reserved __family__ row: which model produced this table.
    //
    // The value column is DOUBLE, so the family travels as its catalog F-number
    // (docs/DRAWS_CONTRACT.md). Without this macro an auditor holding a persisted
    // table would have to know that 7 means `conjugate_anomaly`, which is precisely
    // the knowledge a persisted table is supposed to carry for them. An unknown code
    // reads as 'unknown' rather than NULL: a table written by a newer extension is a
    // fact worth seeing, not a missing value to coalesce away.
    {DEFAULT_SCHEMA,
     "anofox_bayes_family_text",
     {"param", "value", nullptr},
     {{nullptr, nullptr}},
     "CASE max(CASE WHEN param = '__family__' THEN value END)"
     "  WHEN 1 THEN 'hier_negbin'"
     "  WHEN 2 THEN 'censored_aft'"
     "  WHEN 3 THEN 'pooled_gaussian'"
     "  WHEN 5 THEN 'payer_alive'"
     "  WHEN 7 THEN 'conjugate_anomaly'"
     "  ELSE 'unknown' END"},

    {nullptr, nullptr, {nullptr}, {{nullptr, nullptr}}, nullptr}};

} // anonymous namespace

void RegisterBayesMacros(ExtensionLoader &loader) {
	for (idx_t i = 0; ANOFOX_BAYES_MACROS[i].name != nullptr; i++) {
		auto info = DefaultFunctionGenerator::CreateInternalMacroInfo(ANOFOX_BAYES_MACROS[i]);
		info->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		loader.RegisterFunction(*info);
	}
}

} // namespace duckdb
