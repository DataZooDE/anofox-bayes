#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb_compat/aggregate.hpp"

#include "../include/anofox_bayes_extension.hpp"
#include "../include/anofox_bayes_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Shared state
//===--------------------------------------------------------------------===//
//
// All three diagnostics consume the same input -- (value, chain, draw) triples -- and
// differ only in which statistic the Rust core computes from them. One state and one
// templated implementation, therefore, rather than three near-identical copies: the
// vector bookkeeping and the null handling can only be got wrong once.
//
// The draw index is carried explicitly because DuckDB makes no promise about the
// order in which rows reach an aggregate state, and every statistic here is a
// function of the sequence. Fed shuffled rows, R-hat and ESS would report excellent
// numbers for a badly mixed fit.

struct DiagnosticState {
	vector<double> values;
	vector<int32_t> chains;
	vector<int32_t> draws;

	void Reset() {
		values.clear();
		chains.clear();
		draws.clear();
	}
};

void DiagnosticInitialize(data_ptr_t state_p) {
	new (state_p) DiagnosticState();
}

void DiagnosticDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	duckdb_compat::ToUnifiedFormat(state_vector, count, sdata);
	auto states = UnifiedVectorFormat::GetData<data_ptr_t>(sdata);
	for (idx_t i = 0; i < count; i++) {
		reinterpret_cast<DiagnosticState *>(states[sdata.sel->get_index(i)])->~DiagnosticState();
	}
}

void DiagnosticUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat value_data, chain_data, draw_data, sdata;
	duckdb_compat::ToUnifiedFormat(inputs[0], count, value_data);
	duckdb_compat::ToUnifiedFormat(inputs[1], count, chain_data);
	duckdb_compat::ToUnifiedFormat(inputs[2], count, draw_data);
	duckdb_compat::ToUnifiedFormat(state_vector, count, sdata);

	auto values = UnifiedVectorFormat::GetData<double>(value_data);
	auto chains = UnifiedVectorFormat::GetData<int64_t>(chain_data);
	auto draws = UnifiedVectorFormat::GetData<int64_t>(draw_data);
	auto states = UnifiedVectorFormat::GetData<data_ptr_t>(sdata);

	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto chain_idx = chain_data.sel->get_index(i);
		auto draw_idx = draw_data.sel->get_index(i);

		// A NULL anywhere in the triple makes the row unplaceable in a sequence, so
		// it is dropped rather than guessed at.
		if (!value_data.validity.RowIsValid(value_idx) || !chain_data.validity.RowIsValid(chain_idx) ||
		    !draw_data.validity.RowIsValid(draw_idx)) {
			continue;
		}

		auto &state = *reinterpret_cast<DiagnosticState *>(states[sdata.sel->get_index(i)]);
		state.values.push_back(values[value_idx]);
		// Narrowed deliberately. The signature takes BIGINT so that INTEGER upcasts
		// implicitly and the indices produced by row_number() and generate_series()
		// bind without a cast, but a chain or draw index beyond 2^31 is not a real
		// posterior -- it is a join gone wrong -- so it is dropped rather than
		// silently wrapped into a plausible-looking index.
		if (chains[chain_idx] > NumericLimits<int32_t>::Maximum() ||
		    chains[chain_idx] < NumericLimits<int32_t>::Minimum() ||
		    draws[draw_idx] > NumericLimits<int32_t>::Maximum() ||
		    draws[draw_idx] < NumericLimits<int32_t>::Minimum()) {
			state.values.pop_back();
			continue;
		}
		state.chains.push_back(static_cast<int32_t>(chains[chain_idx]));
		state.draws.push_back(static_cast<int32_t>(draws[draw_idx]));
	}
}

void DiagnosticCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	duckdb_compat::ToUnifiedFormat(source_vector, count, source_data);
	duckdb_compat::ToUnifiedFormat(target_vector, count, target_data);
	auto sources = UnifiedVectorFormat::GetData<data_ptr_t>(source_data);
	auto targets = UnifiedVectorFormat::GetData<data_ptr_t>(target_data);

	for (idx_t i = 0; i < count; i++) {
		auto &source = *reinterpret_cast<DiagnosticState *>(sources[source_data.sel->get_index(i)]);
		auto &target = *reinterpret_cast<DiagnosticState *>(targets[target_data.sel->get_index(i)]);
		if (source.values.empty()) {
			continue;
		}
		// Order of concatenation is irrelevant: the core reconstructs the sequence
		// from the (chain, draw) indices, which is the whole reason they are carried.
		target.values.insert(target.values.end(), source.values.begin(), source.values.end());
		target.chains.insert(target.chains.end(), source.chains.begin(), source.chains.end());
		target.draws.insert(target.draws.end(), source.draws.begin(), source.draws.end());
	}
}

template <int32_t KIND>
void DiagnosticFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	duckdb_compat::ToUnifiedFormat(state_vector, count, sdata);
	auto states = UnifiedVectorFormat::GetData<data_ptr_t>(sdata);
	auto out = duckdb_compat::FlatData<double>(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *reinterpret_cast<DiagnosticState *>(states[sdata.sel->get_index(i)]);
		idx_t result_idx = i + offset;

		double value = 0;
		bool defined = false;
		bool ok = anofox_bayes_ffi_diagnostic(KIND, state.values.data(), state.chains.data(), state.draws.data(),
		                                      state.values.size(), &value, &defined);

		// A statistic that does not exist for this input becomes SQL NULL. Emitting
		// a number here -- 1.0 for an uncomputable R-hat, 0 for an unassessable ESS --
		// would let an agent gate on something that was never measured.
		if (!ok || !defined) {
			FlatVector::SetNull(result, result_idx, true);
		} else {
			out[result_idx] = value;
		}
		state.Reset();
	}
}

template <int32_t KIND>
AggregateFunction MakeDiagnostic(const char *name) {
	// BIGINT rather than INTEGER: INTEGER upcasts to it implicitly, so the draws
	// contract's own columns bind, and so do row_number() and generate_series()
	// output, which are BIGINT and would otherwise need an explicit cast at every
	// call site.
	return AggregateFunction(name, {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE,
	                         duckdb_compat::AggregateStateSize<sizeof(DiagnosticState)>(),
	                         duckdb_compat::AggregateInit<DiagnosticInitialize>(), DiagnosticUpdate, DiagnosticCombine,
	                         duckdb_compat::AggregateFinalize<DiagnosticFinalize<KIND>>(), nullptr, nullptr,
	                         DiagnosticDestroy);
}

void RegisterOne(ExtensionLoader &loader, const AggregateFunction &func, const char *description) {
	AggregateFunctionSet set(func.name);
	set.AddFunction(func);

	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

	FunctionDescription d;
	d.description = description;
	// Self-contained: an example that reads FROM draws cannot run anywhere the caller
	// has not already produced that table, and an example that does not run is the
	// usual shape of one that is wrong. The inline relation mirrors the draws-table
	// contract (param, value, chain, draw).
	d.examples = {StringUtil::Format("SELECT param, %s(value, chain, draw) FROM (VALUES "
	                                 "('mu',1.0,0,0),('mu',2.0,0,1)) t(param,value,chain,draw) "
	                                 "GROUP BY param",
	                                 func.name)};
	d.categories = {"bayes", "diagnostics"};
	d.parameter_names = {"value", "chain", "draw"};
	d.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT};
	info.descriptions.push_back(std::move(d));
	duckdb_compat::NameFunctionsFromDescriptions(info);

	loader.RegisterFunction(std::move(info));
}

} // anonymous namespace

void RegisterDiagnosticAggregates(ExtensionLoader &loader) {
	RegisterOne(loader, MakeDiagnostic<ANOFOX_BAYES_DIAGNOSTIC_RHAT>("anofox_bayes_rhat"),
	            "Rank-normalised split R-hat over posterior draws. Values above 1.01 indicate chains that have not "
	            "mixed. NULL when the statistic does not exist (a single chain, or too few draws).");
	RegisterOne(loader, MakeDiagnostic<ANOFOX_BAYES_DIAGNOSTIC_ESS_BULK>("anofox_bayes_ess_bulk"),
	            "Bulk effective sample size: how many independent draws the posterior mean is worth. NULL when the "
	            "draws cannot be assessed.");
	RegisterOne(loader, MakeDiagnostic<ANOFOX_BAYES_DIAGNOSTIC_ESS_TAIL>("anofox_bayes_ess_tail"),
	            "Tail effective sample size: how many independent draws the 5% and 95% posterior quantiles are worth. "
	            "Gate service-level and safety-stock decisions on this rather than on bulk ESS.");

	PostHogTelemetry::Instance().RecordFunctionCall("diagnostics_aggregates");
}

} // namespace duckdb
