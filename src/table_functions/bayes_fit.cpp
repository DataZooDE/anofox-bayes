#include <mutex>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_bayes_extension.hpp"
#include "../include/anofox_bayes_ffi.h"
#include "../include/config_json.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// anofox_bayes_fit(TABLE data, family, config) -> draws
//===--------------------------------------------------------------------===//
//
// A *pure* table in-out function: it returns the draw rows and materialises nothing
// itself. Persistence is the caller's `CREATE TABLE draws AS SELECT ...`, which keeps
// the extension free of global mutable database state, makes every fit reproducible
// from its inputs alone, and makes the whole surface testable in sqllogictest.
//
// Fitting is inherently a whole-relation operation -- a posterior is a function of
// every observation -- so the operator buffers its input and produces all output in
// the finalize phase. DuckDB may call `in_out_function` concurrently regardless of the
// MaxThreads hint (see anofox-forecast's docs/table-in-out-parallel-execution.md), so
// the accumulator is mutex-guarded, and finalize is serialised behind a claim flag so
// exactly one thread emits. The alternative -- several threads emitting during
// FinalExecute -- crashes PhysicalBatchInsert, which assigns them all the same
// sentinel batch index.
//
// The buffering is not a performance compromise being papered over for *time*: the fit
// cannot start before the last row arrives, so streaming would buy no wall clock. It
// does bound memory, and that gap is real and recorded in docs/SCALABILITY.md -- it is
// this file's to close, not the core's, because by the time the core is called the
// relation is already materialised.
//
// `MaxThreads() == 1` therefore does not mean the fit is serial. `conjugate_anomaly`
// fits and samples each group on its own rayon task inside the core, keyed so the
// draws are identical whatever the pool size; DuckDB simply has nothing to partition
// here. See docs/SCALABILITY.md for the measurements and the determinism digests.

struct BayesFitBindData : public TableFunctionData {
	string family;
	string config_json;
	// Input column names and their roles, decided at bind time from the input schema.
	vector<string> numeric_names;
	vector<idx_t> numeric_cols;
	vector<string> key_names;
	vector<idx_t> key_cols;
};

struct BayesFitGlobalState : public GlobalTableFunctionState {
	mutex lock;
	// Column-major accumulation, matching the FFI builder's shape.
	vector<vector<double>> numeric_values;
	vector<vector<bool>> numeric_valid;
	vector<vector<string>> key_values;
	vector<vector<bool>> key_valid;
	idx_t n_rows = 0;

	// Finalize bookkeeping. `claimed` makes exactly one thread the emitter.
	atomic<bool> claimed {false};
	AnofoxBayesFit *fit = nullptr;
	idx_t emitted = 0;
	idx_t total = 0;

	// The run dictionary, fetched once per fit. A run of parameter rows names its
	// parameters by index, so the names are looked up here rather than re-derived per
	// row. These borrow from the fit and are valid until it is freed.
	vector<AnofoxBayesStr> dict_group;
	vector<AnofoxBayesStr> dict_param;
	AnofoxBayesStr model_id {nullptr, 0};
	// The same names as DuckDB vectors, built once so a chunk can *reference* them
	// through a selection vector instead of copying a string per row.
	unique_ptr<Vector> dict_group_vec;
	unique_ptr<Vector> dict_param_vec;
	SelectionVector run_sel {STANDARD_VECTOR_SIZE};

	~BayesFitGlobalState() override {
		if (fit) {
			anofox_bayes_ffi_fit_free(fit);
		}
	}

	idx_t MaxThreads() const override {
		return 1;
	}
};

// How many worker threads the fit may use.
//
// DuckDB's own budget by default, so `SET threads = n` bounds this extension the way
// it bounds everything else. Before this existed the core ran on rayon's global pool,
// sized from the machine's core count and reachable only through an environment
// variable set before process start -- measured, `SET threads = 1` left the fit using
// more than one core. For a database embedded in someone else's process, having no
// in-process way to cap CPU is a defect rather than a missing tuning knob.
//
// `anofox_bayes_threads` overrides it for the case the single knob cannot express:
// the fit is one operator in a query, and a caller may want it sized differently from
// the scan feeding it. 0 means "follow DuckDB".
uint32_t ResolveThreadBudget(ClientContext &context) {
	Value override_value;
	if (context.TryGetCurrentSetting("anofox_bayes_threads", override_value) && !override_value.IsNull()) {
		auto requested = override_value.GetValue<int64_t>();
		if (requested > 0) {
			return static_cast<uint32_t>(requested);
		}
	}
	// NumberOfThreads is the scheduler's own count and already reflects `SET threads`.
	auto threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	return threads > 0 ? static_cast<uint32_t>(threads) : 1;
}

AnofoxBayesStr Borrow(const string &s) {
	AnofoxBayesStr out;
	out.ptr = s.c_str();
	out.len = s.size();
	return out;
}

unique_ptr<FunctionData> BayesFitBind(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<BayesFitBindData>();

	if (input.inputs.size() < 2 || input.inputs[1].IsNull()) {
		throw BinderException("anofox_bayes_fit requires a family name as its second argument");
	}
	result->family = StringValue::Get(input.inputs[1]);
	result->config_json = ConfigValueToJson(input.inputs[2]);

	// Every column of the input relation is offered to the model; the family's config
	// decides which it reads. Splitting by type here rather than by config keeps the
	// binder ignorant of family internals -- a new family needs no binder change.
	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		auto &type = input.input_table_types[i];
		if (type.IsNumeric()) {
			result->numeric_names.push_back(input.input_table_names[i]);
			result->numeric_cols.push_back(i);
		} else {
			result->key_names.push_back(input.input_table_names[i]);
			result->key_cols.push_back(i);
		}
	}

	// The draws contract, docs/DRAWS_CONTRACT.md.
	names = {"model_id", "group_id", "chain", "draw", "param", "value"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER,
	                LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::DOUBLE};

	PostHogTelemetry::Instance().RecordFunctionCall("anofox_bayes_fit");
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> BayesFitInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<BayesFitBindData>();
	auto state = make_uniq<BayesFitGlobalState>();
	state->numeric_values.resize(bind_data.numeric_cols.size());
	state->numeric_valid.resize(bind_data.numeric_cols.size());
	state->key_values.resize(bind_data.key_cols.size());
	state->key_valid.resize(bind_data.key_cols.size());
	return std::move(state);
}

OperatorResultType BayesFitInOut(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                 DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<BayesFitBindData>();
	auto &gstate = data.global_state->Cast<BayesFitGlobalState>();

	// Materialise this chunk locally first, then take the lock once. Locking per row
	// costs more than the copy does.
	vector<UnifiedVectorFormat> numeric(bind_data.numeric_cols.size());
	for (idx_t c = 0; c < bind_data.numeric_cols.size(); c++) {
		input.data[bind_data.numeric_cols[c]].ToUnifiedFormat(input.size(), numeric[c]);
	}
	vector<UnifiedVectorFormat> keys(bind_data.key_cols.size());
	for (idx_t c = 0; c < bind_data.key_cols.size(); c++) {
		input.data[bind_data.key_cols[c]].ToUnifiedFormat(input.size(), keys[c]);
	}

	lock_guard<mutex> guard(gstate.lock);
	for (idx_t c = 0; c < numeric.size(); c++) {
		auto &fmt = numeric[c];
		auto &out_values = gstate.numeric_values[c];
		auto &out_valid = gstate.numeric_valid[c];
		for (idx_t i = 0; i < input.size(); i++) {
			auto idx = fmt.sel->get_index(i);
			bool ok = fmt.validity.RowIsValid(idx);
			// Cast through Value so every numeric input type -- DECIMAL, HUGEINT,
			// FLOAT -- arrives as the DOUBLE the core works in, rather than being
			// reinterpreted from the wrong physical layout.
			double v = 0;
			if (ok) {
				v = input.data[bind_data.numeric_cols[c]].GetValue(i).GetValue<double>();
			}
			out_values.push_back(v);
			out_valid.push_back(ok);
		}
	}
	for (idx_t c = 0; c < keys.size(); c++) {
		auto &fmt = keys[c];
		auto &out_values = gstate.key_values[c];
		auto &out_valid = gstate.key_valid[c];
		for (idx_t i = 0; i < input.size(); i++) {
			auto idx = fmt.sel->get_index(i);
			bool ok = fmt.validity.RowIsValid(idx);
			out_values.push_back(ok ? input.data[bind_data.key_cols[c]].GetValue(i).ToString() : string());
			out_valid.push_back(ok);
		}
	}
	gstate.n_rows += input.size();

	output.SetCardinality(0);
	return OperatorResultType::NEED_MORE_INPUT;
}

// Run the fit. Called once, under the emitter thread's claim.
void RunFit(const BayesFitBindData &bind_data, BayesFitGlobalState &gstate, uint32_t threads) {
	auto *data = anofox_bayes_ffi_data_new(gstate.n_rows);
	if (!data) {
		throw InternalException("anofox_bayes_fit: could not allocate the input relation");
	}

	// The FFI builder copies, so these buffers only need to outlive the add_* calls.
	try {
		for (idx_t c = 0; c < bind_data.numeric_names.size(); c++) {
			// vector<bool> is a bitset with no contiguous storage, so it cannot be
			// handed over directly. uint8_t rather than char or bool: the FFI
			// contract is an explicit byte, because C++ `bool` has no guaranteed
			// representation and Rust `bool` is UB for anything but 0 or 1.
			vector<uint8_t> valid(gstate.numeric_valid[c].begin(), gstate.numeric_valid[c].end());
			if (!anofox_bayes_ffi_data_add_numeric(data, Borrow(bind_data.numeric_names[c]),
			                                       gstate.numeric_values[c].data(), valid.data(), gstate.n_rows)) {
				throw InternalException("anofox_bayes_fit: rejected numeric column '%s'", bind_data.numeric_names[c]);
			}
		}
		for (idx_t c = 0; c < bind_data.key_names.size(); c++) {
			vector<AnofoxBayesStr> slices;
			slices.reserve(gstate.n_rows);
			for (auto &s : gstate.key_values[c]) {
				slices.push_back(Borrow(s));
			}
			vector<uint8_t> valid(gstate.key_valid[c].begin(), gstate.key_valid[c].end());
			if (!anofox_bayes_ffi_data_add_key(data, Borrow(bind_data.key_names[c]), slices.data(), valid.data(),
			                                   gstate.n_rows)) {
				throw InternalException("anofox_bayes_fit: rejected key column '%s'", bind_data.key_names[c]);
			}
		}

		AnofoxBayesFfiError error;
		error.code = 0;
		error.message[0] = 0;
		gstate.fit =
		    anofox_bayes_ffi_fit(Borrow(bind_data.family), Borrow(bind_data.config_json), data, threads, &error);
		if (!gstate.fit) {
			// The core's message already names the offending config slot or column,
			// so it is surfaced verbatim rather than wrapped in something vaguer.
			throw InvalidInputException("anofox_bayes_fit: %s", string(error.message));
		}
	} catch (...) {
		anofox_bayes_ffi_data_free(data);
		throw;
	}
	anofox_bayes_ffi_data_free(data);

	gstate.total = anofox_bayes_ffi_fit_row_count(gstate.fit);
}

OperatorFinalizeResultType BayesFitFinalize(ExecutionContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<BayesFitBindData>();
	auto &gstate = data.global_state->Cast<BayesFitGlobalState>();

	// Exactly one thread emits. Threads that lose the claim finish immediately; if
	// several emitted, PhysicalBatchInsert would see the same sentinel batch index
	// from each and abort the CREATE TABLE AS.
	bool expected = false;
	if (!gstate.claimed.compare_exchange_strong(expected, true)) {
		output.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!gstate.fit) {
		RunFit(bind_data, gstate, ResolveThreadBudget(context.client));
	}

	// The dictionary and the model id are properties of the fit, not of a chunk, so
	// they are fetched once. `model_id` is read off the first row because it is the
	// same value on every row of the fit -- that is what makes it worth hoisting.
	if (gstate.model_id.ptr == nullptr && gstate.total > 0) {
		auto n_params = anofox_bayes_ffi_fit_param_count(gstate.fit);
		gstate.dict_group.resize(n_params);
		gstate.dict_param.resize(n_params);
		if (n_params > 0) {
			anofox_bayes_ffi_fit_param_names(gstate.fit, gstate.dict_group.data(), gstate.dict_param.data());
		}
		AnofoxBayesStr r_group, r_param;
		int32_t r_chain, r_draw;
		double r_value;
		anofox_bayes_ffi_fit_rows(gstate.fit, 0, 1, &gstate.model_id, &r_group, &r_chain, &r_draw, &r_param,
		                          &r_value);
		if (n_params > 0) {
			gstate.dict_group_vec = make_uniq<Vector>(LogicalType::VARCHAR, n_params);
			gstate.dict_param_vec = make_uniq<Vector>(LogicalType::VARCHAR, n_params);
			auto g = FlatVector::GetData<string_t>(*gstate.dict_group_vec);
			auto p = FlatVector::GetData<string_t>(*gstate.dict_param_vec);
			for (idx_t i = 0; i < n_params; i++) {
				g[i] = StringVector::AddString(*gstate.dict_group_vec, gstate.dict_group[i].ptr,
				                               gstate.dict_group[i].len);
				p[i] = StringVector::AddString(*gstate.dict_param_vec, gstate.dict_param[i].ptr,
				                               gstate.dict_param[i].len);
			}
		}
	}

	// The whole-chunk fast path: when the next run is parameter rows and covers a full
	// chunk, the two name columns are *references* into the per-fit dictionary rather
	// than 2048 fresh string copies. Measured, the two `AddString` calls were ~19 % of
	// the emit; a selection vector of consecutive indices replaces them.
	//
	// Only this shape is special-cased. A chunk that mixes runs -- which happens at the
	// metadata block, at group statuses and around sampler statistics -- takes the flat
	// path below, and there are few such chunks against an `n_params` in the thousands.
	{
		const idx_t capacity = STANDARD_VECTOR_SIZE;
		AnofoxBayesRun peek;
		if (gstate.dict_param_vec && anofox_bayes_ffi_fit_run(gstate.fit, gstate.emitted, capacity, &peek) &&
		    peek.kind == 0 && peek.len == capacity && peek.values != nullptr) {
			for (idx_t i = 0; i < capacity; i++) {
				gstate.run_sel.set_index(i, peek.first_param + i);
			}
			output.data[1].Slice(*gstate.dict_group_vec, gstate.run_sel, capacity);
			output.data[4].Slice(*gstate.dict_param_vec, gstate.run_sel, capacity);

			output.data[0].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::GetData<string_t>(output.data[0])[0] =
			    StringVector::AddString(output.data[0], gstate.model_id.ptr, gstate.model_id.len);

			auto chain_fast = FlatVector::GetData<int32_t>(output.data[2]);
			auto draw_fast = FlatVector::GetData<int32_t>(output.data[3]);
			auto value_fast = FlatVector::GetData<double>(output.data[5]);
			memcpy(value_fast, peek.values, capacity * sizeof(double));
			for (idx_t i = 0; i < capacity; i++) {
				chain_fast[i] = peek.chain;
				draw_fast[i] = peek.draw;
			}
			if (peek.has_nan) {
				for (idx_t i = 0; i < capacity; i++) {
					if (std::isnan(value_fast[i])) {
						FlatVector::SetNull(output.data[5], i, true);
					}
				}
			}
			output.SetCardinality(capacity);
			gstate.emitted += capacity;
			if (gstate.emitted >= gstate.total) {
				return OperatorFinalizeResultType::FINISHED;
			}
			gstate.claimed.store(false);
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}
	}

	// No un-slicing is needed before the flat path below, and it is worth saying why
	// rather than defending against it: the executor calls `DataChunk::Reset()` on this
	// chunk before every `FinalExecute` (`pipeline_executor.cpp`), and `Reset` restores
	// each vector from its cache -- flat type, own data pointer, no dictionary. A
	// `SetVectorType(FLAT_VECTOR)` here would be dead code, and worse, it would look
	// like protection while providing none: it changes the type without restoring
	// `data`, so a vector still pointing into the dictionary would then be written
	// through.

	const idx_t capacity = STANDARD_VECTOR_SIZE;
	auto model_out = FlatVector::GetData<string_t>(output.data[0]);
	auto group_out = FlatVector::GetData<string_t>(output.data[1]);
	auto chain_out = FlatVector::GetData<int32_t>(output.data[2]);
	auto draw_out = FlatVector::GetData<int32_t>(output.data[3]);
	auto param_out = FlatVector::GetData<string_t>(output.data[4]);
	auto value_out = FlatVector::GetData<double>(output.data[5]);

	// Emit by *runs* rather than by rows.
	//
	// The draws table is far more regular than a row-at-a-time walk admits, and
	// `anofox_bayes_ffi_fit_run` reports that regularity: inside one draw the parameter
	// rows are contiguous in the core's own buffer, their `chain` and `draw` are
	// constant, and their names are a function of the parameter index. So the `value`
	// column of a block is a `memcpy` straight out of Rust rather than a copy into a
	// staging array and a second copy out of it, and the six per-chunk staging vectors
	// this loop used to allocate are gone entirely.
	//
	// Rows with no such structure -- the metadata block, group statuses, sampler
	// statistics -- arrive as runs of one and take the original row path, which is
	// still the right tool for them: there are a fixed handful of the first two, and
	// at most four statistics per draw against an `n_params` in the thousands.
	//
	// `model_id` is added to the chunk once and reused: it is one value for the whole
	// fit and, at 16 hex characters, past the 12 bytes `string_t` inlines, so calling
	// `AddString` per row is one heap copy per row of the same string.
	const string_t model_str =
	    gstate.model_id.ptr != nullptr
	        ? StringVector::AddString(output.data[0], gstate.model_id.ptr, gstate.model_id.len)
	        : string_t();

	idx_t written = 0;
	while (written < capacity && gstate.emitted < gstate.total) {
		AnofoxBayesRun run;
		if (!anofox_bayes_ffi_fit_run(gstate.fit, gstate.emitted, capacity - written, &run)) {
			break;
		}

		if (run.kind == 0 && run.values != nullptr) {
			// One move for the whole block's values, then the columns that are constant
			// across it.
			memcpy(value_out + written, run.values, run.len * sizeof(double));
			for (idx_t i = 0; i < run.len; i++) {
				const idx_t at = written + i;
				const auto slot = run.first_param + i;
				model_out[at] = model_str;
				group_out[at] =
				    StringVector::AddString(output.data[1], gstate.dict_group[slot].ptr, gstate.dict_group[slot].len);
				param_out[at] =
				    StringVector::AddString(output.data[4], gstate.dict_param[slot].ptr, gstate.dict_param[slot].len);
				chain_out[at] = run.chain;
				draw_out[at] = run.draw;
			}
			// A parameter the model could not estimate arrives as NaN and leaves as SQL
			// NULL. That is the only honest rendering: a number here would be
			// indistinguishable from an estimate, and telling the two apart is the whole
			// point of the refusal path. `has_nan` lets the usual case skip the scan.
			if (run.has_nan) {
				for (idx_t i = 0; i < run.len; i++) {
					if (std::isnan(value_out[written + i])) {
						FlatVector::SetNull(output.data[5], written + i, true);
					}
				}
			}
		} else {
			// The unstructured rows, read one at a time as before.
			AnofoxBayesStr r_model, r_group, r_param;
			(void)r_model;
			int32_t r_chain, r_draw;
			double r_value;
			const idx_t got = anofox_bayes_ffi_fit_rows(gstate.fit, gstate.emitted, run.len, &r_model, &r_group,
			                                            &r_chain, &r_draw, &r_param, &r_value);
			if (got == 0) {
				break;
			}
			model_out[written] = model_str;
			group_out[written] = StringVector::AddString(output.data[1], r_group.ptr, r_group.len);
			param_out[written] = StringVector::AddString(output.data[4], r_param.ptr, r_param.len);
			chain_out[written] = r_chain;
			draw_out[written] = r_draw;
			if (std::isnan(r_value)) {
				FlatVector::SetNull(output.data[5], written, true);
			} else {
				value_out[written] = r_value;
			}
			run.len = got;
		}

		written += run.len;
		gstate.emitted += run.len;
	}
	output.SetCardinality(written);

	if (gstate.emitted >= gstate.total) {
		return OperatorFinalizeResultType::FINISHED;
	}
	// Release the claim so this thread re-enters and emits the next vector.
	gstate.claimed.store(false);
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

} // anonymous namespace

void RegisterBayesFitFunction(ExtensionLoader &loader) {
	TableFunction fit("anofox_bayes_fit", {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::ANY}, nullptr,
	                  BayesFitBind, BayesFitInitGlobal);
	fit.in_out_function = BayesFitInOut;
	fit.in_out_function_final = BayesFitFinalize;

	// Registered as a single overload, not a set. DuckDB refuses to bind a function
	// that has a TABLE parameter *and* multiple overloads ("this is not supported"),
	// so a convenience two-argument form is not available. It would not be much of a
	// convenience anyway: every family requires at least a value column, so a fit
	// with no config is never a valid request.
	loader.RegisterFunction(std::move(fit));
}

} // namespace duckdb
