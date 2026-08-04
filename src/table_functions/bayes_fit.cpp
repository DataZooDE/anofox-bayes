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

	idx_t capacity = STANDARD_VECTOR_SIZE;
	vector<AnofoxBayesStr> model_id(capacity), group_id(capacity), param(capacity);
	vector<int32_t> chain(capacity), draw(capacity);
	vector<double> value(capacity);

	idx_t written = anofox_bayes_ffi_fit_rows(gstate.fit, gstate.emitted, capacity, model_id.data(), group_id.data(),
	                                          chain.data(), draw.data(), param.data(), value.data());
	gstate.emitted += written;

	auto model_out = FlatVector::GetData<string_t>(output.data[0]);
	auto group_out = FlatVector::GetData<string_t>(output.data[1]);
	auto chain_out = FlatVector::GetData<int32_t>(output.data[2]);
	auto draw_out = FlatVector::GetData<int32_t>(output.data[3]);
	auto param_out = FlatVector::GetData<string_t>(output.data[4]);
	auto value_out = FlatVector::GetData<double>(output.data[5]);

	// The three string columns repeat, heavily and by construction. `model_id` is one
	// value for the whole fit; `group_id` takes one value per group and `param` one per
	// parameter, and a draws table is those few names crossed with thousands of draws.
	// The core hands back borrowed `&str` into storage it owns for the life of the fit,
	// so the *same* distinct string always arrives as the same pointer -- which makes
	// the pointer a sound cache key, and a cheaper one than the bytes.
	//
	// Without this, every row calls `AddString`, and a string longer than the 12 bytes
	// `string_t` inlines is a heap copy each time. `model_id` is 16 hex characters, so
	// it was one allocation per row: five million of them for the same value on a
	// 5 000-group fit.
	// `model_id` is one value for the entire fit, and at 16 hex characters it is past
	// the 12 bytes `string_t` inlines -- so the naive loop heap-copied the same string
	// once per row, five million times on a 5 000-group fit. Added once per chunk
	// instead. `group_id` and `param` are deliberately *not* cached: measured, a
	// pointer-keyed memo for them was slower than calling `AddString`, because a
	// 2048-row chunk spans more distinct groups than a small cache holds and both
	// columns are usually short enough to inline anyway.
	const string_t model_str =
	    written > 0 ? StringVector::AddString(output.data[0], model_id[0].ptr, model_id[0].len) : string_t();
	for (idx_t i = 0; i < written; i++) {
		model_out[i] = model_str;
		group_out[i] = StringVector::AddString(output.data[1], group_id[i].ptr, group_id[i].len);
		chain_out[i] = chain[i];
		draw_out[i] = draw[i];
		param_out[i] = StringVector::AddString(output.data[4], param[i].ptr, param[i].len);
		// A parameter the model could not estimate arrives as NaN and leaves as SQL
		// NULL. That is the only honest rendering: a number here would be
		// indistinguishable from an estimate, and telling the two apart is the whole
		// point of the refusal path.
		if (std::isnan(value[i])) {
			FlatVector::SetNull(output.data[5], i, true);
		} else {
			value_out[i] = value[i];
		}
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
