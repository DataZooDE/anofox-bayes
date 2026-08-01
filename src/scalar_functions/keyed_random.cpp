#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_bayes_extension.hpp"
#include "../include/anofox_bayes_ffi.h"

namespace duckdb {

namespace {

// Deterministic randomness for everything downstream of a fit.
//
// DuckDB's `random()` is seeded per session by `setseed()`, not by the fit, so every
// posterior-predictive recipe in the guide used to be irreproducible even though the
// fit that fed it was not. These are pure functions of their arguments instead: the
// noise on a row is a function of the row, so it survives reordering, re-execution
// after a spill, and any thread count.
//
// Registered as VOLATILE? No -- deliberately the opposite. They are stable functions
// of their inputs and DuckDB is free to treat them as such (fold constants, reuse
// results across an expression), which is exactly the property being advertised.

template <double (*Fn)(int64_t, const uint8_t *, size_t, int64_t)>
void KeyedRandomFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	TernaryExecutor::Execute<int64_t, string_t, int64_t, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [&](int64_t seed, string_t key, int64_t draw) {
		    // GetDataUnsafe may be null for an empty string; the Rust side treats a
		    // null pointer with a zero length as an empty key rather than as UB.
		    return Fn(seed, reinterpret_cast<const uint8_t *>(key.GetData()), key.GetSize(), draw);
	    });
}

} // anonymous namespace

void RegisterKeyedRandomFunctions(ExtensionLoader &loader) {
	// (seed, key, draw). The key is arbitrary text identifying the thing being
	// simulated -- a SKU, a lane, a row id -- and the draw index selects the sample.
	// Together they are the coordinates of a value in a fixed random stream.
	ScalarFunctionSet uniform("anofox_bayes_uniform");
	uniform.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::BIGINT},
	                                   LogicalType::DOUBLE, KeyedRandomFunction<anofox_bayes_ffi_uniform>));
	loader.RegisterFunction(uniform);

	ScalarFunctionSet std_normal("anofox_bayes_std_normal");
	std_normal.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::BIGINT},
	                                      LogicalType::DOUBLE, KeyedRandomFunction<anofox_bayes_ffi_std_normal>));
	loader.RegisterFunction(std_normal);
}

} // namespace duckdb
