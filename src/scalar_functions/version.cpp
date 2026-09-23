#include "duckdb.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_bayes_extension.hpp"
#include "../include/anofox_bayes_ffi.h"

namespace duckdb {

namespace {

// Both functions read through the FFI rather than from a C++ constant on purpose:
// they are the end-to-end smoke test that the Rust core is actually linked in. A
// C++-side literal would still return a plausible version from a build where the archive
// was silently dropped -- which is exactly the WASM failure mode documented in
// extension_config.cmake.

void VersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<string_t>(result)[0] = StringVector::AddString(result, anofox_bayes_ffi_version());
}

void DrawsSchemaVersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<int32_t>(result)[0] = anofox_bayes_ffi_draws_schema_version();
}

} // anonymous namespace

void RegisterVersionFunctions(ExtensionLoader &loader) {
	{
		CreateScalarFunctionInfo info(
		    ScalarFunction("anofox_bayes_version", {}, LogicalType::VARCHAR, VersionFunction));
		FunctionDescription d;
		d.description = "Returns the version of the loaded anofox_bayes extension.";
		d.examples = {"SELECT anofox_bayes_version()"};
		d.categories = {"bayes", "meta"};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		CreateScalarFunctionInfo info(ScalarFunction("anofox_bayes_draws_schema_version", {},
		                                             LogicalType::INTEGER, DrawsSchemaVersionFunction));
		FunctionDescription d;
		d.description = "Returns the schema version of the draws-table contract this build writes, so a "
		                "reader can tell whether a persisted draws table it holds is one it understands.";
		d.examples = {"SELECT anofox_bayes_draws_schema_version()"};
		d.categories = {"bayes", "meta"};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
