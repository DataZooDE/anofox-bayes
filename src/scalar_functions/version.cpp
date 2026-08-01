#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_bayes_extension.hpp"
#include "../include/anofox_bayes_ffi.h"

namespace duckdb {

namespace {

// Both functions read through the FFI rather than from a C++ constant on purpose:
// they are the end-to-end smoke test that the Rust core is actually linked in. A
// C++-side literal would still return "0.1.0" from a build where the static archive
// was silently dropped -- which is exactly the WASM failure mode documented in
// extension_config.cmake.

void VersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<string_t>(result)[0] =
	    StringVector::AddString(result, anofox_bayes_ffi_version());
}

void DrawsSchemaVersionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<int32_t>(result)[0] = anofox_bayes_ffi_draws_schema_version();
}

} // anonymous namespace

void RegisterVersionFunctions(ExtensionLoader &loader) {
	ScalarFunction version("anofox_bayes_version", {}, LogicalType::VARCHAR, VersionFunction);
	loader.RegisterFunction(version);

	ScalarFunction draws_schema_version("anofox_bayes_draws_schema_version", {}, LogicalType::INTEGER,
	                                    DrawsSchemaVersionFunction);
	loader.RegisterFunction(draws_schema_version);
}

} // namespace duckdb
