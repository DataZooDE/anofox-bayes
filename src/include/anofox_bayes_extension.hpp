#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

// --- Scalar functions -------------------------------------------------------
void RegisterVersionFunctions(ExtensionLoader &loader);
void RegisterKeyedRandomFunctions(ExtensionLoader &loader);

// --- Aggregate functions ----------------------------------------------------
void RegisterDiagnosticAggregates(ExtensionLoader &loader);

// --- Table functions --------------------------------------------------------
void RegisterBayesFitFunction(ExtensionLoader &loader);

// --- SQL macros -------------------------------------------------------------
void RegisterBayesMacros(ExtensionLoader &loader);

// Extension class required for static linking
class AnofoxBayesExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
