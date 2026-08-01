#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

// --- Scalar functions -------------------------------------------------------
void RegisterVersionFunctions(ExtensionLoader &loader);

// Extension class required for static linking
class AnofoxBayesExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
