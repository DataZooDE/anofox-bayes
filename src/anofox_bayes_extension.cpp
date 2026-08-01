#define DUCKDB_EXTENSION_MAIN

#include "include/anofox_bayes_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

// Single source of truth for the version string reported by both the extension API
// and the SQL surface. EXT_VERSION_ANOFOX_BAYES is injected by the DuckDB extension
// build; the literal is the fallback for local, non-CI builds.
const char *ExtensionVersion() {
#ifdef EXT_VERSION_ANOFOX_BAYES
	return EXT_VERSION_ANOFOX_BAYES;
#else
	return "0.1.0";
#endif
}

} // anonymous namespace

#ifdef ANOFOX_TELEMETRY_ENABLED

namespace {

void OnTelemetryEnabled(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_telemetry_enabled cannot be NULL");
	}
	PostHogTelemetry::Instance().SetEnabled(BooleanValue::Get(parameter));
}

void OnTelemetryKey(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_telemetry_key cannot be NULL");
	}
	PostHogTelemetry::Instance().SetAPIKey(StringValue::Get(parameter));
}

} // anonymous namespace

static void RegisterTelemetryOptions(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	config.AddExtensionOption("anofox_telemetry_enabled", "Enable or disable anonymous usage telemetry",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true), OnTelemetryEnabled);

	config.AddExtensionOption("anofox_telemetry_key", "PostHog API key for telemetry", LogicalType::VARCHAR,
	                          Value("phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t"), OnTelemetryKey);
}

#endif // ANOFOX_TELEMETRY_ENABLED

void LoadInternal(ExtensionLoader &loader) {
#ifdef ANOFOX_TELEMETRY_ENABLED
	RegisterTelemetryOptions(loader);

	auto &telemetry = PostHogTelemetry::Instance();
	telemetry.SetAPIKey("phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t");

	const std::string version = ExtensionVersion();
	telemetry.SetProduct("anofox_bayes", version, "commercial");
	telemetry.AssociateGroup("deployment", PostHogTelemetry::GetDistinctId());
	telemetry.CaptureExtensionLoad("anofox_bayes", version);
#endif // ANOFOX_TELEMETRY_ENABLED

	RegisterVersionFunctions(loader);
}

void AnofoxBayesExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string AnofoxBayesExtension::Name() {
	return "anofox_bayes";
}

std::string AnofoxBayesExtension::Version() const {
	return ExtensionVersion();
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_bayes, loader) {
	duckdb::LoadInternal(loader);
}
}
