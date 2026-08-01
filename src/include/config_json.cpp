#include "config_json.hpp"

#include "duckdb/common/types/value.hpp"

namespace duckdb {

namespace {

void EscapeInto(const string &text, string &out) {
	out += '"';
	for (unsigned char c : text) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (c < 0x20) {
				// JSON forbids raw control characters. \u escapes are the portable
				// encoding, and a column name containing one is pathological enough
				// that correctness matters more than brevity here.
				out += StringUtil::Format("\\u%04x", static_cast<int>(c));
			} else {
				out += static_cast<char>(c);
			}
		}
	}
	out += '"';
}

void RenderInto(const Value &value, string &out) {
	if (value.IsNull()) {
		// An explicit SQL NULL becomes a JSON null, which the core treats as an
		// absent slot -- so `{'group': NULL}` and omitting `group` mean the same
		// thing, which is what a caller building a config conditionally expects.
		out += "null";
		return;
	}

	switch (value.type().id()) {
	case LogicalTypeId::BOOLEAN:
		out += BooleanValue::Get(value) ? "true" : "false";
		return;
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		// DuckDB's own numeric formatting round-trips, and every numeric type is
		// valid JSON in that form.
		out += value.ToString();
		return;
	case LogicalTypeId::STRUCT: {
		out += '{';
		auto &children = StructValue::GetChildren(value);
		auto &names = StructType::GetChildTypes(value.type());
		for (idx_t i = 0; i < children.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			EscapeInto(names[i].first, out);
			out += ':';
			RenderInto(children[i], out);
		}
		out += '}';
		return;
	}
	case LogicalTypeId::MAP: {
		out += '{';
		auto &entries = ListValue::GetChildren(value);
		for (idx_t i = 0; i < entries.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			auto &kv = StructValue::GetChildren(entries[i]);
			EscapeInto(kv[0].ToString(), out);
			out += ':';
			RenderInto(kv[1], out);
		}
		out += '}';
		return;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY: {
		out += '[';
		auto &items = ListValue::GetChildren(value);
		for (idx_t i = 0; i < items.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			RenderInto(items[i], out);
		}
		out += ']';
		return;
	}
	default:
		// Everything else -- VARCHAR, dates, enums -- reaches the core as a string.
		// The core validates against the slot's expected type, so an inappropriate
		// one becomes a precise config error rather than a parse failure here.
		EscapeInto(value.ToString(), out);
		return;
	}
}

} // anonymous namespace

string ConfigValueToJson(const Value &value) {
	if (value.IsNull()) {
		return "{}";
	}
	// A VARCHAR is already JSON by convention: passing it through untouched lets a
	// caller build a config with DuckDB's json functions without round-tripping it
	// through a STRUCT and back.
	if (value.type().id() == LogicalTypeId::VARCHAR) {
		auto text = StringValue::Get(value);
		return text.empty() ? "{}" : text;
	}

	string out;
	RenderInto(value, out);
	return out;
}

} // namespace duckdb
