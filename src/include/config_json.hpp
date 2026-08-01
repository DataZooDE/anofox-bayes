#pragma once

#include "duckdb.hpp"

namespace duckdb {

// Render a DuckDB Value as the JSON configuration the Rust core parses.
//
// The SQL surface accepts a STRUCT or MAP literal -- `{'value': 'cost', 'draws': 2000}`
// -- because that is what a caller naturally writes and what reads well in a query. The
// core takes JSON, because a typed JSON document is what the HLD specifies and what
// gives every slot a dotted path for error reporting. This is the one-way bridge
// between them.
//
// A VARCHAR argument is passed through untouched, so a caller who has already built a
// config with DuckDB's json functions is not forced to round-trip it through a STRUCT.
string ConfigValueToJson(const Value &value);

} // namespace duckdb
