// Hand-maintained C header for the Rust FFI surface of anofox-bayes.
//
// Kept in sync with crates/anofox-bayes-ffi/src/lib.rs by hand rather than by
// cbindgen: the surface is small, and a generated header would add a build-time
// dependency to every platform in the release matrix for no benefit.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stable error codes shared with anofox_bayes_core::errors::ErrorCode.
// Append-only: agent workflows branch on these values.
typedef enum {
	ANOFOX_BAYES_OK = 0,
	ANOFOX_BAYES_UNKNOWN_FAMILY = 1,
	ANOFOX_BAYES_CONFIG = 2,
	ANOFOX_BAYES_MISSING_COLUMN = 3,
	ANOFOX_BAYES_DIMENSION_MISMATCH = 4,
	ANOFOX_BAYES_INSUFFICIENT_DATA = 5,
	ANOFOX_BAYES_SINGULAR_MATRIX = 6,
	ANOFOX_BAYES_NOT_POSITIVE_DEFINITE = 7,
	ANOFOX_BAYES_CONVERGENCE_FAILURE = 8,
	ANOFOX_BAYES_RESERVED_PARAM_NAME = 9,
	ANOFOX_BAYES_INTERNAL = 99
} AnofoxBayesErrorCode;

// Statically allocated, NUL-terminated. Never free the returned pointer.
const char *anofox_bayes_ffi_version(void);

// Version of the long-format draws contract (docs/DRAWS_CONTRACT.md).
int32_t anofox_bayes_ffi_draws_schema_version(void);

// Selector for anofox_bayes_ffi_diagnostic.
#define ANOFOX_BAYES_DIAGNOSTIC_RHAT 0
#define ANOFOX_BAYES_DIAGNOSTIC_ESS_BULK 1
#define ANOFOX_BAYES_DIAGNOSTIC_ESS_TAIL 2

// Compute one convergence diagnostic from unordered (value, chain, draw) triples.
// Returns false only on a misuse (null out-pointers, unknown kind). *out_defined is
// false when the statistic does not exist for this input -- a single chain has no
// R-hat -- which the caller must surface as SQL NULL rather than as a number.
bool anofox_bayes_ffi_diagnostic(int32_t kind, const double *values, const int32_t *chains,
                                 const int32_t *draws, size_t n, double *out_value,
                                 bool *out_defined);

#ifdef __cplusplus
}
#endif
