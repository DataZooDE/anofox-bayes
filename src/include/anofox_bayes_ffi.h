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

// Keyed randomness for the predictive step (docs/GUIDE.md, keyed_rng.rs).
//
// Pure functions of their arguments: same coordinates, same value, on any thread and
// in any evaluation order. `key` may be NULL when `key_len` is 0.
double anofox_bayes_ffi_uniform(int64_t seed, const uint8_t *key, size_t key_len, int64_t draw);
double anofox_bayes_ffi_std_normal(int64_t seed, const uint8_t *key, size_t key_len, int64_t draw);

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

// --- Fitting ---------------------------------------------------------------
//
// Lifecycle:  data_new -> add_numeric/add_key -> fit -> fit_rows -> free.
// Pointers passed in are borrowed for the call only (the builder copies).
// Pointers handed back borrow from the handle and live until it is freed.

// A borrowed string as (pointer, length). Not NUL-terminated: Rust strings are not,
// and copying to add a byte would double the cost of the largest output we produce.
typedef struct {
	const char *ptr;
	size_t len;
} AnofoxBayesStr;

typedef struct {
	int32_t code;
	char message[512];
} AnofoxBayesFfiError;

typedef struct AnofoxBayesData AnofoxBayesData;
typedef struct AnofoxBayesFit AnofoxBayesFit;

AnofoxBayesData *anofox_bayes_ffi_data_new(size_t n_rows);
void anofox_bayes_ffi_data_free(AnofoxBayesData *data);
// Validity masks are uint8_t (0 = NULL, non-zero = present), not `bool`. C++ `bool`
// has no guaranteed size or representation and Rust `bool` is UB for any bit pattern
// other than 0 or 1, so passing one as the other is an ABI accident rather than a
// contract.
bool anofox_bayes_ffi_data_add_numeric(AnofoxBayesData *data, AnofoxBayesStr name, const double *values,
                                       const uint8_t *valid, size_t len);
bool anofox_bayes_ffi_data_add_key(AnofoxBayesData *data, AnofoxBayesStr name, const AnofoxBayesStr *values,
                                   const uint8_t *valid, size_t len);

// Returns NULL on failure with *out_error populated.
AnofoxBayesFit *anofox_bayes_ffi_fit(AnofoxBayesStr family, AnofoxBayesStr config_json, const AnofoxBayesData *data,
                                     AnofoxBayesFfiError *out_error);
void anofox_bayes_ffi_fit_free(AnofoxBayesFit *fit);
size_t anofox_bayes_ffi_fit_row_count(const AnofoxBayesFit *fit);

// Copies up to `max` rows starting at `offset`; returns how many were written.
size_t anofox_bayes_ffi_fit_rows(const AnofoxBayesFit *fit, size_t offset, size_t max, AnofoxBayesStr *out_model_id,
                                 AnofoxBayesStr *out_group_id, int32_t *out_chain, int32_t *out_draw,
                                 AnofoxBayesStr *out_param, double *out_value);

// FitStatus code; -1 for a null handle. Reasons are newline-joined.
int32_t anofox_bayes_ffi_fit_status(const AnofoxBayesFit *fit, AnofoxBayesStr *out_reasons);

#ifdef __cplusplus
}
#endif
