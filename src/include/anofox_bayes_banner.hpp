#pragma once

#include "datazoo_banner_duckdb.hpp"

// Shared identity for the load banner and the issue-link error footer.
//
// The footer is applied here, on the C++ side of the FFI boundary, and
// deliberately NOT inside BayesError in the Rust core: that type is an
// agent-facing contract (see crates/anofox-bayes-core/src/errors.rs, which
// documents that the primary caller is an agent and that variants carry
// machine-readable repair information). Appending prose to its Display would
// corrupt that. Annotating at the boundary still means every error a *human*
// reads carries the link.
extern const datazoo::BannerInfo ANOFOX_BAYES_BANNER;
