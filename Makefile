PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=anofox_bayes
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Rust targets (for local development)
.PHONY: rust_release rust_debug test_rust test_sbc test_scenario lint format_rust format_rust_fix clean_all

rust_release:
	cargo build --release

rust_debug:
	cargo build

# Fast Rust unit tests. Every family, engine and diagnostic has inline
# `#[cfg(test)] mod tests` coverage here; this is the loop you run while coding.
test_rust:
	cargo test --workspace

# Simulation-based calibration. Slow by construction (hundreds of fits per
# family per engine), so it is #[ignore]d and only run explicitly / in CI.
test_sbc:
	cargo test --workspace --release -- --ignored --nocapture

# `test/sql/scenario_counterfactual.test` needs a second extension, so `make test`
# reports it as skipped rather than failing. Install anofox-scenario once and it runs
# with the rest; this target does both.
test_scenario:
	build/release/duckdb -unsigned -c "INSTALL anofox_scenario FROM community;"
	build/release/test/unittest "test/sql/scenario_counterfactual.test"

# What CI gates on, and therefore what to run before pushing. `-D warnings` is CI's
# setting; running anything laxer locally just moves the failure to the pipeline.
lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Named `format_rust` rather than `format`: extension-ci-tools already defines
# `format`/`format-fix` for the C++ side, and overriding them silently drops the
# clang-format pass.
format_rust:
	cargo fmt --all -- --check

format_rust_fix:
	cargo fmt --all

# Clean everything including Rust
clean_all:
	rm -rf build
	cargo clean
