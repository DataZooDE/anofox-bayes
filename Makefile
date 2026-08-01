PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=anofox_bayes
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Rust targets (for local development)
.PHONY: rust_release rust_debug test_rust test_sbc format format-fix clean_all

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

format:
	cargo fmt --all -- --check

format-fix:
	cargo fmt --all

# Clean everything including Rust
clean_all:
	rm -rf build
	cargo clean
