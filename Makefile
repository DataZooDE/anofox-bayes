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
# The counterfactual suite, which needs anofox-scenario alongside this extension.
#
# KNOWN LIMITATION, and the guard below exists so it cannot be mistaken for a pass:
# this suite does not currently run. `require anofox_scenario` skips rather than
# fails when the extension will not load, and anofox-scenario's binaries are
# **unsigned** -- `test/unittest` is a Catch binary with no `-unsigned` flag, so it
# cannot load one. A skipped suite reports as success, which is exactly the failure
# this repository keeps finding elsewhere, so the recipe below turns a skip into a
# non-zero exit.
#
# Two things would fix it, neither of them in this repository: signing the
# anofox-scenario binaries, or having CI provide a `test/unittest` that permits
# unsigned extensions.
#
# Until then the suite is a specification that is verified by hand. What *has* been
# checked: every statement in it executes without error against the two extensions
# loaded together in the DuckDB shell, and the composition it relies on -- fitting
# from an attached scenario catalog, and the fit's `model_id` diverging because the
# data fingerprint covers the branch's rows -- is verified directly.
#
# anofox-scenario is BSL-licensed and served from the DataZoo channel, not from the
# DuckDB community repository: `INSTALL ... FROM community` silently does nothing.
test_scenario:
	@build/release/duckdb -unsigned -c "INSTALL anofox_scenario FROM 'http://get.erpl.io'; LOAD anofox_scenario;" \
		|| { echo "anofox_scenario unavailable - the counterfactual suite cannot run"; exit 1; }
	@build/release/test/unittest "test/sql/scenario_counterfactual.test" | tee /tmp/anofox_scenario_test.log
	@grep -q "skipped" /tmp/anofox_scenario_test.log \
		&& { echo "FAIL: the suite skipped rather than ran"; exit 1; } || true

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
