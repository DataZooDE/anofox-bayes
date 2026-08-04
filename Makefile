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
# It used to be a specification that never ran. `require anofox_scenario` skips rather
# than fails, and a skipped suite reports as success -- exactly the failure this
# repository keeps finding elsewhere. The stated blocker was that anofox-scenario's
# binaries were unsigned while `test/unittest` is a Catch binary with no `-unsigned`
# flag.
#
# That blocker is gone: anofox-scenario is published in the DuckDB community
# repository, so `INSTALL ... FROM community` produces a *signed* binary, and
# `LOAD anofox_scenario` succeeds inside `test/unittest` with no `-unsigned` anywhere.
#
# `require` still skips even then -- it answers "is this statically linked", not "can
# this be loaded" -- so the suite guards on `require-env ANOFOX_SCENARIO` and loads the
# extension explicitly. This target sets that variable, which is what makes the
# difference between a suite that runs and a suite that reports a pass for doing
# nothing. The grep stays as a belt-and-braces check on that promise.
test_scenario:
	@build/release/duckdb -c "FORCE INSTALL anofox_scenario FROM community; LOAD anofox_scenario;" \
		|| { echo "anofox_scenario unavailable - the counterfactual suite cannot run"; exit 1; }
	@ANOFOX_SCENARIO=1 build/release/test/unittest "test/sql/scenario_counterfactual.test" \
		| tee /tmp/anofox_scenario_test.log
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
