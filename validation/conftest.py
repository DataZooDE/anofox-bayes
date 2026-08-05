"""Session fixtures for the PyMC parity suite.

The one job of this file is to get a DuckDB connection with the *locally built*
`anofox_bayes` extension loaded into it, and to skip loudly rather than fail
obscurely when that binary is not there.
"""

from __future__ import annotations

import os
import pathlib

import duckdb
import pytest

# The repository root, i.e. the parent of `validation/`.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Where `make release` puts the loadable extension.
DEFAULT_EXTENSION_PATH = (
    REPO_ROOT / "build" / "release" / "extension" / "anofox_bayes" / "anofox_bayes.duckdb_extension"
)

# The schema version this suite was written against. Bumping the draws contract
# should force a deliberate look at these tests rather than silently changing
# what they mean.
EXPECTED_DRAWS_SCHEMA_VERSION = 1


def _extension_path() -> pathlib.Path:
    """Resolve the extension binary, honouring an explicit override.

    CI builds into a non-default prefix often enough that hard-coding the path
    would make this suite unrunnable there; `ANOFOX_BAYES_EXTENSION` is the
    documented escape hatch.
    """
    override = os.environ.get("ANOFOX_BAYES_EXTENSION")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    return DEFAULT_EXTENSION_PATH


def pytest_terminal_summary(terminalreporter):
    """Print how close each comparison came to its tolerance.

    Green tests that were one Monte Carlo wobble from red, and tolerances with
    100x of unused headroom, are both worth seeing; neither shows up in a pass
    count.
    """
    try:
        from _support import (  # noqa: PLC0415 - test-only import
            ALL_TOLERANCES,
            MARGINS,
        )
    except Exception:  # pragma: no cover - suite skipped before import
        return
    if not MARGINS:
        return

    # Driven off ALL_TOLERANCES rather than a hand-written tuple: a family added
    # to the suite but forgotten here would be sampled, asserted, and then left
    # out of the one table anybody reads before touching a tolerance.
    limits = {
        tol.label: {"mean": tol.mean, "sd": tol.sd, "q05": tol.quantile, "q95": tol.quantile}
        for tol in ALL_TOLERANCES
    }
    stats = ("mean", "sd", "q05", "q95")
    terminalreporter.write_sep("=", "parity margins (delta / tolerance; 1.0 = at the limit)")
    for label in sorted({m[0] for m in MARGINS}):
        rows = [m for m in MARGINS if m[0] == label]
        terminalreporter.write_line(f"{label} ({len(rows)} comparisons)")
        terminalreporter.write_line(
            "    " + f"{'parameter':<34}" + "".join(f"{s:>9}" for s in stats)
        )
        scored = sorted(
            rows,
            key=lambda r: max(r[2][s] / limits[label][s] for s in stats),
            reverse=True,
        )
        for _, name, deltas in scored:
            cells = "".join(f"{deltas[s] / limits[label][s]:>8.1%} " for s in stats)
            terminalreporter.write_line(f"    {name:<34}{cells}")


@pytest.fixture(scope="session")
def extension_path() -> pathlib.Path:
    path = _extension_path()
    if not path.is_file():
        pytest.skip(
            "anofox_bayes extension not found at\n"
            f"    {path}\n"
            "Build it first:\n"
            "    make release        # from the repository root\n"
            "or point the suite at an existing build:\n"
            "    ANOFOX_BAYES_EXTENSION=/path/to/anofox_bayes.duckdb_extension uv run pytest",
            allow_module_level=False,
        )
    return path


@pytest.fixture(scope="session")
def con(extension_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with the extension loaded.

    The extension is built unsigned locally, so unsigned loads have to be
    enabled explicitly; this connection is scoped to the test session and never
    touches disk.
    """
    connection = duckdb.connect(
        database=":memory:", config={"allow_unsigned_extensions": True}
    )
    try:
        connection.execute(f"LOAD '{extension_path}'")
    except duckdb.Exception as exc:  # pragma: no cover - environment problem
        pytest.skip(f"could not load {extension_path}: {exc}")

    (schema_version,) = connection.sql(
        "SELECT anofox_bayes_draws_schema_version()"
    ).fetchone()
    if schema_version != EXPECTED_DRAWS_SCHEMA_VERSION:
        pytest.fail(
            f"draws contract is at schema version {schema_version}, but this parity "
            f"suite was written against version {EXPECTED_DRAWS_SCHEMA_VERSION}. "
            "Re-read docs/DRAWS_CONTRACT.md and update the suite deliberately."
        )
    return connection


@pytest.fixture(scope="session")
def parity_comparison_count():
    """How many statistics the suite actually compared, for the sizing test.

    Read from `MARGINS` rather than counted statically, because the per-family
    comparison lists are built at run time from the fixtures. Each recorded margin
    carries the four statistics `parity_deltas` returns.
    """
    from _support import MARGINS

    return len(MARGINS) * 4
