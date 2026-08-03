"""The one place a demo touches DuckDB.

Every demo in this directory runs against a **real** `anofox_bayes` extension
built from this checkout — the same binary the SQL suite and the PyMC parity
suite load. Nothing here fakes a posterior, and there is no code path that
degrades to one: if the extension is missing, the demo says so and names the
command that builds it.

The discovery rules deliberately mirror `validation/conftest.py` rather than
inventing a second convention, so a developer who has already made the parity
suite run has nothing further to configure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

def _repo_root() -> Path:
    """Walk up until the anofox-bayes checkout is recognisable.

    Counting `parents[n]` would be shorter and was the first version; it broke
    the moment the shared package moved one directory deeper into `lib/`, and it
    would break again for anyone who pip-installed these demos somewhere else.
    Looking for the marker files means the answer stays right under both.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "Makefile").is_file() and (candidate / "crates").is_dir():
            return candidate
    # Nothing recognisable above us -- most likely an installed wheel outside the
    # checkout. Fall back to the ancestor that used to be right, so the error
    # message below still names a plausible path instead of `/`.
    return here.parents[min(4, len(here.parents) - 1)]


REPO_ROOT = _repo_root()

#: Where `make release` puts the loadable extension.
DEFAULT_EXTENSION = (
    REPO_ROOT / "build" / "release" / "extension" / "anofox_bayes" / "anofox_bayes.duckdb_extension"
)
#: Where `make debug` puts it, used only when the release build is absent.
DEBUG_EXTENSION = (
    REPO_ROOT / "build" / "debug" / "extension" / "anofox_bayes" / "anofox_bayes.duckdb_extension"
)

BUILD_HINT = (
    "Build it from the repository root with:\n"
    "    make release -j$(nproc)\n"
    "or point ANOFOX_BAYES_EXTENSION at an existing .duckdb_extension file."
)


class ExtensionMissing(RuntimeError):
    """Raised when the extension cannot be found, with the build command in the message.

    A distinct type rather than a bare `RuntimeError` so a demo's `main` can
    catch exactly this and print it without a traceback: a missing build is a
    setup problem the user fixes in one command, not a crash.
    """


def extension_path() -> Path:
    """Resolve the extension binary, honouring an explicit override.

    `ANOFOX_BAYES_EXTENSION` is the same escape hatch `validation/conftest.py`
    documents, and it wins outright — a developer who set it did so because the
    default is wrong for their build layout.
    """
    override = os.environ.get("ANOFOX_BAYES_EXTENSION")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ExtensionMissing(
                f"ANOFOX_BAYES_EXTENSION points at {path}, which is not a file.\n{BUILD_HINT}"
            )
        return path
    for candidate in (DEFAULT_EXTENSION, DEBUG_EXTENSION):
        if candidate.is_file():
            return candidate
    raise ExtensionMissing(
        f"The anofox_bayes extension was not found at {DEFAULT_EXTENSION} "
        f"or {DEBUG_EXTENSION}.\n{BUILD_HINT}"
    )


@dataclass
class Siblings:
    """Which optional sibling extensions loaded.

    Two demos are honest about wanting a capability this extension does not
    provide: agent 03 would use `anofox_solve` for synthetic-control weights, and
    agent 07 would use `anofox_tabular` for an isolation forest over invoice
    lines. Neither is required — each demo has a plain-SQL path — but a demo that
    silently ran the fallback while describing the full method would be lying
    about its own mechanism, so the flag is carried to the screen.
    """

    solve: bool = False
    tabular: bool = False
    #: Human-readable notes about what was and was not available, shown in-app.
    notes: list[str] = field(default_factory=list)


def _sibling_candidates(name: str) -> list[Path]:
    """Where a sibling checkout would have put its release or debug build.

    Same layout and the same order as
    `anofox-evolve/crates/evolve-agent/src/controller.rs`, which is the
    convention across these repositories.
    """
    repo = REPO_ROOT.parent / name.replace("_", "-")
    return [
        repo / "build" / profile / "extension" / name / f"{name}.duckdb_extension"
        for profile in ("release", "debug")
    ]


def try_load_sibling(con: duckdb.DuckDBPyConnection, name: str, env_var: str) -> bool:
    """Load an optional sibling extension, returning whether it is available.

    Never raises. A sibling that is not built is the expected case for most
    people running these demos, and it is not an error — it changes which path
    the demo takes and what it says on screen.
    """
    override = os.environ.get(env_var)
    candidates = [Path(override)] if override else _sibling_candidates(name)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            con.execute(f"LOAD '{path}'")
            return True
        except duckdb.Error:
            # A sibling built against a different DuckDB refuses at LOAD time.
            # That is exactly the "not available" case, not a demo failure.
            return False
    return False


def connect(want: tuple[str, ...] = ()) -> tuple[duckdb.DuckDBPyConnection, Siblings]:
    """An in-memory DuckDB with the extension loaded.

    `want` names optional siblings by extension id (`anofox_solve`,
    `anofox_tabular`); the returned `Siblings` says which of them arrived.

    Three settings are applied and each is deliberate:

    * `allow_unsigned_extensions`, because the published binaries are unsigned
      and a local build certainly is.
    * `anofox_telemetry_enabled = false` — a demo must not phone home. It is the
      one place in this repository where the default is overridden rather than
      documented.
    * `threads` is left alone. The extension's draws are byte-identical across
      thread counts by construction, and pinning it here would hide a regression
      in that guarantee rather than prevent one.
    """
    path = extension_path()
    con = duckdb.connect(config={"allow_unsigned_extensions": True})
    con.execute(f"LOAD '{path}'")
    con.execute("SET anofox_telemetry_enabled = false")

    siblings = Siblings()
    env = {"anofox_solve": "ANOFOX_SOLVE_EXTENSION", "anofox_tabular": "ANOFOX_TABULAR_EXTENSION"}
    for name in want:
        ok = try_load_sibling(con, name, env[name])
        setattr(siblings, name.removeprefix("anofox_"), ok)
        siblings.notes.append(
            f"{name}: loaded" if ok else f"{name}: not built — using the plain-SQL path"
        )
    return con, siblings


def extension_version(con: duckdb.DuckDBPyConnection) -> str:
    """The extension's own version string, so a demo can show what it ran against."""
    row = con.sql("SELECT anofox_bayes_version()").fetchone()
    return str(row[0]) if row else "unknown"
