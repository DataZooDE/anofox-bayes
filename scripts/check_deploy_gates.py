#!/usr/bin/env python3
"""Check that no deploy job can publish a binary its smoke test rejected.

The smoke test is the only job that loads the shipped .duckdb_extension file in an
official DuckDB CLI. A deploy that runs under a bare `!cancelled()` publishes even
when that test failed, so a broken artifact reaches users anyway. For every job
calling _extension_deploy.yml this asserts that

  * it `needs` the smoke-test job for the same duckdb_version, and
  * its `if:` requires that job's result to be 'success';

and for every job calling _extension_smoke_test.yml, that its `if:` contains
`!cancelled()`, so a failed non-essential build arch does not skip the smoke test
(and with it every deploy). Exit 1 with one line per violation.

usage: check_deploy_gates.py [WORKFLOW.yml]   (default: the main pipeline)
"""
import re
import sys

import yaml

DEPLOY = "_extension_deploy.yml"
SMOKE = "_extension_smoke_test.yml"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        jobs = yaml.safe_load(f)["jobs"]

    def calls(job, wf):
        return str(job.get("uses", "")).endswith("/" + wf)

    def needs(job):
        n = job.get("needs", [])
        return [n] if isinstance(n, str) else list(n)

    def cond(job):
        return re.sub(r"\s+", " ", str(job.get("if", "")))

    smoke = {name: j for name, j in jobs.items() if calls(j, SMOKE)}
    deploy = {name: j for name, j in jobs.items() if calls(j, DEPLOY)}
    errors = []
    if not deploy:
        errors.append(f"{path}: no job calls {DEPLOY}")
    for name, j in smoke.items():
        if "!cancelled()" not in cond(j):
            errors.append(f"{name}: smoke test lacks `if: !cancelled() ...`, so one failed "
                          "build arch skips it and blocks every deploy")
    for name, j in deploy.items():
        version = j.get("with", {}).get("duckdb_version")
        gate = [s for s in needs(j)
                if s in smoke and smoke[s].get("with", {}).get("duckdb_version") == version]
        if not gate:
            errors.append(f"{name}: deploys {version} without needing its smoke test")
            continue
        for s in gate:
            if f"needs.{s}.result == 'success'" not in cond(j):
                errors.append(f"{name}: `if:` does not require needs.{s}.result == 'success'; "
                              "a failed smoke test would still deploy")
    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"OK: {len(deploy)} deploy job(s) gated on their smoke test")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".github/workflows/MainDistributionPipeline.yml"))
