#!/usr/bin/env python3
"""Scaling and threading benchmark for anofox_bayes_fit.

The numbers in docs/SCALABILITY.md come from here. Run it after `make release`;
`--threads` additionally checks that the draws are byte-identical across thread
counts, which is the property that makes a fit auditable.

There are *two* thread counts, and both are checked. `SET threads` sizes DuckDB's
pool; `RAYON_NUM_THREADS` sizes the pool `conjugate_anomaly` fits its groups on.
The second is the one that actually varies the fit's wall time, and therefore the
one whose determinism is worth proving.
"""
import os, subprocess, sys, time, resource, pathlib

DUCK = pathlib.Path(__file__).resolve().parent.parent / "build/release/duckdb"


def fit_sql(groups, periods, draws, threads, digest=False):
    tail = (
        "SELECT md5(string_agg(model_id||group_id||chain||draw||param||"
        "coalesce(value::VARCHAR,'N'),'|' ORDER BY group_id,param,chain,draw)) FROM o;"
        if digest
        else "SELECT count(*) FROM o;"
    )
    return f"""SET threads={threads};
LOAD anofox_bayes;
CREATE TABLE d AS SELECT 'G'||g AS grp, p AS period,
  100.0 + g*0.1 + ((g*7+p*3)%11 - 5)*0.5 AS v
FROM generate_series(1,{groups}) a(g), generate_series(1,{periods}) b(p);
CREATE TABLE o AS SELECT * FROM anofox_bayes_fit((SELECT grp, v FROM d),'conjugate_anomaly',
  {{'value':'v','group':'grp','draws':{draws},'seed':1}});
{tail}"""


def env_with(rayon):
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    if rayon is not None:
        env["RAYON_NUM_THREADS"] = str(rayon)
    return env


def measure(groups, periods, draws, threads, rayon=None):
    t0 = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    r = subprocess.run([str(DUCK), "-c", fit_sql(groups, periods, draws, threads)],
                       capture_output=True, text=True, env=env_with(rayon))
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    ok = r.returncode == 0
    print(f"{groups:>7} {periods:>7} {draws:>7} {threads:>7}  {time.time()-t0:>7.2f}s  "
          f"{max(after, before)/1024:>7.0f} MB  {'ok' if ok else 'FAIL'}")
    if not ok:
        print("   ", r.stderr.strip().splitlines()[0][:160])
    return ok


def scaling():
    print(f"{'groups':>7} {'periods':>7} {'draws':>7} {'threads':>7}  {'wall':>8}  {'peakRSS':>10}")
    for g, p, d in [(100, 104, 1000), (1000, 104, 1000), (5000, 104, 1000),
                    (5000, 104, 4000), (20000, 104, 1000)]:
        if not measure(g, p, d, 8):
            break


def best_of(n, groups, periods, draws, threads, rayon):
    """Fastest of `n` runs. The minimum, not the mean: the noise on a shared machine
    is one-sided, so the fastest run is the closest estimate of the cost of the work
    rather than of the cost of the work plus whatever else was scheduled."""
    walls = []
    for _ in range(n):
        t0 = time.time()
        subprocess.run([str(DUCK), "-c", fit_sql(groups, periods, draws, threads)],
                       capture_output=True, env=env_with(rayon))
        walls.append(time.time() - t0)
    return min(walls)


def digest_of(groups, periods, draws, threads, rayon):
    r = subprocess.run([str(DUCK), "-c", fit_sql(groups, periods, draws, threads, digest=True)],
                       capture_output=True, text=True, env=env_with(rayon))
    hits = [l.strip("│ ") for l in r.stdout.splitlines() if len(l.strip("│ ")) == 32]
    return hits[0] if hits else "ERROR"


def threading():
    print("DuckDB threads (`SET threads`), rayon left at its default:")
    print(f"{'threads':>8} {'wall':>8}")
    for t in [1, 2, 4, 8, 16]:
        print(f"{t:>8} {best_of(3, 2000, 104, 1000, t, None):>7.2f}s")

    print("\nfit threads (`RAYON_NUM_THREADS`), DuckDB fixed at 8:")
    print(f"{'threads':>8} {'wall':>8}")
    for r in [1, 2, 4, 8, 16, None]:
        label = "default" if r is None else r
        print(f"{label:>8} {best_of(3, 2000, 104, 1000, 8, r):>7.2f}s")

    print("\ndeterminism across both thread counts (md5 of the whole draws table):")
    digests = {}
    for label, threads, rayon in [("duckdb=1", 1, None), ("duckdb=8", 8, None),
                                  ("rayon=1", 8, 1), ("rayon=16", 8, 16)]:
        digests[label] = digest_of(2000, 104, 1000, threads, rayon)
        print(f"  {label:>9}: {digests[label]}")
    same = len(set(digests.values())) == 1
    print("  IDENTICAL" if same else "  *** DIFFERENT -- NOT DETERMINISTIC ***")
    return same


if __name__ == "__main__":
    if not DUCK.exists():
        sys.exit(f"{DUCK} not found -- run `make release` first")
    if "--threads" in sys.argv:
        sys.exit(0 if threading() else 1)
    scaling()
