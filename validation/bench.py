#!/usr/bin/env python3
"""Scaling and threading benchmark for anofox_bayes_fit.

The numbers in docs/SCALABILITY.md come from here. Run it after `make release`;
`--threads` additionally checks that the draws are byte-identical across thread
counts, which is the property that makes a fit auditable.
"""
import subprocess, sys, time, resource, pathlib

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


def measure(groups, periods, draws, threads):
    t0 = time.time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    r = subprocess.run([str(DUCK), "-c", fit_sql(groups, periods, draws, threads)],
                       capture_output=True, text=True)
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


def threading():
    print(f"{'threads':>8} {'wall':>8}")
    for t in [1, 2, 4, 8, 16]:
        t0 = time.time()
        subprocess.run([str(DUCK), "-c", fit_sql(2000, 104, 1000, t)], capture_output=True)
        print(f"{t:>8} {time.time()-t0:>7.2f}s")

    print("\ndeterminism across thread counts (md5 of the whole draws table):")
    digests = {}
    for t in [1, 8]:
        r = subprocess.run([str(DUCK), "-c", fit_sql(2000, 104, 1000, t, digest=True)],
                           capture_output=True, text=True)
        hits = [l.strip("│ ") for l in r.stdout.splitlines() if len(l.strip("│ ")) == 32]
        digests[t] = hits[0] if hits else "ERROR"
        print(f"  threads={t}: {digests[t]}")
    same = len(set(digests.values())) == 1
    print("  IDENTICAL" if same else "  *** DIFFERENT -- NOT DETERMINISTIC ***")
    return same


if __name__ == "__main__":
    if not DUCK.exists():
        sys.exit(f"{DUCK} not found -- run `make release` first")
    if "--threads" in sys.argv:
        sys.exit(0 if threading() else 1)
    scaling()
