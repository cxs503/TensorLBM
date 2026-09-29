#!/usr/bin/env python3
"""Flat plate Re-sweep launcher — runs Cs=0.10 at 4 Re on SDAA cards 16-19 in parallel."""
from __future__ import annotations
import json, subprocess, sys, time, os

PYTHON = sys.executable
WORKER = os.path.join(os.path.dirname(__file__), "flatplate_cs_worker.py")
PYTHONPATH = os.path.join(os.path.dirname(__file__), "src")

RE_VALUES = [5e5, 1e6, 5e6, 1e7]
CARD_IDS = [16, 17, 18, 19]
CS = 0.10

# ITTC-1957 reference values
def ittc_cf(re):
    return 0.075 / (__import__('math').log10(re) - 2.0) ** 2

print("=" * 70)
print("FLAT PLATE RE-SWEEP: Cs=0.10")
print("=" * 70)
for re, cid in zip(RE_VALUES, CARD_IDS):
    print(f"  Re={re:.0e} → SDAA card {cid} | ITTC Cf={ittc_cf(re):.6f}")
print()

procs = []
output_files = []

for re, cid in zip(RE_VALUES, CARD_IDS):
    outfile = f"/tmp/flatplate_re_sweep_{int(re):d}.json"
    output_files.append(outfile)

    cmd = [
        PYTHON, WORKER,
        "--cs", str(CS),
        "--re", str(re),
        "--nx", "200", "--ny", "40", "--nz", "40",
        "--n-steps", "2000",
        "--warmup", "500",
        "--u-in", "0.06",
        "--plate-pct", "0.80",
        "--output", outfile,
    ]

    env = os.environ.copy()
    env["SDAA_VISIBLE_DEVICES"] = str(cid)
    env["PYTHONPATH"] = PYTHONPATH

    logfile = f"/tmp/flatplate_re_sweep_{int(re):d}.log"
    print(f"[Launch] Re={re:.0e} → card {cid} | log={logfile}")
    with open(logfile, 'w') as lf:
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    procs.append((re, cid, p, logfile))

print(f"\nLaunched {len(procs)} jobs. Waiting for completion...\n")

# Wait for all
results = {}
for re, cid, p, logfile in procs:
    print(f"[Wait] Re={re:.0e} (card {cid})...")
    p.wait()
    print(f"[Done] Re={re:.0e} (card {cid}) → rc={p.returncode}")

# Collect results
all_results = []
for re, outfile in zip(RE_VALUES, output_files):
    if os.path.exists(outfile):
        with open(outfile) as f:
            r = json.load(f)
        all_results.append(r)
    else:
        print(f"WARNING: Output file missing: {outfile}")

# Write aggregate
aggregate = {
    "description": "Flat plate Re-sweep with Cs=0.10",
    "cs": CS,
    "grid": {"nx": 200, "ny": 40, "nz": 40},
    "n_steps": 2000,
    "warmup": 500,
    "results": all_results,
    "summary": {},
}

for r in all_results:
    re = r["re"]
    aggregate["summary"][f"Re_{re:.0e}"] = {
        "cf_final": r["cf_final"],
        "cf_ittc": r["cf_ittc"],
        "error_pct": r["error_pct"],
        "n_samples": r["n_samples"],
        "wall_clock_s": r["wall_clock_s"],
    }

with open("/tmp/flatplate_re_sweep.json", 'w') as f:
    json.dump(aggregate, f, indent=2)

print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)
for r in all_results:
    re = r["re"]
    cf = r["cf_final"]
    cf_i = r["cf_ittc"]
    err = r["error_pct"]
    print(f"  Re={re:.0e}: Cf={cf:.6f} vs ITTC={cf_i:.6f} error={err:.1f}% "
          f"({r['wall_clock_s']:.0f}s, {r['n_samples']} samples)")
print(f"\nFull results: /tmp/flatplate_re_sweep.json")

# Determine universality
errors = [r["error_pct"] for r in all_results]
if errors:
    max_err = max(errors)
    min_err = min(errors)
    spread = max_err - min_err
    print(f"\nError range: {min_err:.1f}% – {max_err:.1f}% (spread={spread:.1f}%)")
    if spread < 5.0:
        print("CONCLUSION: Cs=0.10 appears UNIVERSAL across this Re range (spread < 5%).")
    else:
        print(f"CONCLUSION: Cs=0.10 shows Re-dependence (spread={spread:.1f}% > 5%).")
