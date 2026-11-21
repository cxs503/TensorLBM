#!/usr/bin/env python3
"""Launcher: 4 thermal/CHT benchmarks on SDAA cards 28-31.

  SDAA:28 → thermal_cavity   (de Vahl Davis, Ra=1e4)
  SDAA:29 → heated_cylinder  (Re=200, Pr=0.71)
  SDAA:30 → conjugate_ht     (channel + heated block)
  SDAA:31 → rayleigh_benard  (Ra=1e4, convection onset)

Output: thermal_common_sdaa{N}.json + .npy for each benchmark.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).parent
WORKER = WORKER_DIR / "_thermal_common_worker.py"
LOG_DIR = WORKER_DIR / "logs_thermal_common"
LOG_DIR.mkdir(exist_ok=True)

CONFIGS = [
    (28, "thermal_cavity"),
    (29, "heated_cylinder"),
    (30, "conjugate_ht"),
    (31, "rayleigh_benard"),
]

OUTPUT_DIR = WORKER_DIR

print("=" * 80)
print("Thermal/CHT Common-Module Benchmarks — 4 Parallel on SDAA 28-31")
print("=" * 80)
print("  28: thermal_cavity   (de Vahl Davis, Ra=1e4, Pr=0.71, Nu_ref≈2.0)")
print("  29: heated_cylinder  (Re=200, Pr=0.71, Nu_ref≈6.5)")
print("  30: conjugate_ht     (channel + heated block, flux continuity <10%)")
print("  31: rayleigh_benard  (Ra=1e4 > Ra_c=1708, detect convection)")
print()

# Clean previous results
for did, case in CONFIGS:
    for ext in [".json", ".npy"]:
        p = OUTPUT_DIR / f"thermal_common_{case}_sdaa{did}{ext}"
        p.unlink(missing_ok=True)

# Launch workers
procs = []
for did, case in CONFIGS:
    out_path = OUTPUT_DIR / f"thermal_common_{case}_sdaa{did}.json"
    log_file = LOG_DIR / f"log_thermal_{case}_sdaa{did}.txt"

    cmd = [sys.executable, str(WORKER), case, str(did), str(out_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKER_DIR / "src")

    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, case, p, out_path, log_file))
    print(f"Launched {case} on SDAA:{did} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} workers launched. Logs: {LOG_DIR}", flush=True)

# Wait for all
t0 = time.time()
while True:
    done = sum(1 for _, _, p, _, _ in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} done", flush=True)
    if done == len(procs):
        break
    time.sleep(30)

# Collect results
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
all_passed = True
for did, case, p, out_path, log_file in procs:
    rc = p.returncode
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        passed = data.get("passed", False)
        metric = data.get("metric", "N/A")
        elapsed = data.get("elapsed_s", 0)
    else:
        passed = False
        metric = f"NO OUTPUT (rc={rc})"
        elapsed = 0
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  SDAA:{did} {case:20s} {status}  {metric}  ({elapsed:.0f}s)")

print("=" * 80)
print(f"Overall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
print("=" * 80)
