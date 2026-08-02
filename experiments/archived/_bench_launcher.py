#!/usr/bin/env python3
"""AllBenchs launcher: 8 parallel benchmarks on SDAA cards.

SDAA cards in different P2P groups (4 cards per group):
  SDAA:0,4,8,12,16,20,24,28  — each from different groups.

Worker: _bench_worker.py
Output: /tmp/all_bench_results.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).parent
WORKER = WORKER_DIR / "_bench_worker.py"
LOG_DIR = Path("/tmp/allbench_logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Configuration: (sdaa_id, case_name) ────────────────────────────────────
# Using SDAA cards in different P2P groups to avoid conflicts
configs = [
    (0,  "naca0012_a0"),
    (4,  "square_prism"),
    (8,  "backward_step"),
    (12, "rect_prism_2_1_1"),
    (16, "s809_a0"),
    (20, "ahmed_25deg"),
    (24, "naca4412_a5"),
    (28, "sphere_d48"),
]

OUTPUT_DIR = Path("/tmp/allbench_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 90)
print("AllBenchs — 8 Parallel Benchmarks")
print("=" * 90)
print("D3Q19 MRT+Smag(Cs=0.05) + wallfn log-law + farfield BC")
print("SDAA cards: 0,4,8,12,16,20,24,28 (split across P2P groups)")
print()

# Clean previous results
for _, case in configs:
    out_path = OUTPUT_DIR / f"{case}.json"
    out_path.unlink(missing_ok=True)

# Launch workers
procs = []
for did, case in configs:
    out_path = OUTPUT_DIR / f"{case}.json"
    log_file = LOG_DIR / f"worker_{did}_{case}.log"

    cmd = [
        sys.executable, str(WORKER),
        str(did), case, str(out_path),
    ]
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

# ── Collect results ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("RESULTS: AllBenchs — 8 Parallel Benchmarks")
print("=" * 90)

results = []
for did, case, p, out_path, log_file in procs:
    if out_path.exists():
        r = json.loads(out_path.read_text())
        results.append(r)
    else:
        log_tail = ""
        if log_file.exists():
            log_tail = log_file.read_text()[-500:]
        print(f"{case} on SDAA:{did} — NO RESULT FILE")
        if log_tail:
            print(f"  log tail: ...{log_tail[-200:]}")
        results.append({
            "case": case,
            "device": f"sdaa:{did}",
            "error": "no result file",
            "status": "FAIL",
        })

# Print results table sorted by Cd_error_pct
print(f"\n{'Case':<22} {'Cd_mean':>10} {'Cd_ref':>10} {'Cd_err%':>8} "
      f"{'Cl_mean':>10} {'Cl_ref':>10} {'Cl_err%':>8} {'xr/h':>8} {'Std':>8} "
      f"{'Steps':>7} {'Time':>8} {'Status':>6}")
print("-" * 130)

def _sort_key(r):
    err = r.get("Cd_error_pct", float("inf"))
    if not isinstance(err, (int, float)) or not (lambda x: x == x)(err):
        return float("inf")
    return err

within_15 = []
for r in sorted(results, key=_sort_key):
    case_name = r.get("case", "?")
    if "error" in r:
        print(f"{case_name:<22} {r['error']}")
        continue

    cd = r.get("Cd_mean", float("nan"))
    cd_s = f"{cd:.4f}" if isinstance(cd, (int, float)) and cd == cd else "DIV"
    ref_cd = r.get("Cd_ref", float("nan"))
    ref_s = f"{ref_cd:.4f}" if isinstance(ref_cd, (int, float)) and ref_cd == ref_cd else "?"
    err_cd = r.get("Cd_error_pct", float("nan"))
    err_s = f"{err_cd:.1f}%" if isinstance(err_cd, (int, float)) and err_cd == err_cd else "N/A"

    cl = r.get("Cl_mean") or float("nan")
    cl_s = f"{cl:.4f}" if isinstance(cl, (int, float)) and cl == cl else "-"
    ref_cl = r.get("Cl_ref") or float("nan")
    ref_cl_s = f"{ref_cl:.4f}" if isinstance(ref_cl, (int, float)) and ref_cl == ref_cl else "-"
    err_cl = r.get("Cl_error_pct") or float("nan")
    err_cl_s = f"{err_cl:.1f}%" if isinstance(err_cl, (int, float)) and err_cl == err_cl else "-"

    xr = r.get("xr_h") or float("nan")
    xr_s = f"{xr:.2f}" if isinstance(xr, (int, float)) and xr == xr else "-"

    std = r.get("Cd_std", 0)
    std_s = f"{std:.4f}" if isinstance(std, (int, float)) else "?"

    ts = r.get("elapsed_s", 0)
    ts_s = f"{ts:.0f}s" if ts else "?"

    status = r.get("status", "?")
    if status == "OK":
        status_s = "✓ OK"
    elif status == "DIV":
        status_s = "✗ DIV"
    else:
        status_s = status

    print(f"{case_name:<22} {cd_s:>10} {ref_s:>10} {err_s:>8} "
          f"{cl_s:>10} {ref_cl_s:>10} {err_cl_s:>8} {xr_s:>8} "
          f"{std_s:>8} {r.get('cd_samples',0):>7} {ts_s:>8} {status_s:>6}")

    # Check if within 15% of reference
    if isinstance(err_cd, (int, float)) and err_cd == err_cd and err_cd <= 15.0:
        within_15.append(case_name)

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Within 15% error: {len(within_15)}/{len(results)} — {within_15 if within_15 else 'NONE'}")

# Sort final results by error%
sorted_results = sorted(results, key=lambda x: (
    x.get("Cd_error_pct", float("inf"))
    if isinstance(x.get("Cd_error_pct"), (int, float)) and x.get("Cd_error_pct") == x.get("Cd_error_pct")
    else float("inf")
))

combined = {
    "title": "AllBenchs — 8 Parallel Benchmarks",
    "setup": "D3Q19 MRT+Smag(Cs=0.05) + wallfn log-law + farfield BC",
    "sliding_window": 300,
    "n_steps_per_case": 2000,
    "sdda_cards": [0, 4, 8, 12, 16, 20, 24, 28],
    "within_15_pct": within_15,
    "results": sorted_results,
}

output_file = Path("/tmp/all_bench_results.json")
output_file.write_text(json.dumps(combined, indent=2))
print(f"\nCombined results saved to: {output_file}")
