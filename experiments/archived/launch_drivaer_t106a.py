#!/usr/bin/env python3
"""Launcher: DrivAer + T106A benchmarks on SDAA cards 8-11.

Runs 4 parallel workers:
  SDAA 8  — DrivAer simplified (Re=1000)
  SDAA 9  — DrivAer simplified (Re=1000) [cross-validation]
  SDAA 10 — T106A cascade (Re=1000)
  SDAA 11 — T106A cascade (Re=1000) [cross-validation]

Worker: drivaer_t106a_worker.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).parent
WORKER = WORKER_DIR / "drivaer_t106a_worker.py"
LOG_DIR = WORKER_DIR
OUTPUT_DIR = WORKER_DIR

# ── Configuration: (sdaa_id, benchmark, tag) ───────────────────────────────
configs = [
    (8,  "drivaer", "drivaer_re1000_sdaa8"),
    (9,  "drivaer", "drivaer_re1000_sdaa9"),
    (10, "t106a",   "t106a_re1000_sdaa10"),
    (11, "t106a",   "t106a_re1000_sdaa11"),
]

print("=" * 90)
print("DrivAer + T106A Cascade Benchmarks — 4 Parallel Workers")
print("=" * 90)
print("D3Q19 MRT+Smag(Cs=0.05) + farfield BC + from_gradient normals")
print("SDAA cards: 8, 9, 10, 11")
print()

# Clean previous results
for _, _, tag in configs:
    out_path = OUTPUT_DIR / f"{tag}.json"
    out_path.unlink(missing_ok=True)

# Launch workers
procs = []
for did, bench, tag in configs:
    out_path = OUTPUT_DIR / f"{tag}.json"
    log_file = LOG_DIR / f"log_{tag}.txt"

    cmd = [
        sys.executable, str(WORKER),
        bench, str(did), str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKER_DIR / "src")

    with open(log_file, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((did, bench, tag, p, out_path, log_file))
    print(f"Launched {bench} on SDAA:{did} (PID={p.pid}) → {out_path.name}", flush=True)

print(f"\nAll {len(procs)} workers launched. Logs: {LOG_DIR}/log_*.txt", flush=True)

# Wait for all
t0 = time.time()
while True:
    done = sum(1 for _, _, _, p, _, _ in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} done", flush=True)
    if done == len(procs):
        break
    time.sleep(30)

# ── Collect results ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)

all_results = {}
for did, bench, tag, p, out_path, log_file in procs:
    rc = p.returncode
    if out_path.exists():
        result = json.loads(out_path.read_text())
        all_results[tag] = result
        if bench == "drivaer":
            print(
                f"  {tag}: Cd_p={result['Cd_pressure']:.4f} "
                f"Cd_f={result['Cd_friction']:.4f} "
                f"Cd_tot={result['Cd_total']:.4f} "
                f"Cl={result['Cl']:.6f} "
                f"(ref={result['Cd_ref']:.4f}) "
                f"err={result['error_pct']:.1f}% "
                f"rc={rc} time={result['elapsed_s']:.0f}s"
            )
        else:  # t106a
            cp_loss = result.get("pressure_loss", "N/A")
            turn = result.get("turning_angle_deg", "N/A")
            cp_str = f"{cp_loss:.4f}" if isinstance(cp_loss, float) else str(cp_loss)
            turn_str = f"{turn:.2f}" if isinstance(turn, float) else str(turn)
            print(
                f"  {tag}: Cd_tot={result['Cd_total']:.4f} "
                f"pressure_loss={cp_str} "
                f"turning_angle={turn_str}deg "
                f"rc={rc} time={result['elapsed_s']:.0f}s"
            )
    else:
        print(f"  {tag}: FAILED (rc={rc}), no output file")

# Cross-validation summary
print("\n" + "-" * 90)
print("CROSS-VALIDATION")
print("-" * 90)

drivaer_results = {tag: r for tag, r in all_results.items() if "drivaer" in tag}
if len(drivaer_results) >= 2:
    cds = [r["Cd_total"] for r in drivaer_results.values()]
    cls = [r["Cl"] for r in drivaer_results.values()]
    cd_mean = sum(cds) / len(cds)
    cl_mean = sum(cls) / len(cls)
    cd_spread = max(cds) - min(cds) if len(cds) > 1 else 0
    print(
        f"  DrivAer: Cd_tot mean={cd_mean:.4f} (spread={cd_spread:.4f}) "
        f"Cl mean={cl_mean:.6f} (n={len(drivaer_results)})"
    )

t106a_results = {tag: r for tag, r in all_results.items() if "t106a" in tag}
if len(t106a_results) >= 2:
    cps = [r.get("pressure_loss", 0) for r in t106a_results.values()]
    turns = [r.get("turning_angle_deg", 0) for r in t106a_results.values()]
    cp_mean = sum(cps) / len(cps)
    turn_mean = sum(turns) / len(turns)
    cp_spread = max(cps) - min(cps) if len(cps) > 1 else 0
    print(
        f"  T106A:   pressure_loss mean={cp_mean:.4f} (spread={cp_spread:.4f}) "
        f"turning mean={turn_mean:.2f}deg (n={len(t106a_results)})"
    )

# Write combined summary
summary_path = OUTPUT_DIR / "drivaer_t106a_summary.json"
summary = {
    "drivaer": {tag: r for tag, r in all_results.items() if "drivaer" in tag},
    "t106a": {tag: r for tag, r in all_results.items() if "t106a" in tag},
}
summary_path.write_text(json.dumps(summary, indent=2))
print(f"\nCombined summary: {summary_path}")
