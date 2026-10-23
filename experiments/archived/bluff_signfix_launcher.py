"""Launcher: spawn cylinder (Re=100/200/500) + sphere (Re=100/1000/10000)
benchmarks on SDAA:22-27 in parallel with sign-fixed pressure drag code.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER_DIR = Path(__file__).parent
CYLINDER_WORKER = WORKER_DIR / "cylinder_worker.py"
SPHERE_WORKER = WORKER_DIR / "sphere_worker.py"
LOG_DIR = Path("/tmp/bluff_signfix_logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Configurations ────────────────────────────────────────────────────────
# (sdaa_id, worker_script, Re, n_steps, warmup, output_json_path, case_name)
configs = [
    # Cylinders: 200×80×4, D=24, Cs=0.05
    (22, CYLINDER_WORKER, 100.0,  2000, 500, "/tmp/signfix_cylinder_re100.json",  "cylinder_Re100"),
    (23, CYLINDER_WORKER, 200.0,  2000, 500, "/tmp/signfix_cylinder_re200.json",  "cylinder_Re200"),
    (24, CYLINDER_WORKER, 500.0,  2000, 500, "/tmp/signfix_cylinder_re500.json",  "cylinder_Re500"),
    # Spheres: 120×60×60, D=24, Cs=0.05
    (25, SPHERE_WORKER,   100.0,  2000, 500, "/tmp/signfix_sphere_re100.json",    "sphere_Re100"),
    (26, SPHERE_WORKER,   1000.0, 2000, 500, "/tmp/signfix_sphere_re1000.json",   "sphere_Re1000"),
    (27, SPHERE_WORKER,   10000.0,2000, 500, "/tmp/signfix_sphere_re10000.json",  "sphere_Re10000"),
]

print("=" * 90)
print("Bluff Body Sign-Fix Re-Test — Cylinder & Sphere")
print("=" * 90)
print("D3Q19 MRT+Smag(Cs=0.05) + wall_function_3d + farfield")
print("Cylinder: D=24, 200×80×4, 2000 steps, warmup=500")
print("Sphere:   D=24, 120×60×60, 2000 steps, warmup=500")
print("SDAA cards: 22-27")
print()

# Clean previous results
for _, _, _, _, _, out_path, _ in configs:
    Path(out_path).unlink(missing_ok=True)

# Launch workers
procs = []
for did, worker, re, n_steps, warmup, out_path, case_name in configs:
    log_file = LOG_DIR / f"worker_{did}_{case_name}.log"
    cmd = [
        sys.executable, str(worker),
        str(did), str(re), str(n_steps), str(warmup), str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKER_DIR / "src")
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, re, case_name, p, out_path, log_file))
    print(f"Launched {case_name} on SDAA:{did} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} workers launched. Logs: {LOG_DIR}", flush=True)

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
print("RESULTS: Bluff Body Sign-Fix Re-Test")
print("=" * 90)

# Buggy results for comparison (from pre-sign-fix runs)
buggy_results = {
    "cylinder_Re100":  {"Cd_mean": 2.3873, "Cd_ref": 1.35},
    "cylinder_Re200":  {"Cd_mean": 1.1952, "Cd_ref": 1.30},
    "cylinder_Re500":  {"Cd_mean": 0.4684, "Cd_ref": 1.20},
    # No previous sphere results with wallfn
}

results = []
for did, re, case_name, p, out_path, log_file in procs:
    rf = Path(out_path)
    if rf.exists():
        r = json.loads(rf.read_text())
        # Add buggy comparison
        bug = buggy_results.get(case_name)
        if bug:
            r["Cd_buggy"] = bug["Cd_mean"]
            r["Cd_change"] = r["Cd_mean"] - bug["Cd_mean"]
        results.append(r)
    else:
        log_tail = ""
        if log_file.exists():
            log_tail = log_file.read_text()[-500:]
        print(f"{case_name} on SDAA:{did} — NO RESULT FILE")
        if log_tail:
            print(f"  log tail: ...{log_tail[-200:]}")
        results.append({
            "case": case_name,
            "device": f"sdaa:{did}",
            "Re": re,
            "error": "no result file",
        })

# Print results table
print(f"\n{'Case':<22} {'Cd_mean':>10} {'Cd_buggy':>10} {'ΔCd':>10} {'Cd_ref':>10} {'Err%':>8} {'Std':>10} {'Steps':>8} {'Time':>8} {'OK':>4}")
print("-" * 106)
for r in sorted(results, key=lambda x: (x.get("case", ""))):
    if "error" in r:
        print(f"{r['case']:<22} {r['error']}")
    else:
        cd = r.get("Cd_mean", float("nan"))
        cd_s = f"{cd:.4f}" if isinstance(cd, (int, float)) and math.isfinite(cd) else "DIV"
        cd_bug = r.get("Cd_buggy", float("nan"))
        cd_bug_s = f"{cd_bug:.4f}" if isinstance(cd_bug, (int, float)) and math.isfinite(cd_bug) else "-"
        delta = r.get("Cd_change", float("nan"))
        delta_s = f"{delta:+.4f}" if isinstance(delta, (int, float)) and math.isfinite(delta) else "-"
        ref = r.get("Cd_ref", float("nan"))
        ref_s = f"{ref:.4f}" if isinstance(ref, (int, float)) and math.isfinite(ref) else "?"
        err = r.get("error_pct", float("nan"))
        err_s = f"{err:.1f}%" if isinstance(err, (int, float)) and math.isfinite(err) else "N/A"
        std = r.get("Cd_std", 0)
        std_s = f"{std:.4f}" if isinstance(std, (int, float)) else "?"
        ts = r.get("elapsed_s", 0)
        ts_s = f"{ts:.0f}s" if ts else "?"
        print(f"{r['case']:<22} {cd_s:>10} {cd_bug_s:>10} {delta_s:>10} {ref_s:>10} {err_s:>8} "
              f"{std_s:>10} {r.get('cd_samples', 0):>8} {ts_s:>8} "
              f"{'✓' if r.get('finite', False) else '✗'}")

# Cylinder reference values
cyl_ref = {100: 1.35, 200: 1.30, 500: 1.20}

# Save combined results
combined = {
    "title": "Bluff Body Sign-Fix Re-Test — Cylinder & Sphere",
    "setup": "D3Q19 MRT+Smag(Cs=0.05) + wall_function_3d + farfield",
    "cylinder": {"domain": "200×80×4", "D": 24},
    "sphere": {"domain": "120×60×60", "D": 24},
    "conditions": "2000 steps, warmup=500, sliding-window average (last 500)",
    "cylinder_reference_cd": {str(k): v for k, v in cyl_ref.items()},
    "sphere_reference_cd": "Schiller-Naumann: 24/Re*(1+0.15*Re^0.687)",
    "sign_fix_note": "Pressure drag sign corrected in wall_function_3d drag_pres computation",
    "buggy_baseline": buggy_results,
    "results": results,
}
output_file = Path("/tmp/bluff_signfix_results.json")
output_file.write_text(json.dumps(combined, indent=2))
print(f"\nResults saved to: {output_file}")
