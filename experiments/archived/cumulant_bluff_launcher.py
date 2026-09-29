"""Launcher: CUMULANT D3Q27 bluff body benchmarks on SDAA 28-31.

Tests:
  SDAA:28 — Cylinder Re=200, D=24, 200×80×4, 3000 steps
  SDAA:29 — Cylinder Re=500, D=24, 200×80×4, 3000 steps
  SDAA:30 — Sphere   Re=1000, D=24, 120×60×60, 3000 steps
  SDAA:31 — Sphere   Re=10000, D=24, 120×60×60, 3000 steps

Compares against D3Q19 MRT+Smag results for Cd accuracy and vortex shedding.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "cumulant_bluff_worker.py"
LOG_DIR = Path("/tmp/cumulant_bluff_logs")
OUTPUT_DIR = Path("/tmp")
LOG_DIR.mkdir(exist_ok=True)

configs = [
    # (device_id, case, re, n_steps, warmup, output_json)
    (28, "cylinder", 200.0,   3000, 500, "/tmp/cumulant_cylinder_re200.json"),
    (29, "cylinder", 500.0,   3000, 500, "/tmp/cumulant_cylinder_re500.json"),
    (30, "sphere",   1000.0,  3000, 500, "/tmp/cumulant_sphere_re1000.json"),
    (31, "sphere",   10000.0, 3000, 500, "/tmp/cumulant_sphere_re10000.json"),
]

# D3Q19 MRT+Smag reference results (from earlier runs)
D3Q19_REF = {
    "cylinder_Re200":  {"Cd_mean": 1.20, "Cd_std": 0.0,   "Cd_ref": 1.30, "error_pct": 8.0,   "vortex_shedding": False},
    "cylinder_Re500":  {"Cd_mean": 0.47, "Cd_std": 0.0,   "Cd_ref": 1.20, "error_pct": 61.0,  "vortex_shedding": False},
    "sphere_Re1000":   {"Cd_mean": 0.80, "Cd_std": 0.0,   "Cd_ref": 0.47, "error_pct": 71.0,  "vortex_shedding": False},
    "sphere_Re10000":  {"Cd_mean": 0.76, "Cd_std": 0.0,   "Cd_ref": 0.40, "error_pct": 90.0,  "vortex_shedding": False},
}

print("=" * 90)
print("CUMULANT D3Q27 Bluff Body Benchmark — Cylinder & Sphere")
print("=" * 90)
print(f"D3Q27 CUMULANT+Smag(Cs=0.05) + wall_function_d3q27 + far_field_bc_27")
print(f"Cylinder: D=24, 200×80×4  |  Sphere: D=24, 120×60×60")
print(f"3000 steps, warmup=500  |  SDAA cards: 28-31")
print()

# Clean previous results
for _, _, _, _, _, out_path in configs:
    Path(out_path).unlink(missing_ok=True)

# Launch workers
procs = []
for did, case, re, n_steps, warmup, out_path in configs:
    log_file = LOG_DIR / f"worker_{did}_{case}_re{int(re)}.log"
    cmd = [
        sys.executable, str(WORKER),
        str(did), case, str(re), str(n_steps), str(warmup), str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, case, re, p))
    print(f"Launched SDAA:{did} {case} Re={int(re)} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} launched. Logs: {LOG_DIR}", flush=True)

# Wait for all
t0 = time.time()
while True:
    done = sum(1 for _, _, _, p in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} done", flush=True)
    if done == len(procs):
        break
    time.sleep(30)

# Collect results
print("\n" + "=" * 90)
print("RESULTS: CUMULANT D3Q27 Bluff Body Drag")
print("=" * 90)

results = []
comparison = []
for did, case, re, p in procs:
    key = f"{case}_Re{int(re)}"
    out_path = f"/tmp/cumulant_{case}_re{int(re)}.json"
    rf = Path(out_path)
    if rf.exists():
        r = json.loads(rf.read_text())
        results.append(r)
        d3q19_ref = D3Q19_REF.get(key, {})
        comparison.append({
            "case": key,
            "D3Q27_Cd_mean": r.get("Cd_mean"),
            "D3Q27_Cd_std": r.get("Cd_std"),
            "D3Q27_error_pct": r.get("error_pct"),
            "D3Q27_vortex_shedding": r.get("vortex_shedding"),
            "D3Q19_Cd_mean": d3q19_ref.get("Cd_mean"),
            "D3Q19_Cd_std": d3q19_ref.get("Cd_std"),
            "D3Q19_error_pct": d3q19_ref.get("error_pct"),
            "D3Q19_vortex_shedding": d3q19_ref.get("vortex_shedding"),
            "Cd_ref": r.get("Cd_ref"),
        })
    else:
        log = LOG_DIR / f"worker_{did}_{case}_re{int(re)}.log"
        log_tail = ""
        if log.exists():
            log_tail = log.read_text()[-500:]
        print(f"SDAA:{did} {case} Re={int(re)} — NO RESULT FILE")
        if log_tail:
            print(f"  log tail: ...{log_tail[-200:]}")
        results.append({
            "case": key, "device": f"sdaa:{did}", "Re": re,
            "error": "no result file",
        })
        comparison.append({
            "case": key, "error": "no result file",
        })

# Print comparison table
print(f"\n{'Case':<22} {'Collision':<30} {'Cd_mean':>8} {'Cd_std':>8} {'Cd_ref':>8} {'Err%':>7} {'Shed?':>6}")
print("-" * 95)

# D3Q19 ref rows
for key, ref in D3Q19_REF.items():
    cm = ref["Cd_mean"]; cs = ref["Cd_std"]
    cm_s = f"{cm:.4f}" if isinstance(cm, (int, float)) and cm == cm else "DIV"
    cs_s = f"{cs:.4f}" if isinstance(cs, (int, float)) and cs == cs else "?"
    ref_cd = ref.get("Cd_ref", "?")
    ref_s = f"{ref_cd:.2f}" if isinstance(ref_cd, (int, float)) else "?"
    print(f"{key:<22} {'D3Q19 MRT+Smag':<30} {cm_s:>8} {cs_s:>8} {ref_s:>8} {ref['error_pct']:>6.1f}% {'NO':>6}")

# D3Q27 CUMULANT rows
for r in sorted(results, key=lambda x: x.get("case", "")):
    if "error" in r:
        print(f"{r['case']:<22} {'CUMULANT+Smag':<30} {'FAILED':>8}")
        continue
    cm = r.get("Cd_mean", float("nan"))
    cd = r.get("Cd_std", 0)
    cm_s = f"{cm:.4f}" if isinstance(cm, (int, float)) and cm == cm else "DIV"
    cd_s = f"{cd:.4f}" if isinstance(cd, (int, float)) and cd == cd else "?"
    err = r.get("error_pct", float("nan"))
    err_s = f"{err:.1f}" if isinstance(err, (int, float)) and err == err else "N/A"
    ref_cd = r.get("Cd_ref", "?")
    ref_s = f"{ref_cd:.2f}" if isinstance(ref_cd, (int, float)) else "?"
    shed = "YES" if r.get("vortex_shedding") else "NO"
    print(f"{r['case']:<22} {'D3Q27 CUMULANT+Smag':<30} {cm_s:>8} {cd_s:>8} {ref_s:>8} {err_s:>6}% {shed:>6}")

# Save combined results
combined = {
    "title": "CUMULANT D3Q27 Bluff Body Benchmark — Cylinder & Sphere Drag",
    "hypothesis": "Non-dissipative CUMULANT preserves vortex shedding better than MRT+Smagorinsky",
    "setup": "D3Q27 CUMULANT+Smag(Cs=0.05) + wall_function_d3q27 + far_field_bc_27",
    "geometry": "Cylinder D=24 (200×80×4), Sphere D=24 (120×60×60)",
    "conditions": "3000 steps, warmup=500, running average Cd post-warmup",
    "d3q19_reference": D3Q19_REF,
    "d3q27_results": results,
    "comparison": comparison,
    "summary": "Cd_std > 0.005 indicates vortex shedding detected",
}
output_file = Path("/tmp/cumulant_bluff_results.json")
output_file.write_text(json.dumps(combined, indent=2))
print(f"\nResults saved to: {output_file}")
