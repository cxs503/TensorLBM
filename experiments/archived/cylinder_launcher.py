"""Launcher: spawn cylinder benchmarks at Re=100/200/500 on SDAA:27-29 in parallel."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "cylinder_worker.py"
LOG_DIR = Path("/tmp/cylinder_logs")
OUTPUT_DIR = Path("/tmp")
LOG_DIR.mkdir(exist_ok=True)

# Map each Re to a specific SDAA card
configs = [
    (27, 100.0, 2000, 500, "/tmp/cylinder_re100.json"),
    (28, 200.0, 2000, 500, "/tmp/cylinder_re200.json"),
    (29, 500.0, 2000, 500, "/tmp/cylinder_re500.json"),
]

print("=" * 90)
print("Cylinder Wall Function Breakdown Benchmark — Re=100, 200, 500")
print("=" * 90)
print(f"D3Q19 MRT+Smag Cs=0.05 + wallfn + farfield")
print(f"D=24, domain 200×80×4, 2000 steps, warmup=500")
print(f"SDAA cards: 27-29")
print()

# Clean previous results
for _, _, _, _, out_path in configs:
    Path(out_path).unlink(missing_ok=True)

# Launch workers
procs = []
for did, re, n_steps, warmup, out_path in configs:
    log_file = LOG_DIR / f"worker_{did}_re{int(re)}.log"
    cmd = [
        sys.executable, str(WORKER),
        str(did), str(re), str(n_steps), str(warmup), str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, re, p))
    print(f"Launched SDAA:{did} Re={int(re)} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} launched. Logs: {LOG_DIR}", flush=True)

# Wait for all
t0 = time.time()
while True:
    done = sum(1 for _, _, p in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} done", flush=True)
    if done == len(procs):
        break
    time.sleep(30)

# Collect results
print("\n" + "=" * 90)
print("SUMMARY: Cylinder Drag — Wall Function Breakdown")
print("=" * 90)

results = []
for did, re, p in procs:
    out_path = f"/tmp/cylinder_re{int(re)}.json"
    rf = Path(out_path)
    if rf.exists():
        r = json.loads(rf.read_text())
        results.append(r)
    else:
        log = (LOG_DIR / f"worker_{did}_re{int(re)}.log")
        log_tail = ""
        if log.exists():
            log_tail = log.read_text()[-500:]
        print(f"SDAA:{did} Re={int(re)} — NO RESULT FILE")
        if log_tail:
            print(f"  log tail: ...{log_tail[-200:]}")
        results.append({
            "case": f"cylinder_Re{int(re)}",
            "device": f"sdaa:{did}",
            "Re": re,
            "error": "no result file",
        })

# Print results table
print(f"\n{'Re':>6} {'Cd_mean':>10} {'Cd_ref':>10} {'Error%':>8} {'Std':>10} {'Steps':>10} {'Time':>8} {'OK':>4}")
print("-" * 72)
for r in sorted(results, key=lambda x: x.get("Re", 0)):
    if "error" in r:
        print(f"{r['Re']:>6.0f} {r['error']}")
    else:
        cd = r.get("Cd_mean", float("nan"))
        cd_s = f"{cd:.4f}" if isinstance(cd, (int, float)) and (cd == cd) else "DIV"
        err = r.get("error_pct", float("nan"))
        err_s = f"{err:.1f}" if isinstance(err, (int, float)) and (err == err) else "N/A"
        std = r.get("Cd_std", 0)
        std_s = f"{std:.4f}" if isinstance(std, (int, float)) else "?"
        ts = r.get("elapsed_s", 0)
        ts_s = f"{ts:.0f}s" if ts else "?"
        print(f"{r['Re']:>6.0f} {cd_s:>10} {r.get('Cd_ref', '?'):>10} {err_s:>8} "
              f"{std_s:>10} {r.get('cd_samples', 0):>10} {ts_s:>8} "
              f"{'✓' if r.get('finite') else '✗'}")

# Save combined results
combined = {
    "title": "Cylinder Wall Function Breakdown — Re sweep 100, 200, 500",
    "setup": "D3Q19 MRT+Smag(Cs=0.05) + wall_function_3d + farfield",
    "geometry": "2D cylinder D=24 lattice cells, extruded to nz=4",
    "domain": "200×80×4",
    "conditions": "2000 steps, warmup=500, running average",
    "reference_cd": {"100": 1.35, "200": 1.30, "500": 1.20},
    "results": results,
}
output_file = Path("/tmp/cylinder_results.json")
output_file.write_text(json.dumps(combined, indent=2))
print(f"\nResults saved to: {output_file}")
