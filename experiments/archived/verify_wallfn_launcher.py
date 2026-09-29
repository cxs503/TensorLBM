"""Launcher: spawn workers, ALL output goes to log files (no PIPE capture)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "verify_wallfn_worker.py"
LOG_DIR = Path("/tmp/wallfn_logs")
LOG_DIR.mkdir(exist_ok=True)

# Configs: (device_id, nx, ny, nz, hull_length, n_steps, use_bounce_back, C_s)
configs = []
device_id = 0
for nx, ny, nz, hl, ns in [(320, 120, 120, 120.0, 1500),
                            (480, 180, 180, 180.0, 2000)]:
    for C_s in [0.1, 0.05]:
        for bb in [True, False]:
            configs.append((device_id, nx, ny, nz, hl, ns, bb, C_s))
            device_id += 1

print("=" * 90)
print("SUBOFF Wall Function Double-Treatment Verification — MRT+Smagorinsky LES")
print("=" * 90)
print(f"Configs: {len(configs)}")
print()

# Clean
for did in range(16):
    Path(f"/tmp/wallfn_worker_{did}.json").unlink(missing_ok=True)
    (LOG_DIR / f"worker_{did}.log").unlink(missing_ok=True)

# Launch workers — output goes to log files
procs = []
for cfg in configs:
    did, nx, ny, nz, hl, ns, bb, cs = cfg
    log_file = LOG_DIR / f"worker_{did}.log"
    cmd = [sys.executable, str(WORKER),
           str(did), str(nx), str(ny), str(nz),
           str(hl), str(ns), str(bb), str(cs)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, nx, bb, cs, p))
    print(f"Launched SDAA:{did} {nx}³ Cs={cs} bb={bb} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} launched. Logs: {LOG_DIR}", flush=True)

t0 = time.time()
while True:
    done = sum(1 for _, _, _, _, p in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} done", flush=True)
    if done == len(procs): break
    time.sleep(30)

# Collect results
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

results = []
for did, nx, bb, cs, p in procs:
    rf = Path(f"/tmp/wallfn_worker_{did}.json")
    if rf.exists():
        r = json.loads(rf.read_text())
        results.append(r)
    else:
        log = (LOG_DIR / f"worker_{did}.log").read_text() if (LOG_DIR / f"worker_{did}.log").exists() else ""
        print(f"SDAA:{did} {nx}³ Cs={cs} bb={bb} — NO RESULT")
        if log: print(f"  tail: {log[-200:]}")
        results.append({"grid": f"{nx}³", "collision": f"MRT+Smag(Cs={cs})",
                        "bounce_back": bb, "error": "no result"})

results.sort(key=lambda r: (r.get("grid",""), -r.get("bounce_back",1), r.get("collision","")))

print(f"\n{'Grid':<18} {'Collision':<18} {'BB':<6} {'Ct_fric':<12} {'Ct_pres':<12} {'Ct_total':<12} {'Err%ITTC':<10} {'OK':<5}")
print("-" * 108)
for r in results:
    if "error" in r:
        print(f"{r['grid']:<18} {r['collision']:<18} {str(r['bounce_back']):<6} ERROR: {r['error']}")
    else:
        ct = r.get('Ct_total', float('nan'))
        ct_s = f"{ct:.6f}" if ct == ct else "DIV"
        err = r.get('error_pct_ITTC', float('nan'))
        err_s = f"{err:.1f}" if err == err else "N/A"
        print(f"{r['grid']:<18} {r['collision']:<18} {str(r['bounce_back']):<6} "
              f"{r['Ct_fric']:<12.6f} {r['Ct_pres']:<12.6f} {ct_s:<12} {err_s:<10} {'✓' if r.get('finite') else '✗'}")

Path("wallfn_verification_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: wallfn_verification_results.json")
