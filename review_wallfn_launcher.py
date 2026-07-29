"""Launch 32 wall_function convergence tests across all SDAA cards."""
import os, subprocess, sys, time, json
from pathlib import Path

TEST_DIR = Path("/tmp/wallfn_review32")
TEST_DIR.mkdir(exist_ok=True)

# Clear old results
for f in TEST_DIR.glob("*.json"):
    f.unlink()

# Test matrix: 4 grids × 2 hull types × 4 C_s = 32 configs
grids = [
    (160, 64, 64, 64.0),
    (200, 80, 80, 80.0),
    (256, 96, 96, 96.0),
    (320, 128, 128, 128.0),
]
hull_types = ["bare_hull", "full"]
cs_vals = [0.05, 0.10, 0.15, 0.20]

configs = []
did = 0
for nx, ny, nz, hl in grids:
    hull_steps = 1500 if nx <= 256 else 2000
    for hull in hull_types:
        for cs in cs_vals:
            configs.append((did, nx, ny, nz, hl, hull, cs, hull_steps))
            did += 1

print(f"Launching {len(configs)} workers on {len(configs)} SDAA cards")
print(f"Grids: {[g[0] for g in grids]}")
print(f"Hulls: {hull_types}, C_s: {cs_vals}")
print()

WORKER = Path(__file__).parent / "_wallfn_worker32.py"
env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent / "src")

procs = []
for did, nx, ny, nz, hl, hull, cs, ns in configs:
    name = f"{hull[:1]}{nx}x{ny}x{nz}_{cs}"
    log = TEST_DIR / f"worker_{did:02d}.log"
    cmd = [sys.executable, str(WORKER),
           str(did), str(nx), str(ny), str(nz), str(hl), hull, str(cs), str(ns)]
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, name, p))
    print(f"[{did:02d}] {name} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} launched. Logs: {TEST_DIR}/worker_*.log", flush=True)

t0 = time.time()
while True:
    time.sleep(30)
    done = sum(1 for _, _, p in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} done", flush=True)
    if done == len(procs):
        break

# Collect results
results = []
for did, name, p in procs:
    rf = TEST_DIR / f"result_{did:02d}.json"
    if rf.exists():
        results.append(json.loads(rf.read_text()))
    else:
        log = TEST_DIR / f"worker_{did:02d}.log"
        tail = ""
        if log.exists():
            lines = log.read_text().splitlines()
            tail = "\n".join(lines[-5:]) if len(lines) > 5 else "\n".join(lines)
        print(f"[{did:02d}] {name} — NO RESULT")
        if tail:
            print(f"  tail: {tail[-200:]}")
        results.append({"name": name, "error": "no result"})

# Sort and summarize
results.sort(key=lambda r: (r.get("hull", ""), r.get("grid", ""), r.get("Cs", 0)))

print("\n" + "=" * 100)
print("WALL FUNCTION CONVERGENCE MATRIX")
print("=" * 100)
ok = [r for r in results if "Ct_total" in r]
if ok:
    hulls = sorted(set(r["hull"] for r in ok))
    for hull in hulls:
        print(f"\n--- {hull} ---")
        print(f"{'Grid':<14} {'Cs':<6} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Err%':<8} {'Steps':<8} {'OK':<5}")
        print("-" * 80)
        for r in ok:
            if r["hull"] != hull:
                continue
            ct = r["Ct_total"]
            err = abs(ct - 0.00405) / 0.00405 * 100
            print(f"{r['grid']:<14} {r['Cs']:<6} {r['Ct_fric']:<10.5f} {r['Ct_pres']:<10.5f} {ct:<10.5f} {err:<8.1f} {r['steps']:<8} {'✓' if r['finite'] else '✗'}")
else:
    print("\nAll failed!")

Path("wallfn_review32_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: wallfn_review32_results.json")
