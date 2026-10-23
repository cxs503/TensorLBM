"""Launch all 7 collision operators on 3 large grids with far-field BC."""
import json, os, subprocess, sys, time
from pathlib import Path

TEST_DIR = Path("/tmp/collision_large")
TEST_DIR.mkdir(exist_ok=True)
for f in TEST_DIR.glob("*.json"): f.unlink()
for f in TEST_DIR.glob("*.log"): f.unlink()

collisions = ["MRT+Smag", "BGK+Smag", "CUMULANT", "KBC", "RLBM", "TRT", "CM"]
grids = [
    (200, 80, 80, 80.0, 1500),   # 0.2s/step, ~300s total
    (256, 96, 96, 96.0, 2000),   # ~0.5s/step, ~1000s total
    (320, 128, 128, 128.0, 2500),# ~1.2s/step, ~3000s total
]
Cs_for = lambda c: 0.05 if "Smag" in c else 0.0

configs = []
did = 0
for col in collisions:
    for nx, ny, nz, hl, ns in grids:
        configs.append((did, col, nx, ny, nz, hl, ns, Cs_for(col)))
        did += 1

print(f"Launching {len(configs)} workers on {len(configs)} SDAA cards")
print(f"Grids: {[(g[0], g[3]) for g in grids]}  Collisions: {collisions}")
print()

WORKER = Path(__file__).parent / "_collision_worker.py"
env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent / "src")

procs = []
for did, col, nx, ny, nz, hl, ns, cs in configs:
    log = TEST_DIR / f"worker_{did:02d}.log"
    name = f"{col:<10} {nx}³"
    cmd = [sys.executable, str(WORKER), str(did), col, str(nx), str(ny), str(nz), str(hl), str(ns), str(cs)]
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, name, p))
    print(f"[{did:02d}] {name} (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} launched. Logs: {TEST_DIR}", flush=True)

t0 = time.time()
while True:
    time.sleep(30)
    done = sum(1 for _, _, p in procs if p.poll() is not None)
    print(f"[{time.time()-t0:.0f}s] {done}/{len(procs)}", flush=True)
    if done == len(procs):
        break

# Results
results = []
for did, name, p in procs:
    rf = TEST_DIR / f"result_{did:02d}.json"
    if rf.exists():
        results.append(json.loads(rf.read_text()))
    else:
        results.append({"collision": name, "error": "no result"})

results.sort(key=lambda r: (r.get("grid",""), -r.get("Ct_total", -999)))

print("\n" + "=" * 100)
print("ALL 7 COLLISIONS × 3 LARGE GRIDS — bare_hull wall_function + far_field BC")
print("=" * 100)
for grid in ["200x80x80", "256x96x96", "320x128x128"]:
    hrs = [r for r in results if r.get("grid") == grid and "error" not in r]
    if not hrs: continue
    print(f"\n--- {grid} ---")
    print(f"{'Collision':<12} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Err%':<8} {'OK':<5}")
    print("-" * 70)
    for r in sorted(hrs, key=lambda x: -x.get("Ct_total", -999)):
        ct = r["Ct_total"]
        err = abs(ct - 0.00405) / 0.00405 * 100
        print(f"{r['collision']:<12} {r['Ct_fric']:<10.5f} {r['Ct_pres']:<10.5f} {ct:<10.5f} {err:<8.1f} {'✓' if r['finite'] else '✗'}")
    failed = [r for r in results if r.get("grid") == grid and "error" in r]
    for r in failed:
        print(f"{r['collision']:<12} ERROR: {r['error']}")

Path("collision_large_grid_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: collision_large_grid_results.json")
