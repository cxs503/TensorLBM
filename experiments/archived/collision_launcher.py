"""Launch collision operator comparison on bare_hull + wall_function."""
import json, os, subprocess, sys, time
from pathlib import Path

TEST_DIR = Path("/tmp/collision_test")
TEST_DIR.mkdir(exist_ok=True)
for f in TEST_DIR.glob("*.json"): f.unlink()

collisions = ["MRT+Smag", "BGK+Smag", "CUMULANT", "KBC", "RLBM", "TRT", "CM"]
grids = [
    (160, 64, 64, 64.0, 1500),
    (200, 80, 80, 80.0, 1500),
]
Cs_for = lambda c: 0.05 if "Smag" in c else 0.0

configs = []
did = 0
for col in collisions:
    for nx, ny, nz, hl, ns in grids:
        configs.append((did, col, nx, ny, nz, hl, ns, Cs_for(col)))
        did += 1

print(f"Launching {len(configs)} workers")
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

t0 = time.time()
while True:
    time.sleep(30)
    done = sum(1 for _, _, p in procs if p.poll() is not None)
    print(f"[{time.time()-t0:.0f}s] {done}/{len(procs)}", flush=True)
    if done == len(procs):
        break

results = []
for did, name, p in procs:
    rf = TEST_DIR / f"result_{did:02d}.json"
    if rf.exists():
        results.append(json.loads(rf.read_text()))
    else:
        print(f"[{did:02d}] {name} — NO RESULT")
        results.append({"collision": name, "error": "no result"})

results.sort(key=lambda r: (r.get("grid",""), r.get("Ct_total",999)))

print("\n" + "=" * 90)
print("COLLISION OPERATOR COMPARISON — bare_hull wall_function")
print("=" * 90)
print(f"{'Collision':<12} {'Grid':<14} {'Cs':<6} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Err%':<8} {'OK':<5}")
print("-" * 90)
for r in results:
    if "error" in r:
        print(f"{r['collision']:<12} {'?':<14} {'?':<6} ERROR: {r['error']}")
    else:
        ct = r["Ct_total"]
        err = r.get("error_pct", abs(ct - 0.00405) / 0.00405 * 100)
        print(f"{r['collision']:<12} {r['grid']:<14} {r.get('Cs',0):<6} "
              f"{r['Ct_fric']:<10.5f} {r['Ct_pres']:<10.5f} {ct:<10.5f} {err:<8.1f} {'✓' if r['finite'] else '✗'}")

Path("collision_comparison_results.json").write_text(json.dumps(results, indent=2))
