"""Domain sensitivity study — test if larger far-field domain improves drag accuracy.

All D3Q19 MRT+Smag Cs=0.05 bare_hull, 2000 steps each.
Test matrix:
  1. 200³ hull=80  (hull=40% domain, blockage ~6.4%) — baseline
  2. 256³ hull=64  (hull=25% domain, blockage ~1.6%)
  3. 320³ hull=80  (hull=25% domain, blockage ~1.6%)
"""
import json, os, subprocess, sys, time
from pathlib import Path

TEST_DIR = Path("/tmp/domain_sensitivity")
TEST_DIR.mkdir(exist_ok=True)
# Clean old results
for f in TEST_DIR.glob("*.json"):
    f.unlink()
for f in TEST_DIR.glob("*.log"):
    f.unlink()

configs = [
    # (device_id, nx, ny, nz, hull_length, n_steps, label)
    (0, 200, 200, 200, 80.0, 2000, "200³_h80_b40pct"),
    (1, 256, 256, 256, 64.0, 2000, "256³_h64_b25pct"),
    (2, 320, 320, 320, 80.0, 2000, "320³_h80_b25pct"),
]

WORKER = Path(__file__).parent / "_matrix_worker.py"
env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent / "src")

procs = []
for did, nx, ny, nz, hl, ns, label in configs:
    log = TEST_DIR / f"worker_{did:02d}.log"
    cmd = [
        sys.executable, str(WORKER),
        str(did),           # device_id
        "D3Q19",            # lattice
        "MRT+Smag",         # collision
        "0.05",             # Cs
        str(nx), str(ny), str(nz),  # grid
        str(hl),            # hull_length
        "bare_hull",        # hull_type
        str(ns),            # n_steps
    ]
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    procs.append((did, label, p, log))
    print(f"[{did:02d}] {label} nx={nx} hl={hl} (PID={p.pid})", flush=True)

print(f"\nLaunched {len(procs)} workers. Waiting for completion...\n", flush=True)

t0 = time.time()
while True:
    time.sleep(30)
    done = sum(1 for _, _, p, _ in procs if p.poll() is not None)
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {done}/{len(procs)} workers done", flush=True)
    if done == len(procs):
        break

# Collect results — _matrix_worker.py writes to /tmp/matrix_final/result_{did:02d}.json
results = []
for did, label, p, log in procs:
    rf = Path(f"/tmp/matrix_final/result_{did:02d}.json")
    if rf.exists():
        r = json.loads(rf.read_text())
        r["label"] = label
        # Compute blockage ratio
        nx_v = int(r["grid"].split("x")[0])
        hl_v = float(r.get("hull", "bare_hull"))  # hull field is the type string
        # Actually compute blockage from grid and hull_length
        # Volume blockage ≈ (hull_length/nx)^3  (ratio of hull to domain in each dimension)
        blockage_pct = (float(sys.argv) if False else 0)  # placeholder
        results.append(r)
        print(f"[{did:02d}] {label}: Ct={r['Ct_total']:.5f} err={r['error_pct']:.1f}% OK={r['finite']}")
    else:
        print(f"[{did:02d}] {label}: NO RESULT FILE (check {log})")
        # Show tail of log
        if log.exists():
            tail = log.read_text()[-500:]
            print(f"  Log tail: {tail}")

# Enrich with metadata
for r, (did, nx, ny, nz, hl, ns, label) in zip(results, configs):
    r["label"] = label
    r["domain_size"] = nx
    r["hull_length"] = hl
    r["hull_pct_of_domain"] = round(hl / nx * 100, 1)
    r["blockage_vol_pct"] = round((hl / nx) ** 3 * 100, 2)

out_path = Path("/tmp/domain_sensitivity.json")
out_path.write_text(json.dumps(results, indent=2))
print(f"\nResults written to {out_path}")
print(json.dumps(results, indent=2))
