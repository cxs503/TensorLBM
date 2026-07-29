"""Launch 8 parallel collision operator comparison tests on SDAA cards 4-11.

Grid: 200³ for all tests. Re=2e6. Far-field + wall-function BC.
Results: /tmp/collision_comparison_long.json
"""

import json, os, subprocess, sys, time
from pathlib import Path

TEST_DIR = Path("/tmp/collision_comparison_long")
TEST_DIR.mkdir(exist_ok=True)
for f in TEST_DIR.glob("*.json"):
    f.unlink()
for f in TEST_DIR.glob("*.log"):
    f.unlink()

WORKER = Path(__file__).parent / "_collision_comparison_worker.py"
env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent / "src")

# 8 configurations: (did, lattice, collision, nx, ny, nz, hull_length, n_steps, Cs, hull_type)
CONFIGS = [
    # D3Q19 tests (SDAA:4 busy → use 12; SDAA:5,6 free)
    (12, "D3Q19", "MRT+Smag",   200, 200, 200, 200, 3000, 0.05, "bare_hull"),
    (5,  "D3Q19", "CUMULANT",   200, 200, 200, 200, 3000, 0.0,  "bare_hull"),
    (6,  "D3Q19", "CASCADED",   200, 200, 200, 200, 3000, 0.0,  "bare_hull"),
    # D3Q27 tests (SDAA:7,9,11 free; 13,14 for memory-hungry CASCADED@160³)
    (7,  "D3Q27", "CUMULANT",   200, 200, 200, 200, 3000, 0.0,  "bare_hull"),
    (13, "D3Q27", "CASCADED",   160, 160, 160, 160, 3000, 0.0,  "bare_hull"),
    (9,  "D3Q27", "CUMULANT",   200, 200, 200, 200, 3000, 0.05, "bare_hull"),
    (14, "D3Q27", "CASCADED",   160, 160, 160, 160, 5000, 0.05, "bare_hull"),
    # Test 8: KVLCC2 not available in codebase → SUBOFF as replacement
    (11, "D3Q27", "CUMULANT",   200, 200, 200, 200, 3000, 0.05, "bare_hull"),
]

print("=" * 100)
print("COLLISION OPERATOR COMPARISON — 8 parallel tests on SDAA:5,6,7,9,11,12,13,14")
print("=" * 100)
print(f"Grid: 200³ (D3Q19), 200³ (D3Q27 CUMULANT), 160³ (D3Q27 CASCADED — memory limit)")
print(f"Re=2e6, u_in=0.06, far-field + wall-function BC")
print(f"Note: KVLCC2 not available; test 8 uses SUBOFF bare_hull instead.")
print(f"Note: SDAA:4 busy; D3Q27 CASCADED@200³ OOM → 160³ on 13,14.")
print()

procs = []
for did, lattice, collision, nx, ny, nz, hl, ns, cs, ht in CONFIGS:
    lt = "Q19" if lattice == "D3Q19" else "Q27"
    smag = f" Cs={cs}" if cs > 0 else ""
    name = f"[SDAA:{did}] {lt} {collision}{smag} {nx}³ {ns}steps"
    log = TEST_DIR / f"worker_{did:02d}.log"
    cmd = [
        sys.executable, str(WORKER),
        str(did), lattice, collision,
        str(nx), str(ny), str(nz), str(hl), str(ns), str(cs), ht,
    ]
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((did, name, p))
    print(f"  {name}  (PID={p.pid})", flush=True)

print(f"\nAll {len(procs)} workers launched. Logs: {TEST_DIR}")
print()

# Wait for completion
t0 = time.time()
while True:
    time.sleep(30)
    done = sum(1 for _, _, p in procs if p.poll() is not None)
    print(f"[{time.time() - t0:.0f}s] {done}/{len(procs)} workers done", flush=True)
    if done == len(procs):
        break

print(f"\nAll workers finished in {time.time() - t0:.0f}s\n", flush=True)

# Collect results
results = []
for did, name, p in procs:
    rf = TEST_DIR / f"result_{did:02d}.json"
    if rf.exists():
        try:
            r = json.loads(rf.read_text())
            results.append(r)
        except (json.JSONDecodeError, IOError) as e:
            results.append({"did": did, "name": name, "error": str(e)})
    else:
        # Try to read last lines from log
        log = TEST_DIR / f"worker_{did:02d}.log"
        last_lines = ""
        if log.exists():
            try:
                lines = log.read_text().strip().split("\n")
                last_lines = "\n".join(lines[-5:])
            except Exception:
                pass
        results.append({
            "did": did, "name": name, "error": "no result file",
            "log_tail": last_lines,
        })

# Sort: D3Q19 first, then D3Q27; within each lattice, by collision type
order = {"MRT+Smag": 0, "CUMULANT": 1, "CASCADED": 2}
results.sort(key=lambda r: (
    0 if r.get("lattice") == "D3Q19" else 1,
    order.get(r.get("collision", ""), 99),
    r.get("Cs", 0),
))

# Print summary
print("=" * 100)
print("RESULTS SUMMARY")
print("=" * 100)
header = f"{'Card':<6} {'Lattice':<7} {'Collision':<12} {'Cs':<5} {'Steps':<7} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Err%':<8} {'OK':<5} {'Time':<8}"
print(header)
print("-" * 100)

for r in results:
    if "error" in r:
        print(f"{r.get('did','?'):<6} {'ERROR':<7} {r.get('name','?'):<40} {r['error']}")
        continue
    did = r.get("did", "?")
    lt = r.get("lattice", "?")
    col = r.get("collision", "?")
    cs = r.get("Cs", 0)
    steps = r.get("steps", 0)
    cf = r.get("Ct_fric", 0)
    cp = r.get("Ct_pres", 0)
    ct = r.get("Ct_total", 0)
    err = r.get("error_pct", 0)
    ok = "✓" if r.get("finite", False) else "✗"
    et = r.get("elapsed_s", 0)
    print(f"{did:<6} {lt:<7} {col:<12} {cs:<5.2f} {steps:<7} {cf:<10.5f} {cp:<10.5f} {ct:<10.5f} {err:<8.1f} {ok:<5} {et:<8.0f}s")

print("-" * 100)

# Detailed breakdown by lattice
for lattice_name in ["D3Q19", "D3Q27"]:
    lr = [r for r in results if r.get("lattice") == lattice_name and "error" not in r]
    if not lr:
        continue
    print(f"\n--- {lattice_name} DETAIL ---")
    for r in sorted(lr, key=lambda x: x.get("Ct_total", 0) if isinstance(x.get("Ct_total"), (int, float)) else 0):
        smag = " + Smag" if r.get("Cs", 0) > 0 else ""
        print(f"  {r['collision']}{smag:>12}: Ct={r.get('Ct_total',0):.5f} "
              f"(fric={r.get('Ct_fric',0):.5f}, pres={r.get('Ct_pres',0):.5f}) "
              f"err={r.get('error_pct',0):.1f}% "
              f"{'STABLE' if r.get('finite') else 'DIVERGED'}")

# Save
out_file = Path("/tmp/collision_comparison_long.json")
out_file.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out_file}")
print(f"Total wall time: {time.time() - t0:.0f}s")
