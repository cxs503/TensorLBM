"""Agent FullMatrix launcher — 16 configs on 16 SDAA cards.

Matrix: 4 collisions × 4 benchmarks = 16 combos.

Collisions (rows):
  C1: D3Q19 MRT+Smag Cs=0.05
  C2: D3Q27 CASCADED
  C3: D3Q27 CUMULANT
  C4: D3Q27 KBC+Smag Cs=0.05

Benchmarks (cols):
  B1: SUBOFF 200×80×80 Re=2e6
  B2: Cylinder 200×80×4 Re=200
  B3: Sphere 200×100×100 Re=1000
  B4: Square prism 200×80×4 Re=22000

Mapping: card 0-15 → (collision_index * 4 + benchmark_index)
"""
import json, math, os, subprocess, sys, time
from pathlib import Path

WORKER = Path(__file__).parent / "_fullmatrix_worker.py"
OUT_DIR = Path("/tmp/fullmatrix_logs")
RESULTS_JSON = Path("/tmp/fullmatrix_results.json")

COLLISION_LABELS = ["C1: D3Q19 MRT+Smag", "C2: D3Q27 CASCADED", "C3: D3Q27 CUMULANT", "C4: D3Q27 KBC+Smag"]
BENCH_LABELS = ["B1: SUBOFF", "B2: Cylinder", "B3: Sphere", "B4: Square prism"]
BENCH_FLOWS = ["suboff", "cylinder", "sphere", "square"]

# Clean
OUT_DIR.mkdir(exist_ok=True)
for f in OUT_DIR.glob("*.json"):
    f.unlink()
for f in OUT_DIR.glob("*.log"):
    f.unlink()

env = os.environ.copy()
env["PYTHONPATH"] = str(Path(__file__).parent / "src")

print("=" * 120)
print("AGENT FULLMATRIX — 16 configs on 16 SDAA cards (0-15)")
print("=" * 120)
print(f"4 collisions × 4 benchmarks = 16 combos, {1500} steps each")
print(f"Collisions: C1=MRT+Smag(D3Q19)  C2=CASCADED(D3Q27)  C3=CUMULANT(D3Q27)  C4=KBC+Smag(D3Q27)")
print(f"Benchmarks: B1=SUBOFF(Re=2e6)  B2=Cylinder(Re=200)  B3=Sphere(Re=1000)  B4=Square(Re=22000)")
print()

# ── Launch all 16 workers ──────────────────────────────────────────
procs = []
for did in range(16):
    ci = did // 4
    bi = did % 4
    name = f"[C{did:02d}] {COLLISION_LABELS[ci]} × {BENCH_LABELS[bi]}"
    log = OUT_DIR / f"worker_{did:02d}.log"
    cmd = [sys.executable, str(WORKER), str(did)]
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((did, name, p, log))
    print(f"  {name} (PID={p.pid})", flush=True)

print(f"\nAll 16 workers launched. Logs: {OUT_DIR}", flush=True)
print()

# ── Wait for completion ────────────────────────────────────────────
t0 = time.time()
while True:
    time.sleep(15)
    done = sum(1 for _, _, p, _ in procs if p.poll() is not None)
    n = len(procs)
    print(f"[{time.time()-t0:.0f}s] {done}/{n} workers done", flush=True)
    if done == n:
        break

print(f"\nAll workers finished in {time.time()-t0:.0f}s\n", flush=True)

# ── Collect results ────────────────────────────────────────────────
results = []
for did, name, p, log_path in procs:
    rf = OUT_DIR / f"result_{did:02d}.json"
    if rf.exists():
        try:
            r = json.loads(rf.read_text())
            results.append(r)
        except Exception as e:
            results.append({"config_id": did, "name": name, "error": str(e)})
    else:
        tail = ""
        if log_path.exists():
            try:
                lines = log_path.read_text().strip().split("\n")
                tail = "\n".join(lines[-5:])
            except Exception:
                pass
        results.append({"config_id": did, "name": name, "error": "no result file", "log_tail": tail})

# ── Build 4×4 matrix ───────────────────────────────────────────────
print()
print("=" * 140)
print("FULLMATRIX RESULTS — 4 COLLISIONS × 4 BENCHMARKS")
print("=" * 140)

# Table header
print(f"\n{'':<22}", end="")
for bi in range(4):
    print(f"  {BENCH_LABELS[bi]:<28}", end="")
print(f"  {'BEST-IN-ROW':<20}")
print("-" * 160)

for ci in range(4):
    row_results = []
    row_errors = []
    for bi in range(4):
        did = ci * 4 + bi
        r = next((r for r in results if r.get("config_id") == did), None)
        row_results.append(r)

        if r and "error" not in r:
            err = r.get("error_pct", float("nan"))
            dp = r.get("dp_slide", r.get("dp_full", 0))
            finite = r.get("finite", False)
            row_errors.append(err if math.isfinite(err) else float("inf"))
            # Build compact cell
            ok = "✓" if finite else "✗"
            cell = f"{dp:.5f} {ok} {err:.1f}%"
        else:
            row_errors.append(float("inf"))
            cell = "ERROR/FAILED"

        print(f"  {COLLISION_LABELS[ci]:<20}", end="") if bi == 0 else None
        print(f"{cell:<28}", end="")

    # Best in row
    if row_errors:
        valid = [(i, e) for i, e in enumerate(row_errors) if math.isfinite(e) and e < float("inf")]
        if valid:
            best_bi, best_err = min(valid, key=lambda x: x[1])
            print(f"  {BENCH_LABELS[best_bi]} ({best_err:.1f}%)", end="")
        else:
            print(f"  ALL FAILED", end="")
    print()

print("-" * 160)

# ── Detailed per-flow breakdown ────────────────────────────────────
for bi, bflow in enumerate(BENCH_FLOWS):
    print(f"\n{'='*80}")
    print(f"  {BENCH_LABELS[bi]} — Best collision operator")
    print(f"{'='*80}")
    print(f"{'Collision':<25} {'dp_full':<10} {'dp_slide':<10} {'error%':<8} {'finite':<8} {'time':<8}")
    print("-" * 70)

    best = None
    for ci in range(4):
        did = ci * 4 + bi
        r = next((r for r in results if r.get("config_id") == did), None)
        if r and "error" not in r:
            dpf = r.get("dp_full", 0)
            dps = r.get("dp_slide", 0)
            err = r.get("error_pct", float("nan"))
            fin = "✓" if r.get("finite") else "✗"
            et = r.get("elapsed_s", 0)
            print(f"{COLLISION_LABELS[ci]:<25} {dpf:<10.5f} {dps:<10.5f} {err:<8.1f} {fin:<8} {et:<8.0f}s")
            if r.get("finite") and math.isfinite(err):
                if best is None or err < best[1]:
                    best = (COLLISION_LABELS[ci], err, dps)
        else:
            err_msg = r.get("error", "no result") if r else "no result"
            print(f"{COLLISION_LABELS[ci]:<25} {'FAILED':<10} {'-':<10} {'FAIL':<8} {'✗':<8} {err_msg}")

    if best:
        print(f"\n  >>> BEST FOR {BENCH_LABELS[bi]}: {best[0]} — dp={best[2]:.5f}, error={best[1]:.1f}%")
    else:
        print(f"\n  >>> ALL OPERATORS FAILED FOR {BENCH_LABELS[bi]}")

# ── Answer the key question ────────────────────────────────────────
print(f"\n{'='*80}")
print("KEY QUESTION: Which collision operator is best for EACH benchmark type?")
print(f"{'='*80}")

for bi, bflow in enumerate(BENCH_FLOWS):
    valid_collisions = []
    for ci in range(4):
        did = ci * 4 + bi
        r = next((r for r in results if r.get("config_id") == did), None)
        if r and "error" not in r and r.get("finite") and math.isfinite(r.get("error_pct", float("nan"))):
            valid_collisions.append((ci, r.get("error_pct"), r.get("dp_slide", 0)))

    if valid_collisions:
        valid_collisions.sort(key=lambda x: x[1])
        best_ci, best_err, best_dp = valid_collisions[0]
        print(f"\n  {BENCH_LABELS[bi]}:")
        print(f"    Best: {COLLISION_LABELS[best_ci]} — dp={best_dp:.5f}, error={best_err:.1f}%")
        for ci, err, dp in valid_collisions[1:]:
            print(f"      vs {COLLISION_LABELS[ci]}: dp={dp:.5f}, error={err:.1f}%")
    else:
        print(f"\n  {BENCH_LABELS[bi]}: ALL OPERATORS FAILED")

# ── Save combined results ──────────────────────────────────────────
combined = {
    "title": "Agent FullMatrix — 16-config collision×benchmark matrix",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "setup": {
        "collisions": [{"label": l, "Cs": [0.05, 0.0, 0.0, 0.05][i],
                        "lattice": ["D3Q19", "D3Q27", "D3Q27", "D3Q27"][i]}
                       for i, l in enumerate(COLLISION_LABELS)],
        "benchmarks": [
            {"label": l, "grid": g, "Re": r}
            for l, g, r in zip(BENCH_LABELS,
                              ["200×80×80", "200×80×4", "200×100×100", "200×80×4"],
                              ["2e6", "200", "1000", "22000"])
        ],
        "common": {"n_steps": 1500, "sliding_window": 300, "farfield_bc": True, "wall_law": "log-law", "sign_fixed": True},
    },
    "results": results,
}
RESULTS_JSON.write_text(json.dumps(combined, indent=2))
print(f"\nSaved: {RESULTS_JSON}")
print(f"Total wall time: {time.time()-t0:.0f}s")
