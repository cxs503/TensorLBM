"""D3Q27 CASCADED launcher — spawns all 8 SDAA workers.

Test matrix:
1. SUBOFF bare_hull 200³, Re=2e6, 5000 steps (SDAA:0) — CASCADED
2. SUBOFF bare_hull 256³, Re=2e6, 3000 steps (SDAA:1) — CASCADED
3. KVLCC2 ship 200³, Re=2e6, 3000 steps (SDAA:2) — CASCADED
4. Wigley ship 200³, Re=2e6, 3000 steps (SDAA:3) — CASCADED
5. KCS ship 200³, Re=2e6, 3000 steps (SDAA:4) — CASCADED
6. Cylinder Re=200, D=24, 200×80×4, 3000 steps (SDAA:5) — CASCADED
7. Sphere Re=1000, D=24, 120×60×60, 3000 steps (SDAA:6) — CASCADED
8. SUBOFF bare_hull 200³, Re=2e6, 5000 steps (SDAA:7) — CUMULANT
"""
import json
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "_d3q27_worker.py"
OUTPUT = Path("/tmp/d3q27_bench_results.json")

# ── Test matrix ────────────────────────────────────────────────────────────
BENCHMARKS = [
    # (did, case,          nx,  ny,  nz,  hl,   n_steps, op)
    (0,   "suboff",        200, 80,  80,  80.0, 5000,    "cascaded"),
    (1,   "suboff",        256, 103, 103, 102.4, 3000,   "cascaded"),
    (2,   "kvlcc2",        200, 80,  80,  100.0, 3000,   "cascaded"),
    (3,   "wigley",        200, 80,  80,  100.0, 3000,   "cascaded"),
    (4,   "kcs",           200, 80,  80,  100.0, 3000,   "cascaded"),
    (5,   "cylinder",      200, 80,  4,   24.0,  3000,   "cascaded"),
    (6,   "sphere",        120, 60,  60,  24.0,  3000,   "cascaded"),
    (7,   "suboff",        200, 80,  80,  80.0,  5000,   "cumulant"),
]

D3Q19_BASELINE = [
    # (lattice, collision, Cs, Ct/Cd, error_pct) — for comparison
    ("D3Q19", "MRT+Smag", 0.05, None, None),  # we'll load from /tmp/collision_comparison_long/
]


def run_one(did: int, case: str, nx: int, ny: int, nz: int, hl: float, n_steps: int, op: str):
    """Run a single benchmark on the specified SDAA card."""
    cmd = [
        sys.executable, str(WORKER),
        str(did), case, str(nx), str(ny), str(nz), str(hl), str(n_steps), op,
    ]
    print(f"\n{'='*70}")
    print(f"STARTING: SDAA:{did} {case} {nx}x{ny}x{nz} hl={hl} steps={n_steps} op={op}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n", flush=True)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def load_baseline():
    """Load D3Q19+MRT+Smag baseline results if available."""
    baseline_path = Path("/tmp/collision_comparison_long")
    results = []
    if baseline_path.exists():
        for f in sorted(baseline_path.glob("result_*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                pass
    return results


def main():
    # Clear previous output
    if OUTPUT.exists():
        OUTPUT.unlink()

    # Load baseline
    baseline = load_baseline()
    if baseline:
        print("=== D3Q19+MRT+Smag BASELINE ===")
        for r in baseline:
            print(f"  {r.get('lattice','?')} {r.get('collision','?')} "
                  f"Cs={r.get('Cs','?')} grid={r.get('grid','?')} "
                  f"Ct={r.get('Ct_total','?'):.5f} err={r.get('error_pct','?'):.1f}%", flush=True)
    else:
        print("No baseline found at /tmp/collision_comparison_long/", flush=True)

    print(f"\n=== D3Q27 CASCADED/CUMULANT BENCHMARKS ===", flush=True)
    print(f"Running {len(BENCHMARKS)} benchmarks on 8 SDAA cards...", flush=True)

    procs = []
    for did, case, nx, ny, nz, hl, n_steps, op in BENCHMARKS:
        p = run_one(did, case, nx, ny, nz, hl, n_steps, op)
        procs.append((did, case, p))
        time.sleep(2)  # stagger starts

    print(f"\nAll {len(procs)} workers launched. Waiting for completion...\n", flush=True)

    # Monitor progress
    start = time.time()
    done = set()
    while len(done) < len(procs):
        for did, case, p in procs:
            if (did, case) in done:
                continue
            ret = p.poll()
            if ret is not None:
                stdout, _ = p.communicate()
                done.add((did, case))
                status = "OK" if ret == 0 else f"FAIL(rc={ret})"
                print(f"\n=== SDAA:{did} {case} {status} ===")
                if stdout:
                    # Print last 20 lines
                    lines = stdout.strip().split("\n")
                    for line in lines[-20:]:
                        print(f"  {line}")
                print()
            else:
                # Still running — show nothing, wait
                pass
        time.sleep(10)

    elapsed = time.time() - start
    print(f"\nAll benchmarks completed in {elapsed:.0f}s", flush=True)

    # ── Summarize ───────────────────────────────────────────────────────
    if OUTPUT.exists():
        results = json.loads(OUTPUT.read_text())
        print(f"\n{'='*70}")
        print("FINAL D3Q27 BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"{'Case':<20s} {'Grid':<12s} {'Op':<10s} {'Ct/Cd':>8s} {'Ref':>8s} {'Err%':>7s} {'Time':>7s} {'Fin':>5s}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x["did"]):
            case = r["case"]
            grid = r["grid"]
            op = r["collision"]
            val = r["Ct_total"]
            ref = r["ref_value"]
            err = r["error_pct"]
            t = r["elapsed_s"]
            fin = "YES" if r["finite"] else "NO"
            err_str = f"{err:.1f}" if isinstance(err, (int, float)) and err == err else "N/A"
            ref_str = f"{ref:.4f}" if isinstance(ref, (int, float)) and ref == ref else "N/A"
            print(f"{case:<20s} {grid:<12s} {op:<10s} {val:>8.5f} {ref_str:>8s} {err_str:>7s} {t:>7.0f}s {fin:>5s}")

        # Compare with baseline
        if baseline:
            print(f"\n{'='*70}")
            print("COMPARISON: D3Q27 vs D3Q19+MRT+Smag")
            print(f"{'='*70}")
            d3q27_suboff = [r for r in results if r["case"] in ("suboff", "suboff_cumulant")]
            d3q19_suboff = [r for r in baseline if r.get("hull") == "bare_hull" and r.get("grid", "").startswith("200")]

            for r27 in d3q27_suboff:
                for r19 in d3q19_suboff:
                    e27 = r27["error_pct"]
                    e19 = r19["error_pct"]
                    print(f"  D3Q27 {r27['collision']}: Ct={r27['Ct_total']:.5f} err={e27:.1f}%")
                    print(f"  D3Q19 {r19['collision']}: Ct={r19['Ct_total']:.5f} err={e19:.1f}%")
                    if isinstance(e27, (int, float)) and isinstance(e19, (int, float)):
                        better = "D3Q27" if e27 < e19 else "D3Q19"
                        print(f"  → {better} is better by {abs(e27-e19):.1f}pp")
    else:
        print("No output file generated.", flush=True)


if __name__ == "__main__":
    main()
