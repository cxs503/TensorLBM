#!/usr/bin/env python3
"""Launch 5 ship hull drag benchmarks in parallel on SDAA cards 8-12.

Each benchmark runs D3Q19 MRT+Smag Cs=0.05 + wall_fn + farfield.
Grid: 200×60×60, Re=2e6, hull_length=80, 3000 steps, warmup=1000.

Results collected into /tmp/ship_bench_results.json
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HULL_TYPES = ["wigley", "series60", "kcs", "kvlcc2", "npl"]
DEVICES = ["sdaa:8", "sdaa:9", "sdaa:10", "sdaa:11", "sdaa:12"]

WORKER_SCRIPT = Path(__file__).parent / "ship_bench_worker.py"


def run_one(hull: str, device: str) -> dict:
    """Run a single benchmark worker subprocess."""
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--hull", hull,
        "--device", device,
        "--cs", "0.05",
        "--n-steps", "3000",
        "--warmup", "1000",
        "--nx", "200", "--ny", "60", "--nz", "60",
        "--re", "2000000",
        "--hull-length", "80",
        "--u-in", "0.06",
    ]
    label = f"[{hull} on {device}]"
    print(f"{label} Starting...", flush=True)
    t0 = time.time()

    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(Path(__file__).parent / "src"),
    }

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"{label} FAILED (rc={proc.returncode}) after {elapsed:.0f}s", flush=True)
        print(f"STDERR: {proc.stderr[-2000:]}", flush=True)
        return {
            "hull_type": hull,
            "device": device,
            "status": "FAILED",
            "return_code": proc.returncode,
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
            "elapsed_s": elapsed,
        }

    # Parse JSON from stdout
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        # Try to find the last JSON object
        lines = proc.stdout.strip().splitlines()
        for line in reversed(lines):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            result = {
                "hull_type": hull,
                "device": device,
                "status": "PARSE_FAILED",
                "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            }

    result["wall_time_launcher_s"] = elapsed
    print(f"{label} Complete: Ct_total={result.get('Ct_total', 'N/A')} "
          f"(err={result.get('error_pct', 'N/A')}%)  in {elapsed:.0f}s", flush=True)
    return result


def main():
    print("=" * 70)
    print("Ship Hull Drag Benchmarks — D3Q19 MRT+Smag Cs=0.05 + wall_fn")
    print("5 hull types on SDAA cards 8-12 in parallel")
    print(f"Grid: 200×60×60  Re=2e6  hull_length=80  3000 steps (warmup=1000)")
    print("=" * 70)

    t_start = time.time()

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_one, hull, dev): (hull, dev)
            for hull, dev in zip(HULL_TYPES, DEVICES)
        }
        results = {}
        for future in as_completed(futures):
            hull, dev = futures[future]
            try:
                result = future.result()
                results[hull] = result
            except Exception as e:
                print(f"[{hull} on {dev}] EXCEPTION: {e}", flush=True)
                results[hull] = {
                    "hull_type": hull,
                    "device": dev,
                    "status": "EXCEPTION",
                    "error": str(e),
                }

    total_elapsed = time.time() - t_start

    # Build summary
    summary = {
        "config": {
            "lattice": "D3Q19",
            "collision": "MRT+Smagorinsky",
            "C_s": 0.05,
            "grid": "200×60×60",
            "Re": 2_000_000,
            "hull_length": 80,
            "u_in": 0.06,
            "n_steps": 3000,
            "warmup": 1000,
            "devices": DEVICES,
        },
        "total_wall_time_s": total_elapsed,
        "results": {},
    }

    for hull in HULL_TYPES:
        r = results.get(hull, {"status": "MISSING"})
        summary["results"][hull] = r

    # Print summary table
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Hull':<12} {'Ct_fric':>10} {'Ct_pres':>10} {'Ct_total':>10} "
          f"{'Ct_ref':>10} {'Error%':>8} {'Status':>10}")
    print("-" * 70)

    for hull in HULL_TYPES:
        r = results.get(hull, {})
        ct_ref = r.get("Ct_reference", 0)
        if r.get("status") in ("FAILED", "EXCEPTION", "MISSING", "PARSE_FAILED"):
            print(f"{hull:<12} {'—':>10} {'—':>10} {'—':>10} "
                  f"{ct_ref:>10.5f} {'—':>8} {r.get('status', '?'):>10}")
        else:
            print(f"{hull:<12} {r.get('Ct_fric', 0):>10.5f} {r.get('Ct_pres', 0):>10.5f} "
                  f"{r.get('Ct_total', 0):>10.5f} {ct_ref:>10.5f} "
                  f"{r.get('error_pct', 0):>7.1f}% {'OK':>10}")

    print("-" * 70)
    print(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()

    # Universal Cs=0.05 assessment
    print("UNIVERSALITY ASSESSMENT (Cs=0.05 across all hull types):")
    print("-" * 50)
    all_ok = True
    for hull in HULL_TYPES:
        r = results.get(hull, {})
        err = r.get("error_pct", float("inf"))
        status = "✓" if err < 5.0 else ("⚠" if err < 10.0 else "✗")
        if err >= 10.0:
            all_ok = False
        print(f"  {hull:<12}: {err:.1f}% error {status}")
    print()
    if all_ok:
        print("✓ Cs=0.05 appears UNIVERSAL across all 5 hull types (<10% error).")
    else:
        print("✗ Cs=0.05 is NOT universal — some hull types exceed 10% error.")
    print()

    # Write results
    out_path = Path("/tmp/ship_bench_results.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
