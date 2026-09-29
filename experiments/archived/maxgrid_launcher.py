#!/usr/bin/env python3
"""Launch all 8 MaxGrid benchmarks in parallel on SDAA cards 0-7.

Usage:
    PYTHONPATH=src python maxgrid_launcher.py
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
WORKER = ROOT / "maxgrid_worker.py"

# All 8 configurations → SDAA card mapping
JOBS = [
    # (case_name, device, output_file)
    ("suboff_384",        "sdaa:0", "/tmp/maxgrid_suboff_384.json"),
    ("suboff_448",        "sdaa:1", "/tmp/maxgrid_suboff_448.json"),
    ("kvlcc2",            "sdaa:2", "/tmp/maxgrid_kvlcc2.json"),
    ("wigley",            "sdaa:3", "/tmp/maxgrid_wigley.json"),
    ("flatplate_cs005",   "sdaa:4", "/tmp/maxgrid_flatplate_cs005.json"),
    ("cylinder_re200",    "sdaa:5", "/tmp/maxgrid_cylinder_re200.json"),
    ("sphere_re100",      "sdaa:6", "/tmp/maxgrid_sphere_re100.json"),
    ("flatplate_cs0",     "sdaa:7", "/tmp/maxgrid_flatplate_cs0.json"),
]

BASELINE_ERRORS = {
    "suboff_384": 24.5,
    "suboff_448": 24.5,
    "kvlcc2": 10.3,
    "wigley": 3.9,
    "flatplate_cs005": 36.6,
    "cylinder_re200": 8.1,
    "sphere_re100": 13.4,
    "flatplate_cs0": 18.0,
}

BASELINE_GRIDS = {
    "suboff_384": "200x80x80",   # approximate baseline
    "suboff_448": "200x80x80",
    "kvlcc2": "200x60x60",
    "wigley": "200x60x60",
    "flatplate_cs005": "200x40x40",
    "cylinder_re200": "200x80x4 (D=24)",
    "sphere_re100": "80x80x80 (D=24)",  # approximate baseline
    "flatplate_cs0": "200x40x40",
}


def run_one(case_name: str, device: str, output_path: str) -> dict:
    """Run a single benchmark worker subprocess."""
    cmd = [
        sys.executable, str(WORKER),
        "--case", case_name,
        "--device", device,
        "--output", output_path,
    ]
    label = f"[{case_name} on {device}]"
    print(f"{label} Starting...", flush=True)
    t0 = time.time()

    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=7200,  # 2-hour timeout
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"{label} TIMEOUT after {elapsed:.0f}s", flush=True)
        return {
            "case": case_name, "device": device,
            "status": "TIMEOUT", "wall_time_s": elapsed,
        }

    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"{label} FAILED (rc={proc.returncode}) after {elapsed:.0f}s", flush=True)
        stderr_tail = proc.stderr[-1000:] if proc.stderr else ""
        print(f"STDERR: {stderr_tail[-500:]}", flush=True)
        return {
            "case": case_name, "device": device,
            "status": "FAILED", "return_code": proc.returncode,
            "stderr_tail": stderr_tail,
            "wall_time_s": elapsed,
        }

    # Parse JSON from output file
    try:
        result = json.loads(Path(output_path).read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"{label} PARSE FAILED: {e}", flush=True)
        return {
            "case": case_name, "device": device,
            "status": "PARSE_FAILED", "error": str(e),
            "wall_time_s": elapsed,
        }

    result["wall_time_launcher_s"] = elapsed
    err = result.get("error_pct", float("nan"))

    # Key metric based on case type
    if case_name.startswith("suboff") or case_name.startswith("kvlcc2") or case_name.startswith("wigley"):
        metric = result.get("Ct_total", float("nan"))
        ref = result.get("Ct_reference", float("nan"))
    elif case_name.startswith("flatplate"):
        metric = result.get("Cf_final", float("nan"))
        ref = result.get("Cf_reference", float("nan"))
    else:
        metric = result.get("Cd_mean", float("nan"))
        ref = result.get("Cd_ref", float("nan"))

    print(f"{label} DONE: err={err:.1f}% (baseline={BASELINE_ERRORS[case_name]}%) "
          f"metric={metric:.5f} ref={ref:.5f} time={elapsed:.0f}s", flush=True)
    return result


def main():
    print("=" * 80)
    print("MaxGrid Benchmark — All 8 Configurations")
    print("D3Q19 MRT+Smag Cs=0.05 + wall_function_3d + far_field_bc_3d")
    print("SDAA cards 0-7 in parallel")
    print("=" * 80)
    print()

    # Clean old results
    for _, _, out_path in JOBS:
        Path(out_path).unlink(missing_ok=True)

    t_start = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_one, case_name, device, out_path): (case_name, device)
            for case_name, device, out_path in JOBS
        }
        results = {}
        for future in as_completed(futures):
            case_name, device = futures[future]
            try:
                result = future.result()
                results[case_name] = result
            except Exception as e:
                print(f"[{case_name} on {device}] EXCEPTION: {e}", flush=True)
                results[case_name] = {
                    "case": case_name, "device": device,
                    "status": "EXCEPTION", "error": str(e),
                }

    total_elapsed = time.time() - t_start

    # Build summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY — MaxGrid Benchmark")
    print("=" * 80)

    header = f"{'Case':<22} {'Grid':>16} {'Metric':>12} {'Ref':>12} {'Error%':>8} {'Baseline%':>10} {'Δ':>8} {'Time':>8} {'Status':>10}"
    print(header)
    print("-" * 80)

    all_ok = 0
    all_fail = 0
    improved = 0
    worsened = 0

    rows = []
    for case_name, _, _ in JOBS:
        r = results.get(case_name, {"status": "MISSING"})
        baseline = BASELINE_ERRORS[case_name]
        baseline_grid = BASELINE_GRIDS[case_name]

        grid = r.get("grid", "?")
        status = r.get("status", "OK" if r else "MISSING")
        err = r.get("error_pct", float("nan"))

        if status in ("FAILED", "EXCEPTION", "TIMEOUT", "MISSING", "PARSE_FAILED"):
            all_fail += 1
            print(f"{case_name:<22} {grid:>16} {'—':>12} {'—':>12} {'—':>8} "
                  f"{baseline:>10.1f} {'—':>8} {'—':>8} {status:>10}")
            rows.append({
                "case": case_name, "grid": grid, "status": status,
                "baseline_error_pct": baseline, "baseline_grid": baseline_grid,
                "error_pct": None, "improved": None,
            })
        else:
            all_ok += 1
            # Determine metric and ref
            if case_name.startswith("suboff") or case_name.startswith("kvlcc2") or case_name.startswith("wigley"):
                metric_val = r.get("Ct_total", float("nan"))
                ref_val = r.get("Ct_reference", float("nan"))
            elif case_name.startswith("flatplate"):
                metric_val = r.get("Cf_final", float("nan"))
                ref_val = r.get("Cf_reference", float("nan"))
            else:
                metric_val = r.get("Cd_mean", float("nan"))
                ref_val = r.get("Cd_ref", float("nan"))

            delta = baseline - err
            improved_str = "✓" if delta > 1.0 else ("✗" if delta < -1.0 else "≈")
            if delta > 1.0:
                improved += 1
            elif delta < -1.0:
                worsened += 1

            wall_time = r.get("wall_time_s", r.get("wall_time_launcher_s", 0))
            time_str = f"{wall_time:.0f}s" if wall_time else "?"

            print(f"{case_name:<22} {grid:>16} {metric_val:>12.5f} {ref_val:>12.5f} "
                  f"{err:>7.1f}% {baseline:>9.1f}% "
                  f"{delta:>+7.1f}%{improved_str} {time_str:>8} {status:>10}")

            rows.append({
                "case": case_name, "grid": grid, "status": status,
                "metric": metric_val, "reference": ref_val,
                "error_pct": err, "baseline_error_pct": baseline,
                "baseline_grid": baseline_grid,
                "delta_pct": delta, "improved": delta > 1.0,
                "wall_time_s": wall_time,
            })

    print("-" * 80)
    print(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Completed: {all_ok}/{len(JOBS)}  Failed: {all_fail}/{len(JOBS)}")
    print(f"Improved (>1%): {improved}  Worsened (>1%): {worsened}  Neutral: {all_ok - improved - worsened}")

    # Key question answer
    print()
    print("=" * 80)
    print("KEY QUESTION: Does grid refinement reduce error?")
    print("=" * 80)
    if all_fail > 0:
        print(f"Answer: INCONCLUSIVE — {all_fail}/{len(JOBS)} jobs failed.")
    elif worsened > improved:
        print(f"Answer: NO — non-monotonic pattern persists ({worsened} cases worsened, {improved} improved).")
    else:
        print(f"Answer: PARTIALLY — {improved} cases improved, {worsened} worsened, "
              f"{all_ok - improved - worsened} neutral.")

    # Specific checks
    suboff_384 = results.get("suboff_384", {})
    suboff_448 = results.get("suboff_448", {})
    e384 = suboff_384.get("error_pct", float("nan"))
    e448 = suboff_448.get("error_pct", float("nan"))
    if e384 == e384 and e448 == e448:
        print(f"  SUBOFF 384→448: {e384:.1f}% → {e448:.1f}% "
              f"({'improved' if e448 < e384 else 'worsened'})")

    print()

    # Write aggregated results
    summary = {
        "title": "MaxGrid Benchmark — Grid Refinement Study",
        "setup": "D3Q19 MRT+Smag Cs=0.05 + wall_function_3d + far_field_bc_3d",
        "total_wall_time_s": total_elapsed,
        "completed": all_ok,
        "failed": all_fail,
        "improved": improved,
        "worsened": worsened,
        "results": rows,
    }
    out_path = Path("/tmp/maxgrid_results.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
