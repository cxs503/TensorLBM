#!/usr/bin/env python3
"""Improved wall-function benchmark launcher — 16 parallel jobs on SDAA 0-15.

Runs all 8 benchmark cases × base vs improved (Musker+vanDriest).

Card allocation:
  0:  suboff_200      base
  1:  suboff_200      improved
  2:  suboff_320      base
  3:  suboff_320      improved
  4:  flatplate_320   base
  5:  flatplate_320   improved
  6:  cylinder        base
  7:  cylinder        improved
  8:  sphere          base
  9:  sphere          improved
  10: kvlcc2          base
  11: kvlcc2          improved
  12: wigley          base
  13: wigley          improved
  14: flatplate_cs0   base
  15: flatplate_cs0   improved
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "improved_worker.py"
SRC = Path(__file__).parent / "src"
LOG_DIR = Path("/tmp/improved_bench_logs")
OUTPUT_FILE = Path("/tmp/improved_bench_results.json")
LOG_DIR.mkdir(exist_ok=True)

# (card_id, case_name, improved_flag, label)
JOBS = [
    (0,  "suboff_200",    0, "SUBOFF 200³ base"),
    (1,  "suboff_200",    1, "SUBOFF 200³ improved"),
    (2,  "suboff_320",    0, "SUBOFF 320³ base"),
    (3,  "suboff_320",    1, "SUBOFF 320³ improved"),
    (4,  "flatplate_320", 0, "Flat-plate 320³ base"),
    (5,  "flatplate_320", 1, "Flat-plate 320³ improved"),
    (6,  "cylinder",      0, "Cylinder base"),
    (7,  "cylinder",      1, "Cylinder improved"),
    (8,  "sphere",        0, "Sphere base"),
    (9,  "sphere",        1, "Sphere improved"),
    (10, "kvlcc2",        0, "KVLCC2 base"),
    (11, "kvlcc2",        1, "KVLCC2 improved"),
    (12, "wigley",        0, "Wigley base"),
    (13, "wigley",        1, "Wigley improved"),
    (14, "flatplate_cs0", 0, "Flat-plate Cs=0 base"),
    (15, "flatplate_cs0", 1, "Flat-plate Cs=0 improved"),
]

OUTPUT_FILES = {f"/tmp/improved_sdaa{card}_{case}_{improved}.json"
                for card, case, improved, _ in JOBS}


def main():
    print("=" * 80)
    print("IMPROVED WALL FUNCTION BENCHMARK — 16 JOBS ON SDAA 0-15")
    print("  Base:    wall_function_3d(f, solid, nu) → log-law")
    print("  Improved: wall_function_3d(f, solid, nu, wall_law='musker', use_van_driest=True)")
    print("=" * 80)
    print()

    # Clean previous output files
    for f in OUTPUT_FILES:
        Path(f).unlink(missing_ok=True)

    # Launch all workers
    procs = {}
    for card_id, case_name, improved, label in JOBS:
        log_file = LOG_DIR / f"improved_{card_id:02d}_{case_name}_{'imp' if improved else 'base'}.log"
        cmd = [
            sys.executable, "-u", str(WORKER),
            str(card_id), case_name, str(improved),
        ]
        env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}

        with open(log_file, "w") as log_fp:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
            )
        procs[(card_id, case_name, improved)] = (proc, log_file, label)
        print(f"  [{card_id:02d}] {label:30s} PID={proc.pid}  log={log_file}")

    print(f"\n  All {len(procs)} jobs launched. Waiting for completion...\n")

    # Poll until all done
    t_start = time.time()
    while True:
        done = sum(1 for proc, _, _ in procs.values() if proc.poll() is not None)
        elapsed = time.time() - t_start
        print(f"  [{elapsed:.0f}s] {done}/{len(procs)} completed", flush=True)
        if done == len(procs):
            break
        time.sleep(30)

    total_elapsed = time.time() - t_start
    print(f"\n  All jobs finished in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

    # Collect results from log files (last JSON line)
    results = {}
    for (card_id, case_name, improved), (proc, log_file, label) in procs.items():
        rc = proc.returncode
        key = f"{case_name}_{'improved' if improved else 'base'}"

        # Parse JSON from log
        try:
            log_text = log_file.read_text()
            # Find the last JSON object in the output
            lines = log_text.strip().splitlines()
            result = None
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict) and "case" in candidate:
                        result = candidate
                        break
                except (json.JSONDecodeError, ValueError):
                    continue

            if result is None:
                result = {
                    "case": case_name,
                    "improved": bool(improved),
                    "card_id": card_id,
                    "_status": "PARSE_FAILED",
                    "_rc": rc,
                    "_log_tail": "\n".join(lines[-10:]) if lines else "",
                }
            else:
                result["_rc"] = rc
                result["card_id"] = card_id
                result["improved"] = bool(improved)

        except FileNotFoundError:
            result = {
                "case": case_name,
                "improved": bool(improved),
                "card_id": card_id,
                "_status": "LOG_MISSING",
            }

        results[key] = result

        status = result.get("_status", "OK")
        if status != "OK":
            print(f"  {label:30s} → {status} (rc={rc})")
        else:
            if "Ct_total" in result:
                print(f"  {label:30s} → Ct={result['Ct_total']:.5f}  ({result.get('wall_time_s',0):.0f}s)")
            elif "Cd_total" in result:
                print(f"  {label:30s} → Cd={result['Cd_total']:.5f}  ({result.get('wall_time_s',0):.0f}s)")
            elif "Cf" in result:
                print(f"  {label:30s} → Cf={result['Cf']:.5f}  ({result.get('wall_time_s',0):.0f}s)")
            else:
                print(f"  {label:30s} → ok ({result.get('wall_time_s',0):.0f}s)")

    print()

    # Build comparison table
    print("=" * 90)
    print("COMPARISON: BASE vs IMPROVED (Musker + van Driest)")
    print("=" * 90)

    comparisons = []
    case_pairs = [
        ("suboff_200",     "SUBOFF 200³",      "Ct_total", "Ct_ref_ITTCx1k", 0.00405),
        ("suboff_320",     "SUBOFF 320³",      "Ct_total", "Ct_ref_ITTCx1k", 0.00405),
        ("flatplate_320",  "Flat Plate 320³",  "Cf",       "Cf_ITTC",         0.00405),
        ("cylinder",       "Cylinder Re=200",  "Cd_total", "Cd_ref",          1.30),
        ("sphere",         "Sphere Re=100",    "Cd_total", "Cd_ref",          1.09),
        ("kvlcc2",         "KVLCC2",           "Ct_total", "Ct_reference",    None),
        ("wigley",         "Wigley",           "Ct_total", "Ct_reference",    None),
        ("flatplate_cs0",  "Flat Plate Cs=0",  "Cf",       "Cf_ITTC",         0.00405),
    ]

    print(f"\n{'Case':<20} {'Metric':>10} {'Base':>12} {'Improved':>12} {'Delta':>12} {'∆%':>8} {'Winner':>10}")
    print("-" * 90)

    for case_key, label, metric, ref_key, ref_val in case_pairs:
        base_key = f"{case_key}_base"
        imp_key = f"{case_key}_improved"
        base = results.get(base_key, {})
        imp = results.get(imp_key, {})

        base_val = base.get(metric)
        imp_val = imp.get(metric)

        if base_val is None or imp_val is None:
            print(f"{label:<20} {'n/a':>10} {'n/a':>12} {'n/a':>12} {'n/a':>12} {'n/a':>8} {'n/a':>10}")
            comparisons.append({
                "case": label, "metric": metric,
                "base": None, "improved": None,
                "delta": None, "delta_pct": None,
                "winner": "n/a",
                "base_status": base.get("_status", "?"),
                "improved_status": imp.get("_status", "?"),
            })
            continue

        delta = imp_val - base_val
        pct = delta / abs(base_val) * 100 if abs(base_val) > 1e-12 else float('nan')

        # Determine winner: lower error vs reference is better
        base_err = abs(base_val - ref_val) if ref_val else None
        imp_err = abs(imp_val - ref_val) if ref_val else None

        if ref_key and ref_val:
            if ref_key in base:
                base_err = abs(base.get("error_pct", base_err)) if base_err is None else base_err
            if ref_key in imp:
                imp_err = abs(imp.get("error_pct", imp_err)) if imp_err is None else imp_err

        if base_err is not None and imp_err is not None:
            if imp_err < base_err:
                winner = "IMPROVED"
            elif base_err < imp_err:
                winner = "BASE"
            else:
                winner = "TIE"
        else:
            # No ref: just compare raw values
            if abs(imp_val) < abs(base_val):
                winner = "IMPROVED↘"
            elif abs(base_val) < abs(imp_val):
                winner = "BASE↗"
            else:
                winner = "TIE"

        winner_str = f"{winner}"
        if not math.isfinite(pct):
            pct_str = "∞"
        else:
            pct_str = f"{pct:+.2f}%"

        print(f"{label:<20} {metric:>10} {base_val:>12.5f} {imp_val:>12.5f} {delta:>+12.6f} {pct_str:>8} {winner_str:>10}")

        comparisons.append({
            "case": label,
            "case_key": case_key,
            "metric": metric,
            "base": base_val,
            "improved": imp_val,
            "delta": delta,
            "delta_pct": pct,
            "winner": winner,
            "base_status": base.get("_status", "OK"),
            "improved_status": imp.get("_status", "OK"),
            "ref_value": ref_val,
            "base_error": base.get("error_pct"),
            "improved_error": imp.get("error_pct"),
        })

    print("-" * 90)

    # Count winners
    wins_imp = sum(1 for c in comparisons if c["winner"] in ("IMPROVED", "IMPROVED↘"))
    wins_base = sum(1 for c in comparisons if c["winner"] in ("BASE", "BASE↗"))
    ties = sum(1 for c in comparisons if c["winner"] == "TIE")
    n_a = sum(1 for c in comparisons if c["winner"] == "n/a")

    print(f"\n  Improved wins: {wins_imp}/8   Base wins: {wins_base}/8   Ties: {ties}   Failed: {n_a}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    if wins_imp > wins_base:
        print(f"  ✓ Musker+vanDriest IMPROVES accuracy in {wins_imp}/{8} cases.")
        print(f"    Recommending adoption of wall_law='musker', use_van_driest=True as default.")
    elif wins_base > wins_imp:
        print(f"  ✗ Base log-law is BETTER in {wins_base}/{8} cases.")
        print(f"    Musker+vanDriest does NOT help — issue is likely elsewhere (pressure drag, grid resolution, etc.)")
    else:
        print(f"  ~ No clear winner ({wins_imp} vs {wins_base}).")
        print(f"    Grid resolution, pressure drag integration, or far-field BC may be the bottleneck.")
    print("=" * 90)

    # Write output
    output = {
        "title": "Improved Wall Function Benchmark — Musker + van Driest",
        "description": "Compares base log-law vs improved Musker continuous profile with van Driest damping",
        "wall_function_base": "wall_function_3d(f, solid, nu)",
        "wall_function_improved": "wall_function_3d(f, solid, nu, wall_law='musker', use_van_driest=True)",
        "total_wall_time_s": total_elapsed,
        "jobs": len(JOBS),
        "cards_used": "SDAA 0-15",
        "results": results,
        "comparisons": comparisons,
        "summary": {
            "improved_wins": wins_imp,
            "base_wins": wins_base,
            "ties": ties,
            "failed": n_a,
            "verdict": (
                "Musker+vanDriest IMPROVES accuracy" if wins_imp > wins_base
                else "Base log-law is better" if wins_base > wins_imp
                else "No clear winner"
            ),
        },
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
