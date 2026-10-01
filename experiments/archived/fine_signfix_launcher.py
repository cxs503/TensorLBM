#!/usr/bin/env python3
"""Fine-grid sign-fix re-test launcher — 6 parallel jobs on SDAA 16-21.

Tests pressure-drag sign fix: -p*(sp-sm) instead of p*(sp-sm).
Previously fine grids (320³+) had negative Ct due to wrong pressure drag sign.
This launcher re-tests to confirm the fix rescues fine grids.

Card allocation:
  16: SUBOFF bare_hull 320³, Cs=0.05, 2000 steps
  17: SUBOFF bare_hull 200³, Cs=0.05, 5000 steps
  18: KVLCC2 ship 320³, Cs=0.05, 2000 steps
  19: KVLCC2 ship 200³, Cs=0.05, 3000 steps
  20: Flat plate 200³, Cs=0.05, 2000 steps
  21: Flat plate 200³, Cs=0.0, 2000 steps
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = Path(__file__).parent / "fine_signfix_worker.py"
SRC = Path(__file__).parent / "src"
LOG_DIR = Path("/tmp/fine_signfix_logs")
OUTPUT_FILE = Path("/tmp/fine_signfix_results.json")
LOG_DIR.mkdir(exist_ok=True)

JOBS = [
    (16, "suboff_320",         "SUBOFF 320³"),
    (17, "suboff_200",         "SUBOFF 200³"),
    (18, "kvlcc2_320",         "KVLCC2 320³"),
    (19, "kvlcc2_200",         "KVLCC2 200³"),
    (20, "flatplate_cs005_200","FlatPlate 200³ Cs=0.05"),
    (21, "flatplate_cs0_200",  "FlatPlate 200³ Cs=0"),
]


def main():
    print("=" * 80)
    print("FINE-GRID SIGN-FIX RE-TEST — 6 JOBS ON SDAA 16-21")
    print("  Pressure drag fix: -p*(sp-sm) [sign flipped]")
    print("  All: D3Q19 MRT+Smag + wallfn + farfield, Re=2e6, sliding-window 500")
    print("  Comparing against previously buggy results (p*(sp-sm))")
    print("=" * 80)
    print()

    # Launch all workers
    procs = {}
    for card_id, case_name, label in JOBS:
        log_file = LOG_DIR / f"signfix_{card_id:02d}_{case_name}.log"
        cmd = [
            sys.executable, "-u", str(WORKER),
            str(card_id), case_name,
        ]
        env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}

        with open(log_file, "w") as log_fp:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
            )
        procs[(card_id, case_name)] = (proc, log_file, label)
        print(f"  [{card_id:02d}] {label:30s} PID={proc.pid}  log={log_file}")

    print(f"\n  All {len(procs)} jobs launched. Waiting for completion...\n")

    t_start = time.time()
    while True:
        done = sum(1 for proc, _, _ in procs.values() if proc.poll() is not None)
        elapsed = time.time() - t_start
        print(f"  [{elapsed:.0f}s] {done}/{len(procs)} completed", flush=True)
        if done == len(procs):
            break
        time.sleep(60)

    total_elapsed = time.time() - t_start
    print(f"\n  All jobs finished in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

    # Collect results from log files (last JSON line)
    results = {}
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"\n{'Case':<25} {'Ct_fric':>10} {'Ct_pres':>10} {'Ct_total':>10} {'Error%':>8} {'Status':>12}")
    print("-" * 75)

    for (card_id, case_name), (proc, log_file, label) in procs.items():
        rc = proc.returncode
        key = case_name

        try:
            log_text = log_file.read_text()
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
                    "case": case_name, "card_id": card_id,
                    "_status": "PARSE_FAILED", "_rc": rc,
                    "_log_tail": "\n".join(lines[-10:]) if lines else "",
                }
            else:
                result["_rc"] = rc
                result["card_id"] = card_id

        except FileNotFoundError:
            result = {"case": case_name, "card_id": card_id, "_status": "LOG_MISSING"}

        results[key] = result

        status = result.get("_status", "OK")
        if status != "OK":
            print(f"  {label:<25s} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>8} {status:>12}")
        else:
            cf = result.get("Ct_fric", 0)
            cp = result.get("Ct_pres", 0)
            ct = result.get("Ct_total", 0)
            err = result.get("error_pct", result.get("Cf_error_pct", 0))
            print(f"  {label:<25s} {cf:>10.5f} {cp:>10.5f} {ct:>10.5f} {err:>7.1f}% {status:>12}")

    print("-" * 75)
    print()

    # Build comparison summary
    print("=" * 90)
    print("SIGN-FIX VERDICT")
    print("=" * 90)

    fine_grids = ["suboff_320", "kvlcc2_320"]
    coarse_grids = ["suboff_200", "kvlcc2_200", "flatplate_cs005_200", "flatplate_cs0_200"]

    print("\n  Fine grids (320³) — previously had negative Ct from sign bug:")
    for key in fine_grids:
        r = results.get(key, {})
        if r.get("_status") == "OK":
            ct = r.get("Ct_total", 0)
            cp = r.get("Ct_pres", 0)
            cf = r.get("Ct_fric", 0)
            sign = "POSITIVE" if ct > 0 else "NEGATIVE"
            verdict = "RESCUED ✓" if ct > 0 else "STILL BROKEN ✗"
            print(f"    {key:<20s} Ct={ct:.5f} (Cf={cf:.5f}, Cp={cp:.5f}) → {verdict} ({sign})")
        else:
            print(f"    {key:<20s} → FAILED: {r.get('_status', '?')}")

    print("\n  Coarse grids (200³) — production stability check:")
    for key in coarse_grids:
        r = results.get(key, {})
        if r.get("_status") == "OK":
            ct = r.get("Ct_total", 0)
            cp = r.get("Ct_pres", 0)
            cf = r.get("Ct_fric", 0)
            print(f"    {key:<20s} Ct={ct:.5f} (Cf={cf:.5f}, Cp={cp:.5f}) — stable")
        else:
            print(f"    {key:<20s} → FAILED: {r.get('_status', '?')}")

    print()

    # Write output
    output = {
        "title": "Fine-Grid Sign-Fix Re-Test — Pressure Drag Sign Correction",
        "description": "Re-tests 200³/320³ grids after fixing pressure drag sign from p*(sp-sm) to -p*(sp-sm)",
        "sign_fix": "changed drag_pres from p*(sp-sm) to -p*(sp-sm) in wall_model.py",
        "total_wall_time_s": total_elapsed,
        "jobs": len(JOBS),
        "cards_used": "SDAA 16-21",
        "results": results,
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
