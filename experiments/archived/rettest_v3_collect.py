#!/usr/bin/env python3
"""Collect results from all rettest_v3 JSON files and print summary."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import rettest_v3_worker as w

OUTPUT_DIR = Path("/tmp/rettest_v3")

def main():
    results = []
    for case_name, dev_id in [
        ("cylinder_Re200", 0),
        ("square_prism_Re100", 1),
        ("square_prism_Re22000", 2),
        ("naca0012_Re6e6", 3),
        ("naca4412_Re3e6", 4),
        ("s809_Re2e6", 5),
        ("backward_step_Re5000", 6),
        ("rect_prism_Re2e4", 7),
        ("flat_plate_Re2e6", 8),
        ("sphere_Re1000", 9),
        ("kvlcc2_Re2e6", 10),
        ("suboff_Re2e6", 11),
        ("ahmed25_Re2e6", 12),
        ("tandem_cyl_Re100", 13),
    ]:
        out_path = OUTPUT_DIR / f"{case_name}.json"
        cfg = w.CASES.get(case_name, {})
        rn = cfg.get("ref_name", "Cd")
        if out_path.exists():
            data = json.loads(out_path.read_text())
            rn = data.get("ref_name", rn)
            status = data.get("status", "?")
            val = data.get(f"{rn}_mean", float('nan'))
            ref = data.get(f"{rn}_ref", cfg.get("ref", float('nan')))
            err = data.get("error_pct", float('nan'))
            grid = data.get("grid", "?")
            elapsed = data.get("elapsed_s", 0)
            results.append(data)
            print(f"  {case_name:30s} sdaa:{dev_id:2d}  {rn}={val:.6f} ref={ref} err={err:.1f}% {status} ({grid}, {elapsed:.0f}s)")
        else:
            results.append({"case": case_name, "status": "MISSING", "error_pct": float('nan')})
            print(f"  {case_name:30s} sdaa:{dev_id:2d}  MISSING (no JSON)")

    n_pass = sum(1 for r in results if r.get("status") == "OK" and r.get("error_pct", 999) < 15)
    n_div = sum(1 for r in results if r.get("status") == "DIV")
    n_fail = sum(1 for r in results if r.get("status") not in ("OK",) or r.get("error_pct", 999) >= 15)
    n_missing = sum(1 for r in results if r.get("status") == "MISSING")

    print(f"\n{'='*80}")
    print(f"SUMMARY V3 (correct Ladd): {n_pass}/{len(results)} PASS (<15% err), {n_div} DIV, {n_fail} FAIL, {n_missing} MISSING")
    print(f"{'='*80}")

    summary = {
        "version": "v3_correct_ladd",
        "total": len(results),
        "pass": n_pass,
        "div": n_div,
        "fail": n_fail,
        "missing": n_missing,
        "cases": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
