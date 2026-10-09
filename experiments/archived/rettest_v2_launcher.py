#!/usr/bin/env python3
"""Launch all 14 benchmark cases in parallel on SDAA cards 0-13."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CASES = [
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
]

OUTPUT_DIR = Path("/tmp/rettest_v2")
WORKER = Path(__file__).parent / "rettest_worker.py"
ENV = dict(os.environ, PYTHONPATH="src")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    t0 = time.time()

    for case_name, dev_id in CASES:
        out_path = OUTPUT_DIR / f"{case_name}.json"
        log_path = OUTPUT_DIR / f"{case_name}.log"
        log_f = open(log_path, "w")
        print(f"Launching {case_name} on sdaa:{dev_id} -> {out_path}")
        p = subprocess.Popen(
            [sys.executable, str(WORKER), case_name, str(dev_id), str(out_path)],
            stdout=log_f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent),
            env=ENV,
        )
        procs.append((case_name, dev_id, p, log_f))

    # Wait for all
    results = []
    for case_name, dev_id, p, log_f in procs:
        rc = p.wait()
        log_f.close()
        out_path = OUTPUT_DIR / f"{case_name}.json"
        if out_path.exists():
            data = json.loads(out_path.read_text())
            results.append(data)
            status = data.get("status", "?")
            val = data.get(f"{data['ref_name']}_mean", float('nan'))
            ref = data.get(f"{data['ref_name']}_ref", float('nan'))
            err = data.get("error_pct", float('nan'))
            print(f"  {case_name:30s} sdaa:{dev_id:2d}  {data['ref_name']}={val:.6f} ref={ref} err={err:.1f}% {status}")
        else:
            print(f"  {case_name:30s} sdaa:{dev_id:2d}  FAILED (rc={rc})")
            results.append({"case": case_name, "status": "CRASH", "error_pct": float('nan')})

    elapsed = time.time() - t0
    n_pass = sum(1 for r in results if r.get("status") == "OK" and r.get("error_pct", 999) < 15)
    n_div = sum(1 for r in results if r.get("status") == "DIV")
    n_fail = sum(1 for r in results if r.get("status") not in ("OK",) or r.get("error_pct", 999) >= 15)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_pass}/{len(results)} PASS (<15% err), {n_div} DIV, {n_fail} FAIL")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"{'='*70}")

    # Write summary
    summary = {
        "total": len(results),
        "pass": n_pass,
        "div": n_div,
        "fail": n_fail,
        "elapsed_s": elapsed,
        "cases": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {OUTPUT_DIR / 'summary.json'}")

if __name__ == "__main__":
    main()
