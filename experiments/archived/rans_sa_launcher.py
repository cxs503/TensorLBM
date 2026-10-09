#!/usr/bin/env python3
"""RANS SA Benchmark Launcher — runs 4 separation-flow tests in parallel on SDAA 12-15.

Cylinder Re=200  (SDAA:12),  Sphere Re=1000 (SDAA:13),
Square Prism Re=22000 (SDAA:14), Backward Step Re=5000 (SDAA:15).

Worker: rans_sa_worker.py — D3Q19 MRT + SASolver + wallfn log-law.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent
WORKER = PROJECT / "rans_sa_worker.py"
OUT_DIR = Path("/tmp/rans_bench")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    {"case": "cylinder",      "device": 12},
    {"case": "sphere",        "device": 13},
    {"case": "square_prism",  "device": 14},
    {"case": "backward_step", "device": 15},
]


def main():
    # Verify test parameters
    print("=" * 70)
    print("RANS SA Benchmark Launcher")
    print("=" * 70)
    for t in TESTS:
        # Build command to verify args
        cmd = [sys.executable, str(WORKER), str(t["device"]), t["case"]]
        print(f"  {t['case']:20s} → SDAA:{t['device']}   (cmd: {' '.join(cmd)})")
    print("=" * 70)
    print()

    # Launch all tests in parallel
    procs = {}
    for t in TESTS:
        out_path = OUT_DIR / f"{t['case']}.json"
        cmd = [sys.executable, str(WORKER), str(t["device"]), t["case"], str(out_path)]
        
        print(f"Launching {t['case']} on SDAA:{t['device']} ...")
        log_path = OUT_DIR / f"{t['case']}.log"
        with open(log_path, "w") as log:
            p = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "PYTHONPATH": str(PROJECT / "src")},
            )
        procs[t["case"]] = p

    print(f"\nAll {len(procs)} tests launched. Waiting for completion...\n")

    # Wait for completion
    start = time.time()
    results = {}
    for case, p in procs.items():
        p.wait()
        elapsed = time.time() - start
        out_path = OUT_DIR / f"{case}.json"
        status = "UNKNOWN"
        if out_path.exists():
            try:
                result = json.loads(out_path.read_text())
                results[case] = result
                status = result.get("status", "ok")
                cd = result.get("Cd_mean", float("nan"))
                err = result.get("error_pct", float("nan"))
                ref = result.get("Cd_ref", float("nan"))
                extra = ""
                if case == "backward_step":
                    xrh = result.get("xr_h_mean", float("nan"))
                    extra = f", xr/h={xrh:.2f}"
                print(f"  {case:20s} rc={p.returncode}  status={status}  "
                      f"Cd={cd:.4f} (ref={ref}) err={err if isinstance(err, str) else f'{err:.1f}%'}"
                      f"{extra}  [{elapsed:.0f}s]")
            except Exception as ex:
                print(f"  {case:20s} rc={p.returncode}  JSON parse error: {ex}")
                results[case] = {"case": case, "status": "parse_error", "error": str(ex)}
        else:
            print(f"  {case:20s} rc={p.returncode}  no output JSON")
            results[case] = {"case": case, "status": "no_output", "returncode": p.returncode}

    # Write combined results
    combined_path = Path("/tmp/rans_bench_results.json")
    combined = {
        "description": "RANS SA model benchmarks on 4 separation-flow cases",
        "model": "D3Q19 MRT + Spalart-Allmaras RANS + wallfn log-law",
        "settings": {
            "n_steps": 2000,
            "warmup": 300,
            "sa_initial_nu_tilde": "10*nu",
            "sa_nu_t_max": 0.5,
            "wall_bc": "nu_tilde=0 at solid",
        },
        "results": results,
        "summary": {},
    }

    # Build summary
    for case, r in results.items():
        cd = r.get("Cd_mean", float("nan"))
        ref = r.get("Cd_ref", float("nan"))
        err = r.get("error_pct", float("nan"))
        st = r.get("status", "unknown")
        summary = {"Cd": cd, "Cd_ref": ref, "error_pct": err, "status": st}
        if case == "backward_step":
            summary["xr_h"] = r.get("xr_h_mean", float("nan"))
        combined["summary"][case] = summary

    combined_path.write_text(json.dumps(combined, indent=2))
    print(f"\nResults written to {combined_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("SUMMARY: Does SA fix any of the 4 failing benchmarks?")
    print("=" * 70)
    for case, r in results.items():
        cd = r.get("Cd_mean", float("nan"))
        ref = r.get("Cd_ref", float("nan"))
        err = r.get("error_pct", float("nan"))
        st = r.get("status", "unknown")
        if isinstance(err, (int, float)) and math.isfinite(err):
            improved = err < 30  # threshold for "improvement"
            print(f"  {case:20s}: Cd={cd:.4f} (ref={ref}) err={err:.1f}%  "
                  f"{'✅ IMPROVED' if improved else '❌ STILL OFF'}")
        else:
            extra_info = ""
            if case == "backward_step":
                xrh = r.get("xr_h_mean", float("nan"))
                extra_info = f"xr/h={xrh:.2f}"
            print(f"  {case:20s}: status={st}  {extra_info}")

    return 0


if __name__ == "__main__":
    import math
    sys.exit(main())
