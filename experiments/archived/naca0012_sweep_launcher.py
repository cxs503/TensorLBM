#!/usr/bin/env python3
"""NACA 0012 fine angle sweep launcher — parallel across 4 SDAA cards."""

import subprocess, sys, time, json, os
from pathlib import Path
from collections import defaultdict

WORKER = Path(__file__).parent / "naca0012_sweep_worker.py"
OUTDIR = Path("/tmp/naca0012_results")
RESULT_JSON = Path("/tmp/naca0012_full_polar.json")

# Map: SDAA card -> list of angles
CARD_ANGLES = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 8],
    3: [10, 12],
}

def run_one(did, alpha):
    """Run one simulation, returns subprocess.Popen."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path(__file__).parent}/src"
    cmd = [sys.executable, str(WORKER), str(did), str(alpha)]
    print(f"[LAUNCH] SDAA:{did} α={alpha}° → {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

def main():
    OUTDIR.mkdir(exist_ok=True)

    # Launch all
    procs = {}  # key=(did, alpha) -> Popen
    for did, angles in CARD_ANGLES.items():
        for alpha in angles:
            procs[(did, alpha)] = run_one(did, alpha)

    t0 = time.time()
    results = {}

    # Wait for all to complete
    for (did, alpha), proc in procs.items():
        out = proc.communicate()[0]
        elapsed = time.time() - t0
        rc = proc.returncode
        tag = f"[SDAA:{did} α={alpha}°]"
        if rc == 0:
            print(f"{tag} DONE (rc={rc}) after {elapsed:.0f}s")
        else:
            print(f"{tag} FAILED rc={rc} after {elapsed:.0f}s")
        # Show last 5 lines of output
        lines = out.strip().split("\n")
        for l in lines[-8:]:
            print(f"  {l}")

    # Collect JSON results
    for (did, alpha) in procs:
        out_file = OUTDIR / f"result_{did:02d}_a{int(alpha)}.json"
        if out_file.exists():
            try:
                results[f"{alpha}"] = json.loads(out_file.read_text())
            except Exception as e:
                print(f"WARNING: failed to parse {out_file}: {e}")
                results[f"{alpha}"] = {"alpha_deg": alpha, "error": str(e)}
        else:
            print(f"WARNING: no output found for SDAA:{did} α={alpha}°")
            results[f"{alpha}"] = {"alpha_deg": alpha, "error": "no output file"}

    # Build polar table
    polar = []
    for a in sorted([0, 1, 2, 3, 4, 5, 6, 8, 10, 12]):
        r = results.get(str(a), {})
        polar.append({
            "alpha_deg": a,
            "Cd_total": r.get("Cd_total"),
            "Cd_fric": r.get("Cd_fric"),
            "Cd_pres": r.get("Cd_pres"),
            "Cl_total": r.get("Cl_total"),
            "Cl_fric": r.get("Cl_fric"),
            "Cl_pres": r.get("Cl_pres"),
            "Cd_experimental": r.get("Cd_experimental"),
            "error_pct": r.get("error_pct"),
            "finite": r.get("finite"),
            "elapsed_s": r.get("elapsed_s"),
        })

    output = {
        "case": "NACA0012",
        "Re": 3e6,
        "Cs": 0.05,
        "grid": "200x80x80",
        "chord_lu": 80.0,
        "u_in": 0.06,
        "steps": 1500,
        "warmup": 500,
        "description": "Fine angle sweep to find stall onset where wall function breaks",
        "polar": polar,
        "stall_onset_analysis": "TBD — see Cd/Cl divergence vs experiment",
    }

    RESULT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nFull polar written to {RESULT_JSON}")

    # Analyze
    print("\n=== NACA 0012 FULL POLAR ===")
    print(f"{'α°':>5s}  {'Cd_LBM':>8s}  {'Cd_exp':>8s}  {'Err%':>7s}  {'Cl_LBM':>8s}  {'finite':>6s}")
    print("-" * 60)
    for p in polar:
        cd_s = f"{p['Cd_total']:.5f}" if p['Cd_total'] is not None else "N/A"
        cd_e = f"{p['Cd_experimental']:.5f}" if p['Cd_experimental'] else "N/A"
        err_s = f"{p['error_pct']:.1f}%" if p['error_pct'] is not None else "N/A"
        cl_s = f"{p['Cl_total']:.5f}" if p['Cl_total'] is not None else "N/A"
        fin = "OK" if p['finite'] else "DIV"
        print(f"{p['alpha_deg']:5.1f}  {cd_s:>8s}  {cd_e:>8s}  {err_s:>7s}  {cl_s:>8s}  {fin:>6s}")

if __name__ == "__main__":
    main()
