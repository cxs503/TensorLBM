"""PG Phase 2: Launch 4 alpha values in parallel on SDAA 0-3.

α=0.3 on sdaa:0, α=0.5 on sdaa:1, α=0.7 on sdaa:2, α=1.0 on sdaa:3
160³ bare_hull, 1000 steps, MRT+Smag Cs=0.05, Re=2e6.

Waits for all to complete, then generates /tmp/pg_parallel_results.json.
"""
import json, subprocess, sys, time
from pathlib import Path

WORKER_SCRIPT = Path(__file__).resolve().parent / "pg_worker.py"
OUTPUT_PATH = Path("/tmp/pg_parallel_results.json")

# Config: (alpha, device_id)
CONFIGS = [
    (0.3, 0),
    (0.5, 1),
    (0.7, 2),
    (1.0, 3),
]

def main():
    print("=" * 60)
    print("PG Phase 2: Parallel alpha sweep on 4 SDAA cards")
    print(f"Worker script: {WORKER_SCRIPT}")
    print("=" * 60)

    processes = []

    for alpha, did in CONFIGS:
        log_path = Path(f"/tmp/pg_worker_a{alpha}_d{did}.log")

        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "src")

        # Use shell redirection so output is flushed to disk immediately
        worker_cmd = (
            f"{sys.executable} {WORKER_SCRIPT} {alpha} {did} "
            f"--steps 1000 --grid 160 "
            f">{log_path} 2>&1"
        )
        print(f"\nLaunching α={alpha} on sdaa:{did} ...")
        proc = subprocess.Popen(
            worker_cmd,
            shell=True,
            env=env,
            cwd=str(Path(__file__).resolve().parent),
        )
        processes.append((alpha, did, proc, log_path))

    print(f"\nAll {len(processes)} workers launched. Waiting for completion...")
    print("(This may take a while — 160³ × 1000 steps on 4 GPUs)\n")

    # Wait for all
    for alpha, did, proc, log_path in processes:
        print(f"Waiting for α={alpha} on sdaa:{did} (PID {proc.pid})...")
        rc = proc.wait()
        if rc == 0:
            print(f"  α={alpha} on sdaa:{did}: COMPLETED (rc={rc})")
        else:
            print(f"  α={alpha} on sdaa:{did}: FAILED (rc={rc})")

    # Read individual JSON results
    print("\n" + "=" * 60)
    print("Collecting results...")
    print("=" * 60)

    all_results = []
    for alpha, did in CONFIGS:
        json_path = Path(f"/tmp/pg_worker_a{alpha}_d{did}.json")
        if json_path.exists():
            data = json.loads(json_path.read_text())
            all_results.append(data)
        else:
            # Try reading from log
            log_path = Path(f"/tmp/pg_worker_a{alpha}_d{did}.log")
            log_text = log_path.read_text() if log_path.exists() else ""
            all_results.append({
                "alpha": alpha, "device_id": did,
                "error": "no_result_file",
                "finite": False,
                "log_tail": log_text[-2000:] if log_text else "",
            })

    # Build summary
    summary = {
        "task": "PG Phase 2: Parallel alpha sweep",
        "grid": "160x160x160",
        "n_steps": 1000,
        "re": 2.0e6,
        "smagorinsky_cs": 0.05,
        "method": "MRT+Smag",
        "alphas_tested": [a for a, _ in CONFIGS],
        "results": [],
    }

    for r in all_results:
        s500 = r.get("step_500", {}) or {}
        s1000 = r.get("step_1000", {}) or {}
        entry = {
            "alpha": r["alpha"],
            "device": r["device_id"],
            "finite": r.get("finite", False),
            "elapsed_s": r.get("elapsed_s", None),
            "ct_pres_mean": r.get("ct_pres_mean"),
            "ct_pres_var": r.get("ct_pres_var"),
            "ct_pres_std": r.get("ct_pres_std"),
            "ct_total_step_500": s500.get("ct_total"),
            "ct_total_step_1000": s1000.get("ct_total"),
            "ct_pres_step_500": s500.get("ct_pres"),
            "ct_pres_step_1000": s1000.get("ct_pres"),
            "ct_fric_step_500": s500.get("ct_fric"),
            "ct_fric_step_1000": s1000.get("ct_fric"),
        }
        if "error" in r:
            entry["error"] = r["error"]
        summary["results"].append(entry)

    # Determine best alpha
    valid = [r for r in summary["results"] if r.get("finite") and r["ct_pres_var"] is not None]
    if valid:
        # Best: lowest Ct_pres variance (most stable pressure signal)
        best_by_var = min(valid, key=lambda x: x["ct_pres_var"])
        # Best: closest Ct_total to target 0.004 at step 1000
        target_ct = 0.004
        valid_with_ct = [r for r in valid if r["ct_total_step_1000"] is not None]
        best_by_ct = min(valid_with_ct, key=lambda x: abs(x["ct_total_step_1000"] - target_ct)) if valid_with_ct else None

        summary["best_alpha_by_variance"] = {
            "alpha": best_by_var["alpha"],
            "ct_pres_var": best_by_var["ct_pres_var"],
            "criterion": "lowest Ct_pres variance = most stable pressure signal",
        }
        if best_by_ct:
            summary["best_alpha_by_ct_proximity"] = {
                "alpha": best_by_ct["alpha"],
                "ct_total_step_1000": best_by_ct["ct_total_step_1000"],
                "target_ct": target_ct,
                "criterion": f"closest Ct_total to target {target_ct}",
            }

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        header = f"{'α':>6} {'Finite':>6} {'Ct_pres_var':>14} {'Ct_tot@500':>12} {'Ct_tot@1000':>12} {'Ct_pres@1000':>12} {'Time(s)':>8}"
        print(header)
        print("-" * len(header))
        for e in sorted(summary["results"], key=lambda x: x["alpha"]):
            print(f"{e['alpha']:>6.1f} {str(e['finite']):>6} "
                  f"{e.get('ct_pres_var', 'N/A'):>14} "
                  f"{e.get('ct_total_step_500', 'N/A'):>12} "
                  f"{e.get('ct_total_step_1000', 'N/A'):>12} "
                  f"{e.get('ct_pres_step_1000', 'N/A'):>12} "
                  f"{e.get('elapsed_s', 'N/A'):>8}")

        print(f"\nBEST by variance: α={best_by_var['alpha']} (Ct_pres_var={best_by_var['ct_pres_var']:.6e})")
        if best_by_ct:
            print(f"BEST by Ct proximity to {target_ct}: α={best_by_ct['alpha']} (Ct_tot@1000={best_by_ct['ct_total_step_1000']:.6f})")
    else:
        summary["best_alpha"] = None
        print("\nERROR: No valid finite results!")

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results saved to {OUTPUT_PATH}")

    # Also print which alpha significantly outperforms base
    print("\n" + "=" * 60)
    print("ANALYSIS: PG Correction Effectiveness")
    print("=" * 60)
    if valid:
        for e in sorted(summary["results"], key=lambda x: x["alpha"]):
            var_str = f"{e.get('ct_pres_var', 0):.6e}" if e.get('ct_pres_var') is not None else "N/A"
            ct1000 = e.get('ct_total_step_1000')
            ct1000_str = f"{ct1000:.6f}" if ct1000 is not None else "N/A"
            print(f"  α={e['alpha']:.1f}: Ct_pres_var={var_str}, Ct_tot@1000={ct1000_str}, finite={e['finite']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
