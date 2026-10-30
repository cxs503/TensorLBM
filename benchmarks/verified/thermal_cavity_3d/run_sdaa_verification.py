#!/usr/bin/env python3
"""SDAA verification script for thermal_cavity_3d benchmark.

This script runs the full verification ladder on SDAA device and generates
evidence files with device information.

Based on m294 diagnosis:
1. Window drift FAIL - need 100k+ steps to reach steady state
2. Monotonicity NOT_RUN - need at least 3 grids per Ra

This script addresses both issues by:
- Running with sufficient steps (100k-800k depending on Ra and grid size)
- Ensuring at least 3 grids per Ra ladder
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch


def get_device_info():
    """Get SDAA device information."""
    info = {
        "device_type": "sdaa",
        "device_count": 0,
        "device_name": None,
        "torch_version": torch.__version__,
    }
    if hasattr(torch, "sdaa"):
        info["device_count"] = torch.sdaa.device_count()
        if info["device_count"] > 0:
            info["device_name"] = torch.sdaa.get_device_name(0)
    return info


def run_case(ra, n, steps, device="sdaa", tag=None):
    """Run a single case and return the result."""
    cmd = [
        sys.executable,
        "run.py",
        "--ra",
        str(ra),
        "--n",
        str(n),
        "--steps",
        str(steps),
        "--device",
        device,
    ]
    if tag:
        cmd.extend(["--tag", tag])

    print(f"\n{'='*60}")
    print(f"Running: Ra={ra}, N={n}, steps={steps}, device={device}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        return None

    print(result.stdout)
    print(f"Elapsed: {elapsed:.1f}s")

    # Load the result
    tag = tag or f"fm_ra{ra:.0e}_n{n}".replace("e+0", "e")
    out_file = Path("out") / f"probe_{tag}.json"
    if out_file.exists():
        with open(out_file) as f:
            return json.load(f)
    return None


def analyze_window_drift(history, window_frac=0.2):
    """Analyze window drift for steady state check."""
    if not history:
        return None

    n_samples = len(history)
    n_window = max(1, int(n_samples * window_frac))
    window = history[-n_window:]

    nu_values = [h["nu"] for h in window]
    nu_mean = sum(nu_values) / len(nu_values)

    # Drift: difference between first and last quarter of window
    n_quarter = max(1, n_window // 4)
    first_quarter = sum(nu_values[:n_quarter]) / n_quarter
    last_quarter = sum(nu_values[-n_quarter:]) / n_quarter

    drift_pct = 100.0 * (last_quarter - first_quarter) / nu_mean if nu_mean > 0 else 0.0

    return {
        "nu_mean": nu_mean,
        "drift_pct": drift_pct,
        "n_samples": n_samples,
        "n_window": n_window,
    }


def main():
    print("=" * 60)
    print("SDAA Verification for thermal_cavity_3d")
    print("=" * 60)

    # Get device info
    device_info = get_device_info()
    print(f"\nDevice Info:")
    print(json.dumps(device_info, indent=2))

    # Define verification ladder
    # Based on m294 diagnosis: need 100k+ steps for steady state
    # and at least 3 grids per Ra for monotonicity
    ladder = [
        # Ra=1e3: 3 grids, 100k-200k steps
        {"ra": 1e3, "n": 32, "steps": 100000, "tag": "sdaa_ra1e3_n32"},
        {"ra": 1e3, "n": 48, "steps": 150000, "tag": "sdaa_ra1e3_n48"},
        {"ra": 1e3, "n": 64, "steps": 200000, "tag": "sdaa_ra1e3_n64"},
        # Ra=1e4: 3 grids, 200k-500k steps
        {"ra": 1e4, "n": 32, "steps": 200000, "tag": "sdaa_ra1e4_n32"},
        {"ra": 1e4, "n": 48, "steps": 300000, "tag": "sdaa_ra1e4_n48"},
        {"ra": 1e4, "n": 64, "steps": 500000, "tag": "sdaa_ra1e4_n64"},
        # Ra=1e5: 3 grids, 300k-800k steps
        {"ra": 1e5, "n": 64, "steps": 400000, "tag": "sdaa_ra1e5_n64"},
        {"ra": 1e5, "n": 96, "steps": 600000, "tag": "sdaa_ra1e5_n96"},
        {"ra": 1e5, "n": 128, "steps": 800000, "tag": "sdaa_ra1e5_n128"},
    ]

    # Reference values (cluster median from literature)
    reference = {
        "1e3": 1.07065,
        "1e4": 2.05552,
        "1e5": 4.3411,
    }

    # Run all cases
    results = {}
    for case in ladder:
        ra = case["ra"]
        n = case["n"]
        steps = case["steps"]
        tag = case["tag"]

        result = run_case(ra, n, steps, device="sdaa", tag=tag)
        if result:
            key = f"ra{ra:.0e}_n{n}"
            results[key] = {
                "config": case,
                "result": result,
                "window_analysis": analyze_window_drift(result.get("history", [])),
            }

    # Generate verification report
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    report = {
        "device_info": device_info,
        "ladder": {},
        "summary": {},
    }

    for ra_str, ref_nu in reference.items():
        ra = float(ra_str)
        ra_key = f"ra{ra:.0e}"
        report["ladder"][ra_key] = {
            "reference_nu": ref_nu,
            "cases": [],
        }

        # Collect cases for this Ra
        for key, data in results.items():
            if key.startswith(ra_key):
                case_result = data["result"]
                window = data["window_analysis"]
                config = data["config"]

                nu_win = window["nu_mean"] if window else None
                err_pct = (
                    100.0 * (nu_win - ref_nu) / ref_nu if nu_win and ref_nu else None
                )

                case_info = {
                    "n": config["n"],
                    "steps": config["steps"],
                    "tag": config["tag"],
                    "nu_win": nu_win,
                    "err_pct": err_pct,
                    "window_drift_pct": window["drift_pct"] if window else None,
                    "nan_at": case_result.get("nan_at"),
                }
                report["ladder"][ra_key]["cases"].append(case_info)

        # Sort by grid size
        report["ladder"][ra_key]["cases"].sort(key=lambda x: x["n"])

        # Check monotonicity
        errs = [c["err_pct"] for c in report["ladder"][ra_key]["cases"] if c["err_pct"] is not None]
        if len(errs) >= 3:
            # Check if |err| is monotonically decreasing
            abs_errs = [abs(e) for e in errs]
            monotone = all(abs_errs[i] >= abs_errs[i + 1] for i in range(len(abs_errs) - 1))
            report["ladder"][ra_key]["monotone"] = monotone
            report["ladder"][ra_key]["finest_err_pct"] = errs[-1]
            report["ladder"][ra_key]["pass"] = monotone and abs(errs[-1]) <= 3.0
        else:
            report["ladder"][ra_key]["monotone"] = False
            report["ladder"][ra_key]["pass"] = False

    # Print summary
    print("\nResults by Ra:")
    for ra_key, data in report["ladder"].items():
        print(f"\n{ra_key} (reference Nu={data['reference_nu']:.5f}):")
        print(f"  {'N':>4} {'steps':>8} {'Nu_win':>10} {'err%':>8} {'drift%':>10} {'NaN':>5}")
        for case in data["cases"]:
            print(
                f"  {case['n']:>4} {case['steps']:>8} {case['nu_win']:>10.5f} "
                f"{case['err_pct']:>8.3f} {case['window_drift_pct']:>10.6f} "
                f"{str(case['nan_at']):>5}"
            )
        print(f"  Monotonicity: {data.get('monotone', 'N/A')}")
        print(f"  Finest err: {data.get('finest_err_pct', 'N/A'):.3f}%")
        print(f"  PASS: {data.get('pass', False)}")

    # Overall verdict
    all_pass = all(data.get("pass", False) for data in report["ladder"].values())
    report["summary"]["all_pass"] = all_pass
    report["summary"]["verdict"] = "VERIFIED" if all_pass else "NOT_VERIFIED"

    print(f"\n{'='*60}")
    print(f"OVERALL VERDICT: {report['summary']['verdict']}")
    print(f"{'='*60}")

    # Save report
    report_file = Path("out/sdaa_verification_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
