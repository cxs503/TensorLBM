#!/usr/bin/env python3
"""SDAA sweep for taylor_couette benchmark.

This script runs the taylor_couette verified benchmark on SDAA device and
collects device evidence, criteria values, and PASS/FAIL for each gate.

The original case uses CUDA with full step counts (~40k steps for ri=24).
For SDAA testing, we use reduced step counts to keep wall clock time reasonable.

NOTE: With reduced steps, the simulation may not fully converge, so some
criteria may FAIL. This is documented honestly in the results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import torch

# --------------------------------------------------------------------------- #
# Device evidence
# --------------------------------------------------------------------------- #
def get_device_evidence() -> dict:
    """Collect SDAA device information."""
    evidence = {
        "torch_version": torch.__version__,
        "sdaa_available": hasattr(torch, "sdaa") and torch.sdaa.is_available(),
    }
    if evidence["sdaa_available"]:
        evidence["sdaa_device_count"] = torch.sdaa.device_count()
        evidence["sdaa_device_name"] = torch.sdaa.get_device_name(0)
        try:
            evidence["sdaa_device_properties"] = str(torch.sdaa.get_device_properties(0))
        except Exception:
            evidence["sdaa_device_properties"] = "N/A"
    return evidence


# --------------------------------------------------------------------------- #
# Run the benchmark
# --------------------------------------------------------------------------- #
def run_taylor_couette_sdaa(ri: int = 24) -> dict:
    """Run taylor_couette on SDAA and return results.
    
    Uses reduced step counts for SDAA testing to keep wall clock time reasonable.
    """
    from verified.taylor_couette.run import run_case

    device = torch.device("sdaa")
    out_json = f"/tmp/taylor_couette_sdaa_ri{ri}.json"

    # Reduced parameters for SDAA testing
    # Full benchmark: min_mult=3.5, max_mult=10.0, avg_steps=1000
    # Reduced: min_mult=0.1, max_mult=0.2, avg_steps=50
    min_mult = 0.1
    max_mult = 0.2
    avg_steps = 50
    sample_interval = 50

    t0 = time.time()
    result = run_case(
        ri=ri,
        eta=0.5,
        re_i=20.0,
        nu=0.05,
        out_json=out_json,
        device=device,
        min_mult=min_mult,
        max_mult=max_mult,
        sample_interval=sample_interval,
        avg_steps=avg_steps,
        seed=0,
        compile_mode="eager",  # Use eager mode for SDAA
    )
    wall_clock = time.time() - t0

    result["wall_clock_seconds"] = wall_clock
    result["device"] = "sdaa"
    result["min_mult_used"] = min_mult
    result["max_mult_used"] = max_mult
    result["avg_steps_used"] = avg_steps
    return result


# --------------------------------------------------------------------------- #
# Criteria evaluation
# --------------------------------------------------------------------------- #
def evaluate_criteria(result: dict) -> dict:
    """Evaluate PASS/FAIL for each criterion gate."""
    gates = {}

    # Gate 1: Profile max error <= 3% (primary metric)
    profile_max_pct = result.get("profile_max_rel_bin_central_pct", float("inf"))
    gates["profile_max_le_3pct"] = {
        "value": profile_max_pct,
        "threshold": 3.0,
        "unit": "%",
        "pass": profile_max_pct <= 3.0,
    }

    # Gate 2: Profile L2 error <= 3%
    profile_l2_pct = result.get("profile_l2_rel_bin_central_pct", float("inf"))
    gates["profile_l2_le_3pct"] = {
        "value": profile_l2_pct,
        "threshold": 3.0,
        "unit": "%",
        "pass": profile_l2_pct <= 3.0,
    }

    # Gate 3: Torque cross-check all within 3%
    t_if_inner = abs(result.get("T_if_inner_err_pct", float("inf")))
    t_if_outer = abs(result.get("T_if_outer_err_pct", float("inf")))
    t_fw_outer = abs(result.get("T_fw_outer_err_pct", float("inf")))
    t_if_balance = abs(result.get("T_if_balance_pct", float("inf")))

    torque_all_within_3pct = (
        t_if_inner <= 3.0
        and t_if_outer <= 3.0
        and t_fw_outer <= 3.0
        and t_if_balance <= 3.0
    )
    gates["torque_crosscheck_all_within_3pct"] = {
        "values": {
            "T_if_inner_err_pct": result.get("T_if_inner_err_pct"),
            "T_if_outer_err_pct": result.get("T_if_outer_err_pct"),
            "T_fw_outer_err_pct": result.get("T_fw_outer_err_pct"),
            "T_if_balance_pct": result.get("T_if_balance_pct"),
        },
        "threshold": 3.0,
        "unit": "%",
        "pass": torque_all_within_3pct,
    }

    # Gate 4: Steady state reached
    gates["steady_state_reached"] = {
        "value": result.get("steady", False),
        "pass": result.get("steady", False),
    }

    # Gate 5: Wall clock time <= 20 minutes (1200 seconds)
    wall_clock = result.get("wall_clock_seconds", float("inf"))
    gates["wall_clock_le_20min"] = {
        "value": wall_clock,
        "threshold": 1200.0,
        "unit": "seconds",
        "pass": wall_clock <= 1200.0,
    }

    # Overall pass
    all_pass = all(g["pass"] for g in gates.values())
    gates["OVERALL"] = {"pass": all_pass}

    return gates


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("SDAA Sweep: taylor_couette benchmark")
    print("=" * 70)

    # Device evidence
    print("\n[1/4] Collecting device evidence...")
    device_evidence = get_device_evidence()
    print(f"  SDAA available: {device_evidence['sdaa_available']}")
    if device_evidence["sdaa_available"]:
        print(f"  Device name: {device_evidence['sdaa_device_name']}")
        print(f"  Device count: {device_evidence['sdaa_device_count']}")

    if not device_evidence["sdaa_available"]:
        print("\nERROR: SDAA device not available. Cannot proceed.")
        output = {
            "case": "taylor_couette",
            "device_evidence": device_evidence,
            "error": "SDAA device not available",
            "verdict": "FAIL",
        }
        out_path = REPO_ROOT / "artifacts" / "sdaa_taylor_couette.json"
        out_path.write_text(json.dumps(output, indent=2))
        sys.exit(1)

    # Run benchmark
    print("\n[2/4] Running taylor_couette on SDAA (ri=24, reduced steps)...")
    print("  Using reduced step counts for SDAA testing (min_mult=0.1, max_mult=0.2)")
    print("  Full benchmark requires ~40k steps; we use ~2k steps for faster testing")
    try:
        result = run_taylor_couette_sdaa(ri=24)
        print(f"  Completed in {result['wall_clock_seconds']:.1f} seconds")
        print(f"  Steps: {result.get('n_steps', 'N/A')}")
        print(f"  Steady: {result.get('steady', 'N/A')}")
    except Exception as e:
        print(f"\nERROR: Benchmark failed with exception: {e}")
        import traceback
        traceback.print_exc()
        output = {
            "case": "taylor_couette",
            "device_evidence": device_evidence,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "verdict": "FAIL",
        }
        out_path = REPO_ROOT / "artifacts" / "sdaa_taylor_couette.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  Error details written to: {out_path}")
        sys.exit(1)

    # Evaluate criteria
    print("\n[3/4] Evaluating criteria...")
    gates = evaluate_criteria(result)
    for gate_name, gate_info in gates.items():
        if gate_name == "OVERALL":
            continue
        status = "PASS" if gate_info["pass"] else "FAIL"
        if "value" in gate_info:
            val = gate_info["value"]
            if isinstance(val, float):
                print(f"  {gate_name}: {val:.4f} {gate_info.get('unit', '')} [{status}]")
            else:
                print(f"  {gate_name}: {val} [{status}]")
        elif "values" in gate_info:
            print(f"  {gate_name}: [{status}]")
            for k, v in gate_info["values"].items():
                if v is not None:
                    print(f"    {k}: {v:.4f} %")
                else:
                    print(f"    {k}: N/A")

    # Build output
    print("\n[4/4] Writing results...")
    output = {
        "case": "taylor_couette",
        "device_evidence": device_evidence,
        "benchmark_result": {
            "ri": result.get("ri"),
            "ro": result.get("ro"),
            "n_grid": result.get("n_grid"),
            "Re_i": result.get("Re_i"),
            "tau": result.get("tau"),
            "omega_i": result.get("omega_i"),
            "n_steps": result.get("n_steps"),
            "steady": result.get("steady"),
            "wall_clock_seconds": result.get("wall_clock_seconds"),
            "device": result.get("device"),
            "min_mult_used": result.get("min_mult_used"),
            "max_mult_used": result.get("max_mult_used"),
            "avg_steps_used": result.get("avg_steps_used"),
            # Primary metrics
            "profile_max_rel_bin_central_pct": result.get("profile_max_rel_bin_central_pct"),
            "profile_l2_rel_bin_central_pct": result.get("profile_l2_rel_bin_central_pct"),
            "percell_max_rel_central_pct": result.get("percell_max_rel_central_pct"),
            "nominal_max_rel_bin_central_pct": result.get("nominal_max_rel_bin_central_pct"),
            # Effective geometry
            "ri_eff": result.get("ri_eff"),
            "ro_eff": result.get("ro_eff"),
            "delta_i": result.get("delta_i"),
            "delta_o": result.get("delta_o"),
            # Torque cross-check
            "T_if_inner_err_pct": result.get("T_if_inner_err_pct"),
            "T_if_outer_err_pct": result.get("T_if_outer_err_pct"),
            "T_fw_outer_err_pct": result.get("T_fw_outer_err_pct"),
            "T_if_balance_pct": result.get("T_if_balance_pct"),
            # Disclosed variants
            "T_lw_inner_ratio_disclosed": result.get("T_lw_inner_ratio_disclosed"),
            "T_lw_outer_ratio_disclosed": result.get("T_lw_outer_ratio_disclosed"),
            "M_ref_from_Bfit": result.get("M_ref_from_Bfit"),
        },
        "criteria_gates": gates,
        "verdict": "PASS" if gates["OVERALL"]["pass"] else "FAIL",
        "notes": (
            "Reduced step counts used for SDAA testing (min_mult=0.1, max_mult=0.2, avg_steps=50). "
            "Full benchmark requires min_mult=3.5, max_mult=10.0, avg_steps=1000 (~40k steps). "
            "With reduced steps, simulation may not fully converge, so some criteria may FAIL. "
            "This demonstrates SDAA functionality, not production-quality convergence."
        ),
    }

    out_path = REPO_ROOT / "artifacts" / "sdaa_taylor_couette.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results written to: {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Device: {device_evidence.get('sdaa_device_name', 'N/A')}")
    print(f"  Case: taylor_couette (ri=24, Re_i=20)")
    print(f"  Wall clock: {result.get('wall_clock_seconds', 0):.1f} s")
    print(f"  Steps: {result.get('n_steps', 'N/A')}")
    print(f"  Steady state: {result.get('steady', 'N/A')}")
    print(f"  Verdict: {output['verdict']}")
    print("=" * 70)

    return 0 if output["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
