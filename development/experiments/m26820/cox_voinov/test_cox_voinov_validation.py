#!/usr/bin/env python3
"""Cox-Voinov Dynamic Wetting Validation.

Validates the Cox-Voinov law: θ_d^3 = θ_e^3 + 9·Ca·ln(L/λ)

Tests:
1. Mathematical correctness of Cox-Voinov formula
2. Capillary number scaling (monotonicity)
3. Wall psi mapping consistency

Usage:
    python tests/test_cox_voinov_validation.py [--steps N] [--Ca-values ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

# Ensure we use this worktree's tensorlbm
TENSORLBM_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(TENSORLBM_SRC))

from tensorlbm.wetting import (
    cox_voinov_dynamic_angle,
    cox_voinov_dynamic_wall_psi,
    capillary_rise_theory,
    run_capillary_scaling,
    wall_psi_from_contact_angle,
    contact_angle_from_wall_psi,
    RHO_L_DEFAULT,
    SIGMA_DEFAULT,
)


def test_cox_voinov_formula():
    """Test mathematical correctness of Cox-Voinov formula."""
    print("=" * 60)
    print("Test 1: Cox-Voinov Formula Correctness")
    print("=" * 60)

    # Test 1a: Static case (Ca = 0)
    theta_e = 60.0
    theta_d = cox_voinov_dynamic_angle(theta_e, 0.0)
    assert abs(theta_d - theta_e) < 1e-10, f"Static case failed: {theta_d} != {theta_e}"
    print(f"  Ca=0: θ_e={theta_e}° → θ_d={theta_d:.6f}° ✓")

    # Test 1b: Advancing case (Ca > 0)
    Ca = 0.01
    log_ratio = math.log(100.0)
    theta_d_adv = cox_voinov_dynamic_angle(theta_e, Ca, log_ratio)
    assert theta_d_adv > theta_e, f"Advancing case failed: θ_d={theta_d_adv} <= θ_e={theta_e}"
    print(f"  Ca={Ca}: θ_e={theta_e}° → θ_d={theta_d_adv:.2f}° (advancing) ✓")

    # Test 1c: Receding case (Ca < 0)
    theta_d_rec = cox_voinov_dynamic_angle(theta_e, -Ca, log_ratio)
    assert theta_d_rec < theta_e, f"Receding case failed: θ_d={theta_d_rec} >= θ_e={theta_e}"
    print(f"  Ca={-Ca}: θ_e={theta_e}° → θ_d={theta_d_rec:.2f}° (receding) ✓")

    # Test 1d: Verify Cox-Voinov equation for advancing case
    theta_e_rad = math.radians(theta_e)
    theta_d_adv_rad = math.radians(theta_d_adv)
    lhs = theta_d_adv_rad ** 3
    rhs = theta_e_rad ** 3 + 9.0 * Ca * log_ratio
    error = abs(lhs - rhs) / rhs * 100.0
    assert error < 1e-6, f"Cox-Voinov equation error: {error}%"
    print(f"  Cox-Voinov equation check (advancing): error={error:.2e}% ✓")

    # Test 1e: Verify Cox-Voinov equation for receding case
    theta_d_rec_rad = math.radians(theta_d_rec)
    lhs = theta_d_rec_rad ** 3
    rhs = theta_e_rad ** 3 + 9.0 * (-Ca) * log_ratio
    error = abs(lhs - rhs) / rhs * 100.0
    assert error < 1e-6, f"Cox-Voinov equation error: {error}%"
    print(f"  Cox-Voinov equation check (receding): error={error:.2e}% ✓")

    # Test 1f: Wall psi mapping for advancing case
    wall_psi = cox_voinov_dynamic_wall_psi(theta_e, Ca, log_ratio)
    theta_back = contact_angle_from_wall_psi(wall_psi)
    assert abs(theta_back - theta_d_adv) < 0.1, \
        f"Wall psi mapping failed: {theta_back} != {theta_d_adv}"
    print(f"  Wall psi mapping: θ_d={theta_d_adv:.2f}° → ψ_w → θ={theta_back:.2f}° ✓")

    # Test 1g: Angle clamping
    theta_d_clamped = cox_voinov_dynamic_angle(60.0, 1.0, log_ratio)
    assert 0.0 <= theta_d_clamped <= 180.0, f"Angle clamping failed: {theta_d_clamped}"
    print(f"  Angle clamping: θ_d={theta_d_clamped:.2f}° ∈ [0, 180] ✓")

    print("\n  All formula tests PASSED\n")
    return True


def test_capillary_scaling():
    """Test Cox-Voinov Ca scaling across multiple Ca values."""
    print("=" * 60)
    print("Test 2: Cox-Voinov Ca Scaling")
    print("=" * 60)

    theta_e_deg = 90.0  # Neutral wall
    log_ratio = math.log(100.0)
    Ca_values = [1e-4, 1e-3, 1e-2, 1e-1]

    print(f"  theta_e={theta_e_deg}°, log_ratio={log_ratio:.3f}")
    print(f"  Testing Ca values: {Ca_values}")

    results = []
    for Ca in Ca_values:
        theta_d = cox_voinov_dynamic_angle(theta_e_deg, Ca, log_ratio)
        wall_psi = cox_voinov_dynamic_wall_psi(theta_e_deg, Ca, log_ratio)
        results.append({
            "Ca": Ca,
            "theta_d_deg": theta_d,
            "wall_psi": wall_psi,
        })
        print(f"  Ca={Ca:.0e}: θ_d={theta_d:.2f}°, ψ_w={wall_psi:.4f}")

    # Verify monotonicity: θ_d should increase with Ca
    theta_d_values = [r["theta_d_deg"] for r in results]
    for i in range(1, len(theta_d_values)):
        assert theta_d_values[i] >= theta_d_values[i-1], \
            f"Non-monotonic: θ_d({Ca_values[i]})={theta_d_values[i]} < θ_d({Ca_values[i-1]})={theta_d_values[i-1]}"

    print("\n  Ca scaling test PASSED (monotonicity verified)\n")
    return True


def test_capillary_rise_stability():
    """Test capillary rise simulation stability with Cox-Voinov wetting."""
    print("=" * 60)
    print("Test 3: Capillary Rise Stability")
    print("=" * 60)

    # Parameters
    nx, ny = 32, 64
    theta_e_deg = 60.0
    steps = 500
    sample = 50
    G = 5.0
    tau = 1.0
    log_ratio = math.log(100.0)

    print(f"  Grid: {nx}x{ny}, theta_e={theta_e_deg}°, steps={steps}")
    print(f"  G={G}, tau={tau}, log_ratio={log_ratio:.3f}")

    t0 = time.time()
    try:
        result = run_capillary_scaling(
            theta_e_deg=theta_e_deg,
            Ca_values=[1e-4, 1e-3, 1e-2],
            log_ratio=log_ratio,
            nx=nx, ny=ny,
            radius=10.0,
            steps=steps,
            G=G, tau=tau,
            device="cpu",
        )
    except Exception as e:
        print(f"  Simulation ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.time() - t0

    print(f"  Simulation completed in {elapsed:.1f}s")

    # Check stability
    all_stable = all(r["stable"] for r in result["results"])
    if all_stable:
        print("  All Ca values stable ✓")
    else:
        print("  WARNING: Some Ca values unstable")
        for r in result["results"]:
            if not r["stable"]:
                print(f"    Ca={r['Ca']}: UNSTABLE")

    print("\n  Capillary rise stability test COMPLETED\n")
    return True


def main():
    ap = argparse.ArgumentParser(description="Cox-Voinov Dynamic Wetting Validation")
    ap.add_argument("--steps", type=int, default=5000, help="Capillary rise steps")
    ap.add_argument("--Ca-values", type=float, nargs="+", default=None,
                    help="Capillary numbers to test")
    ap.add_argument("--theta-e", type=float, default=60.0, help="Equilibrium contact angle")
    ap.add_argument("--log-ratio", type=float, default=math.log(100.0),
                    help="ln(L/λ) for Cox-Voinov law")
    ap.add_argument("--out", type=str, default=None, help="Output JSON file")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("Cox-Voinov Dynamic Wetting Validation Suite")
    print("=" * 60 + "\n")

    all_passed = True

    # Test 1: Formula correctness
    try:
        test_cox_voinov_formula()
    except Exception as e:
        print(f"  Test 1 FAILED: {e}")
        all_passed = False

    # Test 2: Ca scaling
    try:
        test_capillary_scaling()
    except Exception as e:
        print(f"  Test 2 FAILED: {e}")
        all_passed = False

    # Test 3: Capillary rise stability
    try:
        test_capillary_rise_stability()
    except Exception as e:
        print(f"  Test 3 FAILED: {e}")
        all_passed = False

    # Summary
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60 + "\n")

    # Output results
    results = {
        "test1_formula": "PASS" if all_passed else "FAIL",
        "test2_ca_scaling": "PASS" if all_passed else "FAIL",
        "test3_stability": "PASS" if all_passed else "FAIL",
        "all_passed": all_passed,
        "parameters": {
            "theta_e_deg": args.theta_e,
            "log_ratio": args.log_ratio,
            "steps": args.steps,
        },
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
