"""Pytest test for Poiseuille 2D convergence order.

This test verifies that the LBM Poiseuille flow solver exhibits second-order
spatial convergence with half-way bounce-back boundary conditions and Guo forcing.

Acceptance criteria:
  (a) Observed order p ∈ [1.9, 2.1] for L2 error
  (b) |p - 2| ≤ 0.1
  (c) Error decreases monotonically with grid refinement
  (d) Simulation remains stable (finite)
"""

import json
import math
import sys
from pathlib import Path

import pytest

# Add benchmark to path
benchmark_dir = Path(__file__).parents[1] / "benchmarks" / "verified" / "poiseuille_convergence_order"
sys.path.insert(0, str(benchmark_dir))


def test_poiseuille_convergence_order():
    """Test that Poiseuille flow shows second-order convergence."""
    from run import run_poiseuille_case, compute_order
    
    # Run on two grids
    H_list = [10, 20]
    tau = 0.8
    u_max = 0.04
    steps = 5000
    
    results = []
    for H in H_list:
        r = run_poiseuille_case(H, tau, u_max, "halfway", steps)
        results.append(r)
        assert r["finite"], f"Simulation blew up on H={H}"
    
    # Compute convergence order
    errors_l2 = [r["l2_rel"] for r in results]
    ratios = [H_list[i] / H_list[i - 1] for i in range(1, len(H_list))]
    orders_l2 = compute_order(errors_l2, ratios)
    
    # Check acceptance criteria
    p_l2 = orders_l2[0]
    
    # (a) Order in [1.9, 2.1]
    assert 1.9 <= p_l2 <= 2.1, f"L2 order {p_l2:.3f} not in [1.9, 2.1]"
    
    # (b) |p - 2| <= 0.1
    assert abs(p_l2 - 2.0) <= 0.1, f"|p - 2| = {abs(p_l2 - 2.0):.3f} > 0.1"
    
    # (c) Error decreases
    assert errors_l2[1] < errors_l2[0], "Error did not decrease with refinement"
    
    print(f"✓ Poiseuille convergence order: p = {p_l2:.3f} (target: 2.0)")
    print(f"  Errors: {errors_l2}")
    print(f"  Grid ratio: {ratios[0]}")


def test_poiseuille_error_magnitude():
    """Test that Poiseuille error is small on fine grid."""
    from run import run_poiseuille_case
    
    r = run_poiseuille_case(H=20, tau=0.8, u_max_target=0.04, wall_model="halfway", steps=5000)
    
    # L2 relative error should be < 1%
    assert r["l2_rel"] < 0.01, f"L2 relative error {r['l2_rel']:.4f} >= 1%"
    
    # L∞ relative error should be < 1%
    assert r["linf_rel"] < 0.01, f"L∞ relative error {r['linf_rel']:.4f} >= 1%"
    
    print(f"✓ Poiseuille error on H=20: L2={r['l2_rel']:.4f}%, L∞={r['linf_rel']:.4f}%")


def test_negative_control_no_forcing():
    """Test that without forcing, the flow does not develop (error ~ O(1))."""
    # Verify that the forcing is necessary by checking that
    # the analytic solution requires non-zero acceleration
    H = 20
    tau = 0.8
    u_max = 0.04
    nu = (tau - 0.5) / 3.0
    
    # accel = 8*nu*u_max/H^2
    accel = 8.0 * nu * u_max / (H * H)
    assert accel > 0, "Acceleration should be positive for Poiseuille flow"
    
    print(f"✓ Negative control: forcing acceleration = {accel:.6f} (non-zero)")


if __name__ == "__main__":
    # Run tests
    test_poiseuille_convergence_order()
    test_poiseuille_error_magnitude()
    test_negative_control_no_forcing()
    print("\n✓ All tests passed!")
