#!/usr/bin/env python3
"""Cumulant collision (Geier 2015) verification test.

Tests the cumulant collision operator for D3Q19 and D3Q27 lattices
against known properties:
1. Mass and momentum conservation
2. Equilibrium fixed point
3. Numerical stability

Usage:
    python3 cumulant_test.py [--d3q27]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Ensure the local src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))


def test_cumulant_d3q19():
    """Test D3Q19 cumulant collision properties."""
    from tensorlbm.collision_d3q19_advanced import collide_cumulant_d3q19
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

    results = {}

    # Test 1: Mass conservation
    shape = (8, 16, 16)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = 0.05 * torch.ones(shape, dtype=torch.float64)
    uy = 0.03 * torch.ones(shape, dtype=torch.float64)
    uz = 0.02 * torch.ones(shape, dtype=torch.float64)

    f = equilibrium3d(rho, ux, uy, uz)
    f += 0.01 * torch.randn_like(f)  # Add perturbation

    rho_before, ux_before, uy_before, uz_before = macroscopic3d(f)
    mass_before = rho_before.sum().item()
    mom_before = (rho_before * ux_before).sum().item()

    f_collided = collide_cumulant_d3q19(f, 0.95, omega_b=1.0, omega_odd=1.0, omega_even=1.0)

    rho_after, ux_after, uy_after, uz_after = macroscopic3d(f_collided)
    mass_after = rho_after.sum().item()
    mom_after = (rho_after * ux_after).sum().item()

    mass_drift = abs(mass_after - mass_before) / mass_before
    mom_drift = abs(mom_after - mom_before) / abs(mom_before) if mom_before != 0 else 0

    results['d3q19_mass_conservation'] = {
        'mass_before': mass_before,
        'mass_after': mass_after,
        'mass_drift_relative': mass_drift,
        'mom_before': mom_before,
        'mom_after': mom_after,
        'mom_drift_relative': mom_drift,
        'passed': mass_drift < 1e-12 and mom_drift < 1e-12,
    }

    # Test 2: Equilibrium fixed point
    f_eq = equilibrium3d(rho, ux, uy, uz)
    f_eq_collided = collide_cumulant_d3q19(f_eq, 0.95, omega_b=1.0, omega_odd=1.0, omega_even=1.0)
    eq_error = (f_eq_collided - f_eq).abs().max().item()

    results['d3q19_equilibrium_fixed_point'] = {
        'max_error': eq_error,
        'passed': eq_error < 1e-12,
    }

    # Test 3: Numerical stability
    f_unstable = equilibrium3d(rho, 0.2 * torch.ones(shape), 0.2 * torch.ones(shape), 0.2 * torch.ones(shape))
    f_unstable += 0.1 * torch.randn_like(f_unstable)
    f_unstable_collided = collide_cumulant_d3q19(f_unstable, 0.51, omega_b=1.0, omega_odd=1.0, omega_even=1.0)
    is_finite = torch.isfinite(f_unstable_collided).all().item()
    has_negative = (f_unstable_collided < 0).any().item()

    results['d3q19_stability'] = {
        'is_finite': is_finite,
        'has_negative': has_negative,
        'passed': is_finite and not has_negative,
    }

    return results


def test_cumulant_d3q27():
    """Test D3Q27 cumulant collision properties."""
    from tensorlbm.cumulant import collide_cumulant_geier_d3q27
    from tensorlbm.d3q27 import equilibrium27, macroscopic27

    results = {}

    # Test 1: Mass conservation
    shape = (8, 16, 16)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = 0.05 * torch.ones(shape, dtype=torch.float64)
    uy = 0.03 * torch.ones(shape, dtype=torch.float64)
    uz = 0.02 * torch.ones(shape, dtype=torch.float64)

    f = equilibrium27(rho, ux, uy, uz)
    f += 0.01 * torch.randn_like(f)  # Add perturbation

    rho_before, ux_before, uy_before, uz_before = macroscopic27(f)
    mass_before = rho_before.sum().item()
    mom_before = (rho_before * ux_before).sum().item()

    f_collided = collide_cumulant_geier_d3q27(f, 0.95, omega_b=1.0, omega_odd=1.0, omega_even=1.0)

    rho_after, ux_after, uy_after, uz_after = macroscopic27(f_collided)
    mass_after = rho_after.sum().item()
    mom_after = (rho_after * ux_after).sum().item()

    mass_drift = abs(mass_after - mass_before) / mass_before
    mom_drift = abs(mom_after - mom_before) / abs(mom_before) if mom_before != 0 else 0

    results['d3q27_mass_conservation'] = {
        'mass_before': mass_before,
        'mass_after': mass_after,
        'mass_drift_relative': mass_drift,
        'mom_before': mom_before,
        'mom_after': mom_after,
        'mom_drift_relative': mom_drift,
        'passed': mass_drift < 1e-12 and mom_drift < 1e-12,
    }

    # Test 2: Equilibrium fixed point
    f_eq = equilibrium27(rho, ux, uy, uz)
    f_eq_collided = collide_cumulant_geier_d3q27(f_eq, 0.95, omega_b=1.0, omega_odd=1.0, omega_even=1.0)
    eq_error = (f_eq_collided - f_eq).abs().max().item()

    results['d3q27_equilibrium_fixed_point'] = {
        'max_error': eq_error,
        'passed': eq_error < 1e-12,
    }

    # Test 3: Numerical stability
    f_unstable = equilibrium27(rho, 0.2 * torch.ones(shape), 0.2 * torch.ones(shape), 0.2 * torch.ones(shape))
    f_unstable += 0.1 * torch.randn_like(f_unstable)
    f_unstable_collided = collide_cumulant_geier_d3q27(f_unstable, 0.51, omega_b=1.0, omega_odd=1.0, omega_even=1.0)
    is_finite = torch.isfinite(f_unstable_collided).all().item()
    has_negative = (f_unstable_collided < 0).any().item()

    results['d3q27_stability'] = {
        'is_finite': is_finite,
        'has_negative': has_negative,
        'passed': is_finite and not has_negative,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d3q27', action='store_true', help='Test D3Q27 instead of D3Q19')
    parser.add_argument('--out', type=Path, default=None, help='Output file')
    args = parser.parse_args()

    if args.d3q27:
        results = test_cumulant_d3q27()
    else:
        results = test_cumulant_d3q19()

    # Print results
    print("=" * 60)
    print("Cumulant Collision Verification Results")
    print("=" * 60)
    for test_name, test_result in results.items():
        status = "PASS" if test_result.get('passed', False) else "FAIL"
        print(f"\n{test_name}: {status}")
        for key, value in test_result.items():
            if key != 'passed':
                print(f"  {key}: {value}")

    # Save results
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {args.out}")

    # Return exit code based on results
    all_passed = all(r.get('passed', False) for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
