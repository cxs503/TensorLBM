#!/usr/bin/env python3
"""cAC gravity + hydrostatic pressure + mass conservation verification (D3Q19).

Verifies:
1. Hydrostatic pressure profile: p(z) = p0 + rho g z, error < 1%
2. Mass conservation with walls + gravity: relative drift < 0.01% over 6000 steps
3. Phase field integral drift: < 1e-10

Parameters (from PREREG.json):
- D3Q19, FP64, harmonic viscosity
- 128^3 grid (or smaller for quick test)
- Gravity in z direction
- Neumann walls (no wetting bias)

Usage:
    PYTHONPATH=src python3 TensorLBM/scripts/verify_cac_hydrostatic_mass.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Ensure we use the local source
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.cac_lbm3d import (
    CAC3D,
    diagnostics,
    init_phi_slab_3d,
    solid_channel_3d,
    macro,
    rho_of_phi,
)


def run_hydrostatic_test(
    nz: int = 128,
    ny: int = 32,
    nx: int = 32,
    rho_l: float = 1.0,
    rho_g: float = 0.1,
    mu_l: float = 0.1,
    mu_g: float = 0.01,
    sigma: float = 0.01,
    width: float = 4.0,
    mobility: float = 0.05,
    p0: float = 0.01,
    gravity_z: float = -1e-5,
    wall_thickness: int = 2,
    steps: int = 6000,
    sample_every: int = 100,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> dict:
    """Run hydrostatic pressure + mass conservation test.

    Setup:
    - Channel with walls at z=0 and z=nz-1
    - Liquid slab from z=20 to z=nz-20 (leaving gas above and below)
    - Gravity in -z direction
    - Neumann walls (no wetting)
    - Periodic in x, y

    Expected:
    - Hydrostatic pressure: p(z) = p0 + rho * |g| * (z_ref - z)
    - Mass conservation: integral(phi) constant
    - Phase field integral drift < 1e-10
    """
    shape = (nz, ny, nx)
    wall_mask = solid_channel_3d(shape, thickness=wall_thickness, device=device)

    # Liquid slab: from z=20 to z=nz-20
    z_lo = 20.0
    z_hi = float(nz) - 20.0
    phi0 = init_phi_slab_3d(shape, width, z_lo, z_hi, device=device, dtype=dtype)

    sim = CAC3D(
        shape,
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=mu_l,
        mu_g=mu_g,
        sigma=sigma,
        width=width,
        mobility=mobility,
        mu_law="harmonic",
        p0=p0,
        gravity=(0.0, 0.0, gravity_z),
        gravity_mode="rho",
        axes="wpp",  # wall-bounded in z, periodic in x, y
        wall_mask=wall_mask,
        wall_phi_policy="neumann",
        device=device,
        dtype=dtype,
    )
    sim.initialize(phi0)

    # Record initial mass
    m0 = macro(sim, dtype=torch.float64)
    phi0_field = m0["phi"]
    fm = sim.fluid_mask
    phi_integral_0 = float(phi0_field[fm].sum().item())

    history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        sim.step()

        if step % sample_every == 0 or step == 1:
            d = diagnostics(sim)
            d["wall_time_s"] = time.time() - t0

            # Compute hydrostatic pressure profile
            m = macro(sim, dtype=torch.float64)
            p_field = m["p"]
            phi_field = m["phi"]
            rho_field = m["rho"]

            # Average pressure and density in x, y for each z slice
            p_z = p_field[fm].reshape(nz - 2 * wall_thickness, -1).mean(dim=1)
            rho_z = rho_field[fm].reshape(nz - 2 * wall_thickness, -1).mean(dim=1)
            phi_z = phi_field[fm].reshape(nz - 2 * wall_thickness, -1).mean(dim=1)

            # Hydrostatic reference: p(z) = p0 + rho_l * |g| * (z_ref - z)
            # z_ref = interface center (where phi = 0.5)
            z_grid = torch.arange(wall_thickness, nz - wall_thickness, dtype=torch.float64)
            # Find interface center (phi = 0.5)
            phi_diff = (phi_z - 0.5).abs()
            z_interface = float(z_grid[phi_diff.argmin()].item())

            # Use liquid region for hydrostatic check
            liquid_mask = phi_z > 0.9
            if liquid_mask.any():
                z_liquid = z_grid[liquid_mask]
                p_liquid = p_z[liquid_mask]
                rho_liquid = rho_z[liquid_mask].mean().item()

                # Hydrostatic: p(z) = p_ref + rho * |g| * (z_ref - z)
                z_ref = z_liquid[0].item()
                p_ref = p_liquid[0].item()
                p_hydro = p_ref + rho_liquid * abs(gravity_z) * (z_ref - z_liquid)

                # Relative error
                p_max = p_liquid.max().item()
                p_min = p_liquid.min().item()
                p_range = p_max - p_min if p_max != p_min else 1.0
                p_error = float(((p_liquid - p_hydro).abs() / p_range).max().item())
            else:
                p_error = float("nan")
                z_interface = float("nan")

            # Mass conservation
            phi_integral = float(phi_field[fm].sum().item())
            mass_drift = abs(phi_integral - phi_integral_0) / phi_integral_0

            d["p_error_rel"] = p_error
            d["z_interface"] = z_interface
            d["phi_integral"] = phi_integral
            d["phi_integral_0"] = phi_integral_0
            d["mass_drift_rel"] = mass_drift
            d["phase_integral_drift"] = mass_drift  # same as mass_drift for conservative AC

            history.append(d)

    elapsed = time.time() - t0

    # Final diagnostics
    final = history[-1] if history else {}
    final["elapsed_s"] = elapsed
    final["n_steps"] = steps
    final["shape"] = list(shape)

    return {
        "history": history,
        "final": final,
        "phi_integral_0": phi_integral_0,
    }


def analyze_results(results: dict, thresholds: dict) -> dict:
    """Analyze results against pre-registered thresholds."""
    final = results["final"]
    history = results["history"]

    # Hydrostatic pressure error (at late time)
    p_errors = [h["p_error_rel"] for h in history if "p_error_rel" in h and not math.isnan(h["p_error_rel"])]
    p_error_early = p_errors[0] if len(p_errors) > 0 else float("nan")
    p_error_late = p_errors[-1] if len(p_errors) > 0 else float("nan")

    # Mass drift
    mass_drifts = [h["mass_drift_rel"] for h in history if "mass_drift_rel" in h]
    mass_drift_final = mass_drifts[-1] if mass_drifts else float("nan")

    # Phase integral drift
    phase_drifts = [h["phase_integral_drift"] for h in history if "phase_integral_drift" in h]
    phase_drift_final = phase_drifts[-1] if phase_drifts else float("nan")

    verdict = {
        "hydrostatic_error_early_pct": p_error_early * 100 if not math.isnan(p_error_early) else None,
        "hydrostatic_error_late_pct": p_error_late * 100 if not math.isnan(p_error_late) else None,
        "mass_drift_rel_pct": mass_drift_final * 100 if not math.isnan(mass_drift_final) else None,
        "phase_integral_drift": phase_drift_final,
        "thresholds": thresholds,
        "pass_hydrostatic": p_error_late * 100 < thresholds["hydrostatic_pressure_error_pct"] if not math.isnan(p_error_late) else False,
        "pass_mass": mass_drift_final * 100 < thresholds["mass_drift_rel_pct"] if not math.isnan(mass_drift_final) else False,
        "pass_phase": phase_drift_final < thresholds["phase_integral_drift"] if not math.isnan(phase_drift_final) else False,
    }
    verdict["overall_pass"] = verdict["pass_hydrostatic"] and verdict["pass_mass"] and verdict["pass_phase"]

    return verdict


def main():
    parser = argparse.ArgumentParser(description="cAC hydrostatic + mass conservation verification")
    parser.add_argument("--quick", action="store_true", help="Quick test with smaller grid")
    parser.add_argument("--steps", type=int, default=6000, help="Number of steps")
    parser.add_argument("--sample-every", type=int, default=100, help="Sample interval")
    parser.add_argument("--output", type=str, default="docs/m23012/results.json", help="Output file")
    args = parser.parse_args()

    if args.quick:
        nz, ny, nx = 64, 16, 16
        steps = min(args.steps, 1000)
    else:
        nz, ny, nx = 128, 32, 32
        steps = args.steps

    print(f"[m23012] cAC hydrostatic + mass conservation verification")
    print(f"  Grid: {nz}x{ny}x{nx}")
    print(f"  Steps: {steps}")
    print(f"  dtype: float64")
    print(f"  Device: cpu")
    print()

    thresholds = {
        "hydrostatic_pressure_error_pct": 1.0,
        "mass_drift_rel_pct": 0.01,
        "phase_integral_drift": 1e-10,
    }

    results = run_hydrostatic_test(
        nz=nz, ny=ny, nx=nx,
        steps=steps,
        sample_every=args.sample_every,
        dtype=torch.float64,
        device="cpu",
    )

    verdict = analyze_results(results, thresholds)

    print(f"  Hydrostatic error (early): {verdict['hydrostatic_error_early_pct']:.4f}%" if verdict['hydrostatic_error_early_pct'] else "  Hydrostatic error (early): N/A")
    print(f"  Hydrostatic error (late):  {verdict['hydrostatic_error_late_pct']:.4f}%" if verdict['hydrostatic_error_late_pct'] else "  Hydrostatic error (late): N/A")
    print(f"  Mass drift:                {verdict['mass_drift_rel_pct']:.6f}%" if verdict['mass_drift_rel_pct'] is not None else "  Mass drift: N/A")
    print(f"  Phase integral drift:      {verdict['phase_integral_drift']:.2e}" if verdict['phase_integral_drift'] is not None else "  Phase integral drift: N/A")
    print()
    print(f"  Hydrostatic < 1%: {'PASS' if verdict['pass_hydrostatic'] else 'FAIL'}")
    print(f"  Mass drift < 0.01%: {'PASS' if verdict['pass_mass'] else 'FAIL'}")
    print(f"  Phase drift < 1e-10: {'PASS' if verdict['pass_phase'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to lists for JSON serialization
    serializable_results = {
        "final": {k: v for k, v in results["final"].items() if not isinstance(v, torch.Tensor)},
        "history": [{k: v for k, v in h.items() if not isinstance(v, torch.Tensor)} for h in results["history"]],
        "verdict": verdict,
        "thresholds": thresholds,
        "parameters": {
            "nz": nz, "ny": ny, "nx": nx,
            "steps": steps,
            "dtype": "float64",
        },
    }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")

    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
