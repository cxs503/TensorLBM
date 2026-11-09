#!/usr/bin/env python3
"""P40W3-N02 comprehensive cavitation validation campaign.

Uses the isobaric cutoff criterion (p < p_v) to define cavity extent,
which is the classical minimum-pressure criterion for sheet cavitation.
This approach was validated in P41-N02 and is appropriate for the 
homogeneous-mixture solver without strong mass-transfer coupling.

Also attempts to improve pressure resolution with:
1. Better initial conditions (panel-method inspired)
2. Finer grids where stable
3. Longer integration times
"""

import json
import sys
import time
from pathlib import Path

import torch
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
    cavity_metrics,
    reference_data,
    airfoil_surface,
    _interp1,
)

torch.set_num_threads(1)


def isobaric_cavity_metrics(solver, cfg, threshold=0.5):
    """Compute cavity metrics using isobaric cutoff criterion.
    
    The vapor indicator is: alpha_ind = sigmoid((p_v - p) / (0.05 * |p_v|))
    This gives a smooth approximation to the p < p_v region.
    """
    p_v = cfg.p_v
    # Sigmoid-based vapor indicator
    alpha_ind = torch.sigmoid((p_v - solver.p.double()) / (0.05 * abs(p_v)))
    
    return cavity_metrics(
        alpha_ind,
        cfg,
        threshold=threshold,
        geometry=solver.geometry,
    )


def run_case(airfoil: str, sigma: float, cpc: int, n_steps: int, alpha_deg: float = 5.0,
             use_mass_transfer: bool = False):
    """Run a single cavitation case and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {airfoil}, σ={sigma}, cpc={cpc}, α={alpha_deg}°, steps={n_steps}")
    print(f"Mass transfer: {'ON' if use_mass_transfer else 'OFF (isobaric cutoff)'}")
    print(f"{'='*60}")
    
    c_dest = 1.0 if use_mass_transfer else 0.0
    c_prod = 1.0 if use_mass_transfer else 0.0
    
    cfg = CavitationFlowConfig(
        airfoil=airfoil,
        alpha_deg=alpha_deg,
        sigma=sigma,
        cells_per_chord=cpc,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=n_steps,
        cfl=0.2 if cpc <= 30 else 0.15,
        sample_interval=max(10, n_steps // 50),
        c_dest=c_dest,
        c_prod=c_prod,
        pressure_anchor=True,
    )
    
    solver = MixtureCavitationSolver(cfg)
    # Dampen initial velocity perturbations in solid region
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    t0 = time.time()
    try:
        # Run the solver
        for step_i in range(n_steps):
            solver.step()
            if (step_i + 1) % cfg.sample_interval == 0:
                solver.sample()
        
        # Compute final metrics using isobaric cutoff
        m_iso = isobaric_cavity_metrics(solver, cfg, threshold=0.5)
        
        # Also compute with actual alpha if mass transfer was used
        if use_mass_transfer:
            m_alpha = cavity_metrics(
                solver.alpha, cfg, threshold=cfg.cavity_threshold, geometry=solver.geometry
            )
        else:
            m_alpha = m_iso
        
        results = solver.summary()
        results["isobaric_metrics"] = m_iso
        results["alpha_metrics"] = m_alpha
        results["l_cav"] = m_iso["l_cav"]  # Use isobaric as primary
        results["delta_cav"] = m_iso["delta_cav"]
        results["status"] = "success"
        results["wall_time"] = time.time() - t0
        results["p_min"] = float(solver.p.min())
        results["p_max"] = float(solver.p.max())
        results["p_v"] = cfg.p_v
        results["cells_below_pv"] = int((solver.p < cfg.p_v).sum())
        results["alpha_max"] = float(solver.alpha.max())
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        results = {
            "status": "failed",
            "error": str(e),
            "config": {
                "airfoil": airfoil,
                "sigma": sigma,
                "cells_per_chord": cpc,
                "alpha_deg": alpha_deg,
                "n_steps": n_steps,
            },
            "wall_time": time.time() - t0,
        }
    
    return results


def run_validation_matrix():
    """Run the full validation matrix."""
    ref = reference_data()
    
    all_results = {
        "metadata": {
            "mission": "P40W3-N02",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reference_data": ref,
            "method": "isobaric_cutoff",
            "description": "Sheet cavitation predicted by minimum-pressure (isobaric) criterion; "
                          "cloud cavitation by FFT of L_cav(t) time series.",
        },
        "sheet_cavitation": [],
        "cloud_cavitation": [],
        "grid_convergence": [],
    }
    
    # Sheet cavitation: NACA0015 at σ=1.0, 0.8, 0.6
    print("\n" + "="*80)
    print("SHEET CAVITATION: NACA0015")
    print("="*80)
    
    for sigma in [1.0, 0.8, 0.6]:
        result = run_case("NACA0015", sigma, cpc=25, n_steps=500, alpha_deg=5.0)
        all_results["sheet_cavitation"].append(result)
    
    # Cloud cavitation: Clark-Y at σ=0.5, 0.4
    print("\n" + "="*80)
    print("CLOUD CAVITATION: Clark-Y")
    print("="*80)
    
    for sigma in [0.5, 0.4]:
        # Longer integration for shedding detection
        result = run_case("ClarkY", sigma, cpc=25, n_steps=1500, alpha_deg=6.0)
        all_results["cloud_cavitation"].append(result)
    
    # Grid convergence study: NACA0015 at σ=0.8
    print("\n" + "="*80)
    print("GRID CONVERGENCE: NACA0015, σ=0.8")
    print("="*80)
    
    for cpc in [20, 25, 30, 40]:
        result = run_case("NACA0015", 0.8, cpc=cpc, n_steps=500, alpha_deg=5.0)
        all_results["grid_convergence"].append(result)
    
    return all_results


def main():
    print("P40W3-N02 Cavitation Validation Campaign (Isobaric Cutoff)")
    print("=" * 80)
    
    results = run_validation_matrix()
    
    # Save results
    output_dir = Path(__file__).parent.parent / "artifacts" / "p40w3_n02"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    ref = results["metadata"]["reference_data"]
    
    print("\nSheet Cavitation (NACA0015, isobaric cutoff):")
    print("-" * 100)
    print(f"{'σ':<6} {'L_cav/c':<12} {'Brennen':<12} {'Error':<12} {'δ_cav/c':<12} {'Brennen':<12} {'Error':<12} {'p_min':<10} {'p<p_v':<8}")
    print("-" * 100)
    
    for i, r in enumerate(results["sheet_cavitation"]):
        if r["status"] == "success":
            sigma = r["config"]["sigma"]
            l_cav = r["l_cav"]
            delta = r["delta_cav"]
            l_ref = ref["sheet"]["l_cav_exp"][i]
            d_ref = ref["sheet"]["delta_cav_exp"][i]
            l_err = abs(l_cav - l_ref) / l_ref * 100 if l_ref > 0 else 0.0
            d_err = abs(delta - d_ref) / d_ref * 100 if d_ref > 0 else 0.0
            p_min = r.get("p_min", 0.0)
            n_below = r.get("cells_below_pv", 0)
            print(f"{sigma:<6.1f} {l_cav:<12.4f} {l_ref:<12.3f} {l_err:<12.1f}% {delta:<12.4f} {d_ref:<12.3f} {d_err:<12.1f}% {p_min:<10.4f} {n_below:<8}")
        else:
            print(f"{r['config']['sigma']:<6.1f} FAILED: {r.get('error', 'unknown')}")
    
    print("\nCloud Cavitation (Clark-Y):")
    print("-" * 100)
    print(f"{'σ':<6} {'St_cav':<12} {'Franc':<12} {'Error':<12} {'Amplitude':<12} {'Status':<20} {'p_min':<10}")
    print("-" * 100)
    
    for i, r in enumerate(results["cloud_cavitation"]):
        if r["status"] == "success":
            sigma = r["config"]["sigma"]
            st = r.get("strouhal", 0.0)
            st_ref = ref["cloud"]["st_exp"][i]
            amp = r.get("shedding_amplitude", 0.0)
            st_err = abs(st - st_ref) / st_ref * 100 if st_ref > 0 else 0.0
            status = "SHEDDING" if amp > 0.01 else "NO SHEDDING"
            p_min = r.get("p_min", 0.0)
            print(f"{sigma:<6.1f} {st:<12.4f} {st_ref:<12.3f} {st_err:<12.1f}% {amp:<12.4f} {status:<20} {p_min:<10.4f}")
        else:
            print(f"{r['config']['sigma']:<6.1f} FAILED: {r.get('error', 'unknown')}")
    
    print("\nGrid Convergence (NACA0015, σ=0.8):")
    print("-" * 100)
    print(f"{'cpc':<8} {'L_cav/c':<12} {'δ_cav/c':<12} {'Mass Res':<12} {'p_min':<10} {'Status':<12}")
    print("-" * 100)
    
    for r in results["grid_convergence"]:
        if r["status"] == "success":
            cpc = r["config"]["cells_per_chord"]
            l_cav = r["l_cav"]
            delta = r["delta_cav"]
            mass_res = r.get("mass_balance", 0.0)
            p_min = r.get("p_min", 0.0)
            print(f"{cpc:<8} {l_cav:<12.4f} {delta:<12.4f} {mass_res:<12.2e} {p_min:<10.4f} {'OK':<12}")
        else:
            print(f"{r['config']['cells_per_chord']:<8} FAILED: {r.get('error', 'unknown')}")
    
    return results


if __name__ == "__main__":
    main()
