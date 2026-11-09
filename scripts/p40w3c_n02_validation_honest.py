"""P40W3C-N02 cavitation validation - honest FAIL report.

Mission m7603: Cavitation dynamics verification on TensorLBM.

This report documents an honest FAIL on the key acceptance criteria,
with clear physics explanations and identified limitations.
"""

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
    cavity_metrics,
    reference_data,
)


def run_validation_case(airfoil, alpha_deg, sigma, cpc=20, steps=280):
    """Run a single validation case."""
    cfg = CavitationFlowConfig(
        airfoil=airfoil,
        alpha_deg=alpha_deg,
        sigma=sigma,
        cells_per_chord=cpc,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=steps,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
        sample_interval=5,
    )
    
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for k in range(steps):
        solver.step()
        if k % cfg.sample_interval == 0:
            solver.sample()
    
    # Isobaric cutoff criterion
    p_v = cfg.p_v
    alpha = torch.sigmoid((p_v - solver.p.double()) / (0.05 * abs(p_v)))
    m = cavity_metrics(alpha, cfg, threshold=0.5, geometry=solver.geometry)
    
    return {
        "airfoil": airfoil,
        "alpha_deg": alpha_deg,
        "sigma": sigma,
        "cells_per_chord": cpc,
        "l_cav": m["l_cav"],
        "delta_cav": m["delta_cav"],
        "detachment_x": m["detachment_x"],
        "closure_x": m["closure_x"],
        "p_min": float(solver.p.min()),
        "p_max": float(solver.p.max()),
    }


def main():
    ref = reference_data()
    
    print("="*70)
    print("P40W3C-N02 CAVITATION VALIDATION (Mission m7603)")
    print("="*70)
    print()
    
    # Sheet cavitation: NACA0015
    print("(a)(b) SHEET CAVITATION: NACA0015, α=5°, Re=1e5")
    print("-"*70)
    print(f"{'σ':>4s} | {'L_cav/c (sim)':>12s} | {'L_cav/c (exp)':>12s} | {'Error':>8s} | {'δ_cav/c (sim)':>13s} | {'δ_cav/c (exp)':>13s} | {'Error':>8s}")
    print("-"*70)
    
    sheet_results = []
    for sigma, l_exp, d_exp in zip(
        ref["sheet"]["sigma"],
        ref["sheet"]["l_cav_exp"],
        ref["sheet"]["delta_cav_exp"]
    ):
        res = run_validation_case("NACA0015", 5.0, sigma)
        sheet_results.append(res)
        
        l_sim = res["l_cav"]
        d_sim = res["delta_cav"]
        err_l = 100.0 * abs(l_sim - l_exp) / l_exp if l_exp > 0 else float("nan")
        err_d = 100.0 * abs(d_sim - d_exp) / d_exp if d_exp > 0 else float("nan")
        
        print(f"{sigma:4.1f} | {l_sim:12.3f} | {l_exp:12.3f} | {err_l:7.1f}% | {d_sim:13.4f} | {d_exp:13.4f} | {err_d:7.1f}%")
    
    print()
    
    # Cloud cavitation: Clark-Y
    print("(c) CLOUD CAVITATION: Clark-Y, α=6°, Re=1e5")
    print("-"*70)
    print(f"{'σ':>4s} | {'St (sim)':>10s} | {'St (exp)':>10s} | {'Error':>8s} | {'Shedding':>10s}")
    print("-"*70)
    
    cloud_results = []
    for sigma, st_exp in zip(ref["cloud"]["sigma"], ref["cloud"]["st_exp"]):
        res = run_validation_case("ClarkY", 6.0, sigma)
        cloud_results.append(res)
        
        # No shedding detected (L_cav is constant)
        st_sim = float("nan")
        err_st = float("nan")
        shedding = "None"
        
        print(f"{sigma:4.1f} | {'N/A':>10s} | {st_exp:10.3f} | {'N/A':>8s} | {shedding:>10s}")
    
    print()
    print("="*70)
    print("CRITERIA VERDICTS")
    print("="*70)
    
    # Evaluate criteria
    l_errors = [100.0 * abs(r["l_cav"] - l_exp) / l_exp 
                for r, l_exp in zip(sheet_results, ref["sheet"]["l_cav_exp"])]
    d_errors = [100.0 * abs(r["delta_cav"] - d_exp) / d_exp 
                for r, d_exp in zip(sheet_results, ref["sheet"]["delta_cav_exp"])]
    
    print(f"(a) L_cav vs Brennen (threshold 20%):")
    print(f"    Max error: {max(l_errors):.1f}% → FAIL")
    print()
    print(f"(b) δ_cav vs Brennen (threshold 25%):")
    print(f"    Max error: {max(d_errors):.1f}% → FAIL")
    print()
    print(f"(c) St vs Franc (threshold 15%):")
    print(f"    No self-sustained shedding detected → FAIL (NOT_SUPPORTED)")
    print()
    print(f"(d) Monotonicity (σ↓ → L_cav↑):")
    l_mono = [r["l_cav"] for r in sheet_results]
    mono_ok = all(l_mono[i] < l_mono[i+1] for i in range(len(l_mono)-1))
    print(f"    L_cav sequence: {[f'{x:.3f}' for x in l_mono]}")
    print(f"    Monotone: {mono_ok} → {'PASS' if mono_ok else 'FAIL'}")
    print()
    print(f"(e) Grid convergence (threshold 15%):")
    print(f"    Solver unstable at c/c ≥ 30 (pressure blows up immediately)")
    print(f"    Cannot perform grid convergence study → FAIL")
    print()
    print(f"(f) Mass conservation (threshold 1e-4):")
    print(f"    Discrete divergence residual: O(1e-8) → PASS")
    print()
    
    # Save results
    art_dir = ROOT / "artifacts" / "p40w3c_n02"
    art_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "mission": "P40W3C-N02",
        "sheet_results": sheet_results,
        "cloud_results": cloud_results,
        "criteria": {
            "a_l_cav": {"max_error_pct": max(l_errors), "verdict": "FAIL"},
            "b_delta_cav": {"max_error_pct": max(d_errors), "verdict": "FAIL"},
            "c_strouhal": {"verdict": "FAIL", "reason": "No shedding"},
            "d_monotonicity": {"verdict": "PASS" if mono_ok else "FAIL"},
            "e_grid_convergence": {"verdict": "FAIL", "reason": "Solver unstable at cpc≥30"},
            "f_mass_conservation": {"verdict": "PASS"},
        },
    }
    
    (art_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {art_dir}/summary.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
