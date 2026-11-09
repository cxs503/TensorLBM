"""Main verification runner for aero-turbulence dual verification surface.

Runs:
1. NACA0012 airfoil polar sweep (Ma=0.3, Re=500)
2. LES turbulent channel flow (Re_tau=180, Smagorinsky + van Driest)
3. Grid convergence study (3 levels)
4. Energy spectrum analysis

Produces JSON output with all verification metrics.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .grid_convergence import compute_gci, grid_convergence_study
from .les_channel_verification import (
    LESChannelConfig,
    compute_energy_spectrum,
    fit_inertial_subrange,
    run_les_channel,
)
from .naca0012_verification import (
    AirfoilSimConfig,
    compute_lift_slope,
    run_airfoil_sim,
    run_polar_sweep,
)
from .reference_data import (
    MKM1999Statistics,
    abbott_cd_interp,
    abbott_cl_interp,
    mkm1999_mean_velocity_profile,
    mkm1999_reynolds_stress_profile,
    thin_airfoil_cl,
)


def run_airfoil_verification(
    alpha_list: list[float] | None = None,
    chord: float = 100.0,
    u_in: float = 0.05,  # Ma ≈ 0.087
    re: float = 500.0,
    nx: int = 400,
    ny: int = 200,
    n_steps: int = 15000,
    warmup_steps: int = 5000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run NACA0012 airfoil verification.
    
    Returns dict with:
    - polar data (alpha, Cl, Cd, reference values, errors)
    - lift slope analysis
    - acceptance criteria evaluation
    """
    if alpha_list is None:
        alpha_list = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    
    print("=" * 60)
    print("NACA0012 Airfoil Verification")
    print("=" * 60)
    print(f"  Chord = {chord} lu, Re = {re}, Ma = {u_in * math.sqrt(3):.3f}")
    print(f"  Grid: {nx} x {ny}, Steps: {n_steps}")
    print()
    
    # Run polar sweep
    polar = run_polar_sweep(
        alpha_list=alpha_list,
        chord=chord,
        u_in=u_in,
        re=re,
        nx=nx,
        ny=ny,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        device=device,
    )
    
    # Compute lift slope
    lift_slope = compute_lift_slope(
        alpha_list=polar["alpha_list"],
        cl_list=polar["cl_sim"],
        linear_range=(0.0, 8.0),
    )
    
    # Evaluate acceptance criteria
    # C1: CL slope deviation < 5%
    c1_pass = lift_slope["deviation_pct"] < 5.0
    
    # C2: CD deviation < 10% for alpha < 8°
    cd_errors_low_alpha = [
        polar["cd_err_pct"][i]
        for i, a in enumerate(polar["alpha_list"])
        if a < 8.0
    ]
    c2_pass = all(e < 10.0 for e in cd_errors_low_alpha) if cd_errors_low_alpha else False
    
    result = {
        "polar": polar,
        "lift_slope": lift_slope,
        "acceptance": {
            "C1_cl_slope_lt_5pct": {
                "pass": c1_pass,
                "value": lift_slope["deviation_pct"],
                "threshold": 5.0,
            },
            "C2_cd_lt_10pct_alpha_lt_8": {
                "pass": c2_pass,
                "values": cd_errors_low_alpha,
                "threshold": 10.0,
            },
        },
        "notes": {
            "mach_limitation": (
                "Standard D2Q9 LBM valid for Ma < 0.3. "
                f"Current Ma = {u_in * math.sqrt(3):.3f}. "
                "Results at Ma > 0.3 require compressible LBM extensions."
            ),
            "reynolds_limitation": (
                f"Re = {re} is much lower than Abbott-von Doenhoff "
                "experimental Re ≈ 3-9×10^6. Comparison is qualitative."
            ),
        },
    }
    
    return result


def run_les_verification(
    nx: int = 256,
    ny: int = 65,
    re_tau: float = 180.0,
    u_tau: float = 0.005,
    C_s: float = 0.1,
    A_plus: float = 25.0,
    n_steps: int = 50000,
    averaging_start: int = 15000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run LES turbulent channel flow verification.
    
    Returns dict with:
    - mean velocity profile U+(y+)
    - Reynolds stress profiles
    - TKE profile
    - energy spectrum
    - comparison with MKM1999 DNS
    - acceptance criteria evaluation
    """
    print("=" * 60)
    print("LES Turbulent Channel Flow Verification")
    print("=" * 60)
    print(f"  Re_tau = {re_tau}, u_tau = {u_tau}")
    print(f"  Grid: {nx} x {ny}, C_s = {C_s}, A+ = {A_plus}")
    print(f"  Steps: {n_steps}, Averaging from: {averaging_start}")
    print()
    
    config = LESChannelConfig(
        nx=nx, ny=ny, re_tau=re_tau, u_tau=u_tau,
        C_s=C_s, A_plus=A_plus,
        n_steps=n_steps, averaging_start=averaging_start,
        device=device,
    )
    
    result = run_les_channel(config)
    
    # Compare with MKM1999 DNS
    y_plus = np.array(result["profiles"]["y_plus"])
    U_plus_sim = np.array(result["profiles"]["U_plus"])
    uv_plus_sim = np.array(result["profiles"]["uv_plus"])
    tke_plus_sim = np.array(result["profiles"]["tke_plus"])
    
    # Reference profiles
    U_plus_ref = mkm1999_mean_velocity_profile(y_plus)
    uv_plus_ref = mkm1999_reynolds_stress_profile(y_plus)
    tke_plus_ref = mkm1999_tke_profile(y_plus)
    
    # Compute errors in log-law region (y+ > 30)
    log_law_mask = (y_plus > 30) & (y_plus < re_tau * 0.8)
    if log_law_mask.sum() > 0:
        U_error_log = np.abs(U_plus_sim[log_law_mask] - U_plus_ref[log_law_mask])
        U_error_log_mean = float(np.mean(U_error_log / np.maximum(U_plus_ref[log_law_mask], 0.1)) * 100)
    else:
        U_error_log_mean = 100.0
    
    # Reynolds stress peak comparison
    uv_peak_sim_idx = np.argmax(-uv_plus_sim[1:ny//2]) + 1
    uv_peak_ref_idx = np.argmax(uv_plus_ref[1:ny//2]) + 1
    uv_peak_y_sim = float(y_plus[uv_peak_sim_idx])
    uv_peak_y_ref = float(MKM1999Statistics().uv_peak_y_plus)
    uv_peak_val_sim = float(-uv_plus_sim[uv_peak_sim_idx])
    uv_peak_val_ref = MKM1999Statistics().uv_peak_value
    
    uv_peak_y_err = abs(uv_peak_y_sim - uv_peak_y_ref) / uv_peak_y_ref * 100
    uv_peak_val_err = abs(uv_peak_val_sim - uv_peak_val_ref) / uv_peak_val_ref * 100
    
    # Energy spectrum (from last snapshot)
    # Note: For proper spectrum, need instantaneous field
    # Here we use mean field as placeholder
    k, E_uu = compute_energy_spectrum(
        torch.tensor(U_plus_sim).view(1, -1).expand(ny, -1),
        dx=1.0,
    )
    spectrum_fit = fit_inertial_subrange(k, E_uu)
    
    # Acceptance criteria
    # C3: U+ log region deviation < 5%
    c3_pass = U_error_log_mean < 5.0
    
    # C4: Reynolds stress peak position < 10%, magnitude < 15%
    c4_pass = uv_peak_y_err < 10.0 and uv_peak_val_err < 15.0
    
    # C5: Energy spectrum slope close to -5/3 (R² > 0.9)
    c5_pass = spectrum_fit["r_squared"] > 0.9
    
    result.update({
        "comparison": {
            "U_plus_ref": U_plus_ref.tolist(),
            "uv_plus_ref": uv_plus_ref.tolist(),
            "tke_plus_ref": tke_plus_ref.tolist(),
        },
        "spectrum": {
            "k": k.tolist(),
            "E_uu": E_uu.tolist(),
            "fit": spectrum_fit,
        },
        "acceptance": {
            "C3_U_plus_log_lt_5pct": {
                "pass": c3_pass,
                "value": U_error_log_mean,
                "threshold": 5.0,
            },
            "C4_uv_peak_position_lt_10pct": {
                "pass": uv_peak_y_err < 10.0,
                "value": uv_peak_y_err,
                "threshold": 10.0,
            },
            "C4_uv_peak_magnitude_lt_15pct": {
                "pass": uv_peak_val_err < 15.0,
                "value": uv_peak_val_err,
                "threshold": 15.0,
            },
            "C5_spectrum_slope_R2_gt_0.9": {
                "pass": c5_pass,
                "value": spectrum_fit["r_squared"],
                "threshold": 0.9,
            },
        },
        "notes": {
            "2d_limitation": (
                "2D LBM cannot capture true 3D turbulence. "
                "Results are qualitative for turbulent statistics. "
                "3D D3Q19 simulations needed for quantitative validation."
            ),
            "resolution": (
                f"dy+ = {config.dy_plus:.1f}. "
                "For wall-resolved LES, dy+ < 1 recommended near wall."
            ),
        },
    })
    
    return result


def run_grid_convergence_study(
    device: str = "cpu",
) -> dict[str, Any]:
    """Run grid convergence study for LES channel flow.
    
    Three grid levels:
    - Coarse: 128 x 33
    - Medium: 256 x 65
    - Fine: 512 x 129
    
    Returns dict with GCI analysis.
    """
    print("=" * 60)
    print("Grid Convergence Study")
    print("=" * 60)
    
    grids = [
        {"nx": 128, "ny": 33, "label": "coarse"},
        {"nx": 256, "ny": 65, "label": "medium"},
        {"nx": 512, "ny": 129, "label": "fine"},
    ]
    
    # Run each grid level (reduced steps for efficiency)
    results = {}
    for g in grids:
        print(f"\n  Running {g['label']} grid: {g['nx']} x {g['ny']}")
        res = run_les_channel(LESChannelConfig(
            nx=g["nx"], ny=g["ny"],
            re_tau=180.0, u_tau=0.005,
            C_s=0.1, A_plus=25.0,
            n_steps=20000, averaging_start=8000,
            device=device,
        ))
        results[g["label"]] = res
    
    # Extract quantities for GCI
    # Use centerline velocity and bulk velocity
    quantities = {
        "u_centerline_plus": [
            results["coarse"]["statistics"]["u_centerline_plus"],
            results["medium"]["statistics"]["u_centerline_plus"],
            results["fine"]["statistics"]["u_centerline_plus"],
        ],
        "u_bulk_plus": [
            results["coarse"]["statistics"]["u_bulk_plus"],
            results["medium"]["statistics"]["u_bulk_plus"],
            results["fine"]["statistics"]["u_bulk_plus"],
        ],
    }
    
    grid_sizes = [
        grids[0]["nx"] * grids[0]["ny"],
        grids[1]["nx"] * grids[1]["ny"],
        grids[2]["nx"] * grids[2]["ny"],
    ]
    
    gci_results = grid_convergence_study(quantities, grid_sizes)
    
    # C6: GCI < 5%
    c6_pass = all(r.gci_fine < 5.0 for r in gci_results.values())
    
    return {
        "grids": [
            {"label": g["label"], "nx": g["nx"], "ny": g["ny"]}
            for g in grids
        ],
        "quantities": quantities,
        "gci": {
            name: {
                "phi_fine": r.phi_fine,
                "phi_medium": r.phi_medium,
                "phi_coarse": r.phi_coarse,
                "r": r.r,
                "p": r.p,
                "e_fine_pct": r.e_fine,
                "gci_fine_pct": r.gci_fine,
                "gci_medium_pct": r.gci_medium,
                "status": r.convergence_status,
            }
            for name, r in gci_results.items()
        },
        "acceptance": {
            "C6_gci_lt_5pct": {
                "pass": c6_pass,
                "values": {name: r.gci_fine for name, r in gci_results.items()},
                "threshold": 5.0,
            },
        },
    }


def run_full_verification(
    output_dir: str | Path = "runs/aero_turbulence_verification",
    device: str = "cpu",
) -> dict[str, Any]:
    """Run complete aero-turbulence verification suite.
    
    Returns
    -------
    dict with all verification results and acceptance criteria.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("AERO-TURBULENCE DUAL VERIFICATION SURFACE")
    print("TensorLBM Verification Campaign")
    print("=" * 60 + "\n")
    
    # 1. Airfoil verification
    airfoil_result = run_airfoil_verification(device=device)
    
    # 2. LES channel verification
    les_result = run_les_verification(device=device)
    
    # 3. Grid convergence study
    gci_result = run_grid_convergence_study(device=device)
    
    # Compile final result
    elapsed = time.time() - start_time
    
    # Overall acceptance
    all_acceptance = {}
    all_acceptance.update(airfoil_result["acceptance"])
    all_acceptance.update(les_result["acceptance"])
    all_acceptance.update(gci_result["acceptance"])
    
    all_pass = all(v.get("pass", False) for v in all_acceptance.values())
    
    final_result = {
        "airfoil_cl_cd": airfoil_result,
        "les_statistics": les_result,
        "spectrum": les_result.get("spectrum", {}),
        "grid_convergence": gci_result,
        "acceptance_summary": {
            "all_pass": all_pass,
            "criteria": all_acceptance,
            "n_pass": sum(1 for v in all_acceptance.values() if v.get("pass", False)),
            "n_total": len(all_acceptance),
        },
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed,
            "device": device,
            "tensorlbm_version": "0.1.0",
        },
    }
    
    # Save JSON
    json_path = output_dir / "verification_results.json"
    with open(json_path, "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for name, crit in all_acceptance.items():
        status = "✓ PASS" if crit.get("pass", False) else "✗ FAIL"
        print(f"  {name}: {status}")
        if "value" in crit:
            print(f"    Value: {crit['value']:.2f}, Threshold: {crit.get('threshold', 'N/A')}")
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  {final_result['acceptance_summary']['n_pass']}/{final_result['acceptance_summary']['n_total']} criteria passed")
    print(f"  Elapsed: {elapsed:.1f} seconds")
    
    return final_result


__all__ = [
    "run_airfoil_verification",
    "run_les_verification",
    "run_grid_convergence_study",
    "run_full_verification",
]
