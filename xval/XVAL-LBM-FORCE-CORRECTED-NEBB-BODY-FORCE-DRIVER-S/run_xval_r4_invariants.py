#!/usr/bin/env python3
"""Round 4 validation extension: verify predeclared invariants + grid convergence.

Lightweight version for high-load environment.

This script extends the R3 validation by checking:
1. Mass conservation invariant: |Δρ|/ρ < 1e-12 at steady state
2. Arm-to-arm consistency: D2Q9 vs D3Q19 at comparable physical setup
3. Grid convergence rate: O(h²) verification with ny=6,10,20
4. EDM vs Guo scheme consistency

Anchor: XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S
Mission: m21157 (generalist self-start)
"""

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.gpu_core_path import (
    GPUCorePathConfig,
    GPUCorePathSolver,
    channel_analytic_velocity,
)
from tensorlbm.nebb_bodyforce import (
    BodyForceChannel,
    ChannelConfig,
    analytic_poiseuille,
    profile_errors,
)


def _pointwise_max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    interior = slice(1, -1)
    denom = np.maximum(np.abs(ref[interior]), 1e-12)
    return float((np.abs(got[interior] - ref[interior]) / denom).max())


def check_mass_conservation(ny: int = 20, nu: float = 0.2, u_max: float = 0.05):
    """Verify |Δρ|/ρ < 1e-12 at steady state (predeclared invariant)."""
    h = ny - 1
    y_c = (h - 1) / 2.0
    gx = 2.0 * nu * u_max / (y_c * (h - y_c))
    cfg = ChannelConfig(
        nx=8, ny=ny, nu=nu, gx=gx, scheme="edm",
        wall="force_corrected_nebb", steps=50_000, steady_tol=1e-13,
    )
    res = BodyForceChannel(cfg).run()
    
    # Mass conservation: density should remain at reference value (ρ₀=1)
    rho_mean = float(res.rho.mean())
    rho_max_dev = float(np.abs(res.rho - 1.0).max())
    rho_rel_dev = rho_max_dev / 1.0  # ρ₀=1
    
    return {
        "ny": ny,
        "rho_mean": rho_mean,
        "rho_max_deviation": rho_max_dev,
        "rho_relative_deviation": rho_rel_dev,
        "pass_1e12": bool(rho_rel_dev < 1e-12),
    }


def check_arm_to_arm_consistency(ny: int = 20, nu: float = 0.2, u_max: float = 0.05):
    """Verify D2Q9 vs D3Q19 consistency at comparable PHYSICAL setup.
    
    Key insight: D2Q9 NEBB has wall at node plane (y₀=0), so H = ny-1.
    D3Q19 half-way BB has wall at half-way (y₀=0.5), so H = ny-2.
    For fair comparison, we must match the PHYSICAL channel height and compare
    at the same normalized y/H locations.
    """
    # Arm A: D2Q9, H_A = ny-1
    H_A = ny - 1
    y_c_A = (H_A - 1) / 2.0
    gx_A = 2.0 * nu * u_max / (y_c_A * (H_A - y_c_A))
    cfg_d2q9 = ChannelConfig(
        nx=8, ny=ny, nu=nu, gx=gx_A, scheme="edm",
        wall="force_corrected_nebb", steps=50_000, steady_tol=1e-13,
    )
    res_d2q9 = BodyForceChannel(cfg_d2q9).run()
    ux_d2q9 = res_d2q9.ux_corrected.mean(axis=0)
    y_d2q9 = np.arange(ny, dtype=np.float64)  # y = 0, 1, ..., ny-1
    y_norm_d2q9 = y_d2q9 / H_A  # normalized: 0, 1/H, 2/H, ..., 1
    
    # Arm B: D3Q19, H_B = ny-2 (half-way BB)
    # To match H_A, we need ny_B such that ny_B - 2 = H_A, i.e., ny_B = H_A + 2
    ny_B = H_A + 2
    H_B = ny_B - 2
    cfg_d3q19 = GPUCorePathConfig(
        nx=8, ny=ny_B, nz=8, tau=nu * 3.0 + 0.5, u_max_target=u_max, init="rest",
        precision="FP64FP64", backend="eager", steps=3000,
        label=f"consistency_ny{ny_B}",
    )
    solver = GPUCorePathSolver(cfg_d3q19)
    solver.run(cfg_d3q19.steps)
    y_d3q19_lat, u_d3q19 = solver.channel_profile()
    y_d3q19_lat = y_d3q19_lat.numpy()
    ux_d3q19 = u_d3q19.numpy()
    # D3Q19: wall at y=0.5 and y=ny_B-1.5, so fluid at y=1..ny_B-2
    # Normalized: y_norm = (y - 0.5) / H_B
    y_norm_d3q19 = (y_d3q19_lat - 0.5) / H_B
    
    # Now compare at same normalized locations
    # Interpolate D3Q19 to D2Q9's normalized y locations
    ux_d3q19_interp = np.interp(y_norm_d2q9, y_norm_d3q19, ux_d3q19)
    
    # Compare interior (exclude walls)
    interior = slice(1, -1)
    ux_d2q9_int = ux_d2q9[interior]
    ux_d3q19_int = ux_d3q19_interp[interior]
    
    # Arm-to-arm: max|D3Q19 - D2Q9| / max|D2Q9|
    arm_to_arm_err = float(np.abs(ux_d3q19_int - ux_d2q9_int).max() / 
                           np.maximum(np.abs(ux_d2q9_int).max(), 1e-12))
    
    # Also compare both to analytic
    ref_d2q9 = analytic_poiseuille(ny, gx_A, nu, 0.0)  # y₀=0 for NEBB
    
    return {
        "ny_d2q9": ny,
        "ny_d3q19": ny_B,
        "H_matched": H_A,
        "arm_to_arm_max_rel_err": arm_to_arm_err,
        "d2q9_u_max": float(ux_d2q9_int.max()),
        "d3q19_u_max": float(ux_d3q19_int.max()),
        "d2q9_pointwise_err": float(_pointwise_max_rel_err(ux_d2q9, ref_d2q9)),
        "note": "D2Q9 (NEBB, y₀=0, H=ny-1) vs D3Q19 (half-way BB, y₀=0.5, H=ny-2) — matched by physical H",
        "pass_1e2": bool(arm_to_arm_err < 1e-2),  # relaxed threshold due to different wall models
    }


def grid_convergence_study(nu: float = 0.2, u_max: float = 0.05):
    """Verify O(h²) convergence rate with ny=6,10,20."""
    resolutions = [6, 10, 20]
    results = []
    
    for ny in resolutions:
        # D2Q9 NEBB: H = ny-1
        h_d2q9 = ny - 1
        y_c_d2q9 = (h_d2q9 - 1) / 2.0
        gx_d2q9 = 2.0 * nu * u_max / (y_c_d2q9 * (h_d2q9 - y_c_d2q9))
        
        cfg = ChannelConfig(
            nx=8, ny=ny, nu=nu, gx=gx_d2q9, scheme="edm",
            wall="force_corrected_nebb", steps=50_000, steady_tol=1e-13,
        )
        res = BodyForceChannel(cfg).run()
        ref = analytic_poiseuille(ny, gx_d2q9, nu, 0.0)
        got = res.ux_corrected.mean(axis=0)
        err_d2q9 = _pointwise_max_rel_err(got, ref)
        
        # D3Q19: H = ny-2
        if ny >= 6:
            h_d3q19 = ny - 2
            cfg_d3q19 = GPUCorePathConfig(
                nx=8, ny=ny, nz=8, tau=nu * 3.0 + 0.5, u_max_target=u_max, init="rest",
                precision="FP64FP64", backend="eager", steps=3000,
                label=f"convergence_ny{ny}",
            )
            solver = GPUCorePathSolver(cfg_d3q19)
            solver.run(cfg_d3q19.steps)
            y, u_got = solver.channel_profile()
            _, u_ref_all = channel_analytic_velocity(cfg_d3q19)
            u_ref = u_ref_all[1:cfg_d3q19.ny-1].to("cpu").numpy()
            err_d3q19 = _pointwise_max_rel_err(u_got.numpy(), u_ref)
        else:
            err_d3q19 = float("nan")
        
        # Grid spacing (characteristic): use D2Q9's H
        dh = 1.0 / h_d2q9
        
        results.append({
            "ny": ny,
            "dh": dh,
            "d2q9_err": err_d2q9,
            "d3q19_err": err_d3q19,
        })
    
    # Compute convergence rates
    for i in range(1, len(results)):
        dh_prev = results[i-1]["dh"]
        dh_curr = results[i]["dh"]
        err_prev_d2q9 = results[i-1]["d2q9_err"]
        err_curr_d2q9 = results[i]["d2q9_err"]
        err_prev_d3q19 = results[i-1]["d3q19_err"]
        err_curr_d3q19 = results[i]["d3q19_err"]
        
        # Rate = log(err_prev/err_curr) / log(dh_prev/dh_curr)
        if err_curr_d2q9 > 1e-15 and err_prev_d2q9 > 1e-15:
            rate_d2q9 = np.log(err_prev_d2q9 / err_curr_d2q9) / np.log(dh_prev / dh_curr)
        else:
            rate_d2q9 = float("inf")  # machine precision
        
        if (not np.isnan(err_curr_d3q19) and not np.isnan(err_prev_d3q19) and
            err_curr_d3q19 > 1e-15 and err_prev_d3q19 > 1e-15):
            rate_d3q19 = np.log(err_prev_d3q19 / err_curr_d3q19) / np.log(dh_prev / dh_curr)
        else:
            rate_d3q19 = float("inf")
        
        results[i]["convergence_rate_d2q9"] = float(rate_d2q9)
        results[i]["convergence_rate_d3q19"] = float(rate_d3q19)
    
    return results


def check_edm_vs_guo(ny: int = 20, nu: float = 0.2, u_max: float = 0.05):
    """Verify EDM and Guo-BGK schemes give consistent results."""
    h = ny - 1
    y_c = (h - 1) / 2.0
    gx = 2.0 * nu * u_max / (y_c * (h - y_c))
    
    results = {}
    for scheme in ["edm", "guo"]:
        cfg = ChannelConfig(
            nx=8, ny=ny, nu=nu, gx=gx, scheme=scheme,
            wall="force_corrected_nebb", steps=50_000, steady_tol=1e-13,
        )
        res = BodyForceChannel(cfg).run()
        ref = analytic_poiseuille(ny, gx, nu, 0.0)
        got = res.ux_corrected.mean(axis=0)
        err = _pointwise_max_rel_err(got, ref)
        results[scheme] = {
            "pointwise_err": err,
            "u_max": float(got.max()),
            "profile": got.tolist(),
        }
    
    # Compare schemes
    edm_profile = np.array(results["edm"]["profile"])
    guo_profile = np.array(results["guo"]["profile"])
    scheme_diff = float(np.abs(edm_profile - guo_profile).max())
    
    return {
        "ny": ny,
        "edm_err": results["edm"]["pointwise_err"],
        "guo_err": results["guo"]["pointwise_err"],
        "max_abs_diff": scheme_diff,
        "pass_consistent": bool(scheme_diff < 1e-12),
    }


def main():
    out_path = Path("/tmp/xval_r4_invariants.json")
    
    results = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "round": 4,
        "mission": "m21157",
        "purpose": "Verify predeclared invariants + grid convergence study",
    }
    
    print("=" * 70)
    print("Round 4 Validation Extension (Lightweight)")
    print("=" * 70)
    
    # 1. Mass conservation
    print("\n[1/4] Checking mass conservation invariant...")
    mass_results = []
    for ny in [10, 20]:
        m = check_mass_conservation(ny)
        mass_results.append(m)
        status = "✓" if m["pass_1e12"] else "✗"
        print(f"  ny={ny:2d}: |Δρ|/ρ = {m['rho_relative_deviation']:.2e} {status}")
    results["mass_conservation"] = mass_results
    results["mass_conservation_pass"] = all(m["pass_1e12"] for m in mass_results)
    
    # 2. Arm-to-arm consistency
    print("\n[2/4] Checking arm-to-arm consistency (D2Q9 vs D3Q19)...")
    consistency = check_arm_to_arm_consistency(ny=20)
    status = "✓" if consistency["pass_1e2"] else "✗"
    print(f"  Arm-to-arm error: {consistency['arm_to_arm_max_rel_err']:.2e} {status}")
    print(f"  Note: {consistency['note']}")
    results["arm_to_arm_consistency"] = consistency
    
    # 3. Grid convergence
    print("\n[3/4] Grid convergence study (O(h²) verification)...")
    convergence = grid_convergence_study()
    print("  ny   | dh     | D2Q9 err    | rate  | D3Q19 err   | rate")
    print("  -----|--------|-------------|-------|-------------|------")
    for r in convergence:
        rate_d2q9 = f"{r.get('convergence_rate_d2q9', float('nan')):.2f}"
        rate_d3q19 = f"{r.get('convergence_rate_d3q19', float('nan')):.2f}"
        if r.get('convergence_rate_d2q9') == float('inf'):
            rate_d2q9 = "∞ (mach)"
        if r.get('convergence_rate_d3q19') == float('inf'):
            rate_d3q19 = "∞ (mach)"
        if np.isnan(r.get('convergence_rate_d3q19', float('nan'))):
            rate_d3q19 = "N/A"
        print(f"  {r['ny']:3d}  | {r['dh']:.4f} | {r['d2q9_err']:.2e} | {rate_d2q9:>5s} | {r['d3q19_err']:.2e} | {rate_d3q19:>5s}")
    results["grid_convergence"] = convergence
    
    # Check if D3Q19 shows O(h²) rate (D2Q9 is at machine precision)
    d3q19_rates = [r.get('convergence_rate_d3q19', 0) for r in convergence[1:] 
                   if r.get('convergence_rate_d3q19') not in [float('inf'), None] and 
                   not np.isnan(r.get('convergence_rate_d3q19', float('nan')))]
    avg_rate_d3q19 = np.mean(d3q19_rates) if d3q19_rates else 0
    results["grid_convergence_pass"] = bool(1.8 <= avg_rate_d3q19 <= 2.2)
    print(f"  Average D3Q19 convergence rate: {avg_rate_d3q19:.2f} (expected ~2.0)")
    
    # 4. EDM vs Guo consistency
    print("\n[4/4] Checking EDM vs Guo scheme consistency...")
    scheme_check = check_edm_vs_guo(ny=20)
    status = "✓" if scheme_check["pass_consistent"] else "✗"
    print(f"  EDM err: {scheme_check['edm_err']:.2e}")
    print(f"  Guo err: {scheme_check['guo_err']:.2e}")
    print(f"  Max |EDM - Guo|: {scheme_check['max_abs_diff']:.2e} {status}")
    results["edm_vs_guo"] = scheme_check
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    results["summary"] = {
        "mass_conservation_pass": results["mass_conservation_pass"],
        "arm_to_arm_consistency_pass": consistency["pass_1e2"],
        "grid_convergence_pass": results["grid_convergence_pass"],
        "edm_guo_consistency_pass": scheme_check["pass_consistent"],
        "all_invariants_pass": all([
            results["mass_conservation_pass"],
            consistency["pass_1e2"],
            results["grid_convergence_pass"],
            scheme_check["pass_consistent"],
        ]),
    }
    
    for key, val in results["summary"].items():
        status = "✓" if val else "✗"
        print(f"  {key}: {status}")
    
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    
    return 0 if results["summary"]["all_invariants_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
