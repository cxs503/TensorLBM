#!/usr/bin/env python3
"""SUBOFF high-Re turbulence model comparison worker.

Tests Smagorinsky, WALE, Vreman, and Dynamic Smagorinsky LES models on the
SUBOFF bare hull at Re=1000, 1e4, 1e5, 2e6.

Geometry:  SUBOFF bare hull, L=80, nx=200, ny=80, nz=80, u_in=0.06
dpS:       0.5 * u_in^2 * pi * D * L  (wetted area, D=9.334, L=80)
Steps:     5000, averaging window win=500

Reference Cf:
  Re=1000:  Blasius  Cf = 1.328/sqrt(Re) = 0.042
  Re=1e4:   ITTC-1957 Cf = 0.075/(log10(Re)-2)^2 = 0.01875
  Re=1e5:   ITTC-1957 Cf = 0.075/(log10(Re)-2)^2 = 0.00833
  Re=2e6:   ITTC-1957 Ct = 0.075/(log10(Re)-2)^2 = 0.00405

Usage:
  python suboff_turbcomp_worker.py <Re_case> <device_id> <output_dir>
  Re_case: re1000 | re1e4 | re1e5 | re2e6
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.turbulence import (
    collide_smagorinsky_mrt3d,
    collide_wale_mrt3d,
    collide_vreman_mrt3d,
    collide_dynamic_smagorinsky_bgk3d,
)
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm import build_suboff_mask


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES = {
    "re1000": {
        "Re": 1000,
        "tau": 0.5144,
        "ref_cf": 1.328 / math.sqrt(1000),   # Blasius laminar ≈ 0.042
        "ref_name": "Blasius Cf=1.328/sqrt(Re)",
        "models": ["mrt", "smag"],
    },
    "re1e4": {
        "Re": 10000,
        "tau": 0.50144,
        "ref_cf": 0.075 / (math.log10(10000) - 2.0) ** 2,  # ITTC ≈ 0.01875
        "ref_name": "ITTC-1957 Cf=0.075/(log10(Re)-2)^2",
        "models": ["smag", "wale", "vreman"],
    },
    "re1e5": {
        "Re": 100000,
        "tau": 0.500144,
        "ref_cf": 0.075 / (math.log10(100000) - 2.0) ** 2,  # ITTC ≈ 0.00833
        "ref_name": "ITTC-1957 Cf=0.075/(log10(Re)-2)^2",
        "models": ["smag", "wale", "vreman", "dyn_smag"],
    },
    "re2e6": {
        "Re": 2000000,
        "tau": 0.5000072,
        "ref_cf": 0.075 / (math.log10(2000000) - 2.0) ** 2,  # ITTC ≈ 0.00405
        "ref_name": "ITTC-1957 Ct=0.075/(log10(Re)-2)^2",
        "models": ["smag_wf"],
    },
}

# Common simulation parameters
L = 80.0
NX, NY, NZ = 200, 80, 80
U_IN = 0.06
N_STEPS = 5000
WIN = 500
D_WETTED = 9.334  # SUBOFF diameter (lattice units), L/D=8.57


def run_one(device_id, Re, model, tau, ref_cf, ref_name, output_path):
    """Run a single (Re, model) simulation and return results dict."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[sdaa:{device_id} Re={Re} {model}]"

    # Wetted-area dynamic pressure
    dpS = 0.5 * U_IN ** 2 * math.pi * D_WETTED * L
    nu = U_IN * L / Re

    print(
        f"{tag} L={L} grid={NX}x{NY}x{NZ} u_in={U_IN} nu={nu:.6e} "
        f"tau={tau:.6f} dpS={dpS:.6e} ref_cf={ref_cf:.6f} ({ref_name})",
        flush=True,
    )

    t0 = time.time()

    # Build SUBOFF bare hull mask
    solid, meta = build_suboff_mask(
        hull_type="bare_hull",
        nx=NX, ny=NY, nz=NZ,
        length=L,
        device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={meta['L_D_ratio']}", flush=True)

    # Near-wall mask and surface mesh (from_gradient — verified module)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, NZ, NY, NX)

    # Initialize flow field
    rho0 = torch.ones((NZ, NY, NX), device=device)
    ux0 = torch.full((NZ, NY, NX), U_IN, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s), initial_mass={im}", flush=True)

    # Determine if wall function is used
    use_wall_fn = model == "smag_wf"

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    diverged = False

    for step in range(1, N_STEPS + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision
        if model == "mrt":
            f = collide_mrt3d(f, tau=tau)
        elif model == "smag" or model == "smag_wf":
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
        elif model == "wale":
            f = collide_wale_mrt3d(f, tau=tau, C_w=0.5)
        elif model == "vreman":
            f = collide_vreman_mrt3d(f, tau=tau, C_V=0.025)
        elif model == "dyn_smag":
            f = collide_dynamic_smagorinsky_bgk3d(f, tau=tau)
        else:
            raise ValueError(f"Unknown model: {model}")

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        if use_wall_fn:
            # Wall function approach (Re=2e6):
            # stream → wall_function (body force + drag) → far_field → mass_corr
            f = stream3d(f)
            f, drag_fric_raw, drag_pres_wf = wall_function_3d(
                f, solid, nu, y_val=0.5, wall_law="log", near_mask=near,
            )
            f = far_field_bc_3d(f, U_IN)
            if step % 200 == 0:
                f = correct_mass3d(f, im)

            # Drag: friction from wall function, pressure from surface mesh
            cd_f = drag_fric_raw / dpS
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
            cd_p = fx_p
        else:
            # Standard approach: bounce-back → stream → far_field → mass_corr → drag
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, U_IN)
            if step % 200 == 0:
                f = correct_mass3d(f, im)

            # Drag from surface mesh integration
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p = fx_p
            cd_f = fx_f

        cd_tot = cd_p + cd_f
        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)

        # Divergence check
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        # Progress report
        if step % 500 == 0:
            n_avg = min(WIN, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} Cf_ref={ref_cf:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last win steps or all if fewer)
    n_final = min(WIN, len(cd_tot_hist))
    if n_final == 0:
        cd_p_final = cd_f_final = cd_tot_final = float("nan")
    else:
        cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
        cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
        cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final

    err_pct = (
        abs(cd_tot_final - ref_cf) / ref_cf * 100
        if ref_cf > 0 and not diverged
        else float("inf")
    )

    result = {
        "case": tag.strip("[]"),
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "model": model,
        "L": L,
        "grid": f"{NX}x{NY}x{NZ}",
        "u_in": U_IN,
        "nu": nu,
        "tau": tau,
        "n_steps": N_STEPS,
        "n_steps_run": len(cd_tot_hist),
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "dpS_type": "wetted_area_piDL",
        "D_wetted": D_WETTED,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cf_ref": ref_cf,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()) if not diverged else False,
        "diverged": diverged,
        "elapsed_s": elapsed,
    }

    status = "DIVERGED" if diverged else "OK"
    print(
        f"{tag} DONE [{status}] Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cf_ref={ref_cf:.6f} "
        f"err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python suboff_turbcomp_worker.py <Re_case> <device_id> <output_dir>")
        print("  Re_case: re1000 | re1e4 | re1e5 | re2e6")
        sys.exit(1)

    re_case = sys.argv[1]
    device_id = int(sys.argv[2])
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    if re_case not in CASES:
        print(f"Unknown Re_case: {re_case}. Choose from {list(CASES.keys())}")
        sys.exit(1)

    case = CASES[re_case]
    Re = case["Re"]
    tau = case["tau"]
    ref_cf = case["ref_cf"]
    ref_name = case["ref_name"]
    models = case["models"]

    all_results = []
    for model in models:
        out_file = output_dir / f"suboff_{re_case}_{model}_sdaa{device_id}.json"
        print(f"\n{'='*70}")
        print(f"Running {re_case} model={model} on sdaa:{device_id}")
        print(f"{'='*70}\n")
        result = run_one(device_id, Re, model, tau, ref_cf, ref_name, str(out_file))
        all_results.append(result)

    # Write combined results
    combined_path = output_dir / f"suboff_{re_case}_summary_sdaa{device_id}.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nCombined results written to {combined_path}", flush=True)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {re_case} (Re={Re}, tau={tau}, ref Cf={ref_cf:.6f})")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Cd_p':>10} {'Cd_f':>10} {'Cd_tot':>10} "
          f"{'Cf_ref':>10} {'Error%':>8} {'Stable':>6}")
    print("-" * 70)
    for r in all_results:
        stable = "YES" if r["finite"] else "NO"
        print(
            f"{r['model']:<12} {r['Cd_pressure']:>10.6f} {r['Cd_friction']:>10.6f} "
            f"{r['Cd_total']:>10.6f} {r['Cf_ref']:>10.6f} {r['error_pct']:>8.1f} {stable:>6}"
        )


if __name__ == "__main__":
    main()
