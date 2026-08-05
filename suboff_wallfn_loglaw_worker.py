#!/usr/bin/env python3
"""SUBOFF wall function: log-law fix + negative Cd_p fix.

FIX 1: Use log law (not gradient law) for high-Re wall function.
  - wall_law='log' for Re>1000
  - y_val=1.0 (first off-wall cell in log region)
  - u_tau from log law: u = (u_tau/κ) * ln(y+) + B, κ=0.41, B=5.0

FIX 2: Fix negative Cd_p by using far-field p_0 (not near-wall).
  - Background pressure subtraction (Bug 22) over-corrects with WF
  - p_0 = average pressure at far-field (fluid cells NOT near-wall)

TEST 1 (SDAA:12): SUBOFF Re=1e5 with log law, y_val=1.0
  - Reference: Cf=0.00833 (ITTC-1957)
  - Previous: 87.8% (gradient law, Cd_p=-0.001440)
  - Target: <25%

TEST 2 (SDAA:13): SUBOFF Re=2e6 with log law + p_0 fix
  - Reference: Ct=0.00405 (ITTC-1957)
  - Previous: 52.5% (Cd_p=-0.001220, negative!)
  - Target: <30%

TEST 3 (SDAA:14): SUBOFF Re=1e4 with log law, y_val=0.5
  - Reference: Cf=0.01875 (ITTC-1957)
  - Previous: 43.2% (RANS), 20.6% (Smag)
  - Target: <20%

TEST 4 (SDAA:15): Compare p_0 sources (near-wall vs far-field vs domain-avg)
  - Runs Re=1e5 with log law, computes Cd_p with all 3 p_0 methods

Geometry:  SUBOFF bare hull, L=80, nx=200, ny=80, nz=80, u_in=0.06
dpS:       0.5 * u_in^2 * pi * D * L  (wetted area, D=2*R_max)
Normals:   from_suboff (analytical axisymmetric body-of-revolution normal)

Usage:
  python suboff_wallfn_loglaw_worker.py <test> <device_id> <output_path>
  test: t1_log | t2_log_p0fix | t3_log_re1e4 | t4_p0_compare
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
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


# --------------------------------------------------------------------------- #
# Main simulation runner
# --------------------------------------------------------------------------- #

def run(test, device_id, output_path=None):
    """Run a single test configuration."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Common parameters ----
    L = 80.0
    nx, ny, nz = 200, 80, 80
    u_in = 0.06
    cs_smag = 0.05
    n_steps = 5000
    win = 500  # averaging window

    # Test-specific parameters
    if test == "t1_log":
        Re = 100000
        tau = 0.500144
        use_wall_fn = True
        wall_law = "log"
        y_val = 1.0
        p0_method = "far_field"
        tag = f"[SDAA:{device_id} T1-LOG Re=1e5 log y=1.0 p0=far_field]"
    elif test == "t2_log_p0fix":
        Re = 2000000
        tau = 0.5000072
        use_wall_fn = True
        wall_law = "log"
        y_val = 1.0
        p0_method = "far_field"
        tag = f"[SDAA:{device_id} T2-LOG Re=2e6 log y=1.0 p0=far_field]"
    elif test == "t3_log_re1e4":
        Re = 10000
        tau = 0.50144
        use_wall_fn = True
        wall_law = "log"
        y_val = 0.5
        p0_method = "far_field"
        tag = f"[SDAA:{device_id} T3-LOG Re=1e4 log y=0.5 p0=far_field]"
    elif test == "t4_p0_compare":
        Re = 100000
        tau = 0.500144
        use_wall_fn = True
        wall_law = "log"
        y_val = 1.0
        p0_method = "far_field"  # primary, but compute all 3
        tag = f"[SDAA:{device_id} T4-P0CMP Re=1e5 log y=1.0 p0_compare]"
    else:
        raise ValueError(f"Unknown test: {test}")

    nu = u_in * L / Re

    # SUBOFF geometry
    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5

    # Wetted-area dynamic pressure scale
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    # ITTC-1957 reference
    Cf_ref = 0.075 / (math.log10(Re) - 2.0) ** 2

    print(
        f"{tag} test={test} use_wf={use_wall_fn} wall_law={wall_law} "
        f"y_val={y_val} p0_method={p0_method} nx={nx} ny={ny} nz={nz} "
        f"L={L} R={radius:.4f} D={D:.4f} u_in={u_in} nu={nu:.6e} "
        f"tau={tau:.6f} Cs={cs_smag} dpS={dpS:.6e} Cf_ref={Cf_ref:.6f} "
        f"n_steps={n_steps}",
        flush=True,
    )

    t0 = time.time()

    # Build SUBOFF solid mask
    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    # Near-wall mask + surface mesh (from_suboff analytical normals)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)

    # Normal stats
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )

    # NoDynamics solid mask (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # History buffers
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    # For TEST 4: all three p_0 methods
    cd_p_near_hist = []
    cd_p_far_hist = []
    cd_p_dom_hist = []

    step_done = 0
    diverged = False

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        if use_wall_fn:
            # Wall function approach:
            # stream → wall_function (body force) → far_field → mass_corr
            f = stream3d(f)
            f, drag_fric_raw, drag_pres_wf = wall_function_3d(
                f, solid, nu, y_val=y_val, wall_law=wall_law, near_mask=near,
            )
            f = far_field_bc_3d(f, u_in)
            if step % 200 == 0:
                f = correct_mass3d(f, im)

            # Drag: friction from wall function, pressure from surface mesh
            cd_f = drag_fric_raw / dpS
            fx_p, _, _ = drag_pressure_integration(
                f, mesh, dpS, p0_method=p0_method, solid=solid)
            cd_p = fx_p
        else:
            # Standard approach: bounce-back → stream → far_field → mass_corr
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)
            if step % 200 == 0:
                f = correct_mass3d(f, im)

            # Drag from surface mesh integration
            fx_p, _, _ = drag_pressure_integration(
                f, mesh, dpS, p0_method=p0_method, solid=solid)
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p = fx_p
            cd_f = fx_f

        cd_tot = cd_p + cd_f
        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        step_done = step

        # For TEST 4: compute Cd_p with all three p_0 methods
        if test == "t4_p0_compare":
            fx_near, _, _ = drag_pressure_integration(
                f, mesh, dpS, p0_method="near_wall", solid=solid)
            fx_far, _, _ = drag_pressure_integration(
                f, mesh, dpS, p0_method="far_field", solid=solid)
            fx_dom, _, _ = drag_pressure_integration(
                f, mesh, dpS, p0_method="domain_avg", solid=solid)
            cd_p_near_hist.append(fx_near)
            cd_p_far_hist.append(fx_far)
            cd_p_dom_hist.append(fx_dom)

        # Divergence check
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        # Progress report
        if step % 500 == 0:
            n_avg = min(win, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            extra = ""
            if test == "t4_p0_compare" and len(cd_p_near_hist) > 0:
                pn = sum(cd_p_near_hist[-n_avg:]) / n_avg
                pf = sum(cd_p_far_hist[-n_avg:]) / n_avg
                pd = sum(cd_p_dom_hist[-n_avg:]) / n_avg
                extra = (f" p_near={pn:.6f} p_far={pf:.6f} p_dom={pd:.6f}")
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f}{extra} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages
    n_final = min(win, len(cd_tot_hist))
    if n_final == 0:
        cd_p_final = cd_f_final = cd_tot_final = float("nan")
    else:
        cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
        cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
        cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final

    err_pct = (
        abs(cd_tot_final - Cf_ref) / Cf_ref * 100
        if Cf_ref > 0 and not diverged
        else float("inf")
    )

    result = {
        "case": tag.strip("[]"),
        "test": test,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "model": "smag_wf" if use_wall_fn else "smag_bb",
        "wall_law": wall_law,
        "y_val": y_val,
        "p0_method": p0_method,
        "L": L,
        "D": D,
        "radius": radius,
        "L_D_ratio": stats.get("L_D_ratio"),
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "steps_completed": step_done,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "dpS_formula": "0.5*u_in^2*pi*D*L (wetted area)",
        "normal_method": "from_suboff",
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cf_ref": Cf_ref,
        "ref_name": "ITTC-1957 Cf=0.075/(log10(Re)-2)^2",
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "diverged": diverged,
        "elapsed_s": elapsed,
        "avg_window": win,
    }

    # TEST 4: add all three p_0 method results
    if test == "t4_p0_compare" and len(cd_p_near_hist) > 0:
        nf = min(win, len(cd_p_near_hist))
        result["p0_comparison"] = {
            "near_wall": {
                "method": "p0 = avg pressure at near-wall cells",
                "Cd_p": sum(cd_p_near_hist[-nf:]) / nf,
            },
            "far_field": {
                "method": "p0 = avg pressure at fluid cells (not near-wall)",
                "Cd_p": sum(cd_p_far_hist[-nf:]) / nf,
            },
            "domain_avg": {
                "method": "p0 = avg pressure over all fluid cells",
                "Cd_p": sum(cd_p_dom_hist[-nf:]) / nf,
            },
        }
        pn = result["p0_comparison"]["near_wall"]["Cd_p"]
        pf = result["p0_comparison"]["far_field"]["Cd_p"]
        pd = result["p0_comparison"]["domain_avg"]["Cd_p"]
        result["p0_comparison"]["near_minus_far"] = abs(pn - pf)
        result["p0_comparison"]["near_minus_dom"] = abs(pn - pd)
        result["p0_comparison"]["far_minus_dom"] = abs(pf - pd)

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} (ref Cf={Cf_ref:.6f}) "
        f"err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    if test == "t4_p0_compare" and len(cd_p_near_hist) > 0:
        nf = min(win, len(cd_p_near_hist))
        pn = sum(cd_p_near_hist[-nf:]) / nf
        pf = sum(cd_p_far_hist[-nf:]) / nf
        pd = sum(cd_p_dom_hist[-nf:]) / nf
        print(
            f"{tag} P0_CMP near={pn:.6f} far={pf:.6f} dom={pd:.6f} "
            f"|n-f|={abs(pn-pf):.2e} |n-d|={abs(pn-pd):.2e}",
            flush=True,
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python suboff_wallfn_loglaw_worker.py <test> <device_id> <output_path>")
        print("  test: t1_log | t2_log_p0fix | t3_log_re1e4 | t4_p0_compare")
        sys.exit(1)

    test = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    run(test, device_id, output_path)


if __name__ == "__main__":
    main()
