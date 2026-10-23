#!/usr/bin/env python3
"""SUBOFF wall function verification at high Re + friction formula comparison.

TEST 1: SUBOFF Re=1e5 with wall function (gradient law, y_val=0.5) — SDAA:12
        SUBOFF Re=1e5 standard BB baseline                  — SDAA:13
TEST 2: SUBOFF Re=2e6 with wall function (log law, y_val=1.0) — SDAA:14
TEST 3: Friction formula comparison at Re=1000              — SDAA:15
        a) τ=2ν·u_t       (standard half-way BB)
        b) τ=ν·u_t/q      (BFL fix, q=0.5 for standard BB)
        c) τ=ν·du_t/dn    (velocity gradient, forward diff along normal)

Geometry:  SUBOFF bare hull, L=80, nx=200, ny=80, nz=80, u_in=0.06
dpS:       0.5 * u_in^2 * pi * D * L  (wetted area, D=2*R_max)
Normals:   from_suboff (analytical axisymmetric body-of-revolution normal)

Usage:
  python suboff_wallfn_highre_worker.py <test> <device_id> <output_path>
  test: t1_wf | t1_bb | t2_wf | t3_formulas
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
    _shift_along_normal_dominant,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


# --------------------------------------------------------------------------- #
# Friction formula (c): velocity-gradient based
# --------------------------------------------------------------------------- #

def drag_friction_gradient(f, mesh, dpS, nu, delta_n=1.0):
    """Friction drag via velocity gradient along wall normal.

    τ = ν · du_t/dn,  where du_t/dn = (u_t_next - u_t_near) / delta_n

    u_t is the tangential velocity (velocity minus normal component).
    The "next" cell is sampled 1 lattice step further into the fluid along
    the dominant normal direction (via _shift_along_normal_dominant).

    For a linear velocity profile (laminar), this reduces to the standard
    formula τ = 2ν·u_t when y_val = 0.5 and delta_n = 1.0:
        u_t_near = τ_w/ν · 0.5,  u_t_next = τ_w/ν · 1.5
        du_t/dn = (1.5 - 0.5)·τ_w/ν / 1.0 = τ_w/ν
        τ = ν · τ_w/ν = τ_w  ✓

    Returns: (Cd_f_x, Cd_f_y, Cd_f_z) = (ffx, ffy, ffz) / dpS
    """
    rho, ux, uy, uz = macroscopic3d(f)
    nx_n, ny_n, nz_n = mesh.nx_n, mesh.ny_n, mesh.nz_n

    # Tangential velocity at near-wall cell
    u_dot_n = ux * nx_n + uy * ny_n + uz * nz_n
    ut_x = ux - u_dot_n * nx_n
    ut_y = uy - u_dot_n * ny_n
    ut_z = uz - u_dot_n * nz_n

    # Sample velocity at next fluid cell along dominant normal direction
    ux_next = _shift_along_normal_dominant(ux, mesh, steps=1)
    uy_next = _shift_along_normal_dominant(uy, mesh, steps=1)
    uz_next = _shift_along_normal_dominant(uz, mesh, steps=1)

    # Tangential velocity at next cell (using SAME normal from near-wall cell)
    u_dot_n_next = ux_next * nx_n + uy_next * ny_n + uz_next * nz_n
    ut_x_next = ux_next - u_dot_n_next * nx_n
    ut_y_next = uy_next - u_dot_n_next * ny_n
    ut_z_next = uz_next - u_dot_n_next * nz_n

    # Velocity gradient (forward difference, delta_n = 1.0 cell spacing)
    dut_x_dn = (ut_x_next - ut_x) / delta_n
    dut_y_dn = (ut_y_next - ut_y) / delta_n
    dut_z_dn = (ut_z_next - ut_z) / delta_n

    # Wall shear stress: τ = ν · du_t/dn
    tau_x = nu * dut_x_dn
    tau_y = nu * dut_y_dn
    tau_z = nu * dut_z_dn

    mask = mesh.near.float() * mesh.dA
    ffx = (tau_x * mask).sum()
    ffy = (tau_y * mask).sum()
    ffz = (tau_z * mask).sum()
    return (float(ffx.item() / dpS), float(ffy.item() / dpS),
            float(ffz.item() / dpS))


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
    if test == "t1_wf":
        Re = 100000
        tau = 0.500144
        use_wall_fn = True
        wall_law = "gradient"
        y_val = 0.5
        tag = f"[SDAA:{device_id} T1-WF Re=1e5 gradient y=0.5]"
    elif test == "t1_bb":
        Re = 100000
        tau = 0.500144
        use_wall_fn = False
        wall_law = None
        y_val = None
        tag = f"[SDAA:{device_id} T1-BB Re=1e5 standard BB]"
    elif test == "t2_wf":
        Re = 2000000
        tau = 0.5000072
        use_wall_fn = True
        wall_law = "log"
        y_val = 1.0
        tag = f"[SDAA:{device_id} T2-WF Re=2e6 log y=1.0]"
    elif test == "t3_formulas":
        Re = 1000
        tau = 0.5144
        use_wall_fn = False
        wall_law = None
        y_val = None
        tag = f"[SDAA:{device_id} T3-FF Re=1000 formulas]"
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
        f"y_val={y_val} nx={nx} ny={ny} nz={nz} L={L} R={radius:.4f} "
        f"D={D:.4f} u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cf_ref={Cf_ref:.6f} n_steps={n_steps}",
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
    # For TEST 3: all three friction formulas
    cd_f_a_hist = []  # τ = 2ν·u_t (standard)
    cd_f_b_hist = []  # τ = ν·u_t/q (q=0.5)
    cd_f_c_hist = []  # τ = ν·du_t/dn (gradient)

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
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
            cd_p = fx_p
        else:
            # Standard approach: bounce-back → stream → far_field → mass_corr
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in)
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
        step_done = step

        # For TEST 3: compute all three friction formulas
        if test == "t3_formulas":
            # (a) τ = 2ν·u_t (standard, q_wall=None)
            fx_a, _, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=None)
            # (b) τ = ν·u_t/q (q=0.5, same as standard)
            q_half = torch.full((nz, ny, nx), 0.5, dtype=torch.float32, device=device)
            fx_b, _, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=q_half)
            # (c) τ = ν·du_t/dn (velocity gradient)
            fx_c, _, _ = drag_friction_gradient(f, mesh, dpS, nu, delta_n=1.0)
            cd_f_a_hist.append(fx_a)
            cd_f_b_hist.append(fx_b)
            cd_f_c_hist.append(fx_c)

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
            if test == "t3_formulas":
                fa = sum(cd_f_a_hist[-n_avg:]) / n_avg
                fb = sum(cd_f_b_hist[-n_avg:]) / n_avg
                fc = sum(cd_f_c_hist[-n_avg:]) / n_avg
                extra = (f" fa={fa:.6f} fb={fb:.6f} fc={fc:.6f}")
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

    # TEST 3: add all three friction formula results
    if test == "t3_formulas" and len(cd_f_a_hist) > 0:
        nf = min(win, len(cd_f_a_hist))
        result["friction_formulas"] = {
            "a_standard_2nu_ut": {
                "formula": "tau = 2*nu*u_t",
                "Cd_f": sum(cd_f_a_hist[-nf:]) / nf,
            },
            "b_bfl_fix_nu_ut_q": {
                "formula": "tau = nu*u_t/q (q=0.5)",
                "Cd_f": sum(cd_f_b_hist[-nf:]) / nf,
            },
            "c_gradient_nu_dutdn": {
                "formula": "tau = nu*du_t/dn (forward diff, delta_n=1.0)",
                "Cd_f": sum(cd_f_c_hist[-nf:]) / nf,
            },
        }
        fa = result["friction_formulas"]["a_standard_2nu_ut"]["Cd_f"]
        fb = result["friction_formulas"]["b_bfl_fix_nu_ut_q"]["Cd_f"]
        fc = result["friction_formulas"]["c_gradient_nu_dutdn"]["Cd_f"]
        result["friction_formulas"]["a_minus_b"] = abs(fa - fb)
        result["friction_formulas"]["a_minus_c"] = abs(fa - fc)
        result["friction_formulas"]["b_minus_c"] = abs(fb - fc)

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} (ref Cf={Cf_ref:.6f}) "
        f"err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    if test == "t3_formulas" and len(cd_f_a_hist) > 0:
        nf = min(win, len(cd_f_a_hist))
        fa = sum(cd_f_a_hist[-nf:]) / nf
        fb = sum(cd_f_b_hist[-nf:]) / nf
        fc = sum(cd_f_c_hist[-nf:]) / nf
        print(
            f"{tag} FORMULAS a(2νu)={fa:.6f} b(νu/q)={fb:.6f} "
            f"c(νdu/dn)={fc:.6f} |a-b|={abs(fa-fb):.2e} "
            f"|a-c|={abs(fa-fc):.2e}",
            flush=True,
        )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python suboff_wallfn_highre_worker.py <test> <device_id> <output_path>")
        print("  test: t1_wf | t1_bb | t2_wf | t3_formulas")
        sys.exit(1)

    test = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    run(test, device_id, output_path)


if __name__ == "__main__":
    main()
