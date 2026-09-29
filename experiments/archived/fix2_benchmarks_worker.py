#!/usr/bin/env python3
"""Fix remaining deterministic geometry benchmarks — 4 tasks on SDAA cards 4-7.

SDAA:4 — Cylinder Re=3900 3D+RANS smaller grid (D=48, nx=400, ny=240, nz=50, 20% blockage)
SDAA:5 — NACA 0012 Re=1000 from_naca (chord=100, nx=600, ny=300, nz=4)
SDAA:6 — BFL sphere friction 2nd-order (D=40, 180³, bfl_lagrange formula)
SDAA:7 — Channel Re_tau=180 RANS longer (64³, 20000 steps)

Usage:
  python fix2_benchmarks_worker.py <benchmark> <device_id> <output_path>
  benchmark: cyl_re3900_20pct | naca_re1000_from_naca | sphere_bfl_lagrange | channel_retau180_long
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

# ---- Common interface imports ----
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.rans_ke import KESolver, collide_rans_ke, C_MU, C_E1, C_E2
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.bfl_common import (
    bfl_bounce_back_common,
    compute_q_sphere_common,
    compute_q_wall_sphere,
)
from tensorlbm.ibm import ibm_apply_body_force_3d


# ========================================================================== #
# Geometry builders
# ========================================================================== #

def build_cylinder_solid(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along the z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device):
    """NACA 0012 symmetric airfoil (2D extruded in z)."""
    t = 0.12
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xc = (xx - x_le) / chord
    xc = xc.clamp(min=1e-6, max=1.0)
    yt = (
        5.0 * t * (
            0.2969 * torch.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc ** 2
            + 0.2843 * xc ** 3
            - 0.1015 * xc ** 4
        ) * chord
    )
    in_chord = (xx >= x_le) & (xx <= x_le + chord)
    in_profile = (yy - y_c).abs() <= yt
    return in_chord & in_profile


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Boolean solid mask for a sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


# ========================================================================== #
# BENCHMARK 1: Cylinder Re=3900 3D+RANS 20% blockage (SDAA:4)
# ========================================================================== #
def run_cyl_re3900_20pct(device_id, output_path):
    """Cylinder Re=3900 — TRUE 3D + RANS k-epsilon, 20% blockage.

    Previous run with 30% blockage (ny=160) gave 40.6% error.
    Reducing blockage to 20% (ny=240) should improve Cd accuracy.
    Reference: Cd=0.98 (Parnaudeau et al. 2008).
    Target: <25%.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cyl_re3900_20pct SDAA:{device_id}]"

    D = 48.0
    R = D / 2.0
    nx, ny, nz = 400, 240, 50    # 20% blockage (D/ny=0.20), 4.8M cells
    u_in = 0.08
    Re = 3900.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    warmup = 3000
    avg_window = 5000

    cx = nx * 0.25
    cy = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal
    blockage = D / ny

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} blockage={blockage:.2f} "
          f"MRT+RANS-k-epsilon steps={n_steps}", flush=True)
    print(f"{tag} 20% blockage (ny=240) vs previous 30% (ny=160)", flush=True)
    t0 = time.time()

    solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    print(f"{tag} SurfaceMesh.from_cylinder built", flush=True)

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    # Initialize k-epsilon solver
    ke = KESolver(nu=nu, nu_t_max=0.5)
    ke.initialize(ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    print(f"{tag} KESolver initialized, k0={ke._k.mean().item():.6e} "
          f"eps0={ke._eps.mean().item():.6e}", flush=True)

    # Wrap collide_rans_ke for lbm_step_correct
    collide_fn = functools.partial(collide_rans_ke, ke_solver=ke, mask=solid)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step > warmup and math.isfinite(cd_tot):
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)
        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            nu_t_max = float(ke.compute_nu_t(solid).max().item())
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.4f} Cd_f={af:.4f} "
                  f"Cd_tot={at:.4f} Cl={cl:.6f} nu_t_max={nu_t_max:.4e} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, D, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 0.98
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd_tot={cd_tot_f:.4f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "cyl_re3900_20pct", "device": f"sdaa:{device_id}",
        "shape": "cylinder_3d", "lattice": "D3Q19",
        "collision": "MRT+RANS-k-epsilon",
        "constants": {"C_mu": C_MU, "C_e1": C_E1, "C_e2": C_E2},
        "boundary": "halfway_BB+farfield(y±)+periodic(z±)",
        "normal_method": "from_cylinder",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "blockage": blockage,
        "n_steps": n_steps, "warmup": warmup, "avg_window": avg_window,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "key_result": "STABLE" if finite else "DIVERGED",
        "previous_err_pct": 40.6,
        "improvement": "20% blockage (ny=240) vs 30% (ny=160)",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 2: NACA 0012 Re=1000 from_naca (SDAA:5)
# ========================================================================== #
def run_naca_re1000_from_naca(device_id, output_path):
    """NACA 0012 Re=1000 — from_naca analytical normals.

    Previous from_gradient gave 156% error; from_naca gave 90%.
    from_naca is better (analytical normals for symmetric airfoil).
    MRT+Smagorinsky(0.05), far_field p0, quadratic extrap.
    Reference: Cd=0.05.
    Target: <50%.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[naca_re1000_from_naca SDAA:{device_id}]"

    chord = 100.0
    nx, ny, nz = 600, 300, 4
    u_in = 0.05
    Re = 1000.0
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    n_steps = 10000
    warmup = 2000
    cs_smag = 0.05

    x_le = nx * 0.20
    y_c = ny * 0.5
    A_frontal = chord * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} MRT+Smag steps={n_steps}", flush=True)
    print(f"{tag} from_naca analytical normals (NOT from_gradient); "
          f"far_field p0; quadratic extrap", flush=True)
    t0 = time.time()

    solid = build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # from_naca analytical normals (NOT from_gradient) per task spec
    # NACA 0012: m=0 (symmetric), t=0.12
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord, m=0.0, t=0.12)
    print(f"{tag} SurfaceMesh.from_naca built (m=0, t=0.12)", flush=True)

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    collide_fn = functools.partial(collide_smagorinsky_mrt3d, C_s=cs_smag)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field", solid=solid)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_p, cd_f = float(fx_p), float(fx_f)
        cd_tot = cd_p + cd_f
        cl = float(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step > warmup and math.isfinite(cd_tot):
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)
        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.6f} Cd_f={af:.6f} "
                  f"Cd_tot={at:.6f} Cl={cl:.6f} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    nf = min(2000, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, chord, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 0.05
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} "
          f"Cd_tot={cd_tot_f:.6f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "naca_re1000_from_naca", "device": f"sdaa:{device_id}",
        "shape": "naca0012_2d_extruded", "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "boundary": "halfway_BB+farfield(y±)+periodic(z±)",
        "normal_method": "from_naca",
        "p0_method": "far_field", "extrap": "quadratic",
        "grid": f"{nx}x{ny}x{nz}", "chord": chord, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "Cs": cs_smag,
        "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "previous_from_naca_err_pct": 90,
        "previous_from_gradient_err_pct": 156,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 3: BFL sphere friction 2nd-order Lagrange (SDAA:6)
# ========================================================================== #
def run_sphere_bfl_lagrange(device_id, output_path):
    """Sphere Re=100 — BFL with 2nd-order Lagrange friction formula.

    Previous BFL with 1st-order formula (nu*u_t/q) gave 25.2% error.
    Fix: use 2nd-order Lagrange formula: tau = nu*(3*u1 - u2/3)/(2*q).
    This reduces to standard 'lagrange' when q=0.5.
    Target: <15%.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[sphere_bfl_lagrange SDAA:{device_id}]"

    D = 40
    nx, ny, nz = 180, 180, 180
    Re = 100
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 3000
    warmup = 750
    Cd_ref = 1.09
    cs_smag = 0.05
    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
          f"dpS={dpS:.6e} Cd_ref={Cd_ref} use_bfl=True", flush=True)
    print(f"{tag} BFL with 2nd-order Lagrange: tau=nu*(3*u1-u2/3)/(2*q)", flush=True)
    t0 = time.time()

    # Build sphere mask
    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    # BFL q-field (analytical sphere) for bounce-back
    print(f"{tag} computing BFL q-values (analytical sphere)...", flush=True)
    t_q = time.time()
    bfl_mask, bfl_q = compute_q_sphere_common(
        nx, ny, nz, cx, cy, cz, R, device, lattice="D3Q19"
    )
    n_links = int(bfl_mask.sum().item())
    q_at_boundary = bfl_q[bfl_mask]
    bfl_stats = {
        "n_links": n_links,
        "q_min": float(q_at_boundary.min()) if n_links > 0 else None,
        "q_max": float(q_at_boundary.max()) if n_links > 0 else None,
        "q_mean": float(q_at_boundary.mean()) if n_links > 0 else None,
    }
    print(f"{tag} BFL q-field: {n_links} links ({time.time()-t_q:.1f}s) "
          f"q=[{bfl_stats['q_min']:.4f}, {bfl_stats['q_max']:.4f}] "
          f"mean={bfl_stats['q_mean']:.4f}", flush=True)

    # q_wall: normal distance from cell to sphere surface (r - R)
    q_wall = compute_q_wall_sphere(near, cx, cy, cz, R, device)
    q_at_near = q_wall[near]
    print(f"  q_wall: n_near={n_near} "
          f"q_min={float(q_at_near.min()):.4f} q_max={float(q_at_near.max()):.4f} "
          f"q_mean={float(q_at_near.mean()):.4f}", flush=True)

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist, fz_hist = [], [], [], [], []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        # NoDynamics
        sm = solid.unsqueeze(0).expand_as(f)
        f = torch.where(sm, f_pre, f)

        # BFL bounce-back
        f = bounce_back_cells_3d(f, solid)
        f_pre_stream = f.clone()
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        f = bfl_bounce_back_common(
            f, f_pre_stream, bfl_mask, bfl_q, lattice="D3Q19"
        )

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        # BFL friction with 2nd-order Lagrange formula
        fx_f, fy_f, fz_f = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=q_wall, formula="bfl_lagrange")
        cd_tot = fx_p + fx_f
        cl = fy_p + fy_f
        fz_tot = fz_p + fz_f

        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(fx_p)
                cd_f_hist.append(fx_f)
                cd_tot_hist.append(cd_tot)
                cl_hist.append(cl)
                fz_hist.append(fz_tot)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                print(
                    f"{tag} step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                    f"Cl={sum(cl_hist[-n_avg:])/n_avg:.6f} "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist) / n_final if cl_hist else float("nan")
    err_pct = (
        abs(cd_tot_final - Cd_ref) / Cd_ref * 100
        if Cd_ref > 0 and math.isfinite(cd_tot_final)
        else float("nan")
    )

    result = {
        "case": "sphere_Re100_bfl_lagrange",
        "device": f"sdaa:{device_id}",
        "Re": int(Re),
        "D": float(D),
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": float(u_in),
        "nu": float(nu),
        "tau": float(tau),
        "Cs": float(cs_smag),
        "n_steps": int(n_steps),
        "warmup": int(warmup),
        "n_solid": int(n_solid),
        "n_near": int(n_near),
        "dpS": float(dpS),
        "Cd_pressure": float(cd_p_final) if cd_p_final == cd_p_final else None,
        "Cd_friction": float(cd_f_final) if cd_f_final == cd_f_final else None,
        "Cd_total": float(cd_tot_final) if cd_tot_final == cd_tot_final else None,
        "Cl": float(cl_final) if cl_final == cl_final else None,
        "Cd_ref": float(Cd_ref),
        "error_pct": float(err_pct) if err_pct == err_pct else None,
        "bfl_stats": bfl_stats,
        "q_wall_method": "normal_distance_r_minus_R",
        "q_wall_mean": float(q_at_near.mean()),
        "friction_formula": "nu*(3*u1-u2/3)/(2*q) [bfl_lagrange 2nd-order]",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "previous_err_pct": 25.2,
        "previous_formula": "nu*u_t/q_wall (1st-order bfl)",
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cd={Cd_ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result


# ========================================================================== #
# BENCHMARK 4: Channel Re_tau=180 RANS longer (SDAA:7)
# ========================================================================== #
KAPPA = 0.41
B_LOG = 5.0


def log_law_uplus(yp: float) -> float:
    """Log-law: u+ = (1/kappa)*ln(y+) + B."""
    return math.log(yp) / KAPPA + B_LOG if yp > 0 else 0.0


def run_channel_retau180_long(device_id, output_path):
    """Turbulent channel Re_tau=180 with RANS k-epsilon, 20000 steps.

    Previous run with 5000 steps gave 28% Cf error (Cf=0.00940 vs ref 0.00735).
    Key question: Does longer simulation (20000 steps) improve Cf convergence?
    Target: <20%.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[channel_retau180_long SDAA:{device_id}]"

    nx, ny, nz = 64, 64, 64
    nu = 0.00049382716   # gives Re_tau=180 with h=32
    n_steps = 20000
    warmup = 2000

    h = ny / 2.0
    retau_target = 180.0
    u_tau_target = retau_target * nu / h
    body_force = u_tau_target ** 2 / h
    tau = 3.0 * nu + 0.5

    # Solid mask: top/bottom walls
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    fluid = ~solid
    n_fluid = int(fluid.sum().item())

    print(f"{tag} Re_tau target={retau_target:.0f}", flush=True)
    print(f"{tag} Grid: {nx}x{ny}x{nz}, h={h:.1f}, nu={nu:.6f}, tau={tau:.6f}",
          flush=True)
    print(f"{tag} u_tau_target={u_tau_target:.6f}, body_force={body_force:.6e}",
          flush=True)
    print(f"{tag} n_steps={n_steps}, warmup={warmup} (longer than 5000)",
          flush=True)
    print(f"{tag} Fluid cells: {n_fluid}", flush=True)

    # Initialize with turbulent mean profile + perturbations
    rng = torch.Generator(device='cpu')
    u_c_init = min(u_tau_target * 18.0, 0.05)
    y_cpu = torch.arange(ny, dtype=torch.float32)
    y_dist = torch.minimum(y_cpu, h * 2 - 1 - y_cpu).clamp(min=0.5)
    u_profile = u_c_init * (y_dist / h) ** (1.0 / 7.0)
    u_profile[0] = 0.0
    u_profile[-1] = 0.0
    ux_init = u_profile.unsqueeze(0).unsqueeze(-1).expand(nz, ny, nx).clone().to(device)
    ux_init += torch.randn(nz, ny, nx, generator=rng, device='cpu').to(device) * u_c_init * 0.03
    uy_init = torch.randn(nz, ny, nx, generator=rng, device='cpu').to(device) * u_c_init * 0.02
    uz_init = torch.randn(nz, ny, nx, generator=rng, device='cpu').to(device) * u_c_init * 0.03
    ux_init[solid] = 0.0
    uy_init[solid] = 0.0
    uz_init[solid] = 0.0

    rho0 = torch.ones(nz, ny, nx, device=device)
    f = equilibrium3d(rho0, ux_init, uy_init, uz_init, device=device)
    initial_mass = float(rho0.sum().item())

    # Initialize k-epsilon solver
    ke = KESolver(nu=nu, nu_t_max=0.5)
    ke.initialize(ux_init, uy_init, uz_init)
    print(f"{tag} KESolver initialized, k0={ke._k.mean().item():.6e} "
          f"eps0={ke._eps.mean().item():.6e}", flush=True)

    # For channel: use manual loop (periodic in x,z, walls at y=0,ny-1)
    ux_sum = torch.zeros(nz, ny, nx, device=device)
    cf_samples = []
    n_profile_samples = 0

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision with RANS k-epsilon (advances k/eps internally)
        f = collide_rans_ke(f, tau, ke, mask=solid, lattice="D3Q19", collision="MRT")

        # 3. NoDynamics: restore solid cells to pre-collision
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(f.shape[0]):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Streaming (periodic in x,z via wrap-around)
        f = stream3d(f)

        # 5. Compute macroscopic fields
        rho, ux, uy, uz = macroscopic3d(f)

        # 6. Apply uniform body force (Guo forcing)
        fx_field = torch.full((nz, ny, nx), body_force, device=device)
        fx_field[solid] = 0.0
        f = ibm_apply_body_force_3d(f, fx_field,
                                     torch.zeros_like(fx_field),
                                     torch.zeros_like(fx_field))

        # 7. Bounce-back on solid cells
        f = bounce_back_cells_3d(f, solid)

        # 8. Mass correction
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Sample after warmup (running sum to avoid OOM)
        if step > warmup:
            _, ux_s, _, _ = macroscopic3d(f)
            ux_sum += ux_s
            n_profile_samples += 1
            u_bulk = float(ux_s[fluid].mean().item())
            u_tau_sim = math.sqrt(body_force * h)
            cf = 2.0 * u_tau_sim ** 2 / (u_bulk ** 2) if u_bulk > 0 else 0.0
            cf_samples.append(cf)

        if step % 1000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux_c, _, _ = macroscopic3d(f)
            u_bulk_c = float(ux_c[fluid].mean().item())
            u_tau_c = math.sqrt(body_force * h)
            re_tau_c = u_tau_c * h / nu
            cf_c = 2.0 * u_tau_c ** 2 / (u_bulk_c ** 2) if u_bulk_c > 0 else 0.0
            nu_t_max = float(ke.compute_nu_t(solid).max().item())
            finite = bool(torch.isfinite(f).all().item())
            print(f"{tag} step={step:5d} u_bulk={u_bulk_c:.6f} u_tau={u_tau_c:.6f} "
                  f"Re_tau={re_tau_c:.1f} Cf={cf_c:.6f} nu_t_max={nu_t_max:.4e} "
                  f"finite={finite} [{elapsed:.0f}s]", flush=True)
            if not finite:
                print(f"{tag} DIVERGED at step {step}", flush=True)
                break

    elapsed = time.time() - t0

    # Compute mean velocity profile from running sum
    if n_profile_samples > 0:
        ux_mean = ux_sum / n_profile_samples  # (nz, ny, nx)
        ux_y = ux_mean.mean(dim=(0, 2))  # average over x,z → (ny,)

        y_coords = torch.arange(ny, dtype=torch.float32, device=device)
        y_dist = y_coords + 0.5
        y_plus = y_dist * u_tau_target / nu
        u_plus = ux_y / u_tau_target

        profile_data = []
        for j in range(1, ny - 1):
            yp_j = float(y_plus[j].item())
            up_j = float(u_plus[j].item())
            u_j = float(ux_y[j].item())
            profile_data.append({
                "y_cell": j,
                "y_dist": float(y_dist[j].item()),
                "y_plus": yp_j,
                "u_plus": up_j,
                "u": u_j,
                "u_plus_loglaw": log_law_uplus(yp_j),
            })

        log_region = [(d["y_plus"], d["u_plus"], d["u_plus_loglaw"])
                       for d in profile_data if d["y_plus"] > 30]
        if log_region:
            rms_error = math.sqrt(
                sum((up - up_ll)**2 for _, up, up_ll in log_region) / len(log_region))
        else:
            rms_error = float('nan')
        n_samples = n_profile_samples
    else:
        profile_data = []
        rms_error = float('nan')
        n_samples = 0

    u_tau_final = math.sqrt(body_force * h)
    re_tau_final = u_tau_final * h / nu
    cf_mean = sum(cf_samples) / max(len(cf_samples), 1) if cf_samples else float("nan")
    cf_ref = 0.0073461891643709825  # from DNS
    cf_err = abs(cf_mean - cf_ref) / cf_ref * 100 if cf_ref > 0 and math.isfinite(cf_mean) else float("nan")
    finite = bool(torch.isfinite(f).all().item())

    print(f"\n{tag} === FINAL ===  Re_tau={re_tau_final:.1f} Cf={cf_mean:.6f} "
          f"(ref={cf_ref}) Cf_err={cf_err:.1f}% RMS_loglaw={rms_error:.4f} "
          f"finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "channel_retau180_rans_ke_long", "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "MRT+RANS-k-epsilon",
        "constants": {"C_mu": C_MU, "C_e1": C_E1, "C_e2": C_E2},
        "boundary": "periodic(x,z)+bounce_back(y±)+body_force",
        "grid": f"{nx}x{ny}x{nz}",
        "Re_tau_target": retau_target,
        "Re_tau_achieved": re_tau_final,
        "nu": nu, "tau": tau,
        "u_tau_target": u_tau_target,
        "u_tau_final": u_tau_final,
        "body_force": body_force,
        "n_steps": n_steps, "warmup": warmup,
        "n_samples": n_samples,
        "cf_mean": cf_mean,
        "cf_ref": cf_ref,
        "cf_err_pct": cf_err,
        "rms_error_loglaw": rms_error,
        "finite": finite,
        "elapsed_s": elapsed,
        "profile": profile_data,
        "previous_steps": 5000,
        "previous_cf_err_pct": 28.0,
        "key_question": "Does longer simulation (20000 steps) improve Cf?",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# Main
# ========================================================================== #
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python fix2_benchmarks_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: cyl_re3900_20pct | naca_re1000_from_naca | sphere_bfl_lagrange | channel_retau180_long")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    runners = {
        "cyl_re3900_20pct": run_cyl_re3900_20pct,
        "naca_re1000_from_naca": run_naca_re1000_from_naca,
        "sphere_bfl_lagrange": run_sphere_bfl_lagrange,
        "channel_retau180_long": run_channel_retau180_long,
    }

    if benchmark not in runners:
        print(f"Unknown benchmark: {benchmark}")
        print(f"Available: {list(runners.keys())}")
        sys.exit(1)

    runners[benchmark](device_id, output_path)
