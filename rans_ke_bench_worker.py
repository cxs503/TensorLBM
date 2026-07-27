#!/usr/bin/env python3
"""RANS k-epsilon benchmark worker — tests k-ε model on 4 high-Re cases.

SDAA cards 4-7:
  4: Cylinder Re=3900   — compare with Smagorinsky (which DIVERGED)
  5: NACA 0012 Re=6e6   — stable at tau=0.50005, target Cd≈0.008
  6: Cube Re=40000      — compare with LES (12.5%)
  7: Channel Re_tau=180 — compare log-law

Pipeline (external flow):
  solid → get_near_wall_3d → SurfaceMesh.from_xxx → lbm_step_correct →
  drag_pressure_integration + drag_friction_integration

Pipeline (channel flow):
  body-force driven periodic channel with RANS k-ε collision + bounce-back

Usage:
  python rans_ke_bench_worker.py <benchmark> <device_id> <output_path>
  benchmark: cyl_re3900 | naca_re6e6 | cube_re40000 | channel_retau180
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
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.rans_ke import KESolver, collide_rans_ke, C_MU, C_E1, C_E2
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.ibm import ibm_apply_body_force_3d


# ========================================================================== #
# Geometry builders                                                           #
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


def build_cube_mask(nx, ny, nz, cx, cz, D, device):
    """Wall-mounted cube: D×D×D cube on bottom wall (y=0)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    in_cube = (
        (xx >= cx - D / 2) & (xx < cx + D / 2)
        & (yy >= 0) & (yy < D)
        & (zz >= cz - D / 2) & (zz < cz + D / 2)
    )
    wall = yy < 2
    return in_cube | wall


def build_cube_solid_only(nx, ny, nz, cx, cz, D, device):
    """Cube body only (for drag measurement, without bottom wall)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (
        (xx >= cx - D / 2) & (xx < cx + D / 2)
        & (yy >= 0) & (yy < D)
        & (zz >= cz - D / 2) & (zz < cz + D / 2)
    )


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


# ========================================================================== #
# BENCHMARK 1: Cylinder Re=3900  (SDAA:4)                                    #
# ========================================================================== #

def run_cyl_re3900(device_id, output_path):
    """Cylinder Re=3900 with RANS k-epsilon.

    Previous Smagorinsky result: DIVERGED (finite=False, Cd=1.35).
    RANS k-epsilon should be stable.
    Reference: Cd=0.98 (Parnaudeau et al. 2008).
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cyl_re3900 RANS-ke SDAA:{device_id}]"

    D = 24.0
    R = D / 2.0
    nx, ny, nz = 200, 80, 4      # quasi-2D (same as Smagorinsky diverged case)
    u_in = 0.08
    Re = 3900.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 3000
    avg_window = 800

    cx = nx * 0.25
    cy = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+RANS-k-epsilon steps={n_steps}", flush=True)
    print(f"{tag} Smagorinsky DIVERGED here — testing RANS stability", flush=True)
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
    warmup = n_steps // 5

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
        if step % 200 == 0 or step == n_steps:
            n_avg = min(200, len(cd_tot_hist))
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
        "case": "cyl_re3900_rans_ke", "device": f"sdaa:{device_id}",
        "shape": "cylinder_3d", "lattice": "D3Q19",
        "collision": "MRT+RANS-k-epsilon",
        "constants": {"C_mu": C_MU, "C_e1": C_E1, "C_e2": C_E2},
        "boundary": "halfway_BB(f_pre)+farfield(y±)+periodic(z±)",
        "normal_method": "from_cylinder",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "smagorinsky_comparison": "DIVERGED (finite=False, Cd=1.35)",
        "key_result": "STABLE" if finite else "DIVERGED",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 2: NACA 0012 Re=6e6  (SDAA:5)                                     #
# ========================================================================== #

def run_naca_re6e6(device_id, output_path):
    """NACA 0012 Re=6e6 with RANS k-epsilon at tau=0.50005.

    Previous Smagorinsky result: DIVERGED (finite=False, Cd=0.205 nonsense).
    RANS k-epsilon should be stable at tau=0.50005.
    Target: Cd≈0.008 (skin friction dominated at high Re, 0° AoA).
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[naca_re6e6 RANS-ke SDAA:{device_id}]"

    chord = 1000.0
    nx, ny, nz = 800, 200, 4
    u_in = 0.1
    Re = 6e6
    nu = u_in * chord / Re   # = 1.6667e-5
    tau = 3.0 * nu + 0.5     # = 0.50005
    n_steps = 3000
    avg_window = 800

    x_le = nx * 0.15
    y_c = ny * 0.5
    A_frontal = chord * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+RANS-k-epsilon steps={n_steps}", flush=True)
    print(f"{tag} Smagorinsky DIVERGED at tau=0.50005 — testing RANS stability",
          flush=True)
    t0 = time.time()

    solid = build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells={n_solid}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord,
                                 m=0.0, p=0.5, t=0.12)
    print(f"{tag} SurfaceMesh.from_naca built", flush=True)

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

    collide_fn = functools.partial(collide_rans_ke, ke_solver=ke, mask=solid)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field",
            solid=solid)
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
        if step % 200 == 0 or step == n_steps:
            n_avg = min(200, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            nu_t_max = float(ke.compute_nu_t(solid).max().item())
            el = time.time() - t0
            print(f"{tag} step={step}/{n_steps} Cd_p={ap:.6f} Cd_f={af:.6f} "
                  f"Cd_tot={at:.6f} Cl={cl:.6f} nu_t_max={nu_t_max:.4e} "
                  f"({el:.0f}s, {el/step:.3f}s/step)", flush=True)

    elapsed = time.time() - t0
    nf = min(avg_window, len(cd_tot_hist))
    cd_p_f = sum(cd_p_hist[-nf:]) / max(nf, 1)
    cd_f_f = sum(cd_f_hist[-nf:]) / max(nf, 1)
    cd_tot_f = sum(cd_tot_hist[-nf:]) / max(nf, 1)
    cl_f = sum(cl_hist[-nf:]) / max(nf, 1)
    st = detect_strouhal(cl_hist, 1.0, u_in, chord, min_cycles=3) if len(cl_hist) > 100 else None

    cd_ref = 0.008
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.6f} Cd_f={cd_f_f:.6f} "
          f"Cd_tot={cd_tot_f:.6f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "naca_re6e6_rans_ke", "device": f"sdaa:{device_id}",
        "shape": "NACA0012", "lattice": "D3Q19",
        "collision": "MRT+RANS-k-epsilon",
        "constants": {"C_mu": C_MU, "C_e1": C_E1, "C_e2": C_E2},
        "boundary": "halfway_BB(f_pre)+farfield(y±)+periodic(z±)",
        "normal_method": "from_naca(m=0,p=0.5,t=0.12)",
        "grid": f"{nx}x{ny}x{nz}", "chord": chord, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "smagorinsky_comparison": "DIVERGED (finite=False, Cd=0.205 nonsense)",
        "key_result": "STABLE at tau=0.50005" if finite else "DIVERGED",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 3: Cube Re=40000  (SDAA:6)                                        #
# ========================================================================== #

def run_cube_re40000(device_id, output_path):
    """Wall-mounted cube Re=40000 with RANS k-epsilon.

    Previous Smagorinsky (LES) result: Cd=0.963, err=12.5%.
    RANS k-epsilon: compare with LES.
    Reference: Cd=1.1 (Martin & Lim 2008).
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[cube_re40000 RANS-ke SDAA:{device_id}]"

    D = 16.0
    nx, ny, nz = 128, 64, 64
    u_in = 0.08
    Re = 40000.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 2000
    avg_window = 500

    cx = nx * 0.3
    cz = nz * 0.5
    A_frontal = D * D
    dpS = 0.5 * u_in ** 2 * A_frontal

    print(f"{tag} D={D} nx={nx} ny={ny} nz={nz} u_in={u_in} Re={Re} "
          f"nu={nu:.6e} tau={tau:.6f} MRT+RANS-k-epsilon steps={n_steps}", flush=True)
    t0 = time.time()

    solid_total = build_cube_mask(nx, ny, nz, cx, cz, D, device)
    solid_cube = build_cube_solid_only(nx, ny, nz, cx, cz, D, device)
    n_solid = int(solid_total.sum().item())
    print(f"{tag} solid_total={n_solid}", flush=True)

    near = get_near_wall_3d(solid_cube)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall (cube)={n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid_cube, near)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    bc_config = {"far_field_faces": ["y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid_total] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0),
                      torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    # Initialize k-epsilon solver
    ke = KESolver(nu=nu, nu_t_max=0.5)
    ke.initialize(ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    print(f"{tag} KESolver initialized, k0={ke._k.mean().item():.6e} "
          f"eps0={ke._eps.mean().item():.6e}", flush=True)

    collide_fn = functools.partial(collide_rans_ke, ke_solver=ke, mask=solid_total)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid_total, u_in, far_field_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200,
        )

        fx_p, fy_p, _ = drag_pressure_integration(
            f, mesh, dpS, extrap="quadratic", p0_method="far_field",
            solid=solid_cube)
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
        if step % 200 == 0 or step == n_steps:
            n_avg = min(200, len(cd_tot_hist))
            ap = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            af = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            at = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            nu_t_max = float(ke.compute_nu_t(solid_total).max().item())
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

    cd_ref = 1.1
    cd_err = abs(cd_tot_f - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")
    finite = bool(torch.isfinite(f).all().item())
    print(f"\n{tag} === FINAL ===  Cd_p={cd_p_f:.4f} Cd_f={cd_f_f:.4f} "
          f"Cd_tot={cd_tot_f:.4f} (ref={cd_ref}) err={cd_err:.1f}% Cl={cl_f:.6f} "
          f"St={st} finite={finite} time={elapsed:.0f}s", flush=True)

    result = {
        "case": "cube_re40000_rans_ke", "device": f"sdaa:{device_id}",
        "shape": "wall_mounted_cube", "lattice": "D3Q19",
        "collision": "MRT+RANS-k-epsilon",
        "constants": {"C_mu": C_MU, "C_e1": C_E1, "C_e2": C_E2},
        "boundary": "halfway_BB(f_pre)+farfield(y+top,z±)",
        "normal_method": "from_gradient",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "u_in": u_in, "Re": Re,
        "nu": nu, "tau": tau, "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near,
        "Cd_pressure": cd_p_f, "Cd_friction": cd_f_f, "Cd_total": cd_tot_f,
        "Cd_ref": cd_ref, "Cd_err_pct": cd_err,
        "Cl": cl_f, "St": st,
        "n_samples": len(cd_tot_hist),
        "finite": finite,
        "elapsed_s": elapsed, "time_per_step_s": elapsed / n_steps,
        "les_comparison": "Cd_tot=0.963, err=12.47%",
        "key_result": "STABLE" if finite else "DIVERGED",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# BENCHMARK 4: Channel Re_tau=180  (SDAA:7)                                   #
# ========================================================================== #

KAPPA = 0.41
B_LOG = 5.0


def log_law_uplus(yp: float) -> float:
    """Log-law: u+ = (1/kappa)*ln(y+) + B."""
    return math.log(yp) / KAPPA + B_LOG if yp > 0 else 0.0


def run_channel_retau180(device_id, output_path):
    """Turbulent channel Re_tau=180 with RANS k-epsilon.

    Body-force driven periodic channel with RANS k-ε collision.
    Compare mean velocity profile against log-law.
    Previous Smagorinsky result: Cf=0.00888, RMS error=3.74.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[channel_retau180 RANS-ke SDAA:{device_id}]"

    nx, ny, nz = 64, 64, 64
    nu = 0.00049382716   # gives Re_tau=180 with h=32
    n_steps = 5000
    warmup = 1000

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
    print(f"{tag} n_steps={n_steps}, warmup={warmup}", flush=True)
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
    # collide_rans_ke advances k/eps internally, then collides
    # Use running sum for profile (avoid OOM from storing all samples)
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

        if step % 500 == 0 or step == n_steps:
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
    finite = bool(torch.isfinite(f).all().item())

    print(f"\n{tag} === FINAL ===  Re_tau={re_tau_final:.1f} Cf={cf_mean:.6f} "
          f"(ref={cf_ref}) RMS_loglaw={rms_error:.4f} finite={finite} "
          f"time={elapsed:.0f}s", flush=True)

    result = {
        "case": "channel_retau180_rans_ke", "device": f"sdaa:{device_id}",
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
        "rms_error_loglaw": rms_error,
        "finite": finite,
        "elapsed_s": elapsed,
        "profile": profile_data,
        "smagorinsky_comparison": "Cf=0.00888, RMS_loglaw=3.74",
        "key_result": "STABLE" if finite else "DIVERGED",
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ========================================================================== #
# Main                                                                        #
# ========================================================================== #

def main():
    if len(sys.argv) < 4:
        print("Usage: python rans_ke_bench_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: cyl_re3900 | naca_re6e6 | cube_re40000 | channel_retau180")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    runners = {
        "cyl_re3900": run_cyl_re3900,
        "naca_re6e6": run_naca_re6e6,
        "cube_re40000": run_cube_re40000,
        "channel_retau180": run_channel_retau180,
    }
    if benchmark not in runners:
        print(f"Unknown benchmark: {benchmark}")
        print(f"Available: {', '.join(runners.keys())}")
        sys.exit(1)

    runners[benchmark](device_id, output_path)


if __name__ == "__main__":
    main()
