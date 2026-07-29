"""Force computation method comparison with BB fix: 5 methods on 4 cases.

Fixes applied:
  1. BB fix: bounce_back_cells_3d(f, solid, f_pre=f_pre) — pre-collision f
  2. from_cylinder / from_suboff analytical normals (not from_gradient)
  3. Cylinder D=48 (not D=20) for grid convergence
  4. MEM computed BEFORE streaming (save pre-stream f)
  5. lbm_step_correct() pattern: collision → NoDynamics → BB(f_pre) → stream → BC

Tests:
  TEST 1: Cylinder D=48, Re=200, MRT+Smag(Cs=0.05), 5000 steps (SDAA:12)
          MEM vs pressure+friction, Cd_ref=1.30
  TEST 2: Couette, all 5 force methods, target 0.00% (SDAA:13)
  TEST 3: Poiseuille, all 5 force methods, target u_max 0.00% (SDAA:14)
  TEST 4: SUBOFF Re=1000, 5000 steps, MEM vs pressure+friction, Cd_ref=0.042 (SDAA:15)

Usage:
  PYTHONPATH=src python force_methods_test_worker.py --case cylinder  --device sdaa:12
  PYTHONPATH=src python force_methods_test_worker.py --case couette   --device sdaa:13
  PYTHONPATH=src python force_methods_test_worker.py --case poiseuille --device sdaa:14
  PYTHONPATH=src python force_methods_test_worker.py --case suboff    --device sdaa:15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.force_methods import (
    force_momentum_exchange,
    force_stress_integration,
    force_pressure_integration,
    force_virtual_work,
    force_immersed_boundary,
)
from tensorlbm.d3q19 import (
    equilibrium3d as eq3d,
    macroscopic3d as macro3d,
    C as C3D,
    W as W3D,
    OPPOSITE as OPP3D,
)
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_2d,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)

ALL_METHODS = ["mem_standard", "mem_galilean", "stress", "pressure", "virtual_work", "ib"]


# ---------------------------------------------------------------------------
# 3D BGK collision (for Couette/Poiseuille)
# ---------------------------------------------------------------------------

def collide_bgk3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    rho, ux, uy, uz = macro3d(f)
    feq = eq3d(rho, ux, uy, uz, device=f.device)
    return f - (f - feq) / tau


def collide_bgk3d_force(f: torch.Tensor, tau: float, g_x: float) -> torch.Tensor:
    """BGK collision with Guo body force (D3Q19).

    Uses the proper Guo forcing scheme:
      1. Shift equilibrium velocity: u* = u + tau*F/rho
      2. Collision with shifted equilibrium
      3. Add forcing term: (1-1/2tau) * w_q * (c_q-u)/cs2 * F
    """
    rho, ux, uy, uz = macro3d(f)
    # Guo velocity shift: u* = u + tau * F / rho
    ux_shift = ux + tau * g_x / rho.clamp(min=1e-12)
    feq = eq3d(rho, ux_shift, uy, uz, device=f.device)
    f_post = f - (f - feq) / tau
    # Guo forcing term
    c = C3D.to(f.device).float()
    w = W3D.to(f.device).float()
    cs2 = 1.0 / 3.0
    cs4 = cs2 * cs2
    coef = (1.0 - 0.5 / tau)
    for q in range(19):
        cu = c[q, 0] * ux + c[q, 1] * uy + c[q, 2] * uz
        force_q = coef * w[q] * (
            (c[q, 0] - ux) / cs2 + c[q, 0] * cu / cs4
        ) * g_x
        f_post[q] = f_post[q] + force_q
    return f_post


# ---------------------------------------------------------------------------
# Correct LBM step with BB fix (lbm_step_correct pattern)
# ---------------------------------------------------------------------------

def lbm_step_bb_fix(
    f: torch.Tensor,
    collide_fn,
    solid: torch.Tensor,
    tau: float,
    **collide_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LBM step with NoDynamics + half-way BB(f_pre), BEFORE streaming.

    Returns (f_post_bb, f_pre) where f_post_bb is the post-BB pre-stream
    distribution (for MEM computation) and f_pre is the pre-collision
    distribution.
    """
    f_pre = f.clone()
    f = collide_fn(f, tau=tau, **collide_kwargs)
    # NoDynamics: restore solid cells to pre-collision
    sm = solid.unsqueeze(0).expand_as(f)
    for q in range(f.shape[0]):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    # Half-way BB with f_pre (BB fix)
    f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
    return f, f_pre


# ---------------------------------------------------------------------------
# TEST 2: Couette Flow (3D, BB fix, all 5 force methods)
# ---------------------------------------------------------------------------

def run_couette(device: str = "sdaa:13", ny: int = 34, n_steps: int = 3000):
    """3D Couette flow with BB fix. Target: 0.00% error for all methods."""
    dev = torch.device(device)
    torch.sdaa.set_device(dev)
    nx = 128
    nz = 4
    u_wall = 0.05
    nu_lat = 0.02
    tau = 3.0 * nu_lat + 0.5
    H = ny - 2

    # Solid masks
    solid_all = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid_all[:, 0, :] = True    # bottom wall (stationary)
    solid_all[:, -1, :] = True  # top wall (moving)
    solid_bottom = torch.zeros_like(solid_all)
    solid_bottom[:, 0, :] = True

    # Near-wall masks
    near_all = get_near_wall_3d(solid_all)
    near_bottom = get_near_wall_3d(solid_bottom)

    # Initialize with linear profile
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.zeros(nz, ny, nx, device=dev)
    for j in range(1, ny - 1):
        ux0[:, j, :] = u_wall * (j - 0.5) / (ny - 2)
    ux0[solid_all] = 0
    f = eq3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)

    c = C3D.to(dev).float()
    w = W3D.to(dev).float()
    cs2 = 1.0 / 3.0

    force_history = {m: [] for m in ALL_METHODS}
    umax_history = []

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # --- Correct LBM step with BB fix ---
        f, f_pre = lbm_step_bb_fix(f, collide_bgk3d, solid_all, tau)

        # Moving top wall correction (after BB, before streaming)
        rho_top = f[:, :, -1, :].sum(dim=0)  # (nz, nx)
        for q in range(19):
            if c[q, 1] < 0:
                f[q, :, -1, :] = f[q, :, -1, :] - 2.0 * rho_top * w[q] * c[q, 1] * u_wall / cs2

        # Streaming
        f = stream3d(f)

        # Periodic in x and z (handled by torch.roll in stream3d)

        # Compute ALL methods AFTER streaming (post-stream, pre-next-BB)
        # MEM docstring: "Must be post-streaming, pre-bounce-back"
        if step > 200 and step % 50 == 0:
            res_std = force_momentum_exchange(f, solid_bottom, near_bottom, method="standard")
            if math.isfinite(res_std["fx"]):
                force_history["mem_standard"].append(res_std["fx"])
            res_gal = force_momentum_exchange(f, solid_bottom, near_bottom, method="galilean")
            if math.isfinite(res_gal["fx"]):
                force_history["mem_galilean"].append(res_gal["fx"])
            res_stress = force_stress_integration(f, solid_bottom, near_bottom, nu=nu_lat, tau=tau)
            if math.isfinite(res_stress["fx"]):
                force_history["stress"].append(res_stress["fx"])
            res_press = force_pressure_integration(f, solid_bottom, near_bottom)
            if math.isfinite(res_press["fx"]):
                force_history["pressure"].append(res_press["fx"])
            res_vw = force_virtual_work(f, solid_bottom, near_bottom)
            if math.isfinite(res_vw["fx"]):
                force_history["virtual_work"].append(res_vw["fx"])
            res_ib = force_immersed_boundary(f, solid_bottom, near_bottom)
            if math.isfinite(res_ib["fx"]):
                force_history["ib"].append(res_ib["fx"])

            # Check u_max
            rho, ux, uy, uz = macro3d(f)
            umax_history.append(float(ux[:, ny // 2, :].max().item()))

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            print(f"  step={step} ({time.time()-t0:.0f}s)", flush=True)

    # Analytical force on bottom wall
    tau_w = nu_lat * u_wall / H
    F_analytical = tau_w * nx * nz

    print(f"\n{'='*70}")
    print(f"COUETTE (BB fix): ny={ny}, nx={nx}, nz={nz}, u_wall={u_wall}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Analytical: tau_w={tau_w:.6f}, F_x={F_analytical:.6f}")
    print(f"\n{'Method':<25s} {'F_x (mean)':>12s} {'F_x (last)':>12s} {'Error%':>10s}")
    print("-" * 65)
    results = {}
    for method in ALL_METHODS:
        vals = force_history[method]
        if vals:
            mean_f = sum(vals) / len(vals)
            last_f = vals[-1]
            err = abs(mean_f - F_analytical) / (abs(F_analytical) + 1e-12) * 100
            print(f"{method:<25s} {mean_f:>12.6f} {last_f:>12.6f} {err:>9.2f}%")
            results[method] = {"mean": mean_f, "last": last_f, "error_pct": err}
        else:
            print(f"{method:<25s} {'N/A':>12s} {'N/A':>12s} {'N/A':>10s}")
            results[method] = {"mean": 0, "last": 0, "error_pct": None}

    return {
        "case": "couette_bb_fix",
        "grid": f"{nx}x{ny}x{nz}",
        "u_wall": u_wall, "nu": nu_lat, "tau": tau,
        "F_analytical": F_analytical,
        "methods": results,
        "u_max_final": umax_history[-1] if umax_history else None,
        "n_steps": n_steps,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# TEST 3: Poiseuille Flow (3D, BB fix, all 5 force methods)
# ---------------------------------------------------------------------------

def run_poiseuille(device: str = "sdaa:14", ny: int = 34, n_steps: int = 5000):
    """3D Poiseuille flow with BB fix. Target: u_max 0.00%."""
    dev = torch.device(device)
    torch.sdaa.set_device(dev)
    nx = 128
    nz = 4
    u_max = 0.05
    nu_lat = 0.02
    tau = 3.0 * nu_lat + 0.5
    H = ny - 2

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    near = get_near_wall_3d(solid)

    # Body force for Poiseuille: dp/dx = 8*nu*u_max/H^2
    g_x = 8.0 * nu_lat * u_max / (H * H)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.zeros(nz, ny, nx, device=dev)
    f = eq3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)

    force_history = {m: [] for m in ALL_METHODS}
    umax_history = []

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # --- Correct LBM step with BB fix + body force ---
        f_pre = f.clone()
        f = collide_bgk3d_force(f, tau, g_x)
        # NoDynamics
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # BB fix
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # Save pre-stream f for MEM
        f_pre_stream = f.clone()

        # Compute MEM BEFORE streaming
        if step > 500 and step % 100 == 0:
            for method_name, method_key in [("standard", "mem_standard"), ("galilean", "mem_galilean")]:
                res = force_momentum_exchange(f_pre_stream, solid, near, method=method_name)
                val = res["fx"]
                if math.isfinite(val):
                    force_history[method_key].append(val)

        # Streaming
        f = stream3d(f)

        # Compute other methods AFTER streaming
        if step > 500 and step % 100 == 0:
            res_stress = force_stress_integration(f, solid, near, nu=nu_lat, tau=tau)
            if math.isfinite(res_stress["fx"]):
                force_history["stress"].append(res_stress["fx"])
            res_press = force_pressure_integration(f, solid, near)
            if math.isfinite(res_press["fx"]):
                force_history["pressure"].append(res_press["fx"])
            res_vw = force_virtual_work(f, solid, near)
            if math.isfinite(res_vw["fx"]):
                force_history["virtual_work"].append(res_vw["fx"])
            res_ib = force_immersed_boundary(f, solid, near)
            if math.isfinite(res_ib["fx"]):
                force_history["ib"].append(res_ib["fx"])

            rho, ux, uy, uz = macro3d(f)
            umax_history.append(float(ux[:, ny // 2, :].max().item()))

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            u_curr = umax_history[-1] if umax_history else 0
            print(f"  step={step} u_max={u_curr:.6f} (target={u_max:.6f}) ({time.time()-t0:.0f}s)", flush=True)

    # Analytical: force on BOTH walls
    tau_w = 4.0 * nu_lat * u_max / H
    F_analytical = 2.0 * tau_w * nx * nz

    u_max_final = umax_history[-1] if umax_history else 0
    u_max_err = abs(u_max_final - u_max) / u_max * 100 if u_max > 0 else 0

    print(f"\n{'='*70}")
    print(f"POISEUILLE (BB fix): ny={ny}, nx={nx}, nz={nz}, u_max={u_max}, tau={tau:.4f}")
    print(f"{'='*70}")
    print(f"Analytical: tau_w={tau_w:.6f}, F_x(both walls)={F_analytical:.6f}")
    print(f"u_max: simulated={u_max_final:.6f}, analytical={u_max:.6f}, error={u_max_err:.2f}%")
    print(f"\n{'Method':<25s} {'F_x (mean)':>12s} {'Error%':>10s}")
    print("-" * 50)
    results = {}
    for method in ALL_METHODS:
        vals = force_history[method]
        if vals:
            mean_f = sum(vals) / len(vals)
            err = abs(mean_f - F_analytical) / (abs(F_analytical) + 1e-12) * 100
            print(f"{method:<25s} {mean_f:>12.6f} {err:>9.2f}%")
            results[method] = {"mean": mean_f, "error_pct": err}
        else:
            print(f"{method:<25s} {'N/A':>12s} {'N/A':>10s}")
            results[method] = {"mean": 0, "error_pct": None}

    return {
        "case": "poiseuille_bb_fix",
        "grid": f"{nx}x{ny}x{nz}",
        "u_max_target": u_max, "u_max_simulated": u_max_final,
        "u_max_error_pct": u_max_err,
        "nu": nu_lat, "tau": tau,
        "F_analytical": F_analytical,
        "methods": results,
        "n_steps": n_steps,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# TEST 1: Cylinder D=48, Re=200, MRT+Smag (SDAA:12)
# ---------------------------------------------------------------------------

def run_cylinder(device: str = "sdaa:12", n_steps: int = 5000):
    """3D Cylinder D=48 with BB fix. MEM vs pressure+friction. Cd_ref=1.30."""
    dev = torch.device(device)
    torch.sdaa.set_device(dev)

    D = 48
    nx, ny, nz = 400, 160, 4
    radius = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    Re = 200
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal
    Cd_ref = 1.30

    print(f"[SDAA CYL D=48] nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} Cd_ref={Cd_ref}", flush=True)

    t0 = time.time()
    # Build cylinder mask (extruded along z)
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    n_solid = int(solid.sum().item())
    print(f"  solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid, axis='z')
    n_near = int(near.sum().item())
    print(f"  near-wall cells: {n_near}", flush=True)

    # Analytical normals from cylinder geometry
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, radius, axis='z')

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = eq3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    im = float(rho0.sum().item())
    print(f"  init done ({time.time()-t0:.1f}s), mass={im}", flush=True)

    bc_config = {'far_field_faces': ['y-', 'y+'], 'periodic_faces': ['z-', 'z+']}
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    mem_hist = []
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = max(1000, n_steps // 4)

    for step in range(1, n_steps + 1):
        # --- Correct LBM step with BB fix ---
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        # NoDynamics
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # BB fix: use f_pre for correct half-way bounce-back
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # Save pre-stream f for MEM
        f_pre_stream = f.clone()

        # Compute MEM BEFORE streaming (post-BB, pre-stream)
        if step > warmup and step % 50 == 0:
            res_mem = force_momentum_exchange(f_pre_stream, solid, near, method="standard")
            if math.isfinite(res_mem["fx"]):
                mem_hist.append(res_mem["fx"])

        # Streaming
        f = stream3d(f)
        # Far-field BC
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Compute pressure+friction AFTER streaming
        if step > warmup and step % 50 == 0:
            fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, solid=solid, p0_method='far_field')
            fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)
            cl_hist.append(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(100, len(cd_tot_hist))
            if n_avg > 0:
                cd_mem = sum(mem_hist[-n_avg:]) / n_avg / dpS if mem_hist else 0
                print(f"  step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.4f} "
                      f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.4f} "
                      f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.4f} "
                      f"Cd_MEM={cd_mem:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  step={step} (warmup, {time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p = sum(cd_p_hist) / n_final
    cd_f = sum(cd_f_hist) / n_final
    cd_tot = sum(cd_tot_hist) / n_final
    cl = sum(cl_hist) / n_final
    cd_mem = sum(mem_hist) / max(1, len(mem_hist)) / dpS if mem_hist else float('nan')

    err_pf = abs(cd_tot - Cd_ref) / Cd_ref * 100
    err_mem = abs(cd_mem - Cd_ref) / Cd_ref * 100 if math.isfinite(cd_mem) else float('nan')

    print(f"\n{'='*70}")
    print(f"CYLINDER D=48 Re=200 (BB fix): Cd_ref={Cd_ref}")
    print(f"{'='*70}")
    print(f"{'Method':<25s} {'Cd':>12s} {'Error%':>10s}")
    print("-" * 50)
    print(f"{'MEM (pre-stream)':<25s} {cd_mem:>12.4f} {err_mem:>9.1f}%")
    print(f"{'Pressure':<25s} {cd_p:>12.4f}")
    print(f"{'Friction':<25s} {cd_f:>12.4f}")
    print(f"{'Pressure+Friction':<25s} {cd_tot:>12.4f} {err_pf:>9.1f}%")
    print(f"{'Cl':<25s} {cl:>12.4f}")

    return {
        "case": "cylinder_D48_bb_fix",
        "D": D, "nx": nx, "ny": ny, "nz": nz, "Re": Re,
        "u_in": u_in, "tau": tau, "Cs": cs_smag,
        "Cd_ref": Cd_ref,
        "Cd_MEM": cd_mem, "Cd_MEM_error_pct": err_mem,
        "Cd_pressure": cd_p, "Cd_friction": cd_f,
        "Cd_pressure_friction": cd_tot, "Cd_pf_error_pct": err_pf,
        "Cl": cl,
        "n_steps": n_steps, "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# TEST 4: SUBOFF Re=1000 (SDAA:15)
# ---------------------------------------------------------------------------

def run_suboff(device: str = "sdaa:15", n_steps: int = 5000):
    """3D SUBOFF with BB fix. MEM vs pressure+friction. Cd_ref=0.042."""
    dev = torch.device(device)
    torch.sdaa.set_device(dev)

    from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

    L = 80
    nx, ny, nz = 200, 80, 80
    Re = 1000
    u_in = 0.06
    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    dpS = 0.5 * u_in ** 2 * math.pi * D * L
    Cd_ref = 0.042

    print(f"[SDAA SUBOFF] nx={nx} ny={ny} nz={nz} L={L} R={radius:.4f} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} Cd_ref={Cd_ref}", flush=True)

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=dev,
    )
    n_solid = int(solid.sum().item())
    print(f"  solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"  near-wall cells: {n_near}", flush=True)

    # Analytical normals from SUBOFF geometry
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
    print(f"  mesh built ({time.time()-t0:.1f}s)", flush=True)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = eq3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    im = float(rho0.sum().item())
    print(f"  init done ({time.time()-t0:.1f}s), mass={im}", flush=True)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    warmup = max(1000, n_steps // 4)

    mem_hist = []
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []

    for step in range(1, n_steps + 1):
        # --- Correct LBM step with BB fix ---
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        # NoDynamics
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # BB fix
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # Save pre-stream f for MEM
        f_pre_stream = f.clone()

        # Compute MEM BEFORE streaming (less frequently for large grid)
        if step > warmup and step % 200 == 0:
            res_mem = force_momentum_exchange(f_pre_stream, solid, near, method="standard")
            if math.isfinite(res_mem["fx"]):
                mem_hist.append(res_mem["fx"])

        # Streaming
        f = stream3d(f)
        # Far-field BC
        f = far_field_bc_3d(f, u_in)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Compute pressure+friction AFTER streaming
        if step > warmup and step % 100 == 0:
            fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, solid=solid, p0_method='far_field')
            fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)
            cl_hist.append(fy_p + fy_f)

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(50, len(cd_tot_hist))
            if n_avg > 0:
                cd_mem = sum(mem_hist[-n_avg:]) / n_avg / dpS if mem_hist else 0
                print(f"  step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                      f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                      f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                      f"Cd_MEM={cd_mem:.6f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  step={step} (warmup, {time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p = sum(cd_p_hist) / n_final
    cd_f = sum(cd_f_hist) / n_final
    cd_tot = sum(cd_tot_hist) / n_final
    cl = sum(cl_hist) / n_final
    cd_mem = sum(mem_hist) / max(1, len(mem_hist)) / dpS if mem_hist else float('nan')

    err_pf = abs(cd_tot - Cd_ref) / Cd_ref * 100
    err_mem = abs(cd_mem - Cd_ref) / Cd_ref * 100 if math.isfinite(cd_mem) else float('nan')

    print(f"\n{'='*70}")
    print(f"SUBOFF Re=1000 (BB fix): Cd_ref={Cd_ref}")
    print(f"{'='*70}")
    print(f"{'Method':<25s} {'Cd':>12s} {'Error%':>10s}")
    print("-" * 50)
    print(f"{'MEM (pre-stream)':<25s} {cd_mem:>12.6f} {err_mem:>9.1f}%")
    print(f"{'Pressure':<25s} {cd_p:>12.6f}")
    print(f"{'Friction':<25s} {cd_f:>12.6f}")
    print(f"{'Pressure+Friction':<25s} {cd_tot:>12.6f} {err_pf:>9.1f}%")
    print(f"{'Cl':<25s} {cl:>12.6f}")

    return {
        "case": "suboff_bb_fix",
        "L": L, "D": D, "nx": nx, "ny": ny, "nz": nz, "Re": Re,
        "u_in": u_in, "tau": tau, "Cs": cs_smag,
        "Cd_ref": Cd_ref,
        "Cd_MEM": cd_mem, "Cd_MEM_error_pct": err_mem,
        "Cd_pressure": cd_p, "Cd_friction": cd_f,
        "Cd_pressure_friction": cd_tot, "Cd_pf_error_pct": err_pf,
        "Cl": cl,
        "n_steps": n_steps, "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Force methods test with BB fix")
    parser.add_argument("--case", required=True,
                        choices=["couette", "poiseuille", "cylinder", "suboff", "all"],
                        help="Test case to run")
    parser.add_argument("--device", default="sdaa:12", help="SDAA device (e.g., sdaa:12)")
    parser.add_argument("--n-steps", type=int, default=None, help="Override n_steps")
    args = parser.parse_args()

    results = {}

    if args.case in ("cylinder", "all"):
        ns = args.n_steps or 5000
        results["cylinder"] = run_cylinder(device=args.device, n_steps=ns)
    if args.case in ("couette", "all"):
        ns = args.n_steps or 3000
        results["couette"] = run_couette(device=args.device, n_steps=ns)
    if args.case in ("poiseuille", "all"):
        ns = args.n_steps or 5000
        results["poiseuille"] = run_poiseuille(device=args.device, n_steps=ns)
    if args.case in ("suboff", "all"):
        ns = args.n_steps or 5000
        results["suboff"] = run_suboff(device=args.device, n_steps=ns)

    dev_id = args.device.split(":")[1] if ":" in args.device else "0"
    outfile = f"force_methods_results_sdaa{dev_id}.json"
    with open(outfile, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
