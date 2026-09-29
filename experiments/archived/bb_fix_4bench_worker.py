#!/usr/bin/env python3
"""BB-fix retest: 4 benchmarks via common interface (SDAA 8-11).

ALL via common interface ONLY:
  solid → get_near_wall_3d → SurfaceMesh.from_xxx → lbm_step_correct
       → drag_pressure_integration + drag_friction_integration
       → detect_strouhal

Bug 27 fix: lbm_step_correct uses f_pre (pre-collision) for half-way
bounce-back, giving correct no-slip.  Without f_pre, u_t is overestimated
and friction diverges with grid refinement.

BENCHMARK 1: Cylinder Re=200 D=48  (SDAA:8)  — 5000 steps, MRT+Smag(Cs=0.05)
BENCHMARK 2: Sphere  Re=100 D=40  (SDAA:9)  — 3000 steps
BENCHMARK 3: SUBOFF Re=1000 L=80 (SDAA:10) — 5000 steps
BENCHMARK 4: NACA0012 Re=1000 6L (SDAA:11) — 10000 steps

Usage:
  PYTHONPATH=src python bb_fix_4bench_worker.py <benchmark> <device_id> <output_path>
  benchmark: cylinder | sphere | suboff | naca0012
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    get_near_wall_3d,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


# ---------------------------------------------------------------------------
#  Geometry builders
# ---------------------------------------------------------------------------
def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device):
    """Boolean solid mask for a 3D sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


def build_naca0012(chord, nx, ny, x_le, y_c, device, nz=4, t=0.12):
    """Build NACA 0012 (symmetric) solid mask, 2D extruded in z."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    n_samples = 2000
    xc = np.linspace(0.0, 1.0, n_samples)
    yt = 5.0 * t * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    )
    x_surf = x_le + xc * chord
    y_upper = y_c + yt * chord
    y_lower = y_c - yt * chord
    for k in range(nz):
        for i in range(nx):
            xi = float(i)
            if xi < x_surf[0] or xi > x_surf[-1]:
                continue
            y_u = np.interp(xi, x_surf, y_upper)
            y_l = np.interp(xi, x_surf, y_lower)
            j_lo = max(0, int(math.floor(min(y_u, y_l))))
            j_hi = min(ny - 1, int(math.ceil(max(y_u, y_l))))
            if j_hi >= j_lo:
                solid[k, j_lo:j_hi + 1, i] = True
    return solid


# ---------------------------------------------------------------------------
#  BENCHMARK 1: Cylinder Re=200 D=48  (SDAA:8)
# ---------------------------------------------------------------------------
def run_cylinder(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    D = 48.0
    R = D / 2.0
    nx, ny, nz = 400, 160, 4
    u_in = 0.08
    Re = 200.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 5000
    Cd_ref = 1.33  # Henderson 1997, Re=200

    cx = nx * 0.25
    cy = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    tag = f"[SDAA:{device_id} Cylinder-BBfix Re=200 D={D}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} nu={nu:.6e} "
          f"tau={tau:.6f} Cs={cs_smag} n_steps={n_steps} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_cylinder_mask(nx, ny, nz, cx, cy, R, device)
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis='z')
    print(f"{tag} solid={n_solid} near={n_near} mesh=from_cylinder "
          f"({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in,
            far_field_bc_3d, correct_mass_fn=correct_mass3d,
            target_mass=im, step=step, mass_interval=200, C_s=cs_smag,
        )

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)

        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            print(f"{tag} step={step}/{n_steps} Cd_p={cd_p_avg:.4f} "
                  f"Cd_f={cd_f_avg:.4f} Cd_tot={cd_tot_avg:.4f} "
                  f"Cl={cl_avg:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_final = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_final = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cd_err = abs(cd_tot_final - Cd_ref) / Cd_ref * 100

    # Strouhal detection
    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=D, min_cycles=5)

    print(f"\n{tag} === FINAL (Cd_ref={Cd_ref}) ===", flush=True)
    print(f"{tag} Cd_p  = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f  = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot= {cd_tot_final:.6f}  err={cd_err:.1f}%", flush=True)
    print(f"{tag} St    = {st}", flush=True)
    print(f"{tag} time  = {elapsed:.0f}s", flush=True)

    result = {
        "case": "cylinder_bb_fix", "device": f"sdaa:{device_id}",
        "shape": "cylinder", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": cs_smag,
        "boundary": "halfway_BB(f_pre)+farfield",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "Re": Re,
        "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "mesh_type": "from_cylinder", "bb_fix": True,
        "Cd_pressure": float(cd_p_final), "Cd_friction": float(cd_f_final),
        "Cd_total": float(cd_tot_final), "Cd_ref": Cd_ref,
        "Cd_err_pct": float(cd_err), "Cl": float(cl_final),
        "St": st, "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 2: Sphere Re=100 D=40  (SDAA:9)
# ---------------------------------------------------------------------------
def run_sphere(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    D = 40.0
    R = D / 2.0
    nx, ny, nz = 180, 180, 180
    u_in = 0.08
    Re = 100.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 3000
    Cd_ref = 1.09

    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    A_frontal = math.pi * R ** 2
    dpS = 0.5 * u_in ** 2 * A_frontal

    tag = f"[SDAA:{device_id} Sphere-BBfix Re=100 D={D}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} nu={nu:.6e} "
          f"tau={tau:.6f} Cs={cs_smag} n_steps={n_steps} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_sphere_mask(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"{tag} solid={n_solid} near={n_near} mesh=from_sphere "
          f"({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in,
            far_field_bc_3d, correct_mass_fn=correct_mass3d,
            target_mass=im, step=step, mass_interval=200, C_s=cs_smag,
        )

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)

        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            print(f"{tag} step={step}/{n_steps} Cd_p={cd_p_avg:.4f} "
                  f"Cd_f={cd_f_avg:.4f} Cd_tot={cd_tot_avg:.4f} "
                  f"Cl={cl_avg:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_final = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_final = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cd_err = abs(cd_tot_final - Cd_ref) / Cd_ref * 100

    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=D, min_cycles=5)

    print(f"\n{tag} === FINAL (Cd_ref={Cd_ref}) ===", flush=True)
    print(f"{tag} Cd_p  = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f  = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot= {cd_tot_final:.6f}  err={cd_err:.1f}%", flush=True)
    print(f"{tag} St    = {st}", flush=True)
    print(f"{tag} time  = {elapsed:.0f}s", flush=True)

    result = {
        "case": "sphere_bb_fix", "device": f"sdaa:{device_id}",
        "shape": "sphere", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": cs_smag,
        "boundary": "halfway_BB(f_pre)+farfield",
        "grid": f"{nx}x{ny}x{nz}", "D": D, "Re": Re,
        "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "mesh_type": "from_sphere", "bb_fix": True,
        "Cd_pressure": float(cd_p_final), "Cd_friction": float(cd_f_final),
        "Cd_total": float(cd_tot_final), "Cd_ref": Cd_ref,
        "Cd_err_pct": float(cd_err), "Cl": float(cl_final),
        "St": st, "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 3: SUBOFF Re=1000 L=80  (SDAA:10)
# ---------------------------------------------------------------------------
def run_suboff(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    L = 80
    Re = 1000
    u_in = 0.06
    cs_smag = 0.05
    n_steps = 5000

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius

    nx, ny, nz = 200, 80, 80
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in ** 2 * math.pi * D * L
    Cf_ref = 1.328 / math.sqrt(Re)

    tag = f"[SDAA:{device_id} SUBOFF-BBfix Re=1000 L={L}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} D={D:.3f} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} n_steps={n_steps} "
          f"dpS={dpS:.6e} Cf_ref={Cf_ref:.6f}", flush=True)

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
    print(f"{tag} solid={n_solid} near={n_near} mesh=from_suboff "
          f"({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in,
            far_field_bc_3d, correct_mass_fn=correct_mass3d,
            target_mass=im, step=step, mass_interval=200, C_s=cs_smag,
        )

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)

        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            print(f"{tag} step={step}/{n_steps} Cd_p={cd_p_avg:.6f} "
                  f"Cd_f={cd_f_avg:.6f} Cd_tot={cd_tot_avg:.6f} "
                  f"Cl={cl_avg:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_final = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_final = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cd_err = abs(cd_tot_final - Cf_ref) / Cf_ref * 100

    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=D, min_cycles=5)

    print(f"\n{tag} === FINAL (Cf_ref={Cf_ref:.6f}) ===", flush=True)
    print(f"{tag} Cd_p  = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f  = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot= {cd_tot_final:.6f}  err={cd_err:.1f}%", flush=True)
    print(f"{tag} St    = {st}", flush=True)
    print(f"{tag} time  = {elapsed:.0f}s", flush=True)

    result = {
        "case": "suboff_bb_fix", "device": f"sdaa:{device_id}",
        "shape": "suboff", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": cs_smag,
        "boundary": "halfway_BB(f_pre)+farfield",
        "grid": f"{nx}x{ny}x{nz}", "L": L, "D": D, "Re": Re,
        "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "mesh_type": "from_suboff", "bb_fix": True,
        "Cd_pressure": float(cd_p_final), "Cd_friction": float(cd_f_final),
        "Cd_total": float(cd_tot_final), "Cf_ref": Cf_ref,
        "Cd_err_pct": float(cd_err), "Cl": float(cl_final),
        "St": st, "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 4: NACA 0012 Re=1000 6L  (SDAA:11)
# ---------------------------------------------------------------------------
def run_naca0012(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    chord = 100
    nx = 600   # 6 chord
    ny = 300   # 3 chord
    nz = 4
    u_in = 0.05
    Re = 1000
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 10000
    Cd_ref = 0.05  # experimental, friction-dominated at Re=1000

    x_le = int(nx * 0.25)
    y_c = ny // 2
    dpS = 0.5 * u_in ** 2 * chord * nz

    tag = f"[SDAA:{device_id} NACA0012-BBfix Re=1000 6L]"
    print(f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} n_steps={n_steps} "
          f"dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_naca0012(chord, nx, ny, x_le, y_c, device, nz=nz, t=0.12)
    n_solid = int(solid.sum().item())
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    # NACA 0012: m=0, p=0.5 (avoid div-by-zero), t=0.12
    # m=0 → camber=0 regardless of p
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord,
                                 m=0.0, p=0.5, t=0.12)
    print(f"{tag} solid={n_solid} near={n_near} mesh=from_naca "
          f"({time.time()-t0:.1f}s)", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    warmup = n_steps // 5

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in,
            far_field_bc_3d, correct_mass_fn=correct_mass3d,
            target_mass=im, step=step, mass_interval=200, C_s=cs_smag,
        )

        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p, cd_f = fx_p, fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)

        if step % 1000 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / max(n_avg, 1)
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / max(n_avg, 1)
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / max(n_avg, 1)
            cl_avg = sum(cl_hist[-n_avg:]) / max(n_avg, 1)
            print(f"{tag} step={step}/{n_steps} Cd_p={cd_p_avg:.6f} "
                  f"Cd_f={cd_f_avg:.6f} Cd_tot={cd_tot_avg:.6f} "
                  f"Cl={cl_avg:.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = min(max(n_steps // 10, 200), len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / max(n_final, 1)
    cd_f_final = sum(cd_f_hist[-n_final:]) / max(n_final, 1)
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / max(n_final, 1)
    cl_final = sum(cl_hist[-n_final:]) / max(n_final, 1)
    cd_err = abs(cd_tot_final - Cd_ref) / Cd_ref * 100

    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=chord, min_cycles=5)

    print(f"\n{tag} === FINAL (Cd_ref={Cd_ref}) ===", flush=True)
    print(f"{tag} Cd_p  = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f  = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot= {cd_tot_final:.6f}  err={cd_err:.1f}%", flush=True)
    print(f"{tag} St    = {st}", flush=True)
    print(f"{tag} time  = {elapsed:.0f}s", flush=True)

    result = {
        "case": "naca0012_bb_fix", "device": f"sdaa:{device_id}",
        "shape": "naca0012", "lattice": "D3Q19",
        "collision": "MRT+Smag", "Cs": cs_smag,
        "boundary": "halfway_BB(f_pre)+farfield",
        "grid": f"{nx}x{ny}x{nz}", "chord": chord, "Re": Re,
        "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps, "warmup": warmup,
        "n_solid": n_solid, "n_near": n_near, "dpS": dpS,
        "mesh_type": "from_naca", "bb_fix": True,
        "Cd_pressure": float(cd_p_final), "Cd_friction": float(cd_f_final),
        "Cd_total": float(cd_tot_final), "Cd_ref": Cd_ref,
        "Cd_err_pct": float(cd_err), "Cl": float(cl_final),
        "St": st, "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python bb_fix_4bench_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: cylinder | sphere | suboff | naca0012")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "cylinder":
        run_cylinder(device_id, output_path)
    elif benchmark == "sphere":
        run_sphere(device_id, output_path)
    elif benchmark == "suboff":
        run_suboff(device_id, output_path)
    elif benchmark == "naca0012":
        run_naca0012(device_id, output_path)
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
