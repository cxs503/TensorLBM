"""MEM vs Pressure+Friction comparison via common interface.

Key insight: MEM must be computed BEFORE streaming (after bounce-back),
when solid cells still contain the correct bounced-back populations.
After streaming with torch.roll, boundary solid cells get corrupted by
wraparound. P+F is computed AFTER streaming (uses only near-wall fluid
cells, which are correct).

Benchmarks (SDAA 12-15):
  1. Cylinder D=48, Re=200, 5000 steps, from_cylinder, far_field_bc  (Cd=1.30)
  2. Couette, from_gradient, BB fix, 3000 steps                     (0.00%)
  3. Poiseuille, from_gradient, body force, 3000 steps              (u_max 0.00%)
  4. SUBOFF, Re=1000, 5000 steps, from_suboff, far_field_bc         (Cd=0.042)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_sdaa  # noqa: F401

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.momentum_exchange import momentum_exchange_standard


def _float(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)

def _float_list(lst):
    return [_float(x) for x in lst]


# ---------------------------------------------------------------------------
# Proper Guo body-force collision for D3Q19 (x-direction, uniform)
# ---------------------------------------------------------------------------
def collide_bgk3d_guo(f, tau, Fx):
    """BGK collision with proper Guo body force (x-direction, uniform).

    Uses the original Guo (2002) forcing scheme:
      1. Physical velocity: u_phys = u + F/(2*rho)
      2. Equilibrium with u_phys
      3. Collision: f_post = f - (f - feq)/tau
      4. Force term: (1-1/2tau) * w * [(c-u_phys)/cs2 + (c·u_phys)*c/cs4] * F

    Note: Using u_phys = u + F/(2*rho) in BOTH equilibrium and force term
    gives the correct force. Using u* = u + tau*F/rho in equilibrium with
    raw u in force term gives (2 - 1/(2*tau))× the correct force.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    # Guo physical velocity: u_phys = u + F / (2 * rho)
    ux_guo = ux + Fx / (2.0 * rho.clamp(min=1e-12))
    feq = equilibrium3d(rho, ux_guo, uy, uz)
    f_post = f - (f - feq) / tau

    c = C.to(f.device).float()
    w = W.to(f.device).float()
    cs2 = 1.0 / 3.0
    cs4 = cs2 * cs2
    factor = (1.0 - 0.5 / tau)
    # Force term uses physical velocity u_phys (same as equilibrium)
    cu = (c[:, 0].view(19, 1, 1, 1) * ux_guo
          + c[:, 1].view(19, 1, 1, 1) * uy
          + c[:, 2].view(19, 1, 1, 1) * uz)
    force = factor * w.view(19, 1, 1, 1) * (
        (c[:, 0].view(19, 1, 1, 1) - ux_guo) / cs2
        + c[:, 0].view(19, 1, 1, 1) * cu / cs4
    ) * Fx
    return f_post + force


def moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top, f_pre, rho_w=1.0):
    """Half-way bounce-back with moving wall correction (uses f_pre for BB fix)."""
    opp = OPPOSITE.to(f.device)
    f = torch.where(solid.unsqueeze(0), f_pre[opp], f)
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    correction = 6.0 * rho_w * u_top * w * c[:, 0]
    top_mask = top_wall_mask.unsqueeze(0).float()
    f = f + correction.view(19, 1, 1, 1) * top_mask
    return f


# ---------------------------------------------------------------------------
# BENCHMARK 1: Cylinder D=48, Re=200
# ---------------------------------------------------------------------------
def run_cylinder(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    D = 48; Re = 200; u_in = 0.08
    nu_lat = u_in * D / Re; tau = 3.0 * nu_lat + 0.5
    n_steps = 5000; warmup = 1000
    nx = 10 * D; ny = 4 * D; nz = 4
    cx = nx // 4; cy = ny // 2; cz = nz // 2

    tag = f"[CylMEMvsPF SDAA:{device_id}]"
    print(f"{tag} D={D} Re={Re} nx={nx} ny={ny} nz={nz} u_in={u_in} tau={tau:.4f}", flush=True)
    t0 = time.time()

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij")
    R = D / 2.0
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2) <= R ** 2
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis='z', cz=cz)
    A_ref = D * nz; dpS = 0.5 * 1.0 * u_in ** 2 * A_ref
    print(f"{tag} solid={int(solid.sum())} near={int(near.sum())} dpS={dpS:.4f}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    mem_cd_hist = []; pf_p_hist = []; pf_f_hist = []; pf_tot_hist = []

    for step in range(1, n_steps + 1):
        # Custom step: collision → NoDynamics → BB → MEM → stream → far_field → P+F
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # MEM: compute BEFORE streaming (solid cells have correct bounced-back values)
        if step > warmup and step % 50 == 0:
            fx_mem, _, _ = momentum_exchange_standard(f, solid, near)
            cd_mem = fx_mem / dpS
            if math.isfinite(cd_mem):
                mem_cd_hist.append(cd_mem)

        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # P+F: compute AFTER streaming (uses near-wall fluid cells only)
        if step > warmup and step % 50 == 0:
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu_lat)
            if math.isfinite(fx_p + fx_f):
                pf_p_hist.append(fx_p); pf_f_hist.append(fx_f); pf_tot_hist.append(fx_p + fx_f)

        if step % 500 == 0 or step == n_steps:
            n_rec = len(mem_cd_hist)
            if n_rec > 0:
                print(f"{tag} step={step} Cd_MEM={sum(mem_cd_hist)/n_rec:.4f} "
                      f"Cd_PF={sum(pf_tot_hist)/n_rec:.4f} (ref=1.30) ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"{tag} step={step} (warmup) ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0; Cd_ref = 1.30
    n_rec = len(mem_cd_hist)
    cd_mem = sum(mem_cd_hist)/n_rec if n_rec else float('nan')
    cd_p = sum(pf_p_hist)/n_rec if n_rec else float('nan')
    cd_f = sum(pf_f_hist)/n_rec if n_rec else float('nan')
    cd_pf = sum(pf_tot_hist)/n_rec if n_rec else float('nan')
    mem_err = abs(cd_mem-Cd_ref)/Cd_ref*100 if math.isfinite(cd_mem) else float('nan')
    pf_err = abs(cd_pf-Cd_ref)/Cd_ref*100 if math.isfinite(cd_pf) else float('nan')
    diff = abs(cd_mem-cd_pf) if math.isfinite(cd_mem) and math.isfinite(cd_pf) else float('nan')

    print(f"\n{tag} === FINAL === Cd_MEM={cd_mem:.4f}(err={mem_err:.1f}%) Cd_P={cd_p:.4f} "
          f"Cd_F={cd_f:.4f} Cd_PF={cd_pf:.4f}(err={pf_err:.1f}%) |diff|={diff:.6f} t={elapsed:.0f}s", flush=True)
    result = {"case":"cylinder_D48_Re200","device":f"sdaa:{device_id}","D":D,"Re":Re,
              "nx":nx,"ny":ny,"nz":nz,"u_in":_float(u_in),"tau":_float(tau),"nu":_float(nu_lat),
              "n_steps":n_steps,"n_samples":n_rec,"Cd_ref":Cd_ref,
              "Cd_MEM":_float(cd_mem),"Cd_MEM_err_pct":_float(mem_err),
              "Cd_pressure":_float(cd_p),"Cd_friction":_float(cd_f),
              "Cd_PF_total":_float(cd_pf),"Cd_PF_err_pct":_float(pf_err),
              "MEM_PF_diff":_float(diff),"elapsed_s":_float(elapsed)}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# BENCHMARK 2: Couette flow
# ---------------------------------------------------------------------------
def run_couette(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 12; tau = 1.0
    nu = (tau - 0.5) / 3.0; u_top = 0.05
    n_steps = 3000; warmup = 500
    H = ny - 2; Cf_exact = 2.0 * nu / (H * u_top)

    tag = f"[CouetteMEMvsPF SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} u_top={u_top} Cf_exact={Cf_exact:.6f}", flush=True)
    t0 = time.time()

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    top_wall_mask = torch.zeros_like(solid); top_wall_mask[:, -1, :] = True
    near = get_near_wall_3d(solid)
    # Bottom wall (stationary) for force comparison
    near_bot = near.clone(); near_bot[:, ny // 2:, :] = False
    mesh_bot = SurfaceMesh.from_gradient(solid, near_bot)
    A_wall = nx * nz; dpS = 0.5 * 1.0 * u_top ** 2 * A_wall

    rho0 = torch.ones((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    mem_cf_hist = []; pf_f_hist = []; pf_p_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = moving_wall_bounce_back_3d(f, solid, top_wall_mask, u_top, f_pre)

        # MEM: BEFORE streaming (bottom wall, stationary)
        if step > warmup:
            fx_mem, _, _ = momentum_exchange_standard(f, solid, near_bot)
            cf_mem = fx_mem / dpS
            if math.isfinite(cf_mem):
                mem_cf_hist.append(cf_mem)

        f = stream3d(f)
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # P+F: AFTER streaming (bottom wall)
        if step > warmup:
            ffx, _, _ = drag_friction_integration(f, mesh_bot, dpS, nu)
            cdp_x, _, _ = drag_pressure_integration(f, mesh_bot, dpS, solid=solid, p0_method='far_field')
            if math.isfinite(ffx):
                pf_f_hist.append(ffx); pf_p_hist.append(cdp_x)

        if step % 500 == 0:
            n_rec = len(mem_cf_hist)
            mem_avg = sum(mem_cf_hist)/n_rec if n_rec else float('nan')
            pf_avg = sum(pf_f_hist)/n_rec if n_rec else float('nan')
            _, ux, _, _ = macroscopic3d(f)
            print(f"{tag} step={step} Cf_MEM={mem_avg:.6f} Cf_PF={pf_avg:.6f} "
                  f"(exact={Cf_exact:.6f}) u[mid]={float(ux.mean(dim=(0,2))[ny//2]):.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_rec = len(mem_cf_hist)
    cf_mem = sum(mem_cf_hist)/n_rec if n_rec else float('nan')
    cf_pf = sum(pf_f_hist)/n_rec if n_rec else float('nan')
    cf_p = sum(pf_p_hist)/n_rec if n_rec else float('nan')
    mem_err = abs(cf_mem-Cf_exact)/Cf_exact*100 if math.isfinite(cf_mem) else float('nan')
    pf_err = abs(cf_pf-Cf_exact)/Cf_exact*100 if math.isfinite(cf_pf) else float('nan')
    diff = abs(cf_mem-cf_pf) if math.isfinite(cf_mem) and math.isfinite(cf_pf) else float('nan')

    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()
    u_exact = np.array([u_top*(y-0.5)/H if 1<=y<ny-1 else 0 for y in range(ny)], dtype=np.float32)
    u_err = max(abs(u_prof[y]-u_exact[y])/max(abs(u_exact[y]),1e-10)*100 for y in range(1,ny-1)) if n_rec else float('nan')

    print(f"\n{tag} === FINAL === Cf_MEM={cf_mem:.6f}(err={mem_err:.2f}%) Cf_PF={cf_pf:.6f}(err={pf_err:.2f}%) "
          f"Cd_p={cf_p:.6f} |diff|={diff:.8f} u_err={u_err:.2f}% t={elapsed:.0f}s", flush=True)
    result = {"case":"couette_3d","device":f"sdaa:{device_id}","grid":f"{nx}x{ny}x{nz}",
              "tau":_float(tau),"nu":_float(nu),"u_top":_float(u_top),"H":H,"n_steps":n_steps,"n_samples":n_rec,
              "Cf_exact":_float(Cf_exact),"Cf_MEM":_float(cf_mem),"Cf_MEM_err_pct":_float(mem_err),
              "Cf_PF_friction":_float(cf_pf),"Cf_PF_err_pct":_float(pf_err),
              "Cd_PF_pressure":_float(cf_p),"MEM_PF_diff":_float(diff),
              "u_err_max_pct":_float(u_err),
              "u_profile":_float_list(u_prof.tolist()),"u_exact":_float_list(u_exact.tolist()),
              "elapsed_s":_float(elapsed)}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# BENCHMARK 3: Poiseuille flow
# ---------------------------------------------------------------------------
def run_poiseuille(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 80, 12, 12; tau = 1.0
    nu = (tau - 0.5) / 3.0; u_max_target = 0.05
    n_steps = 3000; warmup = 500
    H_full = ny - 2; H_half = H_full / 2.0
    G = 2.0 * nu * u_max_target / (H_half ** 2)
    u_max_exact = G * H_half ** 2 / (2.0 * nu)

    tag = f"[PoiseuilleMEMvsPF SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.6f} G={G:.6e} u_max_exact={u_max_exact:.6f}", flush=True)
    t0 = time.time()

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    A_frontal = H_full * nz; dpS = 0.5 * 1.0 * u_max_target ** 2 * A_frontal
    V_fluid = nx * (ny - 2) * nz; cd_body = G * V_fluid / dpS

    rho0 = torch.ones((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    mem_cd_hist = []; pf_f_hist = []; pf_p_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        # Collision with proper Guo body force (velocity shift: u* = u + tau*F/rho)
        f = collide_bgk3d_guo(f, tau=tau, Fx=G)
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # MEM: BEFORE streaming (both walls, stationary)
        if step > warmup:
            fx_mem, _, _ = momentum_exchange_standard(f, solid, near)
            cd_mem = fx_mem / dpS
            if math.isfinite(cd_mem):
                mem_cd_hist.append(cd_mem)

        f = stream3d(f)
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # P+F: AFTER streaming
        if step > warmup:
            cdp_x, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid, p0_method='far_field')
            cdf_x, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            if math.isfinite(cdf_x):
                pf_f_hist.append(cdf_x); pf_p_hist.append(cdp_x)

        if step % 500 == 0:
            n_rec = len(mem_cd_hist)
            mem_avg = sum(mem_cd_hist)/n_rec if n_rec else float('nan')
            pf_avg = sum(pf_f_hist)/n_rec if n_rec else float('nan')
            _, ux, _, _ = macroscopic3d(f)
            print(f"{tag} step={step} Cd_MEM={mem_avg:.6f} Cd_PF_f={pf_avg:.6f} "
                  f"(Cd_body={cd_body:.6f}) u_max={float(ux.mean(dim=(0,2))[ny//2]):.6f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_rec = len(mem_cd_hist)
    cd_mem = sum(mem_cd_hist)/n_rec if n_rec else float('nan')
    cd_pf_f = sum(pf_f_hist)/n_rec if n_rec else float('nan')
    cd_pf_p = sum(pf_p_hist)/n_rec if n_rec else float('nan')
    mem_err = abs(cd_mem-cd_body)/abs(cd_body)*100 if math.isfinite(cd_mem) else float('nan')
    pf_err = abs(cd_pf_f-cd_body)/abs(cd_body)*100 if math.isfinite(cd_pf_f) else float('nan')
    diff = abs(cd_mem-cd_pf_f) if math.isfinite(cd_mem) and math.isfinite(cd_pf_f) else float('nan')

    _, ux, _, _ = macroscopic3d(f)
    u_prof = ux.mean(dim=(0, 2)).cpu().numpy()
    u_exact = np.array([G/(2*nu)*(y-0.5)*(H_full-(y-0.5)) if 1<=y<ny-1 else 0 for y in range(ny)], dtype=np.float32)
    u_err = max(abs(u_prof[y]-u_exact[y])/max(abs(u_exact[y]),1e-10)*100 for y in range(1,ny-1)) if n_rec else float('nan')
    u_max_sim = float(u_prof[ny//2]); u_max_err = abs(u_max_sim-u_max_exact)/u_max_exact*100

    print(f"\n{tag} === FINAL === u_max={u_max_sim:.6f}(err={u_max_err:.2f}%) "
          f"Cd_MEM={cd_mem:.6f}(err={mem_err:.2f}%) Cd_PF_f={cd_pf_f:.6f}(err={pf_err:.2f}%) "
          f"Cd_p={cd_pf_p:.6f} |diff|={diff:.8f} t={elapsed:.0f}s", flush=True)
    result = {"case":"poiseuille_3d","device":f"sdaa:{device_id}","grid":f"{nx}x{ny}x{nz}",
              "tau":_float(tau),"nu":_float(nu),"G":_float(G),"u_max_target":_float(u_max_target),"u_max_exact":_float(u_max_exact),
              "H_full":H_full,"n_steps":n_steps,"n_samples":n_rec,
              "Cd_body":_float(cd_body),"Cd_MEM":_float(cd_mem),"Cd_MEM_err_pct":_float(mem_err),
              "Cd_PF_friction":_float(cd_pf_f),"Cd_PF_err_pct":_float(pf_err),
              "Cd_PF_pressure":_float(cd_pf_p),"MEM_PF_diff":_float(diff),
              "u_max_sim":_float(u_max_sim),"u_max_err_pct":_float(u_max_err),"u_err_max_pct":_float(u_err),
              "u_profile":_float_list(u_prof.tolist()),"u_exact":_float_list(u_exact.tolist()),
              "elapsed_s":_float(elapsed)}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
# BENCHMARK 4: SUBOFF Re=1000
# ---------------------------------------------------------------------------
def run_suboff(device_id, output_path):
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    Re = 1000; nx, ny, nz = 200, 80, 80; u_in = 0.06
    hull_length = 80; nu_lat = u_in * hull_length / Re; tau = 3.0 * nu_lat + 0.5
    n_steps = 5000; warmup = 1000
    cx = nx * 0.30; cy = ny / 2.0; cz = nz / 2.0
    config = SuboffConfig(); radius = config.r_over_l * hull_length
    D = 2.0 * radius

    tag = f"[SuboffMEMvsPF SDAA:{device_id}]"
    print(f"{tag} Re={Re} nx={nx} ny={ny} nz={nz} u_in={u_in} tau={tau:.4f} L={hull_length} R={radius:.2f}", flush=True)
    t0 = time.time()

    solid, _ = build_suboff_mask(hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
                                  cx=cx, cy=cy, cz=cz, length=hull_length,
                                  radius=radius, config=config, device=device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, hull_length, radius, config=config)
    S_wet = math.pi * D * hull_length; dpS = 0.5 * 1.0 * u_in ** 2 * S_wet
    print(f"{tag} solid={int(solid.sum())} near={int(near.sum())} S_wet={S_wet:.1f} dpS={dpS:.4f}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    mem_cd_hist = []; pf_p_hist = []; pf_f_hist = []; pf_tot_hist = []

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        sm = solid.unsqueeze(0).expand_as(f)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)

        # MEM: BEFORE streaming
        if step > warmup and step % 50 == 0:
            fx_mem, _, _ = momentum_exchange_standard(f, solid, near)
            cd_mem = fx_mem / dpS
            if math.isfinite(cd_mem):
                mem_cd_hist.append(cd_mem)

        f = stream3d(f)
        f = far_field_bc_3d(f, u_in)
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # P+F: AFTER streaming
        if step > warmup and step % 50 == 0:
            fx_p, _, _ = drag_pressure_integration(f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
            fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu_lat)
            if math.isfinite(fx_p + fx_f):
                pf_p_hist.append(fx_p); pf_f_hist.append(fx_f); pf_tot_hist.append(fx_p + fx_f)

        if step % 500 == 0 or step == n_steps:
            n_rec = len(mem_cd_hist)
            if n_rec:
                print(f"{tag} step={step} Cd_MEM={sum(mem_cd_hist)/n_rec:.6f} "
                      f"Cd_PF={sum(pf_tot_hist)/n_rec:.6f} (ref=0.042) ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"{tag} step={step} (warmup) ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0; Cd_ref = 0.042
    n_rec = len(mem_cd_hist)
    cd_mem = sum(mem_cd_hist)/n_rec if n_rec else float('nan')
    cd_p = sum(pf_p_hist)/n_rec if n_rec else float('nan')
    cd_f = sum(pf_f_hist)/n_rec if n_rec else float('nan')
    cd_pf = sum(pf_tot_hist)/n_rec if n_rec else float('nan')
    mem_err = abs(cd_mem-Cd_ref)/Cd_ref*100 if math.isfinite(cd_mem) else float('nan')
    pf_err = abs(cd_pf-Cd_ref)/Cd_ref*100 if math.isfinite(cd_pf) else float('nan')
    diff = abs(cd_mem-cd_pf) if math.isfinite(cd_mem) and math.isfinite(cd_pf) else float('nan')

    print(f"\n{tag} === FINAL === Cd_MEM={cd_mem:.6f}(err={mem_err:.1f}%) Cd_P={cd_p:.6f} "
          f"Cd_F={cd_f:.6f} Cd_PF={cd_pf:.6f}(err={pf_err:.1f}%) |diff|={diff:.6f} t={elapsed:.0f}s", flush=True)
    result = {"case":"suboff_Re1000","device":f"sdaa:{device_id}","Re":Re,
              "nx":nx,"ny":ny,"nz":nz,"L":hull_length,"R":_float(radius),"D":_float(D),
              "u_in":_float(u_in),"tau":_float(tau),"nu":_float(nu_lat),"S_wet":_float(S_wet),
              "n_steps":n_steps,"n_samples":n_rec,"Cd_ref":Cd_ref,
              "Cd_MEM":_float(cd_mem),"Cd_MEM_err_pct":_float(mem_err),
              "Cd_pressure":_float(cd_p),"Cd_friction":_float(cd_f),
              "Cd_PF_total":_float(cd_pf),"Cd_PF_err_pct":_float(pf_err),
              "MEM_PF_diff":_float(diff),"elapsed_s":_float(elapsed)}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} saved {output_path}", flush=True)
    return result


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "cylinder"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    output_path = sys.argv[3] if len(sys.argv) > 3 else f"mem_vs_pf_{case}_sdaa{device_id}.json"
    {"cylinder":run_cylinder,"couette":run_couette,"poiseuille":run_poiseuille,
     "suboff":run_suboff}[case](device_id, output_path)
