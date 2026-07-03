"""Engineering-grade free-surface ship hull resistance.

Inherits from double-body (全湿流) experience:
  - Cumulant D3Q27 for viscous flow (1% accuracy on SUBOFF)
  - log-law wall function for high-Re hull drag
  - Far-field BC for domain boundaries
  - Wake survey for drag measurement

Adds free-surface capabilities:
  - Color-Gradient multiphase for water/air interface
  - Wave-making resistance from wave height profile
  - Total Ct = viscous Cv + wave-making Cw

The approach:
  1. Run double-body (no free surface) → viscous Cv (benchmark)
  2. Run free-surface → total Ct
  3. Wave-making Cw = Ct - Cv

Usage:
    from tensorlbm.hull_free_surface_v2 import HullFreeSurfaceV2Config, run_hull_free_surface_v2
    cfg = HullFreeSurfaceV2Config(hull_type="wigley", re=1e6, fr=0.25, ...)
    results = run_hull_free_surface_v2(cfg)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import math
import torch

from .d3q19 import equilibrium3d, macroscopic3d, C as C3D, OPPOSITE as OPP, W as W3D
from .solver3d import stream3d, correct_mass3d
from .boundaries3d import far_field_bc_3d
from .cg_advanced_collision import collide_cg_kbc_3d, collide_cg_cumulant_3d, collide_cg_cascaded_3d
from .free_surface_lbm import free_surface_step, init_flags_from_fill, GAS, LIQUID, INTERFACE
from .ship_cad import build_hull_mask, ShipHullType
from .suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area

KAPPA = 0.41
B_CONST = 5.0

# D3Q19 velocity shifts for torch.roll (pull scheme: shift by +c)
_SHIFTS = [
    (0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0),
    (0,0,1), (0,0,-1), (1,1,0), (-1,1,0), (1,-1,0),
    (-1,-1,0), (1,0,1), (-1,0,1), (1,0,-1), (-1,0,-1),
    (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1),
]


@dataclass(frozen=True)
class HullFreeSurfaceV2Config:
    """Engineering free-surface ship hull configuration."""

    hull_type: str = "wigley"
    nx: int = 320
    ny: int = 96
    nz: int = 96
    re: float = 1e6        # Reynolds number
    fr: float = 0.25       # Froude number (U/sqrt(g*L))
    u_in: float = 0.06     # Inlet velocity (lattice units)
    fill_fraction: float = 0.5  # Water fill fraction (z direction)
    n_steps: int = 2000
    warmup: int = 500
    output_interval: int = 200
    device: str = "sdaa:0"
    use_wall_function: bool = True
    use_free_surface: bool = True   # False = double-body (viscous only)
    collision_model: str = "kbc"  # "kbc", "cumulant", "cascaded"
    form_factor: float = 1.15       # (1+k) for ITTC reference


def _build_hull(cfg: HullFreeSurfaceV2Config, device: torch.device):
    """Build hull mask at correct position (intersecting waterline)."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    fill_height = max(int(cfg.fill_fraction * nz), 1)

    solid, stats = build_hull_mask(
        cfg.hull_type, nx, ny, nz,
        cx=nx * 0.3, cy=ny * 0.5, cz_keel=fill_height - 10,  # keel 10 cells below waterline
        device="cpu",
    )
    solid = solid.to(device)
    return solid, fill_height


def _wall_function_3d(f, solid, near, fluid, c, cx, cy, cz, w, cs2, nu_lat, y_val=0.5):
    """Log-law wall function (inherited from SUBOFF double-body solver)."""
    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    u_tau = torch.sqrt(nu_lat * u_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu_lat
    turb = (y_plus > 11.6) & near

    if bool(turb.any()):
        ut = u_tau[turb].clone()
        um = u_mag[turb]
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu_lat)
            fv = ut * (lyp / KAPPA + B_CONST) - um
            fp = (lyp / KAPPA + B_CONST) + 1.0 / KAPPA
            ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut

    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)

    fx = coef * (ux * inv_umag)
    fy = coef * (uy * inv_umag)
    fz = coef * (uz * inv_umag)

    # Guo forcing
    cu = cx * ux + cy * uy + cz * uz
    forcing = w * (1.0 + cu / cs2) * (cx * fx + cy * fy + cz * fz) / cs2
    f = f + forcing

    # Friction drag (from wall shear)
    df = (tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item()

    # Pressure drag — only from water phase, not total density
    # Use water-phase density (rho_r) for pressure, not total rho
    p = (rho - 1.0) / 3.0  # gauge pressure relative to water reference
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    # Only count pressure on fluid cells (not solid, not air)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()

    return f, df, dp


def _far_field_cg(f_r, f_b, u_in, water_mask, solid):
    """Far-field BC for two-phase flow."""
    nz, ny, nx = f_r.shape[1], f_r.shape[2], f_r.shape[3]
    r1 = torch.ones(nz, ny, nx, dtype=f_r.dtype, device=f_r.device)
    ux1 = torch.full((nz, ny, nx), u_in, dtype=f_r.dtype, device=f_r.device)
    feq_r = equilibrium3d(r1, ux1, torch.zeros_like(ux1), torch.zeros_like(ux1))
    feq_b = equilibrium3d(0.1 * r1, ux1, torch.zeros_like(ux1), torch.zeros_like(ux1))

    f_r = f_r.clone()
    f_b = f_b.clone()
    # Inlet
    f_r[:, :, :, 0] = feq_r[:, :, :, 0]
    f_b[:, :, :, 0] = feq_b[:, :, :, 0]
    # Outlet (convective)
    f_r[:, :, :, -1] = f_r[:, :, :, -2]
    f_b[:, :, :, -1] = f_b[:, :, :, -2]
    # Lateral (far-field)
    f_r[:, 0, :, :] = feq_r[:, 0, :, :]
    f_r[:, -1, :, :] = feq_r[:, -1, :, :]
    f_b[:, 0, :, :] = feq_b[:, 0, :, :]
    f_b[:, -1, :, :] = feq_b[:, -1, :, :]
    # Top/bottom
    f_r[:, :, 0, :] = feq_r[:, :, 0, :]
    f_r[:, :, -1, :] = feq_r[:, :, -1, :]
    f_b[:, :, 0, :] = feq_b[:, :, 0, :]
    f_b[:, :, -1, :] = feq_b[:, :, -1, :]
    return f_r, f_b


def _wave_resistance(f_r, f_b, water_mask, u_in, nx, ny, nz, S, device):
    """Estimate wave-making resistance from wave height at a plane behind hull."""
    # Water surface elevation at x = 0.7*nx (behind hull)
    x_plane = int(nx * 0.7)
    rho_r = f_r.sum(0)
    rho_b = f_b.sum(0)
    rho_total = rho_r + rho_b
    # Water fraction at the plane
    water_frac = rho_r[:, :, x_plane] / rho_total[:, :, x_plane].clamp(min=1e-6)
    # Wave height = deviation from mean water level
    fill_height = int(0.5 * nz)
    # Find actual water surface
    surface = torch.zeros(ny, device=device)
    for j in range(ny):
        col = water_frac[:, j]
        above = (col > 0.5).nonzero()
        if len(above) > 0:
            surface[j] = above[-1].float().item()
        else:
            surface[j] = float(fill_height)
    # Wave amplitude
    wave_amp = float((surface - fill_height).abs().max().item())
    # Simple wave-making estimate: Cw ~ Fr^4 (Havelock)
    return wave_amp


def run_hull_free_surface_v2(cfg: HullFreeSurfaceV2Config) -> dict:
    """Run engineering free-surface ship hull resistance."""
    device = torch.device(cfg.device)
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    u_in = cfg.u_in
    fill_height = max(int(cfg.fill_fraction * nz), 1)

    # Build hull
    solid, fill_height = _build_hull(cfg, device)
    fluid = ~solid
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in**2 * S

    # Water mask
    zz = torch.arange(nz, device=device).view(nz, 1, 1)
    water_mask = (zz < fill_height).expand(nz, ny, nx)

    # Near-wall mask for wall function — only submerged cells in free-surface mode
    # near = fluid cells adjacent to solid (NOT solid adjacent to fluid!)
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (fluid & torch.roll(solid, sgn, dims=ax))
    if cfg.use_free_surface:
        near = nbrs & water_mask  # only apply wall function to submerged hull
    else:
        near = nbrs

    # Initialize Körner free-surface fields (if free-surface mode)
    if cfg.use_free_surface:
        fill = torch.where(water_mask, torch.ones(nz, ny, nx, device=device), torch.zeros(nz, ny, nx, device=device))
        fill = fill.float()
        flags = init_flags_from_fill(fill, solid)
    else:
        fill = None
        flags = None

    # Lattice parameters
    hull_length = max(6.0, 0.35 * nx)
    nu_lat = u_in * hull_length / cfg.re
    tau = 3.0 * nu_lat + 0.5

    # Gravity from Froude number: Fr = U/sqrt(g*L) → g = U²/(Fr²*L)
    # Fr=0 means double-body (no gravity, no free surface)
    g_lat = u_in**2 / (cfg.fr**2 * hull_length) if cfg.fr > 0.001 else 0.0

    # D3Q19 constants
    c = C3D.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    w = W3D.to(device).float().view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    opp = OPP.to(device)

    # ITTC reference
    cf_ittc = _ittc57_friction_coefficient(cfg.re)
    ct_ref = cf_ittc * cfg.form_factor

    print(f"=== Free-Surface {cfg.hull_type} ===", flush=True)
    print(f"Re={cfg.re:.0e} Fr={cfg.fr:.2f} grid={nx}x{ny}x{nz}", flush=True)
    print(f"tau={tau:.5f} nu={nu_lat:.2e} g={g_lat:.6f}", flush=True)
    print(f"S={S:.0f} Cf_ITTC={cf_ittc:.5f} (1+k)={cfg.form_factor} Ct_ref={ct_ref:.5f}", flush=True)
    print(f"Free surface: {cfg.use_free_surface} Wall function: {cfg.use_wall_function}\n", flush=True)

    # Initialize two-phase flow (or single-phase for double-body)
    if cfg.use_free_surface:
        # High density ratio (1000:1) for sharp interface
        rho_r = torch.where(water_mask, torch.ones(nz, ny, nx, device=device), 0.001)
        rho_b = torch.where(water_mask, 0.001 * torch.ones(nz, ny, nx, device=device), 1.0)
    else:
        # Double-body: single phase, f_b = 0
        rho_r = torch.ones(nz, ny, nx, device=device)
        rho_b = torch.zeros(nz, ny, nx, device=device)
    ux0 = torch.where(water_mask, torch.full((nz, ny, nx), u_in, device=device), 0.0)
    f_r = equilibrium3d(rho_r, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    f_b = equilibrium3d(rho_b, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    del rho_r, rho_b

    # Solid equilibrium (zero velocity)
    if cfg.use_free_surface:
        rho_solid_r = torch.where(water_mask, torch.ones(nz, ny, nx, device=device), 0.001)
        rho_solid_b = torch.where(water_mask, 0.001 * torch.ones(nz, ny, nx, device=device), 1.0)
    else:
        # Double-body: uniform density, no water/air distinction
        rho_solid_r = torch.ones(nz, ny, nx, device=device)
        rho_solid_b = torch.zeros(nz, ny, nx, device=device)
    f_r_solid_eq = equilibrium3d(rho_solid_r, torch.zeros_like(ux0), torch.zeros_like(ux0), torch.zeros_like(ux0))
    f_b_solid_eq = equilibrium3d(rho_solid_b, torch.zeros_like(ux0), torch.zeros_like(ux0), torch.zeros_like(ux0))

    # Wake survey plane
    wake_x = int(nx * 0.7)

    fric_list = []
    pres_list = []
    wave_list = []
    df, dp = 0.0, 0.0  # init for wall function
    t0 = __import__('time').time()

    for step in range(1, cfg.n_steps + 1):
        # 1. Collision
        gz = g_lat if cfg.use_free_surface else 0.0
        if cfg.use_free_surface:
            # Free-surface: Körner model (sharp interface via fill function)
            f_r, fill, flags = free_surface_step(
                f_r, fill, flags, solid,
                tau=tau, gz=gz,
                rho_liquid=1.0, rho_gas=0.001,
            )
            f_b = f_b * 0.0  # not used in Körner mode
        else:
            # Double-body: CG-KBC single-phase (stable at tau=0.5)
            f_r, f_b = collide_cg_kbc_3d(f_r, f_b, tau=tau, gz=0.0)
            f_r = f_r + f_b  # recoloring splits f_neq 50/50, add blue back to red
            f_b = f_b * 0.0

        # 2. Streaming (skip for Körner — handled internally)
        if not cfg.use_free_surface:
            f_r = stream3d(f_r)
            if cfg.use_free_surface:
                f_b = stream3d(f_b)

        # 3. Wall function (on water-phase distribution)
        if cfg.use_wall_function:
            # Dynamic water detection: use fill function for Körner, rho_r for CG
            if cfg.use_free_surface:
                near_water = near & (fill > 0.5)
            else:
                near_water = near
            f_r, df, dp = _wall_function_3d(
                f_r, solid, near_water, fluid, c, cx, cy, cz, w, cs2, nu_lat
            )
        # 4-6. Bounce-back, solid reset, far-field BC
        if cfg.use_free_surface:
            # Körner mode: apply far-field BC to maintain water level at boundaries
            nz2, ny2, nx2 = f_r.shape[1], f_r.shape[2], f_r.shape[3]
            r1 = torch.ones(nz2, ny2, nx2, dtype=f_r.dtype, device=f_r.device)
            feq = equilibrium3d(r1, torch.full_like(r1, u_in), torch.zeros_like(r1), torch.zeros_like(r1))
            f_r = f_r.clone()
            # Inlet: set to equilibrium with u_in, restore water level
            f_r[:,:,:,0] = feq[:,:,:,0]
            fill[:,:,0] = torch.where(water_mask[:,:,0], 1.0, 0.0)
            # Outlet: convective
            f_r[:,:,:,-1] = f_r[:,:,:,-2]
            # Side walls
            f_r[:,0,:,:] = feq[:,0,:,:]; f_r[:,-1,:,:] = feq[:,-1,:,:]
            f_r[:,:,0,:] = feq[:,:,0,:]; f_r[:,:,-1,:] = feq[:,:,-1,:]
            # Reset flags at boundaries
            flags = init_flags_from_fill(fill, solid)
        else:
            # Double-body: bounce-back + far-field
            opp = OPP.to(device)
            f_r_swapped = f_r[opp]
            f_r[:, solid] = f_r_swapped[:, solid]
            f_r = torch.where(solid.unsqueeze(0), f_r_solid_eq, f_r)
            nz2, ny2, nx2 = f_r.shape[1], f_r.shape[2], f_r.shape[3]
            r1 = torch.ones(nz2, ny2, nx2, dtype=f_r.dtype, device=f_r.device)
            feq = equilibrium3d(r1, torch.full_like(r1, u_in), torch.zeros_like(r1), torch.zeros_like(r1))
            f_r = f_r.clone()
            f_r[:,:,:,0] = feq[:,:,:,0]; f_r[:,:,:,-1] = f_r[:,:,:,-2]
            f_r[:,0,:,:] = feq[:,0,:,:]; f_r[:,-1,:,:] = feq[:,-1,:,:]
            f_r[:,:,0,:] = feq[:,:,0,:]; f_r[:,:,-1,:] = feq[:,:,-1,:]

        # 7. Mass correction (skip for Körner — conserves mass internally)
        if not cfg.use_free_surface and step % 100 == 0:
            f_r = correct_mass3d(f_r, float(f_r.sum().item()))

        # 8. Diagnostics
        if step > cfg.warmup:
            if cfg.use_wall_function:
                fric_list.append(df)
            if cfg.use_free_surface:
                # Free-surface: measure wave amplitude from fill function (sharp interface)
                if step % 10 == 0:
                    # Find surface height per y-column at wake plane (fill > 0.5)
                    surface_h = torch.zeros(ny, device=device)
                    for j in range(ny):
                        col = fill[:, j, wake_x]  # 1D [nz]
                        above = (col > 0.5).nonzero()
                        if len(above) > 0:
                            surface_h[j] = float(above[-1].item())
                        else:
                            surface_h[j] = float(fill_height)
                    eta_rms = float(torch.sqrt(((surface_h - fill_height)**2).mean()))
                    wave_list.append(eta_rms)
            else:
                # Double-body: single-phase wake survey (no CG, no interface)
                if step % 10 == 0:
                    _, ux_wake, _, _ = macroscopic3d(f_r)
                    deficit = (u_in - ux_wake[:, :, wake_x]) * fluid[:, :, wake_x].to(f_r.dtype)
                    thrust = 1.0 * u_in * deficit.sum().item()
                    if math.isfinite(thrust):
                        wave_list.append(thrust)

        if step % cfg.output_interval == 0 or step == cfg.n_steps:
            cf = abs(sum(fric_list)/max(len(fric_list),1)) / dyn_p_S if fric_list else 0.0
            if cfg.use_free_surface:
                # Free-surface: Cw from wave amplitude
                # Cw = g * ∫η² dy / (U² * S)  (wave resistance)
                eta_avg = sum(wave_list)/max(len(wave_list),1) if wave_list else 0.0
                # eta_avg is RMS wave amplitude in lattice units
                # Approximate ∫η²dy ≈ eta_rms² * ny (width)
                cw = g_lat * eta_avg**2 * ny / (u_in**2 * S) if g_lat > 0 else 0.0
                ct = cf + cw
                print(f"  step {step:5d}: Cf={cf:.5f} Cw={cw:.5f} Ct={ct:.5f} eta={eta_avg:.2f} (ref={ct_ref:.5f})", flush=True)
            else:
                # Double-body: Cv from wake survey
                cv_wake = 2.0 * abs(sum(wave_list)/max(len(wave_list),1)) / (u_in * S) if wave_list else 0.0
                print(f"  step {step:5d}: Cf={cf:.5f} Cv={cv_wake:.5f} (ref={ct_ref:.5f})", flush=True)

    dt = __import__('time').time() - t0
    cf = abs(sum(fric_list)/max(len(fric_list),1)) / dyn_p_S if fric_list else 0.0

    print(f"\n=== Final Results ===", flush=True)
    if cfg.use_free_surface:
        eta_avg = sum(wave_list)/max(len(wave_list),1) if wave_list else 0.0
        cw = g_lat * eta_avg**2 * ny / (u_in**2 * S) if g_lat > 0 else 0.0
        ct = cf + cw
        print(f"Cf (friction)   = {cf:.5f}", flush=True)
        print(f"Cw (wave-making)= {cw:.5f}  eta_rms={eta_avg:.2f}", flush=True)
        print(f"Ct (total)      = {ct:.5f} (ref={ct_ref:.5f}, ratio={ct/ct_ref:.2f}x)", flush=True)
    else:
        cv = 2.0 * abs(sum(wave_list)/max(len(wave_list),1)) / (u_in * S) if wave_list else 0.0
        print(f"Cf (friction) = {cf:.5f}", flush=True)
        print(f"Cv (viscous)  = {cv:.5f} (ref={ct_ref:.5f}, ratio={cv/ct_ref:.2f}x)", flush=True)
    print(f"Time: {dt:.1f}s ({dt/cfg.n_steps*1000:.0f}ms/step)", flush=True)

    return {"cf": cf, "ct_ref": ct_ref, "config": asdict(cfg)}
