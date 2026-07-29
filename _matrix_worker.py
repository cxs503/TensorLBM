"""Agent Matrix worker — single universal worker for all 8 config combinations.

Handles: D3Q19/D3Q27 × MRT/KBC/CASCADED/CUMULANT × log/gradient wall-law
         × SUBOFF/Cylinder/Sphere geometries.

Usage:
    python _matrix_worker.py <did> <flow> <lattice> <collision> <wall_law> <cs> <nx> <ny> <nz> <geom> <n_steps> <out_path>
"""
from __future__ import annotations

import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

KAPPA = 0.41
B_CONST = 5.0
SLIDE = 300

# ──────────────────────────────────────────────────────────────────
# D3Q19 wall functions
# ──────────────────────────────────────────────────────────────────
def _wallfn19_log(f, solid, nu, y_val=0.5):
    """D3Q19 log-law wall function (Guo body force)."""
    from tensorlbm.d3q19 import macroscopic3d, C as C19

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu
    turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone()
        vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly/KAPPA + B_CONST) - vm
            fp = (ly/KAPPA + B_CONST) + 1.0/KAPPA
            uu = (uu - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut; ium = 1.0/um
    coef = -(tw/y_val) * near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)

    device = f.device
    c19 = C19.to(device).float()
    cx19 = c19[:,0].view(19,1,1,1); cy19 = c19[:,1].view(19,1,1,1); cz19 = c19[:,2].view(19,1,1,1)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12, dtype=f.dtype, device=device).view(19,1,1,1)
    cs2 = 1.0/3.0
    cu = cx19*ux + cy19*uy + cz19*uz
    forcing = w19*(1.0+cu/cs2)*(cx19*fx + cy19*fy + cz19*fz)/cs2
    f = f + forcing
    df = (tw*(ux*ium)*near.to(f.dtype)).sum().item()
    p = (rho-1.0)/3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p*(sp.to(f.dtype)-sm.to(f.dtype))*fluid.to(f.dtype)).sum().item()
    return f, df, dp


def _wallfn19_gradient(f, solid, nu, y_val=0.5):
    """D3Q19 gradient wall-law (τ_w = 2·ν·u/y_val, no log-law)."""
    from tensorlbm.d3q19 import macroscopic3d, C as C19

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    tw = 2.0 * nu * um / y_val  # direct gradient
    ium = 1.0/um
    coef = -(tw/y_val) * near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)

    device = f.device
    c19 = C19.to(device).float()
    cx19 = c19[:,0].view(19,1,1,1); cy19 = c19[:,1].view(19,1,1,1); cz19 = c19[:,2].view(19,1,1,1)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12, dtype=f.dtype, device=device).view(19,1,1,1)
    cs2 = 1.0/3.0
    cu = cx19*ux + cy19*uy + cz19*uz
    forcing = w19*(1.0+cu/cs2)*(cx19*fx + cy19*fy + cz19*fz)/cs2
    f = f + forcing
    df = (tw*(ux*ium)*near.to(f.dtype)).sum().item()
    p = (rho-1.0)/3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p*(sp.to(f.dtype)-sm.to(f.dtype))*fluid.to(f.dtype)).sum().item()
    return f, df, dp


# ──────────────────────────────────────────────────────────────────
# D3Q27 streaming & far-field BC (self-contained)
# ──────────────────────────────────────────────────────────────────
_D3Q27_SHIFTS = [(cx,cy,cz) for cz in [-1,0,1] for cy in [-1,0,1] for cx in [-1,0,1]]

def _stream27_roll(f):
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0,1,2))
    return out

def _far_field_bc_27(f, u_in=0.06):
    from tensorlbm.d3q27 import equilibrium27
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(rho1, torch.full_like(rho1, u_in),
                        torch.zeros_like(rho1), torch.zeros_like(rho1))
    f = f.clone()
    f[:,:,:,0] = feq[:,:,:,0]; f[:,:,:,-1] = f[:,:,:,-2]
    f[:,0,:,:] = feq[:,0,:,:]; f[:,-1,:,:] = feq[:,-1,:,:]
    f[:,:,0,:] = feq[:,:,0,:]; f[:,:,-1,:] = feq[:,:,-1,:]
    return f

# ──────────────────────────────────────────────────────────────────
# D3Q27 wall functions
# ──────────────────────────────────────────────────────────────────
def _wallfn27_log(f, solid, nu, y_val=0.5):
    """D3Q27 log-law wall function."""
    from tensorlbm.d3q27 import macroscopic27, C as C27

    device = f.device
    c = C27.to(device).float()
    cx = c[:,0].view(27,1,1,1); cy = c[:,1].view(27,1,1,1); cz = c[:,2].view(27,1,1,1)
    w27 = torch.tensor([8/27]+[2/27]*6+[1/54]*12+[1/216]*8, dtype=f.dtype, device=device).view(27,1,1,1)

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic27(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu
    turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone(); vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu*(ly/KAPPA + B_CONST) - vm
            fp = (ly/KAPPA + B_CONST) + 1.0/KAPPA
            uu = (uu - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut*ut; ium = 1.0/um
    coef = -(tw/y_val)*near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)

    cs2 = 1.0/3.0
    cu = cx*ux + cy*uy + cz*uz
    forcing = w27*(1.0+cu/cs2)*(cx*fx + cy*fy + cz*fz)/cs2
    f = f + forcing
    df = (tw*(ux*ium)*near.to(f.dtype)).sum().item()
    p = (rho-1.0)/3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p*(sp.to(f.dtype)-sm.to(f.dtype))*fluid.to(f.dtype)).sum().item()
    return f, df, dp


def _wallfn27_gradient(f, solid, nu, y_val=0.5):
    """D3Q27 gradient wall-law (direct velocity gradient, no log-law)."""
    from tensorlbm.d3q27 import macroscopic27, C as C27

    device = f.device
    c = C27.to(device).float()
    cx = c[:,0].view(27,1,1,1); cy = c[:,1].view(27,1,1,1); cz = c[:,2].view(27,1,1,1)
    w27 = torch.tensor([8/27]+[2/27]*6+[1/54]*12+[1/216]*8, dtype=f.dtype, device=device).view(27,1,1,1)

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic27(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    tw = 2.0 * nu * um / y_val
    ium = 1.0/um
    coef = -(tw/y_val)*near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)

    cs2 = 1.0/3.0
    cu = cx*ux + cy*uy + cz*uz
    forcing = w27*(1.0+cu/cs2)*(cx*fx + cy*fy + cz*fz)/cs2
    f = f + forcing
    df = (tw*(ux*ium)*near.to(f.dtype)).sum().item()
    p = (rho-1.0)/3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p*(sp.to(f.dtype)-sm.to(f.dtype))*fluid.to(f.dtype)).sum().item()
    return f, df, dp


# ──────────────────────────────────────────────────────────────────
# Smagorinsky helpers (domain-averaged tau_eff for operators
# that don't support per-cell tau)
# ──────────────────────────────────────────────────────────────────
def _smag_tau_d3q19(f, tau, C_s):
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_3d(f_neq)
    tau_eff_per_cell = _smagorinsky_tau(tau, pi_norm, rho, C_s)
    tau_eff = float(tau_eff_per_cell.mean().item())
    return max(tau, min(tau_eff, tau*10.0))

def _smag_tau_d3q27(f, tau, C_s):
    from tensorlbm.d3q27 import equilibrium27, macroscopic27
    from tensorlbm.turbulence import _neq_stress_norm_27, _smagorinsky_tau
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff_per_cell = _smagorinsky_tau(tau, pi_norm, rho, C_s)
    tau_eff = float(tau_eff_per_cell.mean().item())
    return max(tau, min(tau_eff, tau*10.0))


# ──────────────────────────────────────────────────────────────────
# Collision dispatchers
# ──────────────────────────────────────────────────────────────────
def _collide_d3q19(f, tau, collision, Cs):
    if collision == "MRT":
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d
        return collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
    elif collision == "CASCADED":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        if Cs > 0:
            tau_eff = _smag_tau_d3q19(f, tau, Cs)
            return collide_cascaded_d3q19(f, tau_eff)
        else:
            return collide_cascaded_d3q19(f, tau)
    elif collision == "CUMULANT":
        from tensorlbm.cumulant import collide_cumulant_d3q19
        return collide_cumulant_d3q19(f, tau, C_s=Cs)
    else:
        raise ValueError(f"Unknown D3Q19 collision: {collision}")


def _collide_d3q27(f, tau, collision, Cs):
    if collision == "CUMULANT":
        if Cs > 0:
            from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27
            return collide_cumulant_smag_d3q27(f, tau, C_s=Cs)
        else:
            from tensorlbm.cumulant import collide_cumulant_d3q27
            return collide_cumulant_d3q27(f, tau)
    elif collision == "CASCADED":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        if Cs > 0:
            tau_eff = _smag_tau_d3q27(f, tau, Cs)
            return collide_cascaded_d3q27(f, tau_eff)
        else:
            return collide_cascaded_d3q27(f, tau)
    elif collision == "KBC":
        from tensorlbm.advanced_collision import collide_kbc_d3q27
        if Cs > 0:
            tau_eff = _smag_tau_d3q27(f, tau, Cs)
            return collide_kbc_d3q27(f, tau_eff, C_s=0.0, beta=0.99)
        else:
            return collide_kbc_d3q27(f, tau, C_s=0.0, beta=0.99)
    elif collision == "MRT":
        # D3Q27 MRT via KBC with beta=0 (pure BGK-like relaxation)
        from tensorlbm.advanced_collision import collide_kbc_d3q27
        if Cs > 0:
            tau_eff = _smag_tau_d3q27(f, tau, Cs)
            return collide_kbc_d3q27(f, tau_eff, C_s=0.0, beta=0.5)
        else:
            return collide_kbc_d3q27(f, tau, C_s=0.0, beta=0.5)
    else:
        raise ValueError(f"Unknown D3Q27 collision: {collision}")


# ──────────────────────────────────────────────────────────────────
# Geometry mask builders
# ──────────────────────────────────────────────────────────────────
def _build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx)**2 + (yy - cy)**2 <= radius**2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def _build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 <= radius**2


# ──────────────────────────────────────────────────────────────────
# Reference values
# ──────────────────────────────────────────────────────────────────
REF_CT_SUBOFF = 0.00405
REF_CD_CYLINDER = {200: 1.30}
REF_CD_SPHERE_1000 = 0.47


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    did          = int(sys.argv[1])
    flow         = sys.argv[2]      # suboff | cylinder | sphere
    lattice      = sys.argv[3]      # D3Q19 | D3Q27
    collision    = sys.argv[4]      # MRT | KBC | CASCADED | CUMULANT
    wall_law     = sys.argv[5]      # log | gradient
    Cs           = float(sys.argv[6])
    nx           = int(sys.argv[7])
    ny           = int(sys.argv[8])
    nz           = int(sys.argv[9])
    geom_param   = float(sys.argv[10])  # hull_length or diameter
    n_steps      = int(sys.argv[11])
    out_path     = sys.argv[12]

    # Reynolds number and inflow
    if flow == "suboff":
        u_in, re = 0.06, 2e6
        nu = u_in * geom_param / re
    elif flow == "cylinder":
        u_in, re = 0.08, 200.0
        nu = u_in * geom_param / re
    elif flow == "sphere":
        u_in, re = 0.08, 1000.0
        nu = u_in * geom_param / re
    else:
        raise ValueError(f"Unknown flow: {flow}")

    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    smag_tag = f" Cs={Cs}" if Cs > 0 else ""
    tag = f"[SDAA:{did}] {lattice} {collision}{smag_tag} {wall_law}-law {flow} {nx}x{ny}x{nz}"
    print(f"{tag} Re={re:.0f} u_in={u_in} nu={nu:.2e} tau={tau:.6f}", flush=True)

    t0 = time.time()

    # ── Build geometry ──────────────────────────────────────────
    if flow == "suboff":
        from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
        from tensorlbm.suboff_resistance import _voxel_wetted_area
        cx_h, cy_h, cz_h = nx*0.35, ny/2.0, nz/2.0
        solid, _ = build_suboff_mask(
            hull_type=SuboffHullType.BARE_HULL,
            nx=nx, ny=ny, nz=nz, cx=cx_h, cy=cy_h, cz=cz_h,
            length=geom_param, device=str(device),
        )
        S = _voxel_wetted_area(solid, 1.0)
        A_ref = S
    elif flow == "cylinder":
        radius = geom_param / 2.0
        cx_g = nx*0.25; cy_g = ny*0.5
        solid = _build_cylinder_mask(nx, ny, nz, cx_g, cy_g, radius, device)
        A_ref = geom_param * nz
    elif flow == "sphere":
        radius = geom_param / 2.0
        cx_g = nx*0.25; cy_g = ny*0.5; cz_g = nz*0.5
        solid = _build_sphere_mask(nx, ny, nz, cx_g, cy_g, cz_g, radius, device)
        A_ref = math.pi * radius**2
    else:
        raise ValueError(f"Unknown flow: {flow}")

    dyn_p = 0.5 * 1.0 * u_in**2 * A_ref
    print(f"{tag} A_ref={A_ref:.0f} dyn_p={dyn_p:.6e} solid_cells={solid.sum().item()} ({time.time()-t0:.1f}s)", flush=True)

    # ── Initialise ──────────────────────────────────────────────
    if lattice == "D3Q19":
        from tensorlbm.d3q19 import equilibrium3d
        from tensorlbm.solver3d import correct_mass3d, stream3d
        from tensorlbm.boundaries3d import far_field_bc_3d
        rho0 = torch.ones(nz, ny, nx, device=device)
        ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
        f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
        im = float(rho0.sum().item())
        stream_fn = stream3d
        ff_bc = far_field_bc_3d
        correct_mass = correct_mass3d
        collide_fn = _collide_d3q19
        wallfn = _wallfn19_log if wall_law == "log" else _wallfn19_gradient
    else:  # D3Q27
        from tensorlbm.d3q27 import equilibrium27, correct_mass27
        rho0 = torch.ones(nz, ny, nx, device=device)
        ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
        f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
        im = float(rho0.sum().item())
        stream_fn = _stream27_roll
        ff_bc = _far_field_bc_27
        correct_mass = correct_mass27
        collide_fn = _collide_d3q27
        wallfn = _wallfn27_log if wall_law == "log" else _wallfn27_gradient

    # ── Simulation loop ─────────────────────────────────────────
    warmup = n_steps // 3
    fric, pres = [], []
    all_finite = True
    final_step = 0

    for step in range(1, n_steps + 1):
        try:
            f = collide_fn(f, tau, collision, Cs)
        except Exception as e:
            print(f"{tag} COLLISION ERROR at {step}: {e}", flush=True)
            all_finite = False
            break

        f = stream_fn(f)
        f, df, dp = wallfn(f, solid, nu, y_val=0.5)
        f = ff_bc(f, u_in=u_in) if callable(ff_bc) else ff_bc(f, u_in=u_in)

        if step % 100 == 0:
            f = correct_mass(f, im)

        if step > warmup and math.isfinite(df) and math.isfinite(dp):
            fric.append(df); pres.append(dp)
        final_step = step

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True)
            all_finite = False
            break

        if step % 500 == 0 or step == n_steps:
            wf = fric[-SLIDE:] if len(fric) >= SLIDE else fric
            wp = pres[-SLIDE:] if len(pres) >= SLIDE else pres
            cf = sum(wf)/max(len(wf),1)/dyn_p if wf else 0
            cp = sum(wp)/max(len(wp),1)/dyn_p if wp else 0
            ct = cf + cp
            elap = time.time()-t0
            print(f"{tag} step={step} Ct={ct:.5f} (Cf={cf:.5f} Cp={cp:.5f}) [{elap:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # ── Final statistics ────────────────────────────────────────
    # Full average
    cf_full = sum(fric)/max(len(fric),1)/dyn_p if fric else 0
    cp_full = sum(pres)/max(len(pres),1)/dyn_p if pres else 0
    ct_full = cf_full + cp_full

    # Sliding window
    wf = fric[-SLIDE:] if len(fric) >= SLIDE else fric
    wp = pres[-SLIDE:] if len(pres) >= SLIDE else pres
    cf_slide = sum(wf)/max(len(wf),1)/dyn_p if wf else 0
    cp_slide = sum(wp)/max(len(wp),1)/dyn_p if wp else 0
    ct_slide = cf_slide + cp_slide

    # Error vs reference
    ref: float; ref_lbl: str
    if flow == "suboff":
        ref = REF_CT_SUBOFF; ref_lbl = "SUBOFF AFF-8"
    elif flow == "cylinder":
        ref = REF_CD_CYLINDER.get(int(re), float("nan")); ref_lbl = f"cylinder Re={int(re)}"
    elif flow == "sphere":
        ref = REF_CD_SPHERE_1000; ref_lbl = f"sphere Re={int(re)}"
    else:
        ref = float("nan"); ref_lbl = "unknown"

    err_pct = abs(ct_full - ref)/ref*100 if ref and ref > 0 and math.isfinite(ct_full) else float("nan")

    # Std of sliding window
    if len(wf) > 1:
        ct_std = math.sqrt(sum(((f_i+p_i)/dyn_p - ct_slide)**2 for f_i, p_i in zip(wf, wp))/(len(wf)-1))
    else:
        ct_std = 0.0

    result = {
        "config_id": did,
        "flow": flow,
        "lattice": lattice,
        "collision": collision,
        "wall_law": wall_law,
        "Cs": Cs,
        "grid": f"{nx}x{ny}x{nz}",
        "geom_param": geom_param,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "sliding_window": SLIDE,
        "A_ref": A_ref,
        "Ct_fric_full": cf_full,
        "Ct_pres_full": cp_full,
        "Ct_total_full": ct_full,
        "Ct_fric_slide": cf_slide,
        "Ct_pres_slide": cp_slide,
        "Ct_total_slide": ct_slide,
        "Ct_std_slide": ct_std,
        "ref_value": ref,
        "ref_label": ref_lbl,
        "error_pct": err_pct,
        "samples": len(fric),
        "finite": all_finite,
        "elapsed_s": elapsed,
        "device": f"sdaa:{did}",
    }

    emoji = "✓" if all_finite else "✗"
    print(f"{tag} DONE {emoji} Ct={ct_full:.5f} Ct_slide={ct_slide:.5f}±{ct_std:.5f} "
          f"err={err_pct:.1f}% ({elapsed:.0f}s)", flush=True)

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} wrote result to {out_path}", flush=True)


if __name__ == "__main__":
    main()
