#!/usr/bin/env python3
"""Hybrid wall-function benchmark — single worker for one SDAA card.

Usage: python3 hybrid_worker.py <sdda_id> <case_name>

Cases: cylinder_Re200, square_prism_Re22000, sphere_Re1000, naca0012_Re6e6, suboff_Re2e6
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d, sphere_mask
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_function_common import _near_wall_mask, compute_u_tau, wall_function

def make_cylinder_mask(nx, ny, nz, cx, cy, cz, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32), indexing='ij')
    return (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2

def make_square_prism_mask(nx, ny, nz, cx, cy, cz, side, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32), indexing='ij')
    half = side / 2.0
    return (yy >= cy - half) & (yy <= cy + half) & (zz >= cz - half) & (zz <= cz + half)

def _naca0012_thickness(xi):
    sqrt_xi = np.sqrt(xi)
    return 0.12 / 0.2 * (0.2969 * sqrt_xi - 0.1260 * xi - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.1015 * xi**4)

def make_naca0012_mask(nx, ny, nz, cx, cy, cz, chord, angle_deg=0.0, device=torch.device("cpu")):
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32), indexing='ij')
    x_rel = xx - cx; y_rel = yy - cy
    x_local = x_rel * cos_a + y_rel * sin_a
    y_local = -x_rel * sin_a + y_rel * cos_a
    xi = x_local / chord
    in_chord = (xi >= 0.0) & (xi <= 1.0)
    xi_np = xi.cpu().numpy()
    thick = np.zeros_like(xi_np)
    valid = (xi_np >= 0.0) & (xi_np <= 1.0)
    thick[valid] = _naca0012_thickness(xi_np[valid])
    thick_t = torch.tensor(thick, device=device, dtype=torch.float32)
    half_t = chord * thick_t / 2.0
    inside_xy = (torch.abs(y_local) <= half_t) & in_chord
    inside_z = (zz >= cz - 0.5) & (zz <= cz + 0.5)
    return inside_xy & inside_z

def compute_drags(f, solid, u_tau, u_in, A_ref):
    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    near = _near_wall_mask(solid)
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    drag_fric = float((tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item())
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    fluid = ~solid
    drag_pres = float((p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item())
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_ref
    return drag_fric / dyn_p + drag_pres / dyn_p

CD_REF = {'cylinder_Re200': 1.30, 'square_prism_Re22000': 2.05, 'sphere_Re1000': 0.47,
          'naca0012_Re6e6': 0.008, 'suboff_Re2e6': 0.004}

def main():
    sdda_id = int(sys.argv[1])
    case_name = sys.argv[2]
    device = torch.device(f"sdaa:{sdda_id}")
    torch.sdaa.set_device(device)

    if case_name == 'cylinder_Re200':
        nx, ny, nz = 200, 80, 4; D = 24.0; R = D / 2.0
        cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
        u_in, nu = 0.08, 0.08 * D / 200.0
        A_ref = D * nz; n_steps, warmup = 3000, 500
        solid = make_cylinder_mask(nx, ny, nz, cx, cy, cz, R, device=torch.device('cpu'))
    elif case_name == 'square_prism_Re22000':
        nx, ny, nz = 200, 80, 4; D = 30.0
        cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
        u_in, nu = 0.08, 0.08 * D / 22000.0
        A_ref = D * nz; n_steps, warmup = 3000, 500
        solid = make_square_prism_mask(nx, ny, nz, cx, cy, cz, D, device=torch.device('cpu'))
    elif case_name == 'sphere_Re1000':
        nx, ny, nz = 120, 80, 80; D = 40.0; R = D / 2.0
        cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
        u_in, nu = 0.08, 0.08 * D / 1000.0
        A_ref = math.pi * R * R; n_steps, warmup = 2000, 500
        solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device=torch.device('cpu'))
    elif case_name == 'naca0012_Re6e6':
        nx, ny, nz = 200, 80, 4; chord = 40.0
        cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
        u_in, nu = 0.08, 0.08 * chord / 6_000_000.0
        A_ref = chord * nz; n_steps, warmup = 2000, 500
        solid = make_naca0012_mask(nx, ny, nz, cx, cy, cz, chord, device=torch.device('cpu'))
    elif case_name == 'suboff_Re2e6':
        from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
        from tensorlbm.suboff_resistance import _voxel_wetted_area
        nx = ny = nz = 200; hull_length = 160.0
        cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
        u_in, nu = 0.06, 0.06 * hull_length / 2_000_000.0
        n_steps, warmup = 2000, 500
        solid, _s = build_suboff_mask(hull_type=SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
                                       cx=cx, cy=cy, cz=cz, length=hull_length, device='cpu')
        A_ref = _voxel_wetted_area(solid, 1.0)
    else:
        print(f"Unknown case: {case_name}", flush=True); sys.exit(1)

    print(f"[{case_name}] grid={nx}x{ny}x{nz} u_in={u_in} nu={nu:.6e} steps={n_steps}", flush=True)
    solid = solid.to(device)
    tau = 3.0 * nu + 0.5; cs_smag = 0.05
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full_like(rho0, u_in); ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    initial_mass = float(f.sum().item())
    cd_series = []; t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        rho, ux, uy, uz = macroscopic3d(f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        u_tau = compute_u_tau(u_mag, nu, y_val=0.5, wall_law='hybrid')
        y_plus = u_tau * 0.5 / nu
        cd = compute_drags(f, solid, u_tau, u_in, A_ref)
        cd_series.append(cd if math.isfinite(cd) else 0.0)
        f = wall_function(f, solid, u_tau, y_plus, lattice='D3Q19', nu=nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 200 == 0: f = correct_mass3d(f, initial_mass)
        if step % 500 == 0:
            if not torch.isfinite(f).all():
                elapsed = time.time() - t0
                r = {'case': case_name, 'sdaa': sdda_id, 'status': 'DIVERGED',
                     'step_diverged': step, 'Cd_mean': float('nan'), 'Cd_std': float('nan'),
                     'elapsed_s': elapsed, 'Cd_ref': CD_REF[case_name],
                     'error_pct': float('nan')}
                Path(f"/tmp/hybrid_{case_name}.json").write_text(json.dumps(r))
                print(f"[{case_name}] DIVERGED at step {step}", flush=True)
                return
    elapsed = time.time() - t0
    post = cd_series[warmup:]
    cd_mean = sum(post) / len(post) if len(post) >= 2 else float('nan')
    cd_std = (sum((c - cd_mean)**2 for c in post) / (len(post)-1))**0.5 if len(post) > 1 else 0.0
    ref = CD_REF[case_name]
    err = abs(cd_mean - ref) / ref * 100 if math.isfinite(cd_mean) else float('nan')
    r = {'case': case_name, 'sdaa': sdda_id, 'status': 'OK',
         'grid': f'{nx}x{ny}x{nz}', 'u_in': u_in, 'nu': nu, 'tau': tau,
         'y_val': 0.5, 'n_steps': n_steps, 'warmup': warmup,
         'Cd_mean': cd_mean, 'Cd_std': cd_std, 'cd_samples': len(post),
         'Cd_ref': ref, 'error_pct': err, 'finite': True, 'elapsed_s': elapsed}
    Path(f"/tmp/hybrid_{case_name}.json").write_text(json.dumps(r))
    print(f"[{case_name}] DONE Cd={cd_mean:.4f} ref={ref} err={err:.1f}% time={elapsed:.0f}s", flush=True)

if __name__ == '__main__':
    main()
