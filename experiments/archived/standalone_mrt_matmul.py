#!/usr/bin/env python3
"""SUBOFF benchmark — D3Q19 MRT (matmul) collision.

Standalone file for comparing matmul vs Conv3D MRT performance/accuracy.
Run with:

    PYTHONPATH=src python standalone_mrt_matmul.py --steps 2000

Then compare with the Conv3D version:

    PYTHONPATH=src python standalone_mrt_conv3d.py --steps 2000
"""
import argparse, math, time, torch

U_IN = 0.06; RE = 2_000_000
NX, NY, NZ, HL = 200, 80, 80, 80.0
NU = U_IN * HL / RE; TAU = 3.0 * NU + 0.5; CS = 0.05; REF_CT = 0.00405


def build_solid(device):
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffHullType
    cx, cy, cz = NX * 0.35, NY / 2.0, NZ / 2.0
    solid, _ = build_suboff_mask(SuboffHullType.BARE_HULL, NX, NY, NZ, cx=cx, cy=cy, cz=cz, length=HL, device=device)
    return solid


def wetted_dpS(solid):
    from tensorlbm.suboff_resistance import _voxel_wetted_area
    S = _voxel_wetted_area(solid, 1.0)
    return 0.5 * 1.0 * U_IN**2 * S


def collide_mrt_matmul(f, tau, C_s):
    """Original matmul-based MRT collision (D3Q19)."""
    from tensorlbm.d3q19 import macroscopic3d, equilibrium3d
    from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau, _get_d3q19_mrt_matrices

    M, Mi = _get_d3q19_mrt_matrices(f.device)
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pn = _neq_stress_norm_3d(f_neq)
    te = _smagorinsky_tau(tau, pn, rho, C_s)
    sn = 1.0 / te
    N = f.shape[1] * f.shape[2] * f.shape[3]

    m = M @ f.reshape(19, N)
    me = M @ feq.reshape(19, N)
    dm = m - me
    sf = torch.ones(19, device=f.device); sf[0] = sf[3] = sf[4] = 0.0
    ms = m - sf.unsqueeze(1) * dm
    for k in [9, 11, 13, 14, 15]: ms[k] = m[k] - sn.reshape(N) * dm[k]
    return (Mi @ ms).reshape(19, *f.shape[1:])


def run(device, solid, dpS, n_steps):
    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.boundaries3d import far_field_bc_3d

    nz, ny, nx = solid.shape
    r0 = torch.ones(nz, ny, nx, device=device); u0 = torch.full((nz, ny, nx), U_IN, device=device)
    u0[solid] = 0.0
    f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
    im = float(r0.sum().item())

    win = n_steps // 6; drags = []; t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_mrt_matmul(f, TAU, CS)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, NU)
        f = far_field_bc_3d(f, u_in=U_IN)
        if step % 100 == 0: f = correct_mass3d(f, im)
        if math.isfinite(df): drags.append(df + dp)
        if not torch.isfinite(f).all(): return f"DIVERGED at {step}"
        if step % 500 == 0 and len(drags) >= win:
            ct = sum(drags[-win:]) / win / dpS
            print(f"  step={step:5d} Ct={ct:.5f} ({time.time()-t0:.0f}s)")

    ct_slide = sum(drags[-win:]) / win / dpS if len(drags) >= win else 0
    print(f"  DONE Ct_slide={ct_slide:.5f} err={abs(ct_slide-REF_CT)/REF_CT*100:.1f}% time={time.time()-t0:.0f}s")
    return ct_slide


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()
    dev = torch.device(args.device)
    solid = build_solid(dev)
    dpS = wetted_dpS(solid)
    print(f"D3Q19 MRT (matmul): grid={NX}x{NY}x{NZ} steps={args.steps} solid={solid.sum().item()}")
    run(dev, solid, dpS, args.steps)
