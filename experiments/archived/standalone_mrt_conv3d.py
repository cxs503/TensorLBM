#!/usr/bin/env python3
"""SUBOFF benchmark — D3Q19 MRT (Conv3D) collision.

Identical to standalone_mrt_matmul.py except collision uses
torch.nn.functional.conv3d instead of torch.matmul.

Conv3D 1x1x1 is mathematically equivalent to matmul for MRT:
  M @ f.reshape(19,N)  ≡  conv3d(f.reshape(1,19,nz,ny,nx), M.reshape(19,19,1,1,1))

On CPU: results are BIT-IDENTICAL to matmul.
On SDAA (TecoDNN): ~4.5x faster per M@f, but TecoDNN Conv3D
may have precision differences from TecoBLAS matmul.
On CUDA (cuDNN): both fast and precise.

Run with:
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


def collide_mrt_conv3d(f, tau, C_s):
    """Conv3D-based MRT collision (D3Q19).

    Replaces M @ f.reshape(19,N) with conv3d(f4, Wm).
    Mathematically identical, potentially faster on GPU via DNN libraries.
    """
    from tensorlbm.d3q19 import macroscopic3d, equilibrium3d
    from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau, _get_d3q19_mrt_matrices

    M, Mi = _get_d3q19_mrt_matrices(f.device)
    Wm = M.reshape(19, 19, 1, 1, 1)
    Wmi = Mi.reshape(19, 19, 1, 1, 1)

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pn = _neq_stress_norm_3d(f_neq)
    te = _smagorinsky_tau(tau, pn, rho, C_s)
    sn = 1.0 / te

    f4 = f.unsqueeze(0)
    feq4 = feq.unsqueeze(0)
    conv = torch.nn.functional.conv3d

    m = conv(f4, Wm)
    me = conv(feq4, Wm)
    dm = m - me

    sf = torch.tensor([0.,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1], device=f.device).reshape(1,19,1,1,1)
    ms = m - sf * dm
    sn4 = sn.reshape(1, 1, *sn.shape)
    for k in [9, 11, 13, 14, 15]:
        ms[:, k] = m[:, k] - sn4 * dm[:, k]

    return conv(ms, Wmi).squeeze(0)


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
        f = collide_mrt_conv3d(f, TAU, CS)
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
    print(f"D3Q19 MRT (Conv3D): grid={NX}x{NY}x{NZ} steps={args.steps} solid={solid.sum().item()}")
    run(dev, solid, dpS, args.steps)
