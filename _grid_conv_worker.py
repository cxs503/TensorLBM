"""Grid convergence study — D3Q19 MRT+Smag Cs=0.05 bare_hull running-average drag.

Usage:
    PYTHONPATH=src python _grid_conv_worker.py <did> <nx> <ny> <nz> <hl> <n_steps>
"""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C19
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

KAPPA, B_CONST, REF_CT = 0.41, 5.0, 0.00405


def wallfn(f, solid, nu, y_val=0.5):
    device = f.device
    c = C19.to(device).float()
    cx, cy, cz = c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1)
    fluid = ~solid; near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu; turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone(); vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly/KAPPA + B_CONST) - vm; fp = (ly/KAPPA + B_CONST) + 1.0/KAPPA
            uu = (uu - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut; ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx, fy, fz = coef*(ux*ium), coef*(uy*ium), coef*(uz*ium)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12, dtype=f.dtype, device=device).view(19, 1, 1, 1)
    cs2 = 1.0/3.0; cu = cx*ux + cy*uy + cz*uz
    f = f + w19 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
    df = (tw * (ux*ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp, sm = torch.roll(solid, 1, dims=2), torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


def main():
    did = int(sys.argv[1])
    nx, ny, nz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    hl = float(sys.argv[5]); n_steps = int(sys.argv[6])

    u_in, re, cs = 0.06, 2e6, 0.05
    nu = u_in * hl / re; tau = 3.0 * nu + 0.5
    res_cells = hl / 8.57  # hull diameter resolution
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] grid_conv {nx}³ res={res_cells:.1f}"
    print(f"{tag} tau={tau:.6f} steps={n_steps}", flush=True)
    t0 = time.time()

    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
        cx=cx_g, cy=cy_g, cz=cz_g, length=hl, device=device)
    S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    warmup = n_steps // 3
    fric, pres = [], []

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, df, dp = wallfn(f, solid, nu)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0: f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)

        if step % 1000 == 0 or step == n_steps:
            cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
            cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
            print(f"{tag} step={step:4d} Ct={cf+cp:.5f} f={cf:.4f} p={cp:.4f} n={len(fric)} ({time.time()-t0:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True); break

    cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
    cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
    ct = cf + cp
    err = abs(ct - REF_CT) / REF_CT * 100

    result = {
        "grid": f"{nx}x{ny}x{nz}", "hull_length_lu": hl,
        "diameter_resolution": res_cells,
        "cells_total": nx * ny * nz,
        "steps_total": n_steps, "n_averaged": len(fric),
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct, "error_pct": err,
        "wetted_area": S, "elapsed_s": time.time() - t0,
        "finite": bool(torch.isfinite(f).all().item()),
    }
    out = Path(f"/tmp/grid_conv/result_{did:02d}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{tag} DONE Ct={ct:.5f} err={err:.1f}%", flush=True)


if __name__ == "__main__":
    main()
