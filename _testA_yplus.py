"""Test A: y+ sensitivity — grid convergence with wall-distance diagnostics.

D3Q19 MRT+Smag Cs=0.05 bare_hull at 4 grid sizes (160³,200³,256³,320³).
Outputs y+ histogram + Ct convergence at each logging interval.
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

KAPPA = 0.41; B_CONST = 5.0; REF_CT = 0.00405


def wallfn(f, solid, nu, y_val=0.5):
    device = f.device
    c19 = C19.to(device).float()
    cx = c19[:, 0].view(19, 1, 1, 1)
    cy = c19[:, 1].view(19, 1, 1, 1)
    cz = c19[:, 2].view(19, 1, 1, 1)
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
            fv = uu * (ly/KAPPA + B_CONST) - vm
            fp = (ly/KAPPA + B_CONST) + 1.0/KAPPA
            uu = (uu - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut; ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12, dtype=f.dtype, device=device).view(19,1,1,1)
    cs2 = 1.0/3.0
    cu = cx*ux + cy*uy + cz*uz
    forcing = w19 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
    f = f + forcing
    df = (tw * (ux*ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp, ut, near


def main():
    did = int(sys.argv[1])
    nx = int(sys.argv[2]); ny = int(sys.argv[3]); nz = int(sys.argv[4])
    hl = float(sys.argv[5]); n_steps = int(sys.argv[6])

    u_in, re, cs = 0.06, 2e6, 0.05
    nu = u_in * hl / re; tau = 3.0 * nu + 0.5
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] A: D3Q19 bare {nx}³"
    print(f"{tag} tau={tau:.6f} nu={nu:.2e}", flush=True)

    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx_g, cy=cy_g, cz=cz_g,
        length=hl, device=device)
    S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    warmup = n_steps // 3
    fric, pres = [], []
    yplus_hist = []  # store y+ stats at report steps
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, df, dp, ut, near = wallfn(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0: f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)

        if step % 200 == 0 or step == n_steps:
            cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
            cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
            ct = cf + cp
            err = abs(ct - REF_CT) / REF_CT * 100

            # y+ diagnostics
            yp_vals = (0.5 * ut / nu)[near] if near.any() else torch.zeros(0, device=device)
            if yp_vals.numel() > 0:
                yp_cpu = yp_vals.cpu()
                pct_gt30 = float((yp_cpu > 30).float().mean().item()) * 100
                pct_lt11 = float((yp_cpu < 11.6).float().mean().item()) * 100
                yp_median = float(yp_cpu.median().item())
            else:
                pct_gt30 = pct_lt11 = yp_median = 0.0

            yplus_hist.append({"step": step, "pct_gt30": pct_gt30, "pct_lt11": pct_lt11, "yp_median": yp_median})

            print(f"{tag} step={step:4d} Ct={ct:.5f} err={err:.1f}% "
                  f"y+_med={yp_median:.1f} >30={pct_gt30:.0f}% <11.6={pct_lt11:.0f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True); break

    cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
    cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
    result = {
        "test": "A_yplus_sensitivity",
        "grid": f"{nx}x{ny}x{nz}", "hull_length": hl, "Cs": cs,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": cf + cp,
        "error_pct_ITTC": abs(cf + cp - REF_CT) / REF_CT * 100,
        "steps": step, "finite": bool(torch.isfinite(f).all().item()),
        "yplus_history": yplus_hist,
        "elapsed_s": time.time() - t0,
    }
    out = Path(f"/tmp/test_grid/result_{did:02d}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{tag} DONE Ct={cf+cp:.5f}", flush=True)


if __name__ == "__main__":
    main()
