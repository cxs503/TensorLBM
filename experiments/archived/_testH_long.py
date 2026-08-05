"""Test H: Long-time convergence — D3Q19 MRT+Smag bare_hull 160³ 10000 steps."""
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
from tensorlbm.drag_monitor import DragMonitor

KAPPA = 0.41; B_CONST = 5.0; REF_CT = 0.00405


def wallfn(f, solid, nu, y_val=0.5):
    device = f.device
    c19 = C19.to(device).float()
    cx = c19[:, 0].view(19, 1, 1, 1)
    cy = c19[:, 1].view(19, 1, 1, 1)
    cz = c19[:, 2].view(19, 1, 1, 1)
    fluid = ~solid; near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
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
    return f, df, dp


def main():
    did = int(sys.argv[1])
    nx, ny, nz, hl, n_steps = 160, 64, 64, 64.0, 10000
    u_in, re, cs = 0.06, 2e6, 0.05
    nu = u_in * hl / re; tau = 3.0 * nu + 0.5
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] H: D3Q19 long {n_steps} steps"
    print(f"{tag} tau={tau:.6f}", flush=True)

    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz, length=hl, device=device)
    S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    mon = DragMonitor(warmup=2000)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, df, dp = wallfn(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0: f = correct_mass3d(f, im)
        if step > mon.warmup: mon.add(step, df, dp)

        if step % 1000 == 0 and mon.n > 0:
            s = mon.summary()
            cf = s['Ct_fric_avg'] / dpS; cp = s['Ct_pres_avg'] / dpS; ct = cf + cp
            cs_std = s['Ct_total_std'] / dpS
            ch = s['Ct_change_window']
            conv = "✓" if s['converged'] else ""
            print(f"{tag} step={step:5d} Ct_avg={ct:.5f} f={cf:.4f} p={cp:.4f} std={cs_std:.5f} Δ={ch:.4f} {conv} ({time.time()-t0:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True); break

    s = mon.summary()
    cf = s['Ct_fric_avg'] / dpS; cp = s['Ct_pres_avg'] / dpS; ct = cf + cp
    result = {
        "test": "H_long_convergence",
        "grid": f"{nx}x{ny}x{nz}", "Cs": cs, "steps": step,
        "Ct_fric_avg": cf, "Ct_pres_avg": cp, "Ct_total_avg": ct,
        "Ct_total_std": s['Ct_total_std'] / dpS,
        "Ct_change_window": s['Ct_change_window'],
        "converged": s['converged'],
        "n": mon.n, "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": time.time() - t0,
    }
    out = Path(f"/tmp/test_long/result_{did:02d}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{tag} DONE Ct={ct:.5f}", flush=True)


if __name__ == "__main__":
    main()
