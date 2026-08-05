"""Reproducibility worker: D3Q19 MRT+Smag Cs=0.05 bare_hull 200x80x80, 5000 steps.

Usage:
    PYTHONPATH=src python _repro_worker.py <device_id> [seed]

Writes result to /tmp/repro_results/result_<device_id>.json
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
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
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
    tw = ut * ut
    ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef * (ux * ium)
    fy = coef * (uy * ium)
    fz = coef * (uz * ium)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12, dtype=f.dtype, device=device).view(19, 1, 1, 1)
    cs2 = 1.0/3.0
    cu = cx*ux + cy*uy + cz*uz
    forcing = w19 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
    f = f + forcing
    df = (tw * (ux*ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


def main():
    did = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Fixed optimal config
    lattice = "D3Q19"
    collision = "MRT+Smag"
    cs = 0.05
    nx, ny, nz = 200, 80, 80
    hull_length = 80.0
    hull_type = "bare_hull"
    n_steps = 5000

    u_in, re = 0.06, 2e6
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tag = f"[SDAA:{did}] {lattice} {collision} Cs={cs} {hull_type} {nx}x{ny}x{nz} seed={seed}"
    print(f"{tag}", flush=True)
    print(f"  tau={tau:.6f} nu={nu:.6f} steps={n_steps} warmup={n_steps//3}", flush=True)
    t0 = time.time()

    # Geometry
    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
        cx=cx_g, cy=cy_g, cz=cz_g, length=hull_length, device=device)
    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S
    print(f"  wetted_area={S:.0f} dpS={dpS:.6f}", flush=True)

    # Initial conditions
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    warmup = n_steps // 3  # 1667
    fric, pres = [], []

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, df, dp = wallfn(f, solid, nu)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df)
            pres.append(dp)

        if step % 1000 == 0 or step == n_steps:
            cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
            cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
            ct = cf + cp
            elapsed = time.time() - t0
            print(f"  step={step:4d} Ct={ct:.6f} Cf={cf:.6f} Cp={cp:.6f} n={len(fric)} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

    cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
    cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
    ct = cf + cp
    err = abs(ct - REF_CT) / REF_CT * 100

    result = {
        "device_id": did,
        "seed": seed,
        "lattice": lattice,
        "collision": collision,
        "Cs": cs,
        "hull_type": hull_type,
        "grid": f"{nx}x{ny}x{nz}",
        "hull_length_lu": hull_length,
        "steps_total": n_steps,
        "warmup": warmup,
        "n_averaged": len(fric),
        "Ct_fric": cf,
        "Ct_pres": cp,
        "Ct_total": ct,
        "error_pct": err,
        "wetted_area": S,
        "elapsed_s": time.time() - t0,
        "finite": bool(torch.isfinite(f).all().item()),
    }

    out_dir = Path("/tmp/repro_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"result_{did:02d}.json"
    out_file.write_text(json.dumps(result))
    print(f"\nDONE Ct={ct:.6f} Cf={cf:.6f} Cp={cp:.6f} err={err:.2f}% elapsed={result['elapsed_s']:.0f}s", flush=True)
    print(f"Output: {out_file}", flush=True)


if __name__ == "__main__":
    main()
