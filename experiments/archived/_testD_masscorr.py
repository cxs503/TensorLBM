"""Test D: Mass correction frequency impact on pressure drift.

D3Q27 CUMULANT bare_hull 160³ at 4 correction intervals.
"""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q27 import equilibrium27, macroscopic27, correct_mass27, stream27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.wall_model import wall_function_d3q27
from tensorlbm.boundaries_d3q27 import far_field_bc_27

REF_CT = 0.00405


def main():
    did = int(sys.argv[1])
    interval = int(sys.argv[2])  # mass correction interval

    nx, ny, nz, hl, n_steps = 160, 64, 64, 64.0, 4000
    u_in, re = 0.06, 2e6
    nu = u_in * hl / re; tau = 3.0 * nu + 0.5
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did}] D: mass_corr_every_{interval}"
    print(f"{tag} tau={tau:.6f}", flush=True)

    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz, length=hl, device=device)
    S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    im = float(rho0.sum().item())

    warmup = n_steps // 3
    fric, pres = [], []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_cumulant_d3q27(f, tau)
        f = stream27(f)
        f, df, dp = wall_function_d3q27(f, solid, nu, y_val=0.5)
        f = far_field_bc_27(f, u_in=u_in)
        if step % interval == 0:
            mass_before = float(torch.ones_like(rho0).sum().item())
            f = correct_mass27(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)

        if step % 500 == 0 or step == n_steps:
            cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
            cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
            ct = cf + cp
            print(f"{tag} step={step:4d} Ct={ct:.5f} f={cf:.4f} p={cp:.4f} n={len(fric)} ({time.time()-t0:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True); break

    cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
    cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
    result = {
        "test": "D_mass_correction",
        "grid": f"{nx}x{ny}x{nz}", "correction_interval": interval,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": cf + cp,
        "error_pct": abs(cf + cp - REF_CT) / REF_CT * 100,
        "steps": step, "finite": bool(torch.isfinite(f).all().item()),
        "n_averaged": len(fric), "elapsed_s": time.time() - t0,
    }
    out = Path(f"/tmp/test_mass/result_{did:02d}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"{tag} DONE Ct={cf+cp:.5f}", flush=True)


if __name__ == "__main__":
    main()
