"""Verify double wall-treatment bug using EXACT working configuration (D3Q19 + wall_function_3d)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d


def main():
    device_id = int(sys.argv[1])
    nx, ny, nz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    hull_length = float(sys.argv[5])
    n_steps = int(sys.argv[6])
    use_bb = sys.argv[7] == "True"

    u_in, re = 0.06, 2.0e6
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} D3Q19 {nx}³ bb={use_bb}]"
    print(f"{tag} tau={tau:.6f} nu={nu:.8e}", flush=True)

    t0 = time.time()

    # Build geometry
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
        length=hull_length, device="cpu",
    )
    solid = solid.to(device)

    S = _voxel_wetted_area(solid, 1.0)
    dynamic_pressure_S = 0.5 * 1.0 * u_in ** 2 * S

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    ct_series = []
    for step in range(1, n_steps + 1):
        # 1. Collision: MRT + Smagorinsky LES (EXACT working config)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.1)

        # 2. Stream
        f = stream3d(f)

        # 3. Wall function (IBM body force + drag computation)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)

        # 4. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 5. Bounce-back (ONLY if use_bb=True — DOUBLE wall treatment BUG)
        if use_bb:
            f = bounce_back_cells_3d(f, solid)

        # 6. Mass correction
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Record
        ct_fric = drag_fric / dynamic_pressure_S if dynamic_pressure_S > 0 else 0.0
        ct_pres = drag_pres / dynamic_pressure_S if dynamic_pressure_S > 0 else 0.0
        ct_total = ct_fric + ct_pres
        ct_series.append({"step": step, "ct": ct_total, "ct_fric": ct_fric, "ct_pres": ct_pres})

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at step {step}", flush=True)
            break

        if step % 200 == 0:
            print(f"{tag} step={step} Ct={ct_total:.6f} ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    warmup = max(1, len(ct_series) // 2)
    ct_fric_avg = sum(e["ct_fric"] for e in ct_series[warmup:]) / max(len(ct_series[warmup:]), 1)
    ct_pres_avg = sum(e["ct_pres"] for e in ct_series[warmup:]) / max(len(ct_series[warmup:]), 1)
    ct_total_avg = ct_fric_avg + ct_pres_avg
    ref_ct = 0.00405
    err_pct = abs(ct_total_avg - ref_ct) / ref_ct * 100

    result = {
        "lattice": "D3Q19",
        "collision": "MRT+Smag(Cs=0.1)",
        "grid": f"{nx}x{ny}x{nz}",
        "bounce_back": use_bb,
        "Ct_fric": ct_fric_avg, "Ct_pres": ct_pres_avg, "Ct_total": ct_total_avg,
        "error_pct": err_pct, "steps": len(ct_series),
        "finite": bool(torch.isfinite(f).all().item()), "elapsed_s": elapsed,
        "tau": tau, "nu": nu,
    }

    print(f"{tag} DONE Ct={ct_total_avg:.6f} err={err_pct:.1f}%", flush=True)
    Path(f"/tmp/wallfn_worker_{device_id}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
