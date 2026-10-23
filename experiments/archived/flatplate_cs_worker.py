"""Flat plate Cf validation worker — called per Cs value on a specific SDAA card.

Usage:
    SDAA_VISIBLE_DEVICES=<card_id> PYTHONPATH=src python flatplate_cs_worker.py \
        --cs 0.10 --output /tmp/flatplate_cs010.json

Compares against ITTC-1957 friction line: Cf = 0.075 / (log10(Re) - 2)^2
For Re=2e6: Cf_ittc = 0.00405
"""
from __future__ import annotations
import argparse, json, math, sys, time
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.ibm import ibm_apply_body_force_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


def ittc_cf(re: float) -> float:
    """ITTC-1957 friction line."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


def run(cs: float, nx: int, ny: int, nz: int, re: float,
        u_in: float, plate_pct: float, n_steps: int, warmup: int,
        device: torch.device, output_path: str) -> dict:
    L = float(nx)
    nu_lat = u_in * L / re
    tau = 3.0 * nu_lat + 0.5

    # Flat plate on bottom (y=0), from x_start to nx
    x_start = int((1.0 - plate_pct) * nx)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, x_start:] = True
    plate_area = (nx - x_start) * nz
    dyn_p_A = 0.5 * 1.0 * u_in ** 2 * plate_area

    # ITTC-1957 reference Cf
    cf_ittc = ittc_cf(re)

    # Initialization
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"Flat plate: Re={re:.0e} nx={nx} ny={ny} nz={nz} Cs={cs}")
    print(f"nu={nu_lat:.2e} tau={tau:.6f} u_in={u_in}")
    print(f"Plate: x=[{x_start},{nx}), area={plate_area} cells")
    print(f"ITTC-1957 Cf = {cf_ittc:.5f}")
    print(f"Dynamic pressure × area = {dyn_p_A:.4f}\n")

    samples = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        f, drag_f, drag_p = wall_function_3d(f, solid, nu_lat, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)
        if step > warmup and math.isfinite(drag_f):
            samples.append(drag_f)
        if step % 500 == 0 or step == n_steps:
            cf = (sum(samples) / max(len(samples), 1)) / dyn_p_A if samples else float('nan')
            err_pct = abs(cf - cf_ittc) / cf_ittc * 100.0 if cf_ittc > 0 else float('nan')
            elapsed = time.time() - t0
            print(f"  step {step:4d}: Cf={cf:.6f} (ITTC {cf_ittc:.5f}, err={err_pct:.1f}%) "
                  f"[{elapsed:.0f}s]")

    elapsed = time.time() - t0
    cf_final = (sum(samples) / max(len(samples), 1)) / dyn_p_A if samples else float('nan')
    err_pct = abs(cf_final - cf_ittc) / cf_ittc * 100.0 if cf_ittc > 0 else float('nan')

    result = {
        "cs": cs,
        "re": re,
        "nx": nx, "ny": ny, "nz": nz,
        "n_steps": n_steps,
        "warmup": warmup,
        "u_in": u_in,
        "nu_lat": nu_lat,
        "tau": tau,
        "plate_pct": plate_pct,
        "plate_area_cells": plate_area,
        "n_samples": len(samples),
        "cf_final": cf_final,
        "cf_ittc": cf_ittc,
        "error_pct": err_pct,
        "wall_clock_s": elapsed,
        "device": str(device),
        "cf_history": None,
    }
    print(f"\nFinal Cf={cf_final:.6f} vs ITTC={cf_ittc:.5f} err={err_pct:.1f}% time={elapsed:.0f}s")

    with open(output_path, 'w') as fp:
        json.dump(result, fp, indent=2)
    print(f"Results saved to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs", type=float, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--re", type=float, default=2e6)
    parser.add_argument("--u-in", type=float, default=0.06)
    parser.add_argument("--plate-pct", type=float, default=0.80)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--device", type=str, default="sdaa")
    args = parser.parse_args()

    # Resolve device
    device_str = args.device
    if device_str == "sdaa":
        # Use first visible SDAA device; set via SDAA_VISIBLE_DEVICES
        if hasattr(torch, 'sdaa') and torch.sdaa.device_count() > 0:
            device = torch.device("sdaa:0")
        else:
            device = torch.device("cpu")
            print("WARNING: No SDAA devices visible, falling back to CPU")
    else:
        device = torch.device(device_str)

    run(
        cs=args.cs,
        nx=args.nx, ny=args.ny, nz=args.nz,
        re=args.re, u_in=args.u_in,
        plate_pct=args.plate_pct,
        n_steps=args.n_steps, warmup=args.warmup,
        device=device,
        output_path=args.output,
    )
