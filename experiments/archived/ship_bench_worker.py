"""Ship hull drag benchmark worker — D3Q19 MRT+Smag Cs=0.05 + wall_fn + farfield.

Usage:
    PYTHONPATH=src python ship_bench_worker.py --hull WIGLEY --device sdaa:8
    PYTHONPATH=src python ship_bench_worker.py --hull SERIES60 --device sdaa:9
    ...

Outputs a single JSON line to stdout with the result.
"""
from __future__ import annotations

import json
import math
import sys
import time
from argparse import ArgumentParser

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

# Hull-specific form factors (1+k) for ITTC reference Ct = Cf * (1+k)
_FORM_FACTORS = {
    ShipHullType.WIGLEY: 1.15,
    ShipHullType.SERIES60: 1.18,
    ShipHullType.KCS: 1.20,
    ShipHullType.KVLCC2: 1.25,
    ShipHullType.NPL: 1.10,
}


def run_benchmark(
    hull_type: str,
    device: str,
    nx: int = 200,
    ny: int = 60,
    nz: int = 60,
    re: float = 2e6,
    u_in: float = 0.06,
    hull_length: float = 80.0,
    n_steps: int = 3000,
    warmup: int = 1000,
    smagorinsky_cs: float = 0.05,
) -> dict:
    """Run a single ship hull drag benchmark."""
    hull = ShipHullType(hull_type)
    dev = torch.device(device)
    ff = _FORM_FACTORS[hull]

    # Lattice parameters
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    # Hull placement
    cx = nx * 0.3
    cy = ny * 0.5
    cz_keel = nz * 0.5

    # Build hull mask on CPU first, then move to device
    solid, stats = build_hull_mask(
        hull_type, nx, ny, nz,
        cx=cx, cy=cy,
        cz_keel=cz_keel,
        length=hull_length,
        device="cpu",
    )
    solid = solid.to(dev)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    # ITTC reference
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * ff

    # Initialize populations on CPU, move to device
    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full((nz, ny, nx), u_in)
    uy0 = torch.zeros(nz, ny, nx)
    uz0 = torch.zeros(nz, ny, nx)
    ux0[solid.cpu()] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    f = f.to(dev)
    initial_mass = float(f.sum().item())

    fric_vals = []
    pres_vals = []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        # 1. Collision: D3Q19 MRT + Smagorinsky
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)

        # 2. Streaming
        f = stream3d(f)

        # 3. Wall function (log-law body force)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5)

        # 4. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Collect running-average drag after warmup
        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        # Periodic progress
        if step % 1000 == 0 or step == n_steps:
            if len(fric_vals) > 0:
                cf_avg = sum(fric_vals) / len(fric_vals) / dyn_p_S
                cp_avg = sum(pres_vals) / len(pres_vals) / dyn_p_S
                ct_avg = cf_avg + cp_avg
            else:
                cf_avg = cp_avg = ct_avg = 0.0
            elapsed = time.time() - start_time
            print(f"[{hull.value}] step {step:4d}: Ct_fric={cf_avg:.5f} Ct_pres={cp_avg:.5f} "
                  f"Ct_tot={ct_avg:.5f} (ref {ct_ref:.5f})  elapsed={elapsed:.0f}s",
                  file=sys.stderr, flush=True)

            # Check for NaN
            if not torch.isfinite(f).all():
                print(f"[{hull.value}] NaN detected at step {step} — aborting", file=sys.stderr)
                break

    elapsed = time.time() - start_time

    # Final running-average coefficients
    cf = sum(fric_vals) / max(len(fric_vals), 1) / dyn_p_S if fric_vals else 0.0
    cp = sum(pres_vals) / max(len(pres_vals), 1) / dyn_p_S if pres_vals else 0.0
    ct = cf + cp
    err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float("inf")

    result = {
        "hull_type": hull.value,
        "device": device,
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "C_s": smagorinsky_cs,
        "Re": re,
        "nx": nx, "ny": ny, "nz": nz,
        "hull_length": hull_length,
        "u_in": u_in,
        "tau": tau,
        "nu": nu_lat,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(fric_vals),
        "wetted_area": S,
        "dynamic_pressure_x_area": dyn_p_S,
        "Cf_ITTC": cf_ittc,
        "form_factor_1pk": ff,
        "Ct_reference": ct_ref,
        "Ct_fric": cf,
        "Ct_pres": cp,
        "Ct_total": ct,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "stats": {k: v for k, v in stats.items() if not isinstance(v, (torch.Tensor,))},
    }

    return result


def main():
    parser = ArgumentParser(description="Ship hull drag benchmark worker")
    parser.add_argument("--hull", type=str, required=True,
                        choices=["wigley", "series60", "kcs", "kvlcc2", "npl"])
    parser.add_argument("--device", type=str, default="sdaa:0")
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--nz", type=int, default=60)
    parser.add_argument("--re", type=float, default=2e6)
    parser.add_argument("--u-in", type=float, default=0.06)
    parser.add_argument("--hull-length", type=float, default=80.0)
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--cs", type=float, default=0.05)
    args = parser.parse_args()

    result = run_benchmark(
        hull_type=args.hull,
        device=args.device,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        re=args.re,
        u_in=args.u_in,
        hull_length=args.hull_length,
        n_steps=args.n_steps,
        warmup=args.warmup,
        smagorinsky_cs=args.cs,
    )
    # Output JSON to stdout
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
