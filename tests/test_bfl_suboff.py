"""BFL vs Staircase comparison for SUBOFF bare_hull (D3Q19 MRT+Smag).

Uses fast vectorized operations from solver3d/boundaries3d.
Compares standard bounce_back_cells_3d (staircase) vs BFL interpolated
bounce-back with ellipsoid-analytical q-values.

Reports Ct_total at steps 200/400/600/800/1000.
"""
from __future__ import annotations

import math
import time

import torch

from tensorlbm.d3q19 import C as C19, equilibrium3d, OPPOSITE as OPP19
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.suboff_resistance import _voxel_wetted_area

C19_SHIFTS = [(int(C19[q, 0]), int(C19[q, 1]), int(C19[q, 2])) for q in range(19)]
OPP_LIST = [int(x) for x in OPP19.tolist()]


def far_field_bc_19(f: torch.Tensor, u_in: float = 0.06) -> torch.Tensor:
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium3d(rho1, torch.full_like(rho1, u_in),
                        torch.zeros_like(rho1), torch.zeros_like(rho1))
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, 0, :, :] = feq[:, 0, :, :]
    f[:, -1, :, :] = feq[:, -1, :, :]
    f[:, :, 0, :] = feq[:, :, 0, :]
    f[:, :, -1, :] = feq[:, :, -1, :]
    return f


def compute_forces_me(
    f_post: torch.Tensor, f_pre: torch.Tensor, solid: torch.Tensor,
    c_dev: torch.Tensor,
) -> float:
    """Momentum exchange drag force on solid."""
    total_fx = 0.0
    for d in range(1, 19):
        sx, sy, sz = C19_SHIFTS[d]
        nb_solid = torch.roll(solid, shifts=(-sz, -sy, -sx), dims=(0, 1, 2))
        fluid_bdry = ~solid & nb_solid
        if not fluid_bdry.any():
            continue
        delta = f_pre[d][fluid_bdry] - f_post[d][fluid_bdry]
        total_fx += (delta * c_dev[d, 0]).sum().item()
    return total_fx


def run_simulation(
    *,
    use_bfl: bool,
    nx: int, ny: int, nz: int,
    hull_length: float,
    u_in: float = 0.06,
    re: float = 1e5,
    cs_smag: float = 0.05,
    n_steps: int = 1000,
    device: str = "cpu",
) -> dict:
    dev = torch.device(device)
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    config = SuboffConfig()

    solid, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
        cx=cx_g, cy=cy_g, cz=cz_g, length=hull_length,
        device="cpu", config=config,
    )
    solid = solid.to(dev)

    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S
    c_dev = C19.to(dev).float()[:19]

    bfl_mask = None
    bfl_q = None
    if use_bfl:
        print("  BFL suboff q-field...")
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_suboff(
            nx, ny, nz, cx_g, cy_g, cz_g, hull_length,
            device=dev,
        )
        n_links = int(bfl_mask.sum().item())
        print(f"  Q-field: {n_links} links ({time.time()-t_q:.1f}s)")

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    label = "BFL" if use_bfl else "STAIRCASE"
    print(f"\n=== {label}: Re={re:.0e} tau={tau:.4f} {nx}x{ny}x{nz} L={hull_length} "
          f"Cs={cs_smag} ===")

    results: dict[int, dict[str, float]] = {}
    t0 = time.time()
    fx_samples: list[float] = []

    for step in range(1, n_steps + 1):
        # Reset solid
        f_eq = equilibrium3d(rho0, torch.zeros_like(rho0),
                             torch.zeros_like(rho0), torch.zeros_like(rho0))
        f[:, solid] = f_eq[:, solid]

        # Collide
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f_pre = f.clone()

        # Stream
        f = stream3d(f)

        # Far-field BC
        f = far_field_bc_19(f, u_in=u_in)

        # Wall BC
        if use_bfl and bfl_mask is not None:
            # BFL on boundary links
            for d in range(1, 19):
                if bfl_mask[d].any():
                    f = bouzidi_bounce_back_3d(f, f_pre, bfl_mask[d], bfl_q[d], d)
        else:
            # Standard bounce-back on solid cells
            f = bounce_back_cells_3d(f, solid)

        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > 50:
            fx = compute_forces_me(f, f_pre, solid, c_dev)
            fx_samples.append(fx)

        if step in (200, 400, 500, 600, 800, 1000):
            ct = sum(fx_samples[-50:]) / max(len(fx_samples[-50:]), 1) / dyn_p_S
            results[int(step)] = {"Ct_total": float(ct)}
            print(f"  step {step:4d}: Ct={float(ct):.6f} ({time.time()-t0:.0f}s)")

    print(f"  Done in {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nz", type=int, default=48)
    p.add_argument("--hull-length", type=float, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--re", type=float, default=1e5)
    p.add_argument("--bfl-only", action="store_true")
    p.add_argument("--staircase-only", action="store_true")
    args = p.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    hull_length = args.hull_length or (0.6 * nx)

    print(f"SUBOFF BFL vs Staircase: {nx}x{ny}x{nz} L={hull_length} Re={args.re:.0e}")

    all_results: dict[str, dict] = {}

    if not args.bfl_only:
        all_results["staircase"] = run_simulation(
            use_bfl=False, nx=nx, ny=ny, nz=nz,
            hull_length=hull_length, re=args.re, n_steps=args.steps, device=args.device,
        )

    if not args.staircase_only:
        all_results["bfl"] = run_simulation(
            use_bfl=True, nx=nx, ny=ny, nz=nz,
            hull_length=hull_length, re=args.re, n_steps=args.steps, device=args.device,
        )

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    steps_list = [200, 400, 500, 600, 800, 1000]
    keys = sorted(all_results.keys())
    header = f"{'Step':>6}  " + "  ".join(f"{k.upper():>18}" for k in keys)
    print(header)
    for step in steps_list:
        parts = [f"{step:6d}"]
        for key in keys:
            ct = all_results[key].get(step, {}).get("Ct_total", float("nan"))
            parts.append(f"{ct:18.6f}")
        print("  ".join(parts))

    if len(keys) == 2:
        k0, k1 = keys[0], keys[1]
        print(f"\nStaircase vs BFL Ct_total:")
        for step in steps_list:
            if step in all_results[k0] and step in all_results[k1]:
                cs = all_results[k0][step]["Ct_total"]
                cb = all_results[k1][step]["Ct_total"]
                delta = cs - cb
                pct = delta / max(abs(cs), 1e-12) * 100
                print(f"  step {step:4d}: stair={cs:.6f}  bfl={cb:.6f}  Δ={delta:.6f}  ({pct:+.1f}%)")
