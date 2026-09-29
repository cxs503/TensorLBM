"""SUBOFF Full Hull (bare_hull+sail+fins) drag test: Staircase vs BFL.

D3Q19 MRT+Smag Cs=0.05, wall function + farfield, 200³ grid, Re=2e6.
Runs on two SDAA cards in parallel and writes results to /tmp/suboff_full_results.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Any

import torch

# Add src to path
sys.path.insert(0, "/root/TensorLBM_dev/src")

from tensorlbm.d3q19 import C as C19, equilibrium3d, OPPOSITE as OPP19, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.suboff_resistance import _voxel_wetted_area

KAPPA = 0.41
B_CONST = 5.0

C19_SHIFTS = [(int(C19[q, 0]), int(C19[q, 1]), int(C19[q, 2])) for q in range(19)]


def far_field_bc_19(f: torch.Tensor, u_in: float = 0.06) -> torch.Tensor:
    """Far-field BC for D3Q19: inlet eq, outlet neumann, y/z walls eq."""
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium3d(rho1, torch.full_like(rho1, u_in),
                        torch.zeros_like(rho1), torch.zeros_like(rho1))
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet (x-)
    f[:, :, :, -1] = f[:, :, :, -2]        # outlet (x+)
    f[:, 0, :, :] = feq[:, 0, :, :]        # y-
    f[:, -1, :, :] = feq[:, -1, :, :]      # y+
    f[:, :, 0, :] = feq[:, :, 0, :]        # z-
    f[:, :, -1, :] = feq[:, :, -1, :]      # z+
    return f


def wall_function_19(f: torch.Tensor, solid: torch.Tensor, nu: float, y_val: float = 0.5):
    """D3Q19 log-law wall function with Guo body force.

    Returns: (f_updated, drag_fric, drag_pres)
    """
    device = f.device
    c = C19.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    fluid = ~solid
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    near = nbrs

    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    u_tau = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu
    turb = (y_plus > 11.6) & near
    if bool(turb.any()):
        ut = u_tau[turb].clone()
        um = u_mag[turb]
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu)
            fv = ut * (lyp / KAPPA + B_CONST) - um
            fp = (lyp / KAPPA + B_CONST) + 1.0 / KAPPA
            ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut

    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)
    fx = coef * (ux * inv_umag)
    fy = coef * (uy * inv_umag)
    fz = coef * (uz * inv_umag)

    # D3Q19 Guo body force
    w19 = torch.tensor(
        [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12,
        dtype=f.dtype, device=device,
    ).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx * ux + cy * uy + cz * uz
    forcing = w19 * (1.0 + cu / cs2) * (cx * fx + cy * fy + cz * fz) / cs2
    f = f + forcing

    drag_fric = (tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    drag_pres = (p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, drag_fric, drag_pres


def run_simulation(
    *,
    use_bfl: bool,
    nx: int, ny: int, nz: int,
    hull_length: float,
    hull_type: str = "bare_hull",
    u_in: float = 0.06,
    re: float = 2_000_000.0,
    cs_smag: float = 0.05,
    n_steps: int = 3000,
    warmup: int = 1000,
    y_val: float = 0.5,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a single SUBOFF hull simulation.

    Returns dict with Ct_fric, Ct_pres, Ct_total, force_time_series, grid info, etc.
    """
    dev = torch.device(device)
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    config = SuboffConfig()
    hull_type_enum = SuboffHullType(hull_type)

    label = "BFL" if use_bfl else "STAIRCASE"
    print(f"\n{'='*70}")
    print(f"  {label}: D3Q19 MRT+Smag Cs={cs_smag} + wallfn + farfield  hull={hull_type}")
    print(f"  Re={re:.0e}  tau={tau:.5f}  nu={nu_lat:.6e}")
    print(f"  Grid: {nx}×{ny}×{nz}  L={hull_length}  steps={n_steps}  warmup={warmup}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    # Build hull mask
    t0_total = time.time()
    solid, stats = build_suboff_mask(
        hull_type_enum,
        nx=nx, ny=ny, nz=nz,
        cx=cx_g, cy=cy_g, cz=cz_g,
        length=hull_length,
        device="cpu", config=config,
    )
    solid = solid.to(dev)
    n_solid = int(solid.sum().item())
    print(f"  Solid cells: {n_solid} ({n_solid/(nx*ny*nz)*100:.1f}%)")

    # Wetted area
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S
    print(f"  Wetted area S={S:.1f}  dyn_q*S={dyn_p_S:.3f}")

    # BFL q-field (if needed)
    bfl_mask = None
    bfl_q = None
    if use_bfl:
        print("  Computing BFL q-field...")
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_suboff(
            nx, ny, nz, cx_g, cy_g, cz_g, hull_length,
            hull_type=hull_type,
            device=dev,
        )
        n_links = int(bfl_mask.sum().item())
        print(f"  Q-field: {n_links} links ({time.time() - t_q:.1f}s)")

    # Initialise
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    force_time_series: list[dict[str, float]] = []
    fric_list: list[float] = []
    pres_list: list[float] = []
    t_step_total = 0.0

    for step in range(1, n_steps + 1):
        ts = time.time()

        # Reset solid cells to eq at rest
        f_eq = equilibrium3d(rho0, torch.zeros_like(rho0),
                             torch.zeros_like(rho0), torch.zeros_like(rho0))
        f[:, solid] = f_eq[:, solid]

        # Collide: MRT + Smagorinsky LES
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # Pre-stream copy for BFL (always clone for consistency)
        f_pre = f.clone()

        # Stream
        f = stream3d(f)

        # Far-field BC
        f = far_field_bc_19(f, u_in=u_in)

        # Wall BC: BFL or staircase bounce-back
        if use_bfl and bfl_mask is not None:
            for d in range(1, 19):
                if bfl_mask[d].any():
                    f = bouzidi_bounce_back_3d(f, f_pre, bfl_mask[d], bfl_q[d], d)
        else:
            f = bounce_back_cells_3d(f, solid)

        # Wall function (log-law Guo body force)
        f, df, dp = wall_function_19(f, solid, nu_lat, y_val=y_val)

        # Force BC again after wall function
        f = far_field_bc_19(f, u_in=u_in)

        # Mass correction periodically
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        t_step_total += time.time() - ts

        # Collect force samples (post-warmup only, finite check)
        if step > warmup and math.isfinite(df) and math.isfinite(dp):
            fric_list.append(df)
            pres_list.append(dp)
            force_time_series.append({
                "step": step,
                "drag_fric": df,
                "drag_pres": dp,
                "drag_total": df + dp,
            })

        # Progress report
        if step % 200 == 0 or step == n_steps:
            n_samples = max(len(fric_list), 1)
            cf = sum(fric_list) / n_samples / dyn_p_S
            cp = sum(pres_list) / n_samples / dyn_p_S
            ct = cf + cp
            avg_step = t_step_total / step
            mlups = nx * ny * nz / avg_step / 1e6
            elapsed = time.time() - t0_total
            print(f"  step {step:4d}: Ct_f={cf:.6f} Ct_p={cp:.6f} Ct={ct:.6f} "
                  f"| {avg_step*1000:.0f}ms/step {mlups:.1f}MLUPS | {elapsed:.0f}s elapsed")

    total_elapsed = time.time() - t0_total
    n_samples = max(len(fric_list), 1)
    Ct_fric = sum(fric_list) / n_samples / dyn_p_S
    Ct_pres = sum(pres_list) / n_samples / dyn_p_S
    Ct_total = Ct_fric + Ct_pres

    print(f"\n  FINAL: Ct_fric={Ct_fric:.6f}  Ct_pres={Ct_pres:.6f}  Ct_total={Ct_total:.6f}")
    print(f"  Time: {total_elapsed:.1f}s ({n_samples} samples)")

    return {
        "schema": "tensorlbm.suboff-bfl-cs/v1",
        "boundary": label.lower(),
        "hull_type": hull_type,
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "Cs": cs_smag,
        "wall_function": f"log-law (κ={KAPPA}, B={B_CONST}, y_val={y_val})",
        "Re": re,
        "Ct_fric": Ct_fric,
        "Ct_pres": Ct_pres,
        "Ct_total": Ct_total,
        "finite": math.isfinite(Ct_total),
        "steps_completed": n_steps,
        "warmup": warmup,
        "samples": n_samples,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "hull_length": hull_length,
        "u_in": u_in,
        "tau": tau,
        "nu": nu_lat,
        "wetted_area": S,
        "dynamic_pressure": dyn_p_S,
        "solid_cells": n_solid,
        "device": device,
        "total_elapsed_s": total_elapsed,
        "force_time_series": force_time_series,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SUBOFF Hull drag test")
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--bfl", action="store_true", help="Use BFL boundary (default: staircase)")
    p.add_argument("--hull-type", default="bare_hull",
                   choices=["bare_hull", "with_sail", "full"],
                   help="SUBOFF hull variant (default: bare_hull)")
    p.add_argument("--cs", type=float, default=0.05, help="Smagorinsky constant (default: 0.05)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args()

    result = run_simulation(
        use_bfl=args.bfl,
        hull_type=args.hull_type,
        nx=200, ny=80, nz=80,
        hull_length=100.0,
        u_in=0.06,
        re=2_000_000.0,
        cs_smag=args.cs,
        n_steps=3000,
        warmup=1000,
        y_val=0.5,
        device=args.device,
    )

    if args.output:
        out_path = args.output
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults written to {out_path}")
    else:
        print(f"\nCt_total = {result['Ct_total']:.6f}")
        print(f"Ct_fric  = {result['Ct_fric']:.6f}")
        print(f"Ct_pres  = {result['Ct_pres']:.6f}")
