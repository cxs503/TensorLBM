#!/usr/bin/env python3
"""Standalone SUBOFF drag benchmark — ready to share.

Single-file, zero-dependency beyond PyTorch + TensorLBM.
Just set PYTHONPATH=src and run.

Usage:
    PYTHONPATH=src python standalone_suboff_bench.py [--device sdaa:0] [--steps 5000]

Tests 4 configurations:
  1. D3Q19 MRT+Smag (Cs=0.05)  — production baseline
  2. D3Q19 MRT+Smag (Cs=0.05)  — Musker wall law
  3. D3Q27 CUMULANT (no LES)    — breakthrough, pressure smoothing
  4. D3Q27 CASCADED (no LES)    — most accurate

Expected results (200³, 5000 steps, Re=2e6):
  Config 1: Ct ≈ 0.00406  (0.2% error, production)
  Config 3: Ct ≈ 0.00418  (3.2% error, no LES needed)
  Config 4: Ct ≈ 0.00393  (2.9% error, most accurate)
"""

import argparse
import math
import time
import torch

# ── common parameters ──
U_IN = 0.06
RE = 2_000_000
NX, NY, NZ, HL = 200, 80, 80, 80.0
NU_LAT = U_IN * HL / RE
TAU = 3.0 * NU_LAT + 0.5
CS_SMAG = 0.05
REF_CT = 0.00405  # AFF-8 experimental reference


def build_solid(device: torch.device) -> torch.Tensor:
    """Build SUBOFF bare-hull solid mask."""
    from tensorlbm.suboff_cad import build_suboff_mask, SuboffHullType
    cx, cy, cz = NX * 0.35, NY / 2.0, NZ / 2.0
    solid, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL, NX, NY, NZ,
        cx=cx, cy=cy, cz=cz, length=HL, device=device,
    )
    return solid


def wetted_params(solid: torch.Tensor, u_in: float = U_IN) -> tuple[float, float]:
    """Return (wetted_area, dynamic_pressure * area)."""
    from tensorlbm.suboff_resistance import _voxel_wetted_area
    S = _voxel_wetted_area(solid, 1.0)
    return S, 0.5 * 1.0 * u_in ** 2 * S


def run_d3q19_mrt(
    device: torch.device,
    solid: torch.Tensor,
    dpS: float,
    n_steps: int,
    wall_law: str = "log",
) -> dict:
    """D3Q19 MRT+Smagorinsky — production configuration.

    Usage:
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d
        f = collide_smagorinsky_mrt3d(f, tau=TAU, C_s=CS_SMAG)
    """
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.boundaries3d import far_field_bc_3d

    nz, ny, nx = solid.shape
    r0 = torch.ones(nz, ny, nx, device=device)
    u0 = torch.full((nz, ny, nx), U_IN, device=device)
    u0[solid] = 0.0
    f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
    im = float(r0.sum().item())

    win = n_steps // 6
    drags = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=TAU, C_s=CS_SMAG)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, NU_LAT, wall_law=wall_law)
        f = far_field_bc_3d(f, u_in=U_IN)
        if step % 100 == 0:
            f = correct_mass3d(f, im)
        if math.isfinite(df):
            drags.append(df + dp)
        if not torch.isfinite(f).all():
            return {"error": f"diverged at step {step}"}

    ct_slide = sum(drags[-win:]) / win / dpS if len(drags) >= win else 0.0
    ct_all = sum(drags) / max(len(drags), 1) / dpS if drags else 0.0
    return {
        "Ct_slide": ct_slide,
        "Ct_all": ct_all,
        "err_slide": abs(ct_slide - REF_CT) / REF_CT * 100,
        "wall_time_s": time.time() - t0,
    }


def run_d3q27_cumulant(
    device: torch.device,
    solid: torch.Tensor,
    dpS: float,
    n_steps: int,
) -> dict:
    """D3Q27 CUMULANT (no LES) — breakthrough configuration.

    Usage:
        from tensorlbm.cumulant import collide_cumulant_d3q27
        f = collide_cumulant_d3q27(f, tau=TAU)
    """
    from tensorlbm.cumulant import collide_cumulant_d3q27
    from tensorlbm.wall_model import wall_function_d3q27
    from tensorlbm.d3q27 import equilibrium27, correct_mass27, stream27
    from tensorlbm.boundaries_d3q27 import far_field_bc_27

    nz, ny, nx = solid.shape
    r0 = torch.ones(nz, ny, nx, device=device)
    u0 = torch.full((nz, ny, nx), U_IN, device=device)
    u0[solid] = 0.0
    f = equilibrium27(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
    im = float(r0.sum().item())

    win = n_steps // 6
    drags = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_cumulant_d3q27(f, TAU)
        f = stream27(f)
        f, df, dp = wall_function_d3q27(f, solid, NU_LAT)
        f = far_field_bc_27(f, u_in=U_IN)
        if step % 200 == 0:
            f = correct_mass27(f, im)
        if math.isfinite(df):
            drags.append(df + dp)
        if not torch.isfinite(f).all():
            return {"error": f"diverged at step {step}"}

    ct_slide = sum(drags[-win:]) / win / dpS if len(drags) >= win else 0.0
    ct_all = sum(drags) / max(len(drags), 1) / dpS if drags else 0.0
    return {
        "Ct_slide": ct_slide,
        "Ct_all": ct_all,
        "err_slide": abs(ct_slide - REF_CT) / REF_CT * 100,
        "wall_time_s": time.time() - t0,
    }


def run_d3q27_cascaded(
    device: torch.device,
    solid: torch.Tensor,
    dpS: float,
    n_steps: int,
) -> dict:
    """D3Q27 CASCADED (no LES) — most accurate, slower."""
    from tensorlbm.cascaded_collision import collide_cascaded_d3q27
    from tensorlbm.wall_model import wall_function_d3q27
    from tensorlbm.d3q27 import equilibrium27, correct_mass27, stream27
    from tensorlbm.boundaries_d3q27 import far_field_bc_27

    nz, ny, nx = solid.shape
    r0 = torch.ones(nz, ny, nx, device=device)
    u0 = torch.full((nz, ny, nx), U_IN, device=device)
    u0[solid] = 0.0
    f = equilibrium27(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
    im = float(r0.sum().item())

    win = n_steps // 6
    drags = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_cascaded_d3q27(f, TAU)
        f = stream27(f)
        f, df, dp = wall_function_d3q27(f, solid, NU_LAT)
        f = far_field_bc_27(f, u_in=U_IN)
        if step % 200 == 0:
            f = correct_mass27(f, im)
        if math.isfinite(df):
            drags.append(df + dp)
        if not torch.isfinite(f).all():
            return {"error": f"diverged at step {step}"}

    ct_slide = sum(drags[-win:]) / win / dpS if len(drags) >= win else 0.0
    ct_all = sum(drags) / max(len(drags), 1) / dpS if drags else 0.0
    return {
        "Ct_slide": ct_slide,
        "Ct_all": ct_all,
        "err_slide": abs(ct_slide - REF_CT) / REF_CT * 100,
        "wall_time_s": time.time() - t0,
    }


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Standalone SUBOFF drag benchmark")
    parser.add_argument("--device", default="sdaa:0", help="Torch device (sdaa:0, cuda:0, cpu)")
    parser.add_argument("--steps", type=int, default=2000, help="Total steps")
    parser.add_argument("--configs", default="all", help="Comma-separated: mrt,cumulant,cascaded,mrt_musker or 'all'")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_steps = args.steps

    print(f"Standalone SUBOFF Drag Benchmark")
    print(f"  Device: {args.device}  Steps: {n_steps}")
    print(f"  Grid: {NX}x{NY}x{NZ}  Re: {RE:.0e}  Reference Ct: {REF_CT}")
    print()

    solid = build_solid(device)
    _, dpS = wetted_params(solid)
    solid_cells = solid.sum().item()
    print(f"  Solid cells: {solid_cells}  ({solid_cells / (NX * NY * NZ) * 100:.1f}%)")

    if args.configs == "all":
        configs = ["mrt", "mrt_musker", "cumulant", "cascaded"]
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    results = {}

    for cfg in configs:
        print(f"\n{'=' * 60}")
        if cfg == "mrt":
            print("Config 1: D3Q19 MRT+Smag (log-law) — production")
            results["mrt_log"] = run_d3q19_mrt(device, solid, dpS, n_steps, wall_law="log")
        elif cfg == "mrt_musker":
            print("Config 2: D3Q19 MRT+Smag (Musker) — continuous wall law")
            results["mrt_musker"] = run_d3q19_mrt(device, solid, dpS, n_steps, wall_law="musker")
        elif cfg == "cumulant":
            print("Config 3: D3Q27 CUMULANT (no LES) — breakthrough")
            results["cumulant"] = run_d3q27_cumulant(device, solid, dpS, n_steps)
        elif cfg == "cascaded":
            print("Config 4: D3Q27 CASCADED (no LES) — most accurate")
            results["cascaded"] = run_d3q27_cascaded(device, solid, dpS, n_steps)

        r = results[cfg]
        if "error" in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  Ct_slide={r['Ct_slide']:.5f}  Ct_all={r['Ct_all']:.5f}  err={r['err_slide']:.1f}%  time={r['wall_time_s']:.0f}s")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"SUMMARY (Reference Ct = {REF_CT})")
    print(f"{'Config':<20} {'Ct_slide':<12} {'Error':<10} {'Time':<10}")
    print("-" * 52)
    for name, r in results.items():
        if "error" not in r:
            print(f"{name:<20} {r['Ct_slide']:<12.5f} {r['err_slide']:<10.1f}% {r['wall_time_s']:<10.0f}s")


if __name__ == "__main__":
    main()
