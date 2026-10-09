#!/usr/bin/env python3
"""Hybrid wall-function benchmark launcher — 5 test cases on SDAA 10-14.

TRACK 1: MRT+Smagorinsky + hybrid wall law (y+<11.6 viscous, y+>=11.6 log-law).
Tests:
  SDAA:10 — Cylinder D=24 Re=200    2000 steps
  SDAA:11 — Square prism D=30 Re=22000  2000 steps
  SDAA:12 — Sphere D=40 Re=1000     2000 steps
  SDAA:13 — NACA 0012 Re=6e6        2000 steps
  SDAA:14 — SUBOFF 200³ Re=2e6      2000 steps

Outputs Cd/Ct comparison to /tmp/hybrid_results.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the local src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d, sphere_mask
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_function_common import (
    _near_wall_mask,
    compute_u_tau,
    wall_function,
)

# ── Reference drag coefficient values ──────────────────────────────────────
CD_REF = {
    "cylinder_Re200":      1.30,   # Zdravkovich (1997)
    "square_prism_Re22000": 2.05,  # Lyn et al. (1995)
    "sphere_Re1000":        0.47,  # Schiller-Naumann
    "naca0012_Re6e6":       0.008, # Abbott & von Doenhoff
    "suboff_Re2e6":         0.004, # ITTC experimental Ct
}


# ── Geometry builders ─────────────────────────────────────────────────────
def make_cylinder_mask(
    nx: int, ny: int, nz: int, cx: float, cy: float, cz: float, radius: float,
    device: torch.device,
) -> torch.Tensor:
    """2D cylinder (circle in yz-plane, extruded along x)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Circle in yz-plane, span full x
    return (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2


def make_square_prism_mask(
    nx: int, ny: int, nz: int, cx: float, cy: float, cz: float, side: float,
    device: torch.device,
) -> torch.Tensor:
    """Square prism (square in yz-plane, extruded along x)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    half = side / 2.0
    return (yy >= cy - half) & (yy <= cy + half) & (zz >= cz - half) & (zz <= cz + half)


def _naca0012_thickness(xi):
    """NACA 0012 thickness distribution (0 ≤ xi ≤ 1)."""
    sqrt_xi = np.sqrt(xi)
    return 0.12 / 0.2 * (0.2969 * sqrt_xi - 0.1260 * xi
                          - 0.3516 * xi ** 2 + 0.2843 * xi ** 3
                          - 0.1015 * xi ** 4)


def make_naca0012_mask(
    nx: int, ny: int, nz: int,
    cx: float, cy: float, cz: float, chord: float, angle_deg: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """NACA 0012 airfoil mask (extruded as 2D in yz-plane, spanwise along z).

    The airfoil lies in the xy-plane (chord along x, thickness along y),
    centered at (cx, cy). The extruded solid spans z in [cz-0.5, cz+0.5].
    """
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Airfoil in xy: shift to chord origin, rotate to get local coords
    x_rel = xx - cx
    y_rel = yy - cy
    x_local = x_rel * cos_a + y_rel * sin_a
    y_local = -x_rel * sin_a + y_rel * cos_a

    xi = x_local / chord  # 0..1 chordwise
    in_chord = (xi >= 0.0) & (xi <= 1.0)
    xi_np = xi.cpu().numpy() if xi.is_cpu else xi.cpu().numpy()
    thick = np.zeros_like(xi_np)
    valid = (xi_np >= 0.0) & (xi_np <= 1.0)
    thick[valid] = _naca0012_thickness(xi_np[valid])
    thick_t = torch.tensor(thick, device=device, dtype=torch.float32)
    half_t = chord * thick_t / 2.0

    # Point is inside airfoil if |y_local| <= half_thickness AND in chord
    inside_xy = (torch.abs(y_local) <= half_t) & in_chord
    # Extrude: span ±0.5 around cz
    inside_z = (zz >= cz - 0.5) & (zz <= cz + 0.5)
    solid = inside_xy & inside_z
    return solid


# ── Drag computation ───────────────────────────────────────────────────────
def compute_drags_3d(
    f: torch.Tensor,
    solid: torch.Tensor,
    u_tau: torch.Tensor,
    u_in: float,
    A_ref: float,
):
    """Compute Cd (friction + pressure) for bluff bodies.

    friction drag: integrated τ_w · û_x over near-wall cells
    pressure drag: integrated p · (S_plus - S_minus) over fluid-solid faces
    """
    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    near = _near_wall_mask(solid)
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    drag_fric = float((tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item())

    # Pressure drag: p = (rho - 1) / 3 on faces where solid transitions
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)   # solid at x+1
    sm = torch.roll(solid, -1, dims=2)  # solid at x-1
    fluid = ~solid
    drag_pres = float((p * (sp.to(f.dtype) - sm.to(f.dtype))
                        * fluid.to(f.dtype)).sum().item())

    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_ref
    cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
    cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
    cd_total = cd_fric + cd_pres
    return drag_fric, drag_pres, cd_fric, cd_pres, cd_total


# ── Single simulation ──────────────────────────────────────────────────────
def run_hybrid_test(
    sdda_id: int,
    solid: torch.Tensor,
    u_in: float,
    nu: float,
    y_val: float,
    A_ref: float,
    n_steps: int,
    warmup: int,
    case_name: str,
) -> dict:
    device = torch.device(f"sdaa:{sdda_id}")
    torch.sdaa.set_device(device)
    solid = solid.to(device)
    nz, ny, nx = solid.shape

    tau = 3.0 * nu + 0.5
    cs_smag = 0.05

    # Init
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device)
    initial_mass = float(f.sum().item())

    cd_series: list[float] = []
    ct_series: list[float] = []

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Collision
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        # Stream
        f = stream3d(f)

        # Macroscopic
        rho, ux, uy, uz = macroscopic3d(f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

        # Hybrid wall law: compute u_tau with y+<11.6 threshold
        u_tau = compute_u_tau(u_mag, nu, y_val=y_val, wall_law="hybrid")
        y_plus = u_tau * y_val / nu

        # Drag (before wall fn modifies f)
        _, _, cd_fric, cd_pres, cd_total = compute_drags_3d(
            f, solid, u_tau, u_in, A_ref)

        cd_series.append(cd_total)

        # Wall function (Guo body force)
        f = wall_function(f, solid, u_tau, y_plus, lattice="D3Q19",
                          nu=nu, y_val=y_val)

        # Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # Bounce-back on solid
        f = bounce_back_cells_3d(f, solid)

        # Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Divergence check
        if step % 100 == 0:
            if not torch.isfinite(f).all():
                return {
                    "case": case_name, "sdaa": sdda_id, "status": "DIVERGED",
                    "step_diverged": step, "Cd_mean": float("nan"),
                    "Cd_std": float("nan"),
                }

    elapsed = time.time() - t0

    # Running average after warmup
    post = cd_series[warmup:]
    if len(post) < 2:
        cd_mean = float("nan")
        cd_std = 0.0
    else:
        cd_mean = sum(post) / len(post)
        cd_std = (sum((c - cd_mean) ** 2 for c in post) / (len(post) - 1)) ** 0.5

    ref = CD_REF.get(case_name, float("nan"))
    err_pct = abs(cd_mean - ref) / ref * 100 if ref and math.isfinite(cd_mean) else float("nan")

    return {
        "case": case_name,
        "sdaa": sdda_id,
        "status": "OK",
        "grid": f"{nx}×{ny}×{nz}",
        "Re": round(u_in * A_ref ** 0.5 * 4.0 / nu, 0) if "cylinder" in case_name or "square" in case_name else
              (round(u_in * 2 * (A_ref / math.pi) ** 0.5 / nu, 0) if "sphere" in case_name else None),
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "y_val": y_val,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_ref": ref,
        "error_pct": err_pct,
        "cd_samples": len(post),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }


# ── Launcher ───────────────────────────────────────────────────────────────
def test_case_1_cylinder(sdda_id: int):
    """Cylinder D=24 Re=200, 200×80×4."""
    nx, ny, nz = 200, 80, 4
    D = 24.0
    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    u_in = 0.08
    nu = u_in * D / 200.0
    A_ref = D * nz  # frontal area = diameter × span

    solid = make_cylinder_mask(nx, ny, nz, cx, cy, cz, R, device=torch.device("cpu"))
    return run_hybrid_test(sdda_id, solid, u_in, nu, y_val=0.5, A_ref=A_ref,
                            n_steps=3000, warmup=500,
                            case_name="cylinder_Re200")


def test_case_2_square(sdda_id: int):
    """Square prism D=30 Re=22000, 200×80×4."""
    nx, ny, nz = 200, 80, 4
    D = 30.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    u_in = 0.08
    nu = u_in * D / 22000.0
    A_ref = D * nz

    solid = make_square_prism_mask(nx, ny, nz, cx, cy, cz, D, device=torch.device("cpu"))
    return run_hybrid_test(sdda_id, solid, u_in, nu, y_val=0.5, A_ref=A_ref,
                            n_steps=3000, warmup=500,
                            case_name="square_prism_Re22000")


def test_case_3_sphere(sdda_id: int):
    """Sphere D=40 Re=1000, 120×80×80."""
    nx, ny, nz = 120, 80, 80
    D = 40.0
    R = D / 2.0
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    u_in = 0.08
    nu = u_in * D / 1000.0
    A_ref = math.pi * R ** 2

    solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device=torch.device("cpu"))
    return run_hybrid_test(sdda_id, solid, u_in, nu, y_val=0.5, A_ref=A_ref,
                            n_steps=2000, warmup=500,
                            case_name="sphere_Re1000")


def test_case_4_naca(sdda_id: int):
    """NACA 0012 Re=6e6, 200×80×4."""
    nx, ny, nz = 200, 80, 4
    chord = 40.0  # chord length in cells
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    u_in = 0.08
    nu = u_in * chord / 6_000_000.0
    A_ref = chord * nz  # planform area = chord × span

    solid = make_naca0012_mask(nx, ny, nz, cx, cy, cz, chord, angle_deg=0.0,
                                device=torch.device("cpu"))
    return run_hybrid_test(sdda_id, solid, u_in, nu, y_val=0.5, A_ref=A_ref,
                            n_steps=2000, warmup=500,
                            case_name="naca0012_Re6e6")


def test_case_5_suboff(sdda_id: int):
    """SUBOFF 200³ Re=2e6."""
    from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask

    nx = ny = nz = 200
    hull_length = 160.0
    cx = nx * 0.35
    cy = ny * 0.5
    cz = nz * 0.5
    u_in = 0.06
    nu = u_in * hull_length / 2_000_000.0

    solid, _stats = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hull_length,
        device="cpu",
    )
    # Wetted area for Ct
    from tensorlbm.suboff_resistance import _voxel_wetted_area
    S = _voxel_wetted_area(solid, 1.0)
    A_ref = S  # wetted area for Ct

    result = run_hybrid_test(sdda_id, solid, u_in, nu, y_val=0.5, A_ref=A_ref,
                              n_steps=2000, warmup=500,
                              case_name="suboff_Re2e6")
    # Rename Cd→Ct
    result["Ct_mean"] = result.pop("Cd_mean", float("nan"))
    result["Ct_std"] = result.pop("Cd_std", 0.0)
    result["Ct_ref"] = CD_REF["suboff_Re2e6"]
    result["error_pct_ct"] = (abs(result["Ct_mean"] - result["Ct_ref"]) /
                               result["Ct_ref"] * 100
                               if math.isfinite(result["Ct_mean"]) else float("nan"))
    result["wetted_area"] = S
    return result


# ── Test cases registry ────────────────────────────────────────────────────
TEST_CASES = [
    (10, test_case_1_cylinder,  "cylinder_Re200"),
    (11, test_case_2_square,    "square_prism_Re22000"),
    (12, test_case_3_sphere,    "sphere_Re1000"),
    (13, test_case_4_naca,      "naca0012_Re6e6"),
    (14, test_case_5_suboff,    "suboff_Re2e6"),
]


def main():
    print("=" * 80)
    print("HYBRID WALL-FUNCTION BENCHMARK — 5 Test Cases")
    print("MRT+Smagorinsky + hybrid wall law (y+<11.6 viscous, y+>=11.6 log-law)")
    print("=" * 80)

    results = []

    for sdda_id, test_fn, case_name in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"SDAA:{sdda_id} — {case_name}")
        print(f"{'─' * 60}")
        sys.stdout.flush()

        try:
            r = test_fn(sdda_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"case": case_name, "sdaa": sdda_id, "status": "ERROR",
                 "error": str(e)}
        results.append(r)

        cd = r.get("Cd_mean", r.get("Ct_mean", float("nan")))
        err = r.get("error_pct", r.get("error_pct_ct", float("nan")))
        cd_s = f"{cd:.4f}" if math.isfinite(cd) else "N/A"
        err_s = f"{err:.1f}%" if math.isfinite(err) else "N/A"
        print(f"  Status: {r.get('status')}  Cd/Ct: {cd_s}  Error: {err_s}  "
              f"Time: {r.get('elapsed_s', 0):.0f}s")

    # ── Comparison with log-law baseline ──
    print(f"\n{'=' * 80}")
    print("RESULTS TABLE: Hybrid vs Log-law")
    print(f"{'=' * 80}")
    print(f"{'Case':<25} {'Cd/Ct_mean':>10} {'Cd/Ct_std':>10} {'Ref':>10} "
          f"{'Err%':>8} {'Status':>10}")
    print("-" * 70)

    for r in results:
        cd = r.get("Cd_mean", r.get("Ct_mean", float("nan")))
        cd_std = r.get("Cd_std", r.get("Ct_std", 0.0))
        ref = r.get("Cd_ref", r.get("Ct_ref", float("nan")))
        err = r.get("error_pct", r.get("error_pct_ct", float("nan")))
        cd_s = f"{cd:.4f}" if math.isfinite(cd) else "N/A"
        cd_std_s = f"{cd_std:.4f}" if math.isfinite(cd_std) else "N/A"
        ref_s = f"{ref:.4f}" if math.isfinite(ref) else "N/A"
        err_s = f"{err:.1f}%" if math.isfinite(err) else "N/A"
        status = r.get("status", "?")
        print(f"{r.get('case', '?'):<25} {cd_s:>10} {cd_std_s:>10} {ref_s:>10} "
              f"{err_s:>8} {status:>10}")

    # ── Save output ──
    output = {
        "title": "Hybrid Wall-Function Benchmark — 5 Test Cases",
        "description": (
            "MRT+Smagorinsky (Cs=0.05) + hybrid wall law: "
            "y+<11.6 → viscous sublayer (u_tau = sqrt(nu*u/y)), "
            "y+>=11.6 → log-law (Newton iteration)."
        ),
        "parameters": {
            "n_steps": 2000,
            "warmup": 500,
            "y_val": 0.5,
            "cs_smag": 0.05,
        },
        "reference_coefficients": CD_REF,
        "results": results,
    }
    out_path = Path("/tmp/hybrid_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {out_path}")

    return output


if __name__ == "__main__":
    main()
