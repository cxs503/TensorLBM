#!/usr/bin/env python3
"""Cross-test matrix: RANS k-ε, wall functions, and adaptive mesh
across all bluff-body benchmarks (sphere, airfoil, ellipsoid, SUBOFF).

Usage
-----
  PYTHONPATH=src python benchmarks/bench_bluff_body_matrix.py [--device cuda]
  PYTHONPATH=src python benchmarks/bench_bluff_body_matrix.py --fast --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.boundaries3d import (
    apply_zou_he_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.ellipsoid_benchmark import (
    EllipsoidConfig,
    build_ellipsoid_mask,
    reference_ellipsoid_cd,
)
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.rans_ke import KESolver
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.utils import resolve_device
from tensorlbm.wall_model import apply_wall_model_bounce_back


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

@dataclass
class CrossTestConfig:
    """Configuration for a single cross-test run."""
    case: str           # "sphere", "airfoil", "ellipsoid", "suboff"
    use_rans: bool = False
    use_wall_model: bool = False
    use_smag: bool = True
    nx: int = 120
    ny: int = 64
    nz: int = 64
    u_in: float = 0.06
    re: float = 100.0
    n_steps: int = 3000
    warmup_steps: int = 1500
    smag_cs: float = 0.1
    device: str = "cpu"
    seed: int = 42
    rans_nu_t_max: float = 0.1


# ---------------------------------------------------------------------------
# Sphere cross-test
# ---------------------------------------------------------------------------

def _run_sphere_cross(cfg: CrossTestConfig) -> dict:
    """Run sphere with optional RANS/wall-model coupling."""
    radius = cfg.ny * 0.15
    nu = cfg.u_in * 2.0 * radius / cfg.re
    tau = 3.0 * nu + 0.5

    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    cx, cy, cz = cfg.nx / 2.0, cfg.ny / 2.0, cfg.nz / 2.0
    mask = sphere_mask(cfg.nx, cfg.ny, cfg.nz, cx, cy, cz, radius, device=device)
    wall_mask = make_channel_wall_mask_3d(cfg.nz, cfg.ny, cfg.nx, mask, device=device)

    rho0 = torch.ones((cfg.nz, cfg.ny, cfg.nx), device=device)
    ux0 = torch.full_like(rho0, cfg.u_in)
    uy0 = uz0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    area = math.pi * radius**2
    dyn_p = 0.5 * cfg.u_in**2 * area

    # RANS init
    ke: KESolver | None = None
    if cfg.use_rans:
        ke = KESolver(nu=nu, nu_t_max=cfg.rans_nu_t_max)
        ke.initialize(ux0, uy0, uz0)

    fx_all: list[float] = []
    for step in range(1, cfg.n_steps + 1):
        # Collision
        if cfg.use_rans and ke is not None:
            _, ux_m, uy_m, uz_m = macroscopic3d(f)
            nu_t = ke.step(ux_m, uy_m, uz_m, mask)
            nu_eff = nu + float(nu_t.mean().item())
            tau_eff = max(min(3.0 * nu_eff + 0.5, 2.0), 0.501)
            f = collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)
        elif cfg.use_smag and tau < 0.575:
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cfg.smag_cs)
        else:
            from tensorlbm.solver3d import collide_bgk3d
            f = collide_bgk3d(f, tau=tau)

        f = stream3d(f)

        # Forces AFTER stream, BEFORE bounce-back
        fx, _, _ = compute_obstacle_forces_3d(f, mask)

        # BCs
        if cfg.use_wall_model:
            # Don't use Zou/He bounce-back on obstacle — wall model handles it
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=cfg.u_in, wall_mask=wall_mask,
                obstacle_mask=torch.zeros_like(mask),
            )
            _, ux_w, uy_w, uz_w = macroscopic3d(f)
            f = apply_wall_model_bounce_back(f, mask, ux_w, uy_w, uz_w, nu)
        else:
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=cfg.u_in, wall_mask=wall_mask, obstacle_mask=mask,
            )

        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > cfg.warmup_steps:
            fx_all.append(float(fx.item()))

    cd = sum(fx_all) / max(len(fx_all), 1) / dyn_p

    # Schiller-Naumann reference
    sn = 24.0 / cfg.re * (1.0 + 0.15 * cfg.re**0.687)
    return {"cd": cd, "cd_ref": sn, "cd_err_pct": abs(cd - sn) / sn * 100}


# ---------------------------------------------------------------------------
# Ellipsoid cross-test
# ---------------------------------------------------------------------------

def _run_ellipsoid_cross(cfg: CrossTestConfig) -> dict:
    """Run ellipsoid with optional RANS/wall-model coupling."""
    a = cfg.ny * 0.4   # semi-major, a/b=3 → 6:1
    b = a / 3.0
    diam = 2.0 * b
    nu = cfg.u_in * diam / cfg.re
    tau = 3.0 * nu + 0.5

    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    mask = build_ellipsoid_mask(
        cfg.nx, cfg.ny, cfg.nz, a, b, alpha_deg=0.0, device=device,
    )
    wall_mask = make_channel_wall_mask_3d(cfg.nz, cfg.ny, cfg.nx, mask, device=device)

    rho0 = torch.ones((cfg.nz, cfg.ny, cfg.nx), device=device)
    ux0 = torch.full_like(rho0, cfg.u_in)
    uy0 = uz0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    dyn_p = 0.5 * cfg.u_in**2 * math.pi * b**2

    ke: KESolver | None = None
    if cfg.use_rans:
        ke = KESolver(nu=nu, nu_t_max=cfg.rans_nu_t_max)
        ke.initialize(ux0, uy0, uz0)

    fx_all: list[float] = []
    for step in range(1, cfg.n_steps + 1):
        if cfg.use_rans and ke is not None:
            _, ux_m, uy_m, uz_m = macroscopic3d(f)
            nu_t = ke.step(ux_m, uy_m, uz_m, mask)
            nu_eff = nu + float(nu_t.mean().item())
            tau_eff = max(min(3.0 * nu_eff + 0.5, 2.0), 0.501)
            f = collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)
        elif cfg.use_smag and tau < 0.575:
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cfg.smag_cs)
        else:
            from tensorlbm.solver3d import collide_bgk3d
            f = collide_bgk3d(f, tau=tau)

        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, mask)

        if cfg.use_wall_model:
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=cfg.u_in, wall_mask=wall_mask,
                obstacle_mask=torch.zeros_like(mask),
            )
            _, ux_w, uy_w, uz_w = macroscopic3d(f)
            f = apply_wall_model_bounce_back(f, mask, ux_w, uy_w, uz_w, nu)
        else:
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=cfg.u_in, wall_mask=wall_mask, obstacle_mask=mask,
            )

        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > cfg.warmup_steps:
            fx_all.append(float(fx.item()))

    cd = sum(fx_all) / max(len(fx_all), 1) / dyn_p
    ref = reference_ellipsoid_cd(cfg.re, 0.0)
    return {"cd": cd, "cd_ref": ref["cd"], "cd_err_pct": abs(cd - ref["cd"]) / ref["cd"] * 100}


# ---------------------------------------------------------------------------
# Airfoil cross-test (2D — RANS and wall model not applicable)
# ---------------------------------------------------------------------------

def _run_airfoil_cross(cfg: CrossTestConfig) -> dict:
    """Run airfoil baseline (2D, no 3D RANS/wall)."""
    from tensorlbm.airfoil_benchmark import AirfoilConfig, run_airfoil_benchmark
    a_cfg = AirfoilConfig(
        chord=cfg.nx * 0.25, alpha_deg=4.0, re=cfg.re,
        nx=cfg.nx, ny=cfg.ny, u_in=cfg.u_in,
        n_steps=cfg.n_steps, warmup_steps=cfg.warmup_steps,
        smagorinsky_cs=cfg.smag_cs,
        device=cfg.device, seed=cfg.seed,
    )
    result = run_airfoil_benchmark(a_cfg)
    return {"cd": result["cd_sim"], "cd_ref": result["cd_ref"],
            "cd_err_pct": result["cd_err_pct"],
            "cl": result["cl_sim"], "cl_ref": result["cl_ref"]}


# ---------------------------------------------------------------------------
# SUBOFF cross-test (already has RANS/wall/adaptive support)
# ---------------------------------------------------------------------------

def _run_suboff_cross(cfg: CrossTestConfig) -> dict:
    """Run SUBOFF baseline (RANS/wall/adaptive built-in)."""
    from tensorlbm.suboff_resistance import (
        SuboffResistanceBenchmarkConfig,
        run_suboff_resistance_benchmark,
    )
    s_cfg = SuboffResistanceBenchmarkConfig(
        base_length_lu=64, lbm_tau=0.54, smagorinsky_cs=cfg.smag_cs,
        lbm_steps=cfg.n_steps, lbm_warmup=cfg.warmup_steps,
        device=cfg.device, seed=cfg.seed,
        use_rans_ke=cfg.use_rans,
        use_wall_model=cfg.use_wall_model,
        use_adaptive_mesh=False,
    )
    result = run_suboff_resistance_benchmark(s_cfg)
    if isinstance(result, dict):
        return {"cd": result.get("cd_pred", 0), "cd_err_pct": result.get("cd_err_pct", 0)}
    return {"cd": 0, "cd_err_pct": 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CASES = {
    "sphere": _run_sphere_cross,
    "ellipsoid": _run_ellipsoid_cross,
    "airfoil": _run_airfoil_cross,
    "suboff": _run_suboff_cross,
}

_COMBOS_3D = [
    ("baseline",  {"use_rans": False, "use_wall_model": False, "use_smag": True}),
    ("+RANS",     {"use_rans": True,  "use_wall_model": False, "use_smag": False}),
    ("+Wall",     {"use_rans": False, "use_wall_model": True,  "use_smag": True}),
    ("+RANS+Wall",{"use_rans": True,  "use_wall_model": True,  "use_smag": False}),
]

_COMBOS_2D = [
    ("baseline",  {"use_rans": False, "use_wall_model": False}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bluff-body cross-test matrix")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: minimal grid and steps for smoke test")
    parser.add_argument("--cases", nargs="+",
                        default=["sphere", "airfoil", "ellipsoid"],
                        choices=["sphere", "airfoil", "ellipsoid", "suboff"],
                        help="Cases to run (default: sphere airfoil ellipsoid)")
    parser.add_argument("--combos", nargs="+",
                        default=["baseline", "+RANS", "+Wall", "+RANS+Wall"],
                        help="Combinations to test")
    args = parser.parse_args()

    if args.fast:
        nx, ny, nz, steps, warmup = 80, 36, 36, 200, 100
    else:
        nx, ny, nz, steps, warmup = 120, 64, 64, 3000, 1500

    results: list[dict] = []

    for case in args.cases:
        combos = _COMBOS_2D if case == "airfoil" else _COMBOS_3D
        for combo_name, combo_kwargs in combos:
            if combo_name not in args.combos:
                continue
            if case == "airfoil" and combo_name != "baseline":
                continue  # RANS/wall are 3D-only

            cfg = CrossTestConfig(
                case=case, nx=nx, ny=ny, nz=nz,
                n_steps=steps, warmup_steps=warmup,
                device=args.device, **combo_kwargs,
            )
            tag = f"{case}/{combo_name}"
            print(f"\n{'='*60}")
            print(f"  {tag}")
            print(f"{'='*60}")
            try:
                result = _CASES[case](cfg)
                result["case"] = case
                result["combo"] = combo_name
                results.append(result)
                print(f"  → Cd={result['cd']:.4f} (ref {result['cd_ref']:.4f}) "
                      f"err={result['cd_err_pct']:.1f}%")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results.append({"case": case, "combo": combo_name, "error": str(e)})

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY: Bluff-Body Cross-Test Matrix")
    print(f"  {'='*80}")
    print(f"  {'Case':<16} {'Combo':<14} {'Cd_sim':>8} {'Cd_ref':>8} {'Err%':>8}")
    print(f"  {'-'*56}")
    for r in results:
        if "error" in r:
            print(f"  {r['case']:<16} {r['combo']:<14} {'FAILED':>26}")
        else:
            print(f"  {r['case']:<16} {r['combo']:<14} "
                  f"{r['cd']:8.4f} {r['cd_ref']:8.4f} {r['cd_err_pct']:7.1f}%")
    print(f"  {'-'*56}")


if __name__ == "__main__":
    main()
