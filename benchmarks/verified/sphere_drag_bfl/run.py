"""Sphere drag coefficient benchmark with BFL interpolated bounce-back.

Validates TensorLBM D3Q19 sphere flow against the Schiller–Naumann
empirical correlation across the full Reynolds-number range.

Re scan: 0.1, 1, 10, 50, 100, 200, 500, 1000
Grid convergence: D = 20, 40 lattice diameters

Schiller–Naumann (1933):
    C_D = 24/Re·(1 + 0.15·Re^0.687) + 0.42/(1 + 42500/Re^1.16)

Acceptance criteria:
    Re ≤ 1:    |C_D − C_D,ref| / C_D,ref ≤ 5 %
    1 < Re ≤ 100:  ≤ 8 %
    100 < Re ≤ 1000: ≤ 12 %
    Grid convergence (D=40 vs D=20, Re ≤ 100): ≤ 5 %

Usage::
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py --fast
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py --re 100 --diameter 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# TensorLBM imports
# --------------------------------------------------------------------------- #
from tensorlbm.general_sim import (
    CollisionModel,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
)

# --------------------------------------------------------------------------- #
# Schiller–Naumann correlation
# --------------------------------------------------------------------------- #

def schiller_naumann_cd(re: float) -> float:
    """Schiller–Naumann drag coefficient for a smooth sphere.

    C_D = 24/Re·(1 + 0.15·Re^0.687) + 0.42/(1 + 42500/Re^1.16)

    Valid for Re ≲ 3×10⁵.
    """
    term1 = 24.0 / re * (1.0 + 0.15 * re ** 0.687)
    term2 = 0.42 / (1.0 + 42500.0 / re ** 1.16)
    return term1 + term2


# --------------------------------------------------------------------------- #
# Case definitions
# --------------------------------------------------------------------------- #

RE_SCAN = [0.1, 1, 10, 50, 100, 200, 500, 1000]
DIAMETERS = [20, 40]


def make_case_config(
    re: float,
    diameter: int,
    *,
    lattice: str = "d3q19",
    max_steps: int = 6000,
    device: str = "cpu",
    output_dir: str = "/tmp/sphere_drag_bfl",
) -> GeneralSimConfig:
    """Create a GeneralSimConfig for a sphere drag case.

    Uses physical units: ρ=1000 kg/m³, ν=1e-6 m²/s, L=1 m (sphere diameter).
    The engine converts to lattice units automatically.
    """
    lat = LatticeModel.D3Q27 if lattice == "d3q27" else LatticeModel.D3Q19
    u_phys = re * 1e-6 / 1.0  # U = Re * ν / L

    tag = f"sphere_{lattice}_re{re:g}_D{diameter}"

    return GeneralSimConfig(
        name=tag,
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,  # diameter = 1 m
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=u_phys,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=lat,
            collision=CollisionModel.MRT,
            resolution=diameter,  # D lattice cells
            max_steps=max_steps,
            device=device,
            force_sample_interval=10,
            snapshot_interval=10**9,
            pressure_extrap="quadratic",
            mass_correction=True,
            mass_correction_interval=200,
            divergence_check_interval=50,
        ),
        output=OutputConfig(directory=f"{output_dir}/{tag}", save_macroscopic=False),
    )


def run_case(
    re: float,
    diameter: int,
    *,
    lattice: str = "d3q19",
    threads: int = 16,
    fast: bool = False,
) -> dict:
    """Run a single sphere drag case and return results."""
    torch.set_num_threads(threads)

    max_steps = 3000 if fast else 6000
    cfg = make_case_config(re, diameter, lattice=lattice, max_steps=max_steps)

    eng = GeneralSimEngine(cfg)
    t0 = time.time()
    setup = eng.setup()
    out = eng.run(max_steps)
    wall = time.time() - t0

    # Extract force history
    hist = [
        {k: e[k] for k in ("step", "cd_pressure", "cd_friction", "cd_total", "cl")}
        for e in eng.forces_log
    ]

    # Compute time-averaged Cd from the last 25% of steps
    tail = [e for e in hist if e["step"] > 0.75 * max_steps]
    prev = [e for e in hist if 0.5 * max_steps < e["step"] <= 0.75 * max_steps]

    if tail:
        mean_tail = sum(e["cd_total"] for e in tail) / len(tail)
    else:
        mean_tail = float("nan")

    if prev:
        mean_prev = sum(e["cd_total"] for e in prev) / len(prev)
        drift_pct = abs(mean_tail - mean_prev) / abs(mean_tail) * 100.0 if mean_tail else float("nan")
    else:
        mean_prev = float("nan")
        drift_pct = float("nan")

    mass_drift_ppm = (float(eng.f.sum().item()) - eng._initial_mass) / eng._initial_mass * 1e6

    cd_ref = schiller_naumann_cd(re)
    cd_err_pct = abs(mean_tail - cd_ref) / cd_ref * 100.0 if cd_ref > 0 else float("nan")

    return {
        "re": re,
        "diameter": diameter,
        "lattice": lattice,
        "cd": mean_tail,
        "cd_ref": cd_ref,
        "cd_err_pct": cd_err_pct,
        "cd_prev_mean": mean_prev,
        "plateau_drift_pct": drift_pct,
        "mass_drift_ppm": mass_drift_ppm,
        "diverged": out["diverged"],
        "finite": not out["diverged"],
        "tau": setup["tau"],
        "u_lb": setup["u_lb"],
        "nu_lb": setup["nu_lb"],
        "Ma": setup["Ma"],
        "obstacle_cells": setup["obstacle_cells"],
        "domain_lu": setup["domain_lu"],
        "wall_s": wall,
        "force_history": hist[-20:] if len(hist) > 20 else hist,
    }


def run_full_study(
    *,
    fast: bool = False,
    re_list: list[float] | None = None,
    diameter_list: list[int] | None = None,
    threads: int = 16,
    output_dir: str = "benchmarks/verified/sphere_drag_bfl",
) -> dict:
    """Run the full Re-scan and grid-convergence study."""
    re_list = re_list or RE_SCAN
    diameter_list = diameter_list or DIAMETERS

    if fast:
        re_list = [100.0]
        diameter_list = [40]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    t0 = time.time()
    for re_val in re_list:
        for diam in diameter_list:
            key = f"Re{re_val:g}_D{diam}"
            print(f"[{key}] Re={re_val} D={diam}")

            result = run_case(re_val, diam, threads=threads, fast=fast)
            all_results[key] = result
            summary_rows.append({
                "key": key,
                "re": re_val,
                "diameter": diam,
                "cd": result["cd"],
                "cd_ref": result["cd_ref"],
                "cd_err_pct": result["cd_err_pct"],
                "finite": result["finite"],
                "wall_s": result["wall_s"],
            })

            status = "✓" if result["finite"] else "✗"
            print(f"  {status} Cd={result['cd']:.4f}  ref={result['cd_ref']:.4f}  err={result['cd_err_pct']:.1f}%")

    elapsed = time.time() - t0

    # Grid convergence analysis
    grid_conv: dict[str, dict] = {}
    for re_val in re_list:
        if re_val > 100:
            continue
        k20 = f"Re{re_val:g}_D20"
        k40 = f"Re{re_val:g}_D40"
        if k20 in all_results and k40 in all_results:
            cd20 = all_results[k20]["cd"]
            cd40 = all_results[k40]["cd"]
            if cd20 and cd20 > 0:
                rel_change = abs(cd40 - cd20) / cd20 * 100.0
            else:
                rel_change = float("nan")
            grid_conv[f"Re{re_val:g}"] = {
                "cd_D20": cd20,
                "cd_D40": cd40,
                "rel_change_pct": rel_change,
                "pass_5pct": rel_change <= 5.0,
            }

    # Acceptance check
    acceptance: dict[str, bool] = {}
    for row in summary_rows:
        re_val = row["re"]
        err = row["cd_err_pct"]
        if re_val <= 1.0:
            acceptance[row["key"]] = err <= 5.0
        elif re_val <= 100:
            acceptance[row["key"]] = err <= 8.0
        else:
            acceptance[row["key"]] = err <= 12.0

    report = {
        "study": "sphere_drag_bfl",
        "elapsed_s": round(elapsed, 1),
        "summary": summary_rows,
        "grid_convergence": grid_conv,
        "acceptance": acceptance,
        "all_results": {k: {kk: vv for kk, vv in v.items() if kk != "force_history"}
                        for k, v in all_results.items()},
    }

    # Save JSON
    json_path = output_path / "results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults saved to {json_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sphere drag benchmark (BFL)")
    parser.add_argument("--fast", action="store_true", help="Quick validation (Re=100, D=40)")
    parser.add_argument("--re", type=float, nargs="+", help="Specific Re values")
    parser.add_argument("--diameter", type=int, nargs="+", help="Specific diameters")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads")
    parser.add_argument("--output-dir", default="benchmarks/verified/sphere_drag_bfl")
    args = parser.parse_args()

    run_full_study(
        fast=args.fast,
        re_list=args.re,
        diameter_list=args.diameter,
        threads=args.threads,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
