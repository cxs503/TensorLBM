#!/usr/bin/env python3
"""Measure the effective wall position (``y_w``) of the production channel wall BC.

Why this exists
---------------
Post-processing a wall-resolved channel profile needs the *effective* no-slip
position of the bounce-back wall.  The analysis pipeline used to assume the
halfway value ``y_w = 0.5``.  That assumption is a property of the numerical
scheme (collision operator, relaxation time, streaming/bounce-back ordering),
not of the physics, and it can be measured exactly with a laminar
plane-Poiseuille run: at low Reynolds number the discrete solution is a
parabola whose symmetry line and centreline value fix the effective half height
and therefore both wall positions.

This driver reproduces the production time loop (same kernels, same force
operator, same bounce-back call as
:mod:`tensorlbm.boundary.wall_resolved_channel3d`) but launches from the
analytic parabolic profile, so a laminar steady state is reached in a few
viscous times instead of a full diffusive transient.  It then reports
``y_w`` versus the assumed 0.5, for each requested relaxation time / collision
model.

Usage
-----
    python laminar_wall_origin.py --tau 0.50155 --steps 6000 --out out.json
    python laminar_wall_origin.py --tau-scan 0.5008 0.50155 0.505 0.55 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d
from tensorlbm.boundary.wall_model import guo_body_force_d3q19
from tensorlbm.chunked_collision import NaturalKBCCollisionExecutor, collide_in_z_chunks
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d_low_memory
from tensorlbm.solver3d import stream3d
from tensorlbm.validation.wall_origin import (
    fit_wall_origin_wall_stress,
    poiseuille_wall_origin,
    wall_offset_bias,
)


def run_laminar_channel(
    *,
    ny: int = 33,
    nx: int = 8,
    nz: int = 8,
    tau: float = 0.50155,
    body_force: float = 1.0e-9,
    steps: int = 8000,
    average_from: float = 0.6,
    sample_interval: int = 50,
    wall_offset_guess: float = 0.5,
    collision_model: str = "natural_kbc",
    seed: int = 20261106,
    device: str = "cpu",
    threads: int = 1,
) -> dict:
    """Run a laminar wall-resolved channel and report the effective wall position."""
    if tau <= 0.5:
        raise ValueError("tau must exceed 0.5 for positive viscosity")
    nu = (tau - 0.5) / 3.0
    torch.manual_seed(seed)
    # Tiny calibration grids: intra-op threading costs more than it saves.
    torch.set_num_threads(max(1, int(threads)))
    dev = torch.device(device)

    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    fluid = ~solid
    solid_q = solid.unsqueeze(0)

    # Analytic launch: parabola between the assumed wall positions, which the
    # discrete steady state then corrects to its own effective wall position.
    y_index = torch.arange(ny, device=dev, dtype=torch.float32)
    wall_distance = (y_index - wall_offset_guess).clamp_min(0.0)
    channel_height = (ny - 1) - 2.0 * wall_offset_guess
    ux_profile = (
        body_force / (2.0 * nu) * wall_distance * (channel_height - wall_distance)
    ).clamp_min(0.0)
    ux_profile[0] = 0.0
    ux_profile[-1] = 0.0
    ux = ux_profile.view(1, ny, 1).expand(nz, ny, nx).contiguous()
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    f = equilibrium3d(torch.ones_like(ux), ux, uy, uz)

    executor = NaturalKBCCollisionExecutor(compile_enabled=False)
    profile_sum = torch.zeros(ny, dtype=torch.float64)
    samples = 0
    start_average = int(average_from * steps)
    diagnostics: list[dict] = []
    for step in range(1, steps + 1):
        old = f
        if collision_model == "natural_kbc":
            collided = collide_in_z_chunks(
                f, lambda slab: executor(slab, tau), chunk_cells=262144
            )
        elif collision_model == "cumulant":
            collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown collision model {collision_model!r}")
        post = torch.where(solid_q, old, collided)
        rho, ux_now, uy_now, uz_now = macroscopic3d_low_memory(post)
        force_x = rho * body_force * fluid
        post = guo_body_force_d3q19(
            post,
            force_x,
            torch.zeros_like(force_x),
            torch.zeros_like(force_x),
            ux_now,
            uy_now,
            uz_now,
            direction_chunk_size=4,
        )
        streamed = stream3d(post)
        f = bounce_back_cells_3d(streamed, solid, f_pre=post)
        if step > start_average and step % sample_interval == 0:
            _, sx, _, _ = macroscopic3d_low_memory(f)
            profile_sum += sx.mean(dim=(0, 2)).to(torch.float64)
            samples += 1
        if step % max(steps // 10, 1) == 0:
            _, sx, _, _ = macroscopic3d_low_memory(f)
            plane = sx.mean(dim=(0, 2))
            diagnostics.append(
                {
                    "step": step,
                    "centreline_velocity": float(plane[ny // 2]),
                    "first_fluid_velocity": float(plane[1]),
                    "finite": bool(torch.isfinite(f).all().item()),
                }
            )

    if samples == 0:
        raise RuntimeError("no profile samples collected; increase steps")
    mean_profile = (profile_sum / samples).tolist()
    y = [float(i) for i in range(ny)]

    # Drop the two wall samples, which carry the bounce-back populations only.
    fluid_y = y[1:-1]
    fluid_u = mean_profile[1:-1]
    calibration = poiseuille_wall_origin(fluid_y, fluid_u, nu, body_force)
    # Steady laminar channel: mean momentum balance gives tau_w = rho a h exactly,
    # h = (ny-1)/2 between the *assumed* wall ordinates is only used to size the
    # stress; the estimator is a one-parameter intercept fit, so this is stable.
    half_height_ord = 0.5 * (fluid_y[-1] - fluid_y[0] + 2.0 * wall_offset_guess)
    wall_shear_balance = body_force * half_height_ord
    stress_fit = fit_wall_origin_wall_stress(
        fluid_y, fluid_u, nu, wall_shear_balance, n_cells=2
    )
    y_origin_parabola = calibration["wall_offset_from_first_sample"] + fluid_y[0]
    y_origin_stress = stress_fit.y_origin
    y_origin_measured = y_origin_stress
    calibration["wall_origin_estimators"] = {
        "parabola_global": y_origin_parabola,
        "wall_stress_local": y_origin_stress,
        "spread": abs(y_origin_stress - y_origin_parabola),
        "wall_stress_fit": stress_fit.as_dict(),
        "wall_shear_balance": wall_shear_balance,
    }

    # Sanity: the continuum parabola launched must be the asymptotic shape.
    residual = calibration["residual_rms"]
    centreline = calibration["centreline_velocity"]
    continuum_centreline = body_force * (0.5 * calibration["effective_height"]) ** 2 / (
        2.0 * nu
    )

    bias = wall_offset_bias(
        fluid_y,
        fluid_u,
        nu,
        max(math.sqrt(body_force * 0.5 * calibration["effective_height"]), 1.0e-30),
        y_origin_assumed=wall_offset_guess,
        y_origin_measured=y_origin_measured,
        y_plus_window=(0.0, 1.0e9),
    )

    return {
        "schema": "tensorlbm-laminar-wall-origin-calibration-v1",
        "case": "laminar plane Poiseuille, effective wall position of the production wall BC",
        "setup": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "tau": tau,
            "nu": nu,
            "body_force": body_force,
            "steps": steps,
            "average_from_step": start_average,
            "sample_interval": sample_interval,
            "collision_model": collision_model,
            "wall_bc": "post-streaming bounce-back (top/bottom), periodic x/z",
            "force": "Guo body force, a_x = body_force",
            "wall_offset_assumed": wall_offset_guess,
        },
        "measured": calibration,
        "assumed_vs_measured": {
            "y_w_assumed": wall_offset_guess,
            "y_w_measured": y_origin_measured,
            "delta_cells": y_origin_measured - wall_offset_guess,
            "delta_cells_parabola": y_origin_parabola - wall_offset_guess,
            "exceeds_0p1_cell_trigger": bool(
                abs(y_origin_measured - wall_offset_guess) > 0.1
            ),
            "u_plus_bias_if_assumed": bias,
        },
        "numerical_health": {
            "centreline_discrete": centreline,
            "centreline_continuum_parabola": continuum_centreline,
            "centreline_relative_error": (
                (centreline - continuum_centreline) / continuum_centreline
                if continuum_centreline
                else None
            ),
            "parabola_fit_residual_rms": residual,
            "profile_samples": samples,
            "diagnostics": diagnostics,
        },
        "profile": {"y": y, "u": mean_profile},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ny", type=int, default=33)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.50155)
    parser.add_argument("--tau-scan", type=float, nargs="+", default=None)
    parser.add_argument("--body-force", type=float, default=1.0e-9)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--collision-model", default="natural_kbc",
                        choices=("natural_kbc", "cumulant"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("laminar_wall_origin.json"))
    args = parser.parse_args()

    taus = args.tau_scan if args.tau_scan else [args.tau]
    results = []
    for tau in taus:
        result = run_laminar_channel(
            ny=args.ny,
            nx=args.nx,
            nz=args.nz,
            tau=tau,
            body_force=args.body_force,
            steps=args.steps,
            collision_model=args.collision_model,
            device=args.device,
            threads=args.threads,
        )
        results.append(result)
        measured = result["assumed_vs_measured"]
        print(
            f"tau={tau:.6f} y_w_measured={measured['y_w_measured']:.4f} "
            f"(assumed {measured['y_w_assumed']}) delta={measured['delta_cells']:+.4f} "
            f"centreline_err={result['numerical_health']['centreline_relative_error']:.3e}",
            flush=True,
        )
    payload = {
        "schema": "tensorlbm-laminar-wall-origin-calibration-scan-v1",
        "collision_model": args.collision_model,
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())