#!/usr/bin/env python3
"""IN31-L01 campaign runner: 3D D3Q19 body-force channel, wall-origin corrected.

Cases
-----
``T32`` / ``T48`` / ``T64``
    Turbulent plane channel, Re_tau = 180, D3Q19 + natural KBC, Guo body force,
    post-streaming half-way bounce-back.  The y-ladder 32/48/64 doubles
    ``N_y`` between ``T32`` and ``T64``; the streamwise/spanwise box is kept at
    the Moser-Kim-Mansour aspect ratio (4*pi*delta by 4*pi*delta/3), so the
    wall-normal resolution is the only varied parameter.
``R395``
    Same configured at Re_tau = 395 (coarse wall-normal resolution, no in-repo
    DNS profile: used only for the log-law/three-way closure, never scored
    against MKM data).
``LAM_cs<value>``
    Laminar plane Poiseuille (Re_tau = 2, Smagorinsky ladder) used to *measure*
    the effective wall position ``y_w`` of the production wall BC with
    :func:`tensorlbm.validation.wall_origin.poiseuille_wall_origin`.
``NC_laminar``
    Negative control: the same laminar case with the turbulence detector on; a
    correct detector has to report *no* sustained three-dimensional
    fluctuations.

Every case writes ``<out_dir>/<case>_result.json`` (module output) plus
``<out_dir>/<case>_meta.json`` (timing, closure, wall-origin measurement).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("TENSORLBM_SRC", "src"))

import torch  # noqa: E402

from tensorlbm.boundary.wall_resolved_channel3d import (  # noqa: E402
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)

RE_TAU_TURB = 180.0
# torch.compile(inductor) needs a C++ compiler; the compute nodes have none, so
# the eager natural-KBC kernel is the default here.
COMPILE_KBC = os.environ.get("LBM_COMPILE_KBC", "0") == "1"
U_TAU_TURB = 0.01  # lattice units: Ma_max = 20*u_tau/cs ~ 0.35, cost ~ 1/u_tau

# wall-normal ladder: N_y -> delta = (N_y - 2)/2 cells, dy+ = Re_tau/delta
LEVELS = {
    "T24": dict(ny=24, warm_turns=6.0, avg_turns=14.0, sample_interval=2),
    "T32": dict(ny=32, warm_turns=4.0, avg_turns=12.0, sample_interval=2),
    "T48": dict(ny=48, warm_turns=4.0, avg_turns=10.0, sample_interval=2),
    "R395": dict(ny=48, warm_turns=6.0, avg_turns=12.0, sample_interval=2, re_tau=395.0),
}

CS_LADDER = (0.0, 0.05, 0.10, 0.15, 0.20)


def _turbulent_nu(ny: int, u_tau: float, re_tau: float) -> float:
    return u_tau * 0.5 * (ny - 2) / re_tau


def _turbulent_config(
    name: str,
    *,
    re_tau: float,
    u_tau: float,
    out_dir: Path,
    threads: int,
    step_cap: int | None = None,
) -> tuple[WallResolvedChannel3DConfig, dict]:
    level = LEVELS[name]
    ny = int(level["ny"])
    height = ny - 2
    delta = 0.5 * height
    nx = int(round(4.0 * math.pi * delta))
    nz = int(round(4.0 * math.pi * delta / 3.0))
    eddy_turnover_steps = delta / u_tau
    if step_cap is not None:
        steps = int(step_cap)
        warmup = int(0.2 * steps)
    else:
        warmup = int(round(level["warm_turns"] * eddy_turnover_steps))
        steps = warmup + int(round(level["avg_turns"] * eddy_turnover_steps))
    config = WallResolvedChannel3DConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        re_tau=re_tau,
        u_tau=u_tau,
        steps=steps,
        warmup_steps=warmup,
        sample_interval=int(level["sample_interval"]),
        report_interval=max(50, steps // 200),
        checkpoint_interval=10**9,  # only the final checkpoint is written
        collision_model="natural_kbc",
        compile_natural_kbc=COMPILE_KBC,
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=1.0,
        random_noise_fraction=0.5,
        spectral_mode_count=32,
        spectral_max_wavenumber=4,
        minimum_statistics_eddy_turnovers=5.0,
        stationarity_window_eddy_turnovers=4.0,
        seed=20261106,
        device="cpu",
        wall_origin=0.5,
        momentum_exchange_diagnostic=True,
        output=out_dir / f"{name}_result.json",
        checkpoint=out_dir / f"{name}_state.ckpt",
    )
    plan = {
        "case": name,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "cells": nx * ny * nz,
        "delta": delta,
        "dy_plus": re_tau / delta,
        "streamwise_over_half_height": nx / delta,
        "spanwise_over_half_height": nz / delta,
        "eddy_turnover_steps": eddy_turnover_steps,
        "steps": steps,
        "warmup_steps": warmup,
        "averaging_eddy_turnovers": (steps - warmup) / eddy_turnover_steps,
        "threads": threads,
    }
    return config, plan


def _laminar_case(
    *,
    name: str,
    cs: float,
    out_dir: Path,
    threads: int,
    re_tau: float | None = None,
    u_tau: float = 3.0e-4,
    ny: int = 11,
    steps: int = 30000,
    warmup: int = 18000,
    collision_model: str = "cumulant",
    nu_target: float | None = None,
) -> tuple[WallResolvedChannel3DConfig, dict]:
    if re_tau is None:
        if nu_target is None:
            re_tau = 2.0
        else:
            # Match the molecular viscosity of the turbulent production run so
            # that the measured wall offset belongs to the *same* tau.
            re_tau = u_tau * 0.5 * (ny - 2) / nu_target
    config = WallResolvedChannel3DConfig(
        nx=8,
        ny=ny,
        nz=8,
        re_tau=re_tau,
        u_tau=u_tau,
        steps=steps,
        warmup_steps=warmup,
        sample_interval=10,
        report_interval=2000,
        checkpoint_interval=10**9,
        collision_model=collision_model,
        smagorinsky_cs=cs if collision_model == "cumulant" else 0.0,
        compile_natural_kbc=False,
        initialization_mode="coherent",
        perturbation_fraction=0.0,
        random_noise_fraction=0.0,
        spectral_mode_count=1,
        seed=20261106,
        device="cpu",
        wall_origin=0.5,
        momentum_exchange_diagnostic=True,
        output=out_dir / f"{name}_result.json",
        checkpoint=out_dir / f"{name}_state.ckpt",
    )
    plan = {
        "case": name,
        "ny": ny,
        "nu": config.nu,
        "tau": config.tau,
        "re_tau": re_tau,
        "u_tau": u_tau,
        "smagorinsky_cs": cs,
        "steps": steps,
        "warmup_steps": warmup,
        "threads": threads,
    }
    return config, plan


def _laminar_wall_origin(config: WallResolvedChannel3DConfig, result: dict) -> dict:
    from tensorlbm.validation.wall_origin import poiseuille_wall_origin

    profile = result["statistics"]["mean_velocity_profile"]
    rows = list(range(1, config.ny - 1))
    measured = poiseuille_wall_origin(
        y=rows,
        u=[float(profile[row]) for row in rows],
        nu=config.nu,
        body_force=config.body_force_acceleration,
    )
    measured["wall_origin_assumed"] = float(config.wall_origin)
    measured["deviation_cells"] = measured["wall_position_low"] - 0.5
    measured["wall_units_shift_at_first_node"] = (
        measured["deviation_cells"] * config.u_tau / config.nu
    )
    return measured


def _round(obj, digits: int = 6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    return obj


def _compact(result: dict, meta: dict) -> dict:
    stats = result.get("statistics", {})
    return _round(
        {
            "case": meta["case"],
            "plan": meta["plan"],
            "elapsed_s": meta["elapsed_s"],
            "nu": result.get("derived", {}).get("nu"),
            "tau": result.get("derived", {}).get("tau"),
            "u_tau_nominal": result.get("configuration", {}).get("u_tau"),
            "re_tau": result.get("configuration", {}).get("re_tau"),
            "wall_origin_assumed": result.get("configuration", {}).get("wall_origin"),
            "mean_velocity_profile": stats.get("mean_velocity_profile"),
            "reynolds_stress_profiles_uu_vv_ww_uv": stats.get(
                "reynolds_stress_profiles_uu_vv_ww_uv"
            ),
            "statistics_eddy_turnovers": stats.get("statistics_eddy_turnovers"),
            "profile_samples": stats.get("profile_samples"),
            "crossflow_rms_over_u_tau": stats.get("recent_crossflow_rms_over_u_tau"),
            "wall_exchange": result.get("wall_exchange"),
            "wall_units": result.get("wall_units"),
            "acceptance": result.get("acceptance"),
            "laminar_wall_origin": meta.get("laminar_wall_origin"),
        },
        6,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs_campaign"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("LBM_THREADS", "64")))
    parser.add_argument("--steps", type=int, default=None, help="override step count (probe)")
    parser.add_argument("--u-tau", type=float, default=U_TAU_TURB)
    parser.add_argument("--re-tau", type=float, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case = args.case
    if case in LEVELS:
        re_tau = args.re_tau if args.re_tau is not None else LEVELS[case].get(
            "re_tau", RE_TAU_TURB
        )
        config, plan = _turbulent_config(
            case,
            re_tau=float(re_tau),
            u_tau=float(args.u_tau),
            out_dir=args.out_dir,
            threads=args.threads,
            step_cap=args.steps,
        )
        laminar = False
    elif case.startswith("LAM") or case == "NC_laminar":
        if case == "LAM_kbc":
            config, plan = _laminar_case(
                name=case,
                cs=0.0,
                out_dir=args.out_dir,
                threads=args.threads,
                steps=args.steps or 30000,
                collision_model="natural_kbc",
                nu_target=_turbulent_nu(LEVELS["T32"]["ny"], args.u_tau, RE_TAU_TURB),
            )
        elif case == "NC_laminar":
            config, plan = _laminar_case(
                name=case,
                cs=0.0,
                out_dir=args.out_dir,
                threads=args.threads,
                steps=args.steps or 20000,
                warmup=int((args.steps or 20000) // 2),
                re_tau=2.0,
            )
        elif case.startswith("LAM_nu"):
            config, plan = _laminar_case(
                name=case,
                cs=0.0,
                out_dir=args.out_dir,
                threads=args.threads,
                steps=args.steps or 30000,
                collision_model="natural_kbc",
                nu_target=float(case.split("nu", 1)[1]) * 1.0e-4,
            )
        else:
            cs = float(case.split("cs", 1)[1].replace("p", "."))
            config, plan = _laminar_case(
                name=case,
                cs=cs,
                out_dir=args.out_dir,
                threads=args.threads,
                steps=args.steps or 30000,
                nu_target=_turbulent_nu(LEVELS["T32"]["ny"], args.u_tau, RE_TAU_TURB),
            )
        laminar = True
    else:
        raise SystemExit(f"unknown case {case}")

    print(f"[campaign] {case} plan={json.dumps(plan)}", flush=True)
    started = time.time()
    result = run_wall_resolved_channel3d(config)
    elapsed = time.time() - started

    meta: dict = {
        "case": case,
        "plan": plan,
        "elapsed_s": elapsed,
        "cell_updates_per_s": (
            plan.get("cells", plan.get("nx", 8) * plan.get("ny", 13) * 8)
            * plan["steps"]
            / max(elapsed, 1e-9)
        ),
        "wall_exchange": result.get("wall_exchange"),
        "wall_units": result.get("wall_units"),
        "acceptance": result.get("acceptance"),
        "physical_validation": result.get("physical_validation"),
    }
    if laminar:
        meta["laminar_wall_origin"] = _laminar_wall_origin(config, result)
        meta["nu"] = config.nu
        meta["body_force"] = config.body_force_acceleration
    (args.out_dir / f"{case}_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"[campaign] {case} done in {elapsed:.1f}s", flush=True)
    print(
        "[SUMMARY] " + json.dumps(_compact(result, meta), separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())