#!/usr/bin/env python3
"""P40-N09 campaign runner: statistical-vs-model decomposition of the u+ error.

The node inherits one number from P20-N01 -- the Re_tau = 180 u+ profile of the
LBM channel sits ``15.19 %`` (relative L2) away from Moser-Kim-Mansour (MKM)
-- and asks a *metrology* question: how much of that is finite-averaging
statistical error and how much is a systematic model / domain-length bias?

Design of the case matrix (all arms share one grid, one Re_tau, one u_tau, one
molecular viscosity, one seed policy, and differ in exactly one knob):

===============  ======================  ==========  ==========================
case             box                     C_s         role
===============  ======================  ==========  ==========================
``D4_cs0``       4 pi delta x 4/3 pi d   0.0         reference arm, SGS-off
``D2_cs0``       2 pi delta x 4/3 pi d   0.0         **domain arm** (shorter box)
``D4_cs01``      4 pi delta x 4/3 pi d   0.1         **SGS arm** (closure on)
``LAM_yw``       laminar twin           0.0         measures the wall origin y_w
===============  ======================  ==========  ==========================

``D4_cs0`` vs ``D2_cs0`` isolates the box length; ``D4_cs0`` vs ``D4_cs01``
isolates the sub-grid closure (negative control: with the closure correctly
wired, a C_s = 0.1 run *must* move the u+ profile by more than its own
confidence interval).  The laminar twin runs at the *same* molecular viscosity
(hence the same tau) but a tiny body force, so the measured wall origin y_w
belongs to the production wall format.

Every arm records, in addition to the legacy statistics, the raw material of
:mod:`tensorlbm.validation.statistical_error`:

* ``statistics.profile_batch_moment_means`` -- consecutive batch means of the
  plane-averaged moment profiles (the batch-means CI),
* ``wall_exchange.shear_series`` -- the wall-shear time series (the
  autocorrelation integral time ``T_int``).

The runner itself also writes ``<case>_meta.json`` with the timing ledger and the
``T_int``-based effective-sample ledger, so that a report can quote both without
recomputing anything.
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

RE_TAU = 180.0
U_TAU = 0.01  # lattice: Ma_max = 18 u_tau / cs ~ 0.31
NY = 24  # delta = (ny-2)/2 = 11 cells, dy+ = 16.4 (coarse but uniform)
# Averaging window in *delta/u_tau* turns; >= 3 is the guard of the CI gate.
WARM_TURNS = 6.0
AVG_TURNS = 32.0
BATCH_COUNT = 24
SHEAR_STRIDE = 5

# case -> box length (units of 2 pi delta), collision model, C_s, Re_tau
#
# ``natural_kbc`` is the production wall-resolved kernel and the only one that
# survives this near-wall stiffness (the cumulant arm diverges inside ~1e3 steps
# at tau <= 0.5025; see ``probe_stability_p40n09.py`` and the campaign report).
# The Smagorinsky closure is only available on the cumulant kernel, so the SGS
# separation arms use whatever stable cumulant operating point the probe finds.
CASES = {
    "D4_cs0": dict(box_factor=2.0, cs=0.0, model="natural_kbc", re_tau=180.0),
    "D2_cs0": dict(box_factor=1.0, cs=0.0, model="natural_kbc", re_tau=180.0),
    # SGS negative control: the cumulant kernel is the only one that carries the
    # Smagorinsky closure, and it diverges at the Re_tau = 180 near-wall
    # stiffness (probe_stability_p40n09.py: tau = 0.5018..0.5055 all blow up).
    # The separation is therefore executed at the highest cumulant-stable
    # operating point found by the probe -- Re_tau = 30, same grid, same u_tau --
    # which tests the *wiring* of C_s, not the physics of the Re_tau = 180 case.
    "SGS_cs0": dict(box_factor=2.0, cs=0.0, model="cumulant", re_tau=30.0),
    "SGS_cs01": dict(box_factor=2.0, cs=0.10, model="cumulant", re_tau=30.0),
}


def _geometry(box_factor: float, ny: int) -> tuple[int, int, int, float]:
    """Return ``(nx, ny, nz, delta)`` for a box ``box_factor * 2 pi delta`` long."""
    delta = 0.5 * (ny - 2)
    nx = int(round(box_factor * 2.0 * math.pi * delta))
    nz = int(round(4.0 * math.pi * delta / 3.0))
    return nx, ny, nz, delta


def turbulent_case(
    case: str,
    *,
    out_dir: Path,
    ny: int = NY,
    u_tau: float = U_TAU,
    re_tau: float = RE_TAU,
    steps: int | None = None,
    warm_steps: int | None = None,
    avg_steps: int | None = None,
    batch_count: int = BATCH_COUNT,
    threads: int = 64,
    device: str = "cpu",
    checkpoint_interval: int = 10**9,
    resume: bool = False,
) -> tuple[WallResolvedChannel3DConfig, dict]:
    spec = CASES[case]
    re_tau = float(spec.get("re_tau", re_tau))
    nx, ny, nz, delta = _geometry(spec["box_factor"], ny)
    turnover_steps = delta / u_tau  # delta / u_tau, the DNS "eddy turnover"
    if steps is None:
        warmup = int(round((warm_steps if warm_steps is not None else WARM_TURNS * turnover_steps)))
        averaging = int(
            round(avg_steps if avg_steps is not None else AVG_TURNS * turnover_steps)
        )
        steps = warmup + averaging
    else:
        steps = int(steps)
        warmup = int(steps // 5) if warm_steps is None else int(warm_steps)
    config = WallResolvedChannel3DConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        re_tau=float(re_tau),
        u_tau=float(u_tau),
        steps=steps,
        warmup_steps=warmup,
        sample_interval=5,
        report_interval=max(20, steps // 400),
        checkpoint_interval=int(checkpoint_interval),
        collision_model=spec.get("model", "natural_kbc"),
        smagorinsky_cs=float(spec["cs"]),
        compile_natural_kbc=False,
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=1.0,
        random_noise_fraction=0.5,
        spectral_mode_count=32,
        spectral_max_wavenumber=4,
        minimum_statistics_eddy_turnovers=1.0,
        stationarity_window_eddy_turnovers=2.0,
        seed=20261108,
        device=device,
        wall_origin=0.5,
        momentum_exchange_diagnostic=True,
        profile_batch_count=int(batch_count),
        wall_exchange_series_stride=SHEAR_STRIDE,
        resume=bool(resume),
        output=out_dir / f"{case}_result.json",
        checkpoint=out_dir / f"{case}_state.ckpt",
    )
    plan = {
        "case": case,
        "resume": bool(resume),
        "checkpoint_interval": int(checkpoint_interval),
        "box": f"{spec['box_factor']:.0f} x 2 pi delta",
        "smagorinsky_cs": float(spec["cs"]),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "cells": nx * ny * nz,
        "delta": delta,
        "dy_plus": re_tau / delta,
        "streamwise_over_delta": nx / delta,
        "spanwise_over_delta": nz / delta,
        "u_tau": float(u_tau),
        "nu": config.nu,
        "tau": config.tau,
        "steps": steps,
        "warmup_steps": warmup,
        "averaging_steps": steps - warmup,
        "eddy_turnover_steps": turnover_steps,
        "averaging_eddy_turnovers": (steps - warmup) / turnover_steps,
        "profile_batch_count": int(batch_count),
        "wall_exchange_series_stride": SHEAR_STRIDE,
        "threads": threads,
    }
    return config, plan


def laminar_twin(
    *,
    out_dir: Path,
    ny: int = NY,
    nu_target: float,
    u_tau_laminar: float = 3.0e-4,
    steps: int = 30000,
    warmup: int = 18000,
    threads: int = 8,
) -> tuple[WallResolvedChannel3DConfig, dict]:
    """Laminar Poiseuille at the **production** molecular viscosity (same tau)."""
    height = ny - 2
    re_tau = u_tau_laminar * 0.5 * height / nu_target
    config = WallResolvedChannel3DConfig(
        nx=8,
        ny=ny,
        nz=8,
        re_tau=re_tau,
        u_tau=u_tau_laminar,
        steps=steps,
        warmup_steps=warmup,
        sample_interval=10,
        report_interval=2000,
        checkpoint_interval=10**9,
        collision_model="cumulant",
        smagorinsky_cs=0.0,
        compile_natural_kbc=False,
        initialization_mode="coherent",
        perturbation_fraction=0.0,
        random_noise_fraction=0.0,
        spectral_mode_count=1,
        seed=20261108,
        device="cpu",
        wall_origin=0.5,
        momentum_exchange_diagnostic=True,
        output=out_dir / "LAM_yw_result.json",
        checkpoint=out_dir / "LAM_yw_state.ckpt",
    )
    plan = {
        "case": "LAM_yw",
        "ny": ny,
        "nu": config.nu,
        "tau": config.tau,
        "re_tau": re_tau,
        "u_tau": u_tau_laminar,
        "steps": steps,
        "warmup_steps": warmup,
        "threads": threads,
    }
    return config, plan


def wall_origin_from_laminar(config: WallResolvedChannel3DConfig, result: dict) -> dict:
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
    return measured


def _round(obj, digits: int = 8):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs_p40n09"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("LBM_THREADS", "64")))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warm-steps", type=int, default=None)
    parser.add_argument("--avg-steps", type=int, default=None)
    parser.add_argument("--ny", type=int, default=NY)
    parser.add_argument("--u-tau", type=float, default=U_TAU)
    parser.add_argument("--re-tau", type=float, default=RE_TAU)
    parser.add_argument("--batch-count", type=int, default=BATCH_COUNT)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10**9,
        help="steps between checkpoints; a finite value makes the arm restartable",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue from <case>_state.ckpt.  The batch partition is a pure "
            "function of the step index, so a resumed run lands its samples in "
            "exactly the batches an uninterrupted run would have used."
        ),
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case = args.case
    laminar = case == "LAM_yw"
    if laminar:
        probe_config, _ = turbulent_case(
            "D4_cs0",
            out_dir=args.out_dir,
            ny=args.ny,
            u_tau=args.u_tau,
            re_tau=args.re_tau,
            steps=100,
        )
        config, plan = laminar_twin(
            out_dir=args.out_dir,
            ny=args.ny,
            nu_target=probe_config.nu,
            threads=args.threads,
        )
    elif case in CASES:
        config, plan = turbulent_case(
            case,
            out_dir=args.out_dir,
            ny=args.ny,
            u_tau=args.u_tau,
            re_tau=args.re_tau,
            steps=args.steps,
            warm_steps=args.warm_steps,
            avg_steps=args.avg_steps,
            batch_count=args.batch_count,
            threads=args.threads,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
        )
    else:
        raise SystemExit(f"unknown case {case}")

    print(f"[p40n09] {case} plan={json.dumps(plan)}", flush=True)
    started = time.time()
    result = run_wall_resolved_channel3d(config)
    elapsed = time.time() - started
    cells = plan.get("cells", config.nx * config.ny * config.nz)

    meta: dict = {
        "node": "P40-N09",
        "case": case,
        "resumed_from_checkpoint": bool(args.resume),
        "plan": plan,
        "elapsed_s": elapsed,
        "cell_updates_per_s": cells * config.steps / max(elapsed, 1e-9),
        "configuration": {
            "nu": config.nu,
            "tau": config.tau,
            "u_tau": config.u_tau,
            "re_tau": config.re_tau,
            "smagorinsky_cs": config.smagorinsky_cs,
            "collision_model": config.collision_model,
        },
        "wall_exchange": {
            key: value
            for key, value in result["wall_exchange"].items()
            if key != "shear_series"
        },
        "acceptance": result["acceptance"],
        "statistics": {
            key: result["statistics"][key]
            for key in (
                "profile_samples",
                "statistics_steps",
                "eddy_turnover_steps",
                "statistics_eddy_turnovers",
                "stationarity_half_window_drift_fraction",
                "recent_crossflow_rms_over_u_tau",
            )
            if key in result["statistics"]
        },
    }

    series = result["wall_exchange"].get("shear_series") or []
    if len(series) >= 16:
        from tensorlbm.validation.statistical_error import integral_time

        ledger = integral_time(series, method="sokal")
        # convert the sample-based T_int into physical steps/turnovers
        stride = int(result["wall_exchange"]["shear_series_stride"])
        t_int_steps = ledger["t_int_samples"] * stride
        meta["t_int_ledger"] = {
            "signal": "wall_shear (bounce-back momentum exchange per unit wall area)",
            "stride_steps": stride,
            "t_int_samples": ledger["t_int_samples"],
            "t_int_steps": t_int_steps,
            "t_int_turnovers": t_int_steps / (plan.get("delta", 0.5 * (config.ny - 2)) / config.u_tau),
            "n_samples": ledger["n_samples"],
            "n_effective_samples": ledger["n_effective_samples"],
            "relative_standard_error_of_mean": ledger["relative_standard_error_of_mean"],
            "acf_window": ledger["acf_window"],
            "clamped_at_uncorrelated_floor": ledger["clamped_at_uncorrelated_floor"],
            "observed_steps": len(series) * stride,
            "t_obs_turnovers": len(series) * stride
            / (plan.get("delta", 0.5 * (config.ny - 2)) / config.u_tau),
        }
        meta["t_int_ledger"]["t_obs_over_t_int"] = (
            len(series) / max(ledger["t_int_samples"], 1.0e-9)
        )

    if laminar:
        meta["laminar_wall_origin"] = wall_origin_from_laminar(config, result)

    meta_path = args.out_dir / f"{case}_meta.json"
    meta_path.write_text(json.dumps(_round(meta), indent=2) + "\n", encoding="utf-8")
    print(f"[p40n09] {case} done in {elapsed:.1f}s -> {config.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())