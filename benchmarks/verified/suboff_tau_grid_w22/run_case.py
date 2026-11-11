#!/usr/bin/env python3
"""W22 tau x grid matrix case runner — SUBOFF bare hull (TensorLBM).

Research question (mission m14505, successor of paper-exp-lbm-suboff-tau-6):
    Can the LBM relaxation time tau be decoupled from the grid resolution for
    SUBOFF bare-hull drag, and on which side of the (tau, N) plane does
    negative drag appear?

Design (preregistered in ``prereg.json``, frozen before any case ran)
---------------------------------------------------------------------
* geometry / force pipeline: **imported** from the frozen verified driver
  ``benchmarks/verified/suboff_sail_resistance/run.py`` (``SuboffAppendageEngine``
  + ``run_engine_routed``); this script does not re-implement the solver, the
  bounce-back, the pressure/friction integrals or the reference areas.
* factor 1: tau in {0.5005, 0.505, 0.510, 0.515, 0.520, 0.530}
  pinned exactly through the frozen ``_physical_viscosity_for_tau`` route
  (nu_lb = (tau-0.5)/3 reaches the solver; the achieved tau is read back).
* factor 2: N (cells per hull length) in {96, 144, 192} (ratio 1.5).
* one collision model for the whole matrix: MRT (``CollisionModel.MRT``), D3Q19,
  half-way bounce-back, ``force_method=PRESSURE_FRICTION``,
  ``friction_formula='standard'``, ``pressure_extrap='none'``,
  ``p0_method='near_wall'``, mass correction every 200 steps.
* one initial field for every case: uniform-equilibrium IC from
  ``GeneralSimEngine.setup()`` (identical construction path, u_lb from the
  converter) — no per-case warm start, no field carried between cases.
* identical step count for every case (``--steps``), and identical domain
  padding ratios (so the blockage is grid-independent).

Normalisation (frozen, same caliber as the anchor pack)
------------------------------------------------------
C_T = F_x / (0.5 rho u^2 S), S = pi*D*L (bare-hull wetted area, anchor caliber).
The engine samples forces with the **frontal** dpS = 0.5 u^2 pi R^2, so every
sample is rescaled by dpS_front/dpS_wet.  Both dpS values are written into the
raw npz, so downstream recomputation does not need to trust this file.

Outputs (per case, into ``--out``):
  case_<tag>.json   scalars: tau/N/steps/cells/C_T/C_F/C_P/drift/flags/timings
  case_<tag>.npz    raw force time series (frontal-normalised) + normalisation

Nothing in the frozen benchmark tree is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FROZEN = _REPO / "benchmarks" / "verified" / "suboff_sail_resistance" / "run.py"


def load_frozen_driver():
    spec = importlib.util.spec_from_file_location("frozen_suboff_driver", _FROZEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# preregistered production settings (identical for every case)
DOMAIN_PADDING = (0.25, 1.75, 0.35, 0.35, 0.35, 0.35)  # x-  x+  y-  y+  z-  z+
COLLISION = "mrt"
FRICTION_FORMULA = "standard"
P0_METHOD = "near_wall"
PRESSURE_EXTRAP = "none"
MASS_CORRECTION_INTERVAL = 200
FORCE_SAMPLE_INTERVAL = 5
EVAL_WINDOW = 40  # last N samples (=200 steps at interval 5) averaged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--resolution", type=int, required=True, help="cells per hull length L")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = leave default)")
    ap.add_argument("--out", default=str(_HERE / "raw"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    fd = load_frozen_driver()
    N = args.resolution
    tau_target = args.tau
    u_lb = 0.05  # frozen converter default (pinned by _physical_viscosity_for_tau docstring)

    # physical viscosity that makes (tau-0.5)/3 reach the solver
    nu_phys = fd._physical_viscosity_for_tau(tau_target, N, u_lb)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"tau{tau_target:.4f}_N{N}"

    cfg = fd.GeneralSimConfig(
        name=f"w22_suboff_tau{tau_target:.4f}_N{N}",
        geometry=fd.GeometryConfig(
            source=fd.GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=fd.SUBOFF_LENGTH_M,
            suboff_radius=fd.SUBOFF_RADIUS_M,
        ),
        physics=fd.PhysicsConfig(
            density=1000.0,
            viscosity=nu_phys,
            inlet_velocity=fd.U_PHYS,
            reference_length=fd.SUBOFF_LENGTH_M,
        ),
        solver=fd.SolverConfig(
            lattice=fd.LatticeModel.D3Q19,
            collision=fd.CollisionModel.MRT,
            resolution=N,
            domain_padding=DOMAIN_PADDING,
            max_steps=args.steps,
            warmup_steps=0,
            snapshot_interval=10_000_000,
            force_sample_interval=FORCE_SAMPLE_INTERVAL,
            device=args.device,
            wall_treatment=fd.WallTreatment.AUTO,
            force_method=fd.ForceMethod.PRESSURE_FRICTION,
            pressure_extrap=PRESSURE_EXTRAP,
            p0_method=P0_METHOD,
            friction_formula=FRICTION_FORMULA,
            mass_correction=True,
            mass_correction_interval=MASS_CORRECTION_INTERVAL,
        ),
        output=fd.OutputConfig(
            directory=str(out_dir / f"run_{tag}"),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    t_setup = time.time()
    engine = fd.SuboffAppendageEngine(cfg, hull_type="bare_hull")
    setup = engine.setup()
    setup_s = time.time() - t_setup

    # ---- run (frozen routed step loop) -------------------------------------
    # progress hook: wraps the frozen force sampler (solver path untouched)
    _orig_sample = engine._sample_forces
    _prog_every = int(os.environ.get("W22_PROGRESS_EVERY", "0") or 0)
    if _prog_every > 0:

        def _wrapped(dpS, nu_lb):
            _orig_sample(dpS, nu_lb)
            k = len(engine.forces_log)
            if k % _prog_every == 0:
                e = engine.forces_log[-1]
                print(
                    f"[prog] step={e['step']} cd_t={e['cd_total']:.8g} "
                    f"cd_p={e['cd_pressure']:.8g} cd_f={e['cd_friction']:.8g}",
                    flush=True,
                )

        engine._sample_forces = _wrapped

    t_run = time.time()
    info, elapsed = fd.run_engine_routed(engine, None)
    elapsed = time.time() - t_run

    log = engine.forces_log
    steps_done = int(info["steps"])

    # ---- normalisation (both calibers recorded) ---------------------------
    geo = cfg.geometry
    dx = cfg.physics.reference_length / N
    R_lb = geo.suboff_radius / dx
    L_lb = geo.suboff_length / dx
    dpS_front = 0.5 * u_lb**2 * math.pi * R_lb**2  # engine caliber
    dpS_wet = 0.5 * u_lb**2 * math.pi * (2.0 * R_lb) * L_lb  # anchor caliber pi*D*L
    rescale = dpS_front / dpS_wet

    cd_p_raw = np.array([e["cd_pressure"] for e in log], dtype=np.float64)
    cd_f_raw = np.array([e["cd_friction"] for e in log], dtype=np.float64)
    cd_t_raw = np.array([e["cd_total"] for e in log], dtype=np.float64)
    step_arr = np.array([e["step"] for e in log], dtype=np.int64)

    n_win = min(EVAL_WINDOW, len(log))
    win_p = cd_p_raw[-n_win:]
    win_f = cd_f_raw[-n_win:]
    win_t = cd_t_raw[-n_win:]
    C_P = float(win_p.mean() * rescale)
    C_F = float(win_f.mean() * rescale)
    C_T = float(win_t.mean() * rescale)

    # window drift: |mean(first half) - mean(second half)| / |mean(second half)|
    def drift(a: np.ndarray) -> float:
        n = len(a) // 2
        if n == 0:
            return float("nan")
        m1, m2 = a[:n].mean(), a[n:].mean()
        return float(abs(m1 - m2) / abs(m2)) if abs(m2) > 0 else float("inf")

    nu_lb = (tau_target - 0.5) / 3.0
    Re_lattice = u_lb * N / nu_lb  # Re actually simulated (N as characteristic length)

    case = {
        "tag": tag,
        "mission": "m14505",
        "benchmark": "suboff_tau_grid_w22",
        "hull_type": "bare_hull",
        "collision_model": COLLISION,
        "lattice": "D3Q19",
        "friction_formula": FRICTION_FORMULA,
        "p0_method": P0_METHOD,
        "pressure_extrap": PRESSURE_EXTRAP,
        "force_method": "PRESSURE_FRICTION",
        "domain_padding": list(DOMAIN_PADDING),
        "tau_target": tau_target,
        "tau_achieved": float(setup["tau"]),
        "nu_lb": float(setup["nu_lb"]),
        "u_lb": float(setup["u_lb"]),
        "N": N,
        "R_lb": R_lb,
        "L_lb": L_lb,
        "domain_lu": list(setup["domain_lu"]),
        "n_cells": int(setup["total_cells"]),
        "obstacle_cells": int(setup["obstacle_cells"]),
        "near_wall_cells": int(setup["near_wall_cells"]),
        "steps_requested": args.steps,
        "steps_done": steps_done,
        "force_samples": int(len(log)),
        "eval_window_samples": int(n_win),
        "eval_window_steps": int(n_win * FORCE_SAMPLE_INTERVAL),
        "C_T": C_T,
        "C_F": C_F,
        "C_P": C_P,
        "C_T_sum_check": C_F + C_P,
        "C_T_window_drift": drift(win_t),
        "C_F_window_drift": drift(win_f),
        "C_P_window_drift": drift(win_p),
        "negative_drag": bool(C_T < 0.0),
        "Re_sim": Re_lattice,
        "dpS_front": dpS_front,
        "dpS_wet": dpS_wet,
        "rescale_front_to_wet": rescale,
        "diverged": bool(info.get("diverged", False)),
        "setup_seconds": setup_s,
        "run_seconds": elapsed,
        "ms_per_step": 1000.0 * elapsed / max(steps_done, 1),
        "mlups": (setup["total_cells"] * steps_done / max(elapsed, 1e-9)) / 1e6,
        "device": args.device,
        "host": socket.gethostname(),
        "platform": sys.platform,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "provenance": {
            "frozen_driver": str(_FROZEN.relative_to(_REPO)),
            "frozen_driver_sha256": sha256_file(_FROZEN),
            "runner": str(Path(__file__).resolve().relative_to(_REPO)),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    npz_path = out_dir / f"case_{tag}.npz"
    np.savez_compressed(
        npz_path,
        step=step_arr,
        cd_pressure_frontal=cd_p_raw,
        cd_friction_frontal=cd_f_raw,
        cd_total_frontal=cd_t_raw,
        dpS_front=np.float64(dpS_front),
        dpS_wet=np.float64(dpS_wet),
        tau_target=np.float64(tau_target),
        N=np.int64(N),
        u_lb=np.float64(u_lb),
        nu_lb=np.float64(nu_lb),
        eval_window=np.int64(n_win),
    )
    case["raw_npz"] = str(npz_path.relative_to(_REPO))
    case["raw_npz_sha256"] = sha256_file(npz_path)
    json_path = out_dir / f"case_{tag}.json"
    json_path.write_text(json.dumps(case, indent=1, sort_keys=True))

    print(
        f"[case] {tag}: C_T={C_T:.10g} C_F={C_F:.10g} C_P={C_P:.10g} "
        f"neg={case['negative_drag']} drift={case['C_T_window_drift']:.4%} "
        f"cells={setup['total_cells']} {case['ms_per_step']:.1f} ms/step "
        f"({case['mlups']:.2f} MLUPS) tau_ach={setup['tau']:.8f}",
        flush=True,
    )
    print(f"[case] wrote {json_path} and {npz_path}", flush=True)


if __name__ == "__main__":
    main()