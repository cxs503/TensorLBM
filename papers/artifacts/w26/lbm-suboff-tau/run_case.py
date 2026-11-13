#!/usr/bin/env python3
"""W26 / mission m17569 — SUBOFF bare-hull single-case runner (ladder + tau x N scan).

Successor of ``benchmarks/verified/suboff_tau_grid_w22/run_case.py`` (mission
m14505, which produced 18 MISSING cells because the 18-case matrix never got
compute).  This runner keeps the *frozen* case protocol and adds what this
mission needs:

  * fixed-Reynolds ladder mode (``--re-lattice``): the lattice Reynolds number
    is held fixed and tau follows from the grid, nu_lb = u_lb*N/Re, so a
    tau(N) change is not confounded with a Reynolds change.  This is the mode
    used for the three-grid GCI study.
  * explicit-tau mode (``--tau``): used for the (tau, N) negative-drag scan.
  * block-averaged force history + divergence flags, so time-convergence can be
    judged instead of assumed (the W22 calibration run was still drifting at
    800 steps).
  * optional mid-plane macroscopic snapshot (one ``.npz``) for the field figure.

Everything geometric/physical is *imported* from the frozen verified driver
``benchmarks/verified/suboff_sail_resistance/run.py`` (``SuboffAppendageEngine``,
``run_engine_routed``, ``_physical_viscosity_for_tau``); nothing in the frozen
tree is modified.  Protocol constants (collision model, force pipeline, domain
padding, u_lb, sampling cadence) are re-declared here and asserted against the
frozen driver at run time.

Normalisation (unchanged, anchor caliber)
-----------------------------------------
C_T = F_x / (0.5 rho u^2 S), S = pi*D*L (bare-hull wetted area).  The engine
samples with the frontal area, so each sample is rescaled by
dpS_front/dpS_wet; both dpS values are written into the raw npz so downstream
recomputation never has to trust this file.

Usage
-----
    python3 papers/artifacts/w26/lbm-suboff-tau/run_case.py \
        --resolution 48 --re-lattice 1000 --steps 5760 --threads 8 \
        --out <abs dir inside repo> --tag ladder_N48
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_REPO / "src"), str(_REPO / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FROZEN = _REPO / "benchmarks" / "verified" / "suboff_sail_resistance" / "run.py"

# ---- frozen protocol (copied 1:1 from the W22 preregistration) -------------
DOMAIN_PADDING = (0.25, 1.75, 0.35, 0.35, 0.35, 0.35)  # x-  x+  y-  y+  z-  z+
FRICTION_FORMULA = "standard"
P0_METHOD = "near_wall"
PRESSURE_EXTRAP = "none"
MASS_CORRECTION_INTERVAL = 200
FORCE_SAMPLE_INTERVAL = 5
U_LB = 0.05  # frozen converter default
X_EXTENT_IN_L = 3.0  # domain x length / hull length (0.25 + 1 + 1.75)


def load_frozen_driver():
    spec = importlib.util.spec_from_file_location("frozen_suboff_driver_w26", _FROZEN)
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


def block_means(a: np.ndarray, n_blocks: int = 10) -> list[float]:
    if len(a) == 0:
        return []
    idx = np.array_split(np.arange(len(a)), min(n_blocks, len(a)))
    return [float(a[i].mean()) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tau", type=float, help="explicit relaxation time (tau x N scan)")
    g.add_argument("--re-lattice", type=float, help="fixed lattice Re; tau derived (ladder)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, help="output dir (absolute, inside the repo)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--snapshot", action="store_true", help="dump mid-plane macroscopic slice")
    ap.add_argument("--mission", default="m17569")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))

    fd = load_frozen_driver()
    N = args.resolution
    u_lb = U_LB
    if args.re_lattice is not None:
        nu_lb_target = u_lb * N / args.re_lattice
        tau_target = 0.5 + 3.0 * nu_lb_target
        mode = "fixed_re_ladder"
        re_declared = args.re_lattice
    else:
        tau_target = args.tau
        nu_lb_target = (tau_target - 0.5) / 3.0
        mode = "explicit_tau"
        re_declared = u_lb * N / nu_lb_target

    # physical viscosity that makes nu_lb_target reach the solver (frozen route)
    nu_phys = fd._physical_viscosity_for_tau(tau_target, N, u_lb)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    cfg = fd.GeneralSimConfig(
        name=f"w26_suboff_{tag}",
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

    t_run = time.time()
    info, _ = fd.run_engine_routed(engine, None)
    run_s = time.time() - t_run

    log = engine.forces_log
    steps_done = int(info["steps"])
    finite = bool(torch.isfinite(engine.f).all().item())

    geo = cfg.geometry
    dx = cfg.physics.reference_length / N
    R_lb = geo.suboff_radius / dx
    L_lb = geo.suboff_length / dx
    dpS_front = 0.5 * u_lb**2 * math.pi * R_lb**2  # engine caliber
    dpS_wet = 0.5 * u_lb**2 * math.pi * (2.0 * R_lb) * L_lb  # anchor caliber pi*D*L
    rescale = dpS_front / dpS_wet

    cd_p = np.array([e["cd_pressure"] for e in log], dtype=np.float64)
    cd_f = np.array([e["cd_friction"] for e in log], dtype=np.float64)
    cd_t = np.array([e["cd_total"] for e in log], dtype=np.float64)
    step_arr = np.array([e["step"] for e in log], dtype=np.int64)

    n_win = min(40, len(log))  # last 200 steps at interval 5 (frozen eval window)
    C_P = float(cd_p[-n_win:].mean() * rescale) if n_win else float("nan")
    C_F = float(cd_f[-n_win:].mean() * rescale) if n_win else float("nan")
    C_T = float(cd_t[-n_win:].mean() * rescale) if n_win else float("nan")

    def drift(a: np.ndarray) -> float:
        n = len(a) // 2
        if n == 0:
            return float("nan")
        m1, m2 = a[:n].mean(), a[n:].mean()
        return float(abs(m1 - m2) / abs(m2)) if m2 != 0 else float("inf")

    # downstream recomputation path: derive from the *wetted* caliber directly
    C_T_wet = float((cd_t[-n_win:].mean() * dpS_front) / dpS_wet) if n_win else float("nan")
    C_F_wet = float((cd_f[-n_win:].mean() * dpS_front) / dpS_wet) if n_win else float("nan")

    ts_path = out_dir / f"timeseries_{tag}.npz"
    np.savez_compressed(
        ts_path,
        step=step_arr,
        cd_pressure_raw=cd_p,
        cd_friction_raw=cd_f,
        cd_total_raw=cd_t,
        dpS_front=np.float64(dpS_front),
        dpS_wet=np.float64(dpS_wet),
        u_lb=np.float64(u_lb),
        rho=np.float64(cfg.physics.density),
        R_lb=np.float64(R_lb),
        L_lb=np.float64(L_lb),
        sample_interval=np.int64(FORCE_SAMPLE_INTERVAL),
    )

    snap_path = None
    if args.snapshot:
        from tensorlbm.d3q19 import macroscopic3d  # noqa: PLC0415

        rho_f, ux, uy, uz = macroscopic3d(engine.f)
        nx, ny, nz = engine.f.shape[1:]
        j = ny // 2
        snap_path = out_dir / f"slice_{tag}.npz"
        np.savez_compressed(
            snap_path,
            ux_midplane=ux[:, j, :].detach().cpu().numpy(),
            uy_midplane=uy[:, j, :].detach().cpu().numpy(),
            rho_midplane=rho_f[:, j, :].detach().cpu().numpy(),
            solid_midplane=engine.solid[:, j, :].detach().cpu().numpy(),
            u_lb=np.float64(u_lb),
        )

    cells = int(setup["total_cells"])
    case = {
        "schema": "w26_suboff_case/v1",
        "mission": args.mission,
        "tag": tag,
        "mode": mode,
        "case_key": "suboff_bare_hull",
        "hull_type": "bare_hull",
        "lattice": "D3Q19",
        "collision_model": "MRT",
        "force_method": "PRESSURE_FRICTION",
        "friction_formula": FRICTION_FORMULA,
        "p0_method": P0_METHOD,
        "pressure_extrap": PRESSURE_EXTRAP,
        "mass_correction_interval": MASS_CORRECTION_INTERVAL,
        "force_sample_interval": FORCE_SAMPLE_INTERVAL,
        "domain_padding_in_L": list(DOMAIN_PADDING),
        "N": N,
        "cells": cells,
        "steps_requested": args.steps,
        "steps_done": steps_done,
        "convective_steps": X_EXTENT_IN_L * N / u_lb,
        "passes": float(steps_done / (X_EXTENT_IN_L * N / u_lb)),
        "u_lb": u_lb,
        "tau_target": tau_target,
        "tau_achieved": float(setup["tau"]),
        "nu_lb": float(setup.get("nu_lb", nu_lb_target)),
        "nu_lb_target": nu_lb_target,
        "nu_phys": float(nu_phys),
        "Re_lattice_declared": float(re_declared),
        "Re_lattice_measured": float(u_lb * N / float(setup.get("nu_lb", nu_lb_target))),
        "R_lb": float(R_lb),
        "L_lb": float(L_lb),
        "domain_lu": list(setup["domain_lu"]),
        "obstacle_cells": int(setup["obstacle_cells"]),
        "near_wall_cells": int(setup["near_wall_cells"]),
        "C_T": C_T,
        "C_F": C_F,
        "C_P": C_P,
        "C_T_wet_recompute": C_T_wet,
        "C_F_wet_recompute": C_F_wet,
        "C_T_sum_check": C_F + C_P,
        "C_T_window_drift": drift(cd_t[-n_win:]),
        "C_F_window_drift": drift(cd_f[-n_win:]),
        "negative_drag": bool(C_T < 0.0),
        "C_T_block_means": block_means(cd_t * rescale, 10),
        "C_F_block_means": block_means(cd_f * rescale, 10),
        "C_T_min": float(cd_t.min() * rescale) if len(cd_t) else float("nan"),
        "C_T_max": float(cd_t.max() * rescale) if len(cd_t) else float("nan"),
        "field_finite_at_end": finite,
        "compile_status": info.get("compile_status"),
        "wall_treatment_configured": str(getattr(cfg.solver.wall_treatment, "name", cfg.solver.wall_treatment)),
        "wall_treatment_effective": str(getattr(getattr(engine, "_auto_wall_treatment", None), "name",
                                                getattr(engine, "_auto_wall_treatment", None))),
        "solver_route": ("wall_function" if "wall-function path" in str(info.get("compile_status_reason", ""))
                         else "standard_compiled_step"),
        "wall_clock": {
            "setup_s": setup_s,
            "run_s": run_s,
            "cells_per_s": cells * steps_done / run_s if run_s > 0 else None,
        },
        "threads": args.threads,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "env": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "raw_npz": str(ts_path),
        "raw_npz_sha256": sha256_file(ts_path),
        "slice_npz": str(snap_path) if snap_path else None,
        "frozen_driver": {
            "path": str(_FROZEN.relative_to(_REPO)),
            "sha256": sha256_file(_FROZEN),
        },
    }
    if snap_path:
        case["slice_npz_sha256"] = sha256_file(snap_path)

    case_path = out_dir / f"case_{tag}.json"
    case_path.write_text(json.dumps(case, indent=1, ensure_ascii=False))
    print(
        f"[case] {tag} N={N} steps={steps_done} Re_lat={case['Re_lattice_measured']:.6g} "
        f"tau={case['tau_achieved']:.6f} C_T={C_T:.8g} C_F={C_F:.8g} "
        f"drift={case['C_T_window_drift']:.4g} run_s={run_s:.1f} "
        f"Mcells/s={cells * steps_done / max(run_s, 1e-9) / 1e6:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()