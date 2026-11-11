"""Run ledger for p223w2b-n01: what was actually executed, and what it measured.

Every number in the campaign report comes from the JSON this script writes, so
nothing is hand-copied.  Three entries:

``L``      laminar verification anchor of the new spectral engine
           (forced channel vs the exact Poiseuille parabola);
``T180``   production-size turbulent attempt at Re_tau = 180 (the run that
           decides whether the engine can deliver DNS statistics at all);
``COST``   measured throughput and the resulting cost model for the three
           nominal Reynolds numbers.

Usage::

    PYTHONPATH=src python benchmarks/p223w2b_n01_channel_turb/run_ledger.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.spectral_channel import (  # noqa: E402
    SpectralChannel,
    SpectralChannelConfig,
    module_sha256,
    poiseuille_exact,
)

OUT = ROOT / "artifacts/p223w2b_n01_channel_turb/run_ledger.json"


def _finite(solver: SpectralChannel) -> bool:
    return bool(np.isfinite(solver.u).all())


def laminar_anchor(init_profile: str = "parabola", label: str = "L") -> dict:
    """Forced channel at Re_tau = 10: fixed point and relaxation to the parabola."""
    cfg = SpectralChannelConfig(
        nx=16, ny=33, nz=16, lx=2 * math.pi, lz=math.pi, h=1.0, re_tau=10.0,
        u_tau=1.0, dt=5.0e-3, init_profile=init_profile, noise=0.0, stat_every=10**9,
        label=label,
    )
    solver = SpectralChannel(cfg)
    exact = poiseuille_exact(solver.y, cfg.height, cfg.accel, cfg.nu)
    info = solver.run(4000)
    profile = solver.mean_profile()
    tau_exact = cfg.accel * cfg.h
    tau_rel = (info["wall_shear"] - tau_exact) / tau_exact
    return {
        "case": label,
        "config": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz, "re_tau": cfg.re_tau,
                   "dt": cfg.dt, "steps": 4000, "init": init_profile, "noise": 0.0},
        "linf_profile_error_rel": float(np.abs(profile - exact).max() / exact.max()),
        "wall_shear": info["wall_shear"],
        "wall_shear_expected": tau_exact,
        "wall_shear_rel_error": tau_rel,
        "divergence_max": info["divergence_max"],
        "max_speed": info["max_speed"],
        "throughput_mcell_steps_per_s": info["mcell_steps_per_s"],
        "status": "ok" if abs(tau_rel) < 0.02 else "fail",
    }


def turbulent_attempt(
    label: str,
    nx: int,
    ny: int,
    nz: int,
    re_tau: float,
    dt: float,
    max_steps: int,
    noise: float = 0.05,
) -> dict:
    """Long-window turbulent attempt; records the blow-up step if it dies."""
    cfg = SpectralChannelConfig(
        nx=nx, ny=ny, nz=nz, lx=2 * math.pi, lz=math.pi, re_tau=re_tau, dt=dt,
        noise=noise, init_profile="log", stat_every=10**9, label=label,
    )
    solver = SpectralChannel(cfg)
    metrics = cfg.grid_metrics()
    t0 = time.time()
    div_hist = []
    step = 0
    for step in range(1, max_steps + 1):
        solver.step()
        if step % 10 == 0:
            div_hist.append((step, solver.divergence()))
        if not _finite(solver):
            break
    wall = time.time() - t0
    cells = nx * ny * nz
    alive = _finite(solver)
    mean = solver.mean_profile() if alive else np.full(ny, np.nan)
    return {
        "case": label,
        "config": {"nx": nx, "ny": ny, "nz": nz, "re_tau": re_tau, "dt": dt,
                   "noise": noise, "max_steps": max_steps,
                   "intervals_per_turnover": 1.0 / dt},
        "grid_metrics": metrics,
        "steps_completed": step if alive else step - 1,
        "blowup": (not alive),
        "blowup_time": (step - 1) * dt if not alive else None,
        "status": "ok" if alive else "diverged",
        "divergence_history": div_hist,
        "max_mean_speed": float(np.abs(mean).max()) if alive else None,
        "wall_shear": solver.wall_shear() if alive else None,
        "throughput_mcell_steps_per_s": cells * step / wall / 1e6,
        "wall_seconds": wall,
    }


def cost_model(throughput_mcell: float) -> dict:
    """Steps and wall time needed for N turnover times at (lx,lz) = (2pi,pi)."""
    rows = []
    for target, dt in ((180.0, 2.0e-4), (395.0, 1.2e-4), (590.0, 8.0e-5)):
        nx = nz = int(round(64 * target / 180.0 / 16.0)) * 16
        ny = 97 if target <= 400 else 129
        cells = nx * ny * nz
        turns = 20.0
        steps = int(round(turns / dt))
        rows.append({
            "re_tau": target,
            "nx_nz": nx,
            "ny": ny,
            "cells": cells,
            "dx_plus": 2 * math.pi / nx * target,
            "dz_plus": math.pi / nz * target,
            "dt": dt,
            "turnovers": turns,
            "steps": steps,
            "wall_hours_at_measured_throughput": cells * steps / (throughput_mcell * 1e6) / 3600.0,
        })
    return {
        "throughput_mcell_steps_per_s_measured": throughput_mcell,
        "assumption": "20 large-eddy turnover times, one run, single job",
        "rows": rows,
    }


def main() -> dict:
    lam = laminar_anchor("parabola", "L_fixed_point")
    lam2 = laminar_anchor("log", "L_relax_from_log_profile")
    turb = turbulent_attempt("T180", 64, 97, 64, 180.0, 2.0e-4, 3000)
    throughput = max(turb["throughput_mcell_steps_per_s"], lam["throughput_mcell_steps_per_s"])
    payload = {
        "schema": "tensorlbm-turbulent-channel-run-ledger-v1",
        "engine": "tensorlbm.turbulence.spectral_channel",
        "engine_sha256": module_sha256(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cases": [lam, lam2, turb],
        "cost_model": cost_model(throughput),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    OUT.write_text(text + "\n")
    payload["artifact_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    print(json.dumps({"laminar": {k: lam[k] for k in
                                  ("linf_profile_error_rel", "wall_shear", "wall_shear_rel_error", "divergence_max")},
                      "laminar_relax": {k: lam2[k] for k in
                                  ("linf_profile_error_rel", "wall_shear", "wall_shear_rel_error")},
                      "turbulent": {k: turb[k] for k in ("steps_completed", "blowup", "status", "throughput_mcell_steps_per_s")},
                      "cost_rows": payload["cost_model"]["rows"]}, indent=1))
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()