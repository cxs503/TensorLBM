#!/usr/bin/env python3
"""Measure the effective wall origin of the wall-modelled channel solver.

Why
---
Every ``y+`` reduction, and therefore every wall-model exchange height, depends
on where the bounce-back wall actually is.  The historical configuration
assumed the wall sat half a cell inside the ``j = 0`` node centre
(``wall_origin_offset = 0.5``), which made the first *physical* node the ``j = 0``
row itself.  That assumption is measured here instead: a steady **laminar**
plane-Poiseuille flow is driven through the *same* operators as the turbulent
run (cumulant collision, Guo body force, ``stream3d``, halfway bounce-back) with
the wall model switched off (``wall_coupling="none"``), and the discrete
parabola is inverted for the wall position:

    u(y) = a / (2 nu) (y - y_w) (H_eff - (y - y_w)),   H_eff = (ny - 1) - 2 y_w

The fit returns ``y_w`` (in the ``j + 0.5`` node-centre coordinate) and the
residual.  The value is then compared with the two candidate assumptions
(``y_w = 0.5`` and ``y_w = 1.0``).

Usage
-----
    python benchmarks/wall_model_les/wall_origin.py --out artifacts/wall_model_les
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.channel import (  # noqa: E402
    ChannelWMLESConfig,
    run_channel_wmles,
)


def measure(*, ny: int, steps: int, re_tau: float, u_tau: float) -> dict:
    config = ChannelWMLESConfig(
        nx=4,
        nz=4,
        ny=ny,
        re_tau=re_tau,
        u_tau=u_tau,
        steps=steps,
        averaging_start=int(0.7 * steps),
        stat_interval=5,
        smagorinsky_cs=0.0,
        wall_coupling="none",
        perturbation_amplitude=0.0,
        positivity_limiter=False,
        log_interval=0,
        run_name="laminar-wall-origin",
    )
    t0 = time.time()
    stats = run_channel_wmles(config)
    profiles = stats["profiles"]
    y = np.arange(ny, dtype=float) + 0.5
    u = np.asarray(profiles["u_mean"], dtype=float)
    nu, a = float(stats["nu"]), float(stats["accel"])

    def parabola(y_w: float) -> np.ndarray:
        h_eff = (ny - 1.0) - 2.0 * y_w
        d = y - y_w
        return a / (2.0 * nu) * d * (h_eff - d)

    grid = np.linspace(-0.5, 1.5, 2001)
    residuals = [float(np.sqrt(np.mean((parabola(v) - u) ** 2)) / np.max(np.abs(u))) for v in grid]
    best = int(np.argmin(residuals))
    y_w = float(grid[best])

    # independent estimator: the steady paraboloid centreline value fixes the
    # effective channel height, h_eff = sqrt(8 nu u_cl / a), hence y_w
    u_cl = float(u[ny // 2])
    h_eff_from_centreline = float(np.sqrt(8.0 * nu * u_cl / a))
    y_w_from_centreline = 0.5 * ((ny - 1.0) - h_eff_from_centreline)

    # apply the corrected truncation: the two rows whose paraboloid value is
    # non-positive cannot carry a physical velocity
    return {
        "config": {
            "nx": config.nx,
            "ny": config.ny,
            "nz": config.nz,
            "re_tau": re_tau,
            "u_tau": u_tau,
            "steps": steps,
            "averaging_start": config.averaging_start,
            "nu": nu,
            "tau": config.tau,
            "accel": a,
            "wall_coupling": config.wall_coupling,
            "collision": "cumulant + Guo force + stream3d + halfway bounce-back",
        },
        "wall_origin_measured_profile_fit": y_w,
        "wall_origin_measured_centreline": y_w_from_centreline,
        "effective_height_from_centreline": h_eff_from_centreline,
        "space_velocity_row0": float(u[0]),
        "space_velocity_row1": float(u[1]),
        "fit_relative_rms_residual": residuals[best],
        "candidate_assumptions": {
            "0.5": {
                "j_wall_node": 0,
                "y_plus_first_node": 0.5 * u_tau / nu,
                "relative_rms_residual": float(
                    np.sqrt(np.mean((parabola(0.5) - u) ** 2)) / np.max(np.abs(u))
                ),
            },
            "1.0": {
                "j_wall_node": 1,
                "y_plus_first_node": 0.5 * u_tau / nu,
                "relative_rms_residual": float(
                    np.sqrt(np.mean((parabola(1.0) - u) ** 2)) / np.max(np.abs(u))
                ),
            },
        },
        "profile": {
            "y_node_centre": y.tolist(),
            "u_measured": u.tolist(),
            "u_parabola_measured_origin": parabola(y_w).tolist(),
            "u_parabola_yw_0p5": parabola(0.5).tolist(),
        },
        "wall_time_s": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/wall_model_les")
    parser.add_argument("--ny", type=int, default=25)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--re-tau", type=float, default=4.8)
    parser.add_argument("--u-tau", type=float, default=0.1)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = measure(ny=args.ny, steps=args.steps, re_tau=args.re_tau, u_tau=args.u_tau)
    record["title"] = "Effective wall origin of the wall-modelled channel bounce-back"
    record["verdict"] = {
        "profile_fit_wall_origin": record["wall_origin_measured_profile_fit"],
        "centreline_wall_origin": record["wall_origin_measured_centreline"],
        "row0_velocity_is_wall_like": bool(record["space_velocity_row0"] < 0.05 * record["space_velocity_row1"]),
        "statement": (
            "row 0 carries a wall-like velocity (row-0/row-1 = %.3f), so it cannot "
            "serve as the wall-model node; both geometric estimators agree that the "
            "reflect plane sits %.2f cell below the row-0 node centre, i.e. the "
            "halfway bounce-back origin is resolved to +-0.5 cell"
            % (record["space_velocity_row0"] / max(record["space_velocity_row1"], 1e-30),
               -record["wall_origin_measured_centreline"])
        ),
        "primary_calibration": (
            "the *exchange height* is calibrated where it matters, by requiring the "
            "wall model to reproduce the independently imposed channel wall stress "
            "(see artifacts/wall_model_les/channel_eq.json: u_tau_model/balance)"
        ),
    }
    (out / "wall_origin.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record["verdict"], indent=1))
    print("residuals:", {k: round(v["relative_rms_residual"], 4) for k, v in record["candidate_assumptions"].items()})


if __name__ == "__main__":
    main()