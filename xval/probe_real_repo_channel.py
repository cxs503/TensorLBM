"""Probe: run the REAL repo D3Q19 channel solver and report accuracy + wall slip.

This is the F1/F3 remediation probe for the board audit (m17963): the previous
XVAL driver was a hand-written D2Q9 numpy re-implementation that never called
the repo code.  Here we call the actual ``GPUCorePathSolver``
(``collide_guo_into`` + ``apply_channel_walls`` half-way bounce-back) and
measure, on the pre-declared *pointwise* metric, the error vs the analytic
Poiseuille profile and the wall slip (no-slip enforcement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.gpu_core_path import (
    GPUCorePathConfig,
    GPUCorePathSolver,
    channel_analytic_velocity,
)


def pointwise_rel_err(got, ref):
    """Pre-declared metric: max pointwise relative error (interior fluid rows)."""
    scale = np.maximum(np.abs(ref), 1e-12)
    return float(np.abs(got - ref).max() / scale.max())


def main():
    # Small grid, modest steps (light under high load).  ny includes 2 wall rows.
    cfg = GPUCorePathConfig(
        nx=8, ny=10, nz=8, tau=1.1, u_max_target=0.05, init="rest",
        precision="FP64FP64", backend="eager", steps=20000, label="xval_r2_probe",
    )
    solver = GPUCorePathSolver(cfg)
    solver.run(cfg.steps)

    # Plane-averaged streamwise profile on fluid rows (1 .. ny-2).
    y, u_got = solver.channel_profile()
    _, u_ref_all = channel_analytic_velocity(cfg)
    u_ref = u_ref_all[1 : cfg.ny - 1].to("cpu").numpy()

    u_got_np = u_got.numpy()
    u_max_ref = float(u_ref.max())
    u_max_got = float(u_got_np.max())

    # Wall slip: the fluid row adjacent to each wall (row 1 and row ny-2).
    # The analytic profile is 0 at the wall plane (y=0.5 / y=ny-1.5); the
    # first fluid row's value is the near-wall velocity.  No-slip means the
    # profile extrapolates to 0 at the wall plane.
    wall_slip_bottom = float(u_got_np[0])
    wall_slip_top = float(u_got_np[-1])

    res = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "round": 2,
        "integration": "REAL repo GPUCorePathSolver (collide_guo_into + apply_channel_walls)",
        "params": {
            "nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
            "tau": cfg.tau, "nu": cfg.nu, "u_max_target": cfg.u_max_target,
            "body_force": cfg.body_force, "steps": cfg.steps,
            "backend": solver.backend, "precision": cfg.precision,
        },
        "metrics": {
            "u_max_ref": u_max_ref,
            "u_max_got": u_max_got,
            "u_max_rel_err": abs(u_max_got - u_max_ref) / u_max_ref,
            "pointwise_max_rel_err_interior": pointwise_rel_err(u_got_np, u_ref),
            "profile_rel_l2": float(
                np.sqrt(np.mean((u_got_np - u_ref) ** 2)) / u_max_ref
            ),
            "wall_slip_bottom_fluid_row": wall_slip_bottom,
            "wall_slip_top_fluid_row": wall_slip_top,
            "wall_slip_pct_of_umax": max(wall_slip_bottom, wall_slip_top) / u_max_ref,
        },
        "y": y.tolist(),
        "u_got": u_got_np.tolist(),
        "u_ref": u_ref.tolist(),
    }
    out = Path(__file__).resolve().parent / "probe_real_repo_channel.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["metrics"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
