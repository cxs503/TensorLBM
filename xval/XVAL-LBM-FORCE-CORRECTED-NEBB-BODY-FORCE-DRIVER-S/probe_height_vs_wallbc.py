"""Probe: is ``GPUCorePathConfig.height`` consistent with ``wall_bc='nebb'``?

The half-way bounce-back closure puts the no-slip planes at y = 0.5 and
y = ny - 1.5, so the channel width is ``ny - 2`` (what ``height`` returns).
The wall-node NEBB closure puts the planes *on* the wall rows y = 0 and
y = ny - 1, so the width is ``ny - 1``.  ``body_force`` derives the driving
from ``height``, so if ``height`` ignores ``wall_bc`` the NEBB arm is driven
to ``u_max_target * ((ny-1)/(ny-2))**2`` instead of ``u_max_target``.

This probe measures the ratio directly (no reference profile involved):
ratio = measured centre-line u / u_max_target for wall_bc in (halfway, nebb).
"""

from __future__ import annotations

import json
import sys

from tensorlbm.gpu_core_path import GPUCorePathConfig, GPUCorePathSolver

ROWS = []
for ny in (10, 18, 34):
    for wbc in ("halfway", "nebb"):
        cfg = GPUCorePathConfig(
            nx=8, ny=ny, nz=4, tau=1.1, u_max_target=0.05, init="rest",
            precision="FP64FP64", backend="eager", steps=60000,
            wall_bc=wbc, label=f"probe_{wbc}_{ny}",
        )
        solver = GPUCorePathSolver(cfg)
        solver.run(cfg.steps)
        _, u = solver.channel_profile()
        umax = float(u.max())
        # prediction from the mismatch of the width used for the drive
        pred = ((ny - 1) / (ny - 2)) ** 2 if wbc == "nebb" else 1.0
        ROWS.append({
            "ny": ny, "wall_bc": wbc, "height_cfg": cfg.height,
            "body_force": cfg.body_force, "u_max_target": cfg.u_max_target,
            "u_max_got": umax, "ratio_got": umax / cfg.u_max_target,
            "ratio_pred_width_mismatch": pred,
        })
        print(ROWS[-1], flush=True)

json.dump(ROWS, open(sys.argv[1] if len(sys.argv) > 1 else "probe_height.json", "w"), indent=2)