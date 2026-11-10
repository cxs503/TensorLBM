#!/usr/bin/env python3
"""DEM coupling verification of the drag closure (node P54-W3-N11, item 4).

A single sphere settles from rest in a quiescent fluid, integrated by the real
DEM solver of this repository (``tensorlbm.dem.dem_solver.DEMSolver``), not by
a hand-rolled ODE:

* ``drag_model="oseen_bridge"`` (this work) must reproduce the analytic
  terminal velocity of the same closure -- that checks the *coupling*
  (force -> acceleration -> velocity) rather than the closure algebra, which is
  covered by ``tests/test_drag_closure.py``;
* ``drag_model="stokes"`` is the model that ``m7441`` could justify.  Its error
  against the reference terminal velocity is the number that motivates the
  whole node.

Buoyancy is folded into the effective gravity
``g_eff = g (1 - rho_f/rho_p)`` because ``DEMSolver`` applies the full particle
mass times ``config.gravity`` and has no separate buoyancy term; this is the
standard DEM reduction and is stated in the report.

Outputs ``dem_verification.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tensorlbm.dem.dem_solver import DEMConfig, DEMSolver  # noqa: E402
from tensorlbm.dem.particle import IceParticle  # noqa: E402
from tensorlbm.drag_closure import (  # noqa: E402
    cd_clift_gauvin,
    reynolds_number,
    terminal_velocity,
)

RHO_P, RHO_F, MU, G = 2650.0, 1000.0, 1.0e-3, 9.81
DIAMETERS = (1.0e-4, 1.5e-4, 2.0e-4, 3.0e-4, 4.0e-4, 5.0e-4)


def settle(d: float, model: str, n_steps: int = 40000) -> dict:
    """Settle one sphere with the DEM solver; return the measured terminal speed."""
    g_eff = G * (1.0 - RHO_F / RHO_P)
    a = 0.5 * d
    tau_stokes = 2.0 * RHO_P * a * a / (9.0 * MU)
    dt = tau_stokes / 50.0
    cfg = DEMConfig(
        time_step=dt,
        gravity=torch.tensor([0.0, 0.0, -g_eff], dtype=torch.float64),
        enable_wall_contacts=False,
        drag_model=model,
        fluid_density=RHO_F,
        fluid_viscosity=MU,
    )
    solver = DEMSolver(cfg)
    particle = IceParticle(
        position=torch.zeros(3, dtype=torch.float64),
        velocity=torch.zeros(3, dtype=torch.float64),
        angular_velocity=torch.zeros(3, dtype=torch.float64),
        radius=a,
        density=RHO_P,
    )
    solver.initialize([particle])
    quiet = torch.zeros(3, dtype=torch.float64)
    # integrate until the speed stops changing (or the step budget is spent)
    u_prev = 0.0
    reached_at = None
    for n in range(1, n_steps + 1):
        solver.step(quiet)
        u = -float(particle.velocity[2])
        if n % 200 == 0:
            if abs(u - u_prev) <= 1e-10 * max(u, 1e-30) and u > 0.0:
                reached_at = n
                break
            u_prev = u
    return {
        "d_m": d,
        "model": model,
        "dt_s": dt,
        "n_steps": solver.step_count,
        "t_end_s": solver.time,
        "u_terminal_dem": -float(particle.velocity[2]),
        "converged": reached_at is not None,
    }


def main() -> int:
    rows = []
    for d in DIAMETERS:
        u_ref = terminal_velocity(d, RHO_P, RHO_F, MU, G, model="oseen_bridge")
        u_cg = terminal_velocity(d, RHO_P, RHO_F, MU, G, model="clift_gauvin")
        rec_b = settle(d, "oseen_bridge")
        rec_s = settle(d, "stokes")
        Re = reynolds_number(0.5 * d, u_ref, RHO_F, MU)
        rows.append({
            "d_m": d,
            "Re_terminal": Re,
            "cd_reference": cd_clift_gauvin(Re),
            "u_reference_bridge": u_ref,
            "u_reference_clift_gauvin": u_cg,
            "u_dem_bridge": rec_b["u_terminal_dem"],
            "u_dem_stokes": rec_s["u_terminal_dem"],
            "dem_vs_analytic_pct": 100.0 * (rec_b["u_terminal_dem"] - u_ref) / u_ref,
            "stokes_model_err_pct": 100.0 * (rec_s["u_terminal_dem"] - u_cg) / u_cg,
            "bridge_model_err_pct": 100.0 * (u_ref - u_cg) / u_cg,
            "converged": rec_b["converged"] and rec_s["converged"],
            "n_steps": [rec_b["n_steps"], rec_s["n_steps"]],
            "dt_s": rec_b["dt_s"],
        })
        print(json.dumps(rows[-1]), flush=True)

    in_band = [r for r in rows if 1.0 <= r["Re_terminal"] <= 50.0]
    summary = {
        "parameters": {"rho_p": RHO_P, "rho_f": RHO_F, "mu": MU, "g": G},
        "rows": rows,
        "re_band": [r["Re_terminal"] for r in in_band],
        "max_abs_dem_vs_analytic_pct": max((abs(r["dem_vs_analytic_pct"]) for r in rows), default=None),
        "max_abs_stokes_err_pct_in_band": max((abs(r["stokes_model_err_pct"]) for r in in_band), default=None),
        "max_abs_bridge_err_pct_in_band": max((abs(r["bridge_model_err_pct"]) for r in in_band), default=None),
        "all_converged": all(r["converged"] for r in rows),
    }
    (_HERE.parent / "dem_verification.json").write_text(json.dumps(summary, indent=1))
    print("summary:", json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())