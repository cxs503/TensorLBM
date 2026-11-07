#!/usr/bin/env python3
"""Field snapshot for the report figure: mid-plane speed + control surface.

Runs one settling cell to steady state and stores the mid-plane slices of
``u_x``, ``|u|`` and ``ρ`` plus the control-surface circle, so the report
figure is generated from a real field (no sketch).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE.parent))

import torch

from run import NU_LATTICE, TAU, guo_source  # noqa: E402
from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, macroscopic3d, stream3d  # noqa: E402


def main(R: float = 6.0, L: int = 48, re: float = 0.05, steps: int = 2500) -> int:
    torch.set_num_threads(8)
    centre = (L / 2.0, L / 2.0, L / 2.0)
    u_target = re * NU_LATTICE / (2.0 * R)
    solid = sphere_mask(L, L, L, *centre, R, device=torch.device("cpu"))
    v_fluid = float(L**3) - float(solid.sum())
    f_stokes = 6.0 * math.pi * NU_LATTICE * R * u_target
    g = f_stokes / v_fluid
    rho0 = torch.ones(L, L, L)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
    for _ in range(steps):
        rho, ux, uy, uz = macroscopic3d(f)
        fx = rho * g
        f_new = collide_bgk3d(f, TAU) + guo_source(rho, ux, uy, uz, fx, TAU)
        f_new[:, solid] = f[:, solid]
        f = stream3d(bounce_back_cells_3d(f_new, solid))
    rho, ux, uy, uz = macroscopic3d(f)
    speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
    j = L // 2
    snap = {
        "R": R,
        "L": L,
        "u_target": u_target,
        "g": g,
        "u_x_mid": ux[j].tolist(),
        "speed_mid": speed[j].tolist(),
        "rho_mid": rho[j].tolist(),
        "solid_mid": solid[j].to(torch.int).tolist(),
        "control_radius": R + 2.5,
    }
    (_HERE.parent / "field_snapshot.json").write_text(json.dumps(snap))
    print("wrote field_snapshot.json", L)
    return 0


if __name__ == "__main__":
    args = [float(v) if i < 3 else int(v) for i, v in enumerate(sys.argv[1:])]
    raise SystemExit(main(*args))