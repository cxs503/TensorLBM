#!/usr/bin/env python3
"""Mid-plane flow snapshot for the report figure (D=12, Re=10, L=36).

Runs the same case as ``run.py`` for a shorter window and stores one mid-plane
slice of the streamwise velocity together with the sphere mask, so the report
can show the actual field (not a sketch).  The numbers in the report come from
``results.json``; this file only feeds the contour figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, macroscopic3d, stream3d  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
import run as R  # noqa: E402


def main() -> int:
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    D, L, Re, steps = 12.0, 36, 10.0, 3000
    Rr = 0.5 * D
    nu = R.U_CHAR * D / Re
    tau = 0.5 + 3.0 * nu
    u_target = R.U_CHAR
    centre = (L / 2.0, L / 2.0, L / 2.0)
    solid = sphere_mask(L, L, L, centre[0], centre[1], centre[2], Rr, device=torch.device("cpu"))
    n_solid = int(solid.sum().item())
    v_fluid = float(L) ** 3 - n_solid
    a_vol = (3.0 * n_solid / (4.0 * np.pi)) ** (1.0 / 3.0)
    force_factor = 1.0 - 0.5 / tau
    from tensorlbm.drag_closure import cd_oseen_bridge
    cd0 = cd_oseen_bridge(Re)
    g = 0.5 * cd0 * np.pi * a_vol**2 * u_target**2 / (v_fluid * force_factor)
    rho0 = torch.ones(L, L, L)
    f = equilibrium3d(rho0, torch.full_like(rho0, u_target),
                      torch.zeros_like(rho0), torch.zeros_like(rho0))
    fluid = ~solid
    for step in range(1, steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        u_mean = float(ux[fluid].mean())
        g += 0.02 * (u_target - u_mean)
        f_new = collide_bgk3d(f, tau) + R.guo_source(rho, ux, uy, uz, rho * g, tau)
        f_new[:, solid] = f[:, solid]
        f = stream3d(bounce_back_cells_3d(f_new, solid))
        if step % 500 == 0:
            print(f"step={step} U={u_mean:.5f} g={g:.4e}", flush=True)
    rho, ux, uy, uz = macroscopic3d(f)
    mid = L // 2
    out = {
        "L": L,
        "D": D,
        "Re_target": Re,
        "nu": nu,
        "tau": tau,
        "g_final": g,
        "u_mean": float(ux[fluid].mean()),
        "a_vol": float(a_vol),
        "slice_axis": "z",
        "z_index": mid,
        "ux": ux[mid].cpu().numpy().tolist(),
        "uy": uy[mid].cpu().numpy().tolist(),
        "speed": torch.sqrt(ux[mid] ** 2 + uy[mid] ** 2).cpu().numpy().tolist(),
        "solid": solid[mid].cpu().numpy().astype(int).tolist(),
        "steps": steps,
    }
    path = _HERE.parent / "field_snapshot.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())