#!/usr/bin/env python3
"""Probe: viscosity/`tau` response of the solver chain (the §5 finding).

Two independent measurements of the same defect:

1. **Planar Poiseuille channel** (exact analytic parabola) at three relaxation
   times: the *measured* peak velocity over the analytic value.  At `tau=3.5`
   the chain is accurate to a few percent; at `tau=10.1` the profile degenerates
   into a plug flow with a one-cell wall layer and the peak is 3.3x too high —
   i.e. the wall layer behaves as if the viscosity were ~1 instead of 4.8.
2. **Periodic settling cell** at three relaxation times (`probe.py`, logs):
   `U_t/U_Stokes` = 0.93 (tau 3.5) -> 1.36 (tau 5.3) -> 2.93 (tau 14.9).

Writes ``probe_tau.json``; ``make_report.py`` injects it into the report.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))

import torch

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d  # noqa: E402
from tensorlbm.d3q19 import C, W, equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, macroscopic3d, stream3d  # noqa: E402

CS2, CS4 = 1.0 / 3.0, 1.0 / 9.0


def poiseuille_peak(nu: float, L: int = 16, g: float = 1e-5, steps: int = 1500) -> dict:
    """Steady channel peak velocity vs the analytic ``g H^2/(8 nu)``."""
    tau = 0.5 + 3.0 * nu
    solid = torch.zeros(L, L, L, dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    f = equilibrium3d(
        torch.ones(L, L, L), torch.zeros(L, L, L), torch.zeros(L, L, L), torch.zeros(L, L, L)
    )
    c = C.float()
    w = W.float().view(19, 1, 1, 1)
    for _ in range(steps):
        rho, ux, uy, uz = macroscopic3d(f)
        fx = rho * g
        cu = (
            c[:, 0].view(19, 1, 1, 1) * ux
            + c[:, 1].view(19, 1, 1, 1) * uy
            + c[:, 2].view(19, 1, 1, 1) * uz
        )
        cF = c[:, 0].view(19, 1, 1, 1) * fx
        src = (1.0 - 0.5 / tau) * w * (cF / CS2 + cu * cF / CS4)
        fc = collide_bgk3d(f, tau) + src
        fc[:, solid] = f[:, solid]
        f = stream3d(bounce_back_cells_3d(fc, solid))
    _, u_prof, _, _ = macroscopic3d(f)
    prof = [float(u_prof[L // 2, y, L // 2]) for y in range(L)]
    h = L - 2
    analytic = g / (2.0 * nu) * (h / 2.0) ** 2
    # plug-flow indicator: wall-adjacent node over peak
    return {
        "nu": nu,
        "tau": tau,
        "peak_measured": max(prof),
        "peak_analytic": analytic,
        "ratio": max(prof) / analytic,
        "wall_node_over_peak": prof[1] / max(prof),
        "profile": prof,
    }


def parse_probe_logs() -> list[dict]:
    out = []
    for p in sorted(_HERE.parent.glob("probe_*.log")):
        txt = p.read_text()
        m = re.search(r"U_drift/U_stokes_0 = ([0-9.]+)", txt)
        n = re.search(r"nu=([0-9.]+) tau=([0-9.]+)", txt)
        hdr = re.search(r"R=([0-9.]+) L=([0-9.]+)", txt)
        if m and n:
            out.append(
                {
                    "log": p.name,
                    "R": float(hdr.group(1)) if hdr else None,
                    "L": int(float(hdr.group(2))) if hdr else None,
                    "nu": float(n.group(1)),
                    "tau": float(n.group(2)),
                    "U_sim_over_Stokes": float(m.group(1)),
                }
            )
    return out


def main() -> int:
    torch.set_num_threads(4)
    pois = [poiseuille_peak(nu) for nu in (1.0, 1.6, 3.2, 4.8)]
    cells = parse_probe_logs()
    out = {
        "poiseuille": pois,
        "settling_probes": cells,
        "note": "tau = 0.5 + 3 nu; H = L-2 with half-way walls",
    }
    (_HERE.parent / "probe_tau.json").write_text(json.dumps(out, indent=1))
    for p in pois:
        print(
            f"nu={p['nu']:g} tau={p['tau']:.1f} peak={p['peak_measured']:.4e} "
            f"analytic={p['peak_analytic']:.4e} ratio={p['ratio']:.3f}"
        )
    for c in cells:
        print("probe:", c["log"], f"nu={c['nu']:g}", f"tau={c['tau']:g}", c["U_sim_over_Stokes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())