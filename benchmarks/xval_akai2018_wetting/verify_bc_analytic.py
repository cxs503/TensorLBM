#!/usr/bin/env python3
"""Analytic verification of the geometrical wetting BC in :mod:`tensorlbm.cg_wetting2d`.

The boundary condition is a *local* statement: on the last fluid node before a
wall, the imposed one-sided wall-normal phase difference must equal
``cot(theta) * |d(phi)/dt|``.  This script checks that statement against
analytic interface fields (a straight diffuse interface meeting the wall under
a prescribed angle), for a ladder of angles and two interface widths, using the
same estimator that the solver-level campaign reports (``theta_local``:
median over the wall faces, angle recovered from the imposed gradient).

Output: ``analytic_verification.json`` + ``fig_bc_analytic.png``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tensorlbm.cg_wetting2d import (  # noqa: E402
    geometrical_wetting_ghost,
    wall_local_contact_angle,
)

ANGLES = [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0]
WIDTHS = [2.0, 3.0]
NY = NX = 96


def analytic_phi(theta_deg: float, xi: float, ny: int = NY, nx: int = NX) -> torch.Tensor:
    th = math.radians(theta_deg)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    s = math.sin(th) * (x - nx / 2.0) - math.cos(th) * (y - 0.5)
    return torch.tanh(s / xi)


def main() -> int:
    solid = torch.zeros(NY, NX, dtype=torch.bool)
    solid[0, :] = True
    faces = torch.zeros(NY, NX, dtype=torch.bool)
    faces[1, :] = True

    rows = []
    for xi in WIDTHS:
        for theta in ANGLES:
            phi = analytic_phi(theta, xi)
            ghost = geometrical_wetting_ghost(phi, solid, theta)
            ang = wall_local_contact_angle(phi, ghost, solid, face_subset=faces)
            vals = ang.numpy() if ang.numel() else np.array([np.nan])
            med = float(np.nanmedian(vals))
            rows.append(
                {
                    "theta_deg": theta,
                    "xi": xi,
                    "n_nodes": int(ang.numel()),
                    "theta_readback_median": med,
                    "abs_err_deg": abs(med - theta),
                    "rel_err_pct": 100.0 * abs(med - theta) / theta,
                    "iqr_deg": (
                        float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
                        if ang.numel() > 3
                        else float("nan")
                    ),
                    "max_abs_err_deg": float(np.nanmax(np.abs(vals - theta))) if ang.numel() else float("nan"),
                }
            )
            print(
                f"xi={xi} theta={theta:6.1f} readback={med:7.3f} err={abs(med-theta):6.3f} "
                f"({100.0*abs(med-theta)/theta:5.2f}%) n={int(ang.numel())}",
                flush=True,
            )

    ok = [r for r in rows if r["rel_err_pct"] <= 15.0]
    payload = {
        "anchor": "XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING",
        "what": "analytic verification of the geometric wetting normal rotation (local BC statement)",
        "estimator": "theta_local = median over bottom-wall faces of atan2(|dphi/dt|, phi(face)-phi(ghost))",
        "criteria": {"gate": "each (theta, xi) case: relative error <= 15 %"},
        "summary": {
            "n_cases": len(rows),
            "n_pass_15pct": len(ok),
            "max_rel_err_pct": max(r["rel_err_pct"] for r in rows),
            "median_abs_err_deg": float(np.median([r["abs_err_deg"] for r in rows])),
        },
        "cases": rows,
    }
    (HERE / "analytic_verification.json").write_text(json.dumps(payload, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for xi, marker in zip(WIDTHS, ("o", "s")):
        sub = [r for r in rows if r["xi"] == xi]
        ax.plot(
            [r["theta_deg"] for r in sub],
            [r["theta_readback_median"] for r in sub],
            marker + "-",
            label=f"read-back (interface width xi={xi})",
        )
    ax.plot(ANGLES, ANGLES, "k--", lw=1, label="prescribed theta")
    ax.set_xlabel("prescribed contact angle [deg]")
    ax.set_ylabel("read-back angle from imposed wall gradient [deg]")
    ax.set_title("geometric wetting BC: local normals reproduce the prescribed angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_bc_analytic.png", dpi=150)
    print("wrote analytic_verification.json / fig_bc_analytic.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())