#!/usr/bin/env python3
"""Figure for XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE round 3.

Panel (a),(b): the phase field ``phi`` of the same sessile-drop run (theta=60 deg,
80x96, R=20, 400 steps) with the *legacy* opposite map used by rounds 1-2
``(0,3,2,1,4,7,8,5,6)`` and with the canonical ``solver.OPPOSITE_2D``.  The
legacy map leaves the two vertical D2Q9 directions un-reflected, so the wall is
not a bounce-back: the drop de-wets and a second blob condenses at the far wall.

Panel (c): commanded vs measured contact angle for the corrected run, with the
15 % advisory band.

Usage: python benchmarks/xval_openlbmpm_wetting/make_figure_r3.py --out runs/xval_blob_mechanism_r3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.cg_wetting_geometric import sessile_drop_initial  # noqa: E402
from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.solver import OPPOSITE_2D, stream  # noqa: E402

LEGACY_OPP = (0, 3, 2, 1, 4, 7, 8, 5, 6)
CANON = tuple(int(v) for v in OPPOSITE_2D)


def run(opp, *, theta=60.0, steps=400, ny=80, nx=96, radius=20.0):
    torch.set_num_threads(4)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid3 = solid.unsqueeze(0)
    f_r, f_b = sessile_drop_initial(ny, nx, radius, theta, xi=2.0, dtype=torch.float64)
    for _ in range(steps):
        f_r, f_b = color_gradient_two_tau_step(
            f_r, f_b, tau_r=1.0, tau_b=1.0, A=0.04, beta=0.7, solid_mask=solid,
            visc_mode="mixture", wall_contact_angle=theta, contact_angle_convention="physical")
        f_r, f_b = stream(f_r), stream(f_b)
        f_r = torch.where(solid3, f_r[list(opp)], f_r)
        f_b = torch.where(solid3, f_b[list(opp)], f_b)
    rr, rb = f_r.sum(dim=0), f_b.sum(dim=0)
    return ((rr - rb) / (rr + rb).clamp(min=1e-12)).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/xval_blob_mechanism_r3")
    ap.add_argument("--angles-json", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    phi_legacy = run(LEGACY_OPP)
    phi_canon = run(CANON)
    np.save(out / "phi_legacy_opp_t400.npy", phi_legacy)
    np.save(out / "phi_canonical_opp_t400.npy", phi_canon)

    fig = plt.figure(figsize=(13, 4.6))
    for i, (phi, title) in enumerate((
            (phi_legacy, "(a) legacy OPP (0,3,2,1,4,7,8,5,6): wall is not bounce-back"),
            (phi_canon, "(b) canonical OPPOSITE_2D: drop holds its 60 deg cap"))):
        ax = fig.add_subplot(1, 3, i + 1)
        im = ax.imshow(phi, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y (wall at y=0)")
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\phi$")

    angles = json.loads(Path(args.angles_json).read_text()) if args.angles_json else None
    ax = fig.add_subplot(1, 3, 3)
    if angles:
        cmd = [a["theta_cmd"] for a in angles["results"]]
        cap = [100.0 * (a["theta_cap_tail_deg"] - a["theta_cmd"]) / a["theta_cmd"] for a in angles["results"]]
        circ = [100.0 * (a["theta_circle_tail_deg"] - a["theta_cmd"]) / a["theta_cmd"] for a in angles["results"]]
        idx = np.arange(len(cmd))
        ax.bar(idx - 0.2, cap, 0.4, label=r"cap identity $2\arctan(2H/W)$")
        ax.bar(idx + 0.2, circ, 0.4, label="Kasa circle fit")
        ax.set_xticks(idx, [f"{c:g}$^\\circ$" for c in cmd])
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-15, 15, color="0.85", zorder=0, label="owner band $\\pm$15%")
        ax.set_ylabel("measurement error [%]")
        ax.legend(fontsize=7)
    ax.set_title("(c) end-to-end contact angle, corrected map", fontsize=9)
    fig.tight_layout()
    path = out / "figures" / "r3_bounceback_map_evidence.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())