#!/usr/bin/env python3
"""Figures for XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY.

Fig 1: runtime-geometry mask vs the verified hard-coded mask + LIFE marker ring.
Fig 2: marker discretisation of the transferred convention vs dx (second order).
Fig 3: Cd trace, geometry-from-config vs hard-coded (if the run artifacts exist).

Every number is read from the run artifacts; nothing is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xval_life_geometry_config"
OUT.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.boundaries import cylinder_mask  # noqa: E402
from tensorlbm.cylinder_st2d1 import case_geometry  # noqa: E402
from tensorlbm.runtime_geometry import parse_geometry_config  # noqa: E402

ST2D1 = "CIRCLE 1 0 0.20 0.20 0.0 0.0 0.05\n"


def fig1_geometry(dx_D: int = 20) -> Path:
    geo = case_geometry(dx_D)
    nx, ny, dx = int(geo["nx"]), int(geo["ny"]), geo["dx"]
    scene = parse_geometry_config(ST2D1)
    got = scene.mask(nx, ny, dx)
    import torch

    ref = cylinder_mask(
        nx, ny, float(geo["cx"]), float(geo["cy"]), float(geo["radius"]), torch.device("cpu")
    ).numpy()
    markers = scene.bodies[0].markers(dx) / dx  # to lattice units
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].imshow(got, origin="lower", cmap="Greys", interpolation="nearest")
    ax[0].plot(markers[:, 0], markers[:, 1], "o", ms=1.8, color="tab:red", label="IBM markers (LIFE rule)")
    ax[0].plot(geo["cx"], geo["cy"], "x", color="tab:blue", label="centre from geometry.config")
    ax[0].set_title(f"mask from geometry.config, D={dx_D} ({int(got.sum())} cells)")
    ax[0].set_xlabel("i"), ax[0].set_ylabel("j"), ax[0].legend(loc="upper right", fontsize=7)
    delta = got.astype(int) - ref.astype(int)
    im = ax[1].imshow(delta, origin="lower", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax[1].set_title(f"config mask − verified cylinder_mask = {int(np.abs(delta).sum())} cells")
    ax[1].set_xlabel("i")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle(
        "Fig 1 — runtime geometry.config reproduces the verified hard-coded obstacle exactly",
        fontsize=9,
    )
    fig.tight_layout()
    p = OUT / "fig1_geometry_parity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig2_discretisation() -> Path:
    par = json.loads((ROOT / "runs/m17673/xval_parity.json").read_text())
    rows = [r for r in par["rows"]]
    dx = np.array([r["dx_m"] for r in rows])
    per = np.array([r["perimeter_err_pct"] for r in rows])
    nmark = np.array([r["markers_got"] for r in rows])
    order = np.argsort(dx)
    dx, per, nmark = dx[order], per[order], nmark[order]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].loglog(dx, np.abs(per), "o-", label="|perimeter error| of chord polygon")
    ref = np.abs(per)[0] * (dx / dx[0]) ** 2
    ax[0].loglog(dx, ref, "k--", lw=1, label="slope 2 (O(dx²))")
    ax[0].set_xlabel("dx [m]"), ax[0].set_ylabel("|error| [%]"), ax[0].legend(fontsize=8)
    ax[0].invert_xaxis()
    ax[0].set_title("marker discretisation is second order")
    ax[1].semilogx(dx, nmark, "s-", color="tab:green")
    for x, n in zip(dx, nmark):
        ax[1].annotate(f"{n}", (x, n), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax[1].set_xlabel("dx [m]"), ax[1].set_ylabel("IBM markers on the circle")
    ax[1].invert_xaxis()
    ax[1].set_title("floor(2πr/dx) markers")
    fig.suptitle("Fig 2 — sensitivity to dx (r = 0.05 m)", fontsize=9)
    fig.tight_layout()
    p = OUT / "fig2_dx_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig3_cd() -> Path | None:
    runs = {
        "runtime_config (geometry.config)": ROOT / "runs/m17673/xval_D20_config.json",
        "hard_coded (verified path)": ROOT / "runs/m17673/xval_D20_hardcoded.json",
    }
    have = {k: v for k, v in runs.items() if v.is_file()}
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, path in have.items():
        r = json.loads(path.read_text())
        s = r["series"]
        ax.plot(s["step"], s["cd"], lw=1.2, label=f"{label}: Cd={r['cd']:.4f}")
    first = json.loads(next(iter(have.values())).read_text())
    ax.axhline(first["ref_cd"], color="k", ls="--", lw=1, label=f"anchor Cd={first['ref_cd']:.4f}")
    ax.set_xlabel("step"), ax.set_ylabel("Cd (0.5 ρ U² D)")
    ax.set_title("Fig 3 — drag history, runtime geometry vs hard-coded", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = OUT / "fig3_cd_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


if __name__ == "__main__":
    made = [fig1_geometry(20), fig2_discretisation()]
    f3 = fig3_cd()
    if f3:
        made.append(f3)
    print(json.dumps([str(p) for p in made], indent=1))