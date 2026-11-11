#!/usr/bin/env python3
"""Mid-plane field figure for node ``p387w2-n04`` (flow past a sphere).

Reads the ``.pt`` snapshot written by ``run_validation.py ... --snapshot`` and
draws the streamwise velocity and the out-of-plane vorticity on the mid-plane,
with the sphere mask outlined.  Produces ``artifacts/p387w2_n04/figures/field_midplane.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "p387w2_n04" / "figures"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else ROOT / "artifacts" / "p387w2_n04" / "field_re100_d10.pt"
    snap = torch.load(src, weights_only=False)
    ux = snap["ux"].numpy()
    om = snap["omega_y"].numpy()
    cx, _, _ = snap["center"]
    d = snap["diameter"]
    re = snap["reynolds_measured"]

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    for ax, field, label in (
        (axes[0], ux, "$u_x$  [lattice units]"),
        (axes[1], om, r"$\omega_y = \partial_z u_x - \partial_x u_z$"),
    ):
        vmax = float(abs(field).max()) or 1.0
        im = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[-cx, field.shape[1] - cx, -cx, field.shape[0] - cx], aspect="equal")
        circ = plt.Circle((0.0, 0.0), 0.5 * d, fill=False, ec="k", lw=1.2)
        ax.add_patch(circ)
        ax.set_ylabel("z  [lattice]")
        ax.set_title(label, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    axes[-1].set_xlabel("x  [lattice, sphere centre at 0]")
    fig.suptitle(f"Mid-plane field, Re = {re:.1f}, D = {d:g} lattice, step {snap['steps']}", fontsize=10)
    fig.tight_layout()
    out = OUT / "field_midplane.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())