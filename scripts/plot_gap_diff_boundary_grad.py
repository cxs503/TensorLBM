#!/usr/bin/env python3
"""Figures for GAP-DIFFLBM-BOUNDARY-GRAD.

Reads ``artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/diff_boundary_grad_<mode>.json`` and
writes PNGs next to it:

``fig1_wall_anchors.png``      plane Couette / Poiseuille profiles vs exact
``fig2_dfg_forces.png``        DFG 2-D-2 C_D / C_L histories vs literature bands
``fig3_dfg_field.png``         velocity magnitude + vorticity of the DFG case
``fig4_gradient_matrix.png``   AD vs central-difference relative errors
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ART = Path("artifacts/GAP-DIFFLBM-BOUNDARY-GRAD")


def fig1(payload, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, key, title in (
        (axes[0], "couette", "plane Couette, moving bottom wall"),
        (axes[1], "poiseuille", "plane Poiseuille, static bounce-back walls"),
    ):
        a = payload["anchors"][key]
        got = np.asarray(a["profile_got"], dtype=float)
        exact = np.asarray(a["profile_exact"], dtype=float)
        y = np.arange(1, len(got) + 1)
        ax.plot(exact, y, "k-", lw=2.0, label="exact")
        ax.plot(got, y, "o", ms=3.0, mfc="none", label="LBM")
        ax.set_xlabel("$u_x$ [lattice]")
        ax.set_ylabel("wall-normal index")
        ax.set_title(f"{title}\nmax rel. err = {a['profile_max_rel_err']:.2e}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2(payload, out):
    d = payload["anchors"]["dfg"]
    t = np.asarray(d["samples"]["t"], dtype=float)
    cd = np.asarray(d["samples"]["C_D"], dtype=float)
    cl = np.asarray(d["samples"]["C_L"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    axes[0].plot(t, cd, "-", color="tab:blue", label="LBM")
    axes[0].axhspan(d["C_D_ref"][0], d["C_D_ref"][1], color="tab:orange", alpha=0.3,
                    label="Schafer & Turek 2D-2")
    axes[0].set_ylabel("$C_D$")
    axes[0].set_title(f"DFG 2-D-2, Re=100, D={d['params']['D_cells']:.0f} cells: "
                      f"$\\bar{{C_D}}$={d['C_D_mean']:.3f} (ref 3.22-3.24)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, cl, "-", color="tab:red", label="LBM")
    axes[1].axhspan(-1.0, 1.0, color="tab:orange", alpha=0.3, label="ref amplitude ~1.0")
    axes[1].set_ylabel("$C_L$")
    axes[1].set_xlabel("lattice step")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig3(payload, out):
    snap = payload["anchors"]["dfg"]["field_snapshot"]
    ux = np.asarray(snap["ux"], dtype=float)
    uy = np.asarray(snap["uy"], dtype=float)
    body = np.asarray(snap["body"], dtype=float)
    umag = np.hypot(ux, uy)
    duy_dx = np.gradient(uy, axis=1)
    dux_dy = np.gradient(ux, axis=0)
    omega = duy_dx - dux_dy
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.2))
    for ax, field, title in (
        (axes[0], umag, "velocity magnitude $|\\mathbf{u}|$"),
        (axes[1], omega, "vorticity $\\omega_z = \\partial_x u_y - \\partial_y u_x$"),
    ):
        lim = np.percentile(np.abs(field), 99.0) or 1.0
        im = ax.imshow(field, origin="lower", cmap="RdBu_r" if field is omega else "viridis",
                       vmin=-lim if field is omega else 0.0, vmax=lim, aspect="auto")
        ax.contour(body, levels=[0.5], colors="k", linewidths=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle("DFG 2-D-2 field at the last sampled step (every 2nd cell)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig4(payload, out):
    rows = [r for group in ("gradient_matrix", "resolution_scan", "horizon_scan")
            for r in payload.get(group, [])]
    if not rows:
        return
    labels, err_d, err_l, kinds = [], [], [], []
    for r in rows:
        labels.append(f"{r['closure']}\n{r['parameter'].split()[0]}\n[{r['resolution']}]")
        err_d.append(r["rel_err_drag"])
        err_l.append(r["rel_err_lift"])
        kinds.append(r["resolution"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(rows)), 4.2))
    ax.bar(x - 0.2, err_d, width=0.4, label="drag", color="tab:blue")
    ax.bar(x + 0.2, err_l, width=0.4, label="lift", color="tab:red")
    ax.axhline(1e-2, color="k", ls="--", lw=1.0, label="acceptance 1e-2")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("|AD - central FD| / |FD|")
    ax.set_title("Gradient matrix: reverse-mode AD vs central differences")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    path = ART / f"diff_boundary_grad_{mode}.json"
    payload = json.loads(path.read_text())
    made = []
    for fn, name in ((fig1, "fig1_wall_anchors.png"), (fig2, "fig2_dfg_forces.png"),
                     (fig3, "fig3_dfg_field.png"), (fig4, "fig4_gradient_matrix.png")):
        out = path.parent / name
        try:
            fn(payload, out)
            made.append(str(out))
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"{name}: skipped ({exc})")
    print("\n".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())