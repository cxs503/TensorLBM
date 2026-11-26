#!/usr/bin/env python3
"""Generate figures for XVAL-LBM-FILL-GRADIENT-NORMAL (mission 17757)."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_xval as rx  # noqa: E402

FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)


def fig1_circle_normals():
    """Fill field + A1/A2/A4 normal vectors on the 2D circle (N=128, sharp)."""
    N = 128
    R = N / 4.0
    fill, d = rx.fill_circle(N, R, (0.0, 0.0))
    x, y = rx._grid2(N)
    nx1, ny1 = rx.a1_tensorlbm_2d(fill)
    nx2, ny2 = rx.a2_oss_2d(fill, 0.0)
    m1 = np.stack([nx1.numpy(), ny1.numpy()], axis=-1)
    m2 = np.stack([nx2.numpy(), ny2.numpy()], axis=-1)
    na = np.stack([x / d, y / d], axis=-1)
    band = np.abs(d - R) < 1.5
    sel = (np.linalg.norm(m1, axis=-1) > 1e-6) & band
    ys, xs = np.where(sel)
    idx = [i for i in range(len(ys)) if (ys[i] + xs[i]) % 2 == 0]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(fill.numpy(), origin="lower", cmap="Blues", vmin=0, vmax=1)
    for i in idx:
        yy, xx = ys[i], xs[i]
        ax.quiver(xx, yy, m1[yy, xx, 0], m1[yy, xx, 1],
                  color="red", alpha=0.9, width=0.004,
                  label="A1 TensorLBM" if i == idx[0] else None)
        ax.quiver(xx, yy, m2[yy, xx, 0] * 2, m2[yy, xx, 1] * 2,
                  color="orange", alpha=0.8, width=0.003,
                  label="A2 OSS (x2 mag)" if i == idx[0] else None)
        ax.quiver(xx, yy, na[yy, xx, 0] * 0.6, na[yy, xx, 1] * 0.6,
                  color="green", alpha=0.7, width=0.003,
                  label="A4 analytic" if i == idx[0] else None)
    ax.set_title("Fill-gradient normal, sharp circle R=32 (N=128)\n"
                 "A1 (red) and A2 (orange, 2x for visibility) coincide; A4 (green) = analytic")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_circle_normals.png", dpi=140)
    plt.close(fig)


def fig2_convergence():
    """dir_err vs analytic: sharp resolution ladder + smeared width ladder."""
    res = json.loads((HERE / "results.json").read_text())
    sharp = [c for c in res["cases"] if c["case"].startswith("circle-sharp")]
    smear = [c for c in res["cases"] if c["case"].startswith("circle-smeared")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    Ns = [c["case"].split("N=")[1] for c in sharp]
    errs = [c["A1_vs_analytic"]["dir_err"] * 100 for c in sharp]
    a1.semilogy([int(n) for n in Ns], errs, "o-", color="tab:red")
    a1.axhline(15, color="k", ls="--", lw=1, label="owner 15% line")
    a1.set_xlabel("grid N (circle, sharp fill)")
    a1.set_ylabel("dir_err vs analytic, %  (1-mean|cos|)")
    a1.set_title("Sharp fill: axis-bias floor ~5%")
    a1.legend(fontsize=8)
    ws = [c["case"].split("w=")[1].split(" ")[0] for c in smear]
    errs2 = [c["A1_vs_analytic"]["dir_err"] * 100 for c in smear]
    a2.semilogy([float(w) for w in ws], errs2, "s-", color="tab:blue")
    a2.axhline(15, color="k", ls="--", lw=1, label="owner 15% line")
    a2.set_xlabel("interface width w (cells)")
    a2.set_ylabel("dir_err vs analytic, %")
    a2.set_title("Smeared fill: converges with resolution")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_convergence.png", dpi=140)
    plt.close(fig)


def fig3_sphere():
    """3D sphere: A1 normal vectors on the surface (N=64)."""
    N = 64
    R = N / 4.0
    fill, d = rx.fill_sphere(N, R, (0.0, 0.0, 0.0))
    z, y, x = rx._grid3(N)
    gx, gy, gz = rx.a1_tensorlbm_3d(fill)
    m1 = np.stack([gx.numpy(), gy.numpy(), gz.numpy()], axis=-1)
    band = np.abs(d - R) < 1.5
    sel = (np.linalg.norm(m1, axis=-1) > 1e-6) & band
    zs, ys, xs = np.where(sel)
    idx = [i for i in range(len(zs)) if (zs[i] + ys[i] + xs[i]) % 3 == 0]
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x[sel], y[sel], z[sel], s=1, c="lightblue", alpha=0.35)
    for i in idx:
        ax.quiver(xs[i], ys[i], zs[i],
                   m1[zs[i], ys[i], xs[i], 0], m1[zs[i], ys[i], xs[i], 1],
                   m1[zs[i], ys[i], xs[i], 2], color="red", alpha=0.8)
    ax.set_title("3D sphere R=16 (N=64): A1 fill-gradient normal (red)")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_sphere_normals.png", dpi=140)
    plt.close(fig)


def main():
    fig1_circle_normals()
    fig2_convergence()
    fig3_sphere()
    out = {}
    for f in sorted(FIGDIR.glob("*.png")):
        out[f.name] = base64.b64encode(f.read_bytes()).decode()
    (HERE / "figures_b64.json").write_text(json.dumps(out))
    print("figures:", list(out))


if __name__ == "__main__":
    main()
