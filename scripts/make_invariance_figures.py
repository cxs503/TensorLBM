#!/usr/bin/env python
"""Generate the figures of the invariance-certificate report (P170W2-N08).

Reads the artefact JSONs written by ``run_invariance_campaign.py`` and adds
four field snapshots produced by short, deliberately cheap simulations (they
are illustrative only — the quantitative criteria come from the JSONs).

Outputs (``figures/``):

``fig1_poiseuille_convergence.png``  L2 error vs wall resolution (log-log) for
                                     theta = 0/15/30/45 + profile comparison
``fig2_cylinder_invariance.png``     Cd(theta), St(theta) and the negative control
``fig3_galilean.png``                superposition residual vs Ma, O(Ma^2) law,
                                     decay curves
``fig4_fields.png``                  speed fields: inclined channel, cylinder,
                                     boosted Taylor-Green vortex
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.invariance import (  # noqa: E402
    D2Q9Solver,
    VelocityDirichlet,
    channel_boundary_cells,
    circle,
    poiseuille_profile,
    tilted_channel,
)
from tensorlbm.invariance.cases import _PeriodicBox, tg_exact  # noqa: E402

ART = ROOT / "artifacts" / "invariance" / "P170W2-N08"
FIG = ROOT / "figures"


def load(name: str):
    p = ART / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------- fig 1
def fig_poiseuille(pois: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    colors = {0.0: "C0", 15.0: "C1", 30.0: "C2", 45.0: "C3"}
    for key, entry in sorted(pois["results"].items(), key=lambda kv: float(kv[0])):
        theta = float(key)
        h = np.array(entry["widths"], dtype=float)
        e = np.array(entry["errors"], dtype=float)
        ax.loglog(h, e, "o-", color=colors.get(theta, "k"),
                  label=fr"$\theta={theta:g}^\circ$ (order {entry['mean_order']:.2f})")
    hh = np.array([6.0, 28.0])
    ref = pois["results"]["0"]["errors"][0] * (8.0 / hh) ** 2
    ax.loglog(hh, ref, "k--", lw=1.0, label=r"slope $-2$")
    ax.set_xlabel(r"wall resolution $H$ (lattice units)")
    ax.set_ylabel(r"$L_2$ velocity error")
    ax.set_title("(a) inclined-channel Poiseuille: grid convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # profile comparison at theta = 45 deg (re-run, cheap)
    theta = 45.0
    width = 16
    nx = int(round(6 * width))
    ny = int(round(nx * math.tan(math.radians(theta))
                   + width / math.cos(math.radians(theta)) + 2 * width)) + 2
    geom = tilted_channel(nx, ny, theta, width)
    ux_e, uy_e = poiseuille_profile(geom, 0.01)
    solver = D2Q9Solver(geom, tau=0.8)
    solver.initialise(ux_e, uy_e)
    inlet, outlet = channel_boundary_cells(geom)
    solver.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_e, uy_e, (0, 1)),
                                   VelocityDirichlet(outlet, ux_e, uy_e, (0, -1))])
    for _ in range(12 * width * width):
        solver.step()
    ux_n, uy_n = solver.velocity()
    ax = axes[1]
    # sample a cross-section at mid-channel along the normal
    cx, cy = geom.meta["center"]
    n_hat = geom.normal
    s = np.linspace(-width * 0.6, width * 0.6, 200)
    xs = cx + s * n_hat[0]
    ys = cy + s * n_hat[1]
    jj = np.clip(xs.astype(int), 0, geom.nx - 1)
    ii = np.clip(ys.astype(int), 0, geom.ny - 1)
    eta = 2.0 * np.abs(s) / width
    exact = 1.5 * 0.01 * (1.0 - eta ** 2)
    num = (ux_n * geom.axis[0] + uy_n * geom.axis[1])[ii, jj]
    ax.plot(exact * 1e3, s / width, "k-", lw=2, label="exact parabola")
    ax.plot(num * 1e3, s / width, "r--", lw=1.4, label="LBM (D2Q9, Bouzidi BB)")
    ax.set_xlabel(r"axial velocity $u\cdot\hat t$  ($10^{-3}$)")
    ax.set_ylabel(r"$2y'/H$")
    ax.set_title(r"(b) velocity profile at $\theta=45^\circ$, $H=16$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig1_poiseuille_convergence.png"
    FIG.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- fig 2
def fig_cylinder(cyl: dict, neg: dict | None) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    for re_key, runs in sorted(cyl["results"].items(), key=lambda kv: float(kv[0])):
        th = [r["theta_deg"] for r in runs]
        cd = [r["cd"] for r in runs]
        ref = cd[0]
        ax.plot(th, [100 * (c - ref) / ref for c in cd], "o-",
                label=f"Re={float(re_key):g}: Cd(0)={ref:.3f}")
    if neg:
        for re_key, runs in neg["results"].items():
            th = [r["theta_deg"] for r in runs]
            cd = [r["cd"] for r in runs]
            ref = cd[0]
            ax.plot(th, [100 * (c - ref) / ref for c in cd], "s--", color="0.4",
                    label="negative control (BGK, $\\tau\\to0.5^+$)")
    ax.axhline(3.0, color="k", ls=":", lw=1)
    ax.axhline(-3.0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"lattice orientation $\theta$ (deg)")
    ax.set_ylabel(r"$100\,(C_d(\theta)-C_d(0))/C_d(0)$  (%)")
    ax.set_title("(a) drag-coefficient spread")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    for re_key, runs in sorted(cyl["results"].items(), key=lambda kv: float(kv[0])):
        th = [r["theta_deg"] for r in runs]
        st = [r["st"] for r in runs]
        ref = st[0]
        ax.plot(th, [100 * (s - ref) / ref for s in st], "o-",
                label=f"Re={float(re_key):g}: St(0)={ref:.3f}")
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.axhline(-2.0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"lattice orientation $\theta$ (deg)")
    ax.set_ylabel(r"$100\,(S_t(\theta)-S_t(0))/S_t(0)$  (%)")
    ax.set_title("(b) Strouhal-number spread")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    labels, vals, colors = [], [], []
    for re_key, runs in sorted(cyl["results"].items(), key=lambda kv: float(kv[0])):
        cd = np.array([r["cd"] for r in runs])
        labels.append(f"Re={float(re_key):g}")
        vals.append(100 * np.ptp(cd) / cd[0])
        colors.append("C0")
    if neg:
        for re_key, runs in neg["results"].items():
            cd = np.array([r["cd"] for r in runs])
            labels.append("BGK $\\tau\\to0.5^+$")
            vals.append(100 * np.ptp(cd) / cd[0])
            colors.append("C3")
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(3.0, color="k", ls=":", lw=1, label="C2 tolerance (3%)")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("max Cd spread (%)")
    ax.set_title("(c) peak-to-peak Cd spread vs tolerance")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "fig2_cylinder_invariance.png"
    FIG.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- fig 3
def fig_galilean(gal: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    rows = sorted(gal["results"], key=lambda r: r["ma"])
    rows = [r for r in rows if r["ma"] > 0]
    ma = np.array([r["ma"] for r in rows])
    res = np.array([r["residual_relU"] for r in rows])
    sup = np.array([r.get("superposition_residual_relU") or np.nan for r in rows])

    ax = axes[0]
    ax.loglog(ma, res, "o-", label=r"$|u-(u_{exact}+U)|/U$")
    if np.isfinite(sup).any():
        ax.loglog(ma, sup, "s--", label=r"$|u(U)-u(0)-U|/U$ (superposition)")
    ax.axhline(2e-3, color="k", ls=":", label="C4 tolerance $2\\times10^{-3}$")
    ax.set_xlabel(r"$Ma = U/c_s$")
    ax.set_ylabel("normalised residual")
    ax.set_title("(a) Galilean residual vs Mach number")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    base = [r for r in gal["results"] if r["ma"] == 0.0]
    lam0 = base[0]["decay_rate"] if base else np.nan
    dev = np.array([(r["decay_rate"] - lam0) / lam0 for r in rows])
    pos = dev > 0
    if pos.sum() >= 2:
        p = np.polyfit(np.log(ma[pos]), np.log(dev[pos]), 1)
        ax.loglog(ma[pos], dev[pos], "o", label="measured")
        mm = np.linspace(ma[pos].min(), ma[pos].max(), 50)
        ax.loglog(mm, np.exp(p[1]) * mm ** p[0], "k--",
                  label=fr"fit slope ${p[0]:.2f}$ (C5 $\\in[1.7,2.3]$)")
    ax.set_xlabel(r"$Ma$")
    ax.set_ylabel(r"$(\lambda(Ma)-\lambda(0))/\lambda(0)$")
    ax.set_title(r"(b) decay-rate deviation: $O(Ma^2)$ law")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    for r in rows[:4]:
        t = np.array(r["times"]) if "times" in r else None
        e = np.array(r["energies"]) if "energies" in r else None
        if t is None:
            continue
        ax.semilogy(t, e / e[0], "-", lw=1.0,
                    label=fr"$Ma={r['ma']:.3f}$: $\lambda/\lambda_0={r['decay_rate']/lam0:.5f}$")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$E(t)/E(t_{fit})$")
    ax.set_title("(c) fluctuation-energy decay")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / "fig3_galilean.png"
    FIG.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------- fig 4
def fig_fields() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) inclined channel theta=45
    theta, width = 45.0, 16
    nx = int(round(6 * width))
    ny = int(round(nx * math.tan(math.radians(theta))
                   + width / math.cos(math.radians(theta)) + 2 * width)) + 2
    geom = tilted_channel(nx, ny, theta, width)
    ux_e, uy_e = poiseuille_profile(geom, 0.01)
    s = D2Q9Solver(geom, tau=0.8)
    s.initialise(ux_e, uy_e)
    inlet, outlet = channel_boundary_cells(geom)
    s.set_velocity_dirichlet([VelocityDirichlet(inlet, ux_e, uy_e, (0, 1)),
                              VelocityDirichlet(outlet, ux_e, uy_e, (0, -1))])
    for _ in range(10 * width * width):
        s.step()
    ux, uy = s.velocity()
    speed = np.hypot(ux, uy)
    speed = np.ma.masked_where(geom.solid, speed)
    ax = axes[0]
    im = ax.imshow(speed, origin="lower", cmap="viridis")
    ax.set_title(r"(a) inclined channel $\theta=45^\circ$: $|u|$")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (b) cylinder theta=30
    D, dom = 10.0, 10.0
    n = int(round(D * dom))
    cgeom = circle(n, n, D / 2, theta_deg=30.0)
    axis = np.array(cgeom.axis)
    sc = D2Q9Solver(cgeom, tau=0.5 + 3 * 0.05 * D / 100.0, collision="trt",
                    dtype=np.float32)
    sc.initialise(np.zeros((n, n)), np.zeros((n, n)))
    fx = np.zeros((n, n))
    fy = np.zeros((n, n))
    for k in range(4000):
        _, ux, uy = sc.macroscopic()
        f = 0.05 - float((ux * axis[0] + uy * axis[1])[cgeom.fluid].mean())
        fx[cgeom.fluid] = f * axis[0]
        fy[cgeom.fluid] = f * axis[1]
        sc.step(fx, fy)
    ux, uy = sc.velocity()
    sp = np.ma.masked_where(cgeom.solid, np.hypot(ux, uy))
    ax = axes[1]
    im = ax.imshow(sp, origin="lower", cmap="magma")
    ax.set_title(r"(b) cylinder, lattice at $\theta=30^\circ$: $|u|$")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) boosted Taylor-Green
    n = 64
    box = _PeriodicBox(n)
    tg = D2Q9Solver(box, tau=0.5 + 3 * 0.01)
    ux0, uy0 = tg_exact(n, 0.02, 0.01, 0.0, 0.01)
    tg.initialise(ux0, uy0)
    for _ in range(1500):
        tg.step()
    ux, uy = tg.velocity()
    ax = axes[2]
    im = ax.imshow(ux, origin="lower", cmap="RdBu_r")
    ax.set_title(r"(c) boosted Taylor-Green ($Ma\approx0.02$): $u_x$")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = FIG / "fig4_fields.png"
    FIG.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(exist_ok=True)
    made = []
    pois = load("poiseuille")
    if pois:
        made.append(fig_poiseuille(pois))
    cyl = load("cylinder")
    neg = load("negative_control")
    if cyl:
        made.append(fig_cylinder(cyl, neg))
    gal = load("galilean")
    if gal:
        made.append(fig_galilean(gal))
    made.append(fig_fields())
    for p in made:
        print(f"[fig] {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())