#!/usr/bin/env python
"""IN65-N04 -- figures for the DG verification frontier report.

Reads ``artifacts/IN65_N04/*.json`` (produced by
``scripts/run_dg_frontier_in65.py``) and re-derives the *field* panels from the
same library calls, so every plotted number and every plotted curve comes from
code, never from a hand-copied table.

    PYTHONPATH=src python scripts/plot_dg_frontier_in65.py \
        --data-dir artifacts/IN65_N04 --fig-dir artifacts/IN65_N04/figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.dg_verification import frontier as F  # noqa: E402
from tensorlbm.dg_verification.benchmarks import (  # noqa: E402
    cone_initial,
    exact_isothermal_riemann,
    exact_sod,
)
from tensorlbm.dg_verification.element import ref_element  # noqa: E402
from tensorlbm.dg_verification.scalar import (  # noqa: E402
    advection_rhs_2d,
    node_coords_2d,
    ssp_rk3,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAL, ORANGE, GREY, RED, BLUE = "#1b7f79", "#e08b3c", "#666666", "#c0392b", "#2c6fb0"


def _load(data_dir: str, case: str) -> dict | None:
    path = os.path.join(data_dir, f"{case}.json")
    if not os.path.exists(path):
        print(f"[warn] missing {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def _slope_line(ax, h, e, color, label, offset=1.7):
    """Reference slope triangle at the last data point."""
    h = np.asarray(h, float)
    e = np.asarray(e, float)
    y = e[-1] * (h / h[-1]) ** offset
    ax.plot(h, y, "--", color=color, lw=1.0, alpha=0.7)
    ax.annotate(f"$h^{{{offset:g}}}$", xy=(h[len(h) // 2], y[len(h) // 2]),
                xytext=(4, -12), textcoords="offset points", color=color, fontsize=9)


def fig_convergence(data: dict, fig_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    # (a) rotating field
    ax = axes[0]
    for case, color in (("cone", RED), ("hump", ORANGE), ("gaussian", TEAL)):
        d = data.get(case)
        if not d:
            continue
        for p, ls in ((1, "-o"), (2, "-s"), (3, "-^")):
            rows = [r for r in d["rows"] if r["degree"] == p]
            if not rows:
                continue
            ax.loglog([r["h"] for r in rows], [r["L2"] for r in rows], ls, color=color,
                      ms=4, lw=1.3, alpha=0.9 if case == "gaussian" else 0.55,
                      label=f"{case} P{p}")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$L_2$ error after one revolution")
    ax.set_title("(a) rotating field: cone (C0) vs smooth companions")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # (b) 1-D kink vs smooth order
    ax = axes[1]
    k = data.get("kink")
    if k:
        for tag, color, ls in (("kink", RED, "-o"), ("smooth", TEAL, "-s")):
            for p in (1, 2, 3):
                rows = [r for r in k[tag]["rows"] if r["degree"] == p]
                ax.loglog([r["h"] for r in rows], [r["L2"] for r in rows], ls, color=color,
                          ms=3.5, lw=1.2, alpha=0.85, label=f"1-D {tag} P{p}")
        o = k["smooth"]["orders"]["P2"]["L2"]["order_fit"]
        ok = k["kink"]["orders"]["P2"]["L2"]["order_fit"]
        ax.set_title(f"(b) 1-D order: smooth P2 {o:.2f} vs kinked P2 {ok:.2f}")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$L_2$ error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6, ncol=2)

    # (c) MMS variable coefficient
    ax = axes[2]
    m = data.get("mms")
    if m:
        for p, ls in ((1, "-o"), (2, "-s"), (3, "-^")):
            rows = [r for r in m["rows"] if r["degree"] == p]
            if not rows:
                continue
            h = [r["h"] for r in rows]
            ax.loglog(h, [r["L2_u"] for r in rows], ls, color=TEAL, ms=4, lw=1.3, label=f"u P{p}")
            ax.loglog(h, [r["L2_q"] for r in rows], ls, color=ORANGE, ms=4, lw=1.1,
                      alpha=0.75, label=f"q=grad u P{p}")
        ax.set_title("(c) MMS, rotating a(x): u and LDG gradient q")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$L_2$ error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    path = os.path.join(fig_dir, "fig1_convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _rotate(initial_fn, n: int, p: int, t_end: float = 2.0 * math.pi, cfl: float = 0.3):
    el = ref_element(p)
    dx = dy = 1.0 / n
    X, Y = node_coords_2d(n, n, el, dx, dy)
    u = initial_fn(X, Y)
    axf, ayf = 0.5 - Y, X - 0.5
    dt = cfl * dx / ((2 * p + 1) * math.sqrt(0.5))
    nsteps = int(math.ceil(t_end / dt))
    dt = t_end / nsteps
    for _ in range(nsteps):
        u = ssp_rk3(u, dt, advection_rhs_2d, axf, ayf, el, dx, dy, "upwind")
    return X, Y, u


def fig_cone_fields(fig_dir: str, n: int = 48) -> str:
    """Cell-averaged cloud plots of the LeVeque cone before/after one turn."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    ic = cone_initial
    xc = (np.arange(n) + 0.5) / n
    el = ref_element(1)
    X, Y = node_coords_2d(n, n, el, 1.0 / n, 1.0 / n)
    panels = [("initial cone (C0: kink on the base circle)", _cg(ic(X, Y)))]
    for p in (1, 2):
        Xp, Yp, u = _rotate(ic, n, p)
        err = u - ic(Xp, Yp)
        l2 = float(np.sqrt(np.mean(err ** 2)))
        panels.append((f"after 2$\\pi$, P{p} (N={n}): L2={l2:.2e}", _cg(u)))
    for ax, (title, field) in zip(axes, panels):
        im = ax.pcolormesh(xc, xc, field, shading="auto", cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    path = os.path.join(fig_dir, "fig2_cone_fields.png")
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def _cg(field: np.ndarray) -> np.ndarray:
    """Cell-averaged 2-D view of a nodal DG field for plotting."""
    return field.mean(axis=(-1, -2))


def _sod_profile(n: int, p: int, flux: str = "hllc", t_end: float = 0.2, tvb_M: float = 0.0):
    sol = F.EulerDG1D_HLLC(degree=p, n_cells=n, gamma=1.4, flux=flux, use_limiter=True,
                           tvb_M=tvb_M, bc="outflow")
    xc0 = (np.arange(n) + 0.5) / n
    side = (xc0 < 0.5)[:, None]
    sol.set_state(sol.from_primitive(np.where(side, 1.0, 0.125) * np.ones_like(sol.x),
                                     np.zeros_like(sol.x),
                                     np.where(side, 1.0, 0.1) * np.ones_like(sol.x)))
    sol.advance(t_end, cfl=0.3)
    xc = sol.cell_centers()
    rho, u, p = sol.primitive(sol.U)
    return xc, rho.mean(axis=-1), u.mean(axis=-1), p.mean(axis=-1)


def fig_sod(data: dict, fig_dir: str, n: int = 400) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    xf = np.linspace(0.0, 1.0, 4001)
    ref = exact_sod(xf, 0.2)
    ax = axes[0]
    ax.plot(xf, ref["rho"], "k-", lw=1.4, label="exact (Riemann)")
    ax.plot(xf, ref["p"], "k--", lw=1.0, alpha=0.6, label="exact p")
    for p, color in ((1, TEAL), (2, ORANGE)):
        xc, rho, u, pr = _sod_profile(n, p)
        ax.plot(xc, rho, color=color, lw=1.2, label=f"DG P{p} $\\rho$")
        ax.plot(xc, pr, color=color, lw=0.9, ls="--", alpha=0.8, label=f"DG P{p} p")
    for x, lab in ((0.8504, "shock"), (0.6855, "contact")):
        ax.axvline(x, color=GREY, lw=0.7, ls=":")
        ax.annotate(lab, xy=(x, 1.05), fontsize=7, color=GREY, ha="center")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$, p")
    ax.set_title(f"(a) Sod, ideal gas, t=0.2, n={n}")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for flux, color in (("hllc", TEAL), ("rusanov", RED)):
        for p, ls in ((1, "-o"), (2, "-s")):
            rows = [r for r in data["sod"]["rows"] if r["flux"] == flux and r["degree"] == p]
            if not rows:
                continue
            ax.loglog([r["h"] for r in rows], [r["contact_width_cells"] for r in rows], ls,
                      color=color, ms=4, lw=1.2, label=f"{flux} P{p}")
    ax.set_xlabel("h")
    ax.set_ylabel("contact smearing width [cells]")
    ax.set_title("(b) contact resolution: HLLC vs Rusanov")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[2]
    for flux, color in (("hllc", TEAL), ("rusanov", RED)):
        rows = [r for r in data["sod"]["rows"] if r["flux"] == flux]
        if not rows:
            continue
        ns = sorted({r["n_cells"] for r in rows})
        errs = [max(r["shock_pos_err_pct"] for r in rows if r["n_cells"] == nn) for nn in ns]
        ax.semilogx(ns, errs, "-o", color=color, ms=4, label=f"{flux}")
    ax.axhline(2.0, color=GREY, ls="--", lw=1.0)
    ax.annotate("2 % gate", xy=(min(ns), 2.0), xytext=(2, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("cells")
    ax.set_ylabel("shock position error [% of travel]")
    ax.set_title("(c) pressure-based shock locator vs the 2 % gate")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    path = os.path.join(fig_dir, "fig3_sod.png")
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def fig_lbm(data: dict, fig_dir: str, n_cells: int = 2000, t_plot: int = 200) -> str:
    npz = os.path.join(ROOT, "benchmarks", "verified", "sod_shock_tube", "profiles_nx2000.npz")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax = axes[0]
    if os.path.exists(npz):
        d = np.load(npz)
        x = np.arange(n_cells) + 0.5
        rho_lbm = np.asarray(d[f"t{t_plot}"], float)
        cs = math.sqrt(1.0 / 3.0)
        xd = n_cells // 2
        ref = exact_isothermal_riemann(x, t_plot, x0=xd, rho_l=1.0, rho_r=0.25, cs=cs)
        ax.plot(x, rho_lbm, color=ORANGE, lw=1.2, label=f"verified D2Q9 LBM, t={t_plot}")
        ax.plot(x, ref["rho"], "k--", lw=1.0, label="exact isothermal Riemann")
        sol = F.EulerDG1D_HLLC(degree=1, n_cells=n_cells, length=float(n_cells),
                               isothermal=True, cs2=1.0 / 3.0, flux="hllc",
                               use_limiter=True, bc="periodic")
        xc0 = np.arange(n_cells) + 0.5
        side = (xc0 < xd)[:, None]
        sol.set_state(sol.from_primitive(np.where(side, 1.0, 0.25) * np.ones_like(sol.x),
                                         np.zeros_like(sol.x), np.zeros_like(sol.x)))
        sol.advance(float(t_plot), cfl=0.3)
        ax.plot(sol.cell_centers(), sol.cell_means()[0], color=TEAL, lw=1.0,
                label=f"DG P1 HLLC, t={t_plot}")
        ax.set_xlim(xd - 60, xd + 400)
    ax.set_xlabel("x [lattice cells]")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("(a) isothermal Sod: DG <-> LBM <-> analytic")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    dg = [r for r in data["sod_lbm"]["dg_rows"] if r["degree"] == 1]
    lb = data["sod_lbm"]["lbm_rows"]
    ax.semilogy([r["t"] for r in dg], [r["shock_pos_err_pct"] for r in dg], "-o", color=TEAL,
                ms=4, label="DG P1 shock-position error")
    ax.semilogy([r["t"] for r in lb], [r["shock_pos_err_pct"] for r in lb], "-s", color=ORANGE,
                ms=4, label="LBM shock-position error")
    ax.semilogy([r["t"] for r in dg], [r["shock_thickness_cells"] for r in dg], "--", color=TEAL,
                lw=1.0, label="DG shock thickness [cells]")
    ax.semilogy([r["t"] for r in lb], [r["shock_thickness_cells"] for r in lb], "--", color=ORANGE,
                lw=1.0, label="LBM shock thickness [cells]")
    ax.axhline(2.0, color=GREY, ls=":", lw=1.0)
    ax.set_xlabel("t [lattice units]")
    ax.set_title("(b) wave-position error and shock thickness vs time")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    path = os.path.join(fig_dir, "fig4_lbm_crosscheck.png")
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def fig_square(data: dict, fig_dir: str, n: int = 256, period: int = 4) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))
    ax = axes[0]
    for use_lim, color, lab in ((True, TEAL, "TVB/minmod limited"), (False, RED, "unlimited")):
        sol = F.ScalarDG1D(degree=1, n_cells=n, velocity=1.0, use_limiter=use_lim)
        xx = (sol.x - 0.25) % 1.0
        sol.set_state(np.where(xx < 0.5, 1.0, 0.0))
        sol.advance(float(period), cfl=0.3)
        xc = sol.cell_centers()
        ax.plot(xc, sol.cell_means(), color=color, lw=1.1, label=lab)
    xxc = (np.linspace(0, 1, 2000) - 0.25 - period) % 1.0
    ax.plot(np.linspace(0, 1, 2000), np.where(xxc < 0.5, 1.0, 0.0), "k--", lw=0.9, label="exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u (cell means)")
    ax.set_title(f"(a) square wave after {period} periods, P1, n={n}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    rows = data["square"]["square_rows"]
    for use_lim, color in ((True, TEAL), (False, RED)):
        sub = [r for r in rows if r["limiter"] == use_lim and r["degree"] == 1]
        ns = sorted({r["n_cells"] for r in sub})
        tv = [np.mean([r["TV_ratio"] for r in sub if r["n_cells"] == nn]) for nn in ns]
        ax.semilogx(ns, tv, "-o", color=color, ms=4,
                    label="limited" if use_lim else "unlimited")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("cells")
    ax.set_ylabel("TV(u(T)) / TV(u(0))")
    ax.set_title("(b) total-variation ledger (dissipation)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[2]
    modes = data["square"]["mode_rows"]
    for use_lim, color in ((True, TEAL), (False, RED)):
        sub = [r for r in modes if r["limiter"] == use_lim and r["degree"] == 1]
        ns = sorted({r["n_cells"] for r in sub})
        amp = [100.0 * (1.0 - np.mean([r["amplitude_ratio"] for r in sub if r["n_cells"] == nn]))
               for nn in ns]
        ph = [np.mean([abs(r["phase_err_rad"]) for r in sub if r["n_cells"] == nn]) for nn in ns]
        ax.loglog(ns, np.maximum(amp, 1e-12), "-o", color=color, ms=4,
                  label=f"amplitude loss [%] ({'lim' if use_lim else 'unlim'})")
        ax.loglog(ns, ph, "--s", color=color, ms=4, alpha=0.7,
                  label=f"|phase error| [rad] ({'lim' if use_lim else 'unlim'})")
    ax.set_xlabel("cells")
    ax.set_title("(c) single-mode dissipation / dispersion, P1, one period")
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.25)
    path = os.path.join(fig_dir, "fig5_square_wave.png")
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def fig_ledgers(data: dict, fig_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    rows = data["ledger"]["rows"]
    labels = [f"{r['flux']}\nP{r['degree']}" for r in rows]
    xpos = np.arange(len(rows))
    mass = [max(r["mass_drift"], 1e-18) for r in rows]
    mom = [max(r["momentum_drift"], 1e-18) for r in rows]
    en = [max(r["energy_drift"], 1e-18) for r in rows]
    w = 0.26
    ax.bar(xpos - w, mass, w, color=TEAL, label=r"$|\Delta\int\rho|/\int\rho$")
    ax.bar(xpos, mom, w, color=ORANGE, label=r"$|\Delta\int\rho u|/\int\rho$")
    ax.bar(xpos + w, en, w, color=BLUE, label=r"$|\Delta\int E|/\int E$")
    ax.axhline(1e-10, color=RED, ls="--", lw=1.0)
    ax.annotate("1e-10 gate", xy=(0, 1e-10), xytext=(2, 3), textcoords="offset points",
                fontsize=8, color=RED)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative drift after t=0.5")
    ax.set_title("(a) Euler invariants, periodic Sod (HLLC vs Rusanov)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    rot = data["cone"]
    rows = rot["rows"]
    labels = [f"P{r['degree']}\nn={r['n_cells']}" for r in rows]
    band = [r["frac_err_band"] for r in rows]
    ax.bar(np.arange(len(rows)), band, color=TEAL)
    ax.axhline(0.5, color=RED, ls="--", lw=1.0)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("fraction of squared error in the kink band")
    ax.set_title("(b) rotating cone: the kink band owns the error")
    ax.grid(axis="y", alpha=0.25)
    path = os.path.join(fig_dir, "fig6_ledgers.png")
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "artifacts", "IN65_N04"))
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--skip-fields", action="store_true")
    args = ap.parse_args()
    fig_dir = args.fig_dir or os.path.join(args.data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    data = {c: _load(args.data_dir, c) for c in
            ("cone", "hump", "gaussian", "kink", "mms", "sod", "sod_lbm", "square", "ledger")}
    paths = []
    paths.append(fig_convergence(data, fig_dir))
    if not args.skip_fields:
        paths.append(fig_cone_fields(fig_dir))
    if data.get("sod"):
        paths.append(fig_sod(data, fig_dir))
    if data.get("sod_lbm"):
        paths.append(fig_lbm(data, fig_dir))
    if data.get("square"):
        paths.append(fig_square(data, fig_dir))
    if data.get("ledger") and data.get("cone"):
        paths.append(fig_ledgers(data, fig_dir))
    for p in paths:
        print(f"[fig] {p}")


if __name__ == "__main__":
    main()