"""Figures for the D3Q19 AA-pattern fused stream-collide verification.

Reads ``artifacts/xval_aa_pattern/d3q19_verify.json`` (falling back to the
``--quick`` JSON) and writes ``fig_d3q19_*.png`` next to it.  The two field
panels regenerate a cheap run (a few seconds) because the gate JSON stores
scalars and profiles, not fields.

    python tests/gen_aa_fused_d3q19_figures.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tensorlbm.aa_fused_push_pull_3d as M  # noqa: E402
ART = os.path.join(ROOT, "artifacts", "xval_aa_pattern")

torch.set_grad_enabled(False)


def _load():
    for name in ("d3q19_verify.json", "d3q19_verify_quick.json"):
        p = os.path.join(ART, name)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh), p
    raise SystemExit("no d3q19 verification JSON found — run the driver first")


def fig_gates(data, out):
    """Profiles against the exact solutions, with the error panel."""
    po = data["poiseuille"]
    co = data["couette"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    y = np.arange(po["n_fluid"])
    num = np.array(po["profile"])
    ex = np.array(po["exact"])
    ax[0].plot(po["exact"], y, "k-", lw=1.6,
               label="analytic parabolic profile")
    ax[0].plot(num, y, "o", ms=5, mfc="none", label="D3Q19 fused (Guo forcing)")
    ax[0].set_xlabel(r"$u_x$")
    ax[0].set_ylabel(r"$y$ (lattice)")
    ax[0].set_title(f"Plane Poiseuille, $H$={po['n_fluid']}, tau={po['tau']}\n"
                    f"$L_2$ profile error = {po['l2']:.3%}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    y = np.arange(co["n_fluid"])
    ax[1].plot(co["profile"], y, "o", ms=5, mfc="none", color="C3")
    ax[1].plot(co["exact"], y, "k-", lw=1.6)
    ax[1].set_xlabel(r"$u_x$")
    ax[1].set_ylabel(r"$y$ (lattice)")
    ax[1].set_title("Plane Couette (moving wall)\n"
                    f"$L_2$ error = {co['l2']:.2e}")
    ax[1].grid(alpha=0.3)

    err_p = np.abs(num - ex) / max(abs(ex).max(), 1e-30)
    err_c = np.abs(np.array(co["profile"]) - np.array(co["exact"])) / co["u_wall"]
    ax[2].semilogy(y, np.maximum(err_p, 1e-17), "o-", label="Poiseuille rel. error")
    ax[2].semilogy(y, np.maximum(err_c, 1e-17), "s-", label="Couette rel. error")
    ax[2].axhline(0.03, color="C2", ls="--", lw=1, label="3 % gate")
    ax[2].axhline(0.15, color="C3", ls=":", lw=1, label="15 % (unacceptable)")
    ax[2].set_xlabel(r"$y$ (lattice)")
    ax[2].set_ylabel("|error| / scale")
    ax[2].set_title("pointwise error against exact solutions")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_viscosity(data, out):
    """Shear-wave decay: transport coefficient measured, not assumed."""
    sw = data["shear_wave"]
    t = [s["t"] for s in sw["samples"]]
    a = [s["amp"] for s in sw["samples"]]
    ae = [s["amp_exact"] for s in sw["samples"]]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(t, a, "o", label="D3Q19 fused")
    ax[0].semilogy(t, ae, "k-", lw=1.4,
                   label=r"exact $u_0\cos(ky)e^{-\nu k^2 t}$")
    ax[0].set_xlabel("steps")
    ax[0].set_ylabel("RMS $u_x$")
    ax[0].set_title(f"Decaying shear wave ($k$={sw['k_lat']:.4f}, tau={sw['tau']})")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    labels = ["$\\nu$ analytic", "$\\nu$ measured"]
    vals = [sw["nu_analytic"], sw["nu_eff"]]
    ax[1].bar(labels, vals, color=["0.6", "C0"])
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel(r"$\nu$")
    ax[1].set_title(r"measured $\nu_{eff}$ vs $(\tau-1/2)/3$"
                    f"\nrelative error = {sw['nu_rel_error']:.2%}")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_taylor_green(data, out):
    """Field of the exact vortex + the refinement study."""
    tg = data["taylor_green"]
    runs = {r["n"]: r for r in tg["runs"]}
    n, u0, tau = 32, 0.02, 0.8
    nu = (tau - 0.5) / 3.0
    k_lat = 2.0 * math.pi / n
    steps = runs[n]["steps"] if n in runs else 90
    f = M.taylor_green_initial(1, n, u0, k_lat)
    for _ in range(steps):
        f = M.fused_step(f, tau)
    _, ux, uy, _ = M.macroscopic(f)
    ax_field = ux[0].numpy()
    ay_field = uy[0].numpy()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    cf = ax[0].contourf(ax_field, levels=24, cmap="RdBu_r")
    ax[0].streamplot(np.arange(n), np.arange(n), ax_field, ay_field,
                     color="k", density=1.1, linewidth=0.6)
    fig.colorbar(cf, ax=ax[0], shrink=0.85)
    ax[0].set_title(f"2D-extended Taylor-Green vortex\n$u_x$ + streamlines at t={steps}")
    ax[0].set_xlabel("x (lattice)")
    ax[0].set_ylabel("y (lattice)")

    ns = sorted(runs)
    errs = [runs[k]["l2"] for k in ns]
    ax[1].loglog(ns, errs, "o-", label="$L_2$ relative error")
    ref = errs[0] * (np.array(ns, float) / ns[0]) ** -2.0
    ax[1].loglog(ns, ref, "k--", lw=1, label="2nd-order slope")
    ax[1].axhline(0.03, color="C2", ls="--", lw=1, label="3 % gate")
    ax[1].axhline(0.15, color="C3", ls=":", lw=1, label="15 % (unacceptable)")
    ax[1].set_xlabel("n (nodes per side, one period)")
    ax[1].set_ylabel("$L_2$ relative error")
    ax[1].set_title("grid refinement, fixed physical problem")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    for o in tg["orders"]:
        ax[1].annotate(f"order {o['order']:.2f}",
                       xy=(math.sqrt(o["pair"][0] * o["pair"][1]),
                           math.sqrt(runs[o["pair"][0]]["l2"] * runs[o["pair"][1]]["l2"])),
                       fontsize=7, xytext=(4, 6), textcoords="offset points")

    dec_m = [runs[k]["decay_measured"] for k in ns]
    dec_e = [runs[k]["decay_factor_exact"] for k in ns]
    xx = np.arange(len(ns))
    ax[2].bar(xx - 0.2, dec_e, 0.4, label="exact decay factor", color="0.65")
    ax[2].bar(xx + 0.2, dec_m, 0.4, label="measured", color="C0")
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels([f"n={k}" for k in ns])
    ax[2].set_ylabel(r"RMS amplitude / $u_0/\sqrt{2}$")
    ax[2].set_title("viscous decay over the run window")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_gate_summary(data, out):
    """All gates on one log axis, with the 3 % / 15 % standards."""
    items = []

    def add(label, v):
        if isinstance(v, (int, float)) and v == v:
            items.append((label, float(v)))

    add("Couette $L_2$ (exact)", data["couette"]["l2"])
    add("Poiseuille $L_2$ (exact)", data["poiseuille"]["l2"])
    add(r"shear wave $\nu$ rel. err", data["shear_wave"]["nu_rel_error"])
    add("shear wave profile $L_2$", data["shear_wave"]["profile_l2"])
    for r in data["taylor_green"]["runs"]:
        add(f"Taylor-Green $L_2$ (n={r['n']})", r["l2"])

    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = [i[0] for i in items]
    vals = [max(i[1], 1e-17) for i in items]
    colors = ["C0" if v < 0.03 else ("C1" if v < 0.15 else "C3") for v in vals]
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.axvline(0.03, color="C2", ls="--", lw=1, label="3 % gate")
    ax.axvline(0.15, color="C3", ls=":", lw=1, label="15 % = unacceptable")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:.2e}", va="center", fontsize=8)
    ax.set_xlabel("relative error (log)")
    ax.set_title("D3Q19 AA-pattern fused stream-collide: gate summary")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    data, path = _load()
    fig_gates(data, os.path.join(ART, "fig_d3q19_gates.png"))
    fig_viscosity(data, os.path.join(ART, "fig_d3q19_viscosity.png"))
    fig_taylor_green(data, os.path.join(ART, "fig_d3q19_taylor_green.png"))
    fig_gate_summary(data, os.path.join(ART, "fig_d3q19_gate_summary.png"))
    print("figures from", path)
    print("wrote fig_d3q19_{gates,viscosity,taylor_green,gate_summary}.png")


if __name__ == "__main__":
    main()