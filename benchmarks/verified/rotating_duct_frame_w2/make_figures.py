#!/usr/bin/env python
"""Figures for p158w2-n01 from the benchmark record + raw fields.

Produces (into ``docs/assets/rotating_duct_p158w2_n01/``):

1. ``fig1_ekman_scaling.png``  -- measured Ekman thickness vs rotation rate
   (log-log) with the fitted ``Omega^{-1/2}`` law, both schemes and the
   phase-based independent anchor.
2. ``fig2_ekman_profiles.png`` -- measured vs exact Ekman spiral at one rate
   (both velocity components) and the Guo-EDM difference.
3. ``fig3_couette.png``        -- counter-rotating Taylor-Couette profile vs
   ``A r + B/r`` (inertial and rotating frame) and the torque identity ``T(r)``.
4. ``fig4_duct_secondary.png`` -- cross-plane secondary flow of the rotating
   square duct: streamfunction contours + vectors, LBM vs the finite-difference
   reference.
5. ``fig5_duct_scaling.png``   -- secondary-flow amplitude and field error vs
   rotation rate (linearity and the creeping-reference trend).
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.rotating_interior import circular_couette, rotating_channel_analytic

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "docs", "assets", "rotating_duct_p158w2_n01"))
REC = os.path.join(HERE, "records", "rotating_duct_frame_w2.json")
FLD = os.path.join(HERE, "records", "rotating_duct_frame_w2_fields.npz")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rec = json.load(open(REC))
    fld = np.load(FLD, allow_pickle=False)
    cases = rec["cases"]
    ekman = [c for c in cases if c["kind"] == "ekman"]
    couette = [c for c in cases if c["kind"] == "couette"]
    duct = [c for c in cases if c["kind"] == "duct"]

    # ---------------- fig 1: Ekman scaling law ----------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for scheme, marker in (("guo", "o"), ("edm", "s")):
        sel = [c for c in ekman if c["scheme"] == scheme]
        om = np.array([c["params"]["Omega"] for c in sel])
        df = np.array([c["delta_fit"] for c in sel])
        dp = np.array([c["delta_phase"] for c in sel])
        ax.loglog(om, df, marker, ms=6, label=f"$\\delta_E$ profile fit ({scheme})")
        ax.loglog(om, dp, marker, ms=5, mfc="none", label=f"$\\delta_E$ hodograph phase ({scheme})")
    om = np.array([c["params"]["Omega"] for c in ekman if c["scheme"] == "guo"])
    nu = ekman[0]["params"]["nu"]
    grid = np.logspace(np.log10(om.min() * 0.8), np.log10(om.max() * 1.25), 50)
    expo = rec["criteria"]["C4_ekman_scaling_exponent"]["value"]
    fit0 = np.exp(np.polyfit(np.log(om), np.log([c["delta_fit"] for c in ekman if c["scheme"] == "guo"]), 1)[1])
    ax.loglog(grid, fit0 * grid**expo, "k--", lw=1.2, label=f"fit $\\propto\\Omega^{{{expo:.3f}}}$")
    ax.loglog(grid, np.sqrt(nu / grid), "k:", lw=1.0, label="$\\sqrt{2\\nu/f}=\\sqrt{\\nu/\\Omega}$")
    ax.set_xlabel("frame rotation rate $\\Omega$ (lattice units)")
    ax.set_ylabel("Ekman thickness $\\delta_E$ (lattice units)")
    ax.set_title("Ekman-layer thickness scaling law (exponent $-1/2$ verified)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_ekman_scaling.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig 2: profile + scheme difference ------------------- #
    cand = [c for c in ekman if c["scheme"] == "guo"]
    c = max(cand, key=lambda x: x["delta_nodes"])
    dn = c["delta_nodes"]
    Omega = c["params"]["Omega"]
    G = c["params"]["G"]
    N = c["params"]["grid"][1]
    h = (N - 2) / 2.0
    y = np.arange(1, N - 1, dtype=float) - (N - 1) / 2.0
    ux_a, uz_a = rotating_channel_analytic(y, h, G, Omega, nu)
    ux = fld[f"ekman_{int(dn)}_guo_ux"]
    uz = fld[f"ekman_{int(dn)}_guo_uz"]
    try:
        ux_e = fld[f"ekman_{int(dn)}_edm_ux"]
        uz_e = fld[f"ekman_{int(dn)}_edm_uz"]
    except KeyError:
        ux_e = uz_e = None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].plot(ux_a / 0.02, y / dn, "k-", lw=2, label="exact $u_\\parallel/U_g$")
    axes[0].plot(uz_a / 0.02, y / dn, "k--", lw=2, label="exact $u_\\perp/U_g$")
    axes[0].plot(ux / 0.02, y / dn, "o", ms=3.5, mfc="none", label="LBM $u_\\parallel$ (Guo)")
    axes[0].plot(uz / 0.02, y / dn, "s", ms=3.5, mfc="none", label="LBM $u_\\perp$ (Guo)")
    axes[0].set_xlabel("velocity / geostrophic speed")
    axes[0].set_ylabel("$(y+h)/\\delta_E$")
    axes[0].set_title(f"Ekman spiral, $\\delta_E$ = {dn:.0f} lattice nodes")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    if ux_e is not None:
        axes[1].semilogy(y / dn, np.maximum(np.abs(ux - ux_e), 1e-18) / 0.02, "-", label="$|u_x^{Guo}-u_x^{EDM}|/U_g$")
        axes[1].semilogy(y / dn, np.maximum(np.abs(uz - uz_e), 1e-18) / 0.02, "-", label="$|u_z^{Guo}-u_z^{EDM}|/U_g$")
        axes[1].set_ylim(1e-17, 1e-2)
    axes[1].set_xlabel("$(y+h)/\\delta_E$")
    axes[1].set_title("Guo vs EDM: machine-level agreement")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_ekman_profiles.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig 3: Taylor-Couette ------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    r = fld["couette_r"]
    u = fld["couette_u_th"]
    cc = couette[0]
    p = cc["params"]
    ana = circular_couette(r, p["ri"], p["ro"], p["omega_i"], p["omega_o"])["u_theta"]
    axes[0].plot(r, ana, "k-", lw=2, label="exact $Ar+B/r$")
    axes[0].plot(r, u, "o", ms=4, mfc="none", label="LBM (D2Q9, $\\Omega_f=0$)")
    p1 = couette[1]["params"]
    ana1 = circular_couette(r, p1["ri"], p1["ro"], p1["omega_i"], p1["omega_o"])["u_theta"]
    axes[0].plot(r, ana1, ":", color="0.5", lw=1.2, label="exact (frame rotating)")
    axes[0].plot(r, fld["couette_u_r"], "x", ms=3, label="LBM $u_r$ (no spurious radial flow)")
    axes[0].set_xlabel("$r$ (lattice units)")
    axes[0].set_ylabel("$u_\\theta$")
    axes[0].set_title("counter-rotating Taylor-Couette")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # torque identity
    du = np.gradient(u, r)
    T = 2 * math.pi * nu * r**2 * (du - u / r)
    m = np.isfinite(T)
    axes[1].plot(r[m], T[m], "o-", ms=4, label="$T(r)=2\\pi\\rho\\nu L r^2(u'-u/r)$")
    Tana = 4 * math.pi * nu * abs(cc["B_analytic"])
    axes[1].axhline(-Tana, color="k", ls="--", label=f"$-4\\pi\\mu L|B|$ = {-Tana:.4f}")
    axes[1].axvspan(p["ri"] + 2, p["ro"] - 2, color="0.9", zorder=0)
    axes[1].set_xlabel("$r$ (lattice units)")
    axes[1].set_ylabel("torque (lattice units)")
    axes[1].set_title("torque identity: $r$-independent, matches $4\\pi\\mu L|B|$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_couette.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig 4: duct secondary flow -------------------------- #
    pos = [c for c in duct if c["params"]["Omega"] > 0]
    neg = [c for c in duct if c["params"]["Omega"] < 0]
    i_pos = duct.index(min(pos, key=lambda c: abs(c["params"]["Omega"]))) if pos else 0
    keys = list(fld.keys())
    tag = [k for k in keys if k.startswith("duct_pos") and k.endswith("psi")][0].replace("psi", "")
    uy = fld[tag + "uy"]
    uz = fld[tag + "uz"]
    psi = fld[tag + "psi"]
    fduy = fld[tag + "fd_uy"]
    fduz = fld[tag + "fd_uz"]
    fdpsi = fld[tag + "fd_psi"]
    N = uy.shape[0]
    c1 = np.arange(1, N + 1, dtype=float)
    Y, Z = np.meshgrid(c1, c1, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for ax, ps, uy_, uz_, title in (
        (axes[0], psi, uy, uz, "LBM"),
        (axes[1], fdpsi, fduy, fduz, "FD reference (creeping)"),
    ):
        cs = ax.contourf(Y, Z, ps / np.max(np.abs(ps)), levels=21, cmap="RdBu_r")
        ax.streamplot(c1, c1, uy_.T, uz_.T, color="k", density=1.1, linewidth=0.7, arrowsize=0.7)
        ax.set_title(f"secondary flow, $\\psi/\\max|\\psi|$ ({title})")
        ax.set_xlabel("$y$")
        ax.set_ylabel("$z$")
        fig.colorbar(cs, ax=ax, shrink=0.85)
    diff = np.abs(uy - fduy) + np.abs(uz - fduz)
    cs = axes[2].contourf(Y, Z, diff / max(np.max(np.abs(fduy)), np.max(np.abs(fduz))), levels=21, cmap="viridis")
    axes[2].set_title("$|u^{LBM}-u^{FD}|/\\max|u^{FD}|$")
    axes[2].set_xlabel("$y$")
    axes[2].set_ylabel("$z$")
    fig.colorbar(cs, ax=axes[2], shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_duct_secondary.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig 5: duct scaling --------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    om = np.array([c["params"]["Omega"] for c in duct])
    sm = np.array([c["secondary_max_lbm"] for c in duct])
    o = np.argsort(np.abs(om))
    axes[0].loglog(np.abs(om[o]), sm[o], "o-", label="LBM $\\max|u_{sec}|$")
    axes[0].loglog(np.abs(om[o]), sm[o][0] * np.abs(om[o]) / abs(om[o][0]), "k--", lw=1,
                   label="linear in $\\Omega$")
    axes[0].set_xlabel("$|\\Omega|$")
    axes[0].set_ylabel("secondary-flow amplitude")
    axes[0].set_title("Coriolis secondary flow is linear in rotation")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    ppos = [c for c in duct if c["params"]["Omega"] > 0]
    ppos = sorted(ppos, key=lambda c: c["params"]["Omega"])
    axes[1].plot([c["params"]["Omega"] for c in ppos],
                 [c["field_rel_err_vs_reference"] for c in ppos], "o-",
                 label="LBM vs creeping FD reference")
    axes[1].axhline(0.10, color="r", ls="--", lw=1, label="criterion at the lowest rate (10%)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$\\Omega$")
    axes[1].set_ylabel("relative field error")
    axes[1].set_title("deviation from the creeping reference grows with inertia")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_duct_scaling.png"), dpi=150)
    plt.close(fig)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
