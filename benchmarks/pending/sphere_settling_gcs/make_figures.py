#!/usr/bin/env python3
"""Figures for the GCS settling study — every number read from result.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def load():
    return json.loads((HERE / "result.json").read_text())


def linfit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (k, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return k, b


def runs_of(res, key, value):
    return [r for r in res["runs"] if abs(r[key] - value) < 1e-12]


def fig_settling(res):
    dom = sorted(runs_of(res, "R", 4.0), key=lambda r: r["lambda_R_over_L"])
    ref = sorted(runs_of(res, "Re_target", 0.05), key=lambda r: r["R"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lam = [r["lambda_R_over_L"] for r in dom]
    ut = [r["control_surface"]["2.5"]["u_terminal_over_stokes"] for r in dom]
    k, b = linfit(lam, ut)
    xx = np.linspace(0, max(lam) * 1.08, 50)
    axes[0].plot(lam, ut, "o", ms=7, label="simulation (R=4, Re=0.05)")
    axes[0].plot(xx, k * xx + b, "--", label=f"fit: slope {k:+.3f}, λ→0 intercept {b:.4f}")
    axes[0].axhline(1.0, color="k", lw=1, ls=":", label="analytic Stokes U_t")
    axes[0].set_xlabel(r"confinement $\lambda = R/L$")
    axes[0].set_ylabel(r"$U_t^{\rm sim} / U_t^{\rm Stokes}$")
    axes[0].set_title("(a) terminal velocity vs confinement")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    Rs = [r["R"] for r in ref]
    fr = [r["control_surface"]["2.5"]["drag_over_stokes_at_measured_U"] for r in ref]
    k2, b2 = linfit([1 / R for R in Rs], fr)
    xx2 = np.linspace(0, max(1 / R for R in Rs) * 1.08, 50)
    axes[1].plot([1 / R for R in Rs], fr, "s", ms=7, label="simulation (λ=1/8, Re=0.05)")
    axes[1].plot(xx2, k2 * xx2 + b2, "--", label=f"fit: 1/R→0 intercept {b2:.4f}")
    axes[1].axhline(1.0, color="k", lw=1, ls=":", label="Stokes 6πμaU at measured U")
    axes[1].set_xlabel("1/R  (inverse sphere radius in cells)")
    axes[1].set_ylabel(r"$F / (6\pi\mu R U_{\rm meas})$")
    axes[1].set_title("(b) drag law vs voxel resolution")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_settling_vs_stokes.png", dpi=160)
    plt.close(fig)


def fig_series(res):
    dom = sorted(runs_of(res, "R", 4.0), key=lambda r: r["lambda_R_over_L"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in dom:
        ser = np.asarray(r["control_surface"]["2.5"]["series"], float)
        t = np.arange(len(ser)) * r["sample_every"]
        axes[0].plot(t, ser / ser[-1], lw=1.2, label=f"L={r['L']} (λ={r['lambda_R_over_L']:.3f})")
    axes[0].set_xlabel("step (after sampling start)")
    axes[0].set_ylabel(r"$F_x(t) / F_x(\mathrm{end})$")
    axes[0].set_title("(a) instantaneous control-surface force (tail)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    tags, ptp, rms = [], [], []
    for r in res["runs"]:
        tags.append(r["tag"])
        ptp.append(100 * r["control_surface"]["2.5"]["ptp_over_mean"])
        rms.append(100 * r["control_surface"]["2.5"]["rms_over_mean"])
    order = np.argsort(ptp)
    axes[1].barh(
        [tags[i] for i in order], [ptp[i] for i in order], color="tab:blue", alpha=0.8
    )
    axes[1].axvline(2.0, color="r", ls="--", lw=1, label="2% acceptance line")
    axes[1].set_xlabel("peak-to-peak force oscillation / mean  [%]")
    axes[1].set_title("(b) oscillation amplitude, all runs")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_force_series_oscillation.png", dpi=160)
    plt.close(fig)


def fig_instruments(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in res["runs"]:
        offs = sorted(float(k) for k in r["control_surface"])
        vals = [r["control_surface"][f"{o:g}"]["fx_total"] for o in offs]
        axes[0].plot(offs, vals, "o-", lw=1, ms=5, label=r["tag"])
    axes[0].set_xlabel(r"control-surface offset $r_s - R$  [cells]")
    axes[0].set_ylabel(r"$F_x$  [lattice]")
    axes[0].set_title("(a) control-radius independence")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)

    labels, vals = [], []
    res_j3 = res.get("verdict", {}).get("J3_anchor")
    if res_j3:
        labels.append("GCS $F$")
        vals.append(res_j3["F_total"])
        labels.append(r"discrete anchor $(1-1/2\tau)\rho g V_f$")
        vals.append(res_j3["discrete_anchor"])
        labels.append(r"continuum $\rho g V_f$")
        vals.append(res_j3["continuum_rho_g_Vfluid"])
    res_j6 = res.get("verdict", {}).get("J6_cross_instrument")
    if res_j6:
        labels.append("box-CV (corrected)")
        vals.append(res_j6["box_cv_corrected_fx"])
    if vals:
        axes[1].bar(labels, vals, color=["tab:blue", "k", "gray", "tab:orange"][: len(vals)])
        for i, v in enumerate(vals):
            axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        axes[1].set_title("(b) exact anchor + independent instrument")
        axes[1].set_ylabel(r"$F_x$  [lattice]")
        axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_instruments.png", dpi=160)
    plt.close(fig)


def fig_reynolds(res):
    ran = sorted(
        [r for r in res["runs"] if abs(r["R"] - 6.0) < 1e-12], key=lambda r: r["Re_measured"]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    re_ = [r["Re_measured"] for r in ran]
    ut = [r["control_surface"]["2.5"]["u_terminal_over_stokes"] for r in ran]
    dr = [r["control_surface"]["2.5"]["drag_over_stokes_at_measured_U"] for r in ran]
    ax.semilogx(re_, ut, "o-", label=r"$U_t/U_t^{\rm Stokes}$ (R=6, λ=1/8)")
    ax.semilogx(re_, dr, "s--", label=r"$F/(6\pi\mu R U_{\rm meas})$")
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_xlabel("Re = 2 R U/ν")
    ax.set_ylabel("ratio to Stokes")
    ax.set_title("finite-Re boundary of the method")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_reynolds_scan.png", dpi=160)
    plt.close(fig)


def fig_field():
    p = HERE / "field_snapshot.json"
    if not p.exists():
        return
    snap = json.loads(p.read_text())
    ux = np.asarray(snap["u_x_mid"])
    sp = np.asarray(snap["speed_mid"])
    solid = np.asarray(snap["solid_mid"])
    L = snap["L"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, arr, ttl, unit in (
        (axes[0], ux, r"(a) $u_x$ (streamwise)", r"$u_x$"),
        (axes[1], sp, r"(b) $|\mathbf{u}|$", r"$|\mathbf{u}|$"),
    ):
        im = ax.imshow(arr, origin="lower", cmap="viridis")
        ax.contour(solid, levels=[0.5], colors="w", linewidths=1.2)
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            snap["control_radius"] * np.cos(th) + L / 2 - 0.5,
            snap["control_radius"] * np.sin(th) + L / 2 - 0.5,
            "r--",
            lw=1.2,
            label="control surface $R+2.5$",
        )
        ax.set_title(ttl)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, label=unit, fraction=0.046)
    fig.suptitle(
        f"settling cell mid-plane, R={snap['R']:g}, L={snap['L']}, "
        f"$U_t$ target={snap['u_target']:.5f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig5_field_snapshot.png", dpi=160)
    plt.close(fig)


def main() -> int:
    res = load()
    fig_settling(res)
    fig_series(res)
    fig_instruments(res)
    fig_reynolds(res)
    fig_field()
    print("figures ->", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())