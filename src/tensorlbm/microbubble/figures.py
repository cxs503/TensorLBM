"""Figures for the IN62-N04 report (matplotlib, Agg backend)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .loglaw import u_plus


def _save(fig, out: Path, name: str) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    png = buf.getvalue()
    path = out / f"{name}.png"
    path.write_bytes(png)
    return {"png_base64": base64.b64encode(png).decode("ascii"), "path": str(path)}


def fig_transport(res: dict, out: Path) -> dict:
    tr = res["transport"]
    yn = np.asarray(tr["y"])
    yn = yn / (yn[-1] + 0.5 * (yn[1] - yn[0]))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(tr["alpha_analytic"], yn, "k-", lw=2, label="closed form")
    ax.plot(tr["alpha_fv"], yn, "C0o", ms=3.2, label="finite volume")
    ax.plot(tr["alpha_lbm"], yn, "C3s", ms=3.2, mfc="none", label="D2Q9 ADE LBM")
    ax.set_xlabel(r"void fraction $\alpha$")
    ax.set_ylabel(r"$y/H_{layer}$")
    ax.set_title(rf"void profile, $Pe={tr['Pe']:.2f}$, $\alpha_w={tr['alpha_w']}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_d = _save(fig, out, "fig1_alpha_profile")
    return {"name": "fig1_alpha_profile", **out_d,
            "caption": (rf"Void-fraction profile from the three transport solvers "
                        rf"($Pe={tr['Pe']:.2f}$). Max deviation from the closed "
                        rf"form: FV {100*tr['err_fv_max']:.2f}%, "
                        rf"LBM {100*tr['err_lbm_max']:.2f}%.")}


def fig_laminar(res: dict, out: Path) -> dict:
    lam = res["laminar_lbm"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for c, col in zip(lam["cases"], ("k", "C0", "C2")):
        lab = (r"clean" if c["alpha_layer"] == 0
               else rf"$\alpha={c['alpha_layer']}$, $h/H={c['h_over_H']:.2f}$")
        ye = np.linspace(0.0, 1.0, len(c["u_exact_over_U"]))
        ax.plot(np.interp(c["y"], ye, c["u_exact_over_U"]), c["y"], f"{col}-",
                lw=1.6, label=lab + " (exact)")
        ax.plot(c["u_over_U"], c["y"], f"{col}o", ms=3.0, mfc="none")
    ax.set_xlabel(r"$u/U_{bulk}$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title("mixture LBM vs exact two-layer solution")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    drs = [c["DR_exact"] for c in lam["cases"]]
    drl = [c["DR_lbm"] for c in lam["cases"]]
    x = np.arange(len(drs))
    bx.bar(x - 0.2, drs, 0.4, label="exact", color="k")
    bx.bar(x + 0.2, drl, 0.4, label="LBM", color="C0")
    bx.set_xticks(x)
    bx.set_xticklabels([r"$\alpha$=%.2f" % c["alpha_layer"] for c in lam["cases"]],
                       fontsize=8)
    bx.set_ylabel("DR (matched throughput)")
    bx.set_title("laminar drag reduction: LBM vs exact")
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3, axis="y")
    out_d = _save(fig, out, "fig2_laminar_lbm")
    errs = [abs(c["rel_error_DR"]) for c in lam["cases"][1:]]
    return {"name": "fig2_laminar_lbm", **out_d,
            "caption": (f"Two-phase mixture LBM channel at matched throughput against "
                        f"the exact two-layer solution; max DR error "
                        f"{100*max(errs):.2f}%.")}


def fig_loglaw(res: dict, out: Path) -> dict:
    alphas = [0.0, 0.1, 0.2, 0.3]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    yp = np.geomspace(1.0, 1e3, 300)
    for a in alphas:
        ax.semilogx(yp, u_plus(yp, a), lw=1.8,
                    label=rf"$\alpha_w={a}$ ($\kappa_{{eff}}={0.41*np.sqrt(1-a):.3f}$)")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$u^+$")
    ax.set_title("modified log law (density-weighted mixing length)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    out_d = _save(fig, out, "fig3_loglaw")
    return {"name": "fig3_loglaw", **out_d,
            "caption": (r"Modified log law $u^+=\kappa_{eff}^{-1}\ln y^+ + B_{eff}$ "
                        r"with $\kappa_{eff}=\kappa\sqrt{1-\alpha}$ and the intercept "
                        r"pinned by sublayer continuity at $y^+=11$.")}


def fig_dr(res: dict, out: Path) -> dict:
    rows = res["turbulent"]["rows"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    rts = sorted({r["re_theta"] for r in rows})
    for rt, col in zip(rts, ("C0", "C1", "C2", "C3")):
        sel = [r for r in rows if abs(r["re_theta"] - rt) < 1e-6]
        a = [r["alpha_w"] for r in sel]
        ax.plot(a, [r["DR_bl"] for r in sel], f"{col}-o", ms=3, label=rf"$Re_\theta={rt:.0f}$")
        ax.plot(a, [r["DR_ana"] for r in sel], f"{col}--", lw=1.0)
    ax.set_xlabel(r"wall void fraction $\alpha_w$")
    ax.set_ylabel("DR")
    ax.set_title("DR vs $\\alpha_w$ (solid: mixing-length BL, dashed: log law)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    sel = [r for r in rows if abs(r["re_theta"] - 2000.0) < 1e-6]
    a = [r["alpha_w"] for r in sel]
    bx.plot(a, [r["DR_mixing_length"] for r in sel], "C0-o", ms=3, label="M2 only (BL)")
    bx.plot(a, [r["DR_loglaw"] for r in sel], "C0--", lw=1.1, label="M2 only (log law)")
    bx.plot(a, [1 - (1 - x) for x in a], "C7:", lw=1.2, label="M1 only ($\\alpha_w$)")
    bx.plot(a, [r["DR_bl"] for r in sel], "k-", lw=1.8, label="total (BL)")
    bx.set_xlabel(r"$\alpha_w$")
    bx.set_ylabel("DR")
    bx.set_title(r"mechanism split at $Re_\theta=2000$")
    bx.legend(fontsize=7.5)
    bx.grid(alpha=0.3)
    out_d = _save(fig, out, "fig4_dr_vs_alpha")
    return {"name": "fig4_dr_vs_alpha", **out_d,
            "caption": (r"Drag reduction versus wall void fraction for "
                        r"$Re_\theta=1000..5000$: the numerical boundary-layer route "
                        r"(solid) and the profile-aware log-law anchor (dashed); "
                        r"right panel splits the two mechanisms.")}


def fig_cf(res: dict, out: Path) -> dict:
    rows = res["turbulent"]["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sel0 = [r for r in rows if r["alpha_w"] == 0.0]
    ax.loglog([r["re_theta"] for r in sel0], [r["cf_clean"] for r in sel0],
              "k-o", ms=4, label=r"clean BL solver")
    for aw, col in ((0.1, "C0"), (0.2, "C2"), (0.3, "C3")):
        sel = [r for r in rows if abs(r["alpha_w"] - aw) < 1e-9]
        ax.loglog([r["re_theta"] for r in sel], [r["cf_bubbly"] for r in sel],
                  f"{col}-s", ms=4, label=rf"$\alpha_w={aw}$")
    ax.set_xlabel(r"$Re_\theta$")
    ax.set_ylabel(r"$C_f$")
    ax.set_title("skin friction with and without the bubble layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    out_d = _save(fig, out, "fig5_cf_vs_retheta")
    return {"name": "fig5_cf_vs_retheta", **out_d,
            "caption": r"$C_f(Re_\theta)$ from the mixing-length solver with and without the bubble layer."}


def fig_power(res: dict, out: Path) -> dict:
    pw = res["power"]
    a = np.asarray(pw["alpha_w"])
    etas = np.asarray(pw["eta_c"])
    net = np.asarray(pw["net_power_density"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for j, e in enumerate(etas):
        if e in (0.3, 0.5, 0.6, 0.8):
            ax.plot(a, net[:, j], "-o", ms=3, label=rf"$\eta_c={e}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\alpha_w$")
    ax.set_ylabel(r"$P_{net}$ [W/m$^2$]")
    ax.set_title(rf"net power at $U={pw['U']:.0f}$ m/s, $L={pw['length']:.0f}$ m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    cs = bx.contourf(a, etas, net.T, levels=12, cmap="viridis")
    fig.colorbar(cs, ax=bx, label=r"$P_{net}$ [W/m$^2$]")
    bx.set_xlabel(r"$\alpha_w$")
    bx.set_ylabel(r"$\eta_c$")
    bx.set_title("profitability map")
    out_d = _save(fig, out, "fig6_net_power")
    return {"name": "fig6_net_power", **out_d,
            "caption": (rf"Net power ledger: saved drag power minus compressor power "
                        rf"(gas flow from the transport solve), best "
                        rf"{pw['best_net_power_density']:.1f} W/m$^2$ at "
                        rf"$\alpha_w={pw['best_alpha_w']:.2f}$.")}


def make_all(res: dict, out: Path | str = "artifacts/microbubble_in62_n04") -> list:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figs = [
        fig_transport(res, out),
        fig_laminar(res, out),
        fig_loglaw(res, out),
        fig_dr(res, out),
        fig_cf(res, out),
        fig_power(res, out),
    ]
    return figs