#!/usr/bin/env python
"""P154W3-N11: build figures + a numbers JSON from the run artifacts.

Writes ``artifacts/p154w3_n11/figs/*.png`` and ``artifacts/p154w3_n11/summary.json``.
Nothing here computes physics: every number is read back from the per-case JSON
files written by ``scripts/p154w3_n11_mass_transfer.py``, so the report can be
assembled without hand-copying a single value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p154w3_n11"
FIG = ART / "figs"
SRC = ROOT / "src"
import sys  # noqa: E402

sys.path.insert(0, str(SRC))
import tensorlbm.convective_mass_transfer as cmt  # noqa: E402


def load(name: str) -> dict | None:
    p = ART / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_glob(prefix: str) -> list[dict]:
    out = []
    for p in sorted(ART.glob(f"{prefix}*.json")):
        out.append(json.loads(p.read_text()))
    return out


def fig_calibration() -> dict:
    d = load("calib")
    if not d:
        return {}
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    stats = {}
    for trt, style in ((True, "o-"), (False, "s--")):
        rows = [r for r in d["rows"] if r["trt"] is trt]
        pe = [r["Pe_cell"] for r in rows]
        ratio = [r["ratio"] for r in rows]
        ax.plot(pe, ratio, style, label="TRT, $\\Lambda=3/16$" if trt else "BGK")
        stats["trt" if trt else "bgk"] = {f"{r['Pe_cell']:.0f}": round(r["ratio"], 4) for r in rows}
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("cell Peclet number $Pe_\\Delta = u\\Delta/D$")
    ax.set_ylabel("$D_{eff}/D$")
    ax.set_title("Measured numerical diffusivity of the D2Q5 scalar lattice")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig1_calibration.png", dpi=140)
    plt.close(fig)
    return stats


def fig_leveque() -> dict:
    cases = load_glob("leveque_L_")
    if not cases:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    out = {}
    for c in cases:
        x = np.asarray(c.get("x") or np.maximum(np.arange(len(c["sh_x"])) - 0.5, 0.5))
        sh = np.asarray(c["sh_x"])
        sh_e = np.asarray(c["sh_x_exact"])
        lab = f"{c['label']} ($\\delta_c/\\Delta$ {c['fit_window']['delta_c_min']:.1f}-{c['fit_window']['delta_c_max']:.1f})"
        axes[0].loglog(x, sh, ".", ms=3, label=lab)
        out[c["label"]] = {
            "p_fit": round(c["sh_x_fit"]["p"], 5),
            "p_err_pct": round((c["sh_x_fit"]["p"] / (2.0 / 3.0) - 1.0) * 100.0, 3),
            "median_err_pct": round(c["sh_err_pct"]["median"], 3),
            "err_at_xmax_pct": round(c["sh_err_pct"]["at_x_max"], 3),
            "delta_c_min": round(c["fit_window"]["delta_c_min"], 2),
            "delta_c_max": round(c["fit_window"]["delta_c_max"], 2),
            "steps": c["steps"],
            "converged": c["converged"],
            "collision": c.get("collision", "trt"),
        }
    x0 = np.asarray(cases[0].get("x") or np.maximum(np.arange(len(cases[0]["sh_x"])) - 0.5, 0.5))
    xr = np.logspace(0.3, math.log10(x0[-1]), 200)
    axes[0].loglog(xr, cmt.leveque_local_sh(xr, cases[0]["shear"], cases[0]["D"]), "k-", lw=1.2,
                   label="Leveque exact")
    axes[0].set_xlabel("x (lattice)")
    axes[0].set_ylabel("$Sh_x$")
    axes[0].set_title("Leveque wall: local Sherwood number")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    for c in cases:
        x = np.asarray(c.get("x") or np.maximum(np.arange(len(c["sh_x"])) - 0.5, 0.5))
        rel = (np.asarray(c["sh_x"]) / np.asarray(c["sh_x_exact"]) - 1.0) * 100.0
        ok = np.isfinite(rel)
        axes[1].plot(x[ok], rel[ok], lw=1.2,
                     label=f"{c['label']}")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].axhspan(-3, 3, color="g", alpha=0.08, label="+/-3%")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-20, 20)
    axes[1].set_xlabel("x (lattice)")
    axes[1].set_ylabel("$Sh_x$ deviation from Leveque (%)")
    axes[1].set_title("Resolution trend")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_leveque.png", dpi=140)
    plt.close(fig)
    return out


def fig_leveque_field() -> dict:
    cases = load_glob("leveque_L_s100")
    if not cases:
        return {}
    c = cases[0]
    # reload the field from the stored G would need the npz; instead show the
    # wall-flux / Sh comparison as a profile plot
    return {}


def fig_blasius() -> dict:
    cases = load_glob("blasius_Sc")
    if not cases:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    out = {}
    for c in cases:
        x = np.asarray(c["x"])
        re_x = c["u_inf"] * x / c["nu"]
        sh = np.asarray(c["sh_x"])
        sc = c["Sc"]
        ok = np.isfinite(sh) & (re_x > 1)
        axes[0].loglog(re_x[ok], sh[ok], ".", ms=2.5, label=f"Sc={sc:g}")
        re_plot = np.logspace(0, math.log10(re_x.max()), 100)
        axes[0].loglog(re_plot, 0.332 * np.sqrt(re_plot) * sc ** (1.0 / 3.0), "k--", lw=0.8)
        out[f"Sc{sc:g}"] = {
            "p_fit": round(c["sh_vs_re_fit"]["p"], 5),
            "p_err_pct": round((c["sh_vs_re_fit"]["p"] / 0.5 - 1.0) * 100.0, 3),
            "median_err_pct": round(c["sh_err_pct"]["median"], 3),
            "max_abs_err_pct": round(c["sh_err_pct"]["max_abs"], 3),
            "Re_range": [round(c["fit_window"]["Re_min"], 1), round(c["fit_window"]["Re_max"], 1)],
            "delta_c_min": round(c["fit_window"]["delta_c_min"], 2),
            "delta_c_over_dx": round(c["fit_window"]["delta_c_min"] / 1.0, 2),
            "steps": c["steps"],
            "converged": c["converged"],
            "collision": c.get("collision", "trt"),
            "nan": c.get("nan_in_field"),
        }
    axes[0].set_xlabel("$Re_x$")
    axes[0].set_ylabel("$Sh_x$")
    axes[0].set_title("Blasius plate: $0.332\\,Re_x^{1/2}Sc^{1/3}$ (dashed)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    for c in cases:
        x = np.asarray(c["x"])
        re_x = c["u_inf"] * x / c["nu"]
        rel = (np.asarray(c["sh_x"]) / np.asarray(c["sh_x_exact"]) - 1.0) * 100.0
        ok = np.isfinite(rel) & (re_x >= 10)
        axes[1].semilogx(re_x[ok], rel[ok], lw=1.2, label=f"Sc={c['Sc']:g}")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].axhspan(-3, 3, color="g", alpha=0.08)
    axes[1].set_ylim(-25, 25)
    axes[1].set_xlabel("$Re_x$")
    axes[1].set_ylabel("$Sh_x$ deviation (%)")
    axes[1].set_title("Schmidt-number ladder ($Sc$ = 1 ... 1000)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_blasius.png", dpi=140)
    plt.close(fig)
    # exponent of Sh vs Sc at fixed Re
    return out


def fig_cylinder() -> dict:
    cases = load_glob("cylinder_Re")
    if not cases:
        return {}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    out = {}
    res = sorted({c["Re"] for c in cases})
    scs = sorted({s["Sc"] for c in cases for s in c["scalars"]})
    for sc in scs:
        pts = [(c["Re"], s["Sh"]) for c in cases for s in c["scalars"] if s["Sc"] == sc]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"LBM, Sc={sc:g}")
    re_plot = np.logspace(math.log10(min(res)), math.log10(max(res)), 200)
    for sc in scs:
        ax.plot(re_plot, [cmt.ranz_marshall_mean_sh(r, sc) for r in re_plot], "k--", lw=1.0,
                label=f"Ranz-Marshall, Sc={sc:g}")
        ax.plot(re_plot, [cmt.hilpert_mean_sh(r, sc) for r in re_plot], ":", lw=1.2,
                label=f"Hilpert, Sc={sc:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("mean Sh")
    ax.set_title("Cylinder in cross flow: mass transfer")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cylinder.png", dpi=140)
    plt.close(fig)
    for c in cases:
        out[f"Re{c['Re']:g}"] = {
            "Cd_mean": round(c["Cd_mean"], 4),
            "St": round(c["St"], 4) if isinstance(c["St"], (int, float)) else None,
            "blockage": round(c["blockage"], 4),
            "Sh": {f"Sc{s['Sc']:g}": round(s["Sh"], 4) for s in c["scalars"]},
            "dev_RM_pct": {f"Sc{s['Sc']:g}": round(s["dev_RM_pct"], 2) for s in c["scalars"]},
            "dev_Hilpert_pct": {f"Sc{s['Sc']:g}": round(s["dev_Hilpert_pct"], 2) for s in c["scalars"]},
            "dev_CB_pct": {f"Sc{s['Sc']:g}": round(s["dev_CB_pct"], 2) for s in c["scalars"]},
            "n_avg_steps": c["n_avg_steps"],
        }
    return out


def fig_analogy() -> dict:
    cases = load_glob("analogy_frozen_")
    coupled = load("analogy_coupled")
    if not cases and not coupled:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    out = {}
    for c in cases:
        x = np.asarray(c["x"])
        axes[0].loglog(x, c["j_D"], ".", ms=2.5, label=f"{c['label']} $j_D$")
        axes[0].loglog(x, c["j_H"], "-", lw=0.8, label=f"{c['label']} $j_H$")
        out[c["label"]] = {
            "jD_vs_jH_pct": round(c["jD_vs_jH_pct"], 3),
            "jD_vs_Cf_half_pct": round(c["jD_vs_Cf_half_pct"], 3),
            "sh_D_median": round(c.get("sh_D_median", float("nan")), 4),
            "sh_H_median": round(c.get("sh_H_median", float("nan")), 4),
            "sh_D_vs_analytic_pct": round(c.get("sh_D_vs_analytic_pct", float("nan")), 3),
            "sh_H_vs_analytic_pct": round(c.get("sh_H_vs_analytic_pct", float("nan")), 3),
            "n": c["n"],
        }
    axes[0].loglog(np.asarray(cases[0]["x"]), cases[0]["Cf_half"], "k--", lw=1.0,
                   label="$C_f/2$ (Blasius)")
    axes[0].set_xlabel("x (lattice)")
    axes[0].set_ylabel("$j$ factor")
    axes[0].set_title("Chilton-Colburn: $j_D$ vs $j_H$ vs $C_f/2$")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6)
    if coupled:
        x = np.asarray(coupled["x"])
        axes[1].plot(x, coupled["cf_lbm"], lw=1.5, label="$C_f$ LBM momentum")
        cfb = np.asarray(coupled["cf_blasius"], dtype=float)
        axes[1].plot(x[: cfb.size], cfb, "k--", lw=1.0, label="$C_f$ Blasius")
        for b in coupled["scalars"]:
            jx = np.asarray(b["j_x"], dtype=float)
            xx = x[: jx.size]
            axes[1].plot(xx, 2.0 * jx, lw=1.0, label=f"$2j$ ({b['label']})")
        axes[1].set_xlabel("x (lattice)")
        axes[1].set_ylabel("$C_f$, $2j$")
        axes[1].set_title("Coupled arm: skin friction vs Colburn $j$")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        out["coupled"] = {
            "Re_L": round(coupled["Re_L"], 2),
            "cf_err_pct_median": round(coupled["cf_err_pct_median"], 3),
            "cf_err_pct_max": round(coupled["cf_err_pct_max"], 3),
            "scalars": [{"label": b["label"], "j_median": round(b["j_median"], 5)} for b in coupled["scalars"]],
        }
    fig.tight_layout()
    fig.savefig(FIG / "fig5_analogy.png", dpi=140)
    plt.close(fig)
    return out


def fig_fields() -> dict:
    """Concentration fields (Leveque wall layer + Blasius plate)."""
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0))
    p1 = ART / "field_leveque.npz"
    if p1.exists():
        d = np.load(p1)
        G, x, dc = d["G"], d["x"], d["delta_c"]
        y = np.arange(G.shape[0])
        im = axes[0, 0].pcolormesh(x, y, G, shading="auto", cmap="viridis")
        axes[0, 0].set_yscale("symlog", linthresh=1.0)
        axes[0, 0].set_ylim(0, 30)
        axes[0, 0].set_xlabel("x (lattice)")
        axes[0, 0].set_ylabel("y (lattice)")
        axes[0, 0].set_title("Leveque: concentration field $G$ (symlog y)")
        fig.colorbar(im, ax=axes[0, 0])
        for j in range(20, G.shape[1] - 10, max((G.shape[1] - 30) // 8, 1)):
            axes[0, 1].plot(y, G[:, j], lw=1.0, label=f"x={x[j]:.0f} ($\\delta_c$={dc[j]:.2f})")
        out["leveque"] = {"shape": list(G.shape), "delta_c_max": round(float(dc.max()), 3)}
    p2 = ART / "field_blasius.npz"
    if p2.exists():
        d = np.load(p2)
        G, x = d["G"], d["x"]
        y = np.arange(G.shape[0])
        im = axes[1, 0].pcolormesh(x, y, G, shading="auto", cmap="inferno")
        axes[1, 0].set_ylim(0, 60)
        axes[1, 0].set_xlabel("x (lattice)")
        axes[1, 0].set_ylabel("y (lattice)")
        axes[1, 0].set_title(f"Blasius plate $G$, Sc={float(d['Sc']):g}")
        fig.colorbar(im, ax=axes[1, 0])
        sh, sh_e = d["sh_x"], d["sh_x_exact"]
        ok = np.isfinite(sh) & (x >= 10)
        axes[1, 1].semilogx(x[ok], sh[ok], "o", ms=2.5, label="LBM")
        axes[1, 1].semilogx(x[ok], sh_e[ok], "k-", lw=1.0, label="exact")
        axes[1, 1].set_xlabel("x (lattice)")
        axes[1, 1].set_ylabel("$Sh_x$")
        axes[1, 1].set_title("$Sh_x$: LBM vs $0.332 Re_x^{1/2}Sc^{1/3}$")
        axes[1, 1].grid(alpha=0.3, which="both")
        axes[1, 1].legend()
        out["blasius"] = {"shape": list(G.shape), "Sc": float(d["Sc"])}
    fig.tight_layout()
    fig.savefig(FIG / "fig6_fields.png", dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {
        "calibration": fig_calibration(),
        "leveque": fig_leveque(),
        "blasius": fig_blasius(),
        "cylinder": fig_cylinder(),
        "analogy": fig_analogy(),
        "fields": fig_fields(),
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])
    print(f"[written] {ART/'summary.json'}")
    print(f"[figs] {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()