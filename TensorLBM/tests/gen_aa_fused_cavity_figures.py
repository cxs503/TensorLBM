"""Figures + summary table for the AA-pattern fused kernel acceptance sweep.

Reads ``TensorLBM/artifacts/xval_aa_pattern/cavity_accept_*.json`` (produced by
``benchmarks/xval/aa_fused_cavity_acceptance.py``, one file per scheduler shard)
and writes

* ``cavity_accept_summary.json`` -- the numeric table (no hand-typed digits),
* ``fig3_couette_gate.png``       -- exact-solution gate for the moving wall,
* ``fig4_cavity_profiles.png``    -- centre lines vs Ghia (1982), L = 64/128/192,
* ``fig5_cavity_field.png``       -- field contour + streamlines (converged field),
* ``fig6_cavity_convergence.png`` -- L2 vs L with the 15 % / 3 % gates and the
  metric floor, plus steady-state residual and fluid-mass drift.

Run: ``python tests/gen_aa_fused_cavity_figures.py``
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(os.path.join(HERE, "..", "artifacts", "xval_aa_pattern"))
# Ghia (1982) tables: single source of truth (see
# docs/xval_aa_pattern_ghia_table_defect.md).  The inline copy that used to live
# here had a scrambled Table II, which also corrupted the published figures.
import sys as _sys

_sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "xval"))
from aa_fused_stream_collide_ordering_audit import (  # noqa: E402
    GHIA_U,
    GHIA_V,
)


def load():
    out = {}
    for p in sorted(glob.glob(os.path.join(ART, "cavity_accept_*.json"))):
        with open(p) as fh:
            d = json.load(fh)
        out[d.get("task_name", os.path.basename(p))] = d
    return out


def fig_couette(d):
    if "rel_err_max" not in d:
        return None
    L = d["L"]
    y = (np.arange(1, L + 1) - 0.5) / L
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].plot(np.asarray(d["profile_numeric"]) / d["U"], y, "o", ms=4,
               label="AA fused kernel")
    ax[0].plot(np.asarray(d["profile_analytic"]) / d["U"], y, "-", lw=1.2,
               label="exact $u_x=U\\,(y-1/2)/L$")
    ax[0].set_xlabel("$u_x/U$"); ax[0].set_ylabel("$y/L$")
    ax[0].set_title(f"plane Couette, L={L}, tau={d['tau']:.2f}, {d['steps']} steps")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    err = (np.asarray(d["profile_numeric"]) - np.asarray(d["profile_analytic"])) / d["U"]
    ax[1].semilogy(y, np.abs(err) + 1e-17, "o-", ms=3)
    ax[1].axhline(1e-2, color="r", ls="--", lw=1, label="1 % gate")
    ax[1].set_xlabel("$y/L$"); ax[1].set_ylabel("$|u_{num}-u_{exact}|/U$")
    ax[1].set_title(f"max rel. error {d['rel_err_max']:.1e}")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(ART, "fig3_couette_gate.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_profiles(cav):
    cav = [d for d in cav if "profiles" in d and d.get("fluid_finite")]
    if not cav:
        return None
    cav.sort(key=lambda d: d["L"])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for d in cav:
        ys = np.asarray(d["profiles"]["y_star"])
        ax[0].plot(np.asarray(d["profiles"]["ux"]) / d["U"], ys, "-", lw=1.3,
                   label=f"L={d['L']}  L2={d['L2_u_full']*100:.1f}%")
        ax[1].plot(ys, np.asarray(d["profiles"]["uy"]) / d["U"], "-", lw=1.3,
                   label=f"L={d['L']}  L2={d['L2_v_full']*100:.1f}%")
    ax[0].plot(GHIA_U[:, 1], GHIA_U[:, 0], "ks", ms=5, mfc="none", label="Ghia 1982")
    ax[1].plot(GHIA_V[:, 0], GHIA_V[:, 1], "ks", ms=5, mfc="none", label="Ghia 1982")
    ax[0].set_xlabel("$u_x/U$ (vertical centre line)")
    ax[0].set_ylabel("$y/L$")
    ax[1].set_xlabel("$x/L$")
    ax[1].set_ylabel("$u_y/U$ (horizontal centre line)")
    ax[1].invert_xaxis()
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle("AA-pattern fused stream-collide: lid-driven cavity, Re=100", fontsize=10)
    fig.tight_layout()
    p = os.path.join(ART, "fig4_cavity_profiles.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_field(cav):
    best = None
    for d in cav:
        f = os.path.join(ART, f"cavity_accept_field_L{d['L']}.npz")
        if os.path.exists(f):
            best = (d, f)
    if best is None:
        return None
    d, path = best
    z = np.load(path)
    wt, L, U = int(z["wt"]), int(z["L"]), float(z["U"])
    ux = z["ux"][wt:wt + L, wt:wt + L] / U
    uy = z["uy"][wt:wt + L, wt:wt + L] / U
    sp = np.sqrt(ux ** 2 + uy ** 2)
    x = (np.arange(L) + 0.5) / L
    y = (np.arange(L) + 0.5) / L
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))
    c0 = ax[0].contourf(x, y, sp, levels=24, cmap="viridis")
    fig.colorbar(c0, ax=ax[0], label="$|u|/U$")
    ax[0].set_title(f"velocity magnitude, L={L}, Re=100")
    skip = max(1, L // 32)
    ax[1].streamplot(x, y, ux, uy, density=1.1, linewidth=0.6, color=sp,
                     cmap="viridis", arrowsize=0.7)
    # psi = integral ux dy, contour lines of the stream function
    psi = np.cumsum(ux, axis=0) / L
    cs = ax[1].contour(x, y, psi, levels=10, colors="w", linewidths=0.5)
    ax[1].clabel(cs, inline=True, fontsize=5, fmt="%.4f")
    ax[1].set_title("streamlines / stream function")
    for a in ax:
        a.set_xlabel("$x/L$"); a.set_ylabel("$y/L$"); a.set_aspect("equal")
    fig.suptitle("AA-pattern fused kernel, converged cavity field", fontsize=10)
    fig.tight_layout()
    p = os.path.join(ART, "fig5_cavity_field.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_convergence(cav):
    cav = [d for d in cav if "L2_u_full" in d]
    if not cav:
        return None
    cav.sort(key=lambda d: d["L"])
    Ls = np.array([d["L"] for d in cav], float)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].loglog(Ls, [d["L2_u_full"] for d in cav], "o-", label="$L_2(u_x)$ full")
    ax[0].loglog(Ls, [d["L2_v_full"] for d in cav], "s-", label="$L_2(u_y)$ full")
    ax[0].loglog(Ls, [d["metric_floor"]["floor_u"] for d in cav], "k--", lw=1,
                 label="metric floor (exact profile)")
    ax[0].axhline(0.15, color="r", ls="-", lw=1, label="15 % acceptance gate")
    ax[0].axhline(0.03, color="orange", ls=":", lw=1, label="3 % target")
    ax[0].set_xlabel("$L$"); ax[0].set_ylabel("$L_2$ error vs Ghia (1982)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
    ax[0].set_title("grid convergence, Re=100")

    for d in cav:
        h = [r for r in d["history"] if "residual" in r]
        if h:
            ax[1].loglog([r["step"] for r in h], [max(r["residual"], 1e-16) for r in h],
                         lw=1, label=f"L={d['L']}")
    ax[1].axhline(1e-9, color="r", ls="--", lw=1, label="steady criterion")
    ax[1].set_xlabel("step"); ax[1].set_ylabel(r"$\max|\Delta u|/U$ per 1000 steps")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    ax[1].set_title("steady-state residual")

    for d in cav:
        h = [r for r in d["history"] if "fluid_mass_rel_drift" in r]
        if h:
            ax[2].semilogy([r["step"] for r in h],
                           [max(r["fluid_mass_rel_drift"], 1e-17) for r in h], lw=1,
                           label=f"L={d['L']}")
    ax[2].axhline(1e-10, color="r", ls="--", lw=1, label="1e-10")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("relative fluid-mass drift")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, which="both")
    ax[2].set_title("conservation (fluid nodes)")
    fig.tight_layout()
    p = os.path.join(ART, "fig6_cavity_convergence.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def summary(data):
    tab = {}
    for name, d in data.items():
        if d.get("task") == "couette":
            tab["couette_gate"] = {k: d[k] for k in
                                   ("L", "tau", "steps", "rel_err_max",
                                    "fluid_mass_drift_rel", "gate_1pct")}
        elif d.get("task") == "equality_cavity":
            tab["equality_gate"] = {k: d[k] for k in
                                    ("L", "steps", "max_abs_diff", "bit_equal")}
        elif d.get("task") == "cavity" and "L2_u_full" in d:
            tab[f"cavity_L{d['L']}"] = {
                "L": d["L"], "tau": d["tau"], "steps_run": d["steps_run"],
                "converged_at": d["converged_at"],
                "L2_u_full": d["L2_u_full"], "L2_v_full": d["L2_v_full"],
                "L2_u_interior": d["L2_u_interior"],
                "L2_v_interior": d["L2_v_interior"],
                "metric_floor_u": d["metric_floor"]["floor_u"],
                "fluid_mass_rel_drift": d["fluid_mass_rel_drift"],
                "rho_fluid_range": d["rho_fluid_range"],
                "gate_15pct": d["gate_15pct"], "gate_3pct": d["gate_3pct"],
            }
    with open(os.path.join(ART, "cavity_accept_summary.json"), "w") as fh:
        json.dump(tab, fh, indent=2)
    return tab


def main():
    data = load()
    cav = [d for d in data.values() if d.get("task") == "cavity"]
    made = {}
    for key, fn in (("couette", lambda: fig_couette(data.get("couette", {}))),
                    ("profiles", lambda: fig_profiles(cav)),
                    ("field", lambda: fig_field(cav)),
                    ("convergence", lambda: fig_convergence(cav))):
        p = fn()
        if p:
            made[key] = p
    tab = summary(data)
    print(json.dumps({"figures": made, "table": tab}, indent=2))


if __name__ == "__main__":
    main()