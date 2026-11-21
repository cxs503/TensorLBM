"""Figures for the corrected cavity acceptance metric (Ghia Table II defect).

Reads ``artifacts/xval_aa_pattern/cavity_accept_corrected.json`` (written by
``benchmarks/xval/aa_fused_cavity_metric_recompute.py``) and produces

* ``fig7_ghia_table_defect.png`` -- the horizontal centre-line ``v`` profile at
  the finest grid against the BOUND Table II and against the retired scrambled
  copy, plus the point-wise residual of both;
* ``fig8_cavity_convergence_corrected.png`` -- ``L2`` of both centre lines vs
  grid size with the ``15 %`` / ``3 %`` gates and the metric floor, for the
  corrected table and (dashed) the retired one.

Run: ``python tests/gen_aa_fused_corrected_figures.py``
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(os.path.join(HERE, "..", "artifacts", "xval_aa_pattern"))
U = 0.1


def load():
    with open(os.path.join(ART, "cavity_accept_corrected.json")) as fh:
        return json.load(fh)


def _grids(d):
    """Grids that carry a corrected *and* a retired score, sorted by L."""
    out = []
    for key, rec in d["grids"].items():
        sc = rec.get("corrected") or rec.get("corrected_table_on_retired_profiles")
        if not sc:
            continue
        out.append((rec["L"], key, sc, rec))
    return sorted(out)


def fig7(d, prof_L=None):
    grids = [g for g in _grids(d) if "profiles_corrected" in g[3]]
    if not grids:
        return None
    L, key, sc, rec = grids[-1]
    p = rec["profiles_corrected"]
    ys = np.asarray(p["y_star"])
    v = np.asarray(p["uy"]) / U
    bound = np.asarray(d["tables"]["bound_v"])
    retired = np.asarray(d["tables"]["retired_v"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(v, ys, "-", lw=1.6, color="C0",
               label=f"AA-pattern fused, L={L} (corrected sampling)")
    ax[0].plot(bound[:, 1], bound[:, 0], "ks", ms=5, mfc="none",
               label="Ghia 1982 Table II (BOUND)")
    ax[0].plot(retired[:, 1], retired[:, 0], "rx", ms=7,
               label="retired scrambled copy")
    ax[0].set_xlabel(r"$v/U$")
    ax[0].set_ylabel(r"$x/H$" if False else r"$y/H$ station of the profile axis")
    ax[0].set_title("horizontal centre line, $v(x)$ at $y/H=1/2$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    vb = np.interp(bound[:, 0], ys, v)
    vr = np.interp(retired[:, 0], ys, v)
    ax[1].plot(bound[:, 0], vb - bound[:, 1], "s-", ms=4,
               label=f"vs BOUND   L2 = {sc['bound_L2_v_full']*100:.2f} %")
    ax[1].plot(retired[:, 0], vr - retired[:, 1], "x--", ms=6,
               label=f"vs retired L2 = {sc['retired_L2_v_full']*100:.2f} %")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel(r"$x/H$")
    ax[1].set_ylabel(r"$(v_{num} - v_{ref})/U$")
    ax[1].set_title("point-wise residual: the reference, not the solver, moved")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Ghia Table II harness defect: scrambled stations produce a "
                 "grid-independent 4.5 % floor", fontsize=10)
    fig.tight_layout()
    out = os.path.join(ART, "fig7_ghia_table_defect.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig8(d):
    grids = _grids(d)
    if len(grids) < 2:
        return None
    Ls = [g[0] for g in grids]
    u = [g[2]["bound_L2_u_full"] * 100 for g in grids]
    v = [g[2]["bound_L2_v_full"] * 100 for g in grids]
    v_old = [g[2]["retired_L2_v_full"] * 100 for g in grids]
    fu = [g[3]["metric_floor_bound"]["floor_u"] * 100 for g in grids]
    fv = [g[3]["metric_floor_bound"]["floor_v"] * 100 for g in grids]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(Ls, u, "o-", label=r"$L_2(u_x)$ vs Ghia Table I")
    ax.loglog(Ls, v, "s-", color="C2", label=r"$L_2(u_y)$ vs Ghia Table II (BOUND)")
    ax.loglog(Ls, v_old, "x--", color="C3",
              label=r"$L_2(u_y)$ vs retired scrambled table")
    ax.loglog(Ls, fu, ":", color="C0", lw=1, label="metric floor, $u$")
    ax.loglog(Ls, fv, ":", color="C2", lw=1, label="metric floor, $v$")
    ax.axhline(3.0, color="grey", lw=0.8)
    ax.text(Ls[0], 3.2, "3 % gate", fontsize=8, color="grey")
    ax.axhline(15.0, color="k", lw=0.8, ls="--")
    ax.text(Ls[0], 16.0, "15 % acceptance gate", fontsize=8)
    ax.set_xlabel("$L$ (fluid nodes per side)")
    ax.set_ylabel(r"$L_2$ error  [% of $U$]")
    ax.set_title("lid-driven cavity Re=100: acceptance vs grid")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(ART, "fig8_cavity_convergence_corrected.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    d = load()
    for fn in (fig7, fig8):
        out = fn(d)
        print("wrote", out)


if __name__ == "__main__":
    main()