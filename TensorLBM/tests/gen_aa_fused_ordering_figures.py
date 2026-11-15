"""Figures for the AA-pattern fused stream-collide ordering audit.

Reads ``artifacts/xval_aa_pattern/ordering_audit.json`` (operator ordering,
shear wave) and ``artifacts/xval_aa_pattern/cavity_arm_*.json`` (cavity ablation,
produced by the cluster shards), writes PNGs next to them.
"""

from __future__ import annotations

import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "xval_aa_pattern")

GHIA_U = np.array([
    [0.0000, 0.00000], [0.0547, -0.03717], [0.0625, -0.04192], [0.0703, -0.04775],
    [0.1016, -0.06434], [0.1719, -0.10150], [0.2813, -0.15662], [0.4531, -0.21090],
    [0.5000, -0.20581], [0.6172, -0.13641], [0.7344, 0.00332], [0.8516, 0.23151],
    [0.9531, 0.68717], [0.9609, 0.73722], [0.9688, 0.78871], [0.9766, 0.84123],
    [1.0000, 1.00000],
])
GHIA_V = np.array([
    [0.0000, 0.00000], [0.0625, 0.09233], [0.0703, 0.10091], [0.0781, 0.10890],
    [0.0938, 0.12317], [0.1563, 0.16077], [0.2266, 0.17507], [0.2344, 0.17527],
    [0.5000, 0.05454], [0.7344, -0.24533], [0.8516, -0.18866], [0.9531, -0.03717],
    [0.9609, 0.00730], [0.9688, 0.02526], [0.9766, 0.05248], [1.0000, 0.00000],
])


def fig_ordering(res):
    e1 = res["E1_E2_operator_ordering"]
    trace = e1["divergence_trace_nonuniform"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = ["uniform\n(old test IC)", "non-uniform"]
    fused = [e1["uniform"]["max_rel_diff_aa_fused_vs_canonical"],
             e1["nonuniform"]["max_rel_diff_aa_fused_vs_canonical"]]
    legacy = [e1["uniform"]["max_rel_diff_audited_vs_canonical"],
              e1["nonuniform"]["max_rel_diff_audited_vs_canonical"]]
    x = np.arange(2)
    ax[0].bar(x - 0.2, np.maximum(fused, 1e-18), 0.4,
              label="AA fused pull (fixed) = S[C[f]]", color="#2b7bba")
    ax[0].bar(x + 0.2, np.maximum(legacy, 1e-18), 0.4,
              label="stream-then-collide (retired) = C[S[f]]", color="#c0392b")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel(r"max relative deviation from canonical $S[C[f]]$")
    ax[0].set_title("Operator equivalence: the published test was degenerate")
    ax[0].axhline(1e-14, ls="--", c="k", lw=1, label="old tolerance 1e-14")
    ax[0].legend(fontsize=8)

    steps = [t["step"] for t in trace]
    diffs = [t["max_rel_diff"] for t in trace]
    ax[1].loglog(steps, diffs, "o-", color="#c0392b")
    ax[1].set_xlabel("LBM step")
    ax[1].set_ylabel(r"max relative difference (populations)")
    ax[1].set_title("Two orderings diverge at once (non-uniform IC, $\\tau=0.8$)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(ART, "fig1_ordering_equivalence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_shear(res):
    rows = {r["scheme"]: r for r in res["E3_shear_wave"]}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ny = rows["fused"]["ny"]
    U = rows["fused"]["U"]
    nu = rows["fused"]["nu_target"]
    k = 2 * math.pi / ny
    t = np.linspace(0, rows["fused"]["steps"], 200)
    ax[0].semilogy(t, U * np.exp(-nu * k**2 * t), "k-", lw=2, label="analytic $e^{-\\nu k^2 t}$")
    for name, c in (("fused", "#2b7bba"), ("audited", "#c0392b")):
        r = rows[name]
        tt = np.array([0, r["steps"]])
        aa = np.array([r["amplitude_initial"], r["amplitude_final"]])
        ax[0].semilogy(tt, aa, "o--", color=c,
                       label=f"{name}: $\\nu_{{eff}}/\\nu-1$ = {r['nu_rel_error']*100:+.2f}%")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("shear-wave amplitude")
    ax[0].set_title("Viscous decay of $u=A\\sin(ky)e^{-\\nu k^2t}$ (Re-independent anchor)")
    ax[0].legend(fontsize=8)

    names = ["canonical", "fused", "audited"]
    errs = [abs(rows[n]["nu_rel_error"]) * 100 for n in names]
    ax[1].bar(names, errs, color=["#7f8c8d", "#2b7bba", "#c0392b"])
    ax[1].axhline(3.0, ls="--", c="k", lw=1, label="owner 3% gate")
    ax[1].set_ylabel(r"$|\nu_{eff}/\nu - 1|$  [%]")
    ax[1].set_title("Effective viscosity error (honest negative: ordering is\nnearly invisible in a smooth single mode)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(ART, "fig2_shear_wave_viscosity.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_cavity():
    files = sorted(glob.glob(os.path.join(ART, "cavity_arm_*.json")))
    if not files:
        return []
    arms = [json.load(open(f)) for f in files]
    out = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = ["#2b7bba", "#e67e22", "#8e44ad", "#c0392b", "#16a085"]
    for a, c in zip(arms, colors):
        if "profiles" not in a:
            continue
        prof = a["profiles"]
        ys = np.array(prof["y_star"])
        lab = f"{a['arm']} L={a['L']}"
        l2u = a["checkpoints"][max(a["checkpoints"])]["L2_u_vs_ghia"] * 100
        ax[0].plot(prof["ux_centerline"], ys, "-", color=c, lw=1.4,
                   label=f"{lab}  ($L2_u$={l2u:.1f}%)")
        ax[1].plot(ys, prof["uy_centerline"], "-", color=c, lw=1.4, label=lab)
    ax[0].plot(GHIA_U[:, 1], GHIA_U[:, 0], "ks", ms=4, label="Ghia et al. (1982)")
    ax[1].plot(GHIA_V[:, 0], GHIA_V[:, 1], "ks", ms=4, label="Ghia et al. (1982)")
    ax[0].set_xlabel("$u/U_{lid}$ at $x=0.5$")
    ax[0].set_ylabel("$y/L$")
    ax[0].set_title("Lid-driven cavity Re=100: vertical centerline")
    ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("$x/L$")
    ax[1].set_ylabel("$v/U_{lid}$ at $y=0.5$")
    ax[1].set_title("Horizontal centerline")
    ax[1].legend(fontsize=6.5)
    fig.tight_layout()
    p = os.path.join(ART, "fig3_cavity_profiles_vs_ghia.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    # convergence of L2_u with steps
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for a, c in zip(arms, colors):
        ck = a["checkpoints"]
        st = sorted(int(s) for s in ck)
        vals = [ck[str(s)]["L2_u_vs_ghia"] * 100 for s in st]
        ax.semilogx(st, vals, "o-", color=c, label=f"{a['arm']} L={a['L']}")
    ax.axhline(15.0, ls="--", c="k", lw=1, label="15% acceptance")
    ax.axhline(3.0, ls=":", c="k", lw=1, label="3% gate")
    ax.set_xlabel("LBM step")
    ax.set_ylabel(r"$L_2$ error of $u$ vs Ghia [% of $U_{lid}$]")
    ax.set_title("Cavity convergence by arm (ablation)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(ART, "fig4_cavity_convergence.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def main():
    with open(os.path.join(ART, "ordering_audit.json")) as fh:
        res = json.load(fh)
    made = [fig_ordering(res), fig_shear(res)]
    made += fig_cavity()
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()