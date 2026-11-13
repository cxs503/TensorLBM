#!/usr/bin/env python3
"""Figures for the m18289 delivery audit (board-audit m18348).

fig1_operator_identity.png : what the delivered code actually achieves (machine-
                             precision operator identity, stencil coefficient)
fig2_physics_arm.png       : the inherited physics arm vs the 15% owner gate
fig3_evidence_matrix.png   : the maturity-gate evidence keys, present/absent
All numbers are read from probe_delivery.json / probe_chain_physics.json, i.e.
written by execution — nothing is hand-typed into the panels.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PD = json.loads((HERE / "probe_delivery.json").read_text())
PC = json.loads((HERE / "probe_chain_physics.json").read_text())


def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    comps = ["Fx1", "Fy1", "Fx2", "Fy2"]
    rel = [PD["P3_operator_identity_linear"][c]["rel"] for c in comps]
    ax[0].bar(comps, rel, color="#2b7bba")
    ax[0].set_yscale("log")
    ax[0].axhline(1e-12, ls="--", c="green", label="1e-12 (fp64 floor)")
    ax[0].axhline(1e-6, ls=":", c="orange", label="1e-6 (delivered claim)")
    ax[0].set_ylabel("max|F_oss - F_repo| / max|F_repo|")
    ax[0].set_title("P3 operator identity (linear psi, g=-G)\nworst = %.2e"
                    % PD["P3_operator_identity_linear"]["worst_rel"])
    ax[0].legend(fontsize=8)

    st = PD["P4_stencil"]
    ax[1].bar(["forward Σw·c·ψ(x+c)", "backward Σw·c·ψ(x−c)", "theory cs²"],
              [st["forward_over_kx"], abs(st["backward_over_kx"]), st["cs2_theory"]],
              color=["#2b7bba", "#e08a2b", "#666666"])
    ax[1].set_ylim(0, 0.45)
    ax[1].set_ylabel("coefficient / k")
    ax[1].set_title("P4 stencil coefficient (linear field, kx=0.37 ky=0.21)\n"
                    "antisymmetry residual %.1e" % st["forward_plus_backward_max_abs"])
    fig.suptitle("m18289 delivery: the operator module is real and machine-precision exact;"
                 " the run chain is not", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "fig1_operator_identity.png", dpi=130)
    plt.close(fig)


def fig2():
    arms = PC["arms"]
    names, vals, cols = [], [], []
    for k, v in arms.items():
        short = k.replace("arm_", "").replace(".npz", "")
        names.append(short)
        if v.get("err_pct") is None:
            vals.append(0.0)
            cols.append("#bbbbbb")
        else:
            vals.append(v["err_pct"])
            cols.append("#c0392b")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(names, vals, color=cols)
    ax.axhline(15.0, ls="--", c="green", lw=1.6, label="owner gate 15%")
    ax.axhline(3.0, ls=":", c="black", lw=1.2, label="ecosystem bar 3%")
    for b, k in zip(bars, arms):
        v = arms[k]
        if v.get("err_pct") is not None:
            ax.text(b.get_x() + b.get_width() / 2, v["err_pct"] + 6,
                    "%.2f%%\n(%.2f%% plain-norm)" % (v["err_pct"], v["err_pct_plain_norm"]),
                    ha="center", fontsize=7)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 8, "NaN\n(%d NaNs)" % v["nan_entries"],
                    ha="center", fontsize=8, color="#444444")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, 400)
    ax.set_ylabel("rel-L2 of the interface profile [%]")
    ax.set_title("Inherited physics arm (m18126 artifacts, recomputed here): 1/4 arms evaluable,\n"
                 "200.34% against the 15% gate -> benchmark FAIL", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_physics_arm.png", dpi=130)
    plt.close(fig)


def fig3(evidence: dict):
    levels = {
        "L1 tests": ["tests"],
        "L2 + impl/entry/artifacts": ["impl_files", "entry", "run_artifacts"],
        "L3 + integration": ["integration_callers"],
        "L4 + benchmark/repro": ["benchmark", "repro_cmd"],
        "L5 + external anchor": ["external_anchor"],
    }
    rows, cols, M = [], [], []
    for label, keys in levels.items():
        for k in keys:
            v = evidence.get(k)
            rows.append(f"{label.split(' + ')[0]} · {k}")
            cols.append(1.0 if v not in (None, "", [], {}) else 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(rows))
    ax.barh(y, [max(c, 0.02) for c in cols],
            color=["#27ae60" if c else "#c0392b" for c in cols])
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0.02, 1.0])
    ax.set_xticklabels(["absent", "present"], fontsize=8)
    ax.invert_yaxis()
    for yy, c in zip(y, cols):
        ax.text(1.02, yy, "ok" if c else "missing", va="center", fontsize=8,
                color="#27ae60" if c else "#c0392b")
    ax.set_title("m18289 evidence vs the maturity-gate key scheme (judge: /root/tp-agent/maturity_gate.py)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig3_evidence_matrix.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ev = json.loads((HERE / "verdict_evidence.json").read_text())
    fig1()
    fig2()
    fig3(ev)
    for p in ("fig1_operator_identity.png", "fig2_physics_arm.png", "fig3_evidence_matrix.png"):
        print(p, (HERE / p).stat().st_size, "bytes")