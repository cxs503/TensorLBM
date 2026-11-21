#!/usr/bin/env python3
"""m21631 board-audit figures for m21548 (CASE-BACKSTEP-FVM-RE100).

All panels are built from the auditor's own reproductions stored under
``audit/m21631/audit_repro/`` — NOT from mission m21548 deliverables, which
carry no run artefact at all (``artifacts/case_backward_step_armaly/`` is empty).

  fig1_ux_field_audit_repro.png   mean streamwise velocity + reattachment marker
                                  (h1_re100, 6000 steps, ER=51/26, explicitly NOT converged)
  fig2_x1s_vs_steps.png           X1/s trace vs the two registered references and
                                  their +-10 % bands: shows why a 170-200 k step budget
                                  is required and why 6 k steps cannot be an acceptance point
  fig3_evidence_matrix.png        the gate's evidence keys x level requirements, filled
                                  only where the auditor found a real pointer

Run:  python3 audit/m21631/make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPRO = HERE / "audit_repro"
OUT = HERE / "figs"

REF_EXP = 3.05      # Armaly 1983 experimental digitisation, in-repo refs.json
REF_NUM = 2.878     # Erturk 2008 2-D numerical benchmark, in-repo refs.json
BAND = 0.10         # ledger acceptance for this anchor


def fig1() -> Path:
    z = np.load(REPRO / "field.npz")
    ux = z["ux_mean"]
    solid = z["solid"].astype(bool)
    x_step = float(z["x_step"])
    s = float(z["s"])
    res = json.loads((REPRO / "result.json").read_text())
    x1 = float(res["result"]["x1_over_s"])
    x_re = x_step + x1 * s

    fig, ax = plt.subplots(figsize=(9.0, 2.9), dpi=150)
    masked = np.ma.masked_where(solid, ux)
    lim = float(np.percentile(np.abs(ux[~solid]), 99)) or 1.0
    pc = ax.pcolormesh(masked[::-1], shading="auto", cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [lattice]")
    ax.set_ylabel("y [lattice]")
    ax.axvline(x_step, color="k", lw=0.8, ls="--")
    ax.axvline(x_re, color="g", lw=1.2)
    ax.annotate(f"reattachment x/s = {x1:.3f}\n(h1_re100, 6000 steps, NOT converged)",
                xy=(x_re, ux.shape[0] * 0.25), xytext=(x_re - 300, ux.shape[0] * 0.55),
                fontsize=7, color="g",
                arrowprops=dict(arrowstyle="->", color="g", lw=0.8))
    ax.set_title("audit reproduction (m21631, not a mission artefact): "
                 "mean $u_x$, Re=100, ER=51/26=1.9615", fontsize=8)
    fig.colorbar(pc, ax=ax, label="$u_x$ [lattice]")
    fig.tight_layout()
    p = OUT / "fig1_ux_field_audit_repro.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2() -> Path:
    smoke = json.loads((REPRO / "smoke_result.json").read_text())
    mid = json.loads((REPRO / "result.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)

    ax.axhspan(REF_EXP * (1 - BAND), REF_EXP * (1 + BAND), color="C0", alpha=0.12,
               label="Armaly 1983 exp 3.05 ±10 % (ledger gate)")
    ax.axhline(REF_EXP, color="C0", lw=1.0)
    ax.axhspan(REF_NUM * (1 - BAND), REF_NUM * (1 + BAND), color="C3", alpha=0.12,
               label="Erturk 2008 2-D 2.878 ±10 % (ledger gate)")
    ax.axhline(REF_NUM, color="C3", lw=1.0)

    series = [("g1_re100 smoke 3000 steps ($h_{in}$=8, ER=1.875)", smoke, "C7", "o")]
    s_sm = [(h["step"], h["x1_over_s_instant"]) for h in smoke["convergence_history"]]
    ax.plot(*zip(*s_sm), color="C7", marker="o", ms=3, lw=1.0, label=series[0][0])

    s_mid = [(h["step"], h["x1_over_s_instant"]) for h in mid["convergence_history"]]
    s_mid.append((mid["result"]["steps_run"], mid["result"]["x1_over_s"]))
    ax.plot(*zip(*s_mid), color="C1", marker="s", ms=4, lw=1.2,
            label="$h_{in}$=26 (ER=1.9615) 6000 steps (audit repro)")

    ax.set_xlabel("step")
    ax.set_ylabel("$X_1/s$ (instantaneous, shear zero-crossing)")
    ax.set_title("audit m21631: no series reaches a steady point inside 6 k steps\n"
                 "(mission m21548 has zero run artefacts, so no acceptance point exists)",
                 fontsize=8)
    ax.set_ylim(1.4, 3.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    p = OUT / "fig2_x1s_vs_steps.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig3() -> Path:
    keys = ["impl_files", "entry", "run_artifacts", "integration_callers",
            "benchmark", "repro_cmd", "external_anchor", "tests"]
    need = {"L1": ["tests"], "L2": ["impl_files", "entry", "run_artifacts"],
            "L3": ["impl_files", "entry", "run_artifacts", "integration_callers"],
            "L4": ["impl_files", "entry", "run_artifacts", "integration_callers",
                   "benchmark", "repro_cmd"],
            "L5": keys}
    # mission-side state (m21548), from the audit: 1 = real pointer found, 0 = absent
    mission = {"impl_files": 1, "entry": 1, "run_artifacts": 0, "integration_callers": 0,
               "benchmark": 0, "repro_cmd": 1, "external_anchor": 1, "tests": 0}
    mat = np.zeros((len(need), len(keys)))
    for i, lv in enumerate(need):
        for j, k in enumerate(keys):
            if k in need[lv]:
                mat[i, j] = mission[k]          # 1 satisfied, 0 required-but-missing
            else:
                mat[i, j] = np.nan              # not required at this level

    fig, ax = plt.subplots(figsize=(7.0, 2.8), dpi=150)
    cmap = matplotlib.colors.ListedColormap(["#d9534f", "#5cb85c"])
    ax.imshow(np.where(np.isnan(mat), -1, mat), cmap=cmap, vmin=-1, vmax=1,
              aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(need)))
    ax.set_yticklabels(list(need), fontsize=8)
    for i in range(len(need)):
        for j in range(len(keys)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "·", ha="center", va="center", fontsize=8, color="0.6")
    ax.set_title("m21548 evidence keys vs the maturity-gate scheme "
                 "(green = real pointer, red = required and missing)", fontsize=8)
    fig.tight_layout()
    p = OUT / "fig3_evidence_matrix.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3):
        print(f())


if __name__ == "__main__":
    main()