#!/usr/bin/env python3
"""Assemble the XVAL-vs-vivsim artifacts: comparison table + figures.

Reads the shard JSONs / series produced by
``benchmarks/xval/xval_vivsim_cylinder_re200.py`` and writes

  * ``docs/figures/xval_vivsim_re200_cd_cl.png``    -- Cd/Cl history vs the
    OSS DOI references (with the +-15% band) -- the "comparison curve";
  * ``docs/figures/xval_vivsim_re200_errors.png``   -- error bars per caliber
    and per run, with the owner 15% tolerance line;
  * ``docs/figures/xval_vivsim_re200_grid.png``     -- grid convergence;
  * ``artifacts/xval_vivsim_re200/summary.json``    -- programmatic table.

Everything numeric is read from the shard outputs; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CD_REF = 1.389
CL_REF = 0.718
ST_LIT = 0.197
TOL = 15.0


def load(paths):
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        d["_json"] = str(p)
        series = Path(p).with_suffix(".npz")
        d["_series"] = str(series) if series.exists() else None
        runs.append(d)
    runs.sort(key=lambda r: (r["setup"]["D"], r["setup"]["u_in"]))
    return runs


def fig_history(runs, out: Path):
    cand = [r for r in runs if r["_series"]]
    if not cand:
        return None
    r = max(cand, key=lambda x: x["setup"]["D"])
    z = np.load(r["_series"])
    t, cd, cl = z["t"], z["cd"], z["cl"]
    u, D = r["setup"]["u_in"], r["setup"]["D"]
    tf = t * u / D  # convective times
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    ax[0].plot(tf, cd, lw=0.9, color="tab:blue", label="TensorLBM $C_D$")
    ax[0].axhline(CD_REF, color="k", ls="--", lw=1.2, label=f"vivsim ref {CD_REF}")
    ax[0].fill_between([tf[0], tf[-1]], CD_REF * (1 - TOL / 100), CD_REF * (1 + TOL / 100),
                       color="k", alpha=0.08, label=f"±{TOL:.0f}% band")
    ax[0].axvline(0.5 * (tf[0] + tf[-1]), color="gray", ls=":", lw=1,
                  label="OSS averaging window start")
    ax[0].set_ylabel("$C_D$")
    ax[0].set_xlabel("convective time  $tU_0/D$")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title(f"$C_D$  (D={D}, u={u})", fontsize=10)

    ax[1].plot(tf, cl, lw=0.9, color="tab:red", label="TensorLBM $C_L$")
    for s in (+1, -1):
        ax[1].axhline(s * CL_REF, color="k", ls="--", lw=1.2)
    ax[1].axhline(0.0, color="gray", lw=0.5)
    ax[1].fill_between([tf[0], tf[-1]], CL_REF * (1 - TOL / 100), CL_REF * (1 + TOL / 100),
                       color="k", alpha=0.08)
    ax[1].fill_between([tf[0], tf[-1]], -CL_REF * (1 + TOL / 100), -CL_REF * (1 - TOL / 100),
                       color="k", alpha=0.08, label="±15% band on $|C_L|$ peak")
    ax[1].axvline(0.5 * (tf[0] + tf[-1]), color="gray", ls=":", lw=1)
    ax[1].set_ylabel("$C_L$")
    ax[1].set_xlabel("convective time  $tU_0/D$")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title(f"$C_L$  (ref peak amplitude {CL_REF})", fontsize=10)
    fig.suptitle("XVAL vs haimingz/vivsim — free-stream cylinder Re=200 (DOI-anchored Cd/Cl)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_errors(runs, out: Path):
    labels, e_cd, e_cl, e_clh = [], [], [], []
    for r in runs:
        labels.append(f"D{r['setup']['D']} u{r['setup']['u_in']:g}")
        v = r["verdict"]
        e_cd.append(v["err_pct_cd_mean"])
        e_cl.append(v["err_pct_cl_peak"])
        e_clh.append(v["err_pct_cl_amp_harmonic_vs_CL_REF"])
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(labels)), 4.0))
    ax.bar(x - w, e_cd, w, label="$C_{D,mean}$ vs 1.389 (OSS caliber)", color="tab:blue")
    ax.bar(x, e_cl, w, label="$|C_L|_{max}$ vs 0.718 (OSS caliber)", color="tab:red")
    ax.bar(x + w, e_clh, w, label="$C_{L,amp}$ vs 0.718 (harmonic caliber)",
           color="tab:orange")
    for s in (+TOL, -TOL):
        ax.axhline(s, color="k", ls="--", lw=1.1)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("error vs OSS DOI reference  [%]")
    ax.set_title(f"XVAL error by caliber (dashed = ±{TOL:.0f}% owner tolerance)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_grid(runs, out: Path):
    runs = [r for r in runs if r["runtime"]["nan_at"] is None]
    if len(runs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for u in sorted({r["setup"]["u_in"] for r in runs}):
        rr = sorted([r for r in runs if r["setup"]["u_in"] == u], key=lambda x: x["setup"]["D"])
        if len(rr) < 1:
            continue
        h = [1.0 / r["setup"]["D"] for r in rr]
        ax.plot(h, [r["vivsim"]["cd_mean_last_half"] for r in rr], "o-",
                label=f"$C_D$ u={u:g}")
        ax.plot(h, [r["ours"]["cl_amp_harmonic"] for r in rr], "s--",
                label=f"$C_{{L,amp}}$ u={u:g}")
    ax.axhline(CD_REF, color="tab:blue", ls=":", lw=1)
    ax.axhline(CL_REF, color="tab:orange", ls=":", lw=1)
    ax.set_xlabel("$1/D$  (lattice diameters per cylinder)")
    ax.set_ylabel("coefficient")
    ax.set_xscale("log")
    ax.set_title("grid ordering toward the OSS reference values", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--figdir", default="docs/figures")
    ap.add_argument("--out", default="artifacts/xval_vivsim_re200/summary.json")
    a = ap.parse_args()
    runs = load(a.jsons)
    figdir = Path(a.figdir)
    figs = {
        "history": fig_history(runs, figdir / "xval_vivsim_re200_cd_cl.png"),
        "errors": fig_errors(runs, figdir / "xval_vivsim_re200_errors.png"),
        "grid": fig_grid(runs, figdir / "xval_vivsim_re200_grid.png"),
    }
    rows = []
    for r in runs:
        rows.append({
            "D": r["setup"]["D"],
            "u_in": r["setup"]["u_in"],
            "re": r["setup"]["re"],
            "tau": round(r["setup"]["tau"], 5),
            "grid": [r["setup"]["nx"], r["setup"]["ny"]],
            "steps": r["runtime"]["steps_done"],
            "conv_lengths": round(r["setup"]["conv_lengths"], 1),
            "mlups": round(r["runtime"]["mlups"], 2),
            "cd_mean_oss_caliber": round(r["vivsim"]["cd_mean_last_half"], 4),
            "cl_peak_oss_caliber": round(r["vivsim"]["cl_peak_last_half"], 4),
            "cl_amp_harmonic": round(r["ours"]["cl_amp_harmonic"], 4),
            "cl_amp_period_median": round(r["ours"]["cl_amp_period_median"], 4),
            "st": round(r["ours"]["st"], 4),
            "err_cd_pct": round(r["verdict"]["err_pct_cd_mean"], 2),
            "err_cl_peak_pct": round(r["verdict"]["err_pct_cl_peak"], 2),
            "err_cl_harm_pct": round(r["verdict"]["err_pct_cl_amp_harmonic_vs_CL_REF"], 2),
            "err_st_pct": round(r["verdict"]["st_vs_lit_err_pct"], 2),
            "pass": r["verdict"]["pass"],
            "nan_at": r["runtime"]["nan_at"],
        })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "anchor": "XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS",
        "oss": "haimingz/vivsim @ fb84b12 examples/2d/flow_pass_cylinder.py:78-85",
        "refs": {"cd": CD_REF, "cl": CL_REF, "st_lit_diagnostic_only": ST_LIT},
        "tolerance_pct": TOL,
        "runs": rows,
        "figures": {k: str(v) for k, v in figs.items() if v},
    }, indent=2))
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())