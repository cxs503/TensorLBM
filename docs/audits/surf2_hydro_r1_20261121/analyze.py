#!/usr/bin/env python3
"""SURF2-HYDRO-R1: dual-arm (ctrl vs hydro) full-window analysis.

Reads runs/ctrl/result.json + runs/hydro/result.json (SDAA probe outputs),
applies the pre-registered verdict, and emits:
  - verdict.json   (all numbers, programmatic)
  - fig_depth.png  (depth history vs experiment, pc axis)
  - fig_residual.png (pointwise residual vs experiment)

Pre-registration (seed DE-X-SURF2-HYDRO.R1):
  judgment point = experiment 1.857434 D @ 10.394 ms (pc)
  ctrl must bit-reproduce 1.435220 D @ 11.0906 ms (abs)
  HIT      = hydro peak |err| < 12% AND survival > 10.4 ms
  FALSIFY  = deeper by <= 1 cell (0.125 D ~ 6.7% of exp peak) OR shallower OR early death
"""
import json
import math
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
ROOT = next(p for p in _here.parents if (p / "pyproject.toml").exists())
RUNS = ROOT / "runs"
OUT = RUNS / "SURF2-HYDRO-R1"

EXP_PEAK_D = 1.857434
EXP_PEAK_MS_PC = 10.394
CTRL_PEAK_D = 1.435220
CTRL_PEAK_MS_ABS = 11.0906
CELL_D = 1.0 / 8.0  # cells_per_D = 8
HIT_ERR = 0.12
SURVIVE_MS = 10.4


def load(arm):
    p = RUNS / arm / "result.json"
    j = json.loads(p.read_text())
    h = j["history"]
    rows = [r for r in h if r.get("depth_over_D") is not None]
    pk = max(rows, key=lambda r: r["depth_over_D"])
    return j, pk


def main():
    ctrl, ctrl_pk = load("ctrl")
    hydro, hydro_pk = load("hydro")

    def peak_info(pk, j):
        return {
            "depth_over_D": pk["depth_over_D"],
            "time_ms_abs": pk["time_ms"],
            "step": pk["step"],
        }

    ctrl_pk = peak_info(ctrl_pk, ctrl)
    hydro_pk = peak_info(hydro_pk, hydro)

    # contact offset (abs -> pc)
    def contact_ms(j):
        cs = j.get("contact_step")
        dt = j["scales"]["dt_s"]
        return (cs * 1000 * dt) if cs is not None else None

    ctrl_c = contact_ms(ctrl)
    hydro_c = contact_ms(hydro)

    ctrl_err = (ctrl_pk["depth_over_D"] - EXP_PEAK_D) / EXP_PEAK_D
    hydro_err = (hydro_pk["depth_over_D"] - EXP_PEAK_D) / EXP_PEAK_D
    delta_cells = (hydro_pk["depth_over_D"] - ctrl_pk["depth_over_D"]) / CELL_D

    ctrl_bit_ok = (
        abs(ctrl_pk["depth_over_D"] - CTRL_PEAK_D) < 5e-6
        and abs(ctrl_pk["time_ms_abs"] - CTRL_PEAK_MS_ABS) < 5e-4
    )

    # survival: run must have produced history beyond 10.4 ms abs
    hydro_last_ms = max(r["time_ms"] for r in hydro["history"])
    hydro_survive = hydro_last_ms > SURVIVE_MS

    hit = abs(hydro_err) < HIT_ERR and hydro_survive
    deeper = hydro_pk["depth_over_D"] > ctrl_pk["depth_over_D"]
    quant_only = deeper and (hydro_pk["depth_over_D"] - ctrl_pk["depth_over_D"]) <= CELL_D
    shallower = not deeper
    early_death = not hydro_survive

    if not ctrl_bit_ok:
        verdict = "INVALID_CTRL"
    elif hit:
        verdict = "HIT"
    elif quant_only:
        verdict = "FALSIFIED_QUANTIZATION"
    elif shallower:
        verdict = "FALSIFIED_SHALLOWER"
    elif early_death:
        verdict = "FALSIFIED_EARLY_DEATH"
    else:
        verdict = "FALSIFIED_OTHER"

    out = {
        "case": "santini_case3 (Santini et al. 2015, case 3)",
        "config": {
            "cells_per_D": 8, "gap_over_D": 0.18, "speed": 0.06,
            "release": 1.53, "pool_depth_over_D": 3.0, "u_limit": 0.15,
            "steps_limit": 10000, "z_ref": 24.5,
        },
        "experiment": {"peak_D": EXP_PEAK_D, "peak_ms_pc": EXP_PEAK_MS_PC},
        "ctrl": {**ctrl_pk, "expected_D": CTRL_PEAK_D, "expected_ms_abs": CTRL_PEAK_MS_ABS,
                  "bit_reproduced": bool(ctrl_bit_ok),
                  "err_vs_exp": ctrl_err,
                  "stop_reason": ctrl.get("stop_reason"),
                  "contact_step": ctrl.get("contact_step"),
                  "contact_ms_abs": ctrl_c},
        "hydro": {**hydro_pk, "err_vs_exp": hydro_err,
                   "delta_vs_ctrl_D": hydro_pk["depth_over_D"] - ctrl_pk["depth_over_D"],
                   "delta_vs_ctrl_cells": delta_cells,
                   "stop_reason": hydro.get("stop_reason"),
                   "last_time_ms_abs": hydro_last_ms,
                   "contact_step": hydro.get("contact_step"),
                   "contact_ms_abs": hydro_c,
                   "survived_beyond_10p4ms": bool(hydro_survive)},
        "verdict": verdict,
        "rules": {
            "hit": "hydro |err|<12% AND survival>10.4ms",
            "falsify": "deeper<=1cell(0.125D~6.7%) OR shallower OR early death",
        },
    }
    (OUT / "verdict.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = np.genfromtxt(ROOT / "benchmarks/water_entry/single_drop/fig5_experiment.csv",
                        delimiter=",", names=True)
    t_exp = exp["time_ms"]
    d_exp = exp["depth_over_D"]

    def hist(j, c_off):
        h = [r for r in j["history"] if r.get("depth_over_D") is not None]
        t = np.array([r["time_ms"] - c_off for r in h])
        d = np.array([r["depth_over_D"] for r in h])
        return t, d

    tc, dc = hist(ctrl, ctrl_c or 0.0)
    th, dh = hist(hydro, hydro_c or 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(t_exp, d_exp, "k^--", ms=5, lw=1.2, label="experiment (Santini case-3)")
    ax.plot(tc, dc, "b-", lw=1.4, label="ctrl (no hydro comp.)")
    ax.plot(th, dh, "r-", lw=1.4, label="hydro (TL_FS_HYDRO_Z_REF=24.5)")
    ax.axhline(EXP_PEAK_D, color="k", ls=":", lw=0.8)
    ax.axvline(EXP_PEAK_MS_PC, color="k", ls=":", lw=0.8)
    ax.annotate(f"exp peak {EXP_PEAK_D}D@{EXP_PEAK_MS_PC}ms",
                (EXP_PEAK_MS_PC, EXP_PEAK_D), textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.annotate(f"ctrl {ctrl_pk['depth_over_D']:.4f}D@{ctrl_pk['time_ms_abs'] - (ctrl_c or 0):.3f}ms",
                (ctrl_pk["time_ms_abs"] - (ctrl_c or 0.0), ctrl_pk["depth_over_D"]),
                textcoords="offset points", xytext=(-90, -14), fontsize=8, color="b")
    ax.annotate(f"hydro {hydro_pk['depth_over_D']:.4f}D@{hydro_pk['time_ms_abs'] - (hydro_c or 0.0):.3f}ms",
                (hydro_pk["time_ms_abs"] - (hydro_c or 0.0), hydro_pk["depth_over_D"]),
                textcoords="offset points", xytext=(8, -14), fontsize=8, color="r")
    ax.set_xlabel("time post-contact (ms)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("SURF2-HYDRO-R1: cavity depth, full window (cells=8, gap=0.18, U=0.06)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_depth.png")
    plt.close(fig)

    # residual at experiment sample times (interpolate sim on pc axis)
    def resid(t, d, t_e, d_e):
        di = np.interp(t_e, t, d)
        return (di - d_e) / d_e * 100.0

    rc = resid(tc, dc, t_exp, d_exp)
    rh = resid(th, dh, t_exp, d_exp)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    w = 0.38
    ax.bar(t_exp - w / 2, rc, width=w, color="b", alpha=0.8, label="ctrl residual")
    ax.bar(t_exp + w / 2, rh, width=w, color="r", alpha=0.8, label="hydro residual")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-12, color="g", ls="--", lw=0.8)
    ax.axhline(12, color="g", ls="--", lw=0.8)
    ax.set_xlabel("time post-contact (ms)")
    ax.set_ylabel("residual vs experiment (%)")
    ax.set_title("SURF2-HYDRO-R1: pointwise residual vs experiment (green = ±12% hit band)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_residual.png")
    plt.close(fig)
    print("figures ->", OUT / "fig_depth.png", OUT / "fig_residual.png")


if __name__ == "__main__":
    main()
