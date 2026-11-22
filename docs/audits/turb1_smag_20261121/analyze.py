#!/usr/bin/env python3
"""DE-X-TURB1-SMAG: programmatic verdict + figures (zero hand-copy).

Reads runs/{smag_hydro,smag01,smag02}/result.json, applies the pre-registered
verdict (see DE-X-TURB1-SMAG.md), writes verdict.json + figs.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # worktree root
OUT = Path(__file__).resolve().parent
# full-window results for the two surviving arms; smag02's coarse run IS its
# terminal state (deterministic death at step 792 < coarse cap 1200).
ARMS = ("smag_hydro", "smag01", "smag02")
ARM_FILES = {
    "smag_hydro": "runs/smag_hydro_full/result.json",
    "smag01": "runs/smag01_full/result.json",
    "smag02": "runs/smag02/result.json",
}
EXP_CSV = ROOT / "benchmarks/water_entry/single_drop/fig5_experiment.csv"
VERDICT_DEPTH_MS = 10.39446  # ms pc, experiment peak (17-digit source: 10.39446)
EXP_PEAK_D = 1.857434
ONE_CELL_D = 0.125           # 1 cell = 0.125 D
PEAK_TIME_GATE_MS = 1.0      # pre-registered peak-time gate


def load_arm(name):
    r = json.loads((ROOT / ARM_FILES[name]).read_text())
    h = r["history"]
    dt = h[1]["time_ms"] - h[0]["time_ms"]
    c = r["contact_step"]
    rows = []
    for row in h:
        if "depth_over_D" not in row:
            continue  # the death row (health_limit break) has no depth
        pc = row["time_ms"] - c * dt
        rows.append((pc, row["depth_over_D"], row["time_ms"]))
    return r, rows, dt, c


def interp(rows, t_pc):
    ts = np.array([x[0] for x in rows])
    ds = np.array([x[1] for x in rows])
    if t_pc < ts[0] or t_pc > ts[-1]:
        return None
    return float(np.interp(t_pc, ts, ds))


def main():
    exp = np.genfromtxt(EXP_CSV, delimiter=",", names=True)
    exp_t = exp["time_ms"]
    exp_d = exp["depth_over_D"]

    data = {}
    for name in ARMS:
        r, rows, dt, c = load_arm(name)
        pk = max(rows, key=lambda x: x[1])
        data[name] = {
            "stop_reason": r["stop_reason"],
            "C_s": r.get("C_s"),
            "contact_step": c,
            "dt_ms": dt,
            "peak_depth_D": pk[1],
            "peak_time_ms_abs": pk[2],
            "peak_time_ms_pc": pk[0],
            "last_time_ms_abs": rows[-1][2],
            "last_time_ms_pc": rows[-1][0],
            "depth_at_verdict_pc": interp(rows, VERDICT_DEPTH_MS),
            "rows": rows,
        }
        # late-window (7-13 ms pc) mean residual vs experiment
        sel = [(t, d) for t, d, _ in rows if 7.0 <= t <= 13.0]
        if sel:
            resid = []
            for t, d in sel:
                de = float(np.interp(t, exp_t, exp_d))
                resid.append(d / de - 1.0)
            data[name]["late_window_mean_resid"] = float(np.mean(resid))
        # post-impact decay: depth at fixed pc times after the peak
        for t in (15.0, 18.0, 21.0):
            data[name][f"depth_pc_{t:g}"] = interp(rows, t)

    hydro = data["smag_hydro"]
    verdict = {}
    hits = []
    for name in ("smag01", "smag02"):
        dd = abs(data[name]["peak_depth_D"] - hydro["peak_depth_D"])
        dt_ = abs(data[name]["peak_time_ms_pc"] - hydro["peak_time_ms_pc"])
        hit = dd > ONE_CELL_D or dt_ > PEAK_TIME_GATE_MS
        verdict[name] = {
            "delta_peak_depth_D": data[name]["peak_depth_D"] - hydro["peak_depth_D"],
            "delta_peak_time_ms_pc": data[name]["peak_time_ms_pc"] - hydro["peak_time_ms_pc"],
            "one_cell_D": ONE_CELL_D,
            "hit": bool(hit),
        }
        if hit:
            hits.append(name)
    verdict["outcome"] = "HIT" if hits else "FALSIFY"
    verdict["hit_arms"] = hits
    verdict["verdict_point"] = {"depth_D": EXP_PEAK_D, "time_ms_pc": VERDICT_DEPTH_MS}
    verdict["baseline_hydro"] = {
        "peak_depth_D": hydro["peak_depth_D"],
        "peak_time_ms_pc": hydro["peak_time_ms_pc"],
        "stop_reason": hydro["stop_reason"],
    }
    verdict["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                       for k, v in data.items()}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")

    # ---- figures ----------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9})
    colors = {"smag_hydro": "tab:blue", "smag01": "tab:orange", "smag02": "tab:red"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ARMS:
        rows = data[name]["rows"]
        ax.plot([x[0] for x in rows], [x[1] for x in rows],
                color=colors[name], lw=1.4,
                label=f"{name} (C_s={data[name]['C_s']})")
    ax.plot(exp_t, exp_d, "k^", ms=5, mfc="none", label="experiment (Santini case-3)")
    ax.axvline(VERDICT_DEPTH_MS, color="gray", ls=":", lw=1)
    ax.axhline(EXP_PEAK_D, color="gray", ls=":", lw=1)
    ax.set_xlabel("time after contact (ms, pc)")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("DE-X-TURB1-SMAG: cavity depth vs Smagorinsky C_s (HYDRO baseline)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_depth.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ARMS:
        rows = data[name]["rows"]
        pk = data[name]["peak_time_ms_pc"]
        ax.plot([x[0] for x in rows if x[0] >= pk], [x[1] for x in rows if x[0] >= pk],
                color=colors[name], lw=1.4, label=f"{name}")
    ax.plot(exp_t[exp_t >= 10.39446], exp_d[exp_t >= 10.39446], "k^", ms=5,
             mfc="none", label="experiment")
    ax.set_xlabel("time after contact (ms, pc), post-peak")
    ax.set_ylabel("cavity depth (D)")
    ax.set_title("Post-impact decay (after each arm's own peak)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_decay.png")
    plt.close(fig)

    # late-window residual vs experiment (7-13 ms pc), like SURF2-HYDRO-R1
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for name in ARMS:
        rows = data[name]["rows"]
        sel = [(t, d) for t, d, _ in rows if 7.0 <= t <= 13.0]
        if not sel:
            continue
        ts = [t for t, _ in sel]
        rs = [(d / float(np.interp(t, exp_t, exp_d)) - 1.0) * 100.0 for t, d in sel]
        ax.plot(ts, rs, color=colors[name], lw=1.4, label=f"{name}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("time after contact (ms, pc)")
    ax.set_ylabel("depth residual vs experiment (%)")
    ax.set_title("Late-window (7-13 ms pc) depth residual vs experiment")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_residual.png")
    plt.close(fig)

    # ---- console summary ---------------------------------------------------
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in data.items()}, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
