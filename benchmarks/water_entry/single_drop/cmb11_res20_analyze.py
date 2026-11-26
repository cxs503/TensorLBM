"""DE-X-CMB11-RES20: programmatic extraction + figures for the cells=20 two-arm run.

All numbers are extracted from runs/*/result.json (zero hand transcription).
Usage: python3 cmb11_res20_analyze.py --ctrl runs/cmb11_r20_ctrl/result.json \
    --tuned runs/cmb11_r20_tuned/result.json --out-dir runs/cmb11_r20_ctrl
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_CSV = Path(__file__).with_name("fig5_experiment.csv")
EXP_DEPTH = 1.857434  # D, Santini 2021 (fluids7020050)
EXP_TIME = 10.39446  # ms post-contact
DUAL_DEPTH_GATE = 1.8017  # D
DUAL_TIME_GATE = 11.4334  # ms post-contact

# prior anchors (worktrees of predecessor missions; result.json on disk)
# prior anchors: cluster-fetch artifacts of predecessor missions (m27225/m27256
# worktrees retired), copied verbatim into the worktree for self-contained provenance
_PRIOR_DIR = Path(__file__).resolve().parents[3] / "runs" / "cmb11_prior"
PRIOR = {
    "C8-ctrl": str(_PRIOR_DIR / "C8_ctrl_result.json"),
    "C8-tuned": str(_PRIOR_DIR / "C8_tuned_result.json"),
    "C12-ctrl": str(_PRIOR_DIR / "C12_ctrl_result.json"),
    "C12-tuned": str(_PRIOR_DIR / "C12_tuned_result.json"),
    "C16-ctrl": str(_PRIOR_DIR / "C16_ctrl_result.json"),
    "C16-tuned": str(_PRIOR_DIR / "C16_tuned_result.json"),
}


def load_arm(path):
    d = json.loads(Path(path).read_text())
    h = [e for e in d["history"] if isinstance(e, dict) and "depth_over_D" in e]
    i = max(range(len(h)), key=lambda k: h[k]["depth_over_D"])
    dt_ms = d["scales"]["dt_s"] * 1000.0
    contact_ms = d["contact_step"] * dt_ms
    last = d["history"][-1]
    return {
        "cells": d["cells_per_D"],
        "stop": d["stop_reason"],
        "peak_depth_D": h[i]["depth_over_D"],
        "peak_time_ms_pc": h[i]["time_ms"] - contact_ms,
        "peak_step": h[i]["step"],
        "hist_steps": len(h),
        "contact_ms": contact_ms,
        "end_drift": last.get("volume_drift", float("nan")),
        "mach_max": max(e["mach_max"] for e in h),
        "collision_mode": d.get("collision_mode"),
        "rr_tau": d.get("rr_tau"),
        "C_s": d.get("C_s"),
        "history": h,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arms = {
        "R20-ctrl": load_arm(args.ctrl),
        "R20-tuned": load_arm(args.tuned),
    }
    for name, p in PRIOR.items():
        if Path(p).exists():
            arms[name] = load_arm(p)

    # ---- summary table ----
    rows = []
    for name in [
        "C8-ctrl",
        "C12-ctrl",
        "C16-ctrl",
        "R20-ctrl",
        "C8-tuned",
        "C12-tuned",
        "C16-tuned",
        "R20-tuned",
    ]:
        if name not in arms:
            continue
        a = arms[name]
        rows.append(
            {
                "arm": name,
                "cells": a["cells"],
                "stop": a["stop"],
                "peak_depth_D": round(a["peak_depth_D"], 6),
                "depth_err_pct": round((a["peak_depth_D"] / EXP_DEPTH - 1) * 100, 2),
                "peak_time_ms_pc": round(a["peak_time_ms_pc"], 4),
                "time_err_pct": round((a["peak_time_ms_pc"] / EXP_TIME - 1) * 100, 2),
                "end_drift": a["end_drift"],
                "mach_max": round(a["mach_max"], 4),
                "dual_pass": a["peak_depth_D"] > DUAL_DEPTH_GATE
                and a["peak_time_ms_pc"] < DUAL_TIME_GATE,
            }
        )
    c8 = arms["C8-ctrl"]["peak_depth_D"]
    c12 = arms["C12-ctrl"]["peak_depth_D"]
    c16 = arms["C16-ctrl"]["peak_depth_D"]
    c20 = arms["R20-ctrl"]["peak_depth_D"]
    summary = {
        "experiment": {"depth_D": EXP_DEPTH, "time_ms_pc": EXP_TIME},
        "dual_gates": {"depth_gt": DUAL_DEPTH_GATE, "time_lt": DUAL_TIME_GATE},
        "arms": rows,
        "extrapolation_gate": {
            "ctrl_C8": c8,
            "ctrl_C12": c12,
            "ctrl_C16": c16,
            "ctrl_C20": c20,
            "inc_C8_C12": round(c12 - c8, 6),
            "inc_C12_C16": round(c16 - c12, 6),
            "inc_C16_C20": round(c20 - c16, 6),
            "capped": (c20 - c16) <= 0.08,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- fig 1: depth-time vs experiment ----
    exp = list(csv.DictReader(EXP_CSV.open()))
    ex_t = [float(r["time_ms"]) for r in exp]
    ex_d = [float(r["depth_over_D"]) for r in exp]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(ex_t, ex_d, "k--o", ms=4, lw=1.2, label="experiment (Santini 2021)")
    for name, col, mk in [("R20-ctrl", "tab:blue", "s"), ("R20-tuned", "tab:red", "^")]:
        a = arms[name]
        t = [e["time_ms"] - a["contact_ms"] for e in a["history"]]
        ax.plot(
            t,
            [e["depth_over_D"] for e in a["history"]],
            color=col,
            marker=mk,
            ms=3,
            lw=1.0,
            label=f"{name} ({a['stop']})",
        )
        ax.annotate(
            f"{a['peak_depth_D']:.4f}D\n@{a['peak_time_ms_pc']:.3f}ms",
            (a["peak_time_ms_pc"], a["peak_depth_D"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=col,
        )
    ax.axhline(EXP_DEPTH, color="k", ls=":", lw=0.8)
    ax.axvline(EXP_TIME, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time (ms post-contact)")
    ax.set_ylabel("open-water depth (D)")
    ax.set_title("CMB11-RES20: crater depth vs time, cells=20 two arms vs experiment")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_depth_time.png")
    plt.close(fig)

    # ---- fig 2: four-point resolution trend ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150)
    cells = [8, 12, 16, 20]
    for ax, tag, col, ttl in [
        (ax1, "ctrl", "tab:blue", "BGK (ctrl) peak depth"),
        (ax2, "tuned", "tab:red", "rr_iface+SMAG.075+HYDRO24.5 (tuned) peak depth"),
    ]:
        vals = [arms[f"C{c}-{tag}"]["peak_depth_D"] for c in cells[:3]]
        vals.append(arms[f"R20-{tag}"]["peak_depth_D"])
        ax.plot(cells, vals, "-o", color=col, lw=1.4, label=f"{tag} arm")
        for c, v in zip(cells, vals):
            ax.annotate(
                f"{v:.4f}",
                (c, v),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )
        ax.axhline(EXP_DEPTH, color="k", ls=":", lw=0.9)
        ax.annotate(
            "experiment", (20, EXP_DEPTH), xytext=(-30, 6), textcoords="offset points", fontsize=7.5
        )
        ax.set_xlabel("cells per D")
        ax.set_ylabel("peak depth (D)")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks(cells)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_resolution_trend.png")
    plt.close(fig)
    print("figures:", out / "fig1_depth_time.png", out / "fig2_resolution_trend.png")


if __name__ == "__main__":
    main()
