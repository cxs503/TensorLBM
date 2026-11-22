#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-COL2-REG: depth-vs-time comparison figure (matplotlib, PNG out).

Panel (a): penetration depth d/D vs time (post-contact) for every arm, with
the archived ctrl anchor and the Santini case-3 experimental peak marked.
Panel (b): survival (steps completed / stop reason) vs relaxation time tau,
bracketing the RR stability window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_PEAK_D = 1.857434
EXP_PEAK_MS = 10.39446
CTRL_PEAK_D = 1.4352204748231814
CTRL_PEAK_MS_PC = 10.543339285714286

ARM_STYLE = {
    "rr_base": ("tab:red", "-", "RR @ tau=0.5004 (baseline)"),
    "ctrl_bgk": ("tab:blue", "-", "BGK ctrl @ tau=0.5004"),
    "rr_t49": ("tab:orange", "--", "RR @ tau=0.49"),
    "rr_t48": ("tab:purple", "--", "RR @ tau=0.48"),
    "rr_t45": ("tab:brown", "--", "RR @ tau=0.45"),
    "rr_t42": ("tab:pink", "--", "RR @ tau=0.42"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--arms", default="rr_base,rr_t49,rr_t48,rr_t45,rr_t42,ctrl_bgk")
    ap.add_argument("--out", default="artifacts/COL2-REG/col2reg_depth.png")
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    plotted = 0
    for arm in [s for s in a.arms.split(",") if s]:
        p = runs / arm / "result.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        hist = res.get("history") or []
        contact = res.get("contact_step")
        dt_ms = 1000 * res["scales"]["dt_s"]
        if contact is None:
            continue  # died before contact: no post-contact axis
        ts = [r["time_ms"] - contact * dt_ms for r in hist]
        ds = [r.get("depth_over_D") for r in hist]
        color, ls, label = ARM_STYLE.get(arm, ("gray", ":", arm))
        ax.plot(ts, ds, color=color, ls=ls, lw=1.6, label=label)
        plotted += 1
        # panel b: survival
        stop = res.get("stop_reason")
        tau = res.get("rr_tau")
        tau = 0.5003877 if tau is None else tau
        ok = stop == "window_complete"
        ax2.scatter([tau], [len(hist)], marker="o" if ok else "x",
                    s=70, color="tab:green" if ok else "tab:red", zorder=3)
        ax2.annotate(f"{arm}\n{stop}", (tau, len(hist)),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.plot([EXP_PEAK_MS], [EXP_PEAK_D], "k*", ms=14,
            label=f"exp {EXP_PEAK_D}D@{EXP_PEAK_MS}ms")
    ax.plot([CTRL_PEAK_MS_PC], [CTRL_PEAK_D], "ks", ms=8,
            label=f"ctrl anchor {CTRL_PEAK_D:.4f}D@{CTRL_PEAK_MS_PC:.4f}ms")
    ax.set_xlabel("time after contact (ms)")
    ax.set_ylabel("penetration depth d/D")
    ax.set_title("(a) depth-peak vs ctrl (same 24 ms window)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    ax2.set_xlabel("relaxation time tau")
    ax2.set_ylabel("steps completed")
    ax2.set_title("(b) RR stability window (x = health_limit)")
    ax2.set_ylim(0, 2100)
    ax2.axhline(1974, color="gray", ls=":", lw=1)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    a.out = Path(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"figure -> {a.out} ({plotted} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
