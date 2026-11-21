#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TENS1-CSF-SMOOTH-R2 报告图生成（m26894）：从 runs/*/result.json 程序化绘图，零手抄。
产出 runs/tens1_csf_r2/fig_depth.png + fig_mach.png（报告内嵌 base64 用）。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_PEAK_D, EXP_PEAK_PC_MS = 1.857434, 10.394


def load(arm):
    d = json.load(open(os.path.join(ROOT, "runs", arm, "result.json")))
    h = d["history"]
    cs = d.get("contact_step")
    ct = next((r["time_ms"] for r in h if r["step"] == cs), 0.0)
    t_pc = [r["time_ms"] - ct for r in h]
    depth = [r.get("depth_over_D") or 0.0 for r in h]
    mach = [r.get("mach_max") or 0.0 for r in h]
    return t_pc, depth, mach, ct


def main():
    out = os.path.join(ROOT, "runs", "tens1_csf_r2")
    os.makedirs(out, exist_ok=True)
    tc, dc, mc, ctc = load("ctrl")
    ts, ds, ms, cts = load("csf_smooth")

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ax.plot(tc, dc, color="tab:blue", lw=1.6, label="ctrl (filter off)")
    ax.plot(ts, ds, color="tab:red", lw=1.2, ls="--", label="csf_smooth (filter on)")
    ax.axhline(EXP_PEAK_D, color="k", lw=0.9, ls=":", label="experiment peak 1.857434 D")
    ax.axvline(EXP_PEAK_PC_MS, color="k", lw=0.9, ls=":")
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("open-water depth (D)")
    ax.set_title("Santini case-3 water entry, cells=8: cavity depth vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_depth.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ax.plot(tc, mc, color="tab:blue", lw=1.6, label="ctrl (filter off)")
    ax.plot(ts, ms, color="tab:red", lw=1.2, ls="--", label="csf_smooth (filter on)")
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("max lattice Mach")
    ax.set_title("Max lattice Mach vs time (parasitic-current indicator)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_mach.png"))
    plt.close(fig)
    print("wrote", out, "contact ctrl/csf ms:", round(ctc, 4), round(cts, 4))


if __name__ == "__main__":
    main()
