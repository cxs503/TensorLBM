#!/usr/bin/env python3
"""DE-X-DIAG2-CAVCLSE: figures for the card (matplotlib, no hand-typed numbers).

Usage: python3 diag2_cavclse_figures.py <extract_dir> <out_png>
Reads trajectory.csv / experiment.json / extract.json from the extract dir.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ed, out_png = sys.argv[1], sys.argv[2]
    rows = []
    with open(os.path.join(ed, "trajectory.csv")) as f:
        for r in __import__("csv").DictReader(f):
            rows.append({k: (float(v) if k != "step" else int(v))
                         for k, v in r.items()})
    exp = json.load(open(os.path.join(ed, "experiment.json")))
    summ = json.load(open(os.path.join(ed, "closure_summary.json")))
    t_close_pc = summ.get("t_close_after_contact_ms")
    peak_step = summ.get("peak_step")
    peak_row = next((r for r in rows if r["step"] == peak_step), None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) closure point depth vs time, with experiment
    ax = axes[0]
    ax.plot([r["t_ms_pc"] for r in rows], [r["depth_D"] for r in rows],
            "b-", lw=1.2, label="closure point depth (sim)")
    ax.plot([e["t_ms"] for e in exp], [e["depth_D"] for e in exp],
            "r-o", ms=4, label="experiment (Santini case-3)")
    if peak_row:
        ax.axvline(peak_row["t_ms_pc"], color="gray", ls=":", lw=1)
        ax.annotate(f"depth peak\n{peak_row['depth_D']:.4f} D @ {peak_row['t_ms_pc']:.2f} ms",
                    (peak_row["t_ms_pc"], peak_row["depth_D"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    if t_close_pc is not None:
        ax.axvline(t_close_pc, color="green", ls="--", lw=1)
        ax.annotate(f"t_close = {t_close_pc:.2f} ms (pc)",
                    (t_close_pc, 0.05), color="green", fontsize=8)
    ax.set_xlabel("t (ms, post-contact)")
    ax.set_ylabel("depth (D)")
    ax.set_title("(a) closure-point depth vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) closure point z (sub-cell, axis 0 up: smaller = deeper) vs time
    ax = axes[1]
    ax.plot([r["t_ms_pc"] for r in rows], [r["z"] for r in rows], "b-", lw=1.2)
    if peak_row:
        ax.axvline(peak_row["t_ms_pc"], color="gray", ls=":", lw=1)
    if t_close_pc is not None:
        ax.axvline(t_close_pc, color="green", ls="--", lw=1)
        zc = next((r["z"] for r in rows if abs(r["t_ms_pc"] - t_close_pc) < 1e-6), None)
        if zc is not None:
            ax.annotate("t_close", (t_close_pc, zc), color="green", fontsize=8)
    ax.invert_yaxis()  # deeper (smaller index) at the bottom, like a tank
    ax.set_xlabel("t (ms, post-contact)")
    ax.set_ylabel("z index (0 = bottom, up = larger)")
    ax.set_title("(b) closure point z vs time (inverted: down = deeper)")
    ax.grid(alpha=0.3)

    # (c) closure point x vs time
    ax = axes[2]
    ax.plot([r["t_ms_pc"] for r in rows], [r["x"] for r in rows], "b-", lw=1.0)
    if t_close_pc is not None:
        ax.axvline(t_close_pc, color="green", ls="--", lw=1)
    ax.set_xlabel("t (ms, post-contact)")
    ax.set_ylabel("x index (mid = 15.5)")
    ax.set_title("(c) closure point x vs time")
    ax.grid(alpha=0.3)

    fig.suptitle("DE-X-DIAG2-CAVCLSE: crater-closure point kinematics "
                 "(cells=8, CAVSWEEP ctrl anchor flags)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    print("figure ->", out_png)


if __name__ == "__main__":
    main()
