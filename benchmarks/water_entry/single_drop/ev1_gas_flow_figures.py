#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-EV1-GAS_FLOW-R2 · 结果图（程序化生成，零手抄）。

用法：python3 benchmarks/water_entry/single_drop/ev1_gas_flow_figures.py --tree .
输出：docs/audits/ev1_gas_flow_20261121/figs/gas_flow_depth.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=".")
    a = ap.parse_args()
    tree = Path(a.tree).resolve()

    def load(arm: str):
        d = json.loads((tree / "runs" / arm / "result.json").read_text())
        h = d["history"]
        dt = h[1]["time_ms"] - h[0]["time_ms"]
        c0 = (d.get("contact_step") or 0) * dt
        t = [r["time_ms"] - c0 for r in h]
        dep = [r.get("depth_over_D", float("nan")) for r in h]
        return t, dep, d

    tc, dc, _ = load("EV1-GAS_FLOW-CTRL")
    ta, da, da_raw = load("EV1-GAS_FLOW")

    # absolute-time copies for the pre-impact zoom (arm never reached contact)
    def load_abs(arm: str):
        d = json.loads((tree / "runs" / arm / "result.json").read_text())
        h = d["history"]
        return [r["time_ms"] for r in h], [r.get("depth_over_D", float("nan")) for r in h]

    tca, dca = load_abs("EV1-GAS_FLOW-CTRL")
    taa, daa = load_abs("EV1-GAS_FLOW")

    exp = []
    with open(tree / "benchmarks" / "water_entry" / "single_drop" / "fig5_experiment.csv") as fh:
        for row in csv.DictReader(fh):
            exp.append((float(row["time_ms"]), float(row["depth_over_D"])))
    ex, ed = zip(*exp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.plot(tc, dc, color="tab:blue", lw=1.4, label="CTRL (gas_flow off)")
    ax1.plot(ta, da, color="tab:green", lw=1.4, label="GAS_FLOW (died 0.18 ms, drift gate)")
    ax1.plot(ex, ed, "o", color="tab:red", ms=4, label="Santini case-3 (18 pts)")
    pkc = int(max(range(len(dc)), key=lambda i: dc[i]))
    ax1.plot([tc[pkc]], [dc[pkc]], "b^", ms=7)
    ax1.annotate(
        f"ctrl {dc[pkc]:.4f}D@{tc[pkc]:.2f}ms",
        (tc[pkc], dc[pkc]),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=8,
    )
    ax1.set_xlabel("time (ms, post-contact)")
    ax1.set_ylabel("cavity depth (D)")
    ax1.set_title("Cavity depth vs time (full window)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(tca, dca, color="tab:blue", lw=1.4, label="CTRL")
    ax2.plot(taa, daa, color="tab:green", lw=1.4, label="GAS_FLOW")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 0.03)
    ax2.set_xlabel("time (ms, absolute)")
    ax2.set_ylabel("cavity depth (D)")
    ax2.set_title("Pre-impact window (zoom, 0-1 ms)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = tree / "docs" / "audits" / "ev1_gas_flow_20261121" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gas_flow_depth.png", dpi=150)
    print(f"saved {out / 'gas_flow_depth.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
