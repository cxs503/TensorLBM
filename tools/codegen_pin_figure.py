#!/usr/bin/env python3
"""Render the codegen-pin audit figure from the manifest (no hand-copied numbers).

Panel A: declared codegen configuration per benchmarked kernel -- the four
         distinct register budgets that existed before the pin (2/2, 1/2, 4/2,
         4/--), which is why an unmanaged budget makes throughput deltas
         unattributable.
Panel B: audit coverage -- launch sites routed through the pin helper vs
         inline, and how many declared pins actually fix a register cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "codegen_pin_manifest.json"
OUT = REPO / "artifacts" / "codegen_pin_audit.png"


def main() -> int:
    report = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pins = report["declared_pins"]
    summary = report["summary"]

    names = list(pins)
    warps = [pins[n]["knobs"]["num_warps"] for n in names]
    stages = [
        pins[n]["knobs"]["num_stages"] if pins[n]["knobs"]["num_stages"] is not None else 0
        for n in names
    ]
    caps = [pins[n]["knobs"]["maxnreg"] for n in names]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    xs = range(len(names))
    width = 0.38
    bars_w = ax_a.bar(
        [x - width / 2 for x in xs], warps, width, label="num_warps (declared)", color="#2b6cb0"
    )
    bars_s = ax_a.bar(
        [x + width / 2 for x in xs],
        stages,
        width,
        label="num_stages (declared, 0 = backend default)",
        color="#dd6b20",
    )
    for bar, value in zip(bars_w, warps):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, value + 0.08, str(value), ha="center", fontsize=9
        )
    for bar, value in zip(bars_s, stages):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            str(value) if value else "dflt",
            ha="center",
            fontsize=9,
        )
    ax_a.set_xticks(list(xs))
    ax_a.set_xticklabels(
        [n.replace("triton_fused_obstacle_", "obs_").replace("triton_", "") for n in names],
        rotation=18,
        ha="right",
        fontsize=9,
    )
    ax_a.set_ylabel("declared value")
    ax_a.set_title(
        f"A · declared codegen per kernel ({summary['distinct_pin_fingerprints']} distinct pin fingerprints)\n"
        f"maxnreg fixed in {summary['declared_pins_with_maxnreg']}/{len(pins)} pins -> budget was ptxas's choice",
        fontsize=10,
    )
    ax_a.legend(fontsize=8)
    ax_a.grid(axis="y", alpha=0.3)

    routed = summary["launch_sites_pinned_via_codegen_pin"]
    total = summary["launch_sites"]
    ax_b.barh(["launch sites", "declared pins"], [total, len(pins)], color="#e2e8f0", label="total")
    ax_b.barh(
        ["launch sites", "declared pins"],
        [routed, summary["declared_pins_with_maxnreg"]],
        color="#2f855a",
        label="routed through pin helper / cap fixed",
    )
    ax_b.set_xlabel("count")
    ax_b.set_title(
        "B · pin coverage after this change\n"
        f"{routed}/{total} launch sites via codegen_pin.launch_kwargs, "
        f"drift findings = {summary['drift_findings']}",
        fontsize=10,
    )
    for i, (a, b) in enumerate(
        [(routed, total), (summary["declared_pins_with_maxnreg"], len(pins))]
    ):
        ax_b.text(max(a, b) + 0.2, i, f"{a} / {b}", va="center", fontsize=9)
    ax_b.legend(fontsize=8, loc="lower right")
    ax_b.grid(axis="x", alpha=0.3)

    env = report["environment"]
    fig.suptitle(
        "TensorLBM Triton codegen pin audit — "
        f"triton {env.get('triton_version')}, torch {env.get('torch_version')}, "
        f"cuda_available={env.get('cuda_available')}, maxnreg_supported={env.get('maxnreg_supported')} "
        f"({env.get('machine')})",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print("maxnreg values:", caps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
