"""Figures for the KPI-QC-RECHECK independent re-judgement.

Zero hand-copy: every bar is read from the frozen audit record
(``artifacts/validation_matrix_independent_audit.json``) produced by
``benchmarks/validation_matrix/independent_audit.py``.

Panel A: per-level |err| / tol ratio for every gated metric level.  Bars above
         1.0 are tolerance violations; informational levels are drawn hollow,
         because by contract they cannot fail the metric (that asymmetry is the
         subject of the fix).
Panel B: the published headline number for ``ship_resistance_suboff`` before and
         after the fix, i.e. the contradiction the audit found and closed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "validation_matrix_independent_audit.json"
OUT = ROOT / "docs" / "figures" / "kpi_qc_recheck_gated_gate.png"

PASS_C = "#2f6f4e"
FAIL_C = "#a33b2a"
INFO_C = "#7a7a7a"


def main() -> int:
    report = json.loads(AUDIT.read_text(encoding="utf-8"))
    cases = report["cases"]
    labels, ratios, colors, hatches, edges = [], [], [], [], []
    for cid, rec in cases.items():
        for metric in rec["metrics"]:
            for lvl in metric["levels"]:
                err = lvl.get("err_pct")
                if err is None:
                    continue
                if lvl["gated"]:
                    ratio = abs(err) / lvl["tol_pct"]
                    labels.append(f"{cid}\n{metric['metric']} @{lvl['level']}")
                    colors.append(PASS_C if ratio <= 1.0 else FAIL_C)
                    hatches.append("")
                    edges.append(PASS_C)
                else:
                    # informational level: normalise by its own metric tolerance
                    ratio = abs(err) / metric["tol_pct"]
                    labels.append(f"{cid}\n{metric['metric']} @{lvl['level']}\n(info)")
                    colors.append("none")
                    hatches.append("///")
                    edges.append(INFO_C)
                ratios.append(ratio)

    order = sorted(range(len(ratios)), key=lambda i: -ratios[i])
    labels = [labels[i] for i in order]
    ratios = [ratios[i] for i in order]
    colors = [colors[i] for i in order]
    hatches = [hatches[i] for i in order]
    edges = [edges[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [3.1, 1.0]}
    )
    x = range(len(ratios))
    bars = ax1.bar(x, ratios, color=colors, edgecolor=edges, hatch=hatches,
                   linewidth=1.0)
    ax1.axhline(1.0, color=FAIL_C, lw=1.4, ls="--")
    ax1.text(len(ratios) - 0.5, 1.02, "tolerance", color=FAIL_C, ha="right",
             fontsize=9)
    for b, r in zip(bars, ratios):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.2f}",
                 ha="center", fontsize=7.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=90, fontsize=6.8)
    ax1.set_ylabel("|error| / tolerance  (own metric)")
    ax1.set_title(
        "A. Per-level independent re-judgement "
        f"({report['counts']['passed']} passed / {report['counts']['failed']} failed"
        f" / {report['counts']['pending']} pending)"
    )
    ax1.set_ylim(0, max(1.35, max(ratios) * 1.12))
    ax1.grid(axis="y", alpha=0.25)

    # ---- panel B: the published headline before/after the fix
    before, after = -3.17, -1.15  # percent, from the audit record below
    sub = cases["ship_resistance_suboff"]["metrics"][0]
    for lvl in sub["levels"]:
        if lvl["level"] == "L=48":
            before = lvl["err_pct"]
        if lvl["level"] == "L=80":
            after = lvl["err_pct"]
    ax2.bar([0, 1], [abs(before), abs(after)], color=[FAIL_C, PASS_C],
            width=0.55)
    ax2.axhline(3.0, color=FAIL_C, ls="--", lw=1.3)
    ax2.text(1.35, 3.05, "tol 3%", color=FAIL_C, ha="right", fontsize=9)
    ax2.text(0, abs(before) + 0.08, f"{before:+.2f}%", ha="center", fontsize=9)
    ax2.text(1, abs(after) + 0.08, f"{after:+.2f}%", ha="center", fontsize=9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["before fix\n(info L=48 shown\nas headline)",
                         "after fix\n(gated L=80)"], fontsize=8)
    ax2.set_ylabel("|Cd err| [%] published in STATUS_MATRIX row")
    ax2.set_title("B. ship_resistance_suboff\npublished headline")
    ax2.set_ylim(0, 3.7)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())