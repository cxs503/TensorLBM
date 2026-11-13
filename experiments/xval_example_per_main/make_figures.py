#!/usr/bin/env python3
"""Figures for XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN (from artefacts only).

Reads artifacts/xval_example_per_main/{audit_tensorlbm,audit_listlbm,summary}.json
and writes figures/xval_example_per_main/*.png.  No number is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "artifacts" / "xval_example_per_main"
FIG = REPO_ROOT / "figures" / "xval_example_per_main"


def _load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def fig_metrics(tb: dict, lb: dict) -> Path:
    keys = ["entry_ratio", "dup_line_ratio", "cli_adoption"]
    labels = ["example carries\nown entry point", "boilerplate line\nshare (3-gram)", "CLI adoption\n(no source edit)"]
    tx = [tb[k] for k in keys]
    lx = [lb[k] for k in keys]
    x = range(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], tx, w, label=f"TensorLBM examples/ (n={tb['n_files']})",
                color="#2b6cb0")
    b2 = ax.bar([i + w / 2 for i in x], lx, w,
                label=f"listLBM examples/ (n={lb['n_files']})", color="#c05621")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                    f"{rect.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction (-)")
    ax.set_title("Example-per-main signature, same mechanical scanner on both trees")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    out = FIG / "fig1_antipattern_metrics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_sensitivity(summary: dict) -> Path:
    sens = summary["block_len_sensitivity"]
    xs = sorted(int(k) for k in sens)
    tb = [sens[str(k)]["tensorlbm_dup_line_ratio"] for k in xs]
    lb = [sens[str(k)]["listlbm_dup_line_ratio"] for k in xs]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, tb, "o-", label="TensorLBM examples/", color="#2b6cb0")
    ax.plot(xs, lb, "s--", label="listLBM examples/", color="#c05621")
    for x, y in zip(xs, tb):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(4, -12), fontsize=7)
    for x, y in zip(xs, lb):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xlabel("boilerplate n-gram length L (lines)")
    ax.set_ylabel("duplicate line share (-)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.0)
    ax.set_title("Sensitivity of the duplication metric to the n-gram length")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out = FIG / "fig2_blocklen_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_ab(summary: dict) -> Path:
    ab = summary["ab"]
    ids = list(ab)
    direct = [ab[i]["direct_median_s"] for i in ids]
    runner = [ab[i]["runner_median_s"] for i in ids]
    over = [ab[i]["overhead_pct"] for i in ids]
    x = range(len(ids))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    b1 = ax.bar([i - w / 2 for i in x], direct, w, label="direct interpreter", color="#4a5568")
    b2 = ax.bar([i + w / 2 for i in x], runner, w, label="via example_runner.py", color="#2f855a")
    for i, (d, q, o) in enumerate(zip(direct, runner, over)):
        ax.text(i, max(d, q) + 0.4, f"overhead {o:+.2f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_ylabel("median wall time (s, n=3)")
    ax.set_title("Dispatch transparency: same example, two invocation paths")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    out = FIG / "fig3_dispatch_overhead.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    tb = _load("audit_tensorlbm.json")
    lb = _load("audit_listlbm.json")
    summary = _load("summary.json")
    outs = [fig_metrics(tb, lb), fig_sensitivity(summary)]
    if summary.get("ab"):
        outs.append(fig_ab(summary))
    for out in outs:
        print(out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())