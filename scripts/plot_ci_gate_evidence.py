#!/usr/bin/env python3
"""Archival figure for ``docs/m4645_ci_red_fix_evidence.md``.

    python scripts/plot_ci_gate_evidence.py

Writes ``docs/assets/m4645/ci_gates.png`` (2 panels).

**This script plots an archive, not a live measurement.**  The red-state counts
in ``DATA["red"]`` are the numbers that were printed by the commands quoted in
§1/§2 of the evidence document on 2026-11-04 (branch ``agent/m3643`` @
``a75534da``, loong64 / ruff 0.15.0 / pytest 9.2.0).  They cannot be
re-measured on current ``main`` because main is already healed — re-running
``ruff check`` today prints 0 by construction, so a "live" plot would be a
tautology.  The post-fix column is likewise the archived local re-run output.

Panel (b) is *not* archived: the four analytic errors are read from the error
ledger that ``scripts/verify_lbm_analytic_convergence.py`` regenerates
(``artifacts/lbm_analytic_verification/convergence.json``), so that half of the
figure is reproducible — run that script first if the ledger is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts" / "lbm_analytic_verification" / "convergence.json"
OUT = ROOT / "docs" / "assets" / "m4645" / "ci_gates.png"

# Archived gate counts: (label, red-state count, post-fix count).
# Provenance per row quoted in the evidence document §1/§2 (§4 for the zeros).
DATA = (
    ("ruff check\n[errors]", 95, 0),
    ("ruff format --check\n[files]", 36, 0),
    ("repo layout\n[violations]", 7, 0),
    ("test_bfl_suboff\n[failed]", 4, 0),
)

# Canonical (default-argument) case order used by the test module.
CASES = (
    ("A_shear_wave", r"A: $\nu$"),
    ("B_couette", "B: Couette"),
    ("C_poiseuille", "C: Poiseuille"),
    ("D_friction", r"D: $C_f$"),
)


def _canonical_errors() -> tuple[list[str], list[float], float]:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    canon = payload["canonical"]
    labels, errs = [], []
    for case, label in CASES:
        labels.append(label)
        errs.append(abs(float(canon[case]["err_pct"])))
    return labels, errs, float(payload["rel_tol_pct"])


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # (a) red -> green per CI gate
    labels = [row[0] for row in DATA]
    red = [row[1] for row in DATA]
    green = [row[2] for row in DATA]
    x = range(len(DATA))
    width = 0.38
    b1 = axes[0].bar(
        [i - width / 2 for i in x], red, width, color="#e76f51", label="before (PR branch state)"
    )
    b2 = axes[0].bar(
        [i + width / 2 for i in x], green, width, color="#2a9d8f", label="after (main merged in)"
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("count")
    axes[0].set_title("(a) CI gates: red state -> 0 violations")
    axes[0].set_ylim(0, max(red) * 1.18)
    for bars, vals in ((b1, red), (b2, green)):
        for bar, v in zip(bars, vals):
            axes[0].annotate(
                str(v),
                (bar.get_x() + bar.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    # (b) the four analytic errors the healed gate now actually executes
    labels, errs, tol = _canonical_errors()
    colours = ["#2a9d8f" if e < tol else "#e76f51" for e in errs]
    bars = axes[1].bar(labels, errs, color=colours)
    axes[1].axhline(tol, color="k", ls="--", lw=1.0, label=f"{tol:g}% acceptance")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max relative error [%]")
    axes[1].set_title("(b) analytic suite A-D: error vs 1% acceptance")
    for bar, e in zip(bars, errs):
        axes[1].annotate(
            f"{e:.2e}%",
            (bar.get_x() + bar.get_width() / 2, e),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylim(min(errs) / 20, tol * 20)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
