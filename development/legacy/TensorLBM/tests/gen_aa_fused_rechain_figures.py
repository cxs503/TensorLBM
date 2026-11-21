#!/usr/bin/env python3
"""Figure for the Ghia witness re-chain (fig9).

Panels
------
(a) per-table re-derivation coverage from the sha256-pinned OCR witness, with
    the residual (text-layer-corrupted) cells marked;
(b) the acceptance re-score: L2 errors of the stored converged fields against
    the witness-derived Re=100 table, with the 3 % and 15 % gates.

Reads ``artifacts/xval_aa_pattern/ghia_witness_rechain.json`` (produced by
``benchmarks/xval/ghia_witness_rechain.py``).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, ".."))
ART = os.path.join(_PKG, "artifacts", "xval_aa_pattern")
SRC = os.path.join(ART, "ghia_witness_rechain.json")


def main() -> str:
    r = json.load(open(SRC))
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) coverage per table / Re column
    labels, matched, missing = [], [], []
    for table in ("I", "II"):
        cells = r["magnitude_rechain"][table]
        hits = cells["hits"] if "hits" in cells else []
        cells_full = r["magnitude_rechain_hits"][table]
        for re_key in ("100", "400", "1000"):
            n_hit = sum(1 for h in cells_full if f"/Re{re_key}/" in h["cell"])
            n_tot = sum(1 for h in cells_full if f"/Re{re_key}/" in h["cell"])
            miss = sum(1 for m in cells["misses"] if f"/Re{re_key}/" in m["cell"])
            labels.append(f"{table}\nRe{re_key}")
            matched.append(n_hit)
            missing.append(miss)
    x = np.arange(len(labels))
    ax[0].bar(x, matched, color="#2b6cb0", label="re-derived from witness")
    ax[0].bar(x, missing, bottom=matched, color="#c53030", label="text-layer corrupted")
    for xi, (m, k) in enumerate(zip(matched, missing)):
        ax[0].text(xi, m + k + 0.6, f"{m}/{m + k}", ha="center", fontsize=8)
    ax[0].set_xticks(x, labels, fontsize=8)
    ax[0].set_ylabel("cells checked")
    ax[0].set_title("(a) Ghia table re-derived from OCR witness\n"
                    f"sha256 match={r['witness']['match']}", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].set_ylim(0, 80)

    # (b) acceptance re-score
    grids = sorted(r["rescore"])
    xb = np.arange(len(grids) * 2, dtype=float)
    vals, cols, names = [], [], []
    for i, g in enumerate(grids):
        vals += [100.0 * r["rescore"][g]["L2_u"], 100.0 * r["rescore"][g]["L2_v"]]
        cols += ["#2f855a", "#805ad5"]
        names += [f"{g}\n$L_2u$", f"{g}\n$L_2v$"]
    ax[1].bar(xb, vals, color=cols)
    ax[1].axhline(3.0, color="k", ls="--", lw=1, label="3 % target")
    ax[1].axhline(15.0, color="r", ls=":", lw=1.2, label="15 % acceptance")
    for xi, v in zip(xb, vals):
        ax[1].text(xi, v + 0.3, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(xb, names, fontsize=8)
    ax[1].set_ylabel("L2 error vs Ghia Re=100 [% of lid speed]")
    ax[1].set_title("(b) cavity re-score with the witness-derived table\n"
                    "(fused stream-collide, converged fields)", fontsize=9)
    ax[1].set_ylim(0, 6)
    ax[1].legend(fontsize=7)

    fig.tight_layout()
    dst = os.path.join(ART, "fig9_ghia_witness_rechain.png")
    fig.savefig(dst, dpi=150)
    print(dst)
    return dst


if __name__ == "__main__":
    main()
