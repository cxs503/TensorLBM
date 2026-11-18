"""Generate report figures from the pipeline campaign artifacts.

Reads the profiles + campaign summary written by ``run_campaign.py`` and
renders the profile comparison vs the exact plane-Poiseuille solution.

Usage (from the repo root):
    python3 runs/m25066_pipeline_real/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ANCHOR = "XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE"


def main() -> None:
    prof = json.loads((HERE / "artifacts" / f"{ANCHOR}_profiles.json").read_text())
    summary = json.loads((HERE / "campaign_summary.json").read_text())
    m = summary["metrics"]

    y = np.array(prof["y"])
    u_sim = np.array(prof["u_sim"])
    u_ref = np.array(prof["u_ref"])
    rel_err = np.abs(u_sim - u_ref) / u_ref

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(u_ref, y, "k-", lw=1.5, label="exact plane Poiseuille")
    ax1.plot(u_sim, y, "o-", color="tab:blue", ms=4, lw=1, label="TensorLBM D2Q9 BGK (Zou-He)")
    ax1.set_xlabel("u (lattice units)")
    ax1.set_ylabel("y (lattice units)")
    ax1.set_title(
        f"Converged channel profile vs analytic solution\n"
        f"eps_inf={m['eps_inf']:.4f}   eps_l2={m['eps_l2']:.4f}   (acceptance < 0.15)"
    )
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.semilogy(y, rel_err, "s-", color="tab:red", ms=4, lw=1)
    ax2.axhline(0.15, color="k", ls="--", lw=1, label="15% acceptance bound")
    ax2.set_xlabel("y (lattice units)")
    ax2.set_ylabel("|u_LBM - u_exact| / u_exact")
    ax2.set_title(f"Pointwise relative error (max = {rel_err.max():.4f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = HERE / "fig1_profile_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
