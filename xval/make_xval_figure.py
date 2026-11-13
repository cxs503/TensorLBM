"""Generate XVAL figures from xval_nebb_bodyforce_driver.json (zero hand-copy)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    res = json.loads((HERE / "xval_nebb_bodyforce_driver.json").read_text())
    runs = res["runs"]
    NY = res["params"]["primary"]["NY"]
    y = np.arange(NY)
    ref = np.array(runs["tensorlbm_bgk_guo"]["ux_profile"])  # analytic proxy not stored; recompute
    # recompute analytic from params
    p = res["params"]["primary"]
    H = p["NY"] - 1
    ref = p["Gx"] / (2 * p["nu"]) * y * (H - y)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(y, ref, "k-", lw=2, label="Analytic Poiseuille")
    ax.plot(y, runs["vivsim_bgk_edm"]["ux_profile"], "o", ms=5, label="OSS vivsim (BGK+EDM)")
    ax.plot(y, runs["vivsim_bgk_guo"]["ux_profile"], "s", ms=5, label="OSS vivsim (BGK+Guo)")
    ax.plot(y, runs["tensorlbm_bgk_guo"]["ux_profile"], "^", ms=5, label="TensorLBM (BGK+Guo + NEBB)")
    ax.set_xlabel("y (lattice units)")
    ax.set_ylabel("$u_x$ (lattice units)")
    ax.set_title("Force-corrected NEBB + body-force driver: Poiseuille channel\n"
                 f"NX={p['NX']}, NY={p['NY']}, nu={p['nu']}, Gx={p['Gx']}, T={p['steps']}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(HERE / "xval_nebb_profile.png", dpi=130)
    plt.close(fig)

    # base64 for report
    b64 = base64.b64encode((HERE / "xval_nebb_profile.png").read_bytes()).decode()
    out = {
        "png": "xval/xval_nebb_profile.png",
        "png_base64": b64,
        "caption": "Steady $u_x(y)$ profile: TensorLBM force-corrected NEBB + Guo body-force driver "
                   "vs OSS vivsim (BGK+EDM and BGK+Guo) vs closed-form Poiseuille solution.",
    }
    (HERE / "xval_figure_b64.json").write_text(json.dumps(out))
    print("wrote", HERE / "xval_nebb_profile.png")
    print("verdict:", json.dumps(res["verdict"], indent=2))


if __name__ == "__main__":
    main()
