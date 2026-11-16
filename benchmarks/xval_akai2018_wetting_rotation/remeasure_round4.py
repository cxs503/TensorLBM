#!/usr/bin/env python3
"""Re-measure the round-4 phase fields with the validated estimator.

Round 4 (``benchmarks/xval_akai2018_wetting``) stored one final phase field per
arm (``_last_phi_<arm>.npy``).  Their reported angles came from a Kasa circle
fit of the ``phi = 0`` contour, which ``estimator_validation.py`` shows to be
structurally invalid for shallow caps.  This script re-measures the *same*
saved fields with the cap area + contact-width inversion and writes
``round4_remeasure.json`` plus the report figures.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from run import Y_WALL, count_clusters, theta_circle_sym  # noqa: E402
from tensorlbm.wetting.metrics import contact_angle_circle_fit, spherical_cap_inversion  # noqa: E402

R4 = HERE.parent / "xval_akai2018_wetting"
ARMS = {"ctl60": 90.0, "gk30": 30.0, "gk60": 60.0, "gk90": 90.0}


def main() -> int:
    out = {"source": str(R4.relative_to(HERE.parents[1])), "arms": {}}
    for arm, target in ARMS.items():
        phi = np.load(R4 / f"_last_phi_{arm}.npy")
        rec = {"theta_target": target}
        try:
            cap = spherical_cap_inversion(phi, Y_WALL, level=0.0)
            rec["theta_cap"] = cap["theta_deg"]
            rec["contact_width"] = cap["contact_width"]
        except Exception as exc:
            rec["theta_cap"] = float("nan")
            rec["cap_error"] = repr(exc)
        # the round-4 decision metric, unchanged (trim 0.05 as the old driver used)
        try:
            old = contact_angle_circle_fit(phi, Y_WALL, trim=0.05, level=0.0)
            rec["theta_circle_r4_trim05"] = old["theta_deg"]
            rec["circle_r4"] = {k: old[k] for k in ("x_center", "y_center", "R_fit", "rms", "n_points")}
        except Exception as exc:
            rec["theta_circle_r4_trim05"] = float("nan")
            rec["circle_r4_error"] = repr(exc)
        rec["theta_circle_sym_trim25"] = theta_circle_sym(phi)
        rec["clusters"] = count_clusters(phi)
        rec["err_pct_cap"] = 100.0 * (rec["theta_cap"] - target) / target
        rec["single_cluster"] = len(rec["clusters"]) == 1
        rec["pass_15pct"] = bool(abs(rec["err_pct_cap"]) <= 15.0 and rec["single_cluster"])
        out["arms"][arm] = rec
        print(f"{arm:6s} target {target:5.1f}  cap {rec['theta_cap']:7.2f} "
              f"({rec['err_pct_cap']:+6.1f}%)  circle(r4) {rec['theta_circle_r4_trim05']:7.2f}  "
              f"clusters {rec['clusters']}")

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = json.loads((HERE / "estimator_validation.json").read_text())
    rows = ev["rows"]
    th = sorted({r["theta_true"] for r in rows})
    def agg(key, R, xi):
        out_ = []
        for t in th:
            sel = [r for r in rows if r["theta_true"] == t and r["R"] == R and r["xi"] == xi][0]
            out_.append(sel[key + "_err_deg"])
        return out_

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(th, agg("theta_cap", 30, 2.0), "o-", label=r"$\theta_{cap}$ (area+width), R=30")
    ax[0].plot(th, agg("theta_cap", 12, 2.0), "s--", label=r"$\theta_{cap}$, R=12")
    ax[0].plot(th, agg("theta_circle_sym", 30, 2.0), "^-", color="crimson",
               label="circle fit (round-4 metric), R=30")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axhspan(-3, 3, color="0.85", label="±3°")
    ax[0].set_xlabel("prescribed angle of the analytic cap (deg)")
    ax[0].set_ylabel("measured − prescribed (deg)")
    ax[0].set_title("estimator validation on analytic tanh caps (81 cases)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    labels = list(out["arms"])
    width = np.arange(len(labels))
    ax[1].bar(width - 0.2, [out["arms"][a]["theta_target"] for a in labels], 0.4,
              label="target", color="0.7")
    ax[1].bar(width + 0.2, [out["arms"][a]["theta_cap"] for a in labels], 0.4,
              label=r"re-measured $\theta_{cap}$", color="steelblue")
    for i, a in enumerate(labels):
        if not out["arms"][a]["single_cluster"]:
            ax[1].text(i, out["arms"][a]["theta_cap"] + 3, "satellite drop", color="crimson",
                       fontsize=7, ha="center")
    ax[1].set_xticks(width, labels)
    ax[1].set_ylabel("contact angle (deg)")
    ax[1].set_title("round-4 saved fields, re-measured")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = HERE / "fig_estimator_and_round4.png"
    fig.savefig(p, dpi=130)
    out["figure"] = p.name
    (HERE / "round4_remeasure.json").write_text(json.dumps(out, indent=2, default=float))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
