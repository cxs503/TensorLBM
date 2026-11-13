"""Audit m17963: build figures + machine-check table for the m17759 delivery audit.

Reads (read-only):
  * delivered  JSON: /root/tp-agent/work/m17759/xval/xval_nebb_bodyforce_driver.json
  * audit rerun JSON: <this dir>/rerun/xval_nebb_bodyforce_driver.json
Writes:
  * fig_m17963_profile.png  (ux(y) delivered vs analytic vs OSS, wall-slip annotated)
  * fig_m17963_metrics.png  (predeclared pointwise metric vs substituted global metric)
  * audit_numbers.json      (every number in the report is read from here -> zero hand-copy)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DELIV = Path("/root/tp-agent/work/m17759/xval/xval_nebb_bodyforce_driver.json")
RERUN = HERE / "rerun" / "xval_nebb_bodyforce_driver.json"


def analytic(ny, gx, nu):
    H = ny - 1
    y = np.arange(ny, dtype=float)
    return gx / (2.0 * nu) * y * (H - y)


def main():
    d = json.loads(DELIV.read_text())
    r = json.loads(RERUN.read_text())
    p = d["params"]["primary"]
    ny, gx, nu = p["NY"], p["Gx"], p["nu"]
    y = np.arange(ny)
    ref = analytic(ny, gx, nu)
    tlb = np.array(d["runs"]["tensorlbm_bgk_guo"]["ux_profile"])
    oss = np.array(d["runs"]["vivsim_bgk_edm"]["ux_profile"])
    oss_guo = np.array(d["runs"]["vivsim_bgk_guo"]["ux_profile"])

    # ---- figure 1: profiles ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(y, ref, "k-", lw=2.0, label="closed-form Poiseuille")
    ax.plot(y, oss, "o", ms=6, mfc="none", label="OSS vivsim arm (BGK+EDM), 5.00% global")
    ax.plot(y, tlb, "s--", ms=5, label="TensorLBM arm (BGK+Guo), 9.40% global")
    ax.axhline(0.0, color="0.7", lw=0.8)
    slip = tlb[0]
    ax.annotate(
        "wall slip u_x(y=0)=%.4f\n(= %.2f%% of u_max) -- no-slip not enforced"
        % (slip, 100.0 * slip / ref.max()),
        xy=(0.0, slip), xytext=(1.2, 0.030),
        fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel("y [lattice]")
    ax.set_ylabel(r"$u_x$ [lattice]")
    ax.set_title("m17759 audit: D2Q9 channel, NX=NY=10, nu=0.2, Gx=1e-3, 1e5 steps")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(HERE / "fig_m17963_profile.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: metrics vs the predeclared threshold --------------------
    e_tlb = np.abs(tlb - ref) / np.maximum(np.abs(ref), 1e-12)
    e_oss = np.abs(oss - ref) / np.maximum(np.abs(ref), 1e-12)
    point_tlb = float(e_tlb[1:-1].max())
    point_oss = float(e_oss[1:-1].max())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["TLB\nglobal-norm\n(report verdict)", "TLB\npointwise\n(predeclared)", "OSS\nglobal-norm", "OSS\npointwise\n(predeclared)"]
    vals = [100 * d["runs"]["tensorlbm_bgk_guo"]["err_global_vs_analytic"],
            100 * point_tlb,
            100 * d["runs"]["vivsim_bgk_edm"]["err_global_vs_analytic"],
            100 * point_oss]
    cols = ["#2b8cbe", "#de2d26", "#2b8cbe", "#de2d26"]
    bars = ax0.bar(labels, vals, color=cols)
    ax0.axhline(15.0, color="k", ls="--", lw=1.2, label="owner gate 15%")
    ax0.axhline(3.0, color="0.5", ls=":", lw=1.0, label="ecosystem 3%")
    for b, v in zip(bars, vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + 0.4, "%.2f%%" % v, ha="center", fontsize=8)
    ax0.set_ylabel("error vs closed form [%]")
    ax0.set_title("metric substitution flips PASS/FAIL")
    ax0.legend(frameon=False, fontsize=8)
    ax0.tick_params(labelsize=7)

    grids = ["NY=10", "NY=20"]
    tl_g = [100 * d["runs"]["tensorlbm_bgk_guo"]["err_global_vs_analytic"],
            100 * d["runs"]["grid_NY20_tensorlbm"]["err_global_vs_analytic"]]
    os_g = [100 * d["runs"]["vivsim_bgk_edm"]["err_global_vs_analytic"],
            100 * d["runs"]["grid_NY20_vivsim"]["err_global_vs_analytic"]]
    ax1.plot(grids, tl_g, "s--", label="TensorLBM arm")
    ax1.plot(grids, os_g, "o-", label="OSS arm")
    ax1.axhline(15.0, color="k", ls="--", lw=1.2)
    for g, a, b in zip(grids, tl_g, os_g):
        ax1.text(g, a + 0.3, "%.2f%%" % a, fontsize=8, ha="center")
        ax1.text(g, b - 0.9, "%.2f%%" % b, fontsize=8, ha="center")
    ax1.set_ylabel("global-normalised error [%]")
    ax1.set_title("grid refinement (delivered JSON, audit re-ran identically)")
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_m17963_metrics.png", dpi=130)
    plt.close(fig)

    # ---- numbers table -----------------------------------------------------
    numbers = {
        "delivered_json": str(DELIV),
        "rerun_json": str(RERUN),
        "rerun_payload_identical": bool(d == r),
        "params": p,
        "tlb_global_err": float(d["runs"]["tensorlbm_bgk_guo"]["err_global_vs_analytic"]),
        "tlb_pointwise_interior_err": point_tlb,
        "tlb_wall_ux": float(tlb[0]),
        "tlb_wall_slip_pct_of_umax": float(100.0 * abs(tlb[0]) / ref.max()),
        "tlb_umax": float(d["runs"]["tensorlbm_bgk_guo"]["umax"]),
        "oss_global_err": float(d["runs"]["vivsim_bgk_edm"]["err_global_vs_analytic"]),
        "oss_pointwise_interior_err": point_oss,
        "cross_arm_global_err": float(d["runs"]["cross_arm"]["err_global_vs_vivsim_bgk_edm"]),
        "cross_arm_max_abs_diff": float(d["runs"]["cross_arm"]["max_abs_diff_vs_vivsim_bgk_edm"]),
        "tlb_ny20_global_err": float(d["runs"]["grid_NY20_tensorlbm"]["err_global_vs_analytic"]),
        "oss_ny20_global_err": float(d["runs"]["grid_NY20_vivsim"]["err_global_vs_analytic"]),
        "owner_gate_pct": 15.0,
        "ecosystem_gate_pct": 3.0,
        "predeclared_metric_pass": bool(point_tlb < 0.15),
        "substituted_metric_pass": bool(d["runs"]["tensorlbm_bgk_guo"]["err_global_vs_analytic"] < 0.15),
        "ecosystem_3pct_met": bool(d["runs"]["tensorlbm_bgk_guo"]["err_global_vs_analytic"] < 0.03),
    }
    (HERE / "audit_numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()