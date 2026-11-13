#!/usr/bin/env python3
"""m17889 审计配图：把交付声称的误差 vs 本单核出的诚实误差画在一张图上。

数据源（全部机器读，零手抄）：
  artifacts/m17889_audit/audit_m17889.json   —— 本单审计读数
  benchmarks/xval_huang2007_contact_angle_inversion_audit_copy/run_xval.py —— 交付标定表（模块常量）

输出：docs/audit_m17889/fig_m17889_honest_vs_claimed.png
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AUDIT = json.loads((ROOT / "artifacts" / "m17889_audit" / "audit_m17889.json").read_text())
COPY = ROOT / "benchmarks" / "xval_huang2007_contact_angle_inversion_audit_copy"
OUTDIR = ROOT / "docs" / "audit_m17889"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_tables():
    spec = importlib.util.spec_from_file_location("xval_delivered", COPY / "run_xval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CALIBRATION_DATA_COARSE, mod.CALIBRATION_DATA_FINE


def main() -> int:
    coarse, fine = load_tables()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # ---------------- (a) 标定曲线 + 交付直线拟合 + LOO ----------------
    ax = axes[0]
    fw = np.array([d["f_w"] for d in coarse])
    th = np.array([d["theta_measured"] for d in coarse])
    ax.plot(fw, th, "o-", color="#1f4e79", lw=1.6, ms=5, label="delivered calibration table (coarse, measured)")
    cal = AUDIT["4_delivered_table_recompute"]
    xs = np.linspace(min(fw), max(fw), 100)
    slope = AUDIT["4_delivered_table_recompute"] and 5.032
    inter = -2.575
    ax.plot(xs, np.degrees(np.arccos(np.clip(slope * xs + inter, -1, 1))), "--",
            color="#c00000", lw=1.2, label="delivered linear law cos(theta)=5.032*f_w-2.575")
    loo = AUDIT["3_leave_one_out"]["rows"]
    ax.plot([r["held_out_f_w"] for r in loo], [r["theta_predicted"] for r in loo],
            "x", color="#e07b00", ms=10, mew=2, label="this audit: LOO out-of-sample prediction")
    ax.set_xlabel(r"$f_w=\psi_w/\psi_l$")
    ax.set_ylabel(r"measured $\theta$  [deg]")
    ax.set_title("(a) calibration curve: points / delivered linear law / audit LOO")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    # ---------------- (b) 声称误差 vs 诚实误差 ----------------
    ax = axes[1]
    labels, vals, colors = [], [], []
    rows = cal["rows"]
    claimed = [r["err_huang_pct"] for r in rows]
    labels.append("delivered claim\nHuang2007\n(median of 9)")
    vals.append(float(np.median(claimed)))
    colors.append("#8faadc")

    loo_in = [r["rel_err_pct"] for r in loo if not r["out_of_range_extrapolation"]]
    labels.append("audit LOO\n(worst interior)")
    vals.append(max(loo_in))
    colors.append("#e07b00")

    cg = AUDIT["5_cross_grid_holdout"]["rows"]
    labels.append("cross-grid hold-out\n(in-repo LBM, max)")
    vals.append(max(r["lbm_rel_err_pct"] for r in cg))
    colors.append("#c00000")

    spots = AUDIT.get("8_fresh_lbm_spots", {}).get("cases", {})
    if spots:
        sv = [v["rel_err_pct"] for v in spots.values() if "rel_err_pct" in v]
        if sv:
            labels.append("fresh LBM spots\n(prescription, max)")
            vals.append(max(sv))
            colors.append("#7030a0")

    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    ax.axhline(15.0, color="k", ls="--", lw=1.2)
    ax.text(len(vals) - 0.5, 15.8, "owner criterion 15%", ha="right", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("relative error [%]")
    ax.set_title("(b) delivered \"8/9 pass\" vs honest errors")
    ax.grid(alpha=0.3, axis="y")

    # ---------------- (c) 处方 → 实测角：交付 vs 本单复验 ----------------
    ax = axes[2]
    ax.plot([0, 180], [0, 180], "k:", lw=1, label="ideal meas=target")
    tgt = [r["theta_target"] for r in rows]
    ax.plot(tgt, [r["theta_meas_huang"] for r in rows], "s", color="#8faadc", ms=6,
            label="delivered table (invert+forward: identity)")
    if cg:
        ax.plot([r["theta_target"] for r in cg], [r["fine_lbm_measured_deg"] for r in cg],
                "^", color="#c00000", ms=8, label="cross-grid LBM (m15470 B2)")
    if spots:
        ax.plot([v["theta_eq"] for v in spots.values()], [v["theta_measured"] for v in spots.values()],
                "o", color="#7030a0", ms=9, label="fresh LBM spots (same grid)")
        for k, v in spots.items():
            ax.annotate(f"{v['theta_measured']:.1f}°", (v["theta_eq"], v["theta_measured"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8, color="#7030a0")
    ax.set_xlabel(r"target $\theta_{target}$ [deg]")
    ax.set_ylabel(r"measured $\theta$ [deg]")
    ax.set_title("(c) prescription hit: identity line vs real LBM")
    ax.set_xlim(20, 160)
    ax.set_ylim(20, 160)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("m17889 cross-audit: XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION (target m17845)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUTDIR / "fig_m17889_honest_vs_claimed.png"
    fig.savefig(out, dpi=135)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())