"""Assemble the platform report payload for P158W3-N01 (numbers from the record)."""

from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REC = os.path.join(HERE, "records", "rotating_w3.json")
DOC = os.path.join(ROOT, "docs", "ROTATING_W3_P158W3_N01.md")
FIG = os.path.join(ROOT, "docs", "assets", "rotating_w3")
OUT = "/tmp/report.json"

CAPTIONS = {
    "fig1_couette_profiles.png": "Taylor-Couette shell-averaged u_theta for the three gap-refined grids "
                                 "(left) and the differential cases vs the exact Ar+B/r (right).",
    "fig2_grid_convergence.png": "A1/A2/A6: nominal-radius error (staircase, non-monotone), "
                                 "effective-radius error, bulk shape residual and the dimensionless "
                                 "torque error vs refinement; right: the effective-wall offset "
                                 "r_eff - r_nom (half-link staircase).",
    "fig3_corotating_null.png": "A7 zero-value anchor: co-rotating cylinders keep the rigid-body "
                                "profile (left); the measured torque collapses from the differential-"
                                "case amplitude to ~1e-4 of it (right, log scale).",
    "fig4_ekman_spiral.png": "Ekman hodographs at delta_E = 6..16 nodes (points) against the exact "
                             "spiral (black), and the wall-shear/geostrophic angle vs 1/delta whose "
                             "extrapolation hits the 45 deg theory.",
    "fig5_ekman_resolution.png": "A4 at the resolution limit: delta_fit/delta_analytic error vs the "
                                 "number of nodes per Ekman layer (delta_E = 2, 3, 6, 8, 12, 16).",
    "fig6_stability.png": "Frame-rate sweep: residual and centrifugal-force scale vs Omega_f, and the "
                          "dimensionless location of the blow-up (Omega_f^2 r vs c_s^2, Ekman number).",
}


def main() -> int:
    rec = json.load(open(REC))
    crit = rec["criteria"]
    doc = open(DOC).read()
    figures = []
    for name in sorted(os.listdir(FIG)):
        if not name.endswith(".png"):
            continue
        png = base64.b64encode(open(os.path.join(FIG, name), "rb").read()).decode()
        figures.append({"name": name.replace(".png", ""),
                        "caption": CAPTIONS.get(name, name), "png_base64": png})
    a = lambda k: crit[k]["value"]  # noqa: E731
    summary = (
        f"TensorLBM 旋转参考系阶梯第三波=认证层：A1 剖面 {100*a('A1_couette_profile'):.2f}%(≤2%)、"
        f"A2 无量纲力矩 {100*a('A2_torque_coefficient'):.2f}%(≤3%)、A3 引用 W2 记录 "
        f"{100*a('A3_duct_secondary'):.2f}%(≤10%)、A4 Ekman 厚度 {100*a('A4_ekman_thickness'):.2f}%(δ≥4)、"
        f"δ=2 时 6.76%(分辨率下界)、A5 螺旋角外推 {crit['A5_ekman_spiral_angle_deg']['intercept_deg']:.2f}°"
        f"(Δ{a('A5_ekman_spiral_angle_deg'):.2f}°)、A6 GCI 观测阶 {crit['A6_gci_order']['order_bulk_shape_residual']:.2f}(≥1.5)、"
        f"A7 共转零力矩 {crit['A7_corotating_null']['value']:.1e}(≤2%)；"
        f"负结果：三条亚网格曲壁方案全败（线性外推发散、Bouzidi 定常退化、neq 外推无效），"
        f"阶梯壁 O(h) 不可消除；稳定性边界以 Ω_f²r/c_s² 定位。"
    )
    payload = {
        "title": "P158W3-N01 · 旋转参考系内部流场认证层：GCI + 无量纲力矩 + Ekman 螺旋角 + 旋转稳定性边界",
        "summary": summary,
        "scenario_key": "tensorlbm.rotating_frame.certification",
        "mission_id": 12403,
        "tags": ["TensorLBM", "rotating-frame", "Taylor-Couette", "Ekman-layer", "GCI",
                 "verification", "negative-result"],
        "markdown": doc,
        "figures": figures,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB,", len(figures), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())