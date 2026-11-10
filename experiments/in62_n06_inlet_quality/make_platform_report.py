#!/usr/bin/env python3
"""Assemble the platform report payload for IN62-N06 from the run artifacts.

Writes JSON with ``title`` / ``summary`` / ``scenario_key`` / ``mission_id`` /
``tags`` / ``markdown`` / ``figures`` (PNG bytes embedded as base64).  The body is
the generated ``docs/IN62_N06_INLET_QUALITY.md``; nothing is typed by hand.

Usage::

    python experiments/in62_n06_inlet_quality/make_platform_report.py \
        --run-dir runs/in62_n06_inlet_quality \
        --doc docs/IN62_N06_INLET_QUALITY.md --out /tmp/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

CAPTIONS = {
    "fig1_inlet_plane_quality.png":
        "Inlet-plane Reynolds stresses (normalised by the target R_uu) and the "
        "measured divergence diagnostic for the four inlet variants.",
    "fig2_development_section.png":
        "Development section: R_ij(x) relaxation and the rms pressure "
        "fluctuation (spurious-acoustic negative control) for DFSEM / SEM / DFM.",
    "fig3_isotropic_decay.png":
        "Isotropic decay against the Comte-Bellot & Corrsin (1971) exponent and "
        "the energy spectra at the first and last sample.",
    "fig4_inlet_field_clouds.png":
        "Inlet-plane field clouds: projected u', v', w' and the divergence "
        "source term, unprojected versus projected.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/in62_n06_inlet_quality")
    ap.add_argument("--doc", default="docs/IN62_N06_INLET_QUALITY.md")
    ap.add_argument("--out", default="/tmp/report.json")
    args = ap.parse_args()

    markdown = Path(args.doc).read_text()
    run_dir = Path(args.run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    gates = report.get("gates", {})
    ldev = {}
    if (run_dir / "results_ldev.json").exists():
        ldev = json.loads((run_dir / "results_ldev.json").read_text())
    passed = sum(1 for k, v in gates.items() if k.endswith("_pass") and v is True)
    total = sum(1 for k in gates if k.endswith("_pass"))

    figures = []
    for png in sorted((Path(args.run_dir) / "figures").glob("*.png")):
        figures.append(
            {
                "name": png.name,
                "caption": CAPTIONS.get(png.name, png.stem),
                "png_base64": base64.b64encode(png.read_bytes()).decode("ascii"),
            }
        )

    ldev_cases = ldev.get("cases", {})
    l_dev = (ldev_cases.get("dfsem") or {}).get("predeclared_5pct_shape_only")
    l_dev_iso = (ldev_cases.get("dfsem_iso") or {}).get("predeclared_5pct_shape_only")
    payload = {
        "title": "IN62-N06 LBM 无散合成湍流入口（DFSEM/digital-filter）与开发段自保持定律",
        "summary": (
            f"在 TensorLBM 实现 DFSEM（涡核+后验散度投影，支持任意 R_ij）与 Klein "
            f"digital-filter+Lund 两条入口路径并建立 12 项机器门（{passed}/{total} 通过）："
            f"入口面 R_ij 回收误差 {100 * gates.get('G1_plane_stress_error_max', float('nan')):.2f}%，"
            f"投影后散度指标 D={gates.get('G2_plane_div_ratio_projected', float('nan')):.2e}"
            f"（未投影 {gates.get('G3_negative_control_div_ratio', float('nan')):.3f}，"
            f"降幅 {gates.get('G2_plane_div_reduction', float('nan')):.0f}×）；"
            f"开发段入口回收误差 {100 * gates.get('G5_duct_inlet_recovery_error', float('nan')):.2f}%，"
            f"自保持长度 L_dev(5%, 形状判据)={l_dev} cells"
            f"（各向同性入口 {l_dev_iso}），伪压力比 SEM/DFSEM="
            f"{gates.get('G7_ratio', float('nan')):.2f}；"
            f"各向同性衰减指数 n={gates.get('G8_n_from_lbm', float('nan')):.3f} vs "
            f"CBC71 n=1.30（相对误差 {100 * gates.get('G8_rel_error', float('nan')):.1f}%，"
            f"honest FAIL）。G6 自保持长度与 G8 衰减指数两项未达标，"
            f"其余 {passed}/{total} 门通过，失败原因与适用范围见报告。"
        ),
        "scenario_key": "in62_n06_inlet_quality",
        "mission_id": 7018,
        "tags": ["TensorLBM", "synthetic-turbulence", "DFSEM", "digital-filter",
                 "Lund-transform", "divergence-free", "development-length",
                 "Comte-Bellot-Corrsin", "inlet-quality", "IN62-N06"],
        "markdown": markdown,
        "figures": figures,
    }
    Path(args.out).write_text(json.dumps(payload))
    print(f"wrote {args.out}: {len(markdown)} chars, {len(figures)} figures, "
          f"gates {passed}/{total} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())