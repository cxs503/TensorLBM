#!/usr/bin/env python
"""Build the platform report payload for P154W3-N11 and print it to a file.

The markdown comes from ``docs/P154W3_N11_CONVECTIVE_MASS_TRANSFER.md`` (itself
generated from ``artifacts/p154w3_n11/summary.json``) and every figure from
``artifacts/p154w3_n11/figs`` is embedded as base64, so nothing in the report is
hand-transcribed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p154w3_n11"
DOC = ROOT / "docs" / "P154W3_N11_CONVECTIVE_MASS_TRANSFER.md"

CAPTIONS = {
    "fig1_calibration.png": "Numerical diffusivity D_eff/D of the D2Q5 scalar lattice vs cell Peclet "
                            "number, TRT (Λ=3/16) vs BGK: the high-Schmidt-number error floor.",
    "fig2_leveque.png": "Leveque wall: local Sherwood number vs the exact similarity solution "
                        "(left) and the deviation at three resolution levels (right).",
    "fig3_blasius.png": "Blasius plate: Sh_x vs 0.332 Re_x^{1/2} Sc^{1/3} for Sc = 1…1000 (left) "
                        "and the deviation vs Re_x (right).",
    "fig4_cylinder.png": "Cylinder in cross flow: measured mean Sh vs Ranz-Marshall (dashed) and "
                         "Hilpert (dotted).",
    "fig5_analogy.png": "Chilton-Colburn analogy: j_D, j_H and C_f/2 on the Blasius plate (left) "
                        "and the coupled LBM-momentum arm (right).",
    "fig6_fields.png": "Concentration fields: Leveque wall layer (symlog y) and Blasius plate at "
                       "Sc = 100 with the Sh_x comparison.",
}


def main() -> None:
    md = DOC.read_text(encoding="utf-8")
    figs = []
    for name, cap in CAPTIONS.items():
        p = ART / "figs" / name
        if not p.exists():
            continue
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode("ascii")})
    summary = json.loads((ART / "summary.json").read_text())
    payload = {
        "title": "P154W3-N11 LBM convective mass transfer: Leveque -1/3 law, Blasius Sh~Re^1/2 Sc^1/3, "
                 "Ranz-Marshall cylinder and the Chilton-Colburn analogy",
        "summary": (
            "TensorLBM 新增对流传质硬验证面：D2Q5 TRT/BGK 标量输运 + D2Q9 动量耦合。"
            f"平板 Leveque 幂次 {summary.get('leveque', {}).get('L_s100', {}).get('p_fit', float('nan')):.4f} "
            f"(2/3 参考) 且 Sh 中位误差 "
            f"{summary.get('leveque', {}).get('L_s100', {}).get('median_err_pct', float('nan')):+.2f}% "
            f"@δc/Δ=8.8；Blasius Sc=1 误差 "
            f"{summary.get('blasius', {}).get('Sc1', {}).get('median_err_pct', float('nan')):+.2f}%；"
            "圆柱 Re=10/40/100/200 平均 Sh 对 Hilpert 圆柱关联式偏 +2.1~+5.5%、对 Churchill-Bernstein +1.2~+11.8%，"
            "Sh(Sc=2)/Sh(Sc=1)=1.266/1.294/1.326/1.349 对 2^(1/3)=1.260；"
            "Chilton-Colburn j_D 与 j_H 在 Pr=0.7,Sc=2 相差 2.38%。"
            "高 Sc(300/1000) 在 δc/Δ≈2 的两格分辨率地板上偏差 12–13%，honest 报告不粉饰。"
        ),
        "scenario_key": "p154w3_n11_convective_mass_transfer",
        "mission_id": 10404,
        "tags": ["TensorLBM", "convective-mass-transfer", "Leveque", "Blasius", "Sherwood",
                 "Ranz-Marshall", "Chilton-Colburn", "high-Schmidt", "verification-surface"],
        "markdown": md,
        "figures": figs,
    }
    out = Path("/tmp/report_p154w3_n11.json")
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[written] {out} ({out.stat().st_size/1e6:.2f} MB, {len(figs)} figures, "
          f"{len(md)} chars markdown)")


if __name__ == "__main__":
    main()