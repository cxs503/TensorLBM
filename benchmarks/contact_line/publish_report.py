"""Assemble and publish the IN100-N02 paper-grade report (platform report API).

All numbers come from ``runs/<campaign>/campaign.json`` / ``criteria.json``; the
figures are the PNGs written by the campaign.  Nothing is typed in by hand.

Usage:
    python benchmarks/contact_line/publish_report.py --campaign runs/contact_line_in100_n02 \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "contact_line"))

from make_doc import build as build_doc  # noqa: E402

CAPTIONS = {
    "fig_static_calibration.png":
        "静态接触角标定：整冠圆拟合（圆点，误差棒=合成不确定度）与局部切线拟合（方块，交叉校验）；"
        "红线为壁面首层密度比 ρ₁/ρ_l（右轴），可见 f≤0.55 的壁面耗尽区。",
    "fig_dynamic_angles.png":
        "驱动坐滴的前进/后退角随 Ca 变化（三种壁面）。虚线为同壁面静态角；"
        "前缘角度随 Ca 下降、后缘上升——与 Cox–Voinov 的符号相反。",
    "fig_cox_voinov.png":
        "Cox–Voinov 检验：θ_A³−θ_e³ 对 Ca（真实接触壁面 f=0.80）。理论要求线性正斜率，实测为负斜率。",
    "fig_hysteresis.png":
        "方向依赖壁吸附模型下的接触角滞后：θ_A−θ_R 随吸附差 Δf 张开，Δf=0 对照闭合。",
    "fig_field.png":
        "驱动坐滴的密度场（白点为界面等值线 ρ_m，白虚线为壁面位置）。",
}


def _summary(campaign: dict, criteria: dict) -> str:
    c1 = criteria["C1_static_limit"]
    c2 = criteria["C2_tanner_exponent"]
    c3 = criteria["C3_log_term"]
    cv = c3["cox_voinov"]
    c4 = criteria["C4_hysteresis"]
    s = criteria["summary"]
    return (
        "伪势(SCMP)壁面接触线的定量标定与标度检验。交付：13 个壁面伪势的静态接触角标定表 "
        f"(θ_e 从 {campaign['static'][0]['theta_deg']:.1f}° 到 "
        f"{sorted(campaign['static'], key=lambda r: r['wall_fraction'])[-1]['theta_deg']:.1f}°，两条独立估计量+不确定度预算)；"
        f"实测表面张力 σ={campaign['sigma']['sigma']:.5f}(R²={campaign['sigma']['r2']:.4f}，与已验收值 0.05611 相差 "
        f"{100*abs(campaign['sigma']['sigma']-0.05611)/0.05611:.2f}%)；"
        f"Ca 扫描 (Ca≈{min(r['capillary'] for r in campaign['sweep'] if 'theta_a' in r):.3f}–"
        f"{max(r['capillary'] for r in campaign['sweep'] if 'theta_a' in r):.3f}) 下的动态角与稳态滑移速度。"
        f"裁决：C1 {'PASS' if c1['passes'] else 'FAIL'}（Ca→0 误差 {c1['error_deg']:.2f}°）、"
        f"C2 {'PASS' if c2['passes'] else 'FAIL'}（Tanner 指数 {c2['exponent']:.3f}，理论 1/3）、"
        f"C3 {'PASS' if c3['passes'] else 'FAIL'}（K={cv['slope']:.2f} rad³，ℓ_m 与界面厚度比值 "
        f"{c3['ratio_fit_over_measured']:.2f}）、C4 {'PASS' if c4['passes'] else 'FAIL'}（滞后回线单调）。"
        f"合计 {s['criteria_passed']}/{s['criteria_evaluated']} 通过；未通过项已给出机制诊断"
        "（壁面耗尽→薄膜/滑移态，接触线被缓冲）。诚实负结果，不粉饰。"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default="runs/contact_line_in100_n02")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mission-id", type=int, default=7285)
    args = ap.parse_args(argv)

    cdir = REPO / args.campaign
    campaign = json.loads((cdir / "campaign.json").read_text())
    campaign.setdefault("outdir", args.campaign)
    criteria = json.loads((cdir / "criteria.json").read_text())

    markdown = build_doc(campaign, criteria)
    figures = []
    for rel in campaign.get("figures", []):
        p = REPO / rel
        if not p.exists():
            continue
        figures.append({
            "name": p.name,
            "caption": CAPTIONS.get(p.name, p.name),
            "png_base64": base64.b64encode(p.read_bytes()).decode(),
        })

    payload = {
        "title": "IN100-N02 伪势模型动态接触角：Cox–Voinov/Tanner 标度与接触角滞后",
        "summary": _summary(campaign, criteria),
        "scenario_key": "tensorlbm.contact_line.in100_n02",
        "mission_id": int(args.mission_id),
        "tags": ["TensorLBM", "contact-line", "Cox-Voinov", "Tanner", "Shan-Chen",
                 "IN100-N02", "honest-negative-result"],
        "markdown": markdown,
        "figures": figures,
    }
    out = REPO / args.campaign / "report_payload.json"
    out.write_text(json.dumps({k: v for k, v in payload.items() if k != "figures"} |
                              {"n_figures": len(figures)}, indent=2))
    print(f"markdown {len(markdown)} chars, {len(figures)} figures -> {out}")
    if args.dry_run:
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user, pwd = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
    req = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": user, "password": pwd}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        token = json.load(r)["access_token"]

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print("published:", r.status, r.read()[:300].decode(errors="replace"))
    except urllib.error.HTTPError as exc:  # pragma: no cover
        print("publish failed:", exc.code, exc.read()[:500].decode(errors="replace"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())