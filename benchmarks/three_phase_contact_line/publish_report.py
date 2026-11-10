"""Publish the P262W1-N02 report to the platform report API.

Numbers come from ``runs/p262w1_n02/{results,criteria}.json`` via
:func:`benchmarks.three_phase_contact_line.make_doc.build`; figures are the PNGs
of the same campaign, inlined as base64.

Usage:
    python benchmarks/three_phase_contact_line/publish_report.py \
        --campaign runs/p262w1_n02 [--dry-run] [--mission-id 9874]
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
sys.path.insert(0, str(REPO / "benchmarks" / "three_phase_contact_line"))

from make_doc import build as build_doc  # noqa: E402

CAPTIONS = {
    "contact_angle_relaxation.png":
        "接触角松弛轨迹：四种壁面耦合下的坐滴平衡接触角 θ(t) 与合成不确定度；"
        "点线为 Young 律预测（由热力学积分润湿张力与 Laplace 张力给出，本模型不闭合，故多数缺失）。",
    "cox_voinov_fit.png":
        "Cox–Voinov 检验：θ_d³−θ_Y³ 对 9Ca 的数据点与最小二乘直线，R² 与斜率 K=9ln(L/ℓ_m) 直接标注。",
    "young_equation.png":
        "Young 律作为预测：arccos(Δγ/γ_lg)（圆点）与坐滴实测圆拟合角（方块，误差棒）随壁面耦合变化。",
    "density_fields.png":
        "密度场云图：静态坐滴（左）与受驱坐滴（右），颜色为 ρ，壁面位于下边界两行。",
}


def _summary(res: dict, crit: dict) -> str:
    gl = res["gamma_lg"]
    fit = res["cox_voinov"]["fit"]
    conv = res["convergence"]
    mass = res["mass_conservation"]
    return (
        "TensorLBM 新增冰–水–气三相接触线验证面（contact_line/three_phase.py + 战役脚本 + 18 条 pytest）。"
        f"实测：Laplace 水/气张力 γ_lg={gl['gamma']:.5f}（与库内已验收值 {gl.get('documented_laplace'):.5f} 差 "
        f"{100 * gl.get('relative_difference_vs_documented', float('nan')):+.3f}%）；"
        f"全域总质量漂移 {100 * mass['max_drift']:.4f}%（判据 ≤0.1%）；"
        f"LBM 状态方程 p=c_s²(ρ−(G/2)ψ²) 经共存检验（0.3%）。"
        f"负结果（留档并显式判读）：伪势壁面的润湿张力有两条独立测量路线均被证伪——"
        f"压力张量积分（度量壁面法向压力，偏差≈0.43γ_lg 且对壁面耦合不敏感）与热力学积分"
        f"（内部锚点检验偏大 5 倍，暴露 SC 模型变分不一致），因此 Young 律 F1 判为 withheld/FAIL；"
        f"Cox–Voinov R²={fit.get('r2', float('nan')):.3f}、接触线速度收敛阶 p="
        f"{conv.get('richardson_order', float('nan')):.3f} 按实测如实报告。"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default="runs/p262w1_n02")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mission-id", type=int, default=9874)
    args = ap.parse_args(argv)
    cdir = REPO / args.campaign
    res = json.loads((cdir / "results.json").read_text())
    res.setdefault("outdir", args.campaign)
    crit = json.loads((cdir / "criteria.json").read_text())

    md = build_doc(res, crit)
    figures = []
    for name, caption in CAPTIONS.items():
        p = cdir / name
        if not p.exists():
            continue
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(p.read_bytes()).decode(),
        })
    payload = {
        "title": "P262W1-N02 冰–水–气三相接触线：伪势 LBM 的 Young 律/Cox–Voinov 验证（含两条证伪路线）",
        "summary": _summary(res, crit),
        "scenario_key": "tensorlbm.contact_line.three_phase",
        "mission_id": int(args.mission_id),
        "tags": ["TensorLBM", "contact-line", "three-phase", "ice-water-air", "Young",
                 "Cox-Voinov", "Shan-Chen", "P262W1-N02", "honest-negative-result"],
        "markdown": md,
        "figures": figures,
    }
    out = cdir / "report_payload.json"
    out.write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "figures"} | {"n_figures": len(figures)},
        indent=2,
    ))
    print(f"markdown {len(md)} chars, {len(figures)} figures -> {out}")
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
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            print("published:", r.status, r.read()[:300].decode(errors="replace"))
    except urllib.error.HTTPError as exc:  # pragma: no cover
        print("publish failed:", exc.code, exc.read()[:500].decode(errors="replace"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())