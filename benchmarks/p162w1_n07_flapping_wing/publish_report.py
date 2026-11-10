#!/usr/bin/env python
"""Publish the P162W1-N07 progress report to the TensorPlatform report API.

Reads ``artifacts_p162n07/results.json`` + the report markdown + the figures and POSTs a
paper-grade report (six elements, figures embedded as base64 PNG).

    python publish_report.py --outdir artifacts_p162n07 --figdir figures
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

FIG_CAPTIONS = {
    "fig1_theodorsen.png": "Theodorsen 函数 C(k)=F+iG：左为两条独立特殊函数路径（Hankel / modified Bessel）"
                           "结果重合，右为 Argand 图，标注 k→0 的 C=1 与 k→∞ 的 C=1/2 精确极限。",
    "fig2_garrick.png": "Garrick 闭式推进效率 η(k) 与无黏推力 C_T(k)（含法向力倾斜项）：纯沉浮 η 由 1 单调降到 0.5，"
                        "纯俯仰低 k 无推力（准稳态），羽化复合运动 η 随 k 升。",
    "fig3_lbm_vs_theory.png": "左：小振幅 9 点扫描 LBM 增量 ΔC_T 与 Garrick 闭式对比（推力被黏性阻力淹没）；"
                              "右：推力主导区 h0/c=0.15 的 LBM 效率 vs Garrick 闭式。",
    "fig4_vorticity.png": "LBM 涡量场（h0/c=0.15、k=1 纯沉浮、弦长 32 格）四个相位快照：尾缘脱涡形成反卡门涡街，"
                          "是推力的直接流场证据。",
    "fig5_energy_gci.png": "左：各算例能量账相对残差（红虚线=1e−3 判据）；右：C6 三网格 C_T 误差随弦长加密的收敛。",
}


def sh(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {out.stderr[:400]}")
    return out.stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "artifacts_p162n07"))
    ap.add_argument("--figdir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--report", default=os.path.join(HERE, "..", "..", "docs",
                                                     "P162W1_N07_FLAPPING_WING.md"))
    ap.add_argument("--mission-id", default="9076")
    args = ap.parse_args(argv)

    with open(os.path.join(args.outdir, "results.json")) as fh:
        res = json.load(fh)
    crit = res["criteria"]

    token = sh(["curl", "-s", "-X", "POST", f"{API}/auth/login",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"username": USER, "password": PASS})])
    try:
        tok = json.loads(token)["access_token"]
    except Exception:
        tok = json.loads(token).get("token", "")

    figures = []
    for name, cap in FIG_CAPTIONS.items():
        path = os.path.join(args.figdir, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            figures.append({
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(fh.read()).decode(),
            })

    with open(args.report) as fh:
        md = fh.read()

    verdicts = {k: v.get("verdict") for k, v in crit.items()
                if isinstance(v, dict) and "verdict" in v}
    summary = (
        "TensorLBM 建成扑翼推进振荡翼验证面：Theodorsen C(k) 双路径互校 1e-16、Garrick 推力/功率/效率闭式"
        "（含法向力倾斜项）相量-时域互校 1e-15、k→0 准稳态极限 0.32%、推力反转线与 GCI 工具；"
        "D2Q9+IBM 求解器位级可复现并通过负控三件套。LBM 侧小振幅判据诚实 FAIL："
        "h0/c=0.02 时无黏推力比黏性阻力小两个量级。判定：" + json.dumps(verdicts, ensure_ascii=False)
    )

    payload = {
        "title": "P162W1-N07 TensorLBM 扑翼推进振荡翼验证面：Theodorsen/Garrick 解析锚 + 推力系数/推进效率/能量账 + 三网格 GCI",
        "summary": summary,
        "scenario_key": "flapping_propulsion_oscillating_wing",
        "mission_id": args.mission_id,
        "tags": ["TensorLBM", "flapping-foil", "Theodorsen", "Garrick", "IBM", "GCI",
                 "P162W1-N07", "verification"],
        "markdown": md,
        "figures": figures,
    }
    path = "/tmp/report_m9076.json"
    with open(path, "w") as fh:
        json.dump(payload, fh)
    out = sh(["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
              "-H", f"Authorization: Bearer {tok}",
              "-H", "Content-Type: application/json",
              "-d", f"@{path}"])
    print(out[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())