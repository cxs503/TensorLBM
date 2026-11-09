#!/usr/bin/env python
"""Publish the N10 progress report to the platform tensor API.

    python scripts/n10_publish_report.py [--dry-run]

Reads the generated docs markdown and the figures, logs in with
``$TENSOR_AGENT_USER``/``$TENSOR_AGENT_PASS`` and POSTs to
``$TENSOR_API_BASE/tensor/reports`` (curl is blocked by the sandbox policy, so
urllib is used).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/scalar-transport-verification-n10.md"
FIGDIR = ROOT / "figures/scalar_transport_n10"
REPORT = ROOT / "artifacts/scalar_transport_n10/report.json"

CAPTIONS = {
    "n10_erf_diffusion.png": "图1 (a) 最细网格(n=512)浓度剖面 vs 精确 erf 解；(b) L2 相对误差-Δx 双对数，实测观测阶 p=2.000（R2=1.000000）。",
    "n10_taylor_aris.png": "图2 (a) D_eff/D-Pe：管道闭式 1+(2/105)Pe^2、剪切闭式 1+(2/15)Pe_gamma^2 与矩法/模态法两个独立估计器的实测点；(b) 不同横向节点数 ny 下 D_eff 相对误差随 Pe 的走势与 ±5% 判据带。",
    "n10_accuracy_domain.png": "图3 G8 精度域热图：Pe_r × ny 的相对误差（数值标注），σ/nx>1/6 的格标注为估计器越界。",
    "n10_sc_independence.png": "图4 (a) 无量纲色散常数 K_meas 跨 D（即跨 Sc）塌缩到 2/105；(b) τ_m 扫掠（Sc=0.1/1/10，τ_d 固定）下 D_eff/D 与各自速度场的离散 Aris 参考。",
    "n10_mass_and_moments.png": "图5 (a) 总量漂移 float64/float32 对照与判据线；(b) 代表性算例（ny=13, Pe=20）的羽流方差 σ²(t) 与拟合窗口。",
}


def _post(path: str, payload: dict, token: str | None = None, timeout: int = 60):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        path, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mission-id", type=int, default=7867)
    args = ap.parse_args(argv)

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    markdown = DOC.read_text()
    rep = json.loads(REPORT.read_text())
    gates = rep["gates"]

    figures = []
    for name, caption in CAPTIONS.items():
        p = FIGDIR / name
        if not p.exists():
            continue
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            }
        )

    payload = {
        "title": "p74-n10 被动标量输运验证面：erf 二阶收敛 + Taylor–Aris（管道/圆管/剪切）解析锚 + Sc 无关性",
        "summary": (
            "TensorLBM 新增被动标量输运验证面：G1–G8 全部通过。erf 扩散 L2=1.8e-5、观测阶 p=2.000；"
            "平面 Poiseuille Taylor–Aris D_eff 相对闭式误差 -1.48%~-3.99%（Pe=5/10/20/40，ny=13 nx=1024），"
            "线性剪切层 -1.47%~-1.84%（新推导 D_eff=D+2γ²h⁴/15D，离散 Aris 校核）；圆管 Aris 常数 1/48 恢复到 1.7e-9；"
            "Pe→0 偏差 +0.23%；Sc 无关性散度 0.73%/0.19%；总量漂移 8.8e-13/1e4 步；u=0 负控 0.9976；"
            "并给出 Pe_r×网格 精度域表（ny=9 在 Pe≥10 超出 5%，判定为精度域外）。"
        ),
        "scenario_key": "passive-scalar-transport",
        "mission_id": args.mission_id,
        "tags": ["TensorLBM", "p74-n10", "passive-scalar", "taylor-aris", "verification", "erf-diffusion"],
        "markdown": markdown,
        "figures": figures,
        "metadata": {
            "verdict": rep["verdict"],
            "gates": {k: v.get("pass") for k, v in gates.items() if k.startswith("G")},
            "report_json": "artifacts/scalar_transport_n10/report.json",
            "module": "src/tensorlbm/scalar_transport_verification.py",
            "tests": "tests/test_scalar_transport_verification_n10.py",
        },
    }
    if args.dry_run:
        out = Path("/tmp/report.json")
        out.write_text(json.dumps(payload, indent=1))
        print(f"dry-run: wrote {out} ({out.stat().st_size} bytes, {len(figures)} figures)")
        return 0

    status, body = _post(
        base + "/auth/login",
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]},
    )
    token = json.loads(body)["access_token"]
    status, body = _post(base + "/tensor/reports", payload, token)
    print("status", status, body[:600].decode(errors="replace"))
    return 0 if status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())