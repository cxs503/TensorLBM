"""Publish the IN67-N01 report to the Tensor platform (credentials from env)."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ART = pathlib.Path("artifacts/microbubble_dr_in67_n01")
DOC = pathlib.Path("docs/microbubble_dr/IN67_N01.md")
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")

FIGS = [
    ("fig1_void_field.png", "图1 解析气泡体积分数 α 场与流向速度（可视化算例 d+=10, 64×65, 4000 步）"),
    ("fig5_mechanism_split.png", "图5 三项机理分离：滑移/黏度对比/湍流调制（占单相壁面剪切份额）"),
    ("fig3_drag_vs_size.png", "图3 单气泡 d+ 扫描：DR 对 d+ 与实测 α（含驱动饱和混杂）"),
    ("fig4_void_path.png", "图4 负控：α→0 时 DR→0（零气泡 DR=0.00%）"),
    ("fig6_reynolds_utau.png", "图6 雷诺应力剖面与四个 u_τ 估计（C2 自洽性检查）"),
]


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    login = post("/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })
    token = login.get("access_token") or login.get("token") or login.get("data", {}).get("access_token")
    assert token, f"no token in login response keys={list(login)}"
    report = json.loads((ART / "report.json").read_text())
    crit = report["criteria"]
    figures = [
        {"name": name, "caption": cap,
         "png_base64": base64.b64encode((ART / name).read_bytes()).decode()}
        for name, cap in FIGS
    ]
    payload = {
        "title": "IN67-N01 Shan-Chen 两相微气泡减阻：三项机理分离 + 负控（C2/C3 honest-FAIL）",
        "summary": (
            "在 TensorLBM 新建 microbubble_dr 模块：Shan-Chen 微气泡平面槽道（质量流匹配驱动、"
            "相依赖弛豫时间、动量平衡壁面剪切）、闭合壁面剪切预算 tau_w=A-B 与滑移/黏度/湍流三分解。"
            f"判据：C1 质量守恒 max drift {crit['C1_mass_conservation']['max_mass_drift']:.1e}（PASS）、"
            f"C4 零气泡负控 DR={crit['C4_negative_control_zero_gas']['DR_at_zero_gas']*100:.3f}%（PASS）、"
            f"C5 尺寸扫描记录（PASS）；C2 u_τ 三口径最差 {crit['C2_momentum_consistency']['worst_u_tau_spread']*100:.1f}%、"
            f"C3 三项闭合最差 {crit['C3_mechanism_closure']['worst_relative_closure_error']*100:.0f}% → honest FAIL，"
            "原因可复现：α>1% 时流向非均匀使 A-B 投影误差增大、大气泡工况驱动带宽饱和无法匹配流量。"
            "物理结论（本设置下）：气泡使壁面剪切上升（DR<0），滑移项 ~1e-3 量级，机理主要在气泡诱导雷诺应力与黏度对比；"
            "DNS/实验定量对标显式 WITHHELD。"
        ),
        "scenario_key": "microbubble_dr_shanchen",
        "mission_id": 7149,
        "tags": ["TensorLBM", "IN67-N01", "shan-chen", "two-phase", "microbubble",
                 "drag-reduction", "mechanism-split", "honest-fail"],
        "markdown": DOC.read_text(),
        "figures": figures,
    }
    resp = post("/tensor/reports", payload, token)
    print(json.dumps({"ok": True, "keys": list(resp)[:6],
                      "id": resp.get("id") or resp.get("data", {}).get("id")})[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())