"""Publish the m15894 audit report to the platform /tensor/reports endpoint."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = pathlib.Path("/root/tp-agent/work/m15894")


def post(path, payload, token=None):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]

md = (HERE / "docs/audit-m15894-case-mcmp-contact-angle.md").read_text()
fig = HERE / "audit/audit_m15894/fig_m15894_contact_angle_audit.png"
rep = {
    "title": "m15894 交差核验：m15470 CASE-MCMP-CONTACT-ANGLE 判级 L2（honest FAIL 采信，L4 否证）",
    "summary": (
        "交差核验 m15470（壁面接触角 Young 定律，TensorLBM/SCMP）：三件套逐项核实——run_artifacts 真实"
        "（41 文件/34 算例/sweep 91819B），ref-vs-got 从 JSON 独立复算 0/6（5° 与 10° 双阈值）与 0/3，"
        "审计侧复跑单例逐位一致（Δ=0.0°），pytest 8 passed 复现。裁定 L2：L3 无集成调用方（库改动仅 11 行 "
        "docstring），L4 否证（基准失败、ref 自指、tensor_benchmark_cases 本锚 0 行、tensorbench_results 0 行、"
        "41 个 run artifact 不在 main——PR #1825 只合了 2 个文件）。"
    ),
    "scenario_key": "board_audit",
    "mission_id": 15894,
    "tags": ["audit", "TensorLBM", "contact-angle", "Young-law", "SCMP", "maturity-gate", "honest-FAIL"],
    "markdown": md,
    "figures": [{
        "name": "fig_m15894_contact_angle_audit.png",
        "caption": ("左：Young 设定角 ref vs 实测 got（两网格），0/6 通过，110° 不可表达；"
                    "右：实测壁面律 θ(f_w) 与库内线性 cos 假设对比（拟合斜率 4.2 倍、中性壁 0.512 vs 0.172）。"
                    "数字全部由 contact_angle_sweep.json 现算。"),
        "png_base64": base64.b64encode(fig.read_bytes()).decode(),
    }],
}
res = post("/tensor/reports", rep, tok)
print(json.dumps(res, ensure_ascii=False)[:400])