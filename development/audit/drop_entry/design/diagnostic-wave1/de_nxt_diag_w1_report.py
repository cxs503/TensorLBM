#!/usr/bin/env python3
"""Publish paper-grade progress report for DE-NXT-DIAG-W1 (diagnostic measurement design)."""
import base64, json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "drop_entry" / "BB" / "NEXT" / "TRACK_DIAG_W1.md"
markdown = TRACK.read_text()

payload = {
    "title": "DE-NXT-DIAG-W1 诊断测量专线·处方赛马设计：p_cav(t) 轨迹探针 + 三相动量收支探针（只测量不修复）",
    "summary": (
        "本线为 DE 诊断测量专线设计官（编排脑，不跑仿真）。按 owner 令设计下一波处方赛马（≤2 臂），"
        "方向=诊断测量（只测量不修复），产测量卡喂全部设计官。已排除路线（网格纯收敛/参数微调/u_limit/"
        "ABB 整体移除/rho_g 简单缩放/空穴压亏）不重开。"
        "设计两臂：①CAV_PRESSURE——气穴中心 p_cav(t) 压力-时间轨迹探针，与 Rayleigh-Plesset 球穴坍缩"
        "解析解逐点对比（物理出处=Rayleigh 1917/Plesset），只加压力场采样探针不改内核，cells=8 满窗三元组；"
        "②MOMENTUM_BUDGET——液滴-气-液三相动量收支探针，控制体动量通量积分逐项列收支并检验闭合性"
        "（物理出处=三相动量守恒），只加动量通量积分探针不改内核。"
        "每臂预登记：预测区间+证伪判据。交付 TRACK_DIAG_W1.md（20 行，格式机读通过，goal_met=no）。"
    ),
    "scenario_key": "de_diag_measurement_design",
    "mission_id": 26722,
    "tags": ["DE", "DIAG", "diagnostic-measurement", "cavitation", "Rayleigh-Plesset",
             "momentum-budget", "prescription-racing", "TensorLBM"],
    "markdown": markdown,
    "figures": [],
}
# 纯设计/编排任务，无仿真运行，故无图（报告纪律：纯文档任务无图须写明理由）
payload["figures_note"] = "本单为诊断测量设计官（编排脑），未跑仿真，无场量云图/比较曲线；交付为处方赛马设计卡。"

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    method="POST",
)
login.add_header("Content-Type", "application/json")
with urllib.request.urlopen(login, timeout=60) as h:
    tok = json.loads(h.read().decode())
token = tok.get("access_token") or tok.get("data", {}).get("access_token")
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(), method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("Authorization", "Bearer " + token)
with urllib.request.urlopen(req, timeout=180) as h:
    print("REPORT_OK:", json.loads(h.read().decode()))
