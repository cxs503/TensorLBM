#!/usr/bin/env python3
"""Build /tmp/report.json for mission 16895 (report discipline: 6 elements + figures)."""
import base64
import json
import os

REPO = "/root/tp-agent/work/m16895"
FIGS = [
    ("fig1_gate_utilisation.png",
     "Gate utilisation |metric|/threshold for every recorded MCMP stratified-Poiseuille cell; red line = gate trips."),
    ("fig2_switch_point.png",
     "Operating characteristic: every gate switches from PASS to FAIL exactly at its declared threshold."),
    ("fig3_mutant_kill.png",
     "Mutant kill matrix: probes that distinguish each degenerate gate variant (0 = mutant survives)."),
]
figures = []
for name, cap in FIGS:
    p = os.path.join(REPO, "figures", "gate_oc", name)
    with open(p, "rb") as fh:
        figures.append({"name": name, "caption": cap,
                        "png_base64": base64.b64encode(fh.read()).decode()})

with open(os.path.join(REPO, "docs", "GATE_OC_MCMP_STRATIFIED.md"), encoding="utf-8") as fh:
    markdown = fh.read()

report = {
    "title": "MCMP 分层 Poiseuille 验收判据操作特性审计：6 判据可失败性、阈值三方三角化、92/92 判决机器复算（锚点 FEAT-LBM-VERIFY-BENCH）",
    "summary": (
        "纯后处理审计（无 LBM 重跑）：建立通用判据机器 gate_oc.py（WITHHELD 语义/阈值三角化/"
        "开关点/变体证伪），对 MCMP 分层 Poiseuille 的 6 条验收判据实测——5 条在真实记录上真的返回过 "
        "FAIL（非恒真），92/92 条记录判决由度量值机器复算一致（0 分歧），18/18 退化变体被杀，"
        "判据切换点=声明阈值（1e-3 内）；发现两处记录面缺陷：定常闸门 6/6 全 WITHHELD（记录只存布尔、"
        "不可复算）、r4 判据集缺分层闸门。伪电流判据利用率 13.24–69.97 倍，量化加固该锚点 honest-FAIL。"
    ),
    "scenario_key": "tensorlbm.verification.gate_oc",
    "mission_id": 16895,
    "tags": ["TensorLBM", "MCMP", "verification", "acceptance-gate", "falsification",
             "FEAT-LBM-VERIFY-BENCH"],
    "markdown": markdown,
    "figures": figures,
}

with open("/tmp/report.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False)
print("wrote /tmp/report.json", os.path.getsize("/tmp/report.json"), "bytes,", len(figures), "figures")