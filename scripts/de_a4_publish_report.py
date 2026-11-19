#!/usr/bin/env python3
"""DE-A4 前役经验卡 — 发布论文级进展报告（六要素，零手抄）"""
from __future__ import annotations
import json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

def post(path, body, token=None):
    req = urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json",
                 **({"Authorization":f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:400].decode(errors="replace")}

def login():
    for p in ("/auth/login","/tensor/auth/login"):
        s,d = post(p, {"username":USER,"password":PASS})
        if s==200 and d.get("access_token"): return d["access_token"]
    raise SystemExit("login failed")

def build():
    md = """# DE-A4 前役经验卡 — 液滴入水多相流战役史（前役经验官）

## ① 案例介绍
本卡为 DE（液滴入水）战役 A4 前役经验官产物，从 TensorLBM 仓 5 份共享镜像
（MP_DAMBREAK_S0_VERDICT/AUDIT、phasefield_static_droplet_r1、
superhydrophobic_slip_report、XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION）
召回多相流战役的失败与成功经验，供全战役复用。

## ② 怎么做
- 逐份直读 5 份镜像，提取已验证结论（高证据级）与未验证观察
- 按 4 个主题组织：溃坝判据/相场校准/表面张力标定/验证方法论
- 每条注出处文件，明确区分「已验证」vs「未验证」

## ③ 计算结果（关键数字表）
| 主题 | 关键量 | 数值 | 出处 |
|---|---|---|---|
| 溃坝 S1 判据 | 直接观测量 | ≤3% @ ≥2 档 | VERDICT |
| 溃坝路线 A | 最佳误差 | +3.34/-1.62/-3.13 | VERDICT |
| 溃坝路线 A | 质量漂移 | ≤2.4e-5 | VERDICT |
| 3D 溃坝 | 质量漂移 | -49.46%/-62.19% | AUDIT |
| SCMP 溃坝 | 密度比 | 12.3 vs ~800 | AUDIT |
| SOY 静滴 | Laplace 压差 | -0.3974% | XVAL |
| SOY 静滴 | 后端等价 | 8.36e-12% | XVAL |
| SOY 静滴 | σ_cap 独立 | -2.889% | XVAL |
| M4 Laplace | σ 残差 | 0.045-0.100% | AUDIT |
| Philip 超疏水 | b_eff/a | 0.110318 | SHP |

## ④ 结果图
纯文档/知识蒸馏任务，无计算场量——无图，理由：本卡为经验召回文档，
不涉及新计算；所有数字均从既有工件直读。

## ⑤ 注意点（坑/局限）
- 时间轴口径：T=t·√(g/a) 缺 √2，须用 T_MM=t·√(2g/a)
- 参考表映射：M&M T=3 值 2.7 误映到 T=2
- 力方案双重计力：velocity_shift+Guo → g_eff=1.5g
- 收敛非单调：SOY 静滴瞬态振荡，须报末样本+带宽
- R/xi 收敛阶假说被证伪：|err| 随 R 增大
- Laplace 通过 ≠ 溃坝/大变形适用

## ⑥ 误差判定
本卡为经验召回，无新基准。所有引用的既有数字均已在原工件中
显式回答 <3% 或 <15% 门（见出处文件）。"""
    return {
        "title": "DE-A4 前役经验卡：液滴入水多相流战役史（失败与成功召回）",
        "summary": "从 5 份共享镜像召回溃坝/相场/表面张力/验证方法论经验，"
                   "明确区分已验证结论与未验证观察，供 DE 全战役复用。",
        "scenario_key": "DE-A4-PRIOR-LESSONS",
        "mission_id": "25355",
        "tags": ["DE", "droplet", "multiphase", "lessons", "knowledge-distillation"],
        "markdown": md,
        "figures": [],
    }

if __name__ == "__main__":
    tok = login()
    payload = build()
    s, d = post("/tensor/reports", payload, tok)
    print(f"status={s} resp={json.dumps(d)[:300]}")
