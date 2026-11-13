#!/usr/bin/env python3
"""m18207 报告发布：六要素论文级进展报告 → POST $TENSOR_API_BASE/tensor/reports。"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text())
A6 = V["A5_A6_benchmark"]["A6_replica"]
rows = V["A5_A6_benchmark"]["delivered_criteria_summary"]

md = f"""# m18207 交差核验报告：m18114 · XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE

**结论先行**：判 **L2（confirmed 路径，消账）**，disposition = approve with notes。
交付在「平面双界面」族上真实达标（σ 误差 10.15%/6.09%/2.84%，全在 owner <15% 带内；
18/18 判据本地逐位复现），代码/入口/工件/测试齐备；但**无下游调用方**（L3 不达）、
**基准覆盖只到平面族**（登记行 86/87 未覆盖，行 87 仍 failed）→ L4 不达。落表 verdict id=715。

## ①案例介绍
对称双阱自由能二组分 LBM 平衡压力闭环：`f=(a/2)φ²+(b/4)φ⁴+(κ/2)|∇φ|²`（a=−A,b=B），
压力张量 `P_ab=dδ_ab+κ∂_aφ∂_bφ`，`d=p0−κφ∇²φ−(κ/2)|∇φ|²`。解析锚 `p_coex=A²/(4B)`、
`σ=(2√2/3)√(κA³)/B`。几何：周期域两条平直界面，`nx=128, ny=16, A=B=1`，
`ξ/h∈{{1.5,2,3}}`（κ=Aξ²/2）；LBM 臂 `steps=2000, tau_f=1.0, tau_g=0.7, Γ=0.5, ρ0=1`。
判据出处：交付 `predeclared.json`（事前声明）＋平台登记行 `tensor_benchmark_cases` id=85/86/87；
外部锚 Pooley/Furtado/Carr, PRE 78, 056709 (2008)。

## ②怎么做的
交差核验（非重做），全只读/复算：A1 交付物存在性+sha256 快照；A2 是否进生态 main；
A3 实跑 pytest；A4 实跑 repro_cmd 并逐键比对工件；A5 从 PG 取三条登记基准行逐行核覆盖；
A6 用**交付模块本体**复算登记行 85 配置（A=B=0.1, κ=0.45 → ξ=3, 64×8）；
A7 集成调用方 grep；A8 自述 vs 工件。落表前用平台守护 `maturity_gate.py::judge` 干跑。
工具链 python3+torch(CPU,float64)+pytest+psql+matplotlib。

## ③计算结果
| 项 | 结果 |
|---|---|
| pytest | 7 passed rc=0 |
| repro_cmd | rc=0, wall {V['A4_repro']['wall_s']} s，VERDICT PASS 18/18 |
| 复跑 vs 交付 | checks/rows 逐键相等，JSON sha256 一致（bb130d75…） |

| ξ/h | σ 解析 | σ 积分 | σ 误差 | P_nn 均匀性 | 共存压力误差 | LBM 总压偏差 | \|u\|max |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r['interface_cells']} | {r['sigma_analytic']:.6f} | {r['sigma_integral']:.6f} | "
    f"{r['sigma_relative_error']*100:.2f}% | {r['p_nn_max_deviation_relative']*100:.2f}% | "
    f"{r['coexistence_relative_error']:.2e} | {r['lbm_final_p_nn_total_deviation_relative']*100:.2f}% | "
    f"{r['lbm_final_max_velocity']:.2e} |" for r in rows) + f"""

基准裁判：登记行 85（平带 σ）**部分覆盖**——本单用交付模块复算同配置得 σ=0.194314（−2.84%，带内），
但登记 got 2.01% 取自 population 二阶矩，口径不同；登记行 86（droplet Laplace）与 87（dropR36 稳定性）
**未覆盖**，行 87 仍 failed。A6 复算另实测：iso9/5pt 两格式在平界面退化为恒等
（grad_x 最大差 {A6['scheme_iso9_vs_5pt_grad_x_max_delta']:.2e}，σ 差 {A6['scheme_iso9_vs_5pt_sigma_delta']:.1f}）。

## ④结果图
fig1 σ 相对误差 vs ξ/h（含 owner 15% 带 + O(h²) 参考线）——10.15%→6.09%→2.84%，二阶趋势成立；
fig2 三分辨率 P_nn(x)/p_coex 与 P_tt(x)/p_coex 剖面——P_tt 界面下陷、P_nn 近似均匀，界面残差随分辨率单调下降。

## ⑤注意点
① 双梯度格式在 y 不变场上恒等，JSON 的 iso9/5pt 两行是重复计数，非独立证据；
② σ 口径（解析场张量）≠ 登记行口径（population 二阶矩），基准行需注明；
③ LBM 臂每步重设 g，量的是伪流/机械平衡（静态两相测试），适用边界须写明（交付已前置声明 g 方程缺 −M∇μ 通量）；
④ 交付未进生态 main（git cat-file ∃ → ABSENT，远端无 m18114 分支），合并前下游不可消费；
⑤ 自述瑕疵：文档串声明两张 png 实际一张；result.checks 把 PNG 105928 B 写成 JSON 大小（JSON 实为 {V['A1_deliverables']['artifact_json']['bytes']} B）。

## ⑥误差判定（逐项，显式回答是否 <3%）
| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| σ (ξ/h=1.5) | 1.0 | 10.15% | 否（带内） |
| σ (ξ/h=2.0) | 1.333333 | 6.09% | 否（带内） |
| σ (ξ/h=3.0) | 2.0 | 2.84% | 是 |
| p_coex（Maxwell 数值） | 0.25 | ≤1.74e−8 | 是 |
| P_nn 均匀性 (ξ/h=1.5) | 0 | 23.09%（阈值 25%） | 否（贴边带内） |
| 登记行 85 复算 σ | 0.2 | 2.84%（登记 got 2.01%） | 是 |

**诚实结论**：σ 三项中仅 1 项满足 3%，另两项在 15% 带内但**不满足 3%**，不得宣称全部 <3%；
L2 成立、L3/L4 不达，机器干跑（claim2→gate2；claim3/4→gate2，原因 integration_callers grep 无引用）与人工判定一致。

## 附：核验清单与未做项
未做：登记行 86/87 复算（交付模块无 droplet/Laplace 与 dropR36 能力）；GPU/CUDA 路径；tau_f/Γ 参数扫描。
工件：`docs/board_audit/m18207/{{AUDIT.md, verify_output.json, verdict.sql, run_audit.py, make_figs.py, evidence/, figs/}}`。
"""

figs = []
for name, cap in [("fig1_sigma_convergence.png", "σ 相对误差 vs 界面分辨率 ξ/h（绿带=owner <15% 精度标准，虚线=过 ξ/h=1.5 的 O(h²) 参考）"),
                  ("fig2_pnn_profiles.png", "三分辨率下 P_nn(x)/p_coex 与 P_tt(x)/p_coex（解析平衡场自由能压力张量）；界面残差 23.09%→14.19%→6.70%")]:
    p = HERE / "figs" / name
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

payload = {
    "title": "m18207 交差核验：m18114 XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE 判 L2（无调用方/基准覆盖不全）",
    "summary": f"独立复核 m18114（head c6ed86d2）：pytest 7 passed；repro_cmd rc=0 wall {V['A4_repro']['wall_s']}s "
               f"VERDICT PASS 18/18 且工件逐键复现；σ 误差 10.15/6.09/2.84%（<15% 带内，仅 1 项 <3%）；"
               "无下游调用方（L3 不达）、登记行 86/87 未覆盖且 87 仍 failed（L4 不达）→ 判 L2；"
               "落表 tensor_maturity_verdicts id=715，gate 干跑 claim2→gate2 一致。",
    "scenario_key": "board-audit",
    "mission_id": 18207,
    "tags": ["board-audit", "TensorLBM", "XVAL", "free-energy-LBM", "maturity-verdict", "L2"],
    "markdown": md,
    "figures": figs,
}
Path("/tmp/report.json").write_text(json.dumps(payload))

base = os.environ["TENSOR_API_BASE"]
user, pwd = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
tok = json.loads(subprocess.run(
    ["curl", "-s", "-X", "POST", f"{base}/auth/login", "-H", "Content-Type: application/json",
     "-d", json.dumps({"username": user, "password": pwd})], capture_output=True, text=True).stdout)
token = tok.get("access_token") or tok.get("token")
print("token keys:", list(tok)[:6])
req = urllib.request.Request(f"{base}/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    print("report POST", r.status, r.read().decode()[:400])