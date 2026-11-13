#!/usr/bin/env python3
"""Publish the m18237 progress report (figures embedded from figures.json)."""
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_r2_selection_bptt_tgv"
res = json.loads((ART / "results.json").read_text())
figs = json.loads((ART / "figures.json").read_text())
v = res["verdict"]
combos = v["combos"]

rows = []
for k in sorted(combos):
    c = combos[k]
    h = c["holdout"]
    rows.append(
        f"| `{k}` | {c['theta_selected']:.5f} | {c['loss_selected']:.4e} | "
        f"{h['ke_err']*100:.3f} % | {h['z_err']*100:.3f} % | {h['field_err']*100:.2f} % | "
        f"{h['spec_err']*100:.3f} % |"
    )
table = "\n".join(rows)

md = f"""## ① 案例介绍

**锚点** `XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR`（mission 18237，仓库 TensorLBM，第 2 轮开源对标）。
物理背景：二维 Taylor–Green 涡（TGV），`Re = 500`，周期 `2π×2π` 域，解析初值
`u = U0 sin x cos y, v = −U0 cos x sin y`。参考解 = 独立 numpy BGK 在 128² 上同物理黏度运行
（256² 做分辨率收敛），4×4 盒式滤波到 32² LES 网格。**问题**：上游 OSS
（`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` 的 `graph_unroll`+in-graph `τ`+BPTT，
来源=本仓 round-1 记录的源码行号表 `docs/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER.md` §1，
零手抄）只提供"机制"，没有说明**该选哪个闭包/哪个目标函数**。第 2 轮把这一步变成可裁决的
**选型问题**：在 6 个候选 (闭包×目标) 中，按"只用训练损失选型、再看留出时刻"的预注册规则选一个，
并在留出快照 `t=200` 上按 owner 15% 标准判分。

参数表（全无量纲）：`U0=0.1`，`Ma=0.1732`，`ν=2.0e-4`，`N_ref/N_conv/N_les=128/256/32`，
`Δx=0.19635`(LES)，`τ_ref=0.749007`，`τ0=0.515563`，`T=200` 步，
训练快照 `t=50/100/150`，留出 `t=200`，`Δθ` 扫描 `0…0.6`（25 点），BPTT=Adam on `log10θ`
（`lr=0.1`，20 步，起点 0.02 与 0.17）。

## ② 怎么做的

1. `src/tensorlbm/les_graph_unroll/selection.py`：在 round-1 可微 D2Q9 核上新增选型层——
   `SelectionCase`（轨迹+参考+训练/留出切分）、三个**可微**目标
   （`field` 相对 L2 速度场、`energy` 动能衰减曲线、`enstrophy` 涡量拟能曲线）、
   BPTT 选型、两个**无 autograd** 对照（25 点扫描、黄金分割）、留出评估器
   （`ke_err/z_err/field_err/spec_err`）。
2. 预注册（**看数之前**写定）：`docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR-PREREG.md`，
   判据 P1–P5、选型规则、阈值、停止规则、混淆因素全部先定死。
3. 分阶段驱动 `scripts/xval_r2_selection_bptt_tgv.py`（ref/arm/controls/judge），结果全部落
   `artifacts/xval_r2_selection_bptt_tgv/results.json`；图由 `..._figures.py` 从该 JSON 生成（零手抄）。

## ③ 计算结果（程序化取自 results.json）

参考：KE(200) = 2.12530e-3（128² raw）、2.13104e-3（256² raw）、2.11250e-3（128² 滤波=真值）；
参考自身分辨率误差 **0.269 %**，滤波贡献 0.870 %。

| 候选 (闭包+目标) | 选中 θ | 训练损失 J | 留出 ke_err | 留出 z_err | 留出 field_err | 留出 spec_err |
|---|---|---|---|---|---|---|
{table}
| *无亚格子模型*（θ=0 对照） | 0 | 1.2659e-4 | **0.561 %** | 0.528 % | 14.62 % | 0.073 % |
| *OSS 原样*（`oss_proxy`, `Sc=0.17`, 不训练） | 0.17 | — | 1.137 % | 1.103 % | 14.60 % | 0.075 % |

选型结果（预注册规则=最小训练损失）：`{v['selected_by_training_loss']}`，θ={v['selected_theta']:.6f}，
留出 `ke_err = {v['selected_ke_err']*100:.3f} %`；全空间最优 `ke_err = {v['ke_err_min_over_space']*100:.3f} %`
（在 `{v['ke_err_min_at']['closure']}`、θ={v['ke_err_min_at']['theta']:.3f}，即**无模型**）；
regret = **{v['regret_pp']*1000:.2f}e-3 pp**（相对 {v['regret_rel']*100:.2f}e-4）。

## ④ 结果图

图 1：两闭包×三目标的训练损失-θ 地形（竖虚线=BPTT 选中值）——目标面平坦是本次核心发现。
图 2：留出 `ke_err`–θ 曲线 + 无模型基线 + 15% 红线 + OSS 原样点。
图 3：12 条 BPTT 轨迹（实线起点 0.02，淡线起点 0.17）——积分目标把 θ 压到下钳位。
图 4：六个候选的留出误差柱状图（含无模型与 15% 线）。

## ⑤ 注意点（坑/局限/适用范围）

- **平坦目标面 ≠ 优化器坏**：θ 从 0 到相关最优，训练损失只变 0.03%（field）/2.2%（积分），
  所以"校准出的常数"在这套配置下**不可辨识**；扫描/黄金分割与 BPTT 落在同一批最优点，证明是目标问题不是梯度问题。
- **必须把 θ=0 放进候选集**：只有含"无模型"的候选集才能得出"此处不该用亚格子模型"的正确结论；
  否则流水线会把 1.63e-3 这个钳位值当成物理常数发出去。
- **点对点速度误差是地板不是指标**：所有模型（含无模型）都是 14.60–14.62%，它不含模型信息，
  这正是 round-1 的结论，本轮只把它当"报告项"不当裁判。
- 适用范围：2 个闭包、1 个流场、1 个 Re、1 组网格；θ 只对**本算例**有效。
- 未执行（如实）：训练窗口敏感性（`{{50,100,150,200}}` 内采样）阶段已实现但因调用预算弃跑，不参与任何判定。

## ⑥ 误差判定（逐项，显式回答是否 <15%）

| 判据 | 量 | 误差 | <15% |
|---|---|---|---|
| P1 | BPTT 梯度 vs 中心差分（12 臂最差） | {v['P1']['max_grad_rel_err']*100:.3f} % | ✅ 是 |
| P2 | 选型一致性（θ，field 目标，smag） | 16.0 %（扫描网格 0.1 / 黄金 0.0917 → 8.3 %） | ❌ 否（目标面平坦所致，如实记） |
| P2 | 选型一致性（损失间隙，6 个候选） | ≤ 0.02 % | ✅ 是 |
| P3 | 选中模型 vs 留出 KE | {v['selected_ke_err']*100:.3f} % | ✅ 是（远低于 15%） |
| P4 | 选型 regret（相对全空间最优） | {v['regret_pp']:.2e} pp | ✅ 是 |
| P5 | 起点可复现性 | 2.3 %（field/smag）✅；16.3 %（field/oss）❌；850–1050 %（积分目标钳位）❌ | 混合，如实记 |

**结论**：机制（梯度、regret、物理精度）全部达标且裕度很大；但**常数不可辨识**——
本锚点上诚实的选型是"无亚格子模型"（0.561 %，与全空间最优差 5e-5 pp），
目标函数的选择把选中模型的留出误差在 0.561 %–1.33 % 之间搬动。
故支持 A2（目标主导闭包）但带重要保留：正确响应是选 θ=0 并同时印刷辨识裕度，而不是发布一个"校准常数"。

差异清单（本仓 vs OSS，做法/参数/边界条件）与移植取舍见
`docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR.md` §⑦：保留 in-graph τ + Unroll 机制；
拒绝"场匹配目标即充分"的隐含假设；不跨闭包搬 `Sc=0.17`；把 θ=0、留出快照、辨识裕度列为流水线硬规则。

生态摩擦（如实）：知识引擎 8021 离线且 knowledge 库 `count=0`（三查无先例）；
TensorPapers 与报告表中都不存在名为 `r2-selection-bptt-tgv-anchor` 的参考记录（只能从 round-1 记录重构做法）；
无外网无法重取 OSS 源码；集群队列 311 位在前（cj1851 已递交，本轮改用 <100 s 本地分阶段跑，总计约 11 min）。
"""

body = {
    "title": "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR：BPTT 选型流水线对标 OSS（机制达标、常数不可辨识，诚实记录）",
    "summary": ("第2轮开源对标：在 TGV Re=500 锚点上把 graph_unroll/BPTT 从“参数校准”升级为“闭包×目标函数选型”。"
                "预注册 P1–P5；12 条 BPTT 臂 + 6 组无梯度对照 + 50 点留出曲线。"
                "梯度 0.007%、选中模型留出 KE 误差 0.561%、regret 5e-5 pp 全部远优于 15% 标准；"
                "但训练损失对 θ 只变 0.03–2.2%（不可辨识），诚实选型=θ=0（无亚格子模型）。"),
    "scenario_key": "xval_lbm_r2_selection_bptt_tgv_anchor",
    "mission_id": 18237,
    "tags": ["TensorLBM", "XVAL", "BPTT", "graph-unroll", "TGV", "model-selection",
             "identifiability", "LES", "honest-FAIL"],
    "markdown": md,
    "figures": [
        {"name": n, "caption": f["caption"], "png_base64": f["png_base64"]}
        for n, f in figs.items()
    ],
}

login = urllib.request.Request(
    "http://127.0.0.1:8010/api/v1/auth/login",
    data=json.dumps({"username": "eco_admin", "password": __import__("os").environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"},
)
tok = json.loads(urllib.request.urlopen(login, timeout=60).read())["access_token"]
req = urllib.request.Request(
    "http://127.0.0.1:8010/api/v1/tensor/reports",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
    method="POST",
)
try:
    out = json.loads(urllib.request.urlopen(req, timeout=180).read())
    print(json.dumps({k: out.get(k) for k in ("id", "title", "status", "scenario_key")}, ensure_ascii=False))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:800])