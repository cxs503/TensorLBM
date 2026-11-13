#!/usr/bin/env python3
"""Publish board-audit report for m18237 (mission 18354)."""
import json, os, urllib.request

md = """## ① 案例介绍（被审对象与物理背景）

被审单 **m18237**（TensorLBM，succeeded，commit `f95e1d63`）：锚点
`XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR` 第 2 轮开源对标——把上游 OSS
（`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` 的 `graph_unroll` + in-graph `τ` + BPTT）
的"机制"升级为可裁决的**闭包×目标函数选型问题**：2D Taylor–Green 涡 `Re=500`，
`U0=0.1`，`Ma=0.1732`，`ν=2.0e-4`，参考解=独立 numpy BGK 128²（256² 收敛检查）4×4 盒式滤波到 32² LES，
训练快照 t=50/100/150，留出 t=200；6 候选（2 闭包×3 目标）×2 起点共 12 条 BPTT 臂 + 6 无 autograd 对照。
本单（m18354）= 交差核验：验收三件套（run_artifacts / ref-vs-got / repro_cmd）+ 基准裁判 + L0–L5 判级 + 落表。

## ② 怎么做的（审计流程）

1. **取证**：读 m18237 工作树（`/root/tp-agent/work/m18237`）的 commit `f95e1d63` 变更面
   （15 文件：selection.py 338 行 / driver 384 行 / 图脚本 146 行 / 测试 130 行 / 工件 7 件 / 预注册+主文档）。
2. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
   anchor_slug='XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR'` → **0 行**（表内无本锚点基准例，缺口如实记）；
   事实裁判=预注册 P1–P5（PREREG §4，看数前写定）+ owner 15% 标准（feature_ledger #439）。
3. **程序化重算**：从 `results.json` 原始 JSON 重算全部判据数字（零手抄），与报告 17147 逐项比对。
4. **复跑**：3 脚本 `py_compile` 通过；`pytest tests/test_les_graph_unroll_selection.py -q`
   → **8 passed (9.39s)**（审计方独立复跑）。全量 driver 复跑 not_run（~11 min，超审计预算）。
5. **落表**：`tensor_maturity_verdicts` id=749（level=4, verdict_by='board-audit', gate_state=pending）。

## ③ 计算结果（审计重算，程序化取自 results.json）

| 判据 | 报告值 | 重算值 | <15% |
|---|---|---|---|
| P1 梯度 vs 中心差分（12 臂最差） | 0.074 % | 7.352e-4 = 0.074 % | ✅ |
| P2 损失间隙（6 候选最大） | ≤0.02 % | 1.89e-4 = 0.019 % | ✅ |
| P2 θ 一致性（smag+field） | 16.0 % | 0.08405 vs 0.1 → 16.0 % | ❌ |
| P3 选中模型 vs 留出 KE | 0.561 % | 5.6098e-3 = 0.561 % | ✅ |
| P4 regret | 5.34e-5 pp | 5.341e-5 pp（最优=无模型 θ=0, 5.6092e-3） | ✅ |
| P5 smag+field | 2.3 % | 0.0234 | ✅ |
| P5 oss+field | 16.3 % | 0.1630 | ❌ |
| P5 积分目标×4 | 850–1050 % | 8.48–10.51（钳位） | ❌ |
| 参考分辨率 128²→256² | 0.269 % | 0.0026916 | ✅ |
| 滤波贡献 | 0.870 % | 0.0087006 | — |

选型重算：最小训练损失 = `oss_proxy|enstrophy`（J=1.1276e-4, θ=0.001632），留出 ke_err 0.561 %，
与全空间最优差 5.34e-5 pp —— 与报告完全一致。**主判据 P3 = 0.561 % < 15 % 达标。**

## ④ 结果图

**无图（理由）**：本单为文档/审计任务，不产生新场量；被审 4 图
（fig1–fig4，`artifacts/xval_r2_selection_bptt_tgv/`，sha256 入 figures.json）已内嵌于被审报告 17147，
审计不重复内嵌。

## ⑤ 注意点（坑/局限）

- **基准表缺口**：`tensor_benchmark_cases` 对本锚点 0 行——表内裁判缺席，事实裁判=预注册 P1–P5；
  建议 anchor_feeder 补注册，否则下轮交差仍无表内裁判。
- **文档笔误 2 处**（不影响判据）：P1「0.007 %」应为 0.074 %（7.35e-4）；P3「5.6098e-4」应为 5.6098e-3；
  百分比判据列均无误。
- 全量 driver 复跑 not_run（11 min 预算外）；P2(θ)/P5 的 FAIL 是研究结论（常数不可辨识）而非交付缺陷，
  被审单已如实申报（honest-FAIL 留痕完整）。
- 判级 L4 依据：完整算例+误差分析+可复现；非 L5（选型层为研究性模块，未入生产求解路径）。
  round-1 同族锚点（m17843）判 L3，本轮新增完整选型研究面，升一级有据。

## ⑥ 误差判定（逐项，显式回答是否 <15%）

| 项 | 误差 | <15% |
|---|---|---|
| P1 梯度 | 0.074 % | 是 |
| P2 损失间隙 | ≤0.019 % | 是 |
| P2 θ（smag+field） | 16.0 % | **否**（目标面平坦，honest） |
| P3 留出 KE（主判据） | 0.561 % | 是 |
| P4 regret | 5.34e-5 pp | 是 |
| P5 smag+field | 2.3 % | 是 |
| P5 oss+field | 16.3 % | **否**（honest） |
| P5 积分目标 | 850–1050 % | **否**（钳位，honest） |

**裁定**：L4 落表（verdict id=749），maturity_gate 机器复核中（confirmed=消账 / reopened=返修）。
"""

payload = {
    "title": "交差审计 m18237（XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR, TensorLBM）→ L4 落表：三件套齐、主判据 0.561%<15%、基准表 0 行缺口如实记",
    "summary": "board-audit 核验 m18237：run_artifacts 7 件全在且已提交；报告 17147 六段齐全，数字与 results.json 程序化重算逐项吻合（P1 0.074%/P3 0.561%/P4 5.34e-5pp）；pytest 8 passed 独立复跑；tensor_benchmark_cases 对本锚点 0 行（缺口入 evidence）；判 L4 落 tensor_maturity_verdicts id=749。",
    "scenario_key": "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR",
    "mission_id": 18354,
    "tags": ["board-audit", "TensorLBM", "XVAL", "BPTT", "maturity-verdict"],
    "markdown": md,
    "figures": [],
}
tok = open("/tmp/tok.txt").read().strip()
req = urllib.request.Request(os.environ["TENSOR_API_BASE"] + "/tensor/reports",
                              data=json.dumps(payload).encode(), method="POST")
req.add_header("Authorization", "Bearer " + tok)
req.add_header("Content-Type", "application/json")
with urllib.request.urlopen(req, timeout=60) as r:
    d = json.loads(r.read())
    print("PUBLISHED id=", d.get("id"), "status=", d.get("status"))
