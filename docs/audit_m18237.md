# 交差审计 m18237 — XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR (TensorLBM)

**审计单**: m18354 (board-audit, owner 2026-11-11 任务榜令「结榜要回来交差」)
**被审单**: m18237 (succeeded, commit `f95e1d63`, 分支 agent/m18237)
**裁定**: **L4**（基准验证：完整算例 + 误差分析 + 可复现）→ 落 `tensor_maturity_verdicts`，待 maturity_gate 机器复核。

## ① 取证三件套核验

| 件 | 判据 | 结果 |
|---|---|---|
| run_artifacts | 文件真实存在且已提交 | ✅ 7 件全在 `artifacts/xval_r2_selection_bptt_tgv/`（results.json 87943B / reference.npz 745941B / fig1–4.png / figures.json 444475B），`git show --stat f95e1d63` 确认全部入库 |
| ref vs got 数值 | 报告数字 vs 原始 results.json 程序化重算 | ✅ 全部吻合（见 ③）；发现 2 处文档笔误（不影响判据，见 ⑤） |
| repro_cmd 可执行 | 脚本可编译 + 测试可复跑 | ✅ 3 个脚本 `py_compile` 通过；`pytest tests/test_les_graph_unroll_selection.py -q` → **8 passed (9.39s)**（审计方独立复跑）；全量 driver 复跑 **not_run**（~11 min，超出审计预算，如实记） |

## ② 基准裁判（唯一裁判）

任务令指定查询：
```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
WHERE anchor_slug='XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR';
```
**结果：0 行。** 本锚点在 `tensor_benchmark_cases` 无注册基准例（全表 102 行中仅 round-1 锚
`XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER` 有 2 例）。→ **证据链缺口**：本单的事实裁判是
预注册判据 P1–P5（`docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR-PREREG.md` §4，看数前写定）
+ owner 15% 标准（feature_ledger #439 acceptance）。缺口已写入 verdict evidence，
建议 anchor_feeder 为本锚点补注册基准例（否则下轮交差仍无表内裁判）。

## ③ got 数值逐项核（原始 JSON 重算，零手抄）

| 判据 | 报告值 | results.json 重算 | 容差 | 判定 |
|---|---|---|---|---|
| P1 梯度 vs 中心差分（12 臂最差） | 0.074 % | `max_grad_rel_err=7.352e-4` = 0.074 % | <15 % | ✅ |
| P2 选型一致性（损失间隙，6 候选） | ≤0.02 % | `loss_gap_rel` 最大 1.89e-4 = 0.019 % | <15 % | ✅ |
| P2 选型一致性（θ，smag+field） | 16.0 % | 0.08405 vs sweep 0.1 → 16.0 % | <15 % | ❌（目标面平坦，honest） |
| P3 选中模型 vs 留出 KE | 0.561 % | `selected_ke_err=5.6098e-3` = 0.561 % | <15 % | ✅ |
| P4 选型 regret | 5.34e-5 pp | `regret_pp=5.341e-5`；全空间最优 5.6092e-3（θ=0 无模型） | ≤15 pp | ✅ |
| P5 起点可复现（smag+field） | 2.3 % | `spread_rel=0.0234` | ≤15 % | ✅ |
| P5 起点可复现（oss+field） | 16.3 % | `spread_rel=0.1630` | ≤15 % | ❌（honest） |
| P5 积分目标（4 组合） | 850–1050 % | `spread_rel` 8.48–10.51 | ≤15 % | ❌（钳位，honest） |

参考解自身：KE(200) 128² raw 2.12530e-3 / 256² raw 2.13104e-3 / 128² 滤波 2.11250e-3；
分辨率误差 0.269 %、滤波贡献 0.870 %（与 JSON `ref` 块逐位一致）。
选型重算：6 组合最小训练损失 = `oss_proxy|enstrophy`（J=1.1276e-4, θ=0.001632），
留出 ke_err 0.561 %，与全空间最优（无模型 θ=0）差 5.34e-5 pp —— 与报告一致。

**主判据（owner 15% 标准，feature_ledger #439）：P3 = 0.561 % < 15 %，达标。**
P2(θ)/P5 的 FAIL 是研究结论（常数不可辨识）而非交付缺陷，且均如实申报。

## ④ 报告六段（tensor_reports id=17147, published）

① 案例介绍 ✅（TGV Re=500 参数表全无量纲）② 怎么做的 ✅ ③ 计算结果 ✅（程序化取自
results.json）④ 结果图 ✅（4 图内嵌，figures.json 含 sha256）⑤ 注意点 ✅（含弃跑阶段
如实声明）⑥ 误差判定 ✅（逐项显式回答是否 <15%）。

## ⑤ 判级与差异

**L4 依据**：完整算例（独立 numpy BGK 参考 128²/256² + 32² LES，12 BPTT 臂 + 6 无
autograd 对照 + 留出快照）+ 误差分析（P1–P5 逐条 + 收敛性 0.269 %）+ 可复现
（driver + 8 测试 + 工件 sha256）。非 L5：选型层是研究性模块，未成为生产求解路径。
round-1 同族锚点（m17843）判 L3，本轮新增完整选型研究面，升一级有据。

**差异/保留（不改变级别，gate 复核时注意）**：
1. `tensor_benchmark_cases` 对本锚点 0 行 —— 表内裁判缺席，事实裁判=预注册 P1–P5；
2. 文档笔误 2 处：P1「0.007 %」应为 0.074 %（7.35e-4）；P3「5.6098e-4」应为 5.6098e-3
   —— 百分比判据列均无误，纯排版量级笔误；
3. 全量 driver 复跑 not_run（11 min 预算外）；
4. 生态摩擦 F1–F5（知识引擎离线、OSS 未重取、集群排队 311 位等）被审单已如实记录。

## ⑥ 落表

`verdict_m18237.sql`（本目录）已执行：
`INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18237 XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR', 18354, 4, <JSON>, 'board-audit')`
evidence 键齐 8 项（impl_files/entry/run_artifacts/tests/integration_callers/benchmark/
repro_cmd/external_anchor），值=真实指针。
