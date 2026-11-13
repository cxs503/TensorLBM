# 交差审计 m17592 — XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION（TensorLBM）

**审计单**: m17631（board-audit） · **被审**: m17592（succeeded） · **日期**: 2026-11-13
**工作树**: /root/tp-agent/work/m17592 · **分支**: agent/m17592

## 1. 验收三件套核验

| 件 | 状态 | 证据 |
|---|---|---|
| run_artifacts | ✅ 真存在（6 文件） | `artifacts/xval_block_amr_stream_correction/{xval_results.json 6347B, fig1_conservation.png 97KB, fig2_approach_comparison.png 140KB, figures.json}` + `artifacts/amr_reflux_ladder/{ladder.json, ladder.png}`；时间戳 2026-11-13 07:06-07:07 与作业窗口一致；均为 untracked（未 commit） |
| benchmark ref vs got | ⚠️ 数值在案，裁判行=0 行 | 见 §2；`tensor_benchmark_cases` 裁判行 ****确认 0 行**（精确+ILIKE 变体均 0）——容差为任务自报，非独立注册基准 |
| repro_cmd | ⚠️ 脚本在案，未复跑 | `benchmarks/xval_block_amr_stream_correction/run_xval.py` 存在；审计员未重新执行（时间预算），可执行性 **UNVERIFIED** |

## 2. ref vs got 逐项（零手抄，取自 xval_results.json / XVAL_REPORT.md）

| 指标 | ref | got | 容差 | 判定 | 来源 |
|---|---|---|---|---|---|
| 1-D 质量漂移（reflux on，120 步） | 0（理论） | 3.691e-16 | 1e-10 | ✅ | 本次新跑（ladder.json） |
| 1-D 质量漂移（reflux off，负对照） | ≫1e-10 | 1.429e-02 | — | ✅ 负对照成立 | 本次新跑 |
| 1-D 阶梯 8 级 | 全绿 | 8/8 pass | 全绿 | ✅ | 本次新跑 |
| 1-D 网格收敛 N=48/96/192 | — | L1=2.50e-2/1.34e-2/3.13e-3，阶 0.90/2.10 | 渐近二阶 | ✅ | 本次新跑 |
| 2-D 界面通量不平衡（相对） | 0（理论） | 2.055e-05 | 1e-3 | ✅ | **二手**：docs/amr-reflux-maturation.md（40000 粗步） |
| 2-D 复合质量漂移（reflux on） | 0（理论） | 6.473e-05 | — | ✅ | **二手**：同上 |
| 2-D Ghia 中心线最大偏差 | 0（稳态） | 0.5269（52.69% U_lid） | 15% | ❌ **HONEST FAIL** | 报告 6.1/7 如实声明：40000 步未达稳态（同步步数均匀网格 0.1751），归因瞬态+单侧 F-H 过耗散，非 AMR 界面缺陷 |

**报告六段**：齐全——案例介绍(§1)、方法(§2)、结果(§4)、图(fig1/fig2)、注意点(§6.1 Honest FAIL 三项)、误差判定(§7 逐项 <15% 显式回答)。报告 232 行 8 节，无缺段。

## 3. 判级：L3

- **L2 达成**：真代码（F-H 重缩放 + Berger-Colella reflux，`src/tensorlbm/amr_reflux.py`、`src/tensorlbm/amr_cavity2d.py`，报告 §1.1/§6.2 列明；文件内容审计员未独立复读）。
- **L3**：能力=库模块+专用 benchmark 驱动，**未接入主生产求解器**（报告自述 NumPy 单核 benchmark；torch 后端可用但串行）。
- **L4 部分达成**：1-D 守恒子案例=L4 级（完整算例+误差分析+可复现+负对照+双网格收敛）；2-D 守恒 PASS 但为文档数据（二手）；2-D 精度 HONEST FAIL。
- **L5 不达成**：交付物是交叉验证基准，无新科研级物理。
- **综合 L3**：守恒验证强，但 2-D 精度不过 15% 线、2-D 守恒证据为文档来源，不能给干净 L4。

## 4. 未执行检查（unknown，不冒充 passed）

1. repro_cmd 复跑
2. `tests/test_amr_reflux_ladder.py` 复跑（报告称 44 passed）
3. 实现文件内容独立复读

## 5. 夸大/口径问题（非阻断）

- 表 4.2 将文档来源的 2-D 数字呈现为「TensorLBM 实测」，虽 §4.2 标题已披露来源，仍建议表格内逐行标注 source。
- 工件全部 untracked：若分支未推上 GitHub，工件将丢失——建议随分支 commit。

## 6. 裁定

**confirmed（消账）@ L3**。锚点核心主张（界面流修正守恒）由 1-D 新跑在机器精度内验证（3.691e-16 ≪ 1e-10）且负对照证明修正必要性；唯一失败指标（2-D Ghia 精度）已如实声明且超出 stream-correction 锚点范围。2-D 稳态精度留作后续成熟化任务。

**落表**：`verdict_m17631.sql`已执行成功（INSERT 0 1）：level=3, verdict_by=board-audit, gate_state=pending，待 maturity_gate 机器复核。
