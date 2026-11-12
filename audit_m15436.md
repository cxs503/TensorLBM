# m15436 CASE-RAYLEIGH-TAYLOR 交差审计报告

**审计任务**: m15716（交差审计 | 任务榜）
**审计对象**: m15436（CASE-RAYLEIGH-TAYLOR · 算例池 · Rayleigh-Taylor 失稳 · TensorLBM）
**审计日期**: 2026-11-12
**审计人**: board-audit

---

## 取证清单

| 项目 | 状态 | 详情 |
|---|---|---|
| impl_files | ✓ 存在 | `src/tensorlbm/rayleigh_taylor.py` (502 行)，`benchmarks/case_pool/rayleigh_taylor/run_case.py` |
| entry | ✓ 存在 | `run_rayleigh_taylor(cfg)`, `rt_step(f, g, cfg, phi)` |
| run_artifacts | ✓ 存在 | 6 算例 case.json + history.csv, summary.json, 3 张 PNG 图, tip_velocity.csv |
| tests | ✓ 通过 | `tests/test_rayleigh_taylor.py`: 11/11 passed (11.26s) |
| integration_callers | ✓ 存在 | run_case.py, test_rayleigh_taylor.py |
| benchmark ref vs got | ⚠ 部分 | 线性增长率 6/6 通过(<10%)；终端速度 0/6 失败(+160%~+220000%) |
| repro_cmd | ✓ 可执行 | `python3 benchmarks/case_pool/rayleigh_taylor/run_case.py` |
| external_anchor | ✓ 存在 | ref/ref_lock.json + ref/sources/ (Rayleigh 1883, Taylor 1950, Chandrasekhar 1961, Goncharov 2002) |
| 入 tensor_benchmark_cases | ✗ 缺失 | 0 rows for anchor_slug='CASE-RAYLEIGH-TAYLOR' |
| 平台 L4 裁定 | ✗ 未通过 | CLAIMED_NOT_PROVEN (no tensorbench_results row) |

## 判级：L3（系统集成）

**理由**：
- L0 ✓ 文档成果：README.md 六段齐全，参数表/对照表/收敛性/失败模式/工件清单完整
- L1 ✓ 数学/单元验证：analytic_growth_rate / viscous_growth_rate 公式正确，11 项单元测试通过
- L2 ✓ 独立功能实现：502 行真实代码（D2Q9 BGK + Boussinesq 浮力 + 对流相场），输入→计算→输出
- L3 ✓ 系统集成：接入 TensorLBM 求解器，有 benchmark 驱动和测试调用
- L4 ✗ 基准验证：benchmark 未入 tensor_benchmark_cases 表（0 rows）；终端速度门全败；平台 L4 裁定 CLAIMED_NOT_PROVEN
- L5 ✗ 科研工程：已知失败模式（相场无界）未修复

## 关键发现

1. **线性增长率诚实通过**：6 算例全部在 ±10% 内，系统偏低（-2.7% ~ -9.9%），与"弥散界面削弱有效浮力"一致，λ 加密后误差单调改善（-7.9% → -2.7%）。

2. **终端速度系统性失败**：u_bubble_sim 为参考值的 1.6-2.5 倍（误差 +160% ~ +220000%），u_spike 同理。u_s/u_b 实测 ≈1.0-1.12，理论应为 1.41(A=1/3)/1.73(A=1/2)——模型丢失 Atwood 非对称性。

3. **根因已定位**：相场无界（基准 |φ|_max=2.27，A=1/2 达 341）→ 浮力 F=-Agφ 被放大 → 速度高估 → Ma>1 → NaN 发散。README 已诚实记录，未粉饰。

4. **benchmark 未注册**：tensor_benchmark_cases 表中无 CASE-RAYLEIGH-TAYLOR 条目，所有基准对比为自声明，非平台测量。

## 裁定

- **成熟度**: L3
- **消账建议**: confirmed（L3 达成，L4 缺口已登记为生态缺口）
- **返修项**: 相场锐化（Cahn-Hilliard/自由能项）→ 修复后重跑终端速度门
