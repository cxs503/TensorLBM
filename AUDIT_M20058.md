# 交差审计报告：m20058 CASE-RT-INSTABILITY-ATWOOD

> 审计任务 m20187 · 对象 m20058 · 分支 agent/m20058 · TensorLBM
> 审计日期 2026-11-14 · 审计员 board-audit

## 审计范围

对 m20058 交付的 Rayleigh-Taylor 不稳定性（Atwood 数 A=1/2）算例进行交差核验，逐项验证：
1. 收三件套（run_artifacts / benchmark ref vs got / repro_cmd）是否真实存在
2. 报告六段是否齐全
3. got 数值是否在容差内（基准判据：±15%）
4. 成熟度判级（L0-L5）

## 取证结果

### ① run_artifacts — ✅ 真实存在

| 工件 | 路径 | 大小/内容 |
|---|---|---|
| summary.json | artifacts/runs/summary.json | 42642 bytes, 10 cases |
| acceptance_table.md | artifacts/runs/acceptance_table.md | 10 cases, all PASS |
| case.json (ref) | artifacts/runs/A050_l064_g2e-4/case.json | σ_sim=2.7725e-03, σ_visc=3.0855e-03 |
| history.csv | artifacts/runs/A050_l064_g2e-4/history.csv | 2526 steps |
| instant_rate.csv | artifacts/runs/A050_l064_g2e-4/instant_rate.csv | 瞬时增长率曲线 |
| ref_lock.json | ref/ref_lock.json | 闭式参考值 + Crossref 书目哈希 |
| figures | figures/fig1-4*.png | 4 张图 (166-197 KB) |

### ② benchmark ref vs got — ✅ 数值在容差内

主判据：`|σ_sim/σ_visc − 1| ≤ 15%`

| 案例 | σ_sim | σ_visc (ref) | 误差 | 判定 |
|---|---|---|---|---|
| A050_l064_g2e-4 (参考例) | 2.7725e-03 | 3.0855e-03 | -10.14% | ✔ PASS |
| A050_l064_g2e-4_legacy (bgk) | 2.9367e-03 | 3.0855e-03 | -4.82% | ✔ PASS |
| A050_l064_g1e-4 (半重力) | 1.8768e-03 | 2.1679e-03 | -13.43% | ✔ PASS |
| A050_l096_g3e-4 (1.5x 网格) | 2.7640e-03 | 3.1119e-03 | -11.18% | ✔ PASS |
| A050_l128_g4e-4 (2x 网格) | 2.7169e-03 | 3.1213e-03 | -12.96% | ✔ PASS |

全部 10 例均通过 ±15% 门限。

### ③ repro_cmd — ✅ 可执行

6 条命令 documented in REPORT.md §2：
```
ref/ref_lock.py → run_case.py → run_case.py --aggregate → make_figures.py --snapshots → make_report.py
```

### ④ 报告六段 — ✅ 齐全

| 段 | 内容 | 状态 |
|---|---|---|
| ① 问题与 ref 出处 | 物理背景、Tryggvason 谱系、闭式公式 | ✅ |
| ② 方法与实现 | Allen-Cahn 相场、四拟合窗、瞬时增长率 | ✅ |
| ③ 参数表 | 无量纲数完整（A, g, ν, kδ, Ma 等） | ✅ |
| ④ 对照表 | ref vs got 逐项误差 | ✅ |
| ⑤ 收敛性与敏感性 | 网格阶梯、ν/g/界面模型支臂 | ✅ |
| ⑥ 失败模式与生态摩擦 | F1-F5 + E1-E4 | ✅ |

## 判级

**L4 — 基准验证**

理由：
- L0 ✅ 文档成果（REPORT.md 18081 bytes）
- L1 ✅ 数学/单元验证（test_rayleigh_taylor.py 10+ tests）
- L2 ✅ 独立功能实现（run_case.py 22462 bytes 驱动 tensorlbm.rayleigh_taylor）
- L3 ✅ 系统集成（接入 case_pool 驱动，调用求解器内核）
- L4 ✅ 完整算例（10 例）+ 误差分析（收敛阶梯/敏感性/失败模式）+ 可复现（repro_cmd + artifacts）
- L5 ❌ 未合并主线；外部锚为闭式理论（非实验数据）

## 发现

| 编号 | 严重度 | 内容 |
|---|---|---|
| F1 | info | 系统性 ~10% 低偏差（浮力-界面耦合），在门内，诚实披露 |
| F2 | info | 非线性尖峰终速在本配置下不可测（Boussinesq 对称性 + 盒高限制） |
| F3 | low | tensor_benchmark_cases 无此行（平台 referee 缺失，生态摩擦 E2） |
| F4 | low | 二维势流常数 C 约定不一致（C=1 vs C=3，生态摩擦 E3） |

## 裁定

**confirmed — 消账**

L4 确认。算例完整、误差分析充分、可复现。平台 referee 缺失为生态摩擦，不降能力级。
