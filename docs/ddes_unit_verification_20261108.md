# DDES/SAS 混合湍流模块验证报告（G21 缺口填补）

> **Mission**: m6583 — 自主工时·TensorLBM DDES 模块单元测试验证
> **日期**: 2026-11-08
> **仓**: TensorLBM
> **文件**: `src/tensorlbm/ddes.py`（370 行，G21 孤儿模块）
> **测试**: `tests/test_ddes_unit.py`（22 项，全部通过）

---

## 1. 案例介绍

### 物理背景

Delayed Detached-Eddy Simulation (DDES, Spalart et al. 2006) 是一种混合 RANS-LES 湍流模型，
在近壁区使用 RANS（避免 LES 的极高网格需求），在分离区自动切换到 LES 模式。
Scale-Adaptive Simulation (SAS, Menter & Egorov 2010) 是相关方法，通过 von Kármán 长度尺度
自动适应从 RANS 到 LES 的行为。

在 LBM 框架中，湍流模型通过有效松弛时间 τ_eff 进入：

    ν_eff = ν_molecular + ν_t
    τ_eff = 0.5 + 3·ν_eff

### 模块功能

`ddes.py` 提供以下公共函数：
- `ddes_shielding_function` — DDES 屏蔽函数 f_d ∈ [0,1]
- `ddes_length_scale` — DDES 混合长度尺度 l_DDES
- `ddes_eddy_viscosity` — 亚格子涡粘 ν_t 计算
- `sas_source_term` — SAS 额外源项 Q_SAS
- `apply_ddes_collision` — 带 DDES 的 BGK 碰撞步
- `run_ddes_diagnostics` — 诊断信息汇总

### 参数表

| 参数 | 符号 | 默认值 | 物理意义 |
|------|------|--------|----------|
| C_DES | C_DES | 0.65 | DES 标定常数 |
| C_S | C_S | 0.1 | Smagorinsky 常数 |
| κ | kappa | 0.41 | von Kármán 常数 |
| C_SAS | C_SAS | 2.0 | SAS 模型常数 |
| ν_t_max | — | 0.3 | 涡粘上限（防 τ<0.5） |

---

## 2. 方法

### 测试策略

采用**数学不变量验证**而非全量模拟——每个函数针对其数学性质设计测试：

| 测试组 | 验证内容 | 参考依据 |
|--------|----------|----------|
| T1 | f_d ∈ [0,1] | tanh 值域 |
| T2 | f_d 极限行为 | r_d → ∞ ⇒ f_d → 0; r_d → 0 ⇒ f_d → 1 |
| T3 | l_DDES 边界 | l_LES ≤ l_DDES ≤ l_RANS |
| T4 | ν_t ≥ 0 | 物理约束 |
| T5 | ν_t ≤ ν_t_max | 代码 clamp |
| T6 | LES 模式 f_d ≡ 1, ν_t = (C_S·Δ)²·\|S\| | Smagorinsky 公式 |
| T7 | τ_eff > 0.5 | LBM 稳定性条件 |
| T8 | Q_SAS ≥ 0 | SAS 源项定义 |
| T9-T11 | 微分算子性质 | 数值分析 |
| T12 | DDESResult 分数 ∈ [0,1] | 概率解释 |
| T13 | 碰撞步质量守恒 | BGK 守恒律 |
| T14 | 诊断返回有效结果 | 接口契约 |

### 工具链

- Python 3.11 + PyTorch (CPU, float64)
- pytest 8.x
- 无 GPU 需求，无外部依赖

---

## 3. 计算结果

### 测试执行

```
22 passed, 1 warning in 6.30s
```

全部 22 项测试通过，无失败、无跳过。

### 关键数值验证

| 测试 | 预期 | 实测 | 状态 |
|------|------|------|------|
| f_d 随机输入范围 | [0, 1] | [0, 1] ✅ | 通过 |
| f_d BL 极限 (r_d≈6) | < 0.1 | 0.0000 | 通过 |
| f_d 远场极限 (r_d≈6e-9) | > 0.99 | 1.0000 | 通过 |
| l_DDES (f_d=0.5, l_RANS=2, l_LES=0.65) | 1.325 | 1.325 | 通过 |
| LES ν_t 公式 | (C_S·Δ)²·\|S\| | 匹配 atol=1e-10 | 通过 |
| τ_eff 最小值 | > 0.5 | 0.50003 | 通过 |
| Q_SAS 非负 | ≥ 0 | ≥ 0 | 通过 |
| 质量守恒相对误差 | < 1e-10 | 0.0 | 通过 |

---

## 4. 结果图

本任务为纯单元测试验证，无数值模拟场量，故无云图/曲线生成。
测试覆盖的数学关系已通过断言验证（见上表）。

---

## 5. 注意点

### 已验证
- ✅ DDES 屏蔽函数数学性质正确
- ✅ 长度尺度混合公式正确
- ✅ 涡粘非负且有上限
- ✅ LES 模式退化为标准 Smagorinsky
- ✅ 碰撞步质量守恒
- ✅ SAS 源项非负

### 局限
- ⚠️ 仅验证 2D 函数（ddes.py 本身仅实现 2D）
- ⚠️ 未做全量模拟验证（需 3D 通道流/圆柱绕流 benchmark）
- ⚠️ `apply_ddes_collision` 仅测试了单步碰撞，未测试多步时间积分稳定性
- ⚠️ SAS 模式仅验证源项非负，未验证与 ω 方程耦合

### 适用范围
- 本验证确认模块**数学实现正确**，可用于后续 benchmark 开发
- 工程应用需配合 3D 扩展（G7/G8 缺口）和全量模拟验证

---

## 6. 误差判定

本任务为**数学不变量验证**，所有测试使用精确解析参考值或严格不等式约束：

| 验证项 | 基准 | 误差 | 是否 < 3% |
|--------|------|------|-----------|
| f_d 范围 | [0,1] | 0 | N/A（精确） |
| f_d 极限 | 0/1 | < 1e-10 | ✅ |
| l_DDES 公式 | 解析值 | < 1e-12 | ✅ |
| LES ν_t 公式 | 解析值 | < 1e-10 | ✅ |
| 质量守恒 | 0 | 0 | ✅ |

所有可量化验证项误差远低于 3% 阈值。

---

## 7. G21 缺口状态更新

| 孤儿模块 | 验证状态 | 备注 |
|----------|----------|------|
| ddes.py | ✅ 已验证（22 测试） | 本任务完成 |
| des_turbulence.py | ❌ 待验证 | 重定向到 turbulence/des_turbulence.py |
| rans_wall_bc.py | ❌ 待验证 | 壁函数边界条件 |
| d3q27_collide.py | ❌ 待验证 | D3Q27 碰撞 |
| hybrid_solver.py | ❌ 待验证 | 混合求解器 |
| prism_solver.py | ❌ 待验证 | 棱柱求解器 |

---

**结论**: DDES/SAS 模块数学实现正确，22 项单元测试全部通过。
模块可作为后续 3D 扩展和全量 benchmark 验证的基础。
