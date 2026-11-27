# 修复方案 — m8802 MCMP 两相 Poiseuille 流量误差

**日期**: 2026-11-09
**物理锚**: GAP-LBM-MCMP-POISEUILLE
**弹体**: A 臂 m8280 (QR 判定 20.2%/16.8% FAIL)

---

## 1. 问题定位

### 1.1 现象
- 两相 Poiseuille 流基准：SC MCMP "self" 模式
- H64: max_rel_err = 26.8%, u_max_ratio = 1.265
- H128: max_rel_err = 33.2%, u_max_ratio = 1.315
- 误差随网格加密**增大**（非收敛行为）→ 系统性偏差

### 1.2 机理分析（基于 Step 1 知识检索）

**根因**: `collide_sc_two_component` 的 `u_eq="self"` 模式

```python
# "self" 模式：各组分使用自己的速度
feq1 = equilibrium(rho1, ux1 + tau1 * Fx1 / rho1, ...)
feq2 = equilibrium(rho2, ux2 + tau2 * Fx2 / rho2, ...)
```

**问题链**:
1. 界面处两相共存 (ρ₁ ≈ ρ₂ ≈ 0.5)
2. 不等 τ → 两相速度不同 (u₁ ≠ u₂)
3. 无速度耦合 → 界面处速度不连续
4. 等效于界面滑移边界 → 流动阻力降低
5. 流量高估 20-33%

**对照**: Color-Gradient 方法使用单一速度场 → 0.2% 误差 (PASS)

### 1.3 排除的候选机理

| 候选 | 排除理由 |
|------|----------|
| 力项离散误差 | 标准 SC 力项 O(h²)，不会导致 20%+ 误差 |
| 界面厚度 | tanh-3 界面足够薄，不影响宏观流量 |
| 壁面边界 | bounce-back 标准实现，单相验证通过 |
| 弛豫时间组合 | τ=(1.0, 0.75) 在稳定窗口内 |

## 2. 修复方案

### 2.1 核心改动

在 `collide_sc_two_component` 中新增 `u_eq="viscous"` 模式：

```python
elif u_eq == "viscous":
    # 黏性加权速度耦合 (Martys et al. 1996 / sailfish)
    w1 = rho1 / tau1
    w2 = rho2 / tau2
    u_visc = (w1 * u1 + w2 * u2) / (w1 + w2)
    feq1 = equilibrium(rho1, u_visc + tau1 * F1 / rho1, ...)
    feq2 = equilibrium(rho2, u_visc + tau2 * F2 / rho2, ...)
```

### 2.2 理论依据

1. **速度连续性**: 两相都向 u_visc 弛豫 → 界面速度连续
2. **动量守恒**: 加权弛豫项之和 = 0（由 u_visc 定义保证）
3. **稳定性**: 与 "self" 模式相同参数范围稳定
4. **文献支持**: sailfish 采用相同方案 (tkc:oss-lbm:sailfish-shan-chen-binary)

### 2.3 最小侵入原则

- 仅修改 `src/tensorlbm/multiphase.py` 一个文件
- 新增 `u_eq="viscous"` 分支，不改动现有 "self"/"mixture" 逻辑
- 默认行为不变（`u_eq="self"` 仍为默认值）
- A/B 可回滚：改回 `u_eq="self"` 即恢复原行为

## 3. 验证计划

### 3.1 基准测试

- **几何**: 平面通道 H=64/128，周期边界 x 方向
- **物理**: τ_w=1.0, τ_g=0.75, G_12=-2.5, G_x=5e-6
- **解析解**: `_two_phase_poiseuille_analytical` (应力连续条件)

### 3.2 验收标准

| 指标 | 目标 | 判据 |
|------|------|------|
| max_rel_err (H64) | < 10% | 中心区域速度剖面 |
| max_rel_err (H128) | < 5% | 网格收敛 |
| 质量漂移 | < 1e-6/step | 总质量守恒 |
| 数值稳定 | 不发散 | 全场 finite |

### 3.3 对照实验

| 模式 | 预期结果 |
|------|----------|
| "self" (基线) | 26-33% 误差 (FAIL) |
| "mixture" | 发散 (已知) |
| "viscous" (修复) | < 10% 误差 (PASS) |

## 4. 可证伪预测

- **P1**: H64 max_rel_err < 10%
- **P2**: H128 max_rel_err < 5%
- **P3**: 质量漂移 < 1e-6/step
- **P4**: 全场 finite（不发散）

若 P1-P4 任一不满足 → honest-FAIL，记录剩余候选机理。

## 5. 实施步骤

1. ✅ 修改 `src/tensorlbm/multiphase.py`：新增 `u_eq="viscous"` 模式
2. ⏳ 创建基准测试 `benchmarks/pending/two_phase_poiseuille_viscous/run.py`
3. ⏳ 集群执行：H64 + H128 两层网格
4. ⏳ 结果分析：对比解析解，计算误差
5. ⏳ 知识回写：TensorKnowledge claims (≥3 条)
6. ⏳ 完成报告：论文级六要素

## 6. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| "viscous" 模式不稳定 | 回退到 "self" + 界面修正（备选方案） |
| 误差仍 > 10% | 检查解析解实现；尝试压力张量力项 |
| 质量不守恒 | 检查力项动量传递；添加质量源修正 |
