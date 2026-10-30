# Mission 15 进展报告：SDAA 验证案例扫荡 — cavity_3d_full

> **补报单** | 原单 mission 15 已成功交付但缺报告，本单补发论文级进展报告。
> **仓库**: TensorLBM | **分支**: agent/m15 | **日期**: 2026-07-26

---

## 1. 目的

对 TensorLBM 的 SDAA（自研加速卡）设备执行验证案例扫荡，以 **cavity_3d_full**（三维方腔驱动流）为核心案例，同时覆盖 SUBOFF 阻力基准和 KBC 碰撞诊断。目标：

1. 在 SDAA 设备上独立运行 cavity_3d_full，验证数值精度是否满足 ≤3% 误差门限；
2. 验证网格收敛的单调性（64³ → 96³ 误差严格递减）；
3. 扩展至 SUBOFF 钝体阻力基准（Cumulant / RANS k-ε），评估湍流模型精度；
4. 诊断 KBC（Karlin-Bösch-Chikatariya）熵碰撞的 H 定理保持性。

---

## 2. 方法

### 2.1 案例配置

| 案例 | 碰撞模型 | 网格 | Re | 步数 | 设备 |
|------|---------|------|----|------|------|
| cavity_3d_full | TRT | 64³, 96³ | 1000 | 100,000 | sdaa |
| KBC diagnostic | KBC vs BGK | 16³ | 100 | 20 | cpu |
| SUBOFF bench1 | Cumulant D3Q19 | 320×120×120 | 1,000 | 5,000 | sdaa:20 |
| SUBOFF bench2 | RANS k-ε (MRT) | 200×80×80 | 100,000 | 5,000 | sdaa:21 |
| SUBOFF bench3 | RANS k-ε (MRT) | 200×80×80 | 10,000 | 5,000 | sdaa:22 |

### 2.2 判据口径

- **cavity_3d_full**: 参照 Ku, Hirsh & Taylor (1987) u_min(0.5,0.5) = −0.2751；门限 ≤3%。
- **SUBOFF**: 参照 Blasius Cf=1.328/√Re (Re=1000) 和 ITTC-1957 Cf=0.075/(log₁₀Re−2)² (Re≥10⁴)。
- **KBC**: H 定理单调性（H_after ≤ H_before）与 Cd 符号/量级合理性。
- **Triton 依赖**: 核心模块 (solver3d, boundaries3d, d3q19) 无 Triton 依赖，无需 TRITON_INTERPRET=1 绕行。

### 2.3 四门验证体系 (cavity_3d_full)

1. **library_entry**: 使用 tensorlbm.solver3d / boundaries3d / d3q19 公共 API。
2. **direct_observable**: 直接可观测量（u_min 中心线速度极小值）与参考值比对。
3. **monotone_convergence**: 网格加密后误差严格递减。
4. **reference_locked_first**: 参考值在运行前锁定，无事后调参。

---

## 3. 关键结果

### 3.1 cavity_3d_full — SDAA 验证（✅ 全部通过）

| 网格 | u_min(0.5,0.5) | 误差 vs 参考 | 残差 | 质量漂移 | 耗时(s) |
|------|----------------|-------------|------|---------|---------|
| 64³ | −0.27142 | **1.337%** | 0.0374 | 708.6 | 3,066.7 |
| 96³ | −0.27582 | **0.260%** | 0.0398 | 2,425.4 | 4,283.9 |

- **单调收敛**: 64³→96³ 误差 1.337% → 0.260%（严格递减 ✅）
- **总壁钟时间**: 7,351.5 s
- **四门全通过**: library_entry ✅ | direct_observable ✅ | monotone_convergence ✅ | reference_locked_first ✅
- **3D 效应**: z 中截面 uz_max_abs = 0.00196 (64³) / 0.00130 (96³)，与 2D 解偏差 ~21%（符合 3D 端壁效应预期）

### 3.2 SUBOFF 阻力基准

| 基准 | 碰撞 | Cd_total | Cf_ref | 误差% | 判定 |
|------|------|----------|--------|-------|------|
| B1: 4L+Cumulant Re=1000 | Cumulant D3Q19 | 0.04179 | 0.04200 (Blasius) | **0.50%** | ✅ PASS |
| B2: RANS k-ε Re=1e5 | RANS k-ε→MRT | 0.00662 | 0.00833 (ITTC) | **20.5%** | ❌ FAIL |
| B3: RANS k-ε Re=1e4 | RANS k-ε→MRT | 0.01065 | 0.01875 (ITTC) | **43.2%** | ❌ FAIL |

**关键发现**:
- Cumulant D3Q19 在 4L 域上达到 0.50% 误差，比 MRT+Smagorinsky (2.3%) 好 4.6 倍 — **SUBOFF 最佳结果**。
- RANS k-ε 在 Re=1e5 误差 20.5%（略优于 Smagorinsky 23.9%，劣于 Vreman 16.4%）。
- RANS k-ε 在 Re=1e4 误差 43.2%（远劣于 Smagorinsky 20.6%），模型在过渡 Re 下过度生成涡粘。
- RANS k-ε 显著低估摩擦阻力 (Cd_f)，需壁面函数修正近壁行为。

### 3.3 KBC 熵碰撞诊断（16³ 立方体，20 步）

| 指标 | BGK | KBC | 参考 |
|------|-----|-----|------|
| 最终 Cd | −3.034 | 22.062 | 1.092 |
| Cd 比值 KBC/BGK | — | −7.27 | ~1 |

- **H 定理违反**: 105 个格子出现 H(f*) > H(f)，最大超额 5.96×10⁻⁸
- **H 模式未弛豫**: h_norm 峰值 1.019×10⁻²，末值 7.16×10⁻³（比值 0.70），h 未被 KBC 碰撞有效弛豫
- **结论**: KBC 实现存在缺陷 — Cd 符号错误且量级偏离参考值 20 倍

---

## 4. 审计与局限

### 4.1 审计总结

| 能力 | 证据等级 | 说明 |
|------|---------|------|
| TRT 方腔 (SDAA) | **REAL** | 四门全过，单调收敛，误差 0.26% |
| Cumulant SUBOFF (SDAA) | **REAL** | 0.50% 误差，Blasius 参考 |
| RANS k-ε SUBOFF (SDAA) | **REAL** (精度不足) | 算法真实但误差 >15%，需壁面函数 |
| KBC 熵碰撞 | **MOCK/缺陷** | Cd 符号错误，H 定理违反，h 模式未弛豫 |

### 4.2 局限与诚实 FAIL 声明

1. **RANS k-ε 精度 FAIL (honest-FAIL)**:
   - Re=1e5 误差 20.5% > 15% 门限 → 不可接受
   - Re=1e4 误差 43.2% > 15% 门限 → 不可接受
   - 根因：缺少壁面函数，近壁涡粘过度生成
   - 改进方向：实现非平衡壁面函数或切换至 Vreman LES

2. **KBC 碰撞 FAIL (honest-FAIL)**:
   - Cd 符号错误（22.06 vs 参考 1.09），BGK 也为负值（−3.03）
   - H 定理在 105 个格子被违反
   - 可能原因：16³ 网格过粗（仅 20 步），或 KBC 熵修复步骤实现有误
   - 需进一步在更细网格和更多步数下诊断

3. **cavity_3d_full 质量漂移**:
   - 64³: 708.6, 96³: 2,425.4（随网格增大）
   - 虽不影响速度场精度，但长期积分需关注质量守恒

4. **计算成本**:
   - cavity_3d_full 两网格合计 7,351.5 s（~2 小时），SDAA 单卡吞吐有限
   - SUBOFF 各基准 ~1,000–2,000 s/算例

### 4.3 未覆盖项

- 未执行 128³ 或更细网格的 cavity 验证（受限于 SDAA 内存与时间）
- 未对比 CUDA 与 SDAA 的数值一致性（需双设备环境）
- Vreman LES 在 SUBOFF 上的详细数据未在本次扫荡中完整采集

---

## 5. 结论

Mission 15 成功完成 SDAA 验证案例扫荡：

- ✅ **cavity_3d_full**: 四门全过，96³ 误差 0.26%，SDAA 设备上的 LBM 求解器精度得到验证
- ✅ **SUBOFF Cumulant**: 0.50% 误差，为当前最佳 SUBOFF 结果
- ❌ **SUBOFF RANS k-ε**: 误差 20.5%–43.2%，超过 15% 门限（honest-FAIL）
- ❌ **KBC 熵碰撞**: Cd 符号错误，H 定理违反（honest-FAIL）

**下一步**: 优先修复 RANS k-ε 壁面函数和 KBC 熵修复步骤，并在更细网格下复验。

---

*报告由 TensorPlatform 船舶总体性能智能体自动生成 | 工件来源: artifacts/sdaa_cavity_3d_full.json, artifacts/suboff_*.json, artifacts/kbc_diagnostic_16cube_20steps.json*
