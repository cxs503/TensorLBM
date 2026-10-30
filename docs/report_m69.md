# Mission 69 补报：SDAA 验证案例扫荡 — thermal_cavity_3d 及 SUBOFF/KBC 诊断

> **Mission ID**: 69 · **Track**: W9-A SDAA Sweep · **Repo**: TensorLBM  
> **日期**: 2026-07-26 · **设备**: SDAA ×3 (sdaa:20/21/22) + sdaa (thermal)  
> **结论**: thermal_cavity_3d **PASS**；SUBOFF Cumulant **PASS**；SUBOFF RANS **FAIL**；KBC 诊断发现两个 root cause（未修复，仅诊断）

---

## 1. 案例介绍

### 1.1 thermal_cavity_3d（自然对流方腔）

| 参数 | 值 |
|------|-----|
| 几何 | 3D 立方腔 L=31 (格点) |
| 流体 | Pr=0.71（空气） |
| 瑞利数 | Ra=10³ |
| 网格 | 32³ |
| 步数 | 40 000 |
| 碰撞 | MRT (D3Q19), post-BB, force=post_raw |
| 参考 Nu | 1.07065 (Ra=1e3 文献值) |

物理背景：封闭方腔内浮力驱动自然对流，经典验证基准。Nu 表征无量纲热通量，是判断 LBM 热求解器精度的核心指标。

### 1.2 SUBOFF 水动力体阻力

| Benchmark | Re | 碰撞 | 网格 | 法向方法 |
|-----------|-----|------|------|----------|
| bench1_cumulant | 1 000 | Cumulant D3Q19 | 320×120×120 | from_suboff (解析轴对称) |
| bench2_rans_ke | 100 000 | RANS k-ε (MRT) | 200×80×80 | from_gradient |
| bench3_rans_ke | 10 000 | RANS k-ε (MRT) | 200×80×80 | from_gradient |

SUBOFF 4L 域：L=80, D=9.335, 阻塞率 0.48%。

### 1.3 KBC 碰撞诊断

16³ 立方体 20 步诊断，对比 KBC vs BGK 在球体绕流中的 H 定理遵守情况。

---

## 2. 方法 / 流程

1. **独立运行脚本** `scripts/sdaa_sweep_thermal_cavity_3d.py`：不改业务代码，纯张量放 `sdaa` 设备。
2. **thermal_cavity_3d**：无 triton 依赖，纯 torch tensor ops；窗口平均法取稳态 Nu。
3. **SUBOFF**：三卡并行 (sdaa:20/21/22)，每卡一个 benchmark，5000 步，后 500 步窗口平均 Cd。
4. **KBC 诊断**：`src/tensorlbm/kbc_diagnostic.py` + `tests/test_kbc_diagnostic.py`（20 个 TDD 测试），逐格检查 γ 可容域、H 定理、负分布。
5. **判据口径**：沿用 NVIDIA 侧已有成熟判据——Nu 误差 ≤3%、无 NaN、窗口漂移 <0.1%、冷热不平衡 <1%。

---

## 3. 计算结果

### 3.1 thermal_cavity_3d (Ra=1e³, N=32)

| 指标 | 数值 | 参考 | 误差 | 门限 | 结果 |
|------|------|------|------|------|------|
| Nu (窗口平均) | 1.0576 | 1.0707 | **−1.22%** | ≤3% | ✅ PASS |
| Nu_hot | 1.0576 | — | — | — | — |
| Nu_cold | 1.0576 | — | — | — | — |
| 窗口漂移 | −1.24e-3% | <0.1% | — | <0.1% | ✅ PASS |
| 冷热不平衡 | 0.045% | <1% | — | <1% | ✅ PASS |
| NaN | 无 | — | — | 无 | ✅ PASS |
| 耗时 | 910 s | <1200 s | — | — | ✅ PASS |

### 3.2 SUBOFF 阻力系数

| Benchmark | Cf_num | Cf_ref | 误差 | 门限 | 结果 |
|-----------|--------|--------|------|------|------|
| bench1 Cumulant Re=1e3 | 0.04179 | 0.04200 (Blasius) | **0.50%** | ≤3% | ✅ PASS |
| bench2 RANS k-ε Re=1e5 | 0.00662 | 0.00833 (ITTC-57) | **20.5%** | ≤3% | ❌ FAIL |
| bench3 RANS k-ε Re=1e4 | 0.01065 | 0.01875 (ITTC-57) | **43.2%** | ≤3% | ❌ FAIL |

### 3.3 KBC 诊断 (16³, 20 步)

| 指标 | KBC | BGK | 参考 |
|------|-----|-----|------|
| 最终 Cd | 22.06 | −3.03 | 1.09 |
| H 定理违反格数 (step 20) | 105 | — | 0 |
| γ 范围 (step 4) | [−5.51, 5.20] | — | [−1, 1] |
| h_norm (step 20) | 7.16e-3 | — | → 0 |

---

## 4. 结果图

本任务为 SDAA 验证扫荡 + 诊断，工件中无 npz/png 场量数据（仅 JSON 探针数据），故无法生成云图/曲线。理由：
- thermal_cavity_3d 的 probe 文件仅含 Nu 时间序列标量，无空间场快照；
- SUBOFF 工件仅含 Cd 窗口平均，无压力/速度场；
- KBC 诊断工件为逐格统计量，非场量。

如需可视化，需重新运行案例并开启 FieldSampleReporter 导出 HDF5 场量。

---

## 5. 注意点

1. **thermal_cavity_3d 仅跑了 Ra=1e³ 单网格**：单调性门限（≥3 网格收敛趋势）因 wall-clock 限制未执行，不能声明网格无关性。
2. **RANS k-ε 在高 Re 下严重低估 Cf**：Re=1e5 误差 20.5%、Re=1e4 误差 43.2%。可能原因：
   - 5000 步不足以让 k-ε 湍流模型充分发展；
   - from_gradient 法向计算在粗网格 (200×80×80) 上精度不足；
   - RANS MRT 碰撞的湍流粘度 ν_t 可能未正确耦合到壁面附近。
3. **KBC 两个 root cause 已定位但未修复**：
   - **Primary**: `solve_gamma_entropy()` 中 γ_init 可容域扩展 bug（代码反向扩展了搜索域而非 clamp）；
   - **Secondary**: 高阶非平衡模 h 完全保留（γ_h=1），未松弛，导致障碍物附近非物理积累。
4. **无 triton 依赖**：thermal_cavity_3d 纯 torch ops，无需 TRITON_INTERPRET=1 workaround。

---

## 6. 误差判定

| 关键量 | 基准 | 计算值 | 误差 | 是否 <3% |
|--------|------|--------|------|----------|
| Nu (Ra=1e³) | 1.0707 | 1.0576 | −1.22% | ✅ **是** |
| Cf (SUBOFF Cumulant Re=1e3) | 0.04200 | 0.04179 | 0.50% | ✅ **是** |
| Cf (SUBOFF RANS Re=1e5) | 0.00833 | 0.00662 | 20.5% | ❌ **否** |
| Cf (SUBOFF RANS Re=1e4) | 0.01875 | 0.01065 | 43.2% | ❌ **否** |

**总结**：
- thermal_cavity_3d + SUBOFF Cumulant：**honest-PASS**，关键量误差 <3%。
- SUBOFF RANS k-ε：**honest-FAIL**，误差远超 3%，需增加步数/细化网格/检查 ν_t 壁面耦合。
- KBC：**honest-FAIL**，Cd 异常（22 vs 1.09），root cause 已定位待修复。

---

*报告生成：Mission 69 补报 · TensorLBM SDAA 验证扫荡 · 2026-07-26*
