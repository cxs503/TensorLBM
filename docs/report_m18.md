# Mission 18 补报：SDAA Sweep — couette_3d 及关联基准

## 1. 目的

对 TensorLBM 仓库  案例执行 SDAA（自研加速卡）设备扫荡验证，同时汇总同期完成的 SUBOFF 系列基准（Cumulant / RANS k-ε）与 KBC 熵诊断结果，形成论文级可追溯记录。核心目标：
- 验证 Couette 3D 解析解在 SDAA 设备上的精度与一致性（fp32/fp64）
- 评估 SUBOFF 阻力系数在不同碰撞模型与雷诺数下与参考公式的偏差
- 诊断 KBC（熵稳定）碰撞算子在低步数下的 H-定理行为

## 2. 方法

### 2.1 Couette 3D SDAA Sweep
- **脚本**：（独立运行脚本，不改业务代码）
- **设备**：SDAA ()，PyTorch 2.7.1，sdaa_available=True
- **Triton**：不需要绕行——案例为纯 torch ops，无 triton kernel
- **格子**：D3Q19 BGK，τ=0.8，H=8，ny=10, nx=nz=8
- **边界**：moving top wall + stationary bottom wall，half-way bounce-back
- **判据**：沿用原案例 5 道 gate（G1–G5），l2_rel_err ≤ 3% 为 PASS

### 2.2 SUBOFF 系列基准
- **Bench1**：4L+Cumulant D3Q19，Re=1000，网格 320×120×120，5000 步，from_suboff 法向
- **Bench2**：RANS k-ε (MRT)，Re=10⁵，网格 200×80×80，5000 步，from_gradient 法向
- **Bench3**：RANS k-ε (MRT)，Re=10⁴，网格 200×80×80，5000 步，from_gradient 法向
- **参考**：Blasius Cf=1.328/√Re (Re=1000)；ITTC-1957 Cf=0.075/(log₁₀Re−2)² (Re=10⁴, 10⁵)

### 2.3 KBC 熵诊断
- 16³ 立方体，Re=100，20 步，对比 BGK vs KBC 碰撞
- 监控 H-定理违反数、Cd 比值、h-mode 弛豫

## 3. 关键结果

### 3.1 Couette 3D SDAA Sweep — ✅ ALL PASS

| 指标 | fp32 | fp64 |
|------|------|------|
| L2 相对误差 | 5.87e-06 | 1.62e-14 |
| 最大绝对误差 | 2.62e-07 | 9.02e-16 |
| 最大相对误差(%) | 1.5140e-03 | 1.9244e-12 |
| 顶板速度误差(%) | 1.033147e-04 | 1.9244e-12 |
| 质量漂移(%) | 8.5831e-05 | -1.8652e-12 |
| uz 最大 | 1.86e-09 | 0.0e+00 |
| 稳态 | True | True |
| 耗时(s) | 7.0 | 7.1 |

**Gate 通过情况**：
| Gate | 判据 | 结果 |
|------|------|------|
| G1_direct_observable | l2_rel_err <= 3% for all grids (fp32) | ✅ PASS |
| G2a_fp32_floor | l2_rel_err <= 1e-3 for all fp32 grids | ✅ PASS |
| G2b_fp64_scheme_exactness | l2_rel_err <= 1e-7 for all fp64 grids | ✅ PASS |
| G3_steady_state | steady=true, |mass_drift|<1%, uz_max<1e-6, spanwise_std<1e-6 (fp64) | ✅ PASS |
| G4_library_entry | tensorlbm.d3q19 + tensorlbm.solver3d importable and functional on sdaa | ✅ PASS |

总耗时：14.1s

### 3.2 SUBOFF 阻力基准

| 基准 | 碰撞模型 | Re | Cd_total | Cf_ref | 误差(%) | 判定 | 耗时(s) |
|------|----------|-----|----------|--------|---------|------|---------|
| Bench1 | Cumulant D3Q19 | 1000 | 0.041786 | 0.041995 | 0.50 | ✅ PASS | 1983 |
| Bench2 | RANS k-ε MRT | 100000 | 0.006623 | 0.008333 | 20.53 | ❌ FAIL (>15%) | 1038 |
| Bench3 | RANS k-ε MRT | 10000 | 0.010649 | 0.018750 | 43.21 | ❌ FAIL (>15%) | 1028 |

- Bench1 (Cumulant, Re=1000)：误差 0.50%，远优于 MRT+Smag 的 2.3%，提升 4.6×
- Bench2 (RANS, Re=10⁵)：误差 20.53%，略优于 Smagorinsky 的 23.9%，但劣于 Vreman 的 16.4%
- Bench3 (RANS, Re=10⁴)：误差 43.21%，未收敛至 ITTC-1957 参考线

### 3.3 KBC 熵诊断 — ❌ FAIL

| 指标 | 值 |
|------|-----|
| BGK 最终 Cd | -3.0343 |
| KBC 最终 Cd | 22.0622 |
| 参考 Cd | 1.0917 |
| Cd 比值 KBC/BGK | -7.27 |
| H-定理违反格点数 | 105 / 16³ |
| H-违反最大超额 | 5.960464e-08 |
| h_norm 峰值 | 1.019e-2 |
| h_norm 终值 | 7.157e-3 |

**关键发现**：
1. H-定理违反：105 个格点 H(f*)>H(f)，最大超额 5.96e-8
2. Cd 比值 KBC/BGK = −7.27，两者均严重偏离参考值 1.09
3. H-mode 未被 KBC 碰撞有效弛豫（h_norm 仅衰减 30%）

## 4. 审计与局限

### 4.1 证据等级
| 案例 | 证据等级 | 说明 |
|------|----------|------|
| couette_3d SDAA | **REAL** | 解析解对比，fp64 误差 ~1e-14，所有 gate 通过 |
| SUBOFF Bench1 (Cumulant) | **REAL** | Blasius 参考，0.5% 误差 |
| SUBOFF Bench2 (RANS Re1e5) | **REAL** (精度不足) | ITTC-1957 参考，20.5% 误差 >15% 阈值 |
| SUBOFF Bench3 (RANS Re1e4) | **REAL** (精度不足) | ITTC-1957 参考，43.2% 误差 >15% 阈值 |
| KBC 诊断 | **REAL** (FAIL) | H-定理违反，Cd 严重偏离 |

### 4.2 局限与后续
1. **RANS k-ε 精度不足**：Bench2/Bench3 误差均超 15% 不可接受线。可能原因：(a) 5000 步未充分收敛；(b) from_gradient 法向计算精度不足；(c) k-ε 模型常数需标定。建议增加步数至 20000+ 并对比 from_suboff 法向。
2. **KBC 实现问题**：H-定理违反表明熵稳定机制在 fp32 下存在数值问题，需检查正则化步骤或切换 fp64 验证。
3. **Couette 案例**：仅测试 H=8 单网格，建议扩展至 H=16/32 验证网格收敛性。
4. **SUBOFF 网格分辨率**：Bench2/3 使用 200×80×80，可能不足以解析高 Re 边界层。

### 4.3 Honest-FAIL 声明
本任务中 couette_3d SDAA sweep 全部通过（PASS），但 SUBOFF RANS 基准（Bench2/3）和 KBC 诊断均为 FAIL。这些 FAIL 结论如实保留，不做任何美化。原单交付的脚本和工件未修改。

## 5. 产物清单

| 文件 | 说明 |
|------|------|
|  | SDAA 扫荡脚本（独立运行，不改业务代码） |
|  | Couette 3D SDAA 结果（ALL PASS） |
|  | KBC 熵诊断（FAIL） |
|  | SUBOFF Cumulant Re=1000（PASS, 0.5%） |
|  | SUBOFF RANS Re=10⁵（FAIL, 20.5%） |
|  | SUBOFF RANS Re=10⁴（FAIL, 43.2%） |
|  | 综合对比汇总 |
