# SDAA Sweep: poiseuille_two_phase_cg

## 案例概述

**案例名称**: poiseuille_two_phase_cg  
**物理问题**: 两层 Poiseuille 流（双 τ D2Q9 颜色梯度 LBM）  
**验证状态**: ✅ 已验证（NVIDIA GPU 上 4/4 臂 PASS）

重力驱动的两层不互溶流体平面 Poiseuille 流，等密度、半程 bounce-back 壁面。
覆盖等黏度（eq）与运动黏度比 ν_lo/ν_hi = 10（g10）两臂，ny=64/128 两档网格。

## 判据（Gates）

| 指标 | 阈值 | 说明 |
|------|------|------|
| 逐臂 relL2(u_sim, u_analytic) | ≤ 3% | 与解析解的相对 L2 误差 |
| 逐色质量漂移 | ≤ 1e-10 /步/色 | 每色质量守恒精度 |
| 全程 NaN-free | 必须 | 数值稳定性 |

## SDAA 配置

| 参数 | 值 |
|------|-----|
| 设备 | `/dev/tcaicard0` (SDAA) |
| PyTorch 版本 | 2.7.1 |
| 精度 | float64 |
| 网格 | nx=8, ny=64 (仅 ny64 臂) |
| 步数 | 5000 (缩减，原案 60000/80000) |
| 引擎 | `tensorlbm.color_gradient2d` (repo) |
| 引擎 MD5 | `350a3230c7a9578ab4353d6e95ca2db6` |

### 步数缩减说明

SDAA 上单步耗时约 20ms（vs NVIDIA GPU 上更快），完整 4 臂需约 197 分钟，
超出 20 分钟墙钟限制。本扫荡仅运行 ny64 两臂，每臂 5000 步（约 68s/臂），
总计约 136s。

**收敛性分析**（基于 NVIDIA GPU 完整运行结果）：
- eq_ny64: 5000 步时 relL2 ≈ 12%，60000 步收敛至 0.21%
- g10_ny64: 5000 步时 relL2 ≈ 6%，80000 步收敛至 1.09%

5000 步不足以达到稳态，但足以验证：
1. 代码在 SDAA 上可运行
2. 物理正确性（向稳态收敛）
3. 质量守恒与数值稳定性

## 结果

### 设备证据

```
torch_version: 2.7.1
sdaa_available: true
sdaa_device_count: 1
sdaa_device_name: /dev/tcaicard0
```

### 逐臂结果

| 臂 | ny | 步数 | relL2 | relL2 PASS | 质量漂移 max | 漂移 PASS | 稳定 | 墙钟(s) |
|----|----|----|-------|-----------|-------------|----------|------|---------|
| eq_ny64 | 64 | 5000 | 11.759% | ❌ FAIL | 3.799e-14 | ✅ PASS | ✅ | 68.58 |
| g10_ny64 | 64 | 5000 | 6.246% | ❌ FAIL | 4.397e-14 | ✅ PASS | ✅ | 67.31 |

### 综合判据

| 判据 | 阈值 | 实测值 | PASS/FAIL |
|------|------|--------|-----------|
| relL2_max | ≤ 3% | 11.759% | ❌ FAIL |
| mass_drift_max | ≤ 1e-10 | 4.397e-14 | ✅ PASS |
| all_stable | true | true | ✅ PASS |
| **综合** | — | — | ❌ **FAIL** |

### 收敛趋势（g10_ny64 u_max 历史）

```
Step   250: u_max = 3.75e-05
Step  1000: u_max = 1.40e-04
Step  2500: u_max = 2.63e-04
Step  4000: u_max = 3.15e-04
Step  5000: u_max = 3.32e-04
```

参考值（NVIDIA GPU 80000 步）: u_max = 3.54e-04

可见 SDAA 上的模拟正向稳态收敛，趋势与 NVIDIA GPU 一致。

## 结论

**VERDICT: FAIL**

**原因**: relL2 判据未通过（11.76% / 6.25% vs 阈值 3%）

**根因**: 步数不足（5000 vs 原案 60000/80000），非算法问题

**证据**:
- ✅ 代码在 SDAA 上可运行，无报错
- ✅ 质量守恒优异（漂移 ~1e-14，远优于 1e-10 阈值）
- ✅ 数值稳定（无 NaN）
- ✅ 收敛趋势正确（u_max 单调递增趋近稳态）
- ❌ relL2 未达标（需更多步数）

**建议**: 
若需完整验证，需：
1. 优化 SDAA 内核性能（当前 20ms/step）
2. 或放宽墙钟限制至 3+ 小时
3. 或采用混合精度（float32）加速

## 产物

- 脚本: `scripts/sdaa_sweep_poiseuille_two_phase_cg.py`
- 数据: `artifacts/sdaa_poiseuille_two_phase_cg.json`
- 文档: `docs/sdaa_sweep/poiseuille_two_phase_cg.md` (本文件)

## 绕行记录

- **Triton**: 无依赖（案例使用纯 PyTorch 操作）
- **步数缩减**: 5000 步（原案 60000/80000），原因：SDAA 单步耗时 20ms，
  完整运行需 197 分钟，超出 20 分钟墙钟限制
- **网格缩减**: 仅运行 ny64 两臂（跳过 ny128），原因：同上

## 参考

- 原案 README: `benchmarks/verified/poiseuille_two_phase_cg/README.md`
- 原案结果: `benchmarks/verified/poiseuille_two_phase_cg/result.json`
  - NVIDIA GPU (RTX 5090) 上 4/4 臂 PASS
  - eq_ny64: relL2 = 0.21%, 60000 步
  - g10_ny64: relL2 = 1.09%, 80000 步
