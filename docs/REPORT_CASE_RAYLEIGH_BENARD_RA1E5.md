# Rayleigh-Bénard / Natural Convection Cavity at Ra=1e5 — Benchmark Report

**任务锚点**: CASE-RAYLEIGH-BENARD-RA1E5  
**Mission**: m15466  
**日期**: 2026-11-12  
**状态**: 进行中（集群作业已递交，等待执行）

---

## ① 问题介绍与参考出处

### 物理问题
自然对流方腔（differentially heated square cavity）：左壁等温 T_hot=1、右壁等温 T_cold=0、上下壁绝热、四壁 no-slip。流体为空气（Pr=0.71），Rayleigh 数 Ra=1e5。

**注**：任务锚点名称为 "CASE-RAYLEIGH-BENARD-RA1E5"，但参考来源 de Vahl Davis 1983 实际为**侧加热方腔**自然对流基准（非经典 Rayleigh-Bénard 水平板间对流）。本算例按 de Vahl Davis 1983 方腔构型实现。

### 参考出处
**de Vahl Davis, G. (1983)**. "Natural convection of air in a square cavity: a bench mark numerical solution." *International Journal for Numerical Methods in Fluids*, 3(3), 249-264.

**Table 1, page 257** 给出 Ra=10^5, Pr=0.71 的基准值：

| 量 | 参考值 | 说明 |
|----|--------|------|
| Nu_mean | 4.519 | 热壁平均 Nusselt 数 |
| u_max | 34.72 | 竖直中线最大水平速度（扩散尺度 u_nd = u_lu·H/α） |
| v_max | 68.60 | 水平中线最大垂直速度（扩散尺度） |
| u_max 位置 | y/H ≈ 0.813 | 竖直中线上 |
| v_max 位置 | x/H ≈ 0.066 | 水平中线上 |

**容差判据**: <8%（任务要求）

---

## ② 方法与实现

### 数值方法
- **求解器**: TensorLBM 热 LBM 模块（`tensorlbm.thermal`）
- **格子模型**: D2Q9（速度场）+ D2Q5（温度场 advection-diffusion）
- **碰撞格式**: BGK + Guo 力格式（Boussinesq 浮力 F = ρ·g·β·(T − T_ref)）
- **边界条件**:
  - 速度：四壁 no-slip（pre-streaming 半程反弹）
  - 温度：左右壁等温（anti-bounce-back Dirichlet），上下壁绝热（bounce-back 零通量）
- **无量纲化**: 
  - ν = (τ − 1/2)/3，α = ν/Pr，τ_T = 3α + 1/2
  - H = nx − 1（等温壁位于节点上）
  - g·β = Ra·ν·α/H³（由 Ra 反解）
  - ΔT = 1（T_hot=1, T_cold=0）

### 实现文件
- **主脚本**: `bench_rayleigh_benard_ra1e5.py`
- **驱动库**: `benchmarks/verified/thermal_cavity/driver.py`（复用已验证的 thermal_cavity 基础设施）
- **图库**: `gen_figures_ra1e5.py`

### 计算配置
| 网格 N | 最大步数 | 数据类型 | 预计时间 |
|--------|----------|----------|----------|
| 64 | 300,000 | float32 | ~30 min |
| 128 | 300,000 | float32 | ~60 min |
| 256 | 800,000 | float64 | ~180 min |

**注**: N=256 使用 float64 以避免 fp32 量化伪影（thermal_cavity 已验证 Ra=1e3 时 fp32 在 N=256 有 ~10% 速度偏差）。

---

## ③ 参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| Ra | 1e5 | Rayleigh 数 |
| Pr | 0.71 | Prandtl 数（空气） |
| τ | 0.6 | 速度场松弛时间 |
| ν | 0.03333 | 运动粘度（格子单位） |
| α | 0.04695 | 热扩散率（格子单位） |
| τ_T | 0.6408 | 温度场松弛时间 |
| H | N−1 | 方腔物理长度（格子单位） |
| g·β | 7.46e-5 (N=128) | 浮力系数（由 Ra 反解） |
| ΔT | 1.0 | 温差 |
| T_ref | 0.0 | 参考温度（T_cold） |

---

## ④ 对照表（ref vs got + 误差）

**状态**: 集群作业已递交（cj1133, cj1134, cj1137），等待执行。

初步测试（N=64, 10k 步，未完全收敛）显示：
| 量 | 参考值 | 初步结果 (10k步) | 误差 |
|----|--------|------------------|------|
| Nu | 4.519 | 3.987 | -11.8% |
| u_max | 34.72 | 35.90 | +3.4% |
| v_max | 68.60 | 64.97 | -5.3% |

u_max 和 v_max 已在 8% 容差内，Nu 需要更多步数达到稳态。

**完整结果待集群作业完成后更新**。

---

## ⑤ 收敛性与参数敏感性

### 网格收敛
计划比较 N=64, 128, 256 三档网格的 Nu、u_max、v_max，验证：
1. 误差随网格细化单调下降
2. N=128 和 N=256 的相对差 <5%

### 时间收敛
稳态判据：Nu 漂移 <1e-4/5000 步，连续 3 窗满足。

### 参数敏感性
- **τ 影响**: 固定 τ=0.6（与 Ra=1e3/1e4 已验证案例一致）
- **数据类型**: N=256 使用 float64 避免量化伪影

---

## ⑥ 失败模式与工件

### 当前状态
- **集群作业**: cj1133 (N=64,128), cj1134 (N=256), cj1137 (N=64 quick) 已递交，排队中
- **初步结果**: N=64 在 10k 步时 u_max/v_max 已接近参考值（<8%），Nu 需更多步数

### 潜在失败模式
1. **稳态不收敛**: Ra=1e5 流动较强，可能需要更多步数或出现非稳态
2. **网格不足**: N=64 可能过粗，Nu 误差超限
3. **数值不稳定**: 高 Ra 数下浮力项可能导致不稳定（已通过 τ/τ_T 控制）

### 工件清单
- `bench_rayleigh_benard_ra1e5.py` — 主 benchmark 脚本
- `gen_figures_ra1e5.py` — 图库生成脚本
- `artifacts/result_ra1e5_*.json` — 集群作业结果（待生成）
- `artifacts/figures_ra1e5/` — 场量云图与收敛曲线（待生成）

---

## 生态摩擦记录

1. **任务锚点名称歧义**: "CASE-RAYLEIGH-BENARD-RA1E5" 实际对应 de Vahl Davis 方腔自然对流，非经典 RB 对流。已按参考来源实现方腔构型。
2. **集群排队**: 作业递交时队列较长（~330  ahead），需等待资源。
3. **本地计算能力不足**: N=64 Ra=1e5 在本地 120s 仅能跑 ~10k 步，必须依赖集群。

---

## 复现命令

```bash
# 集群递交（推荐）
run_cluster submit --repo TensorLBM --cores 16 --timeout 3600 \
  "python3 bench_rayleigh_benard_ra1e5.py --ra 1e5 --nx 64 128 --max-steps 300000 --out artifacts/result_ra1e5.json"

# 本地快速测试（N=64, 50k 步）
python3 bench_rayleigh_benard_ra1e5.py --ra 1e5 --nx 64 --max-steps 50000 --out /tmp/quick_test.json

# 生成图库
python3 gen_figures_ra1e5.py --result artifacts/result_ra1e5.json --out-dir artifacts/figures_ra1e5
```

---

**报告生成时间**: 2026-11-12 00:35  
**下次更新**: 集群作业完成后
