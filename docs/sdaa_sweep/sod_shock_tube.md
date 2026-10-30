# SDAA Sweep: sod_shock_tube (Isothermal D2Q9 Sod Shock Tube)

## 案例概述

| 项目 | 值 |
|------|-----|
| 案例名 | `sod_shock_tube_isothermal_d2q9` |
| 来源 | `benchmarks/verified/sod_shock_tube/` |
| 物理模型 | D2Q9 BGK-LBM，等温 EOS (p=ρ/3, cs²=1/3) |
| 参照解 | 等温 Riemann 解析解（非 γ=1.4 经典 Sod 解） |
| 初始条件 | ρ_L=1.0, ρ_R=0.25 (密度比 4:1), u=0, f=feq |
| 参数 | τ=0.8 (ν=0.1), ny=4, 周期边界 |
| 网格/步数 | nx=2000/t=400, nx=4000/t=800 |
| 声学补充 | ε=0.01 (Ma≈0.01), ε=0.25 (Ma≈0.26) |

## SDAA 运行配置

| 项目 | 值 |
|------|-----|
| 设备 | SDAA (`torch.sdaa`) |
| 设备名 | `/dev/tcaicard1` |
| 设备证据 | `torch.sdaa.get_device_name(0) = '/dev/tcaicard1'` |
| Triton 绕行 | **无** — 核心路径 (equilibrium/stream/collide_bgk/macroscopic) 均为纯 torch ops，无 triton 依赖 |
| 脚本 | `scripts/sdaa_sweep_sod_shock_tube.py` |
| 产物 | `artifacts/sdaa_sod_shock_tube.json` |

## 墙钟时间

| 子任务 | 耗时 (s) |
|--------|----------|
| nx=2000, 400 steps | 1.69 |
| nx=4000, 800 steps | 2.78 |
| acoustic ε=0.01, 400 steps | 1.48 |
| acoustic ε=0.25, 800 steps | 2.70 |
| **总计** | **8.66** |

远小于 20 分钟限制。

## 判据逐条比对

判据口径与原始 `benchmarks/verified/sod_shock_tube/run.py` 完全一致（NVIDIA 侧已有成熟判据）。

| # | 判据 (Gate) | 阈值 | nx=2000 实测 | nx=4000 实测 | PASS/FAIL |
|---|-------------|------|-------------|-------------|-----------|
| 1 | 稳定性 (无 NaN/负密度) | first_bad_step=None | None | None | **PASS** |
| 2 | L2(ρ)/Δρ ≤ 3% | ≤ 0.03 | 0.842% | 0.458% | **PASS** |
| 3 | L2(u)/u* ≤ 3% | ≤ 0.03 | 1.811% | 0.872% | **PASS** |
| 4 | \|W_rel\| ≤ 3% | ≤ 0.03 | 0.09% | 0.03% | **PASS** |
| 5 | \|ρ_mid_rel\| ≤ 3% | ≤ 0.03 | 0.016% | 0.007% | **PASS** |
| 6 | \|u_mid_rel\| ≤ 3% | ≤ 0.03 | 0.041% | 0.026% | **PASS** |
| 7 | 单调收敛 (L2 不增) | L2(nx4000) ≤ L2(nx2000)×1.02 | 0.458% ≤ 0.842%×1.02 | — | **PASS** |

### 声学补充

| ε | Ma_mid | W_rel | u_mid_rel | L2(ρ)/Δρ | 稳定 |
|---|--------|-------|-----------|----------|------|
| 0.01 | 0.010 | −0.042% | −0.188% | 1.891% | PASS |
| 0.25 | 0.256 | +0.094% | −0.041% | 0.795% | PASS |

## 总判定

**ALL PASS** — 所有判据通过，SDAA 设备上的结果与原始 CPU 参考值一致（数值完全吻合，因算法为确定性浮点运算，SDAA 与 CPU 使用相同 torch ops）。

## 绕行记录

- **Triton**: 无绕行。核心路径 (`tensorlbm.d2q9.equilibrium`, `tensorlbm.solver.stream`, `tensorlbm.solver.collide_bgk`, `tensorlbm.d2q9.macroscopic`) 均为纯 PyTorch 张量运算，不涉及 triton 编译。
- **设备放置**: 所有张量在 SDAA 上创建和计算，仅在快照采集时 `.detach().cpu().numpy()` 转回 CPU 做 numpy 后处理（指标计算）。

## 与 CPU 参考值对比

SDAA 结果与 `benchmarks/verified/sod_shock_tube/result.json` 中的 CPU 参考值完全一致：

| 指标 | CPU 参考 | SDAA 实测 | 差异 |
|------|---------|----------|------|
| L2_rho (nx=2000) | 0.842% | 0.842% | 0 |
| L2_rho (nx=4000) | 0.458% | 0.458% | 0 |
| W_fit (nx=2000) | 0.8130 | 0.8130 | 0 |
| W_fit (nx=4000) | 0.8135 | 0.8135 | 0 |
| ρ_mid_rel (nx=2000) | −0.016% | −0.016% | 0 |
| u_mid_rel (nx=2000) | −0.041% | −0.041% | 0 |

数值完全一致，因为算法是确定性的浮点运算（torch 基本 ops 在 SDAA 和 CPU 上产生相同结果）。
