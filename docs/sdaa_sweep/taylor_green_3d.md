# SDAA Sweep: Taylor-Green 3D (B17-3D)

## 案例信息

| 项目 | 值 |
|------|-----|
| 案例名 | taylor_green_3d |
| 基准编号 | B17-3D |
| 物理问题 | 3D Taylor-Green 涡衰减（D3Q19 周期域） |
| 初始场 | OpenLB tgv3d 同款：u=(U0 sin(kx)cos(ky)cos(kz), −U0 cos(kx)sin(ky)cos(kz), 0) |
| 衰减率理论 | γ_E_theory = 6νk²（|κ|²=3k²，8 支 Fourier 模） |
| 源码入口 | `benchmarks/verified/taylor_green_3d/run.py` |
| 共性模块 | `tensorlbm.d3q19.equilibrium3d` / `macroscopic3d` + `tensorlbm.solver3d.collide_bgk3d` / `stream3d` |

## 设备与配置

| 项目 | 值 |
|------|-----|
| 设备 | SDAA (`/dev/tcaicard0`) |
| torch_sdaa 版本 | 3.1.1a3+gite3971b9 |
| PyTorch 版本 | 2.7.1 |
| 编译模式 | **eager**（无 torch.compile） |
| Triton 绕行 | **不需要** — d3q19/solver3d 模块为纯 PyTorch 张量运算，无 triton kernel |
| U0 | 0.05 |
| Re | 24 |
| 采样间隔 | 50 步 |

### 编译模式说明

SDAA 的 `teco_inductor` 后端（Torch-SDAA 3.1.1a3）对 LBM 完整步链（collide→stream）的编译支持不完整（`compile_route.py` 文档化了 L5 缺陷：teco 的 `EXPAND` fallback 路径越界）。本案例采用 **eager 模式**运行，与 `--compile-mode eager` 轨迹 bit-exact 一致。

### Triton 说明

本案例使用的四个核心模块（`equilibrium3d`, `macroscopic3d`, `collide_bgk3d`, `stream3d`）均为纯 PyTorch 张量运算，**不依赖 triton**。无需设置 `TRITON_INTERPRET=1`。

## 网格配置

| N | τ | ν | 步数 | 自动步数公式 |
|---|---|---|------|-------------|
| 64 | 0.900 | 0.1333 | 2000 | E/E0→e⁻¹⁰, cap 10000 |
| 96 | 1.100 | 0.2000 | 2000 | 同上 |
| 128 | 1.300 | 0.2667 | 2600 | 同上 |

## 结果表

| N | γ_E_sim (×10⁻³) | γ_E_theory (×10⁻³) | err_E (%) | R² | 墙钟 (s) |
|---|-----------------|-------------------|-----------|-----|---------|
| 64 | 7.732528 | 7.710628 | **+0.2840** | 0.999998 | 81.1 |
| 96 | 5.154720 | 5.140419 | **+0.2782** | 0.999996 | 88.0 |
| 128 | 3.864921 | 3.855314 | **+0.2492** | 0.999995 | 114.5 |

**总墙钟：283.6 秒（≈4.7 分钟）**

## 判据逐条比对

### 1. 动能衰减率偏差 ≤ 3%

| 网格 | err_E | 阈值 | PASS/FAIL |
|------|-------|------|-----------|
| N=64 | +0.2840% | ≤3% | ✅ **PASS** |
| N=96 | +0.2782% | ≤3% | ✅ **PASS** |
| N=128 | +0.2492% | ≤3% | ✅ **PASS** |

### 2. 网格收敛（|err_E| 随 N 单调下降）

| 序列 | 趋势 | PASS/FAIL |
|------|------|-----------|
| 0.2840 → 0.2782 → 0.2492 | 单调下降 | ✅ **PASS** |

### 3. 指数衰减质量（R² ≥ 0.999）

| 网格 | R² | 阈值 | PASS/FAIL |
|------|-----|------|-----------|
| N=64 | 0.999998 | ≥0.999 | ✅ **PASS** |
| N=96 | 0.999996 | ≥0.999 | ✅ **PASS** |
| N=128 | 0.999995 | ≥0.999 | ✅ **PASS** |

### 综合判定

| 判据 | 结果 |
|------|------|
| err_E ≤ 3% | ✅ PASS |
| 单调收敛 | ✅ PASS |
| R² ≥ 0.999 | ✅ PASS |
| **Overall** | **✅ PASS** |

## 绕行记录

| 绕行项 | 说明 |
|--------|------|
| torch.compile | **未使用**。SDAA teco_inductor 对 LBM 步链编译不完整（compile_route.py 文档化），采用 eager 模式。与 `--compile-mode eager` 轨迹一致，不影响数值精度。 |
| TRITON_INTERPRET=1 | **不需要**。本案例核心模块（equilibrium3d/macroscopic3d/collide_bgk3d/stream3d）为纯 PyTorch 张量运算，无 triton kernel 依赖。 |

## 物理备注

- 偏差符号：γ_sim > γ_theory（+0.25~0.28%），与 3D TG 涡拉伸效应一致（u·∇u 的旋度部分驱动 w 分量增长，产生小尺度模使能量耗散略快于线性理论）
- 与 CPU 参考值对比：err_E 差异 <0.01%（float32 精度内一致）
- 质量守恒：漂移 ~3e-5（与 CPU 参考一致）

## 产物

- 运行脚本：`scripts/sdaa_sweep_taylor_green_3d.py`
- 结果 JSON：`artifacts/sdaa_taylor_green_3d.json`
- 本文档：`docs/sdaa_sweep/taylor_green_3d.md`
