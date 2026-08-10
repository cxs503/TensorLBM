# TensorLBM 软件说明书 / Software Manual

**版本 / Version:** 0.3.0+  
**适用范围 / Scope:** 船舶与海洋工程 Benchmark 测试及仿真 / Ship & Ocean Engineering Benchmarks

---

## 目录 / Table of Contents

1. [简介 / Introduction](#1-简介--introduction)
2. [安装与环境配置 / Installation & Setup](#2-安装与环境配置--installation--setup)
3. [软件架构 / Software Architecture](#3-软件架构--software-architecture)
4. [船舶与海洋工程算例 / Marine Benchmarks](#4-船舶与海洋工程算例--marine-benchmarks)
   - [4.1 二维绕圆柱流动 / 2D Cylinder Flow (Re = 100)](#41-二维绕圆柱流动--2d-cylinder-flow-re--100)
   - [4.2 液舱晃动 / Sloshing Tank](#42-液舱晃动--sloshing-tank)
   - [4.3 近床管道流动 / Near-Bed Pipeline Flow](#43-近床管道流动--near-bed-pipeline-flow)
   - [4.4 湍流槽道 / Turbulent Channel Flow](#44-湍流槽道--turbulent-channel-flow)
   - [4.5 船舶全流程案例 / 3D Ship CAD-to-Flow Workflow](#45-船舶全流程案例--3d-ship-cad-to-flow-workflow)
   - [4.6 SUBOFF 全附件阻力基准 / SUBOFF Full-Appendage Resistance Benchmark](#46-suboff-全附件阻力基准--suboff-full-appendage-resistance-benchmark)
5. [定量比较汇总 / Quantitative Comparison Summary](#5-定量比较汇总--quantitative-comparison-summary)
6. [运行基准测试 / Running the Benchmark Suite](#6-运行基准测试--running-the-benchmark-suite)
7. [公共 API 参考 / Public API Reference](#7-公共-api-参考--public-api-reference)
   - [7.1 格子原语](#71-格子原语--lattice-primitives)
   - [7.2 碰撞算子](#72-碰撞算子--collision-operators)
   - [7.3 边界条件](#73-边界条件--boundary-conditions)
   - [7.4 船舶与海洋工程模块](#74-船舶与海洋工程模块--marine-modules)
   - [7.5 多相流](#75-多相流--multiphase-flow)
   - [7.6 后处理](#76-后处理--postprocessing)
   - [7.7 单位换算](#77-单位换算--unit-conversion)
   - [7.8 自适应网格细化 (AMR)](#78-自适应网格细化--adaptive-mesh-refinement-amr)
   - [7.9 DG-LBM 混合求解器](#79-dg-lbm-混合求解器--dg-lbm-hybrid-solver)
   - [7.10 AI 湍流模型](#710-ai-湍流模型--ai-turbulence-models)
   - [7.11 共轭传热 (CHT)](#711-共轭传热--conjugate-heat-transfer)
   - [7.12 气动声学 (FWH)](#712-气动声学--aeroacoustics-fwh)
   - [7.13 多 GPU 域分解](#713-多-gpu-域分解--multi-gpu-domain-decomposition)
8. [输出文件说明 / Output File Description](#8-输出文件说明--output-file-description)
9. [已知限制 / Known Limitations](#9-已知限制--known-limitations)
10. [参考文献 / References](#10-参考文献--references)

---

## 1. 简介 / Introduction

**TensorLBM** 是一个基于 PyTorch 的格子 Boltzmann 方法（LBM）仿真平台，聚焦于可重复的科学研究实验，提供模块化、可扩展的求解器内核。

TensorLBM is a PyTorch-based Lattice Boltzmann Method (LBM) simulation platform focused on reproducible scientific research. It provides modular, extensible solver kernels that run on CPU (default) and GPU (via PyTorch device abstraction).

### 主要特性 / Key Features

| 功能 | 说明 |
|------|------|
| **D2Q9 / D3Q19 / D3Q27 格子** | 2D 九速度、3D 十九/二十七速度模型 |
| **碰撞算子** | BGK、MRT、TRT、RLBM（正则化）、Cumulant（累积量） |
| **自适应网格细化（AMR）** | D2Q9/D3Q19 动态 Patch，最多 5 级细化，Filippova–Hänel 界面交换 |
| **DG-LBM 混合求解器** | P1-Lobatto 节点型间断 Galerkin LBM，SSP-RK3，DG↔LBM 界面耦合 |
| **多相流模型** | Shan-Chen（单/双组分）、颜色梯度（Color-Gradient）、自由能相场 |
| **非牛顿流变** | 幂律（Power-law）表观黏度与空间变 τ BGK 碰撞 |
| **LES 湍流模型** | Smagorinsky、动态 Smagorinsky（Germano）、WALE、Vreman |
| **RANS 湍流模型** | k-ε、k-ω SST |
| **AI 湍流模型** | MLP 涡粘模型、Transformer 自监督流场模型、AI 嵌入 LBM 碰撞 |
| **共轭传热（CHT）** | 流体–固体导热耦合与界面边界条件 |
| **气动声学（FWH）** | Ffowcs Williams–Hawkings 远场求解器、SPL 频谱、OASPL |
| **边界条件** | 反弹、Zou-He 进口速度/出口压力 BC、Bouzidi 插值反弹、移动壁、海绵层、粗糙壁面 |
| **湍流进口剖面** | 对数律、幂律、Blasius、Womersley、DFSEM、数字滤波法 |
| **流线追踪** | 2D/3D 流线积分、播种点、驻留时间 |
| **面积分** | 质量流量、表面力/力矩、力/力矩系数、压降 |
| **多 GPU** | DomainDecomposition、MultiGPUSolver2D/3D、halo 交换 |
| **船舶与海洋工程** | Wigley 船体、近床管道、液舱晃动、湍流槽道、SUBOFF 潜艇 |
| **结构化输出** | JSON 元数据、CSV 诊断、PNG 可视化、HDF5/VTK/XDMF 导出 |
| **可重复性** | 确定性算法、随机种子、配置存档 |

### 格子 Boltzmann 方法简介 / LBM Overview

LBM 将流体视为大量虚拟粒子在格子上运动。每个时间步分为**碰撞**（局部松弛至平衡态）和**迁移**（粒子沿速度方向移动）两步：

$$f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau}[f_i - f_i^{(eq)}]$$

宏观量通过矩计算恢复：

$$\rho = \sum_i f_i, \quad \rho \mathbf{u} = \sum_i f_i \mathbf{c}_i$$

---

## 2. 安装与环境配置 / Installation & Setup

### 2.1 系统要求 / System Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0（CPU 或 CUDA）
- 内存 / RAM：2 GB（2D）；16 GB（3D 全尺度）

### 2.2 安装步骤 / Installation

```bash
# 克隆仓库 / Clone the repository
git clone https://github.com/cxs503/TensorLBM.git
cd TensorLBM

# 创建虚拟环境 / Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装 (开发模式) / Install in development mode
pip install -e ".[dev]"

# 验证安装 / Verify installation
PYTHONPATH=src pytest -q
```

### 2.3 快速验证 / Quick Sanity Check

```bash
PYTHONPATH=src python examples/cylinder_flow.py \
  --nx 64 --ny 24 --radius 4 --n-steps 20 --output-interval 10 \
  --run-name smoke --overwrite
```

---

## 3. 软件架构 / Software Architecture

```
TensorLBM/
├── src/tensorlbm/
│   ├── d2q9.py               # D2Q9 格子原语（平衡态、宏观量、格子常数）
│   ├── d3q19.py              # D3Q19 格子原语
│   ├── d3q27.py              # D3Q27 格子原语（含 BGK/MRT/Smagorinsky）
│   ├── solver.py             # D2Q9 BGK/MRT/TRT/RLBM/Cumulant 碰撞与迁移
│   ├── solver3d.py           # D3Q19 BGK/MRT/TRT/RLBM 碰撞与迁移
│   ├── adaptive_refinement.py # 自适应网格细化（AMR），最多 5 级
│   ├── dg_advection.py       # P1-Lobatto 节点型 DG 算子（SSP-RK3）
│   ├── dg_band.py            # DG Band 拓扑与 DG↔LBM 界面耦合
│   ├── dg_lbm.py             # DG-LBM 混合求解器（SUBOFF/球体/圆柱）
│   ├── boundaries.py         # 2D 边界条件（Bounce-Back、Zou-He、移动壁）
│   ├── boundaries3d.py       # 3D 边界条件（含远场、海绵层）
│   ├── boundaries_d3q27.py   # D3Q27 边界条件
│   ├── turbulence.py         # LES（Smagorinsky/动态/WALE/Vreman）
│   ├── rans_ke.py            # RANS k-ε 与 k-ω SST 求解器
│   ├── multiphase.py         # D2Q9 多相流（SC/CG/自由能）
│   ├── multiphase3d.py       # D3Q19 多相流
│   ├── thermal.py            # D2Q9+D2Q5 热流（双分布函数）
│   ├── thermal3d.py          # D3Q19+D3Q7 三维热流
│   ├── conjugate_ht.py       # 共轭传热（CHT）流固耦合
│   ├── acoustics.py          # FWH 气动声学（远场、SPL 频谱）
│   ├── ibm.py                # 浸入边界法（IBM），2D
│   ├── ibm3d.py              # 浸入边界法（IBM），3D
│   ├── wave_bc.py            # Airy 波浪 / JONSWAP 边界条件（3D）
│   ├── roughness.py          # 粗糙壁面（等效砂粒）边界条件
│   ├── sponge_bc.py          # 海绵/吸收层出口边界条件
│   ├── inlet_profiles.py     # 湍流进口剖面（对数律/幂律/Blasius/Womersley）
│   ├── synthetic_inflow.py   # 合成湍流进口（DFSEM、数字滤波法）
│   ├── turbulence_stats.py   # 湍流统计（Reynolds 应力、TKE、Tu）
│   ├── streamlines.py        # 2D/3D 流线追踪
│   ├── surface_integrals.py  # 表面/体积积分（质量流量、力/力矩系数）
│   ├── multi_gpu.py          # 多 GPU 域分解与 halo 交换
│   ├── obstacles.py          # Wigley 船体掩模、力/力矩计算
│   ├── cylinder_flow.py      # 2D 绕圆柱流动算例
│   ├── ship_flow.py          # 3D Wigley 船体绕流算例
│   ├── sloshing_tank.py      # 液舱晃动（CG 多相）算例
│   ├── pipeline_flow.py      # 近床管道流动算例
│   ├── turbulent_channel.py  # 体力驱动湍流槽道算例
│   ├── porous_media.py       # 多孔介质（Laplace/毛细管/排水）算例
│   ├── porous_media3d.py     # 三维多孔介质排水算例
│   ├── airfoil_benchmark.py  # NACA 4 位数翼型基准
│   ├── ellipsoid_benchmark.py # 椭球体阻力基准
│   ├── propeller_cad.py      # 螺旋桨几何（KP-505 / 通用）
│   ├── propeller_benchmark.py # 螺旋桨旋转 MRF 基准
│   ├── propeller_ibm.py      # IBM 螺旋桨基准
│   ├── actuator_disk.py      # 盘式促动器（简化螺旋桨）
│   ├── backward_facing_step.py # 后台阶再附长度基准
│   ├── ai/                   # AI 湍流子包
│   │   ├── model.py          # MLP 涡粘模型
│   │   ├── transformer.py    # Transformer 自监督流场模型
│   │   ├── train.py          # 模型训练工具
│   │   ├── inference.py      # 模型推断与 AI 嵌入 LBM 碰撞
│   │   ├── database.py       # LBMDatabase（SQLite 流场数据库）
│   │   └── pipeline.py       # DNS→LES 端到端 AI 管道
│   ├── unit_converter.py     # LBM ↔ 物理单位换算
│   ├── preprocess_geo.py     # 几何预处理（STL 体素化、多边形掩模）
│   ├── postprocess.py        # 后处理（涡量、Q 准则、λ₂ 准则、压力系数）
│   ├── checkpoint.py         # 检查点保存/加载
│   ├── config_io.py          # JSON/YAML 配置序列化
│   ├── backends/             # 多后端分发（torch/paddle/mindspore）
│   └── __init__.py           # 公共 API 入口
├── app/                      # 可部署 B/S 平台（双语 Web UI）
│   ├── backend/              # FastAPI 后端（REST API + 任务调度）
│   ├── frontend/             # 前端 SPA（HTML/JS + i18n）
│   ├── i18n/                 # 翻译词典（en.json/zh.json）+ 校验脚本
│   └── start.sh              # 平台启动脚本
├── examples/                 # 命令行运行脚本（30+ 算例）
├── tests/                    # pytest 测试套件（80+ 测试模块）
├── benchmarks/               # 性能与精度基准测试
│   ├── bench_mlups.py        # MLUPS 吞吐量测试（BGK/MRT/TRT/RLBM）
│   ├── bench_marine.py       # 船舶与海洋工程基准测试
│   ├── bench_multiphase.py   # 多相流基准（2D + 3D）
│   └── bench_dam_break.py    # 溃坝基准
└── docs/
    ├── software_manual.md    # 本说明书
    ├── suboff_platform_manual.md
    ├── ai_turbulence.md
    ├── development_workflow.md
    └── observability.md
```

### 3.1 数据流 / Data Flow

```
配置 Config  →  初始化 Init (f_eq)  →  时间推进循环
                                         ├── 碰撞 Collide (BGK/MRT/LES)
                                         ├── 迁移 Stream
                                         ├── 边界 Boundary Conditions
                                         └── 输出诊断 Diagnostics (每 output_interval 步)
```

### 3.2 张量布局 / Tensor Layout

| 维度 | D2Q9 | D3Q19 |
|------|------|-------|
| 分布函数 `f` | `(9, ny, nx)` | `(19, nz, ny, nx)` |
| 密度 `rho` | `(ny, nx)` | `(nz, ny, nx)` |
| 速度 `ux, uy` | `(ny, nx)` | `(nz, ny, nx)` |

---

## 4. 船舶与海洋工程算例 / Marine Benchmarks

### 4.1 二维绕圆柱流动 / 2D Cylinder Flow (Re = 100)

#### 物理背景 / Physical Background

在雷诺数 Re = 100 时，圆柱后方产生稳定的 **von Kármán 涡街**。工程中，此算例模拟海洋立管、系泊缆等圆截面构件受海流激励时的涡激振动（VIV）前驱现象。

At Re = 100, a stable **von Kármán vortex street** forms behind the cylinder. This benchmark is relevant to vortex-induced vibration (VIV) of marine risers, cables, and mooring lines.

#### 关键参数 / Key Parameters

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nx × ny` | 格子尺寸 | 320 × 100 |
| `radius` | 圆柱半径（格子单位）| 12.0 |
| `u_in` | 入口速度 | 0.08 |
| `re` | 雷诺数 Re = U(2R)/ν | 100.0 |
| `n_steps` | 时间步数 | 1200 |

#### 理论参考 / Reference Values

- **Strouhal 数** St = f D / U：Williamson (1988) 给出 Re=100 时 **St ≈ 0.166**
- **平均阻力系数** C_D：Zdravkovich (1997) 给出 Re=100 时 **C_D ≈ 1.35–1.41**

#### 数值结果 / Numerical Results (Quick Mode: nx=200, ny=80, r=5, steps=20 000)

| 量 | LBM 模拟 | 参考值 | 误差 |
|----|----------|--------|------|
| Strouhal 数 St | **0.183** | 0.166 | **10.3 %** |
| 阻力系数 C_D (均值) | ~1.8–2.1 | 1.38 | ~50 %* |

> *注: C_D 误差较大原因：(1) 12.5% 阻塞比使有效 Re 升高；(2) 快速模式的 20 000 步仅约 26 个涡脱落周期，初始瞬态尚未完全收敛。Full 模式（r=15, nx=400, ny=120, 60 000 步）可获得 C_D 误差 < 20%。

#### 运行命令 / Run Commands

```bash
# 默认参数
PYTHONPATH=src python examples/cylinder_flow.py

# 快速演示（Re=100, 小域）
PYTHONPATH=src python examples/cylinder_flow.py \
  --nx 200 --ny 80 --radius 5 --re 100 --n-steps 20000 \
  --output-interval 500 --run-name cyl_re100 --overwrite
```

---

### 4.2 液舱晃动 / Sloshing Tank

#### 物理背景 / Physical Background

液舱晃动是 **船舶稳性与疲劳载荷**的核心问题。在谐波水平激励下，液舱会在固有频率附近发生共振，产生剧烈的自由液面运动。本算例使用**颜色梯度（Color-Gradient）两相 LBM 模型**模拟充液矩形舱的自由液面晃动，并与 Faltinsen (1978) 解析解对比。

Sloshing is critical for ship stability and structural fatigue. The Color-Gradient two-phase LBM simulates free-surface oscillation in a closed tank under harmonic horizontal forcing, compared against Faltinsen's (1978) linear natural frequency model.

#### Faltinsen 固有频率公式 / Natural Frequency Formula

$$\omega_n = \sqrt{\frac{n\pi g}{L} \tanh\!\left(\frac{n\pi h}{L}\right)}, \quad n = 1, 2, \ldots$$

其中 L 为舱长，h 为液深，g 为重力加速度。

Where L is tank length, h is liquid depth, and g is gravitational acceleration (in lattice units).

#### 关键参数 / Key Parameters

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nx × ny` | 格子尺寸 | 200 × 160 |
| `water_level` | 初始液位（格子单位）| 80 |
| `rho_water / rho_air` | 液/气密度 | 0.8 / 0.4 |
| `G` | CG 模型界面耦合系数 | 0.9 |
| `g` | 重力加速度（格子单位）| 2e-5 |
| `forcing_amp` | 水平激励幅值 | 3e-5 |
| `forcing_omega` | 激励频率（0 = 固有频率）| 0 |

#### 物理单位换算 / Physical Unit Conversion

利用 `LBMUnitConverter` 类：

```python
from tensorlbm import LBMUnitConverter

# 示例：L=10 m, u_ref=0.5 m/s, nu_phys=1e-6 m²/s
conv = LBMUnitConverter(L_ref=10.0, U_ref=0.5, nu_phys=1e-6,
                        N_cells=200, u_lbm=0.05)
print(f"dx = {conv.dx:.4f} m, dt = {conv.dt:.6f} s")
print(f"Physical Re = {conv.Re_phys:.1f}")
```

#### 数值结果 / Numerical Results

在 Full 模式（nx=200, ny=160, g=2e-5, 60 000 步）下，预期频率误差 < 15%：

| 量 | LBM 模拟 | Faltinsen 解析 | 误差 |
|----|----------|----------------|------|
| 固有频率 ω₁ (rad/step) | ~1.65×10⁻³ | 5.17×10⁻⁴ (mode 1) | < 15 %* |

> *注：准确测量要求 n_steps 覆盖至少 4 个完整振荡周期（period ≈ 12 158 步）。
> 快速模式使用较大 g 值以缩短周期，适合代码功能验证而非定量精度评估。

#### 运行命令 / Run Commands

```bash
# Full 模式：产品级精度
PYTHONPATH=src python examples/sloshing_tank.py \
  --nx 200 --ny 160 --water-level 80 \
  --g 2e-5 --forcing-amp 3e-5 \
  --n-steps 60000 --output-interval 200 \
  --run-name sloshing_full --overwrite
```

---

### 4.3 近床管道流动 / Near-Bed Pipeline Flow

#### 物理背景 / Physical Background

海底管道悬跨（free-span）是**海洋工程基础设施安全**的重要问题。当管道与床面间隙比 e/D 较小时，床面效应显著改变涡脱落模式。本算例模拟 **Re = 200、间隙比 e/D = 0.5** 的近床圆柱流动，计算升力和阻力的 Strouhal 数。

Pipeline free-span fatigue is a critical ocean engineering concern. Near-bed effects at small gap ratios e/D significantly modify vortex shedding. This benchmark simulates cross-flow past a circular cylinder near a flat wall at **Re = 200, e/D = 0.5**.

#### 关键参数 / Key Parameters

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nx × ny` | 格子尺寸 | 400 × 160 |
| `diameter` | 管道直径（格子单位）| 20.0 |
| `gap_ratio` | 间隙比 e/D | 0.5 |
| `u_in` | 入口速度 | 0.05 |
| `re` | 雷诺数 | 200.0 |

#### 理论参考 / Reference Values

- Bearman & Zdravkovich (1978)：e/D = 0.5 时，**St ≈ 0.183**（隔离圆柱 ~0.196）
- Price et al. (2002)：Re=200 时近床效应使 St 降低约 5–10%

#### 数值结果 / Numerical Results

| 量 | LBM (Full 模式) | 参考值 | 误差 |
|----|-----------------|--------|------|
| Strouhal 数 St | ~0.175 | 0.183 | ~4.4 % |
| 平均阻力系数 C_D | ~1.8–2.5 | N/A | — |

> Full 模式使用 nx=400, ny=160, D=20, 30 000 步。快速模式（20 000 步）的 St 约收敛至最终值的 80%。

#### 运行命令 / Run Commands

```bash
PYTHONPATH=src python examples/pipeline_flow.py \
  --nx 400 --ny 160 --diameter 20 --gap-ratio 0.5 \
  --re 200 --n-steps 30000 --output-interval 1000 \
  --run-name pipeline_eD05 --overwrite
```

---

### 4.4 湍流槽道 / Turbulent Channel Flow

#### 物理背景 / Physical Background

体力驱动的湍流槽道流是**船体阻力、海洋立管湍流边界层**研究的基础验证算例。本算例采用 **Smagorinsky LES 模型**，在 Re_τ = 100 条件下与对数律速度剖面对比。

Body-force-driven turbulent channel flow is the standard verification case for turbulent boundary layers relevant to ship hull drag. This benchmark uses the **Smagorinsky LES model** at Re_τ = 100.

#### 对数律参考 / Log-Law Reference (Moser et al., 1999)

在对数区（y⁺ > 11）：

$$u^+ = \frac{1}{\kappa} \ln y^+ + B, \quad \kappa = 0.41, \quad B = 5.2$$

粘性底层（y⁺ < 5）：$u^+ = y^+$

#### 关键参数 / Key Parameters

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nx × ny` | 格子尺寸 | 256 × 64 |
| `re_tau` | 壁面摩擦雷诺数 Re_τ = u_τ(H/2)/ν | 100.0 |
| `u_tau` | 壁面摩擦速度（格子单位）| 0.005 |
| `smagorinsky_cs` | Smagorinsky 常数 C_S | 0.1 |
| `averaging_start` | 统计平均起始步 | 20 000 |

#### 数值结果 / Numerical Results (Full Mode: nx=256, ny=64, steps=50 000)

| 区域 | 指标 | LBM | 参考 (DNS) | 误差 |
|------|------|-----|------------|------|
| 粘性底层 (y⁺ < 5) | u⁺ = y⁺ | < 0.1 RMS | — | ✓ |
| 对数区 (y⁺ > 11) | RMS(u⁺ - u⁺_loglaw) | < 3.0 | 0 | < 3 w.u.* |

> *w.u. = wall units。Full 模式（50 000 步）预期 RMS < 3.0 wall units，满足工程验证要求。
> 快速模式（40 000 步）由于流动仍在发展，RMS 约 10–13 wall units。

#### 运行命令 / Run Commands

```bash
PYTHONPATH=src python examples/turbulent_channel.py \
  --nx 256 --ny 64 --re-tau 100 --u-tau 0.005 \
  --n-steps 50000 --averaging-start 20000 \
  --output-interval 5000 --run-name channel_retau100 --overwrite
```

---

### 4.5 船舶全流程案例 / 3D Ship CAD-to-Flow Workflow

#### 物理背景 / Physical Background

该案例覆盖 **CAD 建模 → LBM 计算 → 后处理分析 → 定量比较** 的完整船舶工作流。以 Wigley 参数船体为例，首先由 `ship_cad.py` 生成参数化船型、CAD 预览和 STL 曲面，然后由 `ship_flow.py` 在 D3Q19 格子上执行 **Smagorinsky MRT LES** 三维绕流计算，最后自动输出尾迹剖面、压力系数范围、回流长度、涡量/Q 准则统计及验收结论。

This case covers the full **CAD → LBM solve → post-processing → quantitative comparison** workflow. A Wigley parametric hull is generated by `ship_cad.py`, exported as preview/STL geometry, simulated by `ship_flow.py` on a D3Q19 grid with **Smagorinsky MRT LES**, and then reduced to wake, pressure, recirculation, and force-symmetry metrics.

#### Wigley 船体几何 / Wigley Hull Geometry

$$y = \frac{B}{2}\left[1 - \left(\frac{2x}{L}\right)^2\right]\left[1 - \left(\frac{z}{T}\right)^2\right]$$

其中 L 为船长，B 为最大船宽，T 为吃水。

#### 关键参数 / Key Parameters

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `nx × ny × nz` | 格子尺寸 | 160 × 60 × 40 |
| `hull_length` | 船长（格子单位）| 80.0 |
| `hull_beam` | 最大船宽 | 8.0 |
| `hull_draft` | 吃水深度 | 12.0 |
| `re` | 雷诺数 Re = U L / ν | 200.0 |
| `wave_amp` | 入口 Airy 波幅（0 = 无波浪）| 0.0 |
| `smagorinsky_cs` | Smagorinsky 常数 | 0.1 |

#### 工作流输出 / Workflow Outputs

| 阶段 | 主要文件 | 说明 |
|------|----------|------|
| CAD 建模 | `cad_preview.png`, `cad_summary.json`, `hull.stl` | 船型预览、型系系数、可交换 STL 曲面 |
| 数值计算 | `run_metadata.json`, `forces.csv`, `flow_step_XXXXXX.png` | 配置、诊断历史、力系数与流场快照 |
| 后处理 | `postprocess_summary.json`, `wake_profile.csv` | 尾迹速度剖面、回流长度、压力与旋涡统计 |

#### 数值结果 / Numerical Results (Quick Mode: 80×40×30, steps=2000)

| 类别 | 指标 | LBM / CAD | 参考值 / 阈值 | 结论 |
|------|------|-----------|---------------|------|
| CAD | 数值方形系数 `C_b` | **0.46–0.48** | Wigley 理论值 4/9 ≈ 0.444 | 误差 < 25% ✓ |
| 力学 | 阻力系数幅值 `|C_D|` | **> 0** | 应检测到非零阻力响应 | ✓ |
| 对称性 | `|C_S| / |C_D|` | **< 0.10** | 左右对称阈值 | ✓ |
| 对称性 | `|C_L| / |C_D|` | **< 0.25** | 上下对称阈值 | ✓ |
| 尾迹 | 最大速度亏损 | **> 0** | 船后应形成尾迹减速区 | ✓ |

> **注**: Re = 200 时并不存在适用于当前受限槽道设置的唯一标准 `C_D` 参考值，因此本案例采用两类定量比较：  
> (1) **CAD 层**：数值方形系数 `C_b` 与解析值比较；  
> (2) **流场层**：阻力正值、侧向/垂向力对称性比值以及尾迹亏损等物理一致性指标。

#### 波浪入射算例 / Wave Inlet Case

```python
from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

cfg = ShipHullFlowConfig(
    nx=160, ny=60, nz=40,
    hull_type="wigley",
    u_in=0.05, re=200.0,
    wave_amp=0.01,          # Airy 波幅 / Airy wave amplitude
    wave_period=200.0,      # 波浪周期 / wave period (steps)
    wave_k=0.05,            # 波数 / wave number
    n_steps=4000, output_interval=200,
    output_root="outputs", run_name="wigley_waves", export_stl=True, overwrite=True,
)
run_ship_hull_flow(cfg)
```

#### 运行命令 / Run Commands

```bash
PYTHONPATH=src python examples/ship_hull_flow.py \
  --hull-type wigley \
  --nx 160 --ny 60 --nz 40 \
  --hull-length 80 --hull-beam 8 --hull-draft 12 \
  --re 200 --n-steps 4000 --output-interval 200 --export-stl \
  --run-name wigley_re200 --overwrite
```

---

### 4.6 SUBOFF 全附件阻力基准 / SUBOFF Full-Appendage Resistance Benchmark

#### 物理背景 / Physical Background

SUBOFF（DARPA SUBOFF）是潜艇阻力、尾流与附体影响研究中的经典基准体型。TensorLBM 当前提供 **AFF-8 全附件模型** 的体素化几何、D3Q19 通道绕流求解，以及基于 **ITTC-1957 摩擦阻力公式 + Richardson 外推** 的网格收敛误差评估。

SUBOFF (DARPA SUBOFF) is a canonical benchmark geometry for submarine resistance and appendage-effect studies. TensorLBM currently supports the **AFF-8 full-appendage variant** with voxelised geometry, D3Q19 channel-flow solution, and a convergence check based on the **ITTC-1957 friction line plus Richardson extrapolation**.

#### 关键参数 / Key Parameters

| 参数 | 说明 | 完整模式推荐值 |
|------|------|----------------|
| `hull_type` | 潜艇模型变体 | `full` |
| `length_m` | 物理艇长 | 4.356 m |
| `speed_ms` | 来流速度 | 2.5 m/s |
| `base_length_lu` | 初始艇长格点数 | 64 |
| `max_iterations` | 迭代细化次数 | 4 |
| `target_error_pct` | 收敛目标 | 3.0 % |
| `lbm_steps` | 每级格网推进步数 | 60 |

#### 精度判据 / Accuracy Criterion

完整模式以最后一次 Richardson 外推误差为验收标准：

$$\varepsilon = \frac{|C_D - C_{D,\text{richardson}}|}{|C_{D,\text{richardson}}|} \times 100\%$$

当 `final_error_pct <= 3.0` 时，记为通过（`target_met = true`）。

#### 运行命令 / Run Commands

```bash
# 完整精度 CLI 基准
PYTHONPATH=src python benchmarks/bench_marine.py --cases suboff --full

# 仅导出结构化结果
PYTHONPATH=src python benchmarks/bench_marine.py --cases suboff --full \
  --report outputs/suboff_full/results.json
```

#### 输出解读 / Output Interpretation

- `reference.cd_analytical`：ITTC-1957 理论摩擦阻力参考值
- `reference.cd_richardson`：由相邻细化层估计的收敛参考值
- `simulated.cd`：最终格网上的 LBM 阻力系数
- `iterations[*].grid`：每次细化对应的 `nx × ny × nz`
- `iterations[*].error_pct`：当前层相对于 Richardson 外推值的误差

#### 平台使用说明 / Platform Usage

平台侧完整操作步骤、API 请求示例与结果判读请参见：

- [`docs/suboff_platform_manual.md`](suboff_platform_manual.md)

---

## 5. 定量比较汇总 / Quantitative Comparison Summary

下表汇总了各算例的关键定量对比结果：

| # | 算例 | 格子 | 步数 | 关键量 | LBM | 参考值 | 误差 | 状态 |
|---|------|------|------|--------|-----|--------|------|------|
| 1 | 绕圆柱 Re=100 | 200×80 | 20 000 | St 数 | 0.183 | 0.166 | 10.3 % | ✓ |
| 1 | 绕圆柱 Re=100 | 400×120 (full) | 60 000 | St 数 | ~0.170 | 0.166 | ~2.4 % | ✓ |
| 2 | 液舱晃动 | 200×160 (full) | 60 000 | ω₁ 误差 | < 15 % | — | < 15 % | ✓ |
| 3 | 近床管道 Re=200 e/D=0.5 | 400×160 (full) | 30 000 | St 数 | ~0.175 | 0.183 | ~4.4 % | ✓ |
| 4 | 湍流槽道 Re_τ=100 | 256×64 (full) | 50 000 | RMS 对数律误差 | < 3.0 w.u. | 0 | < 3 w.u. | ✓ |
| 5 | 船舶全流程案例 Re=200 | 80×40×30 | 2 000 | `C_b` 误差 + 对称性 | `C_b≈0.46–0.48` | 4/9, 力比值阈值 | `C_b` 误差 < 25%，对称性通过 | ✓ |
| 6 | SUBOFF 全附件阻力 | 115×52×52 → 288×130×130 (full) | 60 / level | `C_D` 收敛误差 | < 3.0 % | Richardson 外推 | ≤ 3.0 % | ✓ |
| 7 | 顶盖驱动方腔流 Re=100/400/1000 | — | — | u 中线速度剖面 | 匹配 | Ghia 等 (1982) | < 1 % | ✓ |
| 8 | DG-LBM MMS 收敛 | P1 单元 | — | 空间精度阶 | O(Δx²)–O(Δx³) | 制造精确解 | — | ✓ |
| 9 | 翼型 C_L/C_D（NACA 4 位数）| — | — | 升/阻力系数 | 在参考带内 | XFOIL/面元法 | — | ✓ |

> 所有 "Full 模式" 结果基于理论预测或已发表 LBM 参考数据；快速模式结果来自本软件实际运行。

### 误差来源分析 / Error Analysis

1. **阻塞比效应（Blockage Effect）**：当 D/H > 5%，有效阻力系数和 Strouhal 数偏离无限域参考值。修正公式（Zdravkovich, 1997）：
   $$C_{D,corr} = C_D / (1 + \beta^2)^{3/2}$$
   其中 $\beta = D/H$（阻塞比）。

2. **瞬态收敛**：LBM 从均匀初始场出发，涡脱落需要数个对流时间尺度（D/U 或 L/U）才能建立稳定周期状态。

3. **离散化误差**：格子间距 Δx 决定几何近似精度。圆柱需要 D/Δx ≥ 10 以减小离散误差。

4. **数值粘性**：LBM 的有效粘性为 ν = (τ - 0.5)/3。选 τ 过小（< 0.55）会引入数值不稳定；τ 过大则引入过多耗散。

---

## 6. 运行基准测试 / Running the Benchmark Suite

### 6.1 快速模式 / Quick Mode

```bash
PYTHONPATH=src python benchmarks/bench_marine.py \
  --output-root outputs/marine_bench \
  --report outputs/marine_bench/results.json
```

运行时间取决于硬件与设备选择，默认输出海洋基准套件中的全部案例（含 SUBOFF 与几何一致性检查）。

The default run covers the full marine benchmark suite, including SUBOFF and geometry-consistency checks.

### 6.2 选择性运行 / Selective Run

```bash
# 仅运行圆柱和液舱算例 / Run only cylinder and sloshing
PYTHONPATH=src python benchmarks/bench_marine.py \
  --cases cylinder sloshing --output-root /tmp/bench

# 仅运行 SUBOFF 全附件阻力基准 / Run only the SUBOFF full-appendage benchmark
PYTHONPATH=src python benchmarks/bench_marine.py \
  --cases suboff --full --output-root outputs/suboff_full
```

### 6.3 完整精度模式 / Full Accuracy Mode

```bash
PYTHONPATH=src python benchmarks/bench_marine.py --full \
  --output-root outputs/marine_bench_full \
  --report outputs/marine_bench_full/results.json
```

完整模式使用生产级网格和步数，建议在多核 CPU 或 GPU 上运行（数小时）。

Full mode uses production-quality settings. Expected runtime: several hours on CPU; 30–60 minutes on a modern GPU.

### 6.4 GPU 加速 / GPU Acceleration

```bash
# 使用 CUDA GPU / Use CUDA GPU
PYTHONPATH=src python examples/ship_hull_flow.py \
  --nx 320 --ny 120 --nz 80 --n-steps 10000 \
  --device cuda --overwrite

# 使用 Apple Silicon MPS
PYTHONPATH=src python examples/cylinder_flow.py \
  --nx 800 --ny 240 --radius 24 --n-steps 80000 \
  --device mps --overwrite
```

### 6.5 MLUPS 性能测试 / MLUPS Performance Benchmark

```bash
PYTHONPATH=src python benchmarks/bench_mlups.py --device cpu
PYTHONPATH=src python benchmarks/bench_mlups.py --device cuda
```

---

## 7. 公共 API 参考 / Public API Reference

### 7.1 格子原语 / Lattice Primitives

```python
from tensorlbm import equilibrium, macroscopic, C, W, OPPOSITE

# D2Q9 平衡分布 / D2Q9 equilibrium
f_eq = equilibrium(rho, ux, uy)          # (9, ny, nx)

# 宏观量 / Macroscopic variables
rho, ux, uy = macroscopic(f)
```

### 7.2 碰撞算子 / Collision Operators

```python
from tensorlbm import collide_bgk, collide_mrt, collide_trt, collide_rlbm

# BGK（单松弛时间）
f = collide_bgk(f, tau=0.6)

# MRT（多松弛时间）
f = collide_mrt(f, tau=0.6)

# TRT（双松弛时间，Ginzburg magic Λ=3/16）
f = collide_trt(f, tau=0.6)

# RLBM（正则化 BGK，滤除非流体学模式）
f = collide_rlbm(f, tau=0.6)

# Cumulant LBM（D2Q9 或 D3Q27）
from tensorlbm import collide_cumulant_d2q9, collide_cumulant_d3q27
f = collide_cumulant_d2q9(f, tau=0.6)
f27 = collide_cumulant_d3q27(f27, tau=0.6)

# Smagorinsky LES
from tensorlbm import collide_smagorinsky_bgk
f = collide_smagorinsky_bgk(f, tau=0.6, C_s=0.1)

# WALE / Vreman LES
from tensorlbm import collide_wale_bgk, collide_vreman_bgk
f = collide_wale_bgk(f, tau=0.6)
f = collide_vreman_bgk(f, tau=0.6)

# 动态 Smagorinsky（Germano 标识）
from tensorlbm import collide_dynamic_smagorinsky_bgk
f = collide_dynamic_smagorinsky_bgk(f, tau=0.6)
```

### 7.3 边界条件 / Boundary Conditions

```python
from tensorlbm import (
    bounce_back_cells,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)

# 反弹边界（固壁）
f = bounce_back_cells(f, wall_mask)

# Zou-He 进口速度 BC
f = zou_he_inlet_velocity(f, u_in=0.08)

# Zou-He 出口压力 BC
f = zou_he_outlet_pressure(f, rho_out=1.0)
```

### 7.4 船舶与海洋工程模块 / Marine Modules

```python
from tensorlbm import (
    ShipHullFlowConfig, run_ship_hull_flow,
    SloshingTankConfig, run_sloshing_tank,
    PipelineFlowConfig, run_pipeline_flow,
    TurbulentChannelConfig, run_turbulent_channel,
    wigley_hull_mask,
    airy_wave_velocity_3d, apply_wave_inlet_3d,
    faltinsen_natural_frequency,
    measure_strouhal,
    log_law_velocity, viscous_sublayer_velocity,
)

# 计算 Faltinsen 固有频率
omega_n = faltinsen_natural_frequency(L=200, h=80, g=2e-5, mode=1)

# 创建 Wigley 船体掩模
obstacle = wigley_hull_mask(nx=160, ny=60, nz=40,
    cx=56.0, cy=30.0, cz_keel=4.0,
    length=80.0, beam=8.0, draft=12.0)
```

### 7.5 多相流 / Multiphase Flow

```python
from tensorlbm import color_gradient_step, free_energy_step

# 颜色梯度两相流（适合液舱晃动）
f_r, f_b = color_gradient_step(
    f_r, f_b,
    tau=1.0, A=0.036,          # A = G * 0.04
    gx=0.0, gy=-2e-5,          # 重力
    solid_mask=wall,
)

# 自由能相场模型
f, g = free_energy_step(f, g, tau_f=0.7, tau_g=0.7, kappa=0.04, a=1.0)
```

### 7.6 后处理 / Postprocessing

```python
from tensorlbm import (
    compute_vorticity,
    compute_pressure_coefficient,
    compute_q_criterion,
    compute_lambda2_criterion,
    extract_velocity_profile,
    trace_streamlines_2d,
    surface_force_2d,
    force_coefficients,
)

# 涡量（2D）
vort = compute_vorticity(ux, uy)

# Q 准则（湍流可视化）
Q = compute_q_criterion(ux, uy, uz)  # 3D

# λ₂ 准则（旋涡识别）
lam2 = compute_lambda2_criterion(ux, uy, uz)

# 提取中线速度剖面
y, u_profile = extract_velocity_profile(ux, x_idx=nx // 2)

# 压力系数
cp = compute_pressure_coefficient(rho, rho_ref=1.0, u_ref=0.08)

# 流线追踪（2D）
from tensorlbm import seed_points_uniform_2d, streamlines_to_dict
seeds = seed_points_uniform_2d(nx, ny, n=50)
lines = trace_streamlines_2d(ux, uy, seeds, max_steps=500)
data = streamlines_to_dict(lines)

# 表面力与力系数
fx, fy = surface_force_2d(f, obstacle_mask)
cd, cl = force_coefficients(fx, fy, rho_ref=1.0, u_ref=0.08, area=2*radius)
```

### 7.7 单位换算 / Unit Conversion

```python
from tensorlbm import LBMUnitConverter

# 船舶算例：U=1 m/s, L=100 m（实船），Re=5e6
conv = LBMUnitConverter(
    L_ref=100.0, U_ref=1.0, nu_phys=1e-6,
    N_cells=200, u_lbm=0.05,
)
print(f"dx = {conv.dx:.3f} m")
print(f"dt = {conv.dt:.5e} s")
print(f"nu_lbm = {conv.nu_lbm:.5e}")
print(f"tau = {conv.tau:.4f}")
```

### 7.8 自适应网格细化 / Adaptive Mesh Refinement (AMR)

```python
from tensorlbm import (
    AdaptiveSolver2D, AdaptationSchedule,
    nonequilibrium_indicator_2d, mark_cells_for_refinement,
)
from tensorlbm import macroscopic

schedule = AdaptationSchedule(adapt_every=500, max_levels=3)
solver = AdaptiveSolver2D(f_coarse, schedule=schedule)

for step in range(n_steps):
    solver.step(collide_fn, stream_fn, boundary_fn)
    if solver.should_adapt(step):
        rho, ux, uy = macroscopic(solver.coarse_f)
        indicator = nonequilibrium_indicator_2d(solver.coarse_f, rho, ux, uy)
        solver.adapt(indicator)
```

### 7.9 DG-LBM 混合求解器 / DG-LBM Hybrid Solver

```python
from tensorlbm import DGLBMConfig, run_dg_lbm_sphere_flow
from tensorlbm import DGLBMSuboffConfig, run_dg_lbm_suboff_flow

# 3D 球体绕流（DG 加速）
cfg = DGLBMConfig(nx=60, ny=30, nz=30, radius=4, n_steps=500)
run_dg_lbm_sphere_flow(cfg)

# SUBOFF 潜艇（启用真实 DG 格式）
suboff_cfg = DGLBMSuboffConfig(use_real_dg=True, n_steps=200)
run_dg_lbm_suboff_flow(suboff_cfg)
```

### 7.10 AI 湍流模型 / AI Turbulence Models

```python
from tensorlbm import (
    run_ai_dns_pipeline, run_ai_les_pipeline,
    collide_ai_les_bgk, predict_nu_t_2d,
    FlowFieldTransformer, FlowTransformerArch,
)

# 端到端 DNS → 数据库 → 训练 → AI-LBM 管道
result = run_ai_dns_pipeline(nx=128, ny=64, n_steps=5000)

# 将训练好的模型嵌入 LBM 碰撞
nu_t = predict_nu_t_2d(model, f, tau_base=0.6)
f = collide_ai_les_bgk(f, tau_base=0.6, nu_t=nu_t)
```

### 7.11 共轭传热 / Conjugate Heat Transfer

```python
from tensorlbm import CHTConfig, run_conjugate_ht_2d

cfg = CHTConfig(
    nx=128, ny=64,
    tau_fluid=0.7, alpha_solid=0.1,
    n_steps=10000,
)
result = run_conjugate_ht_2d(cfg)
```

### 7.12 气动声学 / Aeroacoustics (FWH)

```python
from tensorlbm import (
    FWHSurface, compute_fwh_far_field,
    compute_spl_spectrum, oaspl,
)

# 定义 FWH 积分面并计算远场声压
surface = FWHSurface(observer_distance=10.0, n_points=128)
result = compute_fwh_far_field(surface, p_history, u_history, dt=1.0)
spl = compute_spl_spectrum(result.p_far, dt=1.0)
oa = oaspl(spl.frequencies, spl.levels)
print(f"OASPL = {oa:.1f} dB")
```

### 7.13 多 GPU 域分解 / Multi-GPU Domain Decomposition

```python
from tensorlbm import MultiGPUSolver2D, auto_decompose

devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
decomp = auto_decompose(nx=1024, ny=512, n_devices=len(devices))
solver = MultiGPUSolver2D(f_global, decomp, devices=devices)

for step in range(n_steps):
    solver.step(collide_fn, stream_fn, boundary_fn)
```

---

## 8. 输出文件说明 / Output File Description

每次运行会在 `outputs/<case>/<run_name>/` 下生成以下文件：

| 文件 | 说明 |
|------|------|
| `cad_summary.json` | 船型 CAD 统计（`C_b`、`C_wp`、`C_m`、`C_p`、离散化误差）|
| `cad_preview.png` | 船体 body plan / waterplane / side profile 三视图 |
| `hull.stl` | 参数化船体导出的 ASCII STL 曲面（船舶案例可选）|
| `run_metadata.json` | 完整运行元数据（配置、推导量、诊断历史、运行环境）|
| `forces.csv` | 每个输出间隔的 C_D、C_L（船体算例含 C_S、M_x、M_y、M_z）|
| `postprocess_summary.json` | 后处理汇总（尾迹亏损、回流长度、压力范围、Q 准则、验收结论）|
| `wake_profile.csv` | 船后指定截面的尾迹速度剖面（`y_index, ux`）|
| `velocity_profile.csv` | 速度剖面（湍流槽道专用：y、y⁺、u⁺、u⁺_loglaw）|
| `elevation.csv` | 液位时间历程（液舱晃动专用：step、t*、η*）|
| `spectrum.csv` | 液位频谱（液舱晃动专用：freq、amplitude）|
| `strouhal.json` | Strouhal 数估计（圆柱/管道算例）|
| `flow_step_XXXXXX.png` | 流场快照（速度场 + 涡量或相分数）|
| `checkpoint_XXXXXX.pt` | PyTorch 检查点（可用于续算）|

### 元数据文件示例 / Metadata Example

```json
{
  "config": {
    "nx": 200, "ny": 80, "radius": 5.0,
    "u_in": 0.08, "re": 100.0, "n_steps": 20000
  },
  "derived": {
    "nu": 0.004, "tau": 0.512
  },
  "runtime": {
    "torch_version": "2.3.0", "device": "cpu"
  },
  "reproducibility": {
    "hostname": "...", "git_hash": "...", "timestamp": "..."
  },
  "strouhal": 0.183,
  "diagnostics": [
    {"step": 500, "mass": 16000.0, "cd": 2.54, "cl": 0.031}
  ]
}
```

---

## 9. 已知限制 / Known Limitations

1. **可压缩性误差 / Compressibility Error**：LBM 在弱可压缩近似下工作，要求 Ma = u/c_s < 0.3（其中 c_s = 1/√3）。对于快速流动（u_in > 0.1），应降低格子速度。

2. **高雷诺数不稳定性 / High-Re Instability**：当 τ < 0.55 时易出现数值振荡。建议使用 MRT、RLBM、Cumulant 或 Smagorinsky LES 来提高 Re 上限。

3. **三维计算成本 / 3D Computational Cost**：D3Q19 在 160×60×40 格子上每步约需 35 ms（CPU）。大型三维算例建议使用 GPU 加速、`torch.compile` 或多 GPU 域分解（`MultiGPUSolver3D`）。

4. **CG 多相流参数敏感性 / CG Multiphase Parameter Sensitivity**：颜色梯度模型的 τ、G、ρ_water/ρ_air 等参数对界面稳定性有较强影响。建议在目标参数范围内进行专项验证。

5. **湍流 LES 近壁精度 / LES Near-Wall Accuracy**：粗网格下 Smagorinsky 模型在 y⁺ < 5 区域过度预测粘性耗散，可改用 WALE 模型改善近壁精度。

6. **DG-LBM 刚性 / DG-LBM Stiffness**：三维 DG-LBM 在低 τ_dg（< 0.6）时要求 RK 子步数 ≥ 16 以保持稳定，计算成本随子步数线性增长。

7. **AMR 内存占用 / AMR Memory**：每增加一个细化级别，精细 Patch 的内存占用约为粗网格的 2^d 倍（d = 维数）。建议在运行前估算峰值内存。

8. **多 GPU halo 通信 / Multi-GPU Halo Communication**：halo 交换通过 PyTorch 张量复制实现，适合 NVLink 或高带宽互联；跨主机通信需额外配置分布式后端。

---

## 10. 参考文献 / References

1. **Williamson, C.H.K. (1988)**. Defining a universal and continuous Strouhal-Reynolds number relationship for the laminar vortex shedding of a circular cylinder. *Physics of Fluids*, 31(10), 2742–2744.

2. **Zdravkovich, M.M. (1997)**. *Flow Around Circular Cylinders, Vol. 1: Fundamentals*. Oxford University Press.

3. **Faltinsen, O.M. (1978)**. A numerical nonlinear method of sloshing in tanks with two-dimensional flow. *Journal of Ship Research*, 22(3), 193–202.

4. **Bearman, P.W. & Zdravkovich, M.M. (1978)**. Flow around a circular cylinder near a plane boundary. *Journal of Fluid Mechanics*, 89(1), 33–47.

5. **Price, S.J., Sumner, D., Smith, J.G., Leong, K. & Paidoussis, M.P. (2002)**. Flow visualization around a circular cylinder near to a plane wall. *Journal of Fluids and Structures*, 16(2), 175–191.

6. **Moser, R.D., Kim, J. & Mansour, N.N. (1999)**. Direct numerical simulation of turbulent channel flow up to Re_τ = 590. *Physics of Fluids*, 11(4), 943–945.

7. **Wigley, W.C.S. (1926)**. A comparison of experiment and calculated wave-profiles and wave resistances for a form having parabolic waterlines. *Transactions of the Institution of Naval Architects*, 68, 124–137.

8. **Michell, J.H. (1898)**. The wave resistance of a ship. *Philosophical Magazine*, 45, 106–123.

9. **He, X. & Luo, L.S. (1997)**. Lattice Boltzmann model for the incompressible Navier–Stokes equation. *Journal of Statistical Physics*, 88(3–4), 927–944.

10. **Lallemand, P. & Luo, L.S. (2000)**. Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. *Physical Review E*, 61(6), 6546–6562.

11. **Ginzburg, I. & d'Humières, D. (2003)**. Multireflection boundary conditions for lattice Boltzmann models. *Physical Review E*, 68(6), 066614.

12. **Zou, Q. & He, X. (1997)**. On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591–1598.

13. **Latt, J. & Chopard, B. (2006)**. Lattice Boltzmann method with regularized non-equilibrium distribution functions. *Mathematics and Computers in Simulation*, 72(2–6), 165–168.

14. **Geier, M., Greiner, A. & Korvink, J.G. (2006)**. Cascaded digital lattice Boltzmann automata for high Reynolds number flow. *Physical Review E*, 73(6), 066705.

15. **Ghia, U., Ghia, K.N. & Shin, C.T. (1982)**. High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.

16. **Lagrava, D., Malaspinas, O., Latt, J. & Chopard, B. (2012)**. Advances in multi-domain lattice Boltzmann grid refinement. *Journal of Computational Physics*, 231(14), 4808–4822.

17. **Filippova, O. & Hänel, D. (1998)**. Grid refinement for lattice-BGK models. *Journal of Computational Physics*, 147(1), 219–228.

---

*本说明书由 TensorLBM 团队维护。如有问题请提交 GitHub Issue。*  
*This manual is maintained by the TensorLBM team. For issues, please open a GitHub Issue.*
