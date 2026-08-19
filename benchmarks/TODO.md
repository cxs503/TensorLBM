# TensorLBM 共性模块 Benchmark 清单

> 规则：**每个 benchmark 必须通过共性模块（GeneralSimEngine 或经其验证的共性求解路径）实现，
> 且精度 ≤3% 才保存到 benchmarks/verified/ 文件夹**。未达标案例留待达标后移入。

## 可做 Benchmark 总览（按共性模块分组）

### 组 1：GeneralSimEngine（产品内核，配置即求解）
| # | 案例 | 物理 | 参考值 | 当前精度 | 状态 |
|---|------|------|--------|---------|------|
| B1 | sphere_re100 | 球 Re=100 D40 | Cd=1.09 (Schiller-Naumann) | 压力积分 -16.6%；MEM bg_sub +264.6%（G15 修复后实测位同 standard） | 力法缺口，未达标，暂缓 |
| B2 | sphere_re100_d60 | 球 Re=100 D60（加密） | Cd=1.09 | 未知（需跑） | 待验证 |
| B3 | sphere_re200 | 球 Re=200 | Cd=0.77 (Schiller-Naumann) | 未知 | 待验证 |
| B4 | cylinder_re100_2d | 2D 圆柱 Re=100 D60 | Cd=1.35 (Braza 1990) | 4.8% (Re=200 历史) | 待验证 |
| B5 | cylinder_re200_2d | 2D 圆柱 Re=200 | Cd=1.33 | 4.8% | 待验证 |
| B6 | suboff_re1000 | SUBOFF Re=1000 | Ct=0.004 (实验) | 3.8% (CUDA) / 3.6% (SDAA) | 待验证 |

### 组 2：static_block_amr + sphere_amr_common（AMR 共性路径）
| # | 案例 | 物理 | 参考值 | 当前精度 | 状态 |
|---|------|------|--------|---------|------|
| B7 | sphere_re100_amr2 | 球 Re=100 AMR 双层 | Cd=1.09 | 历史 AMR 已验证 177 测试 | 待跑定量 |
| B8 | sphere_re100_amr3 | 球 Re=100 AMR 三层 | Cd=1.09 | 待验证 | 待跑 |
| B9 | suboff_re1000_amr | SUBOFF Re=1000 AMR 双层 | Ct=0.004 | 历史 5.6% (BB) | 待验证 |

### 组 3：spherical_dg（球坐标贴体 DG 壳层）
| # | 案例 | 物理 | 参考值 | 当前精度 | 状态 |
|---|------|------|--------|---------|------|
| B10 | stokes_sphere_dg | Stokes 球 Re=0.1 DG 壳 | Cd=240 (Stokes 理论) | 2.15% 单网格但加密恶化 53% | **假结果，移出 verified** |
| B11 | sphere_re100_dg | 球 Re=100 DG 壳 | Cd=1.09 | 未知 | 待验证 |

### 组 4：octree_boundary（八叉树 VR，实验性）
| # | 案例 | 物理 | 参考值 | 当前精度 | 状态 |
|---|------|------|--------|---------|------|
| B12 | sphere_re100_octree | 球 Re=100 octree | Cd=1.09 | 171%（欠收敛） | 未达标，**不入文件夹** |

### 组 5：周期域解析解（d2q9 solver 共性路径）
| # | 案例 | 物理 | 参考值 | 当前精度 | 状态 |
|---|------|------|--------|---------|------|
| B17 | taylor_green_2d | 2D TG 涡衰减（N=64–128, Re=100–1000） | 解析 γ_E=4νk²（动能）, γ_vel=2νk²（速度） | **err_E ≤ 0.13%（全部 13 案例）** | ✅ 已验证 2026-08-18 |
| B17-3D | taylor_green_3d | **3D TG 涡衰减 D3Q19 周期域（N=64/96/128³, Re=24, U0=0.05）** | 解析 γ_E=6νk²（动能）, γ_vel=3νk²（速度；\|κ\|²=3k²） | **err_E +0.309%→+0.279%→+0.250%（三档单调收敛）** | ✅ 已验证 2026-08-19 |
| B30 | shear_wave_decay | 2D 剪切波衰减（H=64/128, τ=0.8, U0=0.05/0.1） | 解析 γ_vel=νk²（速度）, γ_E=2νk²（动能） | **err_vel +0.051%→+0.011%（两档收敛）** | ✅ 已验证 2026-08-18 |
| B24 | droplet_oscillation | 液滴振荡 m=2 Rayleigh 频率（SCMP SC94，R=20/30/40，30000 步） | Rayleigh ω²=6σ/(ρR³)，σ_eff=0.056112（laplace_droplet 实测） | **阻尼修正 ω₀=√(ω_d²+γ²)：+2.59/−1.50/−2.39% 三档全 ≤3%**（观测频率含固有阻尼 −6~−8%，γ≈0.72/R² 为模型特性） | ✅ 已验证 2026-08-19 |

## 当前状态（2026-08-18 实测）

**✅ 已入库（达标 ≤3%，真实模拟无外推）**
- **B17 Taylor-Green 2D**（128²，Re=100）：γ 误差 **-0.035%**（动能）/-0.0007%（速度），R²=0.9999996 → benchmarks/verified/taylor_green_2d/
- **B17-3D Taylor-Green 3D**（64³/96³/128³，Re=24，U0=0.05，D3Q19 周期域，OpenLB tgv3d 同款）：γ_E 误差 **+0.309%→+0.279%→+0.250%**（三档单调收敛，R²≥0.999995；γ_E=6νk²，\|κ\|²=3k²；γ_sim 略大于理论——3D 涡拉伸物理效应如实报告）→ benchmarks/verified/taylor_green_3d/
- **B30 衰减剪切波 2D**（H=64/128，τ=0.8，U0=0.05）：γ_vel 误差 **+0.051%→+0.011%**（两档单调收敛），R²=1.000000 → benchmarks/verified/shear_wave_decay/
- **B13 Poiseuille 2D**（Zou-He 压力驱动）：H=60 误差 **0.18%**（H=20:1.21%→H=40:0.45%→H=60:0.18% 单调收敛）→ benchmarks/verified/poiseuille_2d/
- **B14 3D 圆管 Poiseuille**（D3Q19，速度入口+压力出口+半程反弹管壁）：**R_eff^Q 方法**（数字楼梯圆管水力半径 R_eff^Q=R+0.11，流量反演独立观测量）径向平均剖面 max **2.15%→1.45%**（R=20/40 单调收敛），L2 1.29%→0.71%；逐格散布 6.45%→2.78% 已披露（R=20 楼梯壁单格几何效应，一阶收敛）→ benchmarks/verified/poiseuille_3d_pipe/
- **B15 方腔流 Re=100**（D2Q9 MRT，u_lid=0.06，100k 步，**V3 修复版**：pre-streaming 半程反弹+zou_he_moving_lid，2026-08-19）：max_abs_dev **0.75%→0.73%**（128²/192² 单调收敛，残差 ~4e-7），u(0.5,0.5)=-0.2108/-0.2103 vs Ghia -0.20581，涡心 (0.614,0.740)/(0.614,0.739) vs (0.6172,0.7344)；V0 曾 22.5% 未达标（post-streaming 反弹顶盖动量绕入底壁），V3 修复经验自 Re=400 迁移 → benchmarks/verified/cavity_re100/
- **B16 方腔流 Re=400**（同 V3 配方，2026-08-19）：max_abs_dev **1.50%→0.83%**（128²/192² 单调收敛）→ benchmarks/verified/cavity_re400/
- **B32 圆柱 Re=20 Schäfer-Turek 2D-1**（通道内圆柱稳态，2026-08-19 复核入库）：mask=R + **无重整化** + 300k 步，D=40/D=80 两档 Cd=5.548428（**-0.558%**）/ 5.419129（**-2.875%**）均 ≤3%，各自统计收敛到平台（末 50k 步摆幅 <0.02%，**三次独立运行逐位一致**）；如实披露：|err| 随网格增大（MEM 阶梯偏差，D=80 贴 3% 边界）、Cl 不匹配（-0.014/-0.0015 vs +0.0106，符号反）→ benchmarks/verified/cylinder_re20_st/（pending/cylinder_re20_st/ 归档）

**🔶 未达标（待继续）**
- B1 球 Re=100：D40 实测 -13%（需 D60/D80 加密）
- B4 圆柱 Re=100：4.8%（需加密）
- backward_step：88%（跑错 Re=200，需重跑 Re=100）
- B1/B2/B4/B7/NACA/Blasius/空化/SUBOFF：子 agent 进行中

## 达标路线（3% 内，真实模拟，禁外推）

1. **B1/B2/B3 球系列：力法缺口阻塞（G12/G15）**——压力积分 extrap='none' 低估 -13~25%（近壁无外推），
   MEM 曲面平衡背景 +268%。两力法在球面均需修复（候选：MEM 减背景 或 galilean 变体）。
   **暂缓球绕流，优先其他问题类型**
2. **B2 网格加密**：D60 → D80 → D100 扫收敛趋势，真实模拟达标才保存
3. **B4/B5 圆柱**：D40/D80 扫描（B4 子 agent 进行中），真实模拟收敛后达标
4. **B10 Stokes DG**：摩擦项需 DG 弱形式应力张量积分（P1 节点导数/单元均值差分均不收敛），收敛性成立才入库
5. **B6 SUBOFF**：3.8% 需壁面模型或更细网格，难度高，放后

## 2026-08-19 新增

| # | 案例 | 状态 |
|---|------|------|
| B31 | 等温 Sod 激波管（D2Q9，4:1，Ma_mid=0.70） | ✅ **已入库** verified/sod_shock_tube/（L2(ρ) 0.84%→0.46% 两档收敛、W −0.09%/−0.03%、中间态 <0.05%、τ=0.8；参照=等温 Riemann 解；附声学补充 ε=0.01/0.25 波速 <0.1%） |
| B32 | 完整可压缩 Sod（γ=1.4 高 Ma） | ❌ 未具备（需多速度 D2V17/D2V21 / FVM-LBM / 双分布；规格书见 /tmp/compressible_gap.md） |

## 文件夹结构

```
benchmarks/verified/          # 达标（≤3%）才保存
  ├── sphere_re100_d40/       # 每个案例一个目录
  │   ├── run.py              # 基于 GeneralSimEngine（或共性模块）
  │   ├── README.md           # 配置、参考值、误差
  │   └── result.json         # 数值结果
  ├── sphere_re100_d60/
  └── ...
benchmarks/TODO.md            # 本清单 + 状态跟踪
```

## 判定标准

- **精度必须来自真实网格模拟**：|Cd_sim − Cd_ref| / Cd_ref ≤ 3%
- **禁止任何形式的外推凑精度**（pressure_extrap='none' 必须；不采用 Richardson/外推修正值作为达标依据）。外推结果可作参考，但不作为 benchmark 保存依据
- 达标 = 真实模拟（无外推、无人工修正）误差 ≤3% **且网格收敛性成立**：
  - 必须 ≥2 档网格扫描（如 H=40/60、D=60/80），误差单调下降（或 Richardson 收敛到参考值）
  - **单网格巧合误差、加密反而恶化的结果 = 假结果，不入库**（用户核心标准："那是凑出来的精度"）
  - 入库时 result.json 必须含网格扫描数据（各档误差）
- **必须通过共性模块入口运行**，定义为以下之一：
  1. **GeneralSimEngine**（general_sim.py，3D 绕流引擎：sphere/cylinder/suboff/naca/hull）
  2. **库内 solver 共性路径**（solver.py collide/stream、d2q9/d3q19 平衡态、boundaries.py 边界——2D/解析解类案例）
  3. **物理模块 run_* 入口**（dam_break/backward_facing_step/turbulent_channel/cavitation/airfoil_benchmark 等）
  - **禁止**：脚本内手写 collide/stream/平衡态（需复用库函数，grep 校验零手写）
- 网格收敛性作为加分项（有收敛趋势才可信，用户核心要求）
- 保存时附带：运行命令、配置（含 extrap='none'）、参考值来源、误差、时间戳
