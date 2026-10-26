# 平台报告：cavity_3d_full 图档补强（BENCH-CLOSE figdoc m93）

## 1. 物理问题表述

全 3D 立方方腔流（Re=1000）：5 个静止壁面 + 1 个移动顶盖（沿 +x 方向匀速 u_lid=0.06），
腔体三个方向均为有限长度（Lx=Ly=Lz=H，无展向周期性）。顶盖运动驱动腔内三维涡旋
结构，侧壁边界层使中心平面主涡强度弱于 2D 情形约 28%。

**参考基准**：Ku, Hirsh & Taylor (1987) 谱方法，中心线 x=0.5, z=0.5 处
u_min = −0.2751（三维立方腔，非 2D Ghia 值 −0.3829）。

**关键物理特征**：
- **主涡结构**：顶盖运动驱动腔内形成单一主涡，涡心位于 (x≈0.59, y≈0.48) 附近
  （中剖面 z=0.5），较 2D 涡心 (0.5313, 0.5625) 向下游偏移
- **侧壁效应**：z=0 与 z=1 两侧壁产生摩擦阻力，使中心平面主涡强度减弱
  （u_min 从 2D 的 -0.3829 降至 3D 的 -0.2751，修正量 ~28%）
- **展向对称性**：uz 在中心平面 z=0.5 附近满足反对称条件，
  uz_midplane_max_abs ≈ 0.0013-0.0020（数值误差量级，物理对称性良好）

## 2. 几何与控制参数

- **几何**：立方腔 [0,1]³，Nx=Ny=Nz（各向同性网格）
- **雷诺数**：Re = u_lid × L / ν = 1000
- **顶盖速度**：u_lid = 0.06（格子单位），等 Re 配方
- **碰撞格式**：TRT（Two-Relaxation-Time），Λ = 3/16（最优稳定性参数）
  - 注：MRT/RLBM/cumulant/KBC 在 Re=1000 下均发散（14-219 步 NaN），
    TRT 是唯一稳定解（2026-08-20 实测）
- **边界条件**：
  - 5 静止壁（x=0, x=N-1, y=0, z=0, z=N-1）：pre-streaming 半程反弹
  - 顶盖 y=N-1：boundaries3d.zou_he_moving_lid_3d（D3Q19 Zou-He 移动壁）

## 3. 验证结果（来自 verified/cavity_3d_full/result.json）

### 网格收敛序列

| 档位 | u_min | err vs Ku | 3D 修正量 | 残差 | 耗时 |
|------|-------|-----------|-----------|------|------|
| 64³  | −0.27142 | 1.338% | 29.11% | 9.75e−07 | 779s |
| 96³  | −0.27581 | 0.259% | 27.97% | 3.82e−06 | 1800s |

**单调收敛**：1.338% → 0.259% 严格下降（monotone_decrease: true）。

### 四门判据

| 判据 | 状态 | 证据 |
|------|------|------|
| 共性模块入口 | ✅ PASS | src/tensorlbm 库（solver3d + boundaries3d）驱动 |
| 直接观测量 ≤3% | ✅ PASS | 96³ 档 0.259% ≤ 3%；64³ 档 1.338% ≤ 3% |
| ≥2 档单调收敛 | ✅ PASS | 1.338% → 0.259% 严格单调 |
| 参考先锁定后跑 | ✅ PASS | Ku 1987 参考预注册，verified_date=2026-08-20 |

## 4. 图件（m93 图档补强）

图件由 `plot_figs.py` 从 case_643.npz + case_643.json + case_963.json 工件生成
（零手抄，图件面可复现）。详见 `benchmarks/verified/cavity_3d_full/figures.json`。

### 4.1 中剖面速度幅值云图（64³）

![中剖面速度幅值](benchmarks/verified/cavity_3d_full/figures/field_midplane_speed.png)

中剖面 z=0.5 速度幅值云图，展示主涡结构与侧壁边界层。白色箭头为速度矢量
（稀疏采样）。主涡位于 (x≈0.59, y≈0.48)，与 2D Ghia 涡心 (0.5313, 0.5625)
相比向下游偏移，反映 3D 侧壁摩擦效应。

### 4.2 中剖面涡量云图（64³）

![中剖面涡量](benchmarks/verified/cavity_3d_full/figures/field_midplane_vorticity.png)

中剖面 z=0.5 涡量 ω_z = ∂v/∂x − ∂u/∂y 云图，展示涡心位置与角涡结构。
红色区域为正涡量（逆时针旋转），蓝色区域为负涡量（顺时针旋转）。
主涡区域呈现清晰的负涡量核心，角涡结构在 (x≈0.9, y≈0.1) 附近可见。

### 4.3 中心线速度剖面 vs 2D Ghia 参考

![中心线速度剖面](benchmarks/verified/cavity_3d_full/figures/curve_centerline_vs_ghia.png)

中心线 u(x=0.5, z=0.5) 垂直剖面 vs 2D Ghia (1982) Re=1000 参考曲线。
3D TRT 结果（64³/96³）较 2D Ghia 明显减弱（u_min 从 -0.3829 降至 -0.2751），
定量展示 3D 侧壁修正效应（~28%）。网格细化从 64³ 到 96³ 使结果更接近 Ku 1987
参考值（误差从 1.338% 降至 0.259%）。

### 4.4 误差-网格收敛曲线

![误差收敛](benchmarks/verified/cavity_3d_full/figures/curve_convergence.png)

误差-网格收敛曲线：64³ 1.338% → 96³ 0.259%，严格单调下降至 Ku 1987 参考值。
证明 TRT + V3-3D 边界组合在 Re=1000 下具有良好的网格收敛性。

## 5. 本次图档补强动作

`benchmarks/verified/cavity_3d_full/` 原已归档（2026-08-20，m60），但缺三样：
案例细节介绍（README 深度）、云图（场量分布）、曲线（收敛/误差/剖面）。

本次操作（m93 BENCH-CLOSE figdoc）：
1. 撰写 README.md：独立成篇的案例介绍（6220B，9 个 ## 小节），含物理问题、
   几何参数、运行命令、判据参考、PROVENANCE 溯源块
2. 修改 run.py：新增 `--dump` 选项，dump 完整 3D 场到 npz（云图生成用）
3. 重跑 N64 dump 场：case_643.npz（2.7MB，含 ux/uy/uz 三维速度场）
4. 编写 plot_figs.py：从 npz + json 工件生成 4 张 PNG（零手抄）
5. 生成 figures.json manifest：4 张图（2 云图 + 2 曲线），每张 ≥6KB
6. 重造 gates.json：六门全过（原四门 + case_readme + figures，profile=paper）
7. 更新 report_cavity_3d_full.md：嵌入新图件与案例细节

## 6. 审计溯源

- verified_date: 2026-08-20（原归档）
- 图档补强 date: 2026-10-20（m93）
- PROVENANCE 值直读自 `benchmarks/verified/cavity_3d_full/result.json`
- 图件工件：case_643.npz (run.py --dump) + case_643/963.json (run.py both)
- plot 脚本：plot_figs.py（入库，图从工件生成，零手抄）
- gates.json：六门三元组完整（gates_check.py --profile paper rc=0）
- 运行时长：N64 dump 场 87s（GPU5，RTX 5090），N96 未重跑（用原 case_963.json）

## 7. 局限与缓解

- **局限**：N96 dump 场未重跑（云图仅用 N64），因 N96 运行时长 1800s 超出本 mission
  预算；但 N64 已足够展示主涡结构与侧壁效应
- **缓解**：N64 与 N96 误差仅差 1.079%（1.338% → 0.259%），N64 场已收敛至
  参考值 1.3% 以内，物理结构定性正确
- **未来工作**：MRT/RLBM 发散根因待诊断（可能为高 Re 下矩空间数值不稳定），
  需跨战役线协调 Wave-10 碰撞核改造

## 8. 判据表（机器可读）

见 `benchmarks/verified/cavity_3d_full/gates.json`（六门三元组）与
`benchmarks/verified/cavity_3d_full/figures.json`（图件 manifest）。

gates_check.py --profile paper 复算结果：
```json
{"case": "cavity_3d_full", "verdict": "promote", "profile": "paper",
 "all_pass": true, "problems": [], "rc": 0}
```
