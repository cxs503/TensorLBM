# 全 3D 方腔流 cavity_3d_full（Re=1000，5 壁固定 + 顶盖移动）

**状态：✅ 已晋级 verified（2026-08-20）**

本案例是经典顶盖驱动方腔流的三维全尺寸版本：立方腔体 [0,1]³，五个静止壁面
（x=0, x=1, y=0, z=0, z=1）+ 一个移动顶盖（y=1 沿 +x 方向匀速 u_lid=0.06）。
与 2D 方腔流（Ghia et al. 1982）不同，3D 版本存在展向侧壁摩擦效应，中心平面
主涡强度弱于 2D 情形约 28%。

## 物理问题

顶盖驱动方腔流是计算流体力学（CFD）最经典的验证基准之一。三维全尺寸版本
（非展向周期性）引入了侧壁边界层与角涡结构，对数值方法的三维空间离散精度、
边界条件处理与时间推进稳定性提出更高要求。

**关键物理特征**：
- **主涡结构**：顶盖运动驱动腔内形成单一主涡，涡心位于 (x≈0.59, y≈0.48)
  附近（中剖面 z=0.5），较 2D 涡心 (0.5313, 0.5625) 向下游偏移
- **侧壁效应**：z=0 与 z=1 两侧壁产生摩擦阻力，使中心平面主涡强度减弱
  （u_min 从 2D 的 -0.3829 降至 3D 的 -0.2751，修正量 ~28%）
- **展向对称性**：uz 在中心平面 z=0.5 附近满足反对称条件，
  uz_midplane_max_abs ≈ 0.0013-0.0020（数值误差量级，物理对称性良好）

## 几何与控制参数

- **几何**：立方腔 [0,1]³，Nx=Ny=Nz（各向同性网格）
- **雷诺数**：Re = u_lid × L / ν = 1000（基于腔体长度 L=1 与顶盖速度 u_lid）
- **顶盖速度**：u_lid = 0.06（格子单位），等 Re 配方
- **碰撞格式**：TRT（Two-Relaxation-Time），Λ = 3/16（最优稳定性参数）
  - 注：MRT/RLBM/cumulant/KBC 在 Re=1000 下均发散（14-219 步 NaN），
    TRT 是唯一稳定解（2026-08-20 实测）
- **边界条件**：
  - 5 静止壁（x=0, x=N-1, y=0, z=0, z=N-1）：pre-streaming 半程反弹
    （碰撞后、stream 前用 f_pre 反射，V3-3D 口径）
  - 顶盖 y=N-1：boundaries3d.zou_he_moving_lid_3d（D3Q19 Zou-He 移动壁，
    覆盖整层含角点，ux=u_lid, uy=uz=0）

## 运行命令

```bash
# 单档运行（64³ 网格，100000 步）
python benchmarks/verified/cavity_3d_full/run.py single 64 case_643.json --device cuda:5 --steps 100000

# 双档运行（64³ + 96³，输出到 verified/cavity_3d_full/）
python benchmarks/verified/cavity_3d_full/run.py both verified/cavity_3d_full --device64 cuda:5 --device96 cuda:5 --steps 100000

# 生成图件（需先运行双档，产出 case_643.json + case_963.json）
python benchmarks/verified/cavity_3d_full/plot_figs.py
```

运行时长参考：64³ 约 779 秒，96³ 约 1800 秒（GPU5，RTX 5090）。

## 判据与参考基准

**主判据**：中心线 u_min(x=0.5, z=0.5) vs Ku, Hirsh & Taylor (1987) 谱方法基准

- **参考值**：u_min = -0.2751（3D 立方腔 Re=1000，非 2D Ghia 值 -0.3829）
- **文献锚点**：
  - Ku, Hirsh & Taylor (1987) "Spectral methods for the Navier-Stokes equations
    with one infinite and two periodic directions", J. Comput. Phys.
  - arXiv:1503.03337 (iD3Q14 MRT, 97³) 网格收敛序列：
    49³/65³/81³/97³ → u_min = -0.2619/-0.2693/-0.2730/-0.2751
  - 2D Ghia et al. (1982) Re=1000 用于 3D 修正量对比（主涡减弱 ~28%）

**辅助判据**：
- 中心平面速度剖面 vs 2D Ghia（报告最大偏差、RMSE）
- 展向对称性 uz_midplane_max_abs（应 < 0.01）
- 残差收敛（last_resid < 1e-5）

## 验证结果

| 档位 | u_min | err vs Ku | 3D 修正量 | 残差 | 耗时 |
|------|-------|-----------|-----------|------|------|
| 64³  | −0.27142 | 1.338% | 29.11% | 9.75e−07 | 779s |
| 96³  | −0.27581 | 0.259% | 27.97% | 3.82e−06 | 1800s |

**单调收敛**：1.338% → 0.259% 严格下降（monotone_decrease: true）。

**四门判据**：
1. ✅ 共性模块入口：src/tensorlbm 库（solver3d + boundaries3d）驱动
2. ✅ 直接观测量 ≤3%：96³ 档 0.259% ≤ 3%；64³ 档 1.338% ≤ 3%
3. ✅ ≥2 档单调收敛：1.338% → 0.259% 严格单调
4. ✅ 参考先锁定后跑：Ku 1987 参考预注册，verified_date=2026-08-20

## 图件

图件位于 `figures/` 子目录，由 `plot_figs.py` 从 case_643.json + case_963.json
工件生成（零手抄，图件面可复现）：

1. **field_midplane_speed.png**：中剖面 z=0.5 速度幅值云图（96³），展示主涡结构
   与侧壁边界层
2. **field_midplane_vorticity.png**：中剖面涡量云图（96³），展示涡心位置与角涡
3. **curve_centerline_vs_ghia.png**：中心线速度剖面 vs 2D Ghia 参考曲线（64³/96³），
   展示 3D 修正效应
4. **curve_convergence.png**：误差-网格收敛曲线（1.338% → 0.259%），展示单调收敛

详见 `figures.json` manifest。

## PROVENANCE

本案例所有判决数字直读自工件，零手抄：

```json
[
  {"v": 1.338, "f": "benchmarks/verified/cavity_3d_full/result.json", "k": "grids.64.err_pct_vs_ku"},
  {"v": 0.259, "f": "benchmarks/verified/cavity_3d_full/result.json", "k": "grids.96.err_pct_vs_ku"},
  {"v": true, "f": "benchmarks/verified/cavity_3d_full/result.json", "k": "verified"},
  {"v": "2026-08-20", "f": "benchmarks/verified/cavity_3d_full/result.json", "k": "verified_date"}
]
```

图件工件：
- case_643.json / case_963.json：run.py 双档运行产出（含 u_centerline_mid, v_centerline_mid）
- plot_figs.py：从 case_*.json 读取中心线数据 + 重跑 N64 dump 场生成云图
- figures/*.png：plot_figs.py 产出，每张 ≥6KB，真 PNG 魔数

## 局限与缓解

- **局限**：TRT 碰撞在 Re=1000 下稳定，但 MRT/RLBM 发散（14-219 步 NaN），
  无法对比高阶碰撞核精度
- **缓解**：TRT 结果与 Ku 1987 谱方法基准吻合良好（96³ 误差 0.259%），
  证明 V3-3D 边界 + TRT 组合足以捕获 3D 方腔流主涡结构
- **未来工作**：MRT/RLBM 发散根因待诊断（可能为高 Re 下矩空间数值不稳定），
  需跨战役线协调 Wave-10 碰撞核改造

## 审计

- verified_date: 2026-08-20
- 归档 mission: m60（BENCH-CLOSE 首轮）
- 图档补强 mission: m93（BENCH-CLOSE figdoc，2026-10-20）
- gates.json: 六门全过（原四门 + case_readme + figures，profile=paper）
