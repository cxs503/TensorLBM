# 3D 椭圆管 Poiseuille 流（pending）

**状态：❌ 未达标（根因类：单尺度参考几何口径）——s^Q 单尺度比较下 a=20 档逐 bin 最大误差 5.11% > 3%（a=40 档 2.43% 过线），单调收敛成立；根因 = 数字楼梯椭圆壁的各向异性水力几何无法用单一尺度因子表达，非求解器缺陷（实测剖面是精确椭圆抛物面，拟合残差 <0.2%）。**

## 物理问题

三维椭圆截面直管（半轴 a:b = 2:1）内充分发展层流，速度入口 + 压力出口驱动，
管壁无滑移。解析解（稳态、充分发展、不可压、无滑移）：
u(y,z) = U_max·(1 − y²/a_eff² − z²/b_eff²)，U_max = 2ū = 2u_in（质量守恒；
Q = πab·U_max/2，White, *Viscous Fluid Flow*）。

**判定方法 = 面积等效尺度 s^Q**（圆管 R_eff^Q 的椭圆类比，单参数、独立积分观测量、
不拟合剖面）：s^Q = √(2Q/(π·a·b·U_max))，a_eff = a·s^Q、b_eff = b·s^Q；
主指标 = λ 分环（椭圆平均）剖面在中心区（|u_ana| > 0.2·U_max）的最大相对误差，
绝对归一化 U_max = 2u_in（施加量）。双参数剖面拟合 (a_fit, b_fit) 仅作诊断
（若用作比较参考即循环论证）。

### 共性模块

- D3Q19 BGK `solver3d.collide_bgk3d` + `stream3d`（周期 gather）、
  `d3q19.equilibrium3d`/`macroscopic3d`；
- BC 全部库函数：`boundaries3d.zou_he_inlet_velocity_3d`（x=0）、
  `zou_he_outlet_pressure_3d`（x=nx−1，ρ_out=1）、`bounce_back_cells_3d`
  （post-streaming 半程反弹，全部固体格 λ>1）；
- 无外推（extrap: none）、无校正因子；a=20/40 两档（与圆管 a 口径一致），
  τ=0.8，u_in=0.02。

## 计算结果现状（`result.json`，速度入口模式，s^Q 比较）

| a | b | Re | 步数 | s^Q | a_eff | b_eff | 逐 bin max（主指标） | 逐 bin L2 |
|---|---|---|---|---|---|---|---|---|
| 20 | 10 | 5.19 | 25600 | 1.011261 | 20.2252 | 10.1126 | **5.107%** | 2.016% |
| 40 | 20 | 10.38 | 21200 | 1.005101 | 40.2041 | 20.1020 | **2.432%** | 0.853% |

- 两档未全部 ≤3%（a=20 超线），但单调收敛（5.107→2.432%，一阶于 1/a）→
  `passed_3pct_and_converged: false`、`verdict: not_verified`。
- 误差位置：a=20 最大误差在最后一个中心 bin（λ∈[0.7,0.8)）；核心区其余 bin 全部
  ≤1.9% 量级。`converged: true`。
- 自检：中心速度 vs 2u_in/s^Q² 偏差 +0.229%/+0.112%（质量守恒自洽）；
  Q/Q_ana^Q ≡ 1（构造恒等）；拟合残差 0.194%/0.184%（剖面是精确椭圆抛物面）。

披露变体（`per_grid` 全记录）：逐格（per-cell）中心区 max 15.86%/6.18%
（楼梯过渡层单格散布，随细化减半）；形状归一化逐 bin 7.24%/3.36%；
名义半轴 (a,b) 逐 bin 11.81%/5.95%（s^Q 修正所消除的近壁 bin 系统偏差）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

1. **共性模块入口**：✓——碰撞/迁移/BC/反弹全部库函数，零手写物理核。
2. **直接观测量 ≤3%**：✗——a=20 档主指标 5.107% > 3%（a=40 档 2.432% 过线）。
3. **≥2 档网格单调收敛**：✓——5.107% → 2.432% 严格单调（一阶 1/a 收敛）。

## 根因/诊断

- **数字（staircase）椭圆壁是各向异性的**：长轴端（单格楼梯"尖端"，曲率半径
  ~b²/a）有效壁面外凸、短轴端（近平直壁）内缩——双参数剖面拟合（仅诊断）：
  (a_fit−a, b_fit−b) = (+1.176, −0.147) / (+1.06, −0.148)；对照圆管的各向同性
  R_eff^Q − R ≈ +0.11（verified/poiseuille_3d_pipe：20.1088/40.1177）。
  单尺度 s^Q 无法同时表达两个方向的壁面偏移 → 最外中心 bin 超线。
- **求解器与边界条件正确**：实测剖面加权拟合残差 0.194%/0.184%、中心区拟合最大
  误差 0.432%/0.373%、质量守恒自检 <0.3%——失败的是"单一尺度参数化参考几何"
  这一比较口径，而非仿真。
- a=20 档 s^Q 面积等效尺度偏大（a_eff−a = +0.225 vs 圆管同分辨率 +0.109 量级）
  是同一各向异性的积分表现。

## 晋级路径（转 verified 条件；达标即归档 PR 移 pending/poiseuille_3d_ellipse → verified/poiseuille_3d_ellipse）

1. **加档重判（零方法变更）**：补 a=40/80（或 a=80/160）两高档——误差一阶于 1/a
   收敛，高档两档全 ≤3% 且单调即满足两档判据（本记录 a=20/40 口径如实 not_verified）；
2. **或口径升级（需 owner 批）**：两观测量水力反演（流量 Q + 压降 Δp 联合定
   (a_eff, b_eff)，均为独立积分观测量）或各向异性双参数参考——超出"R_eff^Q 类比"
   的任务口径，本记录未采用；
3. 达标即归档 PR 移 `pending/poiseuille_3d_ellipse` → `verified/poiseuille_3d_ellipse`
   （与 verified/poiseuille_3d_pipe 同族归并）。

## 运行方式

```bash
cd benchmarks/pending/poiseuille_3d_ellipse
PYTHONPATH=../../../src python run.py scan <out_dir> --a 20 40 --ratio 2.0 \
    --min-steps 20000 --max-steps 60000 --device cuda:0
# 单例：
PYTHONPATH=../../../src python run.py single 40 case_a40_b20.json
# result.json 由 run.build_summary 从两档 case 聚合（与 scan 输出一致）
```

工件：`result.json`（判定口径 + 全指标变体 + 各向异性诊断 notes）、
`case_a20_b10.json` / `case_a40_b20.json`（实测数据）。参考场推导脚本与过程
工件留服务器暂存（Wave-3 档案）。

<!-- PROVENANCE
[{"v": 20, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "a_list[0]"},
{"v": 40, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "a_list[1]"},
{"v": 2.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "aspect_ratio"},
{"v": 0.8, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "tau"},
{"v": 0.02, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "u_in"},
{"v": 5.188201663686822, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].Re"},
{"v": 10.376403327373644, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].Re"},
{"v": 25600, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].n_steps"},
{"v": 21200, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].n_steps"},
{"v": 1.0112611626509898, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].s_Q"},
{"v": 1.0051013719725987, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].s_Q"},
{"v": 20.225223253019795, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].a_eff"},
{"v": 10.112611626509898, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].b_eff"},
{"v": 40.204054878903946, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].a_eff"},
{"v": 20.102027439451973, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].b_eff"},
{"v": 5.106962011179169, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].max_rel_bin_central_sQ_pct"},
{"v": 2.431766861465193, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].max_rel_bin_central_sQ_pct"},
{"v": 0.020160129797660395, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].l2_rel_err_sQ"},
{"v": 0.008534200036143228, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].l2_rel_err_sQ"},
{"v": 15.859010971642098, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].max_rel_err_central_sQ_pct"},
{"v": 6.18358919377844, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].max_rel_err_central_sQ_pct"},
{"v": 7.242043069325041, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].max_rel_bin_central_sQ_shape_pct"},
{"v": 3.363318508756448, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].max_rel_bin_central_sQ_shape_pct"},
{"v": 11.812324377498678, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].max_rel_bin_central_pct"},
{"v": 5.952348607172287, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].max_rel_bin_central_pct"},
{"v": 0.22893144978715962, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].u_center_pred_err_pct"},
{"v": 0.11240568856921664, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].u_center_pred_err_pct"},
{"v": 0.0019447514011280446, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].l2_fit_rel_err"},
{"v": 0.00184052766976355, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].l2_fit_rel_err"},
{"v": 0.43238234500249395, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].max_rel_fit_central_pct"},
{"v": 0.3729981122496338, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].max_rel_fit_central_pct"},
{"v": 1.1757597297230937, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].a_fit_minus_a"},
{"v": -0.14702991758291262, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].b_fit_minus_b"},
{"v": 1.0574115182136339, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].a_fit_minus_a"},
{"v": -0.14768598649858333, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[1].b_fit_minus_b"},
{"v": 0.9999999999999996, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "per_grid[0].Q_ratio_sQ"},
{"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "converged"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "passed_3pct_and_converged"},
{"v": "not_verified", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "verdict"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json", "k": "verified"},
{"v": 20.108758379009544, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_pipe/result.json", "k": "per_grid[0].R_eff_Q"},
{"v": 40.11765828987995, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_pipe/result.json", "k": "per_grid[1].R_eff_Q"},
{"v": "verified", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_3d_pipe/result.json", "k": "verdict"}]
-->
