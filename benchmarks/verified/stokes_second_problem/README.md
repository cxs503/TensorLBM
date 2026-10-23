# Stokes 第二问题（振荡平板, D2Q9）— 受迫振荡层流解析解 benchmark

**判定: 达标 (VERIFIED)** — 6/6 配置全部门限通过 + 两条 kH 线三档网格单调收敛。

## 解析解

半无限静止流体中，无限平板在自身平面内谐波振荡 u_w(t) = U·cos(ωt)。
稳态周期解析解（Batchelor; Landau-Lifshitz §24）：

    u(y,t) = U·e^{−ky}·cos(ωt − ky),    k = √(ω/(2ν))

幅值随深度指数衰减、相位线性滞后，皆由单一穿透波数 k 决定。
格子单位: ν = (τ−½)/3, t 为步数, Ma = U√3。

## 仿真构成（真实仿真，无外推修正/无人工因子）

- 格子: D2Q9, BGK (τ=0.8, ν=0.1), x 方向周期（`solver.stream`）, nx=8,
  ny=H+2。
- **振荡平板 = 顶行 (y=ny−1)**，每步以瞬时板速 u_lid(t)=U·cos(ωt)
  （0-d tensor，兼容 torch.compile 不触发重编译/守卫）调用库移动盖
  `tensorlbm.lid_driven_cavity.zou_he_moving_lid`（任务指定的 cavity
  移动盖机制）。Zou/He 在边界节点上规定速度 → 深度从板节点起算：
  y(row) = (ny−1) − row。
- **远场 = 底行 (y=0)，自由滑移镜面反射** `f_new[j] = f_pre[SPECULAR[j]]`
  （翻转 c_y、保留 c_x，无壁面剪切），与 verified/stokes_first_problem
  相同的 repo 已验证远场范式（同样的 3 行内联写法）。
- IC: 静止 equilibrium；startup 瞬态跳过
  skip = max(10, ceil(8·τ_diff/T)) 个整周期（τ_diff = H²/(π²ν)，
  最慢通道模态衰减时间；实测 skip ≈ 8.6 个 e-fold，残量 ~e⁻⁸·⁶），
  测量覆盖恰好 4 个整周期 —— 只在完整周期上比较，跳过窗口逐配置记录。
- kH = k·H 固定每条扫描线（k = kH/H，ω = 2k²ν）；域高 ≥6 个衰减长度
  （任务规则 ≥4）。

## 为什么远场必须是自由滑移（本 benchmark 的核心工程发现）

无滑移远壁会污染这族物理的衰减尾部。证据链（都在 GPU 6 上复现）：

1. **同 k 的深域对照**（k=0.06, τ=0.8, Zou-He 板，唯一变量 = 底壁深度）：
   - 壁在 6 个衰减长度（kH=6, H=100）：掩模边缘复振幅误差 **2.65%**
     （相位主导，φ 误差 ~1.5°）；
   - 壁在 18 个衰减长度（H=300，其余全同）：**0.064%** —— 40 倍消失。
     → 污染源就是无滑移远壁，不是平板 BC 也不是分辨率。
2. **不可细化性**（无滑移壁、kH=6 时）：固定 τ=0.8，H=50/100/200 →
   误差 2.40/2.65/2.75%（加密反而升向地板）；扩散型细化（H=200, τ=0.65，
   ν 减半）→ **3.44% 更差**；τ=0.55/1.2 扫描误差仅 ±40%。即该误差与
   (H, τ) 基本无关 —— 它只依赖无量纲壁距 kH（e^{−2(kH−2.3)} 型），
   任何标准细化都消不掉。
3. **库内先例交叉验证**：verified/stokes_first_problem（同一物理族的
   瞬态版）文档记录了同一效应：无滑移顶壁把尾部 ~2%U 的速度拽到 0，
   H=100 时 max_rel 10.8%，改自由滑移后消除。本 case 是其周期稳态版
   （kH=6 时 ~2.7% 地板）。
4. 换自由滑移远场后（本文件采用）：H=50 粗网格 kH=6 误差即从 2.40% 降到
   **0.303%**，且随加密正常收敛。

（此发现同时解释了开发早期"kH=6 相位误差 H 无关"的困惑；诊断脚本与
深域对照数据见下"复现/诊断"节。）

## 门限（先验固定，各网格掩模一致）

1. 幅值: max |A_num/A_ana − 1| ≤3%，行集 A_ana ≥ 10%·U
2. 相位滞后: max |φ_num − φ_ana|/φ_ana ≤3%，行集 φ_ana ≥ 1 rad 且
   A_ana ≥ 10%·U（板附近解析滞后 →0 处相对相位无意义；绝对相位误差
   同时按度报告）
3. 复振幅（合成单指标）: max |U_num/U_ana − 1| ≤3%（同幅值掩模）
4. 网格收敛: 误差沿 H=50/100/200 单调下降（两条 kH 线）

## 网格扫描结果（scan_out/result.json）

| kH | H | 幅值 max | 相位 max (rel; abs) | 复振幅 max | steps |
|---|---|---|---|---|---|
| 6 | 50  | 0.182% | 0.106% (0.14°) | 0.303% | 30,548 |
| 6 | 100 | 0.052% | 0.042% (0.05°) | 0.109% | 122,178 |
| 6 | 200 | 0.017% | 0.023% (0.03°) | 0.053% | 488,698 |
| 8 | 50  | 0.310% | 0.147% (0.17°) | 0.425% | 25,767 |
| 8 | 100 | 0.072% | 0.036% (0.05°) | 0.108% | 103,089 |
| 8 | 200 | 0.010% | 0.008% (0.01°) | 0.020% | 412,335 |

全部 ≤3%（最大 0.425%，粗网格）；两条 kH 线三档网格严格单调下降
（kH=6: 0.303→0.109→0.053%，kH=8: 0.425→0.108→0.020%，每档加密
降 ~2.5–5×）。全程 finite；相位绝对误差最大 0.17°。mass drift 随步数
缓慢积累（H=50: 0.0008–0.005% → H=200: 0.35–0.37%，Zou/He 盖在边界
节点上调节密度、非严格守恒——cavity 机制的已知性质，对速度相量无影响，
数值逐配置记录于 case json）。

## 入口申报（零手写内核，grep 自检通过）

- collide: `tensorlbm.solver.collide_bgk`
- stream: `tensorlbm.solver.stream`
- plate: `tensorlbm.lid_driven_cavity.zou_he_moving_lid`（库移动盖，
  每步时变标量）
- far_field: 自由滑移镜面反射 —— verified/stokes_first_problem 的
  repo 已验证内联范式（SPECULAR 置换 pre-collision 状态；非手写
  collide/stream/equilibrium）
- IC/测量: `tensorlbm.d2q9.equilibrium / macroscopic`
- 编译: `benchmarks/compile_route.route_step`（torch.compile default；
  eager/compiled A/B 同案误差 2.646% vs 2.649%，无编译数值差）

## Gap 清单（如实记录）

1. **库内无"无剪切远场"边界入口**：`tensorlbm.boundaries` 只有
   bounce_back（无滑移）类。本文件按 verified/stokes_first_problem
   的既有范式内联 SPECULAR 反射（3 行，库常量置换），未改库。
   该缺口对 Stokes 族解析 benchmark 是实质性的（见上节：无滑移远壁
   → 不可细化的 ~2.7% 地板 @kH=6）。
2. **无滑移远壁 + `bounce_back_cells` 的组合在本 case 不可用**：
   误差 2.4-3.4% 且任何细化不收敛（证据链见上），故未采用；
   若未来库提供自由滑移远场入口，应改用之。
3. Zou/He 移动盖在振荡驱动下本身表现良好（板近旁行误差 ≤0.1%@H=50，
   随网格收敛），无需额外处理。

## 复现 / 诊断

    cd /nfs/wangxi/runs/bm_widen_20260919/unsteady_analytic/stokes_second_problem
    CUDA_VISIBLE_DEVICES=6 /nfs/wangxi/venvs/tensorlbm/bin/python run.py scan scan_out
    # 单点: run.py single 100 6 /tmp/case.json --device cuda

库只读挂载: `/nfs/wangxi/worktrees/bm_relaunch/src` (main @ 64a979de)。
运行 2026-09-19, GPU 6, torch 2.11.0+cu128。

远壁污染证据的诊断脚本（临时，非交付物）: /nfs/wangxi/tmp/probe_deep.py
（k=0.06 定波深域对照）与 /nfs/wangxi/tmp/diag_stokes.log（H×kH×τ 扫描）。
