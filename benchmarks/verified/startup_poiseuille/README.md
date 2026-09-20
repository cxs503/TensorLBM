# 启动 Poiseuille 流（2D 通道, D2Q9）— 瞬态层流解析解 benchmark

**判定: 达标 (VERIFIED)** — 2/2 配置全部门限通过 + 单调网格收敛。

## 解析解

静止流体通道（壁面 y=0, H），t=0 突加恒定流向体力/压力梯度
（a = const, dp/dx = −ρa）。经典 Fourier 级数瞬态解
（Pedley; Ethier & Critchley; Fachinotti & Le Bot）：

    u(y,t) = u_max·[ 4ŷ(1−ŷ) − (32/π³)·Σ_{n odd} (1/n³)·sin(nπŷ)·e^{−n²π²νt/H²} ]
    ŷ = y/H,   u_max = a·H²/(8ν)

- t→0⁺: Fourier 恒等式 Σ_{odd} sin(nπŷ)/n³ = π³ŷ(1−ŷ)/8 使 u→0
  （数值核验 1e-8）；t→∞: 退回抛物线 u_max·4ŷ(1−ŷ)（核验 1e-16）。
- **注**：常被引用的 Pedley "1 − (32/π³)Σ…" 形式是**中心线**特例
  （ŷ=1/2 处 sin(nπ/2)=(−1)^((n−1)/2)）；全场公式稳态部分是
  4ŷ(1−ŷ) 抛物线，不是 1。两者在实现时混用会差整整一个抛物线形状
  （开发中曾错用，靠 t→0/t→∞ 两个极限独立核查抓出）。
- 指数是本征值形式 (nπ/H)²·νt（模态 sin(nπy/H) 的热算子衰减率）；
  开发中曾漏一个 π（写成 n²πνt/H²），PDE 残差未归一化没抓住，
  由仿真/解析中心线比值 = 3.1414 ≈ π 抓出。教训已记：残差核查必须
  对项量级归一。
- 级数 float64 求值，奇 n 至 999，指数裁剪 700 防下溢；最早记录时刻
  t*=0.05 时 n>10 项 <1e-6，无 Gibbs 污染。
- 参考: R.W. Fox & A.T. McDonald 教材标准题；J.R. Womersley 相应
  瞬态形式；O.I. Fachinotti & O. Le Bot, "On the accuracy of ...
  unsteady Poiseuille flow" 类文献中的级数解。

## 仿真构成（真实仿真，无外推修正/无人工因子）

- 格子: D2Q9, BGK (τ=0.8, ν=0.1), x 周期（`solver.stream`）, nx=16,
  ny=H+2。与 Womersley benchmark 同一通道/驱动机械（同库入口同步序）。
- 壁面: 预流半程 bounce-back（verified/poiseuille_2d 范式，
  `f ← f_pre[OPPOSITE]` 流前作用于壁面行；无滑移面 y=0.5 / ny−1.5，
  H_eff = ny−2 = H，流体行 y_phys = row−0.5）。
- 驱动: `tensorlbm.turbulent_channel._apply_body_force_2d(f, a)`，
  常量 a = 8ν·u_max/H_eff² 使解析终态中心线速度 = u_max = 0.03
  (Ma = 0.052)；步序 collide → half-way BB → force → stream。
- IC: t=0 静止 equilibrium，力从第一步起恒定。**瞬态本身即被测对象，
  无跳过窗口**；比较用同一格子时刻的完整级数解。
- a0 与时间标记全部用 H_eff（半程壁面间隙）。

## 门限（先验固定，各网格一致）

1. 中心线 u_c(t)（每 20 步采样，t* = νt/H² ≥ 0.02）：
   max |u_c_num − u_c_ana| / u_max ≤3%
2. 剖面（t* ∈ {0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5}）：
   L2 相对误差 ≤3% 且逐点最大误差 / u_max ≤3%
3. H→2H 误差单调下降。

## 网格扫描结果（scan_out/result.json, status=VERIFIED）

| H_eff | 中心线 max/u_max | 剖面 L2 max | 剖面逐点/u_max max | steps |
|---|---|---|---|---|
| 59  | 0.288% | 0.291% | 0.288% | 52,235 |
| 119 | 0.165% | 0.225% | 0.155% | 212,435 |

全部 ≤3%，三项指标 H→2H 均下降（1.7×–1.9×）。mass drift ≤0.17%，
全程 finite。中心线历史与剖面表逐条存于各 case json
（`centerline_table_every10`, `profile_table`）。

## 入口申报（零手写内核，grep 自检通过）

- collide: `tensorlbm.solver.collide_bgk`
- stream: `tensorlbm.solver.stream`
- force: `tensorlbm.turbulent_channel._apply_body_force_2d`
- wall: 库 `tensorlbm.d2q9.OPPOSITE` 表 + verified/poiseuille_2d 预流半程 BB 范式（见 gap）
- IC/测量: `tensorlbm.d2q9.equilibrium / macroscopic`
- 编译: `benchmarks/compile_route.route_step`（torch.compile default）

## Gap 清单（如实记录，未绕过）

1. 同 Womersley：**solver.py 无公开 D2Q9 体力入口**（详细见其 README）。
2. 库壁面函数 `boundaries.bounce_back_cells`（流后全程 BB，壁面在节点，
   H_eff=ny−1）在本 case 也能收敛但误差约 4 倍且更慢
   （H=29 实测 1.27% vs 半程范式 0.08%）；Womersley α≥4 时该差距放大
   到不可用（见其 README 的变体对比表）。
3. 无其他阻塞。

## 复现

    cd /nfs/wangxi/runs/bm_widen_20260919/unsteady_analytic/startup_poiseuille
    CUDA_VISIBLE_DEVICES=6 /nfs/wangxi/venvs/tensorlbm/bin/python run.py scan scan_out
    # 或单点: run.py single 119 /tmp/case.json --device cuda

库只读挂载: `/nfs/wangxi/worktrees/bm_relaunch/src` (main @ 64a979de)。
运行 2026-09-19, GPU 6, torch 2.11.0+cu128。
