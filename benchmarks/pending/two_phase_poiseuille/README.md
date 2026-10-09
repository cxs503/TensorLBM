# 两相 Poiseuille：SC-MCMP 双组分分层流（pending）

**状态：❌ 未达标（根因类：多相模型结构性偏差）——SC-MCMP 中心区最大相对误差 26.82%/33.17%（阈值 3%，超出近一个数量级）且细化恶化（反收敛）；不等 τ 混合平衡态格式在协议耦合下 NaN 发散，稳定域内残差由界面条带化主导。同物理问题类已由色梯度模型在 verified/poiseuille_two_phase_cg（PR #327）达标，本记录保留为 SC-MCMP 模型点的失败证据与修复判据基线。**

## 物理问题

上气下水双组分分层 Poiseuille 流（2D 通道，周期方向 x，垂直分层 y）：两侧同一身体力
驱动，参考解 = 库自身 `porous_media._two_phase_poiseuille_analytical`（名义黏度比
M = ν_w/ν_g = 2.0，未做实测黏度比重标定——重标定通道已在诊断中排除，见根因）。

### 共性模块

- `multiphase.collide_sc_two_component`（Shan-Chen 双组分，u_eq 默认 "self"）+
  `porous_media` 模块解析参考（经 `w4b_lib.py` 编排，物理核全部来自库函数，
  `iron_rule.physics_kernel_defs_found = []` 机器检查通过）。
- 配置：τ = (1.0, 0.75)（水/气），G_12 = -2.5（全交换分层），G_x = 5e-6，
  tanh-3 界面（y = half），H64（ny=65, 60000 步）/ H128（ny=129, 180000 步）。

## 计算结果现状

**Wave-4 W4-B Part 1 主判决**（`result.json`，复算与记录逐位一致）：

| 档 | 中心区节点数 | max rel err | mean rel err | u_max 比 | Ma_max | 判定 |
|---|---|---|---|---|---|---|
| H64 | 55 | **26.82%** | 14.21% | 1.26524 | 0.0467 | FAIL |
| H128 | 110 | **33.17%** | 15.57% | 1.31458 | 0.1943 | FAIL |
| 单调 | | 26.82 → 33.17 | | | | FAIL（反收敛） |

中心区 = |u| > 0.2·u_max 掩模（simmax/anamax 两种口径结论相同）。块互溶度
ρ_g-in-water / ρ_w-in-gas = 0.30008/0.15934（H64）、0.30004/0.15148（H128）；
伪流 |uy|_max ≈ 0.0794/0.0797。

**Wave-5 W5-A 修复尝试（后续核查，暂存 `bm_widen_w5_20260921/mcmp_fix/`）**——
`collide_sc_two_component` 增加 `u_eq="mixture"`（Shan–Doolen 1995 质心速度平衡态；
动量交换恒等式/等 τ 总动量守恒/伽利略不变性/单分量极限均验证成立）后：

| 运行 | u_eq | G_12 | τ (水,气) | H | 中心区 max rel err |
|---|---|---|---|---|---|
| R1 / R2 | mixture | -2.5 | (1, 0.75) | 64 / 128 | **发散（首个 NaN 第 29 步）** |
| R1 / R2 @ G-1.4 | mixture | -1.4 | (1, 0.75) | 64 / 128 | 22.94% / 30.14%（反收敛） |
| R1_t1 | mixture | -2.5 | (1, 1) | 64 | 34.83% |
| R3_t1 | self | -2.5 | (1, 1) | 64 | 7.17% |
| R3 / R4（主判决） | self | -2.5 | (1, 0.75) | 64 / 128 | 26.82% / 33.17% |

NaN 起始步（mixture, τ=(1,0.75)）：G=-2.5 第 29 步、G=-2.0 第 46 步、
G=-1.5 第 175 步（G=-1.0 与全部 self 档 20k 步无 NaN）。判决维持 FAIL。

**同问题类已达标（第二代记录，供对照）**：verified/poiseuille_two_phase_cg
（色梯度 `color_gradient2d` 新库模块，PR #327）四臂 relL2 = 0.211%/0.155%
（等 τ M=1, ny 64/128）、1.092%/0.560%（τ=(2.5, 0.7)，黏度比 10——恰是
SC-MCMP 不等 τ 失稳的模型点），全过 3% 门。PR #327 未改动本目录
（两侧 blob 逐位一致，本记录两代并存）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

1. **共性模块入口**：✓——物理核全部库函数（`iron_rule.pass: true`，
   零手写核）。
2. **直接观测量 ≤3%**：✗——26.82%/33.17%，超出阈值近一个数量级。
3. **≥2 档网格单调收敛**：✗——细化恶化（26.82% → 33.17%，反收敛）。

## 根因/诊断

- **不等 τ 的混合格式在强分相下结构性不稳定**：每步净动量源
  (1/τ₂-1/τ₁)·ρ₁ρ₂(u₁-u₂)/(ρ₁+ρ₂)（等 τ 时归零）——G ≤ -1.5 时全 NaN
  （首个 NaN 第 29/46/175 步），与驱动强度（S1：G_x 减半误差不变，
  22.9428% vs 22.9402%，u_max 比减半 0.50009）和初始化无关。
- **稳定域内主导残差 = 界面条带化**：平界面碎裂为水/气交替条带，黏度分层被打乱
  ——等 τ 档 mixture 34.83% vs self 7.17% 佐证（等 τ 时混合与自速的名义差异已消）。
- **分量自速度平衡态是偏差的必要成分而非充分解释**：混合平衡态落地后，残差转移为
  界面条带化主导（上表 R1_t1 vs R3_t1）。
- **有效黏度比反转（披露）**：~30% 块互溶度使实测有效比 M_eff = 1.5999/1.6219
  （H64/H128）偏离名义 M=2.0；用实测比修正解析只移动 u_max 比、不动 26–33% 的
  剖面偏差且方向不对（W4-B 诊断，姊妹记录 capillary_invasion_washburn）。

## 晋级路径（转 verified 条件；达标即归档 PR 移 pending/two_phase_poiseuille → verified/two_phase_poiseuille）

1. **问题类侧（已完成）**：色梯度两层层流 Poiseuille 已按严格标准 verified
   （verified/poiseuille_two_phase_cg，PR #327）——若 owner 裁定问题类关闭，
   本记录留 pending 作 SC-MCMP 模型点失败证据，或在归档时并入 CG 案例历史注记；
2. **SC-MCMP 侧（若复活）**：先修库 `collide_sc_two_component`——(a) 不等 τ 稳定性
   （W5-A 判 NaN 结构性、净动量源显式补偿或隐式格式）；(b) 界面条带化（各向同性
   化/界面稳定项）——当前证据以 PR #305 xfail-strict 测试锁定；修复后按预注册
   H64/H128 两档重跑，中心区 max ≤3% 且单调收敛 → 达标即归档 PR 移
   `pending/two_phase_poiseuille` → `verified/two_phase_poiseuille`。

## 运行方式

```bash
cd benchmarks/pending/two_phase_poiseuille
PYTHONPATH=../../../src python run.py    # 正式驱动（需空闲 CUDA 设备，约 10 分钟）
```

工件：`result.json`（Part 1 判决切片 + 铁律检查）、`out_part1.json`（正式运行原始
输出：逐点剖面/收敛史/互溶度）、`log_part1.txt`、`run.py` + `w4b_lib.py`（共享
驱动库）。W5-A 修复尝试全套（判别测试 `tests/test_mcmp_mixture.py`、11 个运行
JSON、NaN 起始阶梯）留服务器暂存 `/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/`。

<!-- PROVENANCE
[
 {
  "v": [],
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "iron_rule.physics_kernel_defs_found"
 },
 {
  "v": true,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "iron_rule.pass"
 },
 {
  "v": 55,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H64.n_center"
 },
 {
  "v": 110,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H128.n_center"
 },
 {
  "v": 0.26821538643505377,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H64.max_rel_err"
 },
 {
  "v": 0.33173655541170183,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H128.max_rel_err"
 },
 {
  "v": 0.14209958433345918,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H64.mean_rel_err"
 },
 {
  "v": 0.15572332635120198,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H128.mean_rel_err"
 },
 {
  "v": 1.2652425421161708,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H64.u_max_ratio"
 },
 {
  "v": 1.3145829551904207,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H128.u_max_ratio"
 },
 {
  "v": 0.046743934944757225,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H64.Ma_max"
 },
 {
  "v": 0.19428996495289802,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.H128.Ma_max"
 },
 {
  "v": false,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.criteria.criterion_max_rel_err_le_3pct.H64"
 },
 {
  "v": false,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.criteria.criterion_max_rel_err_le_3pct.H128"
 },
 {
  "v": false,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.criteria.criterion_monotone_decreasing"
 },
 {
  "v": "FAIL",
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json",
  "k": "part1.criteria.verdict"
 },
 {
  "v": 1.0,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.tau_w"
 },
 {
  "v": 0.75,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.tau_g"
 },
 {
  "v": -2.5,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.G_12"
 },
 {
  "v": 5e-06,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.G_x"
 },
 {
  "v": 2.0,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.nominal_M"
 },
 {
  "v": 60000,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.n_steps"
 },
 {
  "v": 180000,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H128.n_steps"
 },
 {
  "v": 65,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.ny"
 },
 {
  "v": 129,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H128.ny"
 },
 {
  "v": 0.30008479952812195,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.rho_g_in_water_bulk"
 },
 {
  "v": 0.15934278070926666,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.rho_w_in_gas_bulk"
 },
 {
  "v": 0.30003637075424194,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H128.rho_g_in_water_bulk"
 },
 {
  "v": 0.15148460865020752,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H128.rho_w_in_gas_bulk"
 },
 {
  "v": 0.07936520874500275,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H64.spurious_uy_max"
 },
 {
  "v": 0.0796600952744484,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/out_part1.json",
  "k": "H128.spurious_uy_max"
 },
 {
  "v": "FAIL",
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "overall"
 },
 {
  "v": false,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "mixture_vs_self_AB.H64.mixture.finite"
 },
 {
  "v": false,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "mixture_vs_self_AB.H128.mixture.finite"
 },
 {
  "v": 26.821538643505377,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "mixture_vs_self_AB.H64.self.err_max_pct_recomputed"
 },
 {
  "v": 33.173655541170184,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "mixture_vs_self_AB.H128.self.err_max_pct_recomputed"
 },
 {
  "v": 1.5998667951767587,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "mixture_vs_self_AB.H64.self.M_eff_diagnostic"
 },
 {
  "v": 1.6218740969733796,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "mixture_vs_self_AB.H128.self.M_eff_diagnostic"
 },
 {
  "v": 24.744226741691627,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "revisions.REV_A_weak_G_ladder.R1_G-1.2.err_max_pct"
 },
 {
  "v": 22.942822274652496,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "revisions.REV_A_weak_G_ladder.R1_G-1.4.err_max_pct"
 },
 {
  "v": 30.142553143524086,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "revisions.REV_A_weak_G_ladder.R2_G-1.4.err_max_pct"
 },
 {
  "v": 34.832165106975765,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "revisions.REV_B_equal_tau_attribution.R1_t1.err_max_pct"
 },
 {
  "v": 7.170422375202154,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "revisions.REV_B_equal_tau_attribution.R3_t1.err_max_pct"
 },
 {
  "v": 22.942822274652496,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "verdict_gates.S1_supplementary_gx_halving_at_stable_G-1.4.err_R1_G-1.4"
 },
 {
  "v": 22.940222709213643,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "verdict_gates.S1_supplementary_gx_halving_at_stable_G-1.4.err_R5_G-1.4"
 },
 {
  "v": 0.5000896433129578,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "verdict_gates.S1_supplementary_gx_halving_at_stable_G-1.4.u_max_R5_over_R1"
 },
 {
  "v": 29,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/explore/outs/probe_stability.json",
  "k": "ladder[1].first_nan.step"
 },
 {
  "v": 46,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/explore/outs/probe_stability.json",
  "k": "ladder[3].first_nan.step"
 },
 {
  "v": 175,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/explore/outs/probe_stability.json",
  "k": "ladder[4].first_nan.step"
 },
 {
  "v": 0.002109392211371581,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[0].relL2"
 },
 {
  "v": 0.0015494927478906233,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[1].relL2"
 },
 {
  "v": 0.010918219215161354,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[2].relL2"
 },
 {
  "v": 0.005596650353003679,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[3].relL2"
 },
 {
  "v": 2.5,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[2].tau_r"
 },
 {
  "v": 0.7,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[2].tau_b"
 },
 {
  "v": 0.03,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/poiseuille_two_phase_cg/result.json",
  "k": "gate.relL2_max_per_arm"
 },
 {
  "v": 0.005596650353003679,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/../verified/poiseuille_two_phase_cg/result.json",
  "k": "arms[3].relL2"
 },
 {
  "v": -1.4,
  "f": "/nfs/wangxi/runs/bm_widen_w5_20260921/mcmp_fix/result.json",
  "k": "",
  "txt": true
 }
]
-->
