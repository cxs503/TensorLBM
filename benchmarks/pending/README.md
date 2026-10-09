# benchmarks/pending — 未验证案例档案（计算现状 · 不达标清单 · 晋级路径）

**定位**：本目录收录尚未达到 `verified/` 严格标准的基准案例。每个案例的
README 记录：物理问题、计算结果现状、不达标清单、根因/诊断、晋级路径；
全部判决数字由各文件末尾 PROVENANCE 块机器溯源（值-文件-键三元组，
可直接核验），零手抄。

**严格标准**（与 `verified/` 一致）：
1. 共性模块入口（`src/tensorlbm` 库路径驱动，非独立脚本）；
2. 直接观测量对预注册锁定参考 ≤3%；
3. ≥2 档网格严格单调收敛；
4. 参考先锁定后跑（防参考选购）；修正/还原变换不算直接模拟。

**状态图例**：🟡 达标待晋级（机器门已过，待归档决策）· 🔵 formal 在跑 ·
⚠️ 未按战役标准验证（遗留档/骨架）· ⛔ 参数面不可行 · ❌ 未达标（多数已定源根因）

**晋级机制**：案例补齐上述各门后，由归档 PR 将 `pending/X/` 移入
`verified/X/` 并同步更新本索引。达标不等于自动晋级——涉及参考裁决、
口径修订、目录批次安排的决策项在各案例 README 的「晋级路径」中列明。

## 案例总表（30）

| 案例 | 状态 | 卡点（一句话） | 晋级动作 | 预算 |
|---|---|---|---|---|
| [backward_step](backward_step/) | ❌ | τ 阶梯最好 -4.33% > 3%（BB 滑移伪差已定源；参考簇散布 ~6% 受限） | 参考裁决 + L4 细档（外推 -1.1%）或改判 Re=150/200 点 | CPU 日内 |
| [blasius_flat_plate](blasius_flat_plate/) | ❌ | C_f 主门最好 +27.34%（有限 Re_x=900 物理位移主导；旧档首因 FD 权重 bug 已修） | 延长板长放大有效 Re_x，保持直接口径 | CPU 日内 |
| [bstep_3d](bstep_3d/) | ⚠️ | 唯一完整档 Re=100 +13.18%；re200/re400 两档记录损坏；无预注册阶梯 | 按冻结协议重制或新工况预注册 | ~10–20× 存量 |
| [cylinder_re40_st](cylinder_re40_st/) | ❌ | formal 六档 err 5.43–6.06% 全超 3%；根因=上游域截断（up 10→40D 压 -2.84pp，40D 单点 +2.236% 已入门） | W11-A2：扩上游 up≥40D 六档重跑 | ≈25–35 GPU·h |
| [naca_0012](naca_0012/) | ❌ | Cl 对锁定参考 0.132 全档 +85~99% 且不单调；参考图文自不一致（0.132 vs 0.26，预注册已判受限） | 参考重锁二选一或另寻 OA 一手源 | 秒级/档 |
| [sphere_re100](sphere_re100/) | ❌ | 三代测力路径全 >3%（surface-MEM 两档 +9.15/+8.92%；压力 -16.56%；MEM 爆炸已修） | 并入 sphere_re100_mem 大域路线一并处理 | 0（随 mem 晋级） |
| [sphere_re100_mem](sphere_re100_mem/) | 🟡 | 无——大域孪生 T1 +2.7708% / T2 +2.5401% 双过 3% 且严格单调（位一致链+稀疏核解锁证据齐全） | 归档晋级（建议同批 sphere_re100） | 0 |
| [sphere_re200](sphere_re200/) | ⚠️ | 旧口径 Ladd Cd +268.5%（历史失效同族；未按修复后估计器重跑） | 按冻结协议重制（BFL+wet-node+域形条款） | ≈10–20 GPU·h |
| [sphere_re100_d3q27](sphere_re100_d3q27/) | 🟡 | 无——全门 PASS（err +2.79%→+2.57% 严格单调；twins 逐位） | 归档晋级 | 0 |
| [couette_3d](couette_3d/) | ❌ | 唯一败因 G3 稳态判据：H128 fp32 未达 drift 门（判据坐 fp32 噪声带；fp64 四档 l2 ≤1.7337e-9 且跨机逐位） | 稳态口径裁定（fp64 踪迹或噪声带重锁） | ≤0.15 GPU·h |
| [stokes_second_problem](stokes_second_problem/) | ⚠️ | 骨架（仅 run.py 设计稿，无机器结果，PROVENANCE 空块如实） | Phase-0 停批 → 冻结 prereg → H∈{40,80} | ≤0.1 GPU·h |
| [taylor_aris_dispersion](taylor_aris_dispersion/) | ❌ | 容差 6/6 全过（0.05–0.26%）但单调 0/3（err=负 O(1/H²)+正地板 H≈50 变号） | 判据口径重锁或更大 H 阶梯（决策项） | 已判死 |
| [collision_kernels_3d_tg](collision_kernels_3d_tg/) | ❌ | 容差/衰减全过（≤0.4713%）但单调 0/5（bgk27 对照同败=τ 阶梯结构性；KBC 黏度修复后 patched 阶梯 -0.67→-0.36→-0.16% 收敛） | 按修复后核重判（侧证据已有） | 侧证据已有 |
| [hadamard_rybczynski_droplet](hadamard_rybczynski_droplet/) | ❌ | λ=ρ 锁死致 λ 阶梯不可运行；E1–E6 动态排除+15 案包络 0 存活 | 相面重建（M1，已落库）后重开 | 已判死 |
| [cavity_3d_full](cavity_3d_full/) | 🟡 | 无——达标数据在 verified/cavity_3d_full（1.338→0.259% 单调）；本目录为 superseded 诊断稿 | 归档 PR 清理（删除或标记 superseded） | 0 |
| [cavity_natural_convection](cavity_natural_convection/) | ❌ | 修复前旧档（库 thermal 壁 BC 双缺陷；库已修复且 2D 六案已 verified） | 按新库重跑预注册（对齐 thermal_cavity） | 小 |
| [channel_turb](channel_turb/) | ❌ | 2D 槽道无转捩路径（Re_max 4200 < 5772）；RMS 4.59/5.61% 且细化更差 | 3D 槽道（新案例） | 高 |
| [dit_turbulence](dit_turbulence/) | ❌ | 固定 C_s 跨网格不收敛（N256 +17.05%，C_s 微调不敏感） | dynamic Smagorinsky / 系综 / 更长运行 | 中 |
| [rayleigh_benard](rayleigh_benard/) | ❌ | E=1.224→1.006→3.101% 非单调 + ny256 近临界臂拟合质量（二次延长散布 3.10/3.98% 仍超）；根因=格子算子 Ra_c 上移 | γ_ref 重锁格子算子特征值（新 prereg） | ~8 GPU·h/轮 |
| [rayleigh_taylor](rayleigh_taylor/) | ⛔ | SCMP 无生长窗口：λ=280 档扫到 vapor-spinodal 重力天花板 91%（g=6.5e-6）σ_sim 仍负；10/10 案例无增长 | 高密度比伪势引擎（或晋级引擎重设计后重开） | 高 |
| [dam_break_3d_mm](dam_break_3d_mm/) | ❌ | 引擎级五连根因：12 检查点反收敛（T*=2 时 -2.25%→+7.53%）、质量门 -49.5/-62.2%、前锋推进与记账同开关 | 晋级引擎重设计（权重触发）后两档重跑 | 高 |
| [dam_break_sc](dam_break_sc/) | ❌ | 仅 T=1 过门（a80 +2.49%）；T=2.96 档 -19%，err(T) 对 G 反向响应（构成本构/密度比缺口） | 未试杠杆：压力张量 SC 力 / guest-host 耦合 / 高密度比 EOS | 中 |
| [capillary_invasion_washburn](capillary_invasion_washburn/) | ❌ | Washburn 指数 26.82→33.17% 反收敛；湿润相质量漂移 -60.7/-67.0%；W128 确定性 NaN | 修库（质量守恒/σ 可测/抗负分布）后新 prereg | 中（库为主） |
| [forchheimer](forchheimer/) | ❌ | err_A 1.86–3.06% 平坦无 d 收敛、单调条款无判别力（参考受限；求解器侧已排除） | 锁弱惯性精确参考后重判或关闭 | 少量重跑 |
| [permeability3d](permeability3d/) | 🟡 | φ=0.343 双规则 PASS（+4.63→+1.45% 单调降）；总判 FAIL 由稀释档拖累（fp32 注入地板+惯性修正已定源） | 晋级 φ=0.343 档（或 fp64/fs≥100 补跑稀释档） | 0 或 2φ×四档 |
| [permeability_fibrous](permeability_fibrous/) | ❌ | 三失效模式全定源（hex 参考侧 +2.4/+3.9%；sq 量化地板振荡；S1 真惯性）；单调三处破缺为实质败因 | 菜单：cut-cell 自产表 / 门放宽 1% / 购精确表 / 维持 FAIL；sq 0.30–0.60 现成候选 | cut-cell 全阵为大项 |
| [poiseuille_3d_annulus](poiseuille_3d_annulus/) | ❌ | 主通道 1.16→0.05% 单调全过；唯 δ_i spread 0.094 > 0.05 预注册界（内壁楼梯曲率 O(1/R) 真实漂移，已证非欠收敛） | 条款修订（零计算）或补 R_o=160 档复核 | 0 或一档 |
| [poiseuille_3d_ellipse](poiseuille_3d_ellipse/) | ❌ | a=20 档 s^Q 逐 bin 5.11% > 3%（a=40 档 2.43% 过线）；楼梯椭圆各向异性无单一尺度口径 | 加 a=40/80 高档（两档全 ≤3% 即达标） | 两档仿真 |
| [two_phase_poiseuille](two_phase_poiseuille/) | ❌ | SC-MCMP 结构性（等 τ 界面条带化 34.83% vs self 7.17%；不等 τ NaN）；问题类已由 verified/poiseuille_two_phase_cg 关闭 | SC 复活需先修库（稳定性+条带化）再预注册 | 修库+重跑 |

**统计**：🟡 4 · 🔵 1 · ⚠️ 3 · ⛔ 1 · ❌ 21（共 30）

**范围**：本表覆盖 30 案。marine sweep 线五案（dtmb5415_resistance、
kcs_resistance、propeller_openwater、series60_resistance、
suboff_sail_resistance）与 aircraft_icing_rg15_ipw2 为本表基线后入库的
pending 案例，不在本表，由各自战役线补档；sphere_re100 / sphere_re100_mem /
sphere_re200 三案此前仅有本地档案，本次建档入库（README 即本表所链）。

待决策汇总（owner）：🟡 四案的归档批次（sphere_re100_mem、sphere_re100_d3q27、
permeability3d φ=0.343、cavity_3d_full 清理）· cylinder W11-A2 扩上游 ·
couette/annulus/taylor_aris/collision 的口径类裁定 · naca/backward_step/
forchheimer/fibrous 的参考类裁决。

<!-- PROVENANCE
[
 {
  "v": -4.33,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json",
  "k": "ladder_tau_matched.rows[2].err_psi_vs_erturk_pct"
 },
 {
  "v": -1.1,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json",
  "k": "artifact_decomposition.extrapolated_continuum_limit.err_vs_erturk_pct"
 },
 {
  "v": 27.381498847328196,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json",
  "k": "runs[2].cf_err_pct_corrected_fd"
 },
 {
  "v": 13.184865315755209,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json",
  "k": "result.xr_h_err_pct"
 },
 {
  "v": 5.432410628124695,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json",
  "k": "re40.rows[2].err_pct"
 },
 {
  "v": 6.057580319757072,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/formal/ladder_summary.json",
  "k": "re20.rows[0].err_pct"
 },
 {
  "v": -2.8363725847980747,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json",
  "k": "single_variable_ladder[1].delta_vs_baseline_pp"
 },
 {
  "v": 2.2358346717707533,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/floor_20261008/summary.json",
  "k": "single_variable_ladder[4].err_pct"
 },
 {
  "v": 0.132,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json",
  "k": "reference.cl_ref_locked"
 },
 {
  "v": 0.26,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/naca_0012/result.json",
  "k": "reference.cl_ref_disclosure_text_implied"
 },
 {
  "v": 9.15,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json",
  "k": "mesh_trend_lateral16[2].err_pct"
 },
 {
  "v": 8.92,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result_mem_surface.json",
  "k": "mesh_trend_lateral16[5].err_pct_step4000"
 },
 {
  "v": -16.56,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json",
  "k": "err_pct_vs_SN.cd_total"
 },
 {
  "v": 2.7707656824703957,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json",
  "k": "domain_convergence_signature[5].err_pct"
 },
 {
  "v": 2.5400861325477218,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json",
  "k": "runs[1].err_pct"
 },
 {
  "v": 268.5,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re200/result.json",
  "k": "err_pct_vs_sn"
 },
 {
  "v": 2.791717311316592,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json",
  "k": "t1.err_pct"
 },
 {
  "v": 2.569217284710319,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json",
  "k": "t2.err_pct"
 },
 {
  "v": 1.733723620717403e-09,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json",
  "k": "per_tier_fp64.128.l2_rel_err"
 },
 {
  "v": 0.14812899907521598,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json",
  "k": "part2.W32.fit_rms_rel"
 },
 {
  "v": 0.05,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/backward_step/result.json",
  "k": "ladder_tau_matched.rows[0].u_max"
 },
 {
  "v": 0.04956418210400049,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/taylor_aris_dispersion/result.json",
  "k": "per_case[2].err_pct"
 },
 {
  "v": 0.26236133310775234,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/taylor_aris_dispersion/result.json",
  "k": "per_case[3].err_pct"
 },
 {
  "v": 0.47128549235811595,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/collision_kernels_3d_tg/result.json",
  "k": "kernels.trt27.cases[2].err_e_pct"
 },
 {
  "v": -0.6747755960422056,
  "f": "/nfs/wangxi/runs/bm_widen_w7_20260922/kbc_fix/tg/result.json",
  "k": "cases[0].err_e_pct"
 },
 {
  "v": 0.37382388804141997,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/collision_kernels_3d_tg/result.json",
  "k": "kernels.cumulant_d3q27.cases[1].err_e_pct"
 },
 {
  "v": 0.1825360010008122,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/collision_kernels_3d_tg/result.json",
  "k": "kernels.cumulant_d3q27.cases[0].err_e_pct"
 },
 {
  "v": 1.338,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json",
  "k": "grids.64.err_pct_vs_ku"
 },
 {
  "v": 0.259,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json",
  "k": "grids.96.err_pct_vs_ku"
 },
 {
  "v": 4.586411042880767,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/128x32/result.json",
  "k": "errors.rms_vs_dns"
 },
 {
  "v": 5.610730718550807,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/channel_turb/runs/256x64/result.json",
  "k": "errors.rms_vs_dns"
 },
 {
  "v": 17.05,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dit_turbulence/result.json",
  "k": "verdict.slope_err_pct_per_grid.256"
 },
 {
  "v": 1.221918649335052,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json",
  "k": "verdict.per_tier.hex_vf0.3.errors_pct[2]"
 },
 {
  "v": 1.0002975347565441,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json",
  "k": "result.inlet_profile_check.flux_ratio"
 },
 {
  "v": 0.031010378209105776,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict.json",
  "k": "gates.E_ny.256"
 },
 {
  "v": 0.0397985223664529,
  "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal/verdict_ext.json",
  "k": "arm_rows[12].slope_spread_rel"
 },
 {
  "v": 7.140580075786938e-06,
  "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json",
  "k": "cases.lam280_g6p5e6.feasibility.g_ceiling_vapor_spinodal"
 },
 {
  "v": -2.248,
  "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json",
  "k": "formal_fail_ladder.Z_err_pct.a40.T*=2"
 },
 {
  "v": 7.529,
  "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json",
  "k": "formal_fail_ladder.Z_err_pct.a80.T*=2"
 },
 {
  "v": -49.46,
  "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json",
  "k": "formal_fail_ladder.mass_drift_pct.a40"
 },
 {
  "v": -62.19,
  "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json",
  "k": "formal_fail_ladder.mass_drift_pct.a80"
 },
 {
  "v": 2.49,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json",
  "k": "official_rerun_correct_convention.a80.err_pct_T1"
 },
 {
  "v": 2.96,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json",
  "k": "reference_digitisation_lethe.T[13]"
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
  "v": -60.677107119080844,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json",
  "k": "part2.W32.massw_drift_pct"
 },
 {
  "v": -66.97527211128853,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/capillary_invasion_washburn/result.json",
  "k": "part2.W64.massw_drift_pct"
 },
 {
  "v": 0.018575454200680075,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json",
  "k": "rows[3].err_A"
 },
 {
  "v": 0.03058411665542915,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json",
  "k": "rows[8].err_A"
 },
 {
  "v": 0.3430018424987793,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json",
  "k": "amendment1_ladders.phi0.343.tiers[2].phi_actual"
 },
 {
  "v": 0.04625290433934337,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json",
  "k": "amendment1_ladders.phi0.343.tiers[0].err"
 },
 {
  "v": 0.014468555124078408,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability3d/result.json",
  "k": "amendment1_ladders.phi0.343.tiers[3].err"
 },
 {
  "v": 2.418694925947107,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json",
  "k": "verdict.per_tier.sq_vf0.4.errors_pct[2]"
 },
 {
  "v": 3.851477863002062,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json",
  "k": "verdict.per_tier.sq_vf0.7.errors_pct[2]"
 },
 {
  "v": 0.3383014680396368,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json",
  "k": "verdict.per_tier.hex_vf0.75.errors_pct[1]"
 },
 {
  "v": 0.6218225965950896,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/permeability_fibrous/result.json",
  "k": "verdict.per_tier.hex_vf0.6.errors_pct[2]"
 },
 {
  "v": 1.1634,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json",
  "k": "per_grid[0].eff_max_bin_central_pct"
 },
 {
  "v": 0.0514,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json",
  "k": "per_grid[2].eff_max_bin_central_pct"
 },
 {
  "v": 0.0937,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_annulus/result.json",
  "k": "delta_spread_i"
 },
 {
  "v": 5.106962011179169,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json",
  "k": "per_grid[0].max_rel_bin_central_sQ_pct"
 },
 {
  "v": 2.431766861465193,
  "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/poiseuille_3d_ellipse/result.json",
  "k": "per_grid[1].max_rel_bin_central_sQ_pct"
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
  "v": 6.5e-06,
  "f": "/nfs/wangxi/runs/bm_widen_20260919/rt_instability/result.json",
  "k": "cases.lam280_g6p5e6.config.gy"
 }
]
-->
