# 2D 溃坝 dam_break_sc（Shan-Chen 单组分伪势，Martin & Moyce 前沿 X(T)）（pending）

**状态：❌ 未达标（本构/密度比缺口，2026-09-29 二次判决）——修正时间轴与参考表双重错误后，仅最早检查点可达标（a=80, T=1: +2.49%），误差随 T 单调恶化到 -19%；err(T=1) 与 err(T=2.96) 对 G 的依赖方向相反，任何稳定 (G, τ) 组合都无法三检查点同过；质量守恒的 Guo 力已落库（opt-in）但与 velocity_shift 实质等价。**

## 物理问题

Martin & Moyce (1952) 溃坝（2D）：液柱坍塌前沿 X=x_front/a 对 T_MM=t·sqrt(2g/a)
（M&M 约定），参考 = Lethe 官方后处理脚本同源的数字化表（14 点，T∈[0,2.96]），
检查点 T=1/2/2.96 → X=1.326/2.360/3.670（线性内插，禁止外推）。
模型 = 2D SCMP 伪势（psi=exp(-ρ)，G_lib=5.0 → G_eff=-5.0，τ=1.0），
rho_l=1.957 / rho_v=0.1596（密度比 12.3），g=2e-4（浮力修正 g_eff=1.83689320388349e-4，
主轴仍用 M&M 的 g），1600 步。库模块：`multiphase.collide_sc_single_component` /
`multiphase.psi_exp` / `solver.stream` / `boundaries.bounce_back_cells`。

**口径纪律（本案例的关键贡献）**：历史 +50~82% 误差是双重错误的乘积——
(i) 时间轴 T=t·sqrt(g/a) 缺 sqrt(2)；(ii) 参考表错（把 M&M 的 T=3 值 2.7 映到
T=2）——已全部修正并**撤回旧数字**（`convention_and_table_fix.double_error_retraction`）；
`bench_fs_2d.py` 的 REF_T/REF_Z/T_TARGETS/SQRT_GA 已同步修正。

## 计算结果现状（result.json `official_rerun_correct_convention`，2026-09-29，真跑 CPU）

| 档 | X(T=1) | err% | X(T=2) | err% | X(T=2.96) | err% | 质量漂移 |
|----|--------|------|--------|------|-----------|------|----------|
| a=40 | 1.4656 | **+10.526** | 2.1811 | -7.579 | 2.9401 | **-19.889** | 1.43e-05 |
| a=80 | 1.359 | **+2.49**（唯一过门） | 2.1305 | -9.723 | 2.9695 | **-19.086** | 1.4e-05 |

g_eff 副轴（浮力修正时间轴，仅报告）：a40 +12.177/-3.612/-17.118%。

两档收敛（|X_a40-X_a80|/X_a80）：baseline T=1 7.84% **FAIL**（T=2 2.38%、
T=2.96 0.99% 过）；Guo 最优臂 T=1 4.93% **FAIL**。

Guo 力扫参（a=80，g=2e-4，质量守恒源项；err% @ T=1/2/2.96）：

| 臂 | T=1 | T=2 | T=2.96 |
|----|-----|-----|--------|
| guo G=5.0 τ=0.8 | +5.66 | -7.75 | -17.04 |
| guo G=4.6 τ=0.8 | +31.11 | +16.61 | +1.01 |
| velocity_shift G=5.0 τ=0.8 | +4.71 | -7.75 | -16.70 |

无任何稳定 (G,τ) 三检查点同 ≤3%。稳定性边界：Guo 稳定需 τ≥~0.70 且 G≤5.0
（G=5.2+τ≤0.8 自激 NaN；psi=sqrt(ρ) 在 G_lib=+5 直接 NaN@25 步）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：✓（multiphase/solver/boundaries 库原语；Guo 力为 opt-in
  库内落地 `collide_sc_single_component(..., forcing=...)`）。
- 直接观测量 ≤3%：✗（细档最佳 +5.66/-7.75/-17.04，仅 T=1 过；粗档 +10.5%）。
- ≥2 档网格单调收敛：✗（T=1 档间 7.84%（baseline）/4.93%（Guo）均 >3%）。
- 质量守恒：✓（≤1.4e-5）——失败不是质量账问题，是物理本构问题。

## 根因/诊断

1. **本构/密度比缺口（主因）**：模拟柱坍塌过快（残余高度 H~0.16-0.21 vs
   M&M ~0.42-0.5），势能转前沿动量不足 → T≥2 前沿系统性偏慢。SC94 在 G=5
   给出密度比 ~13（本档 12.3）vs M&M 水/空气 ~800；rho_v~0.16 的"气体"对
   前沿施加显著拖曳。
2. **结构性障碍**：err(T=1) 与 err(T=2.96) 对 G 单调依赖**方向相反**
   （降 G 加速晚期前沿→修 T=2.96 但炸 T=1，见扫参表 G5.0→G4.6），
   单标度调参不可解。
3. **两次撤回（记录卫生）**：(i) 历史 +50~82%（双重口径错误）已撤回；
   (ii) "Guo 力是决定性杠杆"（2026-09-29 commit 1390767）已撤回——原型
   `/root/db_guo.py` 源项漏掉标准 -(u·F)/cs² 项，逐步净造质量
   （sum S=-0.0123/步 → 1600 步漂移 0.11）；质量守恒修正后
   sum S=2.7e-09、漂移 1.7e-05，Guo ≈ velocity_shift，"界面伪力"叙事不成立。
4. 已排除路径：psi=sqrt(ρ)（NaN）、wall_psi=0.4（不收敛）、质量守恒 Guo
   （等价无力）、G 4.6-5.2 / τ 0.7-1.0 / g 1e-4-3e-4 全扫（g=3e-4 时
   T=2/2.96 恶化到 +35.87/+69.62）。

## 晋级路径

- 未试的下一级杠杆（result.json `recommendation`）：Ramshaw-Phathanapirom
  压力张量 SC 力；guest-host 耦合力；高密度比伪势（Carnahan-Starling/
  幂律 EOS，需重测共存密度）。
- 判据口径已冻结（T_MM 轴 + Lethe 表 1.326/2.360/3.670）；新验证轮直接可用。
- 达标即归档 PR 移 `pending/dam_break_sc` → `verified/`。

## 复现

```
cd benchmarks/pending/dam_break_sc
PYTHONPATH=../../../src python run.py --a {40,80} --forcing {velocity_shift,guo}
# 产物: case_a{40,80}.json, hist_a{40,80}.csv, guo_sweep.json
# 重算（无重模拟）: scripts/recompute_mm_errors.py -> ../dam_break_recompute_mm.json
```

<!-- PROVENANCE
[{"v": 12.3, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.density_ratio"},
 {"v": 1.957, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.rho_l"},
 {"v": 0.1596, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.rho_v"},
 {"v": 5.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.G_lib"},
 {"v": 1.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.tau"},
 {"v": 0.0002, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.g"},
 {"v": 0.000183689320388349, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.g_eff"},
 {"v": 1600, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.steps"},
 {"v": 2.96, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "reference_digitisation_lethe.T[13]"},
 {"v": 3.67, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "reference_digitisation_lethe.Z[13]"},
 {"v": 1.326, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_sc a=40 (g=2e-4).rows[0].newB_new.X_ref"},
 {"v": 2.36, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_sc a=40 (g=2e-4).rows[1].newB_new.X_ref"},
 {"v": 3.67, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_sc a=40 (g=2e-4).rows[2].newB_new.X_ref"},
 {"v": 1.4656, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.X_T1"},
 {"v": 10.526, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T1"},
 {"v": 2.1811, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.X_T2"},
 {"v": -7.579, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T2"},
 {"v": 2.9401, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.X_T2p96"},
 {"v": -19.889, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T2p96"},
 {"v": 12.177, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T_MM_geff.T1"},
 {"v": -3.612, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T_MM_geff.T2"},
 {"v": -17.118, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T_MM_geff.T2p96"},
 {"v": 1.43e-05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.mass_drift_max"},
 {"v": 1.359, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.X_T1"},
 {"v": 2.49, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.err_pct_T1"},
 {"v": true, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.pass_T1"},
 {"v": 2.1305, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.X_T2"},
 {"v": -9.723, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.err_pct_T2"},
 {"v": 2.9695, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.X_T2p96"},
 {"v": -19.086, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.err_pct_T2p96"},
 {"v": 1.4e-05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.mass_drift_max"},
 {"v": 7.84, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "two_grid_convergence_a40_vs_a80.baseline_velocity_shift.T1.pct"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "two_grid_convergence_a40_vs_a80.baseline_velocity_shift.T1.pass"},
 {"v": 2.38, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "two_grid_convergence_a40_vs_a80.baseline_velocity_shift.T2.pct"},
 {"v": 0.99, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "two_grid_convergence_a40_vs_a80.baseline_velocity_shift.T2p96.pct"},
 {"v": 4.93, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "two_grid_convergence_a40_vs_a80.guo_G5.0_tau0.8.T1.pct"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "two_grid_convergence_a40_vs_a80.guo_G5.0_tau0.8.T1.pass"},
 {"v": 5.66, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.guo_G5.0_tau0.8[0]"},
 {"v": -7.75, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.guo_G5.0_tau0.8[1]"},
 {"v": -17.04, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.guo_G5.0_tau0.8[2]"},
 {"v": 31.11, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.guo_G4.6_tau0.8[0]"},
 {"v": 16.61, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.guo_G4.6_tau0.8[1]"},
 {"v": 1.01, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.guo_G4.6_tau0.8[2]"},
 {"v": 4.71, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.velocity_shift_G5.0_tau0.8[0]"},
 {"v": -7.75, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.velocity_shift_G5.0_tau0.8[1]"},
 {"v": -16.7, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.a80_sweep_g2e4.velocity_shift_G5.0_tau0.8[2]"},
 {"v": -0.0123, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.decisive_artifact_retraction.numeric_proof.sum_S_proto"},
 {"v": 2.7e-09, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.decisive_artifact_retraction.numeric_proof.sum_S_guo"},
 {"v": 0.11, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.decisive_artifact_retraction.numeric_proof.mass_drift_proto_1600steps"},
 {"v": 1.7e-05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.decisive_artifact_retraction.numeric_proof.mass_drift_guo_1600steps"},
 {"v": 35.87, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.g_dependence_guo_G5_tau0.8.g=3e-4[1]"},
 {"v": 69.62, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.g_dependence_guo_G5_tau0.8.g=3e-4[2]"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "checks.accuracy_le_3pct_fine_grid.pass"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "checks.two_grid_convergence_le_3pct.pass"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "verified"},
 {"v": -5.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "model.G_eff"},
 {"v": 4.6, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "ruled_out[3]", "in_str": true},
 {"v": 5.2, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "ruled_out[3]", "in_str": true},
 {"v": 0.7, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "ruled_out[3]", "in_str": true},
 {"v": 0.8, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_sc/result.json", "k": "guo_forcing_investigation.stability.G", "in_str": true}]
-->
