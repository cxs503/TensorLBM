# DE-A3 病理诊断 · 数据清点（DE-A3B 换血版；只读清点，不含修复建议）
数据源（只读共享镜像）：D=benchmarks/pending/droplet_impact_3d_pool/result.json（13 run）；P=benchmarks/pending/phasefield_water_entry/result.json；E=…/evidence/s2_zeta4_dw100.json。
纪律：数字均标注 run 名+JSON 字段路径（表内列名即 D.runs.<run>.<列>）；倍数/比值为字段四则运算，标注（÷）。

## 1. 已跑 run 清单（D.runs.*，共 13）
solver 全文：cac12=CACSim/D3Q19；其余=free_surface_step/D3Q19, constant atmospheric pressure（下记 FS）。ulat=lattice_speed。
| run | solver | cells_per_D | ulat | span/pool(/D) | postcontact_healthy_ms | impact_speed_relative_error | stop_reason |
|---|---|---|---|---|---|---|---|
| case3_cac12 | CACSim | 12 | 0.04 | 4.0/3.0 | 1.36200 | 0.01941 | mach_limit |
| case3_cal12 | FS | 12 | 0.06 | 4.0/3.0 | 24.00525 | 0.01781 | window_complete |
| case3_cal8 | FS | 8 | 0.06 | 4.0/3.0 | 13.85105 | 0.00656 | health_limit |
| case3_camera8 | FS | 8 | 0.04 | 4.0/3.0 | 24.00525 | 0.01759 | window_complete |
| case3_d12 | FS | 12 | 0.06 | 4.0/3.0 | 1.25661 | 0.09172 | health_limit |
| case3_d8 | FS | 8 | 0.06 | 4.0/3.0 | 1.04582 | 0.08593 | health_limit |
| case3_dt8 | FS | 8 | 0.04 | 4.0/3.0 | 24.00525 | 0.01918 | window_complete |
| case3_flux8 | FS | 8 | 0.06 | 4.0/3.0 | 1.36200 | 0.08592 | step_limit |
| case3_longflight8 | FS | 8 | 0.04 | 4.0/3.0 | 0.71343 | 0.19722 | step_limit |
| case3_matched8 | FS | 8 | 0.04 | 4.0/3.0 | 19.70846 | 0.04100 | health_limit |
| case3_motion8 | FS | 8 | 0.04 | 4.0/3.0 | 24.00525 | 0.23401 | window_complete |
| case3_release8 | FS | 8 | 0.06 | 4.0/3.0 | 0.74180 | 0.00655 | health_limit |
| case3_wide8 | FS | 8 | 0.06 | 6.0/4.5 | 24.00525 | 0.00678 | window_complete |
注：wide8 是唯一 span_over_D=6.0/pool_depth_over_D=4.5，其余均 4.0/3.0；gap_over_D：cac12=0.13、camera8/matched8=0.78、longflight8=0.60、其余=0.18（D.runs.<run>.gap_over_D）。

## 2. 走得最远的 run 与目标差距
目标 = D.time_of_observed_peak_ms = 10.39446 ms（D.reference_observed_peak=1.857434 depth/D 为观测峰值）。
- 最远组：5 个 window_complete（cal12/camera8/dt8/motion8/wide8）postcontact_healthy_ms=24.00525 ms = 目标的 2.309×（24.00525÷10.39446），已越过目标时刻。
- 早期终止组最远：case3_matched8 19.70846 ms = 1.896× 目标（health_limit，未到窗口末端 24.00525）。
- 中间：case3_cal8 13.85105 ms = 1.333× 目标（已越峰时刻，未到窗口末端）。
- 未达目标即终止的 6 个（post÷10.39446，倍差=10.39446÷post）：longflight8 0.069×（差 14.6×）、release8 0.071×（差 14.0×）、d8 0.101×（差 9.9×）、d12 0.121×（差 8.3×）、cac12 0.131×（差 7.6×）、flux8 0.131×（差 7.6×）。

## 3. 终止原因归类（D.runs.<run>.stop_reason）
| stop_reason | 次数 | run（postcontact_healthy_ms） |
|---|---|---|
| window_complete | 5 | cal12/camera8/dt8/motion8/wide8（均 24.00525） |
| health_limit | 5 | matched8(19.70846)、cal8(13.85105)、d12(1.25661)、d8(1.04582)、release8(0.74180) |
| step_limit | 2 | flux8(1.36200)、longflight8(0.71343) |
| mach_limit | 1 | cac12(1.36200) |
典型先后顺序：接触（contact_step 21–186，D.runs.<run>.contact_step）→ 健康窗口 → 终止。8 个未完成 run 中 6 个死于触水后 ≤1.36200 ms；其 comparisons 仅含首点 t=0.39021 ms（matched_points=1），该点 simulation=0.0 vs experiment=0.0389、relative_error=1.0（D.runs.<run>.comparisons[0]）。cal8 记录 14 点、matched8 记录 18 点后仍被 health_limit 终止。mach_limit 仅见于 lattice_speed=0.04（cac12）；两个 step_limit 的 last_recorded_step=160/205，在 13 run 中仅高于 release8(107)/d8(135) 两个 health_limit 早死者（D.runs.<run>.last_recorded_step）。

## 4. 数据可见规律（仅陈述数据内可见项）
- cells vs 存活：无单调关系。cells_per_D=12：1/3 window_complete（cal12）；=8：4/10。最早死亡（longflight8 0.71343 ms）与 5 个全程 run 中的 4 个都在 8 cells 档。D.converged_grid_pairs=[]（空）；D.passed_3pct_and_converged=False。
- lattice_speed：两组都有幸存者（0.04：3/6 完成；0.06：2/7 完成）；mach_limit 仅在 0.04 组出现 1 次。
- 撞击速度误差 vs 存活：不相关。误差最小 release8 0.00655 死于 0.74180 ms；误差最大 motion8 0.23401 反而跑满 24.00525 ms（D.runs.<run>.impact_speed_relative_error / postcontact_healthy_ms）。
- 同配置异结局：cal8 与 release8 所有已记录配置字段相同（cells 8/ulat 0.06/span 4.0/pool 3.0/gap 0.18/release 1.53/contact_step 45），结局 13.85105 vs 0.74180 ms（同为 health_limit）——结局不由任何已记录字段单独决定。
- 域尺寸：release8(4.0/3.0) 死于 0.74180 ms，仅 span/pool 不同的 wide8(6.0/4.5) 跑满；但同域 4.0/3.0 的 cal12 也跑满——域尺寸单独不决定结局。
- 精度：6 个有峰值的 run peak_relative_error=0.13195–0.23341（5/6 >15% 精度线；matched8 0.13195 为唯一 <15% 者，但仍 >3%）；5 个全程 run max_full_curve_relative_error=1.0（首点 sim=0.0）；points_within_tolerance ≤1/18；numeric_pass=False ×13；D.digitization_resolves_3pct_at_every_point=False；D.domain_and_material_match_confirmed=False；D.times_below_experimental_3pct_resolution_ms=[0.39021]。

## 5. phasefield_water_entry（S2 zeta4_dw100，独立案例，无 ms 目标）
- P.status=failed、P.exit_code=1、P.first_nonfinite_step=141；E.verdict=FAIL、E.measurement_completed=False。
- 时序：E.contact_step=100 → E.first_nonfinite_step=141，触水后 41 步（141−100）出现非有限；拟合窗 E.requested_fit_window_steps=[300,1100] 未到达，E.fit_window_steps / slope_rel_error / fit_r2 均 null。
- fy 发散序列（E.series[].fy，step:值）：137:−451.46 → 138:7740.06 → 139:3.136e9 → 140:1.599e35 → 141:null。
- 门槛：E.max_u_over_v=17.160（E.gate=0.5）→ gate_u_max_pass=False、gate_pass=False；E.phi_range=[9.483e-7, 1.069]（上界>1）。
- 观测量：P.paper_observable=半楔压力合力（非全身体 MEM 力），P.paper_secant_h_over_Dw=[0.002,0.004]；本案例以步数计，无 10.394 ms 对应物。

---
## 复核记录（A3 本单 2026-11-23，m25354）
- 全表 13 run × 15 字段及 §2–§4 全部倍数/比值已由脚本从 4 个源 JSON 程序化重导出逐一比对：除上文 2 处已更正表述外**零数值偏差**；提取快照存 worktree m25354 `artifacts/`（a3_extract_1.txt、a3_extract_2.txt、a3_summary.tsv）。
- 本次更正 2 处：①step_limit 的 last_recorded_step 并非全表最低两档（release8=107、d8=135 更低，均为 health_limit）；②matched8 peak_relative_error=0.13195<15%，"全部>15%"更正为 5/6。
- 结论不变：无任何 run 满足 3% 精度线（numeric_pass=False ×13、points_within_tolerance≤1/18、converged_grid_pairs=[]）；phasefield 臂触水后 41 步非有限，拟合窗未进入。
