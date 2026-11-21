# m18429 机生成核验表（verify_m18429.py 输出，禁手改）

## T1 Maxwell 等面积构造（ref 侧）

| 目标比 | T_r | rho_l | rho_v | 构造比 | |面积残差|/p_sat dV | 命中目标(<=1%) |
|---|---|---|---|---|---|---|
| 10 | 0.826562 | 5.887082 | 0.587962 | 10.01269 | 1.39e-13 | True |
| 100 | 0.568437 | 7.885447 | 0.078877 | 99.97081 | 1.05e-12 | True |
| 1000 | 0.410938 | 8.765436 | 0.008808 | 995.15275 | 1.22e-11 | True |

## T2 本单工作树真实存在的 run 工件（planar control, 40000 步）

| tag | steps | ratio_got | ratio_ref | 误差 | 相对误差 | Ma | mass_rate/step | 工件判决 |
|---|---|---|---|---|---|---|---|---|
| planar_ratio10_edm | 40000 | 11.0156 | 10.0127 | +1.0029 | 0.1002 | 0.1141 | 2.254e-09 | FAIL (spurious Ma=0.114 > 0.1) |

## T3 历史战役表（RESULTS.md, m8675 2026-11-10 生成；本单未重跑）

| run | forcing | stable | steps | 测比 | Maxwell 比 | 相对误差 | Ma | 表内判决 |
|---|---|---|---|---|---|---|---|---|
| tier_ratio10_edm_L96R20 | edm | True | 20000 | 10.450 | 10.010 | 0.0437 | 0.115 | FAIL |
| tier_ratio10_guo_L96R20 | guo | True | 20000 | 13.810 | 10.010 | 0.3790 | 0.127 | FAIL |
| tier_ratio100_edm_L96R20 | edm | True | 20000 | 293.300 | 99.970 | 1.9300 | 0.333 | FAIL |
| tier_ratio100_guo_L96R20 | guo | False | 23 | — | 99.970 | — | — | FAIL |
| tier_ratio1000_edm_L96R20 | edm | False | 100 | — | 995.200 | — | — | FAIL |
| tier_ratio1000_guo_L96R20 | guo | False | 16 | — | 995.200 | — | — | FAIL |

## T4 审计侧独立短程复跑（main 检出, driver sha256 同源）

| steps | ratio_got | ratio_ref | 误差 | Ma | stable | 判决 |
|---|---|---|---|---|---|---|
| 200 | 10.3493 | 10.0127 | 0.0336 | 0.1336 | True | FAIL (spurious Ma=0.134 > 0.1; steps_run=200 < 20000 hold requirement) |

## T5 判据逐项（PASS/FAIL 由阈值机器比）

- maxwell_all_targets_hit_1pct = True
- tier10_edm_ratio_err_within_owner_15pct = True
- tier10_edm_spurious_ma_le_pass_gate = False
- tier100_edm_within_owner_15pct = False
- tier1000_edm_stable = False
- control_planar_ratio_err_within_owner_15pct = True
- control_planar_spurious_ma_le_pass_gate = False
- benchmark_cases_registered = 0
- tensor_reports_for_m18366 = 0
- commits_mentioning_m18366 = 0

## T6 工件/代码 provenance

- driver sha256 (artifact) = ae393b704946b0ce6ba0de8712c14625bea539509193d4e268e129250349a355
- driver sha256 (main 检出) = ae393b704946b0ce6ba0de8712c14625bea539509193d4e268e129250349a355
- probe  sha256 (artifact) = 31fbdd7ccf0bd9a55919200691e8c36405837af0a2e7f5f25ef5345732c5b597
- probe  sha256 (main 检出) = 31fbdd7ccf0bd9a55919200691e8c36405837af0a2e7f5f25ef5345732c5b597
- m18366 out/results.json sha256 = 755f39cee943e24576b85765b0d536b168f83fdb99cfa95a03627bd78e3abaea
- RESULTS.md 生成于 2026-11-10T05:02:05.402384+00:00；out/results.json 生成于 2026-11-13T13:57:01.862525+00:00（两者不同代）
- m18366 工作树未提交条目 = 4；git log --all --grep=m18366 命中 = 0
- mission(18366) status/head_commit/branch = succeeded|NULL|NULL
- pilot 臂工件存在? False（工件内 tags=['planar_ratio10_edm']）
