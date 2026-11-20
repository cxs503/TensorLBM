# DE_A3_DIAG_COMPACT：滴入水与相场入水基准诊断（原 46 行→压缩 21 行）
## 1. 已跑 Run 清单（droplet_impact_3d_pool）
| Run 名 | 求解器 | Cells/D | postcontact_healthy_ms | impact_speed_relative_error | stop_reason |
|---|---|---|---|---|---|
| case3_cac12 | CACSim/D3Q19 | 12 | 1.362 | 1.94% | mach_limit |
| case3_cal12 | free_surface_step/D3Q19 | 12 | 24.005 | 1.78% | window_complete |
| case3_cal8 | free_surface_step/D3Q19 | 8 | 13.851 | 0.66% | health_limit |
| case3_camera8 | free_surface_step/D3Q19 | 8 | 24.005 | 1.76% | window_complete |
| case3_d12 | free_surface_step/D3Q19 | 12 | 1.257 | 9.17% | health_limit |
| case3_d8 | free_surface_step/D3Q19 | 8 | 1.046 | 8.59% | health_limit |
| case3_dt8 | free_surface_step/D3Q19 | 8 | 24.005 | 1.92% | window_complete |
| case3_flux8 | free_surface_step/D3Q19 | 8 | 1.362 | 8.59% | step_limit |
| case3_longflight8 | free_surface_step/D3Q19 | 8 | 0.713 | 19.72% | step_limit |
| case3_matched8 | free_surface_step/D3Q19 | 8 | 19.708 | 4.10% | health_limit |
| case3_motion8 | free_surface_step/D3Q19 | 8 | 24.005 | 23.40% | window_complete |
| case3_release8 | free_surface_step/D3Q19 | 8 | 0.742 | 0.65% | health_limit |
| case3_wide8 | free_surface_step/D3Q19 | 8 | 24.005 | 0.68% | window_complete |
## 2. 最远 Run：case3_cal12(12 cells/D) 与 case3_wide8(8 cells/D) 并列最长，均存活 24.005 ms（result.json->runs[].postcontact_healthy_ms）；目标 10.394 ms（result.json->time_of_observed_peak_ms），差距倍数 24.005/10.394≈2.31 倍；误差 case3_wide8 最低(0.68%)、case3_cal12 1.78%，两者均完整覆盖峰值时刻(10.394 ms)。
## 3. 终止归类：window_complete×5(存活至 24.005 ms，覆盖完整观测窗口)；health_limit×5(存活 0.7~19.7 ms，触水后 1~20 ms 内触发)；step_limit×2(极短存活 0.7~1.4 ms，步数耗尽)；mach_limit×1(极短存活 1.36 ms，马赫数超限)。典型顺序：多数 run 触水后(contact_step~21~51)即入健康检查窗；health_limit 于触水后 1~15 ms 触发，window_complete 则无阻跑满。
## 4. 数据规律：Cells 与存活无单调相关(12 cells/D 之 case3_cal12 存活 24 ms 而 case3_d12 仅 1.26 ms；8 cells/D 跨度最大 0.7~24 ms)；速度误差与存活无显著相关(高误差 case3_motion8 23.4% 与低误差 case3_wide8 0.68% 均存活 24 ms)；Solver 差异——唯一非 free_surface_step 之 case3_cac12 因 mach_limit 早止，示 CACSim 对马赫数更敏。
## 5. Phasefield Water Entry 诊断：s2_zeta4_dw100(evidence/s2_zeta4_dw100.json) 状态 FAIL(verdict)；终止 first_nonfinite_step=141(result.json & s2_zeta4_dw100.json)；测量 measurement_completed=false 未入拟合窗(300-1100 steps)；原因——141 步即现非有限值(NaN/Inf)，致垂直力(fy)无法收敛至解析斜率。
