# DE-T4-B1R-R2 · 蓝方工程官 · 存活修复（退回重做）
自检：5/5
CROSS已读：T1_DIGEST、T2_DIGEST、T3_DIGEST、T5_DIGEST
上游验收：PASS（退回单 REJECT_DE-T4-B1R.md 为重做依据；A1/A2/A3/A4 紧缩卡数字可溯）

## 1. 重做依据（REJECT 首行任务键 NEW:DE-T4-R1）
- 红方复跑 cj5678：尖峰胞 rho=0.9746（非深袋）、mach_liquid_max=0.0212（尖峰全在界面胞）
- R1 密度 floor 仅削 2.5% 且 mach 仍爬升=降敏非修复 → 机制结论+门语义修改退回
- 本单：回退 R1 门语义（恢复原始 lattice Mach 测量），重诊断，修动量

## 2. 机制（本单实测，非推断）
- 对照臂（u_limit=0，cj5681）：health_limit@13.851ms，mach 0.3006，尖峰胞 (24,16,19) f=INTERFACE rho=0.9746 fill=0.548 |u|=0.1736（2.9×U）
- 诊断（--diag-out）：t~12.2ms 腔体塌缩在界面带 z=24、y/x∈[12,19]（~8 胞）激发弱不稳定声学模态，周期~4 步，幅值 0.10→0.1736 渐增（~1.5%/周期）
- 液相本体健康（mach_liquid_max 0.0212，与红方一致）；尖峰=界面带振荡，非单胞失控、非密度袋

## 3. 补丁（3 文件，commit 404a46d71 + 0830e5032）
- free_surface_lbm.py：free_surface_step 增 u_limit 参（默认 0=关）；界面胞 |u|>u_limit 时按同密度同方向限幅重置种群（+22 行，求解器其余零改动）
- free_surface_probe.py：--u-limit（默认 2.5*speed=0.15，0 关）+ --diag-out 诊断旗标（默认关，零行为改变）
- d3q27.py：MRT 基 Gram-Schmidt 与秩检查 numpy→torch（loong64 LAPACK SVD 坏致导入崩，cj5679 实证）；矩阵逐位一致（max|Δ|=0.0）

## 4. 实测（cells=8，gap .18，speed .06，capacity+simple-exchange+release 1.53，span4/pool3）
| 量 | 对照(u_limit=0) | 修复(u_limit=0.15) | 出处 |
|---|---|---|---|
| postcontact_healthy_ms | 13.851 | **24.005**（满窗） | score.json |
| stop_reason | health_limit | window_complete | score.json |
| max_mach | 0.3006 | 0.2598 | score.json |
| 限幅触发步数 | — | 23/2020（1.1%） | diag_cal8_fixed.json |
| impact_speed 误差 | 0.66% | 0.66% | score.json |
| 峰值深度 | 未覆盖 | 1.4352 D@10.543ms | score.json |
| 峰值误差 | — | 22.73%（>3%） | score.json |
| 体积漂移 max | 9.99e-5 | 8.55e-5 | score.json |

## 5. 预测区间（预登记，实测前写死）
- 修复臂 [13.851, 24.005]，最可能 24.005；证伪判据 <13.851。实测 24.005=命中上界
- 对照臂 [13.5, 14.2]。实测 13.851=命中

## 6. 自评最弱点
- 限幅是幅值稳定化（cap 伪电流），非根因修复：模态仍振荡（mach 0.13~0.26），只是被削在门内；门若收紧到 0.25 仍会死
- u_limit=2.5*speed 为经验值（真射流~2U，尖峰 2.9U，取中）；未做 u_limit 扫描
- 精度仍 honest-FAIL：peak_error 22.7%>3%、18 点 0 点过 3%（与 T5 已知气穴偏低 13~23% 同源，非本单范围）
- 单网格无收敛对；numeric_pass=False
