# T4_DIGEST · 红方攻击官（对蓝方 DE-T4-B1R 存活修复）· 压缩官蒸馏
> 源卡：BB/teams/T4/RED.md（主，DE-T4-R1，m26319）+ BLUE.md + REJECT_DE-T4-B1R.md · 2026-11-20 · 用途=只供新派队只读
> 自检：5/5（①文件在且非空 ②数字可溯 ③双写commit ④预测区间 ⑤失败如实）
> CROSS已读：T1_DIGEST（CACSim 存活修复，n12=1.362ms 基线恢复）；T2_DIGEST（三开关公共因子，全域臂 24.005ms 但 peak_error 18.21%）；T3_DIGEST（双治冠军，存活 5/7）；T5_DIGEST（单打对照，L/I 单向记账+LIQUID 密度漂移，3% 门未过）
> 上游验收：PASS（RED.md 数字全带出处；REJECT_DE-T4-B1R.md 首行含任务键+理由，存活数字保留、机制结论退回）
## 核心结论
- 红方复跑（cj5678）证伪蓝方「mach 尖峰由亚气相密度袋（rho 0.724）顶穿」因果：死亡步 step1185 尖峰胞 rho_at_mach_max=0.9746（轻度亏空非深袋），u=mom/rho 放大仅 1.026x；mach_liquid_max=0.0212（尖峰全在界面胞）——0.3 门由界面胞 ABB 重构速度主导，与液流可压缩性无关（RED#1/#4）。
- floor 无效且降敏：尖峰胞 rho=0.975 → floor 仅削 2.5%（0.3006→0.2930），floored mach 仍 +0.005/4步 爬升（蓝方 run step1189 已 0.2948，约 step1197 二次触 0.3）；只把死亡从 mach_limit 换成 drift_limit（13.851→13.924，+0.073ms，+0.5%），且 score.py L42 静默改写门语义（RED#2）。
- 精度门未过（蓝方自认）：dt8+floor 峰深 1.424 vs 实验 1.857D = -23.3%，远超 3% 门与 15% 底线，numeric_pass=False（RED#3）。
- 根因在内核 ABB 重构（f_abb=f_eq_gas+f_eq_gas[opp]−f_post[opp]，以 u_pc=mom/rho 入 equilibrium 注入非物理动量），经界面→液体转换把密度亏空带进液相（深袋 0.582 → drift 1.12e-4 触门）；probe floor 属治标（RED#4）。
## 关键数字（全部带原卡出处）
| 量 | 值 | 出处 |
|---|---|---|
| 复跑 postcontact_healthy_ms | 13.851（精确复现 A3 原始 cal8；蓝方 floor 版 13.924 仅 +0.5%） | RED#3 |
| 死亡步 | step1185，stop=health_limit@14.410ms，mach_max=0.3006>0.3 | RED#3 |
| 尖峰胞密度 | rho_at_mach_max=0.9746（非深袋），u 放大 1.026x | RED#1 |
| 液相/界面 Mach | mach_liquid_max=0.0212 / mach_interface_max=0.3006 | RED#3 |
| 最深袋步 | step1190 rho 0.582，mach 反而全场最低 0.156 | RED#1 |
| 尖峰步密度带 | step1185-1189 rho_min≈0.976（蓝方工件） | RED#1 |
| Pearson(mach,1/rho_min) | 0.256（蓝方工件）/0.548（复跑） | RED#1 |
| floor 削幅 | 0.3006→0.2930（-2.5%） | RED#2 |
| 死亡步 drift | 1.06e-5（未触 1e-4；蓝方 floored 晚 5 步死于 1.12e-4） | RED#3 |
| dt8+floor 峰深 | 1.424 vs 实验 1.857D = -23.3% | RED#3 |
| 液相 Mach 全程 | ≤0.137（死亡步仅 0.0212） | RED#4 |
| 预测区间（dt8+floor） | [24.0,24.01]（window_complete，本单未复跑，如实注明） | RED#4 |
| 预测区间（rho≥0.99 工况） | floored 与 raw mach 差 <1%（floor 无感） | RED#4 |
| 证伪判据（尖峰在液相） | 复跑 mach_liquid_max 应 ≥0.29；实测 0.0212，归因成立 | RED#4 |
## 可复用方法/仪器
- 红方修复（probe 侧最小 +18 行）：free_surface_probe.py（agent/m26319）history 增记 mach_floored/mach_liquid_max/mach_interface_max/rho_at_mach_max，gate 保持原始 lattice Mach（协议语义）——把「伪迹」论断变可证伪测量（RED#2）。
- 复跑命令：cells=8 gap.18 speed.06 cap simple rel1.53，原始门（集群 cj5678，n330，8 核）；工件 runs/t4_red/case3_cal8_red.json，蓝方对照 m26312/runs/t4_blue/case3_cal8_floor.json（RED#3）。
## 诚实披露的失败与局限
- 未跑：dt8+floor 复跑、12 cells/D 收敛、内核 ABB 根因修复及全基准回归（预算所限）（RED#5）。
- 残余风险：界面胞 ABB 注入动量仍污染界面动力学（或为峰深 -23% 偏差一部分）；score.py 消费 mach_max 的语义问题在蓝方分支仍存在（RED#5）。
- 治本需动 free_surface_lbm.py ABB/转换路径，本单未动（超最小改动范围）（RED#4）。
