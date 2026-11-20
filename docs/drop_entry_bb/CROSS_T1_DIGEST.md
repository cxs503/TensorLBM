# T1_DIGEST · 跨队蒸馏（DE-X-DIGEST-T1）— 双相(CACSim)路径存活修复
自检：5/5（文件在且非空✓ 数字可溯✓ 双写commit 9aa652ed7✓ 预测区间✓ 失败如实✓）
CROSS已读：无（CROSS/ 目录为空，无 *_DIGEST.md）
上游验收：PASS——AUDIT 修正单（P1+R1+禁令）全部执行；A3#1 锚点 1.362ms 复现成功，无需退回单。
## 核心结论
- 战役=DE，T1 队攻双相(CACSim)路径存活修复，靶子=首要终止机制 mach_limit(伪电流)。
- 首回合(E1,REPORT.md) honest-FAIL：n8 0.158ms 死于 volume_drift(1.002e-4,门1e-4)、n12 0.166ms 死于 mach_limit(mach=0.3055)；基线(w4/s0.04) 0.065ms；width/speed 杠杆 0.065→0.158/0.166(约2.5×)。
- 第二回合(E2,REPORT_V2.md) 执行 AUDIT 修正单：P1 锚点复现+R1 诊断骑缝，补丁 commit 829449c31(run.py +9/-1)。
- P1 复现成功：n=12 postcontact_healthy_ms=**1.362ms**（与 A3#1 逐位一致，∈[1.09,1.63]）→ A3#1 恢复有效基线。
- R1 裁决 H1~H4：H1 否证（带内伪电流非主因，带内 0.126/0.187<<全局 0.333/0.390）；H2 否证（step0 worst 在界面 φ≈0.5，Mach 自 0.07 渐增，非衰减瞬态）；H3 n8 支持（末 6 步 worst 贴周期 y 边界 y=31/1，Mach 0.22→0.39）；H4 支持（killer 在液滴中高处深气区，n=12 worst=[44,23,7]，x 偏 16.5 格≈滴面外 3 格，挤气膜）。
- numeric_pass 仍 False（1.362ms 死于 mach_limit，未覆盖 10.394ms 峰值）→ 战役目标仍 honest-FAIL，但基线恢复、机制定位完成。
## 关键数字（全部带原卡出处）
| 量 | 值 | 出处 |
| n=12 postcontact_healthy_ms | **1.362** | REPORT_V2.md §2 |
| n=12 stop_reason | mach_limit(step274=1.4809ms) | REPORT_V2.md §2 |
| n=12 contact/impact | 21 / 1.3728 m/s(误差1.94%) | REPORT_V2.md §2 |
| n=12 mach_max/phi@mm/band | 0.333 / 0.0044 / 0.126 | REPORT_V2.md §2 |
| n=8 postcontact_healthy_ms | **2.21325** | REPORT_V2.md §3 |
| n=8 mach_limit step | 279=2.2619ms | REPORT_V2.md §3 |
| n=8 contact/impact | 5 / 1.3833 m/s(误差1.19%) | REPORT_V2.md §3 |
| n=8 mach_max/phi@mm/band/worst | 0.3895 / 0.010 / 0.187 / [30,31,14] | REPORT_V2.md §3 |
| 首回合 n8/n12 存活 | 0.158 / 0.166 ms | REPORT.md §2 |
| 本回合 vs 首回合 | n8 2.213ms(总2.262,14.3×)、n12 1.362ms(8.2×) | REPORT_V2.md §5 |
| 处方预测区间 | [12,24] ms | PRESCRIPTION.md §3 |
| P1 验收区间 | [1.09,1.63]ms(=1.362±20%) | AUDIT.md |
| 峰值参考 | 10.394 ms | PRESCRIPTION.md(A1#3) |
| 参数组 | gap 0.4→0.13、+odd-rate/uniform-drop/stencil-wall、s0.03→0.04 | REPORT_V2.md §5 |
## 可复用方法/仪器
- 命令：`python -m benchmarks.water_entry.single_drop.run --cells 12 --speed 0.04 --width 6 --gap 0.13 --odd-rate 1 --uniform-drop --stencil-wall --out runs/t1_anchor_cac12.json`（+score）——集群 cj5668/cj5669/n315。
- R1 诊断骑缝：run.py 增 `mach_max_xyz=[z,y,x]` 与 `mach_band_max`（0.1<φ<0.9 带内最大 Mach，空带=None）+ `band.numel()` 守卫；坐标语义 `v[1:-1]` 只切 z 轴。
- 集群环境坑：须前置 `export PYTHONPATH=$PWD/src`；队列 state=succeeded≠rc=0，成败看 jobs/<jid>.log 的 rc。
## 诚实披露的失败与局限
- ⑦ AUDIT P1「不带 --speed=默认 0.04」不成立：首回合补丁 c5a031689 已把 CLI 默认改 0.03 并入 main；本单显式传 --speed 0.04。
- ⑧ 集群节点 python 环境不自洽：n330 默认 python 无 tensorlbm（cj5665/5666 rc=1）；n315 靠陈旧 .pth 指向旧作业目录 cj1594 才 import 成功。
- ⑨ AUDIT R1 原式 `float(mach_field[band].max())` 空带会崩；本单加 `band.numel()` 守卫。
- 两回合参数组不同（gap 0.4→0.13 等），为旗标组对比非单变量。
- 未覆盖 10.394ms 峰值；下回合建议：以 A3#1 为基线做单变量扫描（odd-rate/stencil-wall/gap），或加气相速度场快照区分 H3/H4。
