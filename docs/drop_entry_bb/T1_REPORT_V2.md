# T1 REPORT_V2 · DE-T1-E2 第二回合 — P1 锚点复现 + R1 诊断骑缝（执行 AUDIT 修正单）
自检：5/5（文件在且非空✓ 数字可溯✓ 双写commit 9aa652ed7✓ 预测区间[1.09,1.63]命中✓ 失败如实✓）
上游验收：PASS——AUDIT 修正单（P1+R1+禁令）全部执行；A3#1 锚点 1.362 ms 复现成功，无需退回单。


## 1. 补丁（commit 829449c31，run.py +9/-1）
- R1：history 行增 `mach_max_xyz=[z,y,x]` 与 `mach_band_max`（0.1<φ<0.9 带内最大 Mach，空带=None）。
- 非有限检查扩展兼容 list/None（原 `math.isfinite(v)` 对新字段会 TypeError）。
- 坐标语义：`v[1:-1]` 只切 z 轴（y/x 周期）→ xyz 为「z 内部、y/x 全域」坐标。

## 2. P1 锚点复现（n=12 全旗标，集群 cj5668/n315）
命令：`python -m benchmarks.water_entry.single_drop.run --cells 12 --speed 0.04 --width 6 --gap 0.13 --odd-rate 1 --uniform-drop --stencil-wall --out runs/t1_anchor_cac12.json`（+score）
| 量 | 值 | json 路径 |
|---|---|---|
| postcontact_healthy_ms | **1.362** | runs/t1_score_e2_n12.json → runs.*.postcontact_healthy_ms |
| stop_reason | mach_limit（step 274 = 1.4809 ms） | runs/t1_anchor_cac12.json → stop_reason/history[274] |
| contact_step / impact_speed | 21 / 1.3728 m/s（误差 1.94%） | 同上 → contact_step/impact_speed_m_s |
| mach_max / phi@mm / band_max | 0.333 / 0.0044 / 0.126 | history[274] |
- **1.362 ms 与 A3#1（1.362）逐位一致**，∈[1.09,1.63] → **A3#1 恢复有效基线**（验收 PASS）。

## 3. n=8 档（同旗标，集群 cj5669/n315）
postcontact_healthy_ms=**2.21325**（score runs/t1_score_e2_n8.json）；mach_limit step 279=2.2619 ms；contact=5；impact 1.3833 m/s（误差 1.19%）；mach_max=0.3895、phi@mm=0.010、band=0.187、worst=[30,31,14]（y=31=周期边界行）。

## 4. R1 裁决（H1~H4）
- H1 带内伪电流（曲率滤波靶）：**否证为主因**——死亡时带内 0.126/0.187 << 全局 0.333/0.390，杀手在带外深气（φ≈0.004~0.010）。
- H2 初始化声波：**否证**——step0 worst 在界面（φ≈0.5），Mach 自 0.07 渐增，非衰减瞬态。
- H3 边界伪影：**n=8 支持**——末 6 步 worst 贴周期 y 边界（y=31/1），Mach 0.22→0.39。
- H4 挤气膜：**支持**——killer 在液滴中高处深气区（n=12 worst=[44,23,7]，x 偏 16.5 格≈滴面外 3 格），液滴下插挤气区。
- 结论：AUDIT 禁令（暂停曲率滤波/气相力截断编码）正确；下一杠杆应攻气相动力学/边界，非界面宽度。

## 5. 对比首回合
- 首回合 n=8（w6/s0.03、无旗标）：总存活 0.158 ms、未触水（volume_drift）→ 本回合 n=8：触水后健康 2.213 ms（总 2.262 ms，14.3×）。
- 首回合 n=12：0.166 ms（mach_limit）→ 本回合 n=12：1.362 ms（8.2×）。
- 注意：两回合参数组不同（gap 0.4→0.13、+odd-rate/uniform-drop/stencil-wall、s0.03→0.04），此为旗标组对比非单变量。

## 6. 新异议
⑦ AUDIT P1「不带 --speed=默认 0.04」在当前树不成立：首回合补丁 c5a031689 已把 CLI 默认改为 0.03 并入 main；照抄会跑 0.03。本单显式传 --speed 0.04（对齐 A3#1 意图）。
⑧ 集群节点 python 环境不自洽：n330 默认 python 无 tensorlbm（cj5665/5666 rc=1 ModuleNotFoundError）；n315 靠陈旧 .pth 指向旧作业目录 cj1594 才 import 成功。命令须前置 `export PYTHONPATH=$PWD/src`。另：队列 state=succeeded ≠ rc=0（run.py 非 window_complete 退出码 1 属预期），成败须看 jobs/<jid>.log 的 rc。
⑨ AUDIT R1 原式 `float(mach_field[band].max())` 空带会崩；本单加 `band.numel()` 守卫（None）。

## 7. 结论
- P1：PASS（1.362 ms 精确复现 A3#1）；R1：数据已裁决 H1~H4（§4）。
- numeric_pass 仍 False（1.362 ms 死于 mach_limit，远未覆盖 10.394 ms 峰值）→ 战役目标仍 honest-FAIL，但基线恢复、机制定位完成。
- 下回合建议：以 A3#1 为基线做单变量扫描（odd-rate/stencil-wall/gap），或加气相速度场快照以区分 H3/H4。
