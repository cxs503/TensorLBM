# T1 处方卡 · 双相(CACSim)路径存活修复 — 首要终止机制 mach_limit(伪电流)

> 靶子：A3 诊断首要终止机制。双相路径唯一 run `case3_cac12` 于 1.362 ms 以 `mach_limit` 终止（A3#1），
> 远未达峰值 10.394 ms（A1#3）。自由面路径已有 5 个 run 存活 24 ms（A3#2），故本处方攻双相路径。

## 1. 假设
一句话因果链：CACSim 双相在触水后界面处产生**伪电流/寄生速度**（表面张力力离散 + 序参量梯度
陡峭所致，A4#3 已证伪电流 u_max 高于良调锚 5-25×），使 `mach=sqrt(3·Σv²)` 局部冲过
`maximum_lattice_mach=0.3`（A2#1c 门），在 ~1.36 ms 触发 `mach_limit` 终止——这是双相路径
零存活、无法覆盖 10.394 ms 峰值的首要终止机制。

## 2. 处方
改什么：`benchmarks/water_entry/single_drop/run.py` `run()` 的 `width`（界面宽度，L103 默认 4.0）
与 `speed`（晶格速度，L101 默认 0.04）两开关。
怎么改：① 将 `width` 从 4.0 提升至 **6.0**（界面胞数↑，序参量梯度变缓，表面张力力离散的寄生
电流按 ~1/width 衰减，mach 峰值回落）；② 同时将 `speed` 从 0.04 降至 **0.03**（`dt=speed·dx/U`，
L35，晶格速度↓→ 同一物理速度的晶格 Mach 数↓，直接压低 mach 上限）。
为什么能活过 10.4 ms：两杠杆正交叠加——width 平滑界面力源（伪电流主因），speed 压低晶格
Mach 基线（门限 0.3 的余量），使触水后 1~15 ms 的 `health_limit`/`mach_limit` 窗口（A3#3）
不再被伪电流击穿，从而跑满 `window_complete`（24 ms）覆盖峰值。

## 3. 预测
改后 `postcontact_healthy_ms` 预期区间：**[12, 24]** ms（覆盖 10.394 ms 峰值，A1#3）。
可证伪判据：① 若 `postcontact_healthy_ms < 10.394`，则处方失败（伪电流未压住）；② 若
`stop_reason` 仍为 `mach_limit` 且 `mach_max>0.3`，则 width/speed 杠杆无效；③ 若
`impact_speed_relative_error` 偏离 ≤3%（A1#3 门）且 `volume_drift≤1e-4`（A2#2b），则物理保真。

## 4. 成本
最小验证跑（可直跑命令行，双相入口 run.py）：
```
cd /root/tp-agent/work/m26266
python -m benchmarks.water_entry.single_drop.run --n 12 --speed 0.03 --width 6.0 --out runs/t1_cac_w6_s03.json
python -m benchmarks.water_entry.single_drop.run --n 8  --speed 0.03 --width 6.0 --out runs/t1_cac_w6_s03_n8.json
python -m benchmarks.water_entry.single_drop.score --output runs/t1_score.json runs/t1_cac_w6_s03.json runs/t1_cac_w6_s03_n8.json
```
（两档网格 12/8 满足 A1#3 收敛门；score 判 `numeric_pass`/`converged`。）

## 5. 风险
最可能死在哪：① `width=6.0` 使界面胞膨胀（A4#3 ⑥ 界面胞计数随细化恶化），`volume_drift`
可能超 1e-4 门；② `speed=0.03` 拉长 `dt`，`limit` 步数（L192）可能提前 `step_limit` 耗尽；
③ 伪电流若由碰撞模型（bgk）而非界面宽度主导，则 width 杠杆无效——需先跑 n8 单档探明
`mach_max` 是否随 width 单调回落，再决定是否上 n12 收敛档。
