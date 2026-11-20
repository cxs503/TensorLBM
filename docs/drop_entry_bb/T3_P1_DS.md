# P1_DS 存活修复处方 — T3 议事厅队·DeepSeek 族
> 靶子：存活修复。锚点=DE-T3-P1。只读四张卡（A1/A2/A3/A4）+ 处方卡点名文件。

## 1. 假设（一句话因果链）
触水后 1~15ms 的 crater/jet 高速度相里，默认 Thuerey table-4.1 I/I 质量交换的 `ii_dead` 死链逻辑（free_surface_lbm.py#L1362-1368）在高速拉伸界面处把质量锁死/漏换，导致 `rho_min<=0` 或 `volume_drift>1e-4`，触发 `health_limit`（free_surface_probe.py#L126-128）——这就是多数 run 在 0.7~19.7ms 崩、而 `case3_wide8`(0.68%)/`case3_cal12`(1.78%) 能活到 24.005ms 的分水岭。

## 2. 处方（改什么·怎么改·为什么）
**改**：`benchmarks/water_entry/single_drop/free_surface_probe.py` 的 `probe()` 调用 `free_surface_step(...)` 时把 `simple_mass_exchange` 从默认 `False` 置 `True`（即 CLI `--simple-exchange`）。
**怎么改**：`free_surface_lbm.py#L1356-1361` 的 `simple_mass_exchange=True` 分支——"Basic Körner I/I flux"，同一 link 两端用相同平均 fill，`mass_delta_interface` 成对在总质量账本里**对消**，无 `ii_dead` 死链、无单侧漏换。
**为什么能活过 10.4ms**：crater/jet 相是界面胞最密集拉伸处，默认死链逻辑在此处制造单侧质量残差（drift 源）与负密度（rho_min 源）；对消式交换把这两个 health_limit 触发源同时掐掉，让界面胞在峰值时刻(10.394ms)仍保持 `rho_min>0` 且 `drift≤1e-4`。A4 溃坝经验同源：refilling EQ→AVG 改善 8.6 点，质量守恒是存活第一杠杆。

## 3. 预测（postcontact_healthy_ms 区间 + 可证伪判据）
- **预测区间**：`postcontact_healthy_ms ∈ [10.394, 24.005]`（覆盖峰值时刻，目标跑满 24ms 观测窗；`case3_wide8` 已证 0.68% 误差可活 24ms，本处方应至少持平）。
- **可证伪判据**：① 跑 `--simple-exchange` 后 `stop_reason=="window_complete"`（退出码 0）且 `geometric_volume_drift≤1e-4` 全时序；② 峰值时刻 10.394ms 处 `depth_over_D` 相对误差 ≤3%；③ 若仍 `health_limit` 且 `rho_min<=0` 或 `drift>1e-4` 在触水后 <10.394ms 触发 → 假设被否，改查 mach 源（`mach_max>0.3`）。

## 4. 成本（最小验证跑·可直跑命令行）
```
cd /root/tp-agent/work/m26270
python benchmarks/water_entry/single_drop/free_surface_probe.py \
  --cells 8 --speed 0.04 --gap 0.4 --simple-exchange \
  --out runs/simple8.json
# 退出码 0 = window_complete 且 volume_pass；1 = 崩
# 对照（默认交换，同配置）：
python benchmarks/water_entry/single_drop/free_surface_probe.py \
  --cells 8 --speed 0.04 --gap 0.4 --out runs/base8.json
```
（8 cells/D 是 DIAG 里跨度最大档，`case3_wide8` 同档已证 0.68% 可活 24ms；单跑 <10min，本机可直跑。）

## 5. 风险（最可能死在哪）
- **死点①**：`simple_mass_exchange` 对消式交换在**大变形/破碎**（jet 顶端甩出液滴）时可能把真实单侧质量转移也一并对消掉，导致 `depth_over_D` 峰值**低估**（误差 >3% 但存活）——需用 `--silhouette` 侧视轮廓交叉核验。
- **死点②**：若崩因实为 `mach_max>0.3`（CACSim 路径 `case3_cac12` 已证对 mach 更敏感），则本处方只治质量不治速度，`--simple-exchange` 仍会 `mach_limit`——需另开 `--nearest`/降 `--speed` 治速度源。
- **死点③**：`simple_mass_exchange` 可能改变界面胞膨胀计数（A4 坑⑥），峰值时刻界面胞数异常 → 需监控 `geometric_volume_drift` 与界面计数。
- **honest-FAIL**：若两档网格（8/12）误差不严格递减，则收敛门不过，处方只证存活不证精度，如实报。
