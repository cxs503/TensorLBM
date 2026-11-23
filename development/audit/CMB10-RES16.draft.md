# DE-X-CMB10-RES16 — 分辨率封顶确认 cells=16 双臂（CMB 系列收官单）
锚点=NEW:DE-X-CMB10-RES16 · 战役=DE·多线并行攻关 · 分支 agent/m27256 · 仓库 TensorLBM · 2026-11-23
CROSS已读：CMB9-IFACETUNE（C12 ctrl 1.522445D / C12 tuned 1.739759D@12.7201ms）· CMB8-IFACECOMBO（C8 ctrl 1.435220D / C8 tuned 1.732906D）· RELX3-RRIFACE · CLUSTER-CANON
自检：6/6（文件在且非空✓ 数字可溯=runs/cmb10_r16_*/result.json✓ 双写commit✓ 预登记区间✓ 失败臂如实✓ 账号合规✓）
## 1. 案例介绍
液滴入水（Santini 2021, fluids7020050）：球形液滴 1.53 m/s 释放、0.18D 间隙入静水池，考察空穴生长-峰值-回弹。几何 span 4D×池深 3D，D3Q19 Körner 自由面 LBM（capacity+simple-exchange，u_limit 0.15）。实验锚=1.857434D@10.39446ms pc。
- 双臂 cells=16 同物理旗标（--gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --steps 10000 满窗）：R16-ctrl=BGK；R16-tuned=rr_iface+SMAG.075+HYDRO24.5（--rr-tau 0.5003877 --rr-iface --cs 0.075 + TL_FS_HYDRO_Z_REF=24.5）。
- 代码=m27225 端口三提交（8c24dc061 系，已并入 main da7f3cf6e，无需重 cherry-pick）；单测 14/14 + 60 步冒烟 rc=0。

## 2. 怎么做的
- 车道=集群正典（run_compute，stage_dir=工作树，8 核/臂，cmd 无 cd，PYTHONPATH=.:src；递交前 du -sh 488M→198M 瘦身，跑完还原）。
- 关键：cells=16 单臂 ~4017 步（dt 减半 0.006080ms/步，24ms 窗需 2× cells=12 步数），单臂 ~290-300min，超 run 型 7200s 上限 ⇒ 用 **shard 型 detached**（脱离会话，超 120min 不杀）；单作业单臂分开递交（账 6270 ctrl / 6271 tuned）。
- 预登记（写死后跑，BB/tmp/CMB10-RES16.prereg.md）：①ctrl 峰深 [1.50,1.60]D（≥1.60=未饱和 / ≤1.50=饱和）②tuned 峰深 [1.73,1.78]D ③ctrl 增量递减(≤0.06)=分辨率轴关闭 / 递增=报 cells=20+ ④双门 >1.8017D 且 <11.4334ms。

## 3. 计算结果（程序化提取 runs/cmb10_r16_*/result.json，零手抄）
| 臂 | 配置 | cells | stop | 峰深(D) | 深误差 | 峰时刻(ms pc) | 时刻误差 | 末端漂移 | mach | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| **R16-ctrl** | BGK | 16 | window_complete@4017 | **1.633265** | -12.07% | 12.6228 | +21.43% | 9.66e-05 | 0.2549 | 锚 |
| **R16-tuned** | rr_iface+SMAG.075+HYDRO24.5 | 16 | **health_limit@3341** | **1.747348** | -5.93% | 11.6196 | +11.80% | 1.002e-04 | 0.1864 | **早死(峰有效)** |

## 4. C8/C12/C16 三点分辨率趋势（收官核心）
| 配置 | C8 | C12 | C16 | C8→C12 | C12→C16 | 趋势 |
|---|---|---|---|---|---|---|
| ctrl 峰深(D) | 1.435220 | 1.522445 | 1.633265 | +0.087225 | **+0.110820** | **递增** |
| tuned 峰深(D) | 1.732906 | 1.739759 | 1.747348 | +0.006853 | +0.007589 | 微增 |
| ctrl 峰时刻(ms pc) | 10.5433 | 10.0204 | 12.6228 | -0.5229 | **+2.6024** | 反转推晚 |
**外推判据判定**：ctrl 增量 0.0872→0.1108 **递增**（非递减）⇒ 预登记判据触发：**分辨率轴未饱和，须报 cells=20+ 建议**（CMB 系列未封顶）。

## 5. 判决（对照预登记双门 DUAL_PASS=峰深>1.8017D 且 峰时刻<11.4334ms）
- **无 DUAL_PASS 臂**（与 CMB 系列一致）。ctrl 双门均未过（深 1.633<1.8017、时 12.62>11.43）；tuned 双门均未过（深 1.747<1.8017、时 11.62>11.43）且早死。
- **预登记逐条**：①ctrl 1.6333≥1.60 ⇒ 判据触发：分辨率贡献超预期=趋势未饱和 ✓（超上界 +0.0333D）。②tuned 1.7473∈[1.73,1.78] ✓。③ctrl 增量递增 ⇒ 报 cells=20+ 建议（见 §7）。④双门无臂过。

## 6. 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| R16-ctrl 峰深 | 1.633265 D | 实验 1.857434 D | -12.07% | **否** |
| R16-ctrl 峰时刻 | 12.6228 ms pc | 实验 10.39446 ms pc | +21.43% | **否** |
| R16-tuned 峰深 | 1.747348 D | 实验 1.857434 D | -5.93% | **否** |
| R16-tuned 峰时刻 | 11.6196 ms pc | 实验 10.39446 ms pc | +11.80% | **否** |
结论：**战役 3% 双门（深度<3% 且时刻<3%）无臂达标**（honest-FAIL）。

## 7. 诚实披露
1. **分辨率轴未饱和（本单核心发现）**：ctrl 增量 C8→C12 +0.0872 → C12→C16 **+0.1108**（递增）⇒ 分辨率对 ctrl 缺口贡献仍在放大，**CMB 系列分辨率轴未封顶，须开 cells=20+ 单**；tuned 配置增量平坦（+0.0069→+0.0076），调优态对分辨率不敏感。
2. **tuned 臂早死（honest-FAIL）**：cells=16 调优态 volume_drift 1.002e-04 越过 1e-4 门于 20.31ms（step 3341），未达 window_complete（~4017 步/24.4ms）；峰（step 1979，11.62ms pc）在死前已现，峰值有效但非满窗。
3. **ctrl 时刻税反转**：C12→C16 峰时刻 +2.60ms（10.02→12.62，推晚），细分辨率把 ctrl 峰推晚，抹掉 C12 的时刻优势（C12 曾 -3.60% 贴门）。
4. **兑换率**：ctrl C12→C16 ΔD=+0.1108 Δt=+2.60ms（+23.5ms/D 时刻税）；tuned C12→C16 ΔD=+0.0076 Δt=-1.10ms（-145ms/D 时刻负税，但早死）。
5. **局限**：cells=16 单分辨率双臂；tuned 早死非满窗；无 --planes 场量（云图缺，深度-时刻曲线为主诊断）；结论不外推其它几何。无 Infrastructure-FAIL（两臂 rc 以集群日志为准）。

## 8. 数据溯源
满窗/早死结果=runs/cmb10_r16_ctrl/result.json、runs/cmb10_r16_tuned/result.json（三元组+history）；图=runs/cmb10_r16_ctrl/fig1_depth_time.png、fig2_resolution_trend.png（深度-时刻 vs 实验 + 分辨率趋势）；实验基准=benchmarks/water_entry/single_drop/fig5_experiment.csv（1.857434D@10.394ms）；集群账=6270(ctrl,n315,window_complete)/6271(tuned,n315,health_limit)。
