# DE-X-CMB8-IFACECOMBO — rr_iface 底 × SMAG × HYDRO 合成冲双 3% 门
锚点=NEW:DE-X-CMB8-IFACECOMBO · 战役=DE·多线并行攻关 · 分支 agent/m27206 · 仓库 TensorLBM · 2026-11-22
CROSS已读：RELX3-RRIFACE（底 1.621574D@10.227161ms pc，兑换率 -1.70ms/D 无时刻税）· CMB6-CSTUNE（Cs=0.075 局部极大+z_ref 单调向下）· CMB5-HYDRORR-STAB（RR 剂量轴证伪，τ 固定 0.5003877）· CMB4-TIMECPL（时刻机制=穿透动量）· CLUSTER-CANON（车道正典）
自检：6/6（文件在且非空✓ 数字可溯=集群正典车道 runs/cmb8_*/result.json✓ 双写commit✓ 预登记区间✓ 失败臂如实✓ 账号格式合规✓）

## 1. 案例介绍（物理背景/几何/参数）
液滴入水（Santini 2021, fluids7020050）：球形液滴 1.53 m/s 释放、0.18D 间隙入静水池，考察空穴生长-峰值-回弹。几何 span 4D×池深 3D，cells=8（8 格/D），D3Q19 Körner 自由面 LBM（capacity+simple-exchange，u_limit 0.15）。
- 底=**rr_iface**（界面局部 RR：RR 二阶应力投影仅 INTERFACE&(0<fill<1) 胞，体相 BGK；Ladd 1994；端口 ba9d05c74 勿重移植，本单 cherry-pick+9b936afa0 探针修复）。
- 叠加=**SMAG**（C_s∈{0.075,0.1}，CMB3 RR×SGS 惯用法：_sgs_tau_eff_3d 逐胞 tau_eff 交 RR 核与体相 BGK 两子分支）+ **HYDRO**（TL_FS_HYDRO_Z_REF∈{22,24.5} 静水压补偿）。实验锚=1.857434D@10.39446ms pc。

## 2. 怎么做的（方法/流程/关键设置/工具链）
- 改动：cherry-pick ba9d05c74+9b936afa0（rr_iface 分支+--rr-tau/--rr-iface/--cav-series+4 单测+160 步冒烟），冲突手工解（保留 main 既有 rr/C_s/cavgas 通道）；新增 _sgs_tau_eff_3d（CMB3 惯用法）+ rr_iface 分支 C_s>0 两子分支共用 tau_eff（C_s=0 位级不变）；probe 增 --cs CLI（None→TL_FS_CS env）。HYDRO 走既有 TL_FS_HYDRO_Z_REF env（未设=位级不变）。
- 单测：test_free_surface_rr_collision.py 9 项 + test_free_surface_rr_iface_combo.py 5 项（CS 通道/轨迹分歧/体相局部性/守恒/零行为）全过；160 步冒烟 max|f_rr-iface−f_bgk|=3.34e-01>1e-6 通过。
- 车道=集群正典（run_compute，stage_dir=工作树，8 核/臂，cmd 无 cd 相对路径，PYTHONPATH=.:src；递交前 du -sh 489M>200M 先瘦身至 49M，跑完还原）。
- 4 交付臂（cells=8 同旗标 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --steps 10000 满窗）+2 锚复核+2 诊断臂（300 步）+1 场量臂（--planes）。

## 3. 计算结果（程序化提取 runs/cmb8_*/result.json，零手抄）
| 臂 | 配置 | stop | 峰深(D) | 深误差 | 峰时刻(ms pc) | 时刻误差 | 末端漂移 | 峰值漂移 | mach_max | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | rr_iface+SMAG.075+HYDRO22 | window_complete | 1.620380 | -12.76% | 12.7687 | +22.84% | 3.07e-05 | 8.32e-05 | 0.1375 | FAIL |
| **B** | rr_iface+SMAG.075+HYDRO24.5 | window_complete | 1.732906 | -6.70% | 11.9053 | +14.54% | 4.51e-05 | 8.35e-05 | 0.1375 | **NEAR_MISS** |
| C | rr_iface+SMAG.1+HYDRO22 | window_complete | 1.618262 | -12.88% | 12.7687 | +22.84% | 3.45e-05 | 8.31e-05 | 0.1555 | FAIL |
| base | rr_iface 底 | window_complete | 1.621536 | -12.70% | 10.2150 | -1.73% | 7.30e-06 | 8.41e-05 | 0.2598 | 锚 |
| ctrl | BGK | window_complete | 1.435220 | -22.73% | 10.5433 | +1.43% | 4.81e-06 | 8.55e-05 | 0.2598 | 锚 |
**锚复核（车道保真）**：ctrl 1.435220D@10.5433ms pc vs 战役锚 1.435220D@10.543339ms pc ⇒ 位级吻合；base 1.621536D@10.2150ms pc vs RELX3 底 1.621574D@10.227161ms pc ⇒ Δ3.8e-05D/Δ0.012ms，移植忠实。
**跨臂兑换率（vs rr_iface 底）**：A ΔD=-0.0012 Δt=+2.5537ms；B ΔD=+0.1114 Δt=+1.6903ms（率 +15.18ms/D）；C ΔD=-0.0033 Δt=+2.5537ms。

## 4. 结果图
- 图1 深度-时刻曲线 vs 实验（比较曲线）：runs/cmb8/cmb8_depth_time.png；图2 中面 fill 场量云图（臂 A，t=10.397ms pc≈实验峰时刻）：runs/cmb8/cmb8_contour.png；图3 中面气穴深度/面积时序（几何归因）：runs/cmb8/cmb8_cavity.png

## 5. 判决（对照预登记双门）
- 预登记：DUAL_PASS=峰深>1.8017D 且峰时刻<11.4334ms；DEPTH_ONLY=峰深>1.8017D 但时刻超；TIME_HOLD=时刻<11.4334ms 且峰深∈[1.70,1.8017)D；FAIL=峰深<1.70D 或早死。**无 DUAL_PASS 臂**。A/C 峰深<1.70D ⇒ FAIL；B 峰深 1.7329D∈[1.70,1.8017) 但峰时刻 11.9053ms≥11.4334ms ⇒ 落预登记四分支之外「双边际」缺口（NEAR_MISS），双门均未过。
- 诊断臂（300 步）：SMAG 单独/HYDRO 单独在 rr_iface 底均存活至 3.648ms（drift 6.4e-05/4.6e-05）⇒ 早死非单组分所致；4 交付臂集群车道全部满窗（无早死）。

## 6. 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 最优臂 B 峰深 | 1.732906 D | 实验 1.857434 D | -6.70% | **否** |
| 最优臂 B 峰时刻 | 11.9053 ms pc | 实验 10.39446 ms pc | +14.54% | **否** |
| 臂 A 峰深 | 1.620380 D | 实验 1.857434 D | -12.76% | 否 |
| 臂 C 峰深 | 1.618262 D | 实验 1.857434 D | -12.88% | 否 |
| ctrl 锚复核峰深 | 1.435220 D | 战役锚 1.435220 D | +0.0000% | 是 |
| base 锚复核峰深 | 1.621536 D | RELX3 底 1.621574 D | -0.0023% | 是 |
结论：**战役 3% 双门（深度<3% 且时刻<3%）无臂达标**；最优臂 B 深度 -6.70%/时刻 +14.54%，双门均未过。

## 7. 诚实披露
1. **合成假设被证伪**：rr_iface 底已拿全场 RR 155% 深度收益（1.6216D），SMAG+HYDRO 叠加仅再收 +0.1114D（最优臂 B），远小于全场 RR 底同叠加 ~0.2D ⇒ 叠加增益不可加，rr_iface 底与 SMAG+HYDRO 深化机制不协同。
2. **时刻税回归**：rr_iface 底峰时刻 10.215ms（早于门），叠加后推晚至 11.9–12.8ms（+1.7~+2.6ms）⇒ SGS 耗散把峰推晚，抹掉底时刻优势。
3. **z_ref 方向反转**：rr_iface 底 z_ref=24.5（B）比 22（A/C）深 0.11D，与全场 RR 底（CMB6：22 更深）相反 ⇒ 响应面随底变，不可跨底外推。
4. **本地早死 vs 集群满窗**：本地 200 步接线跑（同代码）死于 1.45ms（drift 1.02e-4 过门），集群同配置满窗（drift_max 8.3e-05<1e-4）⇒ 组合贴漂移门，环境浮点差可推过/推回门 ⇒ 边际稳定，如实登记。
5. **局限**：cells=8 单分辨率单车道；Cs 步长 0.025 较粗；结论不外推其它分辨率/几何。无失败项；无 Infrastructure-FAIL（8 作业 rc 以日志为准，门禁 DB rc 列恒 0 系 cluster_exec 未传播退出码，CMB4 先例）。

## 8. 数据溯源
| 数据源 | 路径 | 内容 |
|---|---|---|
| 满窗结果 | runs/cmb8_a_cs075_z22、cmb8_b_cs075_z245、cmb8_c_cs01_z22、cmb8_ctrl、cmb8_base/result.json | 三元组+history |
| 场量 | runs/cmb8_a_cs075_z22_p/planes.npz | 中面 fill/uz @4 注册时刻 |
| 气穴几何 | runs/cmb8_*/cav_series.json | 中面气穴面积/宽/深时序 |
| 实验基准 | benchmarks/water_entry/single_drop/fig5_experiment.csv | 1.857434D@10.394ms |
| 集群账 | cj6232–cj6239（n315/n330，rc 以 /root/tp-agent/cluster/jobs/*.log 为准） | 8 作业 |
