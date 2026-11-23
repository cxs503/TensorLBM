# DE-X-CMB9-IFACETUNE — rr_iface 底定向调优四臂（z 方向/时刻税隔离/分辨率/Cs 细扫）
锚点=NEW:DE-X-CMB9-IFACETUNE · 战役=DE·多线并行攻关 · 分支 agent/m27225 · 仓库 TensorLBM · 2026-11-23
CROSS已读：CMB8-IFACECOMBO（前驱：合成证伪，B=1.732906D@11.9053ms 双边际；三机制=增益不可加/时刻税回归/z_ref 随底反转）· RELX3-RRIFACE（底 1.621574D@10.227161ms pc）· CMB6-CSTUNE（Cs-zref 平面，全场底）· CLUSTER-CANON（车道正典）
自检：6/6（文件在且非空✓ 数字可溯=集群正典车道 runs/cmb9_*/result.json✓ 双写commit✓ 预登记区间✓ 失败臂如实✓ 账号格式合规✓）

## 1. 案例介绍（物理背景/几何/参数）
液滴入水（Santini 2021, fluids7020050）：球形液滴 1.53 m/s 释放、0.18D 间隙入静水池，考察空穴生长-峰值-回弹。几何 span 4D×池深 3D，D3Q19 Körner 自由面 LBM（capacity+simple-exchange，u_limit 0.15）。
- 底=**rr_iface**（cherry-pick 870ff09eb 系三提交 a757cf9b0+4a14ae3c6+88e101b9a；τ=0.5003877 显式 --rr-tau，与 CMB8 底同调用）；叠加=SMAG（C_s）+HYDRO（TL_FS_HYDRO_Z_REF）。实验锚=1.857434D@10.39446ms pc。
- 四轴：A z27 方向探（CMB8 反转 24.5>22 的延长线）；B 时刻税隔离（无 SGS）；C 分辨率（cells=12+BGK 锚）；D Cs 细扫（0.05）。

## 2. 怎么做的（方法/流程/关键设置/工具链）
- 改动：cherry-pick 三提交（rr_iface+SMAG+HYDRO 通道）；free_surface_probe.py 冲突手工解=保留 main 既有 cav_probe 只读通道（m27202）+端口 rr_iface/cav_series 通道（正交，默认关位级不变）。
- Gate0：单测 14/14（9 rr_collision+5 rr_iface_combo）+160 步冒烟 max|f_rr-iface−f_bgk|=3.42e-01>1e-6 通过。
- 车道=集群正典（run_compute，stage_dir=工作树，8 核/臂，cmd 无 cd，PYTHONPATH=.:src；递交前 du -sh 488M>200M 瘦身至 166M，跑完还原）。
- 8 作业：账 6245(A)/6246(B)/6247(C)/6248(C-ctrl)/6249(D)/6250(base8)/6256(C 场量 --planes，120min 超时 FAIL)/6265(D 场量 --planes)；cells=8 同旗标 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --steps 10000 满窗；cells=12 同物理参数。

## 3. 计算结果（程序化提取 runs/cmb9_*/result.json，零手抄）
| 臂 | 配置 | cells | stop | 峰深(D) | 深误差 | 峰时刻(ms pc) | 时刻误差 | 末端漂移 | mach | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | rr_iface+SMAG.075+HYDRO27 | 8 | window_complete | 1.724181 | -7.17% | 13.0971 | +26.00% | 4.66e-05 | 0.1375 | NEAR_MISS |
| B | rr_iface+HYDRO24.5 Cs=0 | 8 | window_complete | 1.621666 | -12.69% | 12.7566 | +22.72% | 1.35e-05 | 0.1505 | FAIL |
| **C** | rr_iface+SMAG.075+HYDRO24.5 | 12 | window_complete | **1.739759** | **-6.34%** | 12.7201 | +22.37% | 8.15e-05 | 0.1646 | **NEAR_MISS** |
| C-ctrl | BGK | 12 | window_complete | 1.522445 | -18.04% | 10.0204 | -3.60% | 8.59e-06 | 0.2598 | 锚 |
| D | rr_iface+SMAG.05+HYDRO24.5 | 8 | window_complete | 1.726805 | -7.03% | 13.1579 | +26.59% | 1.77e-05 | 0.1493 | NEAR_MISS |
| base8 | rr_iface 底 | 8 | window_complete | 1.621536 | -12.70% | 10.2150 | -1.73% | 7.30e-06 | 0.2598 | 锚 |
**锚复核（车道保真）**：base8 1.621536D@10.2150ms vs CMB8 底 1.621536D@10.2150ms ⇒ 位级吻合（Δ0.000000D/0.0000ms）。
**跨臂兑换率（vs rr_iface 底 1.6216D@10.227ms）**：A ΔD=+0.1026 Δt=+2.870ms（+27.98ms/D）；B ΔD=+0.0001 Δt=+2.530ms（率≈∞，纯时刻税）；C ΔD=+0.1182 Δt=+2.493ms（+21.10ms/D）；D ΔD=+0.1052 Δt=+2.931ms（+27.86ms/D）。

## 4. 结果图
- 图1 深度-时刻曲线 vs 实验（比较曲线）：runs/cmb9/cmb9_depth_time.png；图2 中面 fill 场量云图（臂 D，t≈实验峰时刻 10.397ms）：runs/cmb9/cmb9_contour.png；图3 中面气穴深度/面积时序（几何归因）：runs/cmb9/cmb9_cavity.png

## 5. 判决（对照预登记双门，DUAL_PASS=峰深>1.8017D 且 峰时刻<11.4334ms）
- **无 DUAL_PASS 臂**。最优臂 C 峰深 1.7398D∈[1.70,1.8017) 但峰时刻 12.7201ms≥11.4334ms ⇒ 双边际（NEAR_MISS）；A/D 同落双边际；B 峰深 1.6217D<1.70D ⇒ FAIL。
- **预登记判据逐臂**：A 预测 A≥B(1.7329) ⇒ 实测 1.7242<B ⇒ 判据触发：z 轴非单调（22→1.6204<24.5→1.7329>27→1.7242），24.5 局部极大，z 轴关闭。B 预测时刻<11.4334 ⇒ 实测 12.7566≥门 ⇒ 判据触发：时刻税非 SMAG 独有。C 预测 C≥1.7329 ⇒ 实测 1.7398 ✓（+0.0069D）；C-ctrl 预测 [1.45,1.55] ⇒ 实测 1.5224 ✓ 且 ≥1.50 ⇒ 判据触发：分辨率对 22.73% 缺口贡献 +0.0872D（-22.73%→-18.04%），未测轴实锤。D 预测 D≤1.7329 ⇒ 实测 1.7268 ✓（0.075 仍略深 0.0061D，Cs 轴 0.075 附近平坦）。

## 6. 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 最优臂 C 峰深 | 1.739759 D | 实验 1.857434 D | -6.34% | **否** |
| 最优臂 C 峰时刻 | 12.7201 ms pc | 实验 10.39446 ms pc | +22.37% | **否** |
| C-ctrl 峰时刻 | 10.0204 ms pc | 实验 10.39446 ms pc | -3.60% | 否（-3.60%） |
| base8 锚复核峰深 | 1.621536 D | CMB8 底 1.621536 D | +0.0000% | 是 |
结论：**战役 3% 双门（深度<3% 且时刻<3%）无臂达标**（honest-FAIL）。

## 7. 诚实披露
1. **时刻税归因翻转（本单核心发现）**：B 臂（无 SGS）峰时刻 12.7566ms = 底 10.2150 + **2.5416ms 纯 HYDRO 税**（ΔD=+0.0001D 零深度）；SMAG.075 叠加其上反把峰提前 -0.8513ms（12.7566→11.9053，CMB8-B）⇒ CMB8 的 +1.69ms 时刻税=HYDRO +2.54ms − SMAG 0.85ms，**时刻税是 HYDRO 效应，SMAG 是时刻负税**。
2. **z 轴非单调**：24.5 为局部极大（27 回落 0.0087D），CMB8 的"反转"是 22→24.5 段，非单调延长；z 轴在 {22,24.5,27} 内关闭。
3. **分辨率轴正贡献但小**：调优配置 cells=8→12 仅 +0.0069D（1.7329→1.7398）；ctrl 底 +0.0872D（1.4352→1.5224）且峰时刻提前 -0.52ms（10.5433→10.0204）⇒ 分辨率对 ctrl 缺口贡献 ~0.9% 实验深度，对调优配置边际。
4. **Cs 轴平坦**：0.05 vs 0.075 Δ=0.0061D（0.075 略优），步长 0.025 内无新极大；Cs 轴在 0.05-0.075 段关闭。
5. **局限**：cells=8 单分辨率（C 臂除外）；z_ref 三点；结论不外推其它几何。无早死臂（6 交付作业全满窗）；账 6256（C 场量 --planes）120min 超时 FAIL（节点负载高，同配置 cj6247 96.4min 成功），场量云图改由 D 臂 cells=8 提供（runs/cmb9_d_p/planes.npz）；无其它 Infrastructure-FAIL（rc 以集群日志为准，门禁 DB rc 列恒 0 系 cluster_exec 未传播退出码，CMB4 先例）。

## 8. 数据溯源
| 数据源 | 路径 | 内容 |
|---|---|---|
| 满窗结果 | runs/cmb9_{a_z27,b_cs0_z245,c12,c12_ctrl,d_cs05_z245,base8}/result.json | 三元组+history |
| 气穴几何 | runs/cmb9_*/cav_series.json | 中面气穴面积/宽/深时序 |
| 场量 | runs/cmb9_d_p/planes.npz | 中面 fill/uz @4 注册时刻（D 臂） |
| 实验基准 | benchmarks/water_entry/single_drop/fig5_experiment.csv | 1.857434D@10.394ms |
| 集群账 | 6245-6250+6256(超时)+6265（n315/n330） | 8 作业 |
