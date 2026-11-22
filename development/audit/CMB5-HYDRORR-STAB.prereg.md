# DE-X-CMB5-HYDRORR-STAB · 预登记（跑前写死，禁改）
锚点=NEW:DE-X-CMB5-HYDRORR-STAB · 战役=DE·组合比较矩阵波 W3（hydro_rr 稳定化） · 分支 agent/m27100 · 仓库 TensorLBM
CROSS已读：CMB3-REGSMAG（hydro_rr 早死原点 1.777132D@15.322ms pc，health_limit）、COL2D-REGDOSE（剂量换算 rr_tau=dose×0.5003877）、COL2-REG（RR 处方原文）

## 死因定位（第一步，程序化取自 m27044 树 runs/cmb3/hydro_rr/result.json history）
- 死因=volume_drift 过 1e-4 门（非 mach：mach_max 全程 ≤0.1233，死亡步 0.1060 ≪ 0.3 门）
- 临界步=1305（t=15.869732 ms abs，15.322 ms pc），volume_drift=1.124033e-4 首次 >1e-4
- 末 200 步漂移斜率=1.0654e-7/step（≈1.296e-9/ms）；漂移为振荡型（非单调爬升），末段 1250→1305 步由 3.36e-6 跳到 1.124e-4
- mach 全程 ≤0.1233（死亡步 0.1060）⇒ 死因=volume_drift 门，非 mach 门

## 处方（协调者物理决策，禁改方向）
RR 剂量插值稳定化：HYDRO 底（z_ref 24.5）上 RR 剂量 {0.75, 0.50} 两臂，满窗 --steps 10000。
剂量换算照 COL2D-REGDOSE 卡：rr_tau = dose × 0.5003877（1× = COL2 rr_base 0.5003877）。
| 臂 | 剂量 | rr_tau | 其余旗标（与 CMB3 车道一致） |
|---|---|---|---|
| cmb5_d075 | 0.75× | 0.375290775 | cells=8 gap .18 speed .06 rs 1.53 pd 3.0 ul .15 C_s=0 hydro 24.5 |
| cmb5_d050 | 0.50× | 0.25019385 | 同上 |
应急臂（仅当两臂均早死）：u_limit 收紧 0.13 第三臂（剂量取存活更久者）。

## 预登记判定（跑前写死，禁改）
- 满窗存活（window_complete）且峰深 > 1.750541D ⇒ 新冠军 HIT
- 满窗存活但峰深 ≤ 1.750541D ⇒ 「稳定化成功但无增益」FAIL
- 全部早死 ⇒ hydro_rr 前沿不可稳定，该线关闭（诚实记录死因链）
- 对照三元组：冠军 combo_hydro 1.750541D@14.313161ms pc（账 6073）｜早死原点 hydro_rr 1.777132D@15.322ms pc（账 6082）｜实验锚 1.857434D@10.394ms pc
- 验收双门（战役口径）：深度误差 <3% 且时刻误差 <3%（vs 实验 1.857434D@10.394ms pc）

## 环境
- 车道：SDAA 共享包装器 /root/tp-agent/cluster/sdaa_fs_probe.py（32 卡自动挑闲）；cells=8 满窗
- 接线：agent/m27100 = main + cherry-pick edbd0a013（RR 碰撞+rr_tau 通道）+ 100169bce（HYDRO z_ref+C_s 通道），冲突解=保留 main 既有参数、仅增 rr_tau/C_s
- 位级锚核验：parity 臂（rr_tau=0.5003877+hydro 24.5，200 步）须与 m27044 runs/cmb3/hydro_rr history 前 200 条逐位一致
