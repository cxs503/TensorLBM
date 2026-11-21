# DE 诊断测量线 W2 处方赛马卡设计（DISSMAP + CAVCLSE）

> 战役 DE · 单滴入深水池 Santini case-3 · 设计官（编排脑，不跑仿真）· 2026-11-21
> 机器可解析交付正本 = `BB/NEXT/TRACK_DIAG_W2.md`（20 行，严格机器格式，自检 5/5）；本件=工作树内成果副本+设计依据。

## 交付卡（与 BB/NEXT/TRACK_DIAG_W2.md 逐字一致）

```
# TRACK DIAG WAVE 2

自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）

goal_met: no
rationale: W1 两臂（CAV_PRESSURE/MOMENTUM_BUDGET）全 Infrastructure-FAIL（集群 torch segfault+fetch 失败）无数据，本波改走已验证 SDAA 车道（位级可复现）；K1/K2 已定谳欠穿透主导+收支不闭合（主导通道=mass_exchange_conversion 248.98%、重力排除），本波布两条新正交探针（空间耗散/转换损失图+气穴闭合点运动学），只测量不修复。

## ARM DISSMAP
task_key: DE-X-DIAG2-DISSMAP
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是诊断测量设计官。任务=布设能量耗散/转换损失空间图探针：逐胞算局部粘性耗散 ε=2νS:S（LBM 二阶矩）与 ABB 质量交换转换损失（K2 通道⑦ conv_loss=1.2627e+00=248.98% E_scale），累加到空间网格（中面 z 切片+按区径向直方图：界面胞/ABB 界面胞/体相液/气）。物理出处=局部粘性耗散+K2-BUDGET-B 通道⑦。最小改动=只加只读探针（不改内核），cells=8 满窗三元组（同 CAVSWEEP ctrl 锚旗标 --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15，SDAA 车道 sdaa_fs_probe.py）。验证=输出空间图 PNG+分区数字表（各区耗散/转换总量+份额）+总闭合残差。
prereg: 预测区间：若界面机制成立，转换损失应集中界面/ABB 胞（界面胞份额≥50%，与 T4-RED ABB 界面动量注入一致）。证伪判据：≥50% 损失在体相液（非界面）→ 界面局域化假设被否。

## ARM CAVCLSE
task_key: DE-X-DIAG2-CAVCLSE
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是诊断测量设计官。任务=布设气穴闭合点运动学探针：逐时步追踪闭合点（最深开放气-液交点/缩颈点）位置 (x,z) 与速度，输出闭合点轨迹与闭合时刻。物理出处=空穴/弹坑闭合动力学（K1-DIAG-B 定谳空穴开放 p_cav≈1.0 单连通=弹坑非 Rayleigh 闭气泡，相关物理=crater closure 非 bubble collapse）。最小改动=只加只读探针（不改内核），cells=8 满窗三元组（同 CAVSWEEP ctrl 锚旗标，SDAA 车道 sdaa_fs_probe.py）。验证=输出闭合点轨迹 (x,z vs t) 数字表+闭合时刻 t_close+闭合速度 U_close。
prereg: 预测区间：气穴闭合时刻 t_close∈[12,20]ms（接触后；d_cav 峰 10.54ms 后闭合，d_drop 19.5ms 回落 0.375D），闭合点向上移（自底向上闭合）。证伪判据：闭合点先上后下（空穴重开）或 t_close 越界 [12,20]ms。
```

## 设计依据（全部可溯定谳卡）

- W1（本线）CAV_PRESSURE/MOMENTUM_BUDGET 两臂全 Infrastructure-FAIL（集群 torch 2.13+loong64 segfault rc139、fetch 未回传）→ 无数据；W2 改走已验证 SDAA 车道（位级可复现，K1/K2/FORCE2/CONT1 同车道）。
- K1-DIAG-B 定谳：欠深=液滴物理欠穿透（d_drop_any=1.3953D<0.8·exp=1.4859D）；空穴过闭合被直接证伪（d_cav_gap≡0，p_cav=1.0 单连通，|p_cav|≤0.0209 格单位≈±1.26kPa）。
- K2-BUDGET-B 定谳：收支不闭合 honest FAIL（|R|/E_scale=1781.68%）；主导通道=7_mass_exchange_conversion（份额 248.98%，门 50%）；重力位能需求仅 0.548%（非缺口来源）；缺口在惯性/动量分配侧。
- T4-RED：界面 ABB 重构（f_abb=f_eq_gas+f_eq_gas[opp]−f_post[opp]，u_pc=mom/rho）注入非物理动量→mach 尖峰；ABBS：ABB 动量承重、非常数项来源。
- 排除路线（不重开）：纯网格收敛（CONV12 地板 8.6–14.3%）/参数微调（CAVSWEEP u_limit 平台）/u_limit/ABB 整体移除（ABBS 臂B 早死）/rho_g 简单缩放/空穴压亏（KERN-B 证伪）。

## 两臂正交性

- DISSMAP 答「能量/动量在哪丢」（空间局域化 248.98% 转换损失通道）；CAVCLSE 答「空穴怎么闭」（闭合点运动学）。
- 与 K1（d_drop/d_cav 分解）/K2（能量收支总量）不重复：本波补空间分布与闭合运动学两个维度。

## 预登记纪律

- 每臂=一个可复现只读探针+数字表；预测区间+证伪判据跑前写死；诚实 FAIL=数据。
- goal_met: no（战役峰误差 22.73% 未达 3% 门）。
