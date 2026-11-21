# TRACK GAS WAVE 1
goal_met: no
rationale: 定压闭包是稳定性要求（ABB 对 rho_g 亚 1% 敏感，KERN-B 定谳），压亏/任何 rho_g 时变路线已死；不碰 rho_g 的加深自由度只剩气相粘性（过度阻尼）与气相库存（界面质量源），两臂赛马。
自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）
## ARM GASVISC
task_key: DE-X-GAS1-GASVISC
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是气相粘性修正官。任务=把气相运动粘性修正到物理空气/水比值，消除空穴气的人工阻尼。
物理出处：现求解器全场单 τ（solver3d.py L1036/1040，无分相 τ），气相被液体粘度过度阻尼约 50×（ν_air/ν_water=1.8e-5/1e-3≈0.018，标准物性表）；空穴内气流被阻尼→腔闭合提前→峰深系统性偏低 13–23%（T5/T4/CONV12 同源残差）。
最小改动：碰撞步按气相掩膜对 GAS 胞用 τ_g=0.5+3·(0.018·ν_l)（默认关=零行为改变）；probe 加 --gas-nu-ratio 旗标（默认 1.0=现状）。
验证：cells=8 满窗三元组（CONV12 修复臂旗标：gap .18 speed .06 capacity simple-exchange release-speed 1.53 span 4 pool-depth 3），ctrl+臂，集群 8 核，绝对时轴。
prereg: 预测 peak_depth∈[1.4352,1.65]D（若气相过阻尼为主因，预期收回 0.42D 缺口的 30–80%）；证伪判据：peak_depth<1.4352D 或 postcontact_healthy_ms<10.394ms（早死）→证伪，honest-FAIL 入账。
## ARM MASSSRC
task_key: DE-X-GAS1-MASSSRC
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是界面质量源（蒸发/卷吸）官。任务=界面加小液→气质量源，经定压闭包的气相库存加深空穴。
物理出处：定压下 V_cav=m_g/ρ_g，气相库存直接定腔体积（KERN-B §6）；高速撞击界面卷吸空气+蒸发（Santini 1998 高速摄影：空穴含卷吸空气）；气相库存是唯一不碰 ρ_g 的加深自由度。
定量论证：峰缺 1.857−1.4352=0.4218D≈现腔体积 29%→源强标定 S≈0.03·m_g0/ms（10.4ms 内释放 30% 初始气量）。
最小改动：free_surface_lbm.py ABB 段加 --evap-rate 旗标（默认 0=关），界面胞按 S/步 液转气，复用 simple-exchange 通道做质量守恒记账。
验证：cells=8 满窗三元组，同上旗标，ctrl+臂，集群 8 核。
prereg: 预测 peak_depth∈[1.4352,1.75]D（上界=补满 29% 体积缺口）；证伪判据：peak_depth<1.4352D 或早死<10.394ms→证伪，honest-FAIL 入账。
