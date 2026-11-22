# TRACK RELX WAVE 3
自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）
CROSS已读：T1/T2/T3/T4/T5_DIGEST（5 张全读）；X/ 定谳卡：RELX1-PHASETAU-R2、RELX2-TRT2-R2、REJECT_RELX2-TRT2、COL1-MRT、DE-X-COL2-REG、DE-X-CMB3-REGSMAG、DE-X-CMB1-HYDRODCA、GAS1-GASVISC 全读；STATUS.md
goal_met: no
rationale: 深化-时刻兑换率实测下界 6.53 ms/D（RR 全场 14.69、界面 τ 分离 6.53、HYDRO 14.85）：深度门需 +0.422214D ⇒ 最省也要 +2.757ms，而 3% 时刻预算仅 ±0.163ms（可接受兑换率 ≤0.386 ms/D，差 17×）⇒ 既有旋钮族双门不可同时满足，本波改攻"结构"（高阶矩处理方式）而非旋钮强度。
奇数模/ABB 通道已被 TRT2（0.29ms 早死）/IFTRT（0.51ms 早死）二次证伪；"液气两相 τ 分离"对气相结构性无效（free_surface_lbm.py:987 气胞恒置 feq_gas、从不碰撞）；kbc/cascaded/cumulant 分支是死代码（.advanced_collision_d3q19 模块不存在，导入实测 ModuleNotFoundError）；真 KBC 的 D3Q19 权重缺陷（本单已修+单测：tests/test_kbc_d3q19_lattice.py，α=1 与 BGK 等价实测 rel 5.0e-05）。
## ARM RRIFACE
task_key: DE-X-RELX3-RRIFACE
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是碰撞算子结构官。任务=把 COL2-REG 已实测的 RR 正则化碰撞由"全场"改为"仅界面胞"(flags==INTERFACE, 0<fill<1)，体相胞保持 BGK，τ 同为 0.5003877，不动任何旋钮强度。
物理出处=Ladd 1994 (PRE 50:1765)：BGK 的虚假粘性源自界面处三阶非平衡矩（∝ 曲率、只在界面出现）⇒ 界面局部投影应拿到全场 RR 的大部分加深收益；而全场 RR 的 +1.763ms 时刻罚若源于体相高阶矩被清零，局部化应显著削罚 ⇒ 直接检验"深化-时刻"兑换率能否被结构改善。
最小改动=①cherry-pick agent/m26958 的 RR 端口（commit edbd0a013：collision="rr" 分支+probe rr_tau+--rr-tau 壳通道）②碰撞段按界面掩码混合 f=torch.where(iface_mask, f_rr, f_bgk)，f_rr 用 collision_d3q19_advanced.collide_regularized_stress_d3q19(f_collide, feq, tau)（caller 供 feq ⇒ Guo 半步位移与 BGK 同构）③probe 增 --rr-iface 旗标（默认关=位级不变）④tools/sdaa_fs_probe*.py 壳通道 ⑤probe 增记气穴平面面积/宽度时序（几何 vs 耗散归因用）。
验证=Gate0 单测（掩膜局部性/默认关位级不变/守恒矩）+160 步冒烟（max|f_rriface−f_bgk|>1e-6，否则 abort）+cells=8 满窗三元组（同旗标 gap .18 speed .06 capacity simple-exchange release 1.53 span 4 pool 3 u_limit .15 steps 10000；锚门：ctrl 臂复现 1.435220D@10.543339ms pc / 24.005ms / mach 0.2598 / drift 4.81e-6 / contact 45）。
prereg: 预测 peak_depth∈[1.50,1.60]D（RR 全场 1.555257D，界面局部应保 ≥70% 收益⇒≥1.519D）；峰时刻(pc)∈[10.9,12.4]ms（罚 ≤+0.9ms ⇒ 兑换率≤8 ms/D，结构改善成立；罚 ≥+1.7ms ⇒ 判决"界面高阶清理必付时刻税"，14.7 ms/D 为族下界）。
证伪判据：peak<1.487D｜早死（mach>0.3 或 drift>1e-4 早于 10.394ms）｜冒烟/单测失败｜锚门不通过（位级锚不成立则本臂作废，不报结论）。
## ARM ENTROPY
task_key: DE-X-RELX3-ENTROPY
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是碰撞算子结构官。任务=把自由面基线 BGK 换成 KBC 三段熵碰撞（src/tensorlbm/kbc_collision.collide_kbc，α 逐胞由熵最小化求解），τ=0.5003877 保持剪切段 ω_ν=1/τ 精确。
物理出处=Karlin–Bösch–Chikatamarka (PRL 106:236304, 2011)：KBC 把 f^neq 正交分解为剪切段（ω_ν=1/τ ⇒ 物理剪切应力零过阻尼）/体相段/ghost 段（≥3 阶，α·ω_g，α 由熵条件求）⇒ 只削虚假粘性噪声而保留界面惯性 ⇒ 预期比"整体清零 ghost"的 RR 全场时刻税更小（RR 把 ghost 惯性一并丢掉）。
最小改动=①核验本单已修的 kbc_collision._weights D3Q19 权重（1/3, 1/18×6, 1/36×12，与 d3q19._W_VALUES 逐项一致；_D3Q19_DIRS 序与 d3q19.C 逐项一致免置换；若 arm 分支未含该修复则应用同一补丁）②free_surface_lbm.py 修死分支 collision=="kbc"（现导入 .advanced_collision_d3q19 不存在⇒ImportError 实测）接真 KBC，并加同构修正 +ω·[feq(u+F/2ρ)−feq(u)]（核内 feq 用 u、求解器用半步位移，否则构成混淆变量）③probe 增 --collision-kbc 旗标（默认关=位级不变）④tools/sdaa_fs_probe*.py 壳通道。
验证=Gate0 等价单测（α=1、ω_b=1/τ ⇒ 与求解器 BGK 路径相对误差 ≤1e-3；本单实测 max abs 1.68e-05 / rel 5.0e-05，float64 核 vs float32 求解器，证明权重/序/力同构）+160 步冒烟（与 BGK 分叉>1e-6、无 NaN、drift<1e-6）+cells=8 满窗三元组（锚门同 ARM1）；成本门：单步 >5×BGK ⇒ 降级为已预登记的 α=0.5 固定档并在卡中如实登记。
prereg: 预测 peak_depth∈[1.49,1.62]D（ghost 段熵最优松弛应在界面削虚假粘性⇒接近 RR 全场 1.555257D）；峰时刻(pc)∈[10.7,12.2]ms（保留 ghost 惯性⇒时刻税 <RR 全场 +1.763ms，兑换率目标 ≤8 ms/D）。
证伪判据：peak<1.487D｜峰时刻罚 ≥+1.7ms ⇒ 判决"高阶矩清理必付时刻税"（碰撞族深化破不了兑换率，RELX 线可结案转他线）｜早死｜Gate0 等价失败或冒烟无分叉（诚实 FAIL，不跑满窗）。
