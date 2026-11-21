# TRACK RELX WAVE 2
自检：5/5（格式机读通过/臂≤2/预登记齐/无重开排除路线/出处可溯）
goal_met: no
rationale: W1 TRT 定谳存疑：τ₊=0.500388、Λ=3/16 ⇒ τ₋=0.5+Λ/(τ₊−0.5)≈483.75（奇数模近乎冻结），正确接线的 TRT 不可能与 BGK 逐位同值 ⇒ W1 臂大概率接线 no-op，「TRT 证伪」无效。
PHASE-TAU R2 证界面弛豫=高增益旋钮（+2.0 收回缺口 56%），但峰时刻罚（+19%）来自偶数模粘性改变。
本波=①TRT 带接线验证门重测（冒烟分歧检查）②界面局部 TRT（只冻奇数模、偶数模保持物理 τ）——攻 ABB 奇数模通道而不改局部粘性。
## ARM TRT2
task_key: DE-X-RELX2-TRT2
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是碰撞算子验证官。任务=重跑全局 TRT（W1 TRTMAGIC 存疑：τ₊=0.500388、Λ=3/16 ⇒ τ₋≈483.75，奇数模近乎冻结，ABB 读 f_post[opp] 必变；W1 与 BGK 逐位同值物理上不可能 ⇒ 大概率接线 no-op）。
物理出处=Ginzburg(2008) TRT 魔数 Λ=3/16（奇数模控制界面/壁面位置误差）；T4-RED 定谳：ABB 动量注入是奇数(反对称)模现象。
最小改动=free_surface_lbm.py 碰撞段加 elif collision=="trt" 分支调已测 collide_trt3d(f_collide, tau)（solver3d.py L539，capability 已验）；free_surface_probe.py 加 --collision 旗标（默认 bgk=位级不变）。
验证=①160 步冒烟：max|f_trt−f_bgk|>1e-12，否则 abort（wiring no-op，honest-FAIL，不跑满窗）②cells=8 满窗三元组（--cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3）。
prereg: 预测 peak_depth∈[1.44,1.70]D（奇数模冻结对 ABB 注入效应符号不确定，区间双向覆盖）；证伪判据：peak<1.44D（无改善）或早死（mach>0.3 或 drift>1e-4 早于 10.394ms）或冒烟门失败。
## ARM IFTRT
task_key: DE-X-RELX2-IFTRT
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是界面弛豫结构官。任务=仅对界面胞(0<=fill<1)施加 TRT：偶数(对称)模保持物理 τ（局部粘性不变），奇数(反对称)模用 τ₋=0.5+Λ/(τ−0.5)≈483.75（Λ=3/16，冻结）；液/气胞保持 BGK。
物理出处=①T4-RED：ABB 重构 f_abb=f_eq_gas+f_eq_gas[opp]−f_post[opp] 读碰撞后奇数结构，奇数模冻结直接改注入 ②PHASE-TAU R2：界面弛豫高增益但峰时刻罚来自偶数模粘性改变 ③Ginzburg(2008)：奇数模控制界面动力学。
机制判别=IFTRT 应加深腔且峰时刻不罚（PHASE-TAU +2.0 的 +19%）：若深度增益来自奇数模/ABB 通道，峰时刻应贴 ctrl 11.091ms；若罚复现 ⇒ 增益是粘性驱动（机制判决）。
最小改动=free_surface_lbm.py BGK 分支加 iface_trt 旗标（默认关=位级不变，同 iface_tau_dose 模式）：逐胞奇偶拆分，偶数模按 τ 弛豫，奇数模界面胞按 τ₋、其余按 τ；probe 加 --iface-trt 旗标。
验证=cells=8 满窗三元组+峰时刻轴（peak_t_ms vs ctrl 11.091ms）+ctrl 位级 parity 臂。
prereg: 预测 peak_depth∈[1.50,1.70]D 且峰时刻∈[10.5,12.0]ms（无 +19% 罚）；证伪判据：peak<1.49D 或峰时刻>12.2ms（罚复现）或早死（mach>0.3/drift>1e-4 早于 10.394ms）。
