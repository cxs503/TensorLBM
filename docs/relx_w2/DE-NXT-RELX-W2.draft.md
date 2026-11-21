# DE-NXT-RELX-W2 草稿（设计官·碰撞与弛豫格式线 波2）
> 续跑必读本草稿，禁重做。状态=分析完成，待写终卡 TRACK_RELX_W2.md。

## 0. 任务
设计 RELX 线 W2 处方赛马（≤2 臂），交付 BB/NEXT/TRACK_RELX_W2.md（严格机器格式）。
方向=碰撞算子结构。候选：BGK→TRT(魔数稳定性)/MRT 优选弛豫率/液气两相弛豫率分离/界面局部弛豫修正。
排除（不许重开）：网格纯收敛/参数微调/u_limit/ABB 整体移除/rho_g 简单缩放/空穴压亏。

## 1. 已读证据（CROSS 5 DIGEST + X 定谳卡 + 源码）
- 基线锚（cells=8, gap .18 speed .06 capacity simple-exchange release 1.53 span 4 pool 3, u_limit 0.15）：
  peak 1.4352D@11.091ms，误差 −22.73% vs 实验 1.8574D，窗口 24.005ms，mach 0.2598，contact 45。
- CONV12：网格地板 8.6–14.3%（纯收敛不可行，需 ~278 档）。
- CAVSWEEP：16 臂参数扫描无旋钮推峰深（u_limit=纯稳定器，RS/spd 单调变浅）。
- ABBS：ABB 动量承重（零注入 0.304ms 死）；去放大项反补偿 3.23 点 → ABB 非常数项来源。
- T4-RED：界面 ABB 动量注入（f_abb=f_eq_gas+f_eq_gas[opp]−f_post[opp]）是奇数(反对称)模现象；
  mach 尖峰全在界面胞（mach_liquid_max=0.0212 / mach_interface_max=0.3006）。
- KERN-B：p_cav↓→变浅（空穴压亏证伪）。
- GASVISC：no-op（GAS 胞 populations 每步复位，气相粘性无物理通道）。
- SURF1-CURV：LSQ 法向→早死（volume_drift 超阈），证伪。
- RELX W1 两臂：
  - TRTMAGIC：TRT 与 BGK 逐位同值（1.4352204748231856D@11.090571428571428ms）→ 卡判"证伪"。
  - PHASE-TAU（fill 加权 τ_iface）：真实高增益旋钮，剂量-效应非单调：
    +0.5→1.2309D（变浅）、+1.0→接触前早死、+2.0→1.6182D（本席）/1.6716D（R2 席，误差 12.9%/10.0%）
    但峰时刻 +19~21%（12.367/12.635ms vs ctrl 11.091ms）。honest-FAIL（3% 门未过）。

## 2. 关键观察（hypothesis-generation）
观察：W1 TRT 臂与 BGK 逐位同值（全 float 精度）。
事实（源码级，本工作树 main 6a957e70a）：
- solver3d.py L539 collide_trt3d：tau_minus = 0.5 + lambda_trt/(tau_plus−0.5)，Λ 默认 3/16。
  τ₊=0.500388（probe: tau=0.5+3*mu_l）⇒ τ₋ = 0.5 + 0.1875/0.000388 ≈ 483.75（奇数模近乎冻结）。
- free_surface_lbm.py 碰撞段（L981-1080）：分支 kbc/cascaded/cumulant/mrt/else-bgk；
  BGK 分支已有 iface_tau_dose（PHASE-TAU）与 gas_nu_ratio（GASVISC）旗标模式（默认关=位级不变）。
  main 无 "trt" 分支；probe L280 硬编码 collision="bgk"（无 --collision 旗标）。
- ABB 段（L1219-1240）：f_abb = f_eq_gas + f_eq_gas[opp] − f_post[opp]，f_post=碰撞后出射分布（L1137 保留）。
假设：
- H1（接线 no-op，最可能）：W1 的 --collision trt 旗标未传到 free_surface_step（或分支加错函数）
  ⇒ 实跑=BGK ⇒ "TRT 证伪"无效。判别实验=160 步冒烟 max|f_trt−f_bgk|（<1min，最便宜）。
- H2（实现 bug）：TRT 分支执行但公式错（τ₋=τ₊）⇒ 无效应。同样被冒烟门捕获。
- H3（物理：奇数模贡献可忽略）：不可能——奇数模承载流（current），冻结必改 ABB 重构与界面速度重建。
结论：H1/H2 用冒烟门判别；H3 被物理排除。TRT 重测=合法新臂（非排除路线，任务书明列候选）。

## 3. 选臂决策（4 候选 → 2 臂）
- (1) TRT → 选（TRT2）：W1 结果疑为 no-op，重测带接线验证门=纠正战役假阴性，高价值。
- (2) MRT 优选弛豫率 → 弃：collide_mrt3d 默认 s_e=1.19/s_eps=1.4/s_q=1.2（Lallemand 标准组）
  是稳定性取向，与"腔深欠深"缺口的机制连接弱；且选率无物理目标=滑向参数微调（排除线）。
- (3) 液气两相弛豫分离 → 弃：W1 已做（PHASE-TAU 界面 + GASVISC 气相，后者 no-op）。
- (4) 界面局部弛豫修正 → 选（IFTRT）：PHASE-TAU 证明界面弛豫高增益但峰时刻罚（+19%）来自
  偶数模（粘性）改变；IFTRT 只冻界面奇数模（ABB 读取的通道）、偶数模保持物理 τ（局部粘性不变）
  ⇒ 若深度增益来自奇数模/ABB 通道，应加深且峰时刻不罚——与 PHASE-TAU 机制可判别。

## 4. 两臂设计
### ARM TRT2（全局 TRT + 接线验证门）
- 处方：free_surface_lbm.py 碰撞段加 elif collision=="trt" 调 collide_trt3d(f_collide, tau)（已测算子）；
  probe 加 --collision（默认 bgk=位级不变）。
- 验证：① 160 步冒烟 max|f_trt−f_bgk|>1e-12，否则 abort（wiring no-op，honest-FAIL，不跑满窗）
  ② cells=8 满窗三元组。
- 预登记：peak_depth∈[1.44,1.70]D（奇数模冻结对 ABB 注入效应符号不确定，区间双向覆盖）；
  证伪：peak<1.44D 或早死（mach>0.3 或 drift>1e-4 早于 10.394ms）或冒烟门失败。
- 稳定性：TRT 稳定域 τ₊>0.5, τ₋>0.5（Ginzburg 2008）均满足；τ₋→∞ 极限=half-way 方案，
  稳定域 Ma≲0.3，本案 mach_max=0.2598 贴线 ⇒ 早死判据必须预登记。
### ARM IFTRT（界面局部 TRT）
- 处方：仅 INTERFACE 胞（0<=fill<1）做奇偶拆分：偶数模按 τ 弛豫（粘性不变），奇数模按 τ₋≈483.75（冻结）；
  液/气胞 BGK。旗标 iface_trt 默认关=位级不变（同 iface_tau_dose 模式）。
- 机制判别：IFTRT 应加深腔且峰时刻不罚（PHASE-TAU +2.0 的 +19% 来自偶数模粘性改变）；
  若罚复现 ⇒ 深度增益是粘性驱动非奇数模驱动（机制判决）。
- 预登记：peak_depth∈[1.50,1.70]D 且峰时刻∈[10.5,12.0]ms；
  证伪：peak<1.49D 或峰时刻>12.2ms（罚复现）或早死。

## 5. 格式核对（终卡）
- 首行 # TRACK RELX WAVE 2；自检：5/5；goal_met: no；rationale ≤3 行。
- 每臂：## ARM <≤10 大写> / task_key: DE-X-RELX2-<短词> / model 三选一写一个（Qwen3.8-27B）/
  budget: 600/43200/10000000 / prescription ≤10 行 / prereg ≤2 行。总长 ≤60 行。
- 模型三选：Qwen3.8-27B / Qwen3.6-35B-A3B / DeepSeek-V4-Flash-0731（设计官=0731，执行臂=qw38/q35）。

## 6. 报告
- 纯设计单（编排脑不跑仿真）：报告六要素中③计算结果=设计依据数字表（全部溯 X 卡/源码行号），
  ④图=无仿真图，须写明理由（设计官不跑仿真；以数字表+源码行号替代）。
- 报告发布：curl POST $TENSOR_API_BASE/tensor/reports（先登录拿 token），mission_id 26830。
