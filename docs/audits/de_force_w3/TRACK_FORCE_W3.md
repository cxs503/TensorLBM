# TRACK FORCE WAVE 3
自检：5/5（格式机读同 W1/W2 模板✓/臂=2✓/预登记齐✓/未重开排除路线✓/出处可溯✓）

goal_met: no
rationale: 力线两波（LADD_WET/MOMENTUM_VOL）无一次可证伪的力级证据：FORCE2 判决书实测 force_volume.py 从未被接线（grep 无调用点、D2Q9 二维、源码自带 "this is wrong"），且预登记的"峰误差"是几何深度不是力；探针无 force 字段，DE 案球体是运动学给定轨迹（velocity=-release_lattice），力核不在动力学闭环里。
本波=①造可证伪的力仪器（D3Q19+解析基准，交办 FORCE2 (a)(b)(c)）②从力评估侧攻 T4-RED 记录的 ABB 注入→回灌正反馈（碰撞算子族归 RELX 线，本波不碰）。
不追 3% 门（现最好 12 档 18.035%，网格地板 8.6–14.3%），只求给全战役一条"力级可证伪"通道+一次结构性处方判决。

## ARM FORCE3D
task_key: DE-X-FORCE3-FORCE3D
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是力评估仪器官。任务=造 D3Q19 三维力通道并接进 DE 探针（FORCE2 交办：(a)三维实现 (b)接入输出力时程 (c)与解析/文献力基准对比）。
物理出处=Ladd(1994) 动量交换（湿节点体积分数加权）；Chen(1998) 控制体动量通量 F=-∮Π·n dS, Π=Σ_i f_i c_i c_i；Möbus(2001) 伽利略帧修正（球速 u_s=0.06 格点，u_s²/U²~O(1)）。
最小改动=①新增纯后处理模块 src/tensorlbm/force_momentum_exchange.py（不碰内核）②free_surface_probe.py 增 --force-out（默认关=逐位零行为改变），history 增 F_z(t)、A_w(t)、两估计量互差。
DE 案的"力"=运动学驱动液滴区的控制体反力（该区动量变化率−规定动量注入），非固体动量交换。
解析基准=脉冲加速球的附加质量 F=m_a·a, m_a=0.5ρV（Landau–Lifshitz 11；黏性修正 O(ν)，D²/ν≈6 s≫窗口）⇒ 小盒子可跑，误差由离散球面 O(Δx/D) 主导。
验证=①基准上两估计量互差+相对附加质量误差表 ②DE 案 cells=8 满窗（--cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3）输出 F_z 时程 ③结论"力核可否作为欠深归因通道"。
prereg: 预测区间=基准上两估计量互差≤5%、相对附加质量误差≤15%；DE 案 |F_ME−F_CV|/F≤15%、注入反馈比 r=|Δp_abb|/(ρ|u_g|A_w)∈[0.05,0.5]。
证伪判据=无力字段产出（grep 无调用点）=仪器未建成 honest-FAIL；或基准误差>15%/互差>10%=力核不可信⇒欠深不得归因于力采样（写死）。

## ARM ABBCONS
task_key: DE-X-FORCE3-ABBCONS
model: Qwen3.8-27B
budget: 600/43200/10000000
prescription: 你是界面力自洽官。任务=消除 ABB 重构的"注入→回灌"正反馈：气相侧平衡态所用界面速度 u_g 必须扣除上一步 ABB 注入的动量。
物理出处=Thürey(2007) ABB 用体积分数一致的（mass-consistent）界面速度而非原始碰撞后动量；Möbus(2001) 边界力不得自我回灌；仓库自有注释 src/tensorlbm/free_surface_lbm.py L1719-1725 自述"ABB 按界面速度注入动量→伪流→被 u_limit 截断"，而 u_limit 已定谳为纯稳定器。
机制=现 u_g 取碰撞后宏观速度（含上一步 ABB 注入的动量）⇒u_g↑⇒注入↑⇒伪流自激；自洽速度 u_g*=u_g−Δp_abb_prev/(ρΔt)（每胞持久张量）打断回路。
最小改动=free_surface_lbm.py ABB 段加 1 个持久量 Δp_abb_prev + 速度修正（旗标关=逐位不变，另设 ctrl 臂做位级 parity）；若本征不稳，退到 Thürey 质量一致界面速度（复用已有质量交换账本）。
验证=cells=8 满窗三元组（同一命令）+ 深度时序逐点表 + 每步注入动量统计（复用 ownership ledger 的 abb_population_delta）。
prereg: 预测区间=峰深∈[1.40,1.62]D（机制中心≈1.50D，误差 13–22%）、r 降至≤0.1、postcontact_healthy_ms=24.005 满窗。
证伪判据=峰深|Δ|<0.05D（该通道对欠深无杠杆 honest-FAIL）；或峰深<1.40D（方向相反，机制被否）；或早死（health_limit 早于 10.394ms）⇒不判定；或 ctrl parity 不成立⇒作废。
