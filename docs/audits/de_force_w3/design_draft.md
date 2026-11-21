# DE-NXT-FORCE-W3 · 力评估与边界专线 · 设计官草稿（预研，未定稿）

状态：草稿（开工 ≤5 次调用内落地）。终稿 = /root/tp-agent/drop_entry/BB/NEXT/TRACK_FORCE_W3.md

## 0. 已读
CROSS/T1–T5 DIGEST 全读；X/ 定谳卡读 FORCE2-MOMENTUM_VOL（力线本波）、KERN-B、CONV12；NEXT/TRACK_FORCE_W1(LADD_WET)、W2(MOMENTUM_VOL) 已读；
BB/STATUS.md 段读（champion 旗标 capacity+simple-exchange+release 1.53；peak_error 18.21%(12档)/22.73%(8档)；全过门未过）。

## 1. 观察（Observation）
- 力线 W1/W2 两臂至今**没有一次可证伪的力级证据**：W2 判决书明写 force_volume.py 从未被接线（grep 无调用点）、
  两臂 2020 步 history SHA256 位级一致、模块是 D2Q9 二维且源码自带 `# This is wrong, need to fix`。
- W1/W2 预登记指标写「峰误差≤10%」，但峰误差=**几何穿透深度**对实验的偏差，**不是力误差** => 处方与被检验量错配。
- 探针 free_surface_probe.py 输出 depth/volume/drift/mach/rho，**不输出力**。
- 残差结构（CONV12）：12 档峰误差 18.035%，地板模型渐近 8.6–14.3%(本系列)/15.5–17.6%(归档系列) => 纯网格到不了 3%。

## 2. 事实 / 假设 / 未知
- 事实：8 档 1.435220D@11.0906ms（实验 1.857434D@10.3945ms，22.731%）；12 档 1.52244D@10.020ms（18.035%）；
  impact_speed 误差 0.66%/1.78%；postcontact_healthy_ms=24.005 满窗。
- 事实：u_limit=0.15 仅限幅（T4-RED）；空穴压亏=变浅+早死（KERN-B 证伪）；ABB 整体移除/rho_g 缩放=已排除。
- 假设（待证）：H1 球为运动刚体且力双向耦合；H2 球为运动学给定轨迹（力只作诊断量）；H3 探针无力通道。
- 未知：球质量/D/Δx/u_s(格点单位)；边界实现（halfway bounce-back? moving-wall?）；力公式是否含 u_s 修正。

## 3. 竞争假说（对「残差 13–23% 偏浅」）
- H-A 力/边界采样偏差：动量交换含 O(u_s²) 非伽利略项，或移动壁处理缺项 => 腔体几何被系统性带偏。
- H-B 界面 ABB 动量注入（T4-RED）：气相重构注入动量 => 界面动力学污染。
- H-C 几何欠分辨（8–12 格/径，界面几何误差 3–5 格）=> 地板 8.6–14.3%。
- H-D 空穴闭合/气体压缩模型缺陷（KERN-B 已部分证伪）。
- H-E 域边界/出口回波 => 末段深度偏差（INT 线）。

## 4. 代码审计（下一步）
free_surface_lbm.py 的固体边界/力计算/是否耦合；probe 输出通道；case 参数（D, m, Δx, u_s）。

## 5. 暂定两臂（待代码审计后定稿）
- ARM FORCE_INSTR：造力仪器（探针默认零行为改变）+ 伽利略自检 + 实验导出力基准。
- ARM <物理处方>：候选 Galilean-invariant 动量交换 / 力采样面位置 / 移动壁处理 / 湿节点权重复核。

## 6. 定稿决策（代码审计后）
代码审计结论（src/tensorlbm/free_surface_lbm.py + benchmarks/water_entry/single_drop/free_surface_probe.py）：
- 球=运动学给定轨迹的液体团（velocity = -release_lattice * (moving & active)），非 SOLID；**不存在动量交换力通道**，力核不在动力学闭环 ⇒ 力评估对峰深无直接杠杆（除非改界面边界）。
- 探针 history 无 force 字段；力只可能作为后处理量产出。
- ABB=Körner/Thürey 气相压力重建（src/tensorlbm/free_surface_lbm.py L1176-1210），表面张力以 Laplace 折入气相密度（Δρ=3σκ，±0.1 截断），u_limit=伪流限幅（L1719-1725 自述"注入动量→伪流"）。
- 域几何：span 4 ⇒ 侧壁净空 2.06 D（实验槽半宽 8.8 D）；pool-depth 3 ⇒ 腔底距底壁 1.14 D；README 自述"缩域和边界对峰值与回缩的影响仍需量化"。

定稿两臂：
- ARM FORCE3D（诊断探针/仪器）：D3Q19 力通道（link 动量交换 Ladd 湿节点权重 + Chen 控制体通量 + Möbus 伽利略帧修正）→ 探针 --force-out（默认关零行为改变）+ 附加质量解析基准（m_a=0.5ρV）+ DE 案 F_z(t)。交办 FORCE2 判决书 (a)(b)(c)。
- ARM ABBCONS（物理处方）：界面力自洽——u_g* = u_g − Δp_abb_prev/(ρΔt)，打断 ABB 注入→回灌正反馈（Thürey 2007 mass-consistent 界面速度 + Möbus 2001 边界力不回灌；仓库自有注释为据）。最小改动在 ABB 段加 1 持久张量。

备选（W4 储备，本波不开）：SUBGFACE=子网格界面力采样面位置（δ_z=(fill−0.5)Δx，杠杆臂 ≤Δx/2=6.25% D vs 残差 13–23%）；BOUND_ECHO=域宽/侧壁回波（README 未量化项，span 4→6→8→12 标度律）。

## 7. 假说与更新历史
- H0（初判）：力线两波无可证伪力级证据 = 指标错配（峰误差=几何）+ 模块未接线 + 无力通道。→ 代码审计证实（H0 成立，FORCE2 判决书 + 源码）。
- H1：球力双耦合 ⇒ 已证伪（球为运动学给定轨迹，velocity=-release_lattice）。
- H2：球运动学给定 ⇒ 成立；推论：力评估只能作诊断量，改几何必须动界面边界（→ ABBCONS 的立论基础）。
- H3：探针无力通道 ⇒ 成立（history 无 force 字段）。
- H-A（力/边界采样偏差致欠深）：降级为"待力级证据"，FORCE3D 即其判别试验；若基准误差>15% 则 H-A 被证伪（欠深不得归因于力采样）。
- H-B（ABB 注入污染界面动力学，T4-RED）：升为主查，ABBCONS 从力评估侧给结构性判决（RELX 从碰撞算子侧并行）。
- H-C（几何欠分辨地板 8.6–14.3%）：CONV12 已定谳，不再重开。
- H-D（空穴压亏）：KERN-B 已证伪，不再重开。
- H-E（域/边界回波）：留 W4 备选（本波不占臂位，避免与其它线抢域尺寸旋钮）。
