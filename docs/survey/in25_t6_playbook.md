# IN25-T6 五措 PLAYBOOK（外部成熟技术吸收条目）

由 `tensorlbm.external_survey.playbook.PLAYBOOK` 程序化导出；每条含动作 / 首步 / 陷阱 / 验收阈值 / 锚点出处。

## P1. 外流阻力判据改用 SUBOFF Table 14 量纲阶梯，禁用‘AFF-8 CT≈0.004’

- **措施**：`large_grid` — 大网格先例（规模/网格收敛研究）
- **动作**：以 Liu & Huang (1998) Table 14 的 6 个拖曳速度点换算 CT(Re) 为主判据；每个算例只与最近 Re 的阶梯点比较，并同时报出湿面积/密度约定。
- **首步**：python -c "from tensorlbm.external_survey.suboff_anchor import scorer_candidates; print(scorer_candidates()[1]['points'])" → 把 points 写进新 benchmark 的 reference.json（含 S_wet 区间与 ρ）
- **陷阱**：湿面积约定未定：S_wet 由 6.00→6.35 m² 使 CT 变 5.8%；ρ 由 999→1025.9 再变 2.6%；文献常引 0.004 需要 S_wet≈5.50 m²（小于裸体）才能成立，属不可复现约定。
- **验收**：|CT_CFD − CT_EFD(最近 Re 点)| ≤ 10%（含面积区间）；< 3% 才可称‘定量对标成功’
- **落库**：TensorLBM
- **锚点**：
  - SUBOFF ladder candidate `SUBOFF-AFF8-CT-LADDER`
  - SUBOFF ladder candidate `SUBOFF-AFF1-CT-LADDER`
  - Peng Jiang et al. (2025) — Effects of appendages on the turbulence and flow noise of a submarine model using high-order scheme (`2503.19674`)

## P2. 并行扩展照抄块结构域分解 + 幽灵层，不先优化全局网格

- **措施**：`parallel_lbm` — 并行 LBM 成熟实践（扩展性/域分解）
- **动作**：把分布式层实现为「块结构 + 幽灵层交换 + 内核生成」三层（waLBerla/Palabos 架构）；通信按块而不是按格，块数 ≥ 节点数×每节点 socket 数。
- **首步**：量化一次 halo 交换的有效载荷比：Q=19 分布在 Q_PAD=32 上 → 有效率 19/32=59.4%，先把该浪费测出来再谈优化
- **陷阱**：Q_PAD 对齐带来的无效 lane 传输；小网格上线程超订（本仓实测 L=16 时 64 线程比 16 线程慢 5 倍）；块数不足导致节点空转
- **验收**：8 卡强扩展效率 ≥ 70%（本仓已入库的 8 卡公开数 83% 作为上限参照）
- **落库**：TensorLBM
- **锚点**：
  - Bauer et al. (2021) — waLBerla: A block-structured high-performance framework for multiphysics simulations (`10.1016/j.camwa.2020.01.007`)
  - Latt et al. (2021) — Palabos: Parallel Lattice Boltzmann Solver (`10.1016/j.camwa.2020.03.022`)
  - Godenschwager et al. (2013) — A framework for hybrid parallel flow simulations with a trillion cells in complex geometries (`10.1145/2503210.2503273`)

## P3. 局部加密只做 2:1 块嵌套 + 体积守恒界面，先在小算例上验界面

- **措施**：`amr` — 局部加密（块结构 AMR/渐变网格）
- **动作**：采用 2:1 嵌套块、时间步比 2 的亚循环，界面用守恒（体积）形式而非纯插值；加密只布置在壁面/尾流压力梯度带。
- **首步**：以「充分发展槽道 + 绕球」两算例做 AMR/均匀细网格对照，记录界面质量残差与 CD 差
- **陷阱**：插值型界面破坏局部质量守恒 → 非物理压力抬升（与 ISBB 泄漏同源，见 2611.07137）；细化比不是 2 时需额外时间插值，误差阶掉到 1
- **验收**：同 Re 下 AMR 与均匀细网格 CD 差 < 2%，界面质量残差 < 1e-6/步
- **落库**：TensorLBM
- **锚点**：
  - Filippova & Hänel (1998) — Grid Refinement for Lattice-BGK Models (`10.1006/jcph.1998.6089`)
  - Lagrava et al. (2012) — Advances in multi-domain lattice Boltzmann grid refinement (`10.1016/j.jcp.2012.03.015`)
  - Dorschner et al. (2016) — Grid refinement for entropic lattice Boltzmann models (`10.1103/physreve.94.053311`)
  - Abbas Fakhari (2026) — Improving the accuracy of the momentum-exchange method in lattice Boltzmann methods (`2611.07137`)

## P4. 高 Re 稳定性先换松弛率谱（MRT/熵），Cs 只做细调并标注为工程默认

- **措施**：`les` — 湍流 LES-LBM（模型常数与 MRT 稳定性）
- **动作**：稳定性预算放在 MRT 松弛率谱/熵碰撞上；Smagorinsky 只用来在 τ 上加涡粘，并对速度做 ±c 限幅防止非平衡应力发散。
- **首步**：固定 Re 扫 Cs ∈ {0.05, 0.10, 0.14, 0.16}，记录发散阈值与 CD 漂移（用同一网格、同一步数）
- **陷阱**：Cs 与网格强耦合：加密等同改变有效 Re；MRT 参数选错会过度耗散、抑制湍流；文献级 Cs 推荐区间在本次核验中 absent，本仓 0.05/0.1 属工程默认而非文献锚
- **验收**：Cs 扫描区间内 CD 变化 < 5% 且无发散；否则判‘Cs 未定标’，不得报定量结论
- **落库**：TensorLBM
- **锚点**：
  - Hou et al. (1995) — A lattice Boltzmann subgrid model for high Reynolds number flows (`10.1090/fic/006/12`)
  - Lallemand & Luo (2000) — Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability (`10.1103/physreve.61.6546`)

## P5. 近壁优先 slip-velocity bounce-back；力评估用 modified MEM + 质量守恒校验

- **措施**：`wall_model` — 壁面模型（近壁处理适用域与误差阶）
- **动作**：高 Re 壁面用 slip-velocity BB 耦合模型一致壁面函数；避免正则化/高阶重构壁法向梯度；曲壁测力改用按真实边界位置修正的动量交换，并每步校验总质量。
- **首步**：同一 Re 下跑 BB vs 正则化两版，比较沿程摩阻分布与三网格收敛一致性；同时打印总质量漂移
- **陷阱**：壁面不在格点中点 → 经典动量交换力误差显著；ISBB 破坏局部质量守恒 → 压力漂移；三阶重构对壁法向梯度高度敏感（平壁也敏感）
- **验收**：y+ ∈ [30,300] 时摩阻误差 < 10%，加密时摩阻单调收敛且质量漂移 < 1e-6/步
- **落库**：TensorLBM
- **锚点**：
  - Ponsin & Lozano (2025) — Wall boundary conditions for lattice Boltzmann simulations of turbulent flows with wall functions (`10.1063/5.0283930`)
  - Jorge Ponsin & Carlos Lozano (2026) — Enhanced Wall Boundary Modeling for Turbulent Flows Using the Lattice Boltzmann Method with Adaptive Cartesian Grids (`2601.18443`)
  - Abbas Fakhari (2026) — Improving the accuracy of the momentum-exchange method in lattice Boltzmann methods (`2611.07137`)
  - Musker (1979) — Explicit Expression for the Smooth Wall Velocity Distribution in a Turbulent Boundary Layer (`10.2514/3.61193`)

