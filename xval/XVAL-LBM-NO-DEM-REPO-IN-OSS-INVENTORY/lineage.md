# ② 调研·方法谱系 — 没有 OSS 孪生时，DEM 该对标什么

`anchor = XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY · mission 18215 · repo TensorLBM`

## 2.1 标准做法（教科书/文献，逐条给 external_anchor）

| 环节 | 标准做法 | 出处（external_anchor） | 本仓实现位置 |
|---|---|---|---|
| 接触本构 | 线性弹簧-阻尼（Kelvin–Voigt）：`f_n = k δ − η v_n`，切向库仑限幅 | Cundall & Strack 1979, *Géotechnique* 29(4) 47–65 | `src/tensorlbm/dem/contact.py:LinearSpringDashpot` |
| 软球 + 流体耦合 | 软球 DEM 与 LBM 双向耦合的奠基工作 | Tsuji, Kawaguchi & Tanaka 1993, *Powder Technol.* 71, 239–250 | `src/tensorlbm/dem/dem_solver.py`（drag → 加速度 → 位置） |
| 恢复系数标定 | 线性模型 `e = exp(−πζ/√(1−ζ²))`，`ζ = η/(2√(k m))` | Di Renzo & Di Maio 2004, *Chem. Eng. Sci.* 59, 525–541 | 本单 `analytic_restitution()` |
| 标定层级 | 解析 → 单元测试 → 整体标定，先闭式后实验 | Coetzee 2017, *Powder Technol.* 310, 104–113 | 本单验收集顺序（A/B/C/D→E） |
| 阻力闭合 | `Cd = 24/Re (1+0.15 Re^0.687)`（Re<1000） | Schiller & Naumann 1935, *VDI Z.* 77, 318–320 | `dem_solver.compute_drag_force` |
| 阻力参考曲线 | Clift–Grace–Weber 标准曲线（小球 Re<1e5 的工程基准） | Clift, Grace & Weber 1978, *Bubbles, Drops and Particles* | `src/tensorlbm/drag_closure.py:cd_clift_gauvin` |
| 沉降标准算例 | 单球沉降（含 Re≈1.5 / 31.9 两档实验） | ten Cate et al. 2002, *Phys. Fluids* 14, 4012–4024 | 本单用其**无界极限**（Stokes/Clift–Gauvin）作闭式锚 |
| 动量交换测力 | Ladd 逐链动量交换 | Ladd 1994, *J. Fluid Mech.* 271, 285–309 | `src/tensorlbm/drag_momentum.py` |
| 周期阵列限域修正 | Hasimoto 周期阵列阻力修正 | Hasimoto 1959, *J. Fluid Mech.* 5, 317–328 | `benchmarks/pending/sphere_drag_closure`（部分采用） |

## 2.2 为什么本单选这四类算例（选型逻辑）

1. **接触恢复系数 / 接触时长 / 峰值重叠**（Arm A/B）：纯闭式，零经验参数，
   能把"接触本构 + 时间积分"从"流体模型"里剥离开单独判死；这是 Coetzee 标定层级的第 0/1 层。
2. **Stokes 极限终速**（Arm C）：与求解器**同一条闭合式**，只证"力→加速度→速度"的
   管路与质量/浮力记账（H3 自明性检查），**不作为物理验收**（见 predeclared 的 H3）。
3. **Schiller–Naumann 沉降 vs Clift–Gauvin 文献终速**（Arm D）：真正的"文献锚"，
   检验 DEM 侧所选阻力律在 Re∈[0.04, 5.7] 上是否落在公认曲线上。
4. **流体侧 LBM 的 Cd(Re)**（Arm E）：不设门，只做**诚实体检**——耦合消费的是流体侧
   的阻力律，若流体侧自身在限域盒里偏 15% 以上，DEM 侧再准也不能宣称为"耦合已验证"。

## 2.3 迁移方案（可直接复用的做法，落 `oss_solutions`）

| 迁移点 | 做法 | 落点 |
|---|---|---|
| M1 | DEM 侧任何无量纲判据，一律先写闭式参考函数（本单 40 行），再与求解器比分 | `xval/.../run_xval.py:analytic_*` |
| M2 | 接触类算例的时间步以 **t_c 的分数**（20/40/80/160）而非绝对秒给定——接触刚度变了不用重取步长 | `contact_arm(steps_per_contact)` |
| M3 | 沉降类算例的时间步以**颗粒弛豫时间 τ_p 的分数**给定（25/100），并报尾窗漂移判收敛 | `settling_arm(dt_div)` |
| M4 | 无 OSS 孪生时的验收集分层：闭式（门）→ 文献曲线（门）→ 同式自明（非门）→ 外部求解器体检（非门） | `predeclared.json:decision.rule` |
| M5 | 检验"无张力限幅"这类实现细节对物理量的偏置：同一 solver 换一个 `clip=False` 的控制模型做 A/B | `_NoClipKelvinVoigt` 控制臂 |

## 2.4 已知限制（本谱系不覆盖的）

* ten Cate 2002 的**原始实验值**未在本单作为数值锚（无原文表格可复核，拒绝手抄数字）；
  本单只用其"无界流体终速"极限，并以 Stokes/Clift–Gauvin 闭式代替。
* 未做体标定（休止角、料仓卸料）；本单验收集中在单元级闭式锚。
* Hertz–Mindlin 非线性接触未验证（本仓实现是线性模型）。