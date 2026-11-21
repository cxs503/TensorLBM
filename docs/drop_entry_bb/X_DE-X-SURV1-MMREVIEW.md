# DE-X-SURV1-MMREVIEW — 谱系对照卡：多相族 × 碰撞 × 湍流 适用性矩阵

> 波次：组合比较矩阵波 W1（章程 `BB/NEXT/TRACK_COMB_W1.md`，owner 令 2026-11-21）
> 本卡定位：**谱系对照（survey）**，不是物理验证单。目标是回答"下一波该选哪条道"，
> 每格结论必须有仓内 `文件:行` 硬验（读 `src/` 核实，禁止凭记忆）。
> 仓：`TensorLBM`，本卡工作树 `agent/m26963`。

---

## 0. 决策与备选（实验设计要素，先声明）

| 项 | 内容 |
|---|---|
| **决策** | 下一波（W2）选哪条模型道来攻「液滴入水峰深 −22.731%」这一未解偏差 |
| **备选 A** | 留在现役族：自由面 VOF（Körner, D3Q19, cells=8）换**碰撞算子**（BGK→Cumulant/KBC/Cascaded/MRT） |
| **备选 B** | 留在现役族：自由面 VOF 换**湍流闭合**（SMAG/WALE/Vreman, `C_s>0`） |
| **备选 C** | 换**多相族**：相场（CH/CAC）路线入水（`benchmarks/pending/phasefield_water_entry/`） |
| **备选 D** | 换**多相族**：色梯度 / 伪势 MCMP / Swift(SOY) 自由能 |
| **本卡不做的** | 不跑全窗、不跑粗带（章程：粗带先行由 PF/COL/TURB 兄弟单承担）；本卡只做**对照与选型** |
| **主判据** | ①该族能否表达本案的密度/动量对比（ρ_water/ρ_air = 852:1）②寄生流量级是否低于本案 3% 深度门③仓内接线成本（现成度） |

**假设**（可证伪）：*峰深 −22.731% 的主因在"模型族表达力"，而不在碰撞算子或湍流闭合*——
即：换碰撞/湍流不会带来可分辨的峰深改善，而换多相族（尤其让气体动量真正进入方程）才是可行道。
**本卡只给证据与可达性判定，不宣称因果**；因果由下一波双臂实验判定。

---

## 1. 统一锚（17 位全精度，逐位取自原文件，禁手抄）

取自战役原卡 `BB/tmp/DE-X-DIAG2-DISSMAP.draft.md:6,48` 与 `BB/tmp/DE-X-K1-DIAG-B.draft.md:93`：

| 量 | 值 | 出处 |
|---|---|---|
| 实验峰深 | `1.857434D@10.394ms(pc)`（原卡 1.857434D@10.39446ms） | DE-NXT-DIAG-W2.draft.md:13（实验峰）、DIAG2-DISSMAP.draft.md:48（误差 −22.731%） |
| ctrl 峰深 | `1.4352204748231814D @ 11.090571428571428ms(abs)` / `10.5433ms(pc)` | K1-DIAG-B.draft.md:93 |
| 窗 | `24.005ms`（postcontact）；绝对窗 `24.55248214285714ms` | INT1-FR-SEAL.draft.md:13 |
| mach | `0.2598` | DIAG2-DISSMAP.draft.md:6 |
| contact | `contact_step = 45` | INT1-FR-SEAL.draft.md:13 |
| drift | `4.8e-6`（原卡 4.81e-6） | DIAG2-DISSMAP.draft.md:6 |
| 冲击速度 | `1.4091844081433575 m/s`（对实验 1.4 m/s 误差 0.656%） | INT1-FR-SEAL.draft.md:13 |
| 峰深误差 | `−22.731%` | DIAG2-DISSMAP.draft.md:48 |
| 网格 | `1 cell = 0.125D ≈ 6.7%` | 章程 TRACK_COMB_W1.md |
| ctrl 旗标 | `--cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3`，`u_limit 0.15` | DIAG2-DISSMAP.draft.md:6 |

**ctrl 的模型身份（硬验）**：`benchmarks/water_entry/single_drop/free_surface_probe.py:501-513`
调用 `tensorlbm.free_surface_lbm.free_surface_step`，参数 `tau=0.5+3*sc["mu_l"]`(:507)、
`surface_tension=sc["sigma"]`(:509)、`rho_gas=1.0`(:510，:344 注释"pressure matched"）、
`collision="bgk"`(:511)、`capacity_redistribution`(:512)、`simple_mass_exchange`(:513)。
⇒ **ctrl = 自由面 VOF（Körner）+ D3Q19 + BGK**，气相无动量。

---

## 2. 目标问题（"入水"到底是什么）

`benchmarks/water_entry/single_drop/protocol.json`（Santini et al. 2022, Fluids 7:50, case 3 单滴）：

| 量 | 值 | 门限 |
|---|---|---|
| 直径 D | 0.00227 m | — |
| 冲击速度 U | 1.4 m/s | 误差 ≤3% |
| 水/空气密度 | 997 / 1.17 kg/m³ ⇒ **密度比 852:1** | — |
| 表面张力 σ | 0.07142 N/m | — |
| 判分 | 逐点相对误差 ≤3%，峰深 ≤3%，≥2 档网格误差严格递减，液体体积漂移 ≤1e-4，**lattice Ma ≤ 0.3** | 全部必过 |

**格点尺度映射**（`benchmarks/water_entry/single_drop/run.py:35-48` `scales()`）：
`rho_g = air/water = 1.17e-3`（气相密度在格点单位里的**物理**编码）、
`dx = D/cells`、`dt = U·dx/1.4`、`sigma`、`g` 全部按 SI 映射。
⇒ 本案对"密度比"的硬要求是 **852:1**；任何多相族若只能把密度对比当"浮力比"（Boussinesq）
或"气相无动量"，就**在原理上无法表达本案的核心物理**（气穴闭合惯性与气相压力）。

**无量纲数**（直接取自 `protocol.json` 的 `case3_measurements`，非推算）：**Re = 3715**（过渡区）、**We = 62.1**、**Fr = 88.0**。
配合 D = 8 格（cell = 0.125D）与 Ma = 0.2598，**本网格的湍流尺度不可解析** ⇒ 湍流闭合轴在本波不是杠杆（详见 §5）。

---

## 3. 方法：硬验口径（不许凭记忆）

1. **仓内契约**：`python -c "from tensorlbm.advanced_collision_contract import collision_capability_matrix; ..."`
   与 `turbulence_capability_matrix()`（本卡已执行，落盘
   `artifacts/de_x_surv1_mmreview/capability_matrix_dump.json`）。
2. **仓内登记**：`tensorlbm.registry.scan_static("src/tensorlbm")`（AST 扫描，不 import）⇒ 25 条能力 id + 成熟度。
3. **源码硬验**：逐族读 `src/tensorlbm/**`，所有断言附 `文件:行`；grep 缺席即"仓内不存在"。
4. **本卡自测探针**：`scripts/de_x_surv1_mmreview_probe.py`（只调仓内现成 benchmark/verification 入口，不改求解器），
   实测可达密度比与寄生流量级；结果 `artifacts/de_x_surv1_mmreview/probe_results.json`。
5. **不引集群作业号**（章程纪律）；作业细节写在 STATUS 通道。

---

## 4. 多相族谱系（8 族 × 4 栏）

> 「可达密度比」= 该族**在动量方程里**能表达的最大液/气密度比；
> 「寄生流量级」= 静止界面/液滴上的最大伪速度（格点单位），即噪声地板。

| # | 族 | 可达密度比（动量方程层） | 寄生流量级 | 入水适配度 | 仓内现成度（文件:行） |
|---|---|---|---|---|---|
| **M1** | **CH4 四阶双重阱自由能相场**（Cahn–Hilliard, quartic） | **不可达 852:1**：动量方程单密度，密度对比只能以 Boussinesq 浮力比进入（`multiphase.py:1266-1267` 参数、`:1322-1324` Boussinesq 缩放） | 平面界面门 `spurious_velocity < 1e-3`（`phasefield/ch_benchmarks.py:249`）；**本卡实测 lee_lin = 3.4319e-05（低于门 29×）、swift = 5.0611e-04（且破坏剖面，§4.1）**；CAC 臂实测基线 **2.79e-4**，接触线放大 3.6–3.8× → **1.0–1.1e-3**（`benchmarks/pending/phasefield_water_entry/NOTES.md` P0/P0b） | **中**：相场入水线仍是 pending；已知阻塞＝壁面相场通量造成的系统性相质量源 **2.5e-4/1e3 步**（超 ≤1e-4 门），且 g=1e-5 时静水压梯度退化到 −1.224（22%） | `phasefield/free_energy.py:30`（`DoubleWellFreeEnergy`）、`multiphase.py:1255`（D2Q9 步进器）、`multiphase3d.py:793`（3D）；registry `lbm.phasefield.cac_hch` = **developing**（`cac_lbm.py:57-66`）；CH 自由能步进器**无 registry 条目** |
| **M2** | **CH6 / 高阶（六阶 sextic）自由能** | — | — | — | **NOT_SUPPORTED**：全仓 `grep -rniE "sextic" src` **零命中**；仓内自由能只有 quartic 双重阱（`multiphase.py:1291` "B: Quartic coefficient"；`phasefield/free_energy.py:30`）⇒ 章程里的 "CH6" 若指六阶项，**仓内没有这个模型**（若指"CH 阶次一致性"，见 M4） |
| **M3** | **HSD 比例控制** | 术语**章程未定义** | — | — | **WITHHELD_UNDEFINED_TERM**：`BB/NEXT/TRACK_COMB_W1.md` 无 "HSD"；`/root/tp-agent/drop_entry` 全目录 grep "HSD" **零命中** ⇒ 本卡不臆造模型。仓内**最近邻**的"比例控制面"是：气相密度 `free_surface_lbm.py:992` `rho_gas`、浮力比 `multiphase.py:1266-1267`、EOS 共存密度比 `pseudopotential_edm3d.py:504` `maxwell_coexistence`、黏度比 `two_phase_viscosity.py:142` `viscosity_ratio_from_tau` |
| **M4** | **Lee–Lin 约定（CH 矩闭合）** | 同 M1 | 同 M1 | **低**：只换相场平衡态闭合，不改动量方程 ⇒ 不直接改峰深 | `phasefield/ch_equilibrium.py:3-12` **自陈**：`lee_lin` 是**局部 Hermite 构造**，"does not claim implementation of every term in a published Lee–Lin hydrodynamic scheme"，且"两种构造都不提供某些高阶 CH 方案所需的时变对流修正" ⇒ 证据等级：局部构造 REAL / 完整方案 **WITHHELD**；无 cs²φ 二阶矩 |
| **M5** | **色梯度（Color-Gradient）** | 2D/3D 均可变，但**无严格不可压**；密度比由初值给（仓内 SCMC 默认 0.7/0.3 ⇒ 2.33:1，`multiphase_benchmarks.py:249-250`） | 待测（口径 `color_gradient2d.py:380` `cg_max_velocity`） | **低**：2D 为主，入水现役臂不用 CG；3D CG 有 MRT+Smagorinsky 接线（`multiphase3d.py:1051`） | `multiphase.py:1135`（2D 单 tau）、`color_gradient2d.py:116`（2D 双 tau + Guo 三项力）、`multiphase3d.py:621`（3D）；**无 registry 条目**；证据：`artifacts/sdaa_poiseuille_two_phase_cg.json` relL2 = 0.1176 / 0.0625 **> gate 0.03（未过门）** |
| **M6** | **伪势 MCMP（双组元 Shan–Chen + MCMP 力）** | 由 EOS 与 G 决定；**本卡实测：G12=−2.5 才分离出 6.707:1（ρ1_liq 0.8963 / ρ2_liq 0.1330），G12=−0.9/−1.5 不分离（ratio≈1.008/1.006，§4.1）**——距本案 852:1 差 127×；伪势 EOS 的 Maxwell 等面积构造本卡**未收敛（NaN，honest FAIL）**，工具见 `pseudopotential_edm3d.py:504` / `mcmp_verification.py:577` | 仓内自带寄生流基准（`mcmp_verification.py:965` `spurious_current_benchmark`，报 `u_max`、`ca_spurious`、`density_ratio`）；**本卡实测共存板 u_max = 9.2427e-03（比相场门 1e-3 高 9.2×）；液滴臂 honest FAIL（guardrail：液滴溶解/与周期镜像合并，§4.1）** | **中-低**：3D SC 存在（`multiphase3d.py:224`），MRT 变体存在（`multiphase3d.py:1168`），但入水现役臂未用；高密度比下 SC 需 MRT 才稳（`multiphase3d.py:1184-1186`） | registry `lbm.multiphase.shan_chen_force.d3q19` = **developing**；验证工具 `mcmp_verification.py` / `mcmp_verification_3d.py:565` |
| **M7** | **Swift 自由能（SOY / sailfish 移植）** | 同 M1（单密度 + 浮力比） | 待测 | **低**：SOY 为 2D D2Q9 专用（`phasefield/soy_two_phase.py:1-6`），round-1 复现误差 3.7–7.2% | `phasefield/soy_two_phase.py`（Swift–Osborn–Yeomans 核，含壁面修正 4 项缺陷修复记录）；`phasefield/free_energy_pressure.py:1-45`（sailfish 移植，压力张量内嵌平衡态，Laplace 可直测） |
| **M8** | **自由面 VOF（Körner 2005）** | **气相无动量**（mass ≠ rho 的独立质量账）：密度对比不进动量方程；`rho_gas=1.0` 是**压力约定**（`free_surface_probe.py:344,510`）⇒ 852:1 的**气相惯性**不可表达 | 无"寄生流"概念；对应噪声地板是**质量漂移** 4.8e-6（锚）与界面转换残差 | **高（现役 ctrl 臂）**：唯一有原生 collision 开关的族 | `free_surface_lbm.py:981`（`free_surface_step`，Körner 全模型：GAS/INTERFACE/LIQUID/SOLID、ABB 气压重建、Körner 质量交换 eq.9 + Thürey 表 4.1 选择器）；2D VOF `free_surface.py:557`；registry `lbm.free_surface.vof_sweep` = **developing** |

### 4.1 本卡自测探针结果（第一手，非记忆）

探针 `scripts/de_x_surv1_mmreview_probe.py`（只调仓内 benchmark/verification 入口，CPU/fp64）：

### 4.1 本卡自测探针结果（第一手，非记忆）

探针 `scripts/de_x_surv1_mmreview_probe.py`（只调仓内 benchmark/verification 入口；CPU/fp64）。
**运行记录（诚实）**：首次递交因调度节点上 `ModuleNotFoundError: No module named 'tensorlbm'`
（仓非 pip 安装）**失败**；修正＝脚本内自插 `src` 到 `sys.path`，复跑**成功**（8 核，10.1 min）。
逐项结果（原始 JSON：`artifacts/de_x_surv1_mmreview/probe_results.json`）：

| 探针 | 参数 | 实测 | 判定 |
|---|---|---|---|
| CH 平面界面（**lee_lin**） | 96×4, xi=3, A=B=0.004, kappa=0.018, Γ=0.5, 1000 步 | `max_physical_speed = 3.4319084020401606e-05`；`phase_mass_drift = 2.7289787042998456e-16`；仓内 5 项 checks **全 true** | **优**：伪流比仓内门 1e-3 低 **29×** |
| CH 平面界面（**swift** 历史路径） | 同上，仅 `geq_convention="swift"` | `max_physical_speed = 5.061062424478316e-04`；checks：`surface_tension_integral = false`、`profile_preserved = false`（末步 `profile_linf_change = 0.717856993201048`） | **swift 路径同参数下破坏剖面与表面张力积分**，伪流比 lee_lin 高 **14.7×** |
| MCMP 共存（平面板, G12=−0.9） | n=64, 3000 步 | `density_ratio = 1.0082406828972772`（未分离），`converged = false`，`u_max = 2.2039172472432256e-04` | 该耦合下**不出相分离** |
| MCMP 共存（G12=−1.5） | 同上 | `density_ratio = 1.0064608849270746`，`converged = false`，`u_max = 1.9040987587545488e-04` | 同上 |
| MCMP 共存（G12=−2.5） | 同上 | `density_ratio = 6.707257309180478`（ρ1_liq 0.8963 / ρ2_liq 0.1330），`flat_interface_dp = 6.579607725147343e-04`，`u_max = 9.242722764611244e-03` | **本探针中 MCMP 最高可达密度比 = 6.7:1**，但伪流 9.24e-3 **≫ 相场门 1e-3（9.2×）** |
| MCMP 液滴寄生流（G12=−1.5, R=12, n=64, 3000 步） | — | `u_max = 5.517379613593221e-04`，`u_max_over_cs = 9.556381815388199e-04`，`dp = 1.1922270059585571e-03`，`density_ratio = 0.9836133742710655`；**guardrails：`centre_not_liquid` / `no_crossing` / `not_converged`** | **honest FAIL**：该参数下液滴已溶解或与周期镜像合并，Laplace/σ 不可测；只保留 u_max 原始值，不据此下结论 |
| 伪势 EOS 共存密度比（CS/PR，Maxwell 等面积） | G = 0.5…3.0 | 全部 `density_ratio = NaN`，`converged = false` | **honest FAIL**：等面积构造在给定 brackets/ψ 下未收敛 ⇒ **本卡不给出伪势 EOS 的可达密度比**（不外推、不编造） |
| 色梯度 / SCMC 静态液滴（2D, 64×64, R=10, 2000 步） | — | 探针报错 `ModuleNotFoundError: No module named 'matplotlib'`（节点缺绘图库，`run_static_droplet` 内部画图） | **环境 FAIL**：本单未测色梯度伪流；仓内旁证 `artifacts/sdaa_poiseuille_two_phase_cg.json` relL2 = 0.1176/0.0625 > gate 0.03（未过门） |

**探针适用域（写死，不外推）**：2D / fp64 / 64×64（或 96×4）/ ≤3000 步；3D 与 cells=8 下伪流通常更高、SC 稳定密度比更低。

**由探针直接支持的谱系判定**：
1. 相场族（lee_lin）伪流地板 ≈ **3.4e-5**，MCMP/SC 伪流 ≈ **9.2e-3** ⇒ 相差 **≈2.7 个量级**；
   本案 3% 深度门对噪声最敏感的族是 **MCMP/SC**，其次 CG（未测，用仓内旁证替代）。
2. **M4（Lee–Lin 约定）优于 legacy swift**：伪流 14.7×、且 swift 破坏剖面与表面张力积分
   ⇒ M4 是 M1 家族里**零结构成本的立即可用改动**（`multiphase.py:1271` 的 `geq_convention` 参数），
   但它只改相场平衡态闭合，**不改动量方程** ⇒ 对本案峰深是**弱**因果通道。
3. **MCMP 的可达密度比在本探针只有 6.7:1（G12=−2.5）**，离本案 852:1 差 **127×** ⇒
   即使升耦合，SC 路线在本案也不具备密度比裕度；这与 §4 中 "M6 需 MRT 才稳" 的接线事实一致。


---

## 5. 碰撞族（5）与湍流族（3）

### 5.1 碰撞族合同等级（`advanced_collision_contract.py:53` 实跑，AVAILABLE 全部）

| 族 | D3Q19 入口 | D3Q27 入口 | registry 成熟度 | 在**多相族内**的可插拔性（关键列） |
|---|---|---|---|---|
| BGK | `solver3d.collide_bgk3d` | `d3q27.collide_bgk27` | D3Q19 **mature** / D3Q27 developing | M1/M7 相场步进器**写死 SRT**（`multiphase.py:1344`）；M5/M6 3D 有；**M8 VOF 原生支持**（`free_surface_lbm.py:1224` 的 `else: # bgk` 分支） |
| TRT | `solver3d.collide_trt3d` | `d3q27.collide_trt27` | D3Q19 **mature** / D3Q27 developing | M5 2D 双 tau（`color_gradient2d.py:116`）形近 TRT 但不是通用 TRT 核；**M8 VOF 不支持**（无 "trt" 分支） |
| MRT | `solver3d.collide_mrt3d` | `d3q27.collide_mrt27` | D3Q19 **mature** / D3Q27 developing | M5/M6 3D 有专用步进器（`multiphase3d.py:1051` / `:1168`）；**M8 VOF 支持**（`free_surface_lbm.py:1185`，且带 per-cell τ_eff） |
| **RR**（RLBM 正则化） | `solver3d.collide_rlbm3d` | `d3q27.collide_rlbm27` | D3Q19 **mature** / D3Q27 developing | **仓内无任何多相族接线**：M1/M5/M6/M8 全部无 "rlbm" 分支（grep 命中 0） |
| Cumulant | `cumulant.collide_cumulant_d3q19` | `cumulant.collide_cumulant_d3q27` | **registry 无 `lbm.collision.cumulant.*` 条目**（未登记成熟度） | **M8 VOF 原生支持**（`free_surface_lbm.py:1181`）；M1/M5/M6 无 |

> **战役杠杆的连带损失（重要）**：`iface_tau_dose`（战役 RELX1 的"界面 τ 剂量"杠杆，
> `free_surface_lbm.py:1241-1250`）**只在 BGK 分支内**；一旦把 collision 切到
> mrt/kbc/cascaded/cumulant，该杠杆**静默失效**（不是报错，是行为改变）。

### 5.2 湍流族合同等级（`turbulence_capability_contract.py:662` 实跑）

| 族 | 仓内实现 | 合同状态 | 说明 |
|---|---|---|---|
| **SMAG** | `turbulence/turbulence.py:216,245,270,345,472,497`（2D/3D/27 × BGK/MRT）+ CG `cg_advanced_collision.py:149` + VOF `free_surface_lbm.py:1226`（BGK 支路＝ctrl 实际走的）/ `:1204`（MRT 支路） | `IMPLEMENTED / CONTRACT_TESTED` ⇒ **status = WITHHELD_NO_PHYSICS_VALIDATION** | 合同测试只验证算子代数（形/守恒/平衡恒等），**不验证湍流物理** |
| **WALE** | `turbulence/turbulence.py:896,929,954,1069,1147` + CG `cg_advanced_collision.py:150` + VOF `free_surface_lbm.py:1233`（BGK 支路）/ `:1211`（MRT 支路） | 同上 **WITHHELD_NO_PHYSICS_VALIDATION** | D3Q19/D3Q27 均有 BGK 与 MRT 版 |
| **VLES** | 仓内**无 "VLES" 符号**（grep 零命中）。最近邻：**Vreman**（`turbulence.py:803,844,984,1014,1039,1235,1310`）与 **DDES/SAS**（`ddes.py:105,177,379`；合同里 DDES **仅 D2Q9 BGK**，状态 `IMPLEMENTED_ONLY / WITHHELD_NO_CONTRACT_TESTS`） | VLES 本身 **NOT_SUPPORTED**；Vreman 同 SMAG/WALE withheld；DDES 更弱 | 若"VL"指 Vreman，则已有；若指 VLES（无粘 LES），仓内无 |
| 旁证 | `core/turbulence.py:1-36` 只是**身份元数据**（`SMAGORINSKY`/`WALE` 常量 + Protocol），文件自陈 "no implementation is implied here" | — | 别被 `core/turbulence.py` 误导以为有实现 |

**湍流轴对本案的适配度（硬算）**：Re = 3715（§2，`case3_measurements.Re`）、D = 8 格、Ma = 0.2598（已顶到 0.3 门）、
冲击时间窗 24 ms。8 格/滴的网格**不可能解析**剪切层/空穴涡系，
且任何闭合都会抬高 Ma（吃掉 0.3 门的余量 0.04）⇒ **湍流轴在本波判为不适配（低）**。

---

## 6. 组合矩阵（8 多相族 × 5 碰撞 × 3 湍流）

图例：`✔` 直接接线｜`△` 需适配（小改动，有明确落点）｜`✘` 族内不支持｜`W` 合同 withheld｜`H` 章程术语未定义。
`ΔC`＝该格对**峰深**（本案主观测量）的因果通道强度（强/中/弱/无）。

### 6.1 多相 × 碰撞

| 多相族 | BGK | TRT | MRT | RR(RLBM) | Cumulant | 证据 |
|---|---|---|---|---|---|---|
| M1 CH4 相场 | ✔（写死 SRT） | ✘ | ✘ | ✘ | ✘ | `multiphase.py:1344` |
| M2 CH6 | ✘ | ✘ | ✘ | ✘ | ✘ | 仓内无 sextic |
| M3 HSD | H | H | H | H | H | 术语未定义 |
| M4 Lee–Lin | ✔（只换 g 平衡态） | ✘ | ✘ | ✘ | ✘ | `ch_equilibrium.py:22-69` |
| M5 色梯度 2D | ✔ | △（双 tau 近似） | ✘ | ✘ | ✘ | `color_gradient2d.py:116` |
| M5 色梯度 3D | ✔ | ✘ | ✔ | ✘ | ✘ | `multiphase3d.py:621,1051` |
| M6 SC/MCMP 2D | ✔ | ✘ | ✘ | ✘ | ✘ | `multiphase.py:689` |
| M6 SC/MCMP 3D | ✔ | ✘ | ✔ | ✘ | ✘ | `multiphase3d.py:224,1168` |
| M7 Swift(SOY) 自由能 | ✔（写死 SRT） | ✘ | ✘ | ✘ | ✘ | `phasefield/soy_two_phase.py` |
| M8 自由面 VOF（**ctrl**） | ✔（现役） | ✘ | ✔ | ✘ | ✔ | `free_surface_lbm.py:1173-1185`（bgk/kbc/cascaded/cumulant/mrt） |

### 6.2 多相 × 湍流

| 多相族 | SMAG | WALE | VLES/Vreman/DDES | 证据 |
|---|---|---|---|---|
| M1 CH4 相场 | △（3D CG 式接线有先例，无相场接线） | △ | ✘ | `cg_advanced_collision.py:149-150` 只证 CG |
| M5 色梯度 3D | ✔（`multiphase3d.py:1065,1101`） | ✘ | ✘ | 只有 Smagorinsky 支路 |
| M6 SC 3D | ✔（`multiphase3d.py:1181,1211`） | ✘ | ✘ | 同上 |
| M8 自由面 VOF（**ctrl**） | ✔（SMAG `free_surface_lbm.py:1226`（BGK 支路＝ctrl 走的）/ `:1204`（MRT 支路）） | ✔（WALE `:1233`/`:1211`） | ✔（Vreman `:1236`/`:1214`；DDES 不在其中） | 三者都已接线，但见 §5.2 适配度判定 |
| 其余 | ✘ | ✘ | ✘ | — |

### 6.3 对峰深（本案主观测量）的因果通道

| 道 | ΔC | 理由（硬验） |
|---|---|---|
| 换**碰撞算子**（VOF 内 BGK→Cumulant/KBC/Cascaded/MRT） | **弱** | 碰撞算子改的是非平衡态松弛，不改界面质量交换与气相惯性；且切走 BGK 会丢掉 `iface_tau_dose` 杠杆（`free_surface_lbm.py:1241-1250`） |
| 加**湍流闭合** | **弱/无** | Re≈3.7e3、8 格/滴、Ma 已 0.2598（门 0.3）：不可解析且吃 Ma 余量（§2、§5.2） |
| 换**多相族**（让气相动量/压力真正进方程） | **中-强** | 现役 VOF 气相无动量（`free_surface_probe.py:344,510`）⇒ 空穴闭合惯性与气相压力不在方程里；这正是战役 CA 伏笔区（CAVSWEEP / DIAG2-CAVCLSE 一线的现象） |
| 换**多相族**（相场 CAC/CH） | **中** | 相场路线在本案仍 pending，且已知有壁面相场通量质量源（2.5e-4/1e3 步，超 ≤1e-4 门，NOTES.md P0-4） |

---

## 7. 结论 → 下一波（W2）选型

**主推（W2 主臂）**：**留在现役族 M8（自由面 VOF），换碰撞算子**——
但**不是**换 TR T/RR（VOF 无接线），而是 **BGK → Cumulant / Cascaded / KBC / MRT 四选一**，
四臂同窗同旗标（`--cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53
--span 4 --pool-depth 3`，`u_limit 0.15`），**位级锚门**照抄 §1，
**预登记**：峰深变化 <3% ⇒ 判"碰撞轴证伪"（这是**有价值的负结果**，直接封死 W2 的碰撞轴）；
若 ≥3% 且方向指向 +（更深），再升一档。

> 纪律提醒：换 collision 后必须**显式记录** `iface_tau_dose` 失效（BGK 专属），
> 否则两臂差异会被误读成"剂量效应"。

**并行第二臂（真正的可行道）**：**补气相惯性**——不是换族，而是给现役 VOF 增加**气相动量/压力耦合**
（自由面 VOF 的气相在现役实现里是"无动量壳"，`free_surface_lbm.py:981-1010` 的 GAS 语义）。
这条线的最小可验证形式：气相壳给一个等效压力源（可参数化强度），看峰深对**气相压力强度**的单调性；
先粗带（n≤80）判方向，再全窗（全窗必经 `sdaa_fs_probe(_split)` 壳）。

**明确降级（本波不选）**：
- **湍流轴**（SMAG/WALE/VLES）：Re=3715 + 8 格/滴 + Ma 0.2598 ⇒ 不可解析；且合同层全部
  `WITHHELD_NO_PHYSICS_VALIDATION`（`turbulence_capability_contract.py:662`）。
- **M2 CH6**：仓内无 sextic 自由能（grep 零命中）⇒ 无从比较（除非先实现 sextic 项，那是开发单不是对照单）。
- **M3 HSD**：章程术语未定义 ⇒ 先请 owner 定义，再排期。
- **M5 色梯度 / M6 MCMP / M7 Swift(SOY)**：入水现役臂无接线；色梯度只有 2D 双 tau 且其 2D 基准
  relL2 = 0.062–0.118 **未过 0.03 门**（`artifacts/sdaa_poiseuille_two_phase_cg.json`）⇒ 不适合做主臂。

**W2 建议排期（4 单）**：
1. `COL4-CUM` VOF+Cumulant 全窗（若 PF/COL/TURB 兄弟单已覆盖则只做汇总）
2. `COL5-MRT` VOF+MRT 全窗（**注意**：`iface_tau_dose` 失效需在卡里写明）
3. `GAS4-INERT` 气相惯性粗带（参数扫描：气相压力强度 × 3 档）
4. `MMP-SURV2` 谱系卡刷新（把 W1 本卡的负结果回填，形成谱系台账）

---

## 8. 局限 / 未完成 / 风险（honest）

1. **本卡不产生新的物理结论**：所有"适配度"判定是**结构与合同层**判定（谁接线了、成熟度多少、密度比能否进方程），
   不等于"换了会变好"。因果归 W2 双臂。
2. **探针为 2D/fp64/小网格**（64×64、≤3000 步）测得的寄生流与共存密度比**不能外推到 3D/cells=8**：
   3D 伪流通常更高、SC 在 3D 的稳定密度比更低。数值见 §4.1，适用域写死在探针脚本文档串里。
3. **未覆盖**：M3（HSD 术语待定义）、M2（sextic 不存在）两族没有可测对象；
   色梯度/SO 的寄生流只测了部分配置。
4. **撞空穴/飞溅的观测定义**本身在战役中被质疑过（NOTES.md "尚未覆盖"段：无接触角边界 ⇒ 喷溅/空腔/闭合的定量结论不可得）。
   本卡的"适配度"只针对**峰深**这一个观测量。
5. **regret 风险**：若 W2 发现"碰撞算子"其实通过**界面质量交换的数值耗散**影响峰深，
   则本卡"碰撞轴弱"的判断偏保守；反之若"气相惯性"在本案其实已被 ABB 气压重建部分覆盖，
   则第二臂的增益会小于预期（现役 VOF 已有 ABB 气压重建，`free_surface_lbm.py:1-16` 自陈）。
6. 本卡**未跑全窗/粗带**：按章程，物理结论由兄弟单（PF1/PF2/PF3/PF4/PF5/COL1-3/TURB1-2/CMB1）承担；
   本卡只做对照，避免与兄弟单重复消耗。

---

## 9. 复算

```bash
# 仓内合同矩阵（本卡 §5 的两把尺）
python - <<'PY'
from tensorlbm.advanced_collision_contract import collision_capability_matrix
from tensorlbm.turbulence.turbulence_capability_contract import turbulence_capability_matrix
print(collision_capability_matrix())
print(turbulence_capability_matrix()["wale"])
PY

# 仓内能力登记（成熟度，AST 静态扫描，不 import）
python -m tensorlbm.registry --static --out /tmp/registry.json
python -c "from pathlib import Path; from tensorlbm.registry import scan_static; \
caps,_=scan_static(Path('src/tensorlbm'), repo_root=Path('.')); \
print(len(caps)); [print(c.id, c.kind, c.maturity) for c in caps]" 

# 本卡探针（第一手数字；CPU/fp64）
PYTHONPATH=src python scripts/de_x_surv1_mmreview_probe.py
# 结果：artifacts/de_x_surv1_mmreview/probe_results.json
```

产物：`BB/X/DE-X-SURV1-MMREVIEW.md`（本卡）、
`docs/drop_entry_bb/X_DE-X-SURV1-MMREVIEW.md`（双写，diff=IDENTICAL）、
`scripts/de_x_surv1_mmreview_probe.py`（探针）、
`artifacts/de_x_surv1_mmreview/{capability_matrix_dump.json,probe_results.json}`。
