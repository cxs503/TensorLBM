# XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN — TensorLBM × lbmpy 交叉对标

**任务** m17410（第 1 轮·开源对标，XVAL）｜**仓库** TensorLBM｜**分支** `agent/m17410`
**对标对象** `lssfau/lbmpy` 2.0 的 Allen-Cahn + CSF(Korteweg) 力链
（`lbmpy/phasefield_allen_cahn/`，Fakhari / van der Graaf, *PhysRevE.96.053301*）
**判据** 误差 < 15%（owner 精度标准）｜**日期** 2026-11-13

---

## ① 问题与参考实现出处

### 1.1 问题

TensorLBM 的 Allen-Cahn 相场模块（`src/tensorlbm/allen_cahn_lbm.py`）把"相场输运核"
判为 REAL，但把**变密度流体耦合**判为 WITHHELD：静态液滴在密度比 1000:1 时第 5 步发散、
10:1 时第 12 步发散（`docs/allen_cahn_phasefield_r1_20261108.md`）。本单要回答的是：
**这条"化学势 → 毛细血管力 F = μ∇φ → 外力项"的力链，与成熟开源实现相比差多少、差在哪。**

### 1.2 参考实现（零手抄：全部由 lbmpy 自身函数产出）

| 项 | 值 / 出处 |
|---|---|
| 包 | `lbmpy` 2.0（`python -c "import lbmpy; lbmpy.__version__"`），`pystencils` 2.0 |
| 力链源文件 | `lbmpy/phasefield_allen_cahn/kernel_equations.py`（`chemical_potential_symbolic` L19、`surface_tension_force` L112、`isotropic_gradient_symbolic` L37） |
| 离散算子源文件 | `lbmpy/phasefield_allen_cahn/derivatives.py`（`laplacian_symbolic` L6、`isotropic_gradient_symbolic` L46） |
| 参数源文件 | `lbmpy/phasefield_allen_cahn/parameter_calculation.py`（`beta` = 12σ/W L77、`kappa` = 3σW/2 L82） |
| sha256 | 三文件哈希见 `artifacts/xval_m17410/xval_allen_cahn_force_chain.json → lbmpy_provenance.source_files` |
| 物理模型 | 化学势 μ = 4βφ(φ−1)(φ−½) − κ∇²φ；CSF 力 F = μ∇φ（isotropic 梯度） |

**做法（零手抄的落地方式）**：不抄公式，而是把 lbmpy 的**符号表达式原样取出**，
用 `sympy.lambdify` + 相邻单元 roll 在给定相场上求值（`scripts/xval_m17410_allen_cahn_csf_force_chain.py::_lbmpy_evaluator`）；
离散权重则由**克罗内克 δ 输入 lbmpy 自己的算子**反读出
（`stencil_weight_table()`）——权重表因此是实测产物而不是文档转录。

---

## ② 方法与实现

### 2.1 约定映射（解析可证，非拟合）

lbmpy 用 φ∈[0,1]、TensorLBM 用 φ∈[−1,1]。取 φ_TL = 2φ_lb − 1：

```
ψ_TL = (β_TL/4)(φ_TL²−1)² = 4β_TL φ²(1−φ)²  =  ψ_lb = φ²(1−φ)²·β_lb   ⇒ β_TL = β_lb/4
κ_TL = κ_lb/4，∇φ_TL = 2∇φ_lb，μ_TL = 2μ_lb（同一坐标下）
⇒ F_TL = μ_TL∇φ_TL = 4 μ_lb ∇φ_lb / 2·... = μ_lb ∇φ_lb = F_lb     （代数恒等）
```

数值验证：`β_TL = 3σ/(2W_TL)`、`κ_TL = 3σW_TL/4`（`free_energy_parameters`），
取 **W_TL = W_lb/2**（lbmpy 的 `interface_thickness` 对应 tanh 剖面 2n/W，即 ℓ = W/2），
即得 β_TL = β_lb/4、κ_TL = κ_lb/4；测试 `test_chemical_potential_scales_by_four_under_the_convention_map` 锁定该关系（rel 1e-12）。

**结论**：两条力链**不是"相似"，而是同一代数对象**；任何差异只能来自**离散算子**与
**力链包含哪些力项**。这就是本对标真正的比较对象。

### 2.2 移植：D3Q19 各向同性算子

新增 `src/tensorlbm/phasefield/isotropic_fd.py`：

```
lap_iso = (1/3)Σ_{6 轴邻} φ + (1/6)Σ_{12 面对角邻} φ − 4φ
grad_x  = (1/6)(φ_E − φ_W) + (1/12)[(φ_BE−φ_BW)+(φ_NE−φ_NW)+(φ_SE−φ_SW)+(φ_TE−φ_TW)]
```

边界语义复用 `operators._neighbor`（periodic 折叠 / no_flux 镜像），因此不会与七点族漂移。
`free_energy.DoubleWellFreeEnergy.chemical_potential` / `force_mu_grad_phi` /
`force_minus_phi_grad_mu` 新增 `operator="central7"|"isotropic_d3q19"`，**默认仍是 central7（向后兼容，有测试锁定）**。

### 2.3 判据（先于实验固定）

| 代号 | 度量 | 阈值 |
|---|---|---|
| M1 | 力场相对 L2 误差 ‖F_TL − F_lb‖/‖F_lb‖ | **< 15%** |
| M2 | 由 ∫F_r dr 反演的 σ_rec vs 输入 σ（3D：Δp = 2σ/R） | < 15% |
| M3 | 界面带内伪切向力占比 ‖F_t‖/‖F_r‖ | 越小越好（无阈值，报告倍数） |
| M4 | 净力 \|ΣF\|/Σ\|F\|（周期盒内应为 0） | < 1e-10 |

### 2.4 算例（相同相场、双侧求值）

| 算例 | 几何 | 网格 (z,y,x) | σ | W_lb | ℓ=W/2 | R | R/ℓ |
|---|---|---|---|---|---|---|---|
| C1-planar | 平面 tanh 界面 | 8×8×64 | 1e-2 | 4 | 2 | — | — |
| C2-droplet | 球滴 | 64³ | 1e-2 | 4 | 2 | 16 | 8 |
| C3-W{3,4,6} | 球滴（厚度敏感性） | 64³ | 1e-2 | 3/4/6 | 1.5/2/3 | 16 | 10.7/8/5.3 |
| C4-R{8,12,16} | 球滴（网格加密，box=4R） | 32³/48³/64³ | 1e-2 | 4 | 2 | 8/12/16 | 4/6/8 |
| C5-σ{2e-2,5e-3} | 球滴（σ 线性检查） | 64³ | 2e-2 / 5e-3 | 4 | 2 | 16 | 8 |

**无量纲参数（完整）**：密度比 ρ_H/ρ_L = 10:1（ρ_H=1, ρ_L=0.1，仅力链归一用）；
黏度比 ν_H/ν_L = 10（ν_H=0.1, ν_L=0.01，`AllenCahnParameters` 必填）；
毛细数 Ca = 0、Weber = 0（全部为准静态/静止场，无流动）、Reynolds = 0；
界面解析度 ℓ/Δx = 2；Laplace 数 La = σRρ/μ² 只与流体步相关，本单不做流体步故不适用（见 ⑥局限）。

---

## ③ 计算结果（程序化注入，JSON/CSV 全量在 `artifacts/xval_m17410/`）

### 3.1 离散权重表（lbmpy 自身算子 vs 本仓移植，δ 响应实测）

| 量 | lbmpy 2.0 | TensorLBM `isotropic_d3q19` | TensorLBM `central7` |
|---|---|---|---|
| Laplacian 中心 | −4 | −4 | −6 |
| Laplacian 轴邻 | 1/3 | 1/3 | 1 |
| Laplacian 面对角 | 1/6 | 1/6 | 0 |
| 梯度 轴邻 | 1/6（±） | 1/6（±） | 1/2（±） |
| 梯度 面对角 | 1/12（±） | 1/12（±） | 0 |
| **逐邻点最大差** | — | **0.0（bit 级一致）** | 非零 |

### 3.2 力链对照表（ref = lbmpy；got = TensorLBM 两种算子族）

| 算例 | 变体 | M1 相对 L2 | M1 判据 15% | M2 σ 反演误差 | M2 判据 | M3 伪切向占比 | M4 净力 |
|---|---|---|---|---|---|---|---|
| C1-planar | central7 | 5.02e-16 | ✅ | — | — | — | 1.1e-14 |
| C1-planar | isotropic | 2.92e-16 | ✅ | — | — | — | 1.1e-14 |
| **C2-droplet** | **central7** | **1.121e-01** | ✅（勉强） | −4.00%（ref 自身 −6.47%） | ✅ | **1.155e-02** | 3.6e-17 |
| **C2-droplet** | **isotropic** | **1.170e-14** | ✅ | −6.47%（与 ref 完全相同） | ✅ | **5.030e-04** | 1.9e-17 |
| C3-W3 | central7 | 2.058e-01 | ❌ | −7.58% | ✅ | 1.920e-02 | 5.9e-19 |
| C3-W3 | isotropic | 6.37e-15 | ✅ | −11.75% | ✅ | 1.317e-03 | 5.9e-17 |
| C3-W6 | central7 | 3.990e-02 | ✅ | −0.86% | ✅ | 5.291e-03 | 7.4e-17 |
| C3-W6 | isotropic | 1.34e-14 | ✅ | −1.98% | ✅ | 1.126e-04 | 6.8e-17 |
| C4-R8 | central7 | 6.882e-02 | ✅ | −2.25% | ✅ | 1.113e-02 | 1.9e-17 |
| C4-R8 | isotropic | 5.49e-15 | ✅ | −4.44% | ✅ | 5.292e-04 | 1.9e-17 |
| C4-R12 | central7 | 8.977e-02 | ✅ | −2.89% | ✅ | 1.145e-02 | 3.3e-17 |
| C4-R12 | isotropic | 7.91e-15 | ✅ | −5.12% | ✅ | 5.119e-04 | 5.1e-17 |
| C5-σ2e-2 | central7 / iso | 1.121e-01 / 1.17e-14 | ✅ / ✅ | −4.00% / −6.47% | ✅ | 同 C2 | 同 C2 |
| C5-σ5e-3 | central7 / iso | 1.121e-01 / 1.17e-14 | ✅ / ✅ | −4.00% / −6.47% | ✅ | 同 C2 | 同 C2 |

**读法（三点，缺一不可）**

1. **移植是逐位正确的**：isotropic 变体 vs lbmpy 的力场差 1e-14～1e-15（浮点噪声级），
   σ 反演误差与 lbmpy **完全相同到全部有效位**（−6.47%/−11.75%/−1.98%），说明差异只剩共享的
   "解析 tanh ≠ 离散平衡解"这一项，与实现无关。
2. **七点族（现状默认）在曲率下确实错**：球滴上相对 L2 误差 11.2%（R/ℓ=8）、20.6%（W=3 时 **超 15% 判据**），
   而伪切向力占比是 lbmpy 的 **23 倍**（1.155e-02 vs 5.03e-04）——这正是寄生流的驱动力。
3. **平面界面不敏感**：C1 两族都到机器精度。独立验证（`test_operators_coincide_on_a_one_dimensional_field`）：
   对一维变化场，七点与各向同性算子差 2e-15，**各向异性项在无曲率时恒为零**。
   ⇒ 结论要写成"曲率下才暴露"，不能写成"七点永远错"。

---

## ④ 收敛性与参数敏感性（至少两点）

**敏感性 1 — 界面厚度 W（R=16 固定）**：central7 误差 20.6%(W=3) → 11.2%(W=4) → 4.0%(W=6)，
近似 ∝ W⁻²·（各向异性四阶项），W=3 时越过 15% 判据；isotropic 全程 1e-14（≈ 与分辨率无关，
其误差项被消去）。**工程含义：界面越薄（越省算力）现状实现越危险。**

**敏感性 2 — 网格加密（W=4 固定，R/ℓ = 4/6/8）**：central7 6.88%(R=8) → 8.98%(R=12) → 11.21%(R=16)。
误差不随加密下降而是缓升——因为 L2 归一化分母与界面元胞数同阶增长，而各向异性误差是**逐面元**
的相对 O(1) 偏差（不是 h² 截断误差主导）。isotropic 恒 5e-15。
**这是"加密不能救各向异性"的直接证据**，也是必须换算子的理由。

**敏感性 3 — σ 幅值（2e-2 / 5e-3）**：两种算子族的所有指标逐位不变（力链对 σ 线性，
误差是纯几何/离散量）——作为"度量不依赖物理标定"的对照。

**M2 的独立性检查**：σ 反演误差在 W=6 时降到 −1.98%（lbmpy 自身值，非 TensorLBM），
说明 ∫F_r dr = 2σ/R 的偏差是**解析剖面未落在离散平衡上**的度量，随 ℓ/Δx 收敛；
它不是实现差异的判据，报告中如实标注。

---

## ⑤ 与 OSS 的差异清单 + 移植取舍

| 维度 | lbmpy（ref） | TensorLBM（本单前） | 处置 |
|---|---|---|---|
| 序参量约定 | φ∈[0,1]，ψ=φ²(1−φ)² | φ∈[−1,1]，ψ=(β/4)(φ²−1)² | **不移植**：解析等价（②2.1），移植=无谓重构 |
| β/κ | β=12σ/W，κ=3σW/2，ℓ=W/2 | β=3σ/(2W)，κ=3σW/4，ℓ=W | 保留各自；映射 W_TL=W_lb/2 写入对标脚本与测试 |
| **Laplacian** | D3Q19 各向同性（−4, 1/3, 1/6） | 七点（−6, 1, 0） | **已移植** `isotropic_fd.isotropic_laplacian_3d` |
| **梯度** | 各向同性（轴 1/6 + 面对角 1/12） | 两点中心（1/2） | **已移植** `isotropic_gradient_3d` |
| CSF 力形式 | F = μ∇φ（Korteweg，与化学势同源） | `allen_cahn_lbm` 已是 μ∇φ；另有 2D `csf_force_model.py` Brackbill κδn̂（**不参与本对标**：仅 2D，且历史上用单位法向丢曲率） | 力形式一致；Brackbill 变体留作后续 3D 化时再对标 |
| **力链力项** | F = F_st(μ∇φ) + F_p(−⅓ρ_devΔρ∇φ) + F_v(黏性应力修正) + 体积力（`hydrodynamic_force` L169/L55/L78） | **仅** F_st + ρg | **未移植（关键缺口）**：需与流体步一起验证，见 ⑥ |
| 力/密度归一 | 力统一除 ρ(φ)=ρ_L+φ(ρ_H−ρ_L) 后入 Guo；变密度力项带 2 次子迭代 | `allen_cahn_step` 除 ρ(φ)，Guo 半步速度位移，无子迭代 | 部分一致；子迭代缺失属流体步范畴 |
| 相场力（界面追踪） | `interface_tracking_force` 另有 (1/3)(1−4(φ−φ₀)²)/W·n̂ 加在相场 LB | 无（相场用有限差分 AC 更新，非 LB） | **不移植**：架构不同（FD-AC vs LB-AC），强行移植会引入双重驱动 |
| 边界条件 | 生成核为周期流；另 `contact_angle.py` 提供接触角边界 | periodic / no_flux + solid_mask | 本单只对标 periodic；no_flux 语义本仓自测覆盖（`test_no_flux_boundary_mirrors_instead_of_wrapping`） |
| 精度控制 | 二阶各向同性 FD | 七点 FD（各向异性） | **已移植**（默认仍 central7，opt-in 切换） |

**移植取舍（明确声明）**：
1. **移植算子，不移植约定**——约定映射解析等价，重写只会制造回归面；
2. **不移植 F_p/F_v** ——它们只在流体步里有意义，必须与变密度流体步联合验证（否则就是新的"未验证声称"）；
   且 `allen_cahn_step` 目前整体 WITHHELD（静态液滴 5 步发散），先修算子再谈力项，顺序不能颠倒；
3. **默认值不改**——`central7` 仍是默认，避免静默改变既有算例；isotropic 为显式 opt-in，并已由 15 个数值断言覆盖。

---

## ⑥ 失败模式、局限、工件、生态摩擦

**失败模式（诚实记录）**
- C1 平面算例两族到机器精度：若只做平面算例，会得出"七点族也没问题"的**错误结论**——
  各向异性必须用**曲率几何**（球滴/柱）才能暴露，这是本单最重要的方法论教训。
- σ 反演指标 M2 一度给出"−194%"的荒谬值：原因是把 `∫F_r dr` 的**符号**当成了误差
  （Korteweg 力指向内部，积分恒负）且早期径向分解把标量当成矢量扣除（`f_t = f − f_r` 而非
  `f − (f·n̂)n̂`）。修法：取绝对值 + 正确矢量投影，修正后与 lbmpy 逐位一致（−6.47% 双侧相同）。
  留档在脚本注释与 JSON 中，避免后人重犯。
- 早期版本 `_symbol_offsets` 用 `Field` 索引 API 有三次试错（`field[off]` → `field(*off)` → `field.center.get_shifted(*off)`）；
  lbmpy 2.0 的 `pystencils` 访问器是 `Access.get_shifted(...)`，且 `Field.Access` 本身是 `sympy.Symbol` 子类。

**局限（不粉饰）**
- **只对标了力链，没有对标流体步**：`allen_cahn_step` 的变密度 BGK 仍为 WITHHELD（静态液滴 5 步发散），
  本单不声称已修复它。F_p/F_v 缺失的量级需与流体步联合测量，属下一轮（XVAL round 2 / 修复轮）。
- 只做 D3Q19；lbmpy 亦支持 D2Q9/D3Q15/D3Q27，本仓仅 3D，未覆盖 2D 对标。
- 未做 `contact_angle` 边界对标（本仓无该能力）。
- 网格上限 64³（本机单核秒级）；更大网格与流体步联合测试应走 `cluster_queue`（本单未触发，计算量 < 10 min）。

**工件（全部在 worktree）**
- 脚本：`scripts/xval_m17410_allen_cahn_csf_force_chain.py`（`--quick` 跳过加密研究）
- 数据：`artifacts/xval_m17410/xval_allen_cahn_force_chain.{json,csv}`（含 sha256 出处、权重表、逐算例指标）
- 图：`fig_force_chain_droplet.png`（径向力剖面 + 误差柱）、`fig_force_chain_refinement.png`（加密）、
  `fig_force_error_slice.png`（逐面元 \|F_TL−F_lb\| 切片）
- 代码：`src/tensorlbm/phasefield/isotropic_fd.py`、`src/tensorlbm/phasefield/free_energy.py`（operator 选择）
- 测试：`tests/test_allen_cahn_isotropic_fd_xval_m17410.py`（10 项，全绿；lbmpy 缺失时 7 项仍可跑）

**复现**
```bash
python scripts/xval_m17410_allen_cahn_csf_force_chain.py          # 全量，~2 min
python -m pytest tests/test_allen_cahn_isotropic_fd_xval_m17410.py -q
```

**生态摩擦**
1. `oss_solutions` / `oss_inventory` 表对本 agent 数据库角色 **permission denied**，KB 三查只能退化为
   `/api/v1/knowledge/search`（返回 `total:0`，store 为空）；`/api/v1/tensor/knowledge/search` 返回
   `engine_offline`（8021 端口 connection refused）。⇒ 对标出处只能以**本机源码 + sha256** 自证，
   无法登记到 oss_solutions 台账（建议平台开放该表的只读或提供写入 API）。
2. 平台 API 登录路径是 `/api/v1/auth/login`（不是 `/api/v1/tensor/auth/login`，后者 401），
   与 bootstrap 中 `$TENSOR_API_BASE/tensor/reports` 的写法不一致，易踩。

---

## 判定

| 判据 | 结果 |
|---|---|
| M1 力链相对 L2 < 15% | **isotropic_d3q19：1.17e-14 ✅（优于判据 12 个数量级）；central7：1.12e-01 ✅ 但 W=3 时 2.06e-01 ❌** |
| M2 σ 反演 < 15% | 两族全部 ✅（最大 −11.75%，且与 lbmpy 自身值逐位相同） |
| M4 净力 | 3.6e-17 ✅ |
| 是否 < 3% | 力链误差：**是**（1e-14）；σ 反演：**否**（−6.47% @W=4，源于解析剖面与离散平衡的差，双侧同值）——如实声明 |

**结论**：TensorLBM 的 AC/CSF 力链与 lbmpy **代数等价、离散逐位可对齐**；差距全部落在
1) 算子族（已补齐，默认未改）、2) 力项集合（F_p/F_v 仍缺，需流体步联合修复）。
本轮把"WITHHELD 的根因"从猜测变成了可测量的两条具体缺口。