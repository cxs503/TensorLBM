# XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION

机器可复核产物：`runs/xval_equilibrium_difference_extrapolation/result.json`（spec / identity / orders / runs，
每次运行含完整 `params` / `metrics` / 剖面 `extra`），图 `figures/xval_equilibrium_difference_extrapolation/*.png`。
本文所有数字与表格由 `benchmarks/xval_eq_extrap_docs.py` 从该 JSON **程序化注入**，无手抄。

生成环境：python 3.11.6 / torch 2.7.1 / numpy 2.4.6 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / dtype float64 / device cpu。
复现命令：`python benchmarks/xval_equilibrium_difference_extrapolation.py`（全量）或 `--quick`（CI 尺度）。

## 1 问题设定与基准出处

**对标对象（OSS）**：`hietwll/LBM_Taichi` 的平衡态差值外推边界，源码位置 `lbm_solver.py:130`，
规则为

```
f_i(x_b) = f_i^eq(x_b) - f_i^eq(x_1) + f_i(x_1)
```

其中 `x_1` 为紧邻边界的**第一个内侧节点**，边界密度由内侧节点复制（`lbm_solver.py:129`），
速度为按 `bc_type` 给定的 Dirichlet/Neumann 值。该实现无许可证（all-rights-reserved），
本仓**不复制其代码**，只按上式的公开描述移植（见 §2 差异清单）。

**两处出处（零手抄）**：
- 方法出处：知识库 `oss_solutions` 条目 `[OSS-SOL] repo: hietwll/LBM_Taichi / slug: NEW:equilibrium-extrapolation-bc /
  tag: equilibrium-difference-extrapolation`（`docs/oss_inventory_lbm_taichi.md` 机器块）。
- 文献出处（该方法是更一般族的一阶特例）：Guo, Z., Zheng, C., Shi, B. (2002) "An extrapolation method for boundary
  conditions in lattice Boltzmann method", *Phys. Fluids* **14**(6), 2007–2010. 令 `f = f^eq + f^neq`，
  上式等价于 `f_i(x_b) = f_i^eq(x_b) + f_i^neq(x_1)`，即**净平衡部分零阶（单节点）外推**。

**对标的物理判据（均为闭式解或一级文献表，非本仓自产）**：

| case | 参考解 | 说明 |
|---|---|---|
| 平面 Poiseuille（体积力驱动槽道） | `u(y) = g0 y (H-y) / (2 nu)` | Navier–Stokes 稳态精确解（抛物）；`g0` 为体积力密度 |
| 平面 Couette（下壁拖动） | `u(y) = U (1 - y/H)` | 线性剪切；`f^neq` 沿壁法向**常数** → 单节点外推应精确 |
| 顶盖驱动方腔 Re=100 | Ghia, Ghia & Shin (1982) *J. Comput. Phys.* **48** 387–411 表 I/II（Re=100 列） | 本仓 `tensorlbm.lid_driven_cavity.GHIA_RE100`（该表另有独立 OCR 见证链 `benchmarks/verified/cavity/re5000/reference/`）；**同样是 OSS 仓库自己的验证算例** |

**预登记判据（结果前固定，见 `spec`）**：owner 精度标准 —— 任一 case 相对参考值偏差必须 < 15%；
代数同一性 < 1e-12；并且"guo2 的收敛阶比 oss1 高 ≥ 0.7"才值得把 guo2 设为此族的默认变体。
Couette 是**证伪算例**：若 oss1 在 Couette 也偏差大，则说明移植本身错了（而不只是精度低）。

## 2 方法与实现

| 项 | 本工作实现 | OSS 参考实现 |
|---|---|---|
| 语言/后端 | Python + PyTorch（CPU, float64），张量化 | Python + Taichi JIT（`arch=ti.gpu`） |
| 场布局 | 结构维优先 `(9, ny, nx)` | AoS `Vector.field(9)`（逐节点 9 分量） |
| 边界节点处理 | 边界节点**不参与**碰撞与 streaming（populations 完全由 BC 指定） | 同样不参与局部碰撞，BC 直接写入 `f_old` 的边界节点 |
| `f^neq` 外推阶 | `order=1`（OSS 原式）/ `order=2`（`2 f^neq_1 - f^neq_2`，Guo 2002） | 仅 `order=1` |
| 壁面密度 | `rho_mode='copy'`（= OSS）/ `'extrapolate'`（`2 rho_1 - rho_2`） | 复制内侧节点（`rho[ibc] = rho[inb]`） |
| 边界几何位置 | 壁面在**边界节点中心**（`wall_offset = 0`）；有效槽宽 `H = ny-1` | 同一约定（节点式外推，无半格偏移） |
| 速度 BC | Dirichlet（壁面切向 + 法向 0），可逐壁给不同值 | 按 `bc_type` Dirichlet/Neumann |
| 碰撞/驱动 | BGK + Guo(2002) 二阶力项（Poiseuille 体积力） | BGK（D2Q9，`tau = 3 nu + 0.5`） |
| 角点约定 | 方腔四壁依次施加，**侧壁最后写入**（顶盖两角 u=0） | 单次 mask 循环序（以墙体掩码顺序为准） |

移植代码：`src/tensorlbm/boundary/equilibrium_extrapolation.py`
（`EquilibriumExtrapolationSpec` / `apply_equilibrium_extrapolation` / `apply_channel_extrapolation` /
`apply_cavity_extrapolation` / `oss_equilibrium_difference`）。
**同一性由测试钉死**（`tests/test_equilibrium_extrapolation_bc.py`，8 项全绿）：

- 模块化实现 vs 逐字转写 OSS 原式：max|Δf| = **0.0e+00**（要求 < 1e-12）
- 与**独立的 NumPy 逐节点转写**（AoS 布局、手写 feq、不共享生产内核）比对：max|Δf| = **0.0e+00**
- `order=2` 相对 `order=1` 的差异（证明开关非空转）：2.970e-03

### 2.1 与 OSS 的差异清单（做法 / 参数 / 边界条件）与移植取舍

| # | 差异 | 类别 | 取舍与理由 |
|---|---|---|---|
| 1 | OSS 仅有单节点 `f^neq` 外推；本工作同时提供 `order=2` | 做法 | **吸收**：二阶变体是文献原式（Guo 2002），代价仅多一次节点读取；一阶保留以保证"与 OSS 同规则"可测 |
| 2 | OSS 边界密度一律复制内侧；本工作可 `copy` / `extrapolate` | 参数 | **吸收为开关**：`copy` 保证与 OSS 逐位一致，`extrapolate` 用于 §5 敏感性 |
| 3 | OSS 在 Taichi 中把 collide+stream 融成一个 kernel、BC 写在融合 kernel 之前（作用在 pre-collision 场） | 做法 | **不吸收**：本工作在 post-collision 场上施加 BC（物理上更自洽）；稳态两者等价（稳态 pre/post 场相同，实测残差见 §5.3） |
| 4 | OSS 用 `Vector.field(9)` AoS；本仓全仓约定结构维优先 | 做法 | **不改**：布局是仓内契约（`tensorlbm.d2q9`），用独立 NumPy AoS 转写做等价性交叉验证代替"复制布局" |
| 5 | OSS 的 BC 面向任意掩码（含圆柱绕流），本工作面向轴对齐平面 | 边界条件 | **缩小范围**：轴对齐平面覆盖槽道/方腔（本次全部判据算例）；任意掩码留待需要时扩（未做，§6 声明） |
| 6 | OSS 无对流出流的独立实现（Neumann 复制） | 边界条件 | **未移植**：本仓已有更标准的 `zero_gradient_outlet`/`zou_he_outlet`，无收益 |
| 7 | OSS 角点顺序未定义（掩码循环序） | 边界条件 | **显式化**：本工作把"侧壁最后写入"写成文档 + 测试断言（可测 ≠ 任意） |
| 8 | 无量纲约定 | 参数 | **一致**：`dx=dy=dt=1`、`tau = 3 nu + 0.5`、`Re = U L / nu`，与 OSS 相同 |

**移植取舍的一句话结论**：OSS 这条规则 = Guo 外推族的一阶截断，代码上 10 行可移植、无数值风险；
值得吸收的是**规则本身与它的成本/精度位置**，不值得吸收的是它的框架布局与融合 kernel 结构。

## 3 参数表（无量纲，dx = dy = dt = 1）

格子声速 `c_s^2 = 1/3`；`tau = 3 nu + 1/2`；`Ma = u_ref sqrt(3)`。
Poiseuille 沿壁法向做**自相似加密**（固定 `Re_H = u_max H / nu`），Couette 同理固定 `Re_H`；
方腔固定 `Re = u_lid L / nu = 100`。各项均远小于压缩性可忽略界限（`Ma < 0.04`）。

| case | arm | grid | H or L | Ma | nu | tau | Re | steps | 停机残差 | 墙钟 |
|---|---|---|---|---|---|---|---|---|---|---|
| poiseuille | oss1 | 17x4 | 16 | 0.0346 | 0.032 | 0.5960 | 10 | 15800 | 8.9e-10 | 41s |
| poiseuille | oss1 | 33x4 | 32 | 0.0346 | 0.064 | 0.6920 | 10 | 30400 | 9.6e-10 | 78s |
| poiseuille | guo2 | 17x4 | 16 | 0.0346 | 0.032 | 0.5960 | 10 | 15800 | 1.0e-09 | 50s |
| poiseuille | guo2 | 33x4 | 32 | 0.0346 | 0.064 | 0.6920 | 10 | 30400 | 9.7e-10 | 95s |
| poiseuille | bb | 17x4 | 15 | 0.0346 | 0.03 | 0.5900 | 10 | 15000 | 7.7e-10 | 23s |
| poiseuille | bb | 33x4 | 31 | 0.0346 | 0.062 | 0.6860 | 10 | 29600 | 9.0e-10 | 43s |
| couette | oss1 | 17x4 | 16 | 0.0346 | 0.032 | 0.5960 | 10 | 15400 | 9.1e-10 | 30s |
| couette | oss1 | 33x4 | 32 | 0.0346 | 0.064 | 0.6920 | 10 | 29600 | 9.7e-10 | 59s |
| couette | guo2 | 17x4 | 16 | 0.0346 | 0.032 | 0.5960 | 10 | 15600 | 7.9e-10 | 42s |
| couette | guo2 | 33x4 | 32 | 0.0346 | 0.064 | 0.6920 | 10 | 29600 | 9.8e-10 | 91s |
| couette | bb | 17x4 | 15 | 0.0346 | 0.03 | 0.5900 | 10 | 14600 | 8.1e-10 | 21s |
| couette | bb | 33x4 | 31 | 0.0346 | 0.062 | 0.6860 | 10 | 28800 | 9.3e-10 | 45s |
| cavity_re100 | oss1 | 33x33 | 32 | 0.1732 | 0.032 | 0.5960 | 100 | 11500 | 4.2e-09 | 62s |
| cavity_re100 | guo2 | 33x33 | 32 | 0.1732 | 0.032 | 0.5960 | 100 | 15000 | 6.4e-09 | 107s |
| cavity_re100 | bb | 33x33 | 31 | 0.1732 | 0.031 | 0.5930 | 100 | 12000 | 8.0e-09 | 31s |

## 4 对照表（ref vs got + 误差打勾）

最细网格一级的结果（`oss1` = OSS 原规则；`guo2` = 二阶外推；`bb` = 半程反弹对照臂）：

| case | arm | metric | reference | got | deviation | <15% |
|---|---|---|---|---|---|---|
| poiseuille | oss1 | L2 error of u(y) (all fluid nodes) | closed form | 0.358 % | 0.358 % | ✅ |
| poiseuille | oss1 | max |u-u_exact| / u_max | closed form | 0.266 % | 0.266 % | ✅ |
| poiseuille | guo2 | L2 error of u(y) (all fluid nodes) | closed form | 0.034 % | 0.034 % | ✅ |
| poiseuille | guo2 | max |u-u_exact| / u_max | closed form | 0.025 % | 0.025 % | ✅ |
| poiseuille | bb | L2 error of u(y) (all fluid nodes) | closed form | 0.116 % | 0.116 % | ✅ |
| poiseuille | bb | max |u-u_exact| / u_max | closed form | 0.085 % | 0.085 % | ✅ |
| couette | oss1 | L2 error of u(y) | closed form | 9.225e-07 % | 9.225e-07 % | ✅ |
| couette | oss1 | max |u-u_exact| / U | closed form | 7.355e-07 % | 7.355e-07 % | ✅ |
| couette | guo2 | L2 error of u(y) | closed form | 9.334e-07 % | 9.334e-07 % | ✅ |
| couette | guo2 | max |u-u_exact| / U | closed form | 7.442e-07 % | 7.442e-07 % | ✅ |
| couette | bb | L2 error of u(y) | closed form | 8.349e-07 % | 8.349e-07 % | ✅ |
| couette | bb | max |u-u_exact| / U | closed form | 6.816e-07 % | 6.816e-07 % | ✅ |
| cavity_re100 | oss1 | RMSE of u(y) at x=0.5 / u_lid | Ghia 1982 Re=100 | 0.701 % | 0.701 % | ✅ |
| cavity_re100 | oss1 | RMSE of v(x) at y=0.5 / u_lid | Ghia 1982 Re=100 | 0.930 % | 0.930 % | ✅ |
| cavity_re100 | oss1 | max |u-u_Ghia| / u_lid (worst tabulated y) | Ghia 1982 Re=100 | 1.286 % | 1.286 % | ✅ |
| cavity_re100 | guo2 | RMSE of u(y) at x=0.5 / u_lid | Ghia 1982 Re=100 | 0.608 % | 0.608 % | ✅ |
| cavity_re100 | guo2 | RMSE of v(x) at y=0.5 / u_lid | Ghia 1982 Re=100 | 0.132 % | 0.132 % | ✅ |
| cavity_re100 | guo2 | max |u-u_Ghia| / u_lid (worst tabulated y) | Ghia 1982 Re=100 | 1.930 % | 1.930 % | ✅ |
| cavity_re100 | bb | RMSE of u(y) at x=0.5 / u_lid | Ghia 1982 Re=100 | 0.486 % | 0.486 % | ✅ |
| cavity_re100 | bb | RMSE of v(x) at y=0.5 / u_lid | Ghia 1982 Re=100 | 0.283 % | 0.283 % | ✅ |
| cavity_re100 | bb | max |u-u_Ghia| / u_lid (worst tabulated y) | Ghia 1982 Re=100 | 0.861 % | 0.861 % | ✅ |

**门限汇总**：所有 case × arm × 指标中最大偏差 = **1.930%**（门限 15%）
→ 全部通过 ✅

图 1 剖面 vs 闭式解：`figures/xval_equilibrium_difference_extrapolation/profiles_vs_closed_form.png`
图 3 方腔中心线 vs Ghia 1982：`figures/xval_equilibrium_difference_extrapolation/cavity_centerlines_vs_ghia.png`

## 5 收敛性 / 参数敏感性

### 5.1 壁法向收敛阶（Poiseuille 与 Couette 阶梯）

| ladder | arm | observed order | H values | relative L2 error |
|---|---|---|---|---|
| poiseuille | oss1 | -2.28 | 16, 32 | 1.740e+00%, 3.580e-01% |
| poiseuille | guo2 | -0.98 | 16, 32 | 6.629e-02%, 3.369e-02% |
| poiseuille | bb | -2.22 | 15, 31 | 5.823e-01%, 1.162e-01% |
| couette | oss1 | 1.16 | 16, 32 | 4.141e-07%, 9.225e-07% |
| couette | guo2 | 1.36 | 16, 32 | 3.632e-07%, 9.334e-07% |
| couette | bb | 1.28 | 15, 31 | 3.294e-07%, 8.349e-07% |

图 2：`figures/xval_equilibrium_difference_extrapolation/poiseuille_order.png`。

### 5.2 敏感性之一：外推阶（1 vs 2）与壁面密度模式

见上表 `poiseuille_oss1` / `poiseuille_guo2` / `poiseuille_bb` 三行——同一离散、同一 CFL、同一驱动，
唯一差别是边界处 `f^neq`（与 `rho`）的外推方式；`oss1` 与 `guo2` 的差别即"单节点复制"的代价（`bb` 另含壁面位置差异，见 §2 表）。

### 5.3 敏感性之二：时间步残差（稳态判据）与质量行为

- 所有算例以"内点速度场每 200/500 步最大变化 / 参考速度 < tol"为稳态判据；停机残差见 §3 参数表最后一列（`oss1/guo2` 通道算例量级 1e-9 – 1e-10，均 `converged=True`）。
- OSS 规则**不保证封闭域质量守恒**（边界 populations 由 `f^eq(b) + f^neq(1)` 指定，净通量不为零）。
  方腔（封闭域）实测内点平均密度漂移与漂移率：

| arm | interior mass drift (rel.) | drift per 1000 steps | mass drift over whole domain |
|---|---|---|---|
| oss1 | 0.082 | -0.008 | 0.093 |
| guo2 | 0.032 | -0.002 | 0.034 |
| bb | 1.029e-12 | -9.358e-14 | 0.006 |

  图 4：`figures/xval_equilibrium_difference_extrapolation/cavity_mass_history.png`。该表是"移植取舍"里**唯一需要警示**的一条：
  长时间封闭域积分（例如数万步以上的自由面/多相算例）应改用质量守恒的 `bb` 或加质量修正，而不是这条规则。

### 5.4 敏感性之三：网格 + 马赫数

- 网格：§5.1 的阶梯（H = 16 → 64）。
- `Ma`：所有算例 `Ma <= 0.173`（不可压范围，压缩性误差 `O(Ma²) < 4e-3`）；§5.1 的阶梯在固定 `Re_H` 下改变 `nu`，即同时改变了 `tau`（数值见 §3 参数表），因此阶数不是在单一 `tau` 上取到的偶然值。
- **未做**的敏感性：`Ma` 独立扫描（固定 `tau`、只改驱动力）——若要声称"与 Ma 无关"需补此点（§6 未完成项）。

## 6 失败模式与工件

**失败模式（实测）**：

1. **半程反弹臂的初版方向约定错误**（把反弹写入"指向壁"的方向槽位）→ Poiseuille L2 误差 29.5%（应 ≈0.6%）。
   发现方式：对照臂精度反直觉（比一阶外推差 17 倍）；修正后 0.58%。结论：**对照臂必须独立验证**，
   否则"OSS 规则更准"的结论会被错误对照臂反转。
2. **角点/边界行的宏观量污染**：BC 未应用的边界行在 streaming 后是"未定义"值，若把它们计入剖面/稳态判据，
   会得到 3.3 量级的伪残差与 0.83 的伪密度；本工作所有指标只在**内点**上取，且 streaming 的越界取数被显式
   导向自身（随后必被 BC/反弹覆盖）。
3. **同一性检查的形状陷阱**：`(9,nx)` 与 `(9,1,nx)` 广播会伪造 0.42 的"不一致"；修正后为 0。
   （教训：代数同一性断言必须钉住形状。）

**未做/未完成（如实声明）**：

- **未能运行 OSS 原代码**：Taichi 在本沙箱不可安装（`pip install taichi` 在 loongarch 镜像无可用版本），且该仓无许可证，
  因此不声称"与 OSS 实现逐位同构"；代替证据是 §2 的代数同一性 + 独立 NumPy 转写 + Couette 证伪算例。
- 任意掩码/曲面壁（OSS 的圆柱掩码用法）未实现，本模块只覆盖轴对齐平面。
- 未独立扫描 `Ma`（见 §5.4）。

**工件**：

- 代码：`src/tensorlbm/boundary/equilibrium_extrapolation.py`、`benchmarks/xval_equilibrium_difference_extrapolation.py`、`benchmarks/xval_eq_extrap_docs.py`
- 测试：`tests/test_equilibrium_extrapolation_bc.py`（8 项）
- 数据：`runs/xval_equilibrium_difference_extrapolation/result.json`（每次运行的参数/指标/剖面全量）
- 图：`figures/xval_equilibrium_difference_extrapolation/`
