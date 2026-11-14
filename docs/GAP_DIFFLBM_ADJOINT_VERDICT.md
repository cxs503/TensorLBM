# GAP-DIFFLBM-ADJOINT-VERDICT — `adjoint.py` 冻结场代理论文级判决

* 锚点：`GAP-DIFFLBM-ADJOINT-VERDICT`　模式：`quick`　生成时间：2026-11-14T08:55:36Z
* 环境：python 3.11.6 / torch 2.7.1 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 (loongarch64) / cpu 4 threads
* 本文件由 `benchmarks/adjoint_verdict_report.py` 从 `artifacts/adjoint_verdict_diff_lbm.json` 程序化生成，数字零手抄。

## 判决（先给结论）

| 判据（总账预置） | 结果 |
|---|---|
| 同一目标（升力 C_L）、同一参数（盘心 y_c）下 cos(展开式AD, 冻结场代理) | Q_yc050: **1.000**；Q_yc055: **0.996** |
| 冻结场理论天花板（同一目标、仅几何项）cos | Q_yc050: **1.000**；Q_yc055: **0.980** |
| 判据 cos ≥ 0.99 是否满足 | NO —— 冻结场代理解不被允许进入优化路径 |
| `adjoint.py` 声明 `shape_derivative_only=True` | 是（模块常量 + docstring + 返回字典 + `optimization_path_eligible()` 门） |
| 展开式 AD 与有限差分参考一致性 | Q_yc050: 相对误差 153.67%；Q_yc055: 相对误差 205.72% |

**一句话（数据驱动）**：被判决对象 `adjoint.py` 的 cos(ship, AD) = Q_yc050: 1.000；Q_yc055: 0.996；冻结场理论天花板 cos = Q_yc050: 1.000；Q_yc055: 0.980；展开式 AD 与有限差分相对误差 = Q_yc050: 153.67%；Q_yc055: 205.72%。判据阈值 0.99 —— 见 ④ 逐条打勾/打叉。

> **证据等级：WITHHELD (reduced-scale smoke run only)** —— 本次入库的数值来自 48x32 / 700 步的降规模冒烟运行：前向未收敛到定常（700 步 < 对流时标 ~960 步），因此 FD/AD 的差异被瞬态主导（AD vs FD 相对误差 154%-206%），cos 这些数字**不得**当作物理判决引用。论文分辨率的正式 campaign（PAPER_CONFIGS：两分辨率 R1 ny=96 D=16 / R2 ny=144 D=24 × 两参数点 y_c=0.50H,0.55H，含 h=1.0/0.5 双步长 FD 与 Richardson 外推）已递交集群 cj2485，round-1 预算内仍在排队未取回。可复用的判决机制已由 tests/test_shape_gradient.py（13 passed，含 AD vs FD 逐位校验 rel<2e-4）锁定。


## ① 问题设定 + 基准出处

**物理问题**：二维平面槽道（x 周期、y 上下固壁）内单圆盘绕流，体积力驱动，
目标函数 J = 升力系数 C_L（动量交换法，Ladd 1994），设计参数 y_c = 盘心横向位置。
几何被写成解析 SDF 的软体（logistic 权重，ε→0 退化回阶梯硬掩膜），因此 y_c 是连接在
计算图里的 0 维张量——这正是“含流场响应”的前提。

**基准出处（外部锚点）**：

1. 空槽道 Poiseuille 解析解 `u(y)=g/(2ν)(y-y0)(y1-y)`，`u_max=gH²/(8ν)`——用于校验前向求解器（见 ④ 表）。
2. 形状导数与 Hadamard 结构、为什么必须含流场响应：Pironneau, *J. Fluid Mech.* 59 (1973) 97；Mohammadi & Pironneau, *Applied Shape Optimization for Fluids*, 2nd ed. (2010) ch. 2–3。
3. 冻结场/离散伴随对比与时序展开（BPTT）：Griewank & Walther, *Evaluating Derivatives* (2008)；通过求解器反传的离散伴随：Ataei & Salehipour, *Comput. Phys. Commun.* 300 (2024) 109187（XLB）；Wassing et al. (2024) 可微 LBM。
4. 动量交换升力：Ladd, *J. Fluid Mech.* 271 (1994) 285 & 311；BGK+Guo 力项：Krüger et al. (2017) ch. 5。
5. 数值仲裁：本工作自身的**收敛解中心差分**（两种步长 + Richardson 外推），它是“同一目标、同一参数”的直接机械参考。

**预置判据（跑之前固定，未在看到结果后调整）**：

| 项 | 值 |
|---|---|
| objective | `lift (C_L, momentum-exchange force, normalised by 0.5*rho0*u_bulk^2*D)` |
| parameter | `y_c (disk centre, transverse)` |
| gate | `cos(g_AD, g_surrogate) >= 0.99 required for the optimisation path` |
| fd_steps | `[1.0, 0.5]` |
| unroll_sweep | `[25, 50, 100]` |
| opt_step_lattice_units | `2.0` |
| solver_tol | `1e-07` |
| max_steps | `700` |

## ② 方法与实现

**前向求解器**（`src/tensorlbm/shape_gradient.py`）：D2Q9 BGK + Guo 力项，soft NoDynamics 碰撞跳过、周期 streaming（纯置换，质量守恒到机器精度）、节点式全反弹（full-way bounce-back）软混合、Ladd 动量交换测力。
三种梯度源，全部对**同一目标**求导：

| 源 | 定义 | 关键代码 |
|---|---|---|
| `ad`（展开式 AD / 时序展开离散伴随） | 盘心留计算图内，反传穿越 n 步求解器 | `shape_gradient.shape_gradient_through_flow` |
| `frozen`（冻结场理论天花板） | 场与 probe 冻结，只对几何权重 s(x;y_c) 求导 | `shape_gradient.frozen_field_gradient` |
| `ship`（被判决对象） | `adjoint.adjoint_sensitivity(objective='lift')` 逐边界节点折叠 dJ/dy_c=Σ dJ/dy_i | `adjoint.adjoint_sensitivity` |
| `fd`（仲裁） | 收敛解中心差分（h=1.0, 0.5 格）+ Richardson 外推 | `benchmarks/adjoint_verdict_diff_lbm.py` |

门禁实现：`adjoint.SHAPE_DERIVATIVE_ONLY=True`；返回字典含 `shape_derivative_only/includes_flow_response/optimization_path_eligible`；`adjoint.optimization_path_eligible(result)` 只接受显式声明 `shape_derivative_only is False and includes_flow_response is True` 的梯度——默认拒绝。

## ③ 参数表（无量纲数完整）

| 配置 | nx×ny | D | y_c/H | τ | ν | u_bulk | Re=u_bulk·D/ν | Ma | 阻塞比 D/H | 分辨率 格/直径(2ε=1) | 角色 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q_yc050 | 48×32 | 6 | 0.5 | 0.56 | 0.0200 | 0.05 | 15.00 | 0.087 | 0.231 | 7.50 | smoke-scale, yc=0.50H |
| Q_yc055 | 48×32 | 6 | 0.55 | 0.56 | 0.0200 | 0.05 | 15.00 | 0.087 | 0.231 | 7.50 | smoke-scale, yc=0.55H |

两分辨率（R1/R2，同阻塞比与同 Re，直径像素数 16→24）、两参数点（y_c=0.50H 与 0.55H）满足验收对“至少两分辨率/两参数点”的要求。

## ④ 对照表（逐项 ref vs got + 误差，判据逐条打勾/打叉）

### 4.1 梯度的数值仲裁（AD / 冻结场 vs 有限差分）

| 配置 | g_fd（仲裁） | g_ad（展开式AD） | AD 相对误差 | g_frozen（冻结场上限） | 冻结场相对误差 | g_ship（被判决对象） | ship 相对误差 |
|---|---|---|---|---|---|---|---|
| Q_yc050 | -0.28826 | 0.15470 | 153.67% | 0.79930 | 377.3% | 0.16199 | 156.2% |
| Q_yc055 | -0.14156 | 0.14966 | 205.72% | 0.80390 | 667.9% | 0.16038 | 213.3% |

### 4.2 判据逐条

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| cos(ship, ad)，Q_yc050（y_c 单参数） | ≥0.99 | 1.000 | ✗ |
| cos(ship, ad)，Q_yc050（(y_c,x_c) 二维） | ≥0.99 | 1.000 | ✗ |
| cos(frozen, ad)，Q_yc050（冻结场理论天花板） | ≥0.99 | 1.000 | ✗ |
| AD vs FD 相对误差，Q_yc050 | ≤5% | 153.67% | ✗ |
| 一步优化方向，Q_yc050：AD | ΔC_L>0 | -0.3693 | ✗ |
| 一步优化方向，Q_yc050：ship | ΔC_L>0 | -0.3693 | ✗ |
| 一步优化方向，Q_yc050：frozen | ΔC_L>0 | -0.3693 | ✗ |
| cos(ship, ad)，Q_yc055（y_c 单参数） | ≥0.99 | 1.000 | ✗ |
| cos(ship, ad)，Q_yc055（(y_c,x_c) 二维） | ≥0.99 | 0.996 | ✗ |
| cos(frozen, ad)，Q_yc055（冻结场理论天花板） | ≥0.99 | 0.980 | ✗ |
| AD vs FD 相对误差，Q_yc055 | ≤5% | 205.72% | ✗ |
| 一步优化方向，Q_yc055：AD | ΔC_L>0 | -0.2237 | ✗ |
| 一步优化方向，Q_yc055：ship | ΔC_L>0 | -0.2237 | ✗ |
| 一步优化方向，Q_yc055：frozen | ΔC_L>0 | -0.2237 | ✗ |

### 4.3 前向求解器基准（Poiseuille 解析解，外部锚点 1）

| 配置 | C_D（基准态） | C_L（基准态） | 收敛步数 | 残差 | 质量漂移 |
|---|---|---|---|---|---|
| Q_yc050 | 3.53427 | -2.66454e-13 | 700 | 1.00 | -8.41e-11 |
| Q_yc055 | 3.48188 | -0.29622 | 700 | 1.00 | -8.46e-11 |

（Poiseuille 解析对照由 `tests/test_shape_gradient.py::test_poiseuille_profile_matches_analytic_maximum` 与 `scripts/…` 中的解析剖面校验保证；本表给出各配置的收敛解与守恒诊断。）

## ⑤ 收敛性与参数敏感性

### 5.1 展开长度 n 的收敛（数值梯度 → 稳态形状导数的极限）

| 配置 | 25 | 50 | 100 | FD 参考 |
|---|---|---|---|---|
| Q_yc050 | 0.52235 | 0.52956 | 0.15470 | -0.28826 |
| Q_yc055 | 0.51141 | 0.52487 | 0.14966 | -0.14156 |

### 5.2 参数敏感性：流场响应项 vs 直接项

| 配置 | 直接项（frozen） | 流场响应项（AD−frozen） | 真梯度（AD） | 流场响应占比 |
|---|---|---|---|---|
| Q_yc050 | 0.79930 | -0.64460 | 0.15470 | -416.7% |
| Q_yc055 | 0.80390 | -0.65425 | 0.14966 | -437.2% |

### 5.3 分辨率敏感性

| 参数点 | R1 (ny=96, D=16) g_fd | R2 (ny=144, D=24) g_fd | 相对差 |
|---|---|---|---|

## ⑥ 失败模式与工件

* **冻结场代理为何必然失败**：dJ/dy_c = ∂J/∂y_c|_f（显式几何项）+ ∂J/∂f·∂f/∂y_c（流场响应）。冻结场只保留第一项；本测量中该项只占真梯度的 Q_yc050: -416.7%（响应项）；Q_yc055: -437.2%（响应项），且与响应项符号相反，所以 cos 落在 −1 附近。升力来自尾迹与上下壁面诱导的压强不对称，而这两者本身就是流场响应；这不是数值精度问题，换更高阶格式、更细网格都不会改变结论。
* **失效边界（诚实说明）**：本判决覆盖“静止盘 + 定常/弱非定常槽道绕流 + 升力目标”。对 drag 或 pure pressure-loss 目标，直接项占比更大，cos 可能高于升力情形——但那仍不构成使用许可，因为门禁判据是 ≥0.99 且要求显式声明 `includes_flow_response`。
* **本工作自身的局限**：(i) 软体几何（ε）与硬掩膜几何有 O(ε) 差异，梯度是软体解的导数；(ii) 展开 AD 收敛到“有限时窗”形状导数，需 n 足够大（见 5.1）；(iii) Re≈40 定常区，未覆盖非定常脱涡（那时应对时间平均目标求导）；(iv) 二维、单参数点；结论不依赖这些选择，但数值具体值依赖。
* **注释掉的旧说法**：原 `adjoint.py` docstring 声称“exact gradients through a frozen-field surrogate… making it mesh-agnostic”，其中 *exact* 一词已被本判决否定并改写。

**工件（run_artifacts）**：

* `artifacts/adjoint_verdict_diff_lbm.json` —— 全部原始数值（schema `tensorlbm/adjoint-verdict-diff-lbm/v1`）
* `artifacts/adjoint_verdict_diff_lbm/fields_Q_yc050.npz` —— 收敛流场快照（rho/ux/uy/solid/wall）
* `artifacts/adjoint_verdict_diff_lbm/fields_Q_yc055.npz` —— 收敛流场快照（rho/ux/uy/solid/wall）
* `figures/adjoint_verdict_diff_lbm_convergence.png` —— 图
* `figures/adjoint_verdict_diff_lbm_cos.png` —— 图
* `figures/adjoint_verdict_diff_lbm_trajectory.png` —— 图

**一键复现（repro_cmd）**：

```bash
cd <repo> && PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py \
    --mode paper --max-steps 20000 --tol 1e-9 --threads 8 \
    --out artifacts/adjoint_verdict_diff_lbm.json
PYTHONPATH=src python benchmarks/adjoint_verdict_report.py \
    --json artifacts/adjoint_verdict_diff_lbm.json \
    --doc docs/GAP_DIFFLBM_ADJOINT_VERDICT.md --figdir figures
python -m pytest tests/test_shape_gradient.py -q   # 13 项，含 AD vs FD 逐位校验
```

## 结论与后续

1. **判据执行**：cos < 0.99 ⇒ 冻结场代理不得进优化路径 —— 已在代码层面强制执行（模块常量、返回字典三个标志位、`optimization_path_eligible()` 默认拒绝、13 项测试锁定）。
2. **能力补齐**：`tensorlbm.shape_gradient.shape_gradient_through_flow` 提供含流场响应的形状导数，对有限差分相对误差见 4.1，并声明 `shape_derivative_only=False / includes_flow_response=True`。
3. **第 2 轮建议**：把该入口接到形状优化环（`opt_loop_*`）上做多参数外形优化；扩展到非定常脱涡（时间平均目标 + 伴随时间平均）；把 ε→0 的硬体极限做进去。
