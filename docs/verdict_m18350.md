# m18350 交差审计报告：m18309 XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING

## ① 案例介绍

**物理背景**：Shan-Chen 单组分多相（SCMP）赝势模型，D2Q9 格子，指数型赝势 ψ(ρ)=ρ₀(1−e^(−ρ/ρ₀))。核心问题是**可插拔 forcing 轴**——同一相互作用力可通过三种数值方案注入碰撞步：
- `velocity_shift`（历史方案，u_eq = u + τF/ρ）
- `edm`（精确差分法，Kupershtekhov 2004）
- `guo`（Guo 2002 源项法，二阶精度）

**几何/参数**：2D 液滴（tanh 初始场），R = 10/14/18 格子单位，L = 128 周期盒子，G = 5.0，τ = 0.8/1.0/1.2，24000 步。

**锚点**：
- A1: Maxwell 等面积构造（解析，ρ_l=1.92887, ρ_v=0.15289）
- A2: Young-Laplace 定律 dp = σ/R（线性拟合 R²）
- A3: 动量注入恒等式 Σcᵢ(f_out − f_BGK) = F（机器精度）
- A4: 方案独立性（静态解不应依赖 forcing 方案）

## ② 方法

**双臂交叉验证**：
- `repo` 臂：TensorLBM 引擎（torch，float32/float64），`collide_sc_single_component(forcing=...)`
- `ref` 臂：独立 numpy 转录（`ref_sc_numpy.py`），与 repo 无共享代码

**关键修复**：`guo_force_term` 和 `collide_sc_single_component` 的 guo 分支中，将 `.float()` 强转改为 dtype-aware 的 `_w_for(f.dtype, device)`，使 float64 工况下的动量恒等式误差从 ~1e-9 降至 ≤1.5e-16。

**验收门**：20 项 pytest 测试（动量恒等式、Maxwell 锚、cross-arm 一致性、artifact gate）。

## ③ 计算结果

| 判据 | 测量值 | 容差 |  verdict |
|---|---|---|---|
| A3 动量恒等式 (repo float64) | ≤1.5e-16 | 机器精度 | ✓ |
| A3 动量恒等式 (ref numpy) | ≤1.5e-13 | 机器精度 | ✓ |
| A2 Laplace 线性 (repo:guo) | R²=1.00000, σ spread=2.02% | R²>0.99, spread<15% | ✓ |
| A2 Laplace 线性 (repo:edm) | R²=0.99811, σ spread=6.52% | R²>0.99, spread<15% | ✓ |
| cross-arm Δp | 0.02-0.03% | <15% | ✓ |
| A1 液相密度 vs Maxwell | 0.24-3.21% | <15% | ✓ |
| **A1 汽相密度 vs Maxwell** | **4.8-27.4% (guo 族 17-19%)** | **<15%** | **✗** |
| **A4 方案间 σ 独立性** | **guo=0.0311 vs edm=0.0485 = 56%** | **<15%** | **✗** |

## ④ 结果图

结果图 `xval_pp_guo_forcing.png`（153910 bytes）由 `make_figure.py` 生成，包含：
- 液滴密度场云图（repo vs ref 臂对比）
- Laplace 线性拟合图（dp vs 1/R，三方案叠加）
- 动量恒等式误差分布

## ⑤ 注意点

1. **汽相偏差（主要 honest-FAIL）**：格子相互作用力对低密度分支有系统性偏差，ρ_v 测得 0.124-0.126 vs Maxwell 0.1529（−18%）。**任何来自此引擎的汽相量携带 ~17-19% 系统偏差**，未经校准不可用于气侧预测。

2. **方案依赖 σ**：guo σ = 0.0311 vs edm/shift σ = 0.0485（相同 G,ψ,τ=1 下差 56%）。假设（未证明）：Guo 源项消除了 velocity-shift 的 −(τ−½) 应力，但残余离散应力以不同方式进入界面压力平衡。需压力张量（Korteweg）测量——**在解决前，可插拔轴不能声称物理中性**。

3. **虚假流**：所有方案 max|u| ≈ 0.15（c_s 的 26%），null 流测试仅满足到 15% c_s。

4. **τ=1 时 EDM = velocity_shift**：代数上恒等（都将平衡速度偏移 F/ρ），已在 `test_edm_equals_shift_at_tau_one` 中逐位验证。

## ⑥ 误差判定

| 关键量 | 基准值 | 测量值 | 相对误差 | <15%? |
|---|---|---|---|---|
| ρ_l (Maxwell) | 1.92887 | 1.9132 (repo_guo_R14) | 0.81% | ✓ |
| ρ_v (Maxwell) | 0.15289 | 0.1252 (repo_guo_R14) | 18.14% | ✗ |
| σ (Laplace slope, repo:guo) | — | 0.03114 | — | — |
| Δp cross-arm (R14) | 0.0020625 (ref) | 0.0020620 (repo) | 0.023% | ✓ |
| 动量恒等式误差 | 0 | ≤1.5e-16 | — | ✓ |

**结论**：液相分支、Laplace 线性、cross-arm 一致性、动量恒等式均通过 15% 门；汽相密度偏差（17-19%）和方案间 σ 差异（56%）为 honest-FAIL，已如实记录并设 gate 防止静默回归。

## 判级：L4

- ✓ L0 文档成果：有（REPORT.md, summary.md）
- ✓ L1 数学/单元验证：有（Maxwell 解析锚、动量恒等式）
- ✓ L2 独立功能实现：有（pluggable forcing 轴，3 方案，真代码）
- ✓ L3 系统集成：有（接入 collide_sc_single_component，被 dam_break_sc 等调用）
- ✓ L4 基准验证：有（完整算例 + 误差分析 + 可复现，repro_cmd 可执行）
- ✗ L5 科研工程成果：不足（两项 honest-FAIL 未解决，需下一轮压力张量测量）
