# XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION — 进展报告（返修轮 1 / m18141）

对标对象：**joconnor22/LIFE @ `2af2428`** 的 IBM epsilon-Gram 隐式修正
（`src/Objects.cpp:235-320 computeEpsilon`、`src/IBMNode.cpp:73-74 spread`、
`inc/Utils.h:220-231 diracDelta`）。
判据（owner）：对标误差 < 15%；本单另有 predeclared 的恒等判据（见 §5）。
证据等级：**REAL**（全部数字由本仓 `run_xval.py` 现跑产生，`results.json`/`benchmark.json` 可追溯）。

---

## 1. 问题、参考出处与方法

**问题**：LIFE 在 IBM 力传递里用“Gram 系统 `A ε = 1`”给每个拉格朗日标记求一个乘子 ε，
spread 权重变为 `f·ε·ds·δ`。要回答：把这套做法移植进 TensorLBM 后，(a) 是否逐语义复刻，
(b) 相对“同核无修正”控制组买到什么、付出什么，(c) 物理算例是否落在文献带。

**参考出处（零手抄；逐行引用 OSS 文件:行）**

| LIFE 语义 | OSS 位置 | 本仓实现 |
|---|---|---|
| 3 点正则 Dirac δ，支撑 \|r\|≤1.5 | `inc/Utils.h:220-231` | `life_delta_3pt` |
| 支撑构建：`inear=round(x)`，5×5 窗口，保留 \|dx\|,\|dy\|<1.5，**不周期回卷** | `src/IBMNode.cpp:140-180` | `build_support_2d` |
| 标记间距 `ds` = 到同体其它标记最小距离 | `src/IBMNode.cpp:186-205` | `compute_ds_2d` |
| spread：`F += f·ε·ds·δ` | `src/IBMNode.cpp:73-74` | `spread_force_2d` / `spread_with_correction` |
| Gram：`A[i,j]=Σ_s w_i(s)δ(|x_j−x_i(s)|)δ(|y_j−y_i(s)|)ds_j`，解 `Aε=1` | `src/Objects.cpp:235-320` | `compute_epsilon_gram` |
| 刚性体仅在 t=0 重算 ε | `src/Objects.cpp:271-272` | 本单每个算例只解一次 |

**本仓实现面（L2 入口 + L3 集成）**

* `src/tensorlbm/ibm_epsilon_gram.py`：δ/支撑/ds/Gram/循环与向量化 spread-interp，
  以及 **集成面** `make_life_ibm_stencil()`（返回仓内 `IBMStencil`，`kernel="life3"`）+
  `spread_with_correction()`（走 `IBMStencil.spread`，即主求解器力传递路径）+
  `life_correction()` / `life_epsilon_gram_diagnostics()` / `selftest()`（`python -m` 入口）。
* 入口：`python -m tensorlbm.ibm_epsilon_gram`（自检）；`xval/<anchor>/run_xval.py`（XVAL 主入口）。
* 集成调用方：`xval/<anchor>/run_xval.py::cylinder_drag` 用 `d2q9.equilibrium/macroscopic`+
  `solver.collide_bgk/stream`+`ibm_force_transfer.IBMStencil/body_source_2d` 跑真实 LBM 时推进；
  `tests/test_ibm_epsilon_gram_solver_smoke.py` 用同一路径做 12 步烟测。

## 2. 参数表（无量纲，全部 lattice units）

| 量 | 值 | 说明 |
|---|---|---|
| δ 核 | LIFE 3pt，支撑 1.5 | A1/A2 相同（唯一变量是 ε） |
| 标记环 | R=10，中心 (32,32)，N=16 / 32 | ds=3.902 / 1.960 |
| 网格（力传递） | 64×64 | 非周期支撑裁剪，无回卷 |
| 圆柱算例 | nx×ny=128×64，D=8，u=0.1，τ=0.524 | ν=(τ−0.5)/3=0.008 |
| Re | uD/ν = **100** | 预声明值 |
| 阻塞比 | D/ny = **0.125** | 周期盒，无壁面 |
| 标记间距 | 0.8，M=31 | 圆柱周长 πD≈25.1 |
| 时间步 | 700（前 500 步瞬态，平均 200 步） | predeclared |
| 精度 | float64，loongarch64 CPU | 见 `results.json.provenance` |

## 3. 做法

1. 按 OSS 行号逐条移植 5 个语义（§1 表）；循环版与向量化版并存，互为交叉验证。
2. **控制组设计**：A1（ε 解 Gram）/ A2（同核、同支撑、同 ds，ε≡1）——唯一变量是 ε；
   A3 = 仓内既有 `build_stencil(kernel="4pt")` 基线（差异清单用）。
3. 集成：把 LIFE 支撑装进仓内 `IBMStencil`，spread 用 `f·ε·ds` 预乘后调用 `IBMStencil.spread`，
   interpolate 保持 OSS 的“无权重插值”语义。
4. 力传递（Part A）：spread→interpolate 往返、沉积总力记账。
5. 物理（Part B）：固定圆柱直接力 IBM，`u_int → 2 次 Guo 修正 → spread(+ε ds) → body_source_2d → stream`。
6. 工件：`results.json`（含 provenance/耗时/逐臂原始 drag_history）、`benchmark.json`（ref vs got 表）、
   `fig1..fig4.png`、`figure_captions.json`、`run.log`。

## 4. 结果（ref vs got，程序化注入，无手抄）

**4.1 解析恒等类（ref 为精确数学值）**

| KPI | ref | got (A1) | 误差 | got (A2 控制) | 判定 |
|---|---|---|---|---|---|
| 1-D δ 分割单位（7 个分数偏移的最大偏差） | 0 | 1.11e-16 | 1.1e-16 | — | ✅ |
| 2-D 支撑权重和−1（格点对齐） | 0 | 0.0 | 0 | — | ✅ |
| spread→interp 往返 L2（N=16 常量场） | 0 | **1.52e-16** | 1.5e-16 | 2.455e-2 | ✅ / 控制组暴露缺陷 |
| spread→interp 往返 L2（N=32 常量场） | 0 | **1.56e-16** | 1.6e-16 | 4.600e-1 | ✅ |
| spread→interp 往返 L2（N=32 径向场） | 0 | **2.188e-3** | 2.2e-3 | 4.610e-1 | ✅（<1e-2） |
| `IBMStencil` spread vs 向量化 | 0 | 0.0 | 0 | — | ✅ 逐位一致 |
| `IBMStencil` interpolate vs 向量化 | 0 | 2.22e-16 | 2.2e-16 | — | ✅ |

修正把往返误差从 2.5e-2…4.6e-1 压到 ≤2.2e-3（**改善 210×–14 个数量级**），
在 N=16 非重叠支撑下达到机器零。

**4.2 修正的代价（如实登记，不粉饰）**

| 布局 | ε 区间 | 沉积总力偏差 \|ΣF−Σf·ds\|/Σ\|f\|ds（常量场） |
|---|---|---|
| N=16（支撑不重叠） | 1.0252（均匀） | 2.52e-2 |
| N=32（支撑部分重叠） | 1.8116–1.8796 | **8.52e-1** |

原因可解析：`Aε=1` 保证的是“常量拉格朗日场经 spread→interp 复现”，非重叠支撑下即往返恒等
（ε·ds·Σw² = 1，故 ε = 1/Σw² > 1）；代价是沉积力矩量被放大 O(ε−1)。A2 则相反：
沉积总力精确（2.3e-16）但往返误差 46%。**两条性质不可兼得**——这是本次对标最关键的发现。

**4.3 物理算例（2D 圆柱，Re=100，同一预声明设置）**

| 臂 | C_D（200 步平均） | 后 100 步均值 | 瞬时 std | vs 无界文献带 1.30–1.40（中心 1.35） |
|---|---|---|---|---|
| A1 ε-Gram | **1.6156** | 1.6431 | 1.393 | +19.7% ❌（>15%） |
| A2 ε=1 | 1.6303 | 1.7417 | 1.280 | +20.8% ❌ |
| A3 仓内 4pt | 1.6817 | 1.8397 | 1.405 | +24.6% ❌ |

三臂互相差 ≤ 3.9%，说明**修正不改变收敛的物理量**（它在直接力迭代里被吸收），
差异来自不同力核的离散（A1 与 A2 差 0.9%）。与无界文献的 +20% 不是修正引入的，
但**也不是阻塞比能解释的**：低阻塞补充算例 B=0.0833（192×96）给出 C_D=1.6218，
与 B=0.125 的 1.6156 相差 0.4%（见 `blockage_sweep.json`，两点拟合 k=−0.09，无物理意义）。
⚠️ 因此 §4.3 的 15% 判据**判 FAIL（honest）**，归因假设“12.5% 阻塞比抬高 C_D”被本次补充算例
**证伪**；剩余候选原因（未逐项验证）：D=8 分辨率偏粗、200 步平均窗口仅 0.4 个脱落周期、
u=0.1 的可压缩误差 O(Ma²)≈3%。方法验收依据仍是 §4.1（<2.2e-3，远优于 15%）。

## 5. 收敛性 / 参数敏感性（≥2 点）

1. **标记间距 N=16→32**（ds 3.90→1.96）：A2 往返误差 2.5e-2→4.6e-1（核重叠恶化），
   A1 恒为 ≤2.2e-3；ε 从 1.025 → 1.81（支撑重叠使 Gram 修正量增大）。
2. **力场形状**（常量 (1,0) vs 径向 (cos,sin)）：N=16 两者均机器零；N=32 径向 2.2e-3
   （非均匀场下 Gram 只保常量复现，残差留在非均匀分量上）。
3. **力核**（A1/A2 LIFE-3pt vs A3 仓内 4pt）：C_D 1.616/1.630/1.682，离散路径间 ≤3.9%。
4. **阻塞比 B=0.125→0.0833**（128×64 → 192×96，同一 D/u/τ/核/修正）：C_D 1.6156→1.6218（+0.4%），
   **阻塞比不解释与无界文献带的 +20%**（原假设证伪，见 §4.3）。
5. **时间平均**（fig4）：后 100 步均值比全程均值高 1.7%（A1）——200 步窗口约 0.4 个脱落周期，
   故 C_D 报标准差 1.39，均值不确定度约 ±2%；这是本算例的主要不确定度来源。

## 6. 失败模式与工件

* **失败模式 F1（已量化）**：Gram 修正放大沉积力 O(ε−1)。稀疏 3 点核 + 高标记密度时可达 85%；
  若下游把“沉积力总和”当物理力（如总动量/推力的直接记账），必须显式除以 ε；本仓内已登记为限制。
* **失败模式 F2**：支撑支持不周期回卷（OSS 语义），边界附近标记支撑变小 → 该处 ε 与内部不同；
  用周期 `IBMStencil` 时靠零填充保持布局，已验证与向量化路径逐位一致。
* **失败模式 F3**：`IBMStencil.spread` 的向量布局是 `(M, C)`（`interpolate` 返回 `(C, M)`），
  标量分支 `(M,)` 会触发广播错误；本单按 `(M, C)` 使用并在测试中固化。
* **不可做项**：OSS LIFE 需 Boost/LAPACK 与完整 FSI 算例，本环境未编译 OSS 二进制；
  对标为“逐行语义移植 + 解析/恒等判据”，非同程序对跑（已写入 predeclared 的比较限制）。

**工件**（均在本目录）

| 文件 | 内容 |
|---|---|
| `results.json` | 全部 KPI、provenance（git sha/env）、逐臂 drag_history、耗时 |
| `benchmark.json` | 25 行 ref/got/abs_err/rel_err/tol/pass（20 通过，5 失败：1 控制组预期失败 + 1 沉积代价 + 3 物理阻塞比） |
| `fig1_roundtrip.png` / `fig2_epsilon_cost.png` / `fig3_cd.png` / `fig4_convergence.png` | 往返误差、ε 与其代价、C_D 对照、均值收敛 |
| `predeclared.json` | 预声明判据（未改动） |
| `run.log` / `sweep.log` | 原始运行日志 |
| `blockage_sweep.json` | 阻塞比敏感性（低阻塞点，见 §5 扩展） |

**复现命令**

```bash
cd <repo> && PYTHONPATH=src python xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION/run_xval.py
PYTHONPATH=src python xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION/make_figures.py
PYTHONPATH=src python -m tensorlbm.ibm_epsilon_gram            # 模块自检（解析判据）
PYTHONPATH=src python -m pytest tests/test_ibm_epsilon_gram.py tests/test_ibm_epsilon_gram_solver_smoke.py -q
PYTHONPATH=src python xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION/sweep_blockage.py --steps 700 --transient 300 --boxes 192x96
```

## 7. 本仓 vs OSS 差异清单与移植取舍

| 维度 | LIFE (OSS) | 本仓移植 | 取舍理由 |
|---|---|---|---|
| 语言/框架 | C++ (`IBMNode`/`Objects`)，Boost/LAPACK | PyTorch float64，`torch.linalg.solve` | 免编译依赖，GPU/CPU 同码；小系统（M≈31）求解开销可忽略 |
| 支撑存储 | 每节点对象数组，长度可变 | 零填充 `(M, max_support)` 矩形 + `IBMStencil` 线性索引 | 复用仓内力传递机；零权不产生贡献，语义等价（测试逐位核对） |
| 边界 | 物理域边界截断 | 同（不周期回卷）；`IBMStencil` 侧零填充 | 保持 OSS 语义；不与仓内周期 stencil 假设冲突 |
| ε 重算 | 刚体 t=0 一次，柔体每步 | 每算例一次（刚性圆柱） | 与 OSS 刚性体语义一致；柔体路径未实现（未使用） |
| spread 权重 | `f·ε·ds·δ` | `IBMStencil.spread(f·ε·ds)` | 与 OSS 同式，同时用上仓内索引优化 |
| 插值 | 无权重 δ | 无权重 δ（同一 stencil 的 `w`） | ε 只作用于 spread 方向，与 OSS 一致 |
| 力核基线 | 仅 3 点核 | 3 点核（LIFE 语义）+ 仓内 hat/4pt 可并存 | 差异清单需要仓内基线（A3） |
| 未移植 | 柔体每步重解、MPI 分解、FSI 耦合外层 | — | 超出本次对标范围，如实声明 |

## 8. 结论与证据分级

* **REAL**：δ/支撑/ds/Gram/往返/物理算例全部由本仓代码现跑，数字见 `results.json`/`benchmark.json`。
* 方法验收：往返误差 ≤2.2e-3（vs 控制组 0.46，解析恒等判据 1e-2），**远优于 15% 精度标准**。
* 物理验收：Re=100、12.5% 阻塞比下 C_D=1.616（A1/A2/A3 一致到 ≤3.9%），
  与**无界**文献带差 +19.7%，判 **FAIL（honest；归因未定）**：低阻塞算例（B=0.0833）给出 1.6218，
  阻塞比假设被证伪；未验证的候选 = 分辨率（D=8）/ 平均窗口（0.4 个脱落周期）/ 可压缩误差。
  改进路径已写成脚本：`sweep_blockage.py`（更多 B 点）+ 加长 `--steps/--transient` + 提高 D。
* 已知代价：修正使沉积总力偏差达 O(ε−1)（N=32 时 85%），必须在下游记账时显式修正。