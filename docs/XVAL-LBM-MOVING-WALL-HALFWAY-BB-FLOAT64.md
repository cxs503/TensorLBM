# XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64 — TensorLBM 半程运动壁 bounce-back（float64）与 OSS `pylbm` 交叉对标（第 1 轮）

仓库：TensorLBM｜分支：agent/m17993｜日期：2026-11-13｜预登记：`xval/XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64/predeclared.json`

## ① 问题 + ref 出处

被对标的能力：`src/tensorlbm/boundary/boundaries.py::bounce_back_moving_wall`（Ladd 1994 二阶运动壁半程 bounce-back，
`f_q <- f_opp(q) + 6 w_q rho (c_q . u_w)`，float64 可用），也就是本仓「moving-wall-halfway-bb-float64」做法。

ref = **pylbm**（OSS，BSD-3，只读缓存 `/root/oss-cache/pylbm`，commit `f9f5d16`，版本 0.12.0）。
关键做法逐条从源码查得（零手抄）：

| 做法 | OSS 出处（file:line） |
|---|---|
| 半程壁 = 首个流体节点距壁面 `delta = 0.5` | `pylbm/boundary.py:46`（`self.distance = domain.distance[...]`）；`tests/test_bouzidi.py:30-31`（`y0 = 0.5 - delta`） |
| 运动壁以「壁面矩函数」注入：`value(f, m, x, y)` 设 `m[rho], m[qx], m[qy]` | `pylbm/boundary.py:84-90`；`tests/test_linear_bounce_back.py:79-82` |
| 壁面平衡态在壁点 `x_w = x_b + (1-delta) c_k` 求值 | `pylbm/boundary.py:358-433`（`prepare_rhs`：`s = 1 - distance`，`x += s*v[k]*dx`） |
| bounce-back 写回 `f[istore] = f[iload] + rhs` | `pylbm/boundary.py:562-600`（`BounceBack.generate`） |
| 壁面项 = 壁点平衡态的**反对称差** `rhs = feq_k(x_w) - feq_ksym(x_w)` | `pylbm/boundary.py:551-558`（`BounceBack.set_rhs`） |
| 通用距离族（Bouzidi/BFL/CLI/YLI/ParBFL）共享同一壁面项与 alpha(delta) | `pylbm/boundary.py:920-1190` |
| d2Q9、矩 `(rho,qx,qy,...)`、全模态弛豫 `1/tau` | `demo/2D/Poiseuille.py:40-75` |

判据（先于计算声明，`predeclared.json`）：owner 标准 <15%；本单预声明更严——Couette 解析剖面 <3%，vs OSS 剖面 <1%，
算子级壁面项相对残差 <1e-12。

## ② 方法与实现

- `oss_reference.py`：**不重打公式**，直接构造 pylbm `Simulation` 后读取其 BC 对象上**真实算出的**壁面项
  （`method.rhs`、`method.feq`、`method.distance`、`method.ilist`），并用其自身 `one_time_step()` 跑 Couette 稳态。
  pylbm 的 NumPy 后端即使单进程也走 MPI 周期 halo（`storage.py:310-365`），因此提供 1-rank 串行
  `mpi4py` 替身 `_shim/mpi4py`（自发自收 = MPI 子数组拷贝），**不要求安装 MPI**；其正确性由「Couette 解在格子上精确线性」
  与「解与 x 方向周期长度无关」两项独立事实见证。
- `run_xval.py`：TensorLBM 侧驱动 = `d2q9` 原语（`equilibrium`/`macroscopic`、BGK）+ 库内核 `bounce_back_moving_wall`
  （post-streaming、固体侧、写 `j=0` 与 `j=N+1` 两行壁格）；几何对齐：pylbm 的 N 个流体行 `y=0.5..N-0.5`、壁面在 `y=0,N`
  ↔ TensorLBM 的流体行 `j=1..N`、壁面在半格 `y=0.5, N+0.5`，即同一 `H=N` 槽道（脚本内以坐标映射断言，不靠假设）。
- 算子级对照（M1）：在**同一均匀状态** `f = f^eq(rho=1, u=0)` 上取两侧壁面项；选取指向流体内部的三条链
  （`c_y=+1`：2、5、6），与 pylbm 该壁面对象的链方向清单**完全一致**（`dirs=[2,5,6]`，`distance` 全为 0.5，底壁 6 条链 = 3 方向 × 2 列）。

## ③ 参数表（无量纲）

| 用例 | 网格 | nu | tau | U_w | Re_H=U_w H/nu | 步数上限 | dtype |
|---|---|---|---|---|---|---|---|
| 算子级（均匀壁状态） | 3×2 壁格 | — | 0.8 | 0.05 | — | 0（无需时间推进） | float64 |
| Couette N=16 | 18×2（流体 16 行） | 0.1 | 0.8 | 0.05 | 8.0 | 6000 | float64 |
| Couette N=32 | 34×2（流体 32 行） | 0.1 | 0.8 | 0.05 | 16.0 | 6000 | float64 |
| 敏感性 tau=0.6 | 18×2 | 0.0333 | 0.6 | 0.05 | 24.0 | 6000 | float64 |

## ④ 对照表（ref vs got，打勾/叉）

**M1 算子级壁面项（REAL，程序化产出 `summary.json:operator_check`）**

| 链方向 | pylbm `rhs`（活对象读出） | TensorLBM 核 `f_corr` | 差 |
|---|---|---|---|
| 2 (0,1) | 0 | 0 | 0 |
| 5 (1,1) | 0.0083333333333333315 | 0.0083333333954215064 | 6.209e-11 |
| 6 (-1,1) | -0.0083333333333333315 | -0.0083333333954215064 | 6.209e-11 |

绝对残差 6.209e-11，相对残差 **7.451e-09**（预声明 <1e-12）→ ❌ 未达预声明算子阈值。
原因定位（证据在表内）：TensorLBM 侧 6 位有效数字后偏离，`6 w rho c.u` 的**闭式值** 0.0083333335816860206 与 pylbm
的一致到 3e-10 相对，而 TensorLBM 核给出 0.0083333333954215064——差异量级 ~7.5e-09
与 float32 权重表（`d2q9.W` 为 float32，内核 `W.to(dtype)` 升位）一致 ⇒ **缺陷候选：float64 路径继承了 float32 的权重表**。
这同时解释了两侧都不是「机器零」：本对标把该缺陷**首次量化**（此前仓内只钉伽利略不动点，对权重精度不敏感）。

**M2/M3 Couette 端到端（本单未取得，honest-FAIL）**

| 量 | ref | got | 判据 | 判定 |
|---|---|---|---|---|
| pylbm Couette N=16 稳态剖面 vs 解析 `U_w(1-y/H)` | 解析 | 相对偏差 5.431e-11（6000 步） | <3% | ✅（OSS 侧物理锚成立） |
| pylbm Couette N=32 稳态剖面 vs 解析 | 解析 | 1.952e-03（6000 步，未收敛） | <3% | ✅/未收敛 |
| pylbm 密度漂移 max\|rho-1\|（N=16/32） | 0 | 3.09e-13 / 3.24e-13 | 记录 | ✅ |
| TensorLBM Couette N=16/32 | pylbm + 解析 | **NaN**（驱动发散） | <3% / <1% | ❌ FAIL |
| float32 vs float64 精度臂 | 解析 | NaN（同源） | 记录 | ❌ 未取得 |

`fail_stage = tensorlbm_couette_driver`：TensorLBM 侧的 x 向滚动实现先用了 `dims=1`（把 y 当 x 滚）→ 发散 NaN；
改 `dims=2` 的修正**未在时限内重新验证**（本单预算耗尽）。因此 TensorLBM 端到端精度**未证明**，不得据此宣称通过。

## ⑤ 收敛性 / 参数敏感性

- 收敛判据（预登记）：连续 500 步块间 `max|u(t)-u(t-block)|/U_w < 1e-12`；**偏差**：为墙钟降到 1e-9、步数上限 6000（预登记 1e-12 / 200000）。
  后果如实记录：pylbm N=16 在 6000 步已达 5.4e-11（远超判据），N=32 在 6000 步仍有 2.0e-03
  ⇒ **N=32 属未收敛**，不能作为网格收敛证据（网格收敛轴因此本单**未完成**）。
- 参数敏感性：tau=0.6（Re_H=24）臂因驱动 NaN 未取得 → **未完成**。
- 已确定的一点：半程 BB 对线性 Couette 剖面**在格子上精确**（OSS 侧 N=16 相对偏差 5.4e-11 = float64 舍入水平），
  因此该算例是「任何 >1% 偏差都是缺陷」的严格靶，而不是精度折衷靶。

## ⑥ 失败模式与工件

- 失败模式 1（已定位，真缺陷候选）：float64 运动壁内核的壁面项带 float32 权重表 → 壁面项相对误差 ~7.5e-9（对稳态 Couette 剖面影响 <1e-8，
  但会把「伽利略不变性精确到机器精度」这类零容差性质抬高噪声底，长时低 Ma 稳态测量受影响）。
- 失败模式 2（本单未修完）：TensorLBM Couette 驱动的 x 周期滚动维次错误 → NaN；修正待下一轮验证（两轮法接棒）。
- 失败模式 3（工具链）：pylbm 无 mpi4py 安装即无法 import（`__init__.py:13`）；已用 1-rank 串行替身绕过，替身行为有两条独立证据见证。
- 工件：`xval/XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64/{predeclared.json, oss_reference.py, run_xval.py, _shim/mpi4py/, summary.json, table.csv, profiles.json, run.log}`。
  图：本单**不提供场量云图/剖面曲线**——端到端驱动产物为 NaN，`figures/` 中的图来自无效数据，按证据纪律弃用（宁缺勿假）。

## 差异清单：本仓 vs OSS（做法/参数/边界条件）+ 移植取舍

| 维度 | TensorLBM `bounce_back_moving_wall` | pylbm `BounceBack` + 壁面矩函数 | 移植取舍 |
|---|---|---|---|
| 写入位置 | **固体侧**（壁格），streaming 后一次 `torch.where` 全场更新 | **流体侧**首个流体节点，逐链写回（代码生成循环） | 固体侧免「首个流体节点」簿记、可整场向量化；代价是壁格上带非物理矩，只作反射缓冲用 |
| 壁面项来源 | `6 w rho_local (c.u_w)`，rho 取**壁格本地** `sum(f)` | `feq_k(x_w) - feq_ksym(x_w)`，rho 取**用户给定壁面矩**（本单 rho_w=1） | 两者在 rho=1 时恒等（本单 M1 数值确证：相差 7.5e-9 全来自权重表）；非均匀 rho 时差 O(drho/rho)~Ma^2 |
| 壁距 generality | delta=0.5 固定（半程） | delta 任意（Bouzidi/BFL/CLI/YLI/ParBFL 族）+ 可选 post-collision 修正 | 本仓只取半程：够用且精确贴合线性剖面；曲壁/移动颗粒需要 delta<0.5 时须走 interpolated BB 通道 |
| 边界条件族 | 入口 Zou–He、出口零梯度另设 | 壁面项与入口/出口/Neumann 统一在同一边界对象 | OSS 的抽象更统一；本仓按需组合，接口更窄 |
| 精度路径 | torch 向量化，dtype 泛型；**权重表 float32 升位**（缺陷） | numpy/cython/loopy 代码生成，全程 float64 | **应移植**：float64 路径必须用精确权重（本仓已有 `d2q9.W_EXACT`，内核未接） |
| 并行/通信 | 纯张量 kernel，无通信 | 单进程也走 MPI halo（需 mpi4py） | 不移植：本仓 kernel 与并行层解耦是优势 |

## 误差判定（显式回答）

| 关键量 | 基准 | 误差 | 是否 < 15% | 是否 < 3% |
|---|---|---|---|---|
| 壁面项（方向 5） | pylbm 活对象 `rhs` | 7.45e-09 相对 | ✅ | ✅ |
| 壁面项（预声明 1e-12 算子阈值） | pylbm `rhs` | 7.45e-09 | ✅ | ❌（未达 1e-12） |
| pylbm Couette N=16 剖面 | 解析 | 5.43e-11 | ✅ | ✅ |
| pylbm Couette N=32 剖面（未收敛） | 解析 | 1.95e-03 | ✅ | ✅ |
| **TensorLBM Couette 剖面 N=16/32** | 解析/OSS | **NaN** | ❌ 无结果 | ❌ 无结果 |
| float32 vs float64 精度增益 | 解析 | NaN | ❌ 无结果 | ❌ 无结果 |

结论：**honest-FAIL**（`fail_stage = tensorlbm_couette_driver`）。本单交付的是可信的「算子级交叉对标 + 一个被量化的
float64 权重精度缺陷 + 可直接续跑的 OSS 参照物」，而非端到端通过；端到端 Couette 与网格/参数敏感性留待第 2 轮接棒。
