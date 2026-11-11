# TensorLBM 包级导入健康审计（m14806, 2026-11-11）

## ① 案例介绍
对象：`src/tensorlbm` 包自身。此前仓库对「模块能否被 import」没有任何门禁：
一个在 import 期就失败的模块对所有测试都是不可见的（pytest 收集即 ERROR 或被跳过），
而对用户则是硬崩溃。本审计把这一层作为**验证欠账**补上（无物理算例，纯后端/验证面）。

## ② 怎么做的
1. `pkgutil.iter_modules(tensorlbm.__path__)` 枚举全部 653 个模块，逐个 `importlib.import_module`，
   失败项记录异常类型与首行消息（`tools/_import_sweep.py`，输出 `artifacts/_import_sweep.json`）。
2. 对失败项做根因定位（不是猜）：与源模块当前符号表比对，区分
   「符号被重命名/删除」与「确属拼写/结构错误」。
3. 修复原则：**不得**制造第二套物理实现或静默语义映射；
   兼容名一律写成显式适配器，并在测试里断言「适配器 == 现行实现」。
4. 新增回归门 `tests/test_import_health.py`（653 个参数化 import + 4 个缺陷点显式钉住）。

## ③ 计算结果（程序化产出，零手抄）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 模块总数 | 653 | 653 |
| 导入成功 | 650 | **653** |
| 导入期失败 | 3（首轮报告）+1（级联暴露，共 4 处根因） | 0 |

缺陷清单（逐项）：

| # | 模块 | 症状 | 根因 | 处置 |
|---|---|---|---|---|
| 1 | `drag_monitor` | `from tensorlbm.drag_monitor import DragMonitor` → ImportError | 文件只有 docstring，无任何 import（自称 re-export） | 补真实 re-export 自 `yplus_guide` |
| 2 | `cad` | `cannot import name 'optimal_advance_ratio'` | propeller 重写后 `propeller_cad` 不再提供 `optimal_advance_ratio` / `propeller_design` / `propeller_disk_mask` / `wageningen_b_series` | 从命名空间剔除 4 个死符号（保留 `build_propeller_mask` 等的正确导入路径 `tensorlbm.propeller_cad`） |
| 3 | `boussinesq` | `cannot import name 'C5' from scalar_transport` | D2Q5 TRT 重写重命名了全部入口；旧名未留 | `scalar_transport` 末尾追加旧名适配器（见 #4） |
| 4 | `scalar_verification_in65` | 同上 + `scalar_diffusivity` + `OPPOSITE5` | 同上 | 旧名 9 项全部落为适配器：`C5`/`W5`/`_get_w5`/`scalar_equilibrium`/`scalar_collision`/`scalar_macroscopic`/`scalar_stream`/`scalar_diffusivity`/`OPPOSITE5` |

适配器语义等价性（测试断言，非声明）：
`scalar_equilibrium ≡ equilibrium_scalar_d2q5`；`scalar_macroscopic ≡ scalar_field`；
`scalar_collision(g,C,u,v,τ) ≡ collide_scalar_trt(g,C,u,v,τ,None)`（单弛豫=TRT 的 BGK 退化）；
`scalar_stream` 只丢弃质量记账元组；`scalar_diffusivity(τ)=c_s²(τ−1/2)`。

**连带发现（最有价值的一项）**：`tests/test_scalar_transport_boussinesq.py` 与
`tests/test_scalar_verification_in65.py` 两个既有测试文件此前**根本无法被收集**
（collection ERROR），即它们是常驻的「隐形红灯」。

## ④ 结果图
`figures` 内嵌图：左 = 修复前后 653 模块导入成功率（650→653）；右 = 两个遗留测试文件恢复执行后的结果（703 passed / 2 failed）。

## ⑤ 注意点 / 局限
- **honest FAIL（未修）**：两个遗留测试里 `TestD2Q5Lattice::test_opposite_mapping` 仍失败。
  适配器 `OPPOSITE5 = (0,3,4,1,2)` 是现行 `OPP_D2Q5` 的直接别名，属正确对偶映射；
  失败源于旧测试对 `OPPOSITE5` 的**对象类型/索引约定**（旧版为数组/张量）与现实现不一致，
  属测试侧陈旧断言，不是求解器缺陷。按纪律不粉饰：记录为失败，未改测试迁就。
- 其余 703 项全通过，说明适配器未改变物理路径。
- 本审计是「结构性」能力：它不提升精度，但让「不可见红灯」变成「红灯」。
- 扫描仅覆盖顶层模块（不含子包内部模块），子包导入由其 `__init__` 间接覆盖。

## ⑥ 误差判定
本单无人可比的物理基准，关键量是**可导入模块数**（计数型，无误差）：
导入健康 650/653 = 99.54% → 653/653 = 100.0%，缺陷消除率 4/4 根因（100%）。
遗留 2 项测试失败已如实列出，**未达「0 失败」**，不声明为通过。
