# 移动壁反弹边界横评：7 份实现 vs 伽利略不变量（TensorLBM, 2026-11-08）

`mission 6419 · repo TensorLBM · branch agent/m6419 · 自找活 selfstart`

**结论一句话**：缺口账 `benchmarks/gaps.md` G18 记的"库内没有移动壁反弹"已经过时——
库里现在有 **7 份** 同一物理律的实现（契约各一、密度口径各一、速度接口各一），
此前没有任何解析基准把它们钉在一起。本单给出一个**零容差**判据（伽利略不动点，
正确 Ladd 映射在均匀随流态上*恒等*），逐一裁决：**5/7 精确通过（≤5e-9，float32 噪声地板）；
1 份发现并修复真实缺陷（缺密度因子 + 忽略传入掩码）；1 份发现未修复缺陷（修正项落在
错误的那半个 link 上，偏差 33.3% ρU），已用 `xfail(strict=True)` 钉住并如实上报。**

---

## 1. 案例介绍（物理背景与对象）

LBM 的移动壁半程反弹（Ladd 1994）只有一个式子：

```
f_q(x_b) = f_opp(q)(x_b) + 6 w_q ρ (c_q · u_w)        (c_s² = 1/3)
```

同一律在本库被复制到 7 处，每处的"前后流契约"不同：

| # | 实现 | 位置 | 契约 | u 接口 | 密度口径 | 掩码范围 |
|---|---|---|---|---|---|---|
| 1 | `rotating_cylinder.moving_wall_bounce_back` | `rotating_cylinder.py:140` | 固体格点内交换（post-stream） | **场** | `f.sum(0)` | 掩码内全部格点 |
| 2 | `herschel_bulkley_lbm.moving_wall_bounce_back_2d` | `herschel_bulkley_lbm.py:440` | 同上 | **标量** | `macroscopic` | 全部格点 |
| 3 | `yield_stress_channel._moving_wall_bounce_back` | `yield_stress_channel.py:47` | 同上 | 标量 | **无 ρ**（已修） | **硬编码最后一行**（已修） |
| 4 | `multiphase_water_entry._moving_wall_bb_2d` | `multiphase_water_entry.py:419` | **流体侧注入**（用 f_pre） | 标量 | `f_pre.sum(0)` | 流体格点上写回 |
| 5 | `propeller_benchmark.moving_wall_bounce_back_3d` | `propeller_benchmark.py:221` | 固体格点内交换 | 场 | `f.sum(0)` | 全部格点 |
| 6 | `cac_moving_body.moving_wall_bounce_full` | `cac_moving_body.py:272` | 固体侧、**仅边界 link** | 标量 | 出流邻点 ρ | 表面 link |
| 7 | `cac_moving_body.moving_wall_bounce` | `cac_moving_body.py:210` | 流体侧注入 | 标量 | 场 | 流体节点 |

这些实现分别服务于旋转圆柱（Magnus）、Bingham 屈服应力 Couette、入水多相（球体下沉）、
螺旋桨 D3Q19 力法、CAC 自由表面移动体——全部属船舶水动力主线（操纵/推进/入水）。
判据若错，对应算例的壁面动量注入就按比例错，且**在 ρ≈1 的等温单相算例里看不出来**。

## 2. 怎么做的（方法/流程/工具链）

**工具链**：`artifacts/moving_wall_audit_20261108/run_audit.py`（本单新增）——
不重写物理，`import` 上述 7 个真实函数，按各自 docstring 说明的契约驱动；
两个实验：

### E1 伽利略不动点（局部、无时间推进、零容差）

若流体与壁一起平移 `u_w`，即 `f = f^eq(ρ, u_w)`，代入 Ladd 映射：

```
f_opp(q) + 6 w_q ρ (c_q·u_w)
 = w_q ρ [1 − 3(c_q·u_w) + 4.5(c_q·u_w)² − 1.5u_w²] + 6 w_q ρ (c_q·u_w)
 = w_q ρ [1 + 3(c_q·u_w) + 4.5(c_q·u_w)² − 1.5u_w²] = f^eq_q(ρ, u_w)
```

**恒等**（对 ρ、Ma、网格、维度均无近似）。因此正确的算子在均匀随流态上必须
**逐位不变、净动量注入为零**；偏差只能来自约定错误。度量
`max|f_after − f^eq| / (ρU)`。三个工况：1 行壁 / 4 行壁（表面 vs 内部）/ ρ=1.2
（暴露"漏乘 ρ"这类只在 ρ≠1 现形的缺陷）。

### E2 稳态平面 Couette（全局、时间推进）

`ny=66`、`τ=1`、`U=0.02`、6000 步、float64 单线程；壁在 0.5 / ny−1.5 半程平面，
解析解 `u(y)=U(y_top−y)/(y_top−y_bot)`。同时报告"实测壁面位置"（拟合直线取到 U 与 0 的 y）
与线性度，用于区分半程/整程两种合法约定。

### 负控（判据有没有鉴别力）

同一 harness 内跑 `oracle(+sign)` 与 `oracle(−sign)`（把修正项符号翻掉）。
负控必须失败，否则"通过"无意义。

### 一个必须记下来的流式处理坑

库的 `solver.stream` 两轴都 roll。**两壁都贴在计算域边界**时，远端壁的交换后
population 会经周期性回卷漏到近端壁（实测把剖面整体平移约 2.3 格、残差放大一个量级，
**连我自己的 oracle 也一样**）。E2 因此在 harness 内部改用"流向周期 / 壁法向不回卷 /
域边界镜面封闭"的局部 streamer。这是 harness 修正，不是代码缺陷——但它同样限制了 E2
的绝对精度（见 §5）。

## 3. 计算结果

### 3.1 E1 伽利略不动点（`galilean_results.json`）

| 实现 | 1 行壁 ρ=1 | 1 行壁 ρ=1.2 | 4 行壁 ρ=1 | 判定 |
|---|---|---|---|---|
| `oracle(+sign)`（本单自写） | **0.0** | **0.0** | **0.0** | 基准 |
| `oracle(−sign)` 负控 | 133.3%（壁格 400%） | 133.3% | 133.3% | **判据有鉴别力** |
| `rotating_cylinder` | 4.97e-9 | 4.97e-9 | 0.0 | 通过 |
| `herschel_bulkley` | 4.97e-9 | 4.97e-9 | 0.0 | 通过 |
| `multiphase_water_entry` | 2.48e-9 | 2.48e-9 | 0.0 | 通过（float32 噪声） |
| `propeller_benchmark` | 2.48e-9 | 2.48e-9 | — | 通过 |
| `cac_moving_body.moving_wall_bounce`（流体侧） | 1.24e-9 | 1.24e-9 | — | 通过 |
| `cac_moving_body.moving_wall_bounce_full` | **33.3%** | **33.3%** | — | **未通过（§4 F1）** |
| `yield_stress_channel`（**修复前**） | 0.0 | **11.11%**（壁格 33.33%） | 66.7% | **未通过（§4 F2，已修）** |
| `yield_stress_channel`（**修复后**） | 4.97e-9 | 4.97e-9 | 66.7%（内部 link，见 §5） | 通过 |

### 3.2 E2 稳态 Couette（`couette_run.log`）

| 单元 | 1 行壁 max_rel | 约定 | 线性度 | 4 行壁 max_rel |
|---|---|---|---|---|
| `oracle(+sign)` | 5.718% | halfway | 3.503% | 3.384% |
| `oracle(surface-only)` | 5.718% | halfway | 3.503% | 3.384% |
| `rotating_cylinder` | **5.718%**（= oracle） | halfway | 3.503% | 3.384% |
| `oracle(−sign)` 负控 | **197.54%** | fullway | 3.503% | 197.351% |

`rotating_cylinder` 与 oracle **逐位同值**（斜率、壁位、线性度全同）⇒ E2 残差 5.718%
是 harness 自身地板（域边界封闭 streamer），不是被测代码的缺陷；负控 197% 说明该地板
比缺陷小 35 倍，E2 仍可用于**相对**裁决。其余 6 个单元因时间预算未跑完 E2（见 §5）。



### 3.3 补记（2026-11-08，完整 E2 运行到齐后追加）

后台全量运行最终写全了 `results.json`（上表 3.2 写就时只跑完 4 个单元，此处按实测更正，并非新增计算）：**10/10 单元均有 E2 结果**，地板 5.718%（oracle 同值）。

| 单元 | E2 max_rel | 相对地板 | 约定 |
|---|---|---|---|
| `oracle(+sign) [written here]` | 5.718% | 1.00× | halfway |
| `oracle(-sign) [negative control]` | 197.540% | 34.55× | fullway |
| `oracle(surface-only) [written here]` | 5.718% | 1.00× | halfway |
| `rotating_cylinder.moving_wall_bounce_back` | 5.718% | 1.00× | halfway |
| `herschel_bulkley_lbm.moving_wall_bounce_back_2d` | 5.718% | 1.00× | halfway |
| `yield_stress_channel._moving_wall_bounce_back` | 5.718% | 1.00× | halfway |
| `multiphase_water_entry._moving_wall_bb_2d` | 5.124% | 0.90× | halfway |
| `propeller_benchmark.moving_wall_bounce_back_3d` | 12.769% | 2.23× | halfway |
| `cac_moving_body.moving_wall_bounce_full` | 98.462% | 17.22× | fullway |
| `cac_moving_body.moving_wall_bounce` | 12.069% | 2.11× | halfway |

三点佐证：① `herschel_bulkley` / `yield_stress`（修复后）与地板**逐位同值**，说明 F2 的修复未引入偏差；② `cac_moving_body.moving_wall_bounce_full` 在 Couette 里 **98.46%**（≈1.7×地板 ... 实为 17×），与 E1 的 33.3% 同向同源，独立佐证 F1；③ 三个 D3Q19 单元（propeller 12.77%、cac 流体侧 12.07%）高于 2D 地板，属 3D 网格 + 同一封闭条件差异，**不足以判缺陷**（未做 3D 地板对照，如实记为未决）。

## 4. 发现（按严重度）

### F1 — 高：`cac_moving_body.moving_wall_bounce_full` 修正项落在错误的那半个 link 上（未修，已上报）

* 现象：E1 偏差 **33.3% ρU**（正确实现 ≤5e-9，静壁为 0）。
* 根因：函数自己的 docstring 写的是"对**从固体流出**的 population 加 Ladd 项"
  （`x_b + c_q ∈ fluid`），代码却用
  `surface = body & links_from(fluid)`（`x_b − c_q ∈ fluid`，即**从流体流入固体**的那半）。
  1 格厚壁上两个集合互斥 ⇒ 出流 population 得不到修正，壁只被"半修正"。
* 影响面：CAC 自由表面移动体（`MovingBodyCAC` 步进器的主路径）、以及任何用它的算例；
  对称几何下两侧误差部分抵消，表现为阻力/力矩系统偏差而非平移错，更难自查。
* 处置：**未改代码**——该函数被 `tests/test_cac_moving_body.py` 与自由表面步进器引用，
  改 link 集合会动到 CAC 数值结果，超出本单"最小可验证改动"的范围。
  已在 `tests/test_moving_wall_convention_audit.py::test_cac_moving_wall_bounce_full_is_galilean_transparent`
  用 `xfail(strict=True)` 钉死：**修好会立刻变红提示拆除标记，不会静默漂移**。

### F2 — 中：`yield_stress_channel._moving_wall_bounce_back` 缺密度因子 + 忽略传入掩码（**已修**）

* 缺陷 1：修正项写成 `6 w (c·u_w)`，少了 `ρ`。ρ=1 时**完全看不出来**（E1 ρ=1 偏差 0），
  ρ=1.2 时偏差 11.11%（= 修正量的 33%）。Bingham 屈服流在瞬态里 ρ 并不恒为 1。
* 缺陷 2：修正只加到硬编码的 `top[-1, :]`，不看调用者传入的 `wall` 掩码 ⇒
  任何"移动壁不在最后一行"的几何都被**静默当成静壁**。
* 修复（`src/tensorlbm/yield_stress_channel.py`）：补 `ρ=f.sum(dim=0)`；
  新增 `moving_mask` 关键字参数，缺省仍取最后一行（`run_bingham_couette` 行为不变），
  与 `wall` 取交后施加修正。两项修复均由 E1（ρ=1 与 ρ=1.2 同时通过）与
  `test_yield_stress_operator_follows_the_passed_mask` 钉住。

### F3 — 低：float32 与"u_w 该喂在哪个格点"的隐性契约

* `multiphase_water_entry` 用 `C2.to(device).float()` 建格子向量 ⇒ 即使 f 是 float64，
  修正量的噪声地板也被抬到 ~5e-9（相对 ρU）。单相极限下无损，但这是**唯一的 float32
  污染源**，建议后续统一 `to(f.dtype)`。
* 该函数的流体侧契约把 `u_w` 读在**流体节点**上（不是物体节点）；调用方若只在前者物体格点
  填速度场，修正恒为 0、退化成静壁。其 runner 传标量所以碰不到，但换几何就会踩。
  这是本单第一轮跑出"零流量"的原因（§5 注意点）。

### F4 — 信息：缺口账 G18 已过时

`benchmarks/gaps.md` G18 仍写"库内无移动壁反弹 / 无镜面反弹"，实际上已有 7 份，且
`boundaries.py` 依然没有统一入口——账本与代码脱节本身是风险（下一个人会照着 G18 再写第 8 份）。

## 5. 注意点 / 局限 / 未完成（honest）

* **E2 有 harness 地板 5.718%**：两壁贴域边界 + 壁法向不回卷导致的封闭条件与半程解析解
  不完全一致（oracle 同值即为证据）。**因此 E2 只能做相对比较，不能作为"绝对精度 <1%"的
  证据**；本单未把它写成"通过 1% 判据"。
* **E2 首轮只跑完 4 个单元**（后台进程一度被进程组收割），最终全量补全见 §3.3；
  3D 单元缺 3D 地板对照，其 12% 量级残差**判为未决而非缺陷**。E1 覆盖全部 7 份，是主证据。
* **E1 的 4 行壁工况不构成"通过/不通过"**：4 行壁时任何"只修正表面 link"的实现都会在
  内部 link 上出现 66.7% 偏差（内部交换本身不是恒等映射），这是**约定差异**而非缺陷；
  该工况只用来指出内部 link 的交换是真实存在的动量路径（F1 的放大机制）。
* 判据是**均匀随流**（零剪切）；它抓符号/密度/掩码/读错格点，抓不到"只在大曲率下才现形"的
  几何误差——曲壁需要另行用 `curved_boundary_calibration` 的 Taylor-Couette 力矩基准裁决。
* 全部计算 CPU 单线程 float64，数值与线程数无关。

## 6. 误差判定（vs 基准，逐项）

| 关键量 | 实现 | 基准 | 误差 | <3% ? |
|---|---|---|---|---|
| 伽利略偏差（ρ=1） | 5 份通过 | 恒等（0） | ≤4.97e-9 | **是** |
| 伽利略偏差（ρ=1.2） | 5 份通过 | 恒等（0） | ≤4.97e-9 | **是** |
| 伽利略偏差（ρ=1.2） | `yield_stress`（修复前） | 恒等（0） | **11.11%** | **否**（已修至 ≤5e-9） |
| 伽利略偏差（ρ=1） | `cac ..._full` | 恒等（0） | **33.3%** | **否**（F1，已上报） |
| 负控偏差 | `oracle(−sign)` | 应远大于判据 | 133% | 判据有鉴别力 |
| Couette 残差 | `rotating_cylinder` | oracle 地板 | 0（逐位同值） | 是（相对） |
| Couette 残差 | `oracle(−sign)` | 解析解 | 197.5% | 否（符合预期） |

## 7. 证据指针

* 审计器：`artifacts/moving_wall_audit_20261108/run_audit.py`（`--local` 只跑 E1）
* 原始证据：`artifacts/moving_wall_audit_20261108/{galilean_results.json, couette_run.log}`
* 图：`fig1_galilean_fixed_point.png`（E1 全单元，log 轴）、`fig2_couette_residual.png`
  （E2 相对地板），由 `make_figures.py` 只读上述 json/log 生成（零手抄）
* 判据：`tests/test_moving_wall_convention_audit.py`（7 用例：5 通过 + 1 xfail(strict) + 负控）
* 修复：`src/tensorlbm/yield_stress_channel.py::_moving_wall_bounce_back`
* 复核命令：`PYTHONPATH=src python3 -m pytest tests/test_moving_wall_convention_audit.py -q`