# XVAL m17902 — fused push-pull AA pattern: kernel wired into the solver + machine referee

**Anchor**: `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN`（OSS `lbm-principles-practice/code` ch.13 GPU `gpu_stream_collide_save`）。
**Commit**: `3c92849c`。**Case**: 2d_taylor_green_decay_d2q9_bgk。

## ① 案例介绍
上游做法：把 streaming 与 collision 融进一个 kernel——pull 八个平移到邻居的**碰撞后**分布量，由这九个数重算 ρ/u，BGK 弛豫后直接写另一组缓冲，全程不存在「已 stream 未 collide」的中间场。
物理算例=二维 Taylor-Green 衰减（D2Q9 BGK，周期，float64）：

| 参数 | 值 |
|---|---|
| ν, τ | 0.1666666667, 1.0000000000 |
| u_max, Ma | 0.02, 0.0346 |
| ρ0 | 1.0 |
| 网格/步数 | steps ∝ n²：n=32:120, n=48:270, n=64:480 |
| 参考解 | 闭式 Taylor-Green（上游 `taylor_green_eval`，格心 X=x+1/2, Y=y+1/2, ρ=ρ0+3P） |
| 判据 | owner 15%；自设 3% |

## ② 怎么做的
锚点在首次交付（m17839, `eee0988a`）被独立审计判 L2：融合模块**零调用方**、**无基准裁判行**。本单闭合这两项，并纠正首交的一处过强主张：

1. 新集成点 `tensorlbm.solver.step_stream_collide(f, tau, kernel="ab"|"fused_push_pull")`——`"ab"` 即既有 `stream(collide_bgk(...))` 路径，`"fused_push_pull"` 路由到 `tensorlbm.fused_push_pull`；未知 kernel 与非法 τ 在任何计算前拒绝。
2. `benchmarks/verified/fused_push_pull_aa_pattern/`：两 kernel × 三级网格阶梯 + 两个负控制；`verify.py` 机器裁判（锁定常数、退出码 0）；`register_case.sql` 幂等写入 `tensor_benchmark_cases`（已落地，行状态 pending）；`make_doc.py` 从 `result.json` 渲染文档（零手抄）。
3. **主张纠正**：两 kernel 原始状态**不是**位级相同——融合核在 stream **之后**碰撞、AB 在**之前**，原始差=一次碰撞算子（O(Ma)）。精确关系是 `f_fused(n) == collide_bgk(f_ab(n))`。

## ③ 计算结果（程序化注入，零手抄）

| 网格 | 步数 | kernel | u_x rel.L2 | u_y rel.L2 | ρ rel.L2 | 质量漂移 | ν 反演误差 |
|---|---|---|---|---|---|---|---|
| 32 | 120 | `ab` | 0.4849% | 0.4849% | 1.224e-01 | 6.66e-15 | 0.314% |
| 32 | 120 | `fused_push_pull` | 0.4849% | 0.4849% | 1.224e-01 | 6.77e-15 | 0.314% |
| 48 | 270 | `ab` | 0.2101% | 0.2101% | 1.908e-02 | 1.50e-14 | 0.134% |
| 48 | 270 | `fused_push_pull` | 0.2101% | 0.2101% | 1.908e-02 | 1.50e-14 | 0.134% |
| 64 | 480 | `ab` | 0.1121% | 0.1121% | 4.069e-02 | 2.65e-14 | 0.071% |
| 64 | 480 | `fused_push_pull` | 0.1121% | 0.1121% | 4.069e-02 | 2.65e-14 | 0.071% |

等价性与收敛（n=64, steps=480）：

| 量 | 值 |
|---|---|
| `max|f_fused − collide_bgk(f_ab)|` | **3.331e-16** |
| `max|f_fused − f_ab|`（一次碰撞标签偏移，**非**误差） | 1.399e-04 |
| 跨 arm 速度 rel.L2 | **3.066e-14** |
| fused vs 解析速度 rel.L2 | 1.121e-03 |
| AB vs 解析速度 rel.L2 | 1.121e-03 |

收敛阶（steps ∝ n²）：

| kernel | 观测阶 |
|---|---|
| `ab` | 2.063, 2.182 |
| `fused_push_pull` | 2.063, 2.182 |

负控制（判据是否咬人）：

| 控制臂 | u_x rel.L2 | ≥15% 判 FAIL |
|---|---|---|
| `nu_bias_25pct` | 32.39% | yes |
| `mirror_we_initial` | 193.65% | yes |

## ④ 结果图
见内嵌 figure（fig1 误差-网格 + 15%/3% 门限；fig2 场量云图（解析/AB/融合/差×1e13）；fig3 质量守恒与 ν 反演）。

## ⑤ 注意点
- **状态标签偏移**：融合核存碰撞后状态，AB 存 stream 后状态；比较原始状态量到的是一次碰撞，不是误差。今后任何「bit-identical」主张必须声明比较层级。
- **仅周期边界**：两 kernel 都不是边界条件实现（上游按 `%n` 环绕）；壁面/入口/外力仍在 `tensorlbm.boundary` 之上叠加。
- **存储主张为结构性**：上游 17 doubles/node；仓内模块支持单缓冲 9/node（向量化 gather 先完成再写）。本单**不主张**任何 MLUPS/带宽数字——测量机 load15>100，表内 wall 为高负载墙钟（AB 64² 107 s vs 融合 6.5 s 是负载噪声，不是性能结论）。
- **上游 CUDA 未执行**（沙箱无 GPU）；等价性对的是该快照的 NumPy/torch 转写，源快照 sha256 已由前次审计核验。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否<3%）
- 最细网格 u_x rel.L2（两 kernel 取最差）：**0.1121% < 3% ✓**（owner 门限 15%）。
- 跨 arm 速度差 **3.066e-14**（< 1e-10 ✓）；同级状态等价 **3.331e-16**（< 1e-12 ✓）。
- 质量漂移最大 **2.65e-14**（< 1e-12 ✓）。
- 两个负控制均 ≥15%（32.39% / 193.65%）→ 判据不空转 ✓。
- 结论：**PASS**，`verify.py` 退出码 0（12 项 gate 全绿）。
