# XVAL · 点对点参考回归 第 2 轮：整步（碰撞+迁移）门 + 冻结参考数据

锚点 `XVAL-LBM-POINTWISE-REFERENCE-REGRESSION`（第 2 轮）｜仓 TensorLBM｜mission 20532
对标对象：**lssfau/lbmpy 的 pointwise-reference-regression 纪律**（同一 PDF 场上逐点比对，禁总量聚合代替）
上游轮次：第 1 轮（merged #2263）只覆盖**单步碰撞核**，迁移半步、30 步轨迹累积、以及对 lbmpy 活体安装的运行时依赖未覆盖。

## ① 案例介绍（物理背景与设置）

LBM 一步 = 碰撞（局部、逐方向）＋迁移（非局部、逐方向索引）。第 1 轮用 lbmpy 的符号碰撞算子做点对点真值，
但**迁移核**（本仓至少有 6 条独立实现）与**多步轨迹**没有任何点对点证据。本轮：

* 格子 D2Q9（c_s²=1/3），周期盒；D3Q19 只用于迁移核普查。
* 主算例：剪切波 u_x = u0 sin(ky)，y 方向 64 点、x 方向 8 点，u0 = 0.02（k = 2π/64，Ma ≈ 3.5e-2）。
* 另设非方域 7×11 非对称随机场（暴露轴/转置类索引缺陷）与大 Mach 32×32 非线性算例。
* 参数表（预登记，见 `stepwise/predeclared.json`）：tau ∈ {0.55, 0.7, 1.0}（ν* = 0.0167/0.0667/0.1667），
  TRT Λ=3/16，MRT 原生率 s_e=1.64, s_ε=1.54, s_q=1.7。
* 碰撞真值来源：lbmpy 2.0 / pystencils 2.0，
  其 `LbmCollisionRule.lambdify` 在 numpy 场上逐点求值（源码 sha256 记录在 summary.json，非手抄）。

## ② 怎么做的（方法与工具链）

```
参考一步  S_ref = Stream_定义 ∘ Collide_lbmpy
本仓一步  S_tl  = tensorlbm.solver.stream ∘ tensorlbm.solver.collide_*
```
1. `stepwise/refcore.py`（纯 numpy，无 lbmpy/无 torch）：`definition_stream` 用逐方向 `np.roll`
   直接实现拉式定义 f_i(x,t+1)=f_i^post(x−c_i,t)，与任何仓内迁移核的索引代码零共享。
2. `stepwise/stepwise_ref.py`：lbmpy 侧只负责**碰撞**（round-1 的同一真值源）；MRT 臂从
   `solver._M_D2Q9_DATA` 反推矩基再在 lbmpy 中重建（基重建 diff = 0，逐位相同）。
3. 7 条整步臂 × 30 步，每步两侧同步推进，只比较**同一时刻**的逐点场；同时测质量/总动量守恒。
4. 迁移核普查：6 条仓内实现各自按**原生布局**（`(9,ny,nx)` / `(ny,nx,9)` / `(B,9,ny,nx)` / `(19,nz,ny,nx)`）
   与定义逐点比对。
5. 物理臂：剪切波 2000 步，幅值与解析 A0·exp(−νk²t) 比对（owner 判据 15%/目标 3%）。
6. **冻结参考**：`stepwise/make_refdata.py` 把参考态（f0、t=1、t=30）写成 `refdata/pointwise_stepwise_d2q9_v1.npz`
   ＋ sha256 manifest（含 npz 与**每个数组**的哈希、lbmpy 源码哈希）。`--check` 复算验证逐位可复现。
   新门 `tests/test_xval_pointwise_stepwise_refdata.py` 只依赖 numpy+torch：**lbmpy 缺失也照跑**
   （第 1 轮的门在无 lbmpy 环境是 skip，等于死门）。

## ③ 计算结果（程序化注入，零手抄）

### 整步点对点（30 步，fp64）— 7/7 PASS
| arm | collision | tau | grid(s) | eps_inf(t=30) | mass drift | mom drift | gate | 判定 |
|---|---|---|---|---|---|---|---|---|
| S1 | srt | 0.55 | [9, 64, 8] | 5.37e-15 | 3.00e-15 | 1.45e-17 | 1e-10 | PASS |
| S2 | srt | 0.7 | [9, 64, 8] | 3.33e-14 | 2.33e-15 | 5.96e-18 | 1e-10 | PASS |
| S3 | srt | 1.0 | [9, 64, 8] | 8.49e-15 | 1.78e-15 | 1.75e-17 | 1e-10 | PASS |
| S4 | trt | 0.7 | [9, 64, 8] | 2.37e-14 | 2.22e-15 | 6.99e-18 | 1e-10 | PASS |
| S5 | mrt_native | 0.7 | [9, 64, 8] | 3.80e-14 | 4.00e-15 | 3.89e-15 | 1e-10 | PASS |
| S6 | srt | 0.7 | [9, 7, 11] | 3.31e-14 | 2.40e-15 | 8.80e-18 | 1e-10 | PASS |
| S7 | srt | 0.7 | [9, 32, 32] | 3.40e-14 | 2.44e-15 | 4.33e-19 | 1e-10 | PASS |

### 迁移核普查 — 6/6 PASS（eps=0 表示逐位相同）
| 实现 | 布局 | 格点 | eps_inf vs 定义 | 判定 |
|---|---|---|---|---|
| `solver.stream` | (9,ny,nx) gather | D2Q9 | 0.00e+00 | PASS |
| `lattice.stream` | (ny,nx,9) per-direction roll | D2Q9 | 0.00e+00 | PASS |
| `les_bptt.stream` | (B,9,ny,nx) conv2d | D2Q9 | 0.00e+00 | PASS |
| `diff_boundary_grad.stream` | (9,ny,nx) | D2Q9 | 0.00e+00 | PASS |
| `h_theorem.stream` | (9,ny,nx) spec-driven | D2Q9 | 0.00e+00 | PASS |
| `channel_kernel_numpy.stream` | (19,nz,ny,nx) | D3Q19 | 0.00e+00 | PASS |

### 物理臂：剪切波衰减 vs 解析 — 2/2 PASS
| arm | tau | nu=(tau-1/2)/3 | nu k^2 | 振幅误差 max | 拟合衰减率误差 | eps(in-repo vs ref) | 判定 |
|---|---|---|---|---|---|---|---|
| P1 | 0.8 | 0.1000 | 9.638e-04 | 0.1505% | 0.0514% | 1.60e-12 | PASS |
| P2 | 1.0 | 0.1667 | 1.606e-03 | 0.0001% | 0.0000% | 5.50e-13 | PASS |

### 冻结参考数据
`refdata/pointwise_stepwise_d2q9_v1.npz`（2f6349a06f4746d1…，21 个数组，float64，
lbmpy 2.0）；复算复现 eps = 0.0（逐位相同）。

## ④ 结果图

* `figures/fig1_stepwise_eps.png`：7 条整步臂逐点误差随步数（对数轴，含 1e-10 门线）。
* `figures/fig2_streaming_audit.png`：6 条迁移核 vs 定义的 eps（0 → 图下限 1e-18）。
* `figures/fig3_shear_decay.png`：剪切波幅值 vs 解析指数（τ=0.8, 1.0，2000 步）。
* `figures/fig4_field_and_error.png`：u_x 云图 + 逐格点 |Δf| 误差场 + x=0 剖面。

## ⑤ 注意点（坑与局限）

1. **迁移真值即定义**：`np.roll` 与任何正确的收集式实现代数等价，因此该臂只能暴露**索引/方向序/轴序**缺陷，
   不能暴露模型级错误；它的价值在于把本仓 6 条独立实现钉在同一把尺上（本轮 6/6 逐位相同）。
2. **碰撞真值仍来自 lbmpy**：若 lbmpy 的可压缩 SRT 约定本身有系统性偏差，两侧看不出来（与第 1 轮同源风险）。
3. 整步门只做 D2Q9（D3Q19 只在迁移普查中出现），三维整步仍是欠账。
4. 冻结数据是**参考侧**（lbmpy+定义迁移），不是本仓输出；篡改由 manifest 的 per-array sha256 拦截。
5. 动量守恒指标必须用**总量**并除以总质量：剪切波的总动量为 0，用 |Σj| 做分母会得到无意义的相对漂移
   （本轮首版即踩此坑，8.4e-1 的"漂移"实为逐点动量场变化，属物理衰减）。

## ⑥ 误差判定（逐项显式回答是否 <3%）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 整步逐点场（30 步，7 臂） | lbmpy 碰撞 + 定义迁移 | eps_inf ≤ 3.80e-14（门 1e-10） | 是（严 4 个量级以上） |
| 剪切波幅值（τ=0.8） | A0 e^{−νk²t} | 最大相对误差 0.15% | 是 |
| 剪切波幅值（τ=1.0） | 同上 | 0.0001% | 是 |
| 衰减率（拟合） | νk² | 误差 ≤ 0.0514% | 是 |
| 迁移核 6 条 | 拉式定义 | eps = 0（逐位） | 是（精确） |

**结论**：物理臂最大误差 0.15% < 3%（远优于 owner 15% 上限）；代码等价臂 7/7、迁移普查 6/6、
守恒与冻结数据复现全绿（35 例测试通过：本轮 18 + 第 1 轮 17）。未发现 TensorLBM 整步/迁移核缺陷——
这是一次**验证欠账清偿**（negative result 也如实记录：本轮没有找到 bug）。

## 复现

```bash
python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise/run_xval.py     # 需 lbmpy
python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise/make_refdata.py  # 冻结参考
python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise/make_figures.py
python -m pytest tests/test_xval_pointwise_stepwise_refdata.py -q             # 无 lbmpy 也跑
```
