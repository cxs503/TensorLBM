# GAP-PP-3-adjoint-ootc：反演半场收敛 + 内核可导入化（实质增量）

- 仓: TensorLBM · 分支 agent/m22781 · 锚点 `GAP-PP-3-adjoint-ootc`
  （out-of-core adjoint / 截断 BPTT，长时域反演内存）
- 前置：mission 20040（`docs/GAP_PP_3_ADJOINT_OOTC_ACCEPTANCE.md`）已过两条梯度/内存门，
  但第 6 节诚实标注 **τ 反演 UNFINISHED（rel 20%）**，且其引用的
  `artifacts/adjoint_ootc_acceptance/`、`figures/fig1..fig5` 在 main 树中不存在
  （工件只落在作业节点，数字不可复算）。本单针对这两点做实质增量。

## 1. 本案（物理背景 / 算例）

D2Q9 BGK 二维周期域内高斯密度脉冲衰减：反演参数 τ（ν=(τ-1/2)/3），
目标 `J(τ)=½‖u_T(τ)−u_ref‖²`，`u_ref = u_T(τ_true=0.8)`（**无偏**参照）。
本单实跑规模 N=32、T=600、k=50（本机计算门 120 s 上限内的最大可信算例）；
梯度/内存两条验收门沿用 mission 20040 的 N=128/T=10⁴ 结果。

## 2. 做了什么（两件，均为能力级）

### 2.1 内核搬进可安装树（可导入性缺陷）

`lbm2d.py / bptt.py / ootc.py` 原本住在嵌套影子树
`TensorLBM/src/tensorlbm/adjoint/`，只能靠测试里的 `sys.path.insert` 摸到，
安装后的 `tensorlbm` 包**取不到**这些实现。现搬迁为正式子包：

    src/tensorlbm/adjoint_ootc/{lbm2d,bptt,ootc,inversion}.py

于是 `from tensorlbm.adjoint_ootc import ootc_gradient, invert_tau` 在装好的库里
直接可用（测试 `TestImportability` 钉死 `__file__` 必须位于 `src/tensorlbm/`、
不得落在影子路径）。影子目录仅留说明文档，不再执行任何代码。
注意与 `tensorlbm.adjoint` 区分：后者是 frozen-field 替代模型
（`shape_derivative_only=True`，禁用于驱动优化）。

### 2.2 反演驱动器：从"爬行"到收敛

第一轮的驱动器是 `alpha=2e-2` 步长折半的陡降，12 步停在 |Δτ|/τ=20%。本单先做
**地形诊断**（`artifacts/adjoint_ootc_acceptance/inversion_convergence.json`）：

| τ | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.10 | 1.20 | 1.40 |
|---|---|---|---|---|---|---|---|---|
| J | 2.95e-3 | 3.45e-4 | **0** | 3.54e-5 | 2.73e-4 | 4.04e-4 | 4.93e-4 | 5.86e-4 |
| dJ/dτ | −5.56e-2 | −9.23e-3 | ~0 | +1.64e-3 | +1.52e-3 | +1.09e-3 | +7.15e-4 | +2.76e-4 |

J 只在极小点邻域凸：`g=dJ/dτ` 在 τ≈0.9 达峰后**衰减到 0+**（脉冲幅度
A(τ)~exp(−c(τ−1/2))，J'=(A−A_ref)A'），即 τ≳0.95 处 `dg/dτ<0`、J 凹。
从该区起步的裸 Newton 步会被推向外侧（实测 1.20→1.32→1.45 撞边界），
这正是第一轮"越走越慢"的机制。

替换为 `invert_tau`：**包围区间保护的 Newton/割线混合** ——
先用 `g(lo)<0<g(hi)` 建立括号（不成立则 `converged=False, reason="root not bracketed"`），
局部割线曲率 `dg/dτ` 为正时走 Newton、为负时退化为 J 下降步（步长按 J 是否下降自适应），
每步都被钳在括号内（信任域 ≤ 半个括号宽）。

## 3. 结果

| 门 / 量 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 恢复 τ（无偏目标，N=32 T=600） | — | 0.799999999999964 | — |
| 相对误差 vs τ_true | < 3% | **4.47e-14** | PASS |
| 收敛标志 | 必须显式 | `converged=True`（gradient tolerance，20 次梯度评估） | PASS |
| OOTC 梯度 vs 前向中心差分 | <1e-5 | 7.95e-08 | PASS |
| 峰值内存随 T（T=300 vs 600） | 与 T 无关 | 0.0078 GiB / 0.0078 GiB | PASS |
| 第一轮对照 | — | rel 2.0e-01，未收敛（预算耗尽） | 已修复 |

**目标规格性偏差（新发现）**：若沿用第一轮的 `u_ref = 0.9·u_T(τ_true)`，
J 的极小点本身偏离 τ_true——本单实测 τ_min=0.816（偏 2.0%），驱动器以
0.27% 精度命中该极小点。故第一轮的 20% 是「驱动器未收敛」+「目标有偏」叠加，
不是梯度精度问题。这一点对论文第 4 节的反演设定有直接影响：反演目标必须用
无幅度缩放的终端场，否则误差预算是被目标定义污染的。

## 4. 图

- `artifacts/adjoint_ootc_acceptance/figures/fig_inv1_landscape.png`
  （a）J(τ) 地形 + 反演迭代轨迹；（b）dJ/dτ 变号与零点
- `artifacts/adjoint_ootc_acceptance/figures/fig_inv2_convergence.png`
  相对误差与归一化梯度随迭代下降，3% 精度线与 1e-5 梯度门同图
- `artifacts/adjoint_ootc_acceptance/figures/fig_inv3_memory.png`
  OOTC 峰值内存与 T 无关

## 5. 注意点 / 局限

1. **规模**：反演实跑 N=32/T=600（本机 120 s 计算门内的最大可信算例）。
   N=128/T=10⁴ 的梯度与内存门沿用 mission 20040 的集群实测（内核未改动，
   仅搬移位置）；反演驱动器的额外开销是 ~20 次梯度评估，与规模无关。
2. **单参数**：τ 是唯一标量参数；边界/多参数反演仍在 BPTT 框架外
   （`return_field=True` 给出的 dJ/df⁰ 可复用于初始场反演）。
3. **非凸远场是物理性的**（幅度指数衰减 ⇒ J 渐近常数），不是数值噪声；
   因此任何 Newton 类方法都必须带括号/信任域，这点写进了模块文档与测试。
4. 未做的：mission 20040 缺失的 `fig1..fig5`（全尺度门图）未在本单重建；
   本单只补了反演半场的可复算证据。建议另立小单：把
   `scripts/verify_ootc_acceptance.py` 的输出目录从节点 /tmp 改到仓内并重跑一次
   （与 `JOUR-DIFFLBM-ARTIFACT-INREPO` 同类债务）。

## 6. 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 恢复 τ（无偏目标） | τ_true=0.8 | rel 4.47e-14 | 是 |
| dJ/dτ（OOTC） | 前向中心差分 h=1e-4 | rel 7.95e-08 | 是 |
| 恢复 τ（0.9 缩放目标） | 该目标的极小点 τ_min=0.816 | rel 2.72e-03 | 是 |
| 恢复 τ（0.9 缩放目标） | τ_true=0.8 | rel 1.72e-02 | 是（但偏差来自目标定义，非求解器） |

**关键量是否 <3%？** 是——反演残差 4.5e-14、梯度一致性 8e-8，均为机器精度量级；
唯一 >1% 的项（1.72%）已定位为目标规格性偏差并单独量化。

## 7. 复算

```bash
PYTHONPATH=src python scripts/ootc_inversion_acceptance.py --N 32 --T 600   # ~71 s
python -m pytest tests/test_adjoint_inversion.py tests/test_adjoint_ootc.py -q
```