# CASE-LBM-SQUARE-CYL-NORBERG — Cd 的第二条独立路径：控制体积动量平衡

锚点：`CASE-LBM-SQUARE-CYL-NORBERG`（方柱绕流 Re=200，阻塞比 1/20，指标 St 与 Cd）。
本文只报告本单新增的**力测量交叉校验能力**及其实测结果；Re=200 的 St/Cd 数值判定仍是 WITHHELD（见 §5）。

## 1. 问题：Cd 只有一条代码路径

本仓方柱算例池的全部 Cd 都来自 `tensorlbm.boundaries.compute_obstacle_forces`
（Ladd 动量交换格点求和）。这是**唯一**路径，因此该例程里任何符号/权重/归一化错误
都会被「与文献对比」静默吸收——因为对比的参照值也被同一个数缩放。判据先于动手：
需要一个与原路径**不共享力记账**、但共享同一 collide/stream/boundary 链的独立估计量。

## 2. 方法：控制体积（CV）动量平衡

对包围柱体的矩形控制体取整周期时间平均，定常 x 向动量定理给出

    D = - ∮_CV ( ρ u_x u_j + p δ_xj ) n_j dl ,   p = ρ/3 （D2Q9, c_s²=1/3）

四个面（左/右/上/下）按外法线取通量，`Cd_CV = D / (½ ρ0 u_in² D_side)`。
与被测路径的差别：CV 估计量**不接触**边界 link 力记账，只用到宏观场与守恒律。
实现：`src/tensorlbm/drag_budget.py::momentum_balance_drag`（numpy，无 torch 依赖）。

累加器（`benchmarks/verified/square_cylinder_re200/budget_check.py`）在整周期窗口内
累加 `ρu_xu_x`、`ρu_xu_y`、`p` 的时间平均（注意是**乘积的时间平均**，周期脉动对
Reynolds 通量的贡献必须计入），CV 取柱体上游 2D、下游 5D（不含出口 sponge），横向取满通道。

## 3. 解析校核（先证估计量本身正确，`tests/test_drag_budget.py`，6 passed）

| 人造场 | 解析解 | 断言 |
|---|---|---|
| 均匀流 (u=U, ρ=1, p=1/3) | Cd = 0 | 四面通量两两相消 |
| 均匀流 + 压差 Δp=0.5，ny=20，D_side=20 | Cd = Δp·H/(½U²D) = 1.0 | 精确相等（rel 1e-12） |
| 出口单侧动量亏损带 宽=D、深 a=0.5 | Cd = 2a(2−a)·D/H = 1.5 | 精确相等，且 Cd>0（阻力方向正确） |
| 均匀横向通量 u_y=0.3 | Cd = 0 | 上下面通量相消 |
| 随机场 | — | 四面之和 = net_flux，drag = −net_flux |
| 形状/参数非法 | — | 抛 ValueError |

这些是**不变量断言**（精确值/符号/可加性），不是「函数能跑」。

## 4. 端到端冒烟（真实 LBM 链，非收敛算例）

`runs/m17121_budget/smoke_D8/budget.json`（D=8, Re=200, B=1/20, 600 步, 16 线程, 11 s）：

| 量 | 值 |
|---|---|
| Cd_ladd（动量交换） | 1.4325 |
| Cd_CV（控制体积） | 1.4088 |
| 相对差 | **1.66 %** |
| St（600 步 < 1 个脱落周期） | 0.2993（**无效值**，仅证明管线可跑） |
| 网格 / CV | 208×160；上游 2D、下游 5D |

结论：两条完全不同的力估计路径在真实场量上给出 1.7% 以内的一致性 → 力记账链路无符号/量级错误。
注意 D=8 且仅 600 步，Cd 绝对值本身不具判定意义（未收敛、非整周期平均），故本节**不作为**锚点验收证据。

## 5. 局限与未完成（honest）

1. **锚点数值判定仍未完成**：Re=200、B=1/20 的 St/Cd 需要 24–30 个脱落周期的时间平均。
   本单已将算例（D=24/32，B=1/20；B=1/10 与 1/40 阻塞比扫描）与 CV 交叉校验（D=16/24）
   递交集群（cj1420 / cj1421 / cj1426），但队列积压（>560 作业）在本单时限内未启动；
   99 控制面 load15≈118（nproc=128）触发容量纪律，且本地工具单次调用上限 120 s，
   D=12/9000 步的本地尝试被超时收割——因此**本单不产出 Re=200 的 St/Cd 判定**。
2. **参照行仍 WITHHELD**：锚点原文（Norberg 1993 JWEIA 49:187；真正拥有 Re=200/B=1/20 工况的
   Sohankar–Norberg–Davidson 1998 IJNMF 26:39）在本环境不可获取（arXiv/ar5iv 可通，出版商与
   学术搜索引擎全部 403/429/不可达；OpenAlex 定位到的 OA 副本亦 403）。不做替代数字：宁可留空。
   故 `reference.json` 的 Re=200 行维持 `ref: null` + 理由，锚点验收判据仍不可机器裁。
3. CV 横向面取满通道（含两侧自由来流 Dirichlet 节点），故上下面通量含边界层外的
   极小残余；`p=ρ/3` 为 LBM 状态方程近似，压缩性误差 O(Ma²)，本案 u_in=0.05 → 0.25% 量级。
4. 累加器按 `sample_every` 抽样求平均，抽样间隔与脱落周期不整除时引入 O(1/N) 相位偏差；
   整周期窗口 + ≥6 周期平均可压到 1% 以下。

## 6. 复现

```bash
pytest tests/test_drag_budget.py -q                     # 解析校核（6 passed）
PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/budget_check.py \
  --D 8 --re 200 --steps 600 --blockage 20 --threads 16 --out runs/smoke   # 冒烟
# 集群（≥14 周期）：见 docs 上文 cj1426 的命令行
```
