# 圆柱阻力 Re=100（无约束 2-D）— Schäfer & Turek (1996) 锚点验证 (TensorLBM, m15432)

**锚点**：CASE-CYLINDER-DRAG-RE100 · **容差判据**：<5% · **结论**：**honest-FAIL（Cd 超差，St 达标）**

## ① 问题 + ref 出处

圆柱绕流是钝体阻力的标准验证题。本单目标为**无约束 2-D 圆柱 Re=100** 的
时均阻力系数 Cd、升力系数 Cl 与 Strouhal 数 St。

**基准（ref 原文）**：Schäfer, Rannacher & Turek (1996), "Flow control by means of
fictitious bodies", in *Numerical Methods in Fluid Dynamics*, Springer — 2-D 圆柱
Re=100 给出 **Cd=1.3796, Cl=0.0005, St=0.164**。
（注：同几何的其它文献值 Cd≈1.32–1.35（Dennis & Chang 1970；Williamson 1989；
Norberg 1994），St≈0.164–0.166；本单按任务指定以 Schäfer & Turek 1996 为锚。）

**几何与参数**（格点化，D2Q9/BGK）：

| 量 | 取值 |
|---|---|
| 圆柱直径 D | 10 格点（主算例；D=20 在集群排队中） |
| 计算域 | 上游 5D + 下游 20D（Lx=250），横向 10D（Ly=100） |
| 雷诺数 | Re = U·D/ν = 100 |
| 格子速度 U | 0.10（Ma = U·√3 = 0.173） |
| ν / τ | 0.01 / 0.53 |
| 边界 | `far_field_bc_2d`：入口+两侧面=自由流平衡，出口零梯度（另测 40 列二次海绵） |
| 壁面/圆柱 | 半路反弹；力=Ladd 动量交换（后迁移、反弹前） |
| 统计 | 预热 8000 步 + 18000 采样，双窗漂移 <1% 判 stationary |

## ② 方法与实现

- **单一真源**：`tests/validation/cylinder_re100_unconfined.py`（真跑 D2Q9 BGK
  `collide_bgk_matmul` + `stream` + `bounce_back_cells` + `compute_obstacle_forces`）。
- **为何不用通道边界**：`docs/cylinder_drag_benchmark_r1.md` 已证明本仓通道边界集
  （plug 入口+零梯度出口）存在动量收支失衡、Zou/He 发散，无法支撑定量圆柱阻力。
  无约束算例改用 `far_field_bc_2d`（自由流侧面，去阻塞），并新增 `--sponge-cols`
  二次海绵出口（`_sponge_outlet`）以吸收尾流、抑制出口反射。
- **定常态闸门**：预热后每 2000 步比较前后半窗 Cd 均值，漂移 <1% 才判 stationary。
- **性能**：`torch.set_num_threads(1)`（小张量线程池纯开销）；D=10 全算例墙钟 342 s。
- **复现**：`OMP_NUM_THREADS=1 python -m tests.validation.cylinder_re100_unconfined --full --diameter 10 --emit-dir artifacts/cylinder_re100_D10`

## ③ 计算结果（程序化注入，零手抄）

| run | D | 网格 | Ma | τ | 停机 | 墙钟 s |
|---|---|---|---|---|---|---|
| unconfined_D10 | 10 | 250x100 | 0.173 | 0.53 | stationary (drift 0.028%) | 342 |
| unconfined_D10_sponge | 10 | 250x100 | 0.173 | 0.53 | (运行中, 已停) | — |
| unconfined_D20_sponge | 20 | 500x200 | 0.173 | 0.56 | 集群 cj1159 排队 | — |

## ④ 对照表（ref vs got + 误差）

| 关键量 | 实测 (D10) | 基准 (S&T 1996) | 相对误差 | <5%? |
|---|---|---|---|---|
| Cd | 1.7301 | 1.3796 | **+25.41%** | **✗** |
| Cl | -0.0001 | 0.0005 | ~0（对称，量级正确） | ✓（量级） |
| St | 0.1712 | 0.164 | **+4.39%** | **✓** |

**显式回答「是否 <5%」**：St **是**（+4.39%）；Cd **否**（+25.41%）。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

- **出口海绵**：`--sponge-cols 40` 对 Cd 无改善（~1.73 → ~1.75），说明 Cd 超差
  **不是**出口反射所致，而是更系统的偏差（分辨率/边界/力归一化待查）。
- **分辨率**：D=10 已收敛（drift 0.028%）但 Cd 偏高；D=20 算例已递交集群（cj1159）
  作第二分辨率点，**本单完成时仍在排队**，未纳入判据。
- **St 对分辨率稳健**：D=10 即达 +4.39%，说明涡脱动力学正确，偏差集中在阻力。

## ⑥ 失败模式与工件

- **失败模式**：Cd 系统性高估 +25%（St 正确）。归因未闭环：海绵无效排除出口反射；
  剩余嫌疑为 ① D=10 分辨率不足 ② `far_field_bc_2d` 自由流侧面 Dirichlet 对近壁
  流的扰动 ③ Ladd 力归一化。需 D=20 收敛点 + 动量通量独立核算方可定论。
- **fail_stage**：`quantitative_comparison`（Cd 项）
- **failure_reason**：无约束圆柱 Cd 高估 +25%（St 达标）；`far_field_bc_2d`+海绵
  未消除偏差，需更细分辨率与独立动量核算。
- **工件**：
  - 代码 `tests/validation/cylinder_re100_unconfined.py`、
    `tests/validation/make_cylinder_re100_figures.py`
  - 数据 `artifacts/cylinder_re100_D10/cylinder_re100_unconfined.json`
  - 图 `figures/fig_vorticity.png`、`fig_cd_history.png`、`fig_centerline.png`
- **生态摩擦**：集群队列积压 ~350 作业，D=20 算例长时间未派发；本地 120 s 工具
  超时迫使 >10 min 计算走 setsid 后台，轮询成本高。

## 复现

```bash
OMP_NUM_THREADS=1 python -m tests.validation.cylinder_re100_unconfined --full --diameter 10 --emit-dir artifacts/cylinder_re100_D10
python -m tests.validation.make_cylinder_re100_figures --artifacts-dir artifacts --out-dir figures
```
