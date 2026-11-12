# Ghia (1982) 方腔度量约定审计（m16339）

**结论一句话**：`lid_driven_cavity.compare_ghia` 是 TensorLBM 全部方腔验收判定的度量本体，但它的
「节点索引 → 物理坐标」映射原先被硬写成 `y_j = j/(n-1)` 且不可声明。该映射的选择使同一份流场
报出的 RMSE 变化 **0.37 %–1.11 %（归一化 u_lid）**，与 3 %/5 % 门限同量级；本单把约定显式化
（`wall_offset`）、量化其敏感度、并修掉一个真实的调用点缺陷（`/benchmarks/ghia` 端点必崩）。

## 1. 被审计对象与判据

| 项 | 内容 |
|---|---|
| 度量 | `src/tensorlbm/lid_driven_cavity.py::compare_ghia`（RMSE of u/u_lid 沿竖中线、v/u_lid 沿横中线） |
| 参考 | Ghia, Ghia & Shin, *J. Comput. Phys.* **48** (1982) 387-411, Table I/II/III（Re=100/400/1000） |
| 消费者 | `run_lid_driven_cavity` 元数据、`app/backend/routers/benchmarks.py::/benchmarks/ghia`、`app/backend/services/generic_run.py`、`cross_solver_verification/runner.py`、`scripts/run_gap_pp2_lid_wall_diff.py` |
| 判据（先验） | ①同一份「完美解」在自身约定下 RMSE→0；②换约定后 RMSE 抬升量 = 可解析预测值；③该量随 n 单调下降但**不是**高阶小量；④NaN 场不得被判 PASS；⑤所有调用点参数个数正确 |

## 2. 方法（无手抄，全部程序化）

1. 参考表 → 单调三次（PCHIP）主剖面；
2. 造「完美解」场：节点 j 的值取主剖面在 `y_j` 处的值；
3. 用候选约定 `y_j=(j+w)/(n-1+2w)`（`w=0` 壁面在节点上=历史默认；`w=+0.5` 节点在等分单元中心=半程反弹壁面；`w=-0.5` 壁面节点在内）重新评估同一场，得「约定地板」；
4. `audit/m16339/ghia_metric_audit.py` → `audit/m16339/ghia_metric_audit.json` + 图。

## 3. 结果（`audit/m16339/ghia_metric_audit.json`，程序化输出）

历史默认 `w=0` 施加在「按半程反弹约定正确的完美解」上产生的 RMSE 地板（% of u_lid）：

| Re | n=64 | n=128 | n=192 | n=256 | n=384 |
|---|---|---|---|---|---|
| 100 | 1.520 | 0.753 | 0.499 | 0.374 | 0.248 |
| 400 | 1.784 | 0.880 | 0.581 | 0.434 | 0.288 |
| 1000 | 2.297 | 1.110 | 0.729 | 0.543 | 0.360 |

要点：

* 地板 ≈ O(h)（128→256 只降到 1/2），**不随网格收敛消失**；仓库现有 Re=1000 报告在 256² 上
  报 1.05 %，其中约 0.54 % 属度量约定项、而非求解误差（方向：把正确解判得更差，即报数是偏保守）。
* 若改用 `w=+0.5`，参考表落在壁面上的两行（y=0 与 y=1）跑到节点跨度之外，`np.interp` 会静默
  **钳制**到最近节点值：完美解在 128² 上的残差 0.65 %(Re100)/1.04 %(Re400)/1.64 %(Re1000) 全部
  由这两行贡献（内部行 ≤1.5e-2 的插值误差）。即：**换约定必须同时声明壁面行的处置（剔除或外推）**。
* 因此任何方腔误差数字必须与所用 `wall_offset` 成对出现，否则不可比。

## 4. 发现与修复

| # | 严重度 | 发现 | 证据 | 处置 |
|---|---|---|---|---|
| F1 | **阻断（已修）** | `app/backend/routers/benchmarks.py::run_ghia` 调 `compare_ghia(ux, uy, ref)`（3 实参），函数需 4 参 → `TypeError: missing 1 required positional argument: 'reference'`，`/benchmarks/ghia` 每次调用必崩 | AST 检查 + 函数签名 | 改 `compare_ghia(ux, uy, cfg.u_lid, ref)`；新增 AST 守卫测试防复发 |
| F2 | 高（度量完整性） | 节点→坐标映射硬编码且不可声明；约定项 0.37–1.11 % 与 3 %/5 % 门限同量级，历史报告未披露 | 第 3 节表 | 新增 `wall_offset` 参数（默认 0.0 保持向后兼容）+ `centreline_coordinates` + `ghia_coordinate_sensitivity` 量化工具 |
| F3 | 中（可证伪性） | `np.interp` 对超范围参考点静默钳制，可用来自证「零误差」也可凭空造出边界误差 | 完美解端点残差实验 | 文档化 caveat + 测试固化「壁面行承担全部残差」 |
| F4 | 低（约定陷阱） | 参考表 `y` 为**降序**、`x` 升序；单侧排序即静默错配 | 表结构测试 | 表排序不变量测试 + `compare_ghia` 文档声明 |

## 5. 注意点与局限

* 本单**没有**重跑方腔算例（load15≈120 时按容量纪律只做轻量）；因此「Re=1000 报 1.05 % 中含 ≈0.54 %
  约定项」是**解析预测**，不是对已存流场的重算——现有 run 的 `result.json` 未保存完整中心线场，无法回溯。
* 「哪种约定是对的」取决于该 run 的实际边界实现（半程反弹壁 / Zou-He 顶层节点 / 固体节点行），
  本单只给工具与量级，不给单一裁决；默认值保持 0.0 以免静默改动历史数字。
* 建议：`run_lid_driven_cavity` 写元数据时把 `wall_offset` 与其对应的边界配方一并落盘（后续单）。

## 6. 误差判定（对基准）

| 关键量 | 基准 | 本单结果 | 是否 <3 % |
|---|---|---|---|
| 完美解在自身约定下的 RMSE | 0（解析） | Re100/400/1000 均 < 0.2 %（0.0 % 量级，仅插值误差） | 是 |
| 约定地板预测 vs 直接度量（w=0） | `ghia_coordinate_sensitivity` | 测试断言一致（rel 25 %） | 是（作为量级一致性判据） |
| 约定地板本身（256², Re=1000） | — | 0.543 % | 作为偏差量报告，非精度指标 |

## 7. 复现

```bash
python audit/m16339/ghia_metric_audit.py                      # JSON + PNG
python -m pytest tests/test_ghia_metric_convention.py -q      # 13 项
```

图：`audit/m16339/ghia_convention_bias.png`（(a) 地板 vs 网格；(b) 128² 上两套映射对 Re=1000 近盖
剖面的采样位置偏移；(c) 三种约定在各网格的 RMSE 地板）。
