# 审计：wall/refinement 组合门（fail-closed gate）的可失败性

- 锚点（m16911）：`tests/ci_quarantine.txt` 中的
  `tests/test_wall_refinement_combination_gate.py::test_unavailable_collision_contract_cannot_be_promoted_by_baseline_shape`
- 工件：`artifacts/wall_refinement_gate/falsifiability_audit.json`（机器生成；本文件所有数字由该 JSON 渲染，零手抄）
- 图：`figures/wall_refinement_gate_admission_matrix.png`
- 工具：`tools/wall_refinement_gate_falsifiability.py`（`PYTHONPATH=src python3 tools/wall_refinement_gate_falsifiability.py`）
- EDIT_WHITELIST：`src/`、`tests/`、`tools/`、`docs/`、`artifacts/`、`figures/`

## 0. 结论摘要

| 编号 | 严重度 | 发现 | 状态 |
|---|---|---|---|
| F1 | **高** | 碰撞族轴（`WITHHELD_UNSUPPORTED_COLLISION`）在公开 API 下不可达，且其唯一测试被 CI 隔离为非严格 `xfail` ⇒ fail-closed 守卫既死又不可见 | 已修（注入式见证 + 全码可达性扫描 + 隔离条目销账） |
| F2 | 中 | `advanced_collision_contract` 模块 docstring 与能力矩阵互相矛盾（文档称只有 BGK/TRT/RLBM 可用，矩阵登记 7 族 × 2 格点） | 已修（docstring 更正为 14/14 与真实入口点） |
| F3 | 中 | 门只能表达 MRT/CM/KBC；BGK/TRT/RLBM/CUMULANT 完全无法构造（`TypeError` 兜住，安全但无判决、无文档）；且门内 `CM`（cascaded 中心矩）与矩阵中 `CUMULANT` 同名不同物，易误用 | 已登记为显式子集守卫测试 + 本文件记录，**不改门语义**（扩大准入面须域内评审） |
| F4 | 低（量化） | 准入面极窄：18/2304 = 0.78 % 组合被允许 | 非缺陷，作为 fail-closed 强度基线记录 |
| F5 | —（裁决旁证） | 另两条隔离条目现场裁决：`test_init_g_with_velocity` = API 漂移；`test_fs_runner_does_not_rebuild_mass_from_fill_between_steps` = 断言失败但打印值相等（dtype/索引待查） | 未修，留待专门单据 |

## 1. 案例介绍：这个门是什么、为什么值得审

`src/tensorlbm/boundary/wall_refinement_combination_gate.py` 不是调度器，而是**能力主张门**：调用方把
`(lattice, collision, wall_treatment, refinement, geometry_ownership, geometry_kind, physics)` 交进来，
只有返回 `GateStatus.ALLOWED` 才允许跑。设计意图 fail-closed：证据不全 → `WITHHELD`，并给出机器可判的
`reasons`（拒因码）与 `missing_required_evidence`。

审它的理由：**没有任何算例能证明"门能拒"**——门要是错了，下游所有壁面函数/AMR 组合验证的可信度都被污染。
这是上游守卫，其失效不可观测。

## 2. 怎么做的（方法与关键设置）

1. **穷举扫描**：对门能表达的全部组合空间（2 格点 × 3 碰撞族 × 4 壁面处理 × 4 加密 × 4 几何归属 ×
   3 几何类型 × 2 物理模型）逐点求判决，统计 `ALLOWED/WITHHELD` 与拒因直方图，共 2304 点。
2. **拒因可达性**：门定义了 9 个 `WITHHELD_*` 拒因码；把"公开输入可达"与"须注入能力矩阵才可达"
   分开统计，永不可达者 = 死代码。
3. **差分臂（因果性证明）**：依次把矩阵里**一个** `(lattice, family)` 单元标为 unavailable 后重扫，
   检查翻转的判决集合**恰好等于**使用该单元的全部判决（多翻/少翻都算失败）。这是"碰撞族轴不是装饰"的机器证据。
4. **不变量测试**：`tests/test_wall_refinement_combination_gate.py` 新增 3 项测试（见 §7）。

## 3. 计算结果（数字表，全部由 artifact JSON 渲染）

**判决分布**：2304 点中 `ALLOWED` = 18、`WITHHELD` = 2286（允许率 0.78 %）。

**拒因直方图**（同一判决可含多个拒因码）：

| 拒因码 | 命中点数 |
|---|---|
| `WITHHELD_D3Q27_WALL_FUNCTION` | 288 |
| `WITHHELD_NON_BASELINE_REFINEMENT` | 576 |
| `WITHHELD_REFINEMENT_IBM` | 576 |
| `WITHHELD_REFINEMENT_MULTIPHASE` | 864 |
| `WITHHELD_UNKNOWN_COMBINATION` | 486 |
| `WITHHELD_WALL_FUNCTION_CURVED_WALL` | 384 |
| `WITHHELD_WALL_FUNCTION_IBM` | 384 |
| `WITHHELD_WALL_FUNCTION_WITH_REFINEMENT` | 864 |

**拒因可达性**：定义 9 个；公开输入可达 8 个；仅注入矩阵可达 ['WITHHELD_UNSUPPORTED_COLLISION']；**永不可达 []**。

**差分臂**（每次只抽掉一格能力）：

| 抽掉的矩阵单元 | 应翻转判决数 | 实际翻转数 | 恰好且仅这些 |
|---|---|---|---|
| D3Q19/MRT | 3 | 3 | OK |
| D3Q19/CM | 3 | 3 | OK |
| D3Q19/KBC | 3 | 3 | OK |
| D3Q27/MRT | 3 | 3 | OK |
| D3Q27/CM | 3 | 3 | OK |
| D3Q27/KBC | 3 | 3 | OK |

全部 6 臂 `flipped_all_and_only_that_family = True`（6/6）。

## 4. 图

`figures/wall_refinement_gate_admission_matrix.png`：左右两联（D3Q19 / D3Q27），纵轴 = 门可表达的碰撞族
（MRT/CM/KBC），横轴 = 壁面处理 × 加密类型（16 列），格内 `A`=ALLOWED / `W`=WITHHELD（绿/红），
形状固定为单层平面静态单相。图上可直接读出：只有"无加密 + NONE/STANDARD_STATIC/COMMON_WALL_FUNCTION"
三种壁面处理是绿的，任何加密（static_local/dynamic_amr/surface_shell）一律红——与 fail-closed 设计一致。

## 5. 注意点（坑与局限）

1. **漂移的期望**：原测试断言 `CollisionFamily.CM` 不可用。今天 `collision_capability_matrix()` 把
   **14/14** 个 (格点, 族) 全标 AVAILABLE（CM 由 `cascaded_collision` 的完整中心矩层级承担，不是
   `advanced_collision` 的二阶重构）。所以原测试是**期望过期**，不是门的逻辑错。正确做法不是删测试，而是把
   守卫改成注入式见证——矩阵哪天再丢一族，守卫仍能拒。
2. **隔离机制掩盖真相**：`tests/conftest.py` 把隔离节点改写成 non-strict `xfail`，于是"测试仍在真失败"
   在 CI 里**不产生任何红色**。F1 就这样存续至今。
3. **不能靠单点断言证明 fail-closed**：单点只能钉住某一形状；本审计用"全码可达性 + 差分因果性"两条不变量
   钉住守卫，而不是"跑通即通过"。
4. **局限（诚实声明）**：本审计验证门的结构性行为（拒得对不对、拒因会不会变死代码），**不**验证被准入组合的
   物理正确性——后者是各算例池条目的职责。
5. **命名陷阱**：门内 `CollisionFamily.CM` = cascaded central moment；矩阵另有 `CUMULANT`，两者不是一回事，
   而门无法表达 `CUMULANT`。使用者若把 "cumulant" 当 `CM` 传进来，会拿到错误族的判决。
6. **未扩展门语义**：把 BGK/TRT/RLBM/CUMULANT 加进门 = 扩大准入面，属域内评审范畴。本单只加"子集守卫测试"
   把现状钉住：一旦有人扩枚举，测试立刻红并指向本文件。

## 6. 误差判定（逐项，显式回答）

本单为**结构性审计**，被判量是布尔判决与计数，不是物理量，故无"相对基准误差"可用，改用**精确判据**：

| 判据 | 目标 | 实测 | 是否精确满足 |
|---|---|---|---|
| 拒因码死代码数 | 0 | 0 | 是（精确 0，非容差） |
| 差分臂"恰好且仅"翻转 | 全臂 | 6/6 臂 | OK |
| 隔离目标测试现状 | 非 xfail 且真绿 | 16 passed, 0 xfail | 是 |
| 门枚举 ⊂ 矩阵族集合 | 子集成立 | 差集 = BGK/TRT/RLBM/CUMULANT（显式断言） | 是 |

**是否 <3% 误差？** 不适用：本单没有与实验/解析基准对照的连续量，全部判据为精确相等/计数。
显式声明原因：审计对象是**代码契约**（布尔门 + 拒因码），不是物理场。

## 7. 验证（跑过什么）

```
python3 -m pytest tests/test_wall_refinement_combination_gate.py -q          # 16 passed（目标测试已离开隔离名单，非 xfail 真绿）
python3 -m pytest tests/test_wall_refinement_combination_gate.py \
    tests/test_advanced_collision_contract.py tests/test_capability_contract_drift.py \
    tests/test_collision_matrix_cross_validation.py tests/test_collision_auto_discovery.py \
    tests/test_cross_module_composition_matrix.py tests/test_general_capability_matrix.py -q
                                                                             # 160 passed, 1 skipped
PYTHONPATH=src python3 tools/wall_refinement_gate_falsifiability.py          # 工件 + 图
```

**未跑（not_run，不得记为通过）**：全量 `pytest tests/`（本机 load15 ≥ 100，容量纪律只做轻量活）；
GPU/SDAA 设备上的条目与其余 87 条隔离条目（缺硬件 / 属专门单据）。
