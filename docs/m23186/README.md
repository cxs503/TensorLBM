# m23186 — TensorLBM 能力注册表试点（锚点 CAP-REG-TLBM）

## 1. 交付了什么

| 交付物 | 路径 | 说明 |
|---|---|---|
| 注册表模块（微服务式自注册 + 双模式 dump CLI） | `src/tensorlbm/registry.py` | `@capability(...)` import 时自注册；`REGISTRY` 单例；`--static` AST 模式不 import 包、免 torch；stdout 单 JSON：`{source_commit, capabilities, problems, summary}` |
| 全仓诚实盘点（25 项） | 18 个模块内的 `@capability` 装饰器 | 碰撞 15 / 多相 3 / 声学 1 / 结冰 1 / AMR 1 / backend 1 / 力采样 1 / 可微标定 1 / 自由面 1 |
| 一致性审计（虚标 diff + 逐条处置） | `scripts/m23186_capability_registry_audit.py` → `docs/m23186/registry_audit.{md,json}` | 接线/测试/证据/文档四类虚标，逐行给处置（改码 / 改文 / 在案） |
| 碰撞族误差带复现 | `scripts/m23186_collision_family_verification.py` → `docs/m23186/collision_family_verification.json` | D3Q19 Poiseuille 解析剖面 + MRT 反 mock 探针 |
| 平台同步 | `scripts/m23186_capability_sync.py` → `docs/m23186/capability_sync_receipt.json` | `POST /api/v1/tensor/capabilities/sync`：`upserted=25, tombstoned=1` |
| 契约测试 | `tests/test_capability_registry.py`（10 项） | 含「免 torch 静态模式」「静态↔import 逐字段一致」「伪 mature 被本地拦」|

## 2. 注册表纪律（与平台 422 门同构）

- `mature` ⇒ 必须 `wired and verified`（服务端 `POST /sync` 同款门；本地在注册时就抛 `MaturityDisciplineError`，不让假成熟走到看板）。另外 `verified` 必须有 `evidence.grade ∈ {ANALYTIC, XVALUE}` 与 `evidence.max_error_pct`。
- `planned` ⇒ 必须带 `evidence.feature_ledger` 锚（否则没有 roadmap 依据），且不得 `verified`。
- `endpoint` ⇒ 只允许 `cli:/route:/api:` 三种 scheme，且只给 `wired=True` 的能力（TensorPhysics 可调用清单）。
- 指针解析：`module_path` / `entry` / `test_path` 全部对工作树求解，指不到就是 `error`（这是散文文档做不到的）。

## 3. 家底盘点（诚实分级）

- **mature（4，全部 verified+wired）**：`bgk / trt / rlbm / mrt` 的 **D3Q19** 四个族。依据是同一口径的 Poiseuille 解析剖面误差：0.4122 / 0.5709 / 0.4122 / 0.4124 %（< 15% 验收线，且 < 1%）。
- **developing（21）**：D3Q27 四族、D2Q9 两族（只有机理测试、无解析误差带）；KBC（真熵条件码，但主 stepper 未接线 ⇒ `wired=False`）；CM/cascaded（引用方仅 `free_surface_lbm` 与契约模块）；`advanced_collision` 的 D3Q27 二阶重建（契约明示非 CM/KBC）；Shan-Chen 力、cAC+HCH 相场框架（高密度比发散在案）、声学 FWH、结冰 Macklin 密度式、AMR reflux、Triton 融合步（本环境无法执行 ⇒ grade=NONE）、动量交换力、可微闭包标定、自由面 VOF。
- **planned（0）**：本单没有新增 roadmap 项；`feature_ledger↔registry` 对账通道已由 `planned ⇒ feature_ledger` 规则预留。
- **MRT「此前是 mock」的处置**：HEAD 上 `solver3d.collide_mrt3d` / `solver.collide_mrt` 都是显式 MRT 矩阵实现（m23153 的 2D MRT 分支尚未并入 HEAD）。为免「看着像真码其实是 BGK 别名」，加了**反 mock 探针**：随机非平衡场上 MRT 与 BGK 差 15.4%、改变 `s_e` 使结果变动 33.2%（别名应为 0）——探针写进注册表证据与单测 `test_mrt_is_not_a_bgk_alias`。

## 4. 虚标差异清单与处置

审计输出 4 条发现，**open_findings = 0**（全部逐条裁定）：

| code | capability | 处置 |
|---|---|---|
| `DOC_CLAIM_CONFLICT` | `lbm.collision.mrt.d3q27` | 在案：`docs/d3q27_composition_evidence_r1.md:17` 的「已验证」说的是 composition **范围**，按行触发；D3Q27 MRT 无解析误差带，developing 是有意的 |
| `DOC_CLAIM_CONFLICT` | `lbm.collision.mrt.d2q9` | 在案：同上，跨句上下文误判 |
| `DOC_CLAIM_CONFLICT` | `lbm.amr.reflux_restrict` ×2 | **改文**：README 的 "is production (`stable`)" 改为 "audited stable … registry `developing`（有机理/守恒证据、尚无外部基准误差带）"；历史报告归档件写「在案，不改写归档」 |
| （本单自查）`lbm.free_surface.vof_sweep` | wired/tested 虚标 | **改码**：`wired=False, tested=False`（src/ 无生产模块引用该函数、cited test 未直接覆盖） |
| （本单自查）`lbm.phasefield.interface_width_probe` | test_path 指错 | **改码**：`tests/test_allen_cahn_phasefield.py`（真正覆盖该函数的测试） |

## 5. 怎么验证的

```bash
PYTHONPATH=src python src/tensorlbm/registry.py --static --strict      # 25 caps, 0 error
PYTHONPATH=src python src/tensorlbm/registry.py --parity               # static↔import parity ok=True
PYTHONPATH=src python scripts/m23186_capability_registry_audit.py      # findings 4, open 0
PYTHONPATH=src python scripts/m23186_collision_family_verification.py  # 4 族误差 + 反 mock
PYTHONPATH=src python scripts/m23186_capability_sync.py                # upserted=25 tombstoned=1
PYTHONPATH=src python -m pytest tests/test_capability_registry.py -q   # 10 passed
```

`test_static_mode_runs_without_importing_the_package` 用「假 torch 包直接 raise ImportError」证明静态模式**真的**不 import 包/不需要 torch（99 上 `capability_sync` 守护按此契约调用：`python <repo>/src/tensorlbm/registry.py --static`，不能用 `-m`，`-m` 会先执行包 `__init__`）。

## 6. 注意点 / 局限

1. **`python -m tensorlbm.registry` 会 import 包**（Python 语义），torch-free 主机必须走文件路径调用；这是本单刻意在 CLI 与文档里写死的一条。
2. **import 模式在多 worktree 机器上会被 editable install 遮罩**：`registry.import_registry()` 会把本工作树 `src/` 前置并在 `imports.shadowed_package` 里如实记下被遮罩的路径。
3. `wired` 的机械判据是「src/ 内存在非测试/非契约模块引用该 entry」。这比「public 导出即算接线」严格：KBC/CM 因此被判 `wired=False`——它们能被 `import tensorlbm` 调到，但没有 stepper 分派。
4. **TensorLBM 没有 console script**（`pyproject.toml` 无 `[project.scripts]`），所以 endpoint 只能用 `api:<dotted callable>` 形式给 TensorPhysics；`cli:` 形态目前无真对象可指——这是 TensorPhysics 可调用清单的现存缺口。
5. 本单只登记 25 项（家底的一个诚实切片：每个学科域至少一项 + 碰撞全族）。仓内仍有大量模块未登记；把它们纳入是下一位的机械工作（装饰器 + 审计脚本即可）。
6. 误差带的算例是 `nx=24, ny=10, nz=2, steps=200` 的粗网格（受单机 10 分钟门限约束）；`tests/test_lbm_verification.py` 的 `ny=12/steps=1000` 口径给出 0.338/0.0012/0.338/0.343 %，注册表里两个口径都写了。

## 7. 下一步（供后续单接续）

- 用同一装饰器把 2003 单积累的其余能力纳入（AMR 各模块、多相各变体、声学/结冰全流程、`backends/{torch,mindspore,paddle}` 三后端）。
- 给 `developing` 里的高价值项补解析/外部基准误差带（D3Q27 四族、Shan-Chen Laplace 压差、FWH 单极/偶极源、AMR 2:1 通量守恒），成熟后即可升 `mature`。
- 为 TensorLBM 增加真正的 console script（`tensorlbm-run --case <x>`），把 endpoint 从 `api:` 升级成 `cli:` 契约。