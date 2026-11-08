# 能力契约审计面扩面：4→8 契约、16→170 格，首次抓到「能力虚报」（2026-11-08, TensorLBM）

`mission 6598 · repo TensorLBM · branch agent/m6598 · 自找活 selfstart`

## 0 结论（一句话）

TensorLBM 有 8 份**带矩阵工厂**的能力契约，但机械闸门 `tensorlbm.capability_contract_drift`
（2026-11-08 上一单 m6499 建）只登记了 4 份、16 个能力格；本单把审计面扩到 **8 份 / 170 格**，
并修掉闸门自身的两个**盲区**，扩面后第一次运行即抓到 **18 个 error（分布在 physics 契约的 9 个格子）**：
其中 **4 格（porous_media 全族）指向的模块、函数、测试文件在本仓历史中从未存在**
（`git log --all -- src/tensorlbm/porous_media_common.py` 为空），**5 格（non_newtonian）指向的
`non_newtonian_common.py`/`non_newtonian_tau_eff`/`test_non_newtonian_common.py` 同样从未存在**。
即：能力契约在**虚报**——它声称 IMPLEMENTED + CONTRACT_TESTED，而没有任何代码或测试可解析。
修复后 **170 格 0 error**（余 2 条 warning 为如实披露），闸门行为由 **80 条测试**钉住
（含「注入缺陷必须报警」的负控）。

## 1 案例介绍（对象与背景）

能力契约是本库对外的**证据面**：每格用字符串声明实现入口（`entrypoint`）与测试证据
（`test_evidence`/`verification_evidence`），`status` 一律 `WITHHELD_*`（fail-closed：机制在、
物理未认证）。闸门把这两个字符串**对仓库解析**，让「指针腐烂」变成可机器裁的红。
上一单（m6499）只覆盖 `ibm / sliding_mesh / fsi / sixdof` 四份契约。

本单审计对象（实测口径）：

| 契约 | 工厂 | 格数 | 层次 | 状态约定 |
|---|---|---|---|---|
| `ibm` | `ibm_capability_matrix()` | 4 | evidence | `WITHHELD_*` |
| `sliding_mesh` | `sliding_mesh_capability_matrix()` | 6 | evidence | `WITHHELD_*` |
| `fsi` | `fsi_capability_matrix()` | 4 | evidence | `WITHHELD_*` |
| `sixdof` | `sixdof_capability_matrix()` | 2 | evidence | `WITHHELD_*` |
| `acoustics` | `acoustics_capability_matrix()` | 20 | evidence | `WITHHELD_*` |
| `physics` | `physics_capability_matrix()` | 45 | evidence | `WITHHELD_*` |
| `amr` | `local_refinement_capability_matrix()` | 75 | entrypoint | `WITHHELD_*` |
| `advanced_collision` | `collision_capability_matrix()` | 14 | entrypoint | `AVAILABLE`（豁免，见 §5） |
| **合计** | | **170** | | |

## 2 怎么做的（方法/流程/工具链）

1. **摊平扩面**：`CONTRACTS` 表 4 → 8 项。新纳 4 份契约各有不同的记录 schema，因此引入
   **两层次（tier）**而不是一刀切：
   * *evidence tier*（`acoustics`/`physics`）：有 `entrypoint` + 证据字段 + 验证级 → 入口解析、
     fail-closed、级词表、证据解析**全查**；
   * *entrypoint tier*（`amr`/`advanced_collision`）：schema **根本没有证据字段** → 只查入口解析。
     对这类契约要证据是编需求，静默跳过是不审计，两者都不对，所以在 `Claim.evidence_field`
     上把「没有证据字段」与「有字段但留空」区分开（后者对已测级仍是 `EVIDENCE_MISSING`）。
2. **修闸门盲区 A（序列矩阵）**：`_iter_capabilities` 只递归 `Mapping`；
   `physics_capability_matrix()` 返回 `list`，于是该契约**一格都收不到**——
   幸亏 `CONTRACT_EMPTY` 是 error 兜住了「假绿」，但契约本身仍未受审。现补 `list/tuple` 分支
   （不遍历 `set`，避免报告顺序不确定）。
3. **修闸门盲区 B（词表陈旧）**：`acoustics` 契约用 `ANALYTIC_VERIFIED`（「复现闭式解」，
   比 CONTRACT_TESTED 更强），闸门词表没有 → 把**诚实的强证据判成 UNKNOWN**。
   现纳入 `TESTED_LEVELS`，但因它仍须指名可解析测试文件，级不能凭空断言。
4. **模块级入口合法化**：`amr` 的入口是 `tensorlbm.amr_common`（模块，非 callable）。
   原闸门会 `ENTRYPOINT_NOT_CALLABLE`，属误报；现接受 `ModuleType`。
5. **策略外显**：`advanced_collision` 是可用性矩阵（`status=AVAILABLE`），若强套
   `WITHHELD_*` 不变量就是误报。故加 `ContractPolicy(fail_closed=..., reason=...)`，
   豁免**必须带理由**并写进 JSON 报告的 `policies` 节——豁免可见，不是静默。
6. **扩面后跑审计 → 逐格核实 → 修契约**（见 §3），每处修复都先确认「真实机制在哪」：
   `git log --all` 查文件是否曾存在；`grep` 查函数是否存在；再定 IMPLEMENTED 还是降级。

## 3 计算结果（数字表；命令可复算）

**扩面后首次运行（修复前）**：

| 指标 | 数值 |
|---|---|
| 受审契约 | 8 |
| 受审能力格 | 170 |
| error | **18** |
| 其中 error 分布 | `ENTRYPOINT_UNRESOLVED` × 9、`EVIDENCE_UNRESOLVED` × 9 |
| 涉及格子 | physics[21,22,24,25]（porous_media 4 格）+ physics[27,30,31,33,34]（non_newtonian 5 格） |

**逐格核实结果与修复**：

| 格（physics） | 原声明 | 核实结论 | 修复 |
|---|---|---|---|
| porous_media D3Q19/BGK | IMPLEMENTED, CONTRACT_TESTED, `porous_media_common.porous_media_step` | 模块/函数/测试文件**从未存在**（`git log --all` 空）；`porous_media.py` 只有介质生成 + 多相孔隙尺度流程，无 partial bounce-back 步 | → `NO_IMPLEMENTATION` |
| porous_media D3Q19/MRT | 同上 | 同上 | → `NO_IMPLEMENTATION` |
| porous_media D3Q27/BGK | 同上 | 同上；且无 D3Q27 路径 | → `NO_IMPLEMENTATION` |
| porous_media D3Q27/MRT | 同上 | 同上 | → `NO_IMPLEMENTATION` |
| non_newtonian D2Q9/BGK | `non_newtonian_common.non_newtonian_tau_eff` | 真实机制 = `non_newtonian.apparent_viscosity_power_law` + `collide_power_law_bgk`；测试 `test_non_newtonian.py`（5 例） | **改指针**，仍 IMPLEMENTED/CONTRACT_TESTED |
| non_newtonian D3Q19/BGK | 同上（称 3-D 应变率） | 真实 3-D 机制 = `herschel_bulkley_lbm.collide_hb_bgk_forced_3d` + `shear_rate_3d` + `hb_relaxation_time`（D3Q19、Papanastasiou 正则化 H-B）；测试 `test_herschel_bulkley.py` | **改指针**，仍 IMPLEMENTED/CONTRACT_TESTED |
| non_newtonian D3Q19/MRT | 「tau_eff 与碰撞无关，调用方在 MRT 应力模里用」 | **本包所有非牛顿核都是 BGK**，无变黏度 MRT 核 | → `NO_IMPLEMENTATION` |
| non_newtonian D3Q27/BGK | `non_newtonian_common...` | 无 D3Q27 非牛顿核（3-D 只有 D3Q19 的 19 速度反方向表） | → `NO_IMPLEMENTATION` |
| non_newtonian D3Q27/MRT | 同上 | 同上 | → `NO_IMPLEMENTATION` |

**修复后**：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 受审契约 / 格 | 4 / 16 | **8 / 170** |
| error | 18（来自 9 格虚报） | **0** |
| warning | 2 | 2（`acoustics[surface_pressure_extraction/D3Q19,D3Q27]`：格子按点阵分键但被引测试文件从不提该点阵——如实披露，不阻断） |
| 契约内 IMPLEMENTED 格 | 45 格中 19 格（其中 9 格虚报） | 45 格中 10 格，**全部**入口 + 证据可解析（新测试钉住） |
| 测试 | 15（闸门）+ 7（physics 契约） | **45（闸门）+ 15（physics 契约）**，四个相关文件合计 **80 passed** |

## 4 结果图

**本任务为结构性审计（字符串对仓库解析），不含场量或收敛曲线，故不提供云图/曲线图**——
所有结论均为可复算的布尔量（`0 error` / `80 passed`），复核命令见 §6。

## 5 注意点 / 局限（踩坑与适用范围）

* **「测试绿」不等于「契约诚实」**：修复前 `tests/test_physics_capability_contract.py` 7 条全绿，
  而它自己断言的 4 个 `require_implemented(porous_media…/non_newtonian…D3Q27)` 正是**虚报的一部分**
  （测试把错误声明固化了）。本单连同测试一起改，并把「IMPLEMENTED 格的入口/证据必须解析」
  写成新测试，否则下次腐烂还是无人管。
* **豁免必须外显**：`advanced_collision` 的 `AVAILABLE` 状态在 fail-closed 不变量下会被判
  `NOT_FAIL_CLOSED`（14 个假红）。用 `CONTRACT_POLICIES` 声明豁免 + 理由并写进报告 JSON；
  测试要求「豁免项必须有 reason」，防止豁免表变成消音器。
* **entrypoint tier 的代价**：`amr`/`advanced_collision` 只查入口解析，**其证据面仍未受审**
  （因为 schema 无证据字段）。这是能力边界，不是疏忽：给这两份契约加证据字段是后续工作。
* **`LATTICE_NOT_IN_EVIDENCE` 仍是 warning**：`acoustics` 的表面压力提取与点阵无关，
  按点阵分键而被引测试不提点阵——它暴露的是**契约分键方式**与证据粒度不匹配，不改判据。
* 本单不动任何物理解算逻辑，只动证据面字符串 + 闸门；**没有任何数值精度结论**，
  因此不涉及平台 15% 门槛（本任务为代码面审计，无基准量可比重）。
* 容量纪律：本机 load15 ≈ 128（nproc=128）≥100，全程只做轻量活（导入 + 字符串解析 + pytest），
  未起新算力进程。

## 6 误差判定 / 判据

| 判据 | 阈值 | 实测 | 是否达标 |
|---|---|---|---|
| 闸门 error 数 | 必须 = 0 | 0（8 契约 / 170 格） | **是** |
| 覆盖契约数 | ≥ 全部有矩阵工厂的契约（8） | 8 | **是** |
| 负控鉴别力 | 注入缺陷必须报警 | 45 条闸门测试含 8 类注入缺陷变体 | **是** |
| 契约 IMPLEMENTED 格可解析率 | 100% | 10/10 入口、10/10 证据文件 | **是** |
| 数值精度 vs 基准 | — | 本任务无物理量输出，不适用（非粉饰） | 不适用 |

复核命令：

```bash
PYTHONPATH=src python3 -m tensorlbm.capability_contract_drift          # 0 error / 2 warning
PYTHONPATH=src python3 -m tensorlbm.capability_contract_drift --json   # policies 节可见豁免
PYTHONPATH=src python3 -m pytest tests/test_capability_contract_drift.py \
    tests/test_physics_capability_contract.py \
    tests/test_amr_capability_contract.py tests/test_acoustics_capability_contract.py -q
git log --all --oneline -- src/tensorlbm/porous_media_common.py        # 空 => 从未存在
```

## 7 证据指针

* 闸门扩展：`src/tensorlbm/capability_contract_drift.py`（`CONTRACTS` 4→8、`ContractPolicy`、
  `CONTRACT_POLICIES`、`Claim.evidence_field`/`.tier`、序列遍历、模块入口、`ANALYTIC_VERIFIED`）
* 契约修复：`src/tensorlbm/physics_capability_contract.py`（9 格：4 降级 + 2 改指针 + 3 降级；
  docstring 的「审计范围」同步去掉两个从未存在的模块）
* 测试：`tests/test_capability_contract_drift.py`（15→45 例）、
  `tests/test_physics_capability_contract.py`（7→15 例，含 `TestEvidencePointersAreReal`）