# 能力契约的证据面自审：entrypoint/测试指针漂移闸门 + 一处真实陈旧指针修复（2026-11-08, TensorLBM）

`mission 6499 · repo TensorLBM · branch agent/m6499 · 自找活 selfstart`

## 0 结论（一句话）

TensorLBM 的 4 份 fail-closed 能力契约（`ibm / sliding_mesh / fsi / sixdof`，共 **16 个能力格**）
是全库对外的**证据面**：每格用两个**字符串**声明实现入口（`entrypoint`）与测试证据
（`test_evidence`/`verification_evidence`）——而**全仓没有任何代码把这些字符串对仓库解析过**。
本单补上机械闸门 `tensorlbm.capability_contract_drift`（7 类 error + 2 类 warning + CLI），
首次运行即抓到一处真实漂移：sliding-mesh 契约 2 个格子的证据指针指向
`test_sliding_mesh_common.py`——**该文件不存在**，测试实际在 `test_sliding_mesh_regression.py`；
同时发现证据文本"mass conservation"属**夸大**（实测只有 Σf<1e6 的爆炸哨兵）。修复后：
**16/16 格 entrypoint 可解析、16/16 格证据文件存在且有测试、16/16 格状态仍为 `WITHHELD_*`**，
审计 **5 error → 0 error**，gate 自身由 15 条测试钉住（其中 8 条是**注入缺陷必须报警**的变体测试，
防止审计"因为什么都没看而 green"）。

## 1 案例介绍（对象与背景）

能力契约是本仓描述"哪条机制真实存在、验证到哪一级"的机器可读接口，
下游（`cross_module_composition_matrix`、前端元数据、平台能力台账）都以它为准。
其设计是 fail-closed：所有格子的 `status` 均为 `WITHHELD_*`（如
`WITHHELD_NO_PHYSICS_VALIDATION`、`WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE`），
即"机制在、但物理未经认证"。**这个设计恰好制造了新的失效模式**：`WITHHELD_*` 让契约看起来
保守且已审计，而它指向的符号/测试文件可能早已改名——审计面就开始"礼貌地说谎"。

审计对象（本单实测口径）：

| 契约 | 矩阵工厂 | 能力格数 | 记录字段（层级/证据） |
|---|---|---|---|
| `ibm_capability_contract` | `ibm_capability_matrix()` | 4（D3Q19/D3Q27 × hat/4pt） | `verification_level` / `test_evidence` |
| `sliding_mesh_capability_contract` | `sliding_mesh_capability_matrix()` | 6（2 种 kind × 3 格子） | `implementation_status` / `verification_evidence` |
| `fsi_capability_contract` | `fsi_capability_matrix()` | 4（D3Q19/D3Q27 × 1way/2way） | `verification_level` / `test_evidence` |
| `sixdof_capability_contract` | `sixdof_capability_matrix()` | 2（symplectic_euler / cummins） | `verification_level` / `test_evidence` |

注意第四列的**字段名不统一**——这正是下文 §5 记录的"审计器自身盲区"来源。

## 2 怎么做的（方法/流程/工具链）

**新增 `src/tensorlbm/capability_contract_drift.py`（本单）**，纯只读审计，不改进任何契约语义：

1. **摊平**：遍历 4 个公开矩阵（递归 Mapping），凡是带 `entrypoint` 且有任一证据字段的对象
   即认作能力记录（duck-typing，兼容两套字段拼写）。
2. **entrypoint 解析**：正则抽 `tensorlbm(\.\w+)+` 点分路径 → `importlib` 逐段导入 + `getattr`
   下钻，失败抛 `EntrypointNotFound`；非 callable 亦记 error。
3. **证据解析**：从证据文本抽 `test_*.py[::node]` → 在仓库 `tests/` 下按名解析（含 rglob），
   再校验 文件存在 / 含 `def test_*` / 指定 node 仍存在。
4. **fail-closed 不变量**：`status` 必须以 `WITHHELD` 开头；验证级必须在受审词表内；
   `NO_IMPLEMENTATION` 格不得带 entrypoint、不得引用测试文件；
   `IMPLEMENTED_ONLY` 格不得携带证据（否则"无契约测试"的扣留码会被伪装成已测）。
5. **error / warning 分级**（error=漂移，warning=如实披露不作硬失败）：
   - error：`NOT_FAIL_CLOSED`、`ENTRYPOINT_UNRESOLVED`、`ENTRYPOINT_NOT_CALLABLE`、
     `ENTRYPOINT_UNPARSEABLE`、`ENTRYPOINT_FOR_UNIMPLEMENTED`、`EVIDENCE_MISSING`、
     `EVIDENCE_UNRESOLVED`、`EVIDENCE_UNPARSEABLE`、`EVIDENCE_EMPTY`、
     `EVIDENCE_NODE_MISSING`、`EVIDENCE_LEVEL_MISMATCH`、`EVIDENCE_FOR_UNIMPLEMENTED`、
     `UNKNOWN_VERIFICATION_LEVEL`、`CONTRACT_EMPTY`；
   - warning：`ENTRYPOINT_SEGMENT_UNRESOLVED`（多入口自由文本段无法在上下文模块解析）、
     `LATTICE_NOT_IN_EVIDENCE`（契约按 `DxQy` 分格，但该格点串从未出现在被引测试文件里）。
6. **反空转保护** `CONTRACT_EMPTY`：某契约一格都没被认出来 = 该契约其实没被审，必须 error。
   **这条闸门在本单第一次运行时就抓到了审计器自己**（见 §5）。
7. **CLI**：`python -m tensorlbm.capability_contract_drift [--json] [--strict]`，
   有 error（或 `--strict` 下有任何 warning）返回码 1，可直接进 CI。
8. **测试** `tests/test_capability_contract_drift.py`（15 例）：仓库现状 3 例 + 变体（mutation）8 例
   + 公共 API fail-closed 1 例 + 证据解析 1 例 + CLI 2 例。变体测试从**真实 claim** 出发
   `dataclasses.replace` 注入缺陷（错文件名/错符号/非 callable/放开 status/证据错配/证据缺失），
   然后断言对应 error code **必须出现**——审计器若"什么都没看"就会红。
9. **修复**（`sliding_mesh_capability_contract.py`，最小改动 3 处）：
   ① 模块 docstring 审计范围里的 `test_sliding_mesh_common.py` → `test_sliding_mesh_regression.py`；
   ②③ 两个 3-D 格子的证据文本更正为实测口径：
   "shape, finiteness, mass-explosion guard (Σf < 1e6 over the multi-step stability sweep),
   2-D/3-D formula parity, single-layer interface interpolation, rotation about x/y/z;
   no mass-conservation equality and no physical validation of torque/efficiency"
   （原文写 "mass conservation"，而 `test_sliding_mesh_regression.py:546` 只断言
   `mass < 1e6`，不是守恒等式——按证据如实降级措辞）。

**工具链**：`pytest`、`ruff`、`python -m tensorlbm.capability_contract_drift --json`、matplotlib。
本地 load15 ≈ 124/128（余量低），本单全部为纯 CPU 轻量审计，未起求解器算例。

## 3 计算结果

### 3.1 审计前后（`artifacts/capability_contract_drift_20261108/`）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 受审能力格 | 16（4 契约） | 16（4 契约） |
| error | **5** | **0** |
| ├ `EVIDENCE_UNRESOLVED`（陈旧测试文件名） | **2** | 0 |
| └ `EVIDENCE_FOR_UNIMPLEMENTED`（审计器初版误报，已修正判据） | 3 | 0 |
| warning | 0 | 0 |
| entrypoint 可解析且 callable | 16/16 | 16/16 |
| 证据文件存在且含测试 | 14/16 | **16/16** |
| status 全部 `WITHHELD_*`（fail-closed） | 16/16 | 16/16 |

数据文件：`audit_before_fix.json`（审计器首跑抓到的原始 findings，含误报归类）、
`audit_after_fix.json`（修复后全量：逐格 `verification_level` / `status` / entrypoint / 证据）、
`coverage.png`（覆盖图）。

### 3.2 测试

| 套件 | 结果 |
|---|---|
| `tests/test_capability_contract_drift.py` | **15 passed**（含 8 条注入缺陷必报警的变体测试） |
| `tests/test_boundary_capability_contract.py` `test_sliding_mesh_regression.py` `test_ibm_common.py` `test_fsi_common.py` `test_sixdof_common.py` `test_public_api_lazy_exports.py` | 110 passed, **1 failed（既有缺陷，非本单引入）** |
| `ruff check`（新增/改动 3 文件） | All checks passed |

失败项如实披露：`tests/test_public_api_lazy_exports.py::test_lazy_table_matches_type_checking_block`
报 `_LAZY_ATTRS` 比 `TYPE_CHECKING` 块**多 14 项**。本单 `git diff --stat` 只触碰
`sliding_mesh_capability_contract.py`（+11/−7）与两个新增文件，**未改 `__init__.py`**，
故该失败是既有状态，本单不覆盖、不粉饰，留给 `__init__` 的属主修。

## 4 结果图

`artifacts/capability_contract_drift_20261108/coverage.png`：横轴=4 份契约，纵轴=受审能力格数
（4/6/4/2），标题同时标出 error 5→0。**为何没有场量云图**：本单对象是"证据面"而非流场，
不存在可用于云图的物理场量；可发表的可视化就是覆盖/误差条形图，故以覆盖图替代，
物理类对比曲线不在本单范畴（无解析解可对标）。

## 5 注意点（踩过的坑 / 局限 / 适用范围）

1. **审计器自己的盲区（最有价值的一坑）**：初版只认 `test_evidence` + `verification_level`，
   于是 `SlidingMeshCapability`（字段名 `verification_evidence` + `implementation_status`）
   **一格都没被识别**，全库审计静默跳过——而漂移恰恰就藏在这份契约里。
   修法有两条并都落地了：① duck-typing 兼容两套字段拼写；② 加 `CONTRACT_EMPTY` 闸门，
   让"某契约零格"成为 error 而非静默跳过。**通用教训：审计器必须先证明它看到了东西。**
2. **自由文本证据只能启发式解析**：`sliding_mesh` 的 entrypoint 是
   `"a.b.c(lattice='D3Q19') / d / e"` 这种多入口散文串，裸标识符要靠"上下文模块"猜。
   因此该类解析失败只给 warning，不作硬失败——否则会变成"文本风格测试"。
3. **本闸门只保证"指针诚实"，不保证"物理正确"**：契约本身已用 `WITHHELD_*` 扣留物理认证，
   审计器守的是这条扣留不许被悄悄放开（`NOT_FAIL_CLOSED` / `EVIDENCE_LEVEL_MISMATCH`）。
   任何"测试绿⇒物理对"的推论仍然禁止。
4. **适用范围**：仅 4 份 `*_capability_contract` 矩阵入表（`CONTRACTS` 常量单点登记）。
   `boundary / amr / turbulence / acoustics` 等契约有自己的测试（如
   `test_boundary_capability_contract.py`），但**未**纳入本审计的 entrypoint 解析；
   扩表只需在 `CONTRACTS` 加一行并确保记录字段可 duck-type——遗留项，如实记录。
5. **`LATTICE_NOT_IN_EVIDENCE` 是启发式**：证据文件没出现该格点串不等于没覆盖（可能是 2D 隐含），
   故为 warning。本仓当前 0 条，说明证据文本确实提到了各自格点串。

## 6 误差判定

本单没有物理场量，不存在"对解析解/文献误差"意义上的精度问题；判据是**机械真值**：

| 关键量 | 基准（真值定义） | 误差 | 是否 <3% |
|---|---|---|---|
| entrypoint 可解析率 | 16/16（应等于格数，事实可枚举） | 0（差集为空） | 是（精确 0） |
| 证据文件存在率 | 16/16 | 0（差集为空） | 是（精确 0） |
| 修复前漂移格数 | 2（由首跑 findings 枚举给出） | — | 修复后归零 |
| 审计器检出能力 | 8/8 注入缺陷全部报出对应 code | 0 漏报 / 0 误报（变体测试逐条断言 code） | 是（精确 0） |

**显式声明**：以上"误差 0"是机械枚举的真值，不是物理精度；本单**没有**任何物理量对实验/解析解的
误差结论，因此"<3%"在此仅表示枚举差集为空，不构成对求解器精度的任何声明。
遗留待办：把 `boundary/amr/turbulence/acoustics` 4 份契约并入同一闸门（需扩展 duck-typing 或
统一字段命名，属重构级改动，留给后续单）。