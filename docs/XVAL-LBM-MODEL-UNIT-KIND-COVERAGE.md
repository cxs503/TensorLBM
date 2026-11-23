# XVAL — model directory as build and CI unit: manifest-kind coverage

> 当前检出版本的模型单元位置和复现命令见[模型单元目录](../benchmarks/model_units/README.md)。下文保留历史试验当时的路径和记录。

**Anchor**: `XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT` (CFD-GO/TCLB).
**Generated**: 2026-11-14T15:17:22+00:00 by `tools/xval_model_unit_kind_coverage.py` (every number below is measured at run time).

## ① 案例/背景

TCLB 的规矩是「模型目录 = 编译单元 = CI 单元」，模型清单由文件系统枚举生成，
任何人不写死列表。这条规矩要买到的性质是：加一个目录=加一个 CI 单元，删一个目录=从 CI 消失，
一个坏模型只能失败它自己那个 job。它要防的失效模式也正是 TCLB 自己的历史事故：
`src/models.R` 枚举 61 个模型目录，而 `cpu_test.yml` 只列 16 个——其余 45 个可编译但从不被测。

本仓在同一个锚点下长出了**两种清单方言**：

| 方言 | 清单文件 | 执行器 | 单元数 |
|---|---|---|---|
| `unit_toml` | `models/lbm/<name>/model.toml` (+`case.py`) | `tensorlbm.model_units` | 3 |
| `unit_json` | `models/units/<id>/model.json` | `tensorlbm.models.ci_unit` | 4 |

而生成式 CI 矩阵（`.github/workflows/model-units.yml` 的 `plan` job）只从第一种方言产出。
后果与 TCLB 的漂移同构：`model.json` 目录在「加目录即 CI 单元」的承诺下被创建、被测试、被报告，
却不在矩阵里——加它不改任何 job，坏它也失败不了任何 job。真实基线：
矩阵覆盖 **3/7 = 42.9%**，未覆盖目录 4 个：`models/units/d2q9_bgk_channel_poiseuille`, `models/units/d2q9_bgk_shear_wave`, `models/units/d2q9_mrt_shear_wave`, `models/units/d2q9_trt_channel_poiseuille`

同区域还查出第二个缺陷（provenance）：每个单元「编译」时的探针跑在子解释器里，
子进程继承的是**环境**的 import 路径。若 site-packages 里有一次陈旧的可编辑安装、
或 `PYTHONPATH` 里挂着另一棵工作树，探针测的就是另一套代码——指纹依旧「确定」、
判据依旧「通过」，但两者描述的都是错误的模型。

## ② 怎么做的

1. 新增 `tensorlbm/model_units/coverage.py`：把「清单方言 → 清单文件名 → 矩阵单元命令」
   声明在一处（`MANIFEST_KINDS` / `cell_command`），模型目录从 **git 索引 ∪ 文件系统** 枚举
   （沿用 TCLB `models.R` 的并集语义：未提交的新单元本地也能建、被删的单元会被抓出）。
   覆盖审计分三类失败：`uncovered`（有目录无单元）、`orphans`（有单元无目录）、`duplicates`（跨方言重名）。
2. `python -m tensorlbm.model_units ci-matrix` 改为输出**合并矩阵**（两种方言各用自己的执行器命令）；
   `makefile` 片段同步覆盖两种方言；`verify`/`doctor` 增加 `manifest_kind_coverage` 硬门。
   工作流本身**一行未改**——矩阵仍由 `fromJSON(needs.plan.outputs.matrix)` 消费，
   这正是「加目录无需改任何公共文件」的证明。
3. `build._probe_env()`：子进程环境把 `<repo>/src` 顶到 `PYTHONPATH` 首位；
   探针内再断言 `tensorlbm.__file__` 的父目录就等于该 checkout，否则 `SystemExit` 报错而非静默测量；
   构建工件新增 `tensorlbm_root`（实际被测量包的路径）作为 provenance。
4. 验收：`tests/test_model_units_coverage.py` 12 条新测试（覆盖门三类失败、跨方言矩阵、
   `tmp_path` 加/删目录、诱饵包 shadow、工件 provenance）+ 原有 `tests/test_model_units.py` 10 条全绿。

## ③ 计算结果

### 覆盖（组织度量）

| 度量 | 修复前 | 修复后 | 门槛 | 判定 |
|---|---|---|---|---|
| C5 矩阵覆盖率 | 3/7 = 42.9% | 7/7 = 100.0% | 100% | PASS |
| C6 未覆盖目录 | 4 | 0 | 0 | PASS |
| C7 孤立单元 | — | 0 | 0 | PASS |
| C8 跨方重复名 | — | 0 | 0 | PASS |

### 探针 provenance（缺陷证据）

| 探针 import 路径 | 解析到的包 | 判读 |
|---|---|---|
| 环境继承（修复前行为） | `/tmp/tmptgd447jy/shadow_site/tensorlbm/__init__.py` | **异仓包（测错代码）** |
| `<repo>/src` 置首（修复后） | `/srv/tensor/agent/work/m20568/src/tensorlbm/__init__.py` | 本仓（正确） |

诱饵包内容：`tensorlbm/__init__.py containing FOREIGN = True`，放在 `PYTHONPATH` 首位；修复后探针把它忽略并解析回本 checkout。

### 逐单元验收（两种方言，各自声明的基准与门槛）

| 单元 | 方言 | 判定量 | 实测 | 门槛 | 实测/门槛 | 结果 | 耗时(s) |
|---|---|---|---|---|---|---|---|
| `poiseuille2d_d2q9_guo` | unit_toml | u_profile_rel_sup_err | 2.706e-04 | 0.15 | 0.0018 | passed | 10.7 |
| `poiseuille3d_d3q19_guo` | unit_toml | u_profile_rel_sup_err | 1.098e-03 | 0.15 | 0.00732 | passed | 7.3 |
| `tgv2d_d2q9_bgk` | unit_toml | decay_rate_rel_err | 2.500e-03 | 0.15 | 0.0167 | passed | 4.7 |
| `d2q9_bgk_channel_poiseuille` | unit_json | profile_l2_rel_err | 1.227e-03 | 0.01 | 0.123 | passed | 32.7 |
| `d2q9_bgk_shear_wave` | unit_json | nu_rel_err | 1.111e-03 | 0.01 | 0.111 | passed | 2.0 |
| `d2q9_mrt_shear_wave` | unit_json | nu_rel_err | 1.170e-03 | 0.01 | 0.117 | passed | 2.5 |
| `d2q9_trt_channel_poiseuille` | unit_json | umax_rel_err | 5.775e-05 | 1.000e-03 | 0.0578 | passed | 36.9 |

### 门禁表（程序化判定，`xval.json` 逐项可查）

| 度量 | 值 | 门槛 | 判定 |
|---|---|---|---|
| C5 matrix coverage, before (cells / directories) | 0.4286 | >= 1 | BASELINE |
| C5 matrix coverage, now (cells / directories) | 1 | >= 1 | PASS |
| C6 uncovered model directories (count) | 0 | <= 0 | PASS |
| C7 orphan matrix cells (count) | 0 | <= 0 | PASS |
| C8 duplicate unit ids across dialects (count) | 0 | <= 0 | PASS |
| C9 probe shadowing reproduced on the ambient path (defect demo) | 1 | >= 1 | PASS |
| C10 probe pinning, repo src first | 1 | >= 1 | PASS |
| C11 units passing their own acceptance gate (fraction) | 1 | >= 1 | PASS |

## ④ 图

![fig1_coverage.png](../artifacts/xval_model_unit_coverage/fig1_coverage.png)

Generated CI-matrix coverage per model directory before/after the manifest-kind closure (grey = original model.toml-only matrix).

![fig2_kpi_margins.png](../artifacts/xval_model_unit_coverage/fig2_kpi_margins.png)

Per-unit measured KPI divided by its declared gate limit (log scale; <1 = inside the gate) for every model directory, both dialects.

![fig3_probe_provenance.png](../artifacts/xval_model_unit_coverage/fig3_probe_provenance.png)

Probe provenance: an ambient import path resolves the decoy tensorlbm (origin of the fingerprint drift), the pinned path resolves this checkout.

## ⑤ 注意点 / 局限

- 本单关闭的是**组织度量级**缺口：矩阵覆盖、探针 provenance。它不改变任何物理解。
  数值侧的证据是「全部 7 个单元按自己声明的解析基准仍通过」——即修复是数值中性的，
  但这里没有新的物理基准，物理精度结论仍以锚点原报告与 board 交差审计为准。
- `model.json` 方言没有 `tier` 字段，合并矩阵里用碰撞算子（bgk/mrt/trt）填该列，仅作标签，不参与判定。
- `ci-matrix` 现在对两种方言都要求目录可解析；如果别处存在第三种清单方言，它不会被静默忽略——
  会作为「非受支持清单」暴露（`discover_directories` 只认 `MANIFEST_KINDS` 中的文件名，
  新增方言必须显式登记，这是有意的 fail-closed 设计）。
- 覆盖率门只证明「目录在矩阵里」，不证明单元有意义的判据；判据质量由每个单元的 `gates` 与
  锚点的逐单元基准负责（本单对 7 个单元全部重跑，见上表）。
- 未跑：GitHub Actions 上的真机执行（本环境无 runner，工作流语义由本地 `verify` + 测试覆盖）。

## ⑥ 误差判定（是否 < 3%？）

组织度量是**精确量**，没有 3% 之说，只有 0 误差要求——本单全部按精确门槛判定：

| 度量 | 基准 | 实测 | 偏差 | <3%? |
|---|---|---|---|---|
| 矩阵覆盖目录数 | 7（仓库内受支持清单目录数，程序枚举） | 7 | 0 个 | 是（0 偏差） |
| 未覆盖/孤立/重名 | 0 | 0/0/0 | 0 | 是（0 偏差） |
| 探针解析包 | `<repo>/src/tensorlbm` | `/srv/tensor/agent/work/m20568/src/tensorlbm/__init__.py` | 0 | 是（逐字节同路径） |
| 逐单元 KPI | 各单元声明的解析基准（见上表） | 见上表 | 全部 `实测/门槛 < 1` | 是：相对各自门槛仍有 1.4–4 个数量级余量（最紧一项见门禁表 C11） |

诚实边界：上表第一行「7」是**仓库内枚举值**，不是外部标准；外部对照（TCLB 侧 61 vs 16 的漂移）
只作方法学参照，未做数值对齐。

## 复现

```bash
PYTHONPATH=src python tools/xval_model_unit_kind_coverage.py
PYTHONPATH=src python -m tensorlbm.model_units ci-matrix      # 合并矩阵（两种方言）
PYTHONPATH=src python -m tensorlbm.model_units verify          # 覆盖 + provenance + 漂移门
PYTHONPATH=src python -m pytest tests/test_model_units.py tests/test_model_units_coverage.py -q
```
