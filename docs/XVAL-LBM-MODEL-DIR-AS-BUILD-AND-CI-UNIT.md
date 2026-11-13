# XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT — TensorLBM × CFD-GO/TCLB

**任务** 17991（第 1 轮·开源对标 XVAL）｜**仓库** TensorLBM｜**生成方式** `python tools/xval_m17991_model_dir_ci_unit.py`（本文件由脚本程序化生成，数字零手抄，源：`artifacts/xval_m17991/xval.json`）

**对标对象** CFD-GO/TCLB 的 model-dir-as-build-and-ci-unit 做法（checkout `/tmp/TCLB` @ `ded67cd768cf7e727bd078af139e3ec7895076e5`）
**判据** owner 精度标准 <15%（对「性质」本身逐项判定，见 ④）

## ① 问题与参考出处（零手抄）

**问题**：TensorLBM 的构建/CI 组织是否具备 TCLB 的「模型目录即构建单元即 CI 单元」性质——新增一个模型是否只需新增一个目录、构建目标与 CI 矩阵是否由目录**枚举生成**、每个模型是否带自己的参考与容差？此前本仓的模型散落在 `src/tensorlbm/*.py`，CI 是单体 job + 手写矩阵，无「单元」概念。

**参考做法（全部由脚本在运行时从 TCLB 源码抽取，附 sha256）**

| 做法 | TCLB 源码位置 | sha256（前 16） |
|---|---|---|
| 单元发现/构建生成 | `src/models.R` | `cf0fe71e807355cf` |
| 单元发现/构建生成 | `makefile` | `247622db3668dbc7` |

- 发现规则（`src/models.R` 抽取的命令原文）：`git ls-files | grep 'conf.mk$'`；`find models -name 'conf.mk'` —— 二者取并集，共 **61** 个模型目录。
- 每模型元数据 `conf.mk`：`ADJOINT`×57, `OPT`×22, `TEST`×33；`TEST=TRUE` 的模型 **20** 个。
- CI 矩阵：`hand-written matrix in .github/workflows (models.R enumeration not used)`，声明 **29** 项（去重 16），未覆盖 **73.8%** 的模型目录。
- 生成规则残留：`makefile` 中 ``，而模板 `src/travis.yml.Rt` 存在=False —— TCLB 自己的 CI 生成链已漂移（见 ⑤ 失败模式）。
- 模型自带验证钩子：1 个（`check.py`）。

## ② 方法与实现（本仓侧）

移植为本仓的 `tensorlbm.model_units`（Python 侧 `conf.mk` 等价物）：

- **单元=目录**：`models/lbm/<name>/model.toml` + `case.py`；目录名即单元名（校验强制）。
- **枚举发现**：`git ls-files '*model.toml'` ∪ `models/lbm/*/model.toml`（照搬 TCLB 并集规则），无中心注册表。
- **生成构建目标**：`python -m tensorlbm.model_units makefile` 生成 `artifacts/model_units/Makefile.models`，`Makefile` 用 `-include` + 依赖 `models/lbm/*/model.toml` 自动重生成（对应 TCLB `makefile.main: ... models/*`），每个单元一个 `make model-<name>`。
- **生成 CI 矩阵**：`.github/workflows/model-units.yml` 的 `plan` job 跑 `ci-matrix --github-output`，`unit` job 用 `strategy.matrix: ${{ fromJSON(needs.plan.outputs.matrix) }}`；workflow 里**没有任何单元名**（测试 `test_workflow_has_no_hand_written_unit_list` 守住）。
- **单元自验证**：每个 `case.py` 用解析解判 KPI（Poiseuille 抛物线剖面、TGV 衰减率），容差写在各自 `model.toml` 的 `acceptance.threshold=0.15`；失败如实记 `failed`，不调参。
- **构建即隔离**：`build` 在子进程里只 import 该单元的 kernel 模块并落 import 闭包 + 平衡态探针 sha256（TCLB 编译出 `CLB/<model>` 的等价物）。

## ③ 参数表（无量纲）

| 单元 | 格子 | 维 | 参数 | 无量纲数 |
|---|---|---|---|---|
| `poiseuille2d_d2q9_guo` | D2Q9 | 2 | ny=32 nx=4 tau=0.800 steps=3000 G=1.0e-05 | ν=0.1000，Re_H=u_max·H/ν=4.1，Ma=u_max/cs=0.0221 |
| `poiseuille3d_d3q19_guo` | D3Q19 | 3 | ny=16 nx=2 tau=0.800 steps=2000 G=1.0e-05 | ν=0.1000，Re_H=u_max·H/ν=0.5，Ma=u_max/cs=0.0055 |
| `tgv2d_d2q9_bgk` | D2Q9 | 2 | n=32 u0=0.04 tau=0.800 steps=400 | ν=0.1000，Re=u0/(k ν)=2.04，Ma_max=0.0980 |

## ④ 对照表（ref = TCLB 实测做法；got = 本仓；✓/✗ 按 owner 15% 判据）

| # | 度量 | ref(TCLB) | got(TensorLBM) | 阈值 | got 判定 | ref 是否达标 |
|---|---|---|---|---|---|---|
| 1 | C1 uncovered fraction of model dirs on the CI matrix | 73.8% | 0.00% | 15% (≤) | ✅ | ❌ |
| 2 | C1b hand-written unit names in the CI configuration | 16 | 0 | 0 (≤) | ✅ | ❌ |
| 3 | C2 files outside the model directory touched to add a model | 2 | 0 | 0 (≤) | ✅ | ❌ |
| 4 | C3 modules imported by one unit build / library surface | n/a | 0.008534 | 5% (≤) | ✅ | — |
| 5 | C4 fraction of units verified against an analytic reference (<15%) | 26.2% | 100.00% | 85% (≥) | ✅ | ❌ |
| 6 | C4b poiseuille2d_d2q9_guo: u_profile_rel_sup_err | n/a | 2.706e-04 | 15% (≤) | ✅ | — |
| 7 | C4b poiseuille3d_d3q19_guo: u_profile_rel_sup_err | n/a | 1.098e-03 | 15% (≤) | ✅ | — |
| 8 | C4b tgv2d_d2q9_bgk: decay_rate_rel_err | n/a | 2.500e-03 | 15% (≤) | ✅ | — |

**结论**：got 列全部达标（all_pass=True）；ref 列在 C1/C1b/C2/C4 上不达标——TCLB 的*做法*（目录即单元、枚举生成、每模型独立编译+测试）是对的，但它*自己的实现*在 CI 矩阵上漂移成了手写列表。本仓移植的是做法并加了防漂移门（`doctor`/`verify`/测试）。

## ⑤ 收敛性/参数敏感性（≥2 点）

见 `artifacts/model_units/*/result.json` 与报告 ⑤ 节正文（网格 ny、步数、τ 三个轴）。

## ⑥ 失败模式与工件

见报告正文 ⑥ 节；工件：`artifacts/xval_m17991/xval.json`、`artifacts/model_units/*.build.json`、`runs/model_units/*/result.json`、`artifacts/xval_m17991/fig*.png`。

