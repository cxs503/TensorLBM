# XVAL-LBM-GITLAB-CI-STAGED-GATES — TensorLBM 门禁策略 vs lbmpy 分阶段门禁（开源对标）

任务：m17842 · 锚点 `XVAL-LBM-GITLAB-CI-STAGED-GATES` · 仓 TensorLBM · 分支 `agent/m17842`
对标对象：`lssfau/lbmpy` 的 `gitlab-ci-staged-gates`（`oss_solutions` row 117）
交付：`xval/gitlab_ci_staged_gates/`（可执行入口 + 上游原始字节 + 实测数据）、`tests/test_ci_staged_gates_xval.py`

---

## ① 问题与参考出处（零手抄）

**问题**：TensorLBM 的 CI 是 GitHub Actions 扁平 job 集（`lint / test / platform-test / repository-layout / architecture-audit / package-build`，无 `needs` 边、无阶段阶梯）。lbmpy 用的是 GitLab 8 阶段阶梯。**该移植哪些做法？我们的门禁在功能上是否已经等价？**

上游实现（**直接取自源码，非二手转述**；sha256 记录在 `oss_refs/manifest.json`）：

| 文件 | URL | 关键事实（读源码得到） |
|---|---|---|
| `.gitlab-ci.yml` | `https://i10git.cs.fau.de/pycodegen/lbmpy/-/raw/master/.gitlab-ci.yml` | 8 stage（Code Quality/Unit Tests/Reporting/Prerelease-Tests/integration/nightly/docs/deploy）；除 `pages` 外**每个 job 都声明 `needs: []`**；`minimal-sympy-master` 用 `allow_failure: true`；`nightly-sympy` 用 `rules: if $CI_PIPELINE_SOURCE == "schedule"`；`lint` 继承 `.qa-base` 的 `except: variables: [$ENABLE_NIGHTLY_BUILDS]`；`pages` 用 `needs: [coverage-report, build-documentation]`；`pycodegen-integration` 整块被注释 |
| `noxfile.py` | 同上 `noxfile.py` | `nox -s lint` = **flake8 src/lbmpy**；`nox -s "testsuite-3.13(cpu)"` = `pytest -v -n <cores> --cov -m "not longrun" --junitxml=report.xml`；`coverage_report` = `coverage combine/report/html/xml`；`quicktest` = `python quicktest.py` |
| `pyproject.toml` | 同上 | `testpaths = ["src","tests","doc/notebooks"]`、`--doctest-modules`、markers 只有 `longrun` / `notebook` |

> 注：`gitlab.com/lssfau/lbmpy` 与 `gitlab.com/pycodegen/lbmpy` 现均 403（登录墙），canonical 上游在 i10git；`oss_solutions` row 117 记的「nightly 由变量开关控制」与源码有出入——源码是 **pipeline source == schedule**，变量 `ENABLE_NIGHTLY_BUILDS` 只用于跳过 lint。此差异已按源码为准。

**开工三查**：`tools/kb_precheck.py`（knowledge_base / negative_knowledge_ledger / oss_solutions）实跑；`oss_solutions` 命中 row 117（本单对标对象）、row 116（三层验证面）、row 152/280（同行 CI 做法）。`negative_knowledge_ledger` 对 TensorLBM 的记录集中在「inflated_delivery（交付经复核仅 L1）」——本单据此把产物做成**入口可重跑 + 测试自证 + 数字全部程序化注入**。

## ② 方法与实现

1. `ci_spec.py`：把 GitLab CI / GitHub Actions 归一成同一套 **gate-policy schema**（job 级 `tier ∈ {blocking, advisory, conditional, manual}`、`needs`（区分「无 needs 键=阶段屏障」与 `needs: []`=DAG 解耦）、`events`、`selection{kind, marker_expr, scope}`、artifact/coverage 契约）。`extends` 继承、`rules/except` 极性、注释掉的 job 块、空 stage 都显式解析，每条记录带 evidence。
2. `oss_refs.py`：拉取上游字节 → sha256 → `manifest.json`；用 `ast` 解析 noxfile，把 `nox -s "testsuite-3.13(cpu)"` 还原为真实的 pytest 参数（`-m "not longrun"`、`-n <cores>`、`--cov`）。网络失败回退缓存并在 manifest 标注，绝不静默替换。
3. `gate_sim.py`：把策略落到**本仓真实测试语料**（`pytest` 收集快照 16496 项，逐项 marker），算 guard set / escape set / 阶梯深度。
4. `measure_selections.py`：对 4 个 `-m` 表达式实跑 `pytest --collect-only`，用 pytest 自己的计数校验模拟器（**这是 <15% 判据的落点**）。
5. `run_xval.py`：4 臂 × 4 事件 → `summary.json` / `table.csv` / 4 图；`predeclared.json` 在跑之前写定（判据/停止规则/混杂因素）。

## ③ 参数表（本任务无无量纲数，全部为计数/比例，单位写明）

| 量 | 值 | 单位 |
|---|---|---|
| 测试语料快照 | 16496 | items（`tests/` + `app/tests/`） |
| slow 层 | 99 | items |
| unit 层 | 859 | items |
| `not slow` 快层 | 16397 | items |
| lbmpy `not longrun` 在本仓的选择 | 16496 | items（无 `longrun` marker） |
| host load average（collection 期间） | 173 / 128 核 | 无量纲 |
| 语料收集 CPU 时间（5 次） | 48.8–54.9 | s（**墙钟被邻居干扰，不作判据**） |
| 上游 CI 文件 sha256 | `f58d6778…`（gitlab-ci）/ `b310529c…`（noxfile） | — |

## ④ 对照表（ref = OSS 策略迁到本仓语料；got = 现策略；判据 <15%）

模拟器保真度（模拟选择数 vs pytest 自报计数，4+1 个表达式）：

| 表达式 | sim | pytest | 相对误差 | <15% |
|---|---|---|---|---|
| `<all>` | 16496 | 16496 | 0.000% | ✅ |
| `not slow` | 16397 | 16397 | 0.000% | ✅ |
| `slow` | 99 | 99 | 0.000% | ✅ |
| `unit` | 859 | 859 | 0.000% | ✅ |
| `not longrun` | 16496 | 16496 | 0.000% | ✅ |

功能覆盖（H1，guard_fraction 相对差 / escape 绝对差 pp）：

| 事件 | ours | OSS-adapted | guard 差 | escape 差 | <15% |
|---|---|---|---|---|---|
| mr | 99.40% | 99.40% | 0.00% | 0.00 pp | ✅ |
| branch_push | 99.40% | 99.40% | 0.00% | 0.00 pp | ✅ |
| main_push | 100.00% | 99.40% | 0.60% | 0.60 pp | ✅ |
| schedule | 100.00% | 99.40% | 0.60% | 0.60 pp | ✅ |

词汇可移植性（H2）：

| 移植方式 | 长层泄漏率 | <15% |
|---|---|---|
| 逐字照抄 `-m "not longrun"` | **100.00%**（99/99 泄漏） | ❌ |
| 声明式改名 `longrun→slow` | 0.00% | ✅ |

结构差异清单（做法 / 参数 / 边界条件）：

| 维度 | lbmpy（OSS） | TensorLBM（现） | 判读 |
|---|---|---|---|
| 阶段阶梯 | 声明 8 阶段，**实际 7 个有活 job**，`integration` 阶段空（job 整块注释） | 无阶梯，6 个 job 并行 | 阶梯是**声明顺序**不是门：`needs: []` 让全部 job 立即启动，真实依赖边只有 2 条（都进 `pages`） |
| 门禁分层 | blocking 4 / advisory 1 / conditional 3 | blocking 6 | 我们**缺 advisory 层**（无「不影响流水线的实验性依赖」门） |
| 长测门控 | `testsuite` 与 `nightly` **都** `-m "not longrun"` → CI 里根本没有跑长测的门 | PR 不跑 slow，main/schedule 全跑 | OSS 侧存在长测覆盖洞；我们这侧反而更严 |
| 矩阵 | `3.10-cupy` + `3.13-cpu`（declared 4 python × {cpu,cupy}，CI 只取 2） | 单解释器单后端 | 我们的跨解释器/后端覆盖=1 |
| 并行 | `pytest -n <all cores>`（xdist） | 串行 | 若并移植需先验证 xdist 安全性（本单未做，见⑥） |
| lint 面 | `flake8 src/lbmpy`（仅库源） | `ruff check/format` + `mypy` 核心 4 模块（src+tests+examples+benchmarks） | 我们 lint 面更宽、且带类型门 |
| 覆盖率 | 每 job `.coverage.${CI_JOB_ID}` → `coverage-report` 阶段 combine，带 coverage regex 门 + cobertura | `--cov-report=xml` 直接上传，无合并 job | 我们**无覆盖率聚合门/阈值门** |
| 制品契约 | junit + coverage xml/html + html_doc + pages（11 项） | junit + coverage.xml（2 项） | 差异主要在报告聚合，不在测试选择 |
| 边界条件 | `lint` 带 `except: variables: [$ENABLE_NIGHTLY_BUILDS]`（nightly 时跳过 lint） | 无跳过语义 | OSS 存在「夜间流水线放宽静态门」的隐藏语义 |

**移植取舍（结论 ALT-2）**：
- **移植**：① tier 分类法（blocking/advisory/conditional）+ ② 「声明但未启用」门禁卫生检查（OSS 自己踩了坑：8 阶段里有 1 个空阶段 + 1 个注释掉的 job，本仓用 `ci_spec` 的 `empty_stages` / `declared-but-disabled` 检测把这条固化成断言）；③ marker 词汇必须**显式适配**而不是照抄（H2 实测 100% 泄漏）。
- **不移植**：① 阶段阶梯本身（`needs: []` 使其无门禁作用，移植只增加概念负担）；② `-n 全核` xdist（需先测本仓测试的 xdist 安全性，本单未验证，不claim）；③ `quicktest.py` 式预发布依赖门（我们没有 py3.13/sympy-master 依赖轴，无对应物）。

## ⑤ 收敛性 / 参数敏感性（两点以上）

1. **词汇轴**：`not longrun`（逐字）→ 泄漏 100%；改名 `not slow` → 0%。同一策略骨架，仅词汇差 1 个 token，KPI 从 ❌ 翻到 ✅，说明该 KPI 对「词汇适配」这唯一变量敏感、对阶梯结构不敏感。
2. **范围适配轴**：把 OSS 测试门的 scope 声明为 `tests/ + app/tests`（我们 corpus 全集）。若不做 scope 适配（把它当全仓门），`platform-test`/`test` 的路径作用域会互相覆盖，guard 计数虚高——测试 `test_guard_coverage_on_default_events` 用 `mr` 逃逸数 = slow 层 99 项把这个作用域钉死。
3. **语料完整性**：6 个模块在控制环境里 import 失败（editable install 指向别的 worktree，见⑥），缺席语料；由于 guard KPI 是集合比值，缺席项同比例影响两臂，H1 的相对差不变（`mr` 臂 0.00%），但**绝对计数**（16496）是「本环境可见语料」而非全仓语料，已声明为局限。

## ⑥ 失败模式 / 生态摩擦（如实记）

- **F1 控制面缺 editable 安装**：`python -c "import tensorlbm"` 指向 `/root/tp-agent/work/m17719/src/tensorlbm`（别的 mission 的 worktree）。不设 `PYTHONPATH=src` 时本地 pytest 收集直接崩（6 个 collection error）。→ 已在本单所有命令与 `predeclared.json` 中固定 `PYTHONPATH=src`；这是**共享控制面的系统性问题**，建议平台侧修 editable 安装或强制 PYTHONPATH。
- **F2 共享主机负载**：collection 期间 load average 173/128 核（邻居 mission 在跑 pytest），墙钟不可用 → 判据全部换成**精确计数/集合运算**，墙钟仅记录 context；这也使「CI 阶梯的运行时收益」在本单**无法测量**（已在 `predeclared.json` interpretation_limits 声明，不 claim）。
- **F3 集群排队**：`run_cluster submit` 返回 `queued_ahead: 309`，本单改走本地有界测量（不需要集群）。
- **F4 上游可达性**：`gitlab.com` 两个 lbmpy 路径 403（登录墙），GitHub mirror 无 `.gitlab-ci.yml`；改从 i10git canonical 取，字节落盘 + sha256 记账。
- **F5 `oss_solutions` row 117 与源码有出入**（nightly 的触发机制），已按源码更正并写入本文档；建议知识库侧纠偏。
- 未完成/未覆盖：xdist 加速未实测（不 claim）；OSS 侧语料（lbmpy 自身测试）未使用，属单工作语料对比，已声明。

## 复现

```bash
cd <worktree> && export PYTHONPATH=src:.
python -m xval.gitlab_ci_staged_gates.measure_selections.py --exprs "not slow" slow unit "not longrun"  # 实测计数
python -m xval.gitlab_ci_staged_gates.run_xval.py                                                     # KPIs + 图 + 判决
python -m pytest tests/test_ci_staged_gates_xval.py -q                                               # 8 项验收测试
```

工作区内的 git 工作树之外的写入不涉及；产物：`xval/gitlab_ci_staged_gates/{summary.json,table.csv,figures/*.png,runs/*.log}`。