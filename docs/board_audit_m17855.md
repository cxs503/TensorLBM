# 交差核验：m17790 · XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN（TensorLBM）

- 审计单：m17855（board-audit / 交差审计，非重做）
- 被审单：m17790（`claimed_by=tp-doc-4`，status=succeeded，2026-11-13 09:17 完成）
- 被审交付：commit `ee1f41f1`（branch `agent/m17790`，**未并入 main**）
- 审计依据：`tensor_benchmark_cases`（无本锚点行）＋ `feature_ledger` 验收条件（<15%）＋ 独立复跑
- 结论：**level = 2，verdict = reopened（返修）**。§5 列出 2 条阻断缺陷、3 条非阻断缺陷。

---

## 1. 案例介绍（这个锚点到底要求什么）

`feature_ledger` 里该锚点的登记（程序化读取，非手抄）：

| 字段 | 值 |
|---|---|
| slug | `XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN` |
| title | sorush-khajepor/listLBM 的 example-per-main-antipattern 做法 |
| acceptance | `<15%（owner 精度标准：对标误差 >15% 不可接受）` |
| status | `open` |

即：这是 **xval（验证类）锚点**，验收判据是**与参考解的对标误差 < 15%**——不是"把脚本重构成框架"这一件事本身。
`tensor_benchmark_cases` 中该锚点**没有登记任何 case/metric/tolerance 行**（查询返回 0 行），
所以本次核验的数值裁判只能取锚点级验收（<15%）＋ 交付自带的外部基准（Ghia et al. 1982, JCP 48:387）。

被审交付声明的物理对象：2D 方形顶盖驱动方腔，D2Q9 BGK/RLBM，`n=65, Re=100, u_lid=0.1`（`τ=0.692`），
对标 Ghia 1982 的 u/v 中心线剖面。

## 2. 怎么做的（核验流程 / 工具链）

1. 读 mission 行（`tensor_dev_missions.id=17790`）：`result.evidence.summary.headline` 声称
   "replaced the example-per-main antipattern (100+ standalone scripts …) with a structured, testable,
   composable framework"；`head_commit=null`、`checks={}`、`branch=null`。
2. 读被审工作树 `/root/tp-agent/work/m17790`（branch `agent/m17790`，`git show --stat ee1f41f1`：7 files, +738）。
3. 独立复跑（不采信交付自述）：
   - `python3 -m pytest TensorLBM/tests/test_examples_framework.py -q` → **12 passed in 6.45 s**
   - `python3 -m pytest tests/test_lid_driven_cavity.py -q`（仓内正统树）→ **20 passed in 14.15 s**
   - 交付默认配置逐 step 追踪 → `scripts/audit_m17855_probe.py trace`
   - 交付默认配置走 registry 端到端 → `scripts/audit_m17855_probe.py guard`
   - 交付文档声明的 CLI → `scripts/audit_m17855_probe.py cli`
   - 产物：`artifacts/board_audit_m17855/probe.json`、`fig_defaults_divergence.png`
4. 证据等级判定用的对照：`tensor_benchmark_cases`（0 行）、`tensor_reports`（0 行）、
   `pyproject.toml` 的 `testpaths/pythonpath`、仓内 `src/` 与 `TensorLBM/src/` 双树结构。

## 3. 计算结果（核验数字表）

| 核验项 | 期望（锚点/声明） | 实测（本次独立复跑） | 判定 |
|---|---|---|---|
| 单测 | 12 tests pass | `12 passed, 1 warning in 6.45s` | ✅ 与声明一致 |
| 单测是否断言精度 | 应有 Ghia 容差断言 | 12 个 test 只断言 registry 行为 / 文件存在；`test_run_smoke` 用 `n=17, Re=10` → `ghia_available:false`（`probe.json:ci_smoke_run`） | ❌ 未触及对标分支 |
| 默认配置物理有效性 | `τ=0.692` 稳定、Ghia L2 < 0.05 | `max_abs_ux` 在 **step 10 = 0.125 > u_lid=0.1**，step 35 = 6.9e107，**step 40 = NaN** | ❌ 发散 |
| 默认配置 Ghia 指标 | `l2_u, l2_v < 0.05` | `l2_u=NaN, l2_v=NaN, max_u=NaN, max_v=NaN`（`probe.json:defaults_registry_run`） | ❌ 无数值 |
| 失败是否被吞 | 失败应 `success=False` | `success=true, error=null`（`max_steps=500`） | ❌ 静默假成功 |
| NaN 守卫是否生效 | 应拦截发散 | 守卫条件 `step % 500 == 0 and step > 0`（lid_driven_cavity.py:91-93），`max_steps<=500` 时**永不触发** | ❌ off-by-one 死守卫 |
| 交付自带产物 | run_artifacts 入库 | commit 7 文件全为代码/文档/测试；无 `outputs/`、无 metrics 工件、`head_commit=null` | ❌ 无 |
| 文档声明的 CLI | `python -m tensorlbm.examples.runner --list` 可跑 | `ModuleNotFoundError: No module named 'tensorlbm.examples'`（rc=1，仓根目录执行；`probe.json:documented_cli`） | ❌ 死路径 |
| 平台报告 | 每单必发（六要素） | `tensor_reports` 中 `mission_id=17790` **0 行**；mission 行 `report_owed=true, report_id=null` | ❌ 未发 |
| 基准登记 | `tensor_benchmark_cases` 有 case/metric/tolerance | 按 `anchor_slug` 查询 **0 行** | ❌ 无 |
| 迁移覆盖面 | 声明替代 "100+ standalone scripts" | `git show --stat`：迁移 1 个 example（`lid_driven_cavity`），仓内 `examples/*.py` 仍为原脚本形态 | ⚠️ 声明 > 实际 |

图 1（`artifacts/board_audit_m17855/fig_defaults_divergence.png`）：默认配置下 `max|u_x|` 随 step 演化，
step 10 即越过顶盖速度 0.1，step 40 起全 NaN——与交付文档 "Defaults chosen for stability" 相反。

## 4. 结构性问题：代码落在了不会被打包/不会被 CI 跑的树里

仓内存在两套源码树：

- `src/tensorlbm/`（1256 个 .py，`pyproject.toml:83` `package-dir = {"" = "src"}`、`pythonpath = ["src"]`、`testpaths = ["tests"]`）——**正统、被安装、被 CI 跑**；
- `TensorLBM/src/tensorlbm/`（76 个 .py，含 `verification/lid_driven_cavity/` 旧快照）——**m17790 把新代码全放这里**。

后果（均可复现）：

1. wheel/安装包不含 `tensorlbm.examples` → 文档第 64-73 行给出的 4 条 CLI 全部 `ModuleNotFoundError`（实测 rc=1）；
2. CI 的 pytest 根 = 仓根 `tests/`（`pyproject.toml:92-93`），**永不采集** `TensorLBM/tests/test_examples_framework.py`；
   这批测试只靠 `TensorLBM/tests/conftest.py` 把 `TensorLBM/src` 插进 `sys.path` 才能跑通；
3. 新 example 从旧快照导入 `..verification.lid_driven_cavity`，而正统树里对标 Ghia 的实现在
   `src/tensorlbm/lid_driven_cavity.py`（`tests/test_lid_driven_cavity.py` 20 passed）——即**包了一层会发散的旧实现，
   放着能跑的正统实现不用**。

## 5. 发现清单（按严重度）

**阻断级（blocking）**

- **B1 默认配置发散且静默成功**：`LidDrivenCavityExample.run` 在 `n=65, Re=100, u_lid=0.1` 下 step 40 NaN，
  仍返回 `success=True`、metrics 全 NaN、`ghia_passed=False`（`TensorLBM/src/tensorlbm/examples/lid_driven_cavity.py:63-158`）。
  锚点验收（<15%）因此**无任何可报数值**；`>15%` 不可接受的判据下等价 FAIL。
- **B2 交付不在可交付路径上**：模块位于非打包树 + CI 不采集 + 未并入 main + 文档 CLI 报错（§4）。
  任何用户/平台按文档使用即失败，实质等于能力未落地。

**非阻断（non-blocking）**

- N1 NaN 守卫死代码：`step % 500 == 0 and step > 0` 使 `max_steps<=500` 时永不检查（:91-93）。
- N2 `except KeyError` 静默降级：Re 不在 Ghia 表内时 metrics 只剩 `{"re","n"}`，无任何提示（:120-122），
  测试恰走这条分支，形成"绿测试覆盖不到物理"的假象。
- N3 单测只断言"文件存在/registry 行为"，没有一条精度或容差断言；`tolerance=0.05` 从未被测试触发。
- N4 报告欠账：`report_owed=true`、`tensor_reports` 无记录（六要素报告缺失）。
- N5 声明与范围不符："replaced 100+ scripts" 实为 1 个 example；`head_commit/checks` 在 finish 证据里为空。

## 6. 判级与落表

| 级别判据（maturity_gate JUDGE_SCHEME 机器口径） | 是否满足 | 依据 |
|---|---|---|
| L0 文档成果 | — | 有代码，不止文档 |
| **L1 数学/单元验证**（evidence.tests 非空且可跑） | **是** | 12 个 registry/CLI/smoke 单测独立复跑通过；无一条物理断言 |
| L2 独立功能实现（impl_files + entry + run_artifacts） | 否 | impl_files 是真代码；但**交付零 run 工件**（commit 7 文件无 outputs/，`head_commit=null`），且**文档入口不可用**（`ModuleNotFoundError`）；本锚点唯一 example 在自带默认配置下 NaN→`success=True` |
| L3 系统集成（+ integration_callers） | 否 | 不在打包树、CI 不采集、无外部调用方（仅测试 conftest 塞 sys.path）、未并入 main |
| L4 基准验证（+ benchmark + repro_cmd） | 否 | `tensor_benchmark_cases` 0 行；got=NaN，ref(Ghia 1982) vs got 的相对误差**无法计算** |
| L5 科研工程成果 | 否 | — |

→ **level = 1，verdict = reopened（返修）**。

定级说明（不粉饰）：框架管道本身是**真代码**（`impl_real` 复核通过，`n=17,Re=10` 路径能落 CSV/metadata），
所以返修成本低；但按机器判据 L2 三键里的 **run_artifacts 与可用 entry 均不成立**，且锚点验收（<15%）无任何
可报数字（got=NaN），因此不按"框架做出来了"给 L2 —— 只按交付自身可机器复核到的最高层 L1 记。
gate 复核后应开返修轮（L1 → 承重开线），返修清单见 §7。

verdict 落表：`verdict_m17855.sql` = `tensor_maturity_verdicts`（repo=TensorLBM, scope=`m17790 XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN`,
mission_id=17855, level=1, verdict_by='board-audit'），evidence 八键
（impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor + verdict/level_rationale）
全部为真实指针。

## 7. 返修清单（给接单 agent，按序）

1. 把 `tensorlbm/examples/*` 迁到正统树 `src/tensorlbm/examples/`，改从 `tensorlbm.lid_driven_cavity`（已 20 passed 的实现）取解算器；
2. 让 example 真正跑出 Ghia 数：`n≥129, Re=100`，`l2_u/l2_v` 落盘并断言 `< 0.05`；把该断言写成 `@pytest.mark.benchmark` 测试；
3. 修 NaN 守卫：`step % 500 == 0` 去掉 `step>0` 之外的错误逻辑或改为每 100 step 检查，且 NaN → `success=False`；
4. 在 `tensor_benchmark_cases` 登记 case（`anchor_slug` + metric=`l2_u/l2_v vs Ghia` + tolerance=`<0.15`），
   并把 run 工件（CSV + metadata.json + 环境指纹）提交到 `artifacts/`；
5. 至少迁移 ≥10 个 example 再声称消除 antipattern；发六要素平台报告。

---

### 复现命令（全部可重跑）

```bash
cd /root/tp-agent/work/m17855
python3 scripts/audit_m17855_probe.py trace    # NaN onset = step 40（~15 s）
python3 scripts/audit_m17855_probe.py guard    # success=True + NaN metrics（~31 s）
python3 scripts/audit_m17855_probe.py smoke    # CI 同款 smoke：ghia_available=false
python3 scripts/audit_m17855_probe.py cli      # ModuleNotFoundError（rc=1）
python3 scripts/audit_m17855_probe.py figure   # 出图
cd /root/tp-agent/work/m17790 && python3 -m pytest TensorLBM/tests/test_examples_framework.py -q   # 12 passed
cd /root/tp-agent/work/m17790 && python3 -m pytest tests/test_lid_driven_cavity.py -q             # 20 passed
```

审计独立性限制：本次只读被审工作树与被审 commit，未修改被审代码；未跑完整 `max_steps=5000` 默认算例
（按 0.24 s/step 约 20 min，且 step 40 已 NaN，故用 `max_steps=500` 复现同一失败，
5000 步时守卫会在 step 500 抛出 `RuntimeError` → `success=False`，两种 `max_steps` 下结论相同：默认配置不可用）。