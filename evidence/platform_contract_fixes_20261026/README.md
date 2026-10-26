# 平台路径、任务状态与结果契约修复（2026-10-26）

基线：`90332d1bc24b4aeffd58bc40dfd62e4a876562c3`（用户已合并的 PR #410）。
工作分支：`arena/dc4c3966-tensorlbm`。本报告描述本地实现与执行证据，不代表远程 CI 已通过。
前一轮只读检查保留在 `../post_merge_review_20261026/`，其中探针输出是**修复前快照**。

## 1. 实施范围

### A. 文件访问边界

- 新增 `app/backend/file_paths.py:safe_job_path`，使用 `resolve()` 和
  `is_relative_to()` 做路径组件包含判断，替代字符串前缀判断。
- Job 文件下载、图片读取，以及后处理的 field-data、field-data-3d、export、export-vtk
  四个 checkpoint 接口复用该边界。
- 除显式路径外，检查隐式 `latest` 所选目录，以及 `load_checkpoint` 实际读取的
  `checkpoint_f.pt`、`checkpoint_meta.json`，防止目录内文件通过符号链接指向外部。
- 正常路径、目录内符号链接、目录内绝对路径继续受支持；不存在的文件仍为 404，越界为 403。
- 非法路径、循环符号链接失败关闭。

**限制：**这不是全平台所有文件读取点的审计，也不是原子文件打开；不能防止具有输出目录
写权限的攻击者进行 TOCTOU 替换。测试仅操作自身临时目录，未针对线上服务或真实数据。

### B. Job 状态、启动与测试夹具

- 动态路由使用当前包的相对导入，不再在 `app.backend.main` 中硬编码加载 `backend.routers`。
- 生产脚本与平台测试以 `backend` 为统一入口，修正 13 处 `from app.backend...` 测试导入；
  两个自行创建客户端的模块改用共享 fixture，Live metrics/DoE 使用共享 Job manager fixture。
- `jobs`、`postprocess`、`solver`、`orchestration` 导入失败会阻止启动；可选路由仍记录警告。
- 增加两个命名空间**分别在独立进程启动**时的实例一致性测试、必需/可选路由失败策略测试，
  以及防止测试重新混用命名空间的 AST 检查。
- 修正旧测试替身的 `to_dict(include_result=...)` 接口；SubOff 夹具不再传不存在的
  `Job(output_dir=...)` 参数，并使用真实 `save_checkpoint()` 写入正确的元数据文件。
  没有修改 Job 构造函数或放宽原有 API 断言。

**限制：**两个命名空间并不是模块别名。在同一进程中同时导入两者仍会产生两个状态容器，
因此不支持混用；没有通过 `sys.modules` 别名隐藏这个事实。使用约定已写入平台用户手册。

### C. Study 结果聚合

- 保持 `list_jobs()` / WebSocket 的轻量结果摘要不变，仅对匹配 study 的任务按 ID 取真实结果。
- 容忍列表生成后任务被删除的情况，跳过失效条目。
- 修正稀疏指标的 `best_job_id`：最佳值索引必须对应含该指标的行，而不是全部完成任务的行。
- 保留标量、序列 final/mean、跨任务 min/max/mean、运行中任务计数及完成态统计约定。

### CI 诊断与预算（独立配置改动）

- 数值库与平台测试保存完整 pytest stdout/stderr、JUnit、耗时排行、Python/pip 环境清单。
- artifact 上传采用 `if: always()`；显式 Bash/pipefail，确保 tee 不吞掉 pytest 失败退出码。
- 平台每测试 600 秒、job 45 分钟；数值库 job 180 分钟，保留 fast 600 秒/full 1800 秒。
  CI 用 `timeout_func_only=false` 将 fixture 设置和清理纳入超时；不修改本地默认行为。
- PR 和非 main push 使用 fast 范围；main push、schedule、workflow_dispatch 保持 full。
- 未修改断言、quarantine 或 continue-on-error 来掩盖测试失败。
- 8 个本地 CI 配置测试验证事件分流及真实 shell 管道的失败退出码/日志保留；尚未触发远程 CI。

## 2. 实际执行结果

同一个新安装的虚拟环境：Python 3.11.2、PyTorch **2.14.1+cu130**、NumPy **2.4.6**，
`torch.cuda.is_available() == False`。完整版本清单在 `environment.txt`。
这不同于上一轮旧 Torch/NumPy 环境，不将两个环境的结果直接归因于代码变化。

### 原有 716 个平台测试：同环境基线对照

| 版本 | Passed | Failed | Skipped | Error |
|---|---:|---:|---:|---:|
| 合并基线 90332d1 | 621 | 39 | 48 | 8 |
| 本轮修复 | 638 | 30 | 48 | 0 |

- **17 个原有用例恢复**：3 Live metrics、1 DoE、3 study、1 FakeJob 接口、9 SubOff 夹具。
- 已逐个比较 testcase 标识：原有测试集合不变，**没有 passed → 非 passed，没有新增 skip**。
- 恢复清单与剩余失败的原始 message 见 `summary.json`。

### 新增 61 个平台契约测试

| 源码 | Passed | Failed |
|---|---:|---:|
| 合并基线 + 相同新增测试 | 24 | 37 |
| 本轮修复 | 61 | 0 |

覆盖 HTTP 越界拒绝、正常下载、显式/隐式 checkpoint、文件/目录/元数据符号链接、
正确 checkpoint 的真实读取、稀疏指标最佳任务、轻量列表、删除竞争与路由启动策略。
HTTP 使用 FastAPI TestClient，不是远程服务测试。

最终完整平台执行：**699 passed / 30 failed / 48 skipped，0 errors，71.66 秒**。
真实退出码为 1，没有包装成成功。

其他执行：

- CI 配置 8 项 + 治理 21 项：**29 passed**（pytest 另记录 81 个 subtests）。
- `ruff check src tests examples benchmarks`：通过。
- `ruff format --check src tests examples benchmarks`：1545 文件通过。
- 新增 helper、平台回归测试、证据汇总脚本 Ruff/format：通过。
- 已修改的旧 app 文件有原有 Ruff 债务：基线 53 项，修复后 50 项，按文件/规则/消息
  比较无新增项；没有把不属于当前 CI lint 范围的整个 app 宣称为 lint 全绿。
- 核心 mypy 四文件：通过；文档一致性、仓库布局、`git diff --check`：通过。
- 未运行完整数值库、前端构建/浏览器测试、远程 CI 或 GPU 验收；小规模 CPU 平台测试
  通过不等于物理精度验收。

## 3. 剩余 30 个失败与下一批工作

1. **28 个前端相关**：旧 JS/DOM/i18n 文件契约，或干净 checkout 缺少 Vue 构建产物。
   下一批应建立 Vue 构建及行为测试，并逐项映射旧断言；不能只删除旧测试。
2. **2 个既有 SubOff 问题**：
   - `TestBoundaryLayerAtStation::test_yplus_computed_when_tau_w_supplied`
   - `TestSuboffSolveEndpoint::test_invalid_hull_type_returns_422`

以上两类都保留失败状态。数值/物理算法、float32 Navier 漂移、GPU 能力与隔离清单治理
没有混入本轮。建议先独立复核路径安全改动与任务契约，再继续前端和上述 SubOff 问题。

## 4. 复现与证据位置

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]' -r app/requirements.txt

# 修复后完整平台；失败是当前预期的真实结果
set -o pipefail
.venv/bin/python -m pytest -q app/tests --timeout=120 -o timeout_func_only=false \
  --durations=20 --junitxml=artifacts/runs/platform-contract-fixes/platform-final.xml \
  2>&1 | tee artifacts/runs/platform-contract-fixes/platform-final.log

.venv/bin/python -m pytest -q tests/test_ci_workflow.py tests/test_repository_layout.py \
  tests/test_repository_report_migration.py tests/test_audit_import_graph.py --timeout=60
```

原始 JUnit、stdout/stderr、安装日志位于忽略目录 `artifacts/runs/platform-contract-fixes/`。
基线从 `git archive 90332d1` 导出到该目录中的 `baseline/`，加入同一份新回归测试，
使用同一个 `.venv/bin/python`、相同 timeout 参数、独立 `TENSORLBM_OUTPUT_ROOT` 运行。
最初一次只导出 app 的全量基线尝试已中止，不纳入结果；用于表格的基线是完整仓库导出。
基线完整 XML 内含早期 48 个新增测试，汇总程序将它们剔除，单独以最终 61 个新增测试的
`baseline-contract-final.xml` 与修复后结果对照，避免重复计数。

```bash
python evidence/platform_contract_fixes_20261026/summarize.py
# 或指定包含 platform-baseline.xml、platform-final.xml、baseline-contract-final.xml
# 及对应日志、environment.txt、governance.xml 的原始证据目录：
python evidence/platform_contract_fixes_20261026/summarize.py /path/to/raw-artifacts
```

`summary.json` 保留结果、恢复/失败清单及原始证据 SHA-256；汇总脚本依赖这些原始文件，
单独下载本证据目录不能重新生成。大型求解输出、虚拟环境和基线导出不纳入 Git。
以上是实现阶段的本地验证快照；当时尚未提交、推送或创建 PR。随后用户授权创建 PR，
提交按路径安全、任务/结果契约、CI 配置、文档证据分组；不执行合并。

### PR 准备阶段复验

再次联合执行 61 个平台契约测试、8 个 CI 配置测试、21 个治理测试：
**90 passed，81 subtests passed，1 warning，12.74 秒**。Ruff、格式（1545 文件）、
仓库布局、文档一致性和 diff 检查再次通过。原始复验日志及 JUnit 位于
`artifacts/runs/platform-contract-pr/`（忽略目录）。本次没有重复全平台测试；上文完整平台
699/30/48 的数字仍引用实现阶段的完整执行，不代表远程 CI 或本次重新运行的结果。
