# PR #410 合并后系统检查与下一阶段优化建议

> 历史检查快照：以下内容及 `probe_results.json` 对应修复前源码。后续实现与实际验收见
> [`../platform_contract_fixes_20261026/README.md`](../platform_contract_fixes_20261026/README.md)。
> `probe_contracts.py` 是针对该基线的诊断，不是修复后验收测试；其 AST 提取方式依赖旧函数
> 的自包含实现，不能直接对已改为共享 helper 的新源码重跑。修复后请运行新增契约测试。

- 日期：2026-10-26。
- 当前提交：`90332d1bc24b4aeffd58bc40dfd62e4a876562c3`。
- 当前工作分支：`arena/dc4c3966-tensorlbm`。
- 本轮是检查与计划：仅新增本证据目录，没有修改生产代码、测试断言、CI 或运行配置。

## 1. 当前状态：静态检查问题已解决，不能继续沿用旧诊断

[PR #410](https://github.com/cxs503/TensorLBM/pull/410) 已于 2026-10-26
15:44:00（UTC+8）合并。PR 的远程 lint job 中 Ruff、格式、核心 mypy、全包
advisory mypy、文档一致性检查均成功；不是只根据 job 总状态推测。

[合并后 main CI](https://github.com/cxs503/TensorLBM/actions/runs/40287599103)
在本轮检查时 lint、repository-layout、architecture-audit、package-build 已通过，
平台与两个 Python 测试 job 尚未完成。PR 自身的 platform-test 失败，两个数值测试
job 在检查时仍运行。CI 状态会继续变化，应以链接中的最终结果为准。

**更新上一轮结论：**旧 PyTorch 本地环境的 `solver.py:437` mypy 报错不能作为
当前远程 CI 的既存问题；本轮已看到远程对应检查通过。不要为消除这个旧环境差异
继续修改求解器。

本轮没有重跑全平台、全数值库或 GPU 测试。上一轮记录的 613 passed / 47 failed /
48 skipped / 8 errors 是历史本地对照结果，不是本轮或当前远程结果。

## 2. 新确认：任务模块的双命名空间产生两份状态（P1）

### 证据

- `app/tests/conftest.py` 从 `backend.main` 创建应用，并从 `backend` 获取 Job manager。
- `app/backend/main.py` 动态导入 `backend.routers.*`。
- 但 `test_gap_improvements.py` 的 Live metrics 用例、`test_engineering_features.py`
  的 DoE 用例从 `app.backend` 取 Job manager。
- 静态统计：`app/tests` 内有 41 处 `backend...`、13 处 `app.backend...` 的
  `from ... import ...` 语句（包括函数内导入；不是所有动态导入的完整计数）。

本轮实际导入两个模块并注册一个临时 Job，结果见 `probe_results.json`：

```text
same_source_file       = true
same_module_object     = false
same_registry_object   = false
same_job_class         = false
served_namespace_finds_job    = true
alternate_namespace_finds_job = false
```

这不是简单的命名风格问题：同一源文件被加载成两个模块，拥有不同的 `_jobs`、锁、
线程池与类对象。它为之前 Live metrics 返回 404、DoE 测试查不到任务提供了具体机制，
比“路由可能有问题”更明确。尚未在本轮运行完整 HTTP 用例或验证修复收益。

### 建议与验收

1. 先明确一个受支持的应用导入/启动入口；现有 `app/start.sh` 与测试主要使用 `backend`。
2. 第一小批优先让相关测试通过共同 fixture 获取 Job manager，不先全仓重命名包。
3. 增加实例一致性检查，确保 app、router、fixture 指向同一 module/registry；审查可选
   第二种启动方式，选择明确拒绝或提供经过测试的兼容，而非静默产生第二套状态。
4. 固定任务 ID、任务完成与诊断推送测试；同一个 ID 必须可由 API 查询，真不存在的 ID
   仍返回 404。保留 DoE 元数据断言，不改成允许空结果。
5. 不以散布 `sys.modules` 别名的方式掩盖问题；别名涉及 import 顺序、类身份、事件循环，
   若需要兼容必须独立测试两种启动顺序。

## 3. 已确认：任务摘要与结果读取契约脱节（P1）

`job_manager.list_jobs()` 有意调用 `to_dict(include_result=False)`，返回 `result={}`，
避免轮询和 WebSocket 携带大数组。这一设计应保留。

但 `postprocess.study_compare()` 仍从这个列表的 `result` 提取 scalar metrics，
所以有真实结果的任务也可能显示空指标。本轮最小探针：

```text
job.result                = {drag_coefficient: 0.8, series: [0.7, 0.8]}
list_jobs()[0]['result']   = {}
```

对比：`orchestration.study_summary()` 先遍历轻量列表，再 `get_job()` 取实体并计算
KPI，说明当前代码已经有更合适的访问模式。另有嵌套 `config.study.group`、平铺
`config.study_group`、独立 `config.doe.group` 的不同消费者，不能盲目合并所有字段。

建议明确三种接口：任务摘要、完整结果/场数据、有限标量指标快照。只修正需要结果的
消费者，**不要把所有 list_jobs()/轮询恢复为携带完整 result**。

验收：scalar/time-series 指标与源结果一致；空结果、任务删除/并发状态变化有明确处理；
大场数组不进入列表或 WebSocket；未知状态与非有限值不能被静默纳入“最佳方案”排名。

## 4. 新确认：路径检查的字符串前缀不是目录边界（部署前优先，P0）

`app/backend/routers/jobs.py:_safe_job_path` 使用：

```python
str(target).startswith(str(root.resolve()))
```

本轮只在临时目录上提取并运行该纯函数，不加载服务器、不读取真实业务文件。结果：

- 正常子文件接受；不相关目录的 `..` 路径被拒绝。
- **同前缀兄弟目录被错误接受**，尽管 `Path.is_relative_to(root)` 明确为 false。

这是路径边界检查缺陷，不等于已经证明远程未授权读取；HTTP 路由、鉴权、文件存在条件
及部署隔离仍影响实际可达性。`postprocess.py` 中另有四处相同模式，需一并审查，
本轮未对这些 HTTP 入口做端到端验证。

下一步建议单独修复：将已 resolve 的路径按路径组件验证包含关系（`relative_to` /
`is_relative_to`），统一 helper，维持原有 403/404 语义。增加正常子路径、普通越界、
同前缀兄弟目录、绝对外部路径、指向外部的 symlink 测试；不要用字符串 `startswith`
替代真实目录边界。还应明确部署下可写目录与符号链接/并发替换的威胁模型，不能仅凭
一次 resolve 检查宣称已覆盖所有文件访问风险。

## 5. 测试与 CI：先让每次红灯都有可定位证据（P1，可并行）

当前 CI：

- 数值库只上传 coverage XML；平台测试没有 JUnit/测试日志 artifact 步骤。
- 未打印保存解析后的 Python/PyTorch/NumPy/Ruff/mypy 环境清单。
- PR 才选 `not slow`；普通工作分支 push 也走 full，与注释中强调 main/nightly 的描述不完全一致。
- `timeout_func_only=true` 不能覆盖所有 fixture/setup 阶段；platform-test 没有显式 pytest timeout。
- 存在 121 行隔离条目、120 个唯一 node ID，采用 non-strict xfail；未逐项验证现在是否还失败。

建议：

1. 所有测试 job 输出 JUnit、测试日志、`--durations` 与版本清单；失败时也上传。
   若用 tee，保留 `pipefail`/真实 pytest 退出码，不能把采集成功当成测试成功。
2. 先测耗时，再拆小型单元/接口、CPU 物理回归、长算例、GPU；不要仅以当前 `slow` 标签
   为“快速”保证，也不要把慢的失败用例直接挪出所有门禁。
3. 明确 PR/分支 push/main/nightly 的范围和 job 总时间预算；增加必要的 fixture 终止与
   子进程清理，并记录 timeout 与断言失败的不同原因。
4. 隔离登记加 owner、issue、设备条件和复查日期；追踪 XPASS，逐步退出隔离而不是清空名单。
5. 维护一个受控稳定环境加定期最新依赖兼容测试。当前 `torch>=2.0`、无版本边界的 numpy
   与未用于 CI 的 uv.lock，不能构成已验证的支持矩阵；也不应把所有环境锁死到旧沙箱版本。

## 6. 平台剩余契约与前端（P1/P2）

- Job fixture 使用不支持的 `output_dir` 构造参数；应通过受控临时输出根/工厂创建符合
  真实生命周期的 Job，不为测试单方面扩展生产构造函数。
- `_FakeJob.to_dict` 缺少 `include_result` 参数；采用共享协议/真实轻量对象或签名受约束
  的替身，避免真实接口和替身漂移。
- 当前 Vue 前端有 build 脚本，但没有 test 脚本；旧测试读取 `app/frontend/static/js/...`
  或在初始 HTML 上查旧 DOM。补 `npm run build` 只能解决缺构建，不能使这些旧 DOM 断言有效。
- 将前端验收拆为源码/组件契约、构建并服务 SPA、浏览器交互（导航、任务状态、多语言）。
  先列出原用例保护的用户行为，再迁移，不能批量删测或改为永远跳过。
- `main.py` 对所有 router import 捕获 Exception 并跳过，可能把必需路由缺失伪装成启动成功；
  区分必需与可选路由，对必需失败明确报错，并增加路由注册完整性检查。

## 7. 数值与代码组织：暂不进行大规模迁移（P2）

本轮重跑静态导入扫描：1077 个模块、2958 条内部边、32 处动态导入、11 个环、
0 个解析错误。静态环不等于运行时故障；该扫描也不会自动证明两个导入名字对应同一模块。
根目录仍为 89 个跟踪文件，已有整理成果未回退。

优先建立每个受支持数值能力的轻量物理门禁：守恒、解析剖面、精度/网格收敛、设备范围、
失败漂移记录。特别是 Navier slip 的 float64 测试通过不表示 float32 漂移已修复。
在性能/结构大改前固定参考算例和误差预算，不采用统一质量补偿或放宽阈值换取绿灯。

在关键 API 与失败基线稳定前，不建议再大搬 `src/tensorlbm/`、缩减顶层导出或拆仓。
可继续低风险文档整理，但不能替代当前更重要的正确性和安全边界工作。

## 8. 建议实施顺序及本轮证据

| 批次 | 范围 | 完成标准 |
|---|---|---|
| A | 文件访问路径边界 | 独立安全回归覆盖同前缀/符号链接等，不扩大允许访问范围 |
| B | 统一 Job 实例来源、夹具/替身契约 | API 与测试共享同一 registry；对应失败前后对照；无新隔离 |
| C | study 指标读取与元数据契约 | 聚合结果正确，轮询仍轻量，并发/删除/空结果有测试 |
| 并行 | CI 证据采集与耗时分层 | 失败可定位且退出码真实，保留 full/nightly 覆盖 |
| D | Vue 测试迁移 | 旧用户行为逐项得到替代覆盖，构建与浏览器测试分离 |
| E | 数值能力验收与增量模块治理 | 物理结果与兼容性有独立证据后再重构 |

本轮实际执行：

```bash
python tests/test_repository_layout.py                 # 12 passed
python tests/test_repository_report_migration.py       # 6 passed
python tests/test_audit_import_graph.py                # 3 passed
python tools/check_repository_layout.py               # passed
python evidence/post_merge_review_20261026/probe_contracts.py
```

`probe_contracts.py` 为标准库小探针，输出保存在 `probe_results.json`。它记录缺陷行为，
**不是期望继续保留缺陷的回归测试**。所有演示任务与文件均在独立进程/临时目录中创建并清理。
静态扫描大文件在忽略的 `artifacts/runs/post-merge-review/`，不重复纳入 Git。
本轮没有运行完整 HTTP 平台套件、数值求解、前端构建或 GPU，也没有提交、推送、修改主分支。
