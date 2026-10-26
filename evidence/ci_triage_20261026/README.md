# CI 失败排查：基线对比与第一批修复

日期：2026-10-26。工作分支：`arena/3ecb5c5b-tensorlbm`。

## 结论与范围

**确有历史失败，但不能把所有红灯笼统当作旧问题而忽略。**

本次以相同本地环境分别运行合并前、合并后的平台测试，718 个用例逐项结果完全一致。
同时复现了相同的两项 Ruff 错误及五个格式问题。第一批修复消除了静态检查问题和两处
生产代码运行时导入错误，使七个原平台失败用例恢复通过；仍有其他失败，未宣布 CI 全绿。

- 合并前基线：`ad3d19ae6e5f55f02ac691f6719d4a5ae68f4cdc`。
- PR #409 合并提交：`49eabc4abc26bfdfe5805890e95fb41d180c95c5`。
- 两个提交间 `src/`、`app/`、`frontend-vue/`、`pyproject.toml`、
  `tests/conftest.py` 和 `tests/ci_quarantine.txt` 没有差异。
- 没有修改物理阈值、删测、增加 xfail、降低 CI 要求或变更 benchmark 状态。

## 远程 CI 已确认的事实

- [合并前 main CI](https://github.com/cxs503/TensorLBM/actions/runs/40182539528)：
  lint、平台测试及 Python 3.11/3.12 测试均失败。
- [合并后 main CI](https://github.com/cxs503/TensorLBM/actions/runs/40282873397)：
  排查时 lint、平台测试失败；目录治理、架构审计、打包通过；两个 Python 测试 job 仍运行中。
- GitHub annotations 只提供退出码等概况。完整 job 日志被重定向到当前网络不可达的
  Azure Blob 域名，因此**未获得完整新 CI 日志，也未确认远程依赖的精确版本**。
- Node.js 20 弃用和 ubuntu-latest 迁移提示是 warning/notice，不能当作已证实的失败根因。

以上状态是排查时快照，不是永久结论。本地运行结果与远程 CI 结果分开陈述。

## 本地环境与基线实验

Python 3.11.2，PyTorch 2.2.2+cu121，NumPy 1.26.4，FastAPI 0.162.0，
Pydantic 2.14.1，Ruff 0.17.2，mypy 2.4.0；CPU、OMP/MKL 单线程，CUDA 不可用。
完整环境见 `environment.txt`（环境记录，不是跨机器通用锁文件）。

特意复用之前数值排查使用的 PyTorch 版本，**这不是远程 CI 的精确镜像**。
安装命令：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' 'torch==2.2.2' 'numpy<2' -r app/requirements.txt
```

基线通过 `git archive ad3d19a` 导出到独立临时目录，使用同一个虚拟环境，并将
`PYTHONPATH` 指向该目录的 `src/`，从该目录运行 `app/tests`。没有切换工作分支，
没有把合并后的测试代码拷回基线。导出目录运行结束后清除。

| 平台测试范围 | passed | failed | skipped | errors |
|---|---:|---:|---:|---:|
| 合并前基线 | 606 | 54 | 48 | 8 |
| 合并后、修复前 | 606 | 54 | 48 | 8 |
| 本轮修复后 | 613 | 47 | 48 | 8 |

逐用例比较：合并前后没有结果变化；本轮修复后恰好七项 failed→passed，没有原 passed
变为其他状态。详见机器生成的 `summary.json`。

## 已修复：静态检查

基线和合并后都有：

- `gen_tau_ladder_report.py:66`：F541，无插值字段的 f-string。
- `tau_anchor_probe.py:39`：I001，导入排序。
- 格式问题：上述两个文件、`run_tau_ladder.py`、`test_audit_import_graph.py`、
  `test_tau_ladder_instrument.py`。

仅使用安全修复和格式化，未运行含 `/nfs` 路径的 benchmark 报告生成器。
对五个文件比较了规范化 AST：忽略导入排序、将纯常量 f-string 规范为常量之后，
表达式一致。该检查不宣称可证明任意模块的导入顺序无副作用。

修复后实际执行：

```bash
.venv/bin/ruff check src tests examples benchmarks
.venv/bin/ruff format --check src tests examples benchmarks
```

均通过。**这只证明 Ruff 两步通过，不等于整个 lint job 已通过。**
本地另跑 mypy core gate，在旧 PyTorch 类型定义下发现
`src/tensorlbm/solver.py:437 [no-any-return]`，尚未处理；由于远程 lint job
在 Ruff 阶段已经失败，不能据此认定远程 mypy 也有完全相同问题。文档一致性检查通过。

## 已修复：两处生产代码导入错误

### SUBOFF 阻力后处理

`resistance_breakdown_3d()` 调用 `torch.zeros_like` / `torch.sqrt`，但模块只在
`TYPE_CHECKING` 下导入 torch，函数本身缺少运行时导入。实际触发 `NameError`，
影响 `TestResistanceBreakdown3D` 的三个用例。

修复：函数内局部导入 torch，保持模块惰性依赖，不改力的计算公式。

### D3Q19 壁面剪切

`wss_from_fneq_3d()` 从 `d3q19` 导入不存在的 `equilibrium`，实际公开函数是
`equilibrium3d`。导致皮肤摩擦/y+ 四个测试抛出 ImportError。

修复：使用 `equilibrium3d`，保留原参数、非平衡应力计算和掩码处理；不添加盲目兼容别名。

新增 `tests/test_suboff_postprocess_runtime.py` 的六个 CPU 用例：float32/float64 下
静止平衡零剪切、带解析 xy 扰动的剪切幅值、静止均匀状态零阻力。六项测试在导出的
原始基线源码上全部失败，在修复后全部通过。这些只是局部代数与接口回归，不是阻力物理验收。

## 尚未修复的失败分类

下表统计修复后的 **47 failed + 8 errors**，分类用于安排调查，不代替每项契约审查。

| 类别 | 数量 | 已观察到的原因/下一步 |
|---|---:|---|
| 前端契约或构建缺失 | 28 | 测试仍读取 `app/frontend/static/...`，实际前端为 Vue；另有 `frontend-vue/dist/index.html` 未生成。仅补构建无法解决旧 DOM/旧脚本断言，需要迁移测试契约并验证用户功能。 |
| Job 测试夹具构造接口 | 9 | 8 个 setup errors + 1 个失败使用不支持的 `Job(output_dir=...)`；应核对现有 Job 初始化/输出目录约定，不能直接放宽生产构造函数。 |
| 试验分组与结果聚合 | 4 | DoE study 标记缺失、study-compare 的 metrics 为空；`list_jobs()` 省略 result，而聚合调用方仍需结果，需审查调用链。 |
| Live metrics API | 3 | 预期 200、实际 404；需检查 job 生命周期、路由匹配及返回契约，而非改断言接受 404。 |
| 测试替身签名 | 1 | `_FakeJob.to_dict()` 不接受生产代码传入的 `include_result`。 |
| 边界层 y+ | 1 | `boundary_layer_at_station()` 的 early-return 分支漏掉承诺的 y_plus；应独立审查几何/采样与退化状态语义。 |
| Hull 类型校验 | 1 | 非法 hull type 返回 200 而不是 422；应检查模型枚举和输入验证。 |
| 本地 NumPy 1 兼容性 | 4 | `np.trapezoid` 需要较新 NumPy；本地特意安装 1.26.4。证明本地支持范围存在问题，但不冒充远程同一失败。 |
| 本地旧 Torch compile | 4 | `torch._dynamo...cached_backends` KeyError；需要在 CI 实际依赖版本复核，暂不改 compile 算法或把 eager 当作 compiled 成功。 |

## 已执行的回归与未完成检查

聚焦回归 **71 passed、24 skipped（CUDA 不可用）**：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_suboff_postprocess_runtime.py tests/test_tau_ladder_instrument.py \
  tests/test_audit_import_graph.py tests/test_wall_shear_profile.py \
  tests/test_navier_slip_bc.py tests/test_public_api_lazy_exports.py \
  app/tests/test_suboff_physics.py::TestResistanceBreakdown3D \
  app/tests/test_suboff_physics.py::TestSkinFrictionHull \
  app/tests/test_suboff_physics.py::TestYPlusHull -q --timeout=60
```

平台全量复测命令（退出码仍为 1，不能表述为平台通过）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest \
  app/tests -q --timeout=30 \
  --junitxml=artifacts/runs/ci-triage/platform-after.xml
```

数值库检查的限制：

- 修复前 `--collect-only -m 'not slow'` 成功收集 6702 项、取消选择 14 项；
  本地安装了平台依赖，因此收集范围比缺少 FastAPI 的 solver CI job 更大。
- 尝试全 fast suite（maxfail=3、单测 timeout=30），总执行等待超过 180 秒。
  主进程仍在计算，不据此认定死锁。随后发送 SIGINT 主动停止，约 203 秒时记录
  **579 passed、3 skipped、5 xfailed、14 deselected**；相关子进程已退出。
  **这是中断结果，不是完整回归通过，也不是 timeout=30 能限制全套测试总时间的证明。**
- 完整数值回归、GPU、前端构建未执行；没有更改现有隔离名单。新的修复尚未推送，
  不能把已有远程运行当作对本轮修复的验证。

## 证据与下一步

- `summary.json`：逐项失败分类、前后计数、七项恢复、AST 对比和原始文件 SHA-256。
- `environment.txt`：实际安装版本。
- `summarize.py`：读取本地 JUnit/Ruff 输出并重新生成摘要，缺少输入时直接失败，
  不会自行编造测试结果。执行：`python evidence/ci_triage_20261026/summarize.py`。
- 完整原始日志/XML 保存在忽略的 `artifacts/runs/ci-triage/`，不重复加入 Git；
  `summary.json` 记录其哈希和位置。全新 clone 需要按命令重新采集或从此次工作区取回。

建议分批推进：

1. 独立审查本轮静态清理、两处运行时导入修复和回归证据。
2. 在新的 CI 运行中打印/保存 pip freeze 与工具版本，上传 JUnit 和完整测试日志；
   避免错误只剩 `exit code 1`，也不要把失败采集步骤吞成成功。
3. 单独修复 Job fixture/test-double 契约和结果聚合问题，保留失败前/后的对照。
4. 作为独立任务迁移旧前端测试，区分 Vue 源码契约、构建、实际浏览器功能验收。
5. 在 CI 的实际 Python/PyTorch/NumPy 版本下复核类型检查和数值库失败，再制定依赖
   支持矩阵；不要为了当前沙箱把全项目锁死到旧 PyTorch。
