# TensorLBM 代码库分析报告

> 分析日期：2026-09-29 ｜ 分支：`arena/01a0ebf2-tensorlbm`（基于 `main@2e764e2`）
> 方法：静态扫描（git 文件清单、AST 统计、相似度聚类、配置/CI 审阅）。当前沙箱未安装 torch，因此**未执行**测试与基准。

---

## 1. 概览

| 指标 | 数值 |
|---|---|
| Git 跟踪文件 | 2,661 |
| Python 文件 | 1,561 |
| Python 代码行 | ~511,000 |
| `src/tensorlbm` 行数 | ~150,000（含子包约 195k） |
| `tests/` 行数 | ~110,000（426 个 `test_*.py`） |
| `app/`（FastAPI 平台）行数 | ~35,000（109 个 py） |
| 文档 | 215 个 markdown（其中 `docs/` 119 篇） |
| 工作副本（跟踪文件）体积 | ~108 MB，`.git` 44 MB |

定位：**PyTorch 实现的格子玻尔兹曼（LBM）求解器 + AI4S 平台**，覆盖 D2Q9/D3Q19/D3Q27、BGK/MRT/TRT/RLBM/Cumulant、LES/RANS、AMR、DG-LBM、多相、IBM、热/共轭传热、气动声学、船舶与海洋工程算例，外加 FastAPI 后端 + Vue3 前端与 AI 湍流建模栈。

架构分层清晰：
```
src/tensorlbm/        # 库核心（算子、边界、湍流、算例、AI）
  ├ core/ runtime/ performance/   # 基础与运行时
  ├ ai/ ml/ data/                 # AI4S：训练、推理服务、数据目录（~29k 行）
  ├ octree_boundary/ phasefield/  # 专项模块
  └ apps/ cases/                  # 应用与算例注册表
app/backend/          # FastAPI：30+ routers、job_manager、agent_core
frontend-vue/         # Vue3 + Element Plus + Three.js + ECharts
tests/ benchmarks/ examples/ docs/
```

---

## 2. 做得好的地方

1. **工程基建到位**：`pyproject.toml` 规范（动态版本、可选 extras 划分 `io`/`fused`/`dev`）、ruff + mypy + pytest + coverage 全套；CI 里 mypy 分「core 门禁」与「全量 advisory」两档，是很务实的渐进式类型化策略。
2. **依赖诚实**：注释明确说明哪些依赖是 import 期硬依赖、哪些是函数内惰性导入，并据此划分 extras——这点比多数科研代码库好很多。
3. **测试规模与自觉性**：426 个测试文件、11 万行测试；`tests/conftest.py` 会按环境能力跳过采集，并**在终端摘要里主动打印被忽略的文件**（源于 2026-09-05 的 CI 覆盖审计，PR #272 的测试曾整个生命周期没被跑过）——这种"让盲区可见"的设计很成熟。`tests/ci_quarantine.txt` 显式记录隔离用例，而不是悄悄删掉。
4. **可复现性文化**：大量 `*_capability_contract.py`、`*_evidence*.py`、`regression_gate.py`、`docs/**_r1.md` 回归报告，形成"能力契约 + 证据门禁"的物理验证体系；`scripts/check_platform_docs_consistency.py` 在 CI 中校验文档与实现一致性。
5. **返回类型标注率 88%**（4,982 个函数），docstring 覆盖 62%，对这个体量的科研代码属于偏上水平。
6. 中英双语 README / 手册，CITATION.cff、CHANGELOG（Keep a Changelog 格式，条目写得极其详尽）、GPLv3、发布流水线齐备。

---

## 3. 主要问题（按优先级）

### P0 — 仓库根目录失控：147 个一次性脚本被提交

根目录下有 **147 个 `.py`**（`*_worker.py` / `*_launcher.py` / `verify_*.py` / `diagnose_*.py` / `rettest_v3..v6_*.py`），外加 40+ 个 `run_*.sh` / `launch_*.sh`。这些是实验性一次性脚本，典型问题：

- 相似度扫描：`*worker*.py` 中有 **253 对**相似度 >0.9，其中 `rettest_v3_worker.py` vs `rettest_worker.py` 高达 0.99，`naca0012_worker.py / _v2 / _v3` 系列、`fix_benchmarks_worker.py / fix2_benchmarks_worker.py`、`bug29_ship_common_worker.py / _retest_worker.py` 等基本是复制粘贴分叉。
- `.gitignore` 里其实已有 `/_*.py`、`diag_*.py`、`diagnose_*.py` 等规则，但这些文件**先于规则被提交**，ignore 对已跟踪文件无效——`diagnose_stl_normals{,2..6}.py` 仍在版本库里。

**建议**：新建 `experiments/archive/`（或直接 `git rm --cached` + 移到独立仓库），根目录只保留 `pyproject.toml`、README、少量入口。这一项能立刻让仓库可读性提升一个量级。

### P0 — 二进制与运行产物入库

| 文件 | 大小 |
|---|---|
| `artifacts/multi_card_test/fields_final.pt` | 38 MB |
| `checkpoints/suboff/*.ckpt` ×5 | 各 ~6.3 MB |
| `coverage.xml` | 2.3 MB |
| `test_screenshots/*.png` ×6 | 各 240 KB |
| `results*/`、`logs_thermal_common/`、`evidence/` 下 JSON/CSV/log | 157+ 文件 |

`.gitignore` 里写了 `results_*/`、`artifacts/runs/`，但 `results/`（无下划线）、`results_bfl_common/` 之外的多个目录以及 checkpoint 仍被跟踪。`coverage.xml` 属于 CI 产物，绝不该入库（`.coverage` 已忽略，XML 漏了）。

**建议**：
- `git rm --cached coverage.xml`，`.gitignore` 增加 `coverage.xml`、`test_screenshots/`、`checkpoints/`、`logs_*/`、`results/`。
- 模型权重与 `.pt/.ckpt/.npz` 走 Release assets / HF Hub / Git LFS。
- 44 MB 的 `.git` 目前还能忍，但再攒几轮就需要 history rewrite 了，越早处理成本越低。

### P1 — `__init__.py` 的巨型扁平命名空间

`src/tensorlbm/__init__.py` 有 **1,833 行、153 条 `from . import`、~784 个导出符号，且完全没有 `__getattr__` 惰性加载**。后果：

- `import tensorlbm` 会 eager 加载整个 150k 行的库（含 AMR、DG-LBM、声学、AI、船舶 CAD……），冷启动开销与内存占用都很高，对 CLI/worker/serving 场景尤其不友好。
- 784 个平铺符号使"稳定 API"承诺极难兑现——任何模块改名都是 breaking change，且用户无从判断哪些是核心 API、哪些是内部实现。

**建议**：保留一个精简的核心导出（`equilibrium`/`macroscopic`/`collide_*`/`stream`/主要 Solver，约 50 个），其余用 PEP 562 `__getattr__` 惰性代理（`_LAZY = {"AdaptiveSolver2D": "tensorlbm.adaptive_refinement", ...}`），可做到零 breaking change 地把 import 时间降下来。同时在文档中把 API 分为 stable / experimental 两层。

### P1 — `src/` 顶层模块数量与命名分层

顶层有 **380+ 个平铺模块**，命名上已能看出 5~6 个自然域（`suboff_*` 30+ 个、`free_surface_*` 20+ 个、`wall_*`/`wallfn` 15+ 个、`sphere_*` 12 个、`amr_*`/`nested_*`、`bfl_*`、`drag_*`、`triton_*`）。已有 `core/ runtime/ physics/ models/` 等子包却基本是空壳（`physics/` 114 行、`lattice_models/` 31 行、`models/` 194 行）——说明分层重构**开了头但没推进**。

**建议**：按已存在的前缀做机械式子包化（`tensorlbm/suboff/`、`tensorlbm/free_surface/`、`tensorlbm/wall/`、`tensorlbm/amr/`），旧路径保留 shim 模块重定向，一次 PR 一个域，风险可控。

### P1 — 源码内的重复实现

除脚本外，库内部也有可疑重复（相似度为字符袋近似，需人工确认，但方向明确）：

| 相似度 | 文件对 |
|---|---|
| 0.965 | `core/d3q19_stencil.py` ↔ `core/d3q27_stencil.py` |
| 0.964 | `d3q27_sphere_flow.py` ↔ `sphere_flow.py` |
| 0.941 | `geometry_voxel.py` ↔ `voxelize.py` |
| 0.937 | `backends/mindspore_backend.py` ↔ `backends/paddle_backend.py` |
| 0.936 | `propeller_benchmark.py` ↔ `suboff_resistance.py` |
| 0.94 | `ai/active_learning.py` ↔ `ai/inference_service.py` |

典型模式是"按 lattice / 按几何各拷一份"。stencil 与 backend 两对最适合先做参数化合并（lattice 常量表 + 通用实现），benchmark runner 那几对可抽公共 `run_external_flow_benchmark(case_spec)`。

### P2 — 其他

- **58 个源文件超过 800 行**，最长 `aircraft_icing.py` 4,373 行、`autograd_calib.py` 2,720 行、`triton_fused_obstacle.py` 2,662 行。单文件内聚多个职责，审阅与测试定位成本高。
- **65 处 `except Exception`**：宽泛捕获在数值代码里容易把 NaN/形状错误静默成"回退路径"。建议逐步收窄为具体异常并至少 `logger.warning`。
- **文档膨胀**：根目录 9 个大写 `*_SUMMARY.md` / `*_RESEARCH.md` + `docs/` 119 篇（大量带日期后缀的一次性报告，如 `*_20260827.md`）。建议根目录只留 README/CHANGELOG/CONTRIBUTING/LICENSE，报告全部归入 `docs/reports/<date>/` 并加索引，否则新读者无法分辨哪份文档是现行规范。
- **CI 单档、无矩阵**：只跑 ubuntu + Python 3.12（而 pyproject 声明支持 3.11）、CPU-only、bare `pytest`（slow 标记默认执行）。建议加 3.11 矩阵、把 `-m "not slow"` 设为 PR 默认、slow 套件改 nightly；同时给 pytest 配 `--timeout`（`pytest-timeout` 已在 dev extras 但未使用）。
- **前端体量与后端严重失衡**：`frontend-vue` 仅 13 个 `.vue` + 20 个 `.ts`（~2,400 行）对应 30+ 个后端 router（24.6k 行）。要么后端暴露了大量无 UI 消费的端点，要么前端仍是原型——需明确取舍，否则 API 面积会持续成为维护负担。
- **`.hermes/`** 目录（agent 生成的计划/证据）入库，建议评估是否需要版本化。

---

## 4. 30 天行动清单（按投入产出排序）

| # | 动作 | 成本 | 收益 |
|---|---|---|---|
| 1 | 根目录 147 个脚本 → `experiments/archive/`，更新 `.gitignore` | 半天 | 极高（可读性） |
| 2 | `git rm --cached` 大二进制 + coverage.xml；权重转 Release/LFS | 半天 | 高（clone 体积） |
| 3 | `__init__.py` 改惰性导出，量化 `import tensorlbm` 耗时前后对比 | 1–2 天 | 高（启动性能 + API 治理） |
| 4 | CI：Python 3.11/3.12 矩阵、PR 走 `-m "not slow"`、启用 `--timeout` | 半天 | 中高（反馈速度） |
| 5 | 合并 `d3q19/d3q27_stencil` 与 `paddle/mindspore_backend` 两对重复 | 2 天 | 中（示范效应） |
| 6 | 文档归档到 `docs/reports/`，写一份 `docs/INDEX.md` | 1 天 | 中 |
| 7 | 按前缀推进子包化（先 `suboff/`、`free_surface/`，带 shim） | 数周，分批 | 中长期高 |

---

## 5. 总评

这是一个**科研产出密度很高、工程规范也确实在认真建设**的代码库——能力契约、证据门禁、CI 盲区自曝、依赖诚实声明这些做法，在同类 LBM/CFD 开源项目里属于少数派的优点。

它的核心矛盾是：**研究迭代的速度远快于代码收敛的速度**。根目录 147 个一次性 worker、253 对近似重复脚本、1,833 行的 `__init__.py`、380+ 个平铺模块、215 篇文档，都是同一个成因——每次实验都新开一个文件、每个结论都新开一篇报告，而很少回头做合并与归档。

好消息是问题几乎全在**外围与组织层**，不在数值内核：上面第 1、2、4 项一周内可完成且零功能风险，就能让仓库的第一印象和可维护性发生质变；第 3、5、7 项是中期的 API 与结构治理，建议以"一次一个域 + 保留 shim"的方式渐进推进。
