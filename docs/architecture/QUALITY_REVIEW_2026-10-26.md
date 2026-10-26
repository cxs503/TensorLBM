# TensorLBM 项目质量与文件组织评审

- 评审日期：2026-10-26
- 评审范围：整个仓库（4,708 个 Git 跟踪文件，工作树约 257 MB，`.git` 约 125 MB）
- 方法：目录/文件清点、CI 与工具链检查、与 PyTorch / JAX / HuggingFace `transformers` / scikit-learn / PyFR 等顶级开源项目的工程实践对标
- 关联文档：本仓库已有的 `STRUCTURAL_AUDIT_2026-10-25.md`、`REPOSITORY_REFACTORING_PLAN.md`、`repository_layout_policy.json`

> 一句话结论：**算法与工程基础设施是强项**（CI 矩阵、mypy strict、策略驱动的布局检查器、架构审计、双语 README、CHANGELOG/CITATION 齐全），但**仓库“卫生”和世界顶级标准有明显差距**：根目录堆了 44 个一次性调试脚本 + 13 张图片 + 12 篇报告；`src/tensorlbm/` 是一个 353 模块的“扁平大包”，`__init__.py` 长达 2,903 行暴露 786 个惰性导出；把 15.9 MB 模型权重、`runs/`、`benchmarks/` 的 878 个结果 JSON、`agent_work/`/`.hermes/` 这类**内部 Agent 工作文件**直接提交进了 Git。这些都应先治理，且都不触碰数值算法。

---

## 1. 已经具备的世界级实践（值得肯定）

这些是很多号称“工程化”的科研项目都没有做到的，先如实记录：

| 实践 | 现状 | 评价 |
|---|---|---|
| CI 多维度门禁 | `ci.yml` 含 lint / format / mypy 核心门禁 + 全量 advisory / 覆盖率 / 平台测试 / 仓库布局门禁 / 架构审计 / 包构建 | 顶级水平 |
| 类型检查 | `mypy --strict` 且核心模块有独立门禁 | 顶级水平 |
| 策略驱动的布局检查器 | `tools/check_repository_layout.py` + `repository_layout_policy.json`，按精确路径祖父化历史文件 | 比绝大多数开源项目都严谨 |
| 导入图静态审计 | `tools/audit_import_graph.py` 已在 CI 产出依赖报告 | 顶级水平 |
| API 分层意识 | `tensorlbm.api`（稳定）/ `experimental` / `suboff` 等子命名空间 | 好 |
| 文档与治理 | 双语 README、CONTRIBUTING、CHANGELOG、CITATION.cff、AUTHORS | 顶级水平 |
| 打包 | 规范 `pyproject.toml`、sdist/wheel 构建、动态版本号 | 好 |
| 基准生命周期 | 区分 `pending`/`verified`/`archived`，要求可追溯证据 | 好 |

---

## 2. 文件组织现状（量化）

| 区域 | 跟踪文件数 | 磁盘 | 说明 |
|---|---:|---:|---|
| `benchmarks/` | 1,709 | 61 MB | **仅 239 个 `.py` 是基准代码**，其余 878 个 `.json`、195 个 `.log/.txt`、89 个 `.png` 是**运行结果与输出** |
| `docs/` | 701 | 144 MB | 216 md + 148 png，体量主要来自图片资源 |
| `tests/` | 495 | 6 MB | 与 `app/tests/` 并存，分类标记不足 |
| `src/` | 478 | 9 MB | **353 个 `.py` 平铺在 `src/tensorlbm/` 根**，子包仅 15 个 |
| `examples/` | 195 | 10 MB | 153 个 `.py` 散落在根，未充分分主题子目录 |
| `runs/` | 392 | 4 MB | **提交进 Git 的运行输出（JSON）** |
| `experiments/` | 244 | 4 MB | 探索性产物，全部入库 |
| `scripts/` | 162 | 2 MB | 部分与根脚本重复 |
| `app/` | 117 | 2 MB | 后端结构清晰（routers/services/schemas/i18n），但单文件过大 |
| `agent_work/` `.hermes/` `agent_memory/` `evidence/` | 44 | — | **内部 Agent/CI 工作文件，本不应入库** |
| 仓库根 | 44 `.py` + 13 `.png` + 12 `.md` | — | 一次性调试脚本、生成图、报告散落根目录 |

关键单文件规模（世界顶级项目通常把模块控制在 ~600–1000 行、单一职责）：

- `src/tensorlbm/aircraft_icing.py` — **4,535 行**
- `app/backend/routers/postprocess.py` — **3,542 行**
- `src/tensorlbm/__init__.py` — **2,903 行**（声明 786 个惰性导出名）
- `src/tensorlbm/autograd_calib.py` — 2,720 行；`triton_fused_obstacle.py` — 2,662 行；`ai/suboff_dataset.py` — 2,635 行
- `app/backend/routers/solver.py` — 2,168 行

---

## 3. 对标世界顶级项目（差距矩阵）

评分：✅ 达标 / 🟡 部分达标 / ❌ 差距明显

| 维度 | 世界顶级项目做法 | TensorLBM 现状 | 评级 |
|---|---|---|---|
| 依赖方向 | 严格单向：应用→库；库绝不反向依赖 | `app/backend/main.py` 运行时把 `src` 注入 `sys.path` 并 `try/except` 动态导入 router，边界渗透 | 🟡 |
| 公共 API 表面积 | 薄 `__init__` + 显式 `__all__` + 稳定子包 | 353 扁平模块，2,903 行 `__init__` 暴露 786 名 | ❌ |
| 模块粒度 | <~1,000 行、单一职责 | 多个 >2,500 行“巨石模块” | ❌ |
| 二进制/大产物 | Git LFS / Release 资产 / HF Hub，**绝不裸入库** | `models/suboff_v0.3.pt` 15.9 MB 入库；`benchmarks/` 61 MB 结果；`docs/` 144 MB | ❌ |
| 内部/工作文件 | 绝不入库（私有 worktree） | `agent_work/`、`.hermes/`、`agent_memory/` 提交进 Git | ❌ |
| 标准工具链 | pre-commit、Makefile/justfile、`.editorconfig`、Dockerfile | **全部缺失**，仅 `pyproject.toml` | 🟡 |
| 测试分类 | marker：`unit`/`integration`/`gpu`/`slow`/`benchmark` | 仅定义 `slow` 一个 marker | 🟡 |
| 根目录卫生 | 只放元数据与入口（README/LICENSE/pyproject/…） | 44 个根 `.py`、13 张根 `.png`、12 篇根 `report_*.md` | ❌ |
| Monorepo 布局 | 清晰 `packages/` 或顶层分离 + 工作区清单 | 3 个可部署组件（库/后端/前端）同仓，无工作区清单 | 🟡 |
| 安全治理 | CODEOWNERS、SECURITY.md | 缺失（虽有 `app/backend/security.py`） | 🟡 |

---

## 4. 关键问题（按优先级）

### P0 — 应立即治理（零算法风险，主要是“别把不该入库的入库”）
1. **内部 Agent/CI 工作文件泄漏**：`agent_work/`（21）、`.hermes/`（含 replay 证据、CI 日志）、`agent_memory/`、`evidence/` 提交进了仓库。这是流程卫生问题，也可能含敏感日志。→ 加入 `.gitignore`，可选 `git rm --cached`。
2. **15.9 MB 模型权重裸入库**：`models/suboff_v0.3.pt` 虽在 `large_files` 例外清单里，但违背仓库自身 `.gitignore` 注释“never in-tree”。→ 迁移到 Git LFS 或 Release/HF Hub 资产，`git filter-repo` 或后续提交移除跟踪。
3. **运行产物入库**：`runs/`（392）、`benchmarks/` 的 878 个结果 JSON/195 日志、`experiments/` 大量输出。`.gitignore` 已忽略 `outputs/`，但 `runs/` 等未统一。→ 结果收敛到被忽略的 `outputs/` 或 `artifacts/runs/`，历史结果归档到 Release/外部存储。

### P1 — 文件组织规范（低风险，需更新既有策略清单）
4. **根目录调试脚本**：44 个 `dbg_*`、`l1_d2_*`、`verify_*`、`bfl_q_audit*` 等。其中 42 个已被 `legacy_root_files` 祖父化——等于“承认脏但放着”。→ 分类后迁入 `tools/scratch/`（可复现工具）或 `experiments/archived/`，并从策略清单移除，让根目录只留元数据。
5. **根目录报告与图片**：12 篇 `report_*.md` + 13 张 `fig_*.png`。→ `report_*.md` 迁 `docs/reports/`；`fig_*.png` 迁 `docs/assets/`（及其引用链接需一并更新）。
6. **策略检查器只管“根文件”，不管“顶层目录”**：`agent_work/`、`.hermes/`、`evidence/`、`runs/`、`experiments/` 等目录级膨胀完全未被约束。→ 给检查器加“顶层目录白名单”，不在白名单的目录直接报错。

### P2 — 结构与边界（需遵循现有治理规则的 Phase 3+）
7. **扁平大包**：353 个模块平铺 + 2,903 行 `__init__`。→ 按物理域（碰撞/边界/AMR/多相/海洋/AI）拆分子包，薄化 `__init__`，用兼容性 shim 保旧导入。**必须在导入图审计 + 公共 API 映射之后做**，本评审不授权批量搬迁。
8. **巨石模块**：>2,500 行的 `aircraft_icing.py`、`postprocess.py`、`autograd_calib.py` 等。→ 按职责拆文件（如 `postprocess` 拆 validator / exporter / routes），单 PR 小步。
9. **应用↔库边界渗透**：`main.py` 运行时 `sys.path` 注入 + 动态导入 + 宽 `except`。→ 改为依赖已安装的 `tensorlbm`（开发态用 `pip install -e .` 或显式 `PYTHONPATH`），移除运行时路径 hack。

### P3 — 工程体验（加分项）
10. **缺标准工具链**：加 `.pre-commit-config.yaml`（ruff + ruff-format + mypy + 布局检查器）、`Makefile`/`justfile`（`make lint/test/format/serve/ci-local`）、`.editorconfig`、`.dockerignore`、可选 `Dockerfile`。
11. **测试分类标记不足**：补充 `unit`/`integration`/`gpu`/`benchmark`/`regression` marker，并在 CI 分 job 运行（与现有 `slow` 体系对齐）。
12. **安全治理**：加 `CODEOWNERS` 与 `SECURITY.md`。
13. **示例组织**：`examples/` 有 153 个根 `.py`，已有分主题子目录但覆盖不全 → 继续按物理场景归类。

---

## 5. 改进路线图（遵循现有治理，不触碰数值算法）

对齐 `REPOSITORY_REFACTORING_PLAN.md` 的安全规则（不改变数值算法；搬迁前先搜引用；保公共导入；小 PR）。

- **Phase A（本评审建议立即做，P0/P1，零算法风险）**
  1. 扩展 `.gitignore`：忽略 `agent_work/ .hermes/ agent_memory/ evidence/`（含已跟踪文件的 `git rm --cached`）。
  2. 收紧大产物：`.pt/.pth/.ckpt` 一律 Git LFS/Release；`runs/`、`benchmarks/**/results`、`experiments/**/out` 忽略，历史结果外移。
  3. 根脚本/报告/图片分类迁移，并同步更新 `repository_layout_policy.json` 的 `legacy_root_files`。
  4. 给 `check_repository_layout.py` 增加“顶层目录白名单”。
- **Phase B（P3 工程体验）**：pre-commit、Makefile/justfile、`.editorconfig`、Dockerfile、测试 marker、CODEOWNERS/SECURITY。
- **Phase C（P2 结构，需先审计）**：导入图 + 公共 API 映射完成后，再做子包拆分、`__init__` 薄化、巨石模块拆分、应用边界硬化。

---

## 6. 不建议现在做的

- **批量移动 `src/tensorlbm/` 核心模块**：会静默破坏 786 个惰性导出与外部导入，必须先有迁移清单与基线对比（既有审计已明确不授权）。
- **任何数值算法改动**：文件组织整改与算法修正必须分 PR。
- **一次性 `git rm` 大量“证据”文件**：证据要保留可溯源，应先迁外部存储而非删除。

---

## 附：本评审的“零风险立即可执行清单”

1. 新建 `.gitignore` 追加：`agent_work/`、`agent_memory/`、`agent_memory_context_update.md`、`.hermes/`、`evidence/`（连同已跟踪文件的 `git rm --cached`）。
2. 新建 `.pre-commit-config.yaml`、`Makefile`、`.editorconfig`、`.dockerignore`。
3. 将根目录 44 个 `dbg_*`/`l1_d2_*`/`verify_*` 等迁入 `tools/scratch/`（保留可复现脚本），更新策略清单。
4. 将 12 篇 `report_*.md` 迁入 `docs/reports/`、13 张 `fig_*.png` 迁入 `docs/assets/`。
5. 为 `check_repository_layout.py` 增加顶层目录白名单校验。
6. 补充 pytest marker（`unit`/`integration`/`gpu`/`benchmark`/`regression`）与对应 CI job。

> 以上 1–5 不改变任何数值算法，且多数已被本仓库既有治理文档认可为“低风险治理”。第 6 项仅改 `pyproject.toml` 与一个 CI job，亦零风险。
