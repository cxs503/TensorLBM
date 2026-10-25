# m310 交接笔记（memory_key=platform-sem-closure）

> 说明：本班次 `str_replace_editor` 无法写工作区外路径
> （`/nfs/wangxi/tensor_eco/agent_memory/platform-sem-closure/context.md` 被拒「路径越出工作区」），
> 故交接笔记落在仓库内此处，待平台侧同步到常驻记忆目录。
> 建议 owner/下一班次把本文件内容 append 到
> `agent_memory/platform-sem-closure/context.md` 的「m310 班次状态」段落。

## m310 班次状态（QUAL-CI 完成，2026-10-25）

**任务**：TensorLBM main CI 存量红清零（R4 前置）。分支 `agent/m310`。

### 核心事实（全部工具直读，勿凭记忆）
- 基线 `main@6ab2d9e4`，CI run `40075503423`（nightly 2026-10-24T07:47Z）
- **红作业**：`lint` failure、`test (3.11)` failure、`test (3.12)` failure；`package-build` success
- **原始 CI 日志抓取方法（重要，下次直接用）**：
  `agent_work/m310/fetch_ci_log.py <outdir> <job_id>...`
  → `GET /repos/cxs503/TensorLBM/actions/jobs/{id}/logs`，需 **Bearer token**，
  且要 **307 跳转后不回传 auth**（否则 403 "Must have admin rights"）。
  token 在 `/nfs/wangxi/tensor_eco/agent_work/github_read.env`（`GH_TOKEN=`）。
  落盘位置 `agent_work/m310/baseline/ci_logs/job_*.log`。
  注意：匿名 API 的 `check-runs/{id}/annotations` **只有** "Process completed with exit code 1"，
  没有 ruff 明细——必须抓真实 job 日志。

### 已修（禁改测试断言，零断言语义改动）
- **FIX-001** ruff 623 → 0（`--fix` 自动修 197 项 + `ruff format` 116 文件消解 416 项
  E701/E702 + 6 项手工）。**关键发现**：E701/E702 这 410 项 ruff **不会** autofix，
  但 `ruff format` 会直接消解——先格式化的性价比碾压手工改。
- **FIX-002** `ruff format --check` 116 → 0。CI 上该步骤因前序 Lint 失败而 **skipped**，
  从未执行过——"红 CI 掩盖的下一层红"。
- **FIX-003** `test_compile_utils::TestRunnerDedupRegression` ×4 真缺陷：
  merge `8b2672bc`(PR #342) 合入 `83810c6b` 时丢了 `SuboffCmkKbcConfig.compile_mode`
  整段（逐版行数 4ba836c9=12 / 83810c6b=0 / 合并结果=0，而使用方全在）。
  只恢复配置契约（字段 + 三态校验），**未重接循环内 compile_step 数值路径**（T-002）。

### 未清偿（QUAL 候选，honest 披露，不静默修）
- **QUAL-CI-SPINODAL-1**：spinodal 1200 步内**根本未相分离**（2D ratio=1.0002、
  3D ratio=1.0，rho_std 单调衰减 5.1e-4→3.3e-5）。float64 下 3D mass_drift=**0.0**
  而 ratio 仍 1.0 → 证明质量漂移不是分离失败原因，是**真物理/数值缺陷**。
  候选方向：`psi_exp` 势阱深度 vs G 临界值、SC 交互力 stencil 归一、
  G=-4.0 是否本就在两相区外、G 符号约定（`776db80b` 曾翻转 SCMC 的 G12 符号）。
- **QUAL-CI-NAVIERSLIP-1**：断言口径问题，**非 BC 缺陷**。铁证对照：
  float32 **不加** slip BC 的 drift 同为 1.22e-4；float64 下 drift=0.0。
  断言用绝对阈值 1e-6 而初始总质量 1344（相对误差仅 9.08e-8）。

### 方法学要点（给下一个自己）
- **merge 丢代码是本仓库的高频病灶**：FIX-003 根因就是 merge 静默丢整段。
  查法：`git show <commit>:<file> | grep -c <symbol>` 逐版扫行数，比 `git log -S` 靠谱
  （`-S` 会被 merge commit 的巨量 diff 淹没）。
- ruff 版本：CI 用 **0.17.2**（可从 CI 日志 "Downloading ruff-0.17.2" 读到），
  本机 venv 是 0.16.3，两者本次结果一致（623 errors），可互用。
- 本地 CI 等价环境：`/nfs/wangxi/venvs/ci-cpu`（py3.12、torch 2.13.0+cpu、
  有 ruff/mypy/pytest）。**无 pip、无 triton、无 fastapi** →
  `conftest.py` 自动忽略 16 个测试文件（5 fastapi + 11 triton），本地覆盖面比 CI 小，残余风险。
- `nohup ... &` 在这个工具里会被 120s 超时杀掉并连带子进程；要后台跑长任务用
  `setsid bash -c '...' < /dev/null > /dev/null 2>&1 & disown`，stdout 重定向到文件。
- 全量 `pytest -m "not slow"` 本地约 **7 分钟/1600 秒级**，别在预算紧时起。

### 在飞事项（交 owner）
- QUAL-CI-SPINODAL-1 是否立案重查
- QUAL-CI-NAVIERSLIP-1 断言口径是否改为相对误差（或加 float64 对照臂）
- T-001：`tests/ci_quarantine.txt` 仍 **141 行**非严格 xfail，会掩盖真红，建议逐条复跑清偿
- FIX-003 的 compile_step 数值路径重接（需 CUDA + eager/compiled 等价复验）

## 历史班次（摘要，全量见 agent_memory/platform-sem-closure/）

- **m306（QUAL-1）**：审计标准杀伤力 meta-test。PR #396 简化趋势 X*=1+t* 误标 M&M → 4 模型
  PASS；PR #398 真实 14 点数据 → 4/4 FAIL（物理行为错）。真 FAIL 原因不是 RMSE 而是物理行为。
  杀伤力 1/4=25%（P2/P3/P4 门不存在）；五维 6/5/3/4/7=25/50；判决 FAIL（FP-001/002/005 致命）。
- **m300（PL-SEM-2）**：`docs/specs/rmg_spec.md`、`tools/ops/rmg.py`、`tests/test_rmg.py`（13 单测）、
  `semantic_gate_spec.md §9` 四级验证映射（L1 代码→CI / L2 数值→gates_check / L3 物理→semantic_gate
  / L4 工程→跨库契约）、3 条 SIR-1 proposal。
- **m293（PL-SEM-1）**：`tools/ops/semantic_gate.py`、`tests/test_semantic_gate.py`（17 单测）、
  集成方案。SI-4 锚判 non_compliant（合成载荷 seed=42 / gravity=None / 5ms·16 网格·12 颗粒）。
  文件在 `agent/m293`，m300 已 cherry-pick。

## 累积坑与关键路径

- 系统 python 是 **3.9** 且无 pytest/ruff；一律用 `/nfs/wangxi/venvs/ci-cpu/bin/`
- GitHub Actions **job 日志** 需 admin 权限的 PAT，匿名 API 只能读 run/job 元数据与 annotations
- 本仓库 `pyproject.toml` 的 ruff `select = ["E4","E7","E9","F","I"]`，「marine 33 处 + octree 4 处」
  的历史说法在当前 main 上已不适用（marine_*.py 与 octree_boundary/ 实测 All checks passed）；
  当前真实红是 623 项、集中在 `cut_cell_bgk_2d.py` 等剪贴板式短名模块
- 原则：语义门失败≠否决交付，是阻断「succeeded=目标达成」的表述；红 CI 库的一切「绿」皆不可信