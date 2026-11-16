# LBM-P0-RUFF — lint ratchet 3052 → 3023 (target ≤3049)

状态：完成（本地验证通过；CI 侧 `lint` job 需在平台跑一次 workflow 才算最终收红）
台账条目：`LBM-P0-RUFF` — “CI lint ratchet 收红 3052→≤3049”，验收 `{ci: lint_job_green, metric: ruff_count<=3049}`

## 1. 案例背景（为什么这个"小活"是真缺陷）

`src tests examples benchmarks` 上的 `ruff check` 在本单开工时实测 **3119** 条，比台账记录的 3052 又涨了 67。
CI 的 `lint` job 里是一条硬门 `ruff check src tests examples benchmarks`，也就是说这个 job
**长期是红的**，红的原因与任何一个 diff 都无关（历史遗留债），于是没人再看它的输出——
门禁实际上已经失效。ratchet（棘轮）是把"必须为 0"改成"绝不许变差"：把实测值固定进基线文件，
只在**回归**时失败，且基线只允许下调。这样债务单调偿还，而不是被冻结或继续膨胀。

更关键的是，3119 条里有 **100 条是 `invalid-syntax`**：4 个已提交的 benchmark 脚本
**根本无法被 Python 解析**（死代码，永远不可能运行）：

| 文件 | 语法错误 | 根因 |
|---|---|---|
| `benchmarks/in99_n03_channel_turb/run_validation.py` | 61 | 字符串里的 `\n` 被写成真实换行（5 处），且双引号 f-string 内再用双引号 |
| `benchmarks/in45_channel_turb/plot_campaign.py` | 13 | `np.array(run["..."]["peaks"] and [ -1.0 ] * 0 + np.array(...))` 括号未闭合 |
| `benchmarks/contact_line/make_doc.py` | 17 | 中文正文里用 ASCII 双引号 `"…"` 嵌套，提前终止字符串（5 行） |
| `benchmarks/in45_hb_turb/run_turbulence_campaign.py` | 9 | f-string 表达式里出现反斜杠转义引号 `{cmd[\"name\"]}`（PEP 701 之前非法，target-version=py311） |

这 4 个文件是已发表算例（IN99-N03 湍流槽道、IN45 槽道/滞回湍流、IN100-N02 接触线）的
**结果后处理与文档生成脚本**——即"验证欠账"的复算入口。语法坏了，复算与图表就再也无法重跑。

## 2. 做了什么

1. **修 4 个不可解析脚本**（纯机械、保持语义）：
   - `run_validation.py`：恢复 `\n` 转义；双引号 f-string 内层改单引号（5+3 处）。
   - `plot_campaign.py`：`X and [-1.0]*0 + np.array(uv_plus)` → `np.array(run.get("uv_plus", []))`。
     原表达式求值上恒等于 `uv_plus + 0.0`，删除的是死判定，不改数值。
   - `contact_line/make_doc.py`：正文内嵌引号 `"…"` → `「…」`（不改文本语义，只改引号字符）。
   - `run_turbulence_campaign.py`：`{cmd[\"name\"]}` → `{cmd['name']}`。
2. **新增 `tools/lint_ratchet.py` + `tools/lint_baseline.json`**：三个棘轮指标
   - `ruff_check_findings`（台账口径 metric），基线 **3023**，台账目标 3049 有 26 条余量；
   - `ruff_format_unformatted_files`（`ruff format --check` 会重写的文件数），基线 2195；
   - `invalid_syntax_findings`，**钉死 0**（不可解析源码不是"债"，是缺陷）。
   CLI：`--check`（回归即 exit 1）、`--update`（只准下调，`--force` 才能上调）、`--json`。
3. **CI 接线**：`.github/workflows/ci.yml` 的 `lint` job 用
   `python tools/lint_ratchet.py --check` 取代原来必然失败的两条 `ruff check` / `ruff format --check`。
   注释里写明这是"债务棘轮而非零容忍门"，并说明基线只降不升。
4. **测试固化（15 项）**：`tests/test_lint_ratchet.py`
   - 逻辑：回归判定 / 改善判定 / 缺失指标按 0 计（不会误判为通过）；
   - 测量真实性：在 tmp_path 上构造真实 `ruff` 可检出的违规与不可解析文件，验证三个计数器非空响应；
   - CLI 行为：`--update` 拒绝上调、能锁定下调；`--check` 回归 exit 1、持平 exit 0；
   - 树不变量：基线结构 & ≤ 台账 3049、`invalid_syntax_findings == 0`、仓库当前实测不超基线、
     4 个被修脚本逐个 `ast.parse` 通过。
5. 该测试**抓到自己写的一个真 bug**：`load_baseline(path=BASELINE_PATH)` 把路径绑在默认参数上，
   运行期改不动 → `--update`/`--check` 在重定位基线下读到旧文件。改为调用期解析。

## 3. 结果（程序化注入，非手抄）

| 指标 | before | after | 目标 | 判定 |
|---|---|---|---|---|
| `ruff check` 总条数 | 3119 | **3023** | ≤3049 | PASS（余 26） |
| `invalid-syntax` 条数 | 100 | **0** | 0 | PASS |
| src | 1443 | 1443 | — | 持平 |
| tests | 901 | 901 | — | 持平 |
| examples | 46 | 46 | — | 持平 |
| benchmarks | 729 | 633 | — | −96 |

图：`docs/figures/LBM_P0_RUFF_ratchet.png`（A: 总数 vs 目标线；B: 不可解析数归零；C: 分树贡献）。
原始数字：`docs/figures/LBM_P0_RUFF_ratchet.json`。工具版本：ruff 0.15.0。

## 4. 注意点 / 局限（不粉饰）

- **这不是"lint 全绿"**。3023 条历史债仍在，只是被棘轮钉住。`ruff format` 尚有 2195 个文件未格式化，
  同样只锁不修（一次全量格式化会让 2000+ 文件的 blame 历史失效，须单独立项）。
- 台账验收里的 `ci: lint_job_green` 需要平台真跑一次 `ci.yml` 才能确认；本单只能证明
  本地 `lint_ratchet --check` 为 0 退出，且 `lint` job 中原先必然失败的两步已被替换。
  `lint` job 里还有 mypy core gate 与 docs-consistency 两步，本单未触碰、未验证。
- `ruff` 版本未锁定（dev extra 里是裸 `ruff`）。规则覆盖随版本变化会让计数漂移；
  `lint_baseline.json` 记录 `measured_with: ruff 0.15.0`，版本不一致时工具打印 NOTE 但不误判失败。
  建议后续在 dev extra 里加 `ruff>=0.15,<0.16`（未在本单做，避免动依赖锁定）。
- `contact_line/make_doc.py` 与 `in99_n03/run_validation.py` 需要本地算例产物
  （`runs/contact_line_in100_n02/`、`runs/...` 与 DNS 参考 CSV）才能端到端跑；这些产物不在树上，
  **本次未做端到端执行**，只做了 `ast.parse` + ruff 解析面验证。`in45_hb_turb` 脚本已实跑
  （生成 10 条集群命令，shell 首行 `echo 'Running Y0_n0.8...'` 正确），`in45_channel_turb/plot_campaign.py`
  已实跑到 `--help`（matplotlib + tensorlbm 导入链正常）。
- 未做 `--unsafe-fixes`、未批量 `ruff --fix`：会把 2300+ 处自动改动混进本单，超出可审查范围。

## 5. 误差判定

本单是工具链/门禁任务，**无物理基准量**，因此"是否 <3%"不适用（无解析解/实验值可比）。
可复核的量化验收是上表两个计数指标，均为程序化测量（`ruff ... --output-format=json` 计数），误差为 0
（确定性）。端到端复算的物理误差不在本单范围。

## 6. 下次自找活的线索

- 按规则族分批还债：`F401` 1455 条中 `src/` 的部分多为 `__init__` 再导出，需配合 `__all__`
  或 `per-file-ignores` 一次性治理 → 可让基线再降一档（建议新条目：`LBM-P1-RUFF-F401`，
  验收 `ruff_check_findings<=2600` 且 `import tensorlbm` 冒烟 + 全量测试绿）。
- `ruff format` 全量格式化需要单独一次"只做 format"的提交（建议 `LBM-P2-RUFF-FMT`，
  验收 `ruff_format_unformatted_files==0` 且测试全绿）。
- 复核 `F821`(13) / `F822`(14) / `F811`(20)：这些往往指向**真实运行期 bug**（未定义名、重复定义），
  比 `F401` 值得优先人工看。