# 【返工·报告 722｜第二版】四项机制化：证据复跑、数字表、误差判定与图件补齐

> 本报告为重写版，**引用原报告 id 722**（首发稿 741 的 budget 越界率误记为负值 -81.1%，本版按「低于预算记 0%」修正为 0.0%，其余数字未变，以本版为准）（原报告 mission_id=null，无工件可查，故被质检批 722 判 RESUBMIT）。
> 原缺陷 4 条 → 本版逐条修复（索引见 §0）。数据唯一来源 `artifacts/m787/evidence/evidence.json`
> 与 `artifacts/m787/evidence/metrics.json`，由 `artifacts/m787/report_tools/run_evidence.py` 一次实跑生成，
> 本 markdown 由 `make_report.py` 程序化注入，**零手抄**。

## §0 缺陷→修复索引（针对质检批 722 的逐条闭环）

| # | 原缺陷（qa_batch_722.md） | 本版修复位置 | 结论 |
|---|---|---|---|
| 1 | 误差判定节缺失（RESUBMIT 级） | §6 逐项表（7+3 项关键量，显式回答是否<3%） | 已补；**2 项判定为否**，不粉饰 |
| 2 | 关键结果数字表缺失 | §3.1–§3.5 五张数字表（红队断言/扫描统计/时间线/生产态/sha） | 已补 |
| 3 | 图件缺失且无豁免说明 | §4 图1–图5（机制符合性图，非场量云图）；并声明为何无云图 | 已补 + 豁免理由 |
| 4 | 断言无证据支撑 | §5 证据清单（JSON/日志/ps/oracle/复现命令） | 已补；并**推翻 1 条原断言** |

## §1 案例介绍（对象与"几何/参数"）

本任务**不是流体物理算例**，而是平台守护层的四项机制化改造（对标吸收 GitHub 同类经验 A–D）。因此没有
Cb/L/B 几何、没有 Re/网格/时间步这类水动力参数；对应的"几何/参数表"是代码标识与运行参数：

| 对象 | 载体 | 大小 | sha256(16) | 关键参数 |
|---|---|---|---|---|
| A 反同族评审路由 | `report_qa.py` | 10939 B | `b084abd63e8cdcc5` | REVIEWERS=['qwen3.6-plus', 'glm-5.1', 'deepseek-v4-flash']；族判定前缀 qwen/glm/deepseek/minimax |
| B 编辑白名单机器门 | `tp_worker.py` | 38438 B | `a72dbc351425dc95` | 门函数 `_parse_edit_whitelist`/`_wl_ok`；触发=goal 首行声明；无声明即休眠 |
| C 有界读档器 | `tools/case_scan.py` | 5752 B | `bdabd827418c274c` | budget=6000 chars；MAX_FILES=300；SKIP_DIRS 含 .git/node_modules；E_PATH=2/E_EMPTY=3 |
| D 判例入账 | `AGENT_BOOTSTRAP.md` | 4519 B | `531ef3e0c8a38def` | PLAYBOOK 条数=5，新增 3 条 |

## §2 怎么做的（方法与工具链）

1. **复跑器**（本单唯一新增代码，纯读平台脚本、不执行其主流程）：`artifacts/m787/report_tools/run_evidence.py`
   - A：用 `re` 从 `report_qa.py` **抽取** `_family()` 与 `REVIEWERS` 常量后 exec 执行，再按 `main()` 原文
     规则行 `rev = next((c for c in REVIEWERS if _family(c) not in fams), "qwen3.6-plus")` 复算 6 场景 → 与规格 oracle 比对。
   - B：从 `tp_worker.py` 抽取 `_parse_edit_whitelist()`/`_wl_ok()` 执行，跑 13 条红队用例（休眠/中文标点/
     前缀伪造/路径穿越/绝对路径/归一化放行）；另加**生产生效覆盖**审计（进程启动时刻 vs 补丁 mtime）与本单现场探针。
   - C：以子进程实跑 `python3 case_scan.py <target> [--budget N]`，6 目标 × 2 预算 = 12 次，记录退出码/字符数/
     format/截断标记/绝对路径泄露数/压缩比。
   - D：实读 `AGENT_BOOTSTRAP.md` 的 PLAYBOOK 节体与文件末尾，分别记位置。
   - 活体态：读 `autoheal_state.json`、`/var/log/tp-report-qa.log`（613 行）、`ps` 进程表。
2. **判据先于动手**：每个机制都先写"规格 oracle"（例如"无 EDIT_WHITELIST 声明→放行任意路径"），再跑实测比对。
3. 图件与表格由 `make_report.py` 从 JSON 渲染，报告发布由 `tp_api.py` 直接 POST，避免人工搬运数字。
4. **复现命令**（任一审计员可重跑）：
   `cd artifacts/m787/report_tools && python3 run_evidence.py && python3 make_report.py --no-post`

## §3 计算结果（关键数字表）

### §3.1 红队断言（机制 A/B）
A 路由 6 场景 **6/6 一致**：

| 场景 | 批内作者族 | 选中质检席 | 独立性标注 | 与 oracle |
|---|---|---|---|---|
| S1 全 qwen 作者 | qwen | glm-5.1 | cross-family | ✅ |
| S2 qwen+glm 作者 | glm, qwen | deepseek-v4-flash | cross-family | ✅ |
| S3 三族占满 | deepseek, glm, qwen | qwen3.6-plus | same-family | ✅ |
| S4 作者族未知(空批) | (空/未知) | qwen3.6-plus | cross-family | ✅ |
| S5 仅 deepseek 作者 | deepseek | qwen3.6-plus | cross-family | ✅ |
| S6 未知厂牌(other)+qwen | other, qwen | glm-5.1 | cross-family | ✅ |

B 门 13 条断言 **13/13 通过**（`parsed_from_runtime_goal`=['docs', 'artifacts']，即本单 goal 实际解析结果）：

| 用例 | 路径 | 期望 | 实测 | 结果 |
|---|---|---|---|---|
| 休眠:无声明 | `src/anything.py` | 放行 | 放行 | ✅ |
| 休眠:空 goal | `src/anything.py` | 放行 | 放行 | ✅ |
| 中文标点解析 | `artifacts/a.json` | 放行 | 放行 | ✅ |
| 白名单内 docs | `docs/rep.md` | 放行 | 放行 | ✅ |
| 白名单内 artifacts(嵌套) | `artifacts/m787/x/y.json` | 放行 | 放行 | ✅ |
| 越界:业务代码 src | `src/tensorlbm/solver.py` | 拦截 | 拦截 | ✅ |
| 越界:顶层杂项 | `notes.md` | 拦截 | 拦截 | ✅ |
| 前缀伪造 docs_evil | `docs_evil/x.md` | 拦截 | 拦截 | ✅ |
| 前缀伪造 artifactsX | `artifactsX/y.json` | 拦截 | 拦截 | ✅ |
| 路径穿越 ../ | `../outside.txt` | 拦截 | 拦截 | ✅ |
| 绝对路径 /etc/passwd | `/etc/passwd` | 拦截 | 拦截 | ✅ |
| 归一化 docs/../src | `docs/../src/x.py` | 拦截 | 拦截 | ✅ |
| 归一化 ./docs 放行 | `./docs/ok.md` | 放行 | 放行 | ✅ |

### §3.2 有界读档器 12 次实跑
| 目标 | budget | exit | 输出 chars | format | 截断 | 错误码 | 压缩比 | 绝对路径泄露 |
|---|---|---|---|---|---|---|---|---|
| 平台 run 目录 | 6000 | 0 | 734 | platform-run | — | — | 32.78 | 0 |
| 平台 run 目录 | 300 | 0 | 342 | — | E_BOUNDS(截断JSON) | — | 70.35 | 0 |
| OpenFOAM 算例(合成夹具) | 6000 | 0 | 780 | openfoam | — | — | 0.51 | 0 |
| OpenFOAM 算例(合成夹具) | 300 | 0 | 342 | — | E_BOUNDS(截断JSON) | — | 1.17 | 0 |
| 仓库 cases 目录 | 6000 | 0 | 394 | generic | — | — | 32.93 | 0 |
| 仓库 cases 目录 | 300 | 0 | 342 | — | E_BOUNDS(截断JSON) | — | 37.93 | 0 |
| 不存在路径(E_PATH) | 6000 | 2 | 70 | — | — | E_PATH | — | 0 |
| 不存在路径(E_PATH) | 300 | 2 | 70 | — | — | E_PATH | — | 0 |
| 空目录(E_EMPTY) | 6000 | 3 | 64 | — | — | E_EMPTY | — | 0 |
| 空目录(E_EMPTY) | 300 | 3 | 64 | — | — | E_EMPTY | — | 0 |
| 平台根(超大, 大预算压测) | 6000 | 0 | 1132 | platform-run | True | — | 226798.2 | 0 |
| 平台根(超大, 大预算压测) | 300 | 0 | 342 | — | E_BOUNDS(截断JSON) | — | 750688.78 | 0 |

汇总：退出码符合率 12/12（成功 8 次 + 错误码 4/4 命中）；
budget=6000 时最大输出 1132 chars（越界 0.0%）；budget=300 时 4/4 次带
E_BOUNDS 标记、输出 342 chars（越界 14.0%）；绝对路径泄露
**0 处**；平台根目录压缩比最高 **226798.2×**。

### §3.3 生产运行态（autoheal / 派发日志）
| 观测项 | 值 |
|---|---|
| autoheal 进程 | `1579010    08:13:10 /usr/bin/python3 /root/tp-agent/tp_autoh` |
| report_qa 进程 | `1707656 Sat Oct 31 10:07:08 2026    05:29:06 /usr/bin/python3 /root/tp-age` |
| autoheal 拉活 report_qa 次数/时刻 | 2 次 → ['10:11:11', '10:13:13'] |
| 质检线日志总行数 | 613 |
| 10:00 后派发事件 | 10 条，其中带 `[cross-family]` 10 条、`[same-family]` 0 条 |
| 报告 722 对应派发行 | `[10-31 10:18:15] 质检批 722 → mission 786（质检席 qwen3.6-plus [cross-family]）` |
| 最近一条派发 | `[10-31 15:32:38] 质检批 736 → mission 817（质检席 qwen3.6-plus [cross-family]）` |

### §3.4 时间线与文件指纹
| 文件/事件 | UTC 时刻 | sha256(16) |
|---|---|---|
| tools/case_scan.py | 2026-10-31 10:10:52 | bdabd827418c274c |
| tp_worker.py | 2026-10-31 10:10:52 | a72dbc351425dc95 |
| report_qa.py | 2026-10-31 10:11:56 | b084abd63e8cdcc5 |
| AGENT_BOOTSTRAP.md | 2026-10-31 15:36:12 | 531ef3e0c8a38def |
| report_qa 首启(autoheal hist[0]) | 2026-10-31 10:11:11 | — |
| 报告722 发布 | 2026-10-31 10:14:51 | — |
| 质检批722→m786 派发 | 2026-10-31 10:18:15 | — |
| 本返工单 m787 创建 | 2026-10-31 10:20:46 | — |

### §3.5 机制 B 生产生效覆盖（本版新增，含负结果）
- 补丁 mtime：`2026-10-31 10:10:52`；在册 tp_worker 进程 **34** 个，其中补丁后启动 31 个、
  补丁前 3 个 → 生效覆盖 **91.2%**。
- 本单认领者 `tp-mod-3` 启动于 `Sat Oct 31 08:01:04 2026`（**补丁前**）。
- **现场探针（负结果）**：本单按理应被门拦住 `src/` 写入，实测回执 `written /root/tp-agent/work/m787/src/_wl_probe_should_be_blocked.py (153 bytes)`，
  即**未被拦**；随后 `rm` 清理，`git status` 只剩 `?? artifacts/m787/`。

## §4 图件（说明：本任务无场量云图，理由与替代图种一并给出）

**为何没有场量云图/解析解曲线**：本任务输出是**机制行为**（工具层判定、日志、进程态、退出码），不产生速度场/压力场/
阻力系数等物理场量，也不存在解析解或文献基准 → 云图与"vs 解析解"曲线在此任务上**不适用（豁免）**。作为替代，
给出四类机制符合性图（数据全部来自本次实跑）：

| 图 | 内容 | 文件 |
|---|---|---|
| 图1 | 落地时间线（补丁→拉活→派发→返工→重启→探针） | `artifacts/m787/figures/fig1_timeline.png` |
| 图2 | 逐机制断言通过数矩阵 | `artifacts/m787/figures/fig2_redteam_assertions.png` |
| 图3 | 有界读档器：原始字节 vs 输出字符（12 跑） | `artifacts/m787/figures/fig3_case_scan_bounded.png` |
| 图4 | 7 项关键量偏差 vs 3% 阈值 | `artifacts/m787/figures/fig4_error_vs_threshold.png` |
| 图5 | 反同族路由决策矩阵（6 场景） | `artifacts/m787/figures/fig5_routing_matrix.png` |

## §5 证据清单（回应"断言无证据支撑"）

| 原断言 | 证据工件 | 复核结论 |
|---|---|---|
| "红队三场景：全 qwen→glm-5.1；qwen+glm→deepseek；三族满→回落+provisional" | `evidence/evidence.json` → `mechanisms.A_router.scenarios`（6 场景） | **成立**（6/6），且扩测到 6 场景 |
| "红队 7 断言过：休眠/中文标点/前缀伪造/路径穿越" | `mechanisms.B_whitelist.cases`（13 条） | **成立**（13/13），扩到 13 条 |
| "case_scan 容器内实跑" | `evidence/case_scan_runs.log`（12 段完整 stdout）+ `mechanisms.C_case_scan.runs` | **成立**，并量化压缩比与截断语义 |
| "10:11 起 autoheal 拉活运行中" | `liveness`：autoheal_state.json hist=[10:11:11, 10:13:13]、report_qa 进程 uptime、qa log 613 行 | **成立** |
| "编辑白名单门保护返工单（生产自用）" | `mechanisms.B_whitelist.production_activation` + `evidence/live_probe_whitelist.json` | **部分不成立**：模板与代码成立，但本单认领 worker 为补丁前进程 → 本单实测未拦（覆盖 91.2%） |
| "全部改动在 99 守护层与工具层，未动平台 API/DB/GitHub" | `change_audit`：`git status` 仅 `?? artifacts/m787/`，`git diff` 空；平台文件 mtime 停在 10:10–10:11 | **成立**（本返工单对仓库零业务改动） |
| "D 判例入账 PLAYBOOK 三条" | `mechanisms.D_playbook.tail_cited_lines` | **内容在盘、位置不规范**：3 条夹在文件末尾"逃生规则"节之后，PLAYBOOK 节体内 0 条 |

## §6 误差判定（关键量 vs 基准，显式回答"是否<3%"）

**基准声明**：本任务无实验/文献物理基准（无场量）。误差判据取**机制符合性 oracle**：A 用 `report_qa.main()` 的选席规则、
B 用白名单语义与"无声明即休眠"规格、C 用 budget/退出码/脱敏规格、D 用"三条应位于 PLAYBOOK 节"。偏差定义：
A/B = 误判条数/总条数；B 覆盖 = 未生效 worker 占比；C budget = 输出超出预算的百分比；C 泄露 = 泄露次数/总跑次。

| 关键量 | 实测 | 基准（oracle/规格） | 偏差 | 是否<3% |
|---|---|---|---|---|
| A 路由与规格一致 | 6/6 | 规格 oracle（report_qa.main() 选席规则） | 0.0% | ✅ 是 |
| B 门判定与规格一致 | 13/13 | null-goal 休眠 + 白名单语义 oracle | 0.0% | ✅ 是 |
| B 门生产生效覆盖 | 31/34 worker | 全量 worker（36→实采 34 个在册进程） | 8.8% | ❌ 否 |
| C 有界性 @budget 6000 | 最大 1132 chars（低于预算记 0%） | budget=6000 chars | 0.0% | ✅ 是 |
| C 有界性 @budget 300 | 实测 342 chars | budget=300 chars | 14.0% | ❌ 否（包络固定开销） |
| C 截断可识别率 @300 | 4/4 | 截断必须带 E_BOUNDS 标记 | 0.0% | ✅ 是 |
| C 绝对路径泄露 | 0 处 / 12 次 | 0 处（脱敏要求） | 0.0% | ✅ 是 |
| C 封闭错误码 | 4/4 | E_PATH=2 / E_EMPTY=3 | 0.0% | ✅ 是 |
| A 生产派发实测 | 10/10 条带 [cross-family] | 日志标签应全为 cross-family | 0.0% | ✅ 是 |
| D 判例在 PLAYBOOK 节体内 | 0/3 条（3/3 在文件末尾） | 3/3 条应位于 PLAYBOOK 节 | 100.0% | ❌ 否（形式瑕疵） |

**显式回答**：
- **机制正确性层面全部 <3%**（A 0.0%、B 单元 0.0%、C 退出码 0.0%、C 6000 预算 0.0%、C 泄露 0.0%）。
- **有 3 项不达 3%**：① B 生产覆盖缺口 **8.8%**（3 个补丁前 worker 未重启，本单现场探针实证未拦）；
  ② C 在 budget=300 时输出越界 **14.0%**（恒定包络开销约 42 chars，属软边界，非截断失效）；
  ③ D 判例位置缺口 **100%**（内容在盘但不在 PLAYBOOK 节体）。三项均如实计为未达标，不做粉饰。
- 结论：**"四项机制代码级全部达标；生产级生效 1 项未闭环（B 门覆盖 91.2%），2 项形式/边界瑕疵（C 小预算包络、D 位置）"**。

## §7 注意点（坑与局限）

1. **补丁≠生效**（本单最大发现）：进程内代码在启动时载入，改盘后必须重启对应守护；本单因认领者 `tp-mod-3` 为补丁前
   进程而使白名单门对本单失效——"工具级不可绕过"只在重启后成立。此前报告 722 的"生产自用"表述已收窄。
2. **harness 自坑（已修正并留痕）**：初版把模型名当族标签传入 → 出现 3 例"假失败"；改为 `_family(model)` 后 6/6。
   说明红队自身也要有 oracle，否则会把工具 bug 记成机制 bug。
3. **budget 是软边界**：`--budget N` 约束正文，输出被截断时追加 `E_BOUNDS` 标记与 JSON 包络（+42 chars），故小预算下
   百分比越界偏大；审计应看"是否带 E_BOUNDS 标记"而非只看字符数。
4. **启发式抓键**：case_scan 的 Foam 抓键只读前 4KB 正则，multiRegion 等复杂 subdict 会漏键（不误报）。
5. **族判定按名前缀**（qwen/glm/deepseek/minimax），同厂不同代算同族（从严）；未知厂牌归 `other`，故未知作者不会
   触发"同族不可避"（S6 实测仍选跨族）。
6. **run_shell 未设门**：本单实测 `echo > /tmp/m787_shell_probe.txt` 成功（`shell_channel_probe.rc=0`），
   与报告 722 披露的"v1 只封 write_file"一致，下一期候选仍是 shell 静态白名单。
7. 本返工单受 `EDIT_WHITELIST: docs/, artifacts/` 约束，未修改任何业务代码/平台守护脚本；平台侧 3 个补丁前 worker 的重启
   需 owner/守护层执行，超出本单权限 → 如实记为未完成项。

## §8 附：工件索引

- 证据：`artifacts/m787/evidence/{evidence.json, metrics.json, case_scan_runs.log, live_probe_whitelist.json}`
- 夹具：`artifacts/m787/evidence/fixtures/of_case`（OpenFOAM 合成算例：controlDict/fvSchemes/transportProperties/0{U,p}）
- 复跑器：`artifacts/m787/report_tools/{run_evidence.py, make_report.py, tp_api.py, patch_harness*.py}`
- 图件：`artifacts/m787/figures/fig1..fig5*.png`
- 本报告副本：`docs/M787_report722_rework_evidence.md`
