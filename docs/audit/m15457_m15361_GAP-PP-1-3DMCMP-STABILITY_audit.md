# 交差核验 · m15361 `GAP-PP-1-3DMCMP-STABILITY`（3D MCMP 稳定窗口与失稳判据工装）

- 审计单：m15457（任务榜交差核验单，非重做）
- 被审单：m15361（succeeded）→ 交付 commit `c5e38bf0`（41 files, +8535/−74）
- 落主干：`54a8c646 Merge #1802 (m15361) via merge-master`（11 files, +2040/−74）；`c5e38bf0` 本身**不是** main 祖先，主干文件 sha 与工作树逐一致（见 §1）
- 平台报告：`tensor_reports.id=14453`（六段齐全，自述 honest FAIL）
- 仓库 / 工作树：TensorLBM · `/root/tp-agent/work/m15361`（审计只读，未改被审单任何文件）
- 审计工件：`artifacts/m15457/`（复现日志、探针 JSON、pytest 输出、图 `fig/m15457_audit.png`）
- 分级结论：**L3**（判据与差异见 §6）；台面裁判 `tensor_benchmark_cases` 对该锚 **0 行**，L4 结构上不可达（§3）

## 1. 三件套核验（run_artifacts / benchmark ref-vs-got / repro_cmd）

| 件 | 结论 | 证据 |
|---|---|---|
| ① run_artifacts 文件 | **PASS（分支内）/ 主干被门禁滤除** | 分支 commit 内 `artifacts/mcmp_3d_stability/{scan_coexist×13, scan_coexist_repro×8, scan_water_air×8, summary.json}`；**主干预 `54a8c646` 仅有 `benchmarks/verified/mcmp_3d_stability/result.json`（797 行，表内嵌）+ 2 张 png**，`artifacts/mcmp_3d_stability/**` 全部未入库（`git ls-tree -r 54a8c646 \| grep mcmp_3d_stability` = 4 项，无 artifacts/） |
| ② benchmark ref vs got | **PRESENT 但仅一层（动量）且裁判缺行** | ref `u = g t = 2.0e-2`（解析）→ got `1.0000051e-2·t`，相对误差 **5.1e-6（0.0005%）** < 3%，独立复现一致（§4.1）。稳定性窗口本身**无外部/文献参考值**；`tensor_benchmark_cases WHERE anchor_slug='GAP-PP-1-3DMCMP-STABILITY'` = **0 rows**（全表 16 行，唯一 MCMP 行 = `GAP-LBM-MCMP-POISEUILLE`（2D 分层 Poiseuille 解析，status=pending，异锚）） |
| ③ repro_cmd 可执行 | **PASS（部分）** | `PYTHONPATH=src python cases/mcmp_3d_stability/run_stability_scan.py --stage all --workers 8`：CLI 参数存在（`run_stability_scan.py:152-153`），审计实跑 `--stage a --workers 4` → **8/8 记录逐位一致**（u_max / u_max_final 与交付完全相同）。但交付的 13 条 `scan_coexist` 中 **G=−3.0…−5.0 这 5 条脚本产生不出来**（`G_GRID` 仅 8 值，`run_stability_scan.py:49`）→ `summary.json` 的 `boundary_kind="interval"/−2.875` 无法由文档命令重建，审计复跑只能得到 `open/−2.75`（§4.2） |

## 2. 报告六段与报告诚实度

`benchmarks/verified/mcmp_3d_stability/REPORT.md` 六段齐全（§1 问题与锚、§2 方法与缺陷修复、§3 参数、§4 逐判据结果、§5 敏感性、§6 失效模式与工件）。三条预声明判据在报告与 `result.json.criteria_verdict` 中一致地记为 FAIL（`boundary_repro_delta=0.125`、`anchor_5000_steps_ok=false`、`delta_rho_ge_100_in_stable_window=false`），`tensor_reports 14453` 摘要亦如实写「Δρ≥100 与 5000 步 anchor 未达成（honest FAIL）」——**无粉饰**，这点合格。

## 3. 基准裁判（唯一裁判）核验

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='GAP-PP-1-3DMCMP-STABILITY';   --> (0 rows)
```
→ 该锚在机器判据池里没有条目：`maturity_gate` 没有可机器复核的 ref/tolerance，"报告好看不算数"在这里等价于**没有任何数值裁判**。审计按 L 判据的 `L4=基准验证` 一项判**不成立**（不是因为误差超标，而是裁判缺行 + 被审单自己的三条判据全 FAIL）。

## 4. 审计侧独立复现（全部本单自跑，非引用被审单输出）

### 4.1 Guo 强制项过注入缺陷：真缺陷、真修复

同一驱动器（均匀 ρ=1、8³、g=1e-4、200 步、τ=1，判据 u=g·t）：

| 路径 | u/(g t) | 来源 |
|---|---|---|
| `use_guo=False`（速度平移） | 0.9999945 | 审计跑 |
| `use_guo=True`，**修复前**（父提交 `fd81edf2` 的 `multiphase3d.py`） | **1.5000165** | 审计跑（与交付自述 1.50002 一致） |
| `use_guo=True`，**修复后**（`c5e38bf0`） | 1.0000051 | 审计跑 |

⇒ 1.5×（τ=1 时 `2−1/(2τ)`）过注入确为真缺陷，修复量级正确；`tests/test_mcmp3d_stability.py::test_guo_forcing_injects_exactly_the_applied_force` 在修复后通过、在父版本文件下失败，是有效回归测试。

### 4.2 复现偏差：交付 summary.json 不可由交付命令重建

| 项 | 交付 `summary.json` | 审计复跑 `--stage a` |
|---|---|---|
| `scan_coexist` 记录数 | 13（含 G=−3.0/−3.5/−4.0/−4.5/−5.0） | **8** |
| `last_stable` | −2.75 | −2.75 |
| `first_unstable` | −3.0 | **None** |
| `boundary_kind` / `boundary_estimate` | interval / **−2.875** | open / **−2.75** |

多出来的 5 条记录来自 `.scratch/probe2.py`（scratch 探针，已入库但不在文档化的 repro 路径上）。⇒ "稳定区间 (−2.75,−3.0)" 这一结论的一半证据不可由交付的 repro 命令再生。

### 4.3 `water_air` 族「全耦合 50 步发散」是初始化瞬态，不是耦合失稳

交付表把 `water_air` 族 8 个耦合值全记 `diverged @ step 50`（reason `non-finite moment`）。审计同参数复跑，仅把界面宽度 `interface_width` 由 2.0 改为 6.0（其它不动）：

| iw | 结果 |
|---|---|
| 2.0（交付设置） | diverged @50，`non-finite moment`，`u_max_peak=0.1067`（第 1 步即 0.107 = Ma 0.19） |
| 6.0 | **300 步不发散**，`u_max_peak=0.1568`，Δρ 正常演化 |

⇒ 该族表的全部发散发生在初始 tanh 剖面力不平衡的松弛瞬态（第 1 步 u_max 已达 0.107），与 `G_12` 无关；把它当作"该族耦合轴上的扫描结果"是**没有信息量的**。报告未做此诊断。

### 4.4 自耦合符号约定的"已验证"声明不成立

`docs/mcmp3d_stability_findings.md` §2 声称符号约定 "Verified in `tests/test_mcmp3d_stability.py::test_self_coupling_sign_convention`"。实测：
- 该测试在交付 commit **失败**：`growth(0.5)=9.18e-4`，断言要求 `>1.0`；
- 审计独立探针（n=8，100 步，扰动幅度 0.05）：`G_self=+0.5 / 0.0 / −0.5` 的 growth 全为 `1.0e-5`（三者相同）⇒ 该指标对符号不敏感（扰动被 BGK 扩散吃掉），断言在结构上不可达；
- 代码层面符号一致性可**由读码**成立：自耦合项 `−G_σσ ψ Σw c ψ` 与交叉项 `−G_12 ψ1 Σw c ψ2` 同算子同符号（`multiphase3d.py:128-199`），线性 ψ 下 EOS 贡献 `(G/2)ψ²`。

⇒ 结论：**约定大概率对（读码可信），但"已被测试验证"是不成立的声明**。

### 4.5 交付测试套件非绿

`PYTHONPATH=src pytest tests/test_mcmp3d_stability.py -q` @ `c5e38bf0`：**2 failed, 3 passed, 17.06 s**
- `test_self_coupling_sign_convention`（§4.4）
- `test_stability_boundary_orders_by_magnitude`：`KeyError: 'G_12=-2.5'` —— `stability_boundary()` 的 `states` 键取 `str(r[axis])`（即 `"-2.5"`，`mcmp3d_stability.py:392`），测试期望 `"G_12=-2.5"`；实现/测试键格式不一致（不影响带 `name` 的实跑记录）。

## 5. 集成调用面（L3 依据）

`collide_sc_two_component_3d` 在主求解器/应用路径上的调用方（均 `file:line` 实查）：

| 调用方 | 是否走被修的 Guo 分支 |
|---|---|
| `src/tensorlbm/dam_break_3d.py:666,677,703`（配置 `use_guo: bool = True`，`dam_break_3d.py:156`） | **是，且为默认路径** |
| `src/tensorlbm/porous_media3d.py:410` | 否（默认 `use_guo=False`） |
| `src/tensorlbm/multiphase_water_entry.py:769` | 否 |
| `src/tensorlbm/multiphase_benchmarks.py:1014` | 否 |
| 公共 API 导出 `src/tensorlbm/__init__.py:519,2553` | — |

⇒ 修复不是孤立工件：3D 溃坝应用**默认**启用 `use_guo=True`，修复前该默认路径动量被过驱动 1.5×（τ=1）。

## 6. 分级判定 = L3

| 级 | 判据 | 结论 |
|---|---|---|
| L0 文档 | — | 超越 |
| L1 数学/单元验证 | 动量解析 `u=g t` | **达成**：5.1e-6，且证明修复前 1.500017 |
| L2 独立功能实现 | 稳定窗口工装真代码、输入→计算→输出、确定性可复现 | **达成**：`mcmp3d_stability.py` 393 行；stage-A 8/8 逐位复现 |
| L3 系统集成 | 接入主求解器/平台 | **达成**：`multiphase3d.py` 主求解器内核 + 默认开启的 `dam_break_3d` 应用路径 + 公共 API 导出（§5）；`G_self=0`/默认 ψ 位级不变（回归测试通过） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现，**裁判行存在** | **不达成**：裁判 0 行（§3）；被审单三条预声明判据全 FAIL；`summary.json` 不可由 repro 重建（§4.2）；worked-family 表为初始化瞬态（§4.3）；交付测试 2/5 红（§4.5） |
| L5 科研工程成果 | 稳定窗口/Δρ≥100 达成 | **不达成**：Δρ≥100 在 linear-ψ 下结构不可达，报告如实记 FAIL；窗口边界复现性判据 FAIL |

> 级别 = 功能实质，不以报告篇幅计。附 106 行 REPORT.md + 224 行 make_report.py 不改变级别；反过来"继承了既有内核"也不抹掉真增量（Guo 缺陷修复 + 自耦合/ψ 接口 + 可复现工装）。

## 7. Findings（按严重度）

- **F1 阻塞（裁判）/ 平台侧·非被审单责任**：`tensor_benchmark_cases` 无 `GAP-PP-1-3DMCMP-STABILITY` 行（16 行全表，0 命中）。该锚的任何交付都无法被机器复核 → 只能由审计独立复算代行。
- **F2 高**：交付 commit `c5e38bf0` 处自带测试 **2/5 失败**（§4.5）；`docs/mcmp3d_stability_findings.md` §2 的"sign convention verified in tests::test_self_coupling_sign_convention"因此是**不成立声明**，需撤回或改为"读码一致性 + 有效测试待补"。
- **F3 高**：`summary.json` 结论（interval / −2.875）**不可由文档化 repro 命令再生**（`G_GRID` 8 值 vs 交付 13 条记录，`run_stability_scan.py:49`）；审计复跑得 open / −2.75。多出的 5 条来自 `.scratch/probe2.py`，属未文档化的第二运行路径。
- **F4 高（解释性）**：`water_air` 族 8/8 `diverged @50` 是 tanh 初值瞬态（iw=2.0 时第 1 步 u_max=0.107），非耦合失稳：同参数 iw=6.0 跑 300 步不发散（§4.3）。该族表对"失稳判据 vs 耦合"无信息量，报告中未诊断。
- **F5 中**：`REPORT.md` 扫描表的 `u_max` 列，对 diverged 行实际填的是 `u_max_peak`（工件里 `u_max=null`，`u_max_peak=0.048076` 等）——列名与量错配（数值本身可溯源，不虚增精度）。
- **F6 中（证据完整性）**：主干 `54a8c646` 丢弃了全部 `artifacts/mcmp_3d_stability/**`（含 summary.json 与逐例 JSON），仅 `result.json` 内嵌表留档；三件套①在主干上只剩单文件形态。
- **F7 低**：`stability_boundary()` 的 `states` 键在记录缺 `name` 时退化为 `str(value)`（`"-2.5"`），与测试期望 `"G_12=-2.5"` 不一致（§4.5 第二条失败根因）。
- **F8 信息（正面）**：Guo 缺陷与修复为真（1.500017 → 1.0000051，§4.1），且落在默认开启的 3D 应用路径上（§5）——这是本单最实的增量。
- **F9 信息（正面）**：stage-A 全部 8 例 u_max/u_max_final 逐位复现，工装确定性达标；三条验收判据如实记 FAIL，无粉饰。

## 8. 未做 / 局限（不得视为通过）

- 未做依赖面回归全扫描：`tests/test_mcmp_mixture_3d.py` 本地 115 s 未跑完（rc=124），`test_dam_break_3d*.py` 未跑 → **"Guo 修复是否影响与会依赖既有验收数"= NOT RUN**。这是本单最大未知：`dam_break_3d` 默认走被修分支，若其历史验收数是在 1.5× 过驱动下调出来的，需重标。
- 未跑 `--stage all`（b/c/d 三段：噪声复现、分辨率、5000 步 anchor）；anchor 段在交付里本就是空 `[]`，审计未代跑。
- 未判定 linear-ψ 下 Δρ 上限的理论值（报告称"结构上 ~4"，属定性论证，未做 Maxwell 构造核验）。
- 未改被审单任何代码（交差核验单，非重做）；F2/F3/F5/F7 的修复归返修轮。

## 9. Disposition

**approve with notes → 判 L3（confirmed 预期）**。能力实质：主求解器 Guo 强制项真缺陷修复（1.5×→1×，落在默认 3D 应用路径）+ 自耦合/非线性 ψ 接口（默认位级不变）+ 确定性稳定窗口工装。**不得升到 L4**：该锚无机器裁判行、三条预声明判据全 FAIL、交付测试 2/5 红、头号工件（summary.json）不可由交付命令再生。返修建议（按性价比）：①补 `tensor_benchmark_cases` 该锚判据行（至少把 `u=g t` 与一个 2D/立面解析锚挂上）；②把 `G_GRID` 与实际扫描一致（或把 −3.0…−5.0 段并入 runner）；③`water_air` 族改用带力平衡的初值或声明为 init 瞬态；④修 `states` 键与两条红测试；⑤撤回 findings §2 的"已验证"表述。