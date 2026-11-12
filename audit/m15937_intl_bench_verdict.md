# m15937 · 交差核验：m15912 (GAP-LBM-INTL-BENCH · TensorLBM 国际对标专项审计)

判定：**L2（独立功能实现），附一条 blocking 缺陷（F1 假 verified 漏口）**。仅凭 L2 消账会放过 F1；
建议在 v1.1 修 F1 + 补一条控制组（"聚合 flag=false 但个别 gate=true"）后我认为本工具才配叫"防自我宣布 verified"。

## 审计对象与身份
- 交付分支 `agent/m15912` @ `01748f48`（20 文件，+? 行；日志标题「国际对标审计产品化 — 证据层级评分器入包 + 校准基准 + 落库」）
- 母单 m15912 status=succeeded, accepted=true, decided_by=eco_admin；报告 tensor_reports id=15053
- 复现沙箱：`git archive 01748f48 → /tmp/a15912`（不与交付工作树共享状态），本仓 `audit/` 存复核产物
- **合并状态：未合入**。`git merge-base --is-ancestor 01748f48 HEAD` = false（工作树 main = `123fe6b8`，ECO 检出 = `53046bcc`）；
  `git merge-tree --write-tree HEAD 01748f48` 干净 → 不是冲突，是 push/merge 通路漏了这单（同批更晚的 m15918 已在 main）

## 一、三件套核验（①取证）
| 项 | 结论 | 证据（真实指针） |
|---|---|---|
| impl_files 非占位 | 真 | `src/tensorlbm/audit/intl_bench.py` 623 行（`grade` L220、`apply_evidence_isolation` L276、`check_citations` L306、`compete` L349、`run_audit` L427、`scorecard` L575） |
| entry | 真 | `python -m tensorlbm.audit`（`src/tensorlbm/audit/__main__.py`）、`tools/audit/run_intl_bench_audit.sh:17`、包 API `src/tensorlbm/api.py:61-62,139-140` |
| run_artifacts | 真，12 件 | `artifacts/intl_bench_v1/run1/{summary.json,module_maturity.json,module_maturity.csv,compete_score.csv,maturity.sql,repro.txt,run.log,manifest.sha256,*.png×3}` |
| tests | 真，10 passed（我复跑 9.79 s） | `tests/test_intl_bench_audit.py`（175 行；层级单调/无裁决不 verified/缺陷不 verified/证据复用降档/上游缺失不崩/CLI 端到端/真实仓无未回链 verified/确定性/api 侧真调用） |
| repro_cmd 可执行 | 是，我复跑一致 | `repro.txt` 的 `PYTHONPATH=src python3 -m tensorlbm.audit --repo-root . --out …` → exit 0；`summary.json` = 45 模块 (verified 4 / partial 24 / implemented 15 / failed 2)，引用 51/51（OpenLB 12、waLBerla 16、Palabos 13、Lettuce 10），`citation_failures: []` |
| 校准基准 | 复跑 PASS，10/10（100.0%） | `benchmarks/intl_bench_calibration/run_benchmark.py` → `RESULT.json`：sphere_re200 ref 0.805615 / got 0.798671 = **-0.862%**（<3%），网格 span 2.728%（<3%）；hadamard 正确拒绝 verified（got=partial） |
| 落库 | 真 | `tensorlbm_module_maturity` 45 行（4/24/15/2）, `tensorlbm_compete_score` 10 行（week 2026-W46） |

## 二、基准裁判（平台台账）
`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='GAP-LBM-INTL-BENCH'` → **0 行**
（全表 25 行，LBM 侧仅 `GAP-LBM-SPHERE` / `GAP-LBM-MCMP-POISEUILLE`）；`tensorbench_results` 本单 **0 行**。
⇒ **本锚点无机器裁判**：报告里的 -0.862% 是自建校准脚本读既有 `benchmarks/verified/sphere_re200/result.json` 得出的，
不是平台裁判的 ref/got 行。故 L4「基准验证」不成立（且 L4 还要求 L3 的 integration_callers）。

## 三、级别（L0-L5）
- L2 = 成立：真代码 输入(仓根)→计算(证据层级评分)→输出(json/csv/sql/png)，有入口、有 12 件运行工件、10 测试。
- L3 = 不成立：`integration_callers` 机器可核不通过——`grep -rl -F 'tensorlbm.audit' /root/tensor-ecosystem/TensorLBM` **无命中**；
  全仓唯一调用方是它自己的测试 `tests/test_intl_bench_audit.py:166`。根因是分支未合（见上），非实质缺失。
- L4 = 不成立：见第二节（无裁判行）+ 校准控制组作者=评分器作者（自洽非外部）。
- 附录级说明：报告 105 行文档 + 3 图不改变级别；交付的实质是"评分器 + 校准 + 落库"，不是新一轮物理对标数据。

## 四、发现（按严重度）
**F1 BLOCKING（假 verified，已独立复现）**：`grade()` 的裁决探测 `truthy_verdict()`（intl_bench.py:177）把**任意嵌套**
`"pass": true` 当作算例裁决。真实在盘实例：`benchmarks/verified/compressible_shock_family/result.json`
`all_gates_pass: false`、`n_gates_pass: 11/12`、`FFS-02 = 1.458 deg > 1.0 deg`，其 `README.md:47` 明写「**FAIL (honest)**」，
但模块 `compressible_shock` 仍被评为 **verified** 并落入 `tensorlbm_module_maturity`（DB 行已核）。
⇒ 交付声明的 "verified 4" 实为 **3 可支撑 + 1 虚标**。复现：`audit/m15937_repro_false_verified.py`
（合成算例 `all_gates_pass: false` + 一个 gate pass=true → `GRADED LEVEL: verified`, `FALSE_VERIFIED: True`，退出码 1）。
校准基准 10 控制组无一覆盖"聚合 flag=false 但个别 gate=true"，故 10/10 全绿**不覆盖此失效模式**（自证盲区）。

**F2 MEDIUM（证据链）**：分支未合入 main（ECO 亦无），merge-tree 干净 → 交付孤儿化；`integration_callers` 永久不可满足，
且工作树回收后只能靠 `git show 01748f48` 找回（m15904 审计已在同类单记过同一通路缺陷，属复发）。

**F3 MEDIUM（可复现性）**：四上游活体校验根 3/4 落在易失路径 —— `/tmp/walberla`、`/tmp/palabos_probe`、`/tmp/up/lettuce`
（仅 `/root/openlb` 稳定）；仓内无 pin/clone 脚本、无 submodule、无 commit 清单。换机或 /tmp 清理后
`check_citations` 全部 fail → CLI exit 3，51/51 不可重建。另：`UPSTREAM[*]["tier"]` 是手写常量（intl_bench.py:23-90），
"上游最高层级/差距等级"没有任何机器推导，仅引用存在性经机器校验。

**F4 LOW（过度声明）**：报告 ⑥ 把"评分器校准一致率 ref 100%"列为基准对照，但 ref 等级由同一作者在
`benchmarks/intl_bench_calibration/controls.json` 手写 → 自洽性检验，非外部裁判；建议措辞改为"自校准"。

**F5 LOW（措辞）**：51/51「活体校验」实际是 symbol/文件存在性 substring 命中（如 `"class BGKdynamics"`），
不证明上游能力分档正确；建议证据串写"引用可解析（存在性）"。

## 五、未做的检查（不冒充 passed）
- 未逐条人工阅读 51 处上游引用与四参照算法实现细节的对应性（只抽读 OpenLB/waLBerla 引用结构 + 全量机器解析）。
- 未重跑任何物理算例（本单不改求解器；sphere_re200 Cd 取既有工件，-0.862% 未独立重算，只核了文件内数字与 ref 出处）。
- 未验 GPU 8.6 / 69 GLUPS 自记性能（交付侧同样只标注"自记"，口径不可比，双方一致）。
- 未向集群递交作业（全部复核 < 15 s，本地即可）。
- 未核 m15912 报告内每一行数字对 `summary.json` 的逐字节一致（抽核 45/4/24/15/2、51/51、10/10、-0.862%、2.728% 均一致）。

## 六、处置（disposition）
- 级别：**L2**（impl/entry/run_artifacts/tests 齐；L3 因未合入不可核，L4 无平台裁判行）。
- 结论：**approve with notes → request changes（针对产品本体，非重做审计）**。阻塞项 F1：`truthy_verdict` 必须以顶层
  `all_gates_pass`/`verdict` 为准（缺失才降级为 nested 探测），并把 `compressible_shock` 回滚为 partial、
  校准补 c11 控制组（聚合 false + 个别 gate true → 期望 partial）。F2 需平台侧重放 push-merge（broker 缺陷不是本单责任）。
- 独立性的边界：同机、同 Python 3.11.6、同一 git 对象；独立性杠杆=从零沙箱重跑 + 自证 fixture 反例 + 平台台账对账。

## 复跑清单（本单证据）
```bash
git -C <repo> archive 01748f48 | tar -x -C /tmp/a15912
cd /tmp/a15912 && OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m tensorlbm.audit --repo-root . --out /tmp/a15912_out   # exit 0
cd /tmp/a15912 && OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/test_intl_bench_audit.py -q                # 10 passed
cd /tmp/a15912 && OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/intl_bench_calibration/run_benchmark.py         # PASS 100%
python3 audit/m15937_repro_false_verified.py /tmp/a15912/src                                                          # FALSE_VERIFIED True
```