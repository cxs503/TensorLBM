# m18218 交差核验报告 — XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS（对象 m18073）

**核验人**：board-audit（独立交差席）｜**对象**：m18073（TensorLBM，status=succeeded，model=deepseek-v4-flash，tool_calls=80，fail_stage=`budget_exhausted+forced_finish`）
**取证基线**：`/root/tp-agent/work/m18073`（head commit `4903d86c`，`result.head_commit=null`）＋ `tensor_dev_missions.id=18073.result` ＋ `tensor_benchmark_cases` ＋ `tensor_reports`
**裁决**：**L2**（独立功能实现）。**gate 结论 = reopened**（红测试 + 裁判表 0 行 + 未入库 + 无报告）。

---

## 1. 三件套取证（逐项核）

| 项 | 要求 | 实况 | 判定 |
|---|---|---|---|
| run_artifacts | 存在、可由代码再生 | `xval/oracle_minimal_d2q9_fs/artifacts/{xval_results.json(2.0 MB), numbers.md, port_fidelity_a16.json, 6×png}` 均在；本次审计按脚本独立重跑三点，数值**逐位复现**（见 §3） | ✅ |
| benchmark ref vs got | 对照 `tensor_benchmark_cases` | 该锚点 **0 行**（全表 94 行/75 锚）；无机器裁判登记。审计改用任务自带锚 Martin & Moyce 1952（`src/artifacts/martin_moyce_1952_dam_break.json`）+ 作者 pre-declared 15 % 门 | ❌ WITHHELD |
| repro_cmd | 可执行、无隐参 | `python3 xval/oracle_minimal_d2q9_fs/run_xval.py --matrix` 可跑；但**落盘 runs 止于 T=1.5，而脚本默认 `--tmax=window_T[1]=4.0`，实际命令行未写入工件**；`env` 块缺 `numpy` 键（当前 `main()` 会写），`workers=10` ≠ 默认 12 ⇒ 工件非由树内该版本代码原样产出 | ⚠️ 部分 |
| 报告六段 | `tensor_reports` 落表 | **mission_id=18073 报告 0 行**（`report_id=null, report_owed=f`）；`DIFF_OSS_VS_TENSORLBM.md` 引用的 `REPORT_XVAL_ORACLE_MINIMAL_D2Q9_FS.md` **树内不存在**；无 ①物理背景/②方法/③参数表/④ref-vs-got/⑤敏感性/⑥失败模式 六段文 | ❌ 缺 |

## 2. 基准裁判逐项（作者 pre-declared 门 = 15 %，M&M 1952 数字化曲线）

| label | a | τ | got max｜rel err｜ | ref | 容差 | 判 |
|---|---|---|---|---|---|---|
| tensorlbm_koerner_a32_tau1 | 32 | 1.0 | **13.806 %** | M&M 实验曲线 | ≤15 % | ✅ 过（但 >3 % 精度标准，见 §6） |
| tensorlbm_koerner_a16_tau1 | 16 | 1.0 | 22.297 % | 同上 | ≤15 % | ❌ |
| oss_port_a32_tau1（照抄上游） | 32 | 1.0 | 124.377 % | 同上 | ≤15 % | ❌ |
| oss_port_a32_tau1_rho_prev_fixed | 32 | 1.0 | 15.961 % | 同上 | ≤15 % | ❌（差 0.96 pp） |
| arm-vs-arm a=16/32 | — | — | 63.636 / 53.797 %（max） | 作者 15 % 互洽门 | ≤15 % | ❌ 假设被否证 |

## 3. 审计独立重跑（`docs/audit_m18218/repro_check.py`，3 任务并行）

| 审计任务 | 结果 | 与交付件比 |
|---|---|---|
| tensorlbm a=32, tmax=1.5（复现头条点） | Z max err **13.806 %**, pass=True, drift 0.043 %, u_max 0.2442 | **逐位一致** |
| tensorlbm a=32, **tmax=4.0（脚本默认/预声明窗口上界）** | 4526 步，末采样 T=3.752，Z max err **13.806 %**, pass=True, drift 0.055 % | 结论不变 ⇒ 截窗不改变裁决，但**记录违反固定步数停表规则** |
| oss_port a=32, rho_prev_alias=False | Z max err **15.961 %**, drift **19.08 %**, u_max **0.8164966** | **逐位一致**；u_max=√(2/3) 恰为 OSS 速度限幅饱和值 |

## 4. 判级：L2 的实证依据

- **L2（独立功能实现，真代码）成立**：`oss_oracle_port.py` 569 行（v/权重/CT 位标志/碰撞/流式/ABB/质量交换/转换阈值逐条对应上游文件，头部给出零手抄对照表）、`run_xval.py` 458 行（两臂 + 预声明门 + 消融）、`port_fidelity.py` 130 行、测试 95 行；输入→计算→输出闭环，审计重跑数值可复现。
- **L3 不成立**：交付物全为 **untracked**（`git status` 仅 `?? tests/... ?? xval/oracle_minimal_d2q9_fs/`，`result.head_commit=null`，`git_status` 字段自证）；仓内除该测试外**零调用点**（`grep -rl oracle_minimal_d2q9_fs` 仅命中测试自身）；被测 `src/tensorlbm/free_surface_lbm_2d.py` 属既有能力，本单只消费不接入。
- **L4 不成立**：裁判表 0 行；测试 **红**（5 例 1 失败）；敏感性两轴未执行完；无报告六段。
- **L5 不成立**：无新物理能力落地，OSS 侧不构成可用参考（§5）。

## 5. 严重度排序发现（可复现）

| # | 严重度 | 发现 | 证据/复现 |
|---|---|---|---|
| F1 | **blocking** | 全部交付物未入库：0 tracked 改动 | `git -C /root/tp-agent/work/m18073 status --short` → 仅两行 `??`；`result.head_commit=null` |
| F2 | **blocking** | 报告缺失（任务榜硬性产物） | `SELECT count(*) FROM tensor_reports WHERE mission_id=18073` = 0；DIFF 所引 `REPORT_XVAL_...md` 树内不存在 |
| F3 | **blocking** | 测试红，且失败原因被**错报**。真实失败：`assert Z[0]==1.0±0.1` → `1.125`（1/8 格量子 > 容差，a=8 下数学上不可满足），**非** result 所称「front-advance 断言」；整文件跑更早炸 `ModuleNotFoundError: tensorlbm.free_surface_lbm_2d`，因 `test_port_matches_upstream_snapshot` 把 `src` 从 `sys.path` 永久移除（测试顺序污染） | `pytest tests/test_xval_oracle_minimal_d2q9_fs.py -q` → 1 failed/3 passed（-x 前）；单跑该例 → `1.125 == 1.0 ± 0.1` |
| F4 | major（科学） | 互洽假设被否证：两臂 15 % 门全破（22.7/63.6/53.8 %）；OSS 臂照抄即 124 %，修 quirk 后仍 15.961 %，且 `u_max=√(2/3)` 精确饱和、液量漂移 207.8 %/19.1 % ⇒ 「oracle」自身不可用作裁判 | `xval_results.json` + 审计重跑 |
| F5 | major | 「port 与上游逐位等价」只在**20 步**内成立：`port_fidelity_a16.json` 记 `steps_compared=20`、`upstream_first_nonfinite_step=30`（上游 30 步内 NaN）。port 加了零除守卫（唯一语义偏差，DIFF §3 Q7 自认但 headline 未限定范围） | `artifacts/port_fidelity_a16.json` |
| F6 | moderate | 预声明敏感性未执行完：DESIGN 声明 a∈{16,32,48}、τ∈{0.6,1.0,1.4}，实际只有 a∈{16,32}、τ=1.0；`fig_tau_sensitivity.png` 对缺失 τ 静默 skip ⇒ 图题言过其实 | `xval_results.json` 8 runs 清单 |
| F7 | moderate | 预声明 control 空转：`controls=[]`（DESIGN 声明静水柱 + 虚假 ｜u｜_max 控制），`numbers.md` 的 Controls 段为空 | `artifacts/xval_results.json`、`numbers.md` |
| F8 | minor | 外部锚摘要不符：DESIGN 记 M&M sha256 `e6300b9a…`，树内实际 `6f766bfa461dc0d7…`（同一文件，摘要过期/错抄） | `sha256sum src/artifacts/martin_moyce_1952_dam_break.json` |
| F9 | minor | 工件 provenance 不一致：`env` 缺 `numpy`、`workers=10`≠代码默认 12 ⇒ 工件产自树外代码版本（数值仍可复现，故非 blocking） | `artifacts/xval_results.json.env` |

## 6. 误差判定（是否 <3 %）

- 头条量 Z(T) 对 M&M：**13.806 % > 3 %，且 >15 % 邻域余量仅 1.2 pp** ⇒ **不满足 3 % 精度标准**，仅满足任务 `≤15 %` 判据。
- 网格收敛：a=16→32 误差 22.297 %→13.806 %，收敛方向对但只有两点，无 a=48，**不能宣称收敛**。
- 两臂互洽：63.6 %/53.8 %，**远超任何容差**。
（无基准项：裁判表 0 行，故不存在「基准表内误差」可报；此处按外锚 M&M 报。）

## 7. 未做的检查 / 范围限制

- OSS 侧仅以 numba-identity shim 跑 CPU 路径（GPU/numba 真路径未验）；上游 30 步后 NaN，**长时行为无法核对**（port 独占发言权）。
- 未做 a=48 / τ=0.6,1.4 补跑（审计时间盒）；未做静水柱 control 补跑。
- 未核对 `make_figures.py` 插图逐点数据（仅核 runs 表与 `numbers.md` 一致性）。
- 未在 x86/集群复跑（仅 99 本地 CPU，torch 2.7.1 / py3.11.6）。

## 8. 处置建议（供 gate 与返修单）

1. `git add` 交付物并提交（F1）；补 `xval/oracle_minimal_d2q9_fs/REPORT_*.md` 六段并发布到 `tensor_reports`（F2）。
2. 修测试：`Z[0]` 容差改为 ≥1/a 格量子（或改断言初值 `(a+1)/a`）；`test_port_matches_upstream_snapshot` 用 `monkeypatch.syspath_prepend`/局部 `sys.path` 复原，消除顺序依赖（F3）。
3. headline 与 DIFF 统一限定「port 等价范围 = 上游非 NaN 前 20 步，零除守卫为声明偏差」（F5）；补 a=48、τ=0.6/1.4 与静水 control 落盘（F6/F7）；把 `--tmax` 写入工件 `cmdline` 字段（F4 记录面）。
4. 若要冲 L4：向 `tensor_benchmark_cases` 注册该锚点的 `case_name/metric/tolerance` 行，并在报告中显式回答 3 % 标准（当前 13.8 % 不达标）。

*本报告数字全部来自工件 JSON/DB 查询与本次重跑，`docs/audit_m18218/evidence_snapshot/SHA256SUMS.txt` 固化被审代码与工件指纹。*