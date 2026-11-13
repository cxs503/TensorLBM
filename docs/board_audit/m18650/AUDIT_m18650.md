# 交差审计 · m18462 / XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR（TensorLBM）

审计单：m18650（board-audit）· 被审单：m18462（succeeded）· 被审提交：`75d5cfbd`（branch `agent/m18462`）
技能：adversarial-review（独立复算 + 尝试复现 + 分级处置）· 结论级别：**L3**（机器 gate 预期 = 2，原因见 F1）

## 0. 六项必交产物索引

| 必交项 | 位置 |
|---|---|
| 被审工件/提交 | `/root/tp-agent/work/m18462` @ `75d5cfbd`（23 files, +6256） |
| 独立性声明 | 审计方与作者同为平台 AI，**未接触**真实外部 OSS 仓库（无网络）；复现使用被审分支的完整 archive，未修改被审 worktree |
| 复算证据 | `runs/m18650/audit_bundle/verify_audit.json`、`benchmark.json`、`acceptance_audit.json` |
| 图 | `docs/board_audit/m18650/fig_audit_m18650.png`（脚本 `make_fig.py`） |
| 落表 | `verdict_m18650.sql` → `tensor_maturity_verdicts` id=763（level 3） |
| 未跑的检查 | 见 §5 |

## 1. 判决（disposition）

**approve with notes / request changes（合并缺口）**：功能实质成立、数值可复现（逐位一致），但
**交付从未进入主线**，且 `benchmark.json` 有两行 `ref/got` 字段错位。建议：开 successor 单
「merge `agent/m18462` + 修 benchmark_rows 两行」。

## 2. 基准判据（唯一裁判）核查

* `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR'`
  → **0 行**（F3：本锚点没有登记任何机器容差，裁判缺位）。故本次以 owner 15% 线 + 解析锚点人工裁定。
* 自报 `benchmark.json`：10 行，`threshold=0.15`，`all_pass=true`。审计方**逐行重算** `|got−ref|/|ref|`：
  **8 行自洽**（≤1.3949%），**2 行字段错位**（F2，复算得 `nan` 与 `1.0`）。

## 3. 独立复算了什么（不是读报告）

1. **解析锚点从零重推**（不用仓内代码）：`K(t)=u0²/4·e^{−4νt}=2.1303594724155285e−03`，
   `Z=½Δx²u0²e^{−4νt}(sinΔx/Δx)²=1.0258263146155207e−05 @nx=128`，`τ32/128/256=0.515563/0.749007/1.496028`
   → 与交付 `ref` **完全一致**：参考解不再是 m18237 那种「自己比自己」（F4，正向发现）。
2. **全流程重跑**（12 arms + 6 controls + judge + anchor + benchmark，`/dev/shm` 副本，逐阶段计时）：
   10 行 benchmark 的 `ref/got/rel_err/pass` **全部逐位相同**；`selected_theta=0.0016321294477651906`、
   `selected_ke_err=0.005609761249705503` 与交付一致；`arms` 仅 `wall_seconds` 不同；`acceptance` 8 行逐位相同、`passed=true`。
3. **测试**：`pytest tests/test_les_graph_unroll_selection.py tests/test_r2_selection_integration.py`
   → **13 passed in 18.88 s**（自报 13 passed / 16.58 s，吻合）。
4. **收敛性**：256² → `3.19e−4`（0.032%），128² → 0.237%，32² → 1.395%，趋势单调，说明 fine grid 是 128² 的合理代表。

## 4. 分级（L0–L5）

| 级 | 判据 | 结论 |
|---|---|---|
| L1 | 数学/单元验证 | ✅ 13 tests + `tgv_analytic_*` 闭式可独立复算 |
| L2 | 独立功能实现 | ✅ `selection.py`(338 行)/`tgv_anchor.py`/`solver2d.py`/`reference2d.py` 真代码，入口 3 个，工件齐全 |
| L3 | 系统集成 | ⚠️ 分支内成立（`tensorlbm` 顶层 lazy export + `verification/tgv_r2_selection.py` 消费方），**但未合并主线**（F1）→ 机器 grep 复核必失 |
| L4 | 基准验证 | 数值上成立（8 行 <15%，可复现）——**但因 F2（两行错位）与 F1（未入主线）不予授予** |
| L5 | 科研工程成果 | ❌ 覆盖仅 2 closures × 1 flow × 1 Re × 1 grid pair，无集群算例 |

**定级 L3**：功能实质 = 真实现 + 真工件 + 可逐位复现的基准；扣级原因是交付/证据卫生，不是造假。

## 5. 发现（按严重度）

* **F1（blocking，交付）**：`75d5cfbd` 只在 `agent/m18462`。主线 worktree `m18650` HEAD `10d62137`(#2166) 与
  ECO 检出 `56aafaaf`(#2133，已陈旧) 均无 `selection.py`/`tgv_anchor.py` → 平台上没有任何消费者能用它；
  机器 `maturity_gate.judge` 因此得 **gate_level=2**（`integration_callers: grep 无引用`）。**先合并再消账**。
* **F2（medium，证据结构）**：`scripts/xval_r2_selection_bptt_tgv.py:258-273` 两行把误差写进了 `got`/`ref`：
  `bptt-gradient-vs-fd`（ref=0.0，got=相对差 7.35e−4）与 `holdout-ke-selected-vs-filtered-ref`（ref=误差 5.61e−3，got=0.0）。
  后果：任何「从 ref/got 重算 rel_err」的 gate 会读到 `nan` / `1.0`；`has_numbers` 类弱校验看不出来。
  **未发现数值美化**：真值在自洽行（0.5611%）与 `results.json.verdict.P1`（7.35e−4）里。
* **F3（medium，机制）**：`tensor_benchmark_cases` 对本锚点 0 行 → 「唯一裁判」缺位，交接审计只能靠解析解 + 15% 线。
* **F4（正向）**：参考解已外锚定到精确 TGV（128² 0.237% / 256² 0.032%），修掉了 m18237 的自指缺陷；这是本轮最实的一步。
* **F5（正向）**：可复现性达标（逐位），无随机种子、float64、`repro_cmd` 有效（但需 `TMPDIR` 指到非满盘）。
* **F6（low，指标语义）**：`verdict.P2.*.rel` 出现 1.63e9（分母为 `theta_sweep=0` 的相对差），作为「一致性%」无意义；
  报告已改用 `loss_gap_rel`，但原始字段应显式标注 bound-hit。
* **F7（medium，科学结论的诚实性）**：选中点是 `θ→0`（`ke_err` 0.5611%）与「无模型」控制（0.5609%）不可区分（regret 5.3e−5 pp），
  即**子网格常数在本锚点上不可辨识**；报告如实写出并给出 P2θ/P5 两个 ❌，**未粉饰**——这是本轮的科学价值（负结果入账）。
* **F8（low，环境）**：审计开始时 `/` 仅剩 0 字节，`torch` import 直接崩（`No usable temporary directory`）；
  用 `TMPDIR=/dev/shm/...` 才跑通。属生态摩擦，记录备查。

## 6. 主张核对（claims）

| 自报主张 | 裁决 |
|---|---|
| 「13 passed in 16.58 s」 | 成立（审计方 13 passed / 18.88 s） |
| 「benchmark all_pass / 阈值 0.15 / 均 <15%」 | 数值成立（最大 1.3949%），但 2/10 行字段错位（F2） |
| 「参考解不再自指，锚定精确 TGV」 | 成立（审计方独立重推一致） |
| 「L4 可机器复核交付」 | **不成立**：工件/复现成立，但主线未合并（F1）+ 裁判缺位（F3）+ 两行错位（F2）→ L3 |
| 「θ 不可辨识 / 选无模型」 | 成立且有证据（0.5611% vs 0.5609%） |

## 7. 未跑的检查（unknown / not_run）

无集群算例；无第二个 Re / 第二种流动 / 3-D 或带壁路径；无独立文献值对照（仅解析解）；
未验证 `agent/m18462` 是否推送 GitHub（本地观察：分支仅存在于本机）；`window` 阶段的训练窗口敏感性未进 judge（作者自报）。

## 8. 复现命令（审计方实际所用）

```
git -C /root/tp-agent/work/m18462 archive 75d5cfbd | tar -x -C /dev/shm/m18650/tree
cd /dev/shm/m18650/tree && export TMPDIR=/dev/shm/m18650 PYTHONPATH=$PWD/src TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 scripts/xval_r2_selection_bptt_tgv.py ref --force      # 11.1 s
python3 scripts/xval_r2_selection_bptt_tgv.py arm <closure> <objective> <start>   # x12, ~35 s each
python3 scripts/xval_r2_selection_bptt_tgv.py controls <closure> <objective>      # x6
python3 scripts/xval_r2_selection_bptt_tgv.py judge            # 95 s
python3 scripts/xval_r2_selection_bptt_tgv.py anchor           # 42 s
python3 scripts/xval_r2_selection_bptt_tgv.py benchmark
python3 -m tensorlbm.verification.tgv_r2_selection --convergence --out .../acceptance_audit.json
python3 -m pytest tests/test_les_graph_unroll_selection.py tests/test_r2_selection_integration.py -q
```

## 9. 处置与返修建议

1. `merge agent/m18462` → F1 消除，gate 应回到 ≥3。
2. 修 `benchmark_rows()`：`bptt-gradient-vs-fd` 改 `ref=FD 梯度值, got=autograd 梯度值`；删掉或改写
   `holdout-ke-selected-vs-filtered-ref`（与自洽行重复），并把 `has_numbers` 之外的自洽校验加进 acceptance 套件。
3. 把本锚点登记进 `tensor_benchmark_cases`（`metric=ke_rel_err@t=200`, `tolerance=0.15`），让裁判归位。