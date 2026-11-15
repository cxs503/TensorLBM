# CASE-BACKSTEP-FVM-RE100（m22231）：判分门溯源修复 —— 收敛点交付**未完成**（honest FAIL）

**锚点** `feature_ledger.CASE-BACKSTEP-FVM-RE100`（算例池·后台阶回流区长度 Re=100，验收 `<10%`，ref Armaly 1983）；**仓** TensorLBM；**分支** `agent/m22231`。

## ① 案例
后台阶层流分离—再附，观测量 = 下壁面主再附长度 `X1/s`（`τ_w(x)` 零点）。参考：Armaly et al. 1983 JFM 127:473（仓内二级锚 `3.05`，原文 PDF 在本沙箱 CLOSED）；交叉锚 Erturk 2008 二维数值 `2.878`（band 2.792–2.965）。`Re = ū·D_h/ν`，`D_h = 2h_in`。

## ② 怎么做的
1. 复核判分器与机器裁判是否同一把尺：读 `tensor_benchmark_cases`（`anchor_slug='CASE-BACKSTEP-FVM-RE100'`）两行：
| `backward_step_armaly (Re=100, 2D numerical cross-check)` | **5%** |
| `backward_step_armaly (Re=100, lower-wall primary reattachment X1/s)` | **10%** |
2. 发现 `verdict.py` 把容差写死 `TOLERANCE_PCT = 15.0`，**从不读** `refs.json::ledger_tolerance_pct = 10`，也不判 5 % 交叉门 →
   偏 12 % 的点会被判分器判 PASS、被裁判判 FAIL。**修复**：新增 `tolerances(refs)` 让主门/交叉门都从 `refs.json` 读（含 `cross_checks[].re/tolerance_pct` 新字段），判分条件由 5 条增至 6 条（加 `cross_check`），输出加 `tolerance_source / cross_check_anchors`。
3. 新增 `tests/test_case_pool_backward_step_armaly_gates.py`（5 项，纯合成产物）把「12 % 点必须 FAIL」「交叉锚独立判」钉死。
4. 顺带清掉审计 F6：`src/tensorlbm/bfs_armaly.py` 尾部 4 次 `macroscopic(f)` 合并为 1 次（A/B `--smoke` 逐位相同）。

## ③ 计算结果
| 项 | 值 |
|---|---|
| 主判分门（代码 / 裁判） | 10% / 10% → 一致（本单后）|
| 交叉判分门 | 5%（Erturk 2.878）|
| 新增回归测试 | 5 passed（0.34 s）|
| 行为等价性 A/B | 8 个诊断量逐位相同（`x1_over_s`、`mass_flux_*`、`rho_min/max`、`max_speed_final`、`u1_min_over_u_bar`）|
| **收敛基准点 X1/s** | **WITHHELD**（见⑤）|

## ④ 结果图
![gate](fig1_gate_provenance.png)
门限溯源：绿=注册裁判 ±10%、红=旧硬编码 ±15%；紫点 = 合成点（+12 % 在旧门下 PASS、裁判 FAIL；+4 % 双门 PASS）。

## ⑤ 注意点 / 未完成（honest FAIL）
本单两次集群作业（`cj3965`@n315、`cj4007`@n330，`--preset h1_re100 --steps 120000`）**rc=0 成功**，但**产物都取不回来**：
`fetch` 的 tar 包只含 git 已跟踪文件（15402 条目里没有本次新写的 `artifacts/…/result.json` 或 `evidence/m22231_h1_re100/*`，而仓内已提交的 `evidence/case_backward_step_armaly_cli/**` 在包里）。
→ **staged 工作树上新产生的未提交文件不会被 fetch 回来**，这条路在本单形态下走不通。因此：
- `X1/s` 本单 **WITHHELD**，账本 `<10 %` 判据**未被满足**，锚点**不应转 done**；
- `artifacts/case_backward_step_armaly/` 仍无 `*/result.json`，`tests/test_case_pool_backward_step_armaly.py` 两条基准断言**仍然 skip**（如实登记，不改写为通过）。

## ⑥ 误差判定（是否 <3 %）
| 关键量 | 本算 | 基准 | 误差 | <3 %? |
|---|---|---|---|---|
| `X1/s` @Re=100 | WITHHELD（无进仓收敛点）| Armaly 3.05 | n/a | **无法回答**（既非「是」也非「否」）|
| 判分门一致性 | 10% / 5%（读 refs.json）| 裁判 10% / 5% | 0 % | **是** |

**判定**：本单是**判分器缺陷修复 + 溯源测试**，不是精度达标单；精度部分诚实记 WITHHELD。
