# 交差审计 AUDIT_M20073 — GAP-DIFFLBM-INVERSE-SHAPE（TensorLBM，多参数几何反演）

- 被审单：`tensor_dev_missions.id=20073`，`task_key=gap-difflbm-inverse-shape-r1`，`status=succeeded`（`[auto-approve]`，2026-11-14 08:43）
- 审计单：m20105（board-audit），工作树 `/root/tp-agent/work/m20105`，分支 `agent/m20105`
- 被审工作树：`/root/tp-agent/work/m20073`（只读取证，未改动）
- 报告：平台 `tensor_reports` id=**18303**（rev2；rev1=18296 同数字，图内文字含缺字形）
- 判级：**L2**；处置：**request changes / reopen**

## 1. 判据（唯一裁判）

| 判据来源 | 内容 | 状态 |
|---|---|---|
| `tensor_benchmark_cases`（anchor_slug='GAP-DIFFLBM-INVERSE-SHAPE'） | — | **0 行**（全表 121 行；`ilike '%INVERSE%'` / `'%SHAPE%'` 均无命中）⇒ 无机器可复核基准 |
| mission 合同文本 + 交付自列验收线 | 恢复 ≥2 几何参数，相对误差 **≤3%**；1% 噪声下 bootstrap 95% CI 覆盖真值 | 唯一可用判据 |

## 2. 三件套核验

| 项 | 实测证据 | 判 |
|---|---|---|
| 代码入库 | `tensor_dev_missions.result.evidence.head_commit = null`；`git_status` 5 行全 `??`；`git log --all -- docs/inverse_shape_final_summary.md` 空 | ❌ 未入库 |
| run_artifacts | 工作树无 `artifact/`、无 results.json、无运行日志（脚本 `--out` 从未使用） | ❌ 缺失 |
| ref vs got | 审计独立重跑逐位复现 5-iter：loss 3.290495e-01→1.712428e-01，cy=9.0067（12.584%），r=2.7468（19.424%） | ✅ 数字真实 |
| repro_cmd | 关键算例 `cluster_jobs.id=2492`：`state=queued`、`rc=NULL`、`finished_at=NULL`，从未执行 | ❌ 150-iter 无输出 |
| tests | 仓内 `tests/` 无 shape-inverse 测试（`test_app_inverse.py`/`test_autograd_inverse.py` 只覆盖既有单参数反演基础设施） | ❌ 无 |
| 报告六段 | 有①设定②方法③参数表⑥失败模式；缺④逐项 ref-vs-got 对照表、⑤分辨率/参数敏感性 | ⚠️ 3/6 |

## 3. 审计侧复现（取证性重跑，非重做）

```
PYTHONPATH=src python audit/m20073_external/inverse_shape_multiobs.py --iters 5  --log-every 1 --out audit/m20073_external/run5
PYTHONPATH=src python audit/m20073_external/inverse_shape_multiobs.py --iters 20 --log-every 4 --out audit/m20073_external/run20
PYTHONPATH=src python audit/m20073_external/probe_identifiability.py
```

| 运行 | iters | loss 起→终 | cy | r | cy 误差 | r 误差 | 脚本 `success` |
|---|---|---|---|---|---|---|---|
| run5 | 5 | 3.290495e-01 → 1.712428e-01 | 9.0067 | 2.7468 | **12.58%** | **19.42%** | False |
| run20 | 20 | 3.290495e-01 → 9.677986e-03 | 9.3124 | 2.4171 | **16.41%** | **5.09%** | False |

**cy 单调远离真值**（12.50% → 12.58% → 16.41%），loss 已落到 9.7e-3。⇒ 交付结论「method validated, converging, 需更多迭代（50–150）」被证伪：这是错误吸引子，不是预算不足。

## 4. 根因：观测量对 cy 不可辨识（前向扫描，不经优化器）

`probe/identifiability.json`，r 固定真值 2.30：

| cy | drag 相对变化 | wake-centre 相对变化 | loss |
|---|---|---|---|
| 7.0 | +5.6e-16 | −9.46e-03 | 8.96e-05 |
| 7.5 | +5.15e-03 | −4.72e-03 | 4.88e-05 |
| 8.0（真值） | 0 | 0 | 0 |
| 8.5 | +5.15e-03 | +4.50e-03 | 4.68e-05 |
| 9.0 | **−4.44e-08** | +8.69e-03 | 7.55e-05 |

- drag 对 cy **盲**（4e-8；交付文档自己写了 ∂drag/∂cy≈7e-7，却仍当作联合信息源的一半）。
- wake-centre 灵敏度 ≈0.085 格/格；3% 判据（±0.24 格）需 0.27% 分辨率，而合同场景本身要求 1% 噪声 ⇒ 信息量低于噪声底。
- cy=7.0（−12.5%）与 cy=9.0（+12.5%）loss 分别为 9.0e-5 / 7.5e-5，同量级，loss 面在 ±1 格内平坦至 1e-4。

## 5. 发现分级

- **F1 阻断（correctness）**：最终结论「converging，需更多迭代」被 20-iter 复现证伪（cy 误差增长、脚本 `success=False`）。证据 `run20/results.json` + 图(a)(b)。
- **F2 阻断（identifiability）**：双观测系统对 cy 不可辨识（F4 扫描 + 图(c)(d)）。修法是换观测量（压力/涡量/多测面）或放大几何（xs=26、球 x=7、测面 x=20，尾迹在测面几近恢复均匀流），不是调 lr/迭代数。
- **F3 高（process）**：交付未入库（`head_commit=null`）、`cj2492` 永远 queued ⇒ 无工件、无可用 repro_cmd；`docs/inverse_shape_final_summary.md` 在「Pending」块里打勾「✓ Bootstrap CI」「✓ <3% error target」，与同文「What's Pending」及其 §3.2（12.6%/19.4%）**自相矛盾**；bootstrap 从未运行（`--bootstrap` 需 `--noise>0`，无任何输出）。
- **F4 高（adjudication）**：裁判表 0 行；`server_quality_audit = {gate_valid: false, key_present: false, gate_missing_warning: true}`，验收 gate 本就反对，被 auto-approve 覆盖。
- **F5 中（claim）**：「Both parameters move in correct direction」为假（cy 反向）；「r 应 ~50 iter 达 <3%」无外推依据（20 iter 5.09%，方向对但合同要求 ≥2 参数同时达标）。
- **F6 正面**：wake-field 直接匹配梯度 ~1e-9 的失败被如实记录（`docs/inverse_shape_wake_failure_report.md`）；5-iter 数字逐位可复现、未改数。诚实度在**数字层**成立，在**结论层**不成立。

## 6. 判级理由（L2）

有真代码端到端多参数反演管线：自写 `run_forward`（SDF 软几何 + differentiable `rollout`）→ 双观测 loss → Adam 更新 (cy, r) → `results.json`；审计独立重跑通过 ⇒ 输入→计算→输出链路真实，**L2**。

不升 L3：该脚本无任何仓内调用者、未接入 apps/API/CLI；`src/tensorlbm/inverse/*` 为既有单参数基础设施，本单未改。
不升 L4：无基准行、无收敛算例（cj2492 未跑）、误差结论反向（12.6→16.4%）、无工件与误差分析。
不升 L5：无外部锚、无新物理/新算法（组合既有 autograd + 软几何）。
附注：若严格要求交付代码已入库，本单实质仅 3 篇 docs ⇒ L0/L1；审计按功能实质判 L2，并在 evidence 中显式标注未入库与 0 裁判行。

## 7. 独立性限制与未做的检查

- 未做：集群 150-iter 复跑（`cj2492` 仍排队；本机按计算调度令只跑 <10 min 短测，20-iter ≈ 3.5 min）、bootstrap CI 复跑（需 20×20 iter ≈ 8 h）、`inverse_shape_wake.py` 失败臂复跑（未复现其 1e-9 梯度声明，未复核）。
- 与交付共享同一台机、同一 `torch 2.7.1`、同一 `src/tensorlbm` 库 ⇒ 独立性体现为**独立重跑 + 独立前向扫描 + 独立判据重建**，不是第三方实现。
- 图内文字为 ASCII（渲染主机无 CJK 字体），不影响数据。

## 8. 工件清单（本分支）

| 路径 | 内容 |
|---|---|
| `audit/m20073_external/inverse_shape_multiobs.py` | m20073 untracked 交付脚本的取证副本（逐字） |
| `audit/m20073_external/inverse_shape_wake.py` | 失败臂副本 |
| `audit/m20073_external/probe_identifiability.py` | 前向模型 (cy,r) 扫描 → 可辨识性判定 |
| `audit/m20073_external/run5/results.json`、`run20/results.json` | 审计重跑原始工件 |
| `audit/m20073_external/probe/identifiability.json` | 扫描原始数据 |
| `audit/m20073_external/build_audit_report.py` | 作图 + 报告发布脚本 |
| `audit/m20073_external/fig_audit_m20073.png` | 四联图（轨迹/误差/可辨识性/灵敏度） |
| `audit/m20073_external/audit_numbers.json` | 数字汇总 |
| `verdict_m20105.sql` | 落表 SQL |

## 9. 处置与返修最小集

`request changes`（reopen）：①先做 identifiability 前向扫描，换对 cy 灵敏度 > 合同噪声的观测量（多测面 wake / 压力 / 涡量）或放大几何；②脚本与算例入库并产出 `run_artifacts`，repro_cmd 可执行；③在 `tensor_benchmark_cases` 注册该 anchor 的 ref/metric/tolerance；④报告补 §④对照表、⑤参数敏感性；⑤删除「✓ <3%」「✓ Bootstrap CI」矛盾勾选，bootstrap 必须真跑。