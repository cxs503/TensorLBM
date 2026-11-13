# AUDIT m17866 — XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC (TensorLBM)

Board-audit 交差核验单 m18146 · verdict **L2** · verdict row `tensor_maturity_verdicts#712`
(auditor: board-audit, gate_state pending at write time).

## 0. 结论速览

| 三件套 | 交付状态 | 审计核实 |
|---|---|---|
| impl（真代码） | 2 个文件，**untracked**（`git status = ??`，`head_commit=null`） | 真代码，325 行 D2Q9 算子，无占位体 |
| run_artifacts | **缺失**（worktree 无 `artifacts/`、无 `figures/`、无任何 JSON） | 审计独立重跑产出工件（下文 §3） |
| ref vs got | 解析解 `u=gx/(2ν)·y(H-y)`；声明 1.0% | 复现 1.000%（20k）/ 1.000%（40k），5.60%（TL roll），质量 +0.0135% |
| repro_cmd | 交付脚本存在但 **TM=100000**，与声明数字所用参数（20k，出自 `.scratch/smoke_xval.py`）不一致 | 可执行，单核 ~2.13 ms/step ≈ 22 min |
| 基准判据 | `tensor_benchmark_cases` 该 anchor **0 行** | 裁判缺席（兄弟 anchor `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN` 有 1 行，pending） |
| 报告 | `tensor_reports` mission_id=17866 **0 行** | 六段报告从未发布 |

**判级：L2**（独立功能实现：真代码 + 输入→计算→输出 + 解析解验证）。
不授 L3（无任何生产侧调用方）；不授 L4（无交付工件 / 无测试 / 无 commit / 无注册判据行）。

## 1. 案例介绍（物理背景与参数）

对象：`haimingz/vivsim` 的 `jax-pure-function-aa-pattern-static-slice` 模式（AA 布局 + 静态切片
`concatenate` 流式 + 纯函数算子 + `jit/donate/scan` 融合）在 TensorLBM PyTorch 后端的对等实现，用
二维强迫 Poiseuille 槽道做解析解交叉验证。

| 参数 | 值 |
|---|---|
| 网格 | ny=nx=10（上下固壁，x 周期） |
| 粘性 / 松弛 | ν=0.2，ω=1/(3ν+0.5)=0.90909 |
| 体力 | gx=0.001（EDM=Guo 项） |
| 边界 | 力修正 NEBB（top/bottom），壁面速度 −g/(2ρ) |
| 精度 | float64，CPU |
| 解析解 | u(y)=gx/(2ν)·y(H−y)，H=ny−1=9 |
| 步数 | 交付脚本 TM=100000；声明数字出自 20000 步 |

## 2. 怎么做的（审计方法）

1. 取证：读 m17866 `result`（`status=succeeded` 但 headline 为 `【honest-FAIL·第1轮】`）、
   `changed_files`、`git status`；确认两个文件 untracked、无 commit。
2. 快照：把两文件按 sha256 复制进本审计树 `refs/`（`xval_vivsim_static_slice.py`
   `be31fcca…`、`run_xval.py` `e4a26366…`），基线 commit `804636b0`。
3. 独立重跑（不改被测代码；为避开整包 import，`rerun/TensorLBM/src/tensorlbm/` 用空
   `__init__.py` + 原样 `d2q9.py` 作 import 垫片，被测文件字节相同）：
   种子随机场算子等价性 → 20k 步三臂 → 40k 步收敛点 → 质量账 → 对抗性负对照。
4. 裁判核对：查 `tensor_benchmark_cases`（该 anchor 0 行）与 `tensor_reports`（17866 0 行）。
5. 判级按「功能实质」而非报告篇幅；机器门 `maturity_gate` 复核 evidence 指针。

## 3. 计算结果（审计实测，数字全部来自 `artifacts/audit_summary.json`）

| 检查 | 结果 |
|---|---|
| `static_slice` vs `roll_std` 算子等价（randn 种子 0） | max\|Δ\| = **0.0** |
| `static_slice` 20k 步 vs 解析解 | **1.000%**（声明 1.0% ✓） |
| `static_slice` 40k 步 | **1.000%**（20k→40k Δ=1.4e-6 ⇒ 时间已收敛） |
| `roll_std` 20k 步 | 1.000%（同一数学） |
| TensorLBM `d2q9.stream`（shifts=−c）20k 步 | **5.60%**（声明 5.6% ✓） |
| 质量账 100 → | 100.013548（**+0.0135%**，声明 ✓） |
| 负对照：去掉 −g/(2ρ) 壁面速度修正 | **0.0135%**（比交付版好 ~74 倍） |

声明数字全部复现；问题不在数字真伪，而在交付完整性与判据缺席。

## 4. 图

- `artifacts/fig_profile.png`：解析解 vs static-slice vs TensorLBM roll 剖面。
- `artifacts/fig_error_arms.png`：四臂误差 + 负对照 + 3%/15% 门限（对数轴）。

（图由 `make_artifacts.py` 从 JSON 程序化生成，零手抄。）

## 5. 注意点 / 踩坑与局限

- **F1 交付无工件**：`head_commit=null` + 两文件 untracked ⇒ 组织内其他地方拿不到该能力。
- **F2 复现参数不一致**：声明数字来自 `.scratch/smoke_xval.py`（20k 步），交付
  `run_xval.py` 用 TM=100000；照文档跑不会得到报告里的数字。
- **F3 判据缺席**：该 anchor 在 `tensor_benchmark_cases` 无行，兄弟 anchor 有行而本单没有登记。
- **F4 负对照缺位且反噬**：审计自造负对照（去掉壁面速度修正）把误差从 1.000% 降到 0.0135%
  ⇒ 1.0% 是**边界条件变体**的产物，不是离散误差地板；同时 15% owner 门无法区分「约定错误的
  TL roll（5.6%）」与「正确的 static-slice（1.0%）」，判据对这类约定缺陷不敏感。
- **F5 上游忠实度未核**：vivsim 源码未 vendored、审计无网络，移植忠实度仅凭模块 docstring。
- 口径提示：`max_rel_err` 实为「以解析解峰值为分母的最大绝对误差」，非逐点相对误差。
- 幂律：`d2q9.stream` 的 `shifts=(-cy,-cx)` 被逐行核对为与交付 `stream_roll` 完全一致（不是移植造假）；
  差异属 pull/push 约定，报告「TensorLBM 现有 stream 5.6%」应读作 harness 约定下的结论。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 量 | 基准 | 审计 got | 容差 | 判定 |
|---|---|---|---|---|
| static-slice 剖面 u_x | 解析 Poiseuille | 1.000%（20k/40k 一致） | <3% 自设 / <15% owner | **<3% ✓**（但见 F4：BC 变体可到 0.0135%） |
| roll_std 剖面 u_x | 同上 | 1.000% | 同上 | <3% ✓ |
| TensorLBM roll 剖面 u_x | 同上 | 5.60% | 同上 | <3% ✗ / <15% ✓（约定差异，非数值发散） |
| 质量账 | 100 | +0.0135% | — | 合格 |
| 注册基准行 | `tensor_benchmark_cases` | **0 行** | — | 无基准可裁 ⇒ 不满足 L4 |

## 7. Disposition

`request changes`：
(1) 提交模块与驱动并补测试（对齐兄弟 anchor 的 `tests/test_aa_fused_xval.py`）；
(2) 让 `run_xval.py` 就是产出声明数字的脚本，或按 TM=100000 重跑并交付 JSON；
(3) 在本 anchor 下登记 `tensor_benchmark_cases` 判据行，并加一条必须超门的负对照；
(4) 修正或明确声明壁面速度修正口径，报告中把 1.0% 归因到该 BC 变体。

审计独立性限制：仅本地 CPU、float64、ny=nx=10 单几何；100k 全量脚本与计时表未跑（集群 cj1814 排队 311）；
上游 vivsim 未读。未跑项一律记 `not_run`，不视为通过。