# 交差审计 m18112 — XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO

* 审计单：m18148（board-audit，交差核验非重做）
* 被审对象：m18112 / TensorLBM / 分支 `agent/m18112` / commit `2a7c492e`（工作树 `/root/tp-agent/work/m18112`）
* OSS 对标对象：`haimingz/vivsim` @ `fb84b12f02e18f3b9dfdf738d6a1a318d81539f9` `vivsim/multidevice.py::stream_cross_devices`（MIT）
* **裁定：L4（基准验证）· committed = confirmed** ｜ verdict_by = board-audit ｜ tone = approve with notes

---

## 0. 结论先行（级别 = 功能实质，不是报告页数）

| 级别 | 判据 | 本次核验 | 结论 |
|---|---|---|---|
| L0 | 只有文档 | — | — |
| L1 | 数学/单元验证 | 27 条测试实跑 5.25 s 全绿；断言是**不变量**（`out[dst]=in[src]`、fail-closed、逐位等价），非"执行一遍" | 过 |
| L2 | 独立功能（真代码 输入→计算→输出） | `ppermute_halo.py` 530 行 + `sharded_executor.py` 203 行，无 placeholder（`grep NotImplementedError\|TODO` = 0 命中）；入口 `tools/xval_ppermute_halo/run_xval.py` 615 行实跑产出工件 | 过 |
| L3 | 系统集成 | `ShardedPpermuteRunner.step()` 每步调仓内真求解器核心 `LBMStepExecutor.step()`（`sharded_executor.py:167`）→ 见 §3 注 1（**无反向生产调用方**） | 过（带注） |
| L4 | 基准验证（完整算例+误差分析+可复现） | ref vs got 数值逐项复算 **≤ 1.875%** < 15% 门；`repro_cmd` 由审计者独立重跑（63.2 s），除墙钟外**全部数值逐位复现**；变异对照证明判据非空洞 | **过** |
| L5 | 科研工程成果 | 无新物理/新方法提出（是移植+交叉验证+上游缺陷复核） | — 不主张 |

**不开 L5 的理由（显式）**：本单价值在工程移植与"上游缺陷被独立复核"，未产生新科学结论；`lax.ppermute` 真通路与多卡计时均为 WITHHELD（环境无 jax/mpi4py），性能主张不成立。按"级别=功能实质"判 L4。

---

## 1. 取证清单（三件套逐项机器核）

| 项 | 指针 | 核验方式 | 结果 |
|---|---|---|---|
| impl_files | `src/tensorlbm/parallel/{__init__,ppermute_halo,sharded_executor}.py` | 实读 + placeholder 扫描 | 真实现（530/203 行，`PpermutePeriodicHalo`/`OssBranchFinding`/`ShardedPpermuteRunner` 等可 import） |
| entry | `tools/xval_ppermute_halo/run_xval.py`（615 行，`--out` 参数化） | 实跑 | 可执行，产出 6 件工件 |
| run_artifacts | `artifacts/xval_ppermute_periodic_halo/{RESULTS.md,summary.json,kpi.csv,fig1..fig3}.png` | `ls` + 内容抽样 | 全部存在非空（fig 39 KB / 73 KB / 31 KB） |
| tests | `tests/test_parallel_ppermute_halo.py`（14 个 def，参数化后 27 条） | 审计者实跑 | `27 passed in 5.25s` |
| integration_callers | `ShardedPpermuteRunner` → `LBMStepExecutor`（仓内 53 求解器共用执行器） | 读码 + grep | 前向集成成立；**反向（有求解器通过 sharded 路径跑）不成立** |
| benchmark | A1 逐位等价 / A2 OSS 分支核账 / A3 解析 e^{-νk²t} / A4 通信计数 | 独立复算见 §2 | 数值真实、可复现 |
| repro_cmd | `PYTHONPATH=src python tools/xval_ppermute_halo/run_xval.py --out artifacts/xval_ppermute_periodic_halo` | 审计者改输出到 `/tmp/audit18112_out` 重跑 | 63.2 s 完成；`summary.json` 除墙钟三项外**逐键相等** |
| external_anchor | OSS 源码 @ pin `fb84b12`（审计者 `git fetch` 直取上游）、NS 精确解 | 原文比对 | 见 §2-C：**上游 x 分支镜像确实存在且 y/x 互不对称** |

**基准裁判（tensor_benchmark_cases）**：`SELECT ... WHERE anchor_slug='XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO'` → **0 rows**（全表 91 行内也不存在该 slug；`tensor_xval_cases`、`campaign_plan_nodes` 同样无该锚）。⇒ 本锚**没有预登记机器判据**，审计只能以任务 goal 内声明的 owner 精度标准（<15%）+ 独立复算为裁判。此为**生态缺口**（见 §4 F-1），不构成被审方造假，但意味着该锚的容差判定目前无法被 gate 自动复核。

---

## 2. 独立复算（不信任被审工具的数值路径）

脚本：`docs/board_audit/m18148/independent_check.py`（结果 `independent_check.json`）
图：`docs/board_audit/m18148/fig_m18148_audit.png`

### A. 解析臂 A3 —— 审计者自写 numpy float64 D3Q19-BGK 复算

| L（步） | 报告声称 \|err\| | 审计独立复算 | 差 | 15% 门 |
|---|---|---|---|---|
| 12（100） | 4.197% | **4.198%** | 0.001 pp | ✔ |
| 16（100） | 1.875% | **1.876%** | <0.001 pp | ✔ |
| 24（60） | 0.626% | **0.626%** | <0.001 pp | ✔ |

终态幅值：审计复算 `5.86855e-03` vs 工件 `5.868591e-03`（相对差 7e-6）——两条完全独立的代码路径（numpy float64 / torch float32）给出同一数字 ⇒ **A3 不是手抄、不是拟合，是真实 LBM 离散截断误差**。

> 审计过程中审计者自身先踩过一个坑：把宏观速度 `u` 只更新 y 分量、其余两个分量保持初值 → 平衡态每步被"重新驱动"，衰减几乎消失（err 148%/67%/15%）。改为每步重算三分量后与工件吻合。这一反例恰好说明 A3 对实现细节敏感、**不容易被"随便跑跑"凑出 1.875%**。

### B. 判据灵敏度（变异对照，falsifiability）

用被审代码做"换错邻居"变异（`device_pairs ⇄ device_pairs_reverse` 互换）后重跑 A1 同款对比：

| 变体 | max\|Δf\| sharded vs 单域 |
|---|---|
| 正确置换表 | `0.0`（逐位） |
| 互换置换表（换错邻居） | **`9.47e-04`** |

⇒ A1 的"逐位一致"判据**对目标失效模式敏感**，不是"两边都跑同一函数所以恒等"的空洞断言。这一条是本审计认为 A1 可信的关键（被审方自己在 2D 臂 `test_oss_x_branch_as_written_diverges_and_swap_fixes_it` 也给了同类阳性对照，但**执行器级（A1）此前没有负对照**，本次由审计补齐）。

### C. OSS 上游缺陷的独立复核（不靠被审方的转写）

审计者直接从上游 `git fetch` pin `fb84b12f...` 取原文：

```python
if dir == 'y':
    f = f.at[UP_DIRS, :, 0].set(jax.lax.ppermute(f[UP_DIRS, :, 0], device_axis, device_pairs))
    f = f.at[DOWN_DIRS, :, -1].set(jax.lax.ppermute(f[DOWN_DIRS, :, -1], device_axis, device_pairs_reverse))
elif dir == 'x':
    f = f.at[RIGHT_DIRS, 0].set(jax.lax.ppermute(f[RIGHT_DIRS, 0], device_axis, device_pairs_reverse))
    f = f.at[LEFT_DIRS, -1].set(jax.lax.ppermute(f[LEFT_DIRS, -1], device_axis, device_pairs))
```

推导（`lax.ppermute` 语义 out[dst]=in[src]，`device_pairs=[(i,(i+1)%n)]`）：
正 +x 方向在 plane 0 需要低侧邻居的**高平面** → 输出 rank k 取自 k−1 → 应 `device_pairs`；
负 x 方向在 plane −1 需要高侧邻居的**低平面** → 取自 k+1 → 应 `device_pairs_reverse`。
上游 x 分支两者**恰好互换**，与 y 分支互为镜像 ⇒ 同一 f 布局下 y/x 不可能同时正确，**x 分支为镜像缺陷**成立；N=2 时 `pairs ≡ pairs_reverse` 故不显形，与"长期存活"的解释一致。被审方 A2 表（`x: right@plane0` 期待 `[3,0,1,2]`、OSS 给 `[1,2,3,0]`；N=4 发散 rel L2=0.303、互换后 0）**与原文一致**，非编造。

### D. 通信记账 A4（解析计数，不读被审代码计数器）

D3Q19 沿 z 穿越面：`|cz|=1` 方向 10 个（+5/−5）⇒ 每 rank 每步 10 平面；16² 平面 float32 = 1024 B ⇒ **10240 B/rank/步**；朴素全 Q staging = 2×19=38 平面 ⇒ 比率 **3.80×**。与工件 A4 表逐项一致。

---

## 3. 严重度排序的发现（severity-ranked）

**F-1（中，非阻塞 · 生态侧）该锚无预登记基准判据**
`tensor_benchmark_cases`/`tensor_xval_cases`/`campaign_plan_nodes` 均无 `XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO` 行，容差只存在于任务 goal 的自然语言（"<15%"）。影响：本例 gate 只能核"benchmark 里有数值"（`has_numbers`），不能核"got 在容差内"。建议（不代做）：把本审计 §2 的 A3 三行 + A1 逐位 + A2 分支核账登记为 case 行，锚才有机器判据。

**F-2（中，非阻塞 · 判据含混）A3 度量的是 BGK 截断误差，不是 halo 保真度**
A1 逐位一致 ⇒ sharded 与单域**同一个解**，故 A3 的 1.875% 完全来自格式本身（审计复算证实：纯单域 numpy 实现同为 1.876%）。因此"vs 解析解 PASS"不能作为 ppermute 移植正确性的证据，只有 A1（等价性）＋变异对照能。工件文案已隐约点明，但表格并置容易让读者误读为"halo 误差 1.875%"。后果：级别不受影响（A1 才是功能判据），但**表述需收紧**。

**F-3（低，非阻塞）无反向生产调用方**
`grep -rn PpermutePeriodicHalo` 只在 `src/tensorlbm/parallel`、`tools/xval_ppermute_halo`、`tests` 命中；没有任何域求解器经 sharded 路径跑。故 L3 的"接入主求解器"是**前向**的（halo 驱动真执行器），而非"主求解器改用该路径"。与上游 vivsim 一样（该函数全仓无调用点），属新特性的正常处态，但不宜表述为"已接入生产求解器"。

**F-4（低，非阻塞）A4 墙钟表不可复现且已知为负收益**
审计重跑得到 ppermute 0.0109 vs 朴素 0.0076 s/步（speedup 0.699，原工件 0.728）——数值随机器负载浮动，且**方向为负**（ppermute 更慢）。工件已诚实标注 WITHHELD 不结论，但建议从"通信成本"表中把墙钟列降级为"参考、非度量"，避免后续被引用为性能证据。

**F-5（低，可修）A1 在执行器级缺负对照**
本次由审计补上（§2-B）。建议被审方把变异对照固化为测试，使"逐位一致"在仓内长期可证伪，而非依赖某次审计。

**F-6（记录）真实 `jax.lax.ppermute` 通路与多卡 NCCL 计时 WITHHELD（无 jax/mpi4py wheel）**
声明与 guard 测试（`test_jax_path_is_withheld_without_jax`）一致，属环境限制非缺陷。**后果**：被审方给出的"零 staging/选择性交换"结论仅在 torch 语义等价模拟下成立，跨设备行为未实证。

**F-7（记录 · 生态摩擦）知识检索引擎 `:8021` 离线**：被审方如实记录为 friction，未以之替代开工三查。

---

## 4. 检查了什么 / 没检查什么

**已检查（本次实跑）**：27 条测试；`repro_cmd` 全量重跑（63.2 s）与工件逐键比对；独立 numpy 实现复算三条网格；变异对照；`tensor_benchmark_cases` 等三表锚点登记查询；上游 OSS 原文 `git fetch` 核对；通信计数解析推导；commit 与工作树一致性（`git rev-parse HEAD == 2a7c492e`、`git status --porcelain` 干净）。

**未检查（unknown / not_run，不得视为通过）**：
* 真 `jax.lax.ppermute` 跨设备路径（无 jax）；真 NCCL/多节点线量与墙钟（无多卡）→ 性能收益仍 WITHHELD；
* 非周期边界（Zou-He 等）下的时序等价性——被审方 fail-closed 拒绝 `boundary_fn`，故该情形**未被实现也未验证**；
* 非整除轴长、N>8、非 z 轴分块在**执行器级**（A1）的覆盖：A1 只跑 axis=0 且 N∈{2,4}（测试）/ {2,4,8}（工件），三轴等价仅在**单元级**（`test_wrap_then_permute_equals_monolithic_stream`，axis=0/1/2 × N=2/3/4）覆盖；
* 长时间稳定性（≤100 步）、High-Ma、可压缩区；
* 上游 vivsim 是否**有其它调用点**（blob 级 `git show` 在本沙箱对大文件超时；仅确认 `multidevice.py` 原文层面的事实，未做全仓调用图核对）。

---

## 5. 被高估的主张（overstated claims）

| 声称 | 实际 | 判定 |
|---|---|---|
| "移植进 TensorLBM（L4）" | 成立（本审计独立复现） | 不夸大 |
| "接入主求解器（L3）" | 前向集成成立、反向无调用方 | **措辞偏强**，建议改为"驱动真求解器核心，尚无求解器经该路径" |
| "线量 3.8× 更低" | 解析计数成立（10 vs 38 平面） | 成立（仅线上量，非墙钟） |
| "jax 真通路保留" | 有实现+guard，环境不可跑 | 已标 WITHHELD，诚实 |
| 报告"六段齐全" | ①-⑥ 齐全含误差判定与"是否<3%"显式回答（1.875%<3% ✔）；④图 3 张 | 成立（篇幅偏短但要素齐） |

---

## 6. 处置（disposition）

**Approve with notes.** 级别 L4，verdict 已落 `tensor_maturity_verdicts`（evidence 指针见 §1）。不返修（gate 判 confirmed 语义），但要求后续轮次消化 F-1/F-2/F-5 三条（判据登记 + 表述收紧 + 负对照入测试）；F-3/F-4/F-6 作为已知边界写入锚台账，不得被引用为生产集成/性能收益证据。

**独立性限制（如实）**：审计者与被审方同栈（同一 TensorLBM 仓、同一硬件），A1/A2 的复算**必须**复用被审代码（变异对照的目的即是把"复用"变成可证伪）；A3 与 A4 走完全独立的第二实现/解析推导；上游缺陷核对走第三方原文（GitHub pin），是本审计独立性最强的一环。审计者未参与 m18112 的实现。