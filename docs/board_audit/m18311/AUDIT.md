# m18311 · 交差审计 — m18193 `XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER`

**对象**：mission 18193（TensorLBM，succeeded），`repo=TensorLBM / scope=m18193 XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER`
**被审交付**：commit `bcf10687`（branch `agent/m18193`）——`src/tensorlbm/porter_forcing.py`（294 行）、`tests/test_porter_forcing.py`（13 项）、`xval/XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER/`（REF/GOT/legacy 三臂 + `artifacts/result.json` 35 个 run + 3 图 + `REPORT.md` 七段 + `predeclared.json`）
**裁判**：`tensor_benchmark_cases`（anchor 行）+ owner 精度标准 <15% + 被审单自落盘的 `predeclared.json`
**裁定**：**L2**（独立功能实现）；L3/L4/L5 均被拒，理由见 §4
**独立性限制**：审计方与被审单同属一个 agent 生态、共用工装；被审 `REPORT.md` 的作者视角已按「先重算、后读报告」的顺序规避——本审计先用 `git archive bcf10687` 取快照、重跑 `run_xval.py`，再逐条比对报告文字（见 §5 F2/F3 两处正因如此才被抓出）。

---

## 1. 取证（三件套逐项核）

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件真存在 | ✅ | `refs/xval/.../artifacts/result.json`，sha256 `22610482…fc0`（`refs/MANIFEST.json`；与 commit `bcf10687` 内文件、与被审工作树现文件三者同哈希） |
| benchmark ref vs got 数值 | ✅ 有 / ❌ 判据不一 | 见 §2、§3 |
| repro_cmd 可执行 | ✅ **逐位复现** | 审计侧重跑 `python3 xval/.../run_xval.py`（~4 min）：35/35 run 键完全一致，`err` 最大相对差 **0.0**（bit-for-bit），`b3` stencil 段 JSON 完全相同 → `.audit/repro/artifacts/result.json`、`audit_repro.json` |

被审工作树 `/root/tp-agent/work/m18193` 在审计期间被其它进程写过（一个 `sc_closed_box.py` 出现又消失）→ 全部取证改用 `git archive bcf10687` 的不可变快照，现工作树仅用于交叉核对哈希。

## 2. 基准裁判：`tensor_benchmark_cases`

**审计前：该 anchor 一行都没有。**

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER';   -- 0 rows
```

即：<15% 当时只是**报告自声明**，没有任何机器裁判。本审计补登 3 行（`insert_rows.py`，幂等）：

| id | case | tolerance | status |
|---|---|---|---|
| 106 | B1 单组元体力驱动流道 vs 解析 Poiseuille | <15%（实测 3.249e-04 @NY=34 / 5.066e-03 @NY=66） | **passed** |
| 107 | B2 两组元分层 Stokes，M∈{1,5,10,100} vs 解析两层解 | <15%（M=1 3.43e-04 过；M=5 21.9% / M=10 40.9% / M=100 80.2% 不过） | **failed** |
| 108 | PORT-FIDELITY：仓内 API vs 独立逐行转写 | <1e-6（实测最差 8.38e-11） | **passed** |

## 3. 数字重算（全部由 `audit_repro.json` 程序化产出，无手抄）

| 量 | 记录值 | 审计重跑 | 判据 | 结论 |
|---|---|---|---|---|
| B1 单组元 NY=34 误差 | 3.2492e-04 | 3.2492e-04 | <15% / <3% | ✅✅ |
| B1 单组元 NY=66 误差 | 5.0660e-03 | 5.0660e-03 | <15% / <3% | ✅ / ❌(拉伸) |
| B2 分层 M=1 NY=34 | 3.4255e-04 | 3.4255e-04 | <15% | ✅ |
| B2 分层 M=5 NY=34 | 2.1884e-01 | 2.1884e-01 | <15% | ❌ |
| B2 分层 M=10 NY=34 | 4.0927e-01 | 4.0927e-01 | <15% | ❌ |
| B2 分层 M=10 NY=66（最细） | 1.8139e-01 | 1.8139e-01 | <15% | ❌（最好也只有 18.1%） |
| B2 分层 M=100 NY=34 | 8.0190e-01 | 8.0190e-01 | <15% | ❌ |
| legacy 控制臂 M=10 NY=66 | 5.6315e-01 | 5.6315e-01 | 区分度 ≥3× | ✅ 3.105×（**仅 NY=66**） |
| REF vs GOT 最差配对 | 报告称"<1e-15 / 4 配置" | 3 对配对，最差 8.38e-11 | <1e-6 | ✅ 门槛过；表述见 F2 |
| 质量漂移 | ≤1.7e-13（99 步量级） | 同 | 守恒 | ✅ |
| 有效粘度（机制） | ν_eff/ν_nominal（轻质层，M=10）= 4.45/3.04/1.23（H=16/32/64） | 同（由 `nu_eff_measured/nu_exact` 现算） | — | ✅ 机制定位成立：扩散界面抹平粘度反差，随界面分辨率单调收敛 |

**判据结论**：单组元 + M=1 **过**；**两组元 M≥5 全部不过 15%** → 被审单的 `honest-FAIL` 自评**与数字一致，未粉饰**（这点值得肯定：它没有拿"比值 3.1×"去冒充精度达标）。

## 4. 判级（L0–L5）：**L2**

| 级 | 判据 | 裁定 |
|---|---|---|
| L1 数学/单元 | `tests` | ✅ 13/13 pass，且断言的是**不变量**而非"跑通即过"：`ΣF_i=0`、`ΣF_i c_i=F`、力 PDF 二阶矩为 0、iso4 `Σw=5/3, A4=1, A6=0.6, A8=3/7`、线性场梯度精确、三种书账动量注入 `½F/F/(1−½τ)F` |
| L2 独立功能 | 真代码+入口+运行工件 | ✅ **授予**：`porter_forcing.py` 294 行 numpy 实体（非占位），`run_xval.py` 输入→计算→输出，工件在、且**确定性可复现（0.0 差）** |
| L3 系统集成 | 主求解器/平台调用方 | ❌ **拒**：`grep -rn porter_forcing src/` = 0 命中；`src/tensorlbm/__init__.py` 无导出；生态仓 `/root/tensor-ecosystem/TensorLBM` 完全无该文件（`git merge-base --is-ancestor bcf10687 HEAD` → **NOT MERGED**）。能力只被 xval 自己的 driver 使用 = 仍是库级原语，不在任何求解回路里 |
| L4 基准验证 | 完整算例+误差分析+可复现 | ❌ **拒**：可复现 ✅、误差分析 ✅，但 **owner <15% 判据在两组元高粘度比子项上不成立**（M=10 40.9%）；且外部锚只有自解解析解（无实验/手册原值）。`tensor_benchmark_cases` 行 id=107 记 `failed` 即为机器裁决 |
| L5 科研工程 | 新物理/手册锚 | ❌ 拒：无新物理、无手册原页；结论层面反而是**被证伪的参考主张**（"四件套支撑大粘度比"在 G=0 时不被数据支持） |

> 门禁自校：把本 evidence 喂给 `maturity_gate.judge()` → `(2, [])`；即使声明 L3 也只会被机器降到 2（`integration_callers` 空）——不存在虚标空间。

## 5. 发现（按严重度）

| id | 级 | 事实 | 处置 |
|---|---|---|---|
| **F1** | 高 | `tensor_benchmark_cases` 无本 anchor 行 → <15% 无机器裁判 | **已修**：补 3 行（106 passed / 107 failed / 108 passed） |
| **F4** | 高 | 交付未合入 main：主干（worktree m18311 与生态仓）无 `porter_forcing.py`；分支 `agent/m18193` 未合并。能力当前对任何求解器不可见 | 记录为 L3 阻断项；本审计按承诺只判级不代合 |
| **F2** | 中 | `REPORT.md` §四称"REF 与 GOT 在 **4 个**配置逐位相同（…/4.703e-01），差异 <1e-15"。实测 `doc_seq@M=10` **没有 GOT 行**（4.703e-01 是孤儿数），真正配对只有 **3** 对；最差配对相对差 **8.38e-11**，不是 <1e-15 | 记录；不改变结论（8.38e-11 ≪ 1e-6 门槛），但"4 配置/1e-15"两处表述属**夸大** |
| **F3** | 中 | 报告开篇"判据要求的 3 倍区分度达成 3.1×（**NY=34**）"。实测 NY=34 比值 **1.481**，3.105× 只在 **NY=66**；`predeclared.json` 的 discrimination 条目未写网格 → 判据本身歧义 | 记录；主张应收窄为"最细网格 NY=66 上 3.105×" |
| **F5** | 低 | 被审工作树在审计期间被并发写入（`sc_closed_box.py` 闪现即消失） | 用 commit 快照 + sha256 manifest 消除污染 |

**未构成缺陷、但需注意的适用范围**（复核后认为报告已如实声明，故不判罚）：B2 用解析解作初值（fixed_point）而非从静止起步——报告 §二 给了理由（轻质层动量扩散时间 `H²/ν≈61000` 步超预算），并保留了 B2t from-rest 作敏感性；`G=0`（无 SC 相互作用力）是放不出真实分离界面、误差 40.9% 的根因，报告 §六 已把它写成接棒方向。

## 6. 没跑的检查（unknown / not_run，不算 passed）

1. 外部实验/手册值锚 —— 两层 Stokes 分层无现成实验基准，**WITHHELD**。
2. 与 openLBMPM 源码的逐行对照 —— 参考仓无许可证 + 外网被策略拦截，**not_run**（只能按平台读码记录 + 论文实现）。
3. 叠加 `G<0` 伪势力的真实两相算例 —— 未做（被审单自己列为下一轮）。
4. 3D、非 D2Q9 格子 —— 未做。

## 7. 处置

**approve with notes → 判 L2 落表（verdict id 734），门禁 `judge(2, …) = (2, [])` ⇒ 预期 `confirmed`（消账）。**
不建议以 L4 消账：精度判据在两组元子项上客观不成立，补百页报告也不改变这一点。
两条真增量留给后续：①B1/M=1 的**可复现库级原语**（L2，已可用）；②**被否定的参考主张** + 定量机制（ν_eff/ν_nominal 4.45→3.04→1.23）——这是可入负知识账的科研收获，不是能力级别。

## 8. 工件索引

```
docs/board_audit/m18311/AUDIT.md                 本报告
docs/board_audit/m18311/audit_repro.json         记录值 vs 审计重跑（35 run、REF/GOT 配对、B1/B2 矩阵、ν_eff）
docs/board_audit/m18311/benchmark_rows.json      补登的 3 条机器裁判行
docs/board_audit/m18311/verdict_evidence.json    落表 evidence（程序化生成，键=impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd）
docs/board_audit/m18311/build_evidence.py        上述两个 JSON 的生成器
docs/board_audit/m18311/insert_rows.py           幂等落库（benchmark 行 + verdict）
docs/board_audit/m18311/refs/                    commit bcf10687 快照 + MANIFEST.json（sha256）+ rerun_result.json（审计侧重跑原件）
docs/board_audit/m18311/figures/                 报告用图（由 audit_repro.json 程序化生成）
```