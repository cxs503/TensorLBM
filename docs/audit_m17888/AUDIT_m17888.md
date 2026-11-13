# 交差审计 m17888 · mission m17844 · anchor `XVAL-LBM-HORIZONTAL-SURVEY-EMPTY`（TensorLBM）

**裁定：L2（独立功能实现）confirmed** — 真代码、真工件、真可复现，但**不授 L3/L4/L5**。
被审对象：`/root/tp-agent/work/m17844` @ commit `95eb6b22`（branch `agent/m17844`，**不在 origin/main**）。
审计人：`board-audit`（本单 m17888），被审树全程只读；所有数字由 `artifacts/m17888_audit/*.py` 重算。

| 项 | 结论 |
|---|---|
| 三件套（run_artifacts / benchmark ref vs got / repro_cmd） | 工件 6/8 存在且**逐位可复现**；ref vs got 齐；repro_cmd **实跑通过** |
| 基准裁判 `tensor_benchmark_cases` | **交付时 0 行**（该 anchor 无裁判行）→ 本次审计登记 1 行，`status=pending`（honest FAIL，不是 passed） |
| 报告六段 | 齐（①–⑧），误差判定显式回答「不满足 <3%，也不满足 15%」= honest-FAIL 无粉饰 |
| 级别 | **L2**：输入→计算→输出真实现 + trace 级可复现；无 src 改动、无测试、无调用方、无裁判行、预声明梯截断 ⇒ L3/L4/L5 不成立 |

## ① 案例与判据（重述，不改判据）

交付自称复现 `oss_solutions id=386`（`oss-inventory/tensorplatform`, tag=`horizontal-survey-empty`）的**做法**：
先做水平开源调查，命中 0 行则如实记「无可移植实现」，改用文献锚自建验证案例。
判据（mission m17844 goal，预登记于 `xval/horizontal_survey_empty/predeclared.json`）：
圆柱振荡流惯性系数 `C_M` vs 文献锚 `C_M=2.00`（Morison 1950 / Sarpkaya 1986 理想流体极限），**误差 <15%**（owner 标准），
生态线 `<3%`；`KC = U_max T / D`、`beta = D^2/(nu T)`；预声明 8 算例、步数 = (1.5+3.0) 周期、无事后调阈值。

## ② 方法与验证路径（本次审计怎么判）

1. **静态**：读 `survey.py / run_arm.py / analyze.py / predeclared.json / docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md` 与 `runs/*/result.json|trace.npz`。
2. **重算**（`artifacts/m17888_audit/audit_m17844.py` → `audit_m17844.json`）：由 `result.json` 的 `kpi` 重算 `err%=100*(C_M-2)/2`；由 `trace.npz` 用**独立重写的 Morison 最小二乘**反算 `C_M/C_D`；由 CSV 重扫调查命中；由 `1/D` 两点重算外推。
3. **复现**（`repro_res16_vs_delivered.json`）：在被审树原样跑 `python3 xval/horizontal_survey_empty/run_arm.py --case res16 --threads 8`，与交付工件逐位比对。
4. **对抗式复现**（`diag_interior_zeroing.py` → `diag_interior_zeroing.json`）：独立重写交付的 §⑥「1.48→1.95」常数加速诊断，做幅度扫描。
5. **表证**：`psql` 查 `tensor_benchmark_cases` / `tensor_maturity_verdicts`；`git merge-base --is-ancestor` 判交付是否入 main；`grep` 判库内调用方与重复能力。

## ③ 计算结果（全部机器重算，零手抄）

| case | KC_ach | beta_ach | D_halfway | C_M(E1) | err% 重算 | 15% 带（halfway） | err%（D_area） | 15% 带（area） |
|---|---|---|---|---|---|---|---|---|
| res16 | 0.9439 | 115.4 | 17.187 | 1.6153 | **-19.24** | ❌ | -9.92 | ✅ |
| base  | 0.9860 | 106.5 | 24.773 | 1.5742 | **-21.29** | ❌ | -15.31 | ❌ |
| kc05  | 0.4864 | 106.5 | 24.773 | 1.4342 | -28.29 | ❌ | -22.85 | ❌ |
| beta50| 0.9827 | 53.3  | 24.773 | 1.6715 | -16.43 | ❌ | -10.08 | ✅ |
| beta200| 0.9849 | 213.1 | 24.773 | 1.5134 | -24.33 | ❌ | -18.59 | ❌ |
| box16 | 0.9797 | 106.5 | 24.773 | 1.1687 | **-41.56** | ❌ | -37.13 | ❌ |
| **res32** | — | — | — | **缺失**（预声明第 3 档收敛梯） | — | — | — | — |
| **kc20** | — | — | — | **缺失**（预声明 KC=2） | — | — | — | — |

* `err%` 重算与交付声明**逐一相同**（不是抄报告：由 `kpi.E1.C_M` 重算）；6 个 `runs/<case>/trace.npz` 独立重拟合 `C_M` 相对差 **≤0.0068%**、`C_D` 一致 ⇒ 工件内部自洽、无手改痕迹。
* 调查仪器核验：`data/*.csv` sha256 与 `PROVENANCE.json` **完全一致**（33 行 census / 261 行 ledger）；`added_mass_pmm` 族三源命中重扫 **0/0**，与 `survey_summary.json` 的 `empty` 一致 ⇒ 「调查为空」是真结论，非仪器失灵（对照组 `immersed_boundary`/`non_newtonian` 非空）。
* **复现**：`res16` 原样重跑 `C_M=1.6152540471358583` vs 交付 `1.6152540471358583`，**逐位相同**（`err` -19.2373 / `C_D` 4.6509240 全同；仅 wall_s 43.0 vs 168.4 不同，机器负载差异）。

## ④ 图（`artifacts/m17888_audit/fig_m17888_evidence.png`，3 联）

* (a) `C_M` vs `1/D_halfway`：交付只有 2 档（res16/base），预声明的 res32 缺；两点 `1/D` 外推 **C_M(∞)=1.481**（**-25.9%**）、C_M 随 D 增大**反向远离**参考值 ⇒ 「分辨率收敛到理想值」**未被证明**，图题「收敛梯与 1/D 外推」在 2 点上不成立。
* (b) 盒周期梯：`L/D 12 → 16` 使 `C_M` 从 1.574 掉到 1.169（**-26%**），而预声明把周期像估为 `O((D/L)^2)=0.7%` ⇒ KPI **未做域收敛**，量级已超 15% 判据本身。
* (c) §⑥「1.48→1.95」独立复现：交付幅度 `a0=0.04` 下第 **22–23 步**即非有限（周期盒内恒定加速度使 Ma→∞），「步 100」在所述条件下**不可达**；小幅度（a0≤1e-4，100 步稳态）实测 **1.589 → 1.835**（a0=1e-3 时 1.743 → 2.014）。

## ⑤ 注意点：按严重度排序的发现

| 级别 | 发现 | 证据 |
|---|---|---|
| **F1 高** | **交付未入 main**：`git merge-base --is-ancestor 95eb6b22 origin/main` = 假；只存在于 `agent/m17844`。用户侧不可见、无调用方、`src/` 零改动、无测试 | `audit_m17844.json:git`；`grep` 库内调用 0（命中全是无关文件） |
| **F2 高** | **交付时无基准裁判行**：`tensor_benchmark_cases` 该 anchor 0 行 ⇒ mission 的「<15% 判据」当时无机器裁判可裁；本次审计补登 1 行 `pending`（honest FAIL，非 passed） | SQL 直查 + 本次登记行 |
| **F3 高** | **预声明梯截断且报告措辞过头**：预声明 8 算例，交付 6（`res32`/`kc20` 的 `runs/` 无 `result.json`，shard_2/shard_4 日志仅 1 行）；报告 §④ 与 `docs` 写「收敛梯 3 档中 0 档达标」而实际只有 2 档；报告 §⑥「res32 与 kc20 仍在跑」在 mission succeeded 结案时已不成立 | `audit_m17844.json:cases`、`ladder`；`runs/shard_2.log` 1 行 |
| **F4 高** | **「缺陷修复 1.48→1.95」这对数字无工件背书且所述条件不可达**：交付未附诊断脚本/输出；审计独立重写后，`a0=0.04` 第 22 步即 NaN，稳态极限为 **1.589→1.835**（不是 1.48→1.95）。**方向成立（内部节点置零确有 +15% 相对改善），具体数值不可复现** ⇒ 该对数字在补齐工件前只能按「未核」计 | `diag_interior_zeroing.json` |
| **F5 中** | **E2 估计量非物理**（`C_M(E2)` 全为负，E1–E2 差 103–222%）⇒ 两个「独立」估计量实际只剩 1 个可用，测力算子**缺一项自洽校验**；而内部节点置零会非守恒地湮灭动量，恰是 E2 差值的可疑来源（交付仅声明「未记账」，未定位） | `table.csv` / `audit_m17844.json:kpi_table.CM_E2` |
| **F6 中** | **域收敛缺失使「-21%」不能当物理结论**：盒梯 26% 的量级大于判据 15%（见 ④b）⇒ 交付只能说「未验证」，不能说「本实现 C_M 偏负 21%」 | `kpi_table.base/box16` |
| **F7 中** | **能力重复而非扩展**：库内已有 `compute_added_mass_2d`（`src/tensorlbm/postprocess.py:835`，`tests/test_postprocess_new.py:27` 有测），交付另写 `morison_ls` 且未复用 ⇒ 缺口只是**验证覆盖**，不是能力；交付未给库增加任何接口 | `audit_m17844.json:duplication` |
| **F8 低（正）** | **无造假**：报告/文档/`mission.result` 三处一致申报 honest-FAIL（-21.29%，明答不满足 <3% 与 15%），未挑选 `D_area` 的 15% 通过约定；调查空集有 sha256 溯源 | 报告 §⑥；`PROVENANCE.json` |
| **F9 低（平台）** | mission `server_quality_audit.gate_valid=false, gate_missing_warning=true` 仍被 auto-approve 结案 ⇒ 平台侧「无裁判行交付」可溜过验收（本行归平台，不入本仓返修） | `tensor/missions/17844` |

**未跑/不可核（记 unknown，不算 pass）**：`res32`、`kc20` 预声明算例（>2 min/案，未续跑）；E2 测力算子缺项的物理定位；OSS 侧无实现（0/0/0）⇒ **外部锚结构性不存在**，`external_anchor=WITHHELD`（文献值不是可执行外部实现）；`β=100/D=24` 时 Stokes 层仅 1.35 格，粘性贡献未解析（交付已预登记为混淆项）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 量 | got | ref | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|
| `C_M`（base, D_halfway, KC≈0.99, β≈107） | 1.5742 | 2.000 | **-21.29%** | **否** | **否** |
| `C_M`（base, D_area 约定） | 1.6937 | 2.000 | -15.31% | 否 | 否（临界） |
| `C_M`（res16） | 1.6153 | 2.000 | -19.24% | 否 | 否 |
| `C_M(∞)` 两点外推 | 1.4813 | 2.000 | -25.93% | 否 | 否 |
| 复现一致性（res16 `C_M` 重跑） | 1.6152540471358583 | 同值 | **0.0%** | 是 | 是 |
| trace 重拟合一致性（6 案） | — | — | ≤0.0068% | 是 | 是 |
| 调查输入 sha256 | 37359bac…/9ab9e3cf… | 同值 | 0 | 是 | 是 |

**结论：物理 KPI 全部不达标（honest-FAIL 成立）；工件/复现一致性 <0.01%。**

## ⑦ 裁定与返修建议

**级别 L2 confirmed**（真实现 + 真工件 + 逐位可复现；无集成/无裁判/梯截断 ⇒ 不授 L3/L4/L5）。
返修清单（`reopened` 应做的）：
1. 把 8 个预声明算例补齐（至少 `res32`、`kc20`），否则收敛梯只能是 2 档、外推不成立；
2. 先做**域收敛**（L/D 12/16/20）再谈 C_M 物理误差——当前域依赖 26% 大于判据；
3. 补一个可执行工件支撑「内部节点置零」修复（当前数值对需按 1.589→1.835 改写，或附诊断脚本+JSON）；
4. 定位 E2 缺项（内部节点置零的非守恒动量），或明写「E2 为已知错误估计量」并给量级界；
5. 复用 `compute_added_mass_2d`、把振荡流 C_M 校验挂进库（才谈 L3）；补测试；
6. 合并入 main 并保留 `tensor_benchmark_cases` 裁判行（本次已登记 `pending`）。

**独立性与局限**：被审树只读（未改 `work/m17844` 任何文件，复现输出写入本单 `artifacts/`）；未续跑缺失算例；未做 E2 缺项的独立推导；文献锚 `C_M=2.00` 是理想流体极限，与 KC≈1、β≈100 的实验值（Sarpkaya 量级 1.6–1.8）并非同一口径——本审计不据此改判（预登记判据是唯一裁判），但列入返修建议：**口径选择需在预声明里钉死**。