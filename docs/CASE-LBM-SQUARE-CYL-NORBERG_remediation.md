# CASE-LBM-SQUARE-CYL-NORBERG 返修：参考口径单一化 + 校验器可机械复核（audit m16183 五项返修）

锚点 `CASE-LBM-SQUARE-CYL-NORBERG` · mission **m16553** · branch `agent/m16553` · repo TensorLBM

本单**不给新物理结论**。上一轮交差审计 `audit_m16183.md`（m16205）对该锚点裁定
**reopened（request changes）**，理由是「工件层面数值达标，锚点层面未验证」：数值全来自
2026-09-29 前序 commit `d9a45c85`，m16183 自身零交付，且仓内 `verify.py` **在现树上跑不动**、
存在**两套参考常量**、验收口径**没锁测力列**、无 `tensor_benchmark_cases` 裁判行、`tests/`
里对该算例的覆盖为空。本单逐项清偿这五项欠账。

## ① 案例介绍（物理背景 / 几何 / 参数表）

二维自由流方柱绕流（正置 α=0°），Re = U·D/ν = 100，目标量两项：平均阻力系数
Cd = Fx/(½ρU²D)、涡脱 Strouhal 数 St = f·D/U。

| 参数 | 值 | 说明 |
|---|---|---|
| 碰撞/格子 | D2Q9 · MRT | τ_field sponge（τ_eff = τ(1+ασ), α=10, power 2） |
| 域 | 40D × 40D（blockage 2.5%） | 方块中心距入口 10D，下游 10D sponge |
| u_in / ν / τ | 0.05 / 0.016 / 0.548 | ν = u_in·D/Re |
| 播种 | St_seed=0.14，2 周期，10% 振幅 | 只触发、不锁频（对称柱需人工非对称扰动起振） |
| 统计窗 | 后 50%（D=32 从 step 30000 起） | 60k 步（D=32）/ 90k 步（D=48） |
| 参考（锁定） | **Cd_ref = 1.48 ∈ [1.44, 1.52]，St_ref = 0.147 ∈ [0.145, 0.149]** | 2D 自由流低阻塞**数值**簇 |
| 排除参考 | Okajima 1982 实验（St 0.143–0.145，无 Cd）、Sohankar 1997 通道（Cd≈2.05）、Sharma&Eswaran 2004（Cd=1.57） | 2D 求解器不得对标 3D 实验/高阻塞口径 |

参考来源（原表）：arXiv 2411.03124v4 Tab.1、2309.09197 Tab.2、2308.08085v3 Tab.7/8、
2404.12123、2405.12834；14 点交叉核对见 `REFERENCE_AUDIT.md`。锚点名含 Norberg（实验口径），
而本算例刻意对标数值口径——这一命名/口径冲突是审计的核心发现之一，**按审计建议不在本单
擅自改名**（改名属锚点层裁决），改为在裁判行 `ref_source` 里把口径写死。

## ② 怎么做的（方法 / 流程 / 工具链）

上一轮审计的返修清单 → 本单动作，逐条对表：

| 审计返修项 | 证据（审计原文） | 本单动作 | 验证 |
|---|---|---|---|
| 2. 修 `verify.py` schema 兼容，使现树可机械复核 | `check("case files found")` 报 `D=[]`，随后 `cases[ds[-1]]["pair"]` 抛 `IndexError` | 新增 `refspec.normalize_case()`：summary schema（`case_D32/48.json`）与 triple schema（`case_D64.json`）统一展平；`verify.py` 全部走该入口，双 schema 均可裁 | `python verify.py` → `VERIFY RESULT: ALL OK`（OK 27 / SKIP 5 / FAIL 0）；改前为 `IndexError` |
| 3. 统一参考常量（run.py 1.48/0.147 vs verify.py 1.4785/0.1456） | 同一份数据给出两套误差数字 | 新增 `refspec.py` 作**唯一**常量源（`REF_CD/REF_ST/band/TOL/测力列策略`）；`run.py`、`verify.py`、`verify_traces.py` 全部改为 import；旧常量降级为 `SUPERSEDED_REF`，**只用于识别历史误差列的出处**，不参与评分 | 结构性强校验 `test_refspec_is_the_only_constant_source`（三文件均不得再赋值 REF_*）；漂移门 `test_reference_drift_is_a_hard_failure` |
| 3b. 测力列敏感性 8.53%：判据须锁死测力定义 | `wet` Cd 1.6139 vs `mem` 1.4870（+8.53%），wet 单独会超 3% 门 | `refspec.PRIMARY_COLUMN = "pair"`（与 `mem` 闭合 0.0%），`wet` 降为 diagnostic；主门在 `verify.py` 头部打印所用列 | `test_force_column_policy_is_material`（断言 pair 过门而 wet 不过门，否则策略无意义） |
| 1. 注册 `tensor_benchmark_cases` 裁判行 | 查询 0 行 → 无人有资格宣布过基准 | `register_case.sql`（幂等：DELETE+INSERT），已在本机执行，`status='pending'` 等平台复核 | `psql` 回读该行；`test_registry_sql_matches_refspec` |
| 4. m16183 补报或接棒 mission 全量重跑并归档 Cl 时序 | 无重跑、无 Cl 时序 | **未完成，如实申报**：本机 D=32（1280²）单步约 0.4 s（8 线程实测 D=8 25 ms/step × 16 倍格点外推），60k 步 ≈ 6–7 h，超出本单窗口；集群作业 cj1329（D=8 冒烟）排队中。**改为把门做好**：`verify.py --repro-dir DIR` 已实现重跑对账门（≤0.5%），一旦有重跑即可机械判 | `test_repro_gate_fires_when_supplied`（+2% 输入必 FAIL，一致输入必 OK）；当前无重跑 → 门报 `[SKIP] ... audit item 4 is open`，**SKIP 不冒充 PASS** |
| 5. `tests/` 增加回归测试 | `grep -rl square_cylinder tests/` 为空 | `tests/benchmarks/test_square_cylinder_case.py`（11 例，stdlib+subprocess，不载 torch） | `pytest tests/benchmarks/test_square_cylinder_case.py -q` → **11 passed** |

**判别性设计（防止测试空转）**：每个失效模式都用「篡改归档副本」实测触发，而不是只跑一遍绿：
把 `case_D32.json` 的 `cd_ref` 改回 1.4785 → 校验器 rc=1；把 `err_pct` 改成两套常量都对不上的值 →
`stored err attributable to a named convention` FAIL；把重跑值放大 2% → 重跑门 FAIL。

工具链：Python 3.11 + stdlib（verify/tests）+ matplotlib（图）+ psql（裁判行）；无 torch 依赖，
因此校验器可在任何无 GPU/无 SDAA 的机器上机械复核。

## ③ 计算结果（数字表，程序化注入零手抄）

锁定口径（`refspec.py`）复算全部既有工件（**未修改任何测量值**）：

| 网格 D | schema | 主测力列 | Cd | vs 1.48 | St | vs 0.147 | 3% 门 |
|---|---|---|---|---|---|---|---|
| 32 | summary | summary | 1.4816 | **+0.11%** | 0.1482 | **+0.82%** | PASS |
| 48 | summary | summary | 1.4844 | **+0.30%** | 0.1480 | **+0.68%** | PASS |
| 64 | triple | pair | 1.4870 | **+0.47%** | 0.1479 | **+0.61%** | PASS |
| 64 | triple | mem（复现列） | 1.4870 | +0.47% | 0.1479 | +0.61% | PASS（pair−mem 闭合 0.000%） |
| 64 | triple | wet（**diagnostic，不评分**） | 1.6139 | **+9.05%** | 0.1479 | +1.58% | 若作主判据 → **FAIL** |

- 跨网格跨度：Cd [1.4816, 1.4844, 1.4870] → span 0.36%；St [0.1482, 0.1480, 0.1479] → span 0.20%
  （解算器自身的网格收敛性远好于 3% 门）。
- **历史误差列的出处已定位**：`case_D64.json` 里存的 `err_cd_pct=0.57 / err_st_pct=1.58` 是
  **被取代的** 1.4785/0.1456 口径算的（(1.487−1.4785)/1.4785 = +0.57%），与 D32/D48 的 1.48/0.147
  并存 = 审计所说的「同一份数据两套误差」。本单不篡改历史数字，而是在校验器里**声明其为 legacy**
  （`DECLARED LEGACY ... not scored`）并全部改按锁定口径复算给分。
- 裁判行（已入库）：`anchor_slug=CASE-LBM-SQUARE-CYL-NORBERG`，`tolerance` 含
  `|Cd err| <= 3% AND |St err| <= 3% (refspec.TOL_PCT = 3.0) on the finest grid`，
  `status='pending'`（等平台复核，不自签 L4）。

## ④ 结果图

`artifacts/m16553/fig_square_cylinder_remediation.png`（`make_remediation_figure.py` 从归档 JSON 重绘，
含 sha256 记录于报告）：
- 左：Cd/St 随 D 的实测点 + 锁定参考带（蓝/橙），并标出 D=64 的 wet 列（红叉，+9.0%，单独过不了门）
  —— 直观显示「测力列必须锁死」为何是实质问题；
- 右：两目标量误差柱 vs 3% 容差线，显示单调收敛与两量余量。

（本单为验证/工程返修，非新仿真，图由既有工件派生，不宣称新数值。）

## ⑤ 注意点 / 局限

1. **完整重跑未做**（审计返修项 4 仍开）：本文所有 Cd/St 仍是 2026-09-29（`d9a45c85`）的测量。
   本单只让它们**可被机器复核**，没有让它们**被独立复现**。要签发 L4 必须补一次全量重跑
   （D=32/48 60k/90k 步）并用 `verify.py --repro-dir` 过 0.5% 对账门——工具已就绪，缺算力窗口。
2. 播种不锁频：St_seed=0.14 只触发起振，最终 St≈0.148；对称方柱的起振依赖人工非对称扰动，
   这是该算例的可复现性脆弱点（未在本单改变）。
3. 锚点名（`...NORBERG`，实验口径）与交付口径（数值簇）不一致，属锚点层命名裁决，本单只把
   口径写进裁判行 `ref_source`，**未改名**。
4. `verify.py` 的几何门（solid cells = D²、surface links = 12D−4）对 summary schema 只能 SKIP
   ——summary 记录不含 `force_columns`，几何不可从该记录重推。SKIP 已显式打印，不计入 PASS。
5. 本单未动 `src/tensorlbm/**` 任何求解器代码，改动仅限算例目录 + 测试 + 一个 SQL + 一个图脚本。

## ⑥ 误差判定（显式回答是否 <3%）

**锁定口径（`refspec.REF_CD=1.48 / REF_ST=0.147`）主测力列 pair/summary：**

| 关键量 | got | 基准 | 误差 | <3%？ |
|---|---|---|---|---|
| Cd (D=32) | 1.4816 | 1.48 | +0.11% | **是** |
| St (D=32) | 0.1482 | 0.147 | +0.82% | **是** |
| Cd (D=48) | 1.4844 | 1.48 | +0.30% | **是** |
| St (D=48) | 0.1480 | 0.147 | +0.68% | **是** |
| Cd (D=64, pair) | 1.4870 | 1.48 | +0.47% | **是** |
| St (D=64, pair) | 0.1479 | 0.147 | +0.61% | **是** |
| Cd (D=64, wet — diagnostic) | 1.6139 | 1.48 | +9.05% | **否**（故 lower-rank 列不作判据） |

最大误差 **1.58%**（D=64 wet 列的 St），主判据最大误差 **0.82%** → **是，全部 <3%**。
但**必须同时声明**：这些数字来自 2026-09-29 的前序运行，本单**没有重跑**，因此「<3%」成立的是
**工件-基准对账**，不是**独立复现**；审计返修项 4（全量重跑）仍为 open，裁判行 `status='pending'`
等平台复核，本单不自签 L4。

## 复现命令

```bash
cd benchmarks/verified/square_cylinder
python verify.py                      # 现树机械复核：OK 27 / SKIP 5 / FAIL 0，exit 0
python verify.py --repro-dir <DIR>    # 有重跑归档时加跑 ≤0.5% 对账门
python make_remediation_figure.py ../../artifacts/m16553/fig_square_cylinder_remediation.png
python -m pytest tests/benchmarks/test_square_cylinder_case.py -q   # 11 passed
psql -d tensorplatform -f register_case.sql                        # 裁判行（幂等）
```