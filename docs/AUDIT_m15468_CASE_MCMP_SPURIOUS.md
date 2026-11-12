# 交差核验：m15468 · CASE-MCMP-SPURIOUS（静滴伪电流强度基准，TensorLBM）

**被审** m15468（succeeded，2026-11-12 00:41:52），交付 commit `587b6fe3` "m15468: matured mission deliverables"（2026-11-12 01:00:59）
**被审树** `/root/tp-agent/work/m15468`（只读取证）｜**审计树** `/root/tp-agent/work/m15892`（本单）
**审计基线** 本工作树 `53046bcc`（Merge #1853）
**判级** **L2 独立功能实现**（真代码端到端出数，2D 数值可精确复现）；**基准验证 L4 不成立**，另有 1 项不可复现工件（3D）与 1 项交付缺件（repro 脚本未入库）。

---

## ① 案例介绍（物理背景 / 几何 / 参数 / ref 出处）

静止液滴在无外力时的精确解是 `u ≡ 0`；残留的 `U_spurious = max|u|` 是伪势多相 LBM 的虚假流，等于毛细驱动流动的可解析速度下限，故它是一个**零解检验**。任务容差判据（原单 goal 明写）：`U_spurious < 1e-3`（格子单位）。

| 参数 | 2D 基准（m15468 采用） | 3D 点（m15468 采用） | 出处 |
|---|---|---|---|
| 格子/引擎 | D2Q9，numpy float64，双层 SC 双组分 | D3Q19，torch float64，指数型 ψ | `src/tensorlbm/verification/spurious_current.py:383` / `src/tensorlbm/pseudopotential_edm3d.py:646` |
| 网格 N / 液滴半径 R | 96 / 24.0（=0.25N） | 32 / 8.0 | `artifacts/mcmp_spurious_2d_quick.json`、`mcmp_spurious_single.json`（被审工件） |
| 耦合 | G12 = +2.5（库内约定 = G_12 = −2.5） | G_eff = −5.0 | 同上 |
| 界面宽度 W / τ | 4.0 / 1.0 | 2.0 / 1.0 | 同上 |
| 力格式 / 强制 | lattice（baseline）、geometric / Guo | difference / kupershtokh(EDM) | 同上 |
| 步数 / 收敛 | 1500 步，`converged=false` | 400 步，`converged=false` | 同上 |
| 无量纲数 | Re=N/A（静态），ρ_l/ρ_v≈6.48（工件 0.8962/0.1383），Ca_sp=μU/σ≈1e-4~1e-2 | ρ_l/ρ_v=12.62（Maxwell 共存 1.9289/0.1529） | 审计由工件密度复算 |
| ref | Krüger et al. (2017) ch.9 量级 1e-3~1e-2 格子单位（**无页码/表号**） | 同 | `docs/CASE_MCMP_SPURIOUS_REPORT.md:18-24` |

**基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-MCMP-SPURIOUS'` → **0 行**（全表 25 行，TensorLBM 仅 `GAP-LBM-MCMP-POISEUILLE`、`GAP-LBM-SPHERE` 两行）。**该锚点没有登记裁判**，故"got 是否在容差内"无表可裁，只能用原单 goal 的 `U_spurious<1e-3` 作事实判据（发现 F4）。

---

## ② 怎么做的（审计方法，非重做）

1. **读交付**：`git show --stat 587b6fe3` → 仅 3 个文件（2 个 JSON 工件 + 1 篇报告），**零源码、零测试**；2D 引擎 `spurious_current.py` 属既有模块（node p362w2-n06），m15468 是"复用现有能力跑算例"（原单 goal 允许）。
2. **三件套逐项核**：run_artifacts（分支上 tracked，见 ③.1）/ benchmark ref-vs-got（无裁判表，见 ①）/ repro_cmd（**脚本从未入库**，见 ③.4）。
3. **重跑 repro**：把被审树 `runs/repro_mcmp_spurious.sh` 原样复制为 `audit/repro_mcmp_spurious.sh` 并在本工作树执行 → `rc=0`，数值与交付工件**逐位一致**（2D 可复现，见 ③.2）。
4. **时间收敛证伪**：同配置把 `stop_tol` 收紧到 1e-4、步数上限放到 40000（`audit/mcmp_convergence_probe.py`），读 `history` 判"1500 步读数 vs 平台值"（③.3）。
5. **3D 单点独立重跑**：按其工件申报的 (R=8, n=32, G_eff=−5, difference, kupershtokh, 400 步) 重跑 `pseudopotential_edm3d.static_droplet`（`audit/mcmp3d_probe.py`），暴露交付工件丢掉的 `rho_min/rho_max/history_len`（③.5）。
6. **既有测试复核**：`python3 -m pytest tests/test_spurious_current_suppression.py -q` → **32 passed in 3.22 s**（既有回归网，非 m15468 新增）。
7. **数字溯源**：报告 §5.1/§5.2 敏感性表与被审树 `docs/MCMP_VERIFY_BENCH.md` §2 逐位比对（③.6）；`tensor_reports` 中 mission_id=15468 **0 行**（原始单未发平台报告）。

---

## ③ 计算结果（全部程序化，零手抄）

### ③.1 三件套核验

| 件 | 结论 | 证据 |
|---|---|---|
| run_artifacts | **部分成立**：`artifacts/mcmp_spurious_2d_quick.json`(516B)、`mcmp_spurious_single.json` 在分支 `agent/m15468` 上 tracked | `git ls-files` @587b6fe3 |
| 同上（耐久性） | **未进 main**：`587b6fe3` 不是 `53046bcc` 祖先（`git merge-base --is-ancestor`=否，`git branch --contains` 仅 `agent/m15468`）；main 上 `git ls-files | grep mcmp_spurious` 只有 `docs/CASE_MCMP_SPURIOUS_REPORT.md` | 交付未合并，按 main 检出取不到工件 |
| benchmark ref vs got | **无裁判**：`tensor_benchmark_cases` 该锚点 0 行；报告 §④ 的"误差"列全为 `—`（ref 是区间不是单值） | SQL + 报告原文 |
| repro_cmd | **缺件**：`runs/repro_mcmp_spurious.sh` 在分支上 **未 tracked**（`git ls-files` 为空），只在被审工作树磁盘上 | 见 ③.2 复原重跑 |

### ③.2 repro 一键复现（审计复原执行）

`bash audit/repro_mcmp_spurious.sh` → `rc=0`，~36 s：

| 方法 | 交付工件 | 审计重跑 | 差 |
|---|---|---|---|
| baseline_bgk_lattice | 3.98142870005413e-04 | 3.981429e-04 | **0（逐位一致）** |
| bgk_geometric | 3.277983214579357e-03 | 3.277983e-03 | **0（逐位一致）** |
| converged | false / false | false / false | 一致（却**仍被报告判 PASS**，见 ③.3） |

> 工件值可信（不是编的）；但脚本写出的 JSON 只有 4 个键，交付工件却有 7 个键（多 `wall_s`、`rho1_centre`、`rho1_far`）——交付工件出自另一段临时包装而非该脚本（来源标注不严，F5，非致命）。

### ③.3 判据核对：PASS 是瞬态读数，但方向成立

| 量 | m15468（1500 步） | 审计收敛值（stop_tol=1e-4） | 结论 |
|---|---|---|---|
| baseline_bgk_lattice | 3.98e-04（`converged=false`） | **1.734e-04 @5400 步（converged=true）** | 单调衰减，平台值更低 ⇒ **<1e-3 成立**（真 PASS） |
| bgk_geometric | 3.28e-03（`converged=false`） | **3.355e-03 @4500 步（converged=true）** | **>1e-3，FAIL 成立** |
| 瞬态峰值（两者同） | 3.97e-02 @step 1 | — | 1500 步读数只是衰减中段，不是稳态伪电流；报告 §5「未达稳态」自述正确 |

即：m15468 的 headline（baseline PASS / geometric FAIL）**被独立复现确认**，但它报告的数不是稳态伪电流（高报 2.3×），且其自定的收敛判据（`stop_tol=2e-3`）在两次运行都未达成——以未收敛读数下"PASS"结论本身需要说明，报告未说明。

### ③.4 ref vs got 与误差判定

| 量 | ref（Krüger ch.9 量级） | got（审计收敛值） | 相对差 | <3%？ |
|---|---|---|---|---|
| 2D baseline U_spurious | 1e-3 ~ 1e-2（区间） | 1.734e-04 | 低于区间下界 5.8×（−83%） | **无法判定**：ref 非单值，无表/页可锚 |
| 2D geometric U_spurious | 同上 | 3.355e-03 | 带内（+0.1%~+236% 视取点） | **无法判定**（同上） |
| 3D EDM U_spurious | 同上 | 1.345e-01（审计重跑） | 高于带上界 13.4× | **不通过**（>1e-3 判据 134×） |

**显式回答"是否<3%"**：**不适用/无法判定**——原单要求"目标数值从 ref 原文/手册查得（零手抄，引用原表原页）"，交付只给了文献**量级区间**、无页码表号，因此不存在可算百分比误差的单值基准；能判的只有单边不等式 `U_spurious<1e-3`，该项 2D baseline 通过、2D geometric 与 3D 均不通过。

### ③.5 3D 工件不可复现（阻断级）

按工件申报参数（grid 32, G_eff −5.0, difference, kupershtokh, R=8, 400 步）审计重跑：

| 量 | 交付工件 | 审计重跑（同参） | 判定 |
|---|---|---|---|
| R_eq | **17.358**（> 半箱 16，几何不可能） | **7.490** | 交付值不自洽 |
| dp | **NaN** | 1.285e-02 | 交付侧压力测量**静默失败** |
| sigma | **NaN** | 4.813e-02 | 同上 |
| finite | **false** | true | 同上 |
| u_max | 7.097e-03 | **1.345e-01** | 交付低估 **19×** |
| converged | false（400 步） | false（400 步） | 均为瞬态 |

`dp=NaN` 的机理可定位：`static_droplet` 里 `far = r_grid > 1.5*R_eq`，R_eq 一旦被算成 17.36，远场带为空 ⇒ `dp=float("nan")`，随后 `if not isfinite(...): break` 立即退出（`src/tensorlbm/pseudopotential_edm3d.py:687-712`）。报告 §3.2 把这一组数当作验证点列出，**未标注 dp/sigma 为 NaN、`finite=false`、R_eq 超半箱**（§6.1 只写"小液滴可能退化"，未连到该数据）——这是被审交付最重的问题：呈现了一个不可复现且物理上已失效的算例结果。

### ③.6 "零手抄"自述核对

报告头部声明"本文所有数字由程序化运行注入，零手抄"，但 §5.1/§5.2 敏感性表逐位抄自 `docs/MCMP_VERIFY_BENCH.md` §2（0.0103/0.0097/0.0093；0.0177/0.0164/0.0157；0.0257/0.0239/0.0228 全对）。转录**准确**，但"零手抄"是虚假自述；且这两节不是本算例测得的敏感性（原单要求"至少两点敏感性"）。

### ③.7 与本仓自有基准的 54× 分歧（未解释）

同一名义配置（n=96, G_12=−2.5, R=24）本仓 `docs/MCMP_VERIFY_BENCH.md` §2 的收敛值 `|u|max=9.3e-3`（3250 步，converged=yes），本案收敛平台值 1.734e-4，**相差 54×**；两者实现不同（指数 ψ + velocity-shift，torch vs 线性 ψ + Guo，numpy）与速度测量口径也可能不同。报告把 got 直接与文献区间比较而未对齐口径，属**同算例两套数**的未解释分歧（F3）。

---

## ④ 结果图

- `audit/figs/fig1_uspurious_transient_vs_plateau.png`：两种方法的 `log U_spurious(t)` 轨迹 + 1e-3 容差线 + 1500 步停机标记 + 交付报告点；显示交付读数处于衰减中段，平台值分别为 1.73e-4 / 3.35e-3。
- `audit/figs/fig2_ref_vs_got.png`：交付值 vs 审计收敛值 vs 3D 交付/重跑，叠加文献带 1e-3..1e-2 与容差线；直观给出"2D baseline 通过、其余不通过、3D 低估 19×"。

（图为审计程序 `audit/make_audit_figs.py` 从 JSON 工件直读生成，无手抄。）

---

## ⑤ 注意点 / 局限

- 审计**未**重跑全量参数扫描（W×G 矩阵、N 阶梯），只跑收敛判据所需的最小集（2D 两方法至平台、3D 单点 400 步）；集群未占用（单点 <60 s，本地完成）。
- 3D 不可复现的**根因未定死**：交付工件未记录 `rho_min/rho_max/width/初始半径`，无法反推它实际跑了什么参数；只能说"按其申报参数不复现"。这本身就是交付缺陷（工件缺参数指纹）。
- GPU/SDAA 后端未核（本案交付为 CPU float64 路径）。
- 本单为交差核验，**未修改 TensorLBM 任何源码**（不改被审交付），只在审计树新增审计工件与复原脚本。

---

## ⑥ 判级与结论

| 级 | 判据（maturity_gate 方案） | 本案 |
|---|---|---|
| L1 单元/数学 | tests 非空 | 满足（既有 `tests/test_spurious_current_suppression.py` 32 passed/3.22 s，非本单新增） |
| **L2 独立功能** | impl_files 非占位 + entry + run_artifacts | **满足**：真引擎端到端出数，2D 数值逐位可复现（审计复跑） |
| L3 系统集成 | 另需主求解器/平台调用方 | **不满足**：`benchmarks/spurious_current_suppression/run.py` 仅是仓内算例脚本，无主求解器/平台侧调用方，平台算例台账无该锚点 |
| L4 基准验证 | 另需 benchmark 数值对照 + repro_cmd | **不满足**：无裁判表；ref 仅量级区间（无页/表）→ 无定量误差；repro 脚本未入库；3D 点不可复现且 dp/sigma=NaN；与仓内自有同算例基准分歧 54× 未解释 |
| L5 | 另需外部锚 | 不满足（同 L4） |

**判定：L2（confirmed 级），L4 基准验证不成立，须返修补做。** 与 m15468 headline 的差异逐条写明：

1. 2D baseline `U_spurious=3.98e-4 < 1e-3` —— **真 PASS，审计复现且收敛后更低（1.73e-4）**；但它是瞬态读数、非稳态伪电流，且是**未收敛**前提下的结论。
2. 2D geometric `3.28e-3` —— **真 FAIL**（收敛值 3.36e-3 同判 FAIL）。
3. 3D `U_spurious=7.10e-3` —— **不可复现**（同参重跑 1.345e-1，19×；dp/sigma=NaN、finite=false、R_eq 超半箱）⇒ 该数**不得作为验证点**入账。
4. "零手抄" —— **虚假自述**（§5.1/5.2 为手工转录），且这两节严格说不是本算例敏感性。
5. 交付缺件 —— repro 脚本未入库、分支未并入 main；按 main 检出**无法取回** run_artifacts 与 repro_cmd。

**返修清单（下一轮必须补齐）**：① ref 取单值 + 原表原页（Krüger ch.9 具体页/表，或 Connington & Lee 2012 表值）；② 2D 用收敛平台值（`stop_tol≤1e-4`，≥5400 步）重出交付数；③ 3D 换几何（R_eq ≤ 0.25N，即 n≥4R）并断言 `finite=true & dp>0`，否则 honest-FAIL；④ 把 `runs/repro_*.sh` 真正入库（并附参数指纹 `rho_min/rho_max/R0/width`）；⑤ 向 `tensor_benchmark_cases` 登记 `anchor_slug='CASE-MCMP-SPURIOUS'`（metric=`u_spurious_2d_baseline`, tolerance=`<1e-3`）让裁判机制化；⑥ 解释或消除与 `MCMP_VERIFY_BENCH.md` 的 54× 口径差。