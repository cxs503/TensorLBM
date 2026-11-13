# 交差核验报告 · m18326 `XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD`（判级 L1）

> 核验单：m18359（board-audit，非重做）｜被核验单：m18326（status=succeeded, repo=TensorLBM, head_commit=null）
> 基准裁判：`tensor_benchmark_cases` 该锚点核验前 **0 行**（本单补登记 1 行 status=failed）；平台机判 `/tensor/bench/l4-verdict/18326` = **CLAIMED_NOT_PROVEN (rows=0)**
> 结论：**L1（数学/单元级数值验证）**；交付核心主张（与 palabos 交叉对标）不成立 → 建议 `reopened` 返修

## ① 案例介绍（物理背景 / 几何 / 参数）

被核验对象是两组分 Shan-Chen 伪势（MCMP）静止液滴的 **交叉对标研究**：TensorLBM 用线性伪势 ψ(ρ)=ρ、velocity-shift 力格式、BGK、τ=1.0、D2Q9 周期域；对标对象声称是 omalaspinas/palabos 的 ψ_exp(ρ)=ρ₀(1−e^(−ρ/ρ₀)), ρ₀=1 做法。测量协议：平板共存定 ρ_liq/ρ_gas → 以共存密度初始化 R∈{12,16,20,24} 液滴 → 采样中心/角点 4×4 箱压差 → Δp=σ/R 过原点拟合。

| 参数 | 交付值 | 出处 |
|---|---|---|
| 网格 n | 96（单值，无 n=128 分支） | refs/…/xval_driver.py:173 `N_MAIN=96` |
| τ | 1.0 | 同上 `TAU` |
| G12 (TL) | −2.5 / −2.75 / −3.0 | `G_LEVELS_TL` |
| 半径集 | 12,16,20,24 | `RADII` |
| 步数上限 | 12000（早停 stop_tol=1e-4） | `STEPS_DROPLET/COEXIST` |
| 判据容差 | σ 相对误差 <15%（owner 标准） | `TOL_PRIMARY=0.15`；本单登记入 `tensor_benchmark_cases` |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **取证**：读 m18326 的 result、`work/m18326` 工作树、`refs/` 快照（xval_driver.py sha256 `a0f974ca3e625a5f`…、xval_summary.json sha256 `e8de5eb5…`），psql 直查 `tensor_benchmark_cases` / `tensor_maturity_verdicts` / `tensor_reports`，并调平台 `/tensor/bench/l4-verdict/18326`、`/tensor/bench/results`、`/tensor/reports`。
2. **独立复跑**（不复用交付脚本，只调仓内同一 code path `simulate_static_droplet`+`measure_laplace`）：`artifacts/audit_m18359/repro_droplet.py`，3 个算例（G12=−2.5 R=16/20、G12=−3.0 R=16），12000 步上限、早停，4 线程 CPU。
3. **崩溃复现**：原样执行交付自述的唯一自检入口 `quick_smoke.py`。
4. **OSS 溯源核对**：在环境中唯一的 palabos 树 `/tmp/palabos_probe`（mtime 2026-11-09，来自盘点单 m8296）里核对被声明的参数组是否存在。
5. **副产品修复**：审计复现出的 `droplet_radius` 越界崩溃做最小修复 + 3 例 pytest。

## ③ 计算结果（数字表，程序化注入）

**表 A · 交付 ref vs got（`tensor_benchmark_cases` 容差 <15%）**

| G12 (TL) | σ_TL 交付 | σ_ref 声称 (Palabos) | 相对误差 | <15% |
|---|---|---|---|---|
| -2.5 | 0.1878 | 0.045 | 317.3% | ❌ |
| -2.75 | 0.2043 | 0.045 | 354.1% | ❌ |
| -3.0 | 0.2119 | 0.065 | 226.0% | ❌ |

**表 B · 审计独立复跑（本单 4 线程 CPU 实测）**

| G12 | R_init | R_meas | Δp | σ(单半径) | u_max | 步数 | 收敛 | 审计墙钟 |
|---|---|---|---|---|---|---|---|---|
| -2.5 | 16 | 14.991 | 0.012094 | 0.1813 | 9.06e-03 | 3500 | True | 22.4 s |
| -2.5 | 20 | 19.329 | 0.009485 | 0.1833 | 9.10e-03 | 3500 | True | 25.6 s |
| -3.0 | 16 | 15.500 | 0.013288 | 0.2060 | 2.31e-02 | 2500 | True | 19.4 s |

**表 C · 交付 vs 审计复跑（同一 code path，σ 一致性）**

| G12 | 交付 σ | 审计复跑 σ | 相对差 |
|---|---|---|---|
| -2.5 | 0.1878 | 0.1821 | -3.0% |
| -3.0 | 0.2119 | 0.2060 | -2.8% |

Δp·R 一致性（G12=−2.5 两半径）：0.1813, 0.1833 → 展布 1.1%

**表 D · 缺陷清单**

| # | 严重度 | 缺陷 |
|---|---|---|
| D1 | blocking | 外部 ref 无 provenance：Palabos 侧从未执行，数值为手打近似 |
| D2 | blocking | 对标对的是不同 EOS（线性 ψ vs ψ_exp）：>15% 由构造保证，非实现误差 |
| D3 | major | G 配对事后选择且重复计数（post-hoc matching） |
| D4 | major | 交付自述的唯一自检入口 quick_smoke.py 崩溃 |
| D5 | major | (repo 侧缺陷，审计副产品) droplet_radius 对未分离场无保护 → IndexError |
| D6 | major | 交付未提交：能力不在仓库里 |
| D7 | major | run_artifacts 只留聚合值，报告表格数字不可由工件重算 |
| D8 | major | 报告 ⑤ 5.1 分辨率敏感性声明无算例支撑 |
| D9 | major | 报告从未发布 + 锚点未登记基准 → 验收无门 |
| D10 | minor | 残留：交付未含 OSS 差异清单以外的移植取舍验证 |

## ④ 结果图

图 1（`xval_m18326_audit.png`）三联图：A = σ vs |G|（交付/审计复跑/声称的 palabos ref 三者同轴）；B = 交付自报相对误差 vs 15% 门槛；C = 审计复跑的 Δp–1/R 线性（Young-Laplace）。A 面板的实质读法：TL 三条曲线互相咬合（差 <3.2%），而 ref 那条红线是**另一套 EOS 的手打数表**，两者之差由模型差异而非实现差异主导。
（本单为交差核验单，物理量场云图不适用——审计对象是他人算例的数值链，故不给场量云图，理由如实记录。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **坑 1（D1 外部 ref 无 provenance，blocking）**：Palabos 侧从未 build/运行，`xval_driver.py:135-160` 是手打近似值（0.045/0.065/0.085；ρ 1.82/0.20、1.88/0.14、1.92/0.09）。声明出处「published Palabos examples + Shan-Chen 1994」被环境实证反证：唯一 palabos 树的 `examples/codesByTopic/shanChenMultiPhase/segregation2D.cpp:113,122,124` 用 **G=−120.0, rho0=200.0, psi0=4.0**（PsiShanChen94），与声明的 G=1.2/ρ₀=1 参数组无对应 → 对标语义（零手抄、从 OSS 源码/运行取值）未兑现。
- **坑 2（D2 跨 EOS 比较，blocking）**：线性 ψ 与 ψ_exp 的状态方程不同，>15% 由构造保证；「354% 不一致」既不能证伪也不能证实实现正确性——15% 判据在这个设计下恒不可达。
- **坑 3（D3 事后配对）**：docstring 预声明 G_Palabos≈1.2/1.4/1.6，代码却按「密度比最近」事后挑点，结果 −2.5 与 −2.75 双双对到同一 ref 点（σ_ref=0.045），两条误差不独立；且用密度比选点、用 σ 评分 = 循环论证。
- **坑 4（D4/D5 自检入口崩）**：`quick_smoke.py` 用 G12=−1.2（低于相分离阈值）→ 共存比 1.00，随后 `mcmp_verification.py:409` IndexError。本单已做最小修复（`truncated_crossing` 守卫）并加 `tests/test_mcmp_droplet_radius_guard.py`（3 passed）；修复后该场景不再抛异常而是带 `truncated_crossing`/`periodic_image_risk`/`anisotropic` guardrail 返回——但**亚阈值场仍会给出无意义半径（本单实测 29.0）**，调用方必须查 guardrail，这是既有语义局限，非本单修复范围。
- **坑 5（D7/D8 工件与报告）**：`xval_summary.json` 只有 3677 B 聚合值，12 个液滴算例的 R/Δp/R²/steps/converged 全部没落盘；报告 ⑤5.1「n∈{96,128} 分辨率研究、网格依赖性<2%」在任何代码/工件里都不存在（`generate_report.py:264` 硬写）。**结论：报告里 ≥ 一半的表格数字无法由工件重算（零手抄断链）**。
- **坑 6（D6/D9 交付形态）**：`head_commit=null`、交付文件全 `??`、main 检出（work/m18359）`git ls-files | grep SHAN-CHEN` = 0；`/tensor/reports?mission_id=18326 total=0`（报告从未发布）。原单 succeeded 来自 `server_quality_audit.gate_valid=false` 的门缺失默认放行。
- **局限**：未真跑 palabos（环境需 build，非本单范围）；未做 n=128 分辨率分支（交付不存在该分支）；全 driver 端到端（3 共存+12 液滴，估 15–25 min）未跑，改以 3 个同 code path 单算例取证。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|
| σ(G12=−2.5) = 0.1878 | 声称 Palabos 0.045 | **+317.3%** | ❌ | ❌ |
| σ(G12=−2.75) = 0.2043 | 声称 Palabos 0.045 | **+354.1%** | ❌ | ❌ |
| σ(G12=−3.0) = 0.2119 | 声称 Palabos 0.065 | **+226.0%** | ❌ | ❌ |
| 审计复跑 σ(G12=−2.5, R16/20 拟合) = 0.1821 | 交付 0.1878 | **-3.0%** | ❌（3.1%>3%） | ✅ |
| 审计复跑 σ(G12=−3.0, R16) = 0.2060 | 交付 0.2119 | **-2.8%** | ✅ | ✅ |
| Δp·R 自洽（G12=−2.5 两半径） | 恒定 | 1.1% | ✅ | ✅ |

**显式回答**：对外部基准（Palabos）**最差 354.1% ≫ 3%，也 ≫ 15%** → 若按 15% 判据，本锚点交付 **FAIL**（3/3 超容差）；但 ref 侧 provenance WITHHELD，故该 FAIL 只能记为「交付自报不一致」，不构成对 TensorLBM 实现的否定。
对**内部可复核链**：审计复跑与交付 σ 最大偏差 **3.0%**（>3% 但 <15%），Δp·R 自洽 1.1% → TL 侧数值链真实可信，**这正是 L1 授级的依据**。

## 判级与建议

- **L1 授级**：数学/单元级数值验证成立（Young-Laplace 线性与 σ 量级经独立复跑重现）。
- **L2 不予**：交付未新增功能实现（4 个脚本=既有 `src/tensorlbm/mcmp_verification.py` 的调用壳，`src/` 零改动）；锚点功能实质（与 palabos 交叉对标）未兑现（OSS 侧未执行 + 手打 ref + 跨 EOS），与 m18300 判例（lbmpy 臂未执行 OSS → L1）同类；交付未提交、自检入口崩、无测试。
- **L3 不予**：全仓 0 调用点、main 无痕迹。
- **L4 不予**：`tensor_benchmark_cases` 核验前 0 行、平台 `CLAIMED_NOT_PROVEN(rows=0)`、无逐例原始工件、repro 入口崩。
- **返修路线 R1–R5**：① 真跑 palabos 或钉死源 sha256+参数行给 ref；② TensorLBM 增 ψ_exp 选项后同模型同 G 比，或改判据为趋势一致并撤回 XVAL 主张；③ 配对准则跑前固定（一对一、按匹配共存密度比）；④ 逐例原始工件 + <60 s pytest 入 CI，修 quick_smoke 的 G12 与 `droplet_radius` 越界；⑤ 提交 + 发布报告 + 基准行改 status。

**verdict 落表**：`tensor_maturity_verdicts` id=742（实测，psql 复查） / repo=TensorLBM / scope=`m18326 XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD` / level=1 / verdict_by=board-audit；`tensor_benchmark_cases` 补登记该锚点 1 行（status=failed，ref_source 明标 WITHHELD）。

---

## 工件清单（本审计单 work/m18359）

| 工件 | 路径 |
|---|---|
| 判级证据 JSON | `artifacts/audit_m18359/verdict_m18326.json` |
| 落表 SQL（可重放） | `artifacts/audit_m18359/verdict_m18326.sql` |
| 独立复跑 3 算例 | `artifacts/audit_m18359/repro_G*.json`（脚本 `repro_droplet.py`） |
| 崩溃修复验证 | `artifacts/audit_m18359/verify_quickcheck_fixed.py` |
| 三联图 | `artifacts/audit_m18359/xval_m18326_audit.png`（`make_figure.py`） |
| 平台 HTTP 取证 | `artifacts/audit_m18359/probe_platform.py` |
| 被核验交付快照 | `docs/board_audit/m18359/refs/`（xval_driver.py sha256 a0f974ca…, xval_summary.json sha256 e8de5eb5…） |
| 缺陷修复 | `src/tensorlbm/mcmp_verification.py`（truncated_crossing 守卫）+ `tests/test_mcmp_droplet_radius_guard.py`（3 passed） |
| 平台报告 | `/tensor/reports` id=17196 |
