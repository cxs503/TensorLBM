# m15891 交差核验：m15467 / CASE-MCMP-LAPLACE（TensorLBM 算例池）

审计对象：mission **m15467**（succeeded），交付提交 **e547c37c**（PR #1822，合并为 `cf959e0d`）
被审锚点：`CASE-MCMP-LAPLACE`（2D 两组分 Shan-Chen 静止液滴 · Young–Laplace 定律 Δp = σ/R）
审计单：mission **m15891**（board-audit，非重做）；审计分支 `agent/m15891`，基线 HEAD `53046bcc`。
结论：**L4（基准验证）** —— 完整算例 + 误差分析 + 可复现，三条容差项在预注册判据下全部通过（最大 2.54% < 5%），
但绝对表面张力 σ 无独立外部锚（Route B 作废），故不达 L5；并发现 4 项非阻塞缺陷（其中 2 项本轮已修）。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

两组分 Shan-Chen（MCMP）模型，线性状态方程 ψ(ρ)=ρ，BGK 碰撞，速度平移（velocity-shift）力格式，
D2Q9 周期方盒 n×n。静止液滴平衡时界面两侧压差服从 2D（圆柱界面）Young–Laplace 定律

$$\Delta p = \frac{\sigma}{R}\quad(\text{sphere 才是 } 2\sigma/R)$$

| 项 | 取值（`summary.json:protocol`，预注册） |
|---|---|
| 耦合常数 G₁₂ | −2.500 / −2.750 / −3.000（三点均在库护栏窗口 −2.3…−3.5 内） |
| 初始半径 | R = 12 / 16 / 20 / 24（n=96，R/n ≤ 0.25 < 库护栏 0.35） |
| 网格 | n = 96 主序列；分辨率研究 n = 96 vs 128（固定 R=16） |
| 步数 / 收敛判据 | 液滴 12000 步，\|u\|max 平台 stop_tol=1e-4；共存 12000 步 |
| 容差（算例池） | **<5%**（`tensor_case_pool` id=16 `MCMP-LAPLACE` tol_policy `<5%`） |
| 自设护栏（非容差） | R²≥0.99、自由线截距 \|b\|R_max/\|a\|≤5%、Route B flatness≤5%、库 guardrail |
| dtype / 设备 | float32 / CPU，1 thread/job |

数据规模：12 个液滴算例（3 耦合 × 4 半径）+ 3 个平板界面算例（Route B）+ 6 个分辨率算例，
报告声明全量 369 s @ jobs=6，环境 git `3508357d`，torch 2.7.1，python 3.11.6。

## ② 怎么做的（独立复算，不采信报告数字）

审计不重跑求解器，而是**从已提交工件独立重算每一个头条数字**，并做一次原始场量回代：

1. **逐项复算容差项**：读 `benchmarks/case_pool/mcmp_laplace/artifacts/summary.json` 的 12 个
   (G₁₂, R_i, Δp_i) 原始对，独立做「单未知量过原点最小二乘」σ₀ = Σx_i y_i / Σx_i²（x_i = 1/R_i，y_i = Δp_i，
   与 `src/tensorlbm/mcmp_verification.py:839 _fit_laplace_level` 的定义逐字一致），重算逐点误差与 max；
   同时独立拟合自由线 Δp = a/R + b 作诊断（截距占比 \|b\|R/|a|、R²、σ_free = a 与 σ₀ 的偏差）。
2. **原始场量回代**：从 `artifacts/hero_fields.npz`（ρ₁, ρ₂, u, G₁₂, R_init）经库函数
   `mcmp_verification.measure_laplace(run, dims=2)` 重算该液滴的 Δp / R / σ，与 summary 对应算例比对
   —— 验证「工件 ↔ 数字」链路不是事后填表。
3. **代码实质核查**：`run_case.py` 是否真驱动库求解器（import 链、是否有硬编码结果）：
   `benchmarks/case_pool/mcmp_laplace/run_case.py:58` import `tensorlbm.mcmp_verification`；
   `:48-72` 协议/阈值常量；`:148-200` 液滴/平板的 Pool 任务；`:400-445` 判定族逐项计算。
   全文件无任何 Δp/σ 字面量常量（grep 复核）。
4. **测试不变量核查**：`tests/benchmarks/test_mcmp_laplace_dims.py`（4 用例，断言 2D 因子 σ=Δp·R、
   3D 因子 0.5Δp·R、非法 dims 抛 ValueError、拟合与测量因子同源），本地实跑 **4 passed in 4.34 s**。
5. **可复现性**：本地实跑 `run_case.py --quick --jobs 8`（冒烟协议，37.3 s）确认执行链可跑通；
   用 `make_report.py` 从已提交 summary.json 重生成 README/docs/4 张 PNG，PNG 的 git blob
   与 e547c37c 历史工件**逐字节相同**（sha 一致）⇒ 报告渲染确定性成立。
6. **权威判据来源**：直查平台表 `tensor_benchmark_cases`（anchor_slug='CASE-MCMP-LAPLACE' → **0 行**）
   与 `tensor_case_pool`（id=16，tol_policy `<5%`，status=dispatched，mission_id=15467）。

工具链：`scripts/audit_m15891_mcmp_laplace.py`（本轮新增，纯 numpy/torch/post-hoc，不调求解器）
→ `artifacts/m15891_mcmp_audit/evidence.json`（全部重算数字）+ `figures/audit_m15891_mcmp_laplace.png`。

```
python scripts/audit_m15891_mcmp_laplace.py                  # 重算 + 出图（<5 s）
python -m pytest tests/benchmarks/test_mcmp_laplace_dims.py -q   # 4 passed
python benchmarks/case_pool/mcmp_laplace/run_case.py --quick --jobs 8 --out /tmp/mcmp_smoke  # 37.3 s
python benchmarks/case_pool/mcmp_laplace/make_report.py      # 确定性重渲染（PNG sha 与历史一致）
```

## ③ 计算结果（审计员独立重算，程序化注入，零手抄）

### 表 A｜容差项（预注册：σ_i = Δp_i·R_i 对过原点 σ₀ 的最大相对偏差 < 5%）

| G₁₂ | R 实测 | Δp 实测 | σ_i = Δp·R | 逐点误差 vs σ₀/R | σ₀（审计员） | σ₀（工件） | max 误差 | 判定 |
|---|---|---|---|---|---|---|---|---|
| −2.500 | 11.49 / 15.65 / 19.66 / 23.61 | 0.016059 / 0.012105 / 0.009722 / 0.008156 | 0.18456 / 0.18947 / 0.19114 / 0.19253 | 1.705 / 0.906 / 1.799 / **2.539 %** | 0.187765 | 0.187765 | 2.54 % | **PASS** |
| −2.750 | 11.60 / 15.68 / 19.67 / 23.61 | 0.017442 / 0.013067 / 0.010510 / 0.008807 | 0.20233 / 0.20494 / 0.20672 / 0.20793 | 0.986 / 0.291 / 1.160 / **1.752 %** | 0.204347 | 0.204347 | 1.75 % | **PASS** |
| −3.000 | 11.65 / 15.70 / 19.67 / 23.60 | 0.018011 / 0.013531 / 0.010885 / 0.009135 | 0.20991 / 0.21240 / 0.21409 / 0.21561 | 0.930 / 0.244 / 1.044 / **1.760 %** | 0.211879 | 0.211879 | 1.76 % | **PASS** |

审计员重算与工件**逐位吻合**（σ₀ 差 < 1e-15，max 误差 2.538977 % vs 2.538977 %）。
判据：`THRESHOLDS = {tolerance: 0.05}`（`run_case.py:72-78`，运行前固定，报告声明运行后未调整）。

### 表 B｜自由线诊断（预注册为「诊断项」，不作容差判定）

| G₁₂ | σ_free = a（斜率） | σ₀（过原点） | σ_free vs σ₀ | 截距 b | 截距占比 \|b\|R_max/\|a\| | R² | 判据 ≤5% |
|---|---|---|---|---|---|---|---|
| −2.500 | 0.176802 | 0.187765 | **−5.84 %** | +7.20e-4 | **9.61 %** | 0.999636 | ✗ |
| −2.750 | 0.196760 | 0.204347 | −3.71 % | +4.95e-4 | **5.94 %** | 0.999964 | ✗ |
| −3.000 | 0.204280 | 0.211879 | −3.59 % | +4.95e-4 | **5.72 %** | 0.999980 | ✗ |

`summary.json.verdict.guard_family = {r2_ge_0.99: true, intercept_frac_le_5pct: **false**,
routeB_flatness_le_5pct: false, routeB_converged: true}` —— 自设护栏族整体 False，工件未隐瞒。

### 表 C｜Route B（机械压力张量，探索性独立绝对 σ，自设有效性闸门 flatness ≤5%）

| G₁₂ | σ_mech | σ_Laplace | flatness | 收敛 | 跨路线偏差 | 状态 |
|---|---|---|---|---|---|---|
| −2.500 | −0.00944 | 0.18777 | 45.6 % | yes | −2088 % | VOID（自身闸门失败） |
| −2.750 | −0.01509 | 0.20435 | 56.3 % | yes | −1454 % | VOID |
| −3.000 | −0.02066 | 0.21188 | 64.1 % | yes | −1037 % | VOID |

Route B 时间收敛性已由交付方验证（5000→20000 步 σ_mech 与 flatness 从首个 checkpoint 起稳定），
故其作废**不是**未收敛，而是离散二点对称压力张量在「反向 gather + 速度平移」力格式下不满足静止界面法向应力守恒。
交付方的归因（本库总力在连续极限下是纯梯度力，无 Korteweg 应力；半格错位项缺失）与数据一致，属 honest negative。

### 表 D｜分辨率研究（固定物理半径 R=16）

| G₁₂ | n=96: R / Δp / σ | n=128: R / Δp / σ | σ 跨分辨率变化 |
|---|---|---|---|
| −2.500 | 15.652 / 0.012105 / 0.18947 | 15.519 / 0.012085 / 0.18755 | −1.01 % |
| −2.750 | 15.684 / 0.013067 / 0.20494 | 15.603 / 0.013038 / 0.20343 | −0.74 % |
| −3.000 | 15.696 / 0.013531 / 0.21240 | 15.643 / 0.013486 / 0.21097 | −0.67 % |

### 表 E｜原始场量回代（hero 液滴，`hero_fields.npz` → `measure_laplace(dims=2)`）

| 量 | 审计员（从 npz） | 工件 summary | 相对差 |
|---|---|---|---|
| Δp | 0.010509619 | 0.010509640 | −2.0e-6（float32 存储） |
| R | 19.669375 | 19.669375 | 0 |
| σ = Δp·R | 0.206717650 | 0.206718054 | −2.0e-6 |
| 密度比 ρ_liq/ρ_gas | 10.549 | 10.549（该算例） | 0 |

hero 算例实际是 **G₁₂ = −2.750, R_init = 20, n = 96**（npz 自带字段），不是文档所写的 −2.500。

### 图

![audit](../figures/audit_m15891_mcmp_laplace.png)

*图 1｜m15891 审计图。(a) 三个耦合层的 Δp vs 1/R 实测点与审计员独立拟合的过原点参考线 σ₀/R；
(b) 半径无关性诊断 (σ_i−σ₀)/σ₀ 与预注册 ±5% 门，图例给出自由线斜率偏差与截距占比（曲率趋势）。*

## ④ 报告六段齐全性（结构核查）

`docs/CASE-MCMP-LAPLACE.md` 与 `benchmarks/case_pool/mcmp_laplace/README.md`（两者字节相同）具备六段：
① 案例介绍（模型/协议/阈值）② 方法（库能力表 + 流程）③ 结果（表 A–D，程序化渲染）④ 收敛性/敏感性（4 组）
⑤ 失败模式与生态摩擦（Route B 作废、2D 因子缺陷、平板初值陷阱）⑥ 误差判定（显式回答）。
平台报告 `tensor_reports id=14544`（9436 字符，4 图内嵌，tags 含 TensorLBM/MCMP/Laplace）六要素齐备。

## ⑤ 注意点与发现（按严重度）

- **F1（medium｜判据缺口）** `tensor_benchmark_cases` 对 `CASE-MCMP-LAPLACE` **0 行**，「唯一裁判」表不存在；
  裁判回退到 `tensor_case_pool` id=16（tol_policy `<5%`，status=dispatched）+ 算例预注册 THRESHOLDS。
  建议登记 `(anchor_slug='CASE-MCMP-LAPLACE', metric='max_i |σ_i-σ0|/σ0', tolerance='0.05')` 及分辨率收敛项。
- **F2（medium｜工件丢失，本轮已修）** PR #1822 合并（`cf959e0d`）把 `benchmarks/case_pool/mcmp_laplace/figures/*.png`
  4 张图**滤掉**（提交 e547c37c 有，合并后 HEAD 无）⇒ main 上 README/docs 的 4 个图链接全是死链。
  本轮 `make_report.py` 从已提交 summary.json 重生成，4 张 PNG 的 git blob 与 e547c37c **逐字节一致**（无数据损失，纯合并工件丢失），已随本分支恢复。
- **F3（medium｜表述风险）** 标题「PASS 2.54% < 5%」只在**预注册的过原点单未知量判据**下成立；
  自由线诊断显示 σ 仍随曲率漂移（σ_free 比 σ₀ 小 3.59–5.84 %，截距占比 5.72–9.61 %，三层全部超自设 5% 护栏）。
  即：**σ 本身在 R=11.5–23.6 区间带 ~5 % 系统性趋势（Tolman 型）**，σ₀ 是区间平均拟合值。
  工件与文档 ⑤ 已如实披露（`guard_family_pass=false`），但只读标题的读者会低估该趋势。
  建议报告标题层并列两个数字。
- **F4（medium-low｜标注错误，本轮已修）** `make_report.py:348` 用 `res["cases"][0]["G_12"]`（第 1 个算例 = −2.5）
  标注 hero 场量工件，实际 hero 是 −2.75 / R_init=20 ⇒ README 与 docs 错标。已改为从 npz 读 G₁₂/R_init，
  重渲染后仅该行变化（diff 1 行 + 新增 17 行 helper），图 3 标题本来就用 npz 值（正确）。
- **F5（medium-low｜遗留旧因子）** `docs/MCMP_VERIFY_BENCH.md` 的绝对 σ 列仍是**旧 2D 因子**（σ=Δp·R/2）产物，
  绝对 σ 系统性偏小 2×；该文档仍被 `docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md` 引用。
  交付方 README 已注明「遗留」并给出重生成命令（`benchmarks/mcmp_verify_bench.py --report`，需重跑模拟）。
  本轮未执行（避免与控制面/集群算力争用；且属重做性质，超出交差核验范围）⇒ **仍为未修的生态风险**。
- **F6（low｜台账滞后）** `tensor_case_pool` id=16 仍 `dispatched`，未更新为 passed。
- **F7（info｜能力边界）** 绝对 σ **无独立外部锚**（Route B 作废）⇒ 本轮证明的是「Δp ∝ 1/R 的半径无关性
  + 2D 因子正确」，**不是**绝对表面张力量值的正确性。这是本算例止步 L4、不评 L5 的主因。
- **F8（info｜复现成本）** 全量协议 369 s > 控制面 120 s 硬超时（整进程组收割）；本轮已把全量重跑递交集群
  `cj1196`（stage_dir=审计分支），交差时仍在排队（前置 ~380 作业）⇒ 无结果，**不作结论**。
  quick 协议（37.3 s）本地跑通并复现同一结论（max 误差 2.50 % < 5 %）。
- **F9（info｜测试质量）** `tests/benchmarks/test_mcmp_laplace_dims.py` 断言的是**不变量**
  （2D 因子 σ=Δp·R；3D = 0.5Δp·R；dims=4 抛 ValueError；拟合与测量因子同源），不是「仅执行代码」；4 passed。
  但测试用合成 tanh 场（无求解器），属单元级锁，不覆盖端到端。
- **F10（info｜审计未做项）** `tests/benchmarks/test_mcmp_verify_bench.py`（库级 MCMP 基准）本地超 120 s 硬超时被杀，
  未取得结果；未重跑 12+3+6 全量算例（交差核验非重做）；未验证 GPU/SDAA 路径。

## ⑥ 误差判定（显式回答）

| 关键量 | 参考 | 最大误差 | 判据 | <3 % ? | <15 % ? |
|---|---|---|---|---|---|
| Δp（逐点，3 耦合 × 4 半径） | 预注册过原点 σ₀/R（Laplace 律单未知量拟合） | **2.54 %**（G=−2.5, R=23.6） | <5 %（算例池） | **是**（三层 max 2.54 / 1.75 / 1.76 %） | 是 |
| σ 半径无关性（同上） | σ₀ | 2.54 % | 5 % | 是 | 是 |
| σ 跨分辨率（n=96→128） | n=96 值 | 1.01 % | 无预注册 | 是 | 是 |
| σ（自由线斜率作参考） | σ₀ | **5.84 %**（G=−2.5） | 诊断项（非容差） | 否 | 是 |
| 绝对 σ | **无独立基准**（Route B 作废，flatness 46/56/64 %） | 不可评 | — | **不作声明** | **不作声明** |

结论：在算例自身预注册判据下 **PASS（2.54 % < 5 %，且 < 3 %）**；但「是否 < 3 %」的回答**依赖参考口径**：
过原点单未知量口径 2.54 %（< 3 %），自由线斜率口径 5.84 %（> 3 %）。两个数字均已在表 A/B 给出，不含糊。

## ⑦ 级别判定：L4（基准验证）

| 级 | 判据 | 本算例 |
|---|---|---|
| L0 文档 | 只有文字 | 否（有执行器 + 工件 + 测试） |
| L1 数学/单元验证 | 公式或单点单元校验 | 已超（4 个不变量测试只是其中一部分） |
| L2 独立功能实现 | 真代码 输入→计算→输出 | **满足**：`run_case.py` 驱动库 `simulate_static_droplet`/`measure_laplace`/`scmc_pressure_tensor`，12+3+6 算例真实计算（原始场量回代证明数字来自场量，相对差 2e-6） |
| L3 系统集成 | 接入主求解器/平台 | **部分**：调用库模块（`mcmp_verification`/`scmc_pressure_tensor`），但无平台路由/主求解器注册；平台侧仅有 `tensor_case_pool` id=16 派单记录 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | **满足**：3 耦合 × 4 半径 + 分辨率研究 + Route B 负结果；预注册判据全过（2.54 % < 5 %）；工件入库、quick 协议 37.3 s 复现、报告/图从 summary.json 确定性重渲染（PNG 逐字节可重现） |
| L5 科研工程成果 | 新增可发表级能力/方法 | **不满足**：绝对 σ 无独立锚（Route B 自设闸门失败），本算例是「既有库能力的标定 + 一个 2D 因子缺陷修复」，不构成新方法/新能力 |

**级别 = L4。** 附百页报告不改变级别；级别取自功能实质。两处降级风险已量化并如实入卷：
（a）绝对 σ 无独立外部基准（F7）；（b）自由线诊断显示曲率趋势 5.7–9.6 %（F3）。

## ⑧ 本轮改动（最小、可验证）

| 文件 | 改动 | 验证 |
|---|---|---|
| `benchmarks/case_pool/mcmp_laplace/figures/*.png`（4 个） | 从 `make_report.py` 确定性重生成，恢复合并丢失的工件 | 4 个 blob 与 e547c37c 逐字节一致 |
| `benchmarks/case_pool/mcmp_laplace/make_report.py` | hero 场量标注改从 npz 读取（F4），新增 `_hero_case()` | 重渲染后 README/docs 仅该行变化 |
| `benchmarks/case_pool/mcmp_laplace/README.md`、`docs/CASE-MCMP-LAPLACE.md` | 由 `make_report.py` 重渲染（1 行修正） | diff 1 行 |
| `scripts/audit_m15891_mcmp_laplace.py` | 新增：独立重算 + 原始场量回代 + 审计图 | `artifacts/m15891_mcmp_audit/evidence.json` |
| `artifacts/m15891_mcmp_audit/evidence.json`、`quick_smoke_summary.json` | 审计证据（重算数字 / quick 复现结果） | 本文件与 verdict evidence 的唯一数字源 |
| `docs/audits/m15891_CASE-MCMP-LAPLACE_audit.md` | 本审计报告 | — |