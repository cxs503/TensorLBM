# m18092 交差审计 — m17968 · XVAL-LBM-INSTRUCTION-DSL-TESTS-SH

审计者：board-audit（m18092，分支 agent/m18092）｜被审对象：mission 17968（repo TensorLBM，status=succeeded，claim tp-impl-5）
审计基准：`tensor_benchmark_cases`（anchor=XVAL-LBM-INSTRUCTION-DSL-TESTS-SH）→ **交付时 0 行**（见 F4），
故本轮以 ①交付物内**先于计算声明**的判据（docs §头部：scheme 语义差 ≤1e-12；物理锚点 <15%，更严 <3%）为裁判，
并以审计者独立重算为准；②独立重算数字补注册为基准行（F4 消解）。

## 结论先行

| 项 | 判定 |
|---|---|
| 三件套实存 | ✅ 真存在（源 1010 行 / 工件 5 个 / repro_cmd 可执行 rc=0） |
| 独立重算 vs 交付数字 | ✅ Poiseuille 0.6856%/0.3483%（交付 0.686/0.348，吻合）；pylbm 对表 M/w 差 0.0、f_eq 1.1e-16（交付 1.67e-16，吻合） |
| 报告六段 | ✅ docs ①–⑥ 齐 + 平台报告 id=16892 实存 |
| 判级 | **L4（基准验证级）** — 但 **L3 产线集成未达**（无主求解器/CLI 调用方） |
| 消账前置 | ⛔ **F3 阻塞**：全部交付物在 m17968 工作树内为 **untracked**（`git status` 6 项 `??`，head_commit=null），未进入任何 commit ⇒ 平台自动推送无从推起，主线无此能力 |

## ① 取证：三件套逐项

**源码（REAL，非 MOCK）** `/root/tp-agent/work/m17968/src/tensorlbm/instruction_dsl.py`（1010 行）
- `parse()`:328 行式文法 → `build_scheme()`:583（构造 M、M⁻¹、权重、τ/ν + 不变量自检）→ `run()`:887。
- `run()` 主体是**真矩空间碰撞**：`q = einsum(M,f)` → Strang 半步力（`q[1]+=0.5gx` 松弛前后各一次）→ `q += s(q_eq−q)` →
  `f = einsum(M⁻¹,q)` → `_stream_periodic`:854 / `_stream_channel`:862（x 周期 + y 半程反弹，solid 行清零）。
  守恒矩 s=0 不松弛。**无硬编码答案、无随机数、无解析解回填**。
- 表达式求值走 AST 白名单（`_compile_expression`:133 / `_evaluate`:155），指令文件不能执行任意代码。

**工件** `/root/tp-agent/work/m17968/artifacts/xval_instruction_dsl/`：`results.json`(7.2 KB)、`summary.md`、
`figures/poiseuille_taylor_green.png`(85,948 B)、`programs/*.lbm` 5 个（含 sha256 digest 记入 results.json）。

**repro_cmd** 实测：`python benchmarks/xval_instruction_dsl/run_xval.py` → rc=0，30 s，
`criteria = {scheme_moment_matrix:True, scheme_weights:True, scheme_equilibrium:True, physics_all_under_15pct:True, physics_all_under_3pct:True}`。
`python -m pytest tests/test_instruction_dsl.py -q` → **30 passed, 11.32 s**（与交付自述一致）。

> 披露：repro 脚本默认**原地重写** `artifacts/xval_instruction_dsl/results.json`，故该文件的 `generated_at/wall_time_s` 已被本次复跑刷新；
> 数值段与复跑前读取值逐位一致（TG 0.013244/0.103566；Poiseuille 0.438545/0.239513），无结果污染。

## ② 独立重算（不调用交付驱动的 KPI 函数）

脚本 `artifacts/board_audit/m18092/independent_recompute.py`：解析**公开发布的** `programs/*.lbm` 文本，
自行写解析抛物线 `u(y)=g(y−1/2)(ny−3/2−y)/(2ν)` 与自写 ln A 最小二乘，**不 import** `poiseuille_error` / `decay_rate_from_history`。

| 主张（docs §4） | 交付值 | 审计独立重算 | 结论 |
|---|---|---|---|
| C1 通道剖面最大相对误差 | 0.686% / 0.348% | **0.68559% / 0.34831%** | ✅ 复现（<3%） |
| C2 TG ν 相对误差 | 0.0132% / 0.1036% | **0.0809% / 0.1822%** | ⚠️ 量级不同（见 F2），仍 <3% |
| C3 pylbm 对表 M / w | 0.0 / 0.0 | **0.0 / 0.0**（自建 dico，权重另对教科书 4/9,1/9…） | ✅ 复现 |
| C3 f_eq max abs diff | 1.665e-16 | **1.110e-16**（8 组随机宏观态） | ✅ 复现（≤1e-12） |
| C4 「二阶收敛」 | ratio 1.97 ≈ 2.25 | ratio **1.968**，隐含阶 **p=1.670** | ❌ 表述过强（F1） |

壁面位置实测（审计自写抛物线求根）：ny22 → 0.512/20.488，ny32 → 0.508/30.492（理论 0.5/ny−1.5）⇒ H=ny−2 几何被判据钉住。

## ③ 判级 L0–L5

- **L1** 数学/单元验证：30 用例覆盖文法/不变量/错误分支/降阶语义/物理锚点/pylbm 对表 ⇒ 超过。
- **L2** 独立功能实现：1010 行真代码，文本输入 → 矩空间推进 → 场量输出，无 MOCK/STUB ⇒ **达成**。
- **L3** 系统集成：`grep -rln instruction_dsl` 全仓仅 `benchmarks/xval_instruction_dsl/*`、`tests/*`、
  `docs/*`、`src/tensorlbm/instruction_dsl.py` **四处自身**；`src/tensorlbm/__init__.py` 未导出，无主求解器/CLI/平台 API 调用方
  ⇒ **未达成**（同 m17869 F3 同型缺口）。
- **L4** 基准验证：2 类解析锚点 × 2 分辨率（H=20/30，N=32²/64²）+ 逐项误差表 + 与 OSS 参考实现算子层对表 +
  sha256 程序指纹 + 一键复跑（审计独立复跑并自写判据重算，Poiseuille 逐位吻合）⇒ **达成**。
- **L5** 科研工程成果：不宣称。

⇒ **level = 4**（基准验证级）；L3 产线集成缺口如实扣在说明里，不以 L4 掩盖。
（级别依 m17869 先例：验证子系统内的独立内核可到 L4，但须显式声明「无产线接入」。）

## ④ 严重度排序的发现（含复现）

**F3 · 阻塞（发货）· 证据确凿**：交付物全为 untracked。`git -C /root/tp-agent/work/m17968 status --porcelain` 输出
6 行 `??`（src/…instruction_dsl.py、tests/…、benchmarks/…、artifacts/…、docs/…、tools/xval_pylbm_stub/）；
`git ls-files --error-unmatch src/tensorlbm/instruction_dsl.py` → `did not match any file(s) known to git`；
mission result 内 `head_commit: null`、`branch: null`。⇒ 能力只存在于一块可回收工作树里，main 与生态检出均无此模块。
修复：把上述 6 项 commit 到 agent/m17968 并入 main（`tools/xval_pylbm_stub` 建议随仓保留并在 WORKTREE 文档注明仅用于离线 import）。

**F1 · 非阻塞（表述过强）**：docs §5「比值 1.97 ≈ (30/20)²=2.25 ⇒ 半程反弹壁面二阶收敛」。
审计重算 ratio=1.9683，若为二阶应 2.25；隐含阶 p=ln(1.968)/ln(1.5)=**1.670**，与 2 差 12.5%。
判据本身（<3%）不受影响，但「二阶收敛」应改述为「两分辨率均 <3%，观测阶约 1.67，二阶尚需第三分辨率（如 H=45）确认」。

**F2 · 非阻塞（可复现性缺口）**：C2 的 TG ν 误差对拟合采样/窗口敏感。交付 `run_case`:249 取 `sample_every=steps//200=2`，
审计取 sample_every=1（400 点全采）→ 0.0132% 变 0.0809%（6×），64² 由 0.1036% 变 0.1822%（1.8×）。
两值都远低于 3% 阈值，PASS 不变；但 results.json 只存 `fit_r2` 不存拟合窗口，导致三位有效数字的 0.0132% **不可独立复现**。
修复：把 `sample_every` 与拟合区间写进 results.json / docs 参数表。

**F4 · 治理缺口（本审计已消解）**：交付时 `tensor_benchmark_cases` 本 anchor 0 行，`server_quality_audit.gate_valid=false`、
`gate_missing_warning=true` ⇒ 机器唯一裁判缺席。审计按先例（id=47/55/64/81）以独立重算数字补注册 3 行，status=passed。

**F5 · 范围限定**：`tools/xval_pylbm_stub/mpi4py` 是单进程 no-op 垫片，仅用于让 pylbm 离线 import，**不是 MPI 实现**；
文档与 results.json `reference.offline_shim` 均已如实声明 ⇒ 不构成造假，但引用「与 OSS 对表」时须保留「单进程、仅算子层（M/权重/f_eq），非整解场 parity」的限定。

**F6 · 观察（未改被审代码）**：docs 指出仓内既有 `bitflag_d2q9.poiseuille_profile` 用 H=ny−1 与其自身 `make_channel` 几何不一致；
本单未修、仅记录 —— 属既有模块缺陷线索，建议另立节点。

## ⑤ 未跑的检查（不计入通过）

- pylbm 与 DSL 的**整解场** parity（只比了算子层 M/权重/f_eq；pylbm 本机未跑流体步）。
- 第三分辨率（H=45 / N=128²）收敛阶验证（F1 的修复证据尚未产出）。
- 集群 loong64 复跑（本单 30 s 级，未占集群资源）。
- 与 TCLB 参考实现的二进制 parity（无需，本 anchor 非 TCLB 主题）。

## ⑥ 处置

**approve with notes — L4（基准验证级），但 F3 未消解前不得标记为「已发货」**。
`maturity_gate` 若判 confirmed，语义应为「基准与能力成立」；发货资格另需 F3（commit/合入）与 F1/F2（表述与可复现性）收口。