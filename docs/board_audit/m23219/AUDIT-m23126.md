# 交差核验：m23126 — XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD（TensorLBM）

审计单：m23219 ｜ 被审单：m23126（status=succeeded，PR #2608 merged，commit `93cb61632`）
审计者：board-audit ｜ 判级：**L2** ｜ 交差结论：**锚点交差不成立**（基准裁判 failed + 外部锚 WITHHELD）

---

## 0. 一句话结论

模块本身是真的（530 行独立实现、12 个有牙齿的测试、exact/acoustic 两阶段被本审计**逐位复现**），
够 L2；但「锚点 PASS」这条交差结论**站不住**：唯一裁判 `tensor_benchmark_cases` 该 anchor 行仍
`status=failed`，登记的 3 档耦合（G=1.2/1.4/1.6）在交付里**一个数都没有**，三个 PASS 的参考分别是
自推导解析式（声速）与自洽律 Δp∝1/R（Laplace），不含外部锚。

| 项 | 交付自报 | 审计核验 | 结论 |
|---|---|---|---|
| run_artifacts（数值） | results.json + 6 chunk + 4 图 | 7 json 在 main；**4 个 png 在 main 上不存在**（merge #2608 门禁滤除）；平台报告 4 图 `png_base64`=null | 部分成立 |
| run_artifacts（数值可复现） | — | `--only exact` 12/12 字段、`--only acoustic` 4 臂，**与随货工件逐位一致** | 成立 |
| benchmark ref vs got | 「锚点 PASS」 | 登记 ref 0.045/0.065/0.085（G=1.2/1.4/1.6，psi_exp）；交付 σ 只在 G=2.0、psi=ρ（fit 0.0899，逐半径 0.0957/0.0980/0.0998）→ 匹配条件下**不可计算**；最近邻错配 | **不成立** |
| repro_cmd 可执行 | 5 条命令 | exact/acoustic 实跑 OK；laplace 受本机 120 s 上限，已递交 cj4697；**文档 steps 与随货 steps 不一致** | 部分成立 |
| 集成调用者 | （未声明 L3） | `grep -rl mcmp_psi TensorLBM` → 只有模块自身/tests/xval/docs，主求解器零调用 | L3 不可达 |
| 外部锚 | 「Palabos 做法」 | 无 palabos 构建/运行/日志/SHA；自认未逐行对照；交付仅 D2Q9 | **WITHHELD** |

---

## 1. 判级（L0–L5）

**判 L2 —— 独立功能实现。**

- **L1 达标**：机精度恒等式（均匀场 |F|=0、自作用排除、Σ(F_A+F_B)≈1e-15、gather≡scatter 1.4e-17、
  力对自身 ψ 线性 1.1e-16），且有 12 个 pytest 判据钉住（`tests/test_mcmp_psi_field.py:31/40/53/61/72/80/133/143`）。
- **L2 达标**：真代码全链 `psi_of → neighbor_vec_sum → mcmp_psi_force → collide_edm → stream → measure_drop`，
  驱动 `xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/run_xval.py:373 main()` 输入→计算→输出，审计逐位复现。
- **L3 不达标**：`src/tensorlbm/multiphase/__init__.py` 无 `from .mcmp_psi import`，全仓无 verification/
  与平台 API 调用者；与生产核 `sc_two_component_force`（`multiphase/__init__.py:494`）内部差 1.4e-9 是
  **等价性对照**，不是集成。
- **L4 不达标**：基准裁判 failed（详见 §3 F1）；σ 绝对值零外部比较；Δp∝1/R 只是自洽律；消融臂 not_run。
- **L5 不达标**：无科研级新结论/新方法。

> 百页报告不改变级别：`docs/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD.md` 六段齐全、诚实披露也不错，
> 但物理量没有外部裁判，级别由功能实质决定。

---

## 2. 本审计实跑（可重放）

```bash
# 1) 测试（m23126 工作树）
cd /srv/tensor/agent/work/m23126/TensorLBM && python -m pytest tests/test_mcmp_psi_field.py -q
#   -> 12 passed, 1 warning in 8.43s

# 2) 复现 exact（写进本单 .audit/m23219/repro，避免污染被审工作树）
cd /srv/tensor/agent/work/m23219/.audit/m23219/repro/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
PYTHONPATH=/srv/tensor/agent/work/m23219/TensorLBM/src python run_xval.py --only exact
PYTHONPATH=/srv/tensor/agent/work/m23219/TensorLBM/src python run_xval.py --only acoustic
#   -> 12/12 字段与 4 臂声速和随货 chunk 逐位一致

# 3) Laplace 整轮超本机 120 s 上限 → 递交集群
#   cj4697: run_xval.py --only laplace_primary --radius 6 --steps 4000；再 R=10 复跑 NaN 臂

# 4) 落表
python3 audit/m23219/insert_verdict_m23126.py
```

---

## 3. 发现（按严重度）

### F1（阻断 L4）「锚点 PASS」与基准裁判冲突
- 裁判行（唯一）：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD'`，
  `status=failed`，容差 `<15%`（引 `xval_driver.py:180 TOL_PRIMARY=0.15`），登记 worst 354.1%。
- 登记参考：G=1.2→0.045 / 1.4→0.065 / 1.6→0.085（psi_exp，手抄 dict，**provenance WITHHELD**）。
- 交付 got：G=2.0、psi=ρ 下 fit σ=0.08991、逐半径 0.09575/0.09796/0.09979 →
  对三档 ref 的误差 **+99.8% / +38.3% / +5.8%**（per-radius 最大 +121.8%/+53.5%/+17.4%）。
  这三级比较**全部是跨 G、跨 EOS** 的措配比较，G=1.6 那栏 +5.8% 不能计 PASS；匹配条件下无数。
- 结论：交差需要的「基准验证」缺席，且登记行仍 failed。返修须先落实 palabos 真锚，或把登记基准
  改为可执行的外部解/解析极限并同步容差。

### F2（中）Young-Laplace 一栏以自洽律冒充参考
- `REPORT.md` 表中该行 reference 写 `Δp ∝ 1/R`，结果 σ=0.0899、R²=0.9994 → PASS。
- 但拟合截距 **+8.15e-4**（相对 Δp≈0.0079–0.0128 是 6–10% 量级），逐半径 σ 单调漂移 4.23%，
  fit 与逐半径均值差 **−8.10%**（交付 §⑥ 自己也写了这个数，却仍判 PASS）。
- 3 点拟合 + 非零截距 ⇒ 只能说明 1/R 标度近似成立，不能证明 σ 的量级正确。

### F3（低）图在仓库态缺失、平台报告未内嵌图
- `git ls-tree HEAD -- TensorLBM/xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/` → 7 json + REPORT.md + run_xval.py，
  **0 个 png**（m23126 commit `93cb61632` 有 4 个；merge #2608 门禁滤除）。
- `tensor_reports id=20779`：figures 4 项，`png_base64` 全 null。

### F4（中）repro_cmd 与随货工件 steps 不一致
- 随货 chunk：R6 `steps=6000`（history 到 step 6000）、R8/R10/R12 `steps=3500`；
  文档复现段写 `--radius 6,8 --steps 4000` / `--radius 10,12 --steps 3500`。
- `results.json` 的 `laplace.steps=3500` 把混设扫描压成单一值，进一步掩盖差异。
- 后果：照文档跑**不能**重放 R6/R8 随货数值；exact/acoustic 两阶段例外（已逐位复现）。

### F5（低）恒真断言
- `tests/test_mcmp_psi_field.py:108`：`assert abs(fa + fb) < 0.05 * abs(fa) or True` —— 恒真。
  带内动量抵消无判据（全局守恒由 `:53` 覆盖，故非阻断）。

### F6（低）稳定性臂机理缺失
- R_init=10 在 3500–4000 步内 NaN，而 R=6/8/12 稳定（`chunk_laplace_primary_R10.json: nan=true, elapsed_s=12.97`）。
  交付只记「honest FAIL」，未给机理 → 适用边界不可外推。审计递交 cj4697 复跑该臂。

### F7（低）维度收窄未在结论处声明
- 锚点是 palabos 3D `shanChenMultiComponentProcessor3D`，交付只有 D2Q9；模块 docstring 与 docs §⑤(4)
  自认未逐行对照，但 `REPORT.md` 的 verdict 表未标注维度差。

---

## 4. 未做/未判定（记 not_run，不算 pass）

| 检查 | 状态 | 原因 |
|---|---|---|
| Laplace 整轮独立重跑 | **not_run**（递交 cj4697，截稿 queued） | 本机 120 s shell 上限；R6 6000 步随货 80.9 s，四臂 >180 s |
| Palabos 侧真锚 | **not_run** | 环境无 palabos 构建/输入卡/日志/SHA；无外网 |
| 消融臂 common_velocity / self_coupling | **not_run** | 交付自记 not_run，本审计未补 |
| 3D（D3Q19/27）语义 | **not_run** | 交付未实现，无从核验 |

## 5. 独立性限制

同环境、同仓、无外网：ref 侧只能沿用 m18359 的 WITHHELD 判定，不能自建 palabos 基线；
Laplace 的独立复现受本机 120 s 限制，故该口径记 not_run 而非 passed（exact/acoustic 才是逐位确认）。

## 6. 处置

**approve-with-notes-for-L2（模块成立）/ request-changes-for-交差（锚点不成立）。**
即：L2 判级确认，m23126 的「psi 场 MCMP 模块」作为能力入账；但该锚点的 XVAL 交差**不予消账**，
返修清单 = F1 落实外部锚（或改登记基准并同步容差）、F3 把 4 张 png 落进 main 并内嵌平台报告、
F4 使 repro_cmd 与随货 steps 一致、F2 把 Laplace 行的 reference 从「Δp∝1/R」改成「无外部 σ 锚」并按实判级。

## 7. 附图（本审计生成，数据全部程序化读取）

- `figs/fig1_laplace_selfconsistency.png` — Δp–1/R 两张图：随货 3 点、带截距拟合（σ=0.0899, b=8.15e-4）
  与过原点口径（σ=0.0978，差 −8.1%）；右图逐半径 σ 漂移。
- `figs/fig2_acoustic_parity.png` — 4 臂 c_meas vs 解析 c_pred，±3%/±15% 带，叠加审计复现点（逐位一致）。
- `figs/fig3_benchmark_judge.png` — 基准裁判判据：登记 3 档 ref vs 交付 σ（fit 与逐半径带），
  标注「条件不匹配 + ref provenance WITHHELD + 容差 15%」。
---

## 8. 落表与裁定（机器复核）

- verdict 落表：`tensor_maturity_verdicts` id=**923**，`repo=TensorLBM`，
  `scope='m23126 XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD'`，`mission_id=23219`，`level=2`，`verdict_by='board-audit'`
  （evidence 见 `audit/m23219/verdict_evidence_m23126.json`；落表脚本 `audit/m23219/insert_verdict_m23126.py`，幂等）。
- `maturity_gate.judge()` 预览（与自动裁定同一实现）：
  `judge(declared=2) -> (2, [])`；`judge(declared=4) -> (2, ['L4 缺: integration_callers(grep 无引用)', 'L3 缺: integration_callers(grep 无引用)'])`；
  `link_original -> 23126`。即机器复核与我方判级一致：**L2 通过、L3/L4 因零集成调用被降级**。
  按 gate 规则（gate_level≥2 → confirmed）L2 会被消账于「独立功能实现」这一层；
  **锚点 cross-validation 这一层不予消账**：`tensor_benchmark_cases` 该行仍 `failed`，外部锚 WITHHELD。
- 平台报告：`tensor_reports` id=**20849**（六段 + 3 图 base64 内嵌），由 `audit/m23219/publish_report.py` 程序化发布。
- 集群作业 `cj4697`（Laplace R=6 / R=10 独立复跑，4 核，TensorLBM@main）：截稿时 **queued 12 min**（队列 133 在飞），
  记为 **not_run**，不放行为 passed；结果出来后可单独核 R6 σ=0.09575 与 R10 NaN 的可复现性。

## 9. 审计产物清单

| 文件 | 内容 |
|---|---|
| `docs/board_audit/m23219/AUDIT-m23126.md` | 本报告（判级、7 条发现、未做项、处置） |
| `audit/m23219/verdict_evidence_m23126.json` | gate 契约 evidence（keys=impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor + findings） |
| `audit/m23219/insert_verdict_m23126.py` | 落表脚本（幂等） |
| `audit/m23219/make_figs.py` | 三张审计图的生成器（读随货工件 + DB 基准行，零手抄） |
| `audit/m23219/publish_report.py` | 平台报告发布脚本（图 base64 内嵌） |
| `docs/board_audit/m23219/figs/*.png` | fig1 自洽性 / fig2 声速 parity / fig3 基准裁判 |
