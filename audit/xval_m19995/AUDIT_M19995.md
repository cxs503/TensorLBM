# AUDIT — m19921 `XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS`（交差核验，非重做）

**审计单**：m19995（board-audit，repo TensorLBM，工作树 `/root/tp-agent/work/m19995`）
**被审对象**：mission 19921，`agent/m19921` @ `403c4818e`（工作树 `/root/tp-agent/work/m19921`），
基础 main `a80287b82`；round-2 交付在 `cc8c081fb`，round-3 独立复核在 `403c4818e`。
**审计产物**：本目录 + `artifacts/m19995/`（`recompute.{py,json}`、`make_figs.py`、
`fig_m19995_two_impl_contact_angle.png`、`runs_smoke/`）、`verdict_m19995.sql`。

---

## 0. 独立性与范围（先说限制）

| 我做过的检查 | 方式 |
|---|---|
| A 记账复核 | **自写代码**（`artifacts/m19995/recompute.py`，只读 `predeclared.json` 文本规则，不 import 被审模块）从 `summary.json` 重算 34 行的 `measurable/pass/estimators_agree`、每臂 tally、跨臂 XVAL 量 |
| B 端到端复算 | 在 main 基线 `a80287b82` 上跑 `run_xval.py one --theta 90 --grid baseline`（两臂），与 `summary.json` 逐字段比 |
| C 测试执行 | `pytest tests/test_two_impl_contact_angle.py -q`（main 上实有文件） |
| D 复现链可执行性 | 实跑 `REPORT.md` §"Reproduce" 的三条命令 |
| E 机器裁判 | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS'` |
| F 集成面 | `grep -rn two_impl_contact_angle src/ TensorLBM/`、`wetting/__init__.py` 导出面 |
| G 全量扫描复算 | 集群作业 `cj2428`（main，`sweep --workers 16` + `summarize` + 逐行 diff）：**交单时仍在排队（queued ~11 min），未取回结果 ⇒ 状态 not_run**；全量 34 行复现结论仅由 B 的两行点检支撑 |

**没有做**（状态 unknown / not_run，不计为通过）：
估计器偏差的独立重推（沿用 `verify_r3/AUDIT_R3.md` 的表作为**被审材料**，本单未重跑它的解析帽标定）；
`verify_r3/verify.py` 未在本单重跑；动态润湿（Cox–Voinov / 滞后）、LBM 碰撞-迁移耦合、
实验/文献定量对标——被审件本身声明不在范围内，本单亦未补；
`tensor_benchmark_cases` 中本锚 0 行，故**没有一条登记容差可以逐项对表**（见 F2）。

---

## 1. 三件套取证（run_artifacts / benchmark ref vs got / repro_cmd）

| 项 | 声称 | 核验结果 |
|---|---|---|
| `run_artifacts` | `REPORT.md` §6 点名 `runs/<tag>/result.json`、`runs/<tag>/psi.npz`、`figures/fig{1,2,3}_*.png` | **不存在**。`xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/` 在 `403c4818e` 只有 `REPORT.md/make_report.py/predeclared.json/run_xval.py/summary.json/verify_r3/`；`runs/` 被 `.gitignore:108` 忽略，`figures/` 空目录。实际留存的分析级真值只有 `summary.json`(34 行) + `verify_r3/verify_results.json`（round-3 复算 6 行） |
| `benchmark ref vs got` | ref = 解析 Young 律 `cos(theta)=(3psi_w-psi_w^3)/2`；gate = predeclared `accept_pct=15.0` | **无登记基准行**：本锚在 `tensor_benchmark_cases` 中 **0 行**（全表 121 行，`anchor_slug ILIKE '%CONTACT%'` 无匹配）。唯一裁判 = 交付自带 `predeclared.json` 的解析锚。审计自算：34 行全部 `pass/measurable/estimators_agree` 标志与交付一致（**mismatch = 0**），可测 25 行 / 过闸 25 行，可测行最大误差 9.35 %，跨臂最大差 6.48 %（<15 %） |
| `repro_cmd` | ①`sweep` ②`summarize` ③`make_report.py` ④`pytest tests/test_two_impl_contact_angle.py` | ①在本单以 `one` 代跑（`sweep` 全量见 `cj2428`）→ **可执行**，数值逐位复现 87.27/90.02；②在干净树 **`no runs`**（无输入可汇总）；③**崩**：`make_report.py:82 UnboundLocalError: cannot access local variable 'im'`（`runs/` 缺失时 `fig_fields()` 的空分支未初始化 `im`）；④**34 passed in 36.71 s**（main `a80287b82`） |

---

## 2. 结论分层（L0–L5）

- **L1 数学/单元验证**：`tests/test_two_impl_contact_angle.py` 34 项通过；含
  `test_young_law_from_first_principles`（1-D 墙自由能数值积分独立推出 Young 律）与
  `test_flux_and_geometric_ghosts_encode_the_same_angle`（两幽灵族场级等价 <0.03）。**成立**。
- **L2 独立功能实现**：`src/tensorlbm/wetting/two_impl_contact_angle.py`（486 行）真实现：
  `wall_ghost_rows`(:133-171) 三臂、`measure_contact_angle`(:282-350) 双估计器、
  `initialise_matched_cap`(:227) 体积配平；输入 `theta` → 弛豫 → 输出角度。**成立**。
- **L3 系统集成**：**不成立**。`grep -rn two_impl_contact_angle src/` 无命中，
  `wetting/__init__.py` 未导出该模块；唯一调用方是 `xval/.../run_xval.py` 自带的
  Allen–Cahn 弛豫驱动，**未接入 TensorLBM 的碰撞/迁移主求解器**（无流体、无耦合润湿）。
- **L4 基准验证**：**成立（解析锚口径）**。完整算例（3 网格 × 5 角 × ξ 敏感 34 行）、
  逐行误差分析（vs 解析 Young 律，15 % 闸）、可复现（本单 `one` 两臂逐位复现；全量 34 行扫描作业 `cj2428` 交单时仍 queued，未作为已确认证据）。
  扣分项：报告点名的 `runs/`/`figures/` 未入库、`summarize`/`make_report` 在干净树不可跑（F1），
  且**无登记基准行**（F2）——所以 L4 的依据是"解析锚 + predeclared 闸"，不是"实验/文献登记基准"。
- **L5 科研工程成果**：**不成立**。静态角、无动态润湿、无流体耦合、无实验对标；`omega=1` 是松弛不是时间积分（报告 §6.4 自认）。

**判级：L4**（L0–L2 亦成立；L3 主求解器集成缺、L5 缺）。

---

## 3. 发现（按严重度）

| id | 严重度 | 发现 | 证据 | 影响面 |
|---|---|---|---|---|
| F1 | **中（可复现）** | 报告 §6 点名的 `runs/`、`figures/` 不在树内（`runs/` 被 `.gitignore:108` 忽略）；交付的 3 步复现链 ②`summarize`→`no runs`、③`make_report.py`→`UnboundLocalError`（`make_report.py:82`） | 实跑记录；`git ls-files xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS`；本单 `artifacts/m19995/` | 交付件自洽性；物理结论不受损（`one` 两臂复算） |
| F2 | **中（证据链）** | 基准裁判表本锚 **0 行** ⇒ 没有登记容差可逐项对表；"benchmark PASS" 实为"解析律自洽 + 自定 15 % 闸" | `tensor_benchmark_cases` 121 行、`anchor_slug ILIKE '%CONTACT%'` 空、`tensorbench_results` 全表无本锚 | L4 的"基准"口径；建议补登记 case（ref=Young 律、metric=逐角相对误差、tolerance=15 %） |
| F3 | **中（信息量）** | **θ=90° 是结构退化点**：`neumann_flux` 的 `amp=0.75σcosθ/κ=0`（`two_impl_contact_angle.py:158`）与 `geometric_ghost` 的 `cotθ=0`（:168）都退化为同一零法向梯度墙 ⇒ 两臂场逐位相同，跨臂差 0.00 % 是**构造必然**，不含独立性信息。baseline/refined 各一格如此（占 7 格中 2 格） | 本单 `one` 两臂输出逐位相同（87.27/90.02/6000 步/+0.038）；代码行号；图 (c) | 跨臂表看起来"1/4 格完美"，真正有信息的是 70/110/135°（0.6–6.48 %） |
| F4 | 轻（披露） | predeclared `estimator_agreement_deg=10` 在 5 行被违（`dirichlet` 90° 两行 11.17°/12.87° 属**可测**行；`neumann_flux` 45°/ξ4-70° 三行属不可测行），而报告结论行写无限定 PASS | 本单 `recompute.json:tally.*.agreement_violations`；`REPORT.md` 勾选框未勾；`verify_r3/AUDIT_R3.md` F1 独立同结论 | 报告措辞；不改数值 |
| F5 | 轻 | `theta_cap`（面积/顶点反演）在解析帽上带 1.2–8.9 % 系统低偏，而它是报告的主量；`theta_circle` 无偏 <0.02 % | 采信 `verify_r3/AUDIT_R3.md` §B（本单**未重跑**其标定，状态 not_run） | 报告 θ 主列偏低 1–4°；判定方向不受影响 |
| F6 | 轻 | round-3 的独立仪器测试 `tests/test_two_impl_contact_angle_r3_estimators.py` 只存在于 `agent/m19921`，**main 上没有**（本单实跑报 file not found） | 本单 pytest 输出 | 回归保护未入库；main 的 34 项测试不受影响 |
| — | 非缺陷 | 45° 强润湿行 NOT_MEASURABLE、`dirichlet` 臂 <90° 失败（墙值层吸走液滴） | 报告 §6.1/§6.2；本单 recompute 同判 | 已诚实披露，属模型/分辨率限制 |

**夸大主张检查**：被审件未声称 L5、未声称动态润湿、未声称实验对标；主张
"两独立实现都在 15 % 内且互差 15 % 内" **在可测行上为真**，但 (i) 结论行抹掉了 F4 的限定，
(ii) 跨臂表的 90° 零差被当成正面证据（F3），(iii) "artefacts: runs/…" 一句在树内为假（F1）。

---

## 4. 处置（disposition）

**APPROVE WITH NOTES**（判 L4，落 `tensor_maturity_verdicts`）。
返修建议（不阻塞本轮结论）：
1. 把 `runs/*/result.json` 与 `figures/*.png` 纳入版本控制（或改写到 `artifacts/xval/...` 白名单路径），
   并修 `make_report.py:82` 的空输入分支；使 `REPORT.md` §Reproduce 三条命令在干净树上可跑。
2. 在 `tensor_benchmark_cases` 登记本锚（`ref_source`=解析 Young 律、`metric`=逐角相对误差、
   `tolerance`=15 %），使 gate 有机器裁判行。
3. 结论行改为 "PASS（5 项 predeclared 闸中 `estimator_agreement` 在 2 个可测 dirichlet 行被违）"；
   跨臂表标注 θ=90° 为退化格（两臂同为零梯度墙）。
4. 把 `verify_r3/` 的独立仪器与 `test_two_impl_contact_angle_r3_estimators.py` 合入 main。

---

## 5. 复现命令（审计侧）

```bash
cd /root/tp-agent/work/m19995
python3 artifacts/m19995/recompute.py            # A：记账重算 → recompute.json
python3 artifacts/m19995/make_figs.py            # 图
PYTHONPATH=$PWD/src python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py \
    one --arm neumann_flux --theta 90 --grid baseline     # B：8 s，复现 87.27/90.02
python3 -m pytest tests/test_two_impl_contact_angle.py -q # C：34 passed / 36.7 s
psql -U postgres -h 127.0.0.1 -d tensorplatform -c \
  "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases \
   WHERE anchor_slug='XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS';"   # E：0 行
```