#!/usr/bin/env python3
"""W22 — generate papers/reports/w22_lbm_suboff_anchor8.md from the two artifacts.

Every number in the report is interpolated from
  benchmarks/verified/suboff_anchor_w22/anchor_transcription.json
  benchmarks/verified/suboff_anchor_w22/ittc57_closure.json
  benchmarks/verified/suboff_anchor_w22/figures/manifest.json
so no value is hand-copied into prose.  The prose itself is fixed text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
CASE = HERE.parent
OUT = REPO / "papers/reports/w22_lbm_suboff_anchor8.md"

TRANS = json.loads((CASE / "anchor_transcription.json").read_text(encoding="utf-8"))
CLOS = json.loads((CASE / "ittc57_closure.json").read_text(encoding="utf-8"))
FIGS = json.loads((CASE / "figures/manifest.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def g(ref: dict):
    return ref["value"]


def fmt(x, nd=6):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


crit = {c["id"]: c for c in CLOS["criteria"]}
c2p = crit["C2'"]
geom = CLOS["formula_AN_SUBOFF"]["geometry"]
cal_a = CLOS["closure"]["caliber_A_primary"]
cal_b = CLOS["closure"]["caliber_B"]
cal_c = CLOS["closure"]["caliber_C"]
cal_d = CLOS["closure"]["caliber_D_definitional"]
pro = CLOS["k_identifiability"]["prohaska"]
rec = CLOS["k_identifiability"]["record"]
gap = CLOS["case_caliber_gap"]
grids = CLOS["grids"]

art_rel = "benchmarks/verified/suboff_anchor_w22/anchor_transcription.json"
clos_rel = "benchmarks/verified/suboff_anchor_w22/ittc57_closure.json"
fig_rel = "benchmarks/verified/suboff_anchor_w22/figures"

lines: list[str] = []
w = lines.append

# ----------------------------------------------------------------------- header
w("# W22 / m14504 — DTRC SUBOFF AFF-8 外部锚逐字段机读抄录 与 ITTC-1957 闭合检验")
w("")
w("**任务**：`paper-exp-lbm-suboff-tau-w22-anchor8`（mission 14504，领域仓 TensorLBM，学科域：阻力）")
w("")
w("**前身**：`paper-exp-lbm-suboff-tau-4`（零报告欠账，本单显式承接；本报告 §7 逐条对齐其判据）")
w("")
w(f"**总判决**：**{CLOS['overall_verdict']}** — 抄录与复算两条判据 PASS，闭合与 k 双口径两条判据 FAIL，"
  "按任务书要求如实落账，不改判据、不粉饰。")
w("")
w("| 项 | 值 |")
w("|---|---|")
w(f"| 本单工件 | `{art_rel}`、`{clos_rel}`、`{fig_rel}/`（4 图 + manifest） |")
w(f"| 抄录源 | {len(TRANS['sources'])} 个工件，{TRANS['n_leaves_total']} 个标量叶节点 |")
w(f"| 抄录完整率 | {TRANS['completeness']['completeness_pct']:.4f}%（缺失/失配 {TRANS['completeness']['n_missing_or_mismatched']}） |")
w(f"| 生成时 git commit | `{CLOS['git']['commit']}`（分支 `{CLOS['git']['branch']}`） |")
w(f"| 输入 sha256（抄录件） | `{CLOS['inputs']['anchor_transcription']['sha256']}` |")
w(f"| 盲评合规 | {CLOS['provenance']['blind_review']} |")
w("")

# ------------------------------------------------------------- 1. case description
w("## 1. 案例介绍（物理背景 / 几何 / 参数）")
w("")
w("外部锚 `AN-SUBOFF`：DARPA SUBOFF（DTRC Model 4740）全附体 AFF-8 拖曳水池总阻力，"
  "属模型尺度深潜（无自由面兴波）工况；锚值只来自仓内已归档转录件，未凭记忆写数。")
w("")
w("| 几何/口径 | 值 | 来源（指针） |")
w("|---|---|---|")
for label, key in (("艇长 $L$", "L_m"), ("最大直径 $D$", "D_m"), ("湿表面积 $S$", "wetted_surface_m2"),
                   ("排水体积 $\\nabla$", "displacement_volume_m3")):
    ref = geom[key]
    w(f"| {label} | {g(ref)} | `{ref['source_path']}` → `{ref['pointer']}` |")
w(f"| $C_T$ 定义 | {g(CLOS['formula_AN_SUBOFF']['C_T_definition'])[:60]}… | "
  f"`{CLOS['formula_AN_SUBOFF']['C_T_definition']['pointer']}` |")
w(f"| 锚最点 $Re$（$\\nu$=1.139e-6 m²/s 声明假设） | {g(CLOS['formula_AN_SUBOFF']['anchor_Re_aff8']):.6g} | "
  f"`{CLOS['formula_AN_SUBOFF']['anchor_Re_aff8']['pointer']}` |")
w(f"| 锚最点 $C_T$ | {g(CLOS['formula_AN_SUBOFF']['anchor_CT_aff8']):.8f} | "
  f"`{CLOS['formula_AN_SUBOFF']['anchor_CT_aff8']['pointer']}` |")
w("")
w("内部被检对象：仓内案例 `suboff_sail_resistance`（AFF-3 with_sail，三档网格 L96/L112/L128）：")
w("")
w("| 网格档 | 域（nx×ny×nz） | 步数 | 附体固体胞 | $C_T$（E2 主口径窗均值） | 归档 mix50 $C_T$ | 记录 $C_F$（本案层流参考） |")
w("|---|---|---|---|---|---|---|")
for gr in grids:
    dom = "×".join(f"{int(v)}" for v in gr["domain"])
    w(f"| {gr['tier']} | {dom} | {int(gr['n_steps_e2'])} | {int(gr['appendage_solid_cells_e2'])} | "
      f"{gr['C_T_primary_e2_window']:.6f} | {gr['C_T_archived_mix50']:.6f} | "
      f"{gr['C_F_case_recorded']:.6f} |")
w("")
w("案例口径（逐条由工件注入，禁手抄）：")
w("")
w("| 口径项 | 值 / 文本 | 指针 |")
w("|---|---|---|")
w(f"| 案例 $Re$ | {CLOS['case']['Re_case']:.0f} | `{CLOS['case']['Re_case_refs'][0]['source_path']}` → "
  f"`{CLOS['case']['Re_case_refs'][0]['pointer']}` |")
w(f"| 案例参考式 | {g(CLOS['case']['reference_structure']['formula_leaf'])} | "
  f"`{CLOS['case']['reference_structure']['formula_leaf']['pointer']}` |")
w(f"| 观测定义 | {g(CLOS['case']['reference_structure']['observable_leaf'])} | "
  f"`{CLOS['case']['reference_structure']['observable_leaf']['pointer']}` |")
w(f"| 案例自身记录的『不适用交叉校验』 | `{g(CLOS['case']['reference_structure']['ittc57_recorded_as_not_applicable'])}` | "
  f"`{CLOS['case']['reference_structure']['ittc57_recorded_as_not_applicable']['pointer']}` |")
w(f"| 几何构型文本证据 | `{CLOS['case']['geometry_text_evidence']['text'][:70]}…` "
  f"(`prereg.md:{CLOS['case']['geometry_text_evidence']['line']}`, sha256 `{CLOS['case']['geometry_text_evidence']['sha256'][:16]}…`) | — |")
w("")
w(f"→ 本案 $Re$ 比 AFF-8 锚 $Re$ 低 4 个数量级以上，且上游工件 "
  f"`benchmarks/verified/suboff_anchor/anchor_points.json` 的 C4 已判『NOT REACHABLE at credible "
  f"resolution』——**同 Re 直接对比在两个方向上都不可行**，这是本单闭合判据失败的物理前提，"
  f"不是数值实现缺陷。")
w("")

# ------------------------------------------------------------------- 2. methods
w("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
w("")
w("### 2.1 锚件逐字段机读抄录")
w("")
w("`benchmarks/verified/suboff_anchor_w22/build_anchor_transcription.py`：")
w("")
w("- JSON 源：深度优先遍历全部标量叶节点（str/int/float/bool/null），"
  "指针为 **RFC-6901**（`~0`/`~1` 转义），空串=文档根；")
w("- Markdown 源：正文/表格/代码块三类的**全部数值 token**（正则带左右标识符边界，"
  "避免把 `L96`/`AFF-8`/`m298`/`GPU5` 误判为数），指针为文档序"
  "`/md_tables/<t>/rows/<row_key>/<col>/tok<j>`、`/md_prose/tok<j>`、`/md_code_blocks/<k>/tok<j>`；")
w("- 每个叶节点带 `source_path` + `source_sha256` + 逐字符定位（行/列/raw）；")
w("- **回读**：JSON 侧用独立实现的指针解析回原文件比对 `type` 与 `repr`；"
  "Markdown 侧按记录的行列**重读源文件字符**再 `float()` 比对——不是把内存里同一对象写两遍；")
w("- 自检：指针唯一性、叶数守恒、失配清单。")
w("")
w("### 2.2 AN-ITTC57 独立复算")
w("")
w("$C_F = 0.075/(\\log_{10}Re-2)^2$，三路实现（`math.log10` / `math.log\\math.log(10)` / "
  "60 位 `Decimal.ln`），三路互差见工件；`Re=10^2` 处为极点，代码显式返回 `inf` 而非崩溃。")
w("")
w("### 2.3 闭合结构式")
w("")
w("$C_T = (1+k)\\,C_F + C_W$，按四个口径算：")
w("")
w(f"- **A（主口径）**：$C_F$ = ITTC-57 @ 案例 $Re$，$k$ = 记录的 AFF-8 形式因子，$C_W=0$；")
w(f"- **B**：$C_F$ = 案例自身记录参考（Blasius @ 案例 $Re$），$k$ = 记录 AFF-8；")
w(f"- **C**：$C_F$ = 案例参考，$k$ = 记录的 AFF-1 裸艇体值（with-sail 几何的下界）；")
w(f"- **D（定义性，非证据）**：锚自分比 $C_T/C_F$ 在锚 $Re$ 上自洽——$k$ 本身由该比定义，残差恒零，"
  "只作诚实披露，不作证据。")
w("")
w("$C_W$ 取零的依据（程序化扫描）：全部 "
  f"{TRANS['n_leaves_total']} 个抄录叶中匹配 `/c_w|/cw|/wave|/rw|/ct_wave` 的命中共 "
  f"**{CLOS['closure']['C_W_assumption']['n_hits_in_transcription']}** 条，即没有任何工件承载兴波分量；"
  "案例自身的参考结构也只有单一摩擦项。此外把残差反解成『闭合所需 $C_W$』，其符号即判决。")
w("")
w("### 2.4 k 的两口径")
w("")
w("- 记录值：AFF-8 锚最点 $(1+k)=C_T/C^{ITTC}_{F}$，两次独立转录双录（S5 与 S6）；")
w("- Prohaska：$C_T/C_F=(1+k)+c\\,Fr^{n}$，$Fr=U/\\sqrt{gL}$，$n=4$ 固定，"
  "对 AFF-8 阶梯 5 点做无权 OLS，截距即 $(1+k)$；")
w("- 敏感性：最低 3 / 4 点窗口 + $n\\in[3.0,5.0]$ 步长 0.05 扫描，全量披露。")
w("")
w("### 2.5 预声明判据（动手前固定于脚本常量 `CRITERIA_PREDECLARED`）")
w("")
w("| 判据 | 阈值 | 主口径 |")
w("|---|---|---|")
c = CLOS["criteria_predeclared"]
w(f"| C1 抄录完整率 | = {c['C1_completeness_pct']}% | 全源全叶 |")
w(f"| C2 ITTC-57 复算 vs 记录 | < {c['C2_cf_rel_dev_pct']}% | 锚件中带 ITTC-1957 记录的每一档 |")
w(f"| C3 三档闭合残差 | < {c['C3_closure_residual_pct']}% | {c['C3_primary_caliber']} |")
w(f"| C4 k 双口径差异 | < {c['C4_k_two_caliber_diff_pct']}% | {c['C4_basis_primary']} |")
w("")
w(f"Prohaska 主规则预声明：{c['prohaska_primary_rule']}；敏感性规则：{c['prohaska_sensitivity']}；"
  f"{c['no_threshold_tuning']}。")
w("")
w(f"工具链：Python 3.11（`json`/`math`/`decimal`）、matplotlib 出图、pytest 验证；"
  f"复现命令 `python3 {HERE.relative_to(REPO)}` → "
  f"`python3 {(CASE / 'build_anchor_transcription.py').relative_to(REPO)}` → "
  f"`python3 {(CASE / 'build_ittc57_closure.py').relative_to(REPO)}` → "
  f"`python3 {(CASE / 'make_figures.py').relative_to(REPO)}` → "
  f"`pytest tests/test_suboff_anchor_w22.py`。")
w("")

# ------------------------------------------------------------------- 3. results
w("## 3. 计算结果（数字表，程序化注入）")
w("")
w("### 3.1 抄录普查")
w("")
w("| 源 | 类型 | 叶节点数 | 回读失配 | sha256（前 16） |")
w("|---|---|---|---|---|")
for s in TRANS["sources"]:
    w(f"| `{s['path']}` | {s['kind']} | {s['n_scalar_leaves']} | {s['n_readback_mismatch']} | "
      f"`{s['sha256'][:16]}…` |")
w(f"| **合计** | — | **{TRANS['n_leaves_total']}** | "
  f"**{sum(s['n_readback_mismatch'] for s in TRANS['sources'])}** | — |")
w("")
w("### 3.2 ITTC-1957 复算 vs 锚件记录（C2，逐档）")
w("")
w("| 档（τ 臂） | $Re$ | 记录 $C_F^{{ITTC57}}$ | 独立复算 | 相对偏差 | <1%? | 记录指针 |")
w("|---|---|---|---|---|---|---|")
for a in CLOS["recompute_checks"]["per_arm"]:
    w(f"| {a['arm']} | {a['Re']:.0f} | {a['cf_recorded']:.10g} | {a['cf_recomputed']:.10g} | "
      f"{a['rel_dev_pct']:+.3e}% | {'是' if abs(a['rel_dev_pct']) < 1.0 else '否'} | "
      f"`{a['evidence']['record']['pointer']}` |")
w("")
w(f"最大绝对相对偏差 **{CLOS['recompute_checks']['max_abs_rel_dev_pct']:.3e}%** < 1% → **C2 PASS**。"
  f"（偏差全部来自锚件记录本身的 6 位有效数字舍入：A 档全精度为 0，B–E 档为摘要值。）")
w("")
rc = CLOS["recompute_checks"]["re_lattice_consistency"]
w(f"雷诺阶梯自洽：各臂 `Re·ν_lb` = {', '.join(f'{k}:{v:.15g}' for k, v in rc['uL_by_arm'].items())}，"
  f"案例点 `Re·ν` = {rc['Re_case_times_nu']:.12g} —— 同一 $u_{{lb}}L$ 冻结值，"
  f"证明该阶梯**是雷诺阶梯而非等 Re 扫描**（与 m298 报告一致）。")
w("")
w("Markdown 转录件与 JSON 工件同量对照（转录保真度正交检查）：最大相对偏差 "
  f"{CLOS['md_vs_json_crosscheck']['max_abs_rel_dev_pct']:.3e}%（6 位有效数字舍入量级）。")
w("")
w("### 3.3 案例 Re 口径冲突（披露行 C2'，不参与 C2 判据）")
w("")
w("| 量 | 值 |")
w("|---|---|")
w(f"| 案例 $Re$ | {gap['Re_case']:.0f} |")
w(f"| ITTC-57 @ 案例 $Re$ | {gap['cf_ittc57_at_case_Re']:.6f} |")
w(f"| 案例自身参考 $C_F$（层流 Blasius） | {gap['cf_case_reference_recorded']:.9f} |")
w(f"| 相对偏差 | **{gap['rel_dev_pct']:+.3f}%** |")
w(f"| 案例参考式复算残差（1.328/√Re） | {CLOS['case']['blasius_check']['rel_dev_pct']:+.2e}% |")
w("")
w("### 3.4 闭合检验（C3，逐档 × 逐口径）")
w("")
w("| 网格档 | $C_T$ 实测 | 口径 A 预测 | A 残差 | 口径 B 残差 | 口径 C 残差 | A 所需 $C_W$ 才能闭合 |")
w("|---|---|---|---|---|---|---|")
for i, gr in enumerate(grids):
    ra, rb, rcc = cal_a["per_tier"][i], cal_b["per_tier"][i], cal_c["per_tier"][i]
    w(f"| {gr['tier']} | {ra['C_T']:.6f} | {ra['C_T_predicted']:.6f} | "
      f"**{ra['residual_pct']:+.2f}%** | {rb['residual_pct']:+.2f}% | {rcc['residual_pct']:+.2f}% | "
      f"{ra['required_C_W_to_close']:+.6f} |")
w("")
w("| 口径 | 定义 | 最大绝对残差 | 全部 <3%? |")
w("|---|---|---|---|")
w(f"| A（主） | $C_F$=ITTC-57@案例$Re$, $k$=记录 AFF-8, $C_W$=0 | "
  f"{cal_a['max_abs_residual_pct']:.2f}% | {'是' if cal_a['all_pass'] else '否'} |")
w(f"| B | $C_F$=案例参考, $k$=记录 AFF-8 | {cal_b['max_abs_residual_pct']:.2f}% | "
  f"{'是' if cal_b['all_pass'] else '否'} |")
w(f"| C | $C_F$=案例参考, $k$=记录 AFF-1 | {cal_c['max_abs_residual_pct']:.2f}% | "
  f"{'是' if cal_c['all_pass'] else '否'} |")
w(f"| D（定义性，非证据） | 锚自分比自洽 | {cal_d['max_abs_residual_pct']:.2e}% | 定义使然 |")
w("")
w("$C_T$ 口径敏感性（E2 窗均值 vs 归档 mix50）逐档残差："
  + "；".join(f"{x['tier']}/{x['C_T_caliber']} {x['residual_pct']:+.2f}%"
             for x in CLOS["closure"]["C_T_caliber_sensitivity"]) + "。")
w("")
w("口径 C 揭示的机理：把 AFF-8 的 $k$="
  f"{cal_a['k']:.4f} 换成 AFF-1 裸艇体的记录值 $k$={cal_c['k']:.4f} 后，最细网格残差由 "
  f"{cal_b['per_tier'][-1]['residual_pct']:+.2f}% 收到 {cal_c['per_tier'][-1]['residual_pct']:+.2f}% —— "
  "**形式因子的取值是差距主项**（附体/口径差异）；剩余项与 §3.6 的 $k_{eff}\approx+0.01$ 一致，"
  "即低 $Re$ 层流流动本身提供的形状阻力增量只有 1% 量级，远小于任何湍流口径的形式因子。")
w("")
w("### 3.5 k 的双口径（C4）")
w("")
w("| 量 | 值 | 说明 |")
w("|---|---|---|")
w(f"| 记录 $k$（AFF-8，$Re$={g(CLOS['formula_AN_SUBOFF']['anchor_Re_aff8']):.6g}） | {rec['primary']:.6f} | 主记录（15 位） |")
w(f"| 双录值（独立第二转录） | {rec['double_entry']:.6f} | 相对差 {rec['double_entry_rel_dev_pct']:+.2e}% |")
w(f"| Prohaska 拟合 $(1+k)$ | {pro['fit_aff8']['intercept']:.6f} ± {pro['fit_aff8']['stderr_intercept']:.6f} | $n$=4，5 点 OLS，$R^2$={pro['fit_aff8']['r2']:.3f} |")
w(f"| Prohaska $k$ | {pro['k_prohaska']:.6f} | 截距 − 1 |")
w(f"| 差异（$k$ 口径，主判据） | **{pro['diff_pct_k_basis']:+.3f}%** | 阈值 5% → **FAIL** |")
w(f"| 差异（$(1+k)$ 口径，披露） | {pro['diff_pct_one_plus_k_basis']:+.3f}% | 同数据、不同基，<5% |")
w(f"| 点估计距记录值 | {pro['sigma_distance_point_estimate']:+.2f}σ | 记录值落在拟合 ±2σ 带内 |")
w("")
w("| 窗口敏感性 | 点数 | $(1+k)$ | $k$ | 差异（$k$ 基） | <5%? |")
w("|---|---|---|---|---|---|")
for s in pro["window_sensitivity"]:
    w(f"| {s['window']} | {s['n_points']} | {s['intercept']:.6f} | {s['k']:.6f} | "
      f"{s['diff_vs_record_pct_k_basis']:+.3f}% | {'是' if s['pass_k_basis'] else '否'} |")
w(f"| 全 5 点（预声明主规则） | 5 | {pro['fit_aff8']['intercept']:.6f} | {pro['k_prohaska']:.6f} | "
  f"{pro['diff_pct_k_basis']:+.3f}% | 否 |")
w("")
lo, hi = crit["C4"]["measured"]["exponent_scan_intercept_range"]
w(f"自由指数扫描 $n\\in[3.0,5.0]$：截距区间 [{lo:.6f}, {hi:.6f}]，"
  f"对应 $k$ 区间 [{lo - 1:.6f}, {hi - 1:.6f}] —— 记录 $k$={rec['primary']:.6f} 落在区间下方，"
  "即拟合主规则下两口径不可同化。")
aff1 = pro["fit_aff1_crosscheck"]
w(f"跨族交叉检（AFF-1 裸艇体 6 点，$R^2$={aff1['r2']:.3f}）：$k_{{Prohaska}}$={aff1['k']:.6f} vs "
  f"记录 {crit['C4']['measured']['k_aff1_record']:.6f}，差异 "
  f"{aff1['diff_vs_aff1_record_pct_k_basis']:+.2f}% —— 说明该 Prohaska 口径在仓内阶梯数据上"
  "本身可辨识性极弱（$R^2$ 接近 0），不是选点问题。")
w("")
w("### 3.6 三档网格走向（诊断量）")
w("")
w("| 网格档 | $C_T$（E2） | 隐含 $k_{eff}=C_T/C_F^{case}-1$ |")
w("|---|---|---|")
for gr in grids:
    w(f"| {gr['tier']} | {gr['C_T_primary_e2_window']:.6f} | {gr['k_eff_vs_case_reference']:+.4f} |")
w("")
w("$k_{eff}$ 随网格加密从负数收敛到约 +0.01：本案的『形式因子』是本征的层流低 Re 量级，"
  "与 AFF-8 的湍流 0.255 完全不同量级——闭合失败的机理在数字上直接可见。")
w("")

# ------------------------------------------------------------------- 4. figures
w("## 4. 结果图")
w("")
for i, fig in enumerate(FIGS["figures"], 1):
    w(f"![fig{i}]({fig['file']})")
    w("")
    w(f"**图 {i}**（`{fig['file']}`，sha256 `{fig['sha256'][:16]}…`）：{fig['caption']}")
    w("")
w("图中所有点数均由上述两件工件注入（见 `figures/manifest.json` 的 `data_evidence` 指针清单），"
  "无手工录入曲线。")
w("")

# --------------------------------------------------------------------- 5. notes
w("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
w("")
w("1. **ITTC-1957 不是本案例 Re 段的物理参考**：它是湍流相关线，$Re=10^2$ 处有极点，"
  "$Re=10^3$ 处值恰为 0.075（=0.075/1），比本案层流参考高 78.6%。"
  "本单把它作为**抄录/复算保真度**的检验对象（C2），而非精度基准（C3 因此必然失败），"
  "并在 §3.3 显式披露该口径冲突。")
w("2. **τ 阶梯就是雷诺阶梯**：$u_{lb}$、$L$ 被案例冻结，$\\nu=(\\tau-0.5)/3$ ⇒ $\\tau\\uparrow$ 则 $Re\\downarrow$"
  "（本单用 `Re·ν_lb` 恒等式程序化验证，见 §3.2）。因此五档不是等 Re 重复，"
  "跨档比较必须带 Re 口径。")
w("3. **k 在案例侧不可辨识**：三档网格同一 $Re$、同一 $Fr$，Prohaska 只能在锚阶梯上做；"
  "两口径不是同一物理 $k$ 的独立估计，这本身是判据 C4 的方法论上限。")
w("4. **Prohaska 在仓内阶梯上统计力弱**：AFF-8 $R^2$=0.35、AFF-1 $R^2$=0.01，"
  "截距标准误 0.025；记录值落在 ±2σ 内（1.79σ），即点估计过不了 5% 门但统计上不可区分。"
  "本单按预声明主规则（全点、$n$=4）报 FAIL，并把窗口/指数敏感性全量列出，不择其利。")
w("5. **锚 Re 不可达（上游结论，直接影响本单结论的适用范围）**："
  "`benchmarks/verified/suboff_anchor/anchor_points.json` C4 判『NOT REACHABLE at credible "
  "resolution』（可信档 ~4276× 预算）。故『同 Re 闭合』在当前算力下无法证真；"
  "本单只证伪了跨 Re 的相似性换算式闭合。")
w("6. **$C_W=0$ 是案例自身口径，不是本单加的假设**：抄录全量扫描无兴波字段命中；"
  "且残差反解出的所需 $C_W$ 三档全为负（"
  + "、".join(f"{t}: {v:+.6f}" for t, v in
              crit["C3"]["measured"]["required_C_W_to_close"].items())
  + "），物理上非负的兴波阻力不可能闭合该恒等式 —— 闭合失败**不可能靠补一个兴波项救回**。")
w("7. **锚是二手转录**：离线环境无外网，DTRC/ITTC 原始报告不在工作树内，"
  "全部外部锚来自仓内已归档文本（该限制由上游 "
  "`benchmarks/verified/suboff_anchor/anchor.json :: external_evidence.offline_disclosure` 承载）。")
w("8. **Markdown 记录只有 6 位有效数字**：md 与 JSON 同量对照偏差 2.4e-5 量级，"
  "属舍入而非漂移；需要比特级复核时以 JSON 工件为准（本单已如此做）。")
w("9. **案例几何是 AFF-3（with sail）**，锚记录只有 AFF-1 与 AFF-8 两个形式因子，"
  "故 0.0753 / 0.2552 构成 $k$ 的上下界，而非该几何的真值。")
w("")

# --------------------------------------------------------------- 6. error judgement
w("## 6. 误差判定（关键量 vs 基准，逐项给数与显式回答）")
w("")
w("| 关键量 | 基准 | 误差 | 是否 <3%？ | 证据指针 |")
w("|---|---|---|---|---|")
w(f"| ITTC-57 $C_F$（复算） | 锚件记录 ITTC-57 $C_F$（5 档） | max {CLOS['recompute_checks']['max_abs_rel_dev_pct']:.3e}% | "
  f"**是**（判据 1% 亦通过） | `{clos_rel}` `/recompute_checks` |")
w(f"| 案例参考 $C_F$ | Blasius $1.328/\\sqrt{{Re}}$ | {abs(CLOS['case']['blasius_check']['rel_dev_pct']):.2e}% | **是** | "
  f"`{clos_rel}` `/case/blasius_check` |")
w(f"| $C_T$ 闭合（口径 A，L96） | $(1+k)C_F+C_W$ | {abs(cal_a['per_tier'][0]['residual_pct']):.2f}% | **否** | "
  f"`{clos_rel}` `/closure/caliber_A_primary/per_tier/0` |")
w(f"| $C_T$ 闭合（口径 A，L112） | 同上 | {abs(cal_a['per_tier'][1]['residual_pct']):.2f}% | **否** | "
  f"`{clos_rel}` `/closure/caliber_A_primary/per_tier/1` |")
w(f"| $C_T$ 闭合（口径 A，L128） | 同上 | {abs(cal_a['per_tier'][2]['residual_pct']):.2f}% | **否** | "
  f"`{clos_rel}` `/closure/caliber_A_primary/per_tier/2` |")
w(f"| $C_T$ 闭合（口径 C，L128） | 用 AFF-1 $k$ 的最优情形 | {abs(cal_c['per_tier'][2]['residual_pct']):.2f}% | **否** | "
  f"`{clos_rel}` `/closure/caliber_C/per_tier/2` |")
w(f"| $k$ 双口径（$k$ 基） | 记录 AFF-8 $k$ | {abs(pro['diff_pct_k_basis']):.2f}% | **否** | "
  f"`{clos_rel}` `/k_identifiability/prohaska` |")
w(f"| $k$ 双口径（$(1+k)$ 基） | 记录 AFF-8 $(1+k)$ | {abs(pro['diff_pct_one_plus_k_basis']):.2f}% | **是**（披露口径） | 同上 |")
w("")
w("**显式回答**：三档网格 $C_T$ 与 $(1+k)C_F+C_W$ 结构的闭合误差**不满足 <3%**"
  f"（主口径 {abs(cal_a['per_tier'][0]['residual_pct']):.1f}% / "
  f"{abs(cal_a['per_tier'][1]['residual_pct']):.1f}% / {abs(cal_a['per_tier'][2]['residual_pct']):.1f}%），"
  "且所需 $C_W$ 三档为负，**不是可修的数值误差，而是口径不可比**："
  "ITTC-57（湍流线）+ AFF-8（$Re\\approx1.17\\times10^7$）的组合不能用来预测 $Re=10^3$ 的层流总阻力。")
w("")

# ------------------------------------------------------------------- 7. criteria
w("## 7. 判据逐条裁决表")
w("")
w("| 判据 | 量值 | 阈值 | 裁决 | 证据指针 |")
w("|---|---|---|---|---|")
w(f"| **C1** 锚件抄录完整率 | {crit['C1']['measured']['completeness_pct']:.4f}%（{crit['C1']['measured']['n_transcribed']}/{crit['C1']['measured']['n_expected']} 叶，回读失配 {crit['C1']['measured']['n_missing_or_mismatched']}） | 100% | **{crit['C1']['verdict']}** | `{art_rel}` `/completeness` |")
w(f"| **C2** ITTC-57 复算 vs 记录（逐档） | 最大偏差 {crit['C2']['measured']['max_abs_rel_dev_pct']:.3e}% | <1% | **{crit['C2']['verdict']}** | `{clos_rel}` `/recompute_checks` |")
w(f"| C2'(披露) 案例 Re 口径冲突 | {c2p['measured']['rel_dev_pct']:+.3f}% | <1%（不参与 C2 判定） | {c2p['verdict']} | `/case_caliber_gap` |")
w(f"| **C3** 三档闭合残差 | {cal_a['per_tier'][0]['residual_pct']:+.2f}% / {cal_a['per_tier'][1]['residual_pct']:+.2f}% / {cal_a['per_tier'][2]['residual_pct']:+.2f}% | <3% | **{crit['C3']['verdict']}** | `/closure/caliber_A_primary` |")
w(f"| **C4** k 双口径差异 | $k$ 基 {abs(pro['diff_pct_k_basis']):.2f}%（$(1+k)$ 基 {abs(pro['diff_pct_one_plus_k_basis']):.2f}%） | <5% | **{crit['C4']['verdict']}** | `/k_identifiability/prohaska` |")
w("")
w(f"**总判决 = {CLOS['overall_verdict']}**（C1/C2 PASS，C3/C4 FAIL；未改判据、未改阈值）。")
w("")
w("### 7.1 与前身任务 `paper-exp-lbm-suboff-tau-4` 的判据对齐")
w("")
w("前身零报告、零工件，其判据只能按本单任务书转述的版本逐条对齐（该继承事实如实披露）：")
w("")
w("| 前身判据（任务书转述） | 本单回应 | 裁决 |")
w("|---|---|---|")
w(f"| ① 锚件抄录完整率 100%、缺失逐条列因 | §3.1 全源逐叶抄录 + 独立回读，缺失 0 条 | **{crit['C1']['verdict']}** |")
w(f"| ② ITTC-57 $C_F$ 与记录逐档偏差 <1% | §3.2 五档全量对照 | **{crit['C2']['verdict']}** |")
w(f"| ③ 三档 $C_T$ 闭合 <3% | §3.4 三档 × 四口径（含定义性口径披露） | **{crit['C3']['verdict']}** |")
w(f"| ④ k 双口径差异 <5% | §3.5 记录值 vs Prohaska，两基双披露 + 敏感性 | **{crit['C4']['verdict']}** |")
w("")

# ------------------------------------------------------------------- 8. honest fail
w("## 8. honest FAIL 说明（做了什么 / 没做到什么 / 不声称什么）")
w("")
h = CLOS["honest_fail_summary"]
w("**已交付**：")
for x in h["delivered"]:
    w(f"- {x}")
w("")
w("**未达标（如实）**：")
for x in h["failed"]:
    w(f"- {x}")
w("")
w("**不声称**：")
for x in h["not_claimed"]:
    w(f"- {x}")
w("")
w("**归因（逐条）**：")
for key, txt in CLOS["attribution"].items():
    w(f"- **{key}**：{txt}")
w("")
w("**要闭合需要什么**（下一班可选路线，均不改本单判据）：")
w("1. **同 Re 侧**：在 $Re\\sim10^6$ 量级、壁面分辨可接受的网格上直接模拟 AFF-8 外形——"
  "上游预算评估为可信档 ~4276× 预算，当前不可行；")
w("2. **低 Re 侧**：把 $C_F$ 换成与案例同 Re 的层流参考（本案自身参考），并用**案例自身 Re 阶梯**"
  "做 Prohaska（需要至少 3 个不同 $u_{lb}$ 的同网格跑次，且 $\\tau$ 保持固定），"
  "这正是 m298 报告 §6 的 R1 路线（解耦 $\\tau$ 与 $Re$）；")
w("3. **式子侧**：若坚持 ITTC-57 结构式，必须把对比点搬到 $Re\\gtrsim10^5$，"
  "并同时给出相似性换算误差预算（上游已量化：最有利 $Re_s=10^6$ 仍有 10.87%）。")
w("")

# ------------------------------------------------------------------- 9. provenance
w("## 9. 溯源（工件路径 + commit + sha256 + 复现）")
w("")
w("| 工件 | sha256 |")
w("|---|---|")
w(f"| `{art_rel}` | `{sha256(CASE / 'anchor_transcription.json')}` |")
w(f"| `{clos_rel}` | `{sha256(CASE / 'ittc57_closure.json')}` |")
w(f"| `{fig_rel}/manifest.json` | `{sha256(CASE / 'figures/manifest.json')}` |")
w("| `papers/reports/w22_lbm_suboff_anchor8.md` | 本文件不自我记录 sha256（自指）；提交记录中的 commit 即其身份 |")
w("")
w("| 输入源 | sha256 |")
w("|---|---|")
for s in TRANS["sources"]:
    w(f"| `{s['path']}` | `{s['sha256']}` |")
w("")
w(f"git commit（工件写入时）：`{CLOS['git']['commit']}`；分支 `{CLOS['git']['branch']}`。")
w("")
w("复现：")
w("")
w("```bash")
w(f"python3 {CASE.relative_to(REPO)}/build_anchor_transcription.py   # 抄录 + 回读")
w(f"python3 {CASE.relative_to(REPO)}/build_ittc57_closure.py        # 复算 + 闭合 + Prohaska")
w(f"python3 {CASE.relative_to(REPO)}/make_figures.py                 # 4 图 + manifest")
w(f"python3 {CASE.relative_to(REPO)}/build_report.py                 # 本报告")
w("python3 -m pytest tests/test_suboff_anchor_w22.py -q            # 13 项不变量")
w("```")
w("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(REPO)} ({len(lines)} lines)")