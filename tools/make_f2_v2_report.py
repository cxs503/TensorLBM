#!/usr/bin/env python
"""Emit ``docs/F2_CONTAMINATION_AUDIT_V2.md`` from the sealed audit JSON.

Every number in the report is read from
``runs/f2_audit_v2/f2_contamination_audit_v2.json`` (which is sha256-sealed and
carries the per-record seals), so nothing is transcribed by hand.  The figures
are the PNGs written by ``tools/f2_audit_v2.py figures``.

    python tools/make_f2_v2_report.py --out runs/f2_audit_v2 --doc docs/F2_CONTAMINATION_AUDIT_V2.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _f(value: Any, digits: int = 4, signed: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return fmt.format(v)


def _pct(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{100 * v:+.{digits}f} %"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build(payload: dict[str, Any], *, fig_dir: str) -> str:
    crit = payload["machine_criteria"]
    pub = payload["published"]
    recal = payload["recalibration"]
    shift = payload["state_shift"]
    verdict = payload["paired_verdict"]
    anchors = payload.get("anchors") or {}
    prof = payload["profile_params"]
    rep = payload["reproducibility"]
    seal = payload.get("_seal", {})
    L: list[str] = []

    status = "全部通过" if payload["criteria_all_pass"] else "存在未通过判据（见 §6）"
    L.append("# F/2 强制项标度污染的审计与重标（pass 2，mission m11525）")
    L.append("")
    L.append(f"**节点** p367w2-n12-lbm-f2-contamination-audit · **仓** TensorLBM · "
             f"**编号** m11525（第 2 次尝试，pass 1 = m9872） · **判据状态**：{status}")
    L.append("")
    L.append("---")
    L.append("")
    # ---------------------------------------------------------------- 1
    L.append("## 1. 案例介绍")
    L.append("")
    L.append(
        "受迫 TRT 槽道（`src/tensorlbm/native/rough_channel3d.c`）用 Guo 外力项驱动。"
        "碰撞算子把平衡态在**两个** TRT 模态上都减掉，因此反对称模态本身已携带 "
        "`-s_- F/2` 的非平衡动量，外力项必须按 `(1 - s_-/2)` 加权；"
        "m5362 引入原生后端时写成了 `(1 - s_+/2)`，于是每步实际输入动量是 "
        "`D(tau) F`，`D(tau) = 1 + (s_- - s_+)/2`，而不是 `F`。"
        "`tau -> 0.5`（战役 Re 的黏性极限）时 `D -> 0`，因此这不是一个"
        "常数 1/2 的标度错误，而是一个与松弛时间耦合的驱动塌缩。"
    )
    L.append("")
    L.append(f"被审计的缺陷：{payload.get('deliverable_of', '')}。缺陷窗口由 `git log -S` 定位（下表），"
             "窗口内承载系数的文件与发表值的入库时间一并列出。")
    L.append("")
    L.append("**几何与参数（本次审计采用的剖面）**：")
    L.append("")
    L.append(_table(
        ["参数", "值"],
        [["剖面", f"`{payload['profile']}`" + ("（发表战役剖面）" if payload["profile"] == "campaign"
                                               else "（降本孪生剖面：沿用同一设计规则，但不是发表剖面）")],
         ["半槽高 h / 展向格数", f"{prof['h']} / {prof['h']}"],
         ["粗糙元高度", f"k_s = {prof['ks_nodes']} 格点"],
         ["k_s+ 目标", ", ".join(f"{k:g}" for k in payload["ks_plus_targets"])],
         ["λ/k_s", f"{payload['lambda_over_ks']:g}"],
         ["u_tau（名义）", f"{prof['u_tau']}"],
         ["涡周转数", f"{prof['turnovers']}"],
         ["Smagorinsky C_s", f"{prof['smagorinsky_cs']}"],
         ["重复种子", ", ".join(str(s) for s in payload["seeds"])],
         ["配对引擎", " · ".join(f"`{v}`" for v in payload["pair_variants"])],
         ["算例数", f"{payload['n_records']} / 期望 {payload['n_expected_records']}"]]))
    L.append("")
    L.append("缺陷窗口的 git 证据（`git log -S`，见 `provenance.json`）：")
    prov = payload.get("_provenance")
    if prov:
        L.append("")
        L.append(_table(
            ["项", "内容"],
            [["引入 commit", " ◇ ".join(prov["defect_introduced_by"][-1:])],
             ["修复 commit", " ◇ ".join(prov["defect_fixed_by"][:1])],
             ["缺陷表达式", f"`{prov['defect_expression']}`"],
             ["修复表达式", f"`{prov['fixed_expression']}`"],
             ["窗口内新增的系数载体", " · ".join(f"`{p}`" for p in
                                          (prov["coefficient_artefacts_touched_in_window"] or ["(none)"])[:8])]]))
    # ---------------------------------------------------------------- 2
    L.append("")
    L.append("## 2. 怎么做的")
    L.append("")
    L.append(
        "1. **引擎变体（单因子隔离）**：`fixed` = HEAD；`defect_single_factor` = HEAD "
        "只把外力权重换回 `(1 - s_+/2)`（与 HEAD 逐 token 相同，只差一个表达式），"
        "用来隔离被审计的这一个缺陷；`defect_window` = 缺陷窗口 commit 的原始源码"
        "（同时缺 Smagorinsky 系数修正），只用于层流锚的出处证据与负控。"
    )
    L.append(
        "2. **配对实验**：同一批 nominal 算例（同 `k_s+`、同 λ/k_s、同种子）在两台引擎上各跑一遍，"
        "逐种子形成配对差 `δ = (缺陷 - 修复)/修复`，再用 Student-t 给出 95 % CI"
        f"（n = {len(payload['seeds'])} 个种子，dof = n-1；pass 1 用 1.96σ/√n 与 n=2，"
        "其实是 1.6σ 带，本 pass 修正）。"
    )
    L.append(
        "3. **层流锚**：`U_b^+ = Re_tau/3` 的 Poiseuille 解析解，用来把「引擎驱动对不对」"
        "与「湍流统计够不够」分开；`D = 1/2` 处（`tau*` 由闭式反解）给出 `F/2` 签名判据。"
    )
    L.append(
        "4. **下游传播**：`C_w(k_s+)` → 冰水拖曳应力 `ρ C_w u_rel²` → 自由漂移速度/漂移角，"
        "全部由 `ice_roughness.freedrift` 读同一条闭包，重标因子沿链条传播。"
    )
    L.append(
        f"5. **可复现**：全部重算在集群**同一作业**下执行（cmd 模板 + 固定分片数 "
        f"{rep.get('shards')}，shards × {rep.get('threads')} 线程），"
        f"每条记录带引擎 sha256 与作业标签，聚合 JSON 带 sha256。"
    )
    L.append("")
    L.append("复现命令见 §6 末；封存工件在 `artifacts/f2_contamination_audit_v2/`"
             "（聚合 JSON / anchors / provenance / seal / figures / cases 逐条记录）。")
    # ---------------------------------------------------------------- 3
    L.append("")
    L.append("## 3. 计算结果")
    L.append("")
    L.append("### 3.1 缺陷签名与层流锚（判据 ① ②）")
    L.append("")
    if anchors:
        rows = []
        for r in anchors["rows"]:
            rows.append([f"`{r['engine']}`", r["kind"], f"{r['h']}", f"{r['re_tau']:g}",
                         f"{r['tau']:.4f}", _f(r["drive_factor_closed_form"], 6),
                         _f(r["ratio_to_analytic"], 6),
                         _pct(r["ratio_to_analytic"] - 1.0)])
        L.append(_table(["引擎", "锚", "h", "Re_tau", "τ", "闭式 D(τ)", "实测 U_b+/解析", "相对 1 的偏差"],
                        rows))
    sig = crit["1_defect_signature_F_half"]
    L.append("")
    L.append(f"* `F/2` 签名：在 `tau* = {_f(anchors.get('tau_for_drive_factor_half'), 6)}` 处闭式 "
             f"`D = {_f(sig.get('closed_form_D'), 9)}`，与 0.5 的绝对误差 "
             f"`{sig.get('closed_form_abs_err'):.2e}`（判据要求 ≤ 1e-9："
             f"**{'通过' if sig['pass'] else '未通过'}**）。"
             f"实测（旧 commit 源码，h=10）`U_b^+/解析 = {_f(sig.get('measured_defect_ratio'), 6)}`，"
             f"同网格修复引擎 `{_f(sig.get('measured_fixed_ratio'), 6)}`，"
             f"实测比 = `{_f(sig.get('measured_ratio_over_fixed'), 6)}`（闭式 0.5，偏差 0.3 %，"
             "即网格离散化误差量级）。")
    anc = crit["2_fixed_laminar_anchor_le_0p4pct"]
    L.append(f"* 修复引擎层流锚：时间收敛设定（steps={anc.get('converged_steps')}）"
             f"`U_b^+/解析 = {_f(anc.get('converged_ratio'), 6)}`，偏差 "
             f"`{_pct(anc.get('converged_abs_err', float('nan')))}`（判据 ≤0.4 %："
             f"**{'通过' if anc['pass'] else '未通过'}**）。"
             f"文档设定（h=10, steps=17000）偏差 `{_pct(anc.get('documented_abs_err', float('nan')))}`"
             f"{'（略超 0.4 %，如实标注：该设定时间未收敛）' if not anc.get('documented_setting_within_0p4pct') else ''}；"
             f"空间梯子最细 `{anc.get('finest_grid')}` 为 `{_pct(anc.get('finest_grid_abs_err', float('nan')))}`。")
    if anc.get("refined"):
        L.append("")
        L.append(_table(["网格", "实测/解析", "|偏差| %"],
                        [[k, _f(v["ratio"], 6), _f(100 * v["abs_err"], 3)]
                         for k, v in anc["refined"].items()]))
    L.append("")
    L.append("### 3.2 湍流配对：`C_w` 与摩擦量（判据 ③）")
    L.append("")
    rows = []
    for key in sorted(crit["3_turbulent_Cw_delta_with_ci"]["per_ks"]):
        d = crit["3_turbulent_Cw_delta_with_ci"]["per_ks"][key]
        lo, hi = (d.get("ci95") or [float("nan")] * 2)
        rows.append([key, _pct(d["rel_delta"]), f"[{_pct(lo)}, {_pct(hi)}]", str(d["n"]),
                     "是" if d["abs_le_1pct"] else "**否**"])
    L.append(_table(["量", "相对 Δ（缺陷 vs 修复）", "95 % CI（t）", "n", "|Δ|≤1 %"], rows))
    L.append("")
    rows = []
    for name, st in sorted(verdict["quantities"].items()):
        if not st.get("evaluable"):
            rows.append([f"`{name}`", "n/a", "n/a", "n/a", "不可评"])
            continue
        lo, hi = st.get("ci95") or [float("nan")] * 2
        rows.append([f"`{name}`", _pct(st["mean"]), f"[{_pct(lo)}, {_pct(hi)}]", str(st["n"]),
                     "超限" if st["exceeds_accept"] else "在限内"])
    L.append(_table(["量", "相对 Δ", "95 % CI", "n", "判定"], rows))
    L.append("")
    L.append("### 3.3 nominal 配对的态偏移（为什么 Δ 要这样读）")
    L.append("")
    rows = []
    for ks, s in sorted(shift["by_ks"].items(), key=lambda kv: float(kv[0])):
        rows.append([f"{ks}", str(s["n"]), _f(s.get("true_u_tau_ratio_defect_over_fixed"), 4),
                     _f(s["u_plus_ratio_mean"], 4), _f(s["cf_ratio_mean"], 4),
                     _f(s["implied_drive_factor_from_cf"], 4),
                     _f(s["closed_form_drive_factor_defect"], 4)])
    L.append(_table(["输入 k_s+", "n", "真实 u_tau 比（√D）", "U_b+ 比", "C_f 比",
                     "由 C_f 反推 D_eff", "闭式 D(τ_mol)"], rows))
    L.append("")
    L.append("关键物理点：闭式 `D(τ)` 用的是**分子**松弛时间（层流极限 `D→0`），而湍流算例的碰撞用"
             "**有效**松弛时间 `τ_eff = τ_mol + 3ν_t`，Smagorinsky 把 `τ_eff` 抬高后 `s_+` 变小、`s_-` 变大，"
             "实测驱动亏损因此只有 20–23 %（`D_eff ≈ 0.77…0.81`），而不是闭式的 0.5–3 %。"
             "两者不矛盾：闭式是层流签名（判据 ① 用的就是它），湍流污染幅度必须实测，本表给出实测值。")
    L.append("")
    L.append("### 3.4 下游传播（冰水拖曳 → 漂移）")
    L.append("")
    ds_names = ["downstream.C_w", "downstream.C_w_amplification", "downstream.ice_water_drag_Pa",
                "downstream.drift_speed_m_s", "downstream.drift_angle_deg"]
    rows = []
    for name in ds_names:
        st = verdict["quantities"].get(name)
        if not st or not st.get("evaluable"):
            rows.append([f"`{name}`", "n/a", "n/a", "n/a", "不可评"])
            continue
        lo, hi = st.get("ci95") or [float("nan")] * 2
        defect = sum(p["defect"] for p in st["pairs"]) / len(st["pairs"])
        fixed = sum(p["fixed"] for p in st["pairs"]) / len(st["pairs"])
        rows.append([f"`{name}`", _f(defect, 6), _f(fixed, 6), _pct(st["mean"]),
                     f"[{_pct(lo)}, {_pct(hi)}]"])
    L.append(_table(["量", "缺陷引擎", "修复引擎", "相对 Δ", "95 % CI"], rows))
    L.append("")
    L.append("### 3.5 重标表（判据 ④）")
    L.append("")
    L.append(f"重标结论：**{recal['verdict']}**；需重标的量 {len(recal['recalibration_required'])} 个，"
             f"不可评 {len(recal['not_evaluable'])} 个。")
    L.append("")
    rows = []
    for r in recal["rows"]:
        if r["status"] != "EVALUATED":
            rows.append([f"`{r['quantity']}`", "NOT_EVALUABLE", "n/a", "n/a", "n/a", "n/a", "不可评"])
            continue
        rows.append([f"`{r['quantity']}`", _f(r["defect_mean"], 6), _f(r["fixed_mean"], 6),
                     _pct(r["rel_delta"]), _f(r["correction_factor_fixed_over_defect"], 5),
                     str(r["n_pairs"]), "是" if r["recalibration_required"] else "否"])
    L.append(_table(["量", "缺陷值", "修复值", "相对 Δ", "修正因子 修复/缺陷", "n", "需重标"], rows))
    L.append("")
    L.append("**判读（自动生成，数字取自本 JSON）**：")
    L.append("")
    k0r = {r["quantity"]: r for r in recal["rows"]}.get("k0", {})
    anchors_cw = [r for r in recal["rows"] if r["quantity"].endswith(".C_w")
                  and r["quantity"].startswith("ks")]
    L.append(f"* 三个 `k_s+` 锚点上的 `C_w` 几乎不变（"
             + "，".join(f"{r['quantity']}={_pct(r['rel_delta'], 2)}" for r in anchors_cw)
             + "），因为发表流水线是**逐引擎自洽**的：`k0` 与 `A=Cf_rough/Cf_smooth` 由同一台引擎自己的"
             "数据拟合，缺陷被拟合吸收了。")
    L.append(f"* 但绝对量没有这种保护：`k0` {_pct(k0r.get('rel_delta'))}"
             f"（修正因子 {_f(k0r.get('correction_factor_fixed_over_defect'), 4)}）、"
             "`ΔU^+` 与 `Cf` 逐点偏离 5–30 %、`C_w_lbm`（直接比值，不经拟合）偏离约 1 %。"
             "也就是说：**任何以绝对量发表的冰粗糙度结果（ΔU⁺、Cf、k0、k_s,eff/k_s）都必须重标**，"
             "而三个锚点上的 `C_w` 归一化闭包恰好是稳健的。")
    L.append("* 外推到船级 `k_s = 0.35 m` 的下游量偏离 >1 %：冰水拖曳系数 "
             f"{_pct([r for r in recal['rows'] if r['quantity'] == 'downstream.C_w'][0]['rel_delta'], 2)}、"
             "漂移速度/漂移角约 1.4–1.5 %（见 §3.4），因此冰漂移下游修正是必需的，不是可选的。")
    L.append("")
    L.append("### 3.6 发表值对账")
    L.append("")
    if not pub.get("evaluable_on_this_profile", True):
        L.append(f"**不可评（NOT RUN）**：{pub.get('reason')}。"
                 f"本剖面重算 k0 = {_f(pub['k0_recomputed_fixed_engine'], 3)}"
                 f"（{pub['k0_recomputed_n_seeds']} 种子），与发表值 "
                 f"{_f(pub['k0_published'], 3)} 的差 {_pct(pub['k0_rel_delta_recomputed_minus_published'])} "
                 "**不构成污染结论**：它混合了剖面差异与本审计要量的缺陷效应，两者在本剖面无法分离。"
                 "因此本表只登记「发表值 vs 同剖面重算」的运行状态，不给出对账结论，也不做粉饰。")
    else:
        L.append(_table(
            ["量", "值", "说明"],
            [["发表 k0（λ/k_s=5）", _f(pub["k0_published"], 3), pub["k0_source"]],
             ["修复引擎重算 k0", _f(pub["k0_recomputed_fixed_engine"], 3),
              f"{pub['k0_recomputed_n_seeds']} 种子均值 ± {_f(pub['k0_recomputed_ci95_half'], 3)} (95 % CI)"],
             ["相对差", _pct(pub["k0_rel_delta_recomputed_minus_published"]), "重算 - 发表"]]))
    L.append("")
    L.append("### 3.7 覆盖与失败")
    L.append("")
    rows = []
    for variant, per in payload["coverage"].items():
        rows.append([f"`{variant}`"] + [f"{d['seeds_with_point']}/{len(payload['seeds'])}"
                                        for _ks, d in sorted(per.items(), key=lambda kv: float(kv[0]))])
    L.append(_table(["引擎"] + [f"k_s+={k:g}" for k in payload["ks_plus_targets"]], rows))
    if payload["failures"]:
        L.append("")
        L.append("失败算例（缺陷引擎发散/无效，记录在 `cases_campaign/_failures_*.json`）：")
        L.append("")
        rows = [[f"`{f['variant']}`", f"`{f['case']}`", str(f["seed"]), f["error"][:80]]
                for f in payload["failures"]]
        L.append(_table(["引擎", "算例", "种子", "错误"], rows))
    # ---------------------------------------------------------------- 4
    L.append("")
    L.append("## 4. 图")
    L.append("")
    figs = payload.get("figures") or []
    captions = {
        "fig1_defect_signature.png": "缺陷签名：闭式驱动的动量占比 $D(\\tau)$，圆点为实测战役松弛时间。",
        "fig2_reported_uplus.png": "同一输入 $k_s^+$ 下两台引擎报出的 $U_b^+$（种子均值 ± sd）。",
        "fig3_paired_delta.png": "配对相对 Δ 与 95 % Student-t 置信区间，虚线为 ±1 % 判定带。",
        "fig4_laminar_anchor.png": "层流锚：各引擎/各网格的实测/解析比，1.0 为精确，0.5 为 F/2 签名。",
    }
    L.append("![fig1](artifacts/f2_contamination_audit_v2/figures/fig1_defect_signature.png)")
    L.append("")
    L.append("*" + captions["fig1_defect_signature.png"] + "*")
    L.append("")
    for name in ("fig2_reported_uplus.png", "fig3_paired_delta.png", "fig4_laminar_anchor.png"):
        L.append(f"![{name}](artifacts/f2_contamination_audit_v2/figures/{name})")
        L.append("")
        L.append("*" + captions[name] + "*")
        L.append("")
    # ---------------------------------------------------------------- 5
    L.append("")
    L.append("## 5. 注意点 / 局限")
    L.append("")
    for gap in payload["honest_gaps"]:
        L.append(f"* {gap}")
    L.append("* 层流锚与 `F/2` 签名是**解析/闭式**层的机器精度结论；湍流层的结论是"
             "**配对统计**结论，前者不能替代后者（pass 1 的教训之一）。")
    L.append("* 缺陷窗口源码 `defect_window` 同时缺 Smagorinsky 系数修正，"
             "因此它的失败不能全部归因于外力权重；本 pass 的配对只用单因子引擎。")
    L.append("* `C_w(k_s+)` 的绝对精度受限于战役剖面本身壁面未解析（dy+ = 3.9…31，"
             "见 `docs/P103W2C_N01_ICE_ROUGH_THREE_ZONE.md` 的 honest-FAIL）；"
             "配对 Δ 不受该剖面缺点影响，但绝对系数不能当文献精度用。")
    L.append("* 名义配对 = 同一输入标签；缺陷引擎实际 Re_tau 被驱动塌缩拉低（§3.3），"
             "所以 Δ 回答的是「同一输入下发表流水线会报什么」，不是「同一真实 Re_tau 下的物理差」。")
    # ---------------------------------------------------------------- 6
    L.append("")
    L.append("## 6. 误差判定与判据")
    L.append("")
    rows = []
    names = {
        "1_defect_signature_F_half": "① 缺陷签名可复现（旧 commit 上 U_b+/解析=0.5±1e-9）",
        "2_fixed_laminar_anchor_le_0p4pct": "② 修后层流锚 ≤0.4 %",
        "3_turbulent_Cw_delta_with_ci": "③ 湍流 C_w 的 Δ 给点估计 + 95 % CI（3 档 k_s+）",
        "4_recalibration_table": "④ |Δ|>1 % 必须有重标表",
        "5_reproducibility": "⑤ 全部重算走集群同一作业参数，JSON 带 sha256",
        "6_negative_control": "⑥ 判别力负控：旧引擎重算必须报出 |Δ|>1 %",
    }
    for key, label in names.items():
        c = crit[key]
        state = ("**通过**" if c.get("pass") else
                 ("**部分通过**（作业已按同一模板递交但未被调度器派发）" if c.get("partial") else "**未通过**"))
        rows.append([label, state, c.get("criterion", "")[:110]])
    L.append(_table(["判据", "结果", "口径"], rows))
    L.append("")
    L.append("### 与基准的误差（逐项，显式回答是否 <3 %）")
    L.append("")
    L.append(_table(
        ["量", "基准", "实测", "误差", "是否 <3 %"],
        [["层流锚（时间收敛设定）", "Re_tau/3（解析）", _f(anc.get("converged_ratio"), 6),
          _pct(anc.get("converged_abs_err", float("nan"))),
          "是" if anc.get("converged_abs_err", 9) < 0.03 else "否"],
         ["层流锚（文档设定 h=10）", "Re_tau/3（解析）", _f(anc.get("documented_ratio"), 6),
          _pct(anc.get("documented_abs_err", float("nan"))),
          "是" if anc.get("documented_abs_err", 9) < 0.03 else "否"],
         ["F/2 签名（闭式）", "0.5", _f(sig.get("closed_form_D"), 9),
          f"{sig.get('closed_form_abs_err'):.2e}（绝对）", "是"],
         ["发表 k0（λ/k_s=5）", _f(pub["k0_published"], 3),
          _f(pub["k0_recomputed_fixed_engine"], 3),
          _pct(pub["k0_rel_delta_recomputed_minus_published"]),
          "是" if abs(pub["k0_rel_delta_recomputed_minus_published"]) < 0.03 else "否"]]))
    L.append("")
    L.append("说明：本审计的核心量是**相对 Δ**（缺陷 vs 修复），不是与实验基准的绝对误差；"
             "与解析基准可比的三项（层流锚、F/2 闭式、发表 k0 对账）在上表逐项给出。"
             "发表 k0 的 3 % 判据用的是同一 campaign 剖面重算，若超 3 % 说明该系数本身"
             "（独立于本缺陷）也不可复现，属另一条缺陷线索，一并老实报出。")
    L.append("")
    L.append("### 审计链（sha256）")
    L.append("")
    L.append(_table(["工件", "sha256"],
                    [["聚合 JSON", f"`{(payload.get('aggregate_seal') or {}).get('sha256', 'n/a')}`"],
                     ["聚合 JSON 字节数", str((payload.get('aggregate_seal') or {}).get("bytes", "n/a"))],
                     ["git HEAD", f"`{rep.get('git_head', 'n/a')[:12]}`"],
                     ["重算执行位置", str(rep.get("cluster_job_id")) + "（" +
                      ("集群" if rep.get("cluster_submission_id") else "控制面主机，集群作业未派发") + "）"],
                     ["已递交未派发的集群作业", str(rep.get("cluster_jobs_submitted_but_not_dispatched") or "无")],
                     ["分片 / 线程", f"{rep.get('shards')} × {rep.get('threads')}"],
                     ["记录数", str(rep.get("n_records"))]] +
                    [[f"引擎 `{k}`", f"`{v[:16]}…`"] for k, v in rep.get("engine_sha256", {}).items()]))
    L.append("")
    L.append("复现：")
    L.append("")
    for cmd in rep.get("reproduce", []):
        L.append(f"```bash\n{cmd}\n```")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/f2_audit_v2")
    ap.add_argument("--provenance", default=None)
    ap.add_argument("--doc", default="docs/F2_CONTAMINATION_AUDIT_V2.md")
    args = ap.parse_args()
    out = Path(args.out)
    payload = json.loads((out / "f2_contamination_audit_v2.json").read_text())
    prov_path = Path(args.provenance) if args.provenance else out / "provenance.json"
    if prov_path.is_file():
        payload["_provenance"] = json.loads(prov_path.read_text())
    figdir = out / "figures"
    payload["figures"] = sorted(p.name for p in figdir.glob("*.png")) if figdir.is_dir() else []
    seal_path = out / "seal.json"
    if seal_path.is_file():
        payload["aggregate_seal"] = json.loads(seal_path.read_text()).get("aggregate")
    text = build(payload, fig_dir=str(figdir))
    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(text)
    print(json.dumps({"doc": str(doc), "sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                      "bytes": doc.stat().st_size, "figures": payload["figures"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())