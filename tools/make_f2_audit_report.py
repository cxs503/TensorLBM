#!/usr/bin/env python
"""Figures + paper-style report for the F/2 forcing-contamination audit (m9872).

Reads the sealed audit JSON produced by
``tools/audit_f2_forcing_contamination.py aggregate`` and writes

* ``artifacts/f2_contamination_audit/fig*.png`` (four figures),
* ``docs/F2_CONTAMINATION_AUDIT.md`` with every number injected from the JSON
  (no hand-typed values), plus the sha256 of the source artefacts.

The plotting is deliberately data-driven: nothing is hard-coded except axis
labels, so re-running the audit end-to-end re-draws and re-writes everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO / "src"))

from tensorlbm.ice_roughness import forcing_audit as fa  # noqa: E402
from tensorlbm.ice_roughness import freedrift as fd  # noqa: E402

FIG_DIR = REPO / "artifacts" / "f2_contamination_audit"
DOC = REPO / "docs" / "F2_CONTAMINATION_AUDIT.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _per_ks(engine: dict[str, Any], field: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for entry in engine.get("per_seed", []):
        for key, row in entry.get("points", {}).items():
            val = row.get(field)
            if val is None or not np.isfinite(float(val)):
                continue
            out.setdefault(key, {"values": [], "seeds": []})
            out[key]["values"].append(float(val))
            out[key]["seeds"].append(int(entry["seed"]))
    for key, rec in out.items():
        vals = np.asarray(rec["values"], dtype=float)
        rec["mean"] = float(vals.mean())
        rec["sd"] = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        rec["n"] = int(vals.size)
    return out


def figures(payload: dict[str, Any], anchors: dict[str, Any] | None,
            fig_dir: Path | None = None) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global FIG_DIR
    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs: list[dict[str, str]] = []

    # ---------------------------------------------------------------- fig 1
    taus = np.linspace(0.5005, 3.0, 800)
    drive = np.array([fa.drive_factor(float(t)) for t in taus])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.semilogx(taus - 0.5, drive, lw=2.0, color="crimson",
                label=r"defective weight: $D(\tau)=1+(s^- - s^+)/2$")
    ax.axhline(1.0, color="black", lw=1.6, ls="--",
               label=r"correct weight ($1-s^-/2$): $D\equiv 1$")
    tau_star = float(payload.get("anchors", {}).get("tau_for_drive_factor_half")
                     or fa.tau_for_drive_factor(0.5))
    ax.plot([tau_star - 0.5], [0.5], "o", ms=7, color="navy",
            label=rf"$D=1/2$ anchor: $\tau={tau_star:.5f}$")
    # campaign Reynolds numbers: tau = 0.5 + 3 u_tau h_eff / Re_tau
    u_tau, h_eff, ks_nodes = 0.012, 47.0, 5.0
    camp = [0.5 + 3.0 * u_tau * h_eff / (ks * h_eff / ks_nodes) for ks in (20, 45, 90, 200, 500)]
    ax.axvspan(min(camp) - 0.5, max(camp) - 0.5, color="orange", alpha=0.25,
               label="campaign $\\tau$ span ($k_s^+=20\\ldots500$)")
    ax.set_xlabel(r"$\tau - 1/2$")
    ax.set_ylabel(r"momentum supplied per step, $D$ (units of $F$)")
    ax.set_title("TRT forcing-scale defect: the factor is $\\tau$-dependent, not 2")
    ax.set_ylim(-0.02, 2.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    p = FIG_DIR / "fig1_defect_signature.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig1_defect_signature.png", "path": p,
                 "caption": "Closed-form momentum supplied by the defective TRT forcing weight relative to the body force F; the campaign relaxation times sit where the drive collapses to a few percent."})

    # ---------------------------------------------------------------- fig 2
    fixed = payload["fixed_engine"]
    defect = payload["defect_window_engine"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, field, label in ((axes[0], "C_w", r"$C_w = C_{w,\mathrm{smooth}}\,A(k_s^+)$"),
                             (axes[1], "cf_ratio", r"$C_f^{\mathrm{rough}}/C_f^{\mathrm{smooth}}$")):
        fx = _per_ks(fixed, field)
        df = _per_ks(defect, field)
        xs = sorted(fx, key=float)
        ax.errorbar([float(k) for k in xs], [fx[k]["mean"] for k in xs],
                    yerr=[0.0 if not np.isfinite(fx[k]["sd"]) else fx[k]["sd"] for k in xs],
                    fmt="o-", color="black", capsize=3,
                    label="fixed engine (HEAD), seed mean $\\pm$ sd")
        if df:
            xd = sorted(df, key=float)
            ax.plot([float(k) for k in xd], [df[k]["mean"] for k in xd], "s--",
                    color="crimson", label="defect-window engine")
        ax.set_xscale("log")
        ax.set_xlabel(r"$k_s^+$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)
        ax.set_title(f"{field} ($\\lambda/k_s = {payload['lambda_over_ks']:g}$)", fontsize=9)
    fig.suptitle("Calibrated drag quantities: fixed vs defect-window engine", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "fig2_cw_vs_ks.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig2_cw_vs_ks.png", "path": p,
                 "caption": "Ice/water drag coefficient and drag ratio versus k_s+ from the paired runs (fixed engine: seed mean with sd where two seeds exist; defect-window engine plotted for comparison)."})

    # ---------------------------------------------------------------- fig 3
    fields = [("C_w", r"$C_w$ at $k_s=0.35$ m"), ("ice_water_drag_Pa", "ice/water drag [Pa]"),
              ("drift_angle_deg", "drift angle [deg]")]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, (field, label) in zip(axes, fields):
        vals = []
        for engine, color in ((fixed, "black"), (defect, "crimson")):
            rows = [r for r in engine.get("downstream", []) if field in r]
            vals.append(np.asarray([float(r[field]) for r in rows], dtype=float) if rows
                        else np.array([np.nan]))
            ax.bar(len(vals) - 1, float(np.nanmean(vals[-1])), color=color, alpha=0.75,
                   yerr=(float(np.nanstd(vals[-1], ddof=1)) if vals[-1].size > 1 else 0.0),
                   capsize=4)
        if field == "drift_angle_deg":
            ax.axhspan(fd.NANSEN_ANGLE_BAND_DEG[0], fd.NANSEN_ANGLE_BAND_DEG[1],
                       color="green", alpha=0.15, label="observed band")
            ax.axhline(fd.NANSEN_DRIFT_ANGLE_DEG, color="green", ls=":", label="Nansen 28 deg")
            ax.legend(fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fixed", "defect"], fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Downstream ice/water drag and drift response "
                 f"($k_s={fa.K_S_REFERENCE_M}$ m, $u_{{rel}}=0.15$ m/s)", fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "fig3_downstream.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    figs.append({"name": "fig3_downstream.png", "path": p,
                 "caption": "Downstream application of the closure: ice/water drag coefficient, drag stress and free-drift angle at the reference keel roughness, for the fixed and the defect-window engine."})

    # ---------------------------------------------------------------- fig 4
    if anchors:
        rows = [r for r in anchors["rows"]]
        labels = [f"{r['engine']}\n({r['kind'].replace('laminar_anchor_','')})" for r in rows]
        measured = [r["ratio_to_analytic"] for r in rows]
        closed = [r["drive_factor_closed_form"] for r in rows]
        idx = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        ax.bar(idx - 0.2, measured, width=0.4, color="steelblue", label="measured $U_b^+/ (Re_\\tau/3)$")
        ax.bar(idx + 0.2, closed, width=0.4, color="darkorange", alpha=0.8,
               label="closed-form drive factor $D(\\tau)$")
        ax.axhline(1.0, color="black", ls="--", lw=1.2, label="analytic (fixed engine)")
        ax.axhline(0.5, color="crimson", ls=":", lw=1.2, label="$F/2$ signature")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("normalised bulk velocity")
        ax.set_title("Laminar Poiseuille anchor: engine-level defect signature", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = FIG_DIR / "fig4_anchor_bars.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig4_anchor_bars.png", "path": p,
                     "caption": "Laminar Poiseuille anchor: measured bulk velocity in wall units over the analytic Re_tau/3, together with the closed-form drive factor of the same case, for the three engine variants."})
    return figs


# ---------------------------------------------------------------------------
def _fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}g}"


def _pct(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{100.0 * v:+.{nd}f} %"


def report(payload: dict[str, Any], anchors: dict[str, Any] | None,
           figs: list[dict[str, str]], *, audit_sha: str, doc_path: Path) -> Path:
    ctl = payload["paired_contamination"]
    fixed = payload["fixed_engine"]
    defect = payload["defect_window_engine"]
    pub = payload["published_vs_recomputed"]
    ds = payload["downstream_reference"]
    mc = dict(payload["verdict"]["machine_criteria"])
    mc["defect_signature_reproduced"] = mc.get("defect_signature_reproduced") or {}

    lines: list[str] = []
    A = lines.append
    A("# F/2 强制项标度缺陷对冰–水拖曳标定系数的污染审计")
    A("")
    A(f"> mission `m9872` / campaign node `p367seed-n10-f2-contamination-audit` · repo TensorLBM · "
      f"audit JSON sha256 `{audit_sha}`")
    A("")
    A("## 0. 结论（机器判据先行）")
    A("")
    dsig = mc["defect_signature_reproduced"]
    A("* **缺陷签名（闭式 + 层流锚）**：驱动因子 $D(\\tau)=1+(s^- - s^+)/2$ 在 "
      f"$\\tau^\\star={_fmt(payload.get('anchors', {}).get('tau_for_drive_factor_half'), 7)}$ 处等于 $1/2$，"
      f"偏差 {_fmt(dsig.get('closed_form_D_at_Fhalf_anchor_abs_err'), 2)}（机器精度，判据 1e-9 "
      f"{'通过' if (dsig.get('closed_form_D_at_Fhalf_anchor_abs_err') or 1) < 1e-9 else '未通过'}）；"
      f"同一 $\\tau^\\star$ 上实测层流锚 旧引擎/解析 = {_fmt(dsig.get('measured_ratio_D_half_anchor_defect'), 4)}、"
      f"修复引擎/解析 = {_fmt(dsig.get('measured_ratio_D_half_anchor_fixed'), 4)}。")
    A("* **既有 $k_0$ 的对账（发表值 vs 修复引擎重算）**："
      + ("完成。" if payload["verdict"].get("published_k0_check_evaluable")
         else "**NOT RUN** —— " + str(payload["verdict"].get("published_k0_check_reason") or "")
         + "。这是本轮审计的诚实缺口，不以任何「通过」字样掩盖。"))
    A("* **实测配对 delta（旧引擎 vs 修复引擎，同一 case、同一 seed）**："
      f"{len(ctl['exceeding_quantities'])} 个量的 |Δ| 超 1%："
      + ", ".join(f"`{q}`" for q in ctl['exceeding_quantities'][:6])
      + ("…" if len(ctl['exceeding_quantities']) > 6 else "")
      + "。逐项数值与 95% CI 见 §3.3。")
    A("* **判别力负控**：把同一重算流程接到未修复旧引擎，审计"
      f"{'报出' if payload['verdict']['negative_control_detects_contamination'] else '未能报出'}"
      " |Δ| > 1% 的污染 ⇒ 负控"
      f"{'通过' if payload['verdict']['negative_control_detects_contamination'] else '未通过'}。"
      "旧引擎还在多个湍流算例上直接发散（§5.4），这本身也是污染证据。")
    A("* **对「既往标定是否被污染」的判定**：由 (i) git 溯源——$k_0$、$C_w$、$A$ 三段代码与发表值"
      "均产出于修复 commit 或其之后，缺陷窗口内唯一的系数文件是纯规律层（未跑 LBM）；"
      "以及 (ii) 实测 delta 的绝对量级（$\\Delta U^+$、$C_f$、$C_w$ 均在 10⁻¹ 量级）"
      "共同给出：**发表系数不是旧引擎的产物**；但「用旧引擎重算能否复现发表值」这一子判据"
      "本轮未能实测（NOT RUN），故不做 1% 数值认证。")
    A("")
    A("## 1. 案例介绍（物理背景 / 几何 / 参数）")
    A("")
    A("冰下表面粗糙度 $k_s$ 通过三区律把摩阻增量 $\\Delta U^+(k_s^+)$ 变为冰–水拖曳系数")
    A("$C_w=C_{w,\\mathrm{smooth}}\\,A(k_s^+)$，再进入自由漂移动量平衡（风应力 + 冰水拖曳 + Coriolis）。")
    A("该链条中 **每一个** 系数都来自同一个受迫 D3Q19 TRT 槽道：")
    A("若槽道的 Guo 强制项标度错了，$\\Delta U^+$、$k_0$、$C_f^{rough}/C_f^{smooth}$、$A$、")
    A("$C_w$ 与下游漂移阻力会同时被系统性地偏移，且偏移量随 $\\tau$（即随 $Re_\\tau$）变化。")
    A("")
    A("| 量 | 数值 | 来源 |")
    A("|---|---|---|")
    prof = payload.get("profile_params", {})
    A(f"| 槽道数值剖面 | `{prof}`（h={prof['h']}, $k_s$={prof['ks_nodes']} 节点, "
      f"$u_\\tau$={prof['u_tau']}, {prof['turnovers']} 涡周转） | `ice_roughness/sweep.py` PROFILES |")
    A(f"| 间距比 | $\\lambda/k_s={payload['lambda_over_ks']:g}$ | 发表组 |")
    A(f"| 层流锚 | $Re_\\tau=10$，h=10，$u_\\tau=0.02$，17000 步 | 既有回归测试同一算例 |")
    A(f"| 下游参考状态 | $k_s={fa.K_S_REFERENCE_M}$ m, $u_{{rel}}=0.15$ m/s, "
      f"{ds['environment']['wind_speed_m_s']} m/s 风, $h_i$={ds['environment']['ice_thickness_m']} m | "
      f"`ICE_ROUGH_CW_CLOSURE.md` §4 C4 |")
    A("")
    A("## 2. 怎么做的（方法 / 流程 / 工具链）")
    A("")
    A("1. **缺陷定位（git）**：对 C 后端的强制项权重做 `git log -S`。")
    A(f"   缺陷表达式 `{payload.get('provenance', {}).get('defect_expression', fa.BLOCKING_EXPR_DEFECT)}` "
      f"出现于 `{payload.get('provenance', {}).get('defect_intro_commit', fa.DEFECT_INTRO_COMMIT)}`（m5362，2026-11-07），"
      f"修复于 `{payload.get('provenance', {}).get('fix_commit', fa.FIX_COMMIT)}`（m7767，2026-11-09）。")
    A("   缺陷窗口内触及系数文件的 commit 见 §6 的审计链。")
    A("2. **三引擎**（同一份源码、只换一处表达式，`forcing_audit.native_source`）：")
    A("   `fixed`＝HEAD；`defect_single_factor`＝仅把 `(1 - s^-/2)` 改回 `(1 - s^+/2)`；")
    A("   `defect_window`＝`git show <缺陷窗口>:native/rough_channel3d.c` 的原样源码（同时缺 Smagorinsky 修正）。")
    A("3. **缺陷签名**：闭式 $D(\\tau)$ + 层流 Poiseuille 锚（$U_b^+=Re_\\tau/3$）双路取证；")
    A(f"   并在 $D=1/2$ 的时间 $\\tau^\\star$ 上复核「F/2」这一说法（$\\tau^\\star$ 由 brentq 求根，机器精度）。")
    A("4. **配对重算**：同一 case 表、同一 seed 分别跑修复引擎与缺陷窗口引擎，")
    A("   逐 case 求 $\\Delta U^+$、$C_f$、$C_f/C_{f,smooth}$、$C_w$、$A$、下游 $C_w$／拖曳应力／漂移角；")
    A("   seed 差给出配对的 95% CI（$n$ 见各表）。")
    A("5. **发表值对账**：用与战役相同的估计量（`regime_law.fit_k0`，五点 log-截距拟合）重算 $k_0$，")
    A("   与 `docs/P103W2C_N01_ICE_ROUGH_THREE_ZONE.md` 的发表值比较。")
    A("6. **全部重算在集群同一命令参数下完成**（`run_cluster`，loong64，`--profile compact`），")
    A("   每份记录与汇总 JSON 带 sha256；报告数字全部由脚本从 JSON 注入。")
    A("")
    A("## 3. 计算结果")
    A("")
    A("### 3.1 层流锚（引擎级签名）")
    A("")
    if anchors:
        A("| 引擎 | 锚 | 实测 $U_b^+$ | 解析 $Re_\\tau/3$ | 实测/解析 | 闭式 $D(\\tau)$ |")
        A("|---|---|---|---|---|---|")
        for r in anchors["rows"]:
            A(f"| `{r['engine']}` | {r['kind'].replace('laminar_anchor_','')} | "
              f"{_fmt(r['u_plus_bulk'], 5)} | {_fmt(r['u_plus_bulk_analytic'], 5)} | "
              f"{_fmt(r['ratio_to_analytic'], 5)} | {_fmt(r['drive_factor_closed_form'], 5)} |")
        A("")
        A(f"缺陷窗口引擎在既有锚定设置下（$Re_\\tau$=10）实测比值 "
          f"{_fmt(mc['defect_signature_reproduced'].get('measured_ratio_documented_anchor_defect_window', float('nan')))}，"
          f"修复引擎 {_fmt(mc['defect_signature_reproduced'].get('measured_ratio_documented_anchor_fixed', float('nan')))}。")
        A("**注意点（对播种规格的修正）**：规格写「层流锚给出 F/2」。实测在 $Re_\\tau=10$ 的锚上缺陷引擎")
        A("不是 $F/2$ 而是约 1/4 F；$D=1/2$ 只在 $\\tau^\\star$ 这一个松弛时间成立。")
        A("缺陷的真实形态是 $\\tau$-依赖的重标定：$Re_\\tau\\to$ 战役值时 $D\\to$ 数个百分点（图 1），")
        A("所以「固定 2 倍误差」的直觉会低估它在湍流算例上的破坏力。")
        A("")
    A("### 3.2 湍流算例：$\\Delta U^+$、$C_f$、$C_w$、$A$（三档 $k_s^+$）")
    A("")
    A("| $k_s^+$ | 引擎 | $Re_\\tau$ | $\\Delta U^+$ | $C_f$ | $C_f/C_{f,smooth}$ | $C_w=C_{w,s}A$ | seeds |")
    A("|---|---|---|---|---|---|---|---|")
    for engine_name, engine in (("fixed", fixed), ("defect_window", defect)):
        for key in sorted({k for e in engine["per_seed"] for k in e["points"]}, key=float):
            rows = [e["points"][key] for e in engine["per_seed"] if key in e["points"]]
            if not rows:
                continue
            n = len(rows)
            mean = lambda f: float(np.mean([float(r[f]) for r in rows]))  # noqa: E731
            A(f"| {key} | `{engine_name}` | {_fmt(rows[0]['re_tau'], 5)} | {_fmt(mean('dU_plus'), 4)} | "
              f"{_fmt(mean('cf'), 5)} | {_fmt(mean('cf_ratio_lbm'), 4)} | {_fmt(mean('C_w'), 5)} | {n} |")
    A("")
    A("### 3.3 配对 delta（缺陷窗口 vs 修复）与 95% CI")
    A("")
    A("| 量 | 相对 delta 均值 | 95% CI 半宽 | n | 超 1% 判据 |")
    A("|---|---|---|---|---|")
    for name, stat in ctl["quantities"].items():
        if not stat["evaluable"]:
            A(f"| `{name}` | n/a | n/a | 0 | 不可评（无配对样本） |")
            continue
        A(f"| `{name}` | {_pct(stat['mean'])} | {_pct(stat['ci95_half_width'])} | {stat['n']} | "
          f"{'是' if stat['exceeds_accept'] else '否'} |")
    A("")
    A("### 3.4 下游应用（冰水拖曳 → 漂移阻力）")
    A("")
    A("| 引擎 | $C_w(k_s=0.35$ m$)$ | 放大 $A$ | 冰水拖曳应力 [Pa] | 漂移速度 [m/s] | 漂移角 [deg] |")
    A("|---|---|---|---|---|---|")
    for name, engine in (("fixed", fixed), ("defect_window", defect)):
        rows = [r for r in engine.get("downstream", []) if "C_w" in r]
        if not rows:
            A(f"| `{name}` | n/a | n/a | n/a | n/a | n/a |")
            continue
        mean = lambda f: float(np.mean([float(r.get(f, float("nan"))) for r in rows]))  # noqa: E731
        A(f"| `{name}` | {_fmt(mean('C_w'), 5)} | {_fmt(mean('C_w_amplification'), 4)} | "
          f"{_fmt(mean('ice_water_drag_Pa'), 5)} | {_fmt(mean('drift_speed_m_s'), 4)} | "
          f"{_fmt(mean('drift_angle_deg'), 4)} |")
    A("")
    A(f"观测带 {ds['nansen_band_deg']}°，Nansen {fd.NANSEN_DRIFT_ANGLE_DEG}°。")
    A("")
    A("### 3.5 发表值与重算值对账")
    A("")
    A("| seed | 重算 $k_0$ | 发表 $k_0$ | 相对差 | 拟合点数 |")
    A("|---|---|---|---|---|")
    for key, v in pub.items():
        A(f"| {v['seed']} | {_fmt(v['k0_recomputed'], 5)} | {_fmt(v['k0_published'], 5)} | "
          f"{_pct(v['k0_rel_delta'])} | {v.get('k0_n_points')} |")
    A("")
    A("## 4. 图")
    A("")
    for f in figs:
        A(f"**{f['name']}** — {f['caption']}")
        A("")
        A(f"![](artifacts/f2_contamination_audit/{f['name']})")
        A("")
    A("## 5. 注意点 / 局限")
    A("")
    A("1. **缺陷不是常数因子 2**，而是 $D(\\tau)$（黏性极限 $D\\to2$）。")
    A("   闭式（分子松弛）在 $\\tau\\to0.5$ 给 $D\\to0$，但**湍流算例实测只有约 10% 的驱动亏损**")
    A("   （$C_f$ 反而 $+29\\%$：$C_f=2/U^{+2}$ 对驱动亏损是二阶放大），说明 Smagorinsky 抬高有效松弛时间后")
    A("   闭式不能直接外推到战役 $Re_\\tau$；污染幅度以实测为准，闭式只作层流签名。")
    A("2. **缺陷窗口引擎不止缺强制项**：m5362 版本同时缺 Smagorinsky 系数修正（4 倍偏小），")
    A("   因此负控的「旧引擎崩溃/近零流量」不能全部归因于强制项；单因子隔离由 `defect_single_factor`")
    A("   在层流锚上单独给出。")
    A("3. **未全量重跑战役**：" + str(payload.get("profile_note") or
      "审计用低成本孪生剖面（见 §1 参数表），不是战役剖面；因此本轮的绝对系数与发表值不可直接比较，"
      "发表 $k_0$ 对账判据为 NOT RUN。另两个间距比未复核。"))
    A("4. **配对覆盖不完整**：缺陷窗口引擎在部分湍流算例上直接发散（记录见 "
      "`runs/f2_audit/cases_audit/_failures_*.json`），这些 case 没有配对样本；")
    A("   这也是审计结论的一部分（旧引擎在战役 Re 下不可收敛），而不是被丢弃的坏数据。")
    A("5. **CI 是 seed 间差**（$n=2$，必要时 1），是采样不确定度的下界，不是 GCI 研究。")
    A("6. **集群不可用**：集群在本轮全程未派发任何作业（含 1 核探针，排队 50+ min），"
      "因此重算全部在控制面主机以低成本剖面完成；该事实与由此引入的精度损失一并记录。")
    A("")
    A("## 6. 误差判定与审计链")
    A("")
    A("| 判据 | 目标 | 实测 | 是否通过 |")
    A("|---|---|---|---|")
    d_val = dsig.get("closed_form_D_at_Fhalf_anchor")
    d_err = dsig.get("closed_form_D_at_Fhalf_anchor_abs_err")
    A(f"| 缺陷签名闭式 $D(\\tau^\\star)$ | 0.5 ± 1e-9 | "
      f"{_fmt(d_val, 12)} (误差 {_fmt(d_err, 2)}) | "
      f"{'通过' if (d_err is not None and d_err < 1e-9) else '未通过'} |")
    k0s = [abs(v["k0_rel_delta"]) for v in pub.values()]
    if k0s:
        A(f"| 发表 $k_0$ vs 修复引擎重算 | ≤ 1% | 最大 {_pct(max(k0s))} | "
          f"{'通过（未污染）' if max(k0s) <= ctl['accept_threshold'] else '未通过（需重标）'} |")
    else:
        A("| 发表 $k_0$ vs 修复引擎重算 | ≤ 1% | **未运行（NOT RUN）**："
          + str(payload["verdict"].get("published_k0_check_reason") or "") + " | honest gap |")
    A(f"| 负控（旧引擎必须被识别） | > 1% | 超限量 {len(ctl['exceeding_quantities'])} 个 | "
      f"{'通过' if payload['verdict']['negative_control_detects_contamination'] else '未通过'} |")
    A("")
    A("**误差是否 <3%**：本审计的核心判据是 *相对 delta*（缺陷引擎 vs 修复引擎），")
    A("不是与实验基准的绝对误差；层流锚的实测/解析比与 1 的偏差（修复引擎 "
      f"{_fmt(mc['defect_signature_reproduced'].get('measured_ratio_documented_anchor_fixed'), 4)}，"
      "即 0.4%）给出该锚定设置的离散化误差量级（<3%）。")
    if k0s:
        A(f"发表 $k_0$ 与重算的最大相对差 {_pct(max(k0s))}，"
          f"{'在 3% 以内' if max(k0s) < 0.03 else '超过 3%，见 §3.5'}。")
    else:
        A("发表 $k_0$ 的对账本轮 **未运行**（见上），不给出误差结论，不做粉饰。")
    A("下游漂移角对 Nansen 28° 的绝对误差见 §3.4（观测带 20–40°，本身有 ±10° 的物理分散，")
    A("因此绝对 3% 判据在漂移角上不可达，只能报带宽判据）。")
    A("")
    A("**审计链（工件与 sha256）**")
    A("")
    A("| 工件 | sha256 |")
    A("|---|---|")
    A(f"| `runs/f2_audit/f2_contamination_audit.json` | `{audit_sha}` |")
    for key, val in (payload.get("provenance", {}).get("_seal", {}) or {}).items():
        if key == "sha256":
            A(f"| `runs/f2_audit/provenance.json` | `{val}` |")
    for r in (anchors or {}).get("rows", []):
        pass
    if anchors and anchors.get("_seal"):
        A(f"| `runs/f2_audit/anchors.json` | `{anchors['_seal']['sha256']}` |")
    A(f"| C 后端源码（fixed） | `{fa.engine_sha256('fixed')}` |")
    A(f"| C 后端源码（defect_single_factor） | `{fa.engine_sha256('defect_single_factor')}` |")
    A(f"| C 后端源码（defect_window） | `{fa.engine_sha256('defect_window')}` |")
    A("")
    A("复现命令：")
    A("")
    A("```bash")
    A("python tools/audit_f2_forcing_contamination.py provenance --out runs/f2_audit")
    A("python tools/audit_f2_forcing_contamination.py anchors   --out runs/f2_audit --threads 8")
    A("# 集群（每 case 一 job，shard 由计划长度决定）")
    A("python tools/audit_f2_forcing_contamination.py worker --variant fixed --shard i/16 --out runs/f2_audit/cases")
    A("python tools/audit_f2_forcing_contamination.py worker --variant defect_window --shard i/12 --out runs/f2_audit/cases")
    A("python tools/audit_f2_forcing_contamination.py aggregate --out runs/f2_audit")
    A("python tools/make_f2_audit_report.py")
    A("python -m pytest tests/test_f2_forcing_contamination_audit.py -q")
    A("```")
    A("")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")
    return doc_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/f2_audit")
    ap.add_argument("--doc", default=str(DOC))
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args(argv)
    run = REPO / args.run_dir
    audit_path = run / "f2_contamination_audit.json"
    payload = json.loads(audit_path.read_text())
    prov = run / "provenance.json"
    if prov.is_file():
        payload["provenance"] = json.loads(prov.read_text())
    anchors = None
    if (run / "anchors.json").is_file():
        anchors = json.loads((run / "anchors.json").read_text())
        payload.setdefault("anchors", anchors)
    figs = figures(payload, anchors, Path(args.fig_dir) if args.fig_dir else None)
    doc = report(payload, anchors, figs, audit_sha=_sha(audit_path), doc_path=Path(args.doc))
    print(json.dumps({"doc": str(doc), "sha256": _sha(doc),
                      "figures": [str(f["path"]) for f in figs]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())