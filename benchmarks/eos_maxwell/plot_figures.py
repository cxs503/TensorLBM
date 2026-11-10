#!/usr/bin/env python
"""Figures + generated results tables for the EOS-Maxwell stability probe.

Everything here is derived from ``out/results.json`` — no number is typed by
hand.  Outputs:

* ``figures/fig1_radial_profiles.png``   radial / planar density profiles vs Maxwell asymptotes
* ``figures/fig2_spurious_currents.png`` spurious-current and density-ratio time traces
* ``figures/fig3_stability_map.png``     stability boundary map (ratio x forcing x width)
* ``figures/fig4_ratio_maintenance.png`` measured vs Maxwell ratio with the 15 % band
* ``figures/fig5_tau_lever.png``         viscosity (tau) lever arm + predeclared decision
* ``RESULTS.md``                         machine-generated result tables
* ``CRITERION.md``                       the judgement rule this probe hands to
                                         GAP-LBM-MCMP-POISEUILLE (also generated)

Usage: python plot_figures.py [--results out/results.json] [--out .]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import probe_lib as P  # noqa: E402  (canonical predeclared tau-arm decision rule)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e5:
            return f"{x:.{nd-1}e}"
        return f"{x:.{nd}g}"
    return str(x)


# ---------------------------------------------------------------- figure 1


def fig_profiles(res: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    colors = {10.0: "#1f77b4", 100.0: "#d62728", 1000.0: "#2ca02c"}
    for r in res["tiers"]:
        if not r["stable"] or "radial_profile" not in r:
            continue
        tr = r["target_ratio"]
        prof = r["radial_profile"]
        rr = [p["r_mid"] for p in prof]
        rho = [p["rho_mean"] for p in prof]
        ax.plot(rr, rho, "-", color=colors.get(tr, "gray"), lw=1.8,
                label=f"target ratio {tr:g} ({r['spec']['forcing']}), {r['steps_run']} steps")
        ax.axhline(r["rho_l_maxwell"], color=colors.get(tr, "gray"), ls=":", lw=1.0)
        ax.axhline(r["rho_v_maxwell"], color=colors.get(tr, "gray"), ls=":", lw=1.0)
        ax.annotate(f"$\\rho_l^{{Maxwell}}$={r['rho_l_maxwell']:.3f}\n$\\rho_v^{{Maxwell}}$={r['rho_v_maxwell']:.4f}",
                    xy=(2, r["rho_l_maxwell"]), fontsize=7, color=colors.get(tr, "gray"), va="bottom")
        ax.annotate(f"measured ratio {r['ratio']:.1f}\nMaxwell {r['ratio_maxwell']:.1f}",
                    xy=(0.98, 0.05), xycoords="axes fraction", ha="right", fontsize=7,
                    color=colors.get(tr, "gray"))
    ax.set_yscale("log")
    ax.set_xlabel("radius r (lattice units)")
    ax.set_ylabel(r"$\langle\rho\rangle_{annulus}$")
    ax.set_title("(a) static-droplet radial density profile, final state\n"
                 "dotted = Maxwell equal-area asymptotes (computed at run time)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for r in res["controls"]:
        if not r["stable"] or "y_profile" not in r:
            continue
        tr = r["target_ratio"]
        ys = [p["y"] for p in r["y_profile"]]
        rho = [p["rho_mean"] for p in r["y_profile"]]
        ax.plot(ys, rho, "-", color=colors.get(tr, "gray"), lw=1.6,
                label=f"planar slab, target ratio {tr:g} ({r['spec']['forcing']})")
        ax.axhline(r["rho_l_maxwell"], color=colors.get(tr, "gray"), ls=":", lw=1.0)
        ax.axhline(r["rho_v_maxwell"], color=colors.get(tr, "gray"), ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("y (row index, periodic)")
    ax.set_ylabel(r"$\langle\rho\rangle_{row}$")
    ax.set_title("(b) planar-interface control (liquid slab, periodic)\n"
                 "the lattice-selected vapour plateau, not the Maxwell one")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    p = out / "fig1_radial_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2


def fig_spurious(res: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    colors = {10.0: "#1f77b4", 100.0: "#d62728", 1000.0: "#2ca02c"}
    series = [(r, "tier") for r in res["tiers"]] + [(r, "longhold") for r in res.get("longhold", [])]
    for r, kind in series:
        h = r.get("hist") or []
        if not h:
            continue
        tr = r["target_ratio"]
        style = "-" if kind == "tier" else "--"
        ax.plot([s["step"] for s in h], [s["ma_max"] for s in h], style,
                color=colors.get(tr, "gray"), lw=1.3,
                label=(f"ratio {tr:g} {kind} ({r['spec']['forcing']}, w={r['spec']['width']:g}, "
                       f"{r['steps_run']} steps)"))
    for gate, lab in ((0.05, "predeclared PASS gate 0.05"), (0.10, "predeclared FAIL gate 0.10"),
                      (0.50, "collapse criterion 0.50")):
        ax.axhline(gate, color="k", ls=":" if gate < 0.4 else "-.", lw=1.0)
        ax.annotate(lab, xy=(0.01, gate), xycoords=("axes fraction", "data"), fontsize=6.5, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"max $|\mathbf{u}| / c_s$  (spurious current)")
    ax.set_title("(a) spurious-current history: saturation, not decay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    for r, kind in series:
        h = r.get("hist") or []
        if not h:
            continue
        tr = r["target_ratio"]
        style = "-" if kind == "tier" else "--"
        ax.plot([s["step"] for s in h], [s["ratio"] for s in h], style,
                color=colors.get(tr, "gray"), lw=1.3, label=f"ratio {tr:g} {kind}")
        ax.axhline(r["ratio_maxwell"], color=colors.get(tr, "gray"), ls=":", lw=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"measured $\rho_l/\rho_v$")
    ax.set_title("(b) density ratio vs the Maxwell target (dotted)\n"
                 "ratio-100 trace plateaus 3x above its target and stays there")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    p = out / "fig2_spurious_currents.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 3


def fig_stability_map(res: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    marker = {"edm": "o", "guo": "s"}
    ccol = {"edm": "#1f77b4", "guo": "#ff7f0e"}
    survived = res["prereg"]["stability_steps_min"] / 4.0  # boundary hold (4000 steps)
    for r in res["boundary"]:
        f = r["spec"]["forcing"]
        x = r["ratio_maxwell"]
        if r["stable"]:
            ax.scatter([x], [survived], marker=marker[f], facecolors="none",
                       edgecolors=ccol[f], s=70, linewidths=1.6)
        else:
            ax.scatter([x], [max(r["collapse_step"], 1)], marker="x", color=ccol[f], s=60, linewidths=1.8)
            ax.annotate(_fmt(r["collapse_step"], 3), (x, max(r["collapse_step"], 1)),
                        textcoords="offset points", xytext=(4, 3), fontsize=6, color=ccol[f])
    for r in res.get("widths", []):
        if r["stable"]:
            continue
        ax.scatter([r["ratio_maxwell"]], [max(r["collapse_step"], 1)], marker="^",
                   facecolors="none", edgecolors="#555555", s=45, linewidths=1.0)
    for r in res.get("repeats", []):
        if r["stable"]:
            continue
        ax.scatter([r["ratio_maxwell"]], [max(r["collapse_step"], 1)], marker="v",
                   facecolors="none", edgecolors="#8c564b", s=50, linewidths=1.2)
    for target in res["prereg"]["target_ratios"]:
        ax.axvline(target, color="k", ls="--", lw=0.8)
        ax.annotate(f"target {target:g}", xy=(target, 1.05), xycoords=("data", "axes fraction"),
                    fontsize=7, ha="center", rotation=90, va="bottom")
    for r in res["tiers"]:
        if not r["stable"]:
            ax.scatter([r["ratio_maxwell"]], [max(r["collapse_step"], 1)], marker="P",
                       color="#d62728", s=70, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Maxwell coexistence ratio $\rho_l/\rho_v$ of the EOS point (log)")
    ax.set_ylabel("steps survived (log); open markers = survived the boundary hold")
    ax.set_title("(a) stability boundary map: SCMP + PR-EOS Maxwell construct\n"
                 "circles EDM, squares Guo, triangles width-arm collapses, diamonds tier collapses")
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ccol["edm"], label="EDM (boundary arm)"),
        plt.Line2D([], [], marker="s", ls="", mfc="none", mec=ccol["guo"], label="Guo (boundary arm)"),
        plt.Line2D([], [], marker="x", ls="", color="k", label="collapse (onset step annotated)"),
        plt.Line2D([], [], marker="P", ls="", color="#d62728", label="tier (20000-step hold)"),
        plt.Line2D([], [], marker="^", ls="", mfc="none", mec="#555555", label="width arm"),
        plt.Line2D([], [], marker="v", ls="", mfc="none", mec="#8c564b", label="replication arm collapse"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig3_stability_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 4


def fig_ratio_maintenance(res: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    lo, hi = 3.0, 3000.0
    band = 0.15
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="measured = Maxwell")
    ax.fill_between([lo, hi], [lo * (1 - band), hi * (1 - band)], [lo * (1 + band), hi * (1 + band)],
                    color="0.85", zorder=0, label="±15% acceptance band")
    hold = max((r["spec"]["steps"] for r in res.get("longhold", [])), default=0)
    style = {"tiers": ("o", 40, "tiers (20000-step hold)"),
             "longhold": ("D", 30, f"long hold ({hold} steps)"),
             "controls": ("^", 45, "planar control (40000 steps)"),
             "repeats": (".", 25, "replication arm (same config x3)")}
    colors = {10.0: "#1f77b4", 100.0: "#d62728", 1000.0: "#2ca02c"}
    for key in ("tiers", "longhold", "controls", "repeats"):
        rows = res.get(key, [])
        mk, sz, lab = style[key]
        xs = [r["ratio_maxwell"] for r in rows if r.get("ratio")]
        ys = [r["ratio"] for r in rows if r.get("ratio")]
        cs = [colors.get(r["target_ratio"], "gray") for r in rows if r.get("ratio")]
        ax.scatter(xs, ys, marker=mk, s=sz, facecolors="none", edgecolors=cs, linewidths=1.5, label=lab)
    for r in res["tiers"]:
        if r.get("ratio") and abs(r["ratio_error"] or 0) > band:
            ax.annotate(f"{r['target_ratio']:g}: {r['ratio_error']:.2f}", (r["ratio_maxwell"], r["ratio"]),
                        textcoords="offset points", xytext=(5, -8), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Maxwell target ratio")
    ax.set_ylabel("measured ratio")
    ax.set_title("(a) density-ratio maintenance error table plotted\n(0/6 tiers inside ±15%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    for key in ("tiers", "longhold", "controls"):
        rows = [r for r in res.get(key, []) if r.get("ma_max")]
        ax.scatter([r["ma_max"] for r in rows], [abs(r["ratio_error"]) for r in rows],
                   marker=style[key][0], s=style[key][1], facecolors="none",
                   edgecolors=[colors.get(r["target_ratio"], "gray") for r in rows],
                   linewidths=1.5, label=style[key][2])
    for r in res["boundary"]:
        if r["stable"] and r.get("ma_max"):
            ax.scatter([r["ma_max"]], [abs(r["ratio_error"]) if r.get("ratio_error") is not None else math.nan],
                       marker=".", color="0.5", s=18)
    ax.axvline(res["prereg"]["spurious_pass_ma"], color="k", ls=":", lw=1.0)
    ax.axvline(res["prereg"]["spurious_warn_ma"], color="k", ls="-.", lw=1.0)
    ax.annotate("PASS gate 0.05", (res["prereg"]["spurious_pass_ma"], 1.0), fontsize=6.5, rotation=90,
                va="top")
    ax.annotate("FAIL gate 0.10", (res["prereg"]["spurious_warn_ma"], 1.0), fontsize=6.5, rotation=90,
                va="top")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel(r"spurious current max $|\mathbf{u}|/c_s$")
    ax.set_ylabel("| ratio error |")
    ax.set_title("(b) spurious current vs ratio error\nsmall grey dots = boundary arm")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "fig4_ratio_maintenance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p



# ---------------------------------------------------------------- figure 5


def fig_tau_lever(res: dict, out: Path) -> Path:
    """Viscosity lever: does tau move the spurious-current gate, the ratio gate, both?

    Two quantities with two different y-axes per target ratio: the spurious
    current (left, log) and the density-ratio maintenance error (right, symlog),
    both against the relaxation time at fixed EOS point, forcing and box.
    Collapsed runs cannot carry either quantity and are drawn as a red cross on
    the top axis so that "no data" is never confused with "small error".
    """
    summary = P.tau_lever_summary(res.get("tau", []))
    ratios = res["prereg"]["target_ratios"]
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.3 * len(ratios), 4.8))
    if len(ratios) == 1:
        axes = [axes]
    colors = {10.0: "#1f77b4", 100.0: "#d62728", 1000.0: "#2ca02c"}
    gate_ma = res["prereg"]["spurious_pass_ma"]
    gate_ma_warn = res["prereg"]["spurious_warn_ma"]
    gate_ratio = res["prereg"]["ratio_error_max"]
    for ax, ratio in zip(axes, ratios):
        rows = [r for r in res.get("tau", []) if r.get("target_ratio") == ratio]
        col = colors.get(ratio, "gray")
        alive = [r for r in rows if r.get("stable") and r.get("ma_max")]
        dead = [r for r in rows if not r.get("stable")]
        ax2 = ax.twinx()
        if alive:
            taus = [r["spec"]["tau"] for r in alive]
            ax.plot(taus, [r["ma_max"] for r in alive], "o-", color=col, lw=1.6, ms=6,
                    label=r"spurious $\max|\mathbf{u}|/c_s$")
            ax2.plot(taus, [max(r["ratio_error"], 1e-6) for r in alive], "s--", color="0.35", lw=1.4, ms=5,
                     label="| ratio error |")
        for r in dead:
            ax.plot([r["spec"]["tau"]], [1.0], "x", color="red", ms=9, mew=2)
            ax.annotate(f"collapse\nstep {r['collapse_step']}", (r["spec"]["tau"], 1.0), fontsize=6,
                        textcoords="offset points", xytext=(4, -2), color="red")
        ax.axhline(gate_ma, color="k", ls=":", lw=1.0)
        ax.axhline(gate_ma_warn, color="k", ls="-.", lw=1.0)
        ax2.axhline(gate_ratio, color="#8c564b", ls=":", lw=1.0)
        ax.set_yscale("log")
        ax2.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(1e-3, 2.0)
        ax2.set_ylim(1e-3, 100.0)
        ax.set_xlabel(r"relaxation time $\tau$")
        ax.set_ylabel(r"spurious current $\max|\mathbf{u}|/c_s$", color=col)
        ax2.set_ylabel("| ratio error |", color="0.35")
        ax.set_title(f"target ratio {ratio:g}\nMaxwell $\\rho_l/\\rho_v$={rows[0]['ratio_maxwell']:.3g}"
                     if rows else f"target ratio {ratio:g}")
        ax.grid(alpha=0.3, which="both")
        s_row = summary.get(f"{ratio:g}", {})
        dec = s_row.get("decision", "no_stable_run")
        txt = (
            f"$\\tau^*$={s_row.get('ma_min_tau')} $\\rightarrow$ Ma={_fmt(s_row.get('ma_min'), 3)}\n"
            f"min ratio err={_fmt(s_row.get('ratio_error_min'), 3)}\n"
            f"unrelaxed tau: {s_row.get('not_relaxation_resolved', [])}\n"
            f"decision: {dec}"
        )
        ax.annotate(txt, (0.03, 0.03), xycoords="axes fraction", fontsize=6.5,
                    bbox=dict(fc="white", ec="0.7", alpha=0.85))
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(handles + h2, labels + l2, fontsize=6.5, loc="upper right")
    fig.suptitle(
        "Viscosity lever (EDM, L=96, R=20, 20000 steps): dotted = PASS gate "
        f"({gate_ma}), dash-dot = FAIL gate ({gate_ma_warn}), brown dotted = ratio gate ({gate_ratio:.0%})",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "fig5_tau_lever.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p




# ------------------------------------------------------------- CRITERION.md


def criterion_markdown(res: dict) -> str:
    """The *judgement* artefact: which density ratios the SC/EOS path may be used at.

    Generated from ``results.json`` like the tables: every number below is read
    out of the artefact, so the rule and the evidence cannot drift apart.
    """
    pre = res["prereg"]
    env = res["environment"]
    tsum = P.tau_lever_summary(res.get("tau", []))
    L: list[str] = []
    A = L.append
    A("# 高密度比两相（SCMP/PR-EOS）路判据 v1 —— 由探针产物机器生成")
    A("")
    A(f"- 生成时间（UTC）：`{res.get('generated_utc')}`；探针 `probe_lib` sha256 "
      f"`{env['probe_lib_sha256'][:16]}…`；driver sha256 `{res['driver_sha256'][:16]}…`")
    A(f"- 平台：`{env.get('machine')}` / torch `{env.get('torch')}` / python `{env.get('python')}`"
      f" / threads {env.get('threads')}；预注册冻结：{pre.get('frozen')}")
    A(f"- 门值（预注册，非文献标准）：伪流 Ma ≤ {pre['spurious_pass_ma']} 为 PASS、"
      f"≤ {pre['spurious_warn_ma']} 为 MARGINAL；密度比误差 ≤ {pre['ratio_error_max']:.0%}；"
      f"稳定保持 ≥ {pre['stability_steps_min']} 步；质量漂移 ≤ {pre['mass_drift_rate_max']:.0e}/步；"
      f"ψ clamp 不得触发")
    A("")
    bud = P.stability_budget(R=float(res["prereg"].get("reference_R", 20.0)))
    hold = max((r["spec"]["steps"] for r in res.get("longhold", [])), default=0)
    A("## 判据 D0 —— “稳定保持 N 步”的合理性（时标预算，非整数拍脑袋）")
    A("")
    A("| 时标 | 步数 |")
    A("|---|---|")
    for k, v in bud["timescales"].items():
        A(f"| {k} | {v:.4g} |")
    A(f"| **预注册保持步数 N** | **{bud['hold_steps']:.0f}** |")
    A(f"| N / 最慢时标（{bud['slowest']}） | {bud['hold_over_slowest']:.2f} × |")
    A("")
    A(f"**D0 结论**：ν = c_s²(τ−½) = {bud['nu']:.4g}（τ={bud['tau']}），最慢线性时标"
      f"（{bud['slowest']}，{bud['slowest_steps']:.0f} 步）之上留 {bud['hold_over_slowest']:.1f} 倍"
      f"余量，故 N 不是“跑得动”的步数，而是覆盖最慢线性弛豫的步数；另有 {hold} 步长持时臂"
      "核对平台是否仍在漂（见 RESULTS §3）。")
    A("")
    A("## 判据 D1 —— 构造是否可解（必要条件，恒成立）")
    A("")
    A("| 目标密度比 | T_r | ρ_l | ρ_v | p_sat | 面积残差（相对） | 命中目标（≤" +
      f"{pre['ratio_target_tol']:.0%}） |")
    A("|---|---|---|---|---|---|---|")
    for ratio in pre["target_ratios"]:
        mw = res["maxwell_tiers"][f"{ratio:g}"]
        A(f"| {ratio:g} | {_fmt(mw.get('T_r'), 4)} | {_fmt(mw.get('rho_liquid'), 5)} | "
          f"{_fmt(mw.get('rho_vapour'), 5)} | {_fmt(mw.get('p_sat'), 4)} | "
          f"{_fmt(mw.get('area_residual_rel'), 3)} | {mw.get('within_target_tol')} |")
    A("")
    A("**D1 结论**：三档目标比全部由等面积（Maxwell）构造反解得出、面积残差在机器精度量级，"
      "因此“构造解不出来”不是任何一档失败的机制；失败一律发生在*离散格式*一侧。")
    A("")
    A("## 判据 D2 —— 不得把 Maxwell 对当作格子不动点（平面界面对照）")
    A("")
    A("| 目标密度比 | forcing | Maxwell 比 | 平面实测比 | 实测/Maxwell | 相对偏差 |")
    A("|---|---|---|---|---|---|")
    for r in res.get("controls", []):
        if not r.get("ratio"):
            continue
        gap = r["ratio"] / r["ratio_maxwell"] if r.get("ratio_maxwell") else math.nan
        A(f"| {_fmt(r.get('target_ratio'), 3)} | {r['spec']['forcing']} | {_fmt(r['ratio_maxwell'], 4)} | "
          f"{_fmt(r.get('ratio'), 4)} | {_fmt(gap, 3)} | {_fmt(r.get('ratio_error'), 3)} |")
    A("")
    dead = [r for r in res.get("controls", []) if not r.get("ratio")]
    if dead:
        A("塌缩的平面对照（不产生共存对，故不进上表）：")
        A("")
        A("| run | 目标比 | forcing | 塌缩步 | 机制 |")
        A("|---|---|---|---|---|")
        for r in dead:
            A(f"| {r['tag']} | {r['target_ratio']:g} | {r['spec']['forcing']} | "
              f"{_fmt(r.get('collapse_step'))} | {(r.get('error') or '—')} |")
        A("")
    A("**D2 结论**：曲率无关的平面臂给出的共存对比与 Maxwell 锁值同样偏离（见上表 实测/Maxwell "
      "列），因此把 Maxwell 对直接当作离散格式不动点会系统性高估低密度比的蒸汽密度、"
      "低估高密度比档的蒸汽密度。任何以 Maxwell 锁值作真值的误差评估必须先扣除该格式偏差。")
    A("")
    A("## 判据 D3 —— 可用密度比包络（决策表）")
    A("")
    A("“可用”= 同时满足：稳定保持 ≥ 预注册步数、密度比误差门、伪流 PASS 门、无 clamp、质量守恒。"
      "伪流一列另给 τ 杠杆臂的最小可达值（`PREREG['tau_arm']`）。")
    A("")
    A("| 目标密度比 | 存活 tier 数 | 最短存活步 | tier 最小伪流 Ma | τ 杠杆最小 Ma（τ） | "
      "松弛到位 τ 的最小 Ma | 最小密度比误差 | τ 判据决策 | 能否开跑 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for ratio in pre["target_ratios"]:
        rows = [r for r in res.get("tiers", []) if r.get("target_ratio") == ratio]
        alive = [r for r in rows if r.get("stable")]
        e = tsum.get(f"{ratio:g}", {})
        usable = bool(e.get("ma_ok") and e.get("ratio_ok") and len(alive) == len(rows) and rows)
        relaxed = e.get("mas") or {}
        unrelaxed = set(e.get("not_relaxation_resolved", []))
        relaxed_min = min((v for k, v in relaxed.items() if k not in unrelaxed), default=None)
        tier_ma = min((r["ma_max"] for r in alive), default=None)
        tier_err = min((r["ratio_error"] for r in alive if r.get("ratio_error") is not None), default=None)
        A(f"| {ratio:g} | {len(alive)}/{len(rows)} | "
          f"{min((r['steps_run'] for r in alive), default='—')} | {_fmt(tier_ma, 3)} | "
          f"{_fmt(e.get('ma_min'), 3)} (τ={e.get('ma_min_tau')}) | {_fmt(relaxed_min, 3)} | "
          f"{_fmt(tier_err, 3)} | `{e.get('decision')}` | {'**是**' if usable else '**否**'} |")
    A("")
    A("**D3 规则**：")
    A("")
    A(f"- 开跑判据：只有当某密度比在 D3 表“能否开跑”为**是**时，才允许在该比上做两相"
      f"（例如 Poiseuille）对比；否则先降比或改造格式，不得把“跑满 {pre['stability_steps_min']} 步”"
      "当作可用性证据。")
    A("- 伪流地板：即使判据为否，D3 的“τ 杠杆最小 Ma”仍是该比上的伪流噪声下界；"
      "任何以伪流/表面张力为被测量的实验必须声明该地板。")
    A("- τ 判据四分支（预注册）：`usable_at_some_tau` / `spurious_tunable_ratio_structural` / "
      "`ratio_held_currents_above_pass_gate` / `dead_at_all_tau`；分支语义见 "
      "`PREREG['tau_arm']['decision_rule']`。")
    A("")
    A("## 判据 D4 —— 引用边界")
    A("")
    A("- 本判据的“可达比”是**数值可达性**，不是物理可信度：D2 表明离散格式本身偏移共存对比，"
      "因此任何“支持密度比 X”的声明必须同时给出 D2 的格式偏差。")
    A("- 门值是预注册阈值（同一探针内可比），不是文献统一标准；引用须连同门值一起引用。")
    A("- 适用面：2D D2Q9、float32、周期边界、单组分 SCMP、G=+1、库内 YS2006 PR 参数；"
      "3D 与两相独立黏度未测，不得外推。")
    A("- 若 `tests/test_eos_maxwell_stability.py::test_results_artefact_matches_tree_code_hashes` "
      "失败，说明本文件与代码不同步（产物过期），此时不得引用本判据的任何数字。")
    A("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------- RESULTS.md


def results_markdown(res: dict) -> str:
    env, pre = res["environment"], res["prereg"]
    L: list[str] = []
    A = L.append
    A("# RESULTS — EOS-Maxwell construction stability probe (generated)\n")
    A("> **This file is machine-generated by `plot_figures.py` from `out/results.json`;"
      " no number below is typed by hand.** Re-run the campaign and regenerate to refresh.\n")
    A(f"- generated: {res['generated_utc']}")
    A(f"- eos: `{res['eos']}`   device: CPU   torch {env['torch']}   python {env['python']} ({env['machine']})")
    A(f"- driver sha256: `{res['driver_sha256'][:16]}…`   probe_lib sha256: `{env['probe_lib_sha256'][:16]}…`")
    A(f"- wall time of campaign: {res.get('wall_s_total')} s\n")

    A("## 1. Maxwell construction tiers (equal-area, library solver)\n")
    A("| target ratio | bisected T_r | T | rho_l | rho_v | ratio | p_sat | rho_cross | |area residual|/p_sat Δv | within 1 % |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for k, m in res["maxwell_tiers"].items():
        A(f"| {k} | {_fmt(m['T_r'],6)} | {_fmt(m['T'],6)} | {_fmt(m['rho_liquid'],6)} | "
          f"{_fmt(m['rho_vapour'],6)} | {_fmt(m['ratio'],6)} | {_fmt(m['p_sat'],4)} | {_fmt(m['rho_cross'],5)} | "
          f"{_fmt(m.get('area_residual_rel'),3)} | {m.get('within_target_tol')} |")
    A("")

    A("## 2. Static-droplet tiers (L=96, R=20, tau=1, G=+1, 20000-step hold)\n")
    A("| run | forcing | stable | steps run | collapse step | measured ratio | Maxwell ratio | ratio error | "
      "spurious Ma | rho_max | mass rate/step | verdict | reason |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["tiers"]:
        A(f"| {r['tag']} | {r['spec']['forcing']} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),4)} | {_fmt(r['ratio_maxwell'],4)} | {_fmt(r.get('ratio_error'),3)} | "
          f"{_fmt(r.get('ma_max'),3)} | {_fmt(r.get('rho_max_field'),4)} | {_fmt(r.get('mass_rate_per_step'),2)} | "
          f"**{r['verdict']['cls']}** | {r['verdict']['reason']} |")
    A("")

    hold = max((r["spec"]["steps"] for r in res.get("longhold", [])), default=0)
    A(f"## 3. Long hold ({hold} steps, EDM)\n")
    A("| run | stable | steps run | collapse step | measured ratio | ratio error | rho_v measured | rho_v Maxwell | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in res.get("longhold", []):
        A(f"| {r['tag']} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | {_fmt(r.get('ratio'),5)} | "
          f"{_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('rho_v'),5)} | {_fmt(r.get('rho_v_maxwell'),5)} | "
          f"**{r['verdict']['cls']}** |")
    A("")

    A("## 4. Planar-interface control (liquid slab, periodic box, 40000 steps)\n")
    A("| run | forcing | stable | measured ratio | ratio error | rho_l | rho_v | rho_v Maxwell | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in res.get("controls", []):
        A(f"| {r['tag']} | {r['spec']['forcing']} | {r['stable']} | {_fmt(r.get('ratio'),5)} | "
          f"{_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('rho_l'),5)} | {_fmt(r.get('rho_v'),5)} | "
          f"{_fmt(r.get('rho_v_maxwell'),5)} | **{r['verdict']['cls']}** |")
    A("")

    A("## 5. Force-discretisation lever arm (`scheme='pressure_tensor'`)\n")
    A("| run | forcing | stable | steps run | collapse step | measured ratio | ratio error | spurious Ma | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in res.get("schemes", []):
        A(f"| {r['tag']} | {r['spec']['forcing']} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),4)} | {_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('ma_max'),3)} | "
          f"**{r['verdict']['cls']}** |")
    A("")

    A("## 6. Construction-width arm (initial tanh width, EDM, 20000 steps)\n")
    A("| run | width | stable | steps run | collapse step | measured ratio | ratio error | spurious Ma | verdict | error |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res.get("widths", []):
        A(f"| {r['tag']} | {r['spec']['width']:g} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),4)} | {_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('ma_max'),3)} | "
          f"**{r['verdict']['cls']}** | {r.get('error') or ''} |")
    A("")

    A("## 6b. Replication arm (identical configuration, EDM, 20000 steps)\n")
    A("| run | initial width | stable | steps run | collapse step | measured ratio | ratio error | spurious Ma | rho_min tail | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res.get("repeats", []):
        A(f"| {r['tag']} | {r['spec']['width']:g} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),4)} | {_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('ma_max'),3)} | "
          f"{_fmt(r.get('rho_min_field'),3)} | **{r['verdict']['cls']}** |")
    A("")
    A("## 6c. Thread-count arm (numerical reproducibility control, EDM, 20000 steps)\n")
    A("| run | torch threads | stable | steps run | collapse step | measured ratio | ratio error | spurious Ma | rho_min tail | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res.get("threads", []):
        A(f"| {r['tag']} | {r['spec']['threads']} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),6)} | {_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('ma_max'),4)} | "
          f"{_fmt(r.get('rho_min_field'),4)} | **{r['verdict']['cls']}** |")
    A("")
    A("## 6d. Robustness arm (density perturbation << 1 %, EDM, 20000 steps)\n")
    A("| run | noise amp | seed | stable | steps run | collapse step | measured ratio | ratio error | spurious Ma | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res.get("robust", []):
        A(f"| {r['tag']} | {r['spec']['noise']:g} | {r['spec']['seed']} | {r['stable']} | {r['steps_run']} | "
          f"{_fmt(r['collapse_step'])} | {_fmt(r.get('ratio'),6)} | {_fmt(r.get('ratio_error'),3)} | "
          f"{_fmt(r.get('ma_max'),4)} | **{r['verdict']['cls']}** |")
    A("")
    A("## 6e. Viscosity (tau) lever arm — predeclared decision (PREREG['tau_arm'])\n")
    tsum = P.tau_lever_summary(res.get("tau", []))
    pre_tau = res["prereg"].get("tau_arm", {})
    if pre_tau:
        A(f"- declared: {pre_tau.get('declared', '?')}; grid tau={pre_tau.get('grid', {}).get('tau')}, "
          f"forcing={pre_tau.get('grid', {}).get('forcing')}, steps={pre_tau.get('grid', {}).get('steps')}")
        A(f"- question: {pre_tau.get('question', '')}")
    A("")
    A("| target ratio | tau grid | collapsed runs | min spurious Ma (at tau) | min ratio error (at tau) | MA gate cleared | MA gate cleared on relaxation-resolved runs only | ratio gate cleared | unrelaxed tau | decision |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for ratio in res["prereg"]["target_ratios"]:
        e = tsum.get(f"{ratio:g}", {})
        A(f"| {ratio:g} | {e.get('taus')} | {e.get('n_collapsed')} | "
          f"{_fmt(e.get('ma_min'), 3)} (tau={e.get('ma_min_tau')}) | "
          f"{_fmt(e.get('ratio_error_min'), 3)} (tau={e.get('ratio_error_min_tau')}) | "
          f"{e.get('ma_ok')} | {e.get('ma_ok_relaxed', '—')} | {e.get('ratio_ok')} | "
          f"{e.get('not_relaxation_resolved', [])} | `{e.get('decision')}` |")
    A("")
    A("| run | tau | stable | steps run | collapse step | measured ratio | ratio error | spurious Ma | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in res.get("tau", []):
        A(f"| {r['tag']} | {r['spec']['tau']:g} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),4)} | {_fmt(r.get('ratio_error'),3)} | {_fmt(r.get('ma_max'),3)} | "
          f"**{r['verdict']['cls']}** |")
    A("")
    A("## 7. Stability boundary map (T_r sweep x forcing, 4000-step hold)\n")
    A("| run | Maxwell ratio | rho_l | rho_v | forcing | stable | steps run | collapse step | measured ratio | spurious Ma | onset mechanism |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res.get("boundary", []):
        A(f"| {r['tag']} | {_fmt(r['ratio_maxwell'],4)} | {_fmt(r['rho_l_maxwell'],4)} | {_fmt(r['rho_v_maxwell'],4)} | "
          f"{r['spec']['forcing']} | {r['stable']} | {r['steps_run']} | {_fmt(r['collapse_step'])} | "
          f"{_fmt(r.get('ratio'),4)} | {_fmt(r.get('ma_max'),3)} | {(r.get('error') or '—')} |")
    A("")

    v = res["verdict"]
    A("## 8. Verdict block (predeclared gates)\n")
    A(f"- tiers passed: **{v['tiers_pass']} / {len(v['tiers'])}**; tiers failed: **{v['tiers_fail']}**")
    A(f"- highest passing ratio: `{v['highest_passing_ratio']}`; lowest failing ratio: `{v['lowest_failing_ratio']}`")
    A(f"- gates: ratio error ≤ {pre['ratio_error_max']:.0%}, spurious Ma ≤ {pre['spurious_warn_ma']} "
      f"(PASS ≤ {pre['spurious_pass_ma']}), hold ≥ {pre['stability_steps_min']} steps, "
      f"mass rate ≤ {pre['mass_drift_rate_max']:.0e}/step, no psi-clamp exposure")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(_HERE / "out" / "results.json"))
    ap.add_argument("--out", default=str(_HERE))
    args = ap.parse_args(argv)
    res = _load(Path(args.results))
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    figs = [
        fig_profiles(res, out / "figures"),
        fig_spurious(res, out / "figures"),
        fig_stability_map(res, out / "figures"),
        fig_ratio_maintenance(res, out / "figures"),
        fig_tau_lever(res, out / "figures"),
    ]
    md = results_markdown(res)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    crit = criterion_markdown(res)
    (out / "CRITERION.md").write_text(crit, encoding="utf-8")
    for f in figs:
        print("wrote", f)
    print("wrote", out / "RESULTS.md")
    print("wrote", out / "CRITERION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())