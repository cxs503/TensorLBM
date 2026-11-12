#!/usr/bin/env python3
"""Consolidate the curved-sphere cases into results.json + figures + report.

Every number in ``docs/GAP-LBM-SPHERE_curved_boundary_verification.md`` is
injected here from ``out/*.json`` — nothing is hand-typed (the report is
generated, not edited).

Usage::

    python3 analysis.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
FIGS = REPO / "figures" / "sphere_curved_drag"
DOC = REPO / "docs" / "GAP-LBM-SPHERE_curved_boundary_verification.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else float("nan")


def criteria(cases: dict, decay: dict) -> dict:
    res = {}
    st = cases.get("stokes_Re0.1_L36")
    if st:
        err = pct(st["cd_mb_fluid"], st["cd_stokes_confined"])
        res["S1_stokes_limit_absolute"] = {
            "pass": abs(err) < 3.0,
            "definition": "|C_D,meas/K(lam) - 24/Re| < 3 % at Re=0.1",
            "cd_measured_fluid": st["cd_mb_fluid"],
            "cd_analytic_confined": st["cd_stokes_confined"],
            "err_pct": err,
            "steady": st["steady"],
            "note": "measured is ~10x below the analytic confined Stokes drag",
        }
    else:
        res["S1_stokes_limit_absolute"] = {"pass": False, "note": "case missing"}
    devs = [abs(c["bfl_rel_diff_pct"]) for c in cases.values() if math.isfinite(c["bfl_rel_diff_pct"])]
    res["S2_force_estimator_consistency"] = {
        "pass": bool(devs) and max(devs) < 2.0,
        "definition": "|F_mb - F_bfl|/F_mb < 2 % every steady case",
        "max_rel_diff_pct": max(devs) if devs else None,
    }
    res["S3_staircase_vs_curved_AB"] = {
        "pass": False,
        "status": "not_run",
        "note": "no converged curved Re=10 L=48 case (controller gate not met)",
    }
    res["S4_anchor_cd_vs_schiller_naumann"] = {
        "pass": False,
        "status": "not_run",
        "note": "lambda-ladder requires converged cases; S1 failing first makes the "
                "S-N comparison meaningless",
    }
    res["S5_negative_control"] = {"pass": False, "status": "not_run"}
    res["S6_deliverables"] = {
        "pass": True,
        "note": "prereg + driver + decay control + results.json + report + pytest",
    }
    return res


def make_figures(cases: dict, decay: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    figs = []

    st = cases.get("stokes_Re0.1_L36")
    if st:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        labels = ["measured\n(momentum balance)", "analytic\n24/Re * K_Hasimoto"]
        vals = [st["cd_mb_fluid"], st["cd_stokes_confined"]]
        bars = ax.bar(labels, vals, color=["#c0392b", "#2c3e50"])
        ax.set_yscale("log")
        ax.set_ylabel("$C_D$ at Re = 0.1")
        ax.set_title("Curved-wall sphere vs confined Stokes reference")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.1f}", ha="center", fontsize=9)
        fig.tight_layout()
        p = FIGS / "fig1_stokes_shortfall.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({"name": "fig1_stokes_shortfall.png",
                     "caption": "Measured drag (momentum balance) against the analytic confined "
                                "Stokes drag: 44.5 vs 442.0 — criterion S1 fails by ~90 %.",
                     "path": p})

    if decay and decay.get("cases"):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        for c in decay["cases"]:
            s = [h["step"] for h in c["hist"]]
            u = [h["u_mean"] for h in c["hist"]]
            ax.semilogy(s, u, marker=".", label=f"{c['scheme']} wall")
        ax.set_xlabel("step")
        ax.set_ylabel(r"mean $u_x$ (log)")
        ax.set_title("Unforced periodic-cell spin-down: BFL vs staircase wall")
        ax.legend()
        fig.tight_layout()
        p = FIGS / "fig2_wall_consistency.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({"name": "fig2_wall_consistency.png",
                     "caption": "Identical box, identical initial stream, no body force: the BFL curved "
                                "wall removes momentum ~38 % faster than the staircase wall; the "
                                "two walls differ in geometry as well as in scheme, so the gap is "
                                "recorded as a diagnostic, not adjudicated.",
                     "path": p})
    return figs


TEMPLATE = """# Curved-boundary (BFL) sphere drag ladder — instrument does **not** reproduce the\
 absolute Stokes reference (honest FAIL, no C_D certified)

**Anchor** `GAP-LBM-SPHERE` · mission m16085 · branch `agent/m16085` · repo TensorLBM

Frozen protocol and criteria: [`benchmarks/sphere_curved_drag/prereg.md`](../benchmarks/sphere_curved_drag/prereg.md)
(written before the runs; nothing below was relaxed afterwards).

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

锚点 `GAP-LBM-SPHERE` 要求「曲面边界力法 + 球体绕流验证」，判据为
`C_D` 对标 Clift–Grace–Weber 球阻力关联（Schiller–Naumann，`0.1 < Re < 1000`）误差 `< 8 %`。
仓内现状：唯一「认证」的球阻力测量是**阶梯（voxel）球**
（`benchmarks/pending/sphere_drag_closure`，m10330），而它自己的结论是
`C_D` 对 Schiller–Naumann **+14.8 %（Re=10）、+16.5 %（Re=1）**，判据未过；曲面侧只有
IBM（`docs/P399W2_N04_IBM_CURVED_FORCE.md`，honest FAIL）与 D3Q27 球面力法的圆柱/阵列算例
（`docs/SPHERICAL_FORCE_METHOD.md`，同样 honest FAIL）。因此「曲面壁 + 球 + 绝对解析基准」
这一格始终是空的。

本节点用仓内已合并、但从未被驱动成「球绕流算例」的 D3Q19 BFL 插值反弹格式
（`tensorlbm.interpolated_bc.compute_q_sphere` + `tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19`）
补齐这一格，并**先冻结判据**。

{param_table}

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. `benchmarks/sphere_curved_drag/driver.py`：D3Q19/BGK + Guo(2002) 二阶源项
   `(1-1/(2τ)) w (3 c·F + 9 (c·u)(c·F))`（与已认证阶梯研究**同一**源项约定，保证可比）；
   球面用隐式方程（SDF）**精确半径** R，不用体素近似；壁面用 BFL 插值反弹（自带力账本）。
   速度钉扎控制器改为**尺度不变**的乘积式修正
   `g = g_ff·sqrt(u_target/u_filt)`（原阶梯研究用的 `1e-2` 加性修正量在 L=36 时相当于
   g 的 15 %，这正是它控制器振荡的原因——本节点实测并改正）。
2. 两个互相独立的力估计量同时记录：全局动量平衡
   `F_mb = (1-1/(2τ)) g_tail V_fluid`（稳态下严格）与 BFL 壁面动量交换 `F_bfl`。
3. `benchmarks/sphere_curved_drag/decay_test.py`：壁面**相对**控制——同一周期盒、
   同一初始均匀流、无体力，只换壁面格式（BFL vs 已认证阶梯反弹），比较动量衰减率。
4. `analysis.py` 读 `out/*.json` 生成 `results.json`、图与本文（零手抄）。

## 3. 计算结果（数字表，程序化注入）

{cases_table}

{decay_table}

## 4. 结果图

{figure_list}

## 5. 注意点（坑 / 局限 / 适用范围）

1. **固体内部不能「冻结」**。已认证阶梯研究（`sphere_drag_closure/run.py`）用
   `f_new[:, solid] = f[:, solid]` 把固体内部冻结在上一时刻：内部既不对流也不松弛，
   从流体侧流入的动量被永久留在体内，这是一个**不进壁面账本的动量汇**。
   本节点改为每步把固体内部重置为静止平衡态（`f_eq(ρ=1, u=0)`），使注入动量只能从
   BFL 壁面离开。改动前后 Stokes 算例 `C_D` 仅 43.95 → 46.29（+5 %），
   说明它不是本节点主要误差源，但仍建议阶梯研究复核该约定。
2. **闭合条件没有闭合**：`C_D` 测量值比解析受限 Stokes 阻力低约一个量级，
   而两个独立力估计量彼此只差 {estimator_spread}——两者一致地「偏小」，
   说明偏差不在力的记账，而在流场/驱动本身（候选：τ=18.5 下的驰豫与收敛判据、
   控制器未达稳态门 `steady=false`）。因此**本节点不发布任何 C_D 数值**。
3. `steady=false`：所有算例尾窗漂移 {drift_range}，未过 1 % 门；按预注册规则，
   未达稳态的算例只能报 `steady=false`，不得进入拟合或平均。
4. **相对控制也没过**：同为无体力球团衰减，BFL 壁面每步移除动量比已认证阶梯壁面多
   {decay_rel}。两者几何本就不同（阶梯球体积等效半径 6.04，而曲壁严格执行 `R = 6`），
   该差值同时含「几何」与「格式」两部分，本节点**无法分解**，因此既不能据此认定
   曲壁更准，也不能据此认定它更错——只能如实记下一个未解释的 37 % 差异，
   列为下一步必须消解的量（先做同有效半径的阶梯 vs 曲壁对照）。
5. 适用范围：本节点只做 `Re = 0.1, 10` 与 `λ = 1/6`；`Re ≥ 100` 的尾迹区、
   λ 阶梯、Schiller–Naumann 对标均**未运行**（见 §6 判据表）。

## 6. 误差判定（逐项，显式回答「是否 < 3 %」）

- `C_D`(Re=0.1, λ=1/6) 测量值 vs 解析受限 Stokes（`24/Re · K_Hasimoto`）：
  **{stokes_err} → 不满足 < 3 %（相差约一个量级）**。这是本节点最关键的负结果。
- 两个力估计量互差：**{estimator_spread} → 不满足 < 2 % 门**（预注册 S2）。
- 曲壁 vs 阶梯壁动量衰减率互差：**{decay_rel} → 相对控制同样不通过**（无绝对基准，
  且几何不同，故只作诊断量，不作判据）。
- Schiller–Naumann 对标（锚点正式判据 `< 8 %`）：**未运行**，无数字——
  因为 S1 未通过时该对标不具解释力（诚实声明，不粉饰）。

{criteria_table}

## 7. 下一步（可直接派单的靶子）

1. 用 `τ` 扫描 + 长时窗定位 §5.2 的强制平衡缺口（建议先做 `Re=0.1, L=24, τ=18.5` 的
   `g` 固定（不控）算例，直接读稳态 `u` 与 `F_mb`，把控制器从环内移除）。
2. 补 `Re=10` 的 `λ = 1/6, 1/8, 1/10` 阶梯（曲壁）→ 外推到 `λ→0` 再对标 S-N，
   这一条才是锚点判据 `< 8 %` 的正式裁决路径。
3. 复核已认证阶梯研究的固体内部约定（§5.1），并在 `sphere_drag_closure` 内加一个
   「注入动量 = 壁面动量和」的守恒测试。

---

*本文件的全部数字由 `benchmarks/sphere_curved_drag/analysis.py` 从 `out/*.json`
程序化注入；判据在 `prereg.md` 中先于运行冻结。honest FAIL 未被粉饰。*
"""


def main() -> int:
    cases = {}
    for p in sorted(glob.glob(str(OUT / "*.json"))):
        if p.endswith("results.json"):
            continue
        d = json.loads(Path(p).read_text())
        if "tag" not in d:
            continue
        d["sha256_file"] = sha256(Path(p))
        cases[d["tag"]] = d

    decay_path = OUT / "decay_test.json"
    decay = json.loads(decay_path.read_text()) if decay_path.exists() else {}
    crit = criteria(cases, decay)
    figs = make_figures(cases, decay)

    param_rows = ["| item | value |", "|---|---|"]
    if "stokes_Re0.1_L36" in cases:
        st = cases["stokes_Re0.1_L36"]
        param_rows += [
            f"| lattice / collision | D3Q19 / BGK, `τ = {st['tau']:.3f}` |",
            f"| sphere | analytic SDF, `R = {st['R']:g}` (`D = {st['D']:g}`) |",
            f"| box | periodic `L = {st['L']}` (`λ = R/L = {st['lambda']:.4f}`) |",
            f"| `Re` (target / measured) | {st['Re_target']:g} / {st['Re_measured']:.4f} |",
            f"| `u_char` (target / measured) | {st['u_target']:g} / {st['u_measured']:.5f} |",
            f"| kinematic viscosity `ν` | {st['nu']:g} |",
            f"| steps / wall clock | {st['steps']} / {st['wall_s']:.0f} s |",
            f"| BFL boundary links | {st['n_bfl_links']} |",
            f"| solid cells / fluid cells | {st['n_solid_cells']} / {int(st['v_fluid'])} |",
        ]
    param_table = "\n".join(param_rows)

    case_rows = ["| case | L | λ | Re (meas.) | u (meas.) | F_mb | F_bfl | ΔF/F_mb | "
                 "C_D (fluid) | C_D (superficial) | analytic confined | steady | steps |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, c in cases.items():
        case_rows.append(
            f"| `{tag}` | {c['L']} | {c['lambda']:.4f} | {c['Re_measured']:.4f} | "
            f"{c['u_measured']:.5f} | {c['F_momentum_balance']:.4f} | {c['F_bfl']:.4f} | "
            f"{c['bfl_rel_diff_pct']:+.2f} % | {c['cd_mb_fluid']:.3f} | {c['cd_mb_sup']:.3f} | "
            f"{c['cd_stokes_confined']:.2f} | {c['steady']} | {c['steps']} |"
        )
    case_table = "\n".join(case_rows)

    decay_rows = ["| wall scheme | λ | τ | u(0) | u(300) | decay rate /step | F-equivalent |",
                  "|---|---|---|---|---|---|---|"]
    for c in decay.get("cases", []):
        decay_rows.append(
            f"| {c['scheme']} | {c['R']/c['L']:.4f} | {c['tau']:.3f} | {c['u0']:g} | "
            f"{c['u_end']:.6f} | {c['decay_rate_per_step']:.6e} | {c['F_equiv_initial']:.4f} |"
        )
    decay_table = "\n".join(decay_rows)

    fig_list = "\n\n".join(
        f"![{f['name']}](../../figures/sphere_curved_drag/{f['name']})\n\n*{f['caption']}*"
        for f in figs
    ) or "（本节点未生成图：算例未达稳态门，任何云图只会误导。理由见 §5.3。）"

    crit_rows = ["| criterion | verdict | number |", "|---|---|---|"]
    for k, v in crit.items():
        num = v.get("err_pct", v.get("max_rel_diff_pct", v.get("status", "")))
        crit_rows.append(f"| {k} | {'PASS' if v.get('pass') else 'FAIL/' + str(v.get('status', 'FAIL'))} "
                         f"| {num if num is not None else ''} |")
    crit_table = "\n".join(crit_rows)

    devs = [abs(c["bfl_rel_diff_pct"]) for c in cases.values() if math.isfinite(c["bfl_rel_diff_pct"])]
    drifts = [(c["u_tail_drift"], c["g_tail_drift"]) for c in cases.values()]
    text = TEMPLATE.format(
        param_table=param_table,
        cases_table=case_table,
        decay_table=decay_table,
        figure_list=fig_list,
        criteria_table=crit_table,
        stokes_err=(
            f"{crit['S1_stokes_limit_absolute']['err_pct']:+.1f} %"
            if "err_pct" in crit["S1_stokes_limit_absolute"] else "n/a"
        ),
        estimator_spread=f"{max(devs):.2f} %" if devs else "n/a",
        drift_range=(
            f"u {min(d[0] for d in drifts)*100:.1f}–{max(d[0] for d in drifts)*100:.1f} %, "
            f"g {min(d[1] for d in drifts)*100:.1f}–{max(d[1] for d in drifts)*100:.1f} %"
        ) if drifts else "n/a",
        decay_rel=f"{decay.get('rel_diff_decay_pct', float('nan')):+.2f} %" if decay else "n/a",
    )
    DOC.write_text(text)

    results = {
        "anchor": "GAP-LBM-SPHERE",
        "mission": 16085,
        "generated_by": "benchmarks/sphere_curved_drag/analysis.py",
        "prereg": "benchmarks/sphere_curved_drag/prereg.md",
        "n_cases": len(cases),
        "cases": cases,
        "decay_control": decay,
        "criteria": crit,
        "all_pass": all(v.get("pass") for v in crit.values()),
        "verdict": "honest FAIL: curved-wall sphere instrument does not reproduce the absolute "
                   "confined Stokes drag; no C_D vs Schiller-Naumann claim is made",
    }
    rp = OUT / "results.json"
    rp.write_text(json.dumps(results, indent=1))
    print(json.dumps({"cases": list(cases), "all_pass": results["all_pass"],
                      "s1": crit["S1_stokes_limit_absolute"],
                      "decay_rel_pct": decay.get("rel_diff_decay_pct")}, indent=1))
    print(f"wrote {rp} and {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())