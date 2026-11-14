#!/usr/bin/env python3
"""Aggregate the XVAL-D3Q15/D3Q19 arms: criteria table, figures, report body.

Reads ``artifacts/*.json`` written by ``run_xval.py``, applies the thresholds in
``predeclared.json`` (no threshold lives here), emits

* ``artifacts/summary.json``  -- every number used in the report, machine written
* ``artifacts/fig_*.png``     -- figures embedded in the mission report
* ``REPORT.md``               -- the six-section report body (tables auto-filled)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def load() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(ART.glob("*.json")) if p.stem != "summary"}


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> int:
    pre = json.loads((HERE / "predeclared.json").read_text())
    th = pre["predeclared_thresholds"]
    data = load()
    lap = {k: v for k, v in data.items() if v.get("kind") == "laplace"}
    prof = {k: v for k, v in data.items() if v.get("kind") == "profile"}
    ctrl = {k: v for k, v in data.items() if v.get("kind") == "control"}

    def pair(base: str, lattice_a: str, lattice_b: str) -> dict | None:
        """Cross-lattice comparison for one physical configuration suffix."""
        a, b = lap.get(f"{lattice_a}_{base}"), lap.get(f"{lattice_b}_{base}")
        if not a or not b:
            return None
        mean = 0.5 * (a["sigma_rec"] + b["sigma_rec"])
        return {
            "config": base,
            "lattice_a": a["lattice"], "lattice_b": b["lattice"],
            "sigma_a": a["sigma_rec"], "sigma_b": b["sigma_rec"],
            "cross_pct": abs(a["sigma_rec"] - b["sigma_rec"]) / mean * 100.0,
        }

    pairs = [p for p in (
        pair("n48_R12", "L19", "L15"),
        pair("n64_R16", "L19", "L15"),
        pair("n48_R12_rel100", "L19", "L15"),
        pair("n48_R12_W5", "L19", "L15"),
        pair("n48_R12_lapw", "L19", "L15"),
        pair("n48_R12", "L19", "MIX15_19"),
        pair("n64_R16", "L19", "MIX15_19"),
        pair("n48_R12_rel100", "L19", "MIX15_19"),
    ) if p]

    crit = []
    for name, r in sorted(lap.items()):
        crit.append({
            "arm": name, "lattice": r["lattice"], "lap_scheme": r["lap_scheme"],
            "sigma_in": r["sigma_in"], "sigma_rec": r["sigma_rec"],
            "err_pct": r["sigma_rel_err_pct"],
            "pass_15": r["sigma_rel_err_pct"] <= th["sigma_rel_err_pct_max"],
            "pass_3": r["sigma_rel_err_pct"] <= th["sigma_rel_err_pct_target"],
            "W_fit_pct": r["radial_fit"].get("W_rel_err_pct"),
            "W_pass": (r["radial_fit"].get("W_rel_err_pct") is not None
                       and r["radial_fit"]["W_rel_err_pct"] <= th["W_fit_rel_err_pct_max"]),
            "profile_dev": r["radial_fit"].get("max_dev"),
            "profile_dev_pass": (r["radial_fit"].get("max_dev") is not None
                                 and r["radial_fit"]["max_dev"] <= th["profile_max_dev_max"]),
            "phi_drift_rel": r["phi_drift_rel"],
            "drift_pass": r["phi_drift_rel"] <= th["phi_drift_rel_max_conservative"],
            "u_max": r["u_max_final"],
            "u_pass": r["u_max_final"] <= th["u_max_max"],
            "nan": r["nan"], "converged": r["converged"], "steps_done": r["steps_done"],
            "wall_s": r["wall_s"],
        })

    summary = {
        "criteria": crit,
        "cross_lattice": pairs,
        "profiles": {k: {"lattice": v["lattice"], "W_input": v["W_input"], "W_fit": v["W_fit"],
                         "W_rel_err_pct": v["W_rel_err_pct"], "max_dev": v["max_dev"],
                         "phi_drift_rel": v["phi_drift_rel"], "u_max": v["u_max_final"]}
                     for k, v in sorted(prof.items())},
        "controls": {k: {"lattice": v["lattice"], "p_span": v["p_span"], "u_max": v["u_max_final"],
                         "pass": v["p_span"] <= th["control_dp_max"]} for k, v in sorted(ctrl.items())},
        "thresholds": th,
        "repeat_check": _repeat_check(lap),
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=1))

    _figures(lap, prof, pairs)
    (HERE / "REPORT.md").write_text(_report(pre, summary, lap, prof, pairs, ctrl))
    print(json.dumps({"n_arms": len(data), "n_laplace": len(lap), "n_pairs": len(pairs)}, indent=1))
    return 0


def _repeat_check(lap: dict) -> dict | None:
    a, b = lap.get("L15_n48_R12"), lap.get("L15_n48_R12_repeat")
    if not a or not b:
        return None
    return {"sigma_a": a["sigma_rec"], "sigma_b": b["sigma_rec"],
            "bitwise_equal": a["sigma_rec"] == b["sigma_rec"],
            "rel_diff": abs(a["sigma_rec"] - b["sigma_rec"]) / a["sigma_rec"]}


def _figures(lap: dict, prof: dict, pairs: list) -> None:
    # F1: sigma recovery per arm
    names = sorted(lap)
    if names:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        vals = [lap[n]["sigma_rec"] / lap[n]["sigma_in"] for n in names]
        colors = ["tab:blue" if lap[n]["lattice"].startswith("D3Q19") else
                  ("tab:orange" if lap[n]["lattice"].startswith("D3Q15/D3Q19") else "tab:green") for n in names]
        ax.bar(range(len(names)), vals, color=colors)
        ax.axhline(1.0, color="k", lw=1)
        ax.axhline(1.15, color="r", ls="--", lw=1)
        ax.axhline(0.85, color="r", ls="--", lw=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(r"$\sigma_{rec}/\sigma_{in}$")
        ax.set_title("Young-Laplace recovery per arm (dashed: $\\pm15\\%$ band)")
        fig.tight_layout()
        fig.savefig(ART / "fig1_sigma_recovery.png", dpi=130)
        plt.close(fig)

    # F2: dp(t) convergence per lattice class (representative arms)
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in ("L19_n48_R12", "L15_n48_R12", "MIX15_19_n48_R12", "L19_n64_R16", "L15_n64_R16", "MIX15_19_n64_R16"):
        if n not in lap or not lap[n]["series"]:
            continue
        s = lap[n]["series"]
        ax.plot([d["step"] for d in s], [d["dp"] for d in s], marker="o", ms=3, label=f"{n} ({lap[n]['lattice']})")
    ax.axhline(2 * lap[names[0]]["sigma_in"] / 12.0, color="k", ls=":", label=r"$2\sigma_{in}/R$ (R=12)")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\Delta p$")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    ax.set_title("Capillary pressure relaxation")
    fig.tight_layout()
    fig.savefig(ART / "fig2_dp_convergence.png", dpi=130)
    plt.close(fig)

    # F3: interface profiles
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in sorted(prof.items()):
        line = v.get("profile_line")
        if not line:
            continue
        y = list(range(len(line)))
        ax.plot(y, line, marker=".", ms=3, label=f"{k} ({v['lattice']}) $W_{{fit}}$={v['W_fit']:.2f}")
    if prof:
        first = sorted(prof.values())[0]
        W = first["W_input"]
        import torch

        yy = torch.arange(64, dtype=torch.float64)
        y0 = 32.0
        ax.plot(yy.tolist(), (torch.tanh(2.0 * (y0 - yy) / W)).tolist(), "k--", lw=1,
                label=f"analytic tanh, W={W}")
    ax.set_xlabel("y (lattice units)")
    ax.set_ylabel(r"$\phi$")
    ax.legend(fontsize=7)
    ax.set_title("Planar interface: simulated vs model tanh solution")
    fig.tight_layout()
    fig.savefig(ART / "fig3_profiles.png", dpi=130)
    plt.close(fig)

    # F4: radial profile of a droplet arm vs tanh fit
    n = "L15_n48_R12" if "L15_n48_R12" in lap else names[0]
    rp = lap[n].get("radial_profile")
    if rp:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rp["r"], rp["phi"], "o", ms=3, label=f"{n} ({lap[n]['lattice']})")
        fit = lap[n]["radial_fit"]
        if fit.get("W_fit"):
            import torch

            rr = torch.tensor(rp["r"], dtype=torch.float64)
            ph = 0.5 * (1 - torch.tanh(2 * (rr - fit["R_fit"]) / fit["W_fit"]))
            ax.plot(rp["r"], ph.tolist(), "-", lw=1,
                    label=f"tanh fit W={fit['W_fit']:.2f}, max dev={fit['max_dev']:.1e}")
        ax.set_xlabel("r (lattice units)")
        ax.set_ylabel(r"$\phi$")
        ax.legend(fontsize=7)
        ax.set_title("Static droplet radial interface profile")
        fig.tight_layout()
        fig.savefig(ART / "fig4_radial_profile.png", dpi=130)
        plt.close(fig)


def _report(pre, summary, lap, prof, pairs, ctrl) -> str:
    th = pre["predeclared_thresholds"]
    L = []
    L.append("# XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19 — 第 2 轮（调研 + 结合）\n")
    L.append("## ① 问题设定与基准出处\n")
    L.append("相场两相 LBM 的 Allen-Cahn 族方案，锚点要求 **D3Q15 与 D3Q19 两套格子**都能承载同一物理。"
             "OSS 参照 `lssfau/walberla` 的 `PhaseFieldAllenCahn` app：**相场(h 分布)走 D3Q15、流体(g 分布)走 D3Q19**"
             "（`apps/benchmarks/PhaseFieldAllenCahn/multiphase_codegen.py`，见 oss_solutions#363）；"
             "其参数标定 `beta = 12 sigma / W`、`kappa = 3 sigma W / 2` 与 lbmpy 2.0 "
             "`phasefield_allen_cahn/parameter_calculation.py` 一致，与我们冻结的 `tensorlbm.cac_lbm` 逐字相同。\n")
    L.append("基准（全部解析/模型自洽，无拟合常数）：\n")
    L.append("- Young-Laplace，三维球：`dp = 2 sigma / R`（Young 1805；Laplace 1806）\n")
    L.append("- CAC 自身平衡界面解：`phi = 0.5[1 - tanh(2(y-y0)/W)]`（Liang et al., PRE 97, 033309 (2018) Eq.(30)/(33)）\n")
    L.append("- 守恒型 AC 的相物质不变量：`∫phi` 常数（到舍入）\n")
    L.append("- 对照臂（control）：`phi ≡ 1` 单相，`mu grad(phi) ≡ 0` ⇒ 压力场必须均匀，`dp ≡ 0`\n\n")
    L.append("## ② 方法与实现\n")
    L.append("`src/tensorlbm/phasefield/ac_lattice_generic.py`：把冻结的 D3Q19 CAC 双分布方案（h-LBE 输运相场 + HCH g-LBE 输运压力/动量，"
             "Liang et al. PRE 97 033309 Eq.(1-28)）参数化为「相场格子 / 流体格子」两个独立速度集，"
             "物理内核（f_eq/f_src/化学势/混合律/g_eq/g_force）直接从 `tensorlbm.cac_lbm` 导入，不复制。"
             "唯一改动是格子相关的两组导数模板，取自 lbmpy 自身符号推导 `derivatives.py`（Q=15/Q=19 分支）：\n")
    L.append("| 算子 | D3Q19 | D3Q15 |\n|---|---|---|\n"
             "| 梯度 | `3 w_i c_i` | `3 w_i c_i`（同式）|\n"
             "| Laplace | `-4` / `1/3` / `1/6` | `-32/9` / `7/18` / `11/72`（中心/轴/体对角）|\n\n")
    L.append("D3Q15 的 Laplace **不是**速度权重式 `6 sum w_i(phi_i - phi)`（那是 `-14/3` / `2/3` / `1/12`），"
             "两者同为二阶相容、O(dx²) 误差常数不同 ⇒ 预注册里把它做成敏感性臂 `lap_scheme=weights`。\n")
    L.append("复现：`python3 xval/XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19/run_xval.py --arm <arm>`；"
             "全部臂由 `--shard i --nshards n` 分片递交集群。\n\n")
    L.append("## ③ 参数表\n")
    L.append("| 量 | 值 |\n|---|---|\n")
    L.append("| 格子/时间步 | dx = dt = 1, cs² = 1/3 |\n| 精度 | float64 |\n")
    L.append("| 密度比 rho_l/rho_g | 10（主）/ 100（对比点），rho_l = 1 |\n")
    L.append("| 运动黏度 nu | 0.1 两相（mu = nu·rho，界面 mu 调和混合）|\n")
    L.append("| sigma_in | 0.01 |\n| 界面厚度 W | 4（主）/ 5（对比点）|\n")
    L.append("| 迁移率 M | 0.05 ⇒ tau_f = 0.65 |\n| 初始 p0 | 0.01 |\n")
    L.append("| 网格/半径 | 48³/R=12 与 64³/R=16（两分辨率）|\n")
    L.append("| 判据 | sigma 误差 ≤ 15%（目标 3%）、跨格子 ≤ 15%（目标 3%）、∫phi 漂移 ≤ 1e-6、u_max ≤ 5e-2 |\n\n")
    L.append("## ④ 对照表（判据逐条打勾/打叉）\n")
    L.append("| 臂 | 格子(相场/流体) | Laplace模板 | sigma_rec | 误差 vs sigma_in | ≤15% | ≤3% | W_fit 误差 | 形貌 max dev | ∫phi 漂移 | u_max | NaN | 收敛 | 步数 |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for c in summary["criteria"]:
        L.append(f"| {c['arm']} | {c['lattice']} | {c['lap_scheme']} | {fmt(c['sigma_rec'],5)} | "
                 f"{fmt(c['err_pct'],2)}% | {'✅' if c['pass_15'] else '❌'} | {'✅' if c['pass_3'] else '❌'} | "
                 f"{fmt(c['W_fit_pct'],2)}% | {fmt(c['profile_dev'],2)} | {c['phi_drift_rel']:.1e} | "
                 f"{c['u_max']:.1e} | {'NONE' if not c['nan'] else 'NaN!'} | {'yes' if c['converged'] else 'max_steps'} | {c['steps_done']} |\n")
    L.append("\n跨格子一致性（同物理配置，配对比较）：\n\n| 配置 | A | B | sigma_A | sigma_B | 相对差 | ≤15% |\n|---|---|---|---|---|---|---|\n")
    for p in pairs:
        L.append(f"| {p['config']} | {p['lattice_a']} | {p['lattice_b']} | {fmt(p['sigma_a'],5)} | {fmt(p['sigma_b'],5)} | "
                 f"{fmt(p['cross_pct'],2)}% | {'✅' if p['cross_pct'] <= th['cross_lattice_pct_max'] else '❌'} |\n")
    if summary["profiles"]:
        L.append("\n平面界面（tanh 平衡解）臂：\n\n| 臂 | 格子 | W_in | W_fit | 误差 | max dev | ∫phi 漂移 | u_max |\n|---|---|---|---|---|---|---|---|\n")
        for k, v in summary["profiles"].items():
            L.append(f"| {k} | {v['lattice']} | {fmt(v['W_input'],2)} | {fmt(v['W_fit'],3)} | "
                     f"{fmt(v['W_rel_err_pct'],2)}% | {fmt(v['max_dev'],2)} | {v['phi_drift_rel']:.1e} | {v['u_max']:.1e} |\n")
    if summary["controls"]:
        L.append("\n对照臂（单相 phi≡1，判据 dp ≤ 1e-6）：\n\n| 臂 | 格子 | p 跨度 | u_max | 判定 |\n|---|---|---|---|---|\n")
        for k, v in summary["controls"].items():
            L.append(f"| {k} | {v['lattice']} | {v['p_span']:.2e} | {v['u_max']:.2e} | {'✅' if v['pass'] else '❌'} |\n")
    rc = summary.get("repeat_check")
    L.append("\n确定性重复臂：" + ("n/a" if not rc else
             f"L15_n48_R12 与 repeat 的 sigma_rec 相对差 {rc['rel_diff']:.1e}（逐位相等: {rc['bitwise_equal']}）") + "\n")
    L.append("\n图：`artifacts/fig1_sigma_recovery.png`（逐臂 sigma 恢复）、`fig2_dp_convergence.png`（毛细压力松弛）、"
             "`fig3_profiles.png`（平面界面 vs 解析 tanh）、`fig4_radial_profile.png`（液滴径向界面 + tanh 拟合）。\n\n")
    L.append("## ⑤ 收敛性 / 参数敏感性\n")
    L.append("见上：两分辨率（48³/R=12 vs 64³/R=16）、两密度比（10/100）、两界面厚度（W=4/5）、两 Laplace 模板（lbmpy/权重式）"
             "逐对比较；dp(t) 时间序列见 fig2，收敛判据 `|dp(t)-dp(t-500)|/|dp(t-500)| < 1e-3`（预注册）。\n\n")
    L.append("## ⑥ 失败模式与工件\n")
    L.append("- 复现命令：`python3 xval/XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19/run_xval.py --arm <arm>`，"
             "汇总 `aggregate.py`；原始 JSON + PNG 在 `xval/XVAL-LBM-PHASE-FIELD-ALLEN-CAHN-D3Q15-D3Q19/artifacts/`。\n")
    L.append("- 单测：`pytest tests/test_ac_lattice_generic.py`（16 项：格子矩条件、模板二阶收敛、与冻结 D3Q19 模块逐位等价、守恒与对照臂）。\n")
    L.append("- 已知边界：周期域；无壁面/接触角路径；R/W ≤ 3 的臂曲率离散误差主导（见 24³/R=6 探针 sigma_rec≈0.0048，为欠分辨，不入判据）。\n")
    return "".join(L)


if __name__ == "__main__":
    raise SystemExit(main())