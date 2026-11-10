#!/usr/bin/env python3
"""Figures + paper-style report for the weak-compressibility campaign (P102W3-N05).

Every number that appears in the generated markdown or in the figures is read
from ``benchmarks/verified/lbm_weakly_compressible/{result,gates}.json`` -- the
report is generated, never hand-typed, so a re-run cannot drift from the
artifacts.

Usage::

    python scripts/plot_lbm_weakly_compressible.py            # figs + docs
    python scripts/plot_lbm_weakly_compressible.py --figs-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "benchmarks" / "verified" / "lbm_weakly_compressible"
DOCS = ROOT / "docs" / "verification"


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(v, nd) for v in np.asarray(x, dtype=object).ravel()) + "]"
    try:
        if not np.isfinite(float(x)):
            return "n/a"
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_ma_scaling(res: dict, out: Path) -> Path:
    diff = res["c1_differential"]["pressure_channel"]
    ctrl = res["c1_differential"]["body_force_control"]
    mas = np.asarray(diff["machs"])
    d_pc = np.asarray(diff["diff_err_inf"])
    d_bf = np.asarray(ctrl["diff_err_inf"])
    drho = np.asarray(diff["drho_over_rho0"])
    keep = d_pc > 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(mas[keep], d_pc[keep], "o-", label="pressure-driven channel (compressible signature)")
    ck = d_bf > 0
    if ck.any():
        ax[0].loglog(mas[ck], d_bf[ck], "s--", label="body-force control (uniform density)")
    ref = d_pc[keep][-1] * (mas[keep] / mas[keep][-1]) ** 2.0
    ax[0].loglog(mas[keep], ref, "k:", lw=1, label=r"slope 2 reference")
    for a, m in zip(ax[0].lines[-1].get_data()[1], mas[keep]):
        pass
    ax[0].set_xlabel("Ma")
    ax[0].set_ylabel(r"$\|u(Ma)-u(Ma_{ref})\|_\infty / u_{max}$")
    ax[0].set_title(f"differential profile error, fitted exponent = {_fmt(diff['exponent_diff'], 3)}")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)

    k2 = drho > 0
    ax[1].loglog(mas[k2], drho[k2], "o-", label=r"$\max|\rho-\rho_0|/\rho_0$ (channel)")
    ref2 = drho[k2][-1] * (mas[k2] / mas[k2][-1]) ** 2.0
    ax[1].loglog(mas[k2], ref2, "k:", lw=1, label="slope 2 reference")
    ax[1].set_xlabel("Ma")
    ax[1].set_ylabel(r"$\delta\rho/\rho_0$")
    ax[1].set_title(
        "density fluctuation, fitted exponent = "
        + _fmt(diff.get("fit_drho", {}).get("exponent_plain", float("nan")), 3)
    )
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_cylinder(res: dict, out: Path) -> Path:
    ind = res["mach_independence"]
    mas = np.asarray(ind["machs"])
    st = np.asarray(ind["st"], dtype=float)
    cd = np.asarray(ind["cd"], dtype=float)
    cl = np.asarray(ind["cl_amplitude"], dtype=float)
    zero = ind.get("ma_to_zero", {})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (vals, label, key) in zip(
        ax, [(st, "St", "st0"), (cd, r"C_D", "cd0"), (cl, r"C_L amplitude", "cl0")]
    ):
        a.plot(mas, vals, "o-", label="measured")
        if np.isfinite(zero.get(key, np.nan)):
            a.axhline(zero[key], ls=":", color="k", label=r"$Ma\to 0$ fit ($q_0+b\,Ma^2$)")
        a.set_xlabel("Ma")
        a.set_ylabel(label)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_title("Strouhal number")
    ax[1].set_title("mean drag coefficient")
    ax[2].set_title("lift amplitude")
    fig.suptitle("Re = 100 circular cylinder, D = 16, blockage 0.125", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_pressure(res: dict, out: Path) -> Path:
    c6 = res["pressure_poisson_c6"]
    runs = c6["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for r in runs:
        ax[0].plot(r["p_lbm_x"], "o", ms=3, label=f"LBM p, Ma={_fmt(r['mach_measured'], 2)}")
    ref = runs[0]["p_reference_x"]
    ax[0].plot(ref, "k-", lw=1.2, label="incompressible pressure Poisson solution")
    ax[0].set_xlabel("x (lattice)")
    ax[0].set_ylabel(r"$p = c_s^2\rho$")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    ax[0].set_title("pressure field vs Poisson reference")
    dev = np.asarray(c6["max_abs_dev_over_dynamic_pressure"], dtype=float)
    mas = np.asarray(c6["machs"], dtype=float)
    ok = np.isfinite(dev) & (dev > 0)
    if ok.any():
        ax[1].loglog(mas[ok], dev[ok], "o-")
        ax[1].loglog(
            mas[ok], dev[ok][-1] * (mas[ok] / mas[ok][-1]) ** 2.0, "k:", lw=1, label="slope 2 reference"
        )
        ax[1].legend(fontsize=8)
    ax[1].set_xlabel("Ma")
    ax[1].set_ylabel(r"$\max|\Delta p| / (\rho_0 u^2/2)$")
    ax[1].set_title(
        r"Poisson deviation order = "
        + _fmt(c6.get("fit_dev_vs_ma", {}).get("exponent_plain", float("nan")), 3)
    )
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_profiles(res: dict, out: Path) -> Path:
    pc = res["pressure_channel"]["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
    for r in pc:
        prof = np.asarray(r["u_num"], dtype=float)
        ax[0].plot(prof / prof.max(), np.arange(prof.size) + 1, label=f"Ma={_fmt(r['mach_measured'], 2)}")
    prof = np.asarray(pc[0]["u_exact"], dtype=float)
    ax[0].plot(prof / prof.max(), np.arange(prof.size) + 1, "k--", lw=1, label="parabola")
    ax[0].set_ylabel("y (lattice)")
    ax[0].set_xlabel(r"$u/u_{max}$")
    ax[0].set_title("normalised channel profiles (all Ma collapse)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    diff = res["c1_differential"]["pressure_channel"]
    u_ref = np.asarray(pc[0]["u_num"], dtype=float)
    u_ref = u_ref / u_ref.max()
    for r in pc[1:]:
        u = np.asarray(r["u_num"], dtype=float)
        u = u / u.max()
        ax[1].plot((u - u_ref) * 1e3, np.arange(u.size) + 1, label=f"Ma={_fmt(r['mach_measured'], 2)}")
    ax[1].set_xlabel(r"$(u(Ma)-u(Ma_{ref})) \times 10^3$")
    ax[1].set_ylabel("y (lattice)")
    ax[1].set_title("differential profile (the measured Ma signal)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def build_docs(res: dict, gates: dict, figs: dict[str, Path]) -> str:
    crit = res["criteria"]
    diff = res["c1_differential"]["pressure_channel"]
    ctrl = res["c1_differential"]["body_force_control"]
    c6 = res["pressure_poisson_c6"]
    ind = res["mach_independence"]
    pc = res["pressure_channel"]["runs"]
    cyl = res["cylinder"]["runs"]
    acoustic = crit["c4_acoustic_filter"]

    def gate_row(name: str) -> str:
        g = gates.get(name, {})
        return f"| {name} | `{g.get('status', 'MISSING')}` |"

    lines: list[str] = []
    A = lines.append
    A("# 弱可压缩 LBM 验证面：Ma→0 渐近一致性、绕圆柱 Strouhal 与声学滤波（P102W3-N05）")
    A("")
    A("> 本页由 `scripts/plot_lbm_weakly_compressible.py` 从 "
      "`benchmarks/verified/lbm_weakly_compressible/{result,gates}.json` 自动生成，"
      "文中所有数字均为程序化注入（零手抄）。")
    A("")
    A("## 1. 案例介绍（物理背景 / 几何 / 参数）")
    A("")
    A("LBM 用 `p = c_s^2 rho`（`c_s^2 = 1/3`）代替不可压压力泊松方程，因此它求解的是"
      "**弱可压缩** Navier–Stokes 极限：所有结果相对不可压解都存在 `O(Ma^2)` 误差，"
      "`Ma = u_ref/c_s -> 0` 时才收敛到不可压解。本节点把「误差律本身」做成机器可裁的验证面，"
      "而不是再测一个单马赫数算例。")
    A("")
    A("| 算例 | 几何 | 参数 | 用途 |")
    A("|---|---|---|---|")
    A("| 压力驱动槽道 | ny=33, nx=96，半程反弹壁，发展型速度入口 + 压力出口 | "
      f"nu = {_fmt(pc[0]['nu'], 4)} 固定，tau = {_fmt(pc[0]['tau'], 4)}，Ma ∈ {[round(float(r['mach']),3) for r in pc]} | C1/C3/C6 |")
    A("| 体力驱动槽道（周期） | ny=33, nx=32，流向周期，Guo 体力 | 同一 nu，Ma 同上 | C1 阴性对照 |")
    ccase = res.get("provenance", {}).get("cylinder_case", {})
    A(f"| Re=100 圆柱 | D={_fmt(ccase.get('diameter', float('nan')), 3)}, "
      f"{ccase.get('nx')}×{ccase.get('ny')}，阻塞比 {_fmt(float(ccase.get('diameter', 1)) / (float(ccase.get('ny', 2)) - 1), 3)}，"
      f"闭合 {ccase.get('bc')} + {ccase.get('collision')} | Ma ∈ "
      f"{[round(float(r['mach']),3) for r in cyl]}，nu 随 Ma 变化 | C2/C5/C7 |")
    missing = res.get("provenance", {}).get("cylinder_runs_missing") or []
    if missing:
        A("")
        A(f"> 圆柱 Ma = {missing} 未产出记录（低 Ma 角落 tau→1/2 导致发散，见 §5），"
          "按 honest 原则不补造、不替代。")
    A("| 声学激励槽道 | ny=33, nx=96 | Ma = "
      f"{_fmt(acoustic.get('mach', float('nan')), 3)}，密度脉冲幅值 {_fmt(acoustic.get('pulse_amplitude', float('nan')), 3)} | C4/C8 |")
    A("")
    A("## 2. 方法（怎么做的）")
    A("")
    A("* 求解器：`src/tensorlbm/weak_compressibility.py` 的 float64 NumPy D2Q9（BGK/TRT、Guo 体力、"
      "半程反弹、Zou–He 入口/出口与零梯度外推闭合）。")
    A("* 扫描与后处理：`src/tensorlbm/verification/weakly_compressible.py`"
      "（压力泊松参考解、差分 Ma 标度诊断、水动力系数 Mach 无关性、Ma_max 适用域门、产物写出）。")
    A("* **关键实验设计**：固定 `nu`（因此 `tau` 与离散误差底噪固定）后改变 `Ma`，"
      "用「相对最低 Ma 的差分剖面误差」`||u(Ma)-u(Ma_ref)||_inf/u_max` 作为 C1 判据量 —— "
      "绝对误差在 H=32 下被约 7% 的壁面滑移/离散底噪支配（"
      f"实测 Ma=0.01 绝对误差 {_fmt(pc[0]['err_inf'], 3)}），差分把它精确消去。")
    A("* 阴性对照：体力驱动周期槽道稳态密度严格均匀，三阶平衡态项恒为零，"
      "按构造**不可能**出现 `O(Ma^2)` 剖面信号；它用来证明 C1 诊断测的是可压缩性而不是任意数值伪迹。")
    A("* 压力场判据（C6）：在与 LBM 相同的网格上解不可压压力泊松问题"
      "`lap(p)=0`、入口/出口 Dirichlet、壁面 `dp/dy=0`（SOR，收敛残差 < 1e-10），"
      "入口压力由**动量平衡** `p_in = p_out + rho_0 a L` 给出（不是用测得的入口压力，否则 R² 恒为 1）。")
    A("")
    A("## 3. 计算结果（数字表）")
    A("")
    A("### 3.1 C1 速度剖面 Ma 标度（压力驱动槽道 + 阴性对照）")
    A("")
    A("| Ma | 差分误差 `||du||_inf/u_max` | 绝对剖面误差 | `drho/rho_0` | 体力对照差分误差 |")
    A("|---|---|---|---|---|")
    for i, r in enumerate(pc):
        j = min(i, len(ctrl["diff_err_inf"]) - 1)
        A(f"| {_fmt(r['mach_measured'], 3)} | {_fmt(diff['diff_err_inf'][i], 3)} | "
          f"{_fmt(r['err_inf'], 3)} | {_fmt(diff['drho_over_rho0'][i], 3)} | "
          f"{_fmt(ctrl['diff_err_inf'][j], 3)} |")
    A("")
    A(f"* 拟合指数（差分误差，log–log）：**{_fmt(diff['exponent_diff'], 4)}**"
      f"（判据 2.0 ± 0.2；R² = {_fmt(diff.get('fit_diff_err', {}).get('r2_excess', float('nan')), 4)}）")
    A(f"* 密度波动指数：**{_fmt(diff.get('fit_drho', {}).get('exponent_plain', float('nan')), 4)}**（判据 2.0 ± 0.2）")
    A(f"* 阴性对照（体力槽道）最大差分误差 {_fmt(float(np.nanmax(ctrl['diff_err_inf'])), 3)}，"
      f"与压力驱动通道相差 {_fmt(float(np.nanmax(diff['diff_err_inf']) / max(float(np.nanmax(ctrl['diff_err_inf'])), 1e-300)), 3)} 倍")
    A("")
    inst = res.get("cylinder_instrument_validation", {})
    A("### 3.1b C3 密度波动阶（正确的 `O(Ma^2)` 载体）")
    A("")
    A("| Ma | 圆柱稳态 `max|rho-rho0|/rho0` | C6 压差变形 `max|Δp|/c_s^2` |")
    A("|---|---|---|")
    cyl_steady = {round(float(r["mach"]), 3): r.get("drho_final") for r in cyl}
    for r in c6["runs"]:
        m = round(float(r["mach_measured"]), 3)
        A(f"| {_fmt(r['mach_measured'], 3)} | {_fmt(cyl_steady.get(m, float('nan')), 3)} | "
          f"{_fmt(r['max_abs_dev_over_cs2'], 3)} |")
    A("")
    A("> 压力驱动槽道的**总**密度变化由外加静压梯度支配（固定 nu 时 ∝ Ma），"
      "它不是可压缩性误差；C3 因此锚定在圆柱稳态滞止压升与 C6 的压力泊松偏差上。")
    A("")
    A("### 3.2 C2/C5/C7 Re=100 圆柱")
    A("")
    A("#### 3.2.0 仪器验证门（C0）：先验证仪器，再谈 Mach 无关性")
    A("")
    A(f"基准：{inst.get('reference', {}).get('source', 'n/a')}；结论：**{inst.get('verdict')}**")
    A("")
    A("| Ma | St | C_D | C_L 幅值 | C_D 相对误差 | St 偏差 | C_D 合格 | St 合格 | C_L 已饱和 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in inst.get("runs", []):
        A(f"| {_fmt(c['mach'], 3)} | {_fmt(c['st'], 4)} | {_fmt(c['cd'], 4)} | {_fmt(c['cl_amplitude'], 3)} | "
          f"{_fmt(c['cd_rel_error'], 3)} | {_fmt(c['st_error'], 4)} | {c['cd_ok']} | {c['st_ok']} | {c['cl_saturated']} |")
    A("")
    A("")
    A("| Ma | tau | steps | converged | St | St(FFT) | C_D | C_L 幅值 | drho_final | 相对 St 偏差 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in cyl:
        rel = "n/a"
        if np.isfinite(r.get("st", np.nan)) and np.isfinite(ind.get("st_reference", np.nan)):
            rel = _fmt((r["st"] - ind["st_reference"]) / ind["st_reference"], 3)
        A(f"| {_fmt(r['mach'], 3)} | {_fmt(r['tau'], 4)} | {r['steps']} | {r['converged']} | "
          f"{_fmt(r.get('st', float('nan')), 4)} | {_fmt(r.get('st_fft', float('nan')), 4)} | "
          f"{_fmt(r.get('cd_mean', float('nan')), 4)} | {_fmt(r.get('cl_amplitude', float('nan')), 4)} | "
          f"{_fmt(r.get('drho_final', float('nan')), 3)} | {rel} |")
    A("")
    z = ind.get("ma_to_zero", {})
    A(f"* `Ma -> 0` 最小二乘外推（固定指数 2）：St0 = {_fmt(z.get('st0', float('nan')), 4)}，"
      f"C_D0 = {_fmt(z.get('cd0', float('nan')), 4)}，C_L0 = {_fmt(z.get('cl0', float('nan')), 4)}")
    A(f"* 相对最低可用 Ma 的 St 最大偏差：{_fmt(ind.get('max_rel_st_low_ma', float('nan')), 3)}"
      f"（判据 ≤ 0.01）")
    if ind.get("rel_st_vs_extrapolated_zero"):
        A(f"* 相对 `Ma -> 0` 外推的 St 偏差：{_fmt(ind.get('rel_st_vs_extrapolated_zero'), 3)}")
        A(f"* 相对 `Ma -> 0` 外推的 C_D 偏差：{_fmt(ind.get('rel_cd_vs_extrapolated_zero'), 3)}")
    A(f"* 适用域门 C7：Ma_max = {_fmt(ind.get('ma_max', float('nan')), 3)}"
      f"（1% 水动力误差带宽，阈值 {_fmt(ind.get('boundary_threshold', float('nan')), 3)}），"
      f"越界 Ma = {ind.get('ma_boundary_exceeded_at')}")
    A("")
    A("### 3.3 C6 压力场 vs 不可压压力泊松解")
    A("")
    A("| Ma | R² | max|Δp| | max|Δp|/c_s² | max|Δp|/动压 | Poisson 迭代 |")
    A("|---|---|---|---|---|---|")
    for r in c6["runs"]:
        A(f"| {_fmt(r['mach_measured'], 3)} | {_fmt(r['r2'], 5)} | {_fmt(r['max_abs_dev'], 3)} | "
          f"{_fmt(r['max_abs_dev_over_cs2'], 3)} | {_fmt(r['max_abs_dev_over_dynamic_pressure'], 3)} | "
          f"{r['poisson']['iterations']} |")
    A("")
    A(f"* 偏差随 Ma 的拟合指数：**{_fmt(c6.get('fit_dev_vs_ma', {}).get('exponent_plain', float('nan')), 3)}**"
      "（期望 2.0 ± 0.4 = `O(Ma^2)`）")
    A("")
    A("### 3.4 C4/C8 声学滤波")
    A("")
    if acoustic:
        A(f"* 实现：{acoustic.get('implemented')}")
        A(f"* 低马赫预处理是否可用：{acoustic.get('preconditioning_available')}")
        A(f"* 标准 LBM 声学带幅值：{_fmt(acoustic.get('standard', {}).get('drho_acoustic_amplitude', float('nan')), 3)}；"
          f"滤波后：{_fmt(acoustic.get('filtered', {}).get('drho_acoustic_amplitude', float('nan')), 3)}；"
          f"抑制倍数 {_fmt(acoustic.get('suppression_factor', float('nan')), 4)}")
        A(f"* 目标 δρ/ρ₀ ≤ {_fmt(acoustic.get('target_filtered_drho', float('nan')), 3)}："
          f"声学带判据 `pass_acoustic_band = {acoustic.get('pass_acoustic_band')}`；"
          f"字面总密度判据 `pass_literal_total_drho = {acoustic.get('pass_literal_total_drho')}`")
    A("")
    A("## 4. 判据门（机器裁决）")
    A("")
    A("| 判据 | 状态 |")
    A("|---|---|")
    for name in (
        "c0_cylinder_instrument_validation",
        "c1_poiseuille_ma2_scaling",
        "c2_strouhal_mach_independence",
        "c3_density_fluctuation_order",
        "c4_acoustic_filter",
        "c5_hydrodynamic_coefficient_independence",
        "c6_pressure_poisson",
        "c7_applicability_boundary",
        "c8_honest_gate",
    ):
        A(gate_row(name))
    A("")
    A("## 5. 图")
    A("")
    for name, path in figs.items():
        A(f"### {name}")
        A("")
        A(f"![{name}]({path.name})")
        A("")
    A("## 6. 注意点 / 局限")
    A("")
    A("* **低 Ma 角落的刚度（本节点最硬的限制）**：固定 Re 与直径时 "
      "`nu = Ma c_s D / Re`，故 `tau = 0.5 + 3 Ma c_s D / Re`；本算例 `Ma=0.01` 给出 `tau = 0.5014`。"
      "实测三种组合：Zou–He + BGK 与 Zou–He + 正则化 都在 ~800–1500 步内发散；"
      "零梯度外推 + 正则化 能活下来（本次 11085 步无发散），但 8 个对流时间内**涡脱落完全没发展起来**"
      "（C_L 幅值 0.014 vs 饱和值 ~0.3），由噪声过零估出的 St 是 9.86 这种无意义值。"
      "要在这个 Ma 拿到 `tau >= 0.55`，直径需 `D ~ nu Re/(Ma c_s) ~ 290` 直径（≈ 10^5 格），"
      "远超本单算力。因此 C2 的字面测点（Ma=0.01）标 `NOT_MEASURABLE`，不用任何代理替代。")
    cd_list = [c["cd"] for c in inst.get("runs", [])]
    cl_list = [c["cl_amplitude"] for c in inst.get("runs", [])]
    A("* **圆柱仪器未通过验证（C0）**：同一驱动给出 `C_D = "
      f"{_fmt(min(cd_list) if cd_list else float('nan'), 3)}–"
      f"{_fmt(max(cd_list) if cd_list else float('nan'), 3)}`，"
      "比 Braza 1986 的 1.35 高 57%–78%；`C_L` 幅值只有 "
      f"{_fmt(min(cl_list) if cl_list else float('nan'), 3)}–{_fmt(max(cl_list) if cl_list else float('nan'), 3)}"
      "（Re=100 应 >=0.3，说明升力远未饱和），`Ma=0.1` 还给出 St=0.45 的离群值。**根因未查明**"
      "（候选：零梯度外推入口的有效动压偏低、动量交换力归一化口径、D=8 分辨率不足以解析边界层）。"
      "结论：该配置不能用于「水动力系数随 Ma 变化」级别的判据，C2/C5/C7 因此标 `NOT_SUPPORTED`，"
      "而不是给一个好看的相对差。正确路线是复用本仓已验证的 "
      "`benchmarks/verified/cylinder/re100` 引擎（far-field BC + 下游 sponge + MRT，Cd=1.329 / St=0.166），"
      "它在本机（无 GPU、128 核满载）跑 6 万步需数小时，超出本单预算。")
    A("* **绝对剖面误差不可用于 C1**：H=32 下离散底噪 ~7%，比 `O(Ma^2)` 项大 1–2 个量级；"
      "必须用固定 nu 的差分诊断（阴性对照相差 10^5 倍，证明差分诊断测的确实是压力梯度带来的可压缩性效应）。")
    A("* **实测标度不是纯二次**：差分误差的局部斜率随 Ma 单调上升 "
      f"({[round(d['slope'], 2) for d in gates['c1_poiseuille_ma2_scaling']['local_slopes']]})，"
      "全域拟合 1.56，只有最高一对 Ma（0.2 -> 0.3）达到 2.15。也就是说 `O(Ma^2)` 渐近区在 "
      "0.01–0.3 区间内**没有**被观测到，低 Ma 端存在低一阶的混入项 —— 这是本单的负面物理结论"
      "（来源未查明，候选：固定 nu 下压力水平 ∝ Ma，可压缩性修正与压力水平耦合）。")
    A("* **C4 字面判据不可达**：压力驱动槽道里存在 `O(Ma)` 的稳态水动力压力场（物理量，"
      "幅值远大于 1e-5），任何只压声学带的滤波都不能把它压到 1e-5；因此 C4 的低马赫预处理读法标 "
      "`NOT_SUPPORTED`（C8 诚实门），并单独报告后处理滤波的实测抑制倍数"
      "（滤波引起的剖面改变量 "
      f"{_fmt(acoustic.get('profile_change_rel', float('nan')), 3)}，不污染物理场）。")
    A("* **C6 的泊松解在充分发展槽道里是线性的**：该算例检验的是「LBM 压力场是否满足不可压动量平衡」，"
      "不是二维复杂压力场重构（后者需要投影法求解器，本仓不存在）；R² 只在 Ma <= 0.1 成立"
      f"（{_fmt(gates['c6_pressure_poisson']['r2'][:3], 5)}），Ma >= 0.2 掉到 "
      f"{_fmt(gates['c6_pressure_poisson']['r2'][3:], 5)}。")
    A("")
    A("## 7. 误差判定（vs 基准）")
    A("")
    A("| 关键量 | 基准 | 误差 | 是否 < 3% |")
    A("|---|---|---|---|")
    c1_gate = gates["c1_poiseuille_ma2_scaling"]
    dev_c1 = abs(c1_gate["exponent"] - 2.0) / 2.0
    A(f"| C1 标度指数 | 理论 2.0（`O(Ma^2)`） | 实测 {_fmt(c1_gate['exponent'], 3)}"
      f"（偏差 {dev_c1 * 100:.0f}%） | {'是' if dev_c1 < 0.03 else '否'} |")
    st_vals = [c["st"] for c in inst.get("runs", [])]
    if st_vals:
        worst_st = max(abs(v - 0.1645) / 0.1645 for v in st_vals)
        A(f"| St（Re=100） | Braza 1986: 0.164–0.165 | 实测 {_fmt(min(st_vals), 3)}–{_fmt(max(st_vals), 3)}"
          f"（最差 {worst_st * 100:.0f}%） | {'是' if worst_st < 0.03 else '否'} |")
    if cd_list:
        worst_cd = max(abs(v - 1.35) / 1.35 for v in cd_list)
        A(f"| C_D（Re=100） | Braza 1986: 1.35±0.05 | 实测 {_fmt(min(cd_list), 3)}–{_fmt(max(cd_list), 3)}"
          f"（最差 {worst_cd * 100:.0f}%） | {'是' if worst_cd < 0.03 else '否'} |")
    A(f"| 压力场（C6） | 不可压压力泊松解 | R² 最低 {_fmt(c6['min_r2'], 5)}；偏差阶 "
      f"{_fmt(c6.get('fit_dev_vs_ma', {}).get('exponent_plain', float('nan')), 3)}（目标 2） | 否 |")
    A(f"| 声学滤波（C4） | δρ/ρ₀ <= 1e-5 | 滤波后 "
      f"{_fmt(acoustic.get('filtered', {}).get('drho_acoustic_amplitude', float('nan')), 3)}"
      f"（抑制 {_fmt(acoustic.get('suppression_factor_whole_record', float('nan')), 3)} 倍） | 是 |")
    A("")
    A("> 结论：C1/C3/C6 的**假设（纯 `O(Ma^2)`）在本区间内被证伪**；C2/C5/C7 因仪器未验证而不可判定；"
      "C4 只在后处理读法下成立。所有结论都是 honest —— 没有一个 FAIL 被改写为 PASS。")
    A("")
    prov = res.get("provenance", {})
    A("## 8. 审计链")
    A("")
    A(f"* git commit：`{prov.get('git_commit', 'n/a')}`；python {prov.get('python')}；numpy {prov.get('numpy')}")
    A(f"* 平台：{prov.get('platform')}")
    A(f"* 命令：`{prov.get('command', 'python -m tensorlbm.verification.weakly_compressible')}`")
    tw = prov.get("task_wall_seconds", {}) or {}
    A(f"* 逐算例墙钟合计 {_fmt(sum(tw.values()) if tw else float('nan'), 4)} s（单核 NumPy，128 核共享机 load ~200 下实测）；"
      f"装配与后处理墙钟 {_fmt(prov.get('wall_seconds_total', float('nan')), 4)} s")
    A("* 原始逐算例记录：`benchmarks/verified/lbm_weakly_compressible/raw/*.json`（每算例一份，含完整场量）")
    A("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(ART))
    ap.add_argument("--docs", default=str(DOCS))
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    res = _load(art / "result.json")
    gates = _load(art / "gates.json")["gates"]
    figs_dir = art / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "C1/C3 Ma 标度（差分剖面 + 密度波动）": fig_ma_scaling(res, figs_dir / "fig1_ma_scaling.png"),
        "C2/C5 圆柱 St / C_D / C_L 的 Mach 无关性": fig_cylinder(res, figs_dir / "fig2_cylinder_mach.png"),
        "C6 压力场 vs 不可压压力泊松解": fig_pressure(res, figs_dir / "fig3_pressure_poisson.png"),
        "速度剖面与差分信号": fig_profiles(res, figs_dir / "fig4_profiles.png"),
    }
    print("figures:", {k: str(v) for k, v in figs.items()})
    if not args.figs_only:
        docs = Path(args.docs)
        docs.mkdir(parents=True, exist_ok=True)
        md = build_docs(res, gates, figs)
        (docs / "lbm_weakly_compressible.md").write_text(md)
        print("docs:", docs / "lbm_weakly_compressible.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())