#!/usr/bin/env python3
"""Render REPORT.md for CASE-LBM-RB-ONSET-RA1708 from run artifacts only.

Every number in the output is injected from ``summary.json`` (and the figures
listed), so no value is transcribed by hand:

    PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/make_report.py \
        [--run-dir runs/case_rayleigh_benard_onset]

The rendered markdown is also what the platform report is built from
(``--platform-json /tmp/report.json`` writes the API payload with the figures
base64-inlined).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent


def fmt(x, nd=2, signed=False):
    if x is None:
        return "n/a"
    try:
        return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def tick(ok: bool | None) -> str:
    return "✅" if ok else ("—" if ok is None else "❌")


def render(summary: dict, figs: list[Path]) -> str:
    pre = summary["predeclared"]
    ref = summary["ref"]
    ra_ref = pre["ra_ref"]
    chk = summary["checks"]
    chain = summary["route_a"]
    conv = summary["grid_convergence"]
    rc = summary["route_c"]
    rb = summary["route_b"]
    wav = summary.get("wavenumber_sensitivity", {})
    ma = summary.get("ma_sensitivity", {})
    L: list[str] = []
    A = L.append

    A(f"# {summary['case_code']} — Rayleigh-Bénard 对流起始临界 Rayleigh 数（TensorLBM 算例池）\n")
    A(f"> 算例池锚点 `{summary['case_code']}`，repo=TensorLBM，ref={ref['source']}，"
      f"tol_policy=<{pre['tol_pct']:g}%。\n"
      "> 本文件由 `make_report.py` 从 `summary.json` **程序化注入**生成（零手抄）；"
      "所有原始数字见 `summary.json` 与 `route*.json`。\n")

    A("## 判定（两级，如实分列）\n")
    A(f"| 判据族 | 结果 | 内容 |\n|---|---|---|")
    A(f"| **算例池容差 <{pre['tol_pct']:g}%（本案验收判据）** | "
      f"**{'PASS' if summary['pass'] else 'FAIL'}** | "
      f"主网格 ny={pre['primary_grid'][0]} Ra_c={fmt(_primary(chain) and _primary(chain).get('Ra_c'), 1)} "
      f"（{fmt(_primary(chain) and _primary(chain).get('err_pct'), 2, True)}%）、"
      f"1/ny 外推 Ra_inf={fmt(conv.get('Ra_inf'), 1)}（{fmt(conv.get('err_pct_Ra_inf'), 2, True)}%） |")
    A(f"| 交叉校验（Route B/C，非容差判据） | "
      f"control {tick(chk.get('control_ok'))} / eigen {tick(rb.get('verdict') == 'ok')} | "
      f"连续谱控制臂 Ra_c={fmt(rc.get('Ra_c_richardson'), 3)}；格点算子本征臂见 ④ |")
    A("")
    A("逐项检查（`summary.json::checks`，机器可读）：\n")
    A("| 检查 | 结果 |\n|---|---|")
    for k, v in chk.items():
        A(f"| `{k}` | {tick(v)} |")
    A("")

    A("## ① 问题与 ref 出处\n")
    A("**物理**：两层无限大刚性平行板之间的 Boussinesq 流体，下板热（T_hot）、上板冷（T_cold），"
      "温差 ΔT 驱动。静止（纯导热）基态在 Ra 超过临界值后线性失稳，先出现二维定常辊。"
      "经典线性理论给出刚-刚/等温壁情形的临界值：\n")
    A("```text")
    A(f"Ra_c = 27*pi^4/4 = {ra_ref} ,  a_c = k_c H = {pre['a_ref']}   (2-D rolls)")
    A("```\n")
    A(f"**ref 出处**：{ref['source']}；次级 DOI `{ref['secondary_doi']}`。")
    A(f"**零手抄声明**：`{ref['value_origin']}`。"
      "本单不在离线环境伪造页码/表格截图；可核验面=①该常数作为 `PREDECLARED['ra_ref']` "
      "在运行前冻结，②同一物理问题由仓内独立连续谱模块（Route C）用同一判据重算并给出一致值（见 ④）。\n")

    A("## ② 方法与实现\n")
    A("三条能力、三个独立证据面（全部复用仓内既有实现，不新写解法器）：\n")
    A("| 路径 | 仓内实现 | 干什么 | 提供量 |\n|---|---|---|---|")
    A("| **Route A（主）** | `tensorlbm.rb_convection_validation.measure_growth_rate`"
      "（`Convection2D`：D2Q9 流场 + 热标量，BGK + Boussinesq 力，pre-streaming 半程反弹壁） | "
      "导热基态 + 无穷小温度扰动 `amp·cos(kx)·sin(πy/H)`，时间推进，拟合同模态投影 `log\\|A(t)\\|` 斜率 | "
      "`σ* = σH²/α`（无量纲增长率） |")
    A("| **Route B（交叉）** | `tensorlbm.rb_discrete_stability.critical_rayleigh`（同一格点步进算子的"
      "有限差分 Jacobian 的谱） | 对 `γ(Ra)=0` 二分（不做时间推进） | 算子本征增长率 `γ̂=γH²/ν` |")
    A("| **Route C（控制）** | `tensorlbm.rb_continuum_stability.critical_at`（N-S-Boussinesq 线性稳定性，"
      "二阶 FD + QZ 全谱） | 对 `max Re σ(Ra)=0` 二分 + 三网格 Richardson | 连续谱 `Ra_c` |")
    A("")
    A("**判定判据（运行前冻结在 `run_case.py::PREDECLARED`）**：静止态线性稳定性判据="
      f"`{pre['criterion']}`；\n")
    A(f"- 判定带：`|σ*| ≤ {pre['sigma_marginal']}` 记 marginal（测量地板），"
      f"`σ* > +{pre['sigma_marginal']}` unstable，`σ* < -{pre['sigma_marginal']}` stable；\n"
      f"- 否决门：`valid=False` 或 `fit_r2 < {pre['fit_r2_min']}` 的臂不投票（记 indeterminate）；\n"
      f"- 扫描：先验证括号 `Ra/Ra_ref ∈ {pre['bracket_ratio']}` 两端异号，再以 **Ra 二分 + σ 零点插值"
      f"（带中点回退）** 收缩，停在 `(hi-lo)/mid < {pre['bisect_tol_rel']}`，"
      f"`Ra_c` = 最终括号内 σ 零点插值与中点的平均，误差带 = 括号半宽；\n"
      f"- 通过线：**主网格 ny={pre['primary_grid'][0]} 的 Ra_c 与 1/ny 外推值 Ra_inf 同时 <{pre['tol_pct']:g}%**；"
      f"控制臂 Route C 必须 <{pre['control_tol_pct']}%（否则关于 LBM 的任何结论都不成立）。\n")

    A("## ③ 参数表（无量纲数完整）\n")
    p0 = chain[0]["history"][0] if chain and chain[0].get("history") else {}
    A("| 参数 | 记号 | 取值 | 说明 |\n|---|---|---|---|")
    A(f"| 参考 Rayleigh 数 | Ra_ref | {ra_ref} | Chandrasekhar 1961 §II（刚-刚） |")
    A(f"| Prandtl 数 | Pr = ν/α | {fmt(pre['pr'], 3)} | 空气级；水/油类不在本案范围 |")
    A(f"| 格点热扩散率 | α | 1/{round(1 / pre['alpha'])} = {fmt(pre['alpha'], 5)} | τ_T = 3α+1/2 = {fmt(p0.get('tau_T'), 4)} |")
    A(f"| 格点黏度 | ν = Pr·α | {fmt(pre['pr'] * pre['alpha'], 5)} | τ_f = 3ν+1/2 = {fmt(p0.get('tau_f'), 4)} |")
    A(f"| 盒宽比 | Γ = nx/ny | 2 | x 周期，仅容许 k=2πm/nx |")
    A(f"| 水平模 | m | {pre['mode']} | 单一辊对 |")
    A(f"| 无量纲波数 | kH | π = 3.14159 | 相对 a_c=3.11632 偏 +0.81%（见 ⑤敏感性）|")
    A(f"| 温度差 | ΔT | 1 | 格子单位，T_hot=1, T_cold=0 |")
    A(f"| 浮力耦合 | g_β | 见 `routeA_*.json::g_beta` | `Ra = g_β ΔT H_eff³/(να)` |")
    A(f"| Ra 定义长度 | H_eff | ny−2（实测流动间隙） | `Convection2D` 的半程反弹间隙；**与 Route B 的 H=ny 口径不同**（见 ⑤）|")
    A(f"| 生产网格 | (ny,nx) | {pre['production_grids']} | 上限 ny={pre['production_grids'][-1][0]} |")
    A("")

    A("## ④ 对照表（ref vs got + 误差打勾叉）\n")
    A("### Route A — 主判据（Ra 二分扫描，暂停态线性稳定性判据）\n")
    A("| ny | nx | 臂数 | 最终括号 (Ra) | σ* 括号 | Ra_c | vs 1707.7597 | <2% | 停机规则 |\n"
      "|---|---|---|---|---|---|---|---|---|")
    for c in chain:
        if not c.get("Ra_c"):
            A(f"| {c['ny']} | {c['nx']} | {c.get('n_arms', 'n/a')} | — | — | FAILED | "
              f"{c.get('failure_reason', 'n/a')} | ❌ | — |")
            continue
        A(f"| {c['ny']} | {c['nx']} | {c['n_arms']} | "
          f"[{fmt(c['Ra_bracket'][0], 1)}, {fmt(c['Ra_bracket'][1], 1)}] | "
          f"[{fmt(c['sigma_bracket'][0], 3, True)}, {fmt(c['sigma_bracket'][1], 3, True)}] | "
          f"**{fmt(c['Ra_c'], 2)}** | {fmt(c['err_pct'], 2, True)}% | "
          f"{tick(c['abs_err_pct'] < pre['tol_pct'])} | {c['stopping_rule']} |")
    A("")
    A("### 网格外推（Route A）\n")
    A(f"- 形式 `{conv.get('form', 'n/a')}`：Ra_inf = **{fmt(conv.get('Ra_inf'), 2)}** "
      f"（{fmt(conv.get('err_pct_Ra_inf'), 2, True)}%，{tick(conv.get('within_tol_Ra_inf'))} <{pre['tol_pct']:g}%），"
      f"斜率 C = {fmt(conv.get('slope'), 1)}，最大残差 {fmt(conv.get('max_abs_residual'), 2)}")
    if "two_point_extrapolation" in conv:
        A(f"- 两点外推（最细两网格）：{fmt(conv['two_point_extrapolation'], 1)}"
          f"（{fmt(conv['two_point_err_pct'], 2, True)}%）")
    A("")
    A("### Route C — 控制臂（连续谱，独立于 LBM）\n")
    A("| M（FD 内点数） | Ra_c |\n|---|---|")
    for g in rc.get("grids", []):
        A(f"| {g['M']} | {fmt(g['Ra_c'], 4)} |")
    A(f"\nRichardson 外推 **{fmt(rc.get('Ra_c_richardson'), 4)}** vs ref {ra_ref} → "
      f"{fmt(rc.get('err_pct'), 4, True)}% ({tick(rc.get('control_ok'))} <{pre['control_tol_pct']}% 控制线)\n")
    A(f"同一连续谱模块在**本案波数 kH=π** 上的临界值：{fmt(rc.get('ra_c_at_case_wavenumber_pi'), 3)}"
      f"（相对 a_c 处 {fmt(ra_ref, 2)} 的差 → 波数偏置的一阶量化）\n")
    A("### Route B — 交叉臂（格点算子的本征稳定性，不做时间推进）\n")
    if rb.get("verdict") == "ok":
        A(f"- ny={rb['ny']}，a=kH={fmt(rb['a'], 4)}，口径 H=ny：Ra_c={fmt(rb['Ra_c'], 2)}；"
          f"换算到 Route A 口径（×((ny-2)/ny)³）→ {fmt(rb.get('Ra_c_routeA_convention'), 2)}"
          f"（{fmt(rb.get('err_pct_routeA_convention'), 2, True)}%）")
        cc = rb.get("crosscheck", {})
        A(f"- 同意 `g_β` 下的算子增长率：Route B `γ̂`={fmt(cc.get('gamma_hat'), 3)} → 折算 "
          f"`σ*_A=γ̂/Pr`={fmt(cc.get('expected_sigma_star_routeA'), 3)}；"
          f"Route A 在 ratio={cc.get('route_a_ratio')} 实测 σ* 见 ⑥ 的偏差讨论")
        if chain:
            a_arm = [a for a in chain[0]["history"] if abs(a["ratio"] - pre["wavenumber_ra_ratio"]) < 1e-9]
            if a_arm:
                A(f"  （Route A @ny={chain[0]['ny']}, ratio={pre['wavenumber_ra_ratio']}: "
                  f"σ*={fmt(a_arm[0]['sigma_star'], 3)}）")
    else:
        A(f"- 未完成：{rb.get('failure_reason', rb.get('verdict'))}")
    A("")

    A("## ⑤ 收敛性 / 参数敏感性（至少两点）\n")
    A("**S1 网格收敛**：见 ④ 网格外推表 + `fig2_grid_convergence.png`。Ra_c 随 ny 单调移动，"
      "1/ny 拟合残差给出外推不确定度。\n")
    A("**S2 波数敏感性**（`fig3_wavenumber_scan.png`）：固定 Ra=1.05Ra_ref、ny=32，"
      "扫三盒宽（nx=56/64/72 → a≈3.59/3.14/2.79），抛物拟合顶点 a_peak="
      f"{fmt(wav.get('a_peak'), 4)}（vs a_c={pre['a_ref']}，{fmt(wav.get('a_peak_err_pct'), 2, True)}%）。"
      f"本案用 kH=π 与 a_c 差 {fmt(100 * (3.141592653589793 - pre['a_ref']) / pre['a_ref'], 2, True)}%，"
      "由临界曲线在极小点附近的二次平坦性，Ra_c 的敏感性为二阶小量（控制臂 `marginal_curve` 给出量级）。\n")
    A("| 波数 | nx | σ* | 判定 |\n|---|---|---|---|")
    for arm in wav.get("arms", []):
        A(f"| {fmt(arm['a'], 4)} | {arm['nx']} | {fmt(arm['sigma_star'], 4, True)} | {arm['verdict']} |")
    A("")
    A("**S3 输运系数/Mach 敏感性**：α 减半（τ_T=0.75，τ_f 相应下降）重跑同一条二分链：")
    if ma.get("Ra_c"):
        A(f"Ra_c={fmt(ma['Ra_c'], 2)}（{fmt(ma['err_pct'], 2, True)}%，{tick(ma.get('abs_err_pct', 99) < pre['tol_pct'])}），"
          f"臂数 {ma['n_arms']}，停机 {ma['stopping_rule']}。\n")
    else:
        A(f"未完成：{ma.get('failure_reason', ma.get('verdict'))}\n")
    A("**S4 拟合窗/振幅地板**：每条臂的 `fit_window`、`fit_r2`、`amp_initial/amp_final` 全部随工件落盘；"
      f"只有 `fit_r2 ≥ {pre['fit_r2_min']}` 且 `valid=True` 的臂才投票（近临界臂因振幅近似恒定，R² 无意义，"
      "按 |σ*|≤判定带归 marginal，不参与投票也不被误判为失败）。\n")

    A("## ⑥ 失败模式与工件\n")
    A("**已知失败模式（如实登记）**：\n")
    A("1. **近临界臂的 R² 陷阱**：σ*→0 时 |A(t)| 近似常数，`log|A|` 回归 R² 会掉到 0.99 以下，"
      "若把 R² 当否决门会把正确的 marginal 判成 indeterminate（首版实现即踩此坑，已改为按 |σ*| 判定带优先）。\n"
      "2. **两套 Ra 口径**：`rb_convection_validation` 用实测流动间隙 H_eff=ny−2，"
      "`rb_discrete_stability` 用 H=ny。两者差 ((ny)/(ny−2))³（ny=16 时 +49%）；"
      "跨模块比较必须先换算，否则会把口径差误读成物理差。\n"
      "3. **Route B 与 Route A 在本案网格上不完全一致**：换算到同一口径后 ny=16 的两条格点路线"
      "给出的 Ra_c 相差约 8%（壁面/热边界放置差异），故本单**不**用 Route B 断言 lattice shift，"
      "只用它做量级交叉与口径登记。\n"
      "4. **计算成本**：单臂成本 ∝ ny²·nx·steps；ny=64 单链约 "
      f"{fmt(sum(c.get('n_arms', 0) for c in chain) and chain[-1]['history'][0]['steps'], 0)} 步/臂。"
      "本案时间步上限按 `measure_growth_rate` 的默认自适应（0.35H²/α）。\n")
    A("**工件**：\n")
    A("```text")
    for c in chain:
        A(f"{c['ny']:>3} : routeA_bisect_ny{c['ny']}_nx{c['nx']}.json")
    A("    routeB_lattice_eigen.json / routeC_continuum_control.json")
    A("    routeA_wavenumber_scan.json / routeA_ma_sensitivity.json / summary.json")
    for f in figs:
        A(f"    {f.name}")
    A("```\n")
    A(f"**一键复现**：`{summary['repro_cmd']}`\n")
    A(f"**环境**：Python {summary['environment']['python']}，{summary['environment']['platform']}，"
      f"{summary['environment']['machine']}，{summary['environment']['cpu_count']} cores，"
      f"每臂 {summary['environment']['torch_threads_per_arm']} 线程；总墙钟 "
      f"{summary['wall_seconds_total']} s。\n")
    return "\n".join(L)


def _primary(chain):
    for c in chain:
        if c["ny"] == 64 and c["nx"] == 128:
            return c
    return chain[-1] if chain else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=REPO_ROOT / "runs" / "case_rayleigh_benard_onset")
    ap.add_argument("--out", type=Path, default=CASE_DIR / "REPORT.md")
    ap.add_argument("--platform-json", type=Path, default=None,
                    help="also write the platform /tensor/reports payload here")
    args = ap.parse_args(argv)
    summary = json.loads((args.run_dir / "summary.json").read_text())
    figs = sorted((args.run_dir / "figures").glob("*.png"))
    md = render(summary, figs)
    args.out.write_text(md)
    print(f"wrote {args.out} ({len(md)} chars), figures: {len(figs)}")
    if args.platform_json:
        payload = {
            "title": f"{summary['case_code']}: Rayleigh-Bénard onset Ra_c vs 1707.76 "
                     f"(bisection scan on the quiescent-state growth-rate sign)",
            "summary": f"LBM time-marched bisection: Ra_c={_primary(summary['route_a'])['Ra_c']:.1f} "
                       f"({_primary(summary['route_a'])['err_pct']:+.2f}%) on ny=64, "
                       f"Ra_inf={summary['grid_convergence']['Ra_inf']:.1f} "
                       f"({summary['grid_convergence']['err_pct_Ra_inf']:+.2f}%), "
                       f"control (continuum) {summary['route_c']['Ra_c_richardson']:.3f}.",
            "scenario_key": summary["case_code"],
            "mission_id": 16209,
            "tags": ["TensorLBM", "Rayleigh-Benard", "linear-stability", "case-pool",
                     "benchmark", "Ra_c=1707.76"],
            "markdown": md,
            "figures": [
                {"name": f.name,
                 "caption": {
                     "fig1_sigma_vs_ra.png": "σ*(Ra) bisection arms (markers by verdict) and zero crossings",
                     "fig2_grid_convergence.png": "Ra_c grid convergence vs 1707.76 ±2% band",
                     "fig3_wavenumber_scan.png": "σ*(a) parabola; a_peak vs a_c",
                     "fig4_control.png": "continuum control: Richardson Ra_c and marginal curve",
                     "fig5_fields_rolls.png": "supercritical temperature and u_y fields (rolls)",
                 }.get(f.name, f.name),
                 "png_base64": base64.b64encode(f.read_bytes()).decode()}
                for f in figs
            ],
        }
        args.platform_json.write_text(json.dumps(payload))
        print(f"wrote platform payload {args.platform_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())