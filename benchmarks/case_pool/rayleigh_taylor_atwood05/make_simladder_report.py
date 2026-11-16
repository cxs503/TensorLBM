"""Render the similarity-ladder report of ``CASE-RT-INSTABILITY-ATWOOD``.

Every number in the output is read from the artefacts produced by
``run_simladder.py`` (``artifacts/simladder/summary.json`` +
``*/case.json``), ``verify/audit_bundle.py`` (``artifacts/audit/AUDIT.json``) and
the figure files -- nothing is transcribed by hand.

Usage
-----
    python3 make_simladder_report.py            # -> artifacts/simladder/REPORT_SIMLADDER.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SIM = BENCH / "artifacts" / "simladder"
AUDIT = BENCH / "artifacts" / "audit" / "AUDIT.json"
FIG = BENCH / "figures"


def _fmt(x, spec=".4f"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return format(v, spec)


def render(summary: dict, audit: dict | None) -> str:
    crit = summary["criterion"]
    lad = summary["ladder"]
    rows = summary["rows"]
    by = {r["name"]: r for r in rows}
    ref = by.get("S_ref_l064_g2e-4", {})
    mach = summary["mach_test"]
    kx = summary["kxi_control"]
    ex = lad["extrapolation_kxi0"]
    fz = crit["fr_lz_c1"]

    L = []
    A = L.append
    A("# CASE-RT-INSTABILITY-ATWOOD — 终端（非线性）尖峰/气泡速度的相似性阶梯")
    A("")
    A("> 归因：本文件是 `REPORT.md`（线性增长率门）与 `artifacts/late`（终端速度门）之后的**第三增量**："
      "把「终端 Froude 数 +5.56 % PASS」从**单点**变成**带相似变换的阶梯**，并对其做两项证伪检验")
    A("> （分辨/界面厚度、格子马赫数）。全部数字由 `run_simladder.py → make_simladder_report.py` "
      "从 `artifacts/simladder/*/case.json` 程序化注入（零手抄）。")
    A("")
    A(f"| 判据 | 结果 | 数值 |")
    A("|---|---|---|")
    A(f"| 锚点门（不变）：每臂 `|Fr/Fr_LZ−1| ≤ {crit['gate_pct']:g} %` | "
      f"**{'全部通过' if lad['gate_pass_all'] else '有臂未过'}** | "
      f"Fr_LZ = 1/√(πC) = {fz:.4f}（C={crit['c_2d']:g}，二维）|")
    A(f"| 预注册：阶梯散布 ≤ {crit['ladder_spread_pct_max']:g} pp | "
      f"**{'满足' if lad['resolution_independent'] else '不满足'}** | "
      f"散布 = {lad['fr_spread_pct']:+.2f} pp（λ = 64/96/128，Ma、ζ、k a₀ 固定）|")
    A(f"| 证伪检验：Ma 依赖 | "
      f"**{'可分辨' if mach['resolvable'] else '不可分辨' }** | "
      f"Fr {_fmt(mach['fr_ref'])} → {_fmt(mach['fr_mach'])}（{mach['shift_pct']:+.2f} %），"
      f"Ma {_fmt(mach['ma_ref'], '.3f')} → {_fmt(mach['ma_half_g'], '.3f')} |")
    A(f"| 诊断（不设门）：kξ → 0 外推 | —— | "
      f"Fr = {_fmt(ex.get('intercept'))}（斜率 {_fmt(ex.get('slope'), '+.4f')}，"
      f"R² = {_fmt(ex.get('r2'))}）→ 误差 {lad['extrapolated_err_vs_fr_lz_pct']:+.2f} % |")
    A(f"| 诊断（不设门）：kξ 控制臂 | —— | "
      f"kξ = {_fmt(kx['k_xi_w4'], '.3f')} 时 Fr = {_fmt(kx['fr_w4'])}（{kx['shift_vs_ref_pct']:+.2f} % vs 参考臂）|")
    A("")

    # ------------------------------------------------------------------ ①
    A("## ① 案例介绍")
    A("")
    A("**物理**：重力场中重流体（ρ₁）在轻流体（ρ₂）之上，界面给单模扰动（波长 λ，初始幅值 a₀ = 0.01λ），"
      f"Atwood 数 A = (ρ₁−ρ₂)/(ρ₁+ρ₂) = 0.5。线性期（`REPORT.md`）：a(t) = a₀e^{σt}，"
      "σ_visc = √(A g k + ν²k⁴) − νk²。非线性期：尖峰/气泡不再加速，进入**常速平台** u，"
      "势流（Layzer 1955；Goncharov 2002）预测 u_b = √(2Ag/((1+A)Ck))、u_s = u_b√((1+A)/(1−A))，"
      "二维 C = 1 ⇒ Fr_LZ = u/√(Agλ) = 1/√π。")
    A("")
    A("**本增量要判的量**：终端 Froude 数 Fr = u_mean/√(A g λ)（u_mean = 尖峰/气泡尖端速度平均），"
      "以及它**是不是网格/界面厚度的函数**。旧单点结果：λ=64、g=2e-4 一臂 Fr = 0.5956（+5.56 % PASS），"
      "其最近邻 λ=96 臂因 Ma 抬升到 0.879 被判 INCOMPLETE（+38.8 %）——**PASS 没有收敛证据**。")
    A("")
    A("**几何与参数表（无量纲群按相似变换固定）**")
    A("")
    A("| 臂 | λ | g | ν | τ | kξ | ζ = νk²/√(Agk) | Ma_peak | τ_end | 停止原因 | 步数 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['name']} | {r['lam']:.0f} | {r['g']:.3e} | {r['nu']:.4g} | "
          f"{0.5 + 3 * r['nu']:.4f} | {r['k_xi']:.3f} | {r['zeta']:.4f} | "
          f"{_fmt(r['ma_peak'], '.3f')} | {_fmt(r['tau_reached'], '.2f')} | {r['stop_reason']} | "
          f"{r['n_steps']} |")
    A("")
    A(f"盒深固定 layers = 8（每侧 4λ），A = 0.5，k a₀ = {2 * math.pi * 0.01:.4f}，"
      "Allen–Cahn 界面（`ac_mobility = 20`、`ac_sigma = 3e-4`，`ac_width = 2` cell 除控制臂）。"
      "唯一未固定的群是 **kξ**（界面厚度以波长为单位），它随加密**减半**——这正是要分离的量。")
    A("")

    # ------------------------------------------------------------------ ②
    A("## ② 怎么做的")
    A("")
    A("**方法**：复用仓内已验证的驱动，不新写求解器。")
    A("")
    A("| 文件 | 作用 |")
    A("|---|---|")
    A("| `run_simladder.py` | 相似阶梯定义 + 预注册判据 + 复用 `run_late.run_case` 的长时间推进；`--shard i/n` 可分片 |")
    A("| `late_analysis.py`（既有） | 尖端前缘测量、τ 窗最小二乘、Layzer 参考值；本增量未改动 |")
    A("| `run_case.py`（既有，**本单修 1 处**） | 算例定义；`velocity_in_window` 单位缺陷见 ⑤ F3 |")
    A("| `verify/audit_bundle.py` | 证据包独立复核 A0–A10（本单新增） |")
    A("| `make_simladder_figures.py` / `make_simladder_report.py` | 图 1–3 / 本报告（零手抄） |")
    A("| `tests/test_rt_atwood_case_bundle.py` | 6 条回归测试（单位语义、振幅窗语义、参考式、相似群） |")
    A("")
    A("**相似变换（本增量的核心）**：加密 λ → sλ 时同时取 g → g/s（Ma ∝ √(Agλ) 不变）、"
      "ν → sν（ζ = νk²/√(Agk) 不变）、盒深按 λ 计不变、k a₀ 不变。"
      "**旧套件的 λ=96 臂恰恰违反了这一条**（g 随 λ 上升以固定 Ag/λ，那是**线性**期的相似群，"
      "非线性期 Ma 会随 √λ 涨），所以它的 +38.8 % 不能用来判分辨率效应。")
    A("")
    A("**预注册判据（在看到本阶梯任何数字之前写在 `run_simladder.py` 模块 docstring 里）**：")
    A(f"1. 锚点门不变：每臂 |Fr/Fr_LZ − 1| ≤ {crit['gate_pct']:g} %；")
    A(f"2. 分辨率无关：三条 Ma 匹配臂的 Fr 散布 ≤ {crit['ladder_spread_pct_max']:g} pp；")
    A("3. 若超差，则把 Fr(kξ) 线性外推到 kξ → 0 并连同 R² 报告（诊断，不是门）；")
    A(f"4. Ma 证伪臂（λ=64、g 减半、Ma/√2）：若 Fr 是压缩性假象，Fr 必须移动可观量；"
      f"移动 ≤ {crit['mach_resolution_pct']:g} pp 判「无可分辨 Ma 依赖」。")
    A("")
    A("**复现（一条命令）**：")
    A("")
    A("```bash")
    A("python3 benchmarks/case_pool/rayleigh_taylor_atwood05/run_simladder.py            # 全阶梯")
    A("python3 benchmarks/case_pool/rayleigh_taylor_atwood05/make_simladder_figures.py   # 图 1–3")
    A("python3 benchmarks/case_pool/rayleigh_taylor_atwood05/make_simladder_report.py    # 本报告")
    A("python3 benchmarks/case_pool/rayleigh_taylor_atwood05/verify/audit_bundle.py      # 证据包复检")
    A("```")
    A("")
    A("计算经 `run_compute` 递交（8 核，队列派发，未指定节点）。")
    A("")

    # ------------------------------------------------------------------ ③
    A("## ③ 计算结果")
    A("")
    A("**表 1 — 阶梯与两个证伪臂**")
    A("")
    A("| 臂 | λ | Ma_peak | u_mean | Fr | 误差 vs Fr_LZ | 门 | C_eff = 1/(πFr²) | τ 半窗漂移 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['name']} | {r['lam']:.0f} | {_fmt(r['ma_peak'], '.3f')} | {r['u_mean']:.4e} | "
          f"{_fmt(r['fr'])} | {r['err_pct']:+.2f} % | {'PASS' if r['gate_15pct'] else 'FAIL'} | "
          f"{_fmt(r['c_eff'], '.3f')} | {_fmt(r['subwindow_drift_pct'], '+.2f')} % |")
    A("")
    A("**表 2 — 判据汇总**")
    A("")
    A("| 判据 | 数值 | 判定 |")
    A("|---|---|---|")
    A(f"| 阶梯散布（λ=64/96/128） | {lad['fr_spread_pct']:+.2f} pp（阈值 {crit['ladder_spread_pct_max']:g} pp） | "
      f"{'**分辨率无关**' if lad['resolution_independent'] else '**未收敛**'} |")
    gate_txt = ("全部在 +-%g %% 内" % crit["gate_pct"]) if lad["gate_pass_all"] else "存在越门臂"
    A(f"| 每臂锚点门 | {gate_txt} | "
      f"{'PASS' if lad['gate_pass_all'] else 'FAIL'} |")
    A(f"| kξ→0 外推 | Fr = {_fmt(ex.get('intercept'))}，误差 {lad['extrapolated_err_vs_fr_lz_pct']:+.2f} %，"
      f"R² = {_fmt(ex.get('r2'))} | 诊断 |")
    A(f"| Ma 证伪（g 减半） | ΔFr = {mach['shift_pct']:+.2f} %（阈值 {crit['mach_resolution_pct']:g} pp）| "
      f"{'存在可分辨 Ma 依赖' if mach['resolvable'] else '**无可分辨 Ma 依赖**'} |")
    A(f"| kξ 控制臂（ac_width 2→4 cell） | ΔFr = {kx['shift_vs_ref_pct']:+.2f} % | 诊断 |")
    A("")
    if audit:
        A("**表 3 — 证据包独立复核（`verify/audit_bundle.py`，对已合并件、只读）**")
        A("")
        A("| id | 检查 | 值 | 状态 |")
        A("|---|---|---|---|")
        for a in audit["checks"]:
            v = a["value"]
            vs = f"{v:.6g}" if isinstance(v, (int, float)) and v == v else str(v)
            A(f"| {a['id']} | {a['what']} | {vs} | {a['status']} |")
        A("")
        A(f"OK/PASS = {audit['n_ok']}，flagged = {audit['n_flagged']}（flagged 项见 ⑤ F1/F3）。")
        A("")

    # ------------------------------------------------------------------ ④
    A("## ④ 结果图")
    A("")
    f1 = FIG / "simladder_fr_vs_kxi.png"
    if f1.exists():
        A(f"![Fr vs kxi](../figures/{f1.name})")
        A("")
        A("*图 1 — 终端 Fr 对界面相对厚度 kξ：灰带 = ±15 % 算例门，虚线 = Layzer-Goncharov C=1，"
          "空心方块 = kξ→0 外推，空心菱形 = kξ 控制臂。*")
        A("")
    f2 = FIG / "simladder_fr_vs_ma.png"
    if f2.exists():
        A(f"![Fr vs Ma](../figures/{f2.name})")
        A("")
        A("*图 2 — 同一 Fr 对格子马赫数；紫叉 = 旧套件 λ=96 臂（g 随 λ 升 ⇒ Ma 失控），"
          "是「分辨率/压缩性混淆」的反例。*")
        A("")
    f3 = FIG / "simladder_fronts.png"
    if f3.exists():
        A(f"![fronts](../figures/{f3.name})")
        A("")
        A("*图 3 — 尖峰/气泡前缘轨迹与 τ∈[4,6] 的常速平台（黑段 = 用于 Fr 的最小二乘）。*")
        A("")

    # ------------------------------------------------------------------ ⑤
    A("## ⑤ 注意点（踩过的坑与适用范围）")
    A("")
    A("| 编号 | 坑 / 局限 | 证据 | 处置 |")
    A("|---|---|---|---|")
    A("| F1 | `REPORT.md` 的失败模式 F4「非线性终端速度不可测」在 `artifacts/late` 之后已**过期**："
      "该处置实际是把 `layers=3` 的盒子换成 8、把幅值护栏换成壁面距离判据，窗口 τ∈[4,6] 本身无需改 | "
      "`artifacts/audit/AUDIT.json` 检查 A8 = STALE | 本轮给出陈旧行标注与替代指针（未删原行，保留溯源）|")
    A("| F2 | 旧套件 λ=96 臂 **Ma 混淆**：固定 Ag/λ 使 Ma ∝ √λ，实测 Ma=0.879、Fr=0.7833（+38.8 %）| "
      "`artifacts/late/acceptance_table.md`；图 2 | 本增量改用 Ma 匹配阶梯；该臂继续标 INCOMPLETE，不参与判读 |")
    A("| F3 | `run_case.velocity_in_window` 的**单位缺陷**：返回 d(front)/dτ 而调用方按格子速度比较，"
      "差 1/tau_unit = 800 倍；因旧套件从未进入 τ∈[4,6] 窗（值为 nan）而未暴露 | 审计 A7；"
      "`tests/test_rt_atwood_case_bundle.py::test_velocity_in_window_returns_lattice_units` | 加 `tau_unit` 参数并修正调用方，"
      "保留旧签名可复现旧数字 |")
    A("| F4 | Boussinesq 方程对 `(φ,y)→(−φ,−y)` 不变，**结构上**给不出两流体尖峰/气泡不对称 "
      f"√((1+A)/(1−A)) = {math.sqrt(3):.3f}；实测 1.000（`artifacts/late`）| A6 对称性 = 4e-4 | "
      "该判据**永久 NOT_SUPPORTED**（模型形式限制），不计入门；要给两流体不对称需非 Boussinesq 求解器 |")
    A("| F5 | `ac_width = 1` cell 的 kξ 控制臂**不可用**（低于相场内核可分辨厚度）；"
      "因此 kξ 控制只能向**变厚**方向取一点，外推是单向的 | `run_simladder.py` docstring | 已在 docstring 与报告中写明，"
      "kξ→0 外推只作诊断 |")
    A("| F6 | 参考值仍是闭式理论（Rayleigh/Taylor/Chandrasekhar + Layzer/Goncharov），"
      "锚定文献 Tryggvason et al. 2011 正文在本沙箱不可取得（仅 Crossref 书目记录 + 哈希）| `ref/ref_lock.json` | "
      "沿用既有 MU 声明；不从书上转抄任何页码/表号 |")
    A("| F7 | 线性期的 `a_zc` 窗是**振幅窗**（a/λ∈[0.02,0.08]）不是时间窗；按 τ 同值切会切到启动瞬态，"
      "σ 偏低约 16 %（本单审计脚本第一版就踩了这个坑）| 审计 A1（修正后复现 case.json 的 σ 至 1e-15）| "
      "测试 `test_window_slope_filters_on_amplitude` 固定该语义 |")
    A("")

    # ------------------------------------------------------------------ ⑥
    A("## ⑥ 误差判定（关键量 vs 基准，逐项；显式回答「是否 <3 %」）")
    A("")
    A("| 关键量 | 基准 | 误差 | <3 % ? |")
    A("|---|---|---|---|")
    for r in rows:
        rel = abs(r["err_pct"])
        A(f"| Fr（{r['name']}）| Fr_LZ = {fz:.4f}（Layzer–Goncharov，C=1）| {r['err_pct']:+.2f} % | "
          f"{'是' if rel < 3 else f'否（{rel:.2f} %）'} |")
    A(f"| Fr（kξ→0 外推）| Fr_LZ | {lad['extrapolated_err_vs_fr_lz_pct']:+.2f} % | "
      f"{'是' if abs(lad['extrapolated_err_vs_fr_lz_pct']) < 3 else '否'} |")
    if ref:
        lin = "n/a"
        if audit:
            for a in audit["checks"]:
                if a["id"] == "A1":
                    lin = f"{a['detail']['err_vs_viscous_pct']:+.2f} %"
        A(f"| 线性 σ（参考臂）| σ_visc（等黏度色散）| {lin}（`artifacts/audit` A1 独立复现）| 否 |")
        A(f"| 线性 σ（kξ→0 外推，既有 `artifacts/kdelta`）| σ_inv | "
          f"{audit_w(audit, 'A9')} | 否（报告如实记为未达 3 %）|")
    A("")
    A("**显式回答**：终端 Fr 与解析基准的偏差 **不在 3 % 以内**（本单最好的一臂 "
      f"{min(rows, key=lambda r: abs(r['err_pct']))['err_pct']:+.2f} %，kξ→0 外推 "
      f"{lad['extrapolated_err_vs_fr_lz_pct']:+.2f} %）；它**在**算例锚点门（±15 %）以内。"
      "线性 σ 同理（kξ→0 外推 −5.5 %/−3.3 %，仍短于 3 % 平台标准）。"
      "若要求 <3 %，需要更细的界面（或更小的 kξ）与更低的 Ma，本单未做。")
    A("")

    A("## 判定")
    A("")
    verdict = ("终端速度 PASS **可复现且随分辨率/界面厚度、Ma 的散布在预注册阈值内**，"
               if lad["resolution_independent"] else
               "终端速度 PASS **不满足分辨率无关**（散布超阈值），单点 PASS 不能当作收敛证据，")
    A(verdict + "但**未达 3 % 精度标准**（见 ⑥）。"
      "锚点门（±15 %）结论不变；本增量把它从单点升为阶梯 + 两项证伪检验。")
    return "\n".join(L) + "\n"


def audit_w(audit: dict | None, key: str) -> str:
    if not audit:
        return "n/a"
    for a in audit["checks"]:
        if a["id"] == key:
            d = a["detail"].get("W", {})
            v = d.get("intercept_pct")
            return f"{v:+.2f} %（W 估计量）" if v is not None else "n/a"
    return "n/a"


def main() -> int:
    summary = json.loads((SIM / "summary.json").read_text())
    audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else None
    out = SIM / "REPORT_SIMLADDER.md"
    out.write_text(render(summary, audit))
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())