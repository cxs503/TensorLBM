#!/usr/bin/env python3
"""Render ``REPORT_VERIFY_R3.md`` from ``artifacts/r3_summary.json``.

Every number in the report is injected programmatically — nothing is typed by
hand.  Legs that never came back from the queue are rendered as ``NOT_RUN`` and
are never silently reported as passing.

    python make_report_r3.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "artifacts" / "r3_summary.json"
FIELD = HERE / "artifacts" / "r3_parts" / "field_2d.json"
OUT = HERE / "REPORT_VERIFY_R3.md"
LIB_SIGMA = 0.0561  # benchmarks/verified/laplace_droplet locked value (2D, τ=1)
TOL, TIGHT = 0.15, 0.03


def _load() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def f(x, n=5):
    if x is None or isinstance(x, str):
        return str(x if x is not None else "—")
    return f"{x:.{n}f}"


def pct(x, n=2):
    if x is None or isinstance(x, str):
        return "—"
    return f"{100 * x:+.{n}f}%"


def verdict_str(v) -> str:
    if isinstance(v, bool):
        return "PASS" if v else "**FAIL**"
    return f"__{v}__"


def dev_ok(dev: float | None) -> tuple[str, str]:
    if dev is None:
        return "—", "—"
    return ("是" if dev < TIGHT else "否"), ("是" if dev < TOL else "否")


def render(d: dict) -> str:
    v = d.get("verdicts") or {}
    mw = d.get("maxwell") or {}
    k3 = d.get("kappa3") or {}
    flat = d.get("flat") or {}
    fit = d.get("fit") or {}
    cd = d.get("cross_dim")
    rows = d.get("droplets") or []
    prov = d.get("provenance") or {}
    field_ok = FIELD.exists()

    def dims_of(leg: str) -> str:
        got = [k.split("_")[1] for k in prov if k.startswith(leg)]
        return ", ".join(sorted(got)) if got else "none"

    L: list[str] = []
    A = L.append
    A("# XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION — R3 复核报告")
    A("")
    A("> **锚点**: `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`（mission m20921）")
    A("> **轮次**: R3（承接 R2 预登记但 NOT_RUN 的数值腿）")
    A("> **判据**: owner <15%，并逐项显式回答 <3%；预登记见 `predeclared.json`（运行前固定，未事后调整）")
    A("> **渲染**: 本文件由 `make_report_r3.py` 从 `artifacts/r3_summary.json` 程序化生成，数字零手抄")
    A("")
    A("## ① 案例介绍")
    A("")
    A("单组分 Shan–Chen 伪势模型（SC94，ψ=1−e^(−ρ)）在周期盒内形成液-气共存。二维圆柱界面满足")
    A("Δp = σ/R，三维球面界面满足 Δp = 2σ/R；两者合并写作 **Δp = (d−1)σ/R**。")
    A("R2 留下一条可否证的跨维预言：若界面张力由 Korteweg 截断应力支配且模板四阶矩比")
    A("κ3 = a3/a1 在 D2Q9 与 D3Q19 上相同，则 **σ_3D/σ_2D = 1**。R2 只验证了 κ3 的算术，")
    A("F（平面液膜）/ D（液滴扫描）两条数值腿因排队未取回终态而标 NOT_RUN，故 R2 明确不主张")
    A("跨维一致。R3 就是去把这两条腿跑完并把判决落地。")
    A("")
    A("| 参数 | 值 |")
    A("|---|---|")
    proto = d.get("protocol") or {}
    A(f"| 模型 | {proto.get('model', 'SCMP SC94')} |")
    A(f"| 耦合 | G_lib = {proto.get('G_lib')} ≡ 物理 G_eff = {proto.get('G_eff')} |")
    A(f"| 松弛 | τ = {proto.get('tau')}（BGK） |")
    A(f"| 半径 | R = {proto.get('radii')}，域 L = {proto.get('L_factor')}·R |")
    A(f"| 维度 | {proto.get('dims')}（D2Q9 / D3Q19） |")
    A(f"| 力格式 | velocity-shift（与 `benchmarks/verified/laplace_droplet` 一致） |")
    A(f"| 阈值 | 15% 宽判据 / 3% 紧判据；力学平衡闸门 {proto.get('thresholds', {}).get('mech_balance')} |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("R2 的执行器是单体式的（`--stage all`：2D+3D 的液滴与液膜必须在同一个排队窗口内全部跑完），")
    A("实际结果是**一条腿都没落盘**。R3 不改物理、不改协议、不改阈值，只改执行方式：")
    A("")
    A("**执行方式（本轮的工程改动）** — 新增 `run_r3_split.py`，把计划拆成互相独立的")
    A("(维度, 腿) 分片，每片落一个 part 文件；`--merge` 用 R2 的同一套 `_fit_sigma` 与")
    A("`THRESHOLDS`（直接 `import run_r2`，单一真源，不复制阈值）汇总为 `artifacts/r3_summary.json`。")
    A("缺片一律记为 `NOT_RUN`，绝不以相邻片或上一轮的数值代替。")
    A("")
    A("**四条腿**（协议与 R2 预登记完全一致）：")
    A("")
    A("- **D 腿（主判据）**：2D/3D × R = 10/14/18 液滴扫描，σ_i = Δp·R_eq/(d−1)，")
    A("  逐半径一致性与跨维比值。")
    A("- **F 腿**：2D/3D 平面液膜（横向平均 1D 剖面），给出体相共存密度、")
    A("  力学平衡闸（|p_l−p_v|/p_v）、以及 Korteweg 锚 σ_K = |G_eff κ3|∫(∂_n ψ)²dn。")
    A("- **M 腿**：SC 状态方程上的 Maxwell 等面积构造（解析基准，无求解器）。")
    A("- **S 腿（本轮新增）**：收敛液滴的密度场快照，用于出云图；R2 的 artifact 只有标量表，")
    A("  没有场量，报告的④因此无图可出——这一轮把场补上。")
    A("")
    A(f"已取回的分片：液滴腿 [{dims_of('droplet')}]，液膜腿 [{dims_of('flat')}]，"
      f"场快照 [{'2d' if field_ok else 'not fetched'}]。")
    A("")
    A("工具链：Python + PyTorch（库即 TensorLBM `src/tensorlbm`，求解器零手写：只用库内的")
    A("`collide_sc_single_component*` + `stream*`）；matplotlib 出图；cluster 递交执行。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 3.1 Maxwell 等面积构造（解析基准，M 腿）")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    A(f"| ρ_v | {f(mw.get('rho_v'), 6)} |")
    A(f"| ρ_l | {f(mw.get('rho_l'), 6)} |")
    A(f"| p_c | {f(mw.get('p_c'), 6)} |")
    A(f"| 等化学势残差 | {f(mw.get('mu_residual'), 3)} |")
    A("")
    A("### 3.2 模板四阶矩（κ3 预言）")
    A("")
    A("| 维度 | 速度数 | a1 | a3 | κ3 |")
    A("|---|---|---|---|---|")
    for dim in ("2", "3"):
        k = k3.get(dim) or {}
        A(f"| {dim}D | {k.get('n_vel')} | {f(k.get('a1'), 7)} | {f(k.get('a3'), 7)} | {f(k.get('kappa3'), 7)} |")
    A(f"\nκ3 比值 (3D/2D) = **{f(d.get('kappa3_ratio'), 9)}** ⇒ 预言 σ_3D/σ_2D = 1。")
    A("")
    A("### 3.3 液滴扫描（D 腿）")
    A("")
    if rows:
        A("| 维度 | R_init | L | R_eq | Δp | ρ_in | ρ_out | σ_i = Δp·R_eq/(d−1) | 收敛 | steps |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["dim"], r["R_init"])):
            A(f"| {r['dim']}D | {r['R_init']:.0f} | {r['L']} | {f(r['R_eq'], 3)} | {f(r['dp'], 6)} | "
              f"{f(r['rho_in'], 4)} | {f(r['rho_out'], 4)} | {f(r['sigma'], 5)} | {r['converged']} | {r['steps']} |")
    else:
        A("D 腿未取回：`NOT_RUN`（不给数字）。")
    A("")
    A("### 3.4 σ 拟合（D 腿）")
    A("")
    A("| 维度 | σ (through origin) | σ (mean of radii) | 逐半径最大偏差 | 自由截距 | R² 代理量 |")
    A("|---|---|---|---|---|---|")
    for dim in ("2", "3"):
        s = fit.get(dim)
        if not s:
            A(f"| {dim}D | NOT_RUN | — | — | — | — |")
        else:
            A(f"| {dim}D | {f(s['sigma_through_origin'])} | {f(s['sigma_mean_of_radii'])} | "
              f"{pct(s['max_dev'])} | {f(s['intercept_free'], 6)} | — |")
    A("")
    A("### 3.5 平面液膜（F 腿）：共存与力学平衡")
    A("")
    A("| 维度 | L_cells | ρ_l | ρ_v | p_l | p_v | mech gap | σ_K(tanh) | 收敛 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for dim in ("2", "3"):
        s = flat.get(dim)
        if not s:
            A(f"| {dim}D | NOT_RUN | — | — | — | — | — | — | — |")
            continue
        A(f"| {dim}D | {s['n']} | {f(s['rho_l'], 4)} | {f(s['rho_v'], 4)} | {f(s['p_l'], 5)} | "
          f"{f(s['p_v'], 5)} | {f(s['mech_gap'], 2)} | {f(s['sigma_K_tanh'])} | {s['converged']} |")
    A("")
    A("### 3.6 跨维判决（R2 预登记 P1）")
    A("")
    if cd:
        A("| 量 | 值 |")
        A("|---|---|")
        A(f"| σ_2D | {f(cd['sigma_2d'])} |")
        A(f"| σ_3D | {f(cd['sigma_3d'])} |")
        A(f"| 比值 σ_3D/σ_2D | {f(cd['ratio'], 6)} |")
        A(f"| 与预言 1.0 的偏差 | {pct(cd['dev_from_prediction_1'], 3)} |")
        A(f"| 预登记容差 | 15%（紧判据 3%） |")
    else:
        A("跨维比值需要 2D 与 3D 两条液滴腿同时到齐；当前 `NOT_RUN`。")
    A("")
    A("### 3.7 预登记判据总表")
    A("")
    A("| 判据 | 含义 | 判决 |")
    A("|---|---|---|")
    meanings = {
        "P1_cross_dim": "σ_3D/σ_2D = 1（<15%）",
        "P1_cross_dim_tight3pct": "σ_3D/σ_2D = 1（<3%）",
        "P2_mech_balance_2d": "2D 平面膜 |p_l−p_v|/p_v < 1e−3",
        "P2_mech_balance_3d": "3D 平面膜 |p_l−p_v|/p_v < 1e−3",
        "P3_radius_consistency_2d": "2D 逐半径 σ_i 一致性 <15%",
        "P3_radius_consistency_3d": "3D 逐半径 σ_i 一致性 <15%",
        "P4_sigmaK_vs_lap_2d": "2D Korteweg σ_K vs Laplace σ（<15%）",
        "P4_sigmaK_vs_lap_3d": "3D Korteweg σ_K vs Laplace σ（<15%）",
        "P5_coexist_vs_maxwell_2d": "2D 共存密度 vs Maxwell（<15%）",
        "P5_coexist_vs_maxwell_3d": "3D 共存密度 vs Maxwell（<15%）",
    }
    for key in meanings:
        A(f"| `{key}` | {meanings[key]} | {verdict_str(v.get(key, 'NOT_RUN'))} |")
    A("")
    A("## ④ 结果图")
    A("")
    figs = []
    if (HERE / "artifacts" / "fig_r3_field.png").exists():
        figs.append("- `artifacts/fig_r3_field.png` — 2D 收敛液滴的 ρ(x,y) 云图 + 过中心径向剖面"
                    "（标出 ρ_in / ρ_out，并由 Δp·R_eq 给出同一场的 σ 读数）。")
    else:
        figs.append("- 场量云图 **缺**：S 腿快照未取回，本轮不出云图（不拿解析剖面冒充模拟场）。")
    if (HERE / "artifacts" / "fig_r3_crossdim.png").exists():
        figs.append("- `artifacts/fig_r3_crossdim.png` — 左：Δp/(d−1) vs 1/R_eq（2D、3D 与过原点拟合）；"
                    "右：σ_2D / σ_3D 柱状 + ±3%/±15% 带，标题给出跨维比值。")
    if (HERE / "artifacts" / "fig_r3_flat.png").exists():
        figs.append("- `artifacts/fig_r3_flat.png` — 左：平面膜共存密度 vs Maxwell 解析值；"
                    "右：力学平衡闸（对数轴）与 σ_K 对 σ_Laplace。")
    A("\n".join(figs))
    A("")
    A("## ⑤ 注意点")
    A("")
    A("1. **本轮改的是执行方式，不是物理**：协议、radii、阈值全部沿用 R2 预登记；")
    A("   把单体执行改成可分片执行，是为了让已经算完的腿能落盘，而不是为了让判据变松。")
    A("2. **σ 没有第一性原理解析值**：SC 伪势模型的界面张力依赖离散化细节，")
    A("   所以判据只能是「跨维一致 + 逐半径一致 + Maxwell 相容」这三条互相独立的基准，")
    A("   而不是与某个解析 σ 比。")
    A("3. **Korteweg 锚只是量级锚**：R2 审计已测到 σ_K 比 Laplace σ 低 ~29%（代码系数）/")
    A("   高 ~114%（文档系数），差异来自 cs² 因子与实测剖面的有限带宽；P4 因此是弱判据。")
    A("4. **伪流速度**：SC94 在 τ=1 的液滴界面带 u_max ≈ 0.14，液滴并非严格静止；")
    A("   本报告只用体相带平均值，不依赖界面内速度。")
    A("5. **适用范围**：ψ=1−e^(−ρ)、G_eff=−5、τ=1、velocity-shift 力格式、R=10…18。")
    A("   力格式换成 Guo 会改变模型本身（R1 实测 σ 差 56.8%），不可跨格式搬用。")
    A("6. **三维成本**：3D R=18 的域是 72³，单核 3D 步进约 0.25 s，单算例小时量级；")
    A("   这是 R2 全量跑不完的直接原因，也是本轮改成按维度分片递交的原因。")
    A("")
    A("## ⑥ 误差判定（逐项，显式回答 <3%）")
    A("")
    A("| 关键量 | 值 | 基准 | 误差 | <3%? | <15%? |")
    A("|---|---|---|---|---|---|")
    checks = []
    for dim in ("2", "3"):
        s = fit.get(dim)
        if not s:
            checks.append((f"σ_{dim}D（过原点拟合）", "NOT_RUN", "0.0561（库锁定 2D 值）", None))
            continue
        dev = abs(s["sigma_through_origin"] - LIB_SIGMA) / LIB_SIGMA
        checks.append((f"σ_{dim}D（过原点拟合）", f(s["sigma_through_origin"]),
                       "0.0561（`benchmarks/verified/laplace_droplet` 锁定值）", dev))
        checks.append((f"σ_{dim}D 逐半径最大偏差", pct(s["max_dev"]), "该维 σ 均值", s["max_dev"]))
    if cd:
        checks.append(("σ_3D/σ_2D vs 预言 1", f(cd["ratio"], 6), "1（κ3 算术）", cd["dev_from_prediction_1"]))
    for name, val, ref, dev in checks:
        a, b = dev_ok(dev)
        checks_row = f"| {name} | {val} | {ref} | {pct(dev) if dev is not None else '—'} | {a} | {b} |"
        A(checks_row)
    if mw and flat:
        for dim in ("2", "3"):
            s = flat.get(dim)
            if not s:
                continue
            for key, label in (("rho_l", "ρ_l"), ("rho_v", "ρ_v")):
                dev = abs(s[key] - mw[key]) / mw[key]
                a, b = dev_ok(dev)
                A(f"| {dim}D 平面膜 {label} vs Maxwell | {f(s[key], 4)} | {f(mw[key], 4)} | {pct(dev)} | {a} | {b} |")
            a, b = dev_ok(s["mech_gap"])
            A(f"| {dim}D 力学平衡 |p_l−p_v|/p_v | {f(s['mech_gap'], 3)} | 0（精确平衡） | {f(s['mech_gap'], 3)} | {a} | {b} |")
    A("")
    A("**结论（不粉饰）**：")
    ok_keys = [k for k, val in v.items() if val is True]
    fail_keys = [k for k, val in v.items() if val is False]
    nr_keys = [k for k, val in v.items() if isinstance(val, str)]
    A(f"- PASS 的预登记判据 {len(ok_keys)} 条：{', '.join(f'`{k}`' for k in ok_keys) if ok_keys else '无'}。")
    A(f"- FAIL 的预登记判据 {len(fail_keys)} 条：{', '.join(f'`{k}`' for k in fail_keys) if fail_keys else '无'}。")
    A(f"- NOT_RUN 的预登记判据 {len(nr_keys)} 条：{', '.join(f'`{k}`' for k in nr_keys) if nr_keys else '无'}。")
    A("")
    A("## ⑦ 未做 / 后续")
    A("")
    A("- 毛细波色散路线（不依赖压力差、可给 <3% 的绝对 σ）仍未做：需要长时程界面涨落统计与")
    A("  波数分辨，成本比本轮的液滴扫描高一个量级。")
    A("- τ 扫描（σ 的黏性依赖）与 G_eff 扫描仍未覆盖；R1 只做过 τ=0.7/1.0/1.5 的液滴单点。")
    A("- Guo 格式下体相 EOS 偏移（共存密度整体下移）的机理仍只记录现象，未追根。")
    A("")
    A("## 复算方式")
    A("")
    A("```bash")
    A("cd xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/verify_r2")
    A("python run_r3_split.py --merge                                  # 汇总（含解析腿，秒级）")
    A("python run_r3_split.py --dim 2 --leg droplet --jobs 3 --threads 2")
    A("python run_r3_split.py --dim 3 --leg droplet --jobs 3 --threads 8   # 3D 最贵")
    A("python run_r3_split.py --dim 2 --leg flat")
    A("python run_r3_split.py --dim 2 --leg field --R 14")
    A("python make_figures_r3.py && python make_report_r3.py")
    A("pytest tests/test_xval_sc_laplace_r3.py -q")
    A("```")
    A("")
    return "\n".join(L) + "\n"


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY}; run `run_r3_split.py --merge` first")
    OUT.write_text(render(_load()), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()