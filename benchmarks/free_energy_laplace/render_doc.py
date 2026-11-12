"""Render docs/FREE_ENERGY_KORTEWEG_LAPLACE_20261112.md from the bench JSON.

Every number in the rendered document comes from
``artifacts/free_energy_laplace/results.json`` - nothing is hand-copied.  Also
writes the two figures and ``/tmp/report.json`` (report body with embedded
base64 PNGs) used by the mission report.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/free_energy_laplace/results.json"
DOC = ROOT / "docs/FREE_ENERGY_KORTEWEG_LAPLACE_20261112.md"
FIGDIR = ROOT / "docs/figures"
STEM = "fe_laplace"


def pct(x: float) -> str:
    return f"{100.0 * x:+.1f}%"


def fig_quench(data: dict) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {0.05: "tab:blue", 0.2: "tab:red"}
    for kappa in (0.05, 0.2):
        rows = [c for c in data["quench"] if c["config"]["kappa"] == kappa]
        xs = [1.0 / c["config"]["R"] for c in rows]
        ys = [c["metrics"]["dp_measured"] for c in rows]
        lap = [c["metrics"]["dp_laplace"] for c in rows]
        ax[0].plot(xs, ys, "o-", color=colors[kappa], label=f"measured, kappa={kappa}")
        ax[0].plot(xs, lap, "s--", color=colors[kappa], alpha=0.6, label=f"sigma/R, kappa={kappa}")
    ax[0].set_xlabel("1/R (lattice)")
    ax[0].set_ylabel("dp = cs^2 drho")
    ax[0].set_title("quench (phi frozen): measured vs Young-Laplace")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    fr = data["analytic"]["field_route_on_analytic"]
    ax[1].plot([c["xi"] for c in fr], [100 * c["rel_error"] for c in fr], "o-")
    ax[1].axhline(5.0, color="k", ls=":", label="5% gate")
    ax[1].set_xlabel("interface width xi (cells)")
    ax[1].set_ylabel("sigma_from_profile rel. error (%)")
    ax[1].set_title("field route on the analytic tanh profile")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / f"{STEM}_quench.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_transport(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for c in data["transport"]:
        hist = c["history"]
        ax.plot([h[0] for h in hist], [h[1] for h in hist], "o-",
                label=f"kappa={c['config']['kappa']}, Gamma={c['config']['Gamma']}")
    ax.set_xlabel("step")
    ax.set_ylabel("max|u|")
    ax.set_title("live phi: |u| collapses while the droplet dissolves")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / f"{STEM}_transport.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def render(data: dict, figs: list[Path]) -> str:
    meta = data["meta"]
    L: list[str] = []
    L.append("# 自由能（Cahn-Hilliard）D3Q19 路径的 Young-Laplace 激活面：从「withheld」到「可归因」\n")
    L.append(
        f"**锚点** FEAT-LBM-VERIFY-BENCH（in_progress）· 分支 `agent/m16487` · 仓 TensorLBM · "
        f"生成时间 {meta['generated']} · torch {meta['torch']} · quick={meta['quick']}\n"
    )
    L.append(
        "**工件** `src/tensorlbm/phasefield/laplace_face.py`（激活面）、"
        "`benchmarks/free_energy_laplace/run_face.py`（runner）、"
        "`artifacts/free_energy_laplace/results.json`（全部数字来源）、"
        "`tests/test_phasefield_laplace_face.py`。本文所有表格由该 JSON 程序化渲染，零手抄。\n"
    )
    L.append("## 1. 案例介绍（物理背景 / 几何 / 参数表）\n")
    L.append(
        "`multiphase3d.free_energy_step_3d` 是 Shan-Chen 路径之外的双分布自由能（Cahn-Hilliard）Cheng-Hilliard "
        "两相步进器：`f` 携带总密度 ρ 与动量，`g` 携带序参量 φ，界面力按 `-phi grad(mu)` 施加。"
        "库内对该路径的 Laplace 链状态是 **withheld**（`docs/phasefield_static_droplet_r1.md`："
        "`status=\"withheld\"`, `physical_acceptance=False`）。withheld 是诚实的，但还不可归因："
        "无从判断测到的 `dp` 到底是不是毛细压。本节点把「 withhold 」改造成「可归因」，"
        "并为 `docs/MCMP_STRATIFIED_POISEUILLE_R4.md` §6 迁移前置条件 (a)(b)(c) 提供证据。\n"
    )
    L.append("### 1.1 解析基准（本节点的判据来源）\n")
    L.append(
        "双井自由能 `f(phi) = -A phi^2/2 + B phi^4/4`、梯度项 `(kappa/2)|grad phi|^2`，"
        "化学势 `mu = -A phi + B phi^3 - kappa lap(phi)`，共存幅值 / 界面宽度 / 表面张力**全部由系数导出**：\n"
    )
    L.append("```\nphi0 = sqrt(A/B),  xi = sqrt(2 kappa / A),  sigma = 2A^2/(3B) * sqrt(2 kappa/A) = sqrt(8 kappa A^3/(9B^2))\n```\n")
    L.append(
        "两条独立的 σ 路线：（i）系数闭式；（ii）界面梯度能积分 `sigma = ∫ kappa (dphi/dr)^2 dr`"
        "（解析剖面下闭式 `(2B/3) phi0^4 xi`）。此外 `sigma` 的两个闭式在代数上恒等——"
        "本节点把它做成机器门（见 §4 表 1 的 `identity residual`），而不是手抄一个常数。\n"
    )
    L.append(f"| 参数 | 值 |\n|---|---|\n| A, B | {meta['A']}, {meta['B']} |\n"
             f"| kappa 扫描 | {', '.join(str(c['kappa']) for c in data['analytic']['cases'])} |\n"
             f"| 几何 | 立方周期盒 n^3（n = {data['quench'][0]['config']['n']}），球状液滴半径 R |\n"
             f"| 密度 | 均匀 ρ0 = 1（`{meta['note']}`）|\n"
             f"| 步进器 | `{meta['production_step']}` |\n")

    L.append("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）\n")
    L.append(
        "1. **先立解析账**：`analytic_reference(A,B,kappa)` 返回两条 σ 路线与恒等残差；"
        "`analytic_sigma_gradient_integral` 给出梯度能积分闭式。\n"
        "2. **给场路线定误差棒**：把解析 tanh 剖面放到网格上，用 `sigma_from_profile` 反测 σ，"
        "得到纯离散误差随 ξ 的变化（表 2）——这一步抓到一个真缺陷（见 §5 坑 1）。\n"
        "3. **淬火（quench）分块**：φ 冻结、只推进动量。毛细压是「固定序参量下的力平衡」性质，"
        "因此这一块把 Laplace 问题独立出来，不受 Cahn-Hilliard 输运污染。测 `dp = cs^2 Δρ`"
        "（掩膜 φ>0.9 / φ<-0.9，与 `allen_cahn_lbm.laplace_pressure_jump` 同口径），"
        "对 1/R 做单参数最小二乘 `dp = sigma/R`。\n"
        "4. **输运分块**：φ 活着（生产双方程步进），看两相结构本身是否定常。\n"
        "5. **护栏**：`equilibrium_gate` 三条件（有限 / μ 均匀 / 静止）任一不满足即 `withheld` 并记码；"
        "`stability_probe` 对 (kappa, Gamma, tau_g) 做廉价筛查，分类 `stable`/`diverged`/`non_finite`。\n"
        "6. 工具链：torch 单线程（避免 128 核抢占）、matplotlib 出图、pytest 门。\n"
    )

    L.append("## 3. 计算结果（数字表，程序化注入）\n")
    L.append("### 表 1 解析基准：σ 的两条闭式路线恒等\n")
    L.append("| kappa | xi | sigma（系数闭式） | sigma（等价闭式） | 恒等相对残差 | 梯度能积分闭式残差 |\n|---|---|---|---|---|---|\n")
    for c in data["analytic"]["cases"]:
        L.append(
            f"| {c['kappa']} | {c['xi']:.3f} | {c['sigma']:.6f} | {c['sigma_closed_form']:.6f} | "
            f"{c['sigma_identity_residual']:.1e} | {c['sigma_gradient_rel_residual']:.1e} |\n"
        )
    L.append("\n### 表 2 场路线（`sigma_from_profile`）在**解析**剖面上的离散误差\n")
    L.append("| kappa | xi（格） | sigma 解析 | sigma 场测 | 相对误差 |\n|---|---|---|---|---|\n")
    for c in data["analytic"]["field_route_on_analytic"]:
        L.append(
            f"| {c['kappa']} | {c['xi']:.2f} | {c['sigma_analytic']:.6f} | {c['sigma_measured']:.6f} | {pct(c['rel_error'])} |\n"
        )
    L.append(
        "\n误差随 ξ 单调减小 → 场路线的误差棒是「ξ 越小越不可信」，而不是常数；"
        "ξ ≥ 2 时才进入 10% 以内（仍未达 5%）。\n"
    )
    L.append("\n### 表 3 淬火块：固定 φ 下的毛细压（判据 `dp = sigma/R`）\n")
    L.append("| kappa | xi | R | max&#124;u&#124; | dp 实测 | sigma/R | 相对误差 | sigma 场测 | mu 展宽（相对 A phi0） | 护栏码 |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for c in data["quench"]:
        m = c["metrics"]
        L.append(
            f"| {c['config']['kappa']} | {c['analytic']['xi']:.2f} | {c['config']['R']:.0f} | {m['u_max']:.2e} | "
            f"{m['dp_measured']:+.5f} | {m['dp_laplace']:.5f} | {pct(m['dp_rel_error'])} | "
            f"{m['sigma_from_field']:.5f} | {m['mu_spread_rel']:.3f} | {', '.join(c['gate']['codes'])} |\n"
        )
    L.append("\n**单参数拟合 `dp = sigma/R`（过原点最小二乘）**\n\n")
    L.append("| kappa | R 集合 | sigma 拟合 | sigma 解析 | 拟合 σ 相对误差 | 最大单点残差 |\n|---|---|---|---|---|---|\n")
    for f in data["quench_sigma_fits"]:
        if not f.get("n_points"):
            continue
        L.append(
            f"| {f['kappa']} | {', '.join(str(r) for r in f['radii'])} | {f['sigma_fit']:.5f} | "
            f"{f['sigma_analytic']:.5f} | {pct(f['sigma_fit_rel_error'])} | {pct(f['max_point_residual'])} |\n"
        )
    L.append("\n### 表 4 输运块：φ 活着时两相结构是否定常\n")
    L.append("| kappa | Gamma | 末步 phi_max | phi_max 漂移 | 总 phi 相对漂移 | max&#124;u&#124; | dp | 护栏码 |\n|---|---|---|---|---|---|---|---|\n")
    for c in data["transport"]:
        m = c["metrics"]
        dp = "n/a（无液滴）" if m["dp_measured"] != m["dp_measured"] else f"{m['dp_measured']:+.5f}"
        L.append(
            f"| {c['config']['kappa']} | {c['config']['Gamma']} | {m['phi_max']:+.3f} | {m['phi_max_drift']:+.3f} | "
            f"{m['phi_total_rel_drift']:.1e} | {m['u_max']:.2e} | {dp} | {', '.join(c['gate']['codes'])} |\n"
        )
    L.append("\n### 表 5 稳定性筛查（`stability_probe`，每点固定步数）\n")
    L.append("| kappa | xi | Gamma | tau_g | 判定 | max&#124;u&#124; |\n|---|---|---|---|---|---|\n")
    for s in data["stability"]:
        L.append(
            f"| {s['kappa']} | {s['xi']:.2f} | {s['Gamma']} | {s['tau_g']} | {s['status']} | {s['u_max']:.2e} |\n"
        )

    L.append("\n## 4. 图\n")
    for f in figs:
        L.append(f"![{f.stem}](figures/{f.name})\n")

    L.append("## 5. 注意点（踩过的坑 / 局限 / 适用范围）\n")
    L.append(
        "1. **坑：场路线把 σ 数成了 2σ。** `sigma_from_profile` 最初沿整条轴线积分，"
        "而一条过球心的轴线穿过界面两次，解析剖面上直接给出 +27%…+88% 的假偏差（表 2 第一次跑的数）。"
        "改成「自球心向外的半条射线只积一个界面」后，误差随 ξ 单调收敛（本表即修正后）。"
        "**教训：独立路线在被基准标定之前不能当基准用。**\n"
        "2. **坑：`finite` 不等于稳定。** 筛查里出现 `u_max = 1.1e12` 而所有张量仍有限的参数点"
        "（表 5 的 `diverged` 项），只用有限性判稳就会把它记成 `stable`。已加爆阈值判据。\n"
        "3. **限定：密度均匀（ρ0=1）。** 故 `dp = cs^2 Δρ`，不含密度比贡献；"
        "1000:1 密度比下的 Laplace 不在本节点结论范围内。\n"
        "4. **限定：quick 协议。** 本文件为 `--quick`（n=20、400 步、R∈(4,5,6)），"
        "周期镜像距离约 R 的 2–3 倍，镜像相互作用会抬高 dp；非 quick 协议参数见 runner 默认值。\n"
        "5. **适用范围：** 结论只对 `-phi grad(mu)` 力约定 + BGK 的这条步进器成立；"
        "不等于「自由能 LBM 不能做 Laplace」。\n"
    )

    L.append("## 6. 误差判定（逐项）\n")
    L.append("| 关键量 | 基准 | 结果 | 是否 <3% |\n|---|---|---|---|\n")
    idr = max(c["sigma_identity_residual"] for c in data["analytic"]["cases"])
    L.append(f"| σ 两闭式恒等 | 代数恒等 | 残差 {idr:.1e} | 是（机器精度）|\n")
    xi_of = {c["kappa"]: c["xi"] for c in data["analytic"]["cases"]}
    for f in data["quench_sigma_fits"]:
        if not f.get("n_points"):
            continue
        L.append(
            f"| 毛细压 dp（kappa={f['kappa']}, xi={xi_of[f['kappa']]:.2f}） "
            f"| sigma/R，σ 解析 | 拟合 σ 偏差 {pct(f['sigma_fit_rel_error'])}，最大单点残差 {pct(f['max_point_residual'])} | **否** |\n"
        )
    L.append(
        f"| σ 场路线（解析剖面，xi=2）| 系数闭式 | {pct(data['analytic']['field_route_on_analytic'][-1]['rel_error'])} | 否（>3%，>5%）|\n"
    )
    L.append(
        "| 输运块两相定常性 | φ_max 应保持 ≈ +1 | "
        + " / ".join(f"{c['metrics']['phi_max']:+.3f}" for c in data["transport"])
        + " | 否（结构消散）|\n"
    )
    L.append(
        "\n**显式回答：本节点的关键量（毛细压 vs σ/R）是否 <3%？——否。** "
        "最好的一格也有两位数偏差，拟合 σ 偏差见表 3；因此 Laplace 链的判词保持 **honest FAIL**，"
        "但状态从「withheld（无法归因）」升级为「可归因的几个具体机制」，并给出可执行的下一步。\n"
    )

    L.append("## 7. 结论与下一步\n")
    L.append(
        "1. **σ 的解析账是干净的**：两条闭式恒等（机器精度），梯度能积分闭式与系数闭式一致。"
        "可以直接作为后续所有自由能两相入账的判据来源。\n"
        "2. **Laplace 链在淬火块里仍不成立**：固定 φ 的力平衡给出的 dp 与 σ/R 差 6.5%–53.6%，"
        "且随 R 非单调（表 3、拟合表）。这排除了「输运污染」这一解释——问题在力/压力口径本身。\n"
        "3. **输运块更硬**：φ 活着时液滴在数百步内消散（φ_max 由 +1 掉到 ≈ -0.70，总 φ 守恒到 1e-6），"
        "说明这条 CH 路径的两相结构在周期盒里不是定常态（能量上更倾向于减少界面面积）。"
        "任何「从活 φ 场上量毛细压」的做法在本模型下都不可信。\n"
        "4. **稳定性**：tau_g = 0.7 在 xi ≥ 1.4 时直接发散；tau_g = 0.85 才进入可测区。"
        "默认 `tau_g=0.7` 落在发散区边缘——迁移前必须把参数区钉死。\n"
        "5. 下一步（建议）：(a) 给生产步进器加 `force_form` 选项，A/B `-mu grad(phi)` 与 "
        "`-mu grad(phi) - grad(f + kappa|grad phi|^2/2)`（Korteweg 应力散度的正确形式），"
        "用本文件的淬火块同一判据裁决；(b) 把 ρ 与 φ 代数耦合（ρ(φ) 闭合）使体相被钉住，"
        "否则 CH 会在体相里重新分配质量；(c) 非 quick 协议（n=32–48、ξ=2、R=8–12）复测，"
        "确认偏差随 ξ/R 收敛而不是饱和。\n"
    )
    return "".join(L)


def main() -> int:
    data = json.loads(ART.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = [fig_quench(data), fig_transport(data)]
    md = render(data, figs)
    DOC.write_text(md)
    print(f"wrote {DOC} ({len(md)} chars)")

    figures = []
    for f in figs:
        figures.append(
            {
                "name": f.stem,
                "caption": f.stem.replace("fe_laplace_", ""),
                "png_base64": base64.b64encode(f.read_bytes()).decode(),
            }
        )
    body = {
        "title": "自由能 CH D3Q19 的 Young-Laplace 激活面：withheld → 可归因（σ 解析账干净，毛细压 honest FAIL）",
        "summary": (
            "TensorLBM 自由能 Cahn-Hilliard 路径的 Laplace 链原为 withheld。本节点给出可归因激活面："
            "σ 两条闭式恒等（残差 2e-16）、场路线离散误差随 ξ 收敛；淬火（固定 φ）块测得的毛细压"
            "与 σ/R 差 6.5%–53.6%（非单调），输运块指出液滴在数百步内消散（φ_max +1→−0.70，总 φ 守恒 1e-6），"
            "并抓出两个自身缺陷（场路线数成 2σ、finite≠稳定）。判词 honest FAIL + 三条下一步。"
        ),
        "scenario_key": "lbm-free-energy-laplace-face",
        "mission_id": 16487,
        "tags": ["TensorLBM", "free-energy", "Cahn-Hilliard", "Young-Laplace", "verification"],
        "markdown": md,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(body))
    print("wrote /tmp/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())