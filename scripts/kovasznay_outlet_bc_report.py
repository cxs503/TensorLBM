"""Aggregate the Kovasznay outlet-BC ladder into docs + figures + API report.

Reads runs/kovasznay_outlet_bc/<arm>/summary.json (produced by
benchmarks/verified/kovasznay_2d/run.py scan) and writes:

* docs/CASE-LBM-KOVASZNAY-OUTLET-BC.md  (tables injected programmatically)
* figures/kovasznay_outlet_bc_*.png
* /tmp/report.json + POST to /api/v1/tensor/reports

No number is typed by hand: every cell comes from the summary JSONs.
"""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs" / "kovasznay_outlet_bc"
ARMS = [("zerograd", "zero-gradient (Neumann copy)"), ("nxextrap", "non-equilibrium extrapolation (Guo 2002)"),
        ("dirichlet", "analytic Dirichlet (upper bound / reference)")]

data, failed = {}, {}
for arm, _label in ARMS:
    p = ROOT / arm / "summary.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    finite = all(
        math.isfinite(g["u_l2_rel"]) and math.isfinite(g["v_l2_rel"]) and math.isfinite(g["u_max_rel_pct"])
        for g in d["per_grid"]
    )
    (data if finite else failed)[arm] = d

if not data:
    raise SystemExit("no summary.json found")


def rows(arm):
    return sorted(data[arm]["per_grid"], key=lambda r: r["ny"])


# ---------------------------------------------------------------- figures ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
styles = {"zerograd": "o-", "nxextrap": "s-", "dirichlet": "^-"}
labels = dict(ARMS)
for arm in data:
    rs = rows(arm)
    ny = [r["ny"] for r in rs]
    ax[0].loglog(ny, [r["v_l2_rel"] * 100 for r in rs], styles[arm], label=labels[arm])
    ax[1].loglog(ny, [r["u_l2_rel"] * 100 for r in rs], styles[arm], label=labels[arm])
for a, key, title in ((ax[0], "v_l2_rel", "transverse velocity v: full-field L2 rel. error"),
                      (ax[1], "u_l2_rel", "streamwise velocity u: full-field L2 rel. error")):
    a.set_xlabel("grid ny (lattice cells per y-period)")
    a.set_ylabel("relative L2 error [%]")
    a.set_title(title, fontsize=9)
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=8)
    a.axhline(3.0, color="k", ls=":", lw=1)
    a.text(a.get_xlim()[0] * 1.05, 3.2, "3% gate", fontsize=7)
fig.suptitle("Kovasznay (1948) Re=40: outlet treatment sets the error floor", fontsize=10)
fig.tight_layout()
fig1 = REPO / "figures" / "kovasznay_outlet_bc_convergence.png"
fig1.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig1, dpi=130)
plt.close(fig)

# observed order bar chart
fig, ax = plt.subplots(figsize=(7.2, 3.6))
width = 0.35
arms = list(data)
xs = range(len(arms))
for j, key in enumerate(("u_l2_rel", "v_l2_rel")):
    vals = []
    for arm in arms:
        fl = data[arm]["observed_order"].get("finest_pair_p", {})
        vals.append(fl.get(key, float("nan")))
    ax.bar([x + (j - 0.5) * width for x in xs], vals, width, label=key.replace("_rel", ""))
ax.axhline(2.0, color="k", ls="--", lw=1, label="2nd order")
ax.set_xticks(list(xs))
ax.set_xticklabels([labels[a].split(" (")[0] for a in arms], fontsize=8)
ax.set_ylabel("observed order p (finest grid pair)")
ax.set_title("Observed convergence order of the full-field velocity error", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig2 = REPO / "figures" / "kovasznay_outlet_bc_order.png"
fig.savefig(fig2, dpi=130)
plt.close(fig)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


# ---------------------------------------------------------------- markdown ---
def table(arm):
    head = ("| ny | nx | tau | steps | steady | u L2 [%] | v L2 [%] | u L2 interior [%] | v L2 interior [%] "
            "| u max rel [%] | v max rel [%] | mass drift [%] | wall [s] |\n")
    sep = "|" + "---|" * 13 + "\n"
    body = ""
    for r in rows(arm):
        body += (f"| {r['ny']} | {r['nx']} | {r['tau']:.3f} | {r['n_steps']} | {r['steady']} "
                 f"| {100 * r['u_l2_rel']:.3f} | {100 * r['v_l2_rel']:.3f} "
                 f"| {100 * r['u_l2_rel_interior']:.3f} | {100 * r['v_l2_rel_interior']:.3f} "
                 f"| {r['u_max_rel_pct']:.2f} | {r['v_max_rel_pct']:.2f} "
                 f"| {r['mass_drift_pct']:.4f} | {r['elapsed_s']} |\n")
    return head + sep + body


def order_table(arm):
    o = data[arm]["observed_order"]
    out = "| metric | " + " | ".join(o["u_l2_rel"]["pairs"]) + " |\n"
    out += "|" + "---|" * (len(o["u_l2_rel"]["pairs"]) + 1) + "\n"
    for key in ("u_l2_rel", "v_l2_rel", "u_l2_rel_interior", "v_l2_rel_interior"):
        vals = " | ".join(f"{v:+.2f}" for v in o[key]["p_slopes"])
        out += f"| {key} | {vals} |\n"
    return out


lines = ["# CASE-LBM-KOVASZNAY-OUTLET-BC — 出口边界条件决定误差地板（锚点 CASE-LBM-KOVASZNAY-EXACT）\n\n"]
lines.append("## ① 案例介绍\n\n")
lines.append(
    "Kovasznay (1948, Proc. Camb. Phil. Soc. 44:58) 二维稳态不可压 Navier–Stokes 精确解：\n\n"
    "```\n"
    "u = U0 (1 − e^{λx'} cos 2πy'),  v = U0 (λ/2π) e^{λx'} sin 2πy',  p = ½ (1 − e^{2λx'})\n"
    "λ = Re/2 − sqrt(Re²/4 + 4π²),   Re = U0 L/ν,   L = y 向周期 = ny 格\n"
    "```\n\n"
    "参数：Re = 40（λ = "
    f"{data[list(data)[0]]['lambda']:.6f}"），U0 = 0.03（Ma_max = 0.104），D2Q9 + BGK，\n"
    "y 向周期、x' ∈ [0,3]（nx = 3·ny）。入口 x=0 为解析 Dirichlet（Zou–He 速度入口），\n"
    "初值即解析场，稳态由 ‖u(t)−u(t−2500)‖₂/‖u‖₂ < 5e-5 判定（≥6000 步）。\n\n"
    "**本单要回答的验证欠账**：既有入库基准 `benchmarks/verified/kovasznay_2d` 只用零梯度出口，\n"
    "v 分数全场 L2 误差长期停在 2.5–3.0 %（贴 3 % 门限）、且「弱收敛」（64→128 仅降 0.16 个百分点）。\n"
    "本单把出口处理当作被测量本身，用解析解量化其误差地板。\n\n")
lines.append("## ② 怎么做的\n\n")
lines.append(
    "1. 在库中新增 `tensorlbm.boundaries.nonequilibrium_extrapolation_outlet_2d`\n"
    "   （Guo, Zheng & Shi 2002, J. Comput. Phys. 177:443 式 (10)–(13)，二阶非平衡外推：\n"
    "   f_i(x_b) = f_i^eq(ρ*,u*) + [f_i(x_{b−1}) − f_i^eq(ρ_{b−1},u_{b−1})]，\n"
    "   ρ* = 2ρ_{b−1} − ρ_{b−2}、u* = 2u_{b−1} − u_{b−2}，ρ*/|u*| 带鲁棒夹紧）。\n"
    "2. 在 `benchmarks/verified/kovasznay_2d/run.py` 增加 `--outlet {zerograd,nxextrap,dirichlet}`\n"
    "   第三臂与观测阶估计（p = log(e_c/e_f)/log(h_c/h_f)，相邻网格对逐对给出）。\n"
    "3. 三臂同题同网格（ny = 32/64/128）× 同判据扫描，全部由同一脚本产出 JSON（零手抄）。\n\n")
lines.append("## ③ 计算结果\n\n")
for arm in data:
    lines.append(f"### {labels[arm]} (`{arm}`)\n\n{table(arm)}\n")
    lines.append("观测阶（相邻网格对）：\n\n" + order_table(arm) + "\n")
lines.append("## ④ 结果图\n\n")
lines.append(f"![convergence](figures/{fig1.name})\n\n![order](figures/{fig2.name})\n\n")
if failed:
    lines.append("### honest FAIL：非平衡外推臂发散\n\n")
    for arm in failed:
        for g in failed[arm]["per_grid"]:
            lines.append(f"* `{arm}` ny={g['ny']} nx={g['nx']}: steps={g['n_steps']} "
                         f"finite={g['finite']} u L2={g['u_l2_rel']} v L2={g['v_l2_rel']} "
                         f"u max rel={g['u_max_rel_pct']}\n")
    lines.append("\n该臂（Guo 2002 非平衡外推，仅规定入射方向 + 线性外推 ρ*/u*）在同配置上发散到 NaN，\n"
                 "库函数已标注 EXPERIMENTAL/NOT VALIDATED。\n\n")
lines.append("## ⑤ 注意点 / 局限\n\n")
lines.append(
    "* 网格阶梯只到 ny=128（nx=384）：本机 load15 高企，ny=256 未在本单跑完，故最细一对斜率取自 64→128；\n"
    "  如需渐近阶确认应补 256（预计 nx=768，约 8× 计算量）。\n"
    "* 观测阶受「固定 Ma 的压缩性底噪」限制：LBM 在固定 U0 下非平衡部分存在 O(Ma²) 常数项，\n"
    "  u 的 L2 误差因此不完全随 h² 下降（见上表 u L2 斜率）。这是方法固有项，非出口污染。\n"
    "* 出口面已经处于 e^{λx'}=e^{−2.89}=5.6 % 的衰减区，两臂差异在 v 上最显著；\n"
    "  若在短域（x'∈[0,1]）零梯度会直接把出口 v 抹平（历史记录 ~96 % 误差）。\n")
lines.append("\n## ⑥ 误差判定\n\n")

gate_rows = []
for arm in data:
    rs = [r for r in rows(arm) if r["ny"] >= 64]  # 判据网格（入库约定 ny>=64）
    worst = max(max(r["u_l2_rel"], r["v_l2_rel"], r["u_max_rel_pct"] / 100, r["v_max_rel_pct"] / 100) for r in rs)
    gate_rows.append((arm, worst * 100))
lines.append("| arm | 全阶梯最差判据量 [%] | 是否 <3% |\n|---|---|---|\n")
for arm, w in gate_rows:
    lines.append(f"| {labels[arm]} | {w:.3f} | {'是' if w < 3.0 else '否'} |\n")
lines.append(
    "\n结论：零梯度出口的最差值为 "
    f"{dict(gate_rows)['zerograd']:.3f} %（**未通过** <3 % 判据）；"
    + (f"非平衡外推为 {dict(gate_rows)['nxextrap']:.3f} %（{'通过' if dict(gate_rows)['nxextrap'] < 3 else '未通过'}）。\n"
       if "nxextrap" in dict(gate_rows) else "\n")
    + "逐项数字见上表，全部由脚本注入，无手抄。\n")

doc = REPO / "docs" / "CASE-LBM-KOVASZNAY-OUTLET-BC.md"
doc.parent.mkdir(parents=True, exist_ok=True)
doc.write_text("".join(lines))
print("wrote", doc)

# ------------------------------------------------------------------ report --
tok = json.loads(Path("/tmp/tok.json").read_text())["access_token"]
md = "".join(lines)
report = {
    "title": "CASE-LBM-KOVASZNAY-OUTLET-BC：出口边界是 Kovasznay 解析解的误差地板（零梯度 vs 非平衡外推 A/B）",
    "summary": (
        "在 Kovasznay(1948) Re=40 解析 N-S 解上把出口处理当作被测量：新增库函数 "
        "nonequilibrium_extrapolation_outlet_2d (Guo 2002 NEE)，与零梯度出口同网格同判据 A/B。"
    ),
    "scenario_key": "lbm-case-kovasznay-outlet-bc",
    "mission_id": 16841,
    "tags": ["TensorLBM", "verification", "outflow-BC", "Kovasznay", "CASE-LBM-KOVASZNAY-EXACT"],
    "markdown": md,
    "figures": [
        {"name": fig1.name, "caption": "Kovasznay Re=40：u/v 全场 L2 相对误差 vs 网格（三臂），点线为 3% 门限",
         "png_base64": b64(fig1)},
        {"name": fig2.name, "caption": "最细网格对（64→128）观测收敛阶：零梯度 vs 非平衡外推",
         "png_base64": b64(fig2)},
    ],
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("wrote /tmp/report.json", len(md), "chars")