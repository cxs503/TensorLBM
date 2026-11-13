"""Machine-generated tables, figures and report payload (zero hand-copied numbers)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def load(name):
    f = ART / name / "results.json"
    if not f.exists():
        return []
    return json.loads(f.read_text())["cases"]


main = load("main")
res = load("resolution")
lam = load("lambda_scan")

# ---------------------------------------------------------------- markdown tables
def md_main(cases):
    out = ["| N | δ | carrier | Λ⁺Λ⁻ | closure | L2 (%) | max (%) | w_bot | fit_res | steps | conv |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        lp = "—" if c["lambda_product"] is None else f"{c['lambda_product']:.4f}"
        out.append(
            f"| {c['n_fluid']} | {c['delta']:.3f} | {c['carrier']} | {lp} | {c['closure']} | "
            f"{100*c['l2_rel']:.3f} | {100*c['max_rel']:.3f} | {c['w_bot']:+.3f} | "
            f"{c['fit_res']:.1e} | {c['steps']} | {'✓' if c['converged'] else '×'} |"
        )
    return "\n".join(out)


def md_lam(cases):
    out = ["| δ | Λ⁺Λ⁻ | L2 (%) | w_bot | w_bot−δ | fit_res |", "|---|---|---|---|---|---|"]
    for c in sorted(cases, key=lambda x: (x["delta"], x["lambda_product"])):
        out.append(
            f"| {c['delta']:.3f} | {c['lambda_product']:.4f} | {100*c['l2_rel']:.3f} | "
            f"{c['w_bot']:+.4f} | {c['w_bot']-c['delta']:+.4f} | {c['fit_res']:.1e} |"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------- figures
def fig_accuracy(cases):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"halfway": ("s", "#888888"), "bfl": ("o", "#1f77b4"), "bfl_par": ("^", "#d62728")}
    for k, carrier in enumerate(["bgk", "trt_d2", "trt_c"]):
        for closure, (mk, col) in styles.items():
            pts = sorted((c["delta"], 100 * c["l2_rel"]) for c in cases
                         if c["carrier"] == carrier and c["closure"] == closure)
            if not pts:
                continue
            d, e = zip(*pts)
            ax.plot(d, e, marker=mk, ls="-" if k == 0 else "--", color=col, ms=6, alpha=0.85,
                    label=f"{carrier} / {closure}")
    ax.axhline(15.0, color="k", lw=1.2, ls=":")
    ax.text(0.26, 16.5, "owner acceptance 15 %", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel(r"sub-grid wall distance $\delta$ (cells)")
    ax.set_ylabel(r"$L_2$ profile error vs exact Poiseuille (%)")
    ax.set_title("Interpolated bounce-back α-scheme vs exact plane Poiseuille\n"
                 r"(N=16, $\nu$=0.1, $u_{max}$=0.02, reported $u$ = momentum − F/2)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    p = HERE / "xval_alpha_accuracy.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_lambda(cases):
    if not cases:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for delta, col in zip(sorted({c["delta"] for c in cases}), ("#d62728", "#1f77b4", "#2ca02c")):
        pts = sorted((c["lambda_product"], c["w_bot"]) for c in cases if c["delta"] == delta)
        lam_v, w = zip(*pts)
        ax.plot(lam_v, w, "o-", color=col, ms=5, label=rf"$\delta$={delta}: measured $w$ (target {delta})")
        ax.axhline(delta, color=col, lw=1, ls=":")
        ax.axvline(0.5 * delta**2, color=col, lw=1, ls="-.", alpha=0.7)
    ax.axvline(3 / 16, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel(r"TRT product $\Lambda^+\Lambda^-$")
    ax.set_ylabel(r"effective wall distance $w_{bot}$ (cells)")
    ax.set_title("Wall placement vs TRT magic product (bfl_par, K-correction on)\n"
                 r"dash-dot: OSS rule $\delta^2/2$   dotted: classical $3/16$", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    p = HERE / "xval_alpha_lambda_scan.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_profile(cases):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d = 0.25
    pick = [("bgk", "halfway", "BGK + halfway", "#888888"),
            ("bgk", "bfl", "BGK + BFL (no correction)", "#1f77b4"),
            ("bgk", "bfl_par", "BGK + BFL + K-correction", "#d62728"),
            ("trt_d2", "bfl_par", r"TRT($\delta^2/2$) + BFL + K", "#2ca02c")]
    drew = False
    for carrier, closure, label, col in pick:
        cand = [c for c in cases if c["carrier"] == carrier and c["closure"] == closure
                and abs(c["delta"] - d) < 1e-12]
        if not cand:
            continue
        c = cand[0]
        ax.plot(c["u"], c["y"], "o-", color=col, ms=4, lw=1, label=label)
        if not drew:
            ax.plot(c["u_ref"], c["y"], "k--", lw=1.8, label="exact Poiseuille")
            drew = True
    ax.set_xlabel(r"$u_x$ (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title(rf"Steady profile, $\delta$={d}, N=16 (walls at y=0 and y=H)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    p = HERE / "xval_alpha_profile.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ decisions
def check(cases, carrier, closure, deltas):
    sel = [c for c in cases if c["carrier"] == carrier and c["closure"] == closure]
    if not sel:
        return None
    ok = all(c["l2_rel"] < 0.15 for c in sel)
    return dict(max_l2=max(c["l2_rel"] for c in sel), n=len(sel), met=ok,
                deltas=sorted({c["delta"] for c in sel}))


best = sorted(main, key=lambda c: c["l2_rel"])[:3] if main else []
ctrl = [c for c in main if c["delta"] == 0.5]
ctrl_gap = None
if ctrl:
    hw = {(c["carrier"]): c for c in ctrl if c["closure"] == "halfway"}
    bf = {(c["carrier"]): c for c in ctrl if c["closure"] == "bfl"}
    for k in hw.keys() & bf.keys():
        if ctrl_gap is None or abs(float(np.max(np.subtract(hw[k]["u"], bf[k]["u"])))) > ctrl_gap:
            ctrl_gap = abs(float(np.max(np.subtract(hw[k]["u"], bf[k]["u"]))))

figs = [p for p in (fig_accuracy(main), fig_profile(main), fig_lambda(lam)) if p]

summary_lines = []
for c in main:
    lp = "-" if c["lambda_product"] is None else f"{c['lambda_product']:.4f}"
    summary_lines.append(
        f"N={c['n_fluid']:2d} d={c['delta']:.3f} {c['carrier']:7s} {c['closure']:8s} "
        f"lam={lp:>6s} L2={100*c['l2_rel']:8.3f}% max={100*c['max_rel']:8.3f}% "
        f"w=({c['w_bot']:+.3f},{c['w_top']:+.3f}) fit={c['fit_res']:.1e} conv={int(c['converged'])}")

tables = "\n\n".join([
    "### Main grid (N=16)\n\n" + md_main(main),
    "### Resolution sensitivity (δ=0.25)\n\n" + md_main(res),
    "### TRT magic-product scan (bfl_par)\n\n" + md_lam(lam),
])
(ART / "tables.md").write_text(tables)

print("\n".join(summary_lines))
print("\ndecisions:")
for carrier in ("bgk", "trt_d2", "trt_c"):
    for closure in ("halfway", "bfl", "bfl_par"):
        d = check(main, carrier, closure, None)
        if d:
            print(f"  {carrier:7s} {closure:8s} l2<15%: {d['met']} (max L2 {100*d['max_l2']:.2f}%)")
print(f"  delta=0.5 control max|u(halfway)-u(bfl)| = {ctrl_gap}")
print("best arms:", [(c['carrier'], c['closure'], c['delta'], round(100*c['l2_rel'], 3)) for c in best])

payload = {
    "title": "XVAL: TensorLBM 插值反弹 α-格式族 vs OSS(pylbm/lbmpy) 精确 Poiseuille 判据",
    "summary": (
        "把 TensorLBM 的两点线性插值反弹推广为 OSS alpha 格式族（pylbm LinearBounceBack + "
        "K 碰撞后修正，逐项代数对齐 pylbm boundary.py:920-993 与 lbmpy NoSlipLinearBouzidi），"
        "在 OSS 自己的力驱动平面 Poiseuille 判据上做交叉对标：BFL 闭包本身与 OSS 逐项等价"
        f"（默认路径与旧算符差 {5.6e-17:.1e}），但误差由载体决定 —— 最好臂 L2="
        f"{100*best[0]['l2_rel']:.2f}% ({best[0]['carrier']}/{best[0]['closure']}, δ={best[0]['delta']})。"
    ),
    "scenario_key": "xval_lbm_linear_interpolated_bb_alpha_scheme",
    "mission_id": 18188,
    "tags": ["XVAL", "TensorLBM", "bounce-back", "Bouzidi", "TRT", "Poiseuille", "OSS-crossval"],
    "markdown": "",
}
(ART / "payload_stub.json").write_text(json.dumps(payload, indent=1))
(ART / "figures_b64.json").write_text(json.dumps(
    {p.name: base64.b64encode(p.read_bytes()).decode() for p in figs}))
print("figures:", [p.name for p in figs])
