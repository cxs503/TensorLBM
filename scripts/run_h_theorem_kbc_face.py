#!/usr/bin/env python3
"""Run the KBC H-theorem verification face (campaign node P67-N02) and emit artifacts.

Produces
--------
``artifacts/h_theorem_kbc/campaign.json``
    Every number quoted in the report (criterion by criterion, per-step series).
``artifacts/h_theorem_kbc/fig_*.png``
    Matplotlib figures (entropy ledger, BGK-vs-KBC stability, alpha envelope,
    Reynolds envelope, viscosity calibration).
``docs/P67_N02_h_theorem_kbc.md``
    Human-readable report (numbers injected programmatically, zero hand copying).

Usage
-----
    PYTHONPATH=src python scripts/run_h_theorem_kbc_face.py [--quick]

``--quick`` shrinks the step counts (smoke test, ~1 min) without changing the
criteria structure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import tensorlbm.h_theorem as hl

ART = Path("artifacts/h_theorem_kbc")
DOCS = Path("docs")


def _threads() -> None:
    n = int(os.environ.get("H_THM_THREADS", "4"))
    torch.set_num_threads(max(1, n))


def _case_definitions(quick: bool) -> dict[str, hl.CampaignConfig]:
    steps = 400 if quick else 10000
    bgk = 40 if quick else 100
    eq_steps = 300 if quick else 2000
    return {
        # A: under-resolved grid-scale shear wave at tau = 0.501 (high Re).
        #    Main C1/C2/C5 run and the C4/C6 stability comparison share this IC.
        "A_high_re_d2q9": hl.CampaignConfig(
            lattice="d2q9",
            shape=(32, 32),
            ic="shear_wave",
            u0=0.15,
            wavelength=4,
            tau=0.501,
            steps_kbc=steps,
            steps_bgk_max=bgk,
            sample=200,
            eq_tau=1.0,
            eq_steps=eq_steps,
            eq_u0=0.01,
            eq_wavelength=16,
        ),
        # B: near-equilibrium decaying wave (Ma -> 0 envelope of C1/C2/C5).
        "B_near_equilibrium_d2q9": hl.CampaignConfig(
            lattice="d2q9",
            shape=(32, 32),
            ic="shear_wave",
            u0=0.01,
            wavelength=16,
            tau=0.7,
            steps_kbc=steps,
            steps_bgk_max=bgk,
            sample=200,
            eq_tau=1.0,
            eq_steps=eq_steps,
            eq_u0=0.01,
            eq_wavelength=16,
        ),
        # C: same high-Re test on the three-dimensional D3Q27 lattice.
        "C_high_re_d3q27": hl.CampaignConfig(
            lattice="d3q27",
            shape=(4, 4, 16) if quick else (8, 8, 16),
            ic="shear_wave",
            u0=0.15,
            wavelength=4,
            tau=0.501,
            steps_kbc=300 if quick else 3000,
            steps_bgk_max=bgk,
            sample=100,
            eq_tau=1.0,
            eq_steps=eq_steps,
            eq_u0=0.01,
            eq_wavelength=8,
        ),
    }


def _figures(cases: dict[str, dict], sweep: list[dict], sweep3d: list[dict], visc: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ART.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    a = cases["A_high_re_d2q9"]
    c1 = a["c1"]
    series = c1["series"]
    steps = [r["step"] for r in series]

    # --- fig 1: entropy functional and KL divergence (C1) ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [r["H"] for r in series], lw=1.2, label="H (Boltzmann, sum)")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("H")
    ax[0].set_title("KBC entropy functional (C1)")
    ax[0].grid(alpha=0.3)
    ax[1].semilogy(steps, [max(r["H_rel"], 1e-30) for r in series], lw=1.2, color="C3")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel(r"$H_{rel}=\sum f_i\ln(f_i/f_i^{eq})$")
    ax[1].set_title("relative entropy (KL) decay")
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("D2Q9 grid-scale shear wave, $\\tau=0.501$, KBC collision")
    fig.tight_layout()
    p = ART / "fig1_entropy_series.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    # --- fig 2: per-step entropy change + tolerance band (C1 evidence) ---
    dh = [r["delta_H_collision"] for r in series if "delta_H_collision" in r]
    steps_dh = [r["step"] for r in series if "delta_H_collision" in r]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps_dh, dh, lw=1.0, color="C0", label="$\\Delta H$ per collision")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(c1["tolerance_abs"], color="C3", ls="--", lw=0.9, label="round-off tolerance")
    ax.set_xlabel("step")
    ax.set_ylabel("$\\Delta H$")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title(
        f"C1: entropy monotonicity, {c1['violations']} violations / {c1['steps']} steps "
        f"(max positive excess {c1['max_positive_excess_abs']:.2e})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = ART / "fig2_delta_h.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    # --- fig 3: BGK divergence vs KBC survival (C4/C6) ---
    bgk = a["c6_bgk"]["rows"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if bgk:
        ax.semilogy([r["step"] for r in bgk], [max(r["u_max"], 1e-16) for r in bgk], "o-", ms=3, label="BGK")
        ax.semilogy([r["step"] for r in bgk], [max(r["f_min"], 1e-16) for r in bgk], "s--", ms=2, label="BGK min f")
    ax.semilogy(steps, [max(r["u_max"], 1e-16) for r in series], lw=1.4, label="KBC")
    ax.axhline(1.0 / 3.0, color="C3", ls=":", lw=1.0, label="distributed-shear stability limit $u_{max}=c_s^2$")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("max |u|  /  min f$_i$")
    ax.set_title("C4/C6: same IC, same $\\tau$: BGK diverges, KBC survives")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = ART / "fig3_bgk_vs_kbc.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    # --- fig 4: Reynolds/amplitude envelope (sweep) ---
    if sweep:
        fig, ax = plt.subplots(figsize=(8, 4))
        u0 = [r["u0"] for r in sweep]
        bgk_div = [r["bgk_diverged_at"] if r["bgk_diverged_at"] else len(a["c6_bgk"]["rows"]) for r in sweep]
        ax.plot(u0, bgk_div, "o-", color="C3", label="BGK: first diverged step (cap = allowed window)")
        ax.plot(u0, [r["kbc_steps"] if r["kbc_diverged_at"] is None else r["kbc_diverged_at"] for r in sweep],
                "s-", color="C0", label="KBC: first diverged step (cap = run length)")
        ax.set_yscale("log")
        ax.set_xlabel("shear-wave amplitude $u_0$ (lattice units)")
        ax.set_ylabel("step")
        ax.set_title("Stability envelope at $\\tau=0.501$, $\\lambda=4$ (grid-scale shear)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = ART / "fig4_stability_envelope.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p))

    # --- fig 5: alpha envelope (C5) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [r["alpha_min"] for r in series], lw=1.0, label="$\\alpha_{min}$")
    ax.plot(steps, [r["alpha_max"] for r in series], lw=1.0, label="$\\alpha_{max}$")
    ax.plot(steps, [r["alpha_mean"] for r in series], lw=1.0, ls="--", label="$\\alpha_{mean}$")
    ax.plot(steps, [r["alpha_at_lower_bound"] for r in series], lw=1.0, ls=":", label="fraction at $\\alpha=0$")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("$\\alpha$")
    ax.set_title("C5: entropy-conditioned $\\alpha$ stays inside [0, 1] for every cell")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = ART / "fig5_alpha_envelope.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    # --- fig 6: C2 residual levels vs the 1e-10 gate ---
    labels = ["telescoping\n$\\Delta H$", "KBC ledger\n$P=D$", "BGK\n$H_2^*=\\sigma^2H_2$"]
    vals = [max(a["c2"]["telescoping_rel"], 1e-20),
            max(a["c2"]["kbc_ledger_rel"], 1e-20),
            max(a["c2"]["bgk_sigma2_rel"], 1e-20)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["C0", "C2", "C1"])
    ax.axhline(a["c2"]["tolerance"], color="C3", ls="--", lw=1.2, label="gate 1e-10")
    ax.axhline(2.2e-16, color="k", ls=":", lw=1.0, label="double-precision eps")
    ax.set_yscale("log")
    ax.set_ylabel("relative residual")
    ax.set_title("C2: exact entropy-production / dissipation identities")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = ART / "fig6_c2_residuals.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    # --- fig 7: viscosity calibration vs analytical reference ---
    if visc:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lams = sorted({r["wavelength"] for r in visc})
        taus = sorted({r["tau"] for r in visc})
        ax[0].plot(taus, [1.0 / 3.0 * (t - 0.5) for t in taus], "k--", label="theory $c_s^2(\\tau-1/2)$")
        for i, lam in enumerate(lams):
            rows = sorted((r for r in visc if r["wavelength"] == lam), key=lambda r: r["tau"])
            ax[0].plot([r["tau"] for r in rows], [r["nu_measured"] for r in rows], "o",
                       color=f"C{i}", label=f"KBC $\\lambda={int(lam)}$")
            ax[1].plot([r["tau"] for r in rows], [100 * r["rel_error"] for r in rows], "o-",
                       color=f"C{i}", label=f"$\\lambda={int(lam)}$")
        ax[0].set_xlabel("$\\tau$")
        ax[0].set_ylabel("$\\nu$")
        ax[0].set_title("shear viscosity from the entropy-conditioned KBC")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].axhline(3.0, color="C3", ls="--", label="3% acceptance line")
        ax[1].axhline(-3.0, color="C3", ls="--")
        ax[1].set_xlabel("$\\tau$")
        ax[1].set_ylabel("error vs theory  [%]")
        ax[1].set_title("viscosity error: $O(k^2)$ spatial truncation")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        p = ART / "fig7_viscosity_calibration.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(str(p))

    # --- fig 9: entropy-monotonicity violation magnitude vs amplitude ---
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for rows, lab, mk in ((sweep, "D2Q9 $\\lambda=4$", "o"), (sweep3d, "D3Q27 $\\lambda=8$", "s")):
        if not rows:
            continue
        xs = [r["u0"] for r in rows]
        ys = [max(r["kbc_max_positive_excess_rel"], 1e-18) for r in rows]
        ax.loglog(xs, ys, mk + "-", label=lab)
    ax.axhline(1e-12, color="C3", ls="--", lw=1.0, label="C1 tolerance (1e-12 of $\\Sigma|f\\ln(f/w)|$)")
    ax.axhline(2.2e-16, color="k", ls=":", lw=1.0, label="double eps")
    ax.set_xlabel("shear-wave amplitude $u_0$")
    ax.set_ylabel("max per-step entropy increase (relative)")
    ax.set_title("C1 violation magnitude vs amplitude (grid-scale shear, $\\tau=0.501$)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p9 = ART / "fig9_violation_scaling.png"
    fig.savefig(p9, dpi=140)
    plt.close(fig)
    made.append(str(p9))

    # --- fig 8: D3Q27 entropy series ---
    c = cases["C_high_re_d3q27"]
    s3 = c["c1"]["series"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["step"] for r in s3], [r["H"] for r in s3], lw=1.2, label="H (D3Q27)")
    ax.set_xlabel("step")
    ax.set_ylabel("H")
    ax.set_title(
        f"D3Q27 KBC: {c['c1']['violations']} C1 violations / {c['c1']['steps']} steps, "
        f"BGK diverged at {c['c6_bgk']['diverged_at']}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = ART / "fig8_d3q27_entropy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))
    return made


def _markdown(cases: dict[str, dict], sweep: list[dict], sweep3d: list[dict], visc: list[dict], elapsed: float) -> str:
    a = cases["A_high_re_d2q9"]
    L: list[str] = []
    L.append("# P67-N02 · KBC 碰撞算子 H 定理验证面\n")
    L.append(
        "本文件由 `scripts/run_h_theorem_kbc_face.py` 程序化生成，所有数字取自 "
        "`artifacts/h_theorem_kbc/campaign.json`，无手抄。\n"
    )
    L.append("## 1. 判据与结论\n")
    L.append("| 判据 | 定义 | 结果 | 关键数字 |")
    L.append("|---|---|---|---|")
    L.append(
        f"| C1 | H 单调不增（每步，0 违反） | {'PASS' if a['c1']['pass'] else 'FAIL'} | "
        f"{a['c1']['violations']} / {a['c1']['steps']} 违反，最大正向超出 {a['c1']['max_positive_excess_abs']:.3e}"
        f"（容差 {a['c1']['tolerance_abs']:.3e}） |"
    )
    L.append(
        f"| C2 | 熵产生-耗散恒等式 ≤1e-10 | {'PASS' if a['c2']['pass'] else 'FAIL'} | "
        f"telescoping {a['c2']['telescoping_rel']:.2e} / KBC ledger {a['c2']['kbc_ledger_rel']:.2e} / "
        f"BGK $\\sigma^2$ {a['c2']['bgk_sigma2_rel']:.2e} |"
    )
    L.append(
        f"| C3 | 平衡态恢复 $H/H_{{eq}}-1\\le10^{{-6}}$ | {'PASS' if a['c3']['pass'] else 'FAIL'} | "
        f"{a['c3']['H_over_H_eq_minus_one']:.3e}（{a['c3']['steps']} 步，$u_{{max}}$={a['c3']['u_max_final']:.2e}） |"
    )
    L.append(
        f"| C4 | $\\tau=0.501$：KBC {a['c4_kbc']['steps']} 步不发散 ∧ BGK ≤{a['c6_bgk']['steps_allowed']} 步发散 | "
        f"{'PASS' if a['c4']['pass'] else 'FAIL'} | KBC 存活={a['c4_kbc']['survived']}；BGK 第 {a['c6_bgk']['diverged_at']} 步发散 |"
    )
    L.append(
        f"| C5 | $\\alpha\\in[0,1]$ 每步每格 | {'PASS' if a['c5']['pass'] else 'FAIL'} | "
        f"越界 {a['c5']['violations']} / {a['c5']['samples']} 采样；$\\alpha\\in[{a['c5']['alpha_min']:.3f},{a['c5']['alpha_max']:.3f}]$ |"
    )
    L.append(
        f"| C6 | BGK 负控失稳 | {'PASS' if a['c6_bgk']['diverged'] else 'FAIL'} | "
        f"第 {a['c6_bgk']['diverged_at']} 步（$u_{{max}}$={a['c6_bgk']['rows'][-1]['u_max']:.3f}） |"
    )
    L.append(
        f"| C7 | $\\alpha\\equiv0$ 退化负控 | {'PASS' if a['c7']['pass'] else 'FAIL'} | "
        f"与熵条件解的最大差 {a['c7']['post_collision_diff_entropic_vs_alpha0']:.3e}；"
        f"$\\alpha=0$ 自身 {a['c7']['alpha_zero_diverged_at']} 步发散 |"
    )
    L.append("")
    L.append("## 2. 三种工况（同一验证面，不同参数）\n")
    L.append("| 工况 | 格点 | $u_0$ | $\\lambda$ | $\\tau$ | 步数 | C1 违反 | BGK 发散步 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, name in (("A_high_re_d2q9", "A 高 Re D2Q9"), ("B_near_equilibrium_d2q9", "B 近平衡 D2Q9"),
                      ("C_high_re_d3q27", "C 高 Re D3Q27")):
        cfg = cases[key]["config"]
        L.append(
            f"| {name} | {'x'.join(str(v) for v in cfg['shape'])} | {cfg['u0']} | {cfg['wavelength']} | "
            f"{cfg['tau']} | {cases[key]['c1']['steps']} | {cases[key]['c1']['violations']} | "
            f"{cases[key]['c6_bgk']['diverged_at']} |"
        )
    L.append("")
    L.append("## 3. 稳定性包线（$\\tau=0.501$, $\\lambda=4$）\n")
    L.append("| $u_0$ | $Re_\\lambda$ | BGK 发散步 | KBC 发散步（上限=运行长度） | KBC C1 违反 |")
    L.append("|---|---|---|---|---|")
    for r in sweep:
        L.append(
            f"| {r['u0']:.2f} | {r['re_lattice']:.0f} | {r['bgk_diverged_at']} | "
            f"{r['kbc_diverged_at'] if r['kbc_diverged_at'] else '未发散'} | {r['kbc_h_violations']} |"
        )
    L.append("")
    L.append("### 3b. 三维（D3Q27, $\\lambda=8$）单调性违反幅度\n")
    L.append("| $u_0$ | BGK 发散步 | KBC 发散步 | C1 违反步数 | 最大正向超出（相对） |")
    L.append("|---|---|---|---|---|")
    for r in sweep3d:
        L.append(
            f"| {r['u0']:.2f} | {r['bgk_diverged_at']} | "
            f"{r['kbc_diverged_at'] if r['kbc_diverged_at'] else '未发散'} | "
            f"{r['kbc_h_violations']} | {r['kbc_max_positive_excess_rel']:.2e} |"
        )
    L.append("")
    L.append("## 4. 粘度标定（对解析基准误差）\n")
    L.append("| $\\lambda$ | $\\tau$ | $\\nu_{theory}$ | $\\nu_{measured}$ | 相对误差 |")
    L.append("|---|---|---|---|---|")
    for r in sorted(visc, key=lambda r: (r["wavelength"], r["tau"])):
        L.append(
            f"| {int(r['wavelength'])} | {r['tau']:.2f} | {r['nu_theory']:.6e} | {r['nu_measured']:.6e} | "
            f"{100 * r['rel_error']:+.3f}% |"
        )
    L.append("")
    L.append(f"_总运行时间 {elapsed:.1f} s（`H_THM_THREADS` 限线程，float64）。_\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced step counts (smoke test)")
    args = ap.parse_args()
    _threads()
    ART.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    cases: dict[str, dict] = {}
    for name, cfg in _case_definitions(args.quick).items():
        t0 = time.time()
        out = hl.run_campaign(cfg)
        out["wall_time_s"] = time.time() - t0
        cases[name] = out
        print(f"[case {name}] {out['verdict']} in {out['wall_time_s']:.1f}s", flush=True)

    base = _case_definitions(args.quick)["A_high_re_d2q9"]
    sweep = hl.run_stability_sweep(
        base,
        u0_list=(0.01, 0.05, 0.10, 0.15, 0.20, 0.25) if not args.quick else (0.05, 0.15, 0.25),
        kbc_steps=120 if args.quick else 400,
        bgk_steps=40 if args.quick else 100,
    )
    print(f"[sweep] {len(sweep)} amplitudes done", flush=True)

    base3d = hl.CampaignConfig(
        lattice="d3q27",
        shape=(4, 4, 16) if args.quick else (8, 8, 16),
        ic="shear_wave",
        u0=0.05,
        wavelength=8,
        tau=0.501,
    )
    sweep3d = hl.run_stability_sweep(
        base3d,
        u0_list=(0.01, 0.05, 0.15) if not args.quick else (0.05, 0.15),
        kbc_steps=80 if args.quick else 200,
        bgk_steps=40 if args.quick else 100,
    )
    print(f"[sweep3d] {len(sweep3d)} amplitudes done", flush=True)

    visc = hl.shear_wave_viscosity_study(
        taus=(0.6, 0.7, 0.8, 1.0, 1.5),
        u0=0.01,
        wavelengths=(16, 32),
        steps=300 if args.quick else 1200,
        sample=30,
    )
    print(f"[viscosity] {len(visc)} tau values done", flush=True)

    figs = _figures(cases, sweep, sweep3d, visc)
    elapsed = time.time() - t_start

    payload = {
        "mission": "P67-N02",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed,
        "torch_threads": torch.get_num_threads(),
        "cases": cases,
        "stability_sweep": sweep,
        "stability_sweep_d3q27": sweep3d,
        "viscosity_study": visc,
        "figures": figs,
        "verdict": {
            "A_high_re_d2q9": cases["A_high_re_d2q9"]["verdict"],
            "B_near_equilibrium_d2q9": cases["B_near_equilibrium_d2q9"]["verdict"],
            "C_high_re_d3q27": cases["C_high_re_d3q27"]["verdict"],
        },
    }
    out_json = ART / "campaign.json"
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=True))
    (DOCS / "P67_N02_h_theorem_kbc.md").write_text(_markdown(cases, sweep, sweep3d, visc, elapsed))
    print(f"wrote {out_json} ({out_json.stat().st_size/1e3:.0f} kB), figures: {len(figs)}", flush=True)
    print(f"TOTAL {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())