#!/usr/bin/env python3
"""Mission P151W1-N04 -- cavitation + ventilation validation chain (figures + JSON).

Runs :func:`tensorlbm.cavitation_chain.validation.run_validation` (all numbers in
``artifacts/p151w1_n04/validation.json``) and renders the four report figures into
``artifacts/p151w1_n04/``.  Nothing is typed in by hand: every plotted point comes
from the JSON written by the same run.

    PYTHONPATH=src python scripts/p151w1_n04_cavitation_chain.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.cavitation_chain.kunz import Kunz0D, kunz_mass_transfer, kunz_timescale  # noqa: E402
from tensorlbm.cavitation_chain.section import isobaric_cavity, suction_pressure  # noqa: E402
from tensorlbm.cavitation_chain.thrust import blade_element_solution, design_for_targets  # noqa: E402
from tensorlbm.cavitation_chain.validation import write_validation  # noqa: E402


def fig_cavity_length(data: dict, out: Path) -> Path:
    sc = data["scaling"]
    ref = sc["reference"]
    mod = sc["model"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.loglog(mod["sigma"], mod["l_cav"], "o-", color="#1f77b4", label="isobaric model (panel Cp)")
    ax.errorbar(
        ref["sigma"], ref["l_cav_exp"], yerr=ref["band_l_cav"], fmt="s",
        color="#d62728", capsize=4, label="NACA0015 water tunnel (digested)",
    )
    fs = data["free_streamline"]["attempts"]
    if fs:
        ax.loglog(
            [a["sigma"] for a in fs], [a["l_cav"] for a in fs], "x--", color="#2ca02c",
            label="free-streamline, not converged",
        )
    ax.set_xlabel(r"cavitation number $\sigma$")
    ax.set_ylabel(r"$L_{cav}/c$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(
        "sheet-cavity length: model $n$=%.2f vs experiment $n$=%.2f (%+.0f %%)"
        % (mod["exponent"]["exponent"], ref["exponent_point"]["exponent"],
           100.0 * sc["exponent_rel_error"]),
        fontsize=9,
    )
    p = out / "fig1_cavity_length_scaling.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_kunz(data: dict, out: Path) -> Path:
    case = data["kunz"]["case"]
    ko = Kunz0D(rho_l=case["rho_l"], rho_v=case["rho_v"], U=case["u_inf_m_s"],
                t_inf=case["t_inf_s"], alpha_v0=1.0e-4)
    q = 0.5 * case["rho_l"] * case["u_inf_m_s"] ** 2
    p_low = case["p_v_pa"] - case["sigma"] * q
    t = np.linspace(0.0, 3.0 * kunz_timescale(case["sigma"], case["t_inf_s"]), 200)
    tr = ko.march(np.full(t.size, p_low), case["p_v_pa"], t[1] - t[0], alpha_v0=1e-4)
    ana = ko.analytic(t, p_low, case["p_v_pa"], alpha_v0=1e-4)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    ax.plot(t / case["t_inf_s"], tr["alpha_v"], "-", color="#1f77b4", label="RK4 (module)")
    ax.plot(t / case["t_inf_s"], ana, "k--", lw=1.2, label="closed form")
    ax.set_xlabel(r"$t/t_\infty$")
    ax.set_ylabel(r"$\alpha_v$")
    ax.set_title("Kunz evaporation branch, $\\sigma$=1", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sig = np.logspace(-2, 1, 60)
    a = 0.3
    ev, cd = kunz_mass_transfer(
        case["p_v_pa"] - sig * q, case["p_v_pa"], 1.0 - a, a,
        case["rho_l"], case["rho_v"], case["u_inf_m_s"],
        t_inf=case["t_inf_s"],
    )
    ax.loglog(sig, np.abs(ev) + 1e-30, "-", color="#d62728", label=r"evaporation $p<p_v$")
    ax.loglog(sig, np.abs(cd) + 1e-30, "-", color="#2ca02c", label=r"condensation $p>p_v$")
    ax.axvline(1.0, color="0.5", ls=":")
    ax.set_xlabel(r"$|\sigma|$ of the pressure departure")
    ax.set_ylabel(r"$|\dot m|$ [kg m$^{-3}$s$^{-1}$]")
    ax.set_title(r"Kunz rate, $\alpha_v$=0.3 (zero outside its branch)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    p = out / "fig2_kunz_mass_transfer.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_thrust(data: dict, out: Path) -> Path:
    th = data["thrust"]
    curve = th["curve"]
    sig = [c["sigma_07"] for c in curve]
    dk = [100.0 * c["delta_kt_ratio"] for c in curve]
    band = th["reference_band_at_sigma_0.6"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(sig, dk, "o-", color="#1f77b4", label=r"$\Delta K_T/K_{T,0}$ (strip chain)")
    ax.axhspan(100 * band[0], 100 * band[1], color="#ff7f0e", alpha=0.2,
               label="empirical 10-15 % band")
    crit = th["sigma_critical_5pct"]["sigma_critical"]
    ax.axvline(crit, color="#d62728", ls="--", lw=1.2,
               label=r"$\sigma_{crit}$ (5 % loss) = %.3f" % crit)
    ax.axvline(0.6, color="0.4", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{0.7R}$")
    ax.set_ylabel(r"$\Delta K_T/K_{T,0}$ [%]")
    ax.set_title("blade-element thrust derating (A_E/A0 = %.2f, $K_{T0}$ = %.3f)"
                 % (th["design"]["blade_area_ratio"], th["curve"][-1]["kt_wetted"]), fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    p = out / "fig3_thrust_derating.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_radial(data: dict, out: Path) -> Path:
    design = design_for_targets()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for s07, col in ((1.0, "#1f77b4"), (0.6, "#d62728")):
        sol = blade_element_solution(s07, design)
        st = sol["strips"]
        x = np.asarray(st["x"])
        axes[0].plot(x, st["sigma"], col, label=r"$\sigma_{0.7R}$=%.1f" % s07)
        axes[1].plot(x, 100.0 * np.asarray(st["relative_lift_loss"]), col,
                     label=r"$\sigma_{0.7R}$=%.1f ($\Delta K_T/K_{T0}$=%.1f %%)"
                     % (s07, 100 * sol["delta_kt_ratio"]))
    axes[0].axhline(1.0, color="0.5", ls=":")
    axes[0].set_xlabel(r"$r/R$")
    axes[0].set_ylabel(r"$\sigma(r)$")
    axes[0].set_title("radial cavitation number", fontsize=9)
    axes[1].set_xlabel(r"$r/R$")
    axes[1].set_ylabel("section lift loss [%]")
    axes[1].set_title("isobaric lift loss along the blade", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    p = out / "fig4_radial_distributions.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_regimes(data: dict, out: Path) -> Path:
    tr = data["transition"]
    sp = suction_pressure("naca0015", 5.0, 400)
    sig = np.linspace(0.35, 2.2, 120)
    l = [isobaric_cavity(float(s), pressure=sp)["l_cav"] for s in sig]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(sig, l, "-", color="#1f77b4", label=r"$L_{cav}/c(\sigma)$")
    ax.axvline(tr["sigma_inception"], color="#9467bd", ls="--",
               label=r"inception $\sigma_i$=%.2f" % tr["sigma_inception"])
    ax.axvline(tr["wing_route"]["sigma_critical"], color="#8c564b", ls="-.",
               label=r"wing route $\sigma_{crit}$=%.2f" % tr["wing_route"]["sigma_critical"])
    ax.axvline(tr["thrust_route"]["sigma_critical"], color="#d62728", ls="--",
               label=r"thrust route $\sigma_{crit}$=%.2f" % tr["thrust_route"]["sigma_critical"])
    ax.axhline(0.5, color="0.4", ls=":", label=r"$(L/c)_{crit}$=0.5")
    ax.axvspan(0.8, 1.2, color="#2ca02c", alpha=0.12, label="measured 0.8-1.2 band")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$L_{cav}/c$")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("sheet / cloud regime map (NACA0015, $\\alpha$=5 deg)", fontsize=9)
    p = out / "fig5_regime_map.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/p151w1_n04")
    args = ap.parse_args()
    out = Path(args.out)
    path = write_validation(out)
    data = json.loads(path.read_text())
    figs = [
        fig_cavity_length(data, out),
        fig_kunz(data, out),
        fig_thrust(data, out),
        fig_radial(data, out),
        fig_regimes(data, out),
    ]
    manifest = {
        "artefact": str(path),
        "figures": [str(f) for f in figs],
        "verdict": data["verdict"],
        "criteria": [
            {"id": c["id"], "pass": c["pass"], "rel_error": c["rel_error"]}
            for c in data["criteria"]
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())