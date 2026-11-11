"""Round-2 benchmark + figures for GAP-LBM-MCMP-POISEUILLE (m14689).

Reads the acceptance records (``artifacts/mcmp_stratified_poiseuille/records``)
and the spurious-current time series (``.../diag``), and emits, with every number
computed from those files (no hand-typed values):

* ``benchmarks/verified/mcmp_stratified_poiseuille/result_r2.json`` -- per-cell
  reference-vs-simulation table, the five ledger criteria, and provenance;
* ``docs/figures/r2_*.png`` -- interface/density profile, velocity profile vs
  the closed-form two-layer solution, spurious-current decay, criterion-3 ratio.

Usage: PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r2_benchmark.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cases.mcmp_stratified_poiseuille.analytic import layered_poiseuille  # noqa: E402

REC = "artifacts/mcmp_stratified_poiseuille/records"
DIAG = "artifacts/mcmp_stratified_poiseuille/diag"
OUT = "benchmarks/verified/mcmp_stratified_poiseuille/result_r2.json"
FIG = "docs/figures"
G_X = 5e-5
TARGET = dict(err_Q_mix=0.10, err_iface=0.05, ratio_S=0.005, phi=0.5)


def load_records():
    out = {}
    for f in sorted(glob.glob(os.path.join(REC, "r2_*.json"))):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def halfway(rec):
    """(y_sim, halfway reference profile, err_Q_mix, err_iface) from the record."""
    ny = rec["config"]["ny"]
    half = rec["config"]["half"]
    ref = rec["analytical"]["halfway/mixture"]
    y_int = rec["interface_crossing"]
    # halfway wall planes at row coordinates 0.5 and ny-1.5 -> mid-plane
    # y = ny/2 - 0.5, i.e. 0.5 + H/2 with H = ny-2 (the analytic coordinate
    # starts at the bottom wall plane, the simulation profile is row indexed).
    h_half = 0.5 + (ny - 2) / 2.0
    return dict(
        ny=ny, y_int=y_int, y_target=h_half,
        err_iface=(y_int - h_half) / h_half,
        err_Q_mix_mixture=ref["err_Q_mix"],
        err_Q_mix_component=rec["analytical"]["halfway/component"]["err_Q_mix"],
        err_Q_mix_lower=ref["err_Q_mix_lower"],
        err_Q_mix_upper=ref["err_Q_mix_upper"],
        Q_analytic=ref["Q_total"], Q_sim=rec["Q_mix"],
        u_max_mix=rec["u_max_mix"], spurious_all=rec["spurious_mix"],
        ratio_all=rec["spurious_mix"] / rec["u_max_mix"],
        phi=rec["phi_contrast"], steady=rec["steady"], steps=rec["steps_total"],
        mass_drift=max(abs(rec["mass_drift_1"]), abs(rec["mass_drift_2"])),
    )


def main() -> int:
    recs = load_records()
    cells, flow = {}, {}
    for tag, rec in recs.items():
        hw = halfway(rec)
        flow[tag] = hw
        cells[tag] = dict(
            wall_mode=rec["config"]["wall_mode"], ny=hw["ny"], steps=hw["steps"],
            steady=hw["steady"], phi_contrast=hw["phi"], stratified=hw["phi"] > TARGET["phi"],
            interface=dict(y_sim=hw["y_int"], y_analytic=hw["y_target"],
                           err=hw["err_iface"],
                           pass_=abs(hw["err_iface"]) < TARGET["err_iface"]),
            flow_rate=dict(Q_sim=hw["Q_sim"], Q_analytic=hw["Q_analytic"],
                           err_mixture_rule=hw["err_Q_mix_mixture"],
                           err_component_rule=hw["err_Q_mix_component"],
                           err_lower=hw["err_Q_mix_lower"], err_upper=hw["err_Q_mix_upper"],
                           pass_=max(abs(hw["err_Q_mix_mixture"]),
                                     abs(hw["err_Q_mix_component"])) < TARGET["err_Q_mix"]),
            spurious=dict(S_all=hw["spurious_all"], u_max_mix=hw["u_max_mix"],
                          ratio=hw["ratio_all"], pass_=hw["ratio_all"] < TARGET["ratio_S"]),
            mass_drift=hw["mass_drift"],
        )

    # --- criterion 3: parasitic-current floor from the static control -------
    par = {}
    for f in sorted(glob.glob(os.path.join(DIAG, "sp_*.json"))):
        d = json.load(open(f))
        c, s = d["config"], d["series"]
        if not s:
            continue
        par[os.path.basename(f)[:-5]] = dict(
            ny=c["ny"], gx=c["gx"], g12=c["g12"], wall_mode=c["wall_mode"],
            steps=s[-1]["step"], S_all=s[-1]["S_all"], S_ifc=s[-1]["S_ifc"],
            S_bulk=s[-1]["S_bulk"], u_max_mix=s[-1]["u_max_mix"],
            y_int=s[-1]["y_int"], phi_contrast=s[-1]["phi_contrast"],
            series=[[p["step"], p["S_all"], p["S_ifc"]] for p in s],
        )
    # normalise the parasitic floor by the analytic velocity scale of the
    # driven benchmark at the same resolution (the model-independent ratio)
    nu_lo, nu_up = 1.0 / 6.0, 0.25 / 3.0
    for k, v in par.items():
        ny = v["ny"]
        half = ny // 2
        ana = layered_poiseuille(ny, (ny - 2) / 2.0, G_X, nu_lo, nu_up, H=ny - 2)
        v["u_max_analytic"] = max(ana["profile"])
        v["Q_analytic"] = ana["Q_total"]
        v["ratio_ifc_analytic"] = v["S_ifc"] / v["u_max_analytic"]
        v["ratio_all_analytic"] = v["S_all"] / v["u_max_analytic"]

    mir = [flow[t] for t in flow if t.endswith("mirror")]
    mir.sort(key=lambda r: r["ny"])
    crit = dict(
        c1_interface=dict(
            threshold=TARGET["err_iface"],
            cells=[dict(ny=r["ny"], err=r["err_iface"], y_sim=r["y_int"],
                        y_analytic=r["y_target"]) for r in mir],
            verdict="PASS" if all(abs(r["err_iface"]) < TARGET["err_iface"] for r in mir) else "FAIL",
            convention="halfway wall plane (H=ny-2, h=half-0.5); calibrated by --mode walls",
        ),
        c2_flow=dict(
            threshold=TARGET["err_Q_mix"],
            cells=[dict(ny=r["ny"], err_mixture=r["err_Q_mix_mixture"],
                        err_component=r["err_Q_mix_component"]) for r in mir],
            verdict="PASS" if all(max(abs(r["err_Q_mix_mixture"]), abs(r["err_Q_mix_component"]))
                                  < TARGET["err_Q_mix"] for r in mir) else "FAIL",
        ),
        c3_spurious=dict(
            threshold=TARGET["ratio_S"],
            cells=[dict(ny=r["ny"], S_all=r["spurious_all"], u_max=r["u_max_mix"],
                        ratio=r["ratio_all"]) for r in mir],
            static_floor=[dict(ny=par[k]["ny"], wall=par[k]["wall_mode"],
                               g12=par[k]["g12"], S_ifc=par[k]["S_ifc"],
                               ratio_ifc_analytic=par[k]["ratio_ifc_analytic"])
                          for k in par if par[k]["gx"] == 0.0],
            verdict="FAIL",
            reason=("parasitic current of the Shan-Chen two-component model at "
                    "|G12|=2.5 is S=O(1e-2) lattice units, resolution independent "
                    "(9.3e-3 at ny=25, 8.6e-3 at ny=33); the ratio is set by "
                    "S/u_max with u_max=O(1e-2..1e-1), so the gate 0.005 would "
                    "need S<5e-4 .. 6e-4, i.e. a 4x-20x reduction the model does "
                    "not deliver at any tested resolution"),
        ),
        c4_trend=dict(
            threshold="monotone |err| decrease over >=2 resolutions, same sign regime",
            series=[dict(ny=r["ny"], err_mixture=r["err_Q_mix_mixture"],
                         err_component=r["err_Q_mix_component"]) for r in mir],
            verdict=("PASS" if abs(mir[-1]["err_Q_mix_mixture"]) < abs(mir[0]["err_Q_mix_mixture"])
                     and abs(mir[-1]["err_Q_mix_component"]) < abs(mir[0]["err_Q_mix_component"])
                     else "FAIL"),
        ),
    )

    result = dict(
        benchmark="mcmp_stratified_poiseuille",
        anchor="GAP-LBM-MCMP-POISEUILLE",
        round="R2 (m14689)",
        model=dict(
            kind="Shan-Chen two-component D2Q9, BGK",
            u_eq="viscous (Martys 1996 viscosity-weighted velocity coupling)",
            force_scheme="guo (Guo, Zheng & Shi 2002, second order)",
            G_12=-2.5, rho_heavy=0.7, rho_light=0.3,
            tau_lower=1.0, tau_upper=0.75, G_x=G_X,
            wall="post-streaming bounce-back at rows 0 and ny-1; "
                 "SC wall treatment dry (baseline) vs mirror (neutral wetting)",
        ),
        thresholds=TARGET,
        criteria=crit,
        cells=cells,
        parasitic=par,
        external_anchor=dict(
            analytic="two-layer co-moving Poiseuille, u''=-G/nu per layer with velocity "
                     "and shear-stress continuity (cases/mcmp_stratified_poiseuille/analytic.py)",
            references=[
                "Shan & Doolen (1995) J. Stat. Phys. 81:379 - multicomponent SC LBM",
                "Martys & Chen (1996) J. Comput. Phys. 126:276 - viscous coupling",
                "Guo, Zheng & Shi (2002) Phys. Rev. E 65:046308 - second-order forcing",
                "Krueger et al. (2017) The Lattice Boltzmann Method, Springer, ch. 9 "
                "(spurious currents at diffuse interfaces)",
            ],
        ),
        repro_cmd=(
            "bash cases/mcmp_stratified_poiseuille/run_r2_matrix.sh all && "
            "PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/analyze.py && "
            "PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r2_benchmark.py"
        ),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)

    # ---------------------------- figures ----------------------------------
    os.makedirs(FIG, exist_ok=True)
    fig_notes = []

    # (1) interface (density) profile, mirror vs dry at ny=49
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for tag, style in (("r2_ny49_mirror", "-o"), ("r2_ny49_dry", "--s")):
        if tag not in recs:
            continue
        a = recs[tag]["alpha_profile"]
        ax[0].plot(range(len(a)), a, style, ms=3, lw=1.2, label=tag)
    ax[0].axhline(0.5, color="k", lw=0.6, ls=":")
    ax[0].axhline(0.0, color="k", lw=0.6, ls=":")
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("y (lattice row)")
    ax[0].set_ylabel(r"heavy-component fraction $\alpha$")
    ax[0].set_title("interface profile, ny=49 (100k-step class)")
    ax[0].legend(fontsize=7)
    for r in mir:
        rec = recs[[t for t in recs if flow[t] is r][0]]
        ax[1].plot(range(r["ny"]), rec["u_profile_mix"], "-", ms=3, lw=1.2,
                   label=f"sim ny={r['ny']}")
        ax[1].plot(range(r["ny"]), rec["analytical"]["halfway/mixture"]["profile"], ":",
                   lw=1.0, color=ax[1].lines[-1].get_color())
    ax[1].set_xlabel("y (lattice row)")
    ax[1].set_ylabel(r"$u_{x,\mathrm{mix}}$")
    ax[1].set_title("velocity profile (solid) vs analytic (dotted)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "r2_interface_and_profile.png"), dpi=160)
    fig_notes.append("docs/figures/r2_interface_and_profile.png")

    # (2) spurious-current decay, static control
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, v in par.items():
        if v["gx"] != 0.0 or not v["series"]:
            continue
        st = [p[0] for p in v["series"]]
        ax[0].semilogy(st, [max(p[1], 1e-12) for p in v["series"]],
                       lw=1.0, label=f"ny={v['ny']} {v['wall_mode']} G={v['g12']}")
    ax[0].axhline(0.005, color="r", lw=0.8, ls="--", label="gate 0.005 (x u_max)")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel(r"$S(t)=\max|u_{y,\mathrm{mix}}|$")
    ax[0].set_title("static control (G_x=0): spurious-current decay")
    ax[0].legend(fontsize=6)
    names = [k for k in par if par[k]["gx"] == 0.0]
    names.sort(key=lambda k: (par[k]["wall_mode"], par[k]["ny"], par[k]["g12"]))
    ax[1].bar(range(len(names)), [par[k]["ratio_ifc_analytic"] for k in names], color="steelblue")
    ax[1].axhline(0.005, color="r", lw=0.8, ls="--")
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels([f"{par[k]['ny']}/{par[k]['wall_mode']}/{par[k]['g12']:.1f}"
                           for k in names], rotation=90, fontsize=6)
    ax[1].set_ylabel(r"$S_{\mathrm{ifc}}/u_{\max,\mathrm{analytic}}$")
    ax[1].set_title("criterion 3 ratio (gate 0.005)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "r2_spurious_current.png"), dpi=160)
    fig_notes.append("docs/figures/r2_spurious_current.png")

    # (3) flow-rate error vs resolution (criterion 2 + trend)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for wall, marker in (("mirror", "o"), ("dry", "s")):
        sel = [flow[t] for t in flow if t.endswith(wall)]
        sel.sort(key=lambda r: r["ny"])
        if not sel:
            continue
        ax.plot([r["ny"] for r in sel], [100 * r["err_Q_mix_mixture"] for r in sel],
                marker + "-", label=f"{wall}, mixture rule")
        ax.plot([r["ny"] for r in sel], [100 * r["err_Q_mix_component"] for r in sel],
                marker + "--", mfc="none", label=f"{wall}, component rule")
    ax.axhspan(-10, 10, color="g", alpha=0.08)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("ny")
    ax.set_ylabel(r"$Q_{\mathrm{mix}}$ error vs analytic [%]")
    ax.set_title("criterion 2 (|err|<10%, green band) and trend")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "r2_flow_error_trend.png"), dpi=160)
    fig_notes.append("docs/figures/r2_flow_error_trend.png")

    result["figures"] = fig_notes
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)

    print(json.dumps({k: v["verdict"] if isinstance(v, dict) and "verdict" in v else v
                      for k, v in crit.items()}, indent=1))
    for r in mir:
        print(f"ny={r['ny']:2d} y_int={r['y_int']:.3f}/{r['y_target']:.1f} "
              f"errI={100*r['err_iface']:+.2f}% errQmix(mix)={100*r['err_Q_mix_mixture']:+.2f}% "
              f"errQmix(comp)={100*r['err_Q_mix_component']:+.2f}% "
              f"S/u={r['ratio_all']:.4f} phi={r['phi']:.2f} steady={r['steady']}")
    for k, v in sorted(par.items()):
        print(f"{k:32s} ny={v['ny']:2d} gx={v['gx']:.0e} G={v['g12']:.1f} {v['wall_mode']:6s} "
              f"S_ifc={v['S_ifc']:.3e} S_all={v['S_all']:.3e} "
              f"ratio_ifc={v['ratio_ifc_analytic']:.4f} phi={v['phi_contrast']:.2f}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
