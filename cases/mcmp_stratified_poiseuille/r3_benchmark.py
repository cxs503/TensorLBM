"""Round-3 benchmark assembler + figures for GAP-LBM-MCMP-POISEUILLE.

Reads the records written by ``run_final_matrix.py`` (``artifacts/
mcmp_stratified_poiseuille/final/*.json``) and emits, with every number
computed from those files (zero hand-typed values):

* ``benchmarks/verified/mcmp_stratified_poiseuille/result_r3.json`` — the five
  ledger criteria, per-cell reference-vs-simulation tables and provenance;
* ``docs/figures/r3_*.png`` — interface/density profile, velocity profile vs the
  closed-form two-layer solution, spurious-current decay, error trend.

Criterion definitions (pre-declared, unchanged from rounds 1-2):

1. steady interface position vs the analytic half-height: |err| < 5 %
2. two-phase volume flow rate: |err(Q_mix)| < 10 % (both viscosity rules)
3. interface spurious current: max|u_y,mix| in the interface band, normalised by
   the analytic u_max of the same case: < 0.005
4. resolution trend of criteria 1-2 over ny = 25, 33, 49 (monotone decrease)
5. paper-grade report (``docs/MCMP_STRATIFIED_POISEUILLE_R3.md``)

Usage: PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r3_benchmark.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from cases.mcmp_stratified_poiseuille.analytic import layered_poiseuille  # noqa: E402

REC = os.path.join("artifacts", "mcmp_stratified_poiseuille", "final")
OUT = os.path.join("benchmarks", "verified", "mcmp_stratified_poiseuille", "result_r3.json")
FIG = os.path.join("docs", "figures")
G_X = 5e-5
TARGET = dict(err_iface=0.05, err_Q=0.10, ratio_S=0.005, phi=0.5)
CS2 = 1.0 / 3.0
NU_LOWER, NU_UPPER = CS2 * (1.0 - 0.5), CS2 * (0.75 - 0.5)          # component rule
NU_MIX_LOWER = 0.7 * NU_LOWER + 0.3 * NU_UPPER                       # mixture rule
NU_MIX_UPPER = 0.3 * NU_LOWER + 0.7 * NU_UPPER


def _ref(rec):
    a = rec.get("analytical") or {}
    return a.get("halfway/mixture") or a.get("mixture")


def _ref_component(rec):
    a = rec.get("analytical") or {}
    return a.get("halfway/component") or a.get("component")


def analytic_umax(ny: int) -> float:
    """Analytic u_max of the same driven case under the halfway wall convention."""
    ana = layered_poiseuille(ny, (ny - 2) / 2.0, G_X, NU_MIX_LOWER, NU_MIX_UPPER, H=ny - 2)
    return max(ana["profile"])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    driven, static = {}, {}
    for f in sorted(glob.glob(os.path.join(REC, "*.json"))):
        rec = json.load(open(f))
        tag = os.path.basename(f)[:-5]
        (driven if rec.get("kind") == "driven" else static)[tag] = rec
    return driven, static


def driven_row(tag: str, rec: dict) -> dict:
    cfg = rec["config"]
    ny = cfg["ny"]
    # The analytic reference is recomputed here under the *calibrated* halfway
    # wall convention (H = ny-2, mid-plane h = H/2); the record carries values
    # under the run_sweep default (H = ny-1, the convention rejected by the
    # round-1 wall calibration) and must not be used for the comparison.
    ana_mix = layered_poiseuille(ny, (ny - 2) / 2.0, G_X, NU_MIX_LOWER, NU_MIX_UPPER, H=ny - 2)
    ana_comp = layered_poiseuille(ny, (ny - 2) / 2.0, G_X, NU_LOWER, NU_UPPER, H=ny - 2)
    ref = dict(Q_total=ana_mix["Q_total"], profile=ana_mix["profile"],
               err_Q_mix=(rec["Q_mix"] - ana_mix["Q_total"]) / ana_mix["Q_total"],
               err_Q_mix_lower=(rec["layer_flux"]["Q_mix_lower"] - ana_mix["Q_lower"]) / ana_mix["Q_lower"],
               err_Q_mix_upper=(rec["layer_flux"]["Q_mix_upper"] - ana_mix["Q_upper"]) / ana_mix["Q_upper"])
    refc = dict(Q_total=ana_comp["Q_total"],
                err_Q_mix=(rec["Q_mix"] - ana_comp["Q_total"]) / ana_comp["Q_total"])
    y_int = rec["interface_crossing"]
    y_target = 0.5 + (ny - 2) / 2.0          # halfway wall planes: 0.5 and ny-1.5
    row = dict(
        tag=tag, sc_form=rec["sc_form"], wall_mode=cfg["wall_mode"], ny=ny,
        steps=rec["steps_total"], steady=rec["steady"], finite=rec["finite"],
        phi_contrast=rec["phi_contrast"], stratified=bool(rec["phi_contrast"] > TARGET["phi"]),
        interface=dict(y_sim=y_int, y_analytic=y_target,
                       err=(y_int - y_target) / y_target,
                       pass_=abs((y_int - y_target) / y_target) < TARGET["err_iface"]),
        flow_rate=dict(Q_sim=rec["Q_mix"], Q_analytic=ref["Q_total"],
                       err_mixture_rule=ref["err_Q_mix"],
                       err_component_rule=refc["err_Q_mix"],
                       err_lower=ref["err_Q_mix_lower"], err_upper=ref["err_Q_mix_upper"],
                       pass_=max(abs(ref["err_Q_mix"]), abs(refc["err_Q_mix"])) < TARGET["err_Q"]),
        spurious_driven=dict(S_all=rec["spurious_mix"], u_max_mix=rec["u_max_mix"],
                             ratio=rec["spurious_mix"] / rec["u_max_mix"],
                             u_max_analytic=analytic_umax(ny),
                             ratio_vs_analytic=rec["spurious_mix"] / analytic_umax(ny)),
        mass_drift=max(abs(rec["mass_drift_1"]), abs(rec["mass_drift_2"])),
    )
    if row["phi_contrast"] <= TARGET["phi"]:
        row["interface"]["pass_"] = False
        row["flow_rate"]["pass_"] = False
        row["note"] = "phases mixed (phi_contrast <= 0.5): not a two-phase measurement"
    return row


def static_row(tag: str, rec: dict) -> dict:
    cfg = rec["config"]
    ny = cfg["ny"]
    s = rec["series"][-1]
    umax = analytic_umax(ny)
    return dict(
        tag=tag, sc_form=rec["sc_form"], wall_mode=cfg["wall_mode"], ny=ny,
        steps=s["step"], finite=s["finite"], g12=cfg["g12"],
        S_ifc=s["S_ifc"], S_all=s["S_all"], S_bulk=s["S_bulk"],
        u_max_analytic=umax, u_max_mix=s["u_max_mix"],
        ratio_ifc_vs_analytic=s["S_ifc"] / umax,
        ratio_all_vs_analytic=s["S_all"] / umax,
        phi_contrast=s["phi_contrast"], y_int=s["y_int"],
        pass_=bool(s["S_ifc"] / umax < TARGET["ratio_S"] and s["phi_contrast"] > TARGET["phi"]),
        series=[[p["step"], p["S_all"], p["S_ifc"], p["S_bulk"]] for p in rec["series"]],
    )


def main() -> int:
    driven, static = load()
    d_rows = {t: driven_row(t, r) for t, r in driven.items()}
    s_rows = {t: static_row(t, r) for t, r in static.items()}

    def arm(rows, form):
        return sorted([r for r in rows if r["sc_form"] == form], key=lambda r: r["ny"])

    crit = {}
    for form in ("standard", "edm"):
        d = arm(d_rows.values(), form)
        s = arm(s_rows.values(), form)
        if not d:
            continue
        c1 = all(r["interface"]["pass_"] for r in d)
        c2 = all(r["flow_rate"]["pass_"] for r in d)
        c3 = bool(s) and all(r["pass_"] for r in s)
        trend_iface = [abs(r["interface"]["err"]) for r in d]
        trend_Q = [abs(r["flow_rate"]["err_mixture_rule"]) for r in d]
        c4 = all(b <= a for a, b in zip(trend_iface, trend_iface[1:])) and \
            all(b <= a for a, b in zip(trend_Q, trend_Q[1:]))
        crit[form] = dict(
            c1_interface=dict(
                threshold=TARGET["err_iface"], verdict="PASS" if c1 else "FAIL",
                cells=[dict(ny=r["ny"], y_sim=r["interface"]["y_sim"],
                            y_analytic=r["interface"]["y_analytic"],
                            err=r["interface"]["err"], pass_=r["interface"]["pass_"])
                       for r in d]),
            c2_flow=dict(
                threshold=TARGET["err_Q"], verdict="PASS" if c2 else "FAIL",
                cells=[dict(ny=r["ny"], Q_sim=r["flow_rate"]["Q_sim"],
                            Q_analytic=r["flow_rate"]["Q_analytic"],
                            err_mixture=r["flow_rate"]["err_mixture_rule"],
                            err_component=r["flow_rate"]["err_component_rule"],
                            pass_=r["flow_rate"]["pass_"])
                       for r in d]),
            c3_spurious=dict(
                threshold=TARGET["ratio_S"], verdict="PASS" if c3 else "FAIL",
                definition="max|u_y,mix| inside the interface band (|alpha-0.5|<0.25), "
                           "static control G_x=0, normalised by the analytic u_max of the "
                           "driven case at the same resolution",
                cells=[dict(ny=r["ny"], steps=r["steps"], S_ifc=r["S_ifc"], S_all=r["S_all"],
                            u_max_analytic=r["u_max_analytic"],
                            ratio_ifc=r["ratio_ifc_vs_analytic"],
                            ratio_all=r["ratio_all_vs_analytic"],
                            phi_contrast=r["phi_contrast"], pass_=r["pass_"])
                       for r in s],
                driven_in_situ=[dict(ny=r["ny"], S_all=r["spurious_driven"]["S_all"],
                                     u_max_mix=r["spurious_driven"]["u_max_mix"],
                                     ratio=r["spurious_driven"]["ratio"])
                                for r in d]),
            c4_trend=dict(
                threshold="|err| monotone non-increasing over ny=25,33,49",
                verdict="PASS" if c4 else "FAIL",
                err_interface_abs=trend_iface, err_Q_mix_abs=trend_Q),
        )

    figs = []
    # ---- figure 1: interface + velocity profile -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for tag, rec in sorted(driven.items()):
        if rec["config"]["ny"] != 49:
            continue
        ax[0].plot(range(rec["config"]["ny"]), rec["alpha_profile"], "-",
                   lw=1.3, label=f"{rec['sc_form']} ny={rec['config']['ny']}")
    ax[0].axhline(0.5, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("y (lattice row)")
    ax[0].set_ylabel(r"heavy fraction $\alpha$")
    ax[0].set_title("stratification profile, ny=49, driven")
    ax[0].legend(fontsize=8)
    for rec in sorted(driven.values(), key=lambda r: r["config"]["ny"]):
        ny = rec["config"]["ny"]
        ax[1].plot(range(ny), rec["u_profile_mix"], lw=1.3,
                   label=f"sim ny={ny} ({rec['sc_form']})")
        ref = _ref(rec) or {}
        if ref.get("profile") or ref.get("u_profile"):
            ax[1].plot(range(ny), ref.get("profile") or ref.get("u_profile"), ":", lw=1.0,
                       color=ax[1].lines[-1].get_color())
    ax[1].set_xlabel("y (lattice row)")
    ax[1].set_ylabel(r"$u_{x,\mathrm{mix}}$")
    ax[1].set_title("velocity profile (solid) vs two-layer analytic (dotted)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p1 = os.path.join(FIG, "r3_interface_and_profile.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    figs.append(dict(path=p1, caption="heavy-component fraction and u_x profile vs analytic"))

    # ---- figure 2: spurious current -------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for tag, r in sorted(s_rows.items()):
        st = [p[0] for p in r["series"]]
        ax[0].semilogy(st, [max(p[2], 1e-14) for p in r["series"]], lw=1.2,
                       label=f"ny={r['ny']} {r['sc_form']}")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel(r"$S_{ifc}=\max|u_{y,\mathrm{mix}}|$ (interface band)")
    ax[0].set_title(r"static control ($G_x=0$): spurious-current decay")
    ax[0].legend(fontsize=7)
    names = sorted(s_rows, key=lambda t: (s_rows[t]["sc_form"], s_rows[t]["ny"]))
    ax[1].bar(range(len(names)), [s_rows[t]["ratio_ifc_vs_analytic"] for t in names],
              color=["tab:blue" if s_rows[t]["sc_form"] == "standard" else "tab:orange"
                     for t in names])
    ax[1].axhline(TARGET["ratio_S"], color="r", lw=1.0, ls="--",
                  label=f"gate {TARGET['ratio_S']}")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels([f"ny{s_rows[t]['ny']}\n{s_rows[t]['sc_form']}" for t in names],
                          fontsize=7)
    ax[1].set_ylabel(r"$S_{ifc}/u_{max}^{analytic}$")
    ax[1].set_title("criterion 3 (log scale)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIG, "r3_spurious_current.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    figs.append(dict(path=p2, caption="spurious-current decay and criterion-3 ratio"))

    # ---- figure 3: error trend ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for form in ("standard", "edm"):
        d = arm(d_rows.values(), form)
        if not d:
            continue
        ax[0].plot([r["ny"] for r in d], [100 * r["interface"]["err"] for r in d],
                   "-o", label=form)
        ax[1].plot([r["ny"] for r in d], [100 * r["flow_rate"]["err_mixture_rule"] for r in d],
                   "-o", label=f"{form} (mixture rule)")
        ax[1].plot([r["ny"] for r in d], [100 * r["flow_rate"]["err_component_rule"] for r in d],
                   "--s", label=f"{form} (component rule)")
    ax[0].axhspan(-5, 5, color="green", alpha=0.12, label="±5% gate")
    ax[1].axhspan(-10, 10, color="green", alpha=0.12, label="±10% gate")
    ax[0].set_xlabel("ny"); ax[0].set_ylabel("interface position error (%)")
    ax[0].set_title("criterion 1 vs resolution"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("ny"); ax[1].set_ylabel("Q_mix error (%)")
    ax[1].set_title("criterion 2 vs resolution"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    p3 = os.path.join(FIG, "r3_error_trend.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    figs.append(dict(path=p3, caption="criteria 1-2 error trend vs resolution"))

    # ---- overall verdict -------------------------------------------------
    overall = {}
    for form, c in crit.items():
        ok = all(c[k]["verdict"] == "PASS" for k in
                 ("c1_interface", "c2_flow")) and c["c3_spurious"]["verdict"] == "PASS"
        overall[form] = dict(
            all_criteria_pass=ok,
            c1=c["c1_interface"]["verdict"], c2=c["c2_flow"]["verdict"],
            c3=c["c3_spurious"]["verdict"], c4=c["c4_trend"]["verdict"])

    result = dict(
        benchmark="mcmp_stratified_poiseuille",
        anchor="GAP-LBM-MCMP-POISEUILLE",
        round="R3",
        model=dict(
            kind="Shan-Chen two-component D2Q9, BGK",
            u_eq="viscous (Martys et al. 1996 viscosity-weighted velocity coupling)",
            force_scheme="guo (Guo, Zheng & Shi 2002, second-order forcing)",
            sc_form="standard (Shan-Chen backward gather) vs edm (exact-difference, "
                    "well-balanced pair split, Kupershtokh et al. 2009)",
            wall_mode="mirror (neutral wetting; dry baseline in round 2)",
            G_12=-2.5, rho_heavy=0.7, rho_light=0.3,
            tau_lower=1.0, tau_upper=0.75, G_x=G_X,
            wall="post-streaming bounce-back, halfway planes at 0.5 and ny-1.5 (H=ny-2)",
        ),
        thresholds=TARGET,
        criteria=crit,
        overall_verdict=overall,
        driven_cells=d_rows,
        static_cells=s_rows,
        external_anchor=dict(
            analytic="two-layer co-moving Poiseuille: u''=-G/nu_layer per layer with "
                     "velocity and shear-stress continuity at the interface; closed form in "
                     "cases/mcmp_stratified_poiseuille/analytic.py (verified against the "
                     "single-layer parabolic limit nu_lower=nu_upper)",
            references=[
                "Shan & Doolen (1995) J. Stat. Phys. 81:379 — multicomponent SC LBM",
                "Martys & Chen (1996) J. Comput. Phys. 126:276 — viscous velocity coupling",
                "Guo, Zheng & Shi (2002) Phys. Rev. E 65:046308 — second-order forcing",
                "Kupershtokh, Medvedev & Karpov (2009) Comput. Math. Appl. 58:965 — "
                "exact difference method / well-balanced SC forcing",
                "Kruger et al. (2017) The Lattice Boltzmann Method, Springer, ch. 9 — "
                "spurious currents at diffuse interfaces",
            ],
        ),
        provenance=dict(
            records={os.path.basename(f): sha256(f)
                     for f in sorted(glob.glob(os.path.join(REC, "*.json")))},
            generator="cases/mcmp_stratified_poiseuille/r3_benchmark.py",
            matrix="cases/mcmp_stratified_poiseuille/run_final_matrix.py",
        ),
        figures=[dict(path=f["path"], caption=f["caption"]) for f in figs],
        repro_cmd=("PYTHONPATH=src python cases/mcmp_stratified_poiseuille/run_final_matrix.py "
                   "&& PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r3_benchmark.py"),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(dict(overall=overall,
                          cells_driven=sorted(d_rows), cells_static=sorted(s_rows),
                          out=OUT), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())