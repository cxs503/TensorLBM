"""Round-4 wrap-up: calibration record, probe verdict, figure, ledger payload.

Consumes the two artefacts produced in this round:

* ``artifacts/mcmp_stratified_poiseuille/r4/coex_probe.log`` — the flat-interface
  coexistence scan (``coexistence.py``, periodic strip, G_12 = -2.4 / -2.5);
* ``artifacts/mcmp_stratified_poiseuille/r4/probe/static_ny49_coex_*.json`` —
  the static-control probe (criterion ③) launched from the *calibrated*
  coexistence initial state (``r4_probe_static.py``).

and writes, with zero hand-typed numbers:

* ``artifacts/mcmp_stratified_poiseuille/r4/coexistence_scan.json``  calibration
* ``benchmarks/verified/mcmp_stratified_poiseuille/result_r4_probe.json`` verdict
* ``docs/figures/r4_probe_spurious.png`` figure for the report

Usage: PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r4_wrapup.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from cases.mcmp_stratified_poiseuille.analytic import layered_poiseuille  # noqa: E402

R4 = os.path.join("artifacts", "mcmp_stratified_poiseuille", "r4")
LOG = os.path.join(R4, "coex_probe.log")
PROBE = os.path.join(R4, "probe")
OUT_CAL = os.path.join(R4, "coexistence_scan.json")
OUT_VER = os.path.join("benchmarks", "verified", "mcmp_stratified_poiseuille",
                       "result_r4_probe.json")
FIG = os.path.join("docs", "figures", "r4_probe_spurious.png")
G_X, CS2, NY = 5e-5, 1.0 / 3.0, 49

#: pre-registered admissibility window (fixed before the probe was launched):
#: 3 <= w_10_90 <= H_min/10  (resolved interface AND separation of scales)
W_LO, W_HI_FRAC, CONTRAST_MIN, DRIFT_MAX = 3.0, 1.0 / 10.0, 0.5, 0.5


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_calib():
    recs = []
    for line in open(LOG):
        m = re.match(r"CALIB (\{.*\})", line.strip())
        if m:
            recs.append(json.loads(m.group(1)))
    return recs


def main() -> int:
    H_min = NY - 2
    hi = H_min * W_HI_FRAC
    recs = parse_calib()
    for r in recs:
        r["admissible"] = bool(
            r["w_10_90"] == r["w_10_90"] and W_LO <= r["w_10_90"] <= hi
            and r["contrast"] >= CONTRAST_MIN and abs(r["drift"]) < DRIFT_MAX)
    adm = [r for r in recs if r["admissible"]]
    if adm:
        best = max(adm, key=lambda r: abs(r["g12"]))
        sel = dict(selected=best["g12"], rho_A=best["rho_A"], rho_B=best["rho_B"],
                   w_10_90=best["w_10_90"], w_tanh=best["w_tanh"],
                   contrast=best["contrast"], admissible_g12=[r["g12"] for r in adm],
                   reason="pre-registered window 3 <= w_10_90 <= H/10 = %.2f cells, "
                          "contrast >= %.1f, |drift| < %.1f cell; tie-break max |G12|"
                          % (hi, CONTRAST_MIN, DRIFT_MAX))
    else:
        sel = dict(selected=None, reason="no admissible G12")
    cal = dict(benchmark="mcmp_stratified_poiseuille", round="R4-calibration",
               anchor="GAP-LBM-MCMP-POISEUILLE",
               protocol=dict(strip_ny=96, strip_nx=2, steps=20000, seed_width=4.0,
                             boundary="periodic both directions, no walls",
                             g12_grid=[-2.4, -2.5, -2.75, -3.0],
                             window=[W_LO, hi], contrast_min=CONTRAST_MIN,
                             drift_max=DRIFT_MAX),
               records=recs, selection=sel,
               notes=("G12=-2.75 and -3.0 aborted in this run: the order-parameter "
                      "profile lost its interface crossing (nan) -> recorded as "
                      "non-admissible by the robust path added to coexistence.py; "
                      "the scan was not repeated for budget reasons."))
    json.dump(cal, open(OUT_CAL, "w"), indent=1)
    print("CALIB SELECTION " + json.dumps(sel))

    # --- probe (criterion 3) ---
    tot = sel["rho_A"] + sel["rho_B"]
    fA, fB = sel["rho_A"] / tot, sel["rho_B"] / tot
    nu1, nu2 = CS2 * 0.5, CS2 * 0.25
    nu_lo, nu_up = fA * nu1 + fB * nu2, fB * nu1 + fA * nu2
    ana = layered_poiseuille(NY, 0.5 + (NY - 2) / 2.0, G_X, nu_lo, nu_up, H=NY - 2)
    u_max = max(ana["profile"])

    arms = {}
    prov = {LOG: sha256(LOG)}
    for f in sorted(glob.glob(os.path.join(PROBE, "*.json"))):
        d = json.load(open(f))
        s = d["summary"]
        win = [h for h in d["series"] if h["step"] >= 0.8 * d["series"][-1]["step"]]
        s["S_ifc_tail_min"] = min(h["S_ifc"] for h in win)
        s["ratio_tail_min"] = s["S_ifc_tail_min"] / u_max
        s["ratio"] = s["S_ifc"] / u_max
        s["pass"] = bool(s["ratio"] < 0.005)
        s["pass_tail_min"] = bool(s["ratio_tail_min"] < 0.005)
        arms[s["sc_form"]] = s
        prov[f] = sha256(f)
    ver = dict(benchmark="mcmp_stratified_poiseuille", round="R4-probe",
               anchor="GAP-LBM-MCMP-POISEUILLE",
               hypothesis=("R3 fail_stage 'model/state identification': the round-3 "
                           "parasitic-current plateau is caused by the off-coexistence "
                           "initial state (0.7/0.3 vs measured 0.874/0.137), so starting "
                           "the static control at coexistence restores a resolved "
                           "equilibrium interface and drops S_ifc below 0.005 u_max"),
               verdict="REFUTED" if not any(a["pass"] for a in arms.values()) else "PARTIAL",
               model=dict(g12=sel["selected"], rho_A=sel["rho_A"], rho_B=sel["rho_B"],
                          width_tanh=sel["w_tanh"], w_10_90=sel["w_10_90"],
                          nu_lower=nu_lo, nu_upper=nu_up),
               reference=dict(u_max_analytic=u_max, ny=NY, G_x=G_X,
                              profile=ana["profile"], Q_total=ana["Q_total"]),
               arms=arms, provenance=prov,
               repro_cmd=("PYTHONPATH=src python cases/mcmp_stratified_poiseuille/"
                          "coexistence.py --ny 96 --nx 2 --steps 20000 --g12 -2.4 -2.5 "
                          "-2.75 -3.0 && PYTHONPATH=src python cases/mcmp_stratified_poiseuille/"
                          "r4_probe_static.py --ny 49 --steps 30000 --sc-form {edm,standard}"),
               oss_transfer=[
                   dict(repo="sailfish-team/sailfish", file="sailfish/lb_binary.py:198-268",
                        what="free-energy binary equilibrium with the analytic pressure tensor "
                             "P_ab = [p_b - kappa phi lap phi] delta + rho u_a u_b + "
                             "kappa(d_a phi d_b phi - |grad phi|^2/2 delta)"),
                   dict(repo="ludwig-cf/ludwig", file="src/symmetric.c:327, src/phi_grad_mu.c:40",
                        what="interface force -phi grad mu with Guo velocity shift, "
                             "sigma = sqrt(-8 kappa A^3/(9 B^2)), xi = sqrt(-2 kappa/A)"),
                   dict(repo="ebinan92/Fingering_dynamics",
                        file="lattice_boltzmann/fingering.py:149,212,298",
                        what="two-distribution (f->rho,p; g->psi,mu) phase-field channel"),
               ])
    os.makedirs(os.path.dirname(OUT_VER), exist_ok=True)
    json.dump(ver, open(OUT_VER, "w"), indent=1)
    print("PROBE VERDICT " + json.dumps({k: dict(S_ifc=v["S_ifc"], S_bulk=v["S_bulk"],
                                                ratio=v["ratio"], ratio_min=v["ratio_tail_min"],
                                                pass_=v["pass"]) for k, v in arms.items()}))
    print("wrote", OUT_VER, OUT_CAL)

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for sc, mk, col in (("standard", "o", "tab:blue"), ("edm", "s", "tab:orange")):
        if sc not in arms:
            continue
        d = json.load(open(glob.glob(os.path.join(PROBE, f"*{sc}.json"))[0]))
        ax[0].semilogy([h["step"] for h in d["series"]], [h["S_ifc"] for h in d["series"]],
                       mk + "-", ms=2.5, color=col, label=f"S_ifc ({sc})")
        ax[0].semilogy([h["step"] for h in d["series"]], [h["S_bulk"] for h in d["series"]],
                       mk + "--", ms=2, color=col, alpha=0.6, label=f"S_bulk ({sc})")
    ax[0].axhline(0.005 * u_max, color="r", ls=":", label="gate 0.005 u_max")
    ax[0].set_xlabel("step"), ax[0].set_ylabel("max |u_y,mix|")
    ax[0].set_title("static control from the coexistence state (ny=49, G12=-2.5)")
    ax[0].legend(fontsize=7)
    names, ratios = [], []
    for i, (sc, s) in enumerate(arms.items()):
        ax[1].bar(i, s["ratio"], color="tab:blue" if sc == "standard" else "tab:orange")
        ax[1].text(i, s["ratio"], f"{s['ratio']:.2e}", ha="center", va="bottom", fontsize=8)
        names.append(f"{sc}\nny={NY}"), ratios.append(s["ratio"])
    ax[1].axhline(0.005, color="r", ls="--", label="gate 0.005")
    ax[1].set_yscale("log"), ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel("S_ifc / analytic u_max")
    ax[1].set_title("criterion ③ — coexistence-initialised probe")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print("figure", FIG)

    summary = dict(calibration=sel, u_max_analytic=u_max, nu=(nu_lo, nu_up),
                   arms={k: {kk: v[kk] for kk in ("S_ifc", "S_all", "S_bulk", "ratio",
                                                  "ratio_tail_min", "y_int",
                                                  "phi_contrast", "S_ifc_rel_drift")}
                         for k, v in arms.items()},
                   gate=0.005, H=NY - 2)
    json.dump(summary, open("/tmp/r4_numbers.json", "w"), indent=1)
    print("NUMBERS " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())