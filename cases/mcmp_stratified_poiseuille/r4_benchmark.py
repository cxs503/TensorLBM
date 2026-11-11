"""Round-4 assembler: criteria ①-④ from the cells + calibration -> result_r4.json.

Zero hand-typed numbers: every value comes from
``artifacts/mcmp_stratified_poiseuille/r4/`` (calibration + per-cell records),
with per-file sha256 in ``provenance``.

Criterion definitions (unchanged from rounds 1-3; only the model state changed):

1. steady interface position vs the analytic half-height: |err| < 5 %
2. two-phase volume flow rate: |err(Q_mix)| < 10 %
3. interface spurious current: max|u_y,mix| in the interface band (static
   control, G_x=0, last sample of the run) divided by the analytic u_max of the
   driven case at the same resolution: ratio < 0.005
4. resolution trend over ny = {49, 65} (monotone decrease of |err| for ①②, and
   ③ satisfied at both resolutions)

Usage: PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/r4_benchmark.py
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

R4 = os.path.join("artifacts", "mcmp_stratified_poiseuille", "r4")
CELLS = os.path.join(R4, "cells")
CALIB = os.path.join(R4, "coexistence_scan.json")
OUT = os.path.join("benchmarks", "verified", "mcmp_stratified_poiseuille", "result_r4.json")
FIG = os.path.join("docs", "figures")
G_X = 5e-5
CS2 = 1.0 / 3.0
TARGET = dict(err_iface=0.05, err_Q=0.10, ratio_S=0.005)
NY = (49, 65)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_nus(model):
    """Layer viscosities from the *measured* coexisting fractions."""
    nu1, nu2 = CS2 * (1.0 - 0.5), CS2 * (0.75 - 0.5)
    tot = model["rho_A"] + model["rho_B"]
    fA, fB = model["rho_A"] / tot, model["rho_B"] / tot
    return fA * nu1 + fB * nu2, fB * nu1 + fA * nu2


def analytic(ny, model):
    nu_lo, nu_up = layer_nus(model)
    return layered_poiseuille(ny, 0.5 + (ny - 2) / 2.0, G_X, nu_lo, nu_up, H=ny - 2), (nu_lo, nu_up)


def load():
    driven, static = {}, {}
    for f in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        rec = json.load(open(f))
        cell = rec["cell"]
        (driven if cell["kind"] == "driven" else static)[(cell["ny"], cell["sc_form"])] = (rec, f)
    return driven, static


def main() -> int:
    cal = json.load(open(CALIB))
    model = dict(cal["selection"])
    driven, static = load()
    prov = {os.path.relpath(f, "."): sha256(f) for _, f in
            list(driven.values()) + list(static.values())}
    prov[os.path.relpath(CALIB, ".")] = sha256(CALIB)

    out = dict(benchmark="mcmp_stratified_poiseuille", round="R4",
               anchor="GAP-LBM-MCMP-POISEUILLE",
               external_anchor="Huang, Krafczyk & Lu 2011 (MCMP analytic); "
                               "Shan & Chen 1994 (coexistence); Kupershtokh et al. 2009 (EDM)",
               model=dict(kind="Shan-Chen two-component D2Q9, BGK, Guo forcing",
                          g12=model["selected"], rho_A=model["rho_A"], rho_B=model["rho_B"],
                          interface_width_tanh=model["w_tanh"],
                          initial_state="coexistence calibration (coexistence.py)",
                          wall="halfway bounce-back, H=ny-2, mirror neutral wetting",
                          G_x=G_X, tau_lower=1.0, tau_upper=0.75),
               thresholds=TARGET, cells={}, crit={}, provenance=prov,
               repro_cmd=("PYTHONPATH=src python cases/mcmp_stratified_poiseuille/"
                          "coexistence.py --out %s && PYTHONPATH=src python cases/"
                          "mcmp_stratified_poiseuille/r4_matrix.py && PYTHONPATH=src python "
                          "cases/mcmp_stratified_poiseuille/r4_benchmark.py" % CALIB))

    iface, flow, spur = {}, {}, {}
    for ny in NY:
        ana, nus = analytic(ny, model)
        y_ref = 0.5 + (ny - 2) / 2.0
        u_max_ref = max(ana["profile"])
        for sc in ("standard", "edm"):
            rec, _ = driven[(ny, sc)]
            y = rec["interface_crossing"]
            iface.setdefault(sc, []).append(dict(
                ny=ny, y_sim=y, y_analytic=y_ref, err=(y - y_ref) / y_ref,
                pass_=abs((y - y_ref) / y_ref) < TARGET["err_iface"],
                Q_state=rec.get("steady"), phi_contrast=rec.get("phi_contrast")))
            flow.setdefault(sc, []).append(dict(
                ny=ny, Q_sim=rec["Q_mix"], Q_analytic=ana["Q_total"],
                err=(rec["Q_mix"] - ana["Q_total"]) / ana["Q_total"],
                pass_=abs((rec["Q_mix"] - ana["Q_total"]) / ana["Q_total"]) < TARGET["err_Q"],
                nu_lower=nus[0], nu_upper=nus[1]))
            srec, _ = static[(ny, sc)]
            last = srec["series"][-1]
            ratio = last["S_ifc"] / u_max_ref
            spur.setdefault(sc, []).append(dict(
                ny=ny, steps=last["step"], S_ifc=last["S_ifc"], S_all=last["S_all"],
                u_max_analytic=u_max_ref, ratio=ratio,
                pass_=ratio < TARGET["ratio_S"], y_int=last["y_int"],
                phi_contrast=last["phi_contrast"],
                S_ifc_rel_drift=(max(h["S_ifc"] for h in srec["series"]
                                     if h["step"] >= 0.8 * last["step"]) -
                                 min(h["S_ifc"] for h in srec["series"]
                                     if h["step"] >= 0.8 * last["step"])) /
                                max(min(h["S_ifc"] for h in srec["series"]
                                        if h["step"] >= 0.8 * last["step"]), 1e-30)))

    def monotone(vals):
        a = [abs(v) for v in vals]
        return all(a[i + 1] < a[i] for i in range(len(a) - 1))

    for sc in ("standard", "edm"):
        c1 = iface[sc]
        c2 = flow[sc]
        c3 = spur[sc]
        out["crit"][sc] = dict(
            c1_interface=dict(threshold=TARGET["err_iface"], cells=c1,
                              verdict="PASS" if all(c["pass_"] for c in c1) else "FAIL"),
            c2_flow=dict(threshold=TARGET["err_Q"], cells=c2,
                         verdict="PASS" if all(c["pass_"] for c in c2) else "FAIL"),
            c3_spurious=dict(threshold=TARGET["ratio_S"], cells=c3,
                             verdict="PASS" if all(c["pass_"] for c in c3) else "FAIL",
                             definition="max|u_y,mix| in interface band |alpha-0.5|<0.25, "
                                        "static control, last sample, / analytic u_max"),
            c4_trend=dict(
                iface_monotone=monotone([c["err"] for c in c1]),
                flow_monotone=monotone([c["err"] for c in c2]),
                spurious_gate_both=all(c["pass_"] for c in c3),
                verdict="PASS" if (monotone([c["err"] for c in c1]) and
                                   monotone([c["err"] for c in c2]) and
                                   all(c["pass_"] for c in c3)) else "FAIL"),
        )
        out["cells"][sc] = dict(iface=c1, flow=c2, spurious=c3)

    out["verdict"] = {sc: dict(**{k: v["verdict"] for k, v in out["crit"][sc].items()})
                      for sc in out["crit"]}
    # comparison with the round-3 baseline (same model, nominal 0.7/0.3 initial state)
    r3p = os.path.join("benchmarks", "verified", "mcmp_stratified_poiseuille", "result_r3.json")
    if os.path.exists(r3p):
        r3 = json.load(open(r3p))
        base = {}
        for sc in ("standard", "edm"):
            try:
                c3 = [c for c in r3["criteria"][sc]["c3_spurious"]["cells"] if c["ny"] == 49][0]
                c1 = [c for c in r3["criteria"][sc]["c1_interface"]["cells"] if c["ny"] == 49][0]
                c2 = [c for c in r3["criteria"][sc]["c2_flow"]["cells"] if c["ny"] == 49][0]
                base[sc] = dict(S_ifc=c3["S_ifc"], ratio=c3["ratio_ifc"],
                                y_err=c1["err"], Q_err=c2["err_mixture"])
            except Exception:
                pass
        out["baseline_r3_ny49"] = base
        prov[os.path.relpath(r3p, ".")] = sha256(r3p)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("VERDICT " + json.dumps(out["verdict"]))
    print("wrote", OUT)

    # ---------------- figures ----------------
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sc, mk in (("standard", "o"), ("edm", "s")):
        for ny in NY:
            rec, _ = driven[(ny, sc)]
            ana, _ = analytic(ny, model)
            ys = list(range(ny))
            axes[0].plot(rec["u_profile_mix"], ys, mk + "-", ms=3, lw=0.9,
                         label=f"ny={ny} {sc}")
            axes[1].plot(ana["profile"], ys, "k--", lw=0.8)
            axes[1].plot(rec["u_profile_mix"], ys, mk, ms=2.5, label=f"ny={ny} {sc}")
    axes[0].set_xlabel("u_x,mix"), axes[0].set_ylabel("y (lattice)")
    axes[0].set_title("driven two-layer Poiseuille: profile"), axes[0].legend(fontsize=7)
    axes[1].set_xlabel("u_x,mix"), axes[1].set_ylabel("y (lattice)")
    axes[1].set_title("simulation vs closed-form two-layer solution")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "r4_profiles.png"), dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for sc, mk in (("standard", "o"), ("edm", "s")):
        for ny in NY:
            srec, _ = static[(ny, sc)]
            ax[0].semilogy([h["step"] for h in srec["series"]],
                           [h["S_ifc"] for h in srec["series"]], mk + "-", ms=2.5,
                           label=f"ny={ny} {sc}")
    ax[0].set_xlabel("step"), ax[0].set_ylabel("S_ifc = max|u_y,mix| in interface band")
    ax[0].set_title("static control: parasitic current decay"), ax[0].legend(fontsize=7)
    xs = []
    for i, (ny, sc) in enumerate([(n, s) for s in ("standard", "edm") for n in NY]):
        c = [c for c in spur[sc] if c["ny"] == ny][0]
        ax[1].bar(i, c["ratio"], color="tab:blue" if sc == "standard" else "tab:orange")
        xs.append((i, f"{sc[:3]}\nny={ny}"))
        ax[1].text(i, c["ratio"], f"{c['ratio']:.2e}", ha="center", va="bottom", fontsize=7)
    ax[1].axhline(TARGET["ratio_S"], color="r", ls="--", label="gate 0.005")
    ax[1].set_yscale("log")
    ax[1].set_xticks([x for x, _ in xs]), ax[1].set_xticklabels([l for _, l in xs], fontsize=7)
    ax[1].set_ylabel("S_ifc / analytic u_max"), ax[1].set_title("criterion ③ per cell")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "r4_spurious_current.png"), dpi=130)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())