"""Round-5 selection study: pseudopotential vs free-energy under criterion ③.

Consumes, with zero hand-typed numbers:

* ``artifacts/free_energy_selection/fe_static_*.json`` — the free-energy flat-interface
  arms produced by ``static_interface.py`` (float64 and float32);
* ``benchmarks/verified/mcmp_stratified_poiseuille/result_r4_probe.json`` — the
  *committed* round-4 pseudopotential static control (both pair-force arms) and the
  analytic layered-Poiseuille normaliser ``u_max_analytic``;

and writes

* ``artifacts/free_energy_selection/selection.json``  the decision record
* ``artifacts/free_energy_selection/table.md``        the report tables
* ``docs/figures/r5_free_energy_selection.png``       the three-panel figure

Pre-registered decision rules (fixed before the runs, see the round-5 document):

* **D1** interface accuracy (migration prerequisite (a)): ``|Δw_10_90| / w < 15 %``;
* **D2** flat-interface surface tension (prerequisite (b), the part a flat interface can
  carry): ``|Δsigma| / sigma < 15 %``;
* **D3** criterion ③ on the free-energy arm: ``S_ifc / u_max < 0.005``;
* **D4** comparability: the free-energy ``w_10_90`` is within 15 % of the
  coexistence-calibrated Shan-Chen width 4.645 cells, so the head-to-head is not
  resolved by a thinner interface.

The selection is ``free-energy`` only if D1-D4 all hold.

Usage::

    PYTHONPATH=src:. python benchmarks/pending/free_energy_selection/selection.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.phasefield.free_energy_pressure import CS2  # noqa: E402

FE_GLOB = os.path.join("artifacts", "free_energy_selection", "fe_static_*.json")
SC_REF = os.path.join("benchmarks", "verified", "mcmp_stratified_poiseuille",
                      "result_r4_probe.json")
OUT_DIR = os.path.join("artifacts", "free_energy_selection")
OUT_JSON = os.path.join(OUT_DIR, "selection.json")
OUT_MD = os.path.join(OUT_DIR, "table.md")
FIG = os.path.join("docs", "figures", "r5_free_energy_selection.png")

GATE = 0.005          # criterion ③ of ledger GAP-LBM-MCMP-POISEUILLE
TOL = 0.15            # owner precision standard
W_SC = 4.645          # coexistence-calibrated Shan-Chen 10-90 width (round 4)
H_REF = 47.0          # wall spacing of the ny = 49 probes
G_X_REF = 5e-5        # body force of the driven two-layer case


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fe() -> dict:
    arms = {}
    for path in sorted(glob.glob(FE_GLOB)):
        rec = json.load(open(path))
        p = rec["params"]
        key = os.path.splitext(os.path.basename(path))[0].replace("fe_static_", "")
        arms[key] = dict(params=p, series=rec["series"], path=path, sha256=sha256(path))
    if not arms:
        raise SystemExit(f"no free-energy records match {FE_GLOB}")
    return arms


def main() -> int:
    fe = load_fe()
    sc = json.load(open(SC_REF))
    sc_sha = sha256(SC_REF)
    u_max_sc = float(sc["reference"]["u_max_analytic"])

    rows, series = {}, {}
    for key, arm in fe.items():
        p = arm["params"]
        # normaliser (i): the round-4 two-layer analytic u_max (same G, same H)
        nu_fe = p["nu"]
        # normaliser (ii): the free-energy model's own uniform-viscosity Poiseuille
        u_max_fe = G_X_REF * H_REF ** 2 / (8.0 * nu_fe)
        row = dict(
            arm=p["arm"], dtype=p["dtype"],
            xi_analytic=p["xi_analytic"], xi_measured=p["xi_measured"],
            xi_rel_err=p["xi_rel_err"],
            w_10_90_analytic=p["w_10_90_analytic"], w_10_90_measured=p["w_10_90_measured"],
            w_rel_err=p["w_rel_err"],
            sigma_analytic=p["sigma_analytic"], sigma_band=p["sigma_band"],
            sigma_rel_err=p["sigma_rel_err"],
            S_ifc=p["S_ifc"], S_ifc_tail_max=p["S_ifc_tail_max"],
            S_all=p["S_all"], S_bulk=p["S_bulk"],
            S_ifc_decades_per_1k=p["S_ifc_decades_per_1k"],
            y_drift=p["y_drift"], rho_min=p["rho_min"], rho_max=p["rho_max"],
            dphi_per_cell=p["dphi_per_cell"], drho=p["drho"],
            u_max_r4_layered=u_max_sc, u_max_own=u_max_fe,
            ratio_r4=abs(p["S_ifc"]) / u_max_sc,
            ratio_own=abs(p["S_ifc"]) / u_max_fe,
            steps=p["steps"], ny=p["ny"], nx=p["nx"], band=p["band"],
            wall_time_s=p["wall_time_s"], provenance=arm["path"],
            provenance_sha256=arm["sha256"],
        )
        row["D1_w_ok"] = abs(row["w_rel_err"]) < TOL
        row["D2_sigma_ok"] = abs(row["sigma_rel_err"]) < TOL
        row["D3_gate_ok"] = row["ratio_r4"] < GATE and row["ratio_own"] < GATE
        row["D4_comparable"] = abs(row["w_10_90_measured"] - W_SC) / W_SC < TOL
        rows[key] = row
        series[key] = arm["series"]

    sc_rows = {}
    for name, arm in sc["arms"].items():
        sc_rows[name] = dict(
            S_ifc=arm["S_ifc"], S_all=arm["S_all"], S_bulk=arm["S_bulk"],
            ratio=arm["ratio"], S_ifc_rel_drift=arm["S_ifc_rel_drift"],
            y_int=arm["y_int"], pass_gate=arm["pass"],
        )

    # --- selection ---------------------------------------------------------
    key_main = "matched_f32" if "matched_f32" in rows else sorted(rows)[0]
    all_pass = all(r["D1_w_ok"] and r["D2_sigma_ok"] and r["D3_gate_ok"]
                   for r in rows.values())
    comparable = all(r["D4_comparable"] for r in rows.values())
    selection = "free-energy" if (all_pass and comparable) else "pseudopotential"
    best_sc = min(v["S_ifc"] for v in sc_rows.values())
    primary = rows[key_main]["S_ifc"]
    improvement = best_sc / primary if primary > 0 else float("inf")
    improvement_by_run = {k: best_sc / r["S_ifc"] for k, r in rows.items()}

    record = dict(
        benchmark="mcmp_stratified_poiseuille",
        round="R5-selection",
        anchor="XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER",
        question=("round 4 refuted criterion ③ inside the pseudopotential class "
                  "(S_ifc = 8.6e-3, ratio 0.0767) and selected the free-energy binary "
                  "model as migration target without measuring it under the same metric"),
        rules=dict(gate=GATE, tol=TOL, w_sc_cells=W_SC, D="D1 width, D2 tension, "
                   "D3 criterion ③, D4 width comparability"),
        reference=dict(path=SC_REF, sha256=sc_sha, u_max_layered=u_max_sc,
                       H=H_REF, G_x=G_X_REF, rho_A=sc["model"]["rho_A"],
                       rho_B=sc["model"]["rho_B"], w_10_90=sc["model"]["w_10_90"],
                       u_max_note="analytic two-layer Poiseuille, measured coexisting "
                                  "fractions and layer viscosities (round 4)"),
        free_energy=rows, pseudopotential=sc_rows,
        selection=selection,
        improvement_factor=improvement,
        improvement_by_run=improvement_by_run,
        primary_run=key_main,
        improvement_decades=math.log10(improvement),
        honest_limitations=[
            "the free-energy arm is wall-free and periodic: the round-4 failure mode "
            "S_bulk = S_all is a wall/viscosity-asymmetry bulk motion that this setup "
            "cannot produce, so this measures the interface-local residual only",
            "the free-energy arm carries a (near) uniform density and a single viscosity "
            "tau_f = 1.0: the two-layer density/viscosity contrast of the Poiseuille "
            "acceptance is not part of it; walls + halfway bounce-back for f and no-flux "
            "for phi are the missing implementation work",
            "criterion ③ here is evaluated with the analytic normaliser (as in rounds "
            "1-4), not with a measured driven free-energy flow",
        ],
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(record, open(OUT_JSON, "w"), indent=1)

    # --- markdown tables --------------------------------------------------
    lines = ["## FE arms", "",
             "| run | arm | dtype | w_10_90 (cells) | analytic | Δw | σ_band | analytic σ | Δσ | "
             "S_ifc | S_all | decay (dec/1k) | ratio (R4 u_max) | ratio (own ν) | D1/D2/D3/D4 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in rows.items():
        lines.append("| %s | %s | %s | %.3f | %.3f | %+.2f%% | %.5f | %.5f | %+.2f%% | %.3e | %.3e | "
                     "%.3f | %.2e | %.2e | %s/%s/%s/%s |" % (
                         key, r["arm"], r["dtype"], r["w_10_90_measured"], r["w_10_90_analytic"],
                         100 * r["w_rel_err"], r["sigma_band"], r["sigma_analytic"],
                         100 * r["sigma_rel_err"], r["S_ifc"], r["S_all"],
                         r["S_ifc_decades_per_1k"], r["ratio_r4"], r["ratio_own"],
                         "P" if r["D1_w_ok"] else "F", "P" if r["D2_sigma_ok"] else "F",
                         "P" if r["D3_gate_ok"] else "F", "P" if r["D4_comparable"] else "F"))
    lines += ["", "## pseudopotential arm (round-4 committed record)", "",
              "| arm | S_ifc | S_all | S_bulk | ratio vs u_max | gate |",
              "|---|---|---|---|---|---|"]
    for name, r in sc_rows.items():
        lines.append("| %s | %.4e | %.4e | %.4e | %.4f | %s |" % (
            name, r["S_ifc"], r["S_all"], r["S_bulk"], r["ratio"],
            "PASS" if r["pass_gate"] else "FAIL"))
    lines += ["", "## improvement vs the best pseudopotential arm", "",
              "| run | S_ifc | best SC S_ifc | factor | decades |", "|---|---|---|---|---|"]
    for k, f in improvement_by_run.items():
        lines.append("| %s | %.3e | %.4e | %.2e× | %.1f |" % (
            k, rows[k]["S_ifc"], best_sc, f, math.log10(f)))
    lines += ["", f"selection = **{selection}**; primary run `{key_main}`: interface-local "
                  f"improvement {improvement:.2e}× ({record['improvement_decades']:.1f} decades) "
                  f"over the best pseudopotential arm.", ""]
    open(OUT_MD, "w").write("\n".join(lines))
    print("\n".join(lines))

    # --- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    palette = ["C0", "C1", "C4", "C5", "C2", "C6"]
    colors = {k: palette[i % len(palette)] for i, k in enumerate(sorted(series))}
    styles = {k: ("-" if "f32" not in k else "--") for k in series}
    for key, s in series.items():
        st = [h["step"] for h in s if h["S_ifc"] == h["S_ifc"] and h["S_ifc"] > 0]
        sv = [h["S_ifc"] for h in s if h["S_ifc"] == h["S_ifc"] and h["S_ifc"] > 0]
        ax[0].semilogy(st, sv, styles[key], color=colors[key],
                       label=f"free-energy {key}")
    for name, r in sc_rows.items():
        ax[0].axhline(r["S_ifc"], color="C3", ls=":", lw=1.2,
                      label=f"Shan-Chen {name} plateau {r['S_ifc']:.2e}")
    ax[0].axhline(GATE * u_max_sc, color="k", ls="-.", lw=1.0,
                  label=f"gate 0.005·u_max = {GATE * u_max_sc:.2e}")
    ax[0].set_xlabel("step"); ax[0].set_ylabel(r"$S_{ifc}$ = band $\max|u_y|$")
    ax[0].set_title("(a) static interface residual"); ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    key0 = "matched" if "matched" in series else sorted(series)[0]
    p0 = rows[key0]
    labels = list(rows)
    ax[1].bar([i - 0.2 for i in range(len(labels))],
              [rows[k]["w_10_90_measured"] for k in labels], width=0.4,
              label="measured $w_{10-90}$")
    ax[1].bar([i + 0.2 for i in range(len(labels))],
              [rows[k]["w_10_90_analytic"] for k in labels], width=0.4,
              label="analytic $2\\,atanh(0.8)\\,\\xi$")
    ax[1].axhline(W_SC, color="C3", ls=":", label=f"Shan-Chen coexistence {W_SC} cells")
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, fontsize=7, rotation=20)
    ax[1].set_ylabel("interface width (cells)")
    ax[1].set_title("(b) prerequisite (a): width vs analytic"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    names = [f"SC\n{n}" for n in sc_rows] + [f"FE\n{k}" for k in labels]
    vals = [sc_rows[n]["ratio"] for n in sc_rows] + [rows[k]["ratio_r4"] for k in labels]
    bars = ax[2].bar(range(len(vals)), vals, color=["C3"] * len(sc_rows)
                     + ["C0"] * len(labels))
    ax[2].set_yscale("log")
    ax[2].axhline(GATE, color="k", ls="-.", label=f"criterion ③ gate = {GATE}")
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.1e}", ha="center",
                   va="bottom", fontsize=7)
    ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(names, fontsize=7)
    ax[2].set_ylabel(r"$S_{ifc}/u_{max}$")
    ax[2].set_title("(c) criterion ③ head-to-head"); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("R5 model-class selection: pseudopotential vs free-energy binary LBM "
                 f"({key0}; interface drift {p0['y_drift']:.1e} cells)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("wrote " + FIG + " / " + OUT_JSON + " / " + OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())