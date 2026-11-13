"""Analysis + figures for XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING.

Reads ``artifacts/result.json`` (written by ``run_xval.py``), builds

* ``artifacts/table.csv``   - one row per (workload, arm, configuration)
* ``artifacts/tables.md``   - the markdown tables the report embeds
* ``figures/*.png``         - deviation bars, velocity profiles, momentum ledger
* ``artifacts/analysis.json`` - the same numbers machine-readably, plus the
  predeclared verdict (ACCEPT / REJECT / CONTROL-FAILED) per comparison.

No number in the report is typed by hand: it is rendered from these files.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cases" / "mcmp_stratified_poiseuille"))

ART = HERE / "artifacts"
FIG = HERE / "figures"

from analytic import layered_poiseuille, layer_viscosities  # noqa: E402

G_X = 5e-5
OWNER_TOL = 0.15   # owner precision standard: > 15% is not acceptable
STRICT = 0.03      # in-house strict bound for the analytic Poiseuille comparison
COLOURS = {
    "GOT-MIX": "#1f77b4", "GOT-SELF": "#2ca02c", "LBM4": "#d62728",
    "LBM4-TAU": "#ff7f0e", "LBM3": "#9467bd", "CTRL-NORM": "#7f7f7f",
    "analytic": "#111111",
}


# ---------------------------------------------------------------------------
def rel(a: float, b: float) -> float:
    """Relative deviation |a-b|/|b|; non-finite inputs give NaN, never 0."""
    if a is None or b is None:
        return float("nan")
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0.0:
        return float("nan")
    return abs(a - b) / abs(b)


def verdict(dev: float, tol: float = OWNER_TOL) -> str:
    if dev != dev:
        return "NO-DATA"
    return "PASS" if dev <= tol else "FAIL"


def find_run(block: dict, arm: str, **match) -> dict | None:
    for r in block.get("runs", []):
        if r.get("arm") != arm:
            continue
        ok = True
        for k, v in match.items():
            got = r.get(k, (r.get("config") or {}).get(k))
            if isinstance(v, float):
                ok = ok and got is not None and abs(float(got) - v) < 1e-12
            else:
                ok = ok and got == v
        if ok:
            return r
    return None


# ---------------------------------------------------------------------------
# W2: analytic reference at the same discretisation
# ---------------------------------------------------------------------------
def analytic_for(ny: int, tau_upper: float, G_12_unused: float = -2.5) -> dict:
    nu_l, nu_u = layer_viscosities(1.0, tau_upper, rule="mixture")
    H = float(ny - 2)  # halfway wall planes: declared convention of the shipped case
    ana = layered_poiseuille(ny, ny // 2, G_X, nu_l, nu_u, H=H)
    band = ana["profile"][1:-1]
    return {
        "profile": ana["profile"],
        "nu_lower": nu_l, "nu_upper": nu_u, "H": H,
        "u_max_band": max(band),
        "Q_band": sum(band),
        "u_max_full": max(ana["profile"]),
        "Q_total": ana["Q_total"],
    }


def poiseuille_errors(run: dict) -> dict:
    cfg = run["config"]
    ny = cfg["ny"]
    ana = analytic_for(ny, cfg["tau_upper"])
    sim = run["u_profile_mix"]
    band_sim = sim[1:-1]
    band_ana = ana["profile"][1:-1]
    num = sum((a - b) ** 2 for a, b in zip(band_sim, band_ana))
    den = sum(b * b for b in band_ana)
    return {
        "u_max_sim": max(abs(v) for v in band_sim),
        "u_max_ana": ana["u_max_band"],
        "err_u_max": rel(max(abs(v) for v in band_sim), ana["u_max_band"]),
        "Q_sim_band": sum(band_sim),
        "Q_ana_band": ana["Q_band"],
        "err_Q": rel(sum(band_sim), ana["Q_band"]),
        "err_profile_L2": math.sqrt(num / den) if den else float("nan"),
        "nu_lower": ana["nu_lower"], "nu_upper": ana["nu_upper"], "H": ana["H"],
        "analytic_profile": ana["profile"],
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def build(data: dict) -> dict:
    w = data["workloads"]
    out: dict[str, Any] = {"rows": [], "tables": {}, "figures": {}}

    # ---------------- W1 static droplet ----------------
    rows_w1 = []
    for block_name in ("W1_droplet_coupling", "W1_droplet_radius"):
        for r in w.get(block_name, {}).get("runs", []):
            if r.get("status") == "SKIPPED":
                rows_w1.append({"block": block_name, "arm": r["arm"], "G_12": None, "R_init": None,
                                "u_max": None, "converged": None, "rho1_centre": None,
                                "rho1_corner": None, "dp": None, "R_meas": None, "sigma": None,
                                "wall_s": None, "note": r.get("reason", "SKIPPED")})
                continue
            lap = r.get("laplace") or {}
            rows_w1.append({
                "block": block_name, "arm": r["arm"], "G_12": r["G_12"], "R_init": r["radius_init"],
                "u_max": r["u_max"], "converged": r["converged"],
                "rho1_centre": r["rho1_centre"], "rho1_corner": r["rho1_corner"],
                "dp": lap.get("dp"), "R_meas": lap.get("R"), "sigma": lap.get("sigma"),
                "steps": r["steps"], "wall_s": r.get("wall_s"),
                "note": "" if r.get("finite") else "NON-FINITE (diverged)",
            })
    out["tables"]["W1"] = rows_w1

    # Laplace consistency per arm: sigma(R) must be R-independent
    lap_cons = {}
    for r in w.get("W1_droplet_radius", {}).get("runs", []):
        sig = (r.get("laplace") or {}).get("sigma")
        if sig is None or not math.isfinite(float(sig)):
            continue
        lap_cons.setdefault(r["arm"], []).append(float(sig))
    out["tables"]["W1_laplace_consistency"] = {
        arm: {
            "sigma_values": vals,
            "mean": sum(vals) / len(vals),
            "residual": (max(abs(v - sum(vals) / len(vals)) for v in vals) / abs(sum(vals) / len(vals))
                         if sum(vals) != 0 else float("nan")),
        }
        for arm, vals in lap_cons.items()
    }

    # W1 deviations: our convention vs each OSS reading (u_max and sigma)
    dev_w1 = []
    for G in (-1.5, -2.5):
        base = find_run(w.get("W1_droplet_coupling", {}), "GOT-MIX", G_12=G, radius_init=12.0)
        if not base:
            continue
        for arm in ("GOT-SELF", "LBM4", "LBM3", "CTRL-NORM"):
            oth = find_run(w.get("W1_droplet_coupling", {}), arm, G_12=G, radius_init=12.0)
            if not oth:
                continue
            dev_w1.append({
                "G_12": G, "arm": arm,
                "dev_u_max": rel(oth.get("u_max"), base.get("u_max")),
                "dev_sigma": rel((oth.get("laplace") or {}).get("sigma"), (base.get("laplace") or {}).get("sigma")),
                "dev_rho1_centre": rel(oth.get("rho1_centre"), base.get("rho1_centre")),
            })
    out["tables"]["W1_deviations"] = dev_w1

    # ---------------- W2 stratified Poiseuille ----------------
    rows_w2 = []
    for block_name in ("W2_poiseuille_primary", "W2_poiseuille_shift_factor",
                       "W2_poiseuille_resolution", "W2_poiseuille_coupling"):
        for r in w.get(block_name, {}).get("runs", []):
            if r.get("status") == "SKIPPED" or "config" not in r:
                rows_w2.append({"block": block_name, "arm": r["arm"], "note": r.get("reason", "no config")})
                continue
            e = poiseuille_errors(r)
            rows_w2.append({
                "block": block_name, "arm": r["arm"], "ny": r["config"]["ny"],
                "tau_upper": r["config"]["tau_upper"], "G_12": r["config"]["G_12"],
                "u_max_sim": e["u_max_sim"], "u_max_ana": e["u_max_ana"], "err_u_max": e["err_u_max"],
                "Q_sim": e["Q_sim_band"], "Q_ana": e["Q_ana_band"], "err_Q": e["err_Q"],
                "err_profile_L2": e["err_profile_L2"], "steady": r.get("steady"),
                "finite": r.get("finite"), "spurious": r.get("spurious_mix"),
                "wall_s": r.get("wall_s"), "note": "",
            })
            r["_err"] = e
    out["tables"]["W2"] = rows_w2

    dev_w2 = []
    for block_name in ("W2_poiseuille_primary", "W2_poiseuille_shift_factor"):
        base = find_run(w.get(block_name, {}), "GOT-MIX")
        if not base or "_err" not in base:
            continue
        for arm in ("GOT-SELF", "LBM4", "LBM3", "LBM4-TAU", "CTRL-NORM"):
            oth = find_run(w.get(block_name, {}), arm)
            if not oth or "_err" not in oth:
                continue
            dev_w2.append({
                "block": block_name, "arm": arm,
                "tau_upper": oth["config"]["tau_upper"], "ny": oth["config"]["ny"],
                "dev_Q": rel(oth["_err"]["Q_sim_band"], base["_err"]["Q_sim_band"]),
                "dev_u_max": rel(oth["_err"]["u_max_sim"], base["_err"]["u_max_sim"]),
                "err_Q_vs_analytic_GOT": base["_err"]["err_Q"],
                "err_Q_vs_analytic_other": oth["_err"]["err_Q"],
            })
    out["tables"]["W2_deviations"] = dev_w2

    # LIB-ENTRY harness control
    lib = find_run(w.get("W2_poiseuille_primary", {}), "LIB-ENTRY")
    got = find_run(w.get("W2_poiseuille_primary", {}), "GOT-MIX")
    if lib and got and "config" in lib and "config" in got:
        out["tables"]["W2_library_control"] = {
            "Q_LIB-ENTRY": lib["Q_mix"], "Q_GOT-MIX": got["Q_mix"],
            "rel_diff_Q": rel(lib["Q_mix"], got["Q_mix"]),
            "umax_LIB-ENTRY": lib["u_max_mix"], "umax_GOT-MIX": got["u_max_mix"],
            "rel_diff_umax": rel(lib["u_max_mix"], got["u_max_mix"]),
            "profile_max_abs_diff": max(
                abs(a - b) for a, b in zip(lib["u_profile_mix"], got["u_profile_mix"])
            ),
            "verdict": "HARNESS-OK" if rel(lib["Q_mix"], got["Q_mix"]) < 1e-6 else "HARNESS-MISMATCH",
        }

    # ---------------- W3 momentum ledger ----------------
    rows_w3 = []
    for r in w.get("W3_momentum_ledger", {}).get("runs", []):
        if r.get("status") == "SKIPPED":
            rows_w3.append({"arm": r["arm"], "note": r.get("reason")})
            continue
        rows_w3.append({
            "arm": r["arm"], "drift_x": r["drift"][0], "drift_y": r["drift"][1],
            "drift_rate_x": r["drift_rate_per_step"][0], "drift_rate_y": r["drift_rate_per_step"][1],
            "rel_drift_x": rel(r["PT"][0], r["P0"][0]) if r["P0"][0] else float("nan"),
            "box_sum_F1_plus_F2_x": r["box_sum_F1_plus_F2_x"],
            "box_sum_comp_weighted_F_x": r["box_sum_component_weighted_F_x"],
            "finite": r["finite"],
        })
    out["tables"]["W3"] = rows_w3

    # ---------------- W4 guard ----------------
    out["tables"]["W4"] = (w.get("W4_near_vacuum_guard") or {}).get("result", {})

    # ---------------- verdict block ----------------
    verdicts = []
    for row in dev_w2:
        verdicts.append({"workload": "W2", "block": row["block"], "arm": row["arm"],
                         "dev_Q": row["dev_Q"], "verdict": verdict(row["dev_Q"])})
    for row in dev_w1:
        verdicts.append({"workload": "W1", "block": "coupling", "arm": row["arm"],
                         "dev_u_max": row["dev_u_max"], "verdict": verdict(row["dev_u_max"])})
    ctrl = [v for v in verdicts if v["arm"] == "CTRL-NORM"]
    out["control_teeth"] = all(
        (v.get("verdict") == "FAIL") for v in ctrl
    ) and bool(ctrl)
    out["verdicts"] = verdicts
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_deviation_bars(out: dict) -> str:
    devs = out["tables"]["W2_deviations"]
    if not devs:
        return ""
    labels, vals, colours = [], [], []
    for d in devs:
        tag = "LBM4-tau1" if d["block"].endswith("primary") else f"LBM4-tau{d['tau_upper']:.2f}"
        labels.append(f"{d['arm']}\n{tag} ny={d['ny']}")
        vals.append(d["dev_Q"])
        colours.append(COLOURS.get(d["arm"], "#333333"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(range(len(vals)), [100 * v for v in vals], color=colours)
    ax.axhline(15, color="k", ls="--", lw=1.2, label="owner criterion 15%")
    ax.axhline(3, color="tab:green", ls=":", lw=1.2, label="in-house strict 3%")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("|delta Q| / Q   [%]  (vs GOT-MIX)")
    ax.set_title("W2 stratified Poiseuille: deviation of each arm from the shipped 'mixture' convention")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, 100 * v * 1.15, f"{100*v:.1f}%",
                ha="center", fontsize=7)
    fig.tight_layout()
    path = FIG / "fig1_deviation_bars.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(HERE))


def fig_poiseuille_profiles(data: dict, out: dict, block: str, name: str, title: str) -> str:
    runs = data["workloads"].get(block, {}).get("runs", [])
    runs = [r for r in runs if "_err" in r]
    if not runs:
        return ""
    ny = runs[0]["config"]["ny"]
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    for r in runs:
        y = list(range(ny))
        ax.plot(r["u_profile_mix"], y, "-o", ms=3, lw=1.4,
                color=COLOURS.get(r["arm"], "#333"), label=r["arm"])
    ax.plot(runs[0]["_err"]["analytic_profile"], list(range(ny)), "k--", lw=1.6, label="analytic (two-layer)")
    ax.axhline(ny // 2, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("u_x (mixture)")
    ax.set_ylabel("y (lattice rows)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(HERE))


def fig_droplet_spurious(data: dict) -> str:
    runs = data["workloads"].get("W1_droplet_coupling", {}).get("runs", [])
    runs = [r for r in runs if r.get("u_max") is not None and math.isfinite(r["u_max"])]
    if not runs:
        return ""
    Gs = sorted({r["G_12"] for r in runs})
    arms = ["GOT-SELF", "GOT-MIX", "LBM4", "LBM3", "CTRL-NORM"]
    width = 0.8 / len(Gs)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, G in enumerate(Gs):
        vals = []
        for a in arms:
            r = find_run({"runs": runs}, a, G_12=G, radius_init=12.0)
            vals.append(r["u_max"] if r and r["u_max"] is not None else float("nan"))
        xs = [j + i * width for j in range(len(arms))]
        ax.bar(xs, vals, width=width, label=f"G_12={G}")
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(arms))])
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_ylabel("max |u_mix|  (parasitic flow, 2500 steps)")
    ax.set_title("W1 static droplet: parasitic-current level per mixing convention")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig2_droplet_spurious.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(HERE))


def fig_momentum_ledger(data: dict) -> str:
    runs = data["workloads"].get("W3_momentum_ledger", {}).get("runs", [])
    runs = [r for r in runs if r.get("history")]
    if not runs:
        return ""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for r in runs:
        steps = [h[0] for h in r["history"]]
        dy = [h[2] for h in r["history"]]
        ax.plot(steps, dy, "-o", ms=3, color=COLOURS.get(r["arm"], "#333"), label=r["arm"])
    ax.set_xlabel("step")
    ax.set_ylabel("change of global momentum sum_y c_y f  (lattice units)")
    ax.set_title("W3 momentum ledger: closed periodic box, no body force")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "fig3_momentum_ledger.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(HERE))


# ---------------------------------------------------------------------------
def markdown_tables(out: dict) -> str:
    L = []
    t = out["tables"]

    L.append("### W1 - static droplet (n=64, tau=1, 2500-step equal-time window)\n")
    L.append("| block | arm | G_12 | R_init | max\\|u_mix\\| | converged | rho1 centre | rho1 corner | dp | R meas | sigma = dp*R |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t["W1"]:
        if r.get("u_max") is None:
            L.append(f"| {r['block']} | {r['arm']} | - | - | - | - | - | - | - | - | {r.get('note','')} |")
            continue
        L.append("| {} | {} | {:.2f} | {:.0f} | {:.4g} | {} | {:.3f} | {:.3f} | {:.4g} | {:.2f} | {:.4g} |".format(
            r["block"], r["arm"], r["G_12"], r["R_init"], r["u_max"],
            "yes" if r["converged"] else "**no**", r["rho1_centre"], r["rho1_corner"],
            r["dp"] or float("nan"), r["R_meas"] or float("nan"), r["sigma"] or float("nan")))

    L.append("\n### W1 - Young-Laplace consistency across R in {8,12,16} (G_12=-2.0)\n")
    L.append("| arm | sigma(R=8) | sigma(R=12) | sigma(R=16) | mean | radius residual |")
    L.append("|---|---|---|---|---|---|")
    for arm, d in t["W1_laplace_consistency"].items():
        vals = d["sigma_values"]
        L.append(f"| {arm} | " + " | ".join(f"{v:.4g}" for v in vals) +
                 f" | {d['mean']:.4g} | {100*d['residual']:.1f}% |")

    L.append("\n### W1 - deviation of each convention from GOT-MIX (u_max / sigma)\n")
    L.append("| G_12 | arm vs GOT-MIX | dev u_max | dev sigma | dev rho1_centre | u_max verdict |")
    L.append("|---|---|---|---|---|---|")
    for d in t["W1_deviations"]:
        L.append("| {:.2f} | {} | {:.1f}% | {:.1f}% | {:.1f}% | {} |".format(
            d["G_12"], d["arm"], 100 * d["dev_u_max"], 100 * d["dev_sigma"],
            100 * d["dev_rho1_centre"], verdict(d["dev_u_max"])))

    L.append("\n### W2 - stratified Poiseuille vs closed-form two-layer solution\n")
    L.append("| block | arm | ny | tau_upper | G_12 | u_max sim | u_max analytic | err u_max | Q sim | Q analytic | err Q | err profile L2 | steady |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t["W2"]:
        if r.get("u_max_sim") is None:
            L.append(f"| {r['block']} | {r['arm']} | - | - | - | - | - | - | - | - | - | - | {r.get('note','')} |")
            continue
        L.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.4g} | {:.4g} | {} {:.2f}% | {:.4g} | {:.4g} | {} {:.2f}% | {:.2f}% | {} |".format(
            r["block"], r["arm"], r["ny"], r["tau_upper"], r["G_12"],
            r["u_max_sim"], r["u_max_ana"], "**" if (r["err_u_max"] or 0) > STRICT else "", 100 * r["err_u_max"],
            r["Q_sim"], r["Q_ana"], "**" if (r["err_Q"] or 0) > STRICT else "", 100 * r["err_Q"],
            100 * r["err_profile_L2"], "yes" if r["steady"] else "no"))

    if "W2_library_control" in t:
        c = t["W2_library_control"]
        L.append("\n### W2 harness control (shipped entry point vs this harness at the same configuration)\n")
        L.append("| Q LIB-ENTRY | Q GOT-MIX | rel diff | u_max LIB-ENTRY | u_max GOT-MIX | rel diff | max abs profile diff | verdict |")
        L.append("|---|---|---|---|---|---|---|---|")
        L.append("| {:.6g} | {:.6g} | {:.2e} | {:.6g} | {:.6g} | {:.2e} | {:.2e} | {} |".format(
            c["Q_LIB-ENTRY"], c["Q_GOT-MIX"], c["rel_diff_Q"], c["umax_LIB-ENTRY"],
            c["umax_GOT-MIX"], c["rel_diff_umax"], c["profile_max_abs_diff"], c["verdict"]))

    L.append("\n### W2 - deviation from GOT-MIX (Q = band-integrated flux)\n")
    L.append("| block | arm | tau_upper | dev Q | dev u_max | err Q vs analytic (GOT) | err Q vs analytic (arm) | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in t["W2_deviations"]:
        L.append("| {} | {} | {:.2f} | {:.1f}% | {:.1f}% | {:.2f}% | {:.2f}% | {} |".format(
            d["block"], d["arm"], d["tau_upper"], 100 * d["dev_Q"], 100 * d["dev_u_max"],
            100 * d["err_Q_vs_analytic_GOT"], 100 * d["err_Q_vs_analytic_other"],
            verdict(d["dev_Q"])))

    L.append("\n### W3 - momentum ledger (n=48, periodic, no body force, 400 steps)\n")
    L.append("| arm | drift P_x | drift P_y | drift/step P_y | box sum (F1+F2)_x | box sum (rho1 F1 + rho2 F2)/rho_tot | finite |")
    L.append("|---|---|---|---|---|---|---|")
    for r in t["W3"]:
        if r.get("drift_x") is None:
            L.append(f"| {r['arm']} | - | - | - | - | - | {r.get('note','')} |")
            continue
        L.append("| {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {} |".format(
            r["arm"], r["drift_x"], r["drift_y"], r["drift_rate_y"],
            r["box_sum_F1_plus_F2_x"], r["box_sum_comp_weighted_F_x"],
            "yes" if r["finite"] else "**no**"))

    w4 = t["W4"]
    if w4.get("arms"):
        L.append("\n### W4 - depleted-cell guard (one 2x2 cell at rho_1=1e-8, 1e-3 momentum)\n")
        L.append("| arm | finite | max\\|u_mix\\| in depleted patch | max\\|u_mix\\| global | rho_tot in patch |")
        L.append("|---|---|---|---|---|")
        for arm, d in w4["arms"].items():
            L.append("| {} | {} | {:.4g} | {:.4g} | {:.3e} |".format(
                arm, "yes" if d["finite"] else "**no**", d["u_mix_patch_max"],
                d["u_mix_global_max"], d["rho_tot_patch"]))
    return "\n".join(L) + "\n"


def write_csv(out: dict) -> None:
    path = ART / "table.csv"
    keys = ["workload", "block", "arm", "config", "u_max_or_Q_sim", "reference",
            "err_or_dev", "verdict"]
    rows = []
    for r in out["tables"]["W2"]:
        if r.get("u_max_sim") is None:
            continue
        rows.append(("W2", r["block"], r["arm"], f"ny={r['ny']} tau_upper={r['tau_upper']} G12={r['G_12']}",
                     f"{r['Q_sim']:.6g}", f"{r['Q_ana']:.6g}", f"{100*r['err_Q']:.3f}", verdict(r["err_Q"])))
    for d in out["tables"]["W2_deviations"]:
        rows.append(("W2", d["block"], d["arm"], f"ny={d['ny']} tau_upper={d['tau_upper']}",
                     "-", "GOT-MIX", f"{100*d['dev_Q']:.3f}", verdict(d["dev_Q"])))
    for r in out["tables"]["W1"]:
        if r.get("u_max") is None:
            continue
        rows.append(("W1", r["block"], r["arm"], f"G12={r['G_12']} R={r['R_init']}",
                     f"{r['u_max']:.6g}", "-", "-", "INFO"))
    for d in out["tables"]["W1_deviations"]:
        rows.append(("W1", "coupling", d["arm"], f"G12={d['G_12']}",
                     f"{100*d['dev_u_max']:.4g}", "GOT-MIX", f"{100*d['dev_u_max']:.3f}",
                     verdict(d["dev_u_max"])))
    for r in out["tables"]["W3"]:
        if r.get("drift_x") is None:
            continue
        rows.append(("W3", "momentum", r["arm"], "n=48", f"{r['drift_y']:.4g}", "0", "-", "INFO"))
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(keys)
        wr.writerows(rows)
    print("wrote", path)


def main() -> int:
    ART.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    data = json.loads((ART / "result.json").read_text())
    out = build(data)

    figures = []
    figures.append(fig_deviation_bars(out))
    figures.append(fig_poiseuille_profiles(data, out, "W2_poiseuille_primary",
                                          "fig4_profile_primary.png",
                                          "W2 tau=(1,1): u_mix profile vs analytic"))
    figures.append(fig_poiseuille_profiles(data, out, "W2_poiseuille_shift_factor",
                                          "fig5_profile_shift_factor.png",
                                          "W2 tau_upper=0.75: shift-coefficient factor"))
    figures.append(fig_droplet_spurious(data))
    figures.append(fig_momentum_ledger(data))
    out["figures"] = [f for f in figures if f]

    md = markdown_tables(out)
    (ART / "tables.md").write_text(md)
    write_csv(out)

    summary = {
        "verdicts": out["verdicts"],
        "control_teeth": out["control_teeth"],
        "library_control": out["tables"].get("W2_library_control"),
        "laplace_consistency": out["tables"]["W1_laplace_consistency"],
        "figures": out["figures"],
    }
    (ART / "analysis.json").write_text(json.dumps(summary, indent=1))
    print(md)
    print("\ncontrol_teeth:", out["control_teeth"])
    print("library control:", out["tables"].get("W2_library_control"))
    print("figures:", out["figures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())