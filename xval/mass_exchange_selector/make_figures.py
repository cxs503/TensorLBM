#!/usr/bin/env python3
"""Figures + machine-generated number tables for
``XVAL-LBM-MASS-EXCHANGE-SELECTOR``.

Every number in ``artifacts/numbers.md`` is read from the JSON the runs wrote —
nothing is hand-typed (owner rule: programmatic injection, zero hand-copy).

    python xval/mass_exchange_selector/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REPO = HERE.parent.parent

MM = json.loads((REPO / "src" / "artifacts" / "martin_moyce_1952_dam_break.json").read_text())
MM_T = [d["t_star"] for d in MM["data"]]
MM_Z = [d["X_star"] for d in MM["data"]]


def load(name, default=None):
    p = ART / name
    if not p.exists():
        return default
    return json.loads(p.read_text())


def fig_front(runs_payload):
    runs = {r["label"]: r for r in runs_payload["runs"] if r.get("status") == "ok"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    colors = {"A_repo": "C0", "C_hybrid": "C3", "B_oss": "C2", "D_kappa": "C4",
              "E_tau": "C5", "F_tau": "C6"}
    for ax, a in zip(axes, (16, 32)):
        ax.plot(MM_T, MM_Z, "k^--", ms=5, lw=1.2, label="Martin & Moyce (1952)")
        for lab, r in sorted(runs.items()):
            if r.get("a") != a or ("a16" not in lab and "a32" not in lab):
                continue
            key = next((k for k in colors if lab.startswith(k)), None)
            if key is None:
                continue
            T = [p["T"] for p in r["series"]]
            Z = [p["Z"] for p in r["series"]]
            nm = {"A_repo": "A repo (kappa=1e-2)", "C_hybrid": "C hybrid (kappa=3e-3)",
                  "B_oss": "B OSS port (Q1 fixed)", "D_kappa": "D repo kappa=3e-2",
                  "E_tau": "E repo tau=0.6", "F_tau": "F repo tau=1.4"}[key]
            ax.plot(T, Z, color=colors[key], lw=1.6, label=nm)
        ax.axvspan(0.5, 4.0, color="0.9", zorder=0)
        ax.set_xlim(0, 4.6)
        ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
        ax.set_title(f"a = {a}  (h0 = 2a, g = 5e-5)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$Z = x_{front}/a$")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("XVAL-LBM-MASS-EXCHANGE-SELECTOR — collapse front vs experiment")
    fig.tight_layout()
    out = FIG / "fig1_front_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _decision_curves(rows, variant):
    """For one synthetic configuration, the discrete decision vs fill."""
    rows = sorted(rows, key=lambda r: r["fill"])
    fills, gas, liq = [], [], []
    for r in rows:
        d = r["decisions"][variant]
        fills.append(r["fill"])
        gas.append(d["n_gas"])
        liq.append(d["n_liq"])
    return np.array(fills), np.array(gas), np.array(liq)


def fig_decision_map(diff):
    if not diff:
        return None
    cfgs = sorted({r["config"] for r in diff["synthetic"]})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    styles = {"repo_native": ("C0", "-"), "oss_native": ("C3", "--")}
    for ax, kind in zip(axes, ("n_gas", "n_liq")):
        for cfg in cfgs:
            rows = [r for r in diff["synthetic"] if r["config"] == cfg]
            n_if = max(rows[0]["n_iface"], 1)
            for variant, (c, ls) in styles.items():
                f, g, l = _decision_curves(rows, variant)
                y = (g if kind == "n_gas" else l) / n_if
                if not y.any():
                    continue
                ax.step(f, y, where="mid", color=c, ls=ls,
                        lw=1.0 if cfg == "layer" else 0.8,
                        alpha=0.55 if cfg != "layer" else 1.0,
                        label=f"{cfg} — {variant.split('_')[0]}")
        ax.set_xlabel("interface fill  $\\varepsilon = m/\\rho$")
        ax.set_title("fraction of the interface cells driven to GAS"
                     if kind == "n_gas" else
                     "fraction of the interface cells driven to LIQUID")
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.1)
    axes[0].set_ylabel("fraction converting")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Selector decision map on identical synthetic states (repo solid, OSS dashed)")
    fig.tight_layout()
    out = FIG / "fig2_decision_map.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_state_disagreement(diff):
    if not diff or not diff["states"]:
        return None
    fig, axes = plt.subplots(1, len(diff["states"]), figsize=(11.5, 4.0), sharey=True)
    if len(diff["states"]) == 1:
        axes = [axes]
    keys = ["oss_native", "oss_noGuard", "oss_kappaRepo", "oss_kappaRepo_noGuard"]
    for ax, blk in zip(axes, diff["states"]):
        for k in keys:
            T = [r["T"] for r in blk["rows"]]
            y = [r["variants"][k]["disagree_pct"] for r in blk["rows"]]
            ax.plot(T, y, marker="o", ms=3, lw=1.2, label=k)
        ax.set_xlabel(r"$T = t\sqrt{g/h_0}$")
        ax.set_title(blk["file"].replace("states_", "").replace(".npz", ""))
        ax.grid(alpha=0.3)
        ax.axhline(5.0, color="r", ls=":", lw=1, label="5 % pre-declared")
    axes[0].set_ylabel("decision disagreement vs repo selector  [% of interface cells]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Paired selector differential on frozen states of the running solver")
    fig.tight_layout()
    out = FIG / "fig3_state_disagreement.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_field():
    files = sorted(ART.glob("states_*.npz"))
    if not files:
        return None
    fig, axes = plt.subplots(1, len(files), figsize=(5.2 * len(files), 4.0))
    if len(files) == 1:
        axes = [axes]
    for ax, f in zip(axes, files):
        z = np.load(f)
        i = len(z["steps"]) - 1
        fl, m, rho = z[f"flags_{i}"], z[f"mass_{i}"], z[f"rho_{i}"]
        fill = np.where(fl == 1, 1.0, np.where(fl == 2, m / np.where(rho > 0, rho, 1.0), 0.0))
        fill = np.where(fl == 3, np.nan, fill)
        im = ax.imshow(fill, origin="lower", cmap="viridis", vmin=0, vmax=1.2)
        ax.set_title(f"{f.stem.replace('states_', '')}\nT = {z['T'][i]:.2f}, fill field")
        ax.set_xlabel("x (lattice)")
        ax.set_ylabel("y (lattice)")
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle("Interface fill at the last frozen checkpoint")
    fig.tight_layout()
    out = FIG / "fig4_field.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_numbers(runs_payload, analysis, diff, fidelity):
    L = []
    L.append("# XVAL-LBM-MASS-EXCHANGE-SELECTOR — numbers (programmatic dump, do not hand-edit)\n")
    L.append("Generated by `make_figures.py` from `artifacts/*.json`.\n")
    d = runs_payload["design"]
    L.append("## Case and dimensionless groups\n")
    c = d["case"]
    L.append("| quantity | value |")
    L.append("|---|---|")
    for k, v in c.items():
        L.append(f"| `{k}` | {v} |")
    L.append("")
    L.append("## E1 — port fidelity of the extracted selector\n")
    if fidelity:
        for dt in ("float64", "float32"):
            f = fidelity.get(dt, {})
            L.append(f"* {dt}: states={f.get('states')}, mask mismatch="
                     f"{f.get('mask_mismatch_total')}, max |dmass|="
                     f"{f.get('mass_max_abs_dev'):.3e}, pass={f.get('pass')}")
        L.append("")
    L.append("## E3 — arm table\n")
    L.append("| label | a | tau | kappa | rho_prev_alias | steps | u_max | Ma=u/cs | "
             "max err % | mean err % | pass 15 % | inventory drift % | conv G2I / I2G | wall s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in analysis["table"]:
        sc = r["score"] or {}
        conv = r["conversions"] or {}
        ma = (r["u_max"] or 0) * np.sqrt(3.0)
        L.append(f"| {r['label']} | {r['a']} | {r['tau']} | {r.get('kappa')} | "
                 f"{r.get('rho_prev_alias')} | {r['steps']} | {r['u_max']:.4f} | "
                 f"{ma:.3f} | {sc.get('max_pct')} | {sc.get('mean_pct')} | "
                 f"{sc.get('max_pct') is not None and sc['max_pct'] < 15.0} | "
                 f"{r['inventory_drift_pct']} | {conv.get('gas2if')} / {conv.get('if2gas')} | "
                 f"{r['wall_seconds']} |")
    L.append("")
    L.append("## Arm-to-arm interchangability (front position)\n")
    L.append("| pair | max |dZ|/Z % | mean % | pass 15 % |")
    L.append("|---|---|---|---|")
    for k, v in analysis["arm_arm"].items():
        L.append(f"| {k} | {v['max_pct']} | {v['mean_pct']} | "
                 f"{(v['max_pct'] is not None and v['max_pct'] < 15.0)} |")
    L.append("")
    if analysis.get("determinism"):
        L.append(f"Determinism (arm A repeated at a=16): bitwise identical = "
                 f"{analysis['determinism']['bitwise_identical']} over "
                 f"{analysis['determinism']['n']} samples.\n")
    if diff:
        L.append("## E2 — selector differential on frozen states\n")
        for blk in diff["states"]:
            L.append(f"### {blk['file']}\n")
            L.append("| T | n_iface | " + " | ".join(
                f"{k} %" for k in ("oss_native", "oss_noGuard", "oss_kappaRepo",
                                   "oss_kappaRepo_noGuard")) + " |")
            L.append("|---|---|" + "---|" * 4)
            for r in blk["rows"]:
                L.append(f"| {r['T']:.2f} | {r['n_iface']} | " + " | ".join(
                    f"{r['variants'][k]['disagree_pct']}" for k in
                    ("oss_native", "oss_noGuard", "oss_kappaRepo",
                     "oss_kappaRepo_noGuard")) + " |")
            L.append("")
            L.append("| T | excess_repo | excess_oss | rel diff % | placement L1 / excess % |")
            L.append("|---|---|---|---|---|")
            for r in blk["rows"]:
                e = r["excess"]
                L.append(f"| {r['T']:.2f} | {e['excess_repo']} | {e['excess_oss']} | "
                         f"{e['excess_rel_diff_pct']} | {e['placement_l1_over_excess_pct']} |")
            L.append("")
        L.append("## E2b — synthetic decision map (thresholds)\n")
        L.append("| config | fill | repo to_gas | oss to_gas | repo to_liq | oss to_liq |")
        L.append("|---|---|---|---|---|---|")
        for r in diff["synthetic"]:
            dcs = r["decisions"]
            if not (dcs["oss_native"]["n_gas"] or dcs["oss_native"]["n_liq"]
                    or dcs["repo_native"]["n_gas"] or dcs["repo_native"]["n_liq"]):
                continue
            L.append(f"| {r['config']} | {r['fill']} | {dcs['repo_native']['n_gas']} | "
                     f"{dcs['oss_native']['n_gas']} | {dcs['repo_native']['n_liq']} | "
                     f"{dcs['oss_native']['n_liq']} |")
        L.append("")
    (ART / "numbers.md").write_text("\n".join(L) + "\n")
    return ART / "numbers.md"


def main():
    runs_payload = load("runs.json")
    analysis = load("analysis.json")
    diff = load("differential.json")
    fidelity = load("fidelity_both.json")
    outs = []
    if runs_payload and analysis:
        outs.append(fig_front(runs_payload))
        outs.append(fig_field())
    if diff:
        outs.append(fig_decision_map(diff))
        outs.append(fig_state_disagreement(diff))
    if runs_payload and analysis:
        outs.append(write_numbers(runs_payload, analysis, diff, fidelity))
    print("wrote:", *[str(o) for o in outs if o], sep="\n  ")


if __name__ == "__main__":
    main()