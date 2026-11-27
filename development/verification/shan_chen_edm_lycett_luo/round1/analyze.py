"""Aggregate the coexistence campaign and render the report figures.

Reads ``artifacts/cases/*.json`` (one file per case, written by
``run_coexistence.py``) and produces

* ``artifacts/results.json``  — flat table + derived metrics (zero hand copying)
* ``artifacts/TABLE.md``      — markdown table for the report
* ``artifacts/fig*.png``      — figures for the report

No numbers in the report are typed by hand: they are emitted from here.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import maxwell_sc as MW  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "artifacts", "cases")
FORCING_ORDER = ("velocity_shift", "edm", "guo")
COLORS = {"velocity_shift": "#d62728", "edm": "#1f77b4", "guo": "#2ca02c"}
EOS_LABEL = {"exp": r"$\psi=\rho_0(1-e^{-\rho/\rho_0})$, G=5",
             "cs": r"$\psi_{CS}$ (TCLB), $T_r$=0.85, G=1"}
TC_CS = 0.09432


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(CASES, "*.json"))):
        with open(p) as fh:
            data = json.load(fh)
        rows.extend(data)
    return rows


def metrics(rows):
    """Per (eos, forcing): tau-spread and worst-case deviation from Maxwell."""
    out = {}
    for r in rows:
        if r["ic"] != "maxwell":
            continue
        key = (r["eos"], r["forcing"])
        out.setdefault(key, []).append(r)
    summary = {}
    for key, rs in out.items():
        # only the fixed-G tau grid {0.6, 1.0, 1.5, 2.0} enters the tau metric
        tau_rs = [r for r in rs if r["G"] == (5.0 if key[0] == "exp" else 1.0)]
        tl = [r["measured"]["rho_l"] for r in tau_rs]
        tg = [r["measured"]["rho_g"] for r in tau_rs]
        summary[f"{key[0]}|{key[1]}"] = {
            "n": len(rs),
            "tau_set": sorted(r["tau"] for r in tau_rs),
            "rho_l_tau_spread_pct": 100.0 * (max(tl) - min(tl)) / np.mean(tl),
            "rho_g_tau_spread_pct": 100.0 * (max(tg) - min(tg)) / np.mean(tg),
            "max_abs_err_l_pct": max(abs(r["err_l_pct"]) for r in rs),
            "max_abs_err_g_pct": max(abs(r["err_g_pct"]) for r in rs),
            "mean_abs_err_l_pct": float(np.mean([abs(r["err_l_pct"]) for r in rs])),
            "mean_abs_err_g_pct": float(np.mean([abs(r["err_g_pct"]) for r in rs])),
        }
    return summary


def table_md(rows):
    hdr = ("| case | EOS | forcing | tau | G | rho_l (meas) | rho_l (Maxwell) | err_l % "
           "| rho_g (meas) | rho_g (Maxwell) | err_g % | max abs u | drift_l | mass drift |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in sorted(rows, key=lambda x: (x["eos"], x["forcing"], x["tau"], x["G"] or 0, x["ic"])):
        m, mw = r["measured"], r["maxwell"]
        lines.append(
            f"| {r['tag']} | {r['eos']} | {r['forcing']} | {r['tau']:.2f} | {r['G']:.2f} "
            f"| {m['rho_l']:.5f} | {mw['rho_l']:.5f} | {r['err_l_pct']:+.2f} "
            f"| {m['rho_g']:.5f} | {mw['rho_g']:.5f} | {r['err_g_pct']:+.2f} "
            f"| {m['max_u']:.2e} | {r['drift_l']:.1e} | {r['mass_drift']:.1e} |")
    return hdr + "\n".join(lines) + "\n"


def fig_coexistence(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    Gs = np.linspace(4.15, 5.6, 60)
    mw_l, mw_g = [], []
    for G in Gs:
        c = MW.maxwell(G, MW.psi_exp, rho_hi=6.0, n_solve=2000)
        mw_l.append(c.rho_l)
        mw_g.append(c.rho_g)
    ax.plot(Gs, mw_l, "-", color="k", lw=1.4, label="Maxwell construction (this work)")
    ax.plot(Gs, mw_g, "-", color="k", lw=1.4)
    for forcing in FORCING_ORDER:
        rs = [r for r in rows if r["eos"] == "exp" and r["forcing"] == forcing
              and r["ic"] == "maxwell" and r["tau"] == 1.0]
        rs.sort(key=lambda r: r["G"])
        ax.plot([r["G"] for r in rs], [r["measured"]["rho_l"] for r in rs], "o",
                color=COLORS[forcing], ms=6, label=f"LBM rho_l, {forcing}")
        ax.plot([r["G"] for r in rs], [r["measured"]["rho_g"] for r in rs], "s",
                color=COLORS[forcing], ms=5, mfc="none", label=f"LBM rho_g, {forcing}")
    ax.set_xlabel("coupling G (>0 attractive)")
    ax.set_ylabel(r"density  $\rho$")
    ax.set_title(r"SCMP coexistence curve, $\psi=\rho_0(1-e^{-\rho/\rho_0})$, $\tau=1$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_tau(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, eos in zip(axes, ("exp", "cs")):
        rs0 = [r for r in rows if r["eos"] == eos and r["ic"] == "maxwell"]
        r0 = rs0[0]
        ax.axhline(r0["maxwell"]["rho_l"], ls="--", color="k", lw=1.1,
                   label="Maxwell $" + r"\\rho_l$" + "$")
        ax.axhline(r0["maxwell"]["rho_g"], ls=":", color="k", lw=1.1,
                   label="Maxwell $" + r"\\rho_g$" + "$")
        for forcing in FORCING_ORDER:
            rs = [r for r in rs0 if r["forcing"] == forcing
                  and r["G"] == (5.0 if eos == "exp" else 1.0)]
            rs.sort(key=lambda r: r["tau"])
            ax.plot([r["tau"] for r in rs], [r["measured"]["rho_l"] for r in rs], "o-",
                    color=COLORS[forcing], ms=5, label=f"{forcing} $\\rho_l$")
            ax.plot([r["tau"] for r in rs], [r["measured"]["rho_g"] for r in rs], "s--",
                    color=COLORS[forcing], ms=4, mfc="none", label=f"{forcing} $\\rho_g$")
        ax.set_xlabel(r"relaxation time $\tau$")
        ax.set_ylabel("plateau density")
        ax.set_title(EOS_LABEL[eos], fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Flat-interface coexistence vs tau: EDM is tau-independent, velocity_shift drifts",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_profiles(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharex=True)
    for ax, eos in zip(axes, ("exp", "cs")):
        for forcing in ("velocity_shift", "edm"):
            for tau, ls in ((0.6, "-"), (2.0, "--")):
                sel = [r for r in rows if r["eos"] == eos and r["forcing"] == forcing
                       and r["ic"] == "maxwell" and r["tau"] == tau
                       and r["G"] == (5.0 if eos == "exp" else 1.0)]
                if not sel:
                    continue
                r = sel[0]
                prof = np.array(r["profile_x"])
                ax.plot(np.arange(prof.size), prof, ls, color=COLORS[forcing], lw=1.1,
                        label=f"{forcing}, tau={tau}")
        r0 = [r for r in rows if r["eos"] == eos][0]
        ax.axhline(r0["maxwell"]["rho_l"], color="k", lw=0.8, ls=":")
        ax.axhline(r0["maxwell"]["rho_g"], color="k", lw=0.8, ls=":")
        ax.set_title(EOS_LABEL[eos], fontsize=9)
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel(r"$\rho$")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle("Density profiles (dotted: Maxwell coexistence densities)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_errors(summary, path):
    keys = sorted(summary)
    labels = [k.replace("|", " / ") for k in keys]
    x = np.arange(len(keys))
    el = [summary[k]["max_abs_err_l_pct"] for k in keys]
    eg = [summary[k]["max_abs_err_g_pct"] for k in keys]
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - w / 2, el, w, color="#4c72b0", label=r"max $|\Delta\rho_l|$ / %")
    ax.bar(x + w / 2, eg, w, color="#dd8452", label=r"max $|\Delta\rho_g|$ / %")
    ax.axhline(15.0, color="r", ls="--", lw=1.2, label="acceptance 15 %")
    ax.axhline(3.0, color="gray", ls=":", lw=1.2, label="3 %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("worst-case deviation from Maxwell / %")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    rows = load()
    if not rows:
        raise SystemExit("no case json found")
    summary = metrics(rows)
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    with open(os.path.join(HERE, "artifacts", "results.json"), "w") as fh:
        json.dump({"cases": rows, "summary": summary}, fh, indent=1)
    with open(os.path.join(HERE, "artifacts", "TABLE.md"), "w") as fh:
        fh.write(table_md(rows))
        fh.write("\n## per-(EOS, forcing) metrics\n\n")
        fh.write("| key | n | tau set | rho_l tau-spread % | rho_g tau-spread % "
                 "| max abs err_l % | max abs err_g % |\n|---|---|---|---|---|---|---|\n")
        for k, v in sorted(summary.items()):
            fh.write(f"| {k} | {v['n']} | {v['tau_set']} | {v['rho_l_tau_spread_pct']:.2f} "
                     f"| {v['rho_g_tau_spread_pct']:.2f} | {v['max_abs_err_l_pct']:.2f} "
                     f"| {v['max_abs_err_g_pct']:.2f} |\n")
    fig_coexistence(rows, os.path.join(HERE, "artifacts", "fig1_coexistence.png"))
    fig_tau(rows, os.path.join(HERE, "artifacts", "fig2_tau_dependence.png"))
    fig_profiles(rows, os.path.join(HERE, "artifacts", "fig3_profiles.png"))
    fig_errors(summary, os.path.join(HERE, "artifacts", "fig4_error_bars.png"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()