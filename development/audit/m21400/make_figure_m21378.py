#!/usr/bin/env python3
"""m21400 audit figure — three panels supporting the m21378 verdict.

Panel A  sigma(tau) at R0=12 for the three forcings vs the library calibration
         0.05611: EDM flat (0.40% full spread), velocity_shift slope (53.7%),
         Guo low by 39%.
Panel B  the same delivered band densities scored under both Shan-Chen pressure
         signs: the minus (library, backward-gather) convention reproduces the
         artifact sigma digit-for-digit, the plus (round-1) convention lands at
         ~196-459x -> the round-1 "sigma ~ 16, PASS" was a convention error.
Panel C  measured band densities vs the analytic Maxwell coexistence values
         (independent recompute in probe_m21378.py, agreement <1e-6 %).

Writes audit/m21400/fig1_m21378_audit.png and prints the numbers it plotted.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMPACT = ROOT / "xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO-R2/artifacts/compact.json"
PROBE = HERE / "probe_m21378_results.json"
SIGMA_LIB = 0.05611
COL = {"velocity_shift": "#1f77b4", "edm": "#d62728", "guo": "#2ca02c"}


def dispersion(vals):
    vals = np.asarray(vals, dtype=float)
    m = vals.mean()
    return (vals.max() - vals.min()) / m * 100.0, (vals.max() - vals.min()) / 2 / m * 100.0


def main():
    compact = json.loads(COMPACT.read_text())
    probe = json.loads(PROBE.read_text())
    cases = compact["cases"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---- Panel A: sigma(tau)
    ax = axes[0]
    for f in ("velocity_shift", "edm", "guo"):
        rows = sorted(
            [(c["tau"], c["sigma"]) for c in cases if c["forcing"] == f and c["R0"] == 12.0 and "sigma" in c]
        )
        taus = [r[0] for r in rows]
        sig = [r[1] for r in rows]
        full, half = dispersion(sig)
        ax.plot(taus, sig, "o-", color=COL[f], label=f"{f} (full spread {full:.2f}%)")
        ax.annotate(f"{sig[-1]:.4f}", (taus[-1], sig[-1]), textcoords="offset points",
                    xytext=(6, -3), fontsize=7, color=COL[f])
    ax.axhline(SIGMA_LIB, color="k", ls=":", lw=1.2, label=f"library calibration {SIGMA_LIB}")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\sigma=\Delta p\,R_{eq}$  (library $p_{mech}$, $R_0=12$)")
    ax.set_title("A. EDM removes the tau-dependence of sigma")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # ---- Panel B: pressure sign convention
    ax = axes[1]
    keys = ["edm@1.0@R12", "edm@0.8@R12", "edm@1.2@R12", "guo@1.0@R12"]
    labels = [k.replace("@R12", "") for k in keys]
    sm = [probe["pressure"][k]["sigma_minus"] for k in keys]
    sp = [probe["pressure"][k]["sigma_plus"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, sp, 0.4, color="#8c564b", label=r"round-1 sign: $p=cs^2\rho+\frac{G}{2}cs^2\psi^2$")
    ax.bar(x + 0.2, sm, 0.4, color="#17becf", label=r"library sign: $p=cs^2\rho-\frac{G}{2}cs^2\psi^2$")
    ax.axhline(SIGMA_LIB, color="k", ls=":", lw=1.2)
    for xi, k in zip(x, keys):
        ax.annotate(f"x{probe['pressure'][k]['ratio_plus_over_minus']:.0f}", (xi, sp[keys.index(k)]),
                    textcoords="offset points", xytext=(-10, 3), fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel(r"$\sigma$ from the same delivered band densities")
    ax.set_title("B. Same densities, two pressure signs (factor ~2e2-5e2)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # ---- Panel C: coexistence vs Maxwell
    ax = axes[2]
    mx = probe["maxwell_mine"]
    ax.axhline(mx["rho_l"], color="#d62728", ls="--", lw=1.2,
               label=f"Maxwell $\\rho_l$={mx['rho_l']:.4f}")
    ax.axhline(mx["rho_v"], color="#1f77b4", ls="--", lw=1.2,
               label=f"Maxwell $\\rho_v$={mx['rho_v']:.4f}")
    for i, f in enumerate(("velocity_shift", "edm", "guo")):
        rows = [c for c in cases if c["forcing"] == f and c["R0"] == 12.0 and "rho_in" in c]
        rows.sort(key=lambda c: c["tau"])
        t = [c["tau"] for c in rows]
        ax.plot(t, [c["rho_in"] for c in rows], "o-", color="#d62728", alpha=0.85, lw=1,
                label=None)
        ax.plot(t, [c["rho_out"] for c in rows], "s-", color="#1f77b4", alpha=0.85, lw=1,
                label=None)
        ax.annotate(f, (t[-1], rows[-1]["rho_out"]), textcoords="offset points", xytext=(4, -10),
                    fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("band density ($R_0=12$, circles $\\rho_{in}$, squares $\\rho_{out}$)")
    ax.set_title("C. coexistence vs analytic Maxwell reference")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("m21400 board audit of m21378 — XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO (R2), verdict L3",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dst = HERE / "fig1_m21378_audit.png"
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    print("wrote", dst, dst.stat().st_size, "B")
    for f in ("velocity_shift", "edm", "guo"):
        vals = [c["sigma"] for c in cases if c["forcing"] == f and c["R0"] == 12.0 and "sigma" in c]
        print(f, "sigma", vals, "full/half spread %", [round(v, 2) for v in dispersion(vals)])


if __name__ == "__main__":
    main()