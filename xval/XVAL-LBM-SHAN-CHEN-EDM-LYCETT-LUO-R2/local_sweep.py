#!/usr/bin/env python3
"""Compact sweep + figures for XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO round 2.

Grid: forcing x tau x R with R in (10, 12, 14), tau in (1.0, 0.8), plus a
tau-scan (0.8, 1.0, 1.2) at R = 12.  Writes artifacts/compact.json and the
figures used by REPORT.md.  Runs on one thread (~90 s); the full 45-case
sweep is dispatched separately via the compute gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_xval as X  # noqa: E402

FORCINGS = ("velocity_shift", "edm", "guo")
TASKS = [(f, tau, R) for tau in (1.0, 0.8) for f in FORCINGS for R in (10.0, 12.0, 14.0)]
TASKS += [(f, 1.2, 12.0) for f in FORCINGS]


def main():
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    cases = []
    for f, tau, R in TASKS:
        r = X.run_case(f, tau, R, max_steps=4000, threads=1)
        cases.append(r)
        print(f"{f:14s} tau={tau:<4} R={R:<5} steps={r['steps']:<5} "
              f"dp={r.get('dp', float('nan')):.6f} sigma={r.get('sigma', float('nan')):.5f} "
              f"rho=({r.get('rho_in', 0):.4f},{r.get('rho_out', 0):.4f}) "
              f"u_max={r.get('u_max', float('nan')):.4f} div={r['diverged']}", flush=True)

    mx = X.maxwell_coexistence(X.CFG["G_lib"])
    legs = {
        "laplace": X.leg_laplace(cases),
        "maxwell": X.leg_maxwell(cases, mx),
        "scheme": X.leg_scheme(cases),
    }
    ident = X.leg_identity()
    out = {"cfg": X.CFG, "maxwell": mx, "identity": ident, "cases": cases, "legs": legs}
    (art / "compact.json").write_text(json.dumps(out, indent=2, default=float))

    print("\n== Laplace legs (sigma through origin) ==")
    for k, v in legs["laplace"].items():
        if "sigma_fit" in v:
            print(f"{k:22s} sigma={v['sigma_fit']:.5f} R2={v['r2_origin']:.5f} "
                  f"spread={v['sigma_spread_pct']:.1f}% radii={v['radii']}")
    print("\n== Maxwell (analytic) ==", {k: round(v, 5) for k, v in mx.items()})
    errs_l = [v["err_rho_l_pct"] for v in legs["maxwell"].values()]
    errs_v = [v["err_rho_v_pct"] for v in legs["maxwell"].values()]
    print(f"coexistence error: rho_l max {max(errs_l):.1f}%  rho_v max {max(errs_v):.1f}%")
    print("\n== identity ==", json.dumps(ident))
    print("\n== scheme discriminance (R=12) ==")
    for k, v in legs["scheme"].items():
        if "R12" in k:
            print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                      for kk, vv in v.items() if kk.startswith("d_")})
    make_figures(out, art)
    print("\nwrote", art / "compact.json")


def make_figures(out, art):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c for c in out["cases"] if not c.get("diverged") and "dp" in c]
    cols = {"velocity_shift": "#1f77b4", "edm": "#d62728", "guo": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for tau, mk in ((1.0, "o"), (0.8, "s")):
        for f in FORCINGS:
            rows = sorted([c for c in cases if c["forcing"] == f and c["tau"] == tau],
                          key=lambda c: c["R_eq"])
            if not rows:
                continue
            x = [1 / c["R_eq"] for c in rows]
            y = [c["dp"] for c in rows]
            sig = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)
            ax.plot(x, y, mk, color=cols[f], label=f"{f} tau={tau}")
            xx = np.linspace(0, max(x) * 1.05, 10)
            ax.plot(xx, sig * xx, "-", color=cols[f], lw=1, alpha=0.7)
    ax.set_xlabel(r"$1/R_{eq}$"); ax.set_ylabel(r"$\Delta p$  (library $p_{mech}$)")
    ax.set_title("Young-Laplace: lines forced through the origin")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for f in FORCINGS:
        rows = sorted([c for c in cases if c["forcing"] == f and c["R0"] == 12.0],
                      key=lambda c: c["tau"])
        if len(rows) < 2:
            continue
        ax.plot([c["tau"] for c in rows], [c["sigma"] for c in rows], "o-", color=cols[f], label=f)
    ax.axhline(0.0561, color="k", ls=":", lw=1, label="documented 0.0561")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\sigma$ at $R=12$")
    ax.set_title("tau-stability of the surface tension")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    mx = out["maxwell"]
    ax.axhline(mx["rho_l"], color="#d62728", ls="--", lw=1, label=f"Maxwell rho_l={mx['rho_l']:.3f}")
    ax.axhline(mx["rho_v"], color="#1f77b4", ls="--", lw=1, label=f"Maxwell rho_v={mx['rho_v']:.4f}")
    for i, f in enumerate(FORCINGS):
        rows = [c for c in cases if c["forcing"] == f and c["tau"] == 1.0]
        if not rows:
            continue
        ax.errorbar([i], [np.mean([c["rho_in"] for c in rows])], fmt="o", color="#d62728")
        ax.errorbar([i], [np.mean([c["rho_out"] for c in rows])], fmt="s", color="#1f77b4")
    ax.set_xticks(range(len(FORCINGS))); ax.set_xticklabels(FORCINGS, fontsize=8)
    ax.set_ylabel("band density")
    ax.set_title("coexistence vs Maxwell (analytic)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(art / "fig1_laplace_tau_coexistence.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()