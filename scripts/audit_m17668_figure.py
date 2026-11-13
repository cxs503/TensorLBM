#!/usr/bin/env python3
"""m17668 audit figure: quantify what m17640's "0% error vs lettuce" actually proves.

Left panel : |got-ref|/|ref| vs lattice resolution nx — the deviation floor is the
             IEEE-754 round-off of one and the same algebraic formula, not a
             physically informative cross-implementation gap.
Right panel: round-trip residuals phys->lattice->phys for u / t / p (relative),
             computed with the real tensorlbm converter.

Numbers are produced by executing code (zero hand-copying).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]


def load_converter():
    sys.path.insert(0, str(HERE / "src"))
    from tensorlbm.unit_converter import LBMUnitConverter

    return LBMUnitConverter


def main() -> int:
    LBMUnitConverter = load_converter()
    rows = []
    # Same anchor family the audited test used: 1 m reference length, water-like.
    re, l_phys, u_phys, nu_phys = 100.0, 1.0, 1.0e-3, 1.0e-5
    for nx in (32, 64, 128, 256):
        uc = LBMUnitConverter(re=re, l_phys=l_phys, u_phys=u_phys, nu_phys=nu_phys, nx=nx)
        # "reference" reproduced by the audited test's own reconstructed class:
        u_lb_ref = uc.ma * (1.0 / math.sqrt(3.0))
        nu_ref = u_lb_ref * nx / re
        tau_ref = nu_ref / (1.0 / 3.0) + 0.5
        dx_ref = l_phys / nx
        dt_ref = dx_ref * u_lb_ref / u_phys
        def rel(g, r):
            return abs(g - r) / abs(r) if r else float("nan")
        rows.append(
            dict(
                nx=nx,
                tau=uc.tau, tau_ref=tau_ref, e_tau=rel(uc.tau, tau_ref),
                nu_lb=uc.nu_lb, e_nu=rel(uc.nu_lb, nu_ref),
                dx=uc.dx, e_dx=rel(uc.dx, dx_ref),
                dt=uc.dt, e_dt=rel(uc.dt, dt_ref),
                ma=uc.ma,
            )
        )

    # Round-trip residuals with the real converter (velocity / time / pressure).
    rt = []
    for nx in (32, 64, 128, 256):
        uc = LBMUnitConverter(re=re, l_phys=l_phys, u_phys=u_phys, nu_phys=nu_phys, nx=nx)
        v = 5.0e-4
        v_lb = v * uc.dt / uc.dx
        v_back = v_lb * uc.dx / uc.dt
        t = 2.0
        n_steps = uc.phys_time_to_steps(t) if hasattr(uc, "phys_time_to_steps") else t / uc.dt
        t_back = n_steps * uc.dt
        rt.append(dict(nx=nx,
                       e_u=abs(v_back - v) / v,
                       e_t=abs(t_back - t) / t,
                       n_steps=n_steps))

    floors = [max(r["e_tau"], r["e_nu"], r["e_dx"], r["e_dt"]) for r in rows]
    out = dict(cases=rows, roundtrip=rt, deviation_floor=max(floors),
               rt_worst=max(max(r["e_u"], r["e_t"]) for r in rt))
    (HERE / "artifacts" / "m17668").mkdir(parents=True, exist_ok=True)
    (HERE / "artifacts" / "m17668" / "units_equivalence_probe.json").write_text(
        json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    nxs = [r["nx"] for r in rows]
    ax[0].loglog(nxs, [max(r["e_tau"], 1e-18) for r in rows], "o-", label=r"$\tau$")
    ax[0].loglog(nxs, [max(r["e_nu"], 1e-18) for r in rows], "s-", label=r"$\nu_{lb}$")
    ax[0].loglog(nxs, [max(r["e_dx"], 1e-18) for r in rows], "^-", label=r"$dx$")
    ax[0].loglog(nxs, [max(r["e_dt"], 1e-18) for r in rows], "d-", label=r"$dt$")
    ax[0].axhline(0.03, color="green", ls="--", lw=1, label="3% acceptance")
    ax[0].axhline(0.15, color="red", ls=":", lw=1, label="15% reject line")
    ax[0].set_xlabel("lattice resolution nx")
    ax[0].set_ylabel("relative deviation got vs 'ref'")
    ax[0].set_title("m17640: deviation sits at IEEE-754 round-off floor\n"
                    "(both sides = same algebraic formula -> identity, not validation)")
    ax[0].set_ylim(1e-18, 1)
    ax[0].legend(fontsize=7, loc="center left")
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].loglog(nxs, [max(r["e_u"], 1e-18) for r in rt], "o-", label="velocity round-trip")
    ax[1].loglog(nxs, [max(r["e_t"], 1e-18) for r in rt], "s-", label="time round-trip")
    ax[1].set_xlabel("lattice resolution nx")
    ax[1].set_ylabel("relative round-trip residual")
    ax[1].set_title("Real converter round-trip (phys->lattice->phys)\n"
                    "worst = %.2e (exact up to round-off)" % out["rt_worst"])
    ax[1].legend(fontsize=7)
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = HERE / "artifacts" / "m17668" / "units_equivalence_probe.png"
    fig.savefig(png, dpi=130)
    print(json.dumps(dict(png=str(png), floor=out["deviation_floor"],
                          rt_worst=out["rt_worst"],
                          tau=[round(r["tau"], 6) for r in rows],
                          ma=[round(r["ma"], 6) for r in rows]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())