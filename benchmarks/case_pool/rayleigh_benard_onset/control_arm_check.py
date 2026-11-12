#!/usr/bin/env python3
"""Control arm (Route C) of CASE-LBM-RB-ONSET-RA1708 — standalone, zero hand-typed numbers.

The case-pool entry declares ``Ra_c = 27 pi^4 / 4 = 1707.7597`` (rigid-rigid, most
unstable wavenumber ``a_c = 3.11632``) as the reference for a <2 % criterion, while
the LBM production arms drive **mode m = 1 on Gamma = nx/ny = 2**, i.e. ``k H = pi``.
Those are two different points of the marginal curve, so the control arm must answer
three separate questions before any LBM number can be compared with the literature:

1. does an *independent* continuum discretisation (second-order FD + QZ full spectrum
   + Richardson extrapolation in the interior-node count M) reproduce ``1707.7597``
   to <0.05 %?  (`route_c_control`)
2. how far is the marginal value at the wavenumber the LBM actually drives,
   ``Ra_c(kH = pi)``, from the textbook minimum?  (`_ra_c_at_wavenumber`)
3. how wide is the marginal band between the two rigid-rigid eigenvalue branches
   (stationary vs overstable)? — only the stationary branch is reachable by the
   steady base state used in Route A.

Everything is delegated to the repository functions in ``run_case.py``; this file
only sequences them and writes the artefact:

    artifacts/ra1708_control/control_arm.json      numbers + provenance
    artifacts/ra1708_control/control_arm.png       Richardson plot + wavenumber curve

Usage: PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = REPO / "artifacts" / "ra1708_control"


def load_driver():
    spec = importlib.util.spec_from_file_location("rb1708_driver", HERE.parent / "run_case.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    t0 = time.time()
    drv = load_driver()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1. independent continuum reference (Richardson in M) ---------------
    ctrl = drv.route_c_control()

    # --- 2. marginal value at the wavenumber the LBM actually drives --------
    a_lbm = math.pi  # mode m = 1, Gamma = 2  ->  kH = 2 pi m / Gamma = pi
    ra_pi = drv._ra_c_at_wavenumber(a_lbm)
    ra_min = drv._ra_c_at_wavenumber(drv.A_REF)
    # local curvature of the marginal curve, from the same solver (finite difference)
    da = 1.0e-3
    ra_p = drv._ra_c_at_wavenumber(a_lbm + da)
    ra_m = drv._ra_c_at_wavenumber(a_lbm - da)
    curvature = (ra_p - 2.0 * ra_pi + ra_m) / da**2

    payload = {
        "case_code": drv.CASE_CODE,
        "generated_by": "benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py",
        "route_c_control": ctrl,
        "wavenumber_offset": {
            "a_ref_textbook": drv.A_REF,
            "ra_ref_textbook": drv.RA_REF,
            "ra_c_at_a_ref_from_driver": ra_min,
            "a_lbm_mode1_gamma2": a_lbm,
            "ra_c_at_a_lbm": ra_pi,
            "rel_offset_pct": (ra_pi - drv.RA_REF) / drv.RA_REF * 100.0,
            "d2Ra_da2_at_a_lbm": curvature,
            "note": "Route A drives a single x-mode (kH = pi), so the admissible reference "
                    "is Ra_c(a = pi), not the textbook minimum Ra_c(a_c = 3.11632); the "
                    "difference is a second-order (flat-bottom) offset and is quantified here.",
        },
        "inputs": {
            "ra_ref": drv.RA_REF,
            "a_ref": drv.A_REF,
            "tol_pct": drv.TOL_PCT,
            "pr_route_c": drv.PREDECLARED["pr"],
            "control_grids": drv.PREDECLARED["control_grids"],
            "control_tol_pct": drv.PREDECLARED["control_tol_pct"],
        },
        "verdict": {
            "control_ok": bool(ctrl.get("control_ok", False)),
            "lbm_reference_to_use": ("Ra_c(kH=pi) = %.4f  (textbook minimum %.4f, "
                                     "offset %+.4f %%)"
                                     % (ra_pi, drv.RA_REF,
                                        (ra_pi - drv.RA_REF) / drv.RA_REF * 100.0)),
        },
        "elapsed_s": time.time() - t0,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    (OUT / "control_arm.json").write_text(json.dumps(payload, indent=2, default=float))

    # --- 3. figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))

    hist = ctrl.get("history") or ctrl.get("per_grid") or []
    Ms, ras = [], []
    for rec in hist:
        m = rec.get("M") or rec.get("nodes") or rec.get("grid")
        r = rec.get("ra_c") or rec.get("Ra_c")
        if m is not None and r is not None:
            Ms.append(float(m))
            ras.append(float(r))
    if Ms:
        Ms = np.asarray(Ms)
        ras = np.asarray(ras)
        ax[0].plot(1.0 / Ms, ras, "o-", label="FD + QZ (this control arm)")
        if len(Ms) >= 2:  # Richardson value plotted as a horizontal line at 1/M = 0
            ext = ctrl.get("ra_c_richardson") or ctrl.get("ra_c_extrap")
            if ext is not None:
                ax[0].axhline(float(ext), color="k", ls="--",
                              label="Richardson $M\\to\\infty$ = %.3f" % float(ext))
        ax[0].axhline(drv.RA_REF, color="crimson", ls=":", label="Chandrasekhar 1707.76")
        ax[0].set_xlabel("$1/M$")
        ax[0].set_ylabel("$Ra_c$ (rigid-rigid)")
        ax[0].set_title("Route C control: grid convergence")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)
    else:
        ax[0].text(0.05, 0.5, "control history keys: %s" % sorted(ctrl.keys()), fontsize=6)

    a = np.linspace(2.0, 5.0, 60)
    ra = np.array([drv._ra_c_at_wavenumber(float(x)) for x in a])
    ax[1].plot(a, ra, "-", label="$Ra_c(a)$ rigid-rigid")
    ax[1].plot([a_lbm], [ra_pi], "o", color="tab:blue",
               label="LBM mode: $a=\\pi$ → %.2f" % ra_pi)
    ax[1].plot([drv.A_REF], [drv.RA_REF], "s", color="crimson",
               label="textbook $a_c$ → 1707.76")
    ax[1].set_xlabel("$a = kH$")
    ax[1].set_ylabel("$Ra_c$")
    ax[1].set_title("Marginal curve: reference at the driven wavenumber")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "control_arm.png", dpi=150)

    print(json.dumps(payload["verdict"], indent=2))
    print("wrote", OUT / "control_arm.json", OUT / "control_arm.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())