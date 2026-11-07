#!/usr/bin/env python
"""IN31-N12 production benchmark: discrete lattice stability of the RB operator.

For a given vertical resolution ``ny`` (gap ``H = ny`` lattice units, horizontal
periodicity over ``nx = nx_factor * ny``) this script

1. converges the discrete conductive base state of the library RB operator,
2. locates the discrete critical Rayleigh number ``Ra_c^lat`` by bisecting
   ``gamma(Ra) = 0`` on the Jacobian spectrum,
3. tabulates ``gamma(Ra)`` on both sides of onset for the lattice operator and,
   at the *same* non-dimensional wavenumber, for the continuum
   Navier-Stokes-Boussinesq problem (``tensorlbm.rb_continuum_stability``),
4. time-marches the non-linear library solver at selected arms and compares the
   measured growth rate with the eigenvalue prediction.

Everything is written to one JSON file (numbers only, no hand copying).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.rb_continuum_stability import (  # noqa: E402
    RA_CRIT_CONTINUOUS,
    growth as continuum_growth,
)
from tensorlbm.rb_discrete_stability import (  # noqa: E402
    RBLatticeParams,
    conductive_base_state,
    growth_rate,
    measure_growth_rate,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nx-factor", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.855)
    ap.add_argument("--tauT", type=float, default=1.0)
    ap.add_argument("--ra-lo", type=float, default=1100.0)
    ap.add_argument("--ra-hi", type=float, default=9000.0)
    ap.add_argument("--ra-tol", type=float, default=2e-3)
    ap.add_argument("--arms", default="0.85,0.95,1.0,1.05,1.15")
    ap.add_argument("--base-steps", type=int, default=8000)
    ap.add_argument("--warm-steps", type=int, default=1200)
    ap.add_argument("--m-cont", type=int, default=200)
    ap.add_argument("--sim-check", type=int, default=2)
    ap.add_argument("--sim-steps", type=int, default=6000)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    nx = int(round(args.nx_factor * args.ny))
    p = RBLatticeParams(ny=args.ny, nx=nx, tau=args.tau, tau_T=args.tauT)
    t0 = time.time()
    print(f"[rb] ny={args.ny} nx={nx} a={p.a:.5f} nu={p.nu:.5f} alpha={p.alpha:.5f} Pr={p.Pr:.4f}", flush=True)

    ra_mid = math.sqrt(args.ra_lo * args.ra_hi)
    base = conductive_base_state(p.with_Ra(ra_mid), steps=args.base_steps)
    print(f"[rb] base residual {base.residual:.3e} after {base.steps} steps", flush=True)

    def gamma_at(Ra: float, base_in=None, warm: bool = True):
        q = p.with_Ra(Ra)
        b = base_in
        if b is None:
            b = conductive_base_state(
                q, steps=args.warm_steps, warm_start=base if warm else None
            )
        r = growth_rate(q, base=b, eps=args.eps)
        r["base_residual"] = b.residual
        return r, b

    # --- discrete critical Rayleigh number by bisection ---------------------
    lo, hi = args.ra_lo, args.ra_hi
    r_lo, b_lo = gamma_at(lo)
    r_hi, b_hi = gamma_at(hi)
    if not (r_lo["gamma_step"] < 0.0 < r_hi["gamma_step"]):
        raise SystemExit(
            f"bracket invalid: gamma(lo)={r_lo['gamma_step']:.3e} gamma(hi)={r_hi['gamma_step']:.3e}"
        )
    history = [
        {"Ra": lo, "gamma_step": r_lo["gamma_step"], "gamma_hat": r_lo["gamma_hat"]},
        {"Ra": hi, "gamma_step": r_hi["gamma_step"], "gamma_hat": r_hi["gamma_hat"]},
    ]
    while hi - lo > args.ra_tol * hi:
        mid = 0.5 * (lo + hi)
        r_mid, b_mid = gamma_at(mid)
        history.append({"Ra": mid, "gamma_step": r_mid["gamma_step"], "gamma_hat": r_mid["gamma_hat"]})
        print(
            f"[rb] bisect Ra={mid:9.1f} gamma_step={r_mid['gamma_step']:+.4e} "
            f"gamma_hat={r_mid['gamma_hat']:+.4f} base_resid={b_mid.residual:.1e}",
            flush=True,
        )
        if r_mid["gamma_step"] < 0.0:
            lo, r_lo = mid, r_mid
        else:
            hi, r_hi = mid, r_mid
        if len(history) > 40:
            break
    ra_c_lat = 0.5 * (lo + hi)
    print(f"[rb] Ra_c^lat = {ra_c_lat:.2f}  ({100*(ra_c_lat/RA_CRIT_CONTINUOUS-1):+.2f}% vs 1707.76)", flush=True)

    # --- arm table ----------------------------------------------------------
    arms = []
    for frac in [float(x) for x in args.arms.split(",")]:
        Ra = frac * ra_c_lat
        r, b = gamma_at(Ra)
        cont = continuum_growth(p.a, Ra, Pr=p.Pr, M=args.m_cont)
        arms.append(
            {
                "frac_of_Ra_c_lat": frac,
                "Ra": Ra,
                "gamma_step": r["gamma_step"],
                "gamma_hat_lattice": r["gamma_hat"],
                "lambda_lead": r["lambda_lead"],
                "n_artifact_modes": r["n_ghost_modes"],
                "base_residual": r["base_residual"],
                "gamma_hat_continuum": cont["sigma"],
                "rel_err_lattice_vs_cont": (r["gamma_hat"] / cont["sigma"] - 1.0)
                if cont["sigma"] != 0.0
                else float("nan"),
                "abs_err_hat": r["gamma_hat"] - cont["sigma"],
            }
        )
        print(
            f"[rb] frac={frac:.2f} Ra={Ra:8.1f} g_hat_lat={r['gamma_hat']:+.4f} "
            f"g_hat_cont={cont['sigma']:+.4f} rel={100*(r['gamma_hat']/cont['sigma']-1):+.2f}%",
            flush=True,
        )

    # --- non-linear cross-check --------------------------------------------
    checks = []
    for row in sorted(arms, key=lambda a: -abs(a["gamma_hat_lattice"]))[: max(0, args.sim_check)]:
        q = p.with_Ra(row["Ra"])
        t1 = time.time()
        sim = measure_growth_rate(q, steps=args.sim_steps, amp=1e-7, sample_every=50)
        checks.append(
            {
                "Ra": row["Ra"],
                "gamma_step_eig": row["gamma_step"],
                "gamma_step_sim": sim["gamma_sim"],
                "rel_err_sim_vs_eig": sim["gamma_sim"] / row["gamma_step"] - 1.0
                if row["gamma_step"] != 0.0
                else float("nan"),
                "fit_rms": sim["fit_rms"],
                "fit_window": sim["fit_window"],
                "sim_steps": sim["steps"],
                "elapsed_s": round(time.time() - t1, 1),
            }
        )
        print(
            f"[rb] sim check Ra={row['Ra']:.1f}: eig {row['gamma_step']:+.4e} "
            f"sim {sim['gamma_sim']:+.4e} rel {100*checks[-1]['rel_err_sim_vs_eig']:+.2f}%",
            flush=True,
        )

    out = {
        "node": "IN31-N12",
        "part": "rb_discrete_linear_stability",
        "params": {
            "ny": args.ny,
            "nx": nx,
            "tau": p.tau,
            "tau_T": p.tau_T,
            "nu": p.nu,
            "alpha": p.alpha,
            "Pr": p.Pr,
            "H": p.H,
            "a_resolved": p.a,
            "eps_fd": args.eps,
            "arm_definition": "Ra = frac * Ra_c^lat",
        },
        "Ra_c_lattice": ra_c_lat,
        "Ra_c_continuum": RA_CRIT_CONTINUOUS,
        "Ra_c_rel_shift": ra_c_lat / RA_CRIT_CONTINUOUS - 1.0,
        "bisection_history": history,
        "arms": arms,
        "sim_checks": checks,
        "base_residual": base.residual,
        "elapsed_s": round(time.time() - t0, 1),
    }
    dest = Path(args.out) if args.out else Path(f"artifacts/in31_n12/rb_lattice_ny{args.ny}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"[rb] wrote {dest} in {out['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
