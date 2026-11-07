#!/usr/bin/env python
"""IN31-N12 part 2: viscosity-ratio / density-ratio decoupling and the
pseudopotential drop-settling comparison against Hadamard-Rybczynski.

Two layers are run and both write numbers only (no hand copying):

* ``decoupling``: for a grid of ``(lam, rho_d/rho_c)`` pairs the relaxation times
  are obtained from ``decoupled_relaxation`` and the viscosity each phase actually
  realises is *measured* with a periodic shear-wave decay; the measured
  ``lam = mu_d/mu_c`` is compared with the prescribed one.
* ``settling``: a three-dimensional D3Q19 two-component Shan-Chen drop settles
  under gravity (drop component only, with a momentum-neutral compensating force on
  the ambient component) and the terminal relative velocity is compared with the
  Hadamard-Rybczynski law at the *measured* radius and effective buoyancy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.pseudopotential_hr import (  # noqa: E402
    DropSetup,
    run_settling,
    verify_decoupling,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "decoupling", "settling"])
    ap.add_argument("--lams", default="0.3,1.0,3.0")
    ap.add_argument("--density-ratios", default="0.5,1.0,2.0")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--R", type=float, default=5.0)
    ap.add_argument("--tau-c", type=float, default=0.8)
    ap.add_argument("--tau-c-decoupling", type=float, default=0.6)
    ap.add_argument("--G12", type=float, default=-1.5)
    ap.add_argument("--background", type=float, default=0.3)
    ap.add_argument("--interface", type=float, default=2.0)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--u-eq", default="self")
    ap.add_argument("--out", default="artifacts/in31_n12/hr_pseudopotential.json")
    args = ap.parse_args()

    out = {"node": "IN31-N12", "part": "pseudopotential_hadamard_rybczynski"}
    t0 = time.time()

    if args.mode in ("all", "decoupling"):
        lams = [float(x) for x in args.lams.split(",")]
        ratios = [float(x) for x in args.density_ratios.split(",")]
        rows = verify_decoupling(lams, ratios, tau_c=args.tau_c_decoupling)
        out["decoupling"] = rows
        for r in rows:
            print(
                f"[hr] lam={r['lam']:.3g} rho_d/rho_c={r['density_ratio']:.2f} "
                f"tau_d={r['tau_d']:.3f} lam_measured={r['lam_measured']:.4f} "
                f"rel={100*r['lam_rel_err']:+.2f}%",
                flush=True,
            )
        # an independent check that lambda does not follow the density ratio
        out["decoupling_density_slope"] = _density_slope(rows, lams)

    if args.mode in ("all", "settling"):
        runs = []
        for lam in [float(x) for x in args.lams.split(",")]:
            setup = DropSetup(
                n=args.n,
                R=args.R,
                lam=lam,
                tau_c=args.tau_c,
                G_12=args.G12,
                background=args.background,
                interface=args.interface,
                g=args.g,
                steps=args.steps,
                u_eq=args.u_eq,
                sample_every=max(10, args.steps // 40),
            )
            try:
                res = run_settling(setup)
            except FloatingPointError as exc:
                runs.append({"lam": lam, "status": "diverged", "error": str(exc)})
                print(f"[hr] lam={lam}: diverged ({exc})", flush=True)
                continue
            hist = res.pop("history")
            res["status"] = "ok"
            res["U_rel_series"] = [h["U_rel"] for h in hist]
            res["step_series"] = [h["step"] for h in hist]
            res["R_eff_series"] = [h["R_eff"] for h in hist]
            res["drho_series"] = [h["drho_eff"] for h in hist]
            runs.append(res)
            print(
                f"[hr] lam={lam:.3g} U_rel={res['U_rel']:+.4e} (spread {res['U_rel_spread']:.1e}) "
                f"U_hr={res['U_hr']:+.4e} rel={100*res['rel_err_vs_hr']:+.1f}% "
                f"R_eff={res['R_eff']:.2f} rho2_in={res['rho2_in']:.3f} rho1_out={res['rho1_out']:.3f}",
                flush=True,
            )
        out["settling"] = runs
        ok = [r for r in runs if r.get("status") == "ok"]
        if ok:
            out["settling_summary"] = {
                "n_ok": len(ok),
                "max_abs_rel_err": max(abs(r["rel_err_vs_hr"]) for r in ok),
                "monotone_decreasing_in_lambda": _monotone_in_lambda(ok),
                "spurious_vs_signal": [
                    {
                        "lam": r["setup"]["lam_nominal"],
                        "U_rel": r["U_rel"],
                        "U_rel_spread": r["U_rel_spread"],
                        "ratio": abs(r["U_rel_spread"] / r["U_rel"]) if r["U_rel"] else float("inf"),
                    }
                    for r in ok
                ],
            }

    out["elapsed_s"] = round(time.time() - t0, 1)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"[hr] wrote {dest} in {out['elapsed_s']}s", flush=True)
    return 0


def _density_slope(rows, lams):
    """How much ``lam_measured`` moves when only the density ratio changes.

    A decoupled engine must show ``|d lam_measured / d(rho_d/rho_c)| ~ 0`` at fixed
    prescribed ``lam``; the locked (pre-fix) mapping would show a slope of 1.
    """
    slopes = []
    for lam in lams:
        sel = [r for r in rows if r["lam"] == lam]
        if len(sel) < 2:
            continue
        x = np.array([r["density_ratio"] for r in sel])
        y = np.array([r["lam_measured"] for r in sel])
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    return {"slopes": slopes, "max_abs": max(abs(s) for s in slopes) if slopes else None}


def _monotone_in_lambda(rows):
    """HR ordering: larger lambda (more viscous drop) -> smaller settling speed.

    Compared on the magnitude: the measured relative velocity is dominated by the
    interface spurious current (see the report), so only the *trend* is meaningful.
    """
    vals = sorted(rows, key=lambda r: r["setup"]["lam_nominal"])
    u = [abs(r["U_rel"]) for r in vals]
    return bool(np.all(np.diff(u) < 0.0))


if __name__ == "__main__":
    raise SystemExit(main())