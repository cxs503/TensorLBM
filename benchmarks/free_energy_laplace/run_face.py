"""Young-Laplace activation face for the D3Q19 free-energy Cahn-Hilliard path.

Runner for :mod:`tensorlbm.phasefield.laplace_face`.  Three blocks:

1. ``analytic``  - the coefficient formulas (``phi0``, ``xi``, two independent
   routes to ``sigma``) plus the discretisation error of the *field* route
   (:func:`sigma_from_profile`) on the analytic profile, so the field route has
   a known error bar before it is used on a simulated field;
2. ``quench``    - fixed ``phi`` (momentum only).  The capillary pressure is a
   property of the force balance at fixed order parameter, so this block asks the
   Young-Laplace question on its own terms and cannot be contaminated by
   Cahn-Hilliard transport;
3. ``transport`` - live ``phi`` (the production two-equation step).  Measures
   whether the two-phase structure is stationary at all.
4. ``stability`` - (``kappa``, ``Gamma``, ``tau_g``) screening, so a Laplace run
   is never spent on a diverging point.

Usage::

    PYTHONPATH=src python benchmarks/free_energy_laplace/run_face.py [--quick]

Artifacts: ``artifacts/free_energy_laplace/results.json`` (every number quoted in
``docs/FREE_ENERGY_KORTEWEG_LAPLACE_20261112.md`` is rendered from it).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.phasefield.laplace_face import (
    analytic_reference,
    analytic_sigma_gradient_integral,
    radial_setup,
    run_static_droplet,
    sigma_from_profile,
    stability_probe,
    tanh_profile,
)

A = 0.1
B = 0.1


def block_analytic() -> dict:
    out: dict = {"A": A, "B": B, "cases": [], "field_route_on_analytic": []}
    for kappa in (0.02, 0.05, 0.1, 0.2):
        ref = analytic_reference(A, B, kappa)
        ref["sigma_gradient_integral"] = analytic_sigma_gradient_integral(A, B, kappa)
        ref["sigma_gradient_rel_residual"] = (
            abs(ref["sigma_gradient_integral"] - ref["sigma"]) / ref["sigma"]
        )
        ref["kappa"] = kappa
        out["cases"].append(ref)
    # discretisation error of the field route: analytic tanh on a radial cut
    n = 64
    r, centre = radial_setup(n)
    for kappa in (0.02, 0.05, 0.1, 0.2):
        ref = analytic_reference(A, B, kappa)
        # keep the interface away from the periodic seam of the cut (R = n/4)
        phi = tanh_profile(r, R=n / 4.0, phi0=ref["phi0"], xi=ref["xi"])
        measured = sigma_from_profile(phi, centre=centre, kappa=kappa)
        out["field_route_on_analytic"].append(
            {
                "kappa": kappa,
                "xi": ref["xi"],
                "xi_over_dx": ref["xi"],
                "sigma_analytic": ref["sigma"],
                "sigma_measured": measured,
                "rel_error": abs(measured - ref["sigma"]) / ref["sigma"],
            }
        )
    return out


def block_quench(quick: bool) -> list[dict]:
    cases = []
    n = 20 if quick else 24
    steps = 400 if quick else 800
    for kappa in (0.05, 0.2):
        for R in ((4.0, 5.0, 6.0) if quick else (5.0, 6.0, 8.0)):
            t0 = time.time()
            res = run_static_droplet(
                n=n,
                R=R,
                A=A,
                B=B,
                kappa=kappa,
                Gamma=0.1,
                tau_g=0.85,
                steps=steps,
                check_every=steps // 4,
                freeze_phi=True,
            )
            res["wall_seconds"] = round(time.time() - t0, 1)
            cases.append(res)
    return cases


def block_transport(quick: bool) -> list[dict]:
    cases = []
    n = 20 if quick else 24
    steps = 300 if quick else 600
    for kappa, Gamma, tau_g in ((0.05, 0.1, 0.85), (0.05, 0.02, 0.85), (0.02, 0.1, 0.85)):
        t0 = time.time()
        res = run_static_droplet(
            n=n,
            R=6.0,
            A=A,
            B=B,
            kappa=kappa,
            Gamma=Gamma,
            tau_g=tau_g,
            steps=steps,
            check_every=steps // 4,
            freeze_phi=False,
        )
        res["wall_seconds"] = round(time.time() - t0, 1)
        cases.append(res)
    return cases


def block_stability(quick: bool) -> list[dict]:
    rows = []
    steps = 60 if quick else 100
    for kappa in (0.02, 0.05, 0.1, 0.2):
        for tau_g in (0.7, 0.85):
            for Gamma in (0.05, 0.1):
                rows.append(
                    stability_probe(
                        n=20 if quick else 24,
                        R=6.0,
                        A=A,
                        B=B,
                        kappa=kappa,
                        Gamma=Gamma,
                        tau_g=tau_g,
                        steps=steps,
                    )
                )
    return rows


def summarise(payload: dict) -> dict:
    """Aggregate the quench block into a one-parameter sigma fit per kappa."""
    fits = []
    for kappa in sorted({c["config"]["kappa"] for c in payload["quench"]}):
        rows = [c for c in payload["quench"] if c["config"]["kappa"] == kappa]
        xs, ys, ids = [], [], []
        for c in rows:
            m = c["metrics"]
            if m is None or m["dp_measured"] != m["dp_measured"]:
                continue
            xs.append(1.0 / c["config"]["R"])
            ys.append(m["dp_measured"])
            ids.append(c["config"]["R"])
        if not xs:
            fits.append({"kappa": kappa, "n_points": 0})
            continue
        # dp = sigma / R  -> one-parameter least squares through the origin
        sigma_fit = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        sigma_analytic = analytic_reference(A, B, kappa)["sigma"]
        residuals = [
            abs(y - sigma_fit * x) / (sigma_fit * x) for x, y in zip(xs, ys)
        ]
        fits.append(
            {
                "kappa": kappa,
                "radii": ids,
                "sigma_fit": sigma_fit,
                "sigma_analytic": sigma_analytic,
                "sigma_fit_rel_error": abs(sigma_fit - sigma_analytic) / sigma_analytic,
                "max_point_residual": max(residuals),
                "n_points": len(xs),
            }
        )
    return {"quench_sigma_fits": fits}


def main() -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--out",
        default="artifacts/free_energy_laplace/results.json",
    )
    args = ap.parse_args()

    payload = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "torch": torch.__version__,
            "quick": bool(args.quick),
            "A": A,
            "B": B,
            "model": "D3Q19 double-well free-energy (Cahn-Hilliard) step",
            "production_step": "multiphase3d.free_energy_step_3d + solver3d.stream3d",
            "note": (
                "uniform density (rho0 = 1) so p = cs2 rho and dp has no "
                "density-ratio contribution"
            ),
        },
        "analytic": block_analytic(),
    }
    payload["quench"] = block_quench(args.quick)
    payload["transport"] = block_transport(args.quick)
    payload["stability"] = block_stability(args.quick)
    payload.update(summarise(payload))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {out}")
    for fit in payload["quench_sigma_fits"]:
        print(json.dumps(fit, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())