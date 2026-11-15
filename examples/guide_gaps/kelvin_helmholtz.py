#!/usr/bin/env python3
"""Guide-gap ``kelvin_helmholtz``: numerical growth rate vs the inviscid dispersion relation.

Run::

    python3 examples/guide_gaps/kelvin_helmholtz.py --quick
    python3 examples/guide_gaps/kelvin_helmholtz.py --full

Writes ``artifacts/guide_gaps/kelvin_helmholtz.json``.

What is judged
--------------
1. **growth rate** ``omega_i(k)`` of the discretized linear two-layer problem
   (Chebyshev collocation of ``Phi'' = k^2 Phi`` in each layer + interface
   kinematic/dynamic conditions, see :mod:`kelvin_helmholtz_solver`) against the
   inviscid linear dispersion relation with surface tension, at >= 3 wavenumbers;
2. **phase speed** ``Re(omega)/k`` against ``Ubar``;
3. **grid convergence** over the number of Chebyshev points (spectral: the error
   ratio must be far better than the 2nd-order expectation);
4. **surface-tension cutoff** ``k_c`` -- above it the flow must come out stable;
5. **degenerate limit** ``sigma = 0`` (pure Kelvin-Helmholtz, and the equal-density
   case ``omega_i = k dU / 2``) as a regression guard on the surface-tension term;
6. **most unstable wavenumber** from a numerical ``k`` scan against ``2 k_c/3``.

Scope (honest): this closes the *linear* stability side of the guide entry.  A
nonlinear two-phase (LBM) Kelvin-Helmholtz growth-rate benchmark -- the other
half of "KH capability" -- is **not** delivered here; see the report.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _common as C  # noqa: E402
import kelvin_helmholtz_solver as kh  # noqa: E402

CASE = "kelvin_helmholtz"
RHO1, RHO2 = 1.0, 0.5
U1, U2 = 1.0, -1.0
SIGMA = 0.02
K_LIST = (2.0, 5.0, 10.0, 20.0, 30.0)
K_LIST_QUICK = (2.0, 10.0, 20.0)


def standard_layer() -> kh.LayerKH:
    return kh.LayerKH(rho1=RHO1, rho2=RHO2, U1=U1, U2=U2, sigma=SIGMA)


def run(quick: bool = True) -> dict:
    t0 = time.time()
    layer = standard_layer()
    ks = K_LIST_QUICK if quick else K_LIST
    n_main = 24

    rows = []
    for k in ks:
        omega, diag = kh.dispersion_numeric(layer, k, n=n_main, return_all=True)
        ana = layer.analytic(k)
        rows.append(
            {
                "k": k,
                "omega_real": omega.real,
                "omega_imag": omega.imag,
                "omega_real_analytic": ana.real,
                "omega_imag_analytic": ana.imag,
                "growth_rel_err": C.rel_err(omega.imag, ana.imag),
                "phase_speed": omega.real / k,
                "phase_speed_analytic": ana.real / k,
                "phase_rel_err": C.rel_err(omega.real / k, ana.real / k),
                "n": n_main,
                "L": diag["L"],
            }
        )

    # ---- grid convergence (spectral) --------------------------------------- #
    ns = (8, 12, 16, 24) if quick else (8, 12, 16, 20, 24, 32)
    conv = []
    for n in ns:
        omega, _ = kh.dispersion_numeric(layer, 10.0, n=n)
        conv.append({"n": n, "growth": omega.imag, "rel_err": C.rel_err(omega.imag, layer.analytic_growth(10.0))})
    orders = [
        kh.observed_order(conv[i]["rel_err"], conv[i + 1]["rel_err"], r=conv[i + 1]["n"] / conv[i]["n"])
        for i in range(len(conv) - 1)
    ]

    # ---- far-field truncation cross-check ---------------------------------- #
    omega_far, _ = kh.dispersion_numeric(layer, 10.0, n=32, L=35.0 / 10.0)
    far_err = C.rel_err(omega_far.imag, layer.analytic_growth(10.0))

    # ---- surface-tension cutoff -------------------------------------------- #
    k_above = 1.2 * layer.k_cutoff
    omega_above, _ = kh.dispersion_numeric(layer, k_above, n=n_main)

    # ---- sigma = 0 degenerate branches ------------------------------------- #
    layer0 = kh.LayerKH(rho1=RHO1, rho2=RHO2, U1=U1, U2=U2, sigma=0.0)
    omega0, _ = kh.dispersion_numeric(layer0, 5.0, n=20)
    ana0 = layer0.analytic(5.0)
    layer_eq = kh.LayerKH(rho1=1.0, rho2=1.0, U1=U1, U2=U2, sigma=0.0)
    omega_eq, _ = kh.dispersion_numeric(layer_eq, 7.0, n=20)
    eq_analytic = 7.0 * abs(U1 - U2) / 2.0

    # ---- most unstable wavenumber ------------------------------------------ #
    k_scan, g_scan = kh.max_growth(layer, n=16, search=200)
    k_scan_fine, g_scan_fine = _refine_kmax(layer, k_scan, n=20)

    src = (
        "Inviscid two-layer Kelvin-Helmholtz dispersion relation "
        "(Chandrasekhar 1961 Sec.101; Drazin & Reid 1981 Sec.22): "
        "omega = k Ubar +/- i k sqrt(rho1 rho2 dU^2/(rho1+rho2)^2 - sigma k/(rho1+rho2))"
    )
    checks = []
    for row in rows:
        checks.append(
            C.check(
                "growth_rate_k%g" % row["k"],
                "numerical growth rate Im(omega) at k=%.3g" % row["k"],
                row["omega_imag"],
                row["omega_imag_analytic"],
                0.03,
                src,
            )
        )
    checks += [
        C.check(
            "phase_speed",
            "phase speed Re(omega)/k, worst over k in %s" % (list(ks),),
            max(abs(r["phase_speed"] - r["phase_speed_analytic"]) / abs(r["phase_speed_analytic"]) for r in rows),
            0.0,
            0.03,
            src,
        ),
        C.check(
            "growth_rate_worst",
            "worst growth-rate relative error over k in %s" % (list(ks),),
            max(r["growth_rel_err"] for r in rows),
            0.0,
            0.03,
            src,
        ),
        C.check(
            "spectral_convergence_shortfall",
            "shortfall of the observed convergence order against the 2nd-order "
            "expectation (0 = the scheme is at least 2nd order in the mesh size; "
            "Chebyshev collocation is expected to be spectral)",
            max(0.0, 2.0 - float(np.nanmin(orders))),
            0.0,
            1e-12,
            "Chebyshev collocation of an analytic eigenfunction -> exponential "
            "convergence; observed orders %s" % (["%.2f" % o for o in orders],),
            note="error ratio n=%d -> %d is %.3g" % (ns[-2], ns[-1], conv[-2]["rel_err"] / conv[-1]["rel_err"]),
        ),
        C.check(
            "surface_tension_cutoff",
            "growth rate at k=1.2 k_c (analytically stable, Im omega = 0)",
            max(float(omega_above.imag), 0.0),
            0.0,
            1e-6,
            src,
            note="k_c = %.4f, longest surface-tension-stabilised wavelength" % layer.k_cutoff,
        ),
        C.check(
            "sigma_zero_branch",
            "growth rate with sigma=0 at k=5 vs k |dU| sqrt(rho1 rho2)/(rho1+rho2)",
            omega0.imag,
            ana0.imag,
            0.03,
            src,
            note="regression guard on the surface-tension term",
        ),
        C.check(
            "equal_density_limit",
            "growth rate with rho1=rho2, sigma=0, k=7 vs k |dU|/2",
            omega_eq.imag,
            eq_analytic,
            0.03,
            "vortex-sheet limit of the same relation (rho1 = rho2, sigma = 0)",
        ),
        C.check(
            "most_unstable_wavenumber",
            "numerical argmax_k Im(omega) vs 2 k_c/3",
            k_scan_fine,
            layer.k_most_unstable,
            0.03,
            src,
        ),
        C.check(
            "far_field_truncation",
            "growth-rate error at k=10 with L=35/k, n=32 (shows that L=25/k sets "
            "the e^-25 floor of the main runs)",
            far_err,
            0.0,
            1e-9,
            src,
        ),
    ]

    payload = {
        "case": CASE,
        "quantity": "numerical growth rate Im(omega) at k=10 vs inviscid dispersion relation",
        "computed": next(r["omega_imag"] for r in rows if r["k"] == 10.0),
        "reference": next(r["omega_imag_analytic"] for r in rows if r["k"] == 10.0),
        "rel_err": next(r["growth_rel_err"] for r in rows if r["k"] == 10.0),
        "evidence_grade": "REAL",
        "reference_source": src,
        "checks": checks,
        "dispersion": {
            "rho1": RHO1,
            "rho2": RHO2,
            "U1": U1,
            "U2": U2,
            "sigma": SIGMA,
            "k_cutoff": layer.k_cutoff,
            "k_most_unstable_analytic": layer.k_most_unstable,
            "k_most_unstable_numeric": k_scan_fine,
            "growth_at_kmax_numeric": g_scan_fine,
            "k": [r["k"] for r in rows],
            "omega_imag_numeric": [r["omega_imag"] for r in rows],
            "omega_imag_analytic": [r["omega_imag_analytic"] for r in rows],
            "omega_real_numeric": [r["omega_real"] for r in rows],
            "omega_real_analytic": [r["omega_real_analytic"] for r in rows],
            "rel_err": [r["growth_rel_err"] for r in rows],
            "rows": rows,
        },
        "grid_convergence": {
            "k": 10.0,
            "n": [c["n"] for c in conv],
            "rel_err": [c["rel_err"] for c in conv],
            "observed_order": [float(o) for o in orders],
            "method": "Chebyshev collocation, spectral (exponential) convergence",
        },
        "degenerate": {
            "sigma_zero": {
                "k": 5.0,
                "omega_imag_numeric": omega0.imag,
                "omega_imag_analytic": ana0.imag,
            },
            "equal_density": {
                "k": 7.0,
                "omega_imag_numeric": omega_eq.imag,
                "omega_imag_analytic": eq_analytic,
            },
            "cutoff": {
                "k": k_above,
                "omega_imag_numeric": float(omega_above.imag),
            },
            "far_field": {"L_over_k": 35.0, "n": 32, "rel_err": far_err},
        },
        "params": {
            "n_chebyshev": n_main,
            "L_over_k": 25.0,
            "wall_time_s": time.time() - t0,
        },
        "scope": "linear inviscid dispersion only; nonlinear two-phase LBM "
        "growth-rate benchmark not delivered (documented in docs/guide_gaps/REPORT.md)",
        "command": "python3 examples/guide_gaps/kelvin_helmholtz.py --quick"
        if quick
        else "python3 examples/guide_gaps/kelvin_helmholtz.py --full",
        "meta": {"quick": quick, "solver": "examples/guide_gaps/kelvin_helmholtz_solver.py"},
    }
    return payload


def _refine_kmax(layer: kh.LayerKH, k0: float, n: int = 20) -> tuple[float, float]:
    """Parabolic refinement of the growth-rate maximum around ``k0``."""
    lo, hi = 0.6 * k0, 1.4 * k0
    ks = np.linspace(lo, hi, 81)
    gs = np.array([kh.growth_rate(layer, float(k), n=n) for k in ks])
    i = int(np.argmax(gs))
    if 0 < i < len(ks) - 1:
        d1 = (gs[i + 1] - gs[i - 1]) / (ks[i + 1] - ks[i - 1])
        d2 = (gs[i + 1] - 2 * gs[i] + gs[i - 1]) / ((ks[i + 1] - ks[i]) ** 2)
        if d2 < 0:
            k_fit = float(ks[i] - d1 / d2)
            if ks[i - 1] <= k_fit <= ks[i + 1]:
                return k_fit, float(gs[i] - 0.5 * d1**2 / d2)
    return float(ks[i]), float(gs[i])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer wavenumbers (default)")
    ap.add_argument("--full", action="store_true", help="full sweep")
    args = ap.parse_args(argv)
    quick = not args.full
    payload = run(quick=quick)
    path = C.write_artifact(CASE, payload)
    print(f"[{CASE}] artifact -> {path}")
    print(
        "  primary: %s = %.8f (reference %.8f, rel_err %.3e)"
        % (payload["quantity"], payload["computed"], payload["reference"], payload["rel_err"])
    )
    for row in payload["checks"]:
        print(
            "  %-28s computed=%14.8g ref=%14.8g rel_err=%10.3e  %s"
            % (row["name"], row["computed"], row["reference"], row["rel_err"],
               "PASS" if row["pass"] else "FAIL")
        )
    gc = payload["grid_convergence"]
    print("  spectral convergence:", list(zip(gc["n"], ["%.2e" % e for e in gc["rel_err"]])))
    print("  observed orders:", ["%.2f" % o for o in gc["observed_order"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())