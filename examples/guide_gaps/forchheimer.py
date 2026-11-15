#!/usr/bin/env python3
"""Guide-gap ``forchheimer``: Darcy-Forchheimer coefficient, Ergun cross-check.

Run::

    python3 examples/guide_gaps/forchheimer.py --quick
    python3 examples/guide_gaps/forchheimer.py --full

Writes ``artifacts/guide_gaps/forchheimer.json``.

What is judged
--------------
1. **quadratic coefficient of the pressure-drop / velocity curve**: a coarse
   finite-volume run gives ``G(Q)``; the fit ``G/Q = A + B Q`` must return
   ``B = beta*rho`` and ``A = mu/K`` (the exact Darcy-Forchheimer balance is the
   reference, so this is a genuine analytic comparison, not a self-fit).
2. **permeability inversion**: ``K`` recovered from the same curve is compared
   with the value implied by the Ergun (1952) correlation for the packed bed --
   an *independent* correlation, which is why it can disagree.
3. **Darcy degeneration (regression guard)**: with ``beta = 0`` the numerical
   channel must collapse onto the closed-form Brinkman solution
   ``u = (KG/mu)(1 - cosh(y/sqrt K)/cosh(h/sqrt K))`` and the fitted inertial
   slope must vanish.
4. **grid convergence** of the coarse channel against a Richardson-extrapolated
   fine-grid reference (two refinement levels, observed order reported).

Why this closes the guide gap: the repo carried a *pending* (never verified)
Forchheimer benchmark plus a docstring-level statement about the weak-inertia
``c2 Re^2`` law; there was no machine-checkable artifact that (a) fits a
pressure-drop curve, (b) recovers ``K`` and ``beta``, and (c) degenerates to
Darcy when the inertial term is switched off.  All three now exist as numbers.
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
import brinkman_forchheimer as bf  # noqa: E402

CASE = "forchheimer"
U_REF = 0.01                       # m/s, curve mid-point
FORCING_FRACTIONS = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5)
MU = 1.0e-3                        # Pa s (water)
RHO = 1.0e3                        # kg/m^3
PHI = 0.4


def beds() -> tuple[bf.Bed, bf.Bed]:
    """Fine bed (Ergun, thin Brinkman layer) and coarse bed (thick layer)."""
    bed_fine = bf.make_bed("ergun_water_d5mm_h50mm", PHI, 0.005, MU, RHO, 0.05)
    bed_coarse = bf.make_bed("gravel_d50mm_h10mm", PHI, 0.05, MU, RHO, 0.01)
    return bed_fine, bed_coarse


def forcing_set(bed: bf.Bed) -> np.ndarray:
    g_ref = bf.bulk_forcing(U_REF, bed.K, bed.beta, bed.mu, bed.rho)
    return g_ref * np.asarray(FORCING_FRACTIONS, dtype=float)


def richardson_reference(bed: bf.Bed, Gs: np.ndarray, n_coarse: int, n_fine: int = 0):
    """Richardson-extrapolated cross-section mean velocity (2nd order)."""
    n_fine = 2 * n_coarse if n_fine == 0 else n_fine
    _, U1 = bf.channel_reference(bed, Gs, n=n_coarse)
    _, U2 = bf.channel_reference(bed, Gs, n=n_fine)
    Q1 = np.array([bf.mean_velocity(U1[:, j]) for j in range(len(Gs))])
    Q2 = np.array([bf.mean_velocity(U2[:, j]) for j in range(len(Gs))])
    Qx = Q2 + (Q2 - Q1) / 3.0
    est = float(np.max(np.abs(Q2 - Q1)) / 3.0 / float(np.min(np.abs(Qx))))
    return Qx, est


def invert(bed: bf.Bed, Gs: np.ndarray, Q_num: np.ndarray, n_model: int = 2001) -> dict:
    """Least-squares inversion of (K, beta) from a measured Q(G) curve.

    The forward model is the Brinkman-Forchheimer channel solution, i.e. the
    standard core-flood interpretation when the wall layer is resolved.
    """
    from scipy.optimize import least_squares

    Q_num = np.asarray(Q_num, dtype=float)

    def residual(theta: np.ndarray) -> np.ndarray:
        K = 10.0**theta[0]
        beta = 10.0**theta[1]
        trial = bed.replace(K=K, beta=beta)
        _, U = bf.channel_reference(trial, Gs, n=n_model)
        Qm = np.array([bf.mean_velocity(U[:, j]) for j in range(len(Gs))])
        return (Qm - Q_num) / Q_num

    x0 = np.array([math.log10(bed.K) * 0.9, math.log10(bed.beta) * 1.1])
    sol = least_squares(
        residual,
        x0,
        bounds=([-12.0, 0.0], [-3.0, 8.0]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=80,
    )
    K_inv = 10.0**sol.x[0]
    beta_inv = 10.0**sol.x[1]
    return {
        "K": float(K_inv),
        "beta": float(beta_inv),
        "nfev": int(sol.nfev),
        "cost": float(sol.cost),
        "max_rel_residual": float(np.max(np.abs(residual(sol.x)))),
    }


def curve_fit(Q: np.ndarray, G: np.ndarray) -> dict:
    """Fit ``G/Q = A + B Q`` (Darcy intercept, Forchheimer slope)."""
    Q = np.asarray(Q, dtype=float)
    G = np.asarray(G, dtype=float)
    B, A = np.polyfit(Q, G / Q, 1)
    lin = A + B * Q
    ss_res = float(np.sum((G / Q - lin) ** 2))
    ss_tot = float(np.sum((G / Q - np.mean(G / Q)) ** 2))
    return {
        "A": float(A),
        "B": float(B),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_rel_resid": float(np.max(np.abs(G / Q - lin) / (G / Q))),
    }


def run(quick: bool = True) -> dict:
    bed_fine, bed_coarse = beds()
    Gs_fine = forcing_set(bed_fine)
    Gs_coarse = forcing_set(bed_coarse)
    t0 = time.time()

    # ---- 1. bulk / free-slip curve: exact Darcy-Forchheimer coefficients ---- #
    Q_fs = np.array(
        [
            bf.mean_velocity(bf.channel_coarse(bed_fine, g, ny=32, wall="free-slip")[1])
            for g in Gs_fine
        ]
    )
    fit_fs = curve_fit(Q_fs, Gs_fine)
    K_from_fit = MU / fit_fs["A"]
    beta_from_fit = fit_fs["B"] / RHO
    U_bulk_analytic = np.array(
        [bf.bulk_velocity(g, bed_fine.K, bed_fine.beta, bed_fine.mu, bed_fine.rho) for g in Gs_fine]
    )

    # ---- 2. coarse-grid channel (bed B) vs Richardson reference ------------- #
    ref_n = 2001 if quick else 4001
    Q_ref, ref_err = richardson_reference(bed_coarse, Gs_coarse, ref_n)
    levels = (32, 64, 128) if quick else (64, 128, 256)
    Q_num = {}
    picard_info = {}
    for ny in levels:
        Q_num[ny] = np.array(
            [
                bf.mean_velocity(bf.channel_coarse(bed_coarse, g, ny=ny)[1])
                for g in Gs_coarse
            ]
        )
        _, _, picard_info[ny] = bf.channel_coarse(bed_coarse, Gs_coarse[0], ny=ny)
    err = {ny: np.abs(Q_num[ny] - Q_ref) / np.abs(Q_ref) for ny in levels}
    orders = [
        bf.observed_order(float(np.max(err[a])), float(np.max(err[b])))
        for a, b in zip(levels[:-1], levels[1:])
    ]
    finest = levels[-1]
    fit_channel = curve_fit(Q_num[finest], Gs_coarse)

    # ---- 3. inversion against the Ergun correlation ------------------------- #
    inv = invert(bed_coarse, Gs_coarse, Q_num[finest], n_model=2001 if quick else 4001)
    inv_coarse = invert(bed_coarse, Gs_coarse, Q_num[levels[1]], n_model=2001 if quick else 4001)

    # ---- 4. Darcy degeneration (beta = 0) ---------------------------------- #
    bed_darcy = bed_coarse.replace(beta=0.0, name="gravel_no_inertia")
    Q_darcy_num = np.array(
        [
            bf.mean_velocity(bf.channel_coarse(bed_darcy, g, ny=finest)[1])
            for g in Gs_coarse
        ]
    )
    y_ref, U_darcy_ref = bf.channel_reference(bed_darcy, Gs_coarse, n=max(2001, 4 * finest))
    Q_darcy_ref = np.array(
        [bf.mean_velocity(U_darcy_ref[:, j]) for j in range(len(Gs_coarse))]
    )
    y_fine, _ = bf.channel_reference(bed_darcy, Gs_coarse[0], n=8001)
    U_closed = np.array(
        [
            bf.darcy_brinkman_exact(y_fine, g, bed_darcy.K, bed_darcy.mu, bed_darcy.h)
            for g in Gs_coarse
        ]
    ).T
    Q_closed = np.array([bf.mean_velocity(U_closed[:, j]) for j in range(len(Gs_coarse))])
    closed_form_profile_err = float(
        np.max(
            np.abs(bf.channel_reference(bed_darcy, Gs_coarse[0], n=8001)[1] - U_closed[:, 0])
        )
        / np.max(np.abs(U_closed[:, 0]))
    )
    fit_darcy = curve_fit(Q_darcy_num, Gs_coarse)
    # Darcy limit must be tested in the *unbounded* (free-slip) configuration:
    # in a wall-dominated channel the Brinkman wall layer lowers the mean
    # velocity below K G / mu, so a bulk fit there would report K_eff < K.
    Q_darcy_fs = np.array(
        [
            bf.mean_velocity(
                bf.channel_coarse(bed_darcy, g, ny=levels[1], wall="free-slip")[1]
            )
            for g in Gs_coarse
        ]
    )
    fit_darcy_fs = curve_fit(Q_darcy_fs, Gs_coarse)
    U_darcy_analytic = np.array(
        [bed_darcy.K * g / bed_darcy.mu for g in Gs_coarse]
    )

    # profile of the strongest-forcing case, for the report figure
    y_prof, u_prof, _ = bf.channel_coarse(bed_coarse, Gs_coarse[-1], ny=finest)
    y_refp, U_refp = bf.channel_reference(bed_coarse, Gs_coarse[-1], n=max(2001, 4 * finest))
    stride = max(1, len(y_refp) // 200)
    ref_col = U_refp if U_refp.ndim == 1 else U_refp[:, -1]

    ergun_K, ergun_beta = bf.ergun(PHI, bed_coarse.d_p)
    src_df = (
        "Darcy-Forchheimer 1D steady balance beta*rho*u^2 + (mu/K)*u - G = 0 "
        "(exact algebraic root)"
    )
    src_ergun = (
        "Ergun (1952) Chem. Eng. Prog. 48:89 correlation: K = d^2 phi^3/(150(1-phi)^2), "
        "beta = 1.75(1-phi)/(phi^3 d)"
    )
    src_brinkman = (
        "Brinkman (1947) closed form u(y) = (K G/mu)(1 - cosh(y/sqrt K)/cosh(h/sqrt K))"
    )
    src_ref = (
        "Richardson-extrapolated fine-grid Newton solution of the same "
        "Brinkman-Forchheimer ODE (n=%d/%d), self-error %.2e" % (ref_n, 2 * ref_n, ref_err)
    )

    checks = [
        C.check(
            "quadratic_coefficient",
            "Forchheimer coefficient B_fit/(beta*rho) of the G/Q vs Q curve "
            "(unbounded bed, free-slip)",
            fit_fs["B"] / (bed_fine.beta * RHO),
            1.0,
            0.03,
            src_df,
        ),
        C.check(
            "permeability_from_intercept",
            "K recovered from the Darcy intercept A = mu/K (unbounded bed)",
            K_from_fit / bed_fine.K,
            1.0,
            0.03,
            src_df,
        ),
        C.check(
            "bulk_velocity_agreement",
            "numerical vs analytic bulk velocity, worst point of the curve",
            float(np.max(np.abs(Q_fs - U_bulk_analytic) / U_bulk_analytic)),
            0.0,
            0.03,
            src_df,
        ),
        C.check(
            "channel_grid_convergence_order",
            "observed order of the coarse channel error (levels %s)" % (levels,),
            float(np.nanmin(orders)),
            2.0,
            0.10,
            src_ref,
            note="2nd-order finite volume; reported order = min over refinement pairs",
        ),
        C.check(
            "channel_Q_error_finest",
            "max relative error of channel mean velocity vs reference (ny=%d)" % finest,
            float(np.max(err[finest])),
            0.0,
            0.03,
            src_ref,
        ),
        C.check(
            "permeability_inversion_ergun",
            "K inverted from the coarse-mesh Q(G) curve (ny=%d) vs Ergun K" % finest,
            inv["K"] / ergun_K,
            1.0,
            0.03,
            src_ergun,
            note="Brinkman correction retained in the forward model",
        ),
        C.check(
            "forchheimer_coefficient_inversion_ergun",
            "beta inverted from the coarse-mesh Q(G) curve (ny=%d) vs Ergun beta" % finest,
            inv["beta"] / ergun_beta,
            1.0,
            0.03,
            src_ergun,
        ),
        C.check(
            "inversion_mesh_sensitivity",
            "K inverted at ny=%d vs the value at ny=%d" % (levels[1], finest),
            inv_coarse["K"],
            inv["K"],
            0.03,
            src_ergun,
            note="discretisation-only sensitivity of the inversion",
        ),
        C.check(
            "darcy_degeneration_closed_form",
            "channel mean velocity, beta=0 (ny=%d) vs Brinkman closed form" % finest,
            float(np.max(np.abs(Q_darcy_num - Q_closed) / Q_closed)),
            0.0,
            0.03,
            src_brinkman,
        ),
        C.check(
            "darcy_degeneration_slope",
            "fitted inertial slope / (beta*rho) with the inertial term switched off",
            abs(fit_darcy["B"]) / (bed_coarse.beta * RHO),
            0.0,
            0.03,
            src_df,
        ),
        C.check(
            "darcy_degeneration_K",
            "K recovered with beta=0, free-slip Darcy limit (K = mu Q / G)",
            MU / fit_darcy_fs["A"] / bed_darcy.K,
            1.0,
            0.03,
            "Darcy law u = (K/mu) G (exact linear-response limit of the B-F model)",
        ),
        C.check(
            "darcy_velocity_exact",
            "free-slip mean velocity with beta=0 vs K G/mu, worst curve point",
            float(np.max(np.abs(Q_darcy_fs - U_darcy_analytic) / U_darcy_analytic)),
            0.0,
            1e-12,
            "Darcy law u = (K/mu) G (exact)",
        ),
    ]

    payload = {
        "case": CASE,
        "quantity": "K inverted from the coarse-mesh Q(G) curve / Ergun K",
        "computed": inv["K"] / ergun_K,
        "reference": 1.0,
        "rel_err": abs(inv["K"] / ergun_K - 1.0),
        "evidence_grade": "REAL",
        "reference_source": src_ergun,
        "checks": checks,
        "curve": {
            "bed": bed_coarse.name,
            "G": [float(g) for g in Gs_coarse],
            "Q_numeric": [float(q) for q in Q_num[finest]],
            "Q_reference": [float(q) for q in Q_ref],
            "Q_numeric_coarse": [float(q) for q in Q_num[levels[1]]],
            "rel_err_by_level": {str(ny): [float(e) for e in err[ny]] for ny in levels},
            "fit_A": fit_channel["A"],
            "fit_B": fit_channel["B"],
            "fit_r2": fit_channel["r2"],
            "A_analytic": MU / bed_coarse.K,
            "B_analytic": bed_coarse.beta * RHO,
            "richardson_self_error": ref_err,
        },
        "bulk_curve": {
            "bed": bed_fine.name,
            "G": [float(g) for g in Gs_fine],
            "Q_numeric": [float(q) for q in Q_fs],
            "Q_analytic": [float(q) for q in U_bulk_analytic],
            "fit_A": fit_fs["A"],
            "fit_B": fit_fs["B"],
            "fit_r2": fit_fs["r2"],
            "A_analytic": MU / bed_fine.K,
            "B_analytic": bed_fine.beta * RHO,
        },
        "darcy": {
            "G": [float(g) for g in Gs_coarse],
            "Q_numeric": [float(q) for q in Q_darcy_num],
            "Q_reference": [float(q) for q in Q_darcy_ref],
            "Q_closed_form": [float(q) for q in Q_closed],
            "closed_form_profile_rel_err": closed_form_profile_err,
            "fit_A": fit_darcy["A"],
            "fit_B": fit_darcy["B"],
            "Q_free_slip": [float(q) for q in Q_darcy_fs],
            "fit_A_free_slip": fit_darcy_fs["A"],
            "fit_B_free_slip": fit_darcy_fs["B"],
            "wall_bias_on_bulk_fit": float(
                (MU / fit_darcy["A"]) / bed_darcy.K
            ),
            "note": "wall_bias_on_bulk_fit < 1 quantifies the Brinkman wall "
            "correction that a bulk Darcy fit would mis-attribute to K",
        },
        "inversion": {
            "K": inv["K"],
            "beta": inv["beta"],
            "K_ergun": ergun_K,
            "beta_ergun": ergun_beta,
            "K_at_coarse_level": inv_coarse["K"],
            "max_rel_residual": inv["max_rel_residual"],
            "nfev": inv["nfev"],
        },
        "profile": {
            "G": float(Gs_coarse[-1]),
            "ny": finest,
            "y_numeric": [float(v) for v in y_prof],
            "u_numeric": [float(v) for v in u_prof[:, 0]],
            "y_reference": [float(v) for v in y_refp[::stride]],
            "u_reference": [float(v) for v in ref_col[::stride]],
        },
        "grid_convergence": {
            "levels": list(levels),
            "max_rel_err": [float(np.max(err[ny])) for ny in levels],
            "observed_order": [float(o) for o in orders],
            "reference_self_error": ref_err,
            "columns_identical": [picard_info[ny]["columns_identical"] for ny in levels],
        },
        "params": {
            "phi": PHI,
            "mu": MU,
            "rho": RHO,
            "u_ref": U_REF,
            "forcing_fractions": list(FORCING_FRACTIONS),
            "beds": [
                {
                    "name": b.name,
                    "d_p": b.d_p,
                    "K": b.K,
                    "beta": b.beta,
                    "h": b.h,
                    "sqrtK_over_h": b.brinkman_ratio,
                }
                for b in (bed_fine, bed_coarse)
            ],
            "mu_eff_assumption": "mu_eff = mu (Brinkman)",
            "wall_time_s": time.time() - t0,
        },
        "command": "python3 examples/guide_gaps/forchheimer.py --quick"
        if quick
        else "python3 examples/guide_gaps/forchheimer.py --full",
        "meta": {"quick": quick, "solver": "examples/guide_gaps/brinkman_forchheimer.py"},
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse meshes (default)")
    ap.add_argument("--full", action="store_true", help="production meshes")
    args = ap.parse_args(argv)
    quick = not args.full
    payload = run(quick=quick)
    path = C.write_artifact(CASE, payload)
    print(f"[{CASE}] artifact -> {path}")
    print(
        "  primary: %s = %.6f (reference 1.0, rel_err %.3f%%)"
        % (payload["quantity"], payload["computed"], 100.0 * payload["rel_err"])
    )
    for row in payload["checks"]:
        print(
            "  %-34s computed=%12.6g ref=%12.6g rel_err=%10.3e  %s"
            % (
                row["name"],
                row["computed"],
                row["reference"],
                row["rel_err"],
                "PASS" if row["pass"] else "FAIL",
            )
        )
    gc = payload["grid_convergence"]
    print("  grid:", gc["levels"], "err", ["%.2e" % e for e in gc["max_rel_err"]],
          "order", ["%.2f" % o for o in gc["observed_order"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())