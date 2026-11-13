#!/usr/bin/env python3
"""H2-FRAZIL-RHEOLOGY, particle-resolved face: relative viscosity in a true
moving-wall Couette cell, judged at the *measured* lattice volume fraction.

Why this file exists (audit ``docs/AUDIT_m14882_H2_FRAZIL_RHEOLOGY.md``)
-----------------------------------------------------------------------
The first attempt at this face (m14882) was judged L1 for four reasons, all of
which are addressed here by construction rather than by a looser tolerance:

1. **Wrong cell.** It drove a *body force* through a stationary channel and then
   fitted the shear rate of the resulting Poiseuille parabola, while calling the
   result Couette.  The pure-fluid calibration came out ``gamma_meas/gamma_target
   = 2.37`` at ``R^2 = 10^-6``-clean profiles: a self-consistent but wrongly
   calibrated apparatus.  Here the wall momentum is exchanged through a genuine
   moving-wall :class:`~tensorlbm.ice_suspension_rheology.couette.ShearCell`
   (:class:`~tensorlbm.frazil_powerlaw_mature`), whose ``c = 0`` case has the
   exact solution ``tau = rho nu gamma_dot``; that identity is checked first and
   the run aborts at the criterion rather than reporting a bias.
2. **Illegal ``eta_r`` definition.** It took the ratio of the *nominal* rates at
   equal body force.  Here ``eta_r`` is the ratio of two *measured* stresses at
   their own *measured* rates, each divided by the pure-fluid value at the same
   cell -- a first-order cancellation of the profile-fit bias, and the residual
   is quantified by the pure controls.
3. **Wrong abscissa.** It compared ``eta_r`` against closures evaluated at the
   *declared* ``phi`` (0.05/0.20) while the placed suspension only reached
   0.017-0.061.  Here every comparison uses ``phi_lattice``, the union solids
   fraction of the voxelised suspension referred to the physical cell volume --
   the same fraction the flow actually sees.
4. **No independent anchor.** Here the flat-plate closures are the anchors and
   they are evaluated at ``phi_lattice``; the frozen-configuration assumption
   (crystals advected rigidly, no re-arrangement) is declared, not hidden.

Two stress estimators are carried through the whole campaign (plane momentum
flux in the particle-free wall band, and the per-wall momentum-exchange force).
:func:`~tensorlbm.frazil_powerlaw_mature.select_stress_estimator` picks the
production one from the pure-fluid budget; both are reported at every point.

Machine-checked criteria (thresholds pre-declared in ``CRITERIA`` below)
-----------------------------------------------------------------------
``c1_pure_couette``      both estimators reproduce ``rho nu gamma_dot`` (2 %)
``c2_dilute_einstein``   ``eta_r`` at ``phi_lattice ~ 0.05`` vs ``1 + 2.5 phi``
                         (15 %; the standard tolerance of this node)
``c3_krieger_dougherty`` ``eta_r`` at ``phi_lattice ~ 0.20`` vs KD (15 %)
``c4_rate_independence`` a Newtonian hard-sphere suspension must give the same
                         ``eta_r`` at every shear rate (10 % spread)

Usage (repository root)::

    PYTHONPATH=src python scripts/run_h2_frazil_couette.py --out artifacts/h2_frazil_rheology

Writes ``couette_measurement.json`` and ``fig4_couette_eta_r.png`` next to the
other H2 artifacts.  Exit code 0 only when all criteria pass; a failing run
still writes its artifacts (honest FAIL is data, not a crash).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensorlbm import frazil_powerlaw_mature as face  # noqa: E402

#: Production cell: a physical cube of ``n_xy`` cells plus a two-cell wall band
#: at each wall, and crystals of a fixed physical size (``a_cells`` cells).
BOX = {"n_xy": 16, "n_z": 20, "a_cells": 2.0}
#: Finer cell of the grid-convergence pair (same physical suspension).
BOX_FINE = {"n_xy": 24, "n_z": 28, "a_cells": 3.0}

NU = 0.2                      # lattice kinematic viscosity of the carrier
SEED = 0                      # frozen microstructure; reproducible by design
GAMMA_MAIN = 1.0e-3           # the shear rate the criterion point is measured at
GAMMA_LADDER = (5.0e-4, 1.0e-3, 2.0e-3)   # Re_p = 0.023, 0.045, 0.090 at a = 2.5
PHI_DILUTE = 0.05
PHI_MID = 0.20
PHI_RATE = 0.10               # concentration used for the rate-independence check

#: Pre-declared criteria.  Nothing below is tightened or loosened after the fact.
CRITERIA = {
    "c1_pure_couette": {
        "metric": "max over pure controls of |tau/(rho nu gamma_dot) - 1| (both estimators)",
        "threshold": 0.02,
    },
    "c2_dilute_einstein": {
        "metric": "|eta_r - (1 + 2.5 phi_lattice)| / (1 + 2.5 phi_lattice) at phi ~ 0.05",
        "threshold": 0.15,
    },
    "c3_krieger_dougherty": {
        "metric": "|eta_r - KD(phi_lattice)| / KD(phi_lattice) at phi ~ 0.20",
        "threshold": 0.15,
    },
    "c4_rate_independence": {
        "metric": "max(eta_r)/min(eta_r) - 1 over the shear-rate ladder at phi ~ 0.10",
        "threshold": 0.10,
    },
}


def _steps(box: dict, factor: float = 1.0) -> tuple[int, int]:
    """``(n_steps, n_warmup)`` from the wall-normal momentum diffusion time.

    ``t_nu = H**2 / (pi**2 nu)`` with ``H`` the effective gap; the run keeps one
    diffusion time of spin-up and ``1.5`` of time averaging.
    """
    h = float(box["n_xy"])
    t_nu = h * h / (np.pi**2 * NU)
    warm = int(round(factor * t_nu / 20.0) * 20)
    return int(round(2.0 * warm)), warm


def _suspension(phi: float, box: dict, *, seed: int = SEED):
    """Place and voxelise the frozen microstructure; returns ``(particles, info)``."""
    susp = face.physical_suspension(
        phi, radius_fraction=box["a_cells"] / float(box["n_xy"]), seed=seed
    )
    return face.voxelise(susp, n_xy=box["n_xy"], n_z=box["n_z"])


def _point(particles, box: dict, gamma_dot: float, *, factor: float = 1.0) -> dict:
    steps, warm = _steps(box, factor)
    rec = face.measure_point(
        particles, n_xy=box["n_xy"], n_z=box["n_z"], nu=NU, gamma_dot=gamma_dot,
        steps=steps, warmup=warm,
    )
    rec["box"] = dict(box)
    return rec


def _eta_r(susp: dict, pure: dict, estimator: str) -> float:
    """``eta_r`` from two *measured* stresses at their own measured rates."""
    g_s, g_p = float(susp["gamma_dot"]), float(pure["gamma_dot"])
    if not (g_s > 0.0 and g_p > 0.0):
        return float("nan")
    return (float(susp[estimator]) / g_s) / (float(pure[estimator]) / g_p)


def run(out: Path, *, factor: float = 1.0, with_convergence: bool = True) -> dict:
    t0 = time.time()
    pure = {}
    for g in GAMMA_LADDER:
        rec = face.pure_reference(
            n_xy=BOX["n_xy"], n_z=BOX["n_z"], nu=NU, gamma_dot=g,
            steps=_steps(BOX, factor)[0], warmup=_steps(BOX, factor)[1],
        )
        rec["box"] = dict(BOX)
        pure[f"{g:.3e}"] = rec
    selector = face.select_stress_estimator(list(pure.values()))
    est = selector["chosen"]

    points = []
    cases = [(PHI_DILUTE, GAMMA_MAIN), (PHI_MID, GAMMA_MAIN)]
    cases += [(PHI_RATE, g) for g in GAMMA_LADDER]
    for phi, g in cases:
        parts, info = _suspension(phi, BOX)
        rec = _point(parts, BOX, g, factor=factor)
        rec["phi_target"] = float(phi)
        rec["phi_lattice"] = float(info["phi_lattice"])       # physical cell volume
        rec["phi_lattice_full_box"] = float(info["phi_lattice_full_box"])
        rec["phi_declared"] = float(info["phi_declared"])
        rec["radius_cells"] = float(info["radius_cells"])
        rec["n_particles"] = int(info["n_particles"])
        pure_ref = pure[f"{g:.3e}"]
        for e in ("tau_flux", "tau_wall"):
            rec[f"eta_r_{e}"] = _eta_r(rec, pure_ref, e)
        points.append(rec)

    # ---- grid-convergence pair at the dilute point ------------------------
    convergence = None
    if with_convergence:
        parts_c, info_c = _suspension(PHI_DILUTE, BOX)
        parts_f, info_f = _suspension(PHI_DILUTE, BOX_FINE)
        pure_f = face.pure_reference(
            n_xy=BOX_FINE["n_xy"], n_z=BOX_FINE["n_z"], nu=NU, gamma_dot=GAMMA_MAIN,
            steps=_steps(BOX_FINE, factor)[0], warmup=_steps(BOX_FINE, factor)[1],
        )
        rec_f = _point(parts_f, BOX_FINE, GAMMA_MAIN, factor=factor)
        rec_f["phi_target"] = float(PHI_DILUTE)
        rec_f["phi_lattice"] = float(info_f["phi_lattice"])
        rec_f["phi_declared"] = float(info_f["phi_declared"])
        rec_f["radius_cells"] = float(info_f["radius_cells"])
        rec_f["n_particles"] = int(info_f["n_particles"])
        for e in ("tau_flux", "tau_wall"):
            rec_f[f"eta_r_{e}"] = _eta_r(rec_f, pure_f, e)
        coarse = next(r for r in points
                      if r["phi_target"] == PHI_DILUTE and abs(r["gamma_dot_nominal"] - GAMMA_MAIN) < 1e-15)
        convergence = {
            "coarse": {"box": dict(BOX), "phi_lattice": coarse["phi_lattice"],
                       "eta_r": coarse[f"eta_r_{est}"], "radius_cells": coarse["radius_cells"]},
            "fine": {"box": dict(BOX_FINE), "phi_lattice": rec_f["phi_lattice"],
                     "eta_r": rec_f[f"eta_r_{est}"], "radius_cells": rec_f["radius_cells"]},
            "abs_diff": abs(rec_f[f"eta_r_{est}"] - coarse[f"eta_r_{est}"]),
            "rel_diff": abs(rec_f[f"eta_r_{est}"] - coarse[f"eta_r_{est}"])
                        / max(abs(coarse[f"eta_r_{est}"]), 1e-12),
        }

    # ---- criteria ---------------------------------------------------------
    checks: dict[str, dict] = {}
    pure_err = max(max(abs(p["flux_rel_err"]), abs(p["wall_rel_err"])) for p in pure.values())
    checks["c1_pure_couette"] = {
        "measured": float(pure_err),
        "threshold": CRITERIA["c1_pure_couette"]["threshold"],
        "passed": bool(pure_err <= CRITERIA["c1_pure_couette"]["threshold"]),
    }

    def _at(phi_target, g=GAMMA_MAIN):
        return next(r for r in points
                    if r["phi_target"] == phi_target and abs(r["gamma_dot_nominal"] - g) < 1e-15)

    rec_d = _at(PHI_DILUTE)
    phi_d = rec_d["phi_lattice"]
    eta_d = rec_d[f"eta_r_{est}"]
    ref_d = 1.0 + 2.5 * phi_d
    err_d = abs(eta_d - ref_d) / ref_d
    checks["c2_dilute_einstein"] = {
        "measured": float(err_d), "threshold": CRITERIA["c2_dilute_einstein"]["threshold"],
        "passed": bool(err_d <= CRITERIA["c2_dilute_einstein"]["threshold"]),
        "detail": {"phi_lattice": float(phi_d), "eta_r": float(eta_d),
                   "ref_einstein": float(ref_d), "estimator": est},
    }

    rec_m = _at(PHI_MID)
    phi_m = rec_m["phi_lattice"]
    eta_m = rec_m[f"eta_r_{est}"]
    ref_m = float(face.krieger_dougherty(phi_m))
    ref_b = float(face.einstein_batchelor_green(phi_m))
    err_m = abs(eta_m - ref_m) / ref_m
    checks["c3_krieger_dougherty"] = {
        "measured": float(err_m), "threshold": CRITERIA["c3_krieger_dougherty"]["threshold"],
        "passed": bool(err_m <= CRITERIA["c3_krieger_dougherty"]["threshold"]),
        "detail": {"phi_lattice": float(phi_m), "eta_r": float(eta_m),
                   "ref_krieger_dougherty": ref_m, "ref_einstein_batchelor_green": ref_b,
                   "err_vs_ebg": float(abs(eta_m - ref_b) / ref_b), "estimator": est},
    }

    rate = [_at(PHI_RATE, g)[f"eta_r_{est}"] for g in GAMMA_LADDER]
    spread = max(rate) / min(rate) - 1.0
    checks["c4_rate_independence"] = {
        "measured": float(spread), "threshold": CRITERIA["c4_rate_independence"]["threshold"],
        "passed": bool(spread <= CRITERIA["c4_rate_independence"]["threshold"]),
        "detail": {"phi_lattice": [float(_at(PHI_RATE, g)["phi_lattice"]) for g in GAMMA_LADDER],
                   "gamma_dot": [float(_at(PHI_RATE, g)["gamma_dot"]) for g in GAMMA_LADDER],
                   "eta_r": [float(v) for v in rate]},
    }

    n_pass = sum(1 for c in checks.values() if c["passed"])
    report = {
        "node": "H2-FRAZIL-RHEOLOGY",
        "face": "particle-resolved moving-wall Couette (reworked after the m14882 audit)",
        "parameters": {
            "nu": NU, "seed": SEED, "box": BOX, "box_fine": BOX_FINE,
            "gamma_main": GAMMA_MAIN, "gamma_ladder": list(GAMMA_LADDER),
            "phi_dilute": PHI_DILUTE, "phi_mid": PHI_MID, "phi_rate": PHI_RATE,
            "steps_factor": factor,
            "steps": {"coarse": _steps(BOX, factor), "fine": _steps(BOX_FINE, factor)},
            "phi_max_krieger_dougherty": 0.6,
            "intrinsic_viscosity": 2.5,
        },
        "criteria_declared": CRITERIA,
        "estimator_selection": selector,
        "pure_controls": pure,
        "points": points,
        "convergence": convergence,
        "acceptance": {"criteria": checks, "n_pass": n_pass,
                       "n_defined": len(checks),
                       "verdict": "PASS" if n_pass == len(checks) else "FAIL"},
        "repro": {
            "command": "PYTHONPATH=src python scripts/run_h2_frazil_couette.py "
                       "--out artifacts/h2_frazil_rheology",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "wall_seconds": round(time.time() - t0, 1),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "couette_measurement.json").write_text(json.dumps(report, indent=2))
    try:
        make_figure(report, out / "fig4_couette_eta_r.png")
    except Exception as exc:  # pragma: no cover - plotting is not the deliverable
        report["figure_error"] = repr(exc)
    return report


def make_figure(report: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = report["estimator_selection"]["chosen"]
    pts = report["points"]
    phi = [p["phi_lattice"] for p in pts]
    eta = [p[f"eta_r_{est}"] for p in pts]
    grid = np.linspace(0.0, 0.30, 200)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].plot(grid, 1.0 + 2.5 * grid, "C0:", label=r"Einstein $1+2.5\phi$")
    ax[0].plot(grid, face.einstein_batchelor_green(grid), "C1--",
               label=r"Batchelor-Green $1+2.5\phi+6.2\phi^2$")
    ax[0].plot(grid, face.krieger_dougherty(grid), "k-", label=r"Krieger-Dougherty")
    ax[0].plot(phi, eta, "o", ms=7, mfc="none", color="C3",
               label="LBM Couette measurement (this run)")
    ax[0].set_xlabel(r"$\phi_{lattice}$ (measured)")
    ax[0].set_ylabel(r"$\eta_r$")
    ax[0].set_title("(a) measured relative viscosity vs closures at measured $\\phi$")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ladder = [p for p in pts if p["phi_target"] == report["parameters"]["phi_rate"]]
    g = [p["gamma_dot"] for p in ladder]
    ax[1].semilogx(g, [p[f"eta_r_{est}"] for p in ladder], "o-", color="C3",
                   label=r"$\phi=%.2f$" % report["parameters"]["phi_rate"])
    ax[1].axhline(float(np.mean([p[f"eta_r_{est}"] for p in ladder])), color="k", ls=":",
                  label="mean over the ladder")
    ax[1].set_xlabel(r"measured shear rate $\dot\gamma$ [lattice]")
    ax[1].set_ylabel(r"$\eta_r$")
    ax[1].set_title("(b) rate independence (Newtonian hard spheres)")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    acc = report["acceptance"]
    fig.suptitle("H2-FRAZIL-RHEOLOGY Couette face: %d/%d criteria pass (estimator %s)"
                 % (acc["n_pass"], acc["n_defined"], est), y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/h2_frazil_rheology")
    ap.add_argument("--steps-factor", type=float, default=1.0,
                    help="scale the diffusion-time budget (quick smoke runs)")
    ap.add_argument("--no-convergence", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = (_REPO_ROOT / out).resolve()
    rep = run(out, factor=args.steps_factor, with_convergence=not args.no_convergence)
    acc = rep["acceptance"]
    for cid, c in acc["criteria"].items():
        print(f"{cid:26s} measured={c['measured']:.6g} threshold={c['threshold']:.3g} "
              f"{'PASS' if c['passed'] else 'FAIL'}")
    est = rep["estimator_selection"]["chosen"]
    for p in rep["points"]:
        print(f"phi_target={p['phi_target']:.2f} phi_lat={p['phi_lattice']:.4f} "
              f"gamma={p['gamma_dot']:.3e} eta_r={p[f'eta_r_{est}']:.4f}")
    print(f"verdict {acc['verdict']} ({acc['n_pass']}/{acc['n_defined']}) -> {out}")
    return 0 if acc["n_pass"] == acc["n_defined"] else 1


if __name__ == "__main__":
    raise SystemExit(main())