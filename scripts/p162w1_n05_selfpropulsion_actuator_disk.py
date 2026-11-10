"""P162W1-N05 driver: self-propulsion actuator-disk model in TensorLBM.

Two families of work are exposed:

``--case analytic``
    Runs every analytic layer of
    :mod:`tensorlbm.actuator_disk_selfpropulsion` -- open-water anchor fidelity,
    ``K_T`` monotonicity (C5), BEM-vs-anchor adjudication (C3), the
    Newton-Raphson / Brent self-propulsion iteration (C2) and the momentum-theory
    energy identity -- and writes one JSON bundle.

``--case lbm:<name>``
    Runs one actuator-disk LBM case (``main``, ``negctl``, ``grid-coarse``,
    ``grid-fine``, ``load-uniform``, ``load-elliptic``, ``load-betz``) and writes
    the diagnostics JSON (``c1_error``, momentum-recovered thrust, energy
    residual, ...).

Usage
-----
    PYTHONPATH=src python scripts/p162w1_n05_selfpropulsion_actuator_disk.py \
        --case analytic --out-dir artifacts/p162w1_n05
    PYTHONPATH=src python scripts/p162w1_n05_selfpropulsion_actuator_disk.py \
        --case lbm:main --out-dir artifacts/p162w1_n05

All numbers written to disk are computed; nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from tensorlbm.actuator_disk_selfpropulsion import (
    LatticeConfig,
    ResistanceModel,
    adjudicate_coefficients,
    disk_force_integrals,
    build_disk_force,
    DiskForceSpec,
    kp505_anchor,
    radial_profile,
    axial_profile,
    rankine_disk,
    run_actuator_disk_lbm,
    solve_self_propulsion,
    solve_self_propulsion_brent,
)

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

#: Design advance ratio of the KP505 propeller (open-water efficiency peak).
J_DESIGN = 0.7

#: Reference Reynolds number of the lattice case, Re_D = U D / nu.  Held fixed
#: between refinements so the two grids solve the *same* non-dimensional problem.
RE_D = 60.0

#: Ship particulars for the self-propulsion balance (KCS-like, full scale).
SHIP = dict(length=230.0, wetted_area=9500.0, form_factor=1.15, rho=1025.0,
            nu=1.19e-6, wave_coeff=0.0, gravity=9.81)
SHIP_SPEED = 12.0 * 0.514444  # 12 kn in m/s
PROPELLER_D = 6.0             # full-scale propeller diameter [m]
WAKE_FRACTION = 0.25
THRUST_DEDUCTION = 0.15


def tau_from_re(diameter: float, u_inlet: float, reynolds: float) -> float:
    """Relaxation time that gives ``Re_D = u D / nu`` in the lattice."""
    nu = u_inlet * diameter / reynolds
    return 0.5 + 3.0 * nu


def lattice_case(name: str, quick: bool = False, d_scale: float = 1.0,
                 step_scale: float = 1.0) -> LatticeConfig:
    """Build the :class:`LatticeConfig` for a named case."""
    anchor = kp505_anchor()
    u0 = 0.05
    j = J_DESIGN
    kt = float(anchor.kt(j))
    kq = float(anchor.kq(j))

    spec = {
        # name: (D, nx/D, ny/D, loading, axial, thrust scale, torque scale, Re_D)
        # C1/C6/C7 use a pure-thrust disk (the Rankine actuator disk has no swirl);
        # the "swirl" case carries the propeller torque so that K_Q can be
        # recovered from the flow and adjudicated against the open-water anchor.
        # "re-high" repeats the main case at Re_D = 300: Rankine momentum theory is
        # an inviscid limit, so the far-wake comparison must be made where the
        # viscous spreading length is small compared with the disk radius.
        "main": (20.0, 10.0, 4.0, "uniform", "cosine", 1.0, 0.0, RE_D),
        "re-high": (20.0, 10.0, 4.0, "uniform", "cosine", 1.0, 0.0, 300.0),
        "swirl": (20.0, 10.0, 4.0, "uniform", "cosine", 1.0, 1.0, RE_D),
        "negctl": (20.0, 10.0, 4.0, "uniform", "cosine", 0.0, 0.0, RE_D),
        "grid-coarse": (12.0, 8.0, 3.5, "uniform", "cosine", 1.0, 0.0, RE_D),
        "grid-fine": (24.0, 8.0, 3.5, "uniform", "cosine", 1.0, 0.0, RE_D),
        "load-uniform": (16.0, 8.0, 3.5, "uniform", "cosine", 1.0, 0.0, RE_D),
        "load-elliptic": (16.0, 8.0, 3.5, "elliptic", "cosine", 1.0, 0.0, RE_D),
        "load-betz": (16.0, 8.0, 3.5, "betz", "cosine", 1.0, 0.0, RE_D),
        "load-box": (16.0, 8.0, 3.5, "uniform", "box", 1.0, 0.0, RE_D),
    }
    if name not in spec:
        raise SystemExit(f"unknown case {name!r}; choose from {sorted(spec)}")
    D, nxD, nyD, loading, axial, tscale, qscale, re_d = spec[name]

    D = D * (0.35 if quick else 1.0) * d_scale
    if quick:
        D = max(6.0, round(D))
    nx = int(round(nxD * D))
    ny = nz = int(round(nyD * D))

    n = u0 / (j * D)                       # lattice revolutions per step
    thrust = kt * 1.0 * n * n * D ** 4 * tscale
    torque = kq * 1.0 * n * n * D ** 5 * qscale

    cx = 2.5 * D
    n_steps = int(round(1.5 * step_scale * (nx - cx) / u0))
    if name == "negctl":
        n_steps = int(round((800 if not quick else 200) * step_scale))
    if quick:
        n_steps = 250

    return LatticeConfig(
        nx=nx, ny=ny, nz=nz, diameter=D, hub_ratio=0.18, u_inlet=u0,
        tau=tau_from_re(D, u0, re_d), thrust=thrust, torque=torque,
        loading=loading, axial=axial, half_thickness=0.75,
        disk_x_over_D=2.5, n_steps=n_steps,
        sample_steps=min(1500, max(60, n_steps // 4)),
        report_every=max(1, n_steps // 8), far_station_over_D=4.0, name=name,
    )


# ---------------------------------------------------------------------------
# Analytic bundle
# ---------------------------------------------------------------------------


def analytic_bundle() -> dict:
    """Run all analytic layers and return one JSON-able bundle."""
    from tensorlbm.selfpropulsion import (
        KP505Geometry,
        SectionPolar,
        blade_element_momentum,
        kp505_reference_table,
    )

    t0 = time.time()
    anchor = kp505_anchor()
    table = kp505_reference_table()
    j_tab = np.asarray(table["J"], float)
    kt_tab = np.asarray(table["KT"], float)
    kq_tab = np.asarray(table["KQ"], float)
    kt_fit = np.asarray(anchor.kt(j_tab), float)
    kq_fit = np.asarray(anchor.kq(j_tab), float)

    # ---- C5: monotonic decrease of K_T in J ----
    j_sweep = np.linspace(0.10, 1.05, 200)
    kt_sweep = np.asarray(anchor.kt(j_sweep), float)
    dkt = np.gradient(kt_sweep, j_sweep)
    anchor_monotone = bool(np.all(dkt < 0.0))

    # ---- BEM (independent physics: geometry + section polars) ----
    geom = KP505Geometry()
    polar = SectionPolar()
    bem_curve = []
    for j in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        res = blade_element_momentum(geom, polar, float(j), n_rps=1.0, rho=999.5)
        bem_curve.append(
            {"J": float(j), "KT": float(res["kt"]), "KQ": float(res["kq"]),
             "eta0": float(res["eta0"])}
        )
    bem_monotone = bool(all(
        bem_curve[i + 1]["KT"] < bem_curve[i]["KT"] for i in range(len(bem_curve) - 1)
    ))

    # ---- C3: adjudication at the design advance ratio ----
    res70 = blade_element_momentum(geom, polar, J_DESIGN, n_rps=1.0, rho=999.5)
    adj = adjudicate_coefficients(
        J_DESIGN,
        {
            "open_water_anchor": (float(anchor.kt(J_DESIGN)), float(anchor.kq(J_DESIGN))),
            "bem_from_geometry": (float(res70["kt"]), float(res70["kq"])),
        },
    )

    # ---- C2: self-propulsion point ----
    resistance = ResistanceModel(**SHIP)
    common = dict(rho=SHIP["rho"], wake_fraction=WAKE_FRACTION,
                  thrust_deduction=THRUST_DEDUCTION)
    sp_gross = solve_self_propulsion(SHIP_SPEED, anchor, resistance, PROPELLER_D,
                                     mode="gross", **common)
    sp_eff = solve_self_propulsion(SHIP_SPEED, anchor, resistance, PROPELLER_D,
                                   mode="effective", **common)
    sp_brent = solve_self_propulsion_brent(SHIP_SPEED, anchor, resistance, PROPELLER_D,
                                           mode="effective", **common)
    # ---- convergence-order-independent cross-check of Newton vs Brent ----
    newton_vs_brent = abs(sp_eff["n"] - sp_brent["n"]) / sp_brent["n"]

    # ---- momentum-theory energy identity (C6 analytic anchor) ----
    area = math.pi * (0.5 * PROPELLER_D) ** 2
    thrust_eff = sp_eff["thrust"] * (1.0 - THRUST_DEDUCTION)
    rk = rankine_disk(thrust_eff, SHIP["rho"], SHIP_SPEED, area)
    ke_flux = 0.5 * SHIP["rho"] * area * rk["u_disk"] * (rk["u_far"] ** 2 - SHIP_SPEED ** 2)
    energy_identity_rel = abs(rk["power_input"] - ke_flux) / rk["power_input"]

    # ---- force-distribution integral checks ----
    shape = (48, 48, 60)
    integral_checks = {}
    for kind in ("uniform", "elliptic", "betz"):
        spec = DiskForceSpec(diameter=16.0, thrust=1.0, torque=0.25, loading=kind,
                            axial="cosine", center=(20.0, 24.0, 24.0))
        fld = build_disk_force(spec, shape)
        integral_checks[kind] = {
            "thrust_applied": fld["thrust_applied"],
            "torque_applied": fld["torque_applied"],
            "thrust_rel_error": abs(fld["thrust_applied"] - 1.0),
            "torque_rel_error": abs(fld["torque_applied"] - 0.25) / 0.25,
            "cells": fld["n_cells"],
        }
    axial_checks = {}
    for kind in ("box", "cosine", "gauss"):
        spec = DiskForceSpec(diameter=16.0, thrust=1.0, torque=0.25, axial=kind,
                            center=(20.0, 24.0, 24.0))
        fld = build_disk_force(spec, shape)
        axial_checks[kind] = {
            "thrust_applied": fld["thrust_applied"],
            "thrust_rel_error": abs(fld["thrust_applied"] - 1.0),
            "cells": fld["n_cells"],
        }

    return {
        "kind": "analytic",
        "runtime_s": time.time() - t0,
        "anchor": {
            "source": anchor.source,
            "j_range": anchor.j_range,
            "kt_coeffs_low_to_high": anchor.kt_coeffs,
            "kq_coeffs_low_to_high": anchor.kq_coeffs,
            "fit_rmse": {"KT": anchor.fit_rmse[0], "KQ": anchor.fit_rmse[1]},
            "max_abs_error_vs_table": {
                "KT": float(np.max(np.abs(kt_fit - kt_tab))),
                "KQ": float(np.max(np.abs(kq_fit - kq_tab))),
            },
            "max_rel_error_vs_table": {
                "KT": float(np.max(np.abs(kt_fit - kt_tab) / kt_tab)),
                "KQ": float(np.max(np.abs(kq_fit - kq_tab) / kq_tab)),
            },
            "table": {"J": j_tab.tolist(), "KT": kt_tab.tolist(), "KQ": kq_tab.tolist()},
            "fit": {"J": j_tab.tolist(), "KT": kt_fit.tolist(), "KQ": kq_fit.tolist()},
        },
        "C5_monotonicity": {
            "anchor_monotone_decreasing": anchor_monotone,
            "bem_monotone_decreasing": bem_monotone,
            "min_dKT_dJ": float(np.min(dkt)),
            "max_dKT_dJ": float(np.max(dkt)),
            "j_sweep": j_sweep.tolist(),
            "kt_sweep": kt_sweep.tolist(),
            "bem_curve": bem_curve,
        },
        "C3_adjudication": adj,
        "C2_selfpropulsion": {
            "ship": SHIP,
            "speed_m_s": SHIP_SPEED,
            "propeller_diameter_m": PROPELLER_D,
            "wake_fraction": WAKE_FRACTION,
            "thrust_deduction": THRUST_DEDUCTION,
            "gross": sp_gross,
            "effective": sp_eff,
            "brent_effective": sp_brent,
            "newton_vs_brent_rel_diff": newton_vs_brent,
            "resistance_at_speed_N": resistance.resistance(SHIP_SPEED),
        },
        "C6_analytic": {
            "rankine": rk,
            "ke_flux": ke_flux,
            "power_input": rk["power_input"],
            "identity_rel_error": energy_identity_rel,
            "thrust_input": thrust_eff,
        },
        "force_integral_checks": {"radial": integral_checks, "axial": axial_checks},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True,
                    help="'analytic' or 'lbm:<name>' with name in "
                         "main|negctl|grid-coarse|grid-fine|load-uniform|"
                         "load-elliptic|load-betz|load-box")
    ap.add_argument("--out-dir", default="artifacts/p162w1_n05")
    ap.add_argument("--d-scale", type=float, default=1.0,
                    help="scale all geometry lengths (resolution) by this factor")
    ap.add_argument("--step-scale", type=float, default=1.0,
                    help="scale the number of time steps by this factor")
    ap.add_argument("--quick", action="store_true",
                    help="tiny grids / few steps, for smoke tests only")
    args = ap.parse_args(argv)

    # Torch CPU threading: oversubscription is the single biggest performance
    # trap for a 19-population stencil kernel, so the thread count is explicit.
    threads = int(os.environ.get("LBM_THREADS", "4"))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:  # pragma: no cover
        pass

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.case.startswith("lbm:"):
        name = args.case.split(":", 1)[1]
        cfg = lattice_case(name, quick=args.quick, d_scale=args.d_scale,
                           step_scale=args.step_scale)
        print(f"[lbm:{name}] nx={cfg.nx} ny={cfg.ny} nz={cfg.nz} D={cfg.diameter} "
              f"tau={cfg.tau:.4f} steps={cfg.n_steps} thrust={cfg.thrust:.4e} "
              f"torque={cfg.torque:.4e}", flush=True)
        t0 = time.time()
        result = run_actuator_disk_lbm(cfg)
        result["kind"] = "lbm"
        result["wall_time_s"] = time.time() - t0
        result["quick"] = bool(args.quick)
        path = out / f"lbm_{name}.json"
        print(
            "  summary: u_disk={:.5f} (theory {:.5f}) u_far={:.5f} (theory {:.5f}) "
            "c1_err={:.2e} T_mom={:.5e} (req {:.5e}) E_res={:.2e} u_max={:.4f} "
            "rho=[{:.4f},{:.4f}]".format(
                result["u_disk_lbm"], result["rankine"]["u_disk"],
                result["u_far_lbm"], result["rankine"]["u_far"],
                result["c1_error"], result["thrust_momentum"], result["config"]["thrust"],
                result["energy"]["residual_rel"], result["u_max_abs"],
                result["rho_range"][0], result["rho_range"][1],
            ),
            flush=True,
        )
    else:
        result = analytic_bundle()
        path = out / "analytic.json"

    path.write_text(json.dumps(result, indent=2, sort_keys=False))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())