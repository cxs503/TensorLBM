#!/usr/bin/env python3
"""XVAL: TensorLBM  vs  joconnor22/LIFE ``compiletime-params-plus-runtime-geometry-config``.

Anchor: ``XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY``.

The cross-validated practice (LIFE, read from its sources at
``/tmp/LIFE-master`` — see ``src/tensorlbm/runtime_geometry.py`` docstring for
the file:line map) is a two-file case convention:

* ``inc/params.h`` — grid, ``omega``, BCs, step count as **compile-time**
  constants: any edit forces ``make clean && make``;
* ``input/geometry.config`` — body list as **runtime** text: ``CIRCLE`` /
  ``FILAMENT`` lines, per-body array spacing, **degrees** for angles,
  ``FLEXIBLE`` → FEM body.

What is measured here (pre-declared, see ``THRESHOLDS``):

DECISION  D1: is the runtime-geometry path a drop-in replacement for the
              hard-coded mask in a *verified* in-repo case?
          D2: does a geometry parsed from LIFE's own ``geometry.config`` line
              reproduce the reference physics anchor?
          D3: is the marker discretisation of the transferred convention
              consistent with the dx-refinement claimed by the source?

ALTERNATIVES: (a) keep hard-coded masks per case (status quo); (b) copy
LIFE's C++ reader verbatim; (c) reimplement the format in the Python package
(this script).  (a) is the control, (c) is the candidate.

METRICS:  Δmask (cells), Cd error vs the Schäfer–Turek 2D-1 Re=20 anchor
          (Nabh 1998 spectral, pinned in ``benchmarks/verified/cylinder/``),
          plateau drift, marker-count / perimeter error vs the analytic
          circle.

THRESHOLDS (predeclared): Δmask <= 0 cells is *expected* (exactness, not a
          tolerance); |Cd_err| < 15 % is the owner accuracy gate and < 5 % is
          the case's own verified tolerance; the runtime-vs-hard-coded ΔCd
          must be < 1 % (same physics, geometry plumbing must be inert).
          Stopping rule: a run counts only if the trailing 10 k-step block
          means agree within 0.1 % (``stationary`` true); otherwise it is
          reported as NOT CONVERGED instead of being compared.

Usage::

    PYTHONPATH=src python3 xval/life_geometry_config/xval_life_geometry_config.py \
        --mode parity
    PYTHONPATH=src python3 xval/life_geometry_config/xval_life_geometry_config.py \
        --mode run --grid D20 --source runtime_config --steps 120000 \
        --out runs/m17673/xval_D20_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorlbm.cylinder_st2d1 import (
    GEOMETRY,
    REFERENCE,
    case_geometry,
    run_cylinder_st2d1,
)
from tensorlbm.runtime_geometry import (
    CaseParams,
    CircleBody,
    FilamentBody,
    Scene,
    SimulationSetup,
    parse_geometry_config,
)

HERE = Path(__file__).resolve().parent

#: LIFE ``input/geometry.config`` lines for the Turek–Hron set-up, reproduced
#: verbatim from ``/tmp/LIFE-master/input/geometry.config`` (2 lines).
LIFE_TUREKHRON_CONFIG = """\
# LIFE (joconnor22) input/geometry.config — Turek-Hron FSI set-up
CIRCLE 1 0 0.2 0.2 0.0 0.0 0.05
FILAMENT 1 1 0.25 0.2 0.0 0.0 0.35 0.02 0.0 FLEXIBLE 20 CLAMPED 10000.0 1.4e6
"""

#: The Schäfer–Turek 2D-1 channel (TensorLBM verified case) expressed in the
#: same convention: only the cylinder, 2.2 m x 0.41 m channel.
ST2D1_CONFIG = """\
# TensorLBM Schäfer-Turek 2D-1 channel, LIFE geometry.config convention
CIRCLE 1 0 0.20 0.20 0.0 0.0 0.05
"""

#: Pre-declared thresholds (do not tune after seeing results).
THRESHOLDS: dict[str, Any] = {
    "mask_cells_exact": 0,
    "cd_err_pct_gate": 15.0,  # owner accuracy standard
    "cd_err_pct_case": 5.0,  # verified-case tolerance
    "delta_cd_vs_hardcoded_pct": 1.0,
    "plateau_tol": 1.0e-3,
    "perimeter_err_pct_max": 5.0,
}

GRIDS = {"D10": 10, "D20": 20, "D40": 40}


# ---------------------------------------------------------------------------
# geometry parity (no solver)
# ---------------------------------------------------------------------------
def analytic_circle_markers(radius: float, dx: float) -> dict[str, float]:
    """Analytic oracle for LIFE's marker discretisation of one circle."""
    n = int(math.floor(2.0 * math.pi * radius / dx))
    theta = 2.0 * math.pi * np.arange(n) / n
    pts = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    chord = float(np.sum(np.linalg.norm(np.diff(np.vstack((pts, pts[:1])), axis=0), axis=1)))
    return {
        "n_markers": float(n),
        "perimeter_chord": chord,
        "perimeter_exact": 2.0 * math.pi * radius,
        "perimeter_err_pct": (chord - 2.0 * math.pi * radius) / (2.0 * math.pi * radius) * 100.0,
    }


def parity_report() -> dict[str, Any]:
    """Compare the transferred convention against LIFE's documented formulas."""
    import torch

    from tensorlbm.boundaries import cylinder_mask

    out: dict[str, Any] = {"thresholds": THRESHOLDS, "rows": [], "sensitivities": []}

    # --- S0: LIFE's own file parses, bodies/ids/markers as the source says --
    scene = parse_geometry_config(LIFE_TUREKHRON_CONFIG, source="LIFE:input/geometry.config")
    out["life_config_bodies"] = [
        {"id": b.body_id, "keyword": b.keyword} for b in scene.bodies
    ]

    for label, D in GRIDS.items():
        geo = case_geometry(D)
        dx, nx, ny = geo["dx"], int(geo["nx"]), int(geo["ny"])
        sc = parse_geometry_config(ST2D1_CONFIG, source="ST2D1 geometry.config")
        ref_mask = cylinder_mask(
            nx, ny, float(geo["cx"]), float(geo["cy"]), float(geo["radius"]), torch.device("cpu")
        ).numpy()
        got_mask = sc.mask(nx, ny, dx, origin=(0.0, 0.0))
        delta = int(np.sum(got_mask ^ ref_mask))
        ana = analytic_circle_markers(0.05, dx)
        got_n = sc.bodies[0].marker_count(dx)
        row = {
            "grid": label,
            "D_cells": D,
            "nx": nx,
            "ny": ny,
            "dx_m": dx,
            "mask_cells_ref": int(ref_mask.sum()),
            "mask_cells_got": int(got_mask.sum()),
            "mask_delta_cells": delta,
            "markers_ref_formula": int(ana["n_markers"]),
            "markers_got": got_n,
            "marker_count_err_pct": 0.0,
            "perimeter_chord_m": ana["perimeter_chord"],
            "perimeter_exact_m": ana["perimeter_exact"],
            "perimeter_err_pct": ana["perimeter_err_pct"],
            "pass_mask": delta <= THRESHOLDS["mask_cells_exact"],
            "pass_perimeter": abs(ana["perimeter_err_pct"]) <= THRESHOLDS["perimeter_err_pct_max"],
        }
        out["rows"].append(row)

    # --- S1: sensitivity #1 — marker/perimeter discretisation vs dx --------
    for label, D in GRIDS.items():
        dx = case_geometry(D)["dx"]  # D_cells = diameter/dx = 0.1/dx
        ana = analytic_circle_markers(0.05, dx)
        out["sensitivities"].append(
            {
                "kind": "dx_refinement",
                "grid": label,
                "dx_m": dx,
                "n_markers": int(ana["n_markers"]),
                "perimeter_err_pct": ana["perimeter_err_pct"],
            }
        )

    # --- S2: sensitivity #2 — angle convention (degrees → radians) ---------
    for angle in (0.0, 30.0, 45.0, 90.0):
        fb = FilamentBody(
            body_id=7,
            start=(0.25, 0.2),
            length=0.35,
            height=0.02,
            angle_deg=angle,
            flexible=True,
            n_elements=20,
            bc="CLAMPED",
            rho=1.0e4,
            youngs_modulus=1.4e6,
        )
        pts = fb.markers(1.0e-3)
        length = float(np.linalg.norm(pts[-1] - pts[0]))
        out["sensitivities"].append(
            {
                "kind": "angle_convention",
                "angle_deg": angle,
                "angle_rad": fb.angle_rad,
                "tip": [float(pts[-1][0]), float(pts[-1][1])],
                "marker_span_m": length,
                "span_err_pct": (length - fb.length) / fb.length * 100.0,
                "n_markers": fb.marker_count(1.0e-3),
            }
        )

    # --- S3: mechanism check — geometry edits never touch params fingerprint
    params = CaseParams(nx=int(case_geometry(20)["nx"]), ny=int(case_geometry(20)["ny"]),
                        tau=0.8, steps=10, dx=case_geometry(20)["dx"])
    setup = SimulationSetup(params=params, scene=parse_geometry_config(ST2D1_CONFIG))
    fp0 = setup.param_fingerprint
    setup2 = setup.add_body(CircleBody(body_id=5, centre=(0.9, 0.2), radius=0.05))
    setup3 = setup2.remove_body(5)
    try:
        setup.apply_params(tau=0.81)
        rebuild_refused = False
    except RuntimeError:
        rebuild_refused = True
    out["mechanism"] = {
        "fingerprint_before": fp0,
        "fingerprint_after_geometry_edits": setup3.param_fingerprint,
        "fingerprint_unchanged_by_geometry": fp0 == setup3.param_fingerprint,
        "param_edit_refused_without_rebuild": rebuild_refused,
        "fingerprint_after_rebuild": setup.apply_params(tau=0.81, rebuild=True).param_fingerprint,
    }
    return out


# ---------------------------------------------------------------------------
# physics runs
# ---------------------------------------------------------------------------
def run_one(grid: str, source: str, steps: int, seed_note: str = "") -> dict[str, Any]:
    import torch

    D = GRIDS[grid]
    geo = case_geometry(D)
    dx = float(geo["dx"])
    nx, ny = int(geo["nx"]), int(geo["ny"])

    params = CaseParams(
        nx=nx,
        ny=ny,
        tau=0.8,
        steps=steps,
        dx=dx,
        origin=(0.0, 0.0),
        bc_left="velocity",
        bc_right="pressure",
        bc_bottom="wall",
        bc_top="wall",
    )
    scene = parse_geometry_config(ST2D1_CONFIG, source="ST2D1 geometry.config")
    scene.validate_domain(nx, ny, dx, origin=params.origin)
    setup = SimulationSetup(params=params, scene=scene)
    mask = setup.scene.mask(nx, ny, dx, origin=params.origin) if source == "runtime_config" else None

    t0 = time.time()
    res = run_cylinder_st2d1(
        D_cells=D,
        steps=steps,
        progress=False,
        keep_fields=True,
        sample_every=50,
        avg_window=10_000,
        resid_interval=5_000,
        stop_on_plateau=True,
        min_steps=min(40_000, steps),
        plateau_tol=THRESHOLDS["plateau_tol"],
        solid_mask=mask,
    )
    res["elapsed_s"] = time.time() - t0
    res["provenance"] = setup.provenance(dx)
    res["provenance"]["geometry_source"] = res.get("geometry_source", source)
    res["provenance"]["xval_anchor"] = "XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY"
    res["provenance"]["note"] = seed_note
    fields = res.pop("fields", None)
    return {"result": res, "fields": fields}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["parity", "run"], default="parity")
    ap.add_argument("--grid", default="D20")
    ap.add_argument("--source", choices=["hard_coded", "runtime_config"], default="runtime_config")
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.mode == "parity":
        rep = parity_report()
        txt = json.dumps(rep, indent=1)
        print(txt)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(txt)
        return 0

    payload = run_one(args.grid, args.source, args.steps)
    res, fields = payload["result"], payload["fields"]
    out = Path(args.out) if args.out else HERE / f"xval_{args.grid}_{args.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    if fields is not None:
        np.savez_compressed(
            out.with_suffix(".fields.npz"),
            rho=fields["rho"],
            ux=fields["ux"],
            uy=fields["uy"],
            vorticity=fields["vorticity"],
            solid=fields["solid"],
            dx=fields["dx"],
        )
    print(
        json.dumps(
            {
                "out": str(out),
                "source": res.get("geometry_source"),
                "cd": res["cd"],
                "err_cd_pct": res["err_cd_pct"],
                "n_steps": res["n_steps"],
                "stop_reason": res["stop_reason"],
                "stationary": res["stationary"],
                "drift": res["cd_plateau_drift"],
                "wall_s": res["wall_s"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())