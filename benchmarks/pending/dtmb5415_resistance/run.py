#!/usr/bin/env python3
"""DTMB 5415 bare-hull resistance benchmark runner (NOT yet verified).

Purpose
-------
Try to build a DTMB 5415 resistance benchmark in the repo's verified family
(shared modules: STL geometry -> ``GeneralSimEngine`` -> ``lbm_step_correct``
routed through ``benchmarks.compile_route`` -> ``drag_pressure`` force
integration, wetted-area normalisation, standard vs mix50 friction).

Geometry
--------
* ``--geometry stl`` (default): the genuine DTMB 5415 underwater bare hull,
  extracted (waterline clip at T = 6.15 m) from the zenodo / CNR-INSEAN
  official 5415 baseline point cloud -- see ``_build_geometry.py`` for
  provenance.  The clip leaves the surface **open** at the waterline, which
  makes the voxeliser fill a pillar below z = T (Cb ~ 18, 35x the
  displacement -- this is what made m438 report Cd_total ~ 1.18 instead of
  0.045).  ``_close_waterline.py`` therefore writes the watertight
  ``geometry/dtmb5415_underwater_closed.stl`` (+ ``.capinfo.json``) and that
  file is the default; the raw clipped ``dtmb5415_underwater.stl`` is still
  accepted and reproduces the broken body.  Voxelised through the shared
  ``tensorlbm.stl_geometry`` module by ``GeneralSimEngine``
  (``GeometrySource.STL_FILE``).
* ``--geometry series60``: parametric Series 60 (Cb=0.60) proxy via
  ``GeometrySource.PARAMETRIC_HULL`` -- the ONLY hull shape ``ship_cad`` can
  generate; it is NOT DTMB 5415 (no bulbous bow / sonar dome, no transom).

Reference frames (all recorded; see README for why only some are valid)
-------------------------------------------------------------------
* Blasius laminar flat plate Cf = 1.328/sqrt(Re) at the run Re (the repo's
  Re=1000 family frame used by the verified ``suboff_re1000`` case).
* ITTC-1957 Cf = 0.075/(log10 Re - 2)^2.
* DTMB 5415 model-test total resistance C_T vs Fr (INSEAN/Olivieri 2001,
  Stern 2000; second-hand constants, NOT a first-class repo reference).

Usage
-----
  PYTHONPATH=src python run.py [--geometry stl|series60] [--resolution 48]
      [--steps 8000] [--device sdaa:0] [--collision mrt|smagorinsky]
      [--friction standard|mix50|faces] [--out DIR]
      [--compile-mode default|eager]
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))  # compile_route

import numpy as np  # noqa: E402
import torch  # noqa: E402
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    compile_status_of,
    route_step,
)

from tensorlbm.boundaries3d import far_field_bc_3d  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    drag_friction_integration,
    drag_pressure_integration,
)
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402

# ---------------------------------------------------------------------------
# DTMB 5415 full-scale principal dimensions (standard_models_detailed.py,
# Stern et al. 1996 DTMB 5415 model tests)
# ---------------------------------------------------------------------------
LPP = 142.0  # m, length between perpendiculars
LWL = 153.0  # m, length at waterline
BEAM = 19.06  # m
DRAFT = 6.15  # m
CB = 0.507  # block coefficient
U_PHYS = 1.0e-3  # m/s (sets dt/Re only; low-Re family)

# DTMB 5415 model-test total resistance (INSEAN towing tank; second-hand
# constants from a sibling repo CI test citing Olivieri 2001 / Stern 2000).
# C_T normalised by wetted surface; Cf_ITTC57 at Re ~ 1.2e7.
C_T_EXP = {0.28: 4.2e-3, 0.35: 5.1e-3, 0.41: 6.8e-3}
CF_ITTC57_EXP = 2.91e-3  # at Re ~ 1.2e7  (Fr=0.28)
CR_RESIDUAL_EXP = 0.94e-3  # form + wave residual (Fr=0.28)
EXP_FR = 0.28

# ---------------------------------------------------------------------------
# Acceptance caliber (see docs/sdaa_triage/DTMB5415_DIAGNOSIS.md)
# ---------------------------------------------------------------------------
#: Platform acceptance band.  It is only meaningful when the simulated and the
#: reference quantity are the same dimensionless number at the same similarity
#: point (same Re / Fr / reference area).
CRITERION_BAND_PCT = 3.0
#: Geometry gates: the hull is thin (T/Lpp = 4.3 %), so at benchmark
#: resolutions the draft can degenerate to ~1 cell.  Below 2 cells the voxel
#: body is a staircase plank, not the hull.
GEOM_MIN_T_OVER_DX = 2.0
GEOM_CB_TOL = 0.15
#: A Cd averaged over a force window that still drifts more than this between
#: its two halves is not a steady value and is not reported as one.
FORCE_STEADY_TOL_PCT = 2.0
DISPLACEMENT_M3 = CB * LPP * BEAM * DRAFT


def _default_device() -> str:
    try:
        import torch_sdaa  # noqa: F401

        if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
            return "sdaa:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def stl_wetted_area_breakdown(stl_path: Path, l_cells: int, z_wl: float) -> dict:
    """Wetted surface of the underwater hull, in m2 and lattice units2.

    The waterline-closed STL produced by ``_close_waterline.py`` carries an
    extra *cap* disk at ``z = T`` that keeps the mesh watertight (required by
    the inside/outside voxeliser) but is **not** a wetted surface: it must be
    excluded from the drag normalisation, otherwise S_wet roughly doubles
    (+2955 m2) and every Cd halves.  The cap triangles are the last ones of the
    file and their count is recorded in ``<stl>.capinfo.json``; using that
    sidecar keeps the exclusion exact, whereas a geometric (near-horizontal)
    test also drops ~18 m2 of real hull surface just under the waterline.
    Both areas are reported so nothing is silently dropped.
    """
    from tensorlbm.stl_geometry import read_stl

    verts, faces, _ = read_stl(str(stl_path))
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    cz = verts[faces, 2].mean(axis=1)
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)

    # The cap triangles are appended last by ``_close_waterline.py``, which
    # also writes ``<stl>.capinfo.json`` (``cap_triangle_count``).  Using the
    # sidecar keeps the exclusion exact - a purely geometric test would also
    # drop the few near-horizontal hull triangles that sit just under the
    # waterline.  Without a sidecar the file is treated as a plain open
    # surface (no cap), which is the pre-existing behaviour.
    side = Path(str(stl_path) + ".capinfo.json")
    n_cap, cap_src = 0, "none"
    if side.exists():
        try:
            n_cap = int(json.loads(side.read_text()).get("cap_triangle_count", 0))
            cap_src = "sidecar"
        except Exception:  # pragma: no cover - malformed sidecar
            n_cap, cap_src = 0, "sidecar-unreadable"
    wet = cz <= z_wl
    cap_mask = np.zeros_like(wet)
    if 0 < n_cap <= wet.size:
        cap_mask[-n_cap:] = True
        wet = wet & ~cap_mask

    dx = LPP / l_cells
    return {
        "stl": str(stl_path),
        "cap_source": cap_src,
        "z_waterline_m": z_wl,
        "wetted_area_m2": float(area[wet].sum()),
        "cap_area_m2": float(area[cap_mask].sum()),
        "lattice_dx_m": dx,
        "wetted_area_lb": float(area[wet].sum()) / dx**2,
        "cap_area_lb": float(area[cap_mask].sum()) / dx**2,
        "n_tri_wetted": int(wet.sum()),
        "n_tri_cap": int(cap_mask.sum()),
    }


def _stl_wetted_area_lb(stl_path: Path, l_cells: int, z_wl: float) -> tuple[float, float]:
    """(S_wet_phys_m2, S_wet_lb) of the underwater hull; see breakdown above."""
    d = stl_wetted_area_breakdown(stl_path, l_cells, z_wl)
    return d["wetted_area_m2"], d["wetted_area_lb"]


def geometry_resolution_report(
    *,
    n_solid_cells: int,
    dx_m: float,
    stair_wall_faces: int,
    smooth_wetted_lb: float,
    near_wall_cells: int | None = None,
) -> dict:
    """Is the *body that is actually integrated* the DTMB 5415 hull?

    Two independent checks, both at the resolution of the run:

    ``T/dx``  the draft spans at least ``GEOM_MIN_T_OVER_DX`` cells.  The hull
              is thin (T/Lpp = 4.3 %), so at 32-48 cells per Lpp the draft is
              1-2 cells and the body degenerates into a staircase plank.
    ``Cb``    voxel displacement vs ``Cb*L*B*T`` within ``GEOM_CB_TOL``.

    ``stair_over_smooth_wetted_area`` additionally quantifies the bias that a
    smooth-area normalisation suffers when the integrated staircase surface is
    much larger (it multiplies the raw wall-force sum by exactly that ratio).
    """
    vol = n_solid_cells * dx_m**3
    ratio = vol / DISPLACEMENT_M3
    t_over_dx = DRAFT / dx_m
    b_over_dx = BEAM / dx_m
    return {
        "n_solid_cells": int(n_solid_cells),
        "dx_m": dx_m,
        "L_over_dx": LPP / dx_m,
        "B_over_dx": b_over_dx,
        "T_over_dx": t_over_dx,
        "near_wall_cells": near_wall_cells,
        "displacement_voxel_m3": vol,
        "displacement_reference_m3": DISPLACEMENT_M3,
        "cb_voxel": vol / (LPP * BEAM * DRAFT),
        "cb_reference": CB,
        "cb_error_pct": (ratio - 1.0) * 100.0,
        "stair_wall_faces": int(stair_wall_faces),
        "smooth_wetted_lb": smooth_wetted_lb,
        "stair_over_smooth_wetted_area": (
            stair_wall_faces / smooth_wetted_lb if smooth_wetted_lb else float("nan")
        ),
        "thickness_resolved": bool(t_over_dx >= GEOM_MIN_T_OVER_DX),
        "beam_resolved": bool(b_over_dx >= 8.0),
        "cb_within_tolerance": bool(abs(ratio - 1.0) <= GEOM_CB_TOL),
        "verdict": (
            "VALID"
            if (t_over_dx >= GEOM_MIN_T_OVER_DX and abs(ratio - 1.0) <= GEOM_CB_TOL)
            else "UNDER_RESOLVED"
        ),
    }


def _window_stats(values) -> dict:
    """Drift of a sampled force history: 2nd-half mean vs 1st-half mean."""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size < 4:
        return {
            "n": int(v.size),
            "mean": float(v.mean()) if v.size else float("nan"),
            "drift_pct": float("nan"),
            "steady": False,
        }
    h = v.size // 2
    m1, m2 = v[:h].mean(), v[h:].mean()
    drift = (m2 - m1) / abs(m2) * 100.0 if m2 else float("inf")
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "first_half_mean": float(m1),
        "second_half_mean": float(m2),
        "drift_pct": float(drift),
        "steady": bool(abs(drift) <= FORCE_STEADY_TOL_PCT),
    }


def resistance_criteria(
    *,
    re: float,
    cd_p: float,
    cd_f: float,
    cd_tot: float,
    cf_blasius: float,
    cf_ittc57: float,
    ct_exp: float,
    exp_fr: float,
    fr_run: float,
    no_wave_floor_pct: float,
    geometry: dict,
    force_window_drift_pct: float,
) -> dict:
    """Split the 3 % acceptance band into *comparable* and *non-comparable* frames.

    The m438 SDAA validation applied ``|error| <= 3 %`` to ``Cd_total`` vs the
    Fr = 0.28 model-test ``C_T``.  That comparison is not well posed - the run
    is Re = 1000, single-phase, no free surface - so the band can never be met
    (the fs-free structural floor is already ``no_wave_floor_pct``).  Each
    frame below states what is being compared with what, whether that is
    like-for-like, and only then a pass/fail.  ``verdict`` is the case-level
    statement.
    """

    def _err(sim: float, ref: float) -> float:
        return (sim - ref) / ref * 100.0 if ref else float("nan")

    err_f_blasius = _err(cd_f, cf_blasius)
    err_f_ittc = _err(cd_f, cf_ittc57)
    err_tot_exp = _err(cd_tot, ct_exp)

    frames = {
        "friction_vs_blasius_laminar": {
            "comparable": True,
            "quantity_sim": "Cd_friction (wall shear over wetted area, mix50)",
            "quantity_ref": "Blasius laminar flat plate Cf = 1.328/sqrt(Re)",
            "re": re,
            "sim": cd_f,
            "ref": cf_blasius,
            "err_pct": err_f_blasius,
            "band_pct": CRITERION_BAND_PCT,
            "pass": bool(abs(err_f_blasius) <= CRITERION_BAND_PCT),
            "note": (
                "same Reynolds number, same reference length, friction against "
                "friction: this is the repo's calibrated in-frame friction "
                "check (SUBOFF Re=1000 L=96: Cd_tot +0.25 % vs 0.0420)"
            ),
        },
        "friction_vs_ittc1957": {
            "comparable": False,
            "quantity_sim": "Cd_friction",
            "quantity_ref": "ITTC-1957 Cf = 0.075/(log10 Re - 2)^2",
            "sim": cd_f,
            "ref": cf_ittc57,
            "err_pct": err_f_ittc,
            "reason": (
                "ITTC-1957 is the turbulent flat-plate line; at Re = 1000 the "
                "boundary layer is laminar, so the reference is the wrong "
                "branch, not a target"
            ),
            "pass": None,
        },
        "total_vs_ittc1957": {
            "comparable": False,
            "quantity_sim": "Cd_total = Cd_pressure + Cd_friction",
            "quantity_ref": "ITTC-1957 Cf (friction only)",
            "sim": cd_tot,
            "ref": cf_ittc57,
            "err_pct": _err(cd_tot, cf_ittc57),
            "reason": (
                "category error: form/pressure drag is added on the left and "
                "absent on the right.  It was this pairing that produced the "
                "1400-2700 % 'errors' reported by the m438 sweep"
            ),
            "pass": None,
        },
        "total_vs_exp_CT_fr0.28": {
            "comparable": False,
            "quantity_sim": "Cd_total (Re = %.0f, Fr = %.4f, single-phase)" % (re, fr_run),
            "quantity_ref": "model-test C_T at Fr = %.2f, Re ~ 1.2e7" % exp_fr,
            "sim": cd_tot,
            "ref": ct_exp,
            "err_pct": err_tot_exp,
            "reason": (
                "four orders of magnitude in Re and no free surface: wave-making "
                "resistance is absent by construction.  Even a perfect friction "
                "result lands at %.1f %% of the experimental C_T" % no_wave_floor_pct
            ),
            "structural_floor_pct": no_wave_floor_pct,
            "pass": None,
        },
    }

    in_frame = frames["friction_vs_blasius_laminar"]
    if geometry["verdict"] != "VALID":
        verdict = "WITHHELD_GEOMETRY"
        why = (
            "the voxelised body is not the DTMB 5415 hull at this resolution "
            "(T/dx = %.2f, Cb error %+.1f %%, staircase/smooth wetted-area "
            "ratio %.2f): no drag coefficient computed on it is meaningful"
            % (
                geometry["T_over_dx"],
                geometry["cb_error_pct"],
                geometry["stair_over_smooth_wetted_area"],
            )
        )
    elif abs(force_window_drift_pct) > FORCE_STEADY_TOL_PCT:
        verdict = "WITHHELD_UNSTEADY"
        why = (
            "force window still drifting %+.1f %% between the two halves of the "
            "averaging window (> %.1f %%): Cd is not a steady value yet"
            % (force_window_drift_pct, FORCE_STEADY_TOL_PCT)
        )
    elif in_frame["pass"]:
        verdict = "PASS_IN_FRAME"
        why = (
            "friction coefficient agrees with the Blasius laminar line to "
            "%+.2f %% (< %.1f %%); the Fr-similar experimental C_T frame is "
            "declared NOT COMPARABLE instead of failed"
            % (in_frame["err_pct"], CRITERION_BAND_PCT)
        )
    else:
        verdict = "HONEST_FAIL"
        why = (
            "in-frame friction error %+.2f %% exceeds the %.1f %% band at "
            "T/dx = %.2f" % (in_frame["err_pct"], CRITERION_BAND_PCT, geometry["T_over_dx"])
        )

    return {
        "band_pct": CRITERION_BAND_PCT,
        "frame_used_for_pass_fail": "friction_vs_blasius_laminar",
        "frames": frames,
        "geometry_verdict": geometry["verdict"],
        "force_window_drift_pct": force_window_drift_pct,
        "verdict": verdict,
        "verdict_reason": why,
        "m438_criterion": "|Cd_total - C_T(Fr=0.28)| / C_T <= 3 %",
        "m438_criterion_well_posed": False,
        "m438_root_cause": [
            "Cd_total compared against a friction-only / Fr-similar reference "
            "(category + similarity mismatch)",
            "waterline-clipped STL is open -> voxel body Cb ~ 18 (35x "
            "displacement) -> staircase wetted area ~15x smooth -> Cd inflated "
            "by the same factor",
            "no geometry gate and no force-window steady check in the runner",
        ],
    }


def run_engine_routed(engine: GeneralSimEngine, compile_mode: str | None):
    """Drive the engine per-step chain through ``compile_route`` (repo standard)."""
    sol = engine.config.solver
    out = engine.config.output

    if engine._auto_wall_treatment == WallTreatment.WALL_FUNCTION:
        print("[compile_route] wall-function path -> eager engine.run()", flush=True)
        t0 = time.time()
        info = engine.run()
        info.setdefault("compile_status", "eager")
        info.setdefault("compile_mode_effective", "eager")
        info.setdefault("compile_status_reason", "wall-function path")
        return info, time.time() - t0

    tau = engine.uc.tau
    u_in = engine.uc.u_lb
    collide_fn, collide_kwargs = engine._get_collide_fn()
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
    solid = (
        engine.solid
        if engine.solid is not None
        else torch.zeros_like(engine.f[0], dtype=torch.bool)
    )

    correct_mass_fn = None
    target_mass = None
    if sol.mass_correction:
        try:
            from tensorlbm.solver3d import correct_mass3d

            correct_mass_fn = correct_mass3d
            target_mass = engine._initial_mass
        except ImportError:
            pass

    dpS = engine._compute_dpS()
    div_check = max(1, int(os.environ.get("TL_ISFINITE_INTERVAL", sol.divergence_check_interval)))

    def _step(f: torch.Tensor) -> torch.Tensor:
        return lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs)

    step_fn = route_step(_step, compile_mode, name=f"dtmb5415[L{sol.resolution}]")

    n_steps = sol.max_steps
    t0 = time.time()
    for step in range(1, n_steps + 1):
        engine.f = step_fn(engine.f)
        engine.step_count += 1
        if (
            correct_mass_fn is not None
            and target_mass is not None
            and step % sol.mass_correction_interval == 0
        ):
            engine.f = correct_mass_fn(engine.f, target_mass)
        if out.save_forces and step % sol.force_sample_interval == 0:
            engine._sample_forces(dpS, engine.uc.nu_lb)
        if step % div_check == 0 and not torch.isfinite(engine.f).all():
            break
    elapsed = time.time() - t0
    cs = compile_status_of(step_fn)
    info = {
        "status": "completed",
        "steps": engine.step_count,
        "force_samples": len(engine.forces_log),
        "diverged": not torch.isfinite(engine.f).all().item(),
        "compile_status": cs["compile_status"],
        "compile_mode_effective": cs["compile_mode_effective"],
        "compile_status_reason": cs["compile_status_reason"],
    }
    return info, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="stl", choices=["stl", "series60"])
    _closed = _HERE / "geometry" / "dtmb5415_underwater_closed.stl"
    _default_stl = _closed if _closed.exists() else _HERE / "geometry" / "dtmb5415_underwater.stl"
    ap.add_argument("--stl-path", default=str(_default_stl))
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument(
        "--friction",
        default="mix50",
        choices=["standard", "2nd_order", "central", "lagrange", "bfl_smooth", "faces", "mix50"],
    )
    ap.add_argument(
        "--p0", default="near_wall", choices=["near_wall", "far_field", "domain_avg", "inlet"]
    )
    ap.add_argument("--out", default=None)
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    if not args.device:
        args.device = _default_device()
    compile_mode = compile_mode_from_args(args)

    L = args.resolution
    collision = (
        CollisionModel.SMAGORINSKY_MRT if args.collision == "smagorinsky" else CollisionModel.MRT
    )
    viscosity = U_PHYS * LPP / 1000.0  # Re = u*L/nu = 1000

    if args.geometry == "stl":
        geometry = GeometryConfig(
            source=GeometrySource.STL_FILE, stl_path=args.stl_path, stl_units="m"
        )
        geom_note = f"DTMB5415 underwater bare hull STL ({Path(args.stl_path).name})"
    else:
        geometry = GeometryConfig(
            source=GeometrySource.PARAMETRIC_HULL, hull_type="series60", hull_length=LPP
        )
        geom_note = "PARAMETRIC_HULL series60 (Cb=0.60 proxy -- NOT 5415)"

    out_dir = Path(
        args.out or str(_REPO_ROOT / f"results_dtmb5415_L{L}_{args.geometry}_{args.friction}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"dtmb5415_L{L}_{args.geometry}_{args.friction}",
        geometry=geometry,
        physics=PhysicsConfig(
            density=1000.0, viscosity=viscosity, inlet_velocity=U_PHYS, reference_length=LPP
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=args.steps,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=args.device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method=args.p0,
            friction_formula=args.friction,
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
            divergence_check_interval=50,
        ),
        output=OutputConfig(
            directory=str(out_dir), formats=[], save_macroscopic=False, save_forces=True
        ),
    )

    print(
        f"=== DTMB 5415 L={L} geometry={args.geometry} friction={args.friction} "
        f"steps={args.steps} device={args.device} ===",
        flush=True,
    )
    engine = GeneralSimEngine(config)
    setup = engine.setup()
    print(
        "setup:",
        json.dumps(
            {
                k: setup[k]
                for k in (
                    "Re",
                    "tau",
                    "u_lb",
                    "nu_lb",
                    "domain_lu",
                    "obstacle_cells",
                    "near_wall_cells",
                    "total_cells",
                    "device",
                    "auto_collision",
                    "auto_wall_treatment",
                )
            },
            indent=1,
        ),
        flush=True,
    )

    run_info, elapsed = run_engine_routed(engine, compile_mode)
    print(
        f"run: {run_info['status']} in {elapsed:.0f}s "
        f"({elapsed / max(run_info['steps'], 1) * 1000:.1f} ms/step) "
        f"compile={run_info.get('compile_status')}",
        flush=True,
    )

    # ---- post-process -----------------------------------------------------
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    if args.geometry == "stl":
        s_wet_phys, s_wet_lb = _stl_wetted_area_lb(Path(args.stl_path), L, DRAFT)
    else:
        R_lb = (BEAM / 2) / (LPP / L)
        s_wet_lb = math.pi * (2 * R_lb) * float(L)
        s_wet_phys = math.pi * BEAM * LPP

    wetted_bd = (
        stl_wetted_area_breakdown(Path(args.stl_path), L, DRAFT)
        if args.geometry == "stl"
        else {
            "stl": None,
            "wetted_area_m2": s_wet_phys,
            "cap_area_m2": 0.0,
            "lattice_dx_m": LPP / L,
            "wetted_area_lb": s_wet_lb,
            "cap_area_lb": 0.0,
            "n_tri_wetted": None,
            "n_tri_cap": None,
        }
    )

    L_lb = float(L)
    dpS_wet = 0.5 * u_lb**2 * s_wet_lb
    dpS_front = 0.5 * u_lb**2 * L_lb**2  # engine default for STL (resolution²)
    rescale = dpS_front / dpS_wet

    log = engine.forces_log
    n_win = min(1000, len(log))
    win = log[-n_win:]
    cd_p = sum(e["cd_pressure"] for e in win) / max(n_win, 1) * rescale
    cd_f = sum(e["cd_friction"] for e in win) / max(n_win, 1) * rescale
    cd_tot = cd_p + cd_f

    # final field: all friction formulas + p0 methods
    f_final = engine.f
    mesh = engine.mesh
    solid = engine.solid
    final_checks = {}
    for p0 in ("near_wall", "far_field", "domain_avg", "inlet"):
        fxp, _, _ = drag_pressure_integration(
            f_final, mesh, dpS_wet, extrap="none", p0_method=p0, solid=solid
        )
        row = {"cd_p": fxp}
        for formula in ("standard", "faces", "mix50"):
            fxf, _, _ = drag_friction_integration(
                f_final, mesh, dpS_wet, nu_lb, formula=formula, solid=solid
            )
            row[f"cd_f_{formula}"] = fxf
        row["cd_tot_standard"] = row["cd_p"] + row["cd_f_standard"]
        row["cd_tot_mix50"] = row["cd_p"] + row["cd_f_mix50"]
        final_checks[p0] = row

    # ---- geometry validation at the *run* resolution ---------------------
    solid_np = engine.solid.cpu().numpy().astype(np.int8)
    stair_faces = int(
        np.abs(np.diff(solid_np, axis=2)).sum()
        + np.abs(np.diff(solid_np, axis=1)).sum()
        + np.abs(np.diff(solid_np, axis=0)).sum()
    )
    geom_val = geometry_resolution_report(
        n_solid_cells=int(solid_np.sum()),
        dx_m=LPP / L,
        stair_wall_faces=stair_faces,
        smooth_wetted_lb=s_wet_lb,
        near_wall_cells=setup["near_wall_cells"],
    )

    # ---- acceptance frames ----------------------------------------------
    Re = setup["Re"]
    cf_blasius = 1.328 / math.sqrt(Re) if Re > 0 else float("nan")
    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2 if Re > 1e2 else float("nan")
    fr_run = U_PHYS / math.sqrt(9.81 * LPP)
    ct_exp = C_T_EXP[EXP_FR]
    # best achievable with a no-wave single-phase solver: friction(+form) only
    friction_only_bound = CF_ITTC57_EXP + (1.12 - 1.0) * CF_ITTC57_EXP
    err_exp_bound = (friction_only_bound - ct_exp) / ct_exp * 100

    win_stats = _window_stats([e["cd_friction"] for e in win])
    win_stats_p = _window_stats([e["cd_pressure"] for e in win])
    criteria = resistance_criteria(
        re=Re,
        cd_p=cd_p,
        cd_f=cd_f,
        cd_tot=cd_tot,
        cf_blasius=cf_blasius,
        cf_ittc57=cf_ittc,
        ct_exp=ct_exp,
        exp_fr=EXP_FR,
        fr_run=fr_run,
        no_wave_floor_pct=err_exp_bound,
        geometry=geom_val,
        force_window_drift_pct=win_stats_p["drift_pct"],
    )
    err_blasius = criteria["frames"]["friction_vs_blasius_laminar"]["err_pct"]
    err_ittc = criteria["frames"]["friction_vs_ittc1957"]["err_pct"]
    err_exp = criteria["frames"]["total_vs_exp_CT_fr0.28"]["err_pct"]

    result = {
        "case": "DTMB 5415 bare-hull resistance",
        "benchmark": "dtmb5415_resistance",
        "verified": False,
        "geometry": geom_note,
        "geometry_source": (
            "zenodo/CNR-INSEAN official DTMB 5415 baseline point cloud -> "
            "underwater clip (T=6.15m); external, see _build_geometry.py"
            if args.geometry == "stl"
            else "ship_cad Series60 parametric proxy"
        ),
        "device": args.device,
        "L_cells": L,
        "domain_lu": setup["domain_lu"],
        "n_cells": setup["total_cells"],
        "solid_cells": setup["obstacle_cells"],
        "near_wall_cells": setup["near_wall_cells"],
        "Re": Re,
        "Fr_run": fr_run,
        "collision": args.collision,
        "friction_formula_primary": args.friction,
        "n_steps": run_info["steps"],
        "elapsed_s": round(elapsed, 1),
        "compile_status": run_info.get("compile_status"),
        "compile_mode_effective": run_info.get("compile_mode_effective"),
        "dpS_wet": dpS_wet,
        "S_wet_phys_m2": s_wet_phys,
        "S_wet_lb": s_wet_lb,
        "Cd_pressure": cd_p,
        "Cd_friction": cd_f,
        "Cd_total": cd_tot,
        "references": {
            "blasius_Cf_Re1000": cf_blasius,
            "ittc57_Cf": cf_ittc,
            "exp_CT_Fr0.28": ct_exp,
            "exp_Fr": EXP_FR,
            "exp_Cf_ITTC57": CF_ITTC57_EXP,
            "exp_residual_form_wave": CR_RESIDUAL_EXP,
        },
        "errors_pct": {
            "vs_blasius_friction_Cd_f": err_blasius,
            "vs_ittc57_friction_Cd_f": err_ittc,
            "vs_exp_CT_total_Cd": err_exp,
            "note": (
                "only the first entry is a like-for-like comparison; the other "
                "two are kept for traceability with the m438 sweep and are "
                "flagged non-comparable in 'criteria'"
            ),
        },
        "criteria": criteria,
        "verdict": criteria["verdict"],
        "verdict_reason": criteria["verdict_reason"],
        "geometry_validation": geom_val,
        "wetted_area_breakdown": wetted_bd,
        "force_window": {
            "cd_friction": win_stats,
            "cd_pressure": win_stats_p,
        },
        "no_wave_friction_bound": {
            "value": friction_only_bound,
            "err_vs_exp_CT_pct": err_exp_bound,
            "note": (
                "single-phase (no free surface) LBM cannot produce wave-making; "
                "even a perfect friction+form result caps here, so the Fr-similar "
                "experimental C_T frame is unreachable by construction."
            ),
        },
        "final_field_checks": final_checks,
        "diverged": run_info["diverged"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"\nCd_p={cd_p:.6f} Cd_f={cd_f:.6f} Cd_tot={cd_tot:.6f} "
        f"(Blasius={cf_blasius:.6f}, ITTC57={cf_ittc:.6f}, exp_CT={ct_exp:.6f})",
        flush=True,
    )
    print(
        f"err Cd_f vs Blasius={err_blasius:.2f}% (in-frame, band {CRITERION_BAND_PCT:.1f}%) | "
        f"Cd_f vs ITTC57={err_ittc:.2f}% (not comparable) | "
        f"Cd_tot vs exp C_T={err_exp:.2f}% (not comparable)",
        flush=True,
    )
    print(
        f"geometry: T/dx={geom_val['T_over_dx']:.2f} B/dx={geom_val['B_over_dx']:.2f} "
        f"Cb={geom_val['cb_voxel']:.3f} ({geom_val['cb_error_pct']:+.1f}%) "
        f"A_stair/A_smooth={geom_val['stair_over_smooth_wetted_area']:.2f} "
        f"-> {geom_val['verdict']}",
        flush=True,
    )
    print(
        f"force window drift: Cd_f {win_stats['drift_pct']:+.2f}% "
        f"Cd_p {win_stats_p['drift_pct']:+.2f}% (steady tol {FORCE_STEADY_TOL_PCT:.1f}%)",
        flush=True,
    )
    print(f"VERDICT: {criteria['verdict']} -- {criteria['verdict_reason']}", flush=True)
    print(f"result -> {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
