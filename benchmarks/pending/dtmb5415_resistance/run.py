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
* ``--geometry stl`` (default): the genuine DTMB 5415 underwater bare hull
  ``geometry/dtmb5415_underwater.stl``, extracted (waterline clip at
  T = 6.15 m) from the zenodo / CNR-INSEAN official 5415 baseline point
  cloud -- see ``_build_geometry.py`` for provenance.  Voxelised through the
  shared ``tensorlbm.stl_geometry`` module by ``GeneralSimEngine``
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
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
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
LPP = 142.0        # m, length between perpendiculars
LWL = 153.0        # m, length at waterline
BEAM = 19.06       # m
DRAFT = 6.15       # m
CB = 0.507         # block coefficient
U_PHYS = 1.0e-3    # m/s (sets dt/Re only; low-Re family)

# DTMB 5415 model-test total resistance (INSEAN towing tank; second-hand
# constants from a sibling repo CI test citing Olivieri 2001 / Stern 2000).
# C_T normalised by wetted surface; Cf_ITTC57 at Re ~ 1.2e7.
C_T_EXP = {0.28: 4.2e-3, 0.35: 5.1e-3, 0.41: 6.8e-3}
CF_ITTC57_EXP = 2.91e-3          # at Re ~ 1.2e7  (Fr=0.28)
CR_RESIDUAL_EXP = 0.94e-3        # form + wave residual (Fr=0.28)
EXP_FR = 0.28


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


def _stl_wetted_area_lb(stl_path: Path, l_cells: int, z_wl: float) -> tuple[float, float]:
    """True wetted surface (both sides) of the underwater STL, in lattice units².

    Returns (S_wet_phys_m2, S_wet_lb).
    """
    from tensorlbm.stl_geometry import read_stl

    verts, faces, _ = read_stl(str(stl_path))
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    cz = verts[faces, 2].mean(axis=1)
    tri = verts[faces[cz <= z_wl]]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    s_phys = float(area.sum())
    dx = LPP / l_cells
    s_lb = s_phys / dx**2
    return s_phys, s_lb


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
    ap.add_argument("--stl-path", default=str(_HERE / "geometry" / "dtmb5415_underwater.stl"))
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument(
        "--friction", default="mix50",
        choices=["standard", "2nd_order", "central", "lagrange", "bfl_smooth", "faces", "mix50"],
    )
    ap.add_argument("--p0", default="near_wall",
                    choices=["near_wall", "far_field", "domain_avg", "inlet"])
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

    print(f"=== DTMB 5415 L={L} geometry={args.geometry} friction={args.friction} "
          f"steps={args.steps} device={args.device} ===", flush=True)
    engine = GeneralSimEngine(config)
    setup = engine.setup()
    print("setup:", json.dumps(
        {k: setup[k] for k in (
            "Re", "tau", "u_lb", "nu_lb", "domain_lu", "obstacle_cells",
            "near_wall_cells", "total_cells", "device",
            "auto_collision", "auto_wall_treatment")}, indent=1), flush=True)

    run_info, elapsed = run_engine_routed(engine, compile_mode)
    print(f"run: {run_info['status']} in {elapsed:.0f}s "
          f"({elapsed/max(run_info['steps'],1)*1000:.1f} ms/step) "
          f"compile={run_info.get('compile_status')}", flush=True)

    # ---- post-process -----------------------------------------------------
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    if args.geometry == "stl":
        s_wet_phys, s_wet_lb = _stl_wetted_area_lb(Path(args.stl_path), L, DRAFT)
    else:
        R_lb = (BEAM / 2) / (LPP / L)
        s_wet_lb = math.pi * (2 * R_lb) * float(L)
        s_wet_phys = math.pi * BEAM * LPP

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

    Re = setup["Re"]
    cf_blasius = 1.328 / math.sqrt(Re) if Re > 0 else float("nan")
    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2 if Re > 1e2 else float("nan")

    # experimental frame (Fr similarity) -- NOT self-consistent with Re=1000
    fr_run = U_PHYS / math.sqrt(9.81 * LPP)
    ct_exp = C_T_EXP[EXP_FR]
    err_blasius = (cd_tot - cf_blasius) / cf_blasius * 100
    err_ittc = (cd_tot - cf_ittc) / cf_ittc * 100
    err_exp = (cd_tot - ct_exp) / ct_exp * 100
    # best achievable with a no-wave single-phase solver: friction(+form) only
    friction_only_bound = CF_ITTC57_EXP + (1.12 - 1.0) * CF_ITTC57_EXP
    err_exp_bound = (friction_only_bound - ct_exp) / ct_exp * 100

    result = {
        "case": "DTMB 5415 bare-hull resistance",
        "benchmark": "dtmb5415_resistance",
        "verified": False,
        "geometry": geom_note,
        "geometry_source": (
            "zenodo/CNR-INSEAN official DTMB 5415 baseline point cloud -> "
            "underwater clip (T=6.15m); external, see _build_geometry.py"
            if args.geometry == "stl" else "ship_cad Series60 parametric proxy"
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
            "vs_blasius": err_blasius,
            "vs_ittc57": err_ittc,
            "vs_exp_CT": err_exp,
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
    print(f"\nCd_p={cd_p:.6f} Cd_f={cd_f:.6f} Cd_tot={cd_tot:.6f} "
          f"(Blasius={cf_blasius:.6f}, ITTC57={cf_ittc:.6f}, exp_CT={ct_exp:.6f})", flush=True)
    print(f"err vs Blasius={err_blasius:.2f}%  vs ITTC57={err_ittc:.2f}%  "
          f"vs exp C_T={err_exp:.2f}%  (no-wave bound err={err_exp_bound:.2f}%)", flush=True)
    print(f"result -> {out_dir/'result.json'}", flush=True)


if __name__ == "__main__":
    main()