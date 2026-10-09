#!/usr/bin/env python3
"""Series 60 (Cb=0.60) bare-hull resistance benchmark runner.

Reuses the verified ``suboff_re1000`` end-to-end chain:
  GeneralSimEngine setup -> compile-routed ``lbm_step_correct`` step ->
  ``drag_pressure.drag_pressure_integration`` / ``drag_friction_integration``
  -> wetted-area normalisation (repo Re=1000 family).

What is new
-----------
* ``GeometrySource.PARAMETRIC_HULL`` is *not* implemented by the base
  ``GeneralSimEngine`` (``_build_solid_mask`` has no branch for it and returns
  an EMPTY mask -- the `dtmb5415_resistance` ``--geometry series60`` path is
  therefore a no-op).  ``Series60Engine`` subclasses the engine and supplies
  the Series 60 solid mask (``ship_cad.series60_hull_mask``), the domain
  bounding box and gradient-derived surface normals.
* The reference wetted area is the *analytic* wetted surface of the parametric
  Series 60 hull (lateral both-sides + waterline "deck"): in this single-phase
  model the waterplane is a solid wall in contact with fluid, so it is part of
  the wetted surface (this is the explicit no-free-surface caliber).

Reference frames (multi-source; see README)
-------------------------------------------
* Blasius laminar Cf = 1.328/sqrt(Re) at the run Re (the repo Re=1000 family
  frame used by the verified ``suboff_re1000`` case) -- friction-dominated.
* ITTC-1957 Cf = 0.075/(log10 Re - 2)^2 (the friction line used by the Todd
  Series 60 model-test reduction).
* Todd (1963) DTMB Series 60 Cb=0.60 model-test C_T(Fr) -- turbulent model
  Re ~ 1e6-1e7 and requires wave-making; recorded but EXPLICITLY NOT applicable
  to a single-phase Re=1000 laminar run (no free surface -> no wave drag).

Usage:
  PYTHONPATH=src python run.py --resolution 72 --steps 12000 --device sdaa:0
      [--collision mrt] [--friction mix50] [--out DIR]
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
from tensorlbm.ship_cad import _series60_half_beam, series60_hull_mask  # noqa: E402

# --------------------------------------------------------------------------- #
# Series 60 model geometry (Todd 20-ft model convention) and physics
# --------------------------------------------------------------------------- #
L_M = 6.096          # m, Todd Series 60 standard model length (20 ft)
L_B = 6.5            # length/beam
B_T = 2.74           # beam/draft  (Series 60 Cb=0.60 proportions)
B_M = L_M / L_B      # beam [m]
T_M = B_M / B_T      # draft [m]
CB = 0.60
U_PHYS = 1.0e-3      # m/s (sets dt / Re only; Re=1000 family)

# Todd (1963) Series 60 Cb=0.60 model-test total-resistance coefficient, Cb model
# L=20 ft towed in fresh water.  Values are commonly-cited anchors for the
# Cb=0.60 series (turbulent model Re ~ 2.6e6 at Fr=0.20).  SECOND-HAND,
# approximate -- recorded for cross-check only, NOT used as the pass/fail
# reference (see README): single-phase LBM cannot make waves.
TODD_CT = {0.10: 3.30e-3, 0.15: 3.60e-3, 0.20: 4.30e-3, 0.25: 5.60e-3, 0.30: 7.60e-3}
TODD_MODEL_L_M = 6.096
TODD_NU_M2S = 1.139e-6  # fresh water ~15.5 C


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


# --------------------------------------------------------------------------- #
# Engine subclass: series60 parametric solid + bounding box + normals
# --------------------------------------------------------------------------- #
class Series60Engine(GeneralSimEngine):
    """GeneralSimEngine whose PARAMETRIC_HULL path builds the Series 60 hull."""

    # placed at cx = nx*0.25, cy = ny*0.5, keel at nz*0.5 - T_lb/2
    def _geometry_bounding_box(self) -> tuple[float, ...]:
        return (-L_M / 2.0, L_M / 2.0, -B_M / 2.0, B_M / 2.0, -T_M / 2.0, T_M / 2.0)

    def _build_solid_mask(self, nx, ny, nz, device):
        geo = self.config.geometry
        dx = self.config.physics.reference_length / self.config.solver.resolution
        length_lb = geo.hull_length / dx
        beam_lb = B_M / dx
        draft_lb = T_M / dx
        self._length_lb = length_lb
        self._beam_lb = beam_lb
        self._draft_lb = draft_lb
        cx = nx * 0.25
        cy = ny * 0.5
        cz_keel = nz * 0.5 - draft_lb / 2.0
        self._cx, self._cy, self._cz_keel = cx, cy, cz_keel
        return series60_hull_mask(
            nx, ny, nz, cx, cy, cz_keel, length_lb, beam_lb, draft_lb, device
        )

    def _build_surface_mesh(self, device):
        # Gradient normals for the general hull (base else-branch does this,
        # but PARAMETRIC_HULL is not routed there explicitly -> be explicit).
        from tensorlbm.drag_pressure import SurfaceMesh

        return SurfaceMesh.from_gradient(self.solid, self.near)


def analytic_wetted_areas(length_lb: float, beam_lb: float, draft_lb: float) -> dict:
    """Analytic Series 60 wetted surface (lattice units^2).

    Lateral (both sides) + waterline deck.  The hull is described by
    half-beam_b(ξ,ζ) = (B/2)(1-ξ^2)^0.51 ζ^0.30 over ξ∈[-1,1], ζ∈[0,1].
    """
    x = np.linspace(-1.0, 1.0, 2001)
    z = np.linspace(0.0, 1.0, 1001)
    Xg, Zg = np.meshgrid(x, z, indexing="ij")
    Yg = _series60_half_beam(Xg, Zg)
    dYdx = np.gradient(Yg, x, axis=0)
    dYdz = np.gradient(Yg, z, axis=1)
    # physical partials (Yphys=(B/2)Yg, xphys=(L/2)ξ, zphys=T ζ)
    dYdx_p = (beam_lb / 2.0) * dYdx / (length_lb / 2.0)
    dYdz_p = (beam_lb / 2.0) * dYdz / draft_lb
    gram = np.sqrt(1.0 + dYdx_p**2 + dYdz_p**2)
    s_lat = 2.0 * float(np.trapezoid(np.trapezoid(gram, z, axis=1) * draft_lb, x)) * (length_lb / 2.0)
    # deck: flat top at ζ=1, area = ∫ 2*(B/2)*(1-ξ^2)^0.51 dξ * (L/2)
    deck_hb = _series60_half_beam(x, np.ones_like(x))
    s_deck = 2.0 * float(np.trapezoid(deck_hb, x)) * (beam_lb / 2.0) * (length_lb / 2.0)
    return {
        "S_lateral_lb2": s_lat,
        "S_deck_lb2": s_deck,
        "S_total_lb2": s_lat + s_deck,
    }


def near_wall_face_count(solid: torch.Tensor) -> int:
    """Number of fluid-solid faces (voxel staircase wet faces)."""
    n = 0
    for dim in (0, 1, 2):
        n += int((solid & ~torch.roll(solid, 1, dim)).sum().item())
        n += int((solid & ~torch.roll(solid, -1, dim)).sum().item())
    return n


# --------------------------------------------------------------------------- #
# Compile-routed driver (identical to verified suboff_re1000)
# --------------------------------------------------------------------------- #
def run_engine_routed(engine: GeneralSimEngine, compile_mode: str | None):
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
    nu_lb = engine.uc.nu_lb
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

    step_fn = route_step(_step, compile_mode, name=f"series60[L{sol.resolution}]")

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
            engine._sample_forces(dpS, nu_lb)
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
    ap.add_argument("--resolution", type=int, default=72, help="cells per hull length L")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument(
        "--friction",
        default="mix50",
        choices=["standard", "2nd_order", "central", "lagrange", "faces", "mix50"],
    )
    ap.add_argument("--p0", default="near_wall",
                    choices=["near_wall", "far_field", "domain_avg", "inlet"])
    ap.add_argument("--area-caliber", default="total",
                    choices=["total", "lateral", "cylinder"],
                    help="wetted-area reference: analytic total (lat+deck), lateral only, or pi*B*L")
    ap.add_argument("--save-field", action="store_true")
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
    viscosity = U_PHYS * L_M / 1000.0  # Re = u*L/nu = 1000

    out_dir = Path(args.out or str(_REPO_ROOT / f"results_series60_L{L}_{args.friction}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"series60_re1000_L{L}_{args.friction}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_HULL, hull_type="series60", hull_length=L_M
        ),
        physics=PhysicsConfig(
            density=1000.0, viscosity=viscosity, inlet_velocity=U_PHYS, reference_length=L_M
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            # streamwise padding kept at the verified suboff convention (1L up,
            # 4L down); lateral/vertical padding reduced 1.0L -> 0.6L because the
            # Series 60 hull occupies a large fraction of the domain at these
            # aspect ratios (blockage ~0.6% at 0.6L, negligible) and the full
            # 1.0L lateral domain makes the cost prohibitive at the resolutions
            # needed to resolve the thin draft (T/L = 1/17.8).
            domain_padding=(1.0, 4.0, 0.6, 0.6, 0.6, 0.6),
            max_steps=args.steps,
            warmup_steps=None,
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
        f"=== Series60 Cb=0.60 Re=1000 L={L} collision={args.collision} "
        f"friction={args.friction} steps={args.steps} device={args.device} ===",
        flush=True,
    )
    engine = Series60Engine(config)
    setup = engine.setup()
    print("setup:", json.dumps(
        {k: setup[k] for k in (
            "Re", "tau", "u_lb", "nu_lb", "domain_lu", "obstacle_cells",
            "near_wall_cells", "total_cells", "device",
            "auto_collision", "auto_wall_treatment")}, indent=1), flush=True)
    if setup["obstacle_cells"] == 0:
        raise SystemExit("FATAL: empty solid mask — Series60 geometry not built")

    # geometry diagnostics
    L_lb = engine._length_lb
    B_lb = engine._beam_lb
    T_lb = engine._draft_lb
    areas = analytic_wetted_areas(L_lb, B_lb, T_lb)
    n_faces = near_wall_face_count(engine.solid)
    from tensorlbm.ship_cad import hull_block_coefficient
    cb_num = hull_block_coefficient(engine.solid, B_lb, T_lb, L_lb)
    geom = {
        "L_M": L_M, "B_M": B_M, "T_M": T_M, "L_B": L_B, "B_T": B_T,
        "L_lb": L_lb, "B_lb": B_lb, "T_lb": T_lb,
        "half_beam_cells": B_lb / 2.0, "draft_cells": T_lb,
        "Cb_theory": CB, "Cb_numeric": cb_num,
        "S_lateral_lb2": areas["S_lateral_lb2"],
        "S_deck_lb2": areas["S_deck_lb2"],
        "S_total_lb2": areas["S_total_lb2"],
        "S_cylinder_piBL_lb2": math.pi * B_lb * L_lb,
        "n_wet_faces": n_faces,
        "n_wet_faces_area_lb2": float(n_faces),
        "near_wall_cells": int(engine.near.sum().item()),
    }
    print("geometry:", json.dumps(geom, indent=1), flush=True)

    run_info, elapsed = run_engine_routed(engine, compile_mode)
    print(
        f"run finished: {run_info['status']} in {elapsed:.0f}s "
        f"({elapsed / max(run_info['steps'], 1) * 1000:.1f} ms/step) "
        f"compile={run_info.get('compile_status')}", flush=True,
    )

    # ---- post-process ----
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    dpS_front = engine._compute_dpS()  # generic resolution^2 (engine default)

    Re = setup["Re"]
    cf_blasius = 1.328 / math.sqrt(Re)
    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2
    fr_run = U_PHYS / math.sqrt(9.81 * L_M)

    log = engine.forces_log
    f_final = engine.f
    mesh = engine.mesh
    solid = engine.solid
    n_win = min(1000, len(log))
    win = log[-n_win:]
    raw_p = sum(e["cd_pressure"] for e in win) / max(n_win, 1)
    raw_f = sum(e["cd_friction"] for e in win) / max(n_win, 1)

    # every candidate wetted-area caliber (lattice units^2), evaluated on the
    # SAME field so the choice of reference area is auditable in one run.
    calibers = {
        "analytic_total": areas["S_total_lb2"],      # smooth lateral + waterline deck
        "analytic_lateral": areas["S_lateral_lb2"],  # smooth lateral only (ships' S_wet)
        "wet_faces": float(n_faces),                 # voxel staircase wet faces
        "cylinder_piBL": math.pi * B_lb * L_lb,      # barge/cylinder proxy
    }
    cad = {}
    for name, S_ref in calibers.items():
        dpS_c = 0.5 * u_lb**2 * S_ref
        resc = dpS_front / dpS_c
        cp = raw_p * resc
        cf = raw_f * resc
        row = {
            "S_ref_lb2": S_ref,
            "rescale": resc,
            "cd_p_window": cp,
            "cd_f_mix50_window": cf,
            "cd_tot_mix50_window": cp + cf,
        }
        fxp, _, _ = drag_pressure_integration(
            f_final, mesh, dpS_c, extrap="none", p0_method=args.p0, solid=solid)
        row["cd_p_final"] = fxp
        for formula in ("standard", "faces", "mix50"):
            fxf, _, _ = drag_friction_integration(
                f_final, mesh, dpS_c, nu_lb, formula=formula, solid=solid)
            row[f"cd_f_{formula}"] = fxf
        row["cd_tot_standard"] = fxp + row["cd_f_standard"]
        row["cd_tot_faces"] = fxp + row["cd_f_faces"]
        row["cd_tot_mix50"] = fxp + row["cd_f_mix50"]
        row["err_total_mix50_pct_vs_Blasius"] = (row["cd_tot_mix50"] - cf_blasius) / cf_blasius * 100.0
        row["err_friction_mix50_pct_vs_Blasius"] = (row["cd_f_mix50"] - cf_blasius) / cf_blasius * 100.0
        cad[name] = row

    caliber_map = {
        "total": "analytic_total",
        "lateral": "analytic_lateral",
        "faces": "wet_faces",
        "cylinder": "cylinder_piBL",
    }
    primary = caliber_map[args.area_caliber]
    S_ref = calibers[primary]
    dpS_wet = 0.5 * u_lb**2 * S_ref
    rescale = dpS_front / dpS_wet
    cd_p = raw_p * rescale
    cd_f = raw_f * rescale
    cd_tot = cd_p + cd_f
    n_win2 = min(500, len(log))
    win2 = log[-n_win2:]
    cd_tot2 = (sum(e["cd_total"] for e in win2) / max(n_win2, 1)) * rescale

    # final-field recompute across formulas / p0 (primary caliber)
    final_checks = {}
    for p0 in ("near_wall", "far_field", "domain_avg", "inlet"):
        fx_p, _, _ = drag_pressure_integration(
            f_final, mesh, dpS_wet, extrap="none", p0_method=p0, solid=solid)
        row = {"cd_p": fx_p}
        for formula in ("standard", "2nd_order", "central", "lagrange", "faces", "mix50"):
            fxf, _, _ = drag_friction_integration(
                f_final, mesh, dpS_wet, nu_lb, formula=formula, solid=solid)
            row[f"cd_f_{formula}"] = fxf
        row["cd_tot_standard"] = row["cd_p"] + row["cd_f_standard"]
        row["cd_tot_mix50"] = row["cd_p"] + row["cd_f_mix50"]
        final_checks[p0] = row

    if args.save_field:
        torch.save(f_final.cpu(), out_dir / "final_field.pt")

    tod = {}
    for fr, ct in TODD_CT.items():
        v = fr * math.sqrt(9.81 * TODD_MODEL_L_M)
        re_t = v * TODD_MODEL_L_M / TODD_NU_M2S
        tod[f"Fr{fr:.2f}"] = {
            "C_T": ct,
            "model_Re": re_t,
            "C_F_ittc57": 0.075 / (math.log10(re_t) - 2.0) ** 2,
        }

    err_blasius = (cd_tot - cf_blasius) / cf_blasius * 100.0
    ref = {
        "primary_caliber": primary,
        "blasius_Cf": cf_blasius,
        "ittc57_Cf_Re1000": cf_ittc,
        "ittc57_note": "turbulent line; not physical at Re=1000, recorded for the Todd reduction",
        "todd_series60_Cb0.60": tod,
        "todd_note": (
            "Todd (1963) DTMB Series 60 Cb=0.60 model tests: turbulent model Re~2.6e6, "
            "C_T includes wave-making. Single-phase LBM has NO free surface -> cannot "
            "reproduce C_T at finite Fr; recorded for cross-check only."
        ),
    }

    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(log) * frac)
        seg = log[max(0, k - n_win):k]
        if seg:
            conv[f"{int(frac * 100)}%"] = round(
                sum(e["cd_total"] for e in seg) / len(seg) * rescale, 6)

    result = {
        "case": "Series 60 (Cb=0.60) bare-hull resistance Re=1000",
        "benchmark": "series60_resistance",
        "device": args.device,
        "geometry": "ship_cad series60 parametric hull (Cb=0.60)",
        "geometry_note": (
            "PARAMETRIC_HULL series60 built by Series60Engine override; base "
            "GeneralSimEngine has no PARAMETRIC_HULL mask branch (returns empty)."
        ),
        "grid": f"{setup['domain_lu'][0]}x{setup['domain_lu'][1]}x{setup['domain_lu'][2]}",
        "domain_lu": setup["domain_lu"],
        "n_cells": setup["total_cells"],
        "L_cells": L,
        "Re": Re,
        "Fr_run": fr_run,
        "u_lb": u_lb,
        "nu_lb": nu_lb,
        "tau": setup["tau"],
        "collision": args.collision,
        "friction_formula_primary": args.friction,
        "compile_mode": compile_mode,
        "compile_mode_effective": run_info.get("compile_mode_effective"),
        "compile_status": run_info.get("compile_status"),
        "compile_status_reason": run_info.get("compile_status_reason"),
        "n_steps": run_info["steps"],
        "elapsed_s": round(elapsed, 1),
        "ms_per_step": round(elapsed / max(run_info["steps"], 1) * 1000, 2),
        "area_caliber": args.area_caliber,
        "geometry_stats": geom,
        "dpS_front_engine": dpS_front,
        "dpS_wet": dpS_wet,
        "rescale_front_to_wet": rescale,
        "pressure_extrap": "none",
        "p0_method": args.p0,
        "Cd_pressure": cd_p,
        "Cd_friction": cd_f,
        "Cd_total": cd_tot,
        "Cd_total_last500": cd_tot2,
        "references": ref,
        "cad_per_caliber": cad,
        "error_pct_vs_Blasius": err_blasius,
        "error_pct_friction_vs_Blasius": (cd_f - cf_blasius) / cf_blasius * 100.0,
        "convergence_windows": conv,
        "final_field_checks": final_checks,
        "diverged": run_info["diverged"],
        "finite": bool(torch.isfinite(engine.f).all().item()),
        "solid_cells": int(engine.solid.sum().item()),
        "near_wall_cells": int(engine.near.sum().item()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in (
        "Cd_pressure", "Cd_friction", "Cd_total", "references",
        "error_pct_vs_Blasius", "error_pct_friction_vs_Blasius",
        "convergence_windows")}, indent=1), flush=True)
    print(f"RESULT Cd_p={cd_p:.6f} Cd_f={cd_f:.6f} Cd_tot={cd_tot:.6f} "
          f"(Blasius {cf_blasius:.6f}) err={err_blasius:+.2f}%", flush=True)
    print(f"result -> {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()