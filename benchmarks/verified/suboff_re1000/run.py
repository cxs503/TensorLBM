#!/usr/bin/env python3
"""B6: DARPA SUBOFF bare-hull drag, Re=1000 — GeneralSimEngine benchmark runner.

Reference conventions (repo family, consistent with historical runs):
  - Normalization: wetted area S = pi*D*L  (D = 2*R_max, L = hull length),
    dpS = 0.5 * u_lb^2 * pi * D_lb * L_lb
  - Reference: Blasius laminar flat-plate Cf = 1.328/sqrt(Re) = 0.0420 (Re=1000)
    (the "Ct ~ 0.004" number in benchmarks/TODO.md belongs to the AFF-8 full
    scale Re=2e6 experiment, NOT to Re=1000 — see SUMMARY_REPORT.txt)
  - pressure_extrap = 'none'  (verified-benchmark rule, no extrapolation)

Verified-benchmark standard (2026-08-19): the whole time-stepping chain is
routed through the shared compile module ``benchmarks.compile_route``
(``route_step`` -> ``tensorlbm.compile_utils.compile_step``), i.e.
``torch.compile`` by default and ``--compile-mode eager`` for the A/B path.
The step-dependent branches (mass-correction cadence, force sampling,
divergence guard) stay OUTSIDE the compiled closure, in the eager driver loop.

Usage:
  python run.py [--resolution 80] [--steps 20000] [--device cuda:2]
                [--collision mrt|smagorinsky] [--out DIR]
                [--friction standard|mix50|...] [--compile-mode default|eager]

Simulation runs through GeneralSimEngine (common-module entry point);
force post-processing reuses the same common modules
(drag_pressure.drag_pressure_integration / drag_friction_integration)
with the wetted-area reference area.
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks


def _default_device() -> str:
    """Pick the first available accelerator on this host.

    Historical runs hardcoded ``cuda:2``; this host is a Hygon/SDAA node
    (32 SDAA devices, no CUDA).  Prefer SDAA, then CUDA, else CPU.
    """
    try:
        import torch_sdaa  # noqa: F401

        if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
            return "sdaa:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


import torch

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries3d import far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    drag_friction_integration,
    drag_pressure_integration,
    suboff_smooth_q,
)
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402
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

SUBOFF_LENGTH_M = 4.356  # DARPA SUBOFF bare hull length [m]
SUBOFF_RADIUS_M = 0.254  # max radius [m]  (L/D = 8.57)
U_PHYS = 1.0e-3  # m/s (any value; sets dt only)


def wetted_dpS(u_lb: float, radius_lb: float, length_lb: float) -> float:
    """dpS with wetted-area reference S = pi*D*L (repo Re=1000 family)."""
    return 0.5 * u_lb**2 * math.pi * (2.0 * radius_lb) * length_lb


def frontal_dpS(u_lb: float, radius_lb: float) -> float:
    """dpS with frontal-area reference S = pi*R^2 (engine default)."""
    return 0.5 * u_lb**2 * math.pi * radius_lb**2


def run_engine_routed(engine: GeneralSimEngine, compile_mode: str | None) -> tuple[dict, float]:
    """Drive ``GeneralSimEngine``'s time-stepping chain through ``compile_route``.

    Replicates ``GeneralSimEngine.run`` for the plain bounce-back path so the
    whole per-step chain (collision -> NoDynamics -> half-way BB -> streaming
    -> far-field BC) is routed through the shared
    :func:`benchmarks.compile_route.route_step` (``torch.compile`` by default),
    per the verified-benchmark standard (benchmarks/compile_route.py).

    Every step-dependent branch — the mass-correction cadence, the force
    sampling, the snapshot cadence and the divergence guard — stays OUTSIDE the
    compiled closure, in the eager driver loop (compile_route rule #2).  The
    compiled closure is accordingly the *pure* ``f -> f'`` chain; mass
    correction is re-applied here with exactly the cadence
    ``step % mass_correction_interval == 0`` that ``lbm_step_correct`` used
    internally, so the evolution is unchanged and only the routing differs.

    The wall-function / BFL / thermal / VOF paths are not part of the compiled
    chain; if the auto-selected wall treatment is the wall function this falls
    back to the engine's own (eager) loop and says so in the log.

    Returns ``(run_info, elapsed_s)`` — same ``run_info`` dict as
    ``GeneralSimEngine.run``.
    """
    sol = engine.config.solver
    out = engine.config.output

    if engine._auto_wall_treatment == WallTreatment.WALL_FUNCTION:
        print(
            "[compile_route] suboff_re1000: wall-function path not in compiled "
            "chain -> eager engine.run()",
            flush=True,
        )
        t0 = time.time()
        info = engine.run()
        return info, time.time() - t0

    tau = engine.uc.tau
    nu_lb = engine.uc.nu_lb
    u_in = engine.uc.u_lb

    collide_fn, collide_kwargs = engine._get_collide_fn()
    far_field_fn = functools.partial(
        far_field_bc_3d, bc_config=engine._build_bc_config()
    )
    solid = (
        engine.solid
        if engine.solid is not None
        else torch.zeros_like(engine.f[0], dtype=torch.bool)
    )

    # Mass correction: step-dependent branch -> kept in the eager loop below.
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
    div_check = max(
        1, int(os.environ.get("TL_ISFINITE_INTERVAL", sol.divergence_check_interval))
    )

    def _step(f: torch.Tensor) -> torch.Tensor:
        # Whole per-step chain, pure tensor function f -> f' (no host sync,
        # no step index): collide -> NoDynamics -> half-way BB -> stream -> BC.
        return lbm_step_correct(
            f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs
        )

    step_fn = route_step(_step, compile_mode, name=f"suboff_re1000[L{sol.resolution}]")

    n_steps = sol.max_steps
    t0 = time.time()
    for step in range(1, n_steps + 1):
        engine.f = step_fn(engine.f)
        engine.step_count += 1
        # ---- step-dependent branches stay eager (outside the compiled graph) ----
        if (
            correct_mass_fn is not None
            and target_mass is not None
            and step % sol.mass_correction_interval == 0
        ):
            engine.f = correct_mass_fn(engine.f, target_mass)
        if out.save_forces and step % sol.force_sample_interval == 0:
            engine._sample_forces(dpS, nu_lb)
        if out.save_macroscopic and step % sol.snapshot_interval == 0:
            engine._save_snapshot()
        if step % div_check == 0 and not torch.isfinite(engine.f).all():
            break
    elapsed = time.time() - t0

    info = {
        "status": "completed",
        "steps": engine.step_count,
        "snapshots": len(engine.snapshots),
        "force_samples": len(engine.forces_log),
        "diverged": not torch.isfinite(engine.f).all().item(),
    }
    return info, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=80, help="cells per hull length L")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default=None, help="torch device, e.g. sdaa:0 / cuda:0 (auto if unset)")
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument(
        "--friction",
        default="standard",
        choices=[
            "standard",
            "2nd_order",
            "central",
            "lagrange",
            "bfl",
            "bfl_smooth",
            "bfl_lagrange",
            "faces",
            "mix50",
        ],
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
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0  # Re = u*L/nu = 1000

    out_dir = Path(
        args.out
        or str(_REPO_ROOT / f"results_bench_b6_suboff_re1000_L{L}_{args.collision}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"bench_b6_suboff_re1000_L{L}_{args.collision}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),  # streamwise 6L, lateral 2L+D
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
        ),
        output=OutputConfig(
            directory=str(out_dir),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    print(
        f"=== B6 SUBOFF Re=1000 L={L} collision={args.collision} "
        f"steps={args.steps} device={args.device} ===",
        flush=True,
    )
    engine = GeneralSimEngine(config)
    setup_info = engine.setup()
    print(
        "setup:",
        json.dumps(
            {
                k: setup_info[k]
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
        f"run finished: {run_info['status']} in {elapsed:.0f}s "
        f"({elapsed / max(run_info['steps'], 1) * 1000:.1f} ms/step) "
        f"compile_mode={compile_mode!r}",
        flush=True,
    )

    # ---- post-process: wetted-area coefficients ----
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L)  # 4.6667 for L=80
    dpS_wet = wetted_dpS(u_lb, R_lb, float(L))
    dpS_front = frontal_dpS(u_lb, R_lb)
    rescale = dpS_front / dpS_wet  # frontal->wetted factor

    # Primary metric: window mean of engine force samples (rescaled to wetted)
    log = engine.forces_log
    n_win = min(1000, len(log))
    win = log[-n_win:]
    cd_p_wet = sum(e["cd_pressure"] for e in win) / n_win * rescale
    cd_f_wet = sum(e["cd_friction"] for e in win) / n_win * rescale
    cd_tot_wet = cd_p_wet + cd_f_wet
    # last-5000-step window too (500 samples)
    n_win2 = min(500, len(log))
    win2 = log[-n_win2:]
    cd_tot_wet2 = (sum(e["cd_total"] for e in win2) / n_win2) * rescale

    # Final-field recomputation with several friction formulas / p0 methods
    final_checks = {}
    f_final = engine.f
    mesh = engine.mesh
    solid = engine.solid

    # q_smooth: true distance from each near-wall cell centre to the SMOOTH
    # hull surface (BFL correction; formula='bfl_smooth').
    # Engine places the hull axis along x at (nx*0.25, ny*0.5, nz*0.5).
    nz_g, ny_g, nx_g = solid.shape
    q_smooth = suboff_smooth_q(
        solid, engine.near, nx_g * 0.25, ny_g * 0.5, nz_g * 0.5, float(L), R_lb
    )
    q_near = q_smooth[engine.near]
    q_stats = {
        "mean": float(q_near.mean().item()),
        "min": float(q_near.min().item()),
        "max": float(q_near.max().item()),
    }

    # Uniform-field faces/standard ratio (pure geometry): with u = const the
    # two formulas differ only by stair-face count vs near-cell weighting.
    # The real flow realises only a fraction of this geometric gain
    # (SUBOFF L=96: uniform 1.4551 vs actual 1.269) — the interpolation
    # weight for the ratio-calibrated scheme is derived from it.
    u_uni = torch.full_like(solid, u_lb, dtype=torch.float32)
    f_uni = equilibrium3d(
        torch.ones_like(u_uni), u_uni, torch.zeros_like(u_uni), torch.zeros_like(u_uni)
    )
    cd_f_uni_std = drag_friction_integration(f_uni, mesh, dpS_wet, nu_lb, formula="standard")[0]
    cd_f_uni_faces = drag_friction_integration(
        f_uni, mesh, dpS_wet, nu_lb, formula="faces", solid=solid
    )[0]
    ratio_uniform = cd_f_uni_faces / cd_f_uni_std
    del f_uni, u_uni

    for p0 in ("near_wall", "far_field", "domain_avg", "inlet"):
        fx_p, _, _ = drag_pressure_integration(
            f_final, mesh, dpS_wet, extrap="none", p0_method=p0, solid=solid
        )
        row = {"cd_p": fx_p}
        for formula in (
            "standard",
            "2nd_order",
            "central",
            "lagrange",
            "bfl_smooth",
            "faces",
            "mix50",
        ):
            kwargs = {"solid": solid}
            if formula == "bfl_smooth":
                kwargs["q_wall"] = q_smooth
            fx_f, _, _ = drag_friction_integration(
                f_final, mesh, dpS_wet, nu_lb, formula=formula, **kwargs
            )
            row[f"cd_f_{formula}"] = fx_f
        row["cd_tot_standard"] = row["cd_p"] + row["cd_f_standard"]
        row["cd_tot_mix50"] = row["cd_p"] + row["cd_f_mix50"]
        # Ratio-calibrated weight: the actual flow realises only
        # (faces/std - 1) of the uniform-field geometric gain
        # (uniform_ratio - 1); w = 1 - gain_actual/gain_geometric.
        gain_actual = row["cd_f_faces"] / row["cd_f_standard"] - 1.0
        w_ratio = 1.0 - gain_actual / (ratio_uniform - 1.0)
        row["gain_faces_over_standard"] = gain_actual
        row["uniform_faces_standard_ratio"] = ratio_uniform
        row["w_ratio_calibrated"] = w_ratio
        row["cd_f_weighted_ratio"] = (
            w_ratio * row["cd_f_standard"] + (1.0 - w_ratio) * row["cd_f_faces"]
        )
        row["cd_tot_weighted_ratio"] = row["cd_p"] + row["cd_f_weighted_ratio"]
        final_checks[p0] = row

    # Persist the converged field so later formula studies can recompute
    # without re-running the simulation (1.8 GB float32).
    field_path = out_dir / "final_field.pt"
    torch.save(f_final.cpu(), field_path)
    print(f"final field saved to {field_path}", flush=True)

    cf_ref = 1.328 / math.sqrt(1000.0)
    ref_name = "Blasius Cf=1.328/sqrt(Re)=0.0420 (wetted-area pi*D*L)"
    err_pct = (cd_tot_wet - cf_ref) / cf_ref * 100.0

    # time-convergence: window means at several checkpoints
    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(log) * frac)
        seg = log[max(0, k - n_win) : k]
        if seg:
            conv[f"{int(frac * 100)}%"] = round(
                sum(e["cd_total"] for e in seg) / len(seg) * rescale, 6
            )

    result = {
        "case": "B6 SUBOFF bare hull Re=1000",
        "benchmark": "suboff_re1000",
        "device": args.device,
        "grid": f"{setup_info['domain_lu'][0]}x{setup_info['domain_lu'][1]}x{setup_info['domain_lu'][2]}",
        "domain_lu": setup_info["domain_lu"],
        "n_cells": setup_info["total_cells"],
        "L_cells": L,
        "R_lb": R_lb,
        "L_D": SUBOFF_LENGTH_M / (2 * SUBOFF_RADIUS_M),
        "Re": setup_info["Re"],
        "u_lb": u_lb,
        "nu_lb": nu_lb,
        "tau": setup_info["tau"],
        "collision": args.collision,
        "Cs": 0.05 if args.collision == "smagorinsky" else None,
        "compile_mode": compile_mode,
        "n_steps": run_info["steps"],
        "elapsed_s": round(elapsed, 1),
        "ms_per_step": round(elapsed / max(run_info["steps"], 1) * 1000, 2),
        "dpS_type": "wetted_area_0.5*u^2*pi*D*L",
        "dpS_wetted": dpS_wet,
        "dpS_engine_frontal": dpS_front,
        "rescale_frontal_to_wetted": rescale,
        "pressure_extrap": "none",
        "p0_method": args.p0,
        "friction_formula_primary": args.friction,
        "Cd_pressure": cd_p_wet,
        "Cd_friction": cd_f_wet,
        "Cd_total": cd_tot_wet,
        "Cd_total_last5000": cd_tot_wet2,
        "Cf_ref": cf_ref,
        "ref_name": ref_name,
        "ref_note": (
            "Note: benchmarks/TODO.md lists 'Ct=0.004 (exp)' for B6; "
            "that value is the AFF-8 full-scale Re=2e6 total-drag "
            "coefficient. At Re=1000 the repo family reference is "
            "Blasius Cf=1.328/sqrt(Re)=0.0420 (wetted-area pi*D*L) — "
            "the same frame in which historical errors 3.8% (CUDA) / "
            "3.6% (SDAA) were measured."
        ),
        "error_pct_vs_Blasius": err_pct,
        "window_samples": n_win,
        "window_steps": n_win * 10,
        "convergence_windows": conv,
        "final_field_checks": final_checks,
        "q_smooth_stats": q_stats,
        "final_field_saved": True,
        "finite": bool(torch.isfinite(engine.f).all().item()),
        "diverged": run_info.get("diverged", False),
        "solid_cells": int(engine.solid.sum().item()) if engine.solid is not None else None,
        "near_wall_cells": int(engine.near.sum().item()) if engine.near is not None else None,
        "engine_modules": setup_info.get("modules_used", []),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "Cd_pressure",
                    "Cd_friction",
                    "Cd_total",
                    "Cd_total_last5000",
                    "Cf_ref",
                    "error_pct_vs_Blasius",
                    "convergence_windows",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"results written to {result_path}", flush=True)

    # quick summary line for logs
    print(
        f"RESULT Cd_p={cd_p_wet:.6f} Cd_f={cd_f_wet:.6f} "
        f"Cd_tot={cd_tot_wet:.6f} (ref {cf_ref:.6f}) err={err_pct:+.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
