#!/usr/bin/env python3
"""SUBOFF + sail (AFF-3) / full-appendage (AFF-8) drag, Re=1000 — benchmark runner.

Extends the verified ``suboff_re1000`` bare-hull chain (geometry + mix50
friction + compile routing + wetted-area drag normalization) to the
**with-sail** (AFF-3) and **full** (AFF-8) DARPA SUBOFF configurations.

What is reused verbatim from ``benchmarks/verified/suboff_re1000/run.py``:
  - ``GeneralSimEngine`` setup (same PARAMETRIC_SUBOFF geometry source, same
    domain padding (1,4,1,1,1,1), same units/Re=1000 Laufer state),
  - the compile-routed drive loop (``run_engine_routed`` -> ``route_step`` ->
    ``tensorlbm.compile_utils.compile_step``),
  - ``drag_pressure_integration`` (p0='near_wall', extrap='none') and
    ``drag_friction_integration`` (primary ``mix50``),
  - the wetted-area normalization convention.

What is new (the appendage extension):
  - the solid mask is built with
    ``suboff_cad.build_suboff_mask(hull_type=...)`` instead of the engine's
    hard-coded ``bare_hull`` (subclass ``SuboffAppendageEngine``);
  - surface normals use a **hybrid** mesh: the analytic body-of-revolution
    normals (``SurfaceMesh.from_suboff``) on the hull, and generic
    gradient normals only on the appendage near cells.  The pure
    body-of-revolution normal would give the sail purely radial (y,z)
    normals and therefore silently delete the sail's streamwise pressure
    force;
  - the reference area extends the verified ``pi*D*L`` cylinder proxy by the
    analytic appendage *own* wetted-area proxy from
    ``suboff_cad.suboff_statistics`` (``appendage_own_wetted_area_lu2``).

Reference (multi-source cross-check) — see README:
  primary : Blasius laminar flat-plate Cf = 1.328/sqrt(Re) at Re=1000 on the
            total wetted area (same caliber as the verified bare-hull case).
  cross   : ITTC-1957 Cf = 0.075/(log10 Re - 2)^2 and the DARPA AFF-8
            tow-tank Ct=0.004 are MODEL/FULL-scale TURBULENT references
            (Re ~ 2e6-1.2e7); they are recorded but explicitly NOT applicable
            at Re=1000 (laminar) — see ``reference_cross_check`` in result.json.

Usage:
  python run.py --hull-type with_sail --resolution 80 --steps 12000 \
                --device sdaa:2 --collision mrt --friction mix50 \
                --compile-mode default --out results_suboff_sail_L80
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
    """Pick the first available accelerator on this host (SDAA first)."""
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
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    compile_status_of,
    route_step,
)

from tensorlbm.boundaries3d import far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    suboff_smooth_q,
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

SUBOFF_LENGTH_M = 4.356  # DARPA SUBOFF bare hull length [m]
SUBOFF_RADIUS_M = 0.254  # max radius [m]  (L/D = 8.57)
U_PHYS = 1.0e-3  # m/s (any value; sets dt only)


# --------------------------------------------------------------------------- #
# Appendage-aware engine (override only geometry + surface normals)
# --------------------------------------------------------------------------- #
class SuboffAppendageEngine(GeneralSimEngine):
    """``GeneralSimEngine`` whose PARAMETRIC_SUBOFF path builds an appendage hull.

    Overrides exactly two methods of the common-module entry point; every
    other step of the setup (domain size, near-wall mask, equilibrium IC) and
    the whole run/post-processing chain is the shared code path.
    """

    def __init__(self, config: GeneralSimConfig, hull_type: str = "with_sail"):
        super().__init__(config)
        self._hull_type = hull_type
        self._appendage_stats: dict = {}
        self._appendage_near_cells = 0

    def _build_suboff_solid(self, nx: int, ny: int, nz: int, device) -> torch.Tensor:
        from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

        geo = self.config.geometry
        dx = self.config.physics.reference_length / self.config.solver.resolution
        length_lb = geo.suboff_length / dx
        radius_lb = geo.suboff_radius
        if radius_lb is not None:
            radius_lb = radius_lb / dx
        solid, stats = build_suboff_mask(
            hull_type=self._hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            cx=nx * 0.25,
            cy=ny * 0.5,
            cz=nz * 0.5,
            length=length_lb,
            radius=radius_lb,
            config=SuboffConfig(),
            device=str(device),
        )
        self._appendage_stats = stats
        return solid

    def _build_surface_mesh(self, device):
        """Hybrid normals: analytic hull normals + gradient normals on appendages."""
        geo = self.config.geometry
        sol = self.config.solver
        dx = self.config.physics.reference_length / sol.resolution
        length_lb = geo.suboff_length / dx
        radius_lb = (geo.suboff_radius or geo.suboff_length * (1.0 / (2.0 * 8.57))) / dx
        cx_lb = self.nx * 0.25
        cy_lb = self.ny * 0.5
        cz_lb = self.nz * 0.5

        base = SurfaceMesh.from_suboff(
            self.solid, self.near, cx_lb, cy_lb, cz_lb, length_lb, radius_lb
        )
        if self._hull_type == "bare_hull":
            return base

        # Appendage solid = solid cells radially outside the local hull of
        # revolution.  Their adjacent fluid near-cells get gradient normals so
        # the sail/fin streamwise pressure faces are integrated (the analytic
        # body-of-revolution normal would be purely radial there).
        from tensorlbm.suboff_cad import suboff_radius_profile

        nz, ny, nx = self.solid.shape
        dev = self.solid.device
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=dev, dtype=torch.float32),
            torch.arange(ny, device=dev, dtype=torch.float32),
            torch.arange(nx, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        xi = ((xx - (cx_lb - length_lb / 2.0)) / length_lb).clamp(0.0, 1.0)
        r_loc = torch.from_numpy(suboff_radius_profile(xi.detach().cpu().numpy())).to(
            dev
        ).float() * radius_lb
        r = torch.sqrt((yy - cy_lb) ** 2 + (zz - cz_lb) ** 2)
        app_solid = self.solid & (r > (r_loc + 0.5))

        app_near = torch.zeros_like(self.near)
        for dim, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            app_near |= torch.roll(app_solid, shift, dim)
        app_near = app_near & self.near
        self._appendage_near_cells = int(app_near.sum().item())

        grad = SurfaceMesh.from_gradient(self.solid, self.near)
        nx_n = torch.where(app_near, grad.nx_n, base.nx_n)
        ny_n = torch.where(app_near, grad.ny_n, base.ny_n)
        nz_n = torch.where(app_near, grad.nz_n, base.nz_n)
        # Renormalise (both branches are already unit-ish; keep dA=1 default).
        return SurfaceMesh(self.near, nx_n, ny_n, nz_n)


# --------------------------------------------------------------------------- #
# Reference-area conventions
# --------------------------------------------------------------------------- #
def wetted_dpS(u_lb: float, radius_lb: float, length_lb: float) -> float:
    """dpS with the verified family wetted-area reference S = pi*D*L."""
    return 0.5 * u_lb**2 * math.pi * (2.0 * radius_lb) * length_lb


def frontal_dpS(u_lb: float, radius_lb: float) -> float:
    """dpS with frontal-area reference S = pi*R^2 (engine default)."""
    return 0.5 * u_lb**2 * math.pi * radius_lb**2


def ittc1957_cf(re: float) -> float:
    """ITTC 1957 model-ship correlation line (TURBULENT)."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


# --------------------------------------------------------------------------- #
# Compile-routed driver (identical to verified suboff_re1000)
# --------------------------------------------------------------------------- #
def run_engine_routed(engine: GeneralSimEngine, compile_mode: str | None) -> tuple[dict, float]:
    sol = engine.config.solver
    out = engine.config.output

    if engine._auto_wall_treatment == WallTreatment.WALL_FUNCTION:
        print(
            "[compile_route] suboff_appendage: wall-function path not in compiled "
            "chain -> eager engine.run()",
            flush=True,
        )
        t0 = time.time()
        info = engine.run()
        info.setdefault("compile_status", "eager")
        info.setdefault("compile_mode_effective", "eager")
        info.setdefault("compile_status_reason", "wall-function path: eager engine.run()")
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

    name = f"suboff_{engine._hull_type}[L{sol.resolution}]"
    step_fn = route_step(_step, compile_mode, name=name)

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
        if out.save_macroscopic and step % sol.snapshot_interval == 0:
            engine._save_snapshot()
        if step % div_check == 0 and not torch.isfinite(engine.f).all():
            break
    elapsed = time.time() - t0

    compile_status = compile_status_of(step_fn)
    info = {
        "status": "completed",
        "steps": engine.step_count,
        "snapshots": len(engine.snapshots),
        "force_samples": len(engine.forces_log),
        "diverged": not torch.isfinite(engine.f).all().item(),
        "compile_status": compile_status["compile_status"],
        "compile_mode_effective": compile_status["compile_mode_effective"],
        "compile_status_reason": compile_status["compile_status_reason"],
    }
    return info, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hull-type", default="with_sail", choices=["bare_hull", "with_sail", "full"])
    ap.add_argument("--resolution", type=int, default=80, help="cells per hull length L")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--device", default=None, help="torch device, e.g. sdaa:0 (auto if unset)")
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument(
        "--friction",
        default="mix50",
        choices=["standard", "2nd_order", "central", "lagrange", "bfl", "bfl_smooth",
                 "bfl_lagrange", "faces", "mix50"],
    )
    ap.add_argument("--p0", default="near_wall", choices=["near_wall", "far_field", "domain_avg", "inlet"])
    ap.add_argument("--save-field", action="store_true", help="persist final field (~1.8 GB)")
    ap.add_argument("--out", default=None)
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    if not args.device:
        args.device = _default_device()
    compile_mode = compile_mode_from_args(args)

    L = args.resolution
    hull_type = args.hull_type
    collision = (
        CollisionModel.SMAGORINSKY_MRT if args.collision == "smagorinsky" else CollisionModel.MRT
    )
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0  # Re = u*L/nu = 1000

    out_dir = Path(
        args.out
        or str(_REPO_ROOT / f"results_e2e_suboff_{hull_type}_L{L}_{args.collision}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"bench_suboff_{hull_type}_re1000_L{L}_{args.collision}",
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
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
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
        f"=== SUBOFF {hull_type} Re=1000 L={L} collision={args.collision} "
        f"steps={args.steps} device={args.device} ===\n",
        flush=True,
    )
    engine = SuboffAppendageEngine(config, hull_type=hull_type)
    setup_info = engine.setup()
    print(
        "setup:",
        json.dumps(
            {
                k: setup_info[k]
                for k in (
                    "Re", "tau", "u_lb", "nu_lb", "domain_lu", "obstacle_cells",
                    "near_wall_cells", "total_cells", "device", "auto_collision",
                    "auto_wall_treatment",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"appendage stats: {json.dumps(engine._appendage_stats)}", flush=True)
    print(f"appendage near cells (gradient normals): {engine._appendage_near_cells}", flush=True)

    run_info, elapsed = run_engine_routed(engine, compile_mode)
    print(
        f"run finished: {run_info['status']} in {elapsed:.0f}s "
        f"({elapsed / max(run_info['steps'], 1) * 1000:.1f} ms/step) "
        f"compile_mode={compile_mode!r} effective={run_info.get('compile_mode_effective')!r} "
        f"status={run_info.get('compile_status')!r}",
        flush=True,
    )

    # ---- post-process: wetted-area coefficients ----
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    dx = SUBOFF_LENGTH_M / L
    R_lb = SUBOFF_RADIUS_M / dx
    dpS_bare = wetted_dpS(u_lb, R_lb, float(L))  # pi*D*L (verified family)

    from tensorlbm.suboff_cad import SuboffConfig, suboff_statistics

    st = suboff_statistics(hull_type, float(L), R_lb, SuboffConfig())
    a_app_lb2 = float(st.get("appendage_own_wetted_area_lu2") or 0.0)
    S_total_lb2 = math.pi * (2.0 * R_lb) * float(L) + a_app_lb2
    dpS_wet = 0.5 * u_lb**2 * S_total_lb2  # total wetted area (bare cylinder proxy + appendage)
    dpS_front = frontal_dpS(u_lb, R_lb)
    rescale = dpS_front / dpS_wet  # engine frontal -> total-wetted factor

    log = engine.forces_log
    n_win = min(1000, len(log))
    win = log[-n_win:]
    cd_p_wet = sum(e["cd_pressure"] for e in win) / n_win * rescale
    cd_f_wet = sum(e["cd_friction"] for e in win) / n_win * rescale
    cd_tot_wet = cd_p_wet + cd_f_wet
    n_win2 = min(500, len(log))
    win2 = log[-n_win2:]
    cd_tot_wet2 = (sum(e["cd_total"] for e in win2) / n_win2) * rescale

    # final-field recomputation (several friction formulas / p0 methods)
    final_checks = {}
    f_final = engine.f
    mesh = engine.mesh
    solid = engine.solid

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
        for formula in ("standard", "2nd_order", "central", "lagrange", "bfl_smooth",
                        "faces", "mix50"):
            kwargs = {"solid": solid}
            if formula == "bfl_smooth":
                kwargs["q_wall"] = q_smooth
            fx_f, _, _ = drag_friction_integration(
                f_final, mesh, dpS_wet, nu_lb, formula=formula, **kwargs
            )
            row[f"cd_f_{formula}"] = fx_f
        row["cd_tot_standard"] = row["cd_p"] + row["cd_f_standard"]
        row["cd_tot_mix50"] = row["cd_p"] + row["cd_f_mix50"]
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

    if args.save_field:
        field_path = out_dir / "final_field.pt"
        torch.save(f_final.cpu(), field_path)
        print(f"final field saved to {field_path}", flush=True)

    # ---- references -------------------------------------------------------
    cf_ref = 1.328 / math.sqrt(1000.0)  # Blasius laminar flat plate
    # Cross-check references (NOT applicable at Re=1000 — recorded for caliber audit)
    ittc_1e7 = ittc1957_cf(1.0e7)
    ittc_2e6 = ittc1957_cf(2.0e6)
    err_pct = (cd_tot_wet - cf_ref) / cf_ref * 100.0

    # drag increment vs the verified bare hull (same pi*D*L frame)
    bare_verified_cd_tot = 0.04151330249437245  # verified suboff_re1000 L=80
    cd_tot_bare_frame = cd_tot_wet * (dpS_wet / dpS_bare)  # normalize by pi*D*L only
    increment_ratio = cd_tot_bare_frame / bare_verified_cd_tot

    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(log) * frac)
        seg = log[max(0, k - n_win) : k]
        if seg:
            conv[f"{int(frac * 100)}%"] = round(
                sum(e["cd_total"] for e in seg) / len(seg) * rescale, 6
            )

    result = {
        "case": f"SUBOFF {hull_type} Re=1000",
        "benchmark": "suboff_sail_resistance",
        "hull_type": hull_type,
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
        "compile_mode_effective": run_info.get("compile_mode_effective"),
        "compile_status": run_info.get("compile_status"),
        "compile_status_reason": run_info.get("compile_status_reason"),
        "n_steps": run_info["steps"],
        "elapsed_s": round(elapsed, 1),
        "ms_per_step": round(elapsed / max(run_info["steps"], 1) * 1000, 2),
        "dpS_type": "wetted_area_0.5*u^2*(pi*D*L + A_appendage)",
        "S_bare_piDL_lb2": math.pi * (2.0 * R_lb) * float(L),
        "S_appendage_own_lb2": a_app_lb2,
        "S_total_wetted_lb2": S_total_lb2,
        "appendage_stats": engine._appendage_stats,
        "appendage_near_cells_gradient_normals": engine._appendage_near_cells,
        "dpS_wetted": dpS_wet,
        "dpS_engine_frontal": dpS_front,
        "rescale_frontal_to_totalwetted": rescale,
        "pressure_extrap": "none",
        "p0_method": args.p0,
        "friction_formula_primary": args.friction,
        "Cd_pressure": cd_p_wet,
        "Cd_friction": cd_f_wet,
        "Cd_total": cd_tot_wet,
        "Cd_total_last500": cd_tot_wet2,
        "Cd_total_bare_frame_piDL": cd_tot_bare_frame,
        "drag_increment_ratio_vs_verified_bare": increment_ratio,
        "Cf_ref_blasius": cf_ref,
        "reference_cross_check": {
            "ittc1957_Cf_Re1e7": ittc_1e7,
            "ittc1957_Cf_Re2e6": ittc_2e6,
            "darpa_aff8_experimental_Ct_Re2e6": 0.0040,
            "note": (
                "ITTC-1957 and the DARPA AFF-8 tow-tank Ct=0.004 are turbulent "
                "model/full-scale (Re~2e6-1.2e7) references. At Re=1000 the flow is "
                "laminar, so the primary reference is Blasius Cf=1.328/sqrt(Re)=0.041995 "
                "on the total wetted area — the same caliber as the verified bare-hull case."
            ),
        },
        "error_pct_vs_Blasius_totalwetted": err_pct,
        "window_samples": n_win,
        "window_steps": n_win * 10,
        "convergence_windows": conv,
        "final_field_checks": final_checks,
        "q_smooth_stats": q_stats,
        "final_field_saved": bool(args.save_field),
        "finite": bool(torch.isfinite(engine.f).all().item()),
        "diverged": run_info.get("diverged", False),
        "solid_cells": int(engine.solid.sum().item()) if engine.solid is not None else None,
        "near_wall_cells": int(engine.near.sum().item()) if engine.near is not None else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = out_dir / f"result_L{L}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "Cd_pressure", "Cd_friction", "Cd_total", "Cd_total_last500",
                    "Cf_ref_blasius", "error_pct_vs_Blasius_totalwetted",
                    "drag_increment_ratio_vs_verified_bare", "convergence_windows",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"results written to {result_path}", flush=True)
    print(
        f"RESULT {hull_type} L={L} Cd_p={cd_p_wet:.6f} Cd_f={cd_f_wet:.6f} "
        f"Cd_tot={cd_tot_wet:.6f} (ref {cf_ref:.6f}) err={err_pct:+.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()