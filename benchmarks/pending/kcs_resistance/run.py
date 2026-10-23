#!/usr/bin/env python3
"""KCS resistance benchmark — parametric KRISO Container Ship hull, Re=1000.

Caliber (see README.md and REFERENCE_AUDIT.md for the full argument)
-------------------------------------------------------------------
The real KCS (KRISO Container Ship) experimental resistance lives at
Fr=0.26, Re~1.5e7 (model) / 2.8e9 (full scale).  That caliber is *not*
reachable by this solver:

  * the free surface / wave-making (the dominant residual-resistance term
    at Fr=0.26) is absent in a single-phase LBM, exactly the structural
    blocker that sank DTMB5415 (benchmarks/STATUS.md);
  * Re~1e7..1e9 needs turbulence modelling + tens of cells across the
    boundary layer, far beyond the reachable laminar LBM envelope;
  * the in-repo ``ship_cad`` KCS is a *parametric approximation*
    (half-beam = (B/2)(1-xi^2)^0.45 zeta^0.24, Cb~0.651), NOT the
    Tokyo-2015/ITTC KCS offset hull.

So this benchmark verifies the **friction-dominated** low-Re regime, in
the SAME caliber as the verified ``suboff_re1000`` benchmark (STATUS.md):

  - Re = u*L/nu = 1000 (laminar), L = KCS Lpp.
  - Reference = Blasius laminar flat-plate average skin friction
    Cf = 1.328/sqrt(Re) = 0.041995, evaluated at the run Reynolds number
    (ITTC-1957's Cf = 0.075/(log10(Re)-2)^2 is a *turbulent* line and is
    meaningless at Re=1000; it is documented, not used).
  - Normalisation = the hull's smooth wetted surface, dpS = 0.5*u_lb^2*S_lu.
  - No free surface, no wave-making -> the KCS hull is modelled as either a
    *half-body* (submerged form, keel->waterline, waterplane face solid) or a
    *double body* (mirror about the waterline; the standard zero-Froude
    free-surface limit) -- see --variant.

Force post-processing reuses the common modules
(drag_pressure.drag_pressure_integration / drag_friction_integration) with
the wetted-area reference, and the whole time-stepping chain is routed
through benchmarks/compile_route.route_step (verified-benchmark standard).

Usage:
  python run.py [--resolution 96] [--steps 20000] [--device sdaa:0]
                [--variant double|half] [--collision mrt]
                [--friction mix50] [--out DIR] [--compile-mode default]
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

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks


def _default_device() -> str:
    """First available accelerator on this host (SDAA, then CUDA, else CPU)."""
    try:
        import torch_sdaa  # noqa: F401

        if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
            return "sdaa:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


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
from tensorlbm.ship_cad import _kcs_half_beam  # noqa: E402

# ---------------------------------------------------------------------------
# KCS principal particulars (full scale, metres)
# Confirmed by two independent sources:
#   * MDPI J. Mar. Sci. Eng. 2020, 8, 745, Table 1 (Tokyo-2015 KCS database)
#   * github.com/B4kif/holtrop-mennen-kcs-resistance (Holtrop-Mennen inputs)
# ---------------------------------------------------------------------------
KCS_LPP_M = 230.0  # length between perpendiculars
KCS_BWL_M = 32.2  # waterline beam
KCS_T_M = 10.8  # design draft
KCS_CB = 0.6505
U_PHYS = 1.0e-3  # m/s (sets dt only), same convention as suboff_re1000
RE_TARGET = 1000.0

# Domain padding in units of Lpp: (x_up, x_down, y_side, y_side, z_below, z_above)
PADDING = (1.0, 3.0, 0.4, 0.4, 0.4, 0.4)


def _kcs_g(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Normalised half-beam g = (1-xi^2)^0.45 * zeta^0.24 (clipped)."""
    xi_c = np.clip(xi, -1.0, 1.0)
    zc = np.clip(zeta, 0.0, 1.0)
    return np.clip((1.0 - xi_c**2) ** 0.45, 0.0, 1.0) * np.clip(zc**0.24, 0.0, 1.0)


def smooth_side_area(L_lb: float, B_lb: float, T_lb: float) -> float:
    """Analytic area of ONE side of the below-waterline hull (lattice units^2).

    Surface r(xi,zeta) = (L/2 xi, (B/2) g, T zeta); |r_xi x r_zeta| dxi dzeta
    integrated with Gauss-Legendre quadrature (handles the integrable
    (1-xi^2)^-0.55 / zeta^-0.76 endpoint singularities of the derivatives).
    """
    from numpy.polynomial.legendre import leggauss

    xg, xw = leggauss(3000)
    zg, zw = leggauss(3000)
    xi = xg[None, :]
    zeta = zg[:, None]
    g = _kcs_g(xi * np.ones_like(zeta), zeta * np.ones_like(xi))
    # partials
    xi_c = np.clip(xi, -1.0, 1.0)
    denom_x = np.clip(1.0 - xi_c**2, 1e-12, None)
    g_xi = -0.9 * xi_c * denom_x ** (-0.55) * np.clip(zeta**0.24, 0.0, 1.0)
    zc = np.clip(zeta, 1e-12, 1.0)
    g_zeta = np.clip(denom_x**0.45, 0.0, 1.0) * 0.24 * zc ** (-0.76)
    a = B_lb / 2.0
    b = L_lb / 2.0
    jac = np.sqrt((a * g_xi * T_lb) ** 2 + (b * T_lb) ** 2 + (b * a * g_zeta) ** 2)
    # merge the two 1-D weight vectors
    w = (zw[:, None] * xw[None, :])
    return float((jac * w).sum())


def smooth_wetted_area(variant: str, L_lb: float, B_lb: float, T_lb: float) -> dict:
    """Smooth (grid-independent) wetted area of the modelled body, lu^2."""
    s_one = smooth_side_area(L_lb, B_lb, T_lb)
    if variant == "half":
        # two hull sides + the flat waterplane face (the waterplane is a solid
        # wall in the single-phase half-body model: no free surface)
        from numpy.polynomial.legendre import leggauss

        zg, zw = leggauss(3000)
        a_top = (B_lb * L_lb / 2.0) * float(
            (np.clip(1.0 - zg**2, 0.0, 1.0) ** 0.45 * zw).sum()
        )
        return {"S_side": 2.0 * s_one, "S_top": a_top, "S_total": 2.0 * s_one + a_top}
    # double body: below-WL hull + its mirror (two halves, two sides each)
    return {"S_side": 4.0 * s_one, "S_top": 0.0, "S_total": 4.0 * s_one}


def build_kcs_solid(
    variant: str, nx: int, ny: int, nz: int, device, L_lb: float, B_lb: float, T_lb: float
) -> torch.Tensor:
    """Build the KCS solid mask directly in lattice units (engine override)."""
    u = PADDING[0]
    b = PADDING[4]
    x0 = u * L_lb
    cx = x0 + L_lb / 2.0
    cy = ny / 2.0
    z_base = b * L_lb
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xi = ((xx - cx) / (L_lb / 2.0)).cpu().numpy()
    if variant == "half":
        zeta = ((zz - z_base) / T_lb).cpu().numpy()
        hb = _kcs_half_beam(xi, zeta)  # in_hull handled inside
        hb_t = torch.tensor(hb, device=device, dtype=torch.float32)
        return torch.abs(yy - cy) <= hb_t * (B_lb / 2.0)
    # double body: zeta_eff = 1 - |z - z_wl|/T, z_wl = z_base + T
    z_wl = z_base + T_lb
    zeta_eff = (1.0 - torch.abs(zz - z_wl) / T_lb).cpu().numpy()
    hb = _kcs_half_beam(xi, zeta_eff)
    hb_t = torch.tensor(hb, device=device, dtype=torch.float32)
    return torch.abs(yy - cy) <= hb_t * (B_lb / 2.0)


class KCSEngine(GeneralSimEngine):
    """GeneralSimEngine with the parametric KCS hull injected as geometry."""

    def __init__(self, config, variant: str, L_lb: float, B_lb: float, T_lb: float):
        super().__init__(config)
        self._variant = variant
        self._L_lb = L_lb
        self._B_lb = B_lb
        self._T_lb = T_lb

    def _geometry_bounding_box(self):
        # hull bbox in physical units: bow at x=0, centred on y, keel at z=0.
        # For the double body the height is 2*T.
        H = KCS_T_M if self._variant == "half" else 2.0 * KCS_T_M
        return (0.0, KCS_LPP_M, -KCS_BWL_M / 2.0, KCS_BWL_M / 2.0, 0.0, H)

    def _build_solid_mask(self, nx, ny, nz, device):
        return build_kcs_solid(
            self._variant, nx, ny, nz, device, self._L_lb, self._B_lb, self._T_lb
        )


def run_engine_routed(engine, compile_mode):
    """Drive the engine's time-stepping chain through compile_route (as suboff)."""
    sol = engine.config.solver
    out = engine.config.output

    if engine._auto_wall_treatment == WallTreatment.WALL_FUNCTION:
        print("[compile_route] KCS: wall-function path not compiled -> eager", flush=True)
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
    solid = engine.solid

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

    def _step(f):
        return lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs)

    step_fn = route_step(_step, compile_mode, name=f"kcs_resistance[L{sol.resolution}]")

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
    compile_status = compile_status_of(step_fn)
    info = {
        "status": "completed",
        "steps": engine.step_count,
        "force_samples": len(engine.forces_log),
        "diverged": not torch.isfinite(engine.f).all().item(),
        "compile_status": compile_status["compile_status"],
        "compile_mode_effective": compile_status["compile_mode_effective"],
        "compile_status_reason": compile_status["compile_status_reason"],
        "dpS_engine": dpS,
    }
    return info, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=96, help="cells per Lpp")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variant", default="double", choices=["half", "double"])
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument(
        "--friction",
        default="mix50",
        choices=["standard", "2nd_order", "central", "lagrange", "bfl", "faces", "mix50"],
    )
    ap.add_argument("--p0", default="near_wall", choices=["near_wall", "far_field", "domain_avg", "inlet"])
    ap.add_argument("--out", default=None)
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    if not args.device:
        args.device = _default_device()
    compile_mode = compile_mode_from_args(args)

    L = args.resolution
    variant = args.variant
    collision = (
        CollisionModel.SMAGORINSKY_MRT if args.collision == "smagorinsky" else CollisionModel.MRT
    )
    viscosity = U_PHYS * KCS_LPP_M / RE_TARGET  # Re = u*L/nu = 1000

    # lattice dimensions
    dx = KCS_LPP_M / L
    L_lb = float(L)
    B_lb = KCS_BWL_M / dx
    T_lb = KCS_T_M / dx

    area = smooth_wetted_area(variant, L_lb, B_lb, T_lb)

    out_dir = Path(
        args.out or str(_REPO_ROOT / f"results_bench_kcs_re1000_L{L}_{variant}_{args.collision}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"bench_kcs_re1000_L{L}_{variant}_{args.collision}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_HULL,
            hull_type="kcs",
            hull_length=KCS_LPP_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=U_PHYS,
            reference_length=KCS_LPP_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            domain_padding=PADDING,
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
            directory=str(out_dir), formats=[], save_macroscopic=False, save_forces=True
        ),
    )

    print(
        f"=== KCS resistance Re=1000 L={L} variant={variant} collision={args.collision} "
        f"steps={args.steps} device={args.device} ===",
        flush=True,
    )
    engine = KCSEngine(config, variant, L_lb, B_lb, T_lb)
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
    print(f"L_lb={L_lb} B_lb={B_lb:.3f} T_lb={T_lb:.3f} smooth_area={json.dumps(area)}", flush=True)

    run_info, elapsed = run_engine_routed(engine, compile_mode)
    print(
        f"run finished: {run_info['status']} in {elapsed:.0f}s "
        f"({elapsed / max(run_info['steps'], 1) * 1000:.1f} ms/step) "
        f"effective={run_info.get('compile_mode_effective')!r} "
        f"status={run_info.get('compile_status')!r}",
        flush=True,
    )

    # ---- post-process: rescale engine dpS -> smooth wetted area ----
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    S_smooth = area["S_total"]
    dpS_engine = run_info["dpS_engine"]
    dpS_wet = 0.5 * u_lb**2 * S_smooth
    rescale = dpS_engine / dpS_wet  # engine cd -> wetted-area cd

    log = engine.forces_log
    n_win = min(1000, len(log))
    win = log[-n_win:]
    cd_p = sum(e["cd_pressure"] for e in win) / n_win * rescale
    cd_f = sum(e["cd_friction"] for e in win) / n_win * rescale
    cd_tot = cd_p + cd_f

    # final-field recompute with all friction formulas / p0 methods
    final_checks = {}
    f_final = engine.f
    mesh = engine.mesh
    solid = engine.solid
    for p0 in ("near_wall", "far_field", "domain_avg", "inlet"):
        fx_p, _, _ = drag_pressure_integration(
            f_final, mesh, dpS_wet, extrap="none", p0_method=p0, solid=solid
        )
        row = {"cd_p": fx_p}
        for formula in ("standard", "faces", "mix50", "2nd_order", "central", "lagrange"):
            kwargs = {"solid": solid}
            fx_f, _, _ = drag_friction_integration(
                f_final, mesh, dpS_wet, nu_lb, formula=formula, **kwargs
            )
            row[f"cd_f_{formula}"] = fx_f
        row["cd_tot_standard"] = row["cd_p"] + row["cd_f_standard"]
        row["cd_tot_faces"] = row["cd_p"] + row["cd_f_faces"]
        row["cd_tot_mix50"] = row["cd_p"] + row["cd_f_mix50"]
        final_checks[p0] = row

    cf_ref = 1.328 / math.sqrt(RE_TARGET)
    err_tot = (cd_tot - cf_ref) / cf_ref * 100.0
    err_f_mix50 = (final_checks[args.p0]["cd_f_mix50"] - cf_ref) / cf_ref * 100.0

    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(log) * frac)
        seg = log[max(0, k - n_win) : k]
        if seg:
            conv[f"{int(frac * 100)}%"] = round(sum(e["cd_total"] for e in seg) / len(seg) * rescale, 6)

    result = {
        "case": f"KCS parametric hull ({variant}) Re=1000",
        "benchmark": "kcs_resistance",
        "device": args.device,
        "variant": variant,
        "grid": f"{setup_info['domain_lu'][0]}x{setup_info['domain_lu'][1]}x{setup_info['domain_lu'][2]}",
        "domain_lu": setup_info["domain_lu"],
        "n_cells": setup_info["total_cells"],
        "L_cells": L,
        "L_lb": L_lb,
        "B_lb": B_lb,
        "T_lb": T_lb,
        "KCS_particulars": {"Lpp_m": KCS_LPP_M, "B_m": KCS_BWL_M, "T_m": KCS_T_M, "Cb": KCS_CB},
        "Re": setup_info["Re"],
        "u_lb": u_lb,
        "nu_lb": nu_lb,
        "tau": setup_info["tau"],
        "collision": args.collision,
        "compile_mode_effective": run_info.get("compile_mode_effective"),
        "compile_status": run_info.get("compile_status"),
        "n_steps": run_info["steps"],
        "elapsed_s": round(elapsed, 1),
        "ms_per_step": round(elapsed / max(run_info["steps"], 1) * 1000, 2),
        "smooth_area_lu": area,
        "dpS_engine": dpS_engine,
        "dpS_wetted": dpS_wet,
        "rescale_engine_to_wetted": rescale,
        "friction_formula_primary": args.friction,
        "p0_method": args.p0,
        "Cd_pressure": cd_p,
        "Cd_friction": cd_f,
        "Cd_total": cd_tot,
        "Cf_ref_Blasius": cf_ref,
        "error_pct_total_vs_Blasius": err_tot,
        "error_pct_friction_mix50_vs_Blasius": err_f_mix50,
        "convergence_windows": conv,
        "final_field_checks": final_checks,
        "solid_cells": int(engine.solid.sum().item()),
        "near_wall_cells": int(engine.near.sum().item()),
        "finite": bool(torch.isfinite(engine.f).all().item()),
        "diverged": run_info.get("diverged", False),
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
                    "Cf_ref_Blasius",
                    "error_pct_total_vs_Blasius",
                    "convergence_windows",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"results written to {result_path}", flush=True)
    print(
        f"RESULT Cd_p={cd_p:.6f} Cd_f={cd_f:.6f} Cd_tot={cd_tot:.6f} "
        f"(ref {cf_ref:.6f}) err_tot={err_tot:+.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()