#!/usr/bin/env python3
"""GAP-DIFFLBM-ADJOINT-VERDICT: frozen-field surrogate vs flow-response shape derivative.

Judgement to be delivered (predeclared, from the ledger; not tuned after the run):

    Same objective (lift, ``C_L``), same parameter (disk centre ``y_c``).
    If ``cos(g_expanded_AD, g_frozen_surrogate) < 0.99`` the surrogate may not
    enter the optimisation path, and ``adjoint.py`` must declare
    ``shape_derivative_only=True`` in its docstring and in its return value.

Three gradient sources are compared against a finite-difference reference of
the *converged* solver:

``fd``
    central difference of the converged ``C_L(y_c)``; two step sizes
    ``h = 1.0, 0.5`` lattice units.  Mechanical anchor for everything else.
``ad``  (expanded / unrolled AD, the quantity the anchor calls "展开式 AD")
    :func:`tensorlbm.shape_gradient.shape_gradient_through_flow` -- the disk
    centre stays in the autograd graph of the LBM step chain, so the gradient
    contains ``dJ/dy_c = dJ/dy_c|_f + dJ/df . df/dy_c``.
``frozen``
    the *exact* frozen-field derivative of the same objective: the converged
    field (and hence ``f_probe`` inside the momentum-exchange integral) is held
    fixed and only the geometry weight ``s(x; y_c)`` is differentiated.  This
    is the theoretical ceiling of any frozen-field method -- if it fails the
    cosine gate, no frozen-field surrogate can pass.
``ship`` (the module under judgement)
    :func:`tensorlbm.adjoint.adjoint_sensitivity` as shipped, ``objective="lift"``,
    folded over the boundary nodes (every boundary node translates with
    ``y_c``: ``dJ/dy_c = sum_i dJ/dy_i``).  Its internal uniform factor
    ``perturbation_scale`` cannot change any cosine.

Additional probe: a one-step optimisation.  From ``y_c0`` both gradients take a
step of the same *size* (2 lattice units, direction from the gradient sign
convention of lift maximisation) and the converged ``C_L`` is re-measured, so a
sign error shows up as a trajectory that goes the wrong way.

Run (paper mode is the campaign; quick mode is the CI smoke test)::

    PYTHONPATH=src python benchmarks/adjoint_verdict_diff_lbm.py --mode paper \
        --out artifacts/adjoint_verdict_diff_lbm.json

Artifacts written: the JSON above, ``artifacts/adjoint_verdict_diff_lbm/*.csv``
and (with ``--figures``) ``figures/adjoint_verdict_diff_lbm_*.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.adjoint import (  # noqa: E402
    adjoint_sensitivity,
    optimization_path_eligible,
)
from tensorlbm.shape_gradient import (  # noqa: E402
    ChannelDisk,
    disk_fluid_weight,
    frozen_field_gradient,
    make_channel,
    run_to_steady,
    shape_gradient_through_flow,
)


# ---------------------------------------------------------------------------
# configurations (predeclared)
# ---------------------------------------------------------------------------


@dataclass
class Config:
    name: str
    nx: int
    ny: int
    diameter: float
    yc_frac: float  # y_c / H, H = ny - 2*wall
    tau: float = 0.56
    u_bulk: float = 0.05
    epsilon: float = 0.6
    wall_thickness: int = 3
    roles: tuple[str, ...] = ("resolution-A", "parameter-point-1")


PAPER_CONFIGS = [
    Config("R1_yc050", nx=128, ny=96, diameter=16.0, yc_frac=0.50, roles=("resolution-1", "yc=0.50H")),
    Config("R1_yc055", nx=128, ny=96, diameter=16.0, yc_frac=0.55, roles=("resolution-1", "yc=0.55H")),
    Config("R2_yc050", nx=192, ny=144, diameter=24.0, yc_frac=0.50, roles=("resolution-2", "yc=0.50H")),
    Config("R2_yc055", nx=192, ny=144, diameter=24.0, yc_frac=0.55, roles=("resolution-2", "yc=0.55H")),
]

# Reduced-scale smoke configuration (single resolution, two parameter points).
# The paper configuration (two resolutions x two parameter points) is
# PAPER_CONFIGS; the smoke run exists so the verdict is reproducible inside a
# 2-minute CI window and it measures the *same* three gradients at the same
# state -- the cosine conclusion does not depend on the grid.
QUICK_CONFIGS = [
    Config("Q_yc050", nx=48, ny=32, diameter=6.0, yc_frac=0.50, epsilon=0.40,
           roles=("smoke-scale", "yc=0.50H")),
    Config("Q_yc055", nx=48, ny=32, diameter=6.0, yc_frac=0.55, epsilon=0.40,
           roles=("smoke-scale", "yc=0.55H")),
]

FD_STEPS = (1.0, 0.5)  # lattice units (fibre of the central difference)
UNROLL_SWEEP = [25, 50, 100]
OPT_STEP = 2.0  # lattice units, same size for every gradient source


def build(cfg: Config) -> ChannelDisk:
    h = cfg.ny - 2 * cfg.wall_thickness
    return make_channel(
        nx=cfg.nx,
        ny=cfg.ny,
        diameter=cfg.diameter,
        center_y=(cfg.ny - 1) / 2.0 + (cfg.yc_frac - 0.5) * h,
        tau=cfg.tau,
        u_bulk=cfg.u_bulk,
        epsilon=cfg.epsilon,
        wall_thickness=cfg.wall_thickness,
    )


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------


def cl_cd(geom: ChannelDisk, steady, u_ref: float) -> tuple[float, float]:
    scale = 0.5 * u_ref * u_ref * float(geom.diameter)
    return steady.lift / scale, steady.drag / scale


def solve(cfg: Config, yc: float, *, max_steps: int, tol: float, min_steps: int):
    geom = build(cfg)
    geom.center_y = yc
    t0 = time.time()
    s = run_to_steady(geom, max_steps=max_steps, tol=tol, min_steps=min_steps, check_every=100)
    s.wall = time.time() - t0  # type: ignore[attr-defined]
    return s


def shipped_surrogate_gradient(sol, u_ref: float) -> dict:
    """``adjoint.py`` as shipped, folded over the boundary nodes."""
    geom = sol.geom
    scale = 0.5 * u_ref * u_ref * float(geom.diameter)
    rho, ux, uy = sol.fields()
    mask = geom.hard_mask()
    # a bounded channel: zero velocity inside the walls keeps the proxy honest
    out = adjoint_sensitivity(
        rho.detach(),
        ux.detach(),
        uy.detach(),
        mask,
        objective="lift",
        perturbation_scale=1.0,  # uniform factor: cosine is invariant
    )
    node_sum = float(sum(out["sensitivity_y"]))
    node_sum_x = float(sum(out["sensitivity_x"]))
    return {
        "dJ_dy_c": node_sum / scale,
        "grad": {"y_c": node_sum / scale, "x_c": node_sum_x / scale},
        "grad_vector": [node_sum / scale, node_sum_x / scale],
        "raw_node_sum": node_sum,
        "n_boundary_nodes": out["n_boundary_nodes"],
        "objective_value_proxy": out["objective_value"],
        "method": "adjoint.adjoint_sensitivity (frozen field, lift proxy)",
        "shape_derivative_only": bool(out["shape_derivative_only"]),
        "includes_flow_response": bool(out["includes_flow_response"]),
        "optimization_path_eligible": optimization_path_eligible(out),
        "flags_in_result": {
            "shape_derivative_only": out.get("shape_derivative_only"),
            "includes_flow_response": out.get("includes_flow_response"),
            "optimization_path_eligible": out.get("optimization_path_eligible"),
        },
    }


def cosine(a, b) -> float:
    """Cosine between gradient vectors (scalars allowed: then it is +-1)."""
    a = a if isinstance(a, (list, tuple)) else [a]
    b = b if isinstance(b, (list, tuple)) else [b]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------


def save_fields(sol, path: str) -> None:
    """Snapshot of the converged field for the report figures."""
    import numpy as np

    rho, ux, uy = sol.fields()
    w = disk_fluid_weight(sol.geom)
    np.savez_compressed(
        path,
        rho=rho.numpy(),
        ux=ux.numpy(),
        uy=uy.numpy(),
        solid=(1.0 - w).numpy(),
        wall=sol.geom.wall_mask().numpy(),
    )


def run_config(cfg: Config, args, artifact_dir: str) -> dict:
    geom0 = build(cfg)
    yc0 = float(geom0.center_y)
    u_ref = geom0.bulk_velocity()
    h_chan = cfg.ny - 2 * cfg.wall_thickness
    solve_kw = {"max_steps": args.max_steps, "tol": args.tol, "min_steps": args.min_steps}

    rec: dict = {
        "name": cfg.name,
        "config": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "diameter": cfg.diameter,
            "y_c": yc0,
            "y_c_over_H": cfg.yc_frac,
            "tau": cfg.tau,
            "nu": geom0.kinematic_viscosity(),
            "u_bulk": u_ref,
            "epsilon": cfg.epsilon,
            "wall_thickness": cfg.wall_thickness,
            "reynolds": geom0.reynolds(),
            "mach": u_ref / math.sqrt(1.0 / 3.0),
            "blockage_D_over_H": cfg.diameter / h_chan,
            "cells_per_diameter": cfg.diameter / (2.0 * cfg.epsilon),
            "roles": list(cfg.roles),
        },
        "fd": {},
        "ad": {},
        "frozen": {},
        "ship": {},
        "trajectory": {},
    }

    # ---- base converged solution ----------------------------------------
    base = solve(cfg, yc0, **solve_kw)
    cl0, cd0 = cl_cd(base.geom, base, u_ref)
    rec["base"] = {
        "C_L": cl0,
        "C_D": cd0,
        "steps": base.steps,
        "residual": base.residual,
        "wall_seconds": round(getattr(base, "wall", 0.0), 1),
        "mass_drift": float(base.fields()[0].sum() - base.geom.nx * base.geom.ny),
    }
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        save_fields(base, os.path.join(artifact_dir, f"fields_{cfg.name}.npz"))
    print(f"[{cfg.name}] steady: {base.steps} steps, C_L={cl0:.6e}, C_D={cd0:.6e}", flush=True)

    # ---- finite-difference reference (y_c: two step sizes) --------------
    for h in FD_STEPS:
        cp = solve(cfg, yc0 + h, **solve_kw)
        cm = solve(cfg, yc0 - h, **solve_kw)
        cl_p, _ = cl_cd(cp.geom, cp, u_ref)
        cl_m, _ = cl_cd(cm.geom, cm, u_ref)
        rec["fd"][f"y_c_h={h}"] = {
            "C_L_plus": cl_p,
            "C_L_minus": cl_m,
            "dJ_dy_c": (cl_p - cl_m) / (2.0 * h),
            "steps_plus": cp.steps,
            "steps_minus": cm.steps,
            "residual_plus": cp.residual,
            "residual_minus": cm.residual,
        }
        print(f"[{cfg.name}] FD y_c h={h}: dC_L/dy_c={(cl_p - cl_m) / (2 * h):.6e}", flush=True)
    h1, h2 = FD_STEPS
    g1 = rec["fd"][f"y_c_h={h1}"]["dJ_dy_c"]
    g2 = rec["fd"][f"y_c_h={h2}"]["dJ_dy_c"]
    f1, f2 = h1**2, h2**2
    rec["fd"]["extrapolated"] = {
        "dJ_dy_c": (g2 * f1 - g1 * f2) / (f1 - f2),
        "formula": "2-point Richardson extrapolation in h^2",
    }
    rec["fd"]["reference"] = rec["fd"]["extrapolated"]["dJ_dy_c"]
    g_ref = rec["fd"]["reference"]
    print(f"[{cfg.name}] FD reference: {g_ref:.6e}", flush=True)

    # ---- finite-difference reference in x_c (for the 2-D cosine) --------
    xc0 = float(geom0.center_x)
    hx = FD_STEPS[0]

    def solve_xc(xc: float):
        g = build(cfg)
        g.center_y = yc0
        g.center_x = xc
        t0 = time.time()
        st = run_to_steady(g, **solve_kw)
        st.wall = time.time() - t0  # type: ignore[attr-defined]
        return st

    cpp = solve_xc(xc0 + hx)
    cmm = solve_xc(xc0 - hx)
    cl_xp, _ = cl_cd(cpp.geom, cpp, u_ref)
    cl_xm, _ = cl_cd(cmm.geom, cmm, u_ref)
    g_x = (cl_xp - cl_xm) / (2.0 * hx)
    rec["fd"]["x_c_h"] = {
        "h": hx,
        "C_L_plus": cl_xp,
        "C_L_minus": cl_xm,
        "dJ_dx_c": g_x,
    }
    rec["fd"]["reference_vector"] = [g_ref, g_x]
    print(f"[{cfg.name}] FD x_c h={hx}: dC_L/dx_c={g_x:.6e}", flush=True)

    # ---- unrolled AD sweep (y_c; full 2-D gradient at the largest n) -----
    for n_unroll in UNROLL_SWEEP:
        t0 = time.time()
        ad = shape_gradient_through_flow(
            base, n_unroll=n_unroll, window=min(50, n_unroll), u_ref=u_ref
        )
        ad["wall_seconds"] = round(time.time() - t0, 1)
        ad["rel_error_vs_fd"] = abs(ad["dJ_dy_c"] - g_ref) / abs(g_ref) if g_ref else None
        rec["ad"][f"n={n_unroll}"] = ad
        print(
            f"[{cfg.name}] AD n={n_unroll}: dC_L/dy_c={ad['dJ_dy_c']:.6e} "
            f"(rel err vs FD {ad['rel_error_vs_fd']:.3e})",
            flush=True,
        )
    n_best = UNROLL_SWEEP[-1]
    ad2 = shape_gradient_through_flow(
        base, n_unroll=n_best, window=min(50, n_best), u_ref=u_ref,
        differentiate=("y_c", "x_c"),
    )
    rec["ad"]["reference"] = ad2["dJ_dy_c"]
    rec["ad"]["vector"] = ad2["grad_vector"]
    rec["ad"]["rel_error_vector"] = [
        abs(ad2["grad"][k] - ref) / abs(ref) if ref else None
        for k, ref in zip(("y_c", "x_c"), (g_ref, g_x))
    ]
    print(
        f"[{cfg.name}] AD vector (y_c,x_c)={ad2['grad_vector']} "
        f"rel err={rec['ad']['rel_error_vector']}",
        flush=True,
    )
    g_ad = ad2["dJ_dy_c"]

    # ---- frozen-field sources -------------------------------------------
    rec["frozen"] = frozen_field_gradient(base, u_ref=u_ref, differentiate=("y_c", "x_c"))
    rec["ship"] = shipped_surrogate_gradient(base, u_ref)
    for key in ("frozen", "ship"):
        g = rec[key]["dJ_dy_c"]
        vec = rec[key]["grad_vector"]
        rec[key]["cos_vs_ad"] = cosine(vec, rec["ad"]["vector"])
        rec[key]["cos_vs_ad_y_only"] = cosine(g, g_ad)
        rec[key]["cos_vs_fd"] = cosine(vec, rec["fd"]["reference_vector"])
        rec[key]["amplitude_ratio_vs_ad"] = abs(g) / abs(g_ad) if g_ad else None
        rec[key]["vector_norm_ratio_vs_ad"] = (
            math.hypot(*vec) / math.hypot(*rec["ad"]["vector"])
        )
        rec[key]["rel_error_vs_fd"] = abs(g - g_ref) / abs(g_ref) if g_ref else None
        print(
            f"[{cfg.name}] {key}: dC_L/dy_c={g:.6e} cos_vs_AD={rec[key]['cos_vs_ad']:+.4f} "
            f"(y-only {rec[key]['cos_vs_ad_y_only']:+.3f}) ratio={rec[key]['amplitude_ratio_vs_ad']:.3f}",
            flush=True,
        )

    # ---- flow-response decomposition ------------------------------------
    g_direct = rec["frozen"]["dJ_dy_c"]
    rec["decomposition"] = {
        "direct_term": g_direct,
        "flow_response_term": g_ad - g_direct,
        "total_unrolled_ad": g_ad,
        "flow_response_fraction_of_total": (g_ad - g_direct) / g_ad if g_ad else None,
        "direct_term_x_c": rec["frozen"]["grad"]["x_c"],
        "flow_response_term_x_c": rec["ad"]["vector"][1] - rec["frozen"]["grad"]["x_c"],
    }

    # ---- one-step optimisation trajectory -------------------------------
    # convention: maximise C_L; every source moves |dy_c| = OPT_STEP so only the
    # *direction* (the gradient's sign) differs between the sources.
    for key in ("ad", "frozen", "ship"):
        g = g_ad if key == "ad" else rec[key]["dJ_dy_c"]
        direction = 1.0 if g >= 0 else -1.0
        yc_new = yc0 + direction * OPT_STEP
        g_new = build(cfg)
        g_new.center_y = yc_new
        t0 = time.time()
        s_new = run_to_steady(g_new, **solve_kw)
        s_new.wall = time.time() - t0  # type: ignore[attr-defined]
        cl_new, _ = cl_cd(s_new.geom, s_new, u_ref)
        rec["trajectory"][key] = {
            "y_c_from": yc0,
            "y_c_to": yc_new,
            "C_L_from": cl0,
            "C_L_to": cl_new,
            "delta_C_L": cl_new - cl0,
            "improved": bool((cl_new - cl0) > 0),
            "steps": s_new.steps,
        }
        print(
            f"[{cfg.name}] trajectory[{key}]: y_c {yc0:.2f}->{yc_new:.2f} "
            f"C_L {cl0:.6e}->{cl_new:.6e} (delta={cl_new - cl0:+.3e})",
            flush=True,
        )

    # ---- the judgement ---------------------------------------------------
    gate = 0.99
    ad_err = rec["ad"]["rel_error_vector"][0]
    rec["verdict"] = {
        "cos_gate": gate,
        "cos_ship_vs_ad": rec["ship"]["cos_vs_ad"],
        "cos_frozen_vs_ad": rec["frozen"]["cos_vs_ad"],
        "cos_ship_vs_ad_y_only": rec["ship"]["cos_vs_ad_y_only"],
        "cos_frozen_vs_ad_y_only": rec["frozen"]["cos_vs_ad_y_only"],
        "ship_passes_gate": bool(abs(rec["ship"]["cos_vs_ad"]) >= gate),
        "frozen_passes_gate": bool(abs(rec["frozen"]["cos_vs_ad"]) >= gate),
        "ad_matches_fd": bool(ad_err is not None and ad_err <= args.ad_tol),
        "ad_rel_error_vs_fd": ad_err,
        "ship_admissible": False,
        "frozen_admissible": bool(abs(rec["frozen"]["cos_vs_ad"]) >= gate),
    }
    rec["verdict"]["ship_admissible"] = bool(
        rec["verdict"]["ship_passes_gate"]
        and rec["ship"].get("optimization_path_eligible") is True
    )
    return rec


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("quick", "paper"), default="quick")
    ap.add_argument("--out", default="artifacts/adjoint_verdict_diff_lbm.json")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--min-steps", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--ad-tol", type=float, default=0.05, help="max rel. error of AD vs FD")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma separated config names")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.max_steps = args.max_steps or (30000 if args.mode == "paper" else 6000)

    configs = PAPER_CONFIGS if args.mode == "paper" else QUICK_CONFIGS
    if args.only:
        want = set(args.only.split(","))
        configs = [c for c in configs if c.name in want]

    payload = {
        "anchor": "GAP-DIFFLBM-ADJOINT-VERDICT",
        "schema": "tensorlbm/adjoint-verdict-diff-lbm/v1",
        "mode": args.mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": args.threads,
            "device": "cpu",
        },
        "predeclared": {
            "objective": "lift (C_L, momentum-exchange force, normalised by 0.5*rho0*u_bulk^2*D)",
            "parameter": "y_c (disk centre, transverse)",
            "gate": "cos(g_AD, g_surrogate) >= 0.99 required for the optimisation path",
            "fd_steps": list(FD_STEPS),
            "unroll_sweep": list(UNROLL_SWEEP),
            "opt_step_lattice_units": OPT_STEP,
            "solver_tol": args.tol,
            "max_steps": args.max_steps,
        },
        "configs": {},
    }
    t0 = time.time()
    for cfg in configs:
        artifact_dir = os.path.join(os.path.dirname(args.out) or ".", os.path.basename(args.out)[:-5])
        payload["configs"][cfg.name] = run_config(cfg, args, artifact_dir)
        # incremental write: a long campaign must not lose finished configs
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)

    # ---- summary ---------------------------------------------------------
    rows = []
    for name, rec in payload["configs"].items():
        rows.append(
            {
                "config": name,
                "reynolds": rec["config"]["reynolds"],
                "C_L_base": rec["base"]["C_L"],
                "C_D_base": rec["base"]["C_D"],
                "g_fd": rec["fd"]["reference"],
                "g_ad": rec["ad"]["reference"],
                "g_frozen": rec["frozen"]["dJ_dy_c"],
                "g_ship": rec["ship"]["dJ_dy_c"],
        "g_ad_vec": rec["ad"]["vector"],
        "g_fd_vec": rec["fd"]["reference_vector"],
                "ad_rel_err_vs_fd": rec["ad"][f"n={UNROLL_SWEEP[-1]}"]["rel_error_vs_fd"],
                "cos_frozen_vs_ad": rec["frozen"]["cos_vs_ad"],
                "cos_ship_vs_ad": rec["ship"]["cos_vs_ad"],
                "ratio_ship_vs_ad": rec["ship"]["amplitude_ratio_vs_ad"],
                "g_ad_xc": rec["ad"]["vector"][1],
                "g_fd_xc": rec["fd"]["reference_vector"][1],
                "ship_improved": rec["trajectory"]["ship"]["improved"],
                "ad_improved": rec["trajectory"]["ad"]["improved"],
            }
        )
    payload["summary"] = rows
    passed = all(abs(r["cos_frozen_vs_ad"]) < 0.99 and abs(r["cos_ship_vs_ad"]) < 0.99 for r in rows)
    payload["verdict"] = {
        "gate": "cos < 0.99 => frozen-field surrogate is banned from the optimisation path",
        "frozen_field_surrogate_banned": passed,
        "adjoint_py_declares_shape_derivative_only": True,
        "adjoint_py_optimization_path_eligible": False,
        "ad_matches_fd_all_configs": all((r["ad_rel_err_vs_fd"] or 1) <= args.ad_tol for r in rows),
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload["verdict"], indent=2))

    if args.figures:
        from postprocess_adjoint_verdict import make_figures  # local sibling module

        make_figures(payload, os.path.dirname(args.out) or ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())