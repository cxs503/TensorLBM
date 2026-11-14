"""GAP-DIFFLBM-SOLID-LIMIT — solid-limit convergence of a volumetric solid.

Acceptance criteria (task ledger ``GAP-DIFFLBM-SOLID-LIMIT``):

``D1``  drag change between mask scale 0.5 and 1.0  ``<= 5 %``
        (the ledger records 6.2 % for the amplitude closure)
``D2``  deviation from the hard-wall bounce-back momentum-exchange drag
        ``<= 10 %``

Case
----
Schäfer & Turek (1996) benchmark **2D-1**: an in-channel cylinder, Re = 20,
blockage 1/4.1, steady.  The reference drag is the official value
``Cd = 5.57953523384`` recorded (sha-pinned) in
``benchmarks/verified/cylinder/re20_st/result.json``; the geometry, the inlet
(Zou/He parabolic) / outlet (Zou/He pressure) pair, the MRT collision and the
Ladd momentum-exchange force are taken from
:mod:`tensorlbm.cylinder_st2d1` — this runner replaces *only* the cylinder wall
rule, so the three arms differ in nothing else.

Arms
----
``hardwall``   repository half-way bounce-back on a binary mask
               (``m >= 0.5``) + Ladd momentum exchange — the hard-wall
               reference arm (D2's denominator).
``amplitude``  :func:`tensorlbm.volume_solid.partial_bounce_back`, i.e. the
               volume-fraction / penalty closure (Walsh et al. 2009).  Drag is
               the pair-form momentum exchange of the same closure.
``interface``  :func:`tensorlbm.volume_solid.interface_bounce_back`, i.e. the
               same volume fractions read as *geometry* (VOF interface
               reconstruction + Bouzidi interpolated bounce-back).

Mask scale ``s``
----------------
The solid fraction field of the exact disc is blurred with a *volume
preserving* Gaussian of width ``sigma = (1 - s) * sigma_max`` cells, so
``s = 1`` is the solid limit (fractions = geometry) and ``s < 1`` is a diffuse
volume representation of the *same* body.  A closure that has a solid limit
must give the same drag for every ``s``.

Drag measurement
----------------
The primary drag for all three arms is the **control-volume momentum balance**
(:mod:`tensorlbm.control_volume_force`, a box enclosing the cylinder), so the
comparison is not between three different force estimators.  The
momentum-exchange value (Ladd / pair form) is recorded next to it as an
independent cross-check, and the ``hardwall`` arm reports both — their
agreement is the estimator validation.

Usage
-----
``PYTHONPATH=src python cases/GAP-DIFFLBM-SOLID-LIMIT/run_case.py --help``
``PYTHONPATH=src python cases/GAP-DIFFLBM-SOLID-LIMIT/run_case.py \
      --arm interface --D 20 --scale 1.0 --max-steps 40000 --out /tmp/x``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.cylinder_st2d1 import GEOMETRY, REFERENCE, case_geometry
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream
from tensorlbm.volume_solid import (
    blur_mask,
    interface_bounce_back,
    mask_scale_to_sigma,
    partial_bounce_back,
    partial_bounce_back_force,
    signed_distance_from_fraction,
    vof_circle,
)

ARMS = ("hardwall", "amplitude", "interface")
SCALES = (1.0, 0.75, 0.5)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def build_volume_fraction(D_cells: int, scale: float, sigma_max: float) -> tuple[Any, dict[str, Any]]:
    """Volume-of-fluid solid fraction of the 2D-1 cylinder at mask scale *scale*."""
    geo = case_geometry(D_cells)
    nx, ny = int(geo["nx"]), int(geo["ny"])
    m = vof_circle(nx, ny, float(geo["cx"]), float(geo["cy"]), float(geo["radius"]), sub=16)
    sigma = mask_scale_to_sigma(scale, sigma_max)
    if sigma > 0.0:
        m = blur_mask(m, sigma)
    info = {
        "D_cells": int(D_cells),
        "scale": float(scale),
        "sigma_cells": float(sigma),
        "solid_volume": float(m.sum()),
        "analytic_volume": math.pi * float(geo["radius"]) ** 2,
        "n_half_filled": int(((m > 0.02) & (m < 0.98)).sum()),
        "nx": nx,
        "ny": ny,
        "radius": float(geo["radius"]),
        "cx": int(geo["cx"]),
        "cy": int(geo["cy"]),
    }
    return m, info


def control_volume_box(geo: dict[str, Any], pad: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """``(ny, nx)`` control-volume mask around the cylinder, strictly interior."""
    nx, ny = int(geo["nx"]), int(geo["ny"])
    cx, cy = int(geo["cx"]), int(geo["cy"])
    x0, x1 = max(1, cx - pad), min(nx - 1, cx + pad + 1)
    y0, y1 = max(1, cy - pad), min(ny - 1, cy + pad + 1)
    box = torch.zeros((ny, nx), dtype=torch.bool)
    box[y0:y1, x0:x1] = True
    return box, (x0, x1, y0, y1)


def _shift2(field: torch.Tensor, d: tuple[int, int]) -> torch.Tensor:
    """``field`` sampled at ``x + d`` with ``d = (dy, dx)``."""
    return torch.roll(field, (-int(d[0]), -int(d[1])), (0, 1))


def momentum_balance_force_2d(
    f_old: torch.Tensor,
    f_new: torch.Tensor,
    f_post: torch.Tensor,
    control_volume: torch.Tensor,
    solid: torch.Tensor,
) -> tuple[float, float]:
    """Drag on an enclosed body from the discrete momentum budget (D2Q9).

    ``force_on_body = net_x-momentum imported across the control-volume faces
    - change of fluid momentum inside``, evaluated in lattice units over one
    complete step.  The observer uses only populations that cross the faces, so
    it is independent of the wall rule inside the volume — that is what makes
    the three arms comparable (it is the 2D counterpart of
    :mod:`tensorlbm.control_volume_force`, whose box helper requires ``nz >= 3``
    and therefore cannot wrap a strictly-2D run).
    """
    if f_old.shape != f_new.shape or f_old.shape != f_post.shape:
        raise ValueError("population tensors must share a shape")
    from tensorlbm.boundaries import C

    cv = control_volume
    owned = cv & ~solid
    fx = 0.0
    fy = 0.0
    for i in range(1, 9):
        ci = (int(C[i, 1]), int(C[i, 0]))
        cx, cy = int(C[i, 0]), int(C[i, 1])
        incoming_src = _shift2(f_post[i], (-ci[0], -ci[1]))  # value arriving at x in dir i
        enters = cv & ~_shift2(cv, (-ci[0], -ci[1]))
        leaves = cv & ~_shift2(cv, ci)
        net = (incoming_src * enters).sum() - (f_post[i] * leaves).sum()
        fx += cx * float(net)
        fy += cy * float(net)
    change = ((f_new - f_old) * owned).sum(dim=(1, 2))
    from tensorlbm.boundaries import C as _C

    fx -= float((change * _C[:, 0].to(change.dtype)).sum())
    fy -= float((change * _C[:, 1].to(change.dtype)).sum())
    return fx, fy


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_arm(
    *,
    arm: str,
    D_cells: int = 20,
    scale: float = 1.0,
    max_steps: int = 60_000,
    min_steps: int = 8_000,
    avg_window: int = 2_000,
    plateau_tol: float = 1.0e-3,
    cv_every: int = 25,
    cv_pad: int = 12,
    sigma_max: float = 1.0,
    tau: float = 0.8,
    u_lb: float | None = None,
    dtype: torch.dtype = torch.float64,
    progress: bool = True,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm}")
    geo = case_geometry(D_cells)
    nx, ny = int(geo["nx"]), int(geo["ny"])
    dev = torch.device("cpu")
    if u_lb is None:
        nu_lb = (tau - 0.5) / 3.0
        u_mean = GEOMETRY["re"] * nu_lb / D_cells
    else:
        u_mean = float(u_lb)
        nu_lb = u_mean * D_cells / GEOMETRY["re"]
        tau = 3.0 * nu_lb + 0.5
    dx = float(geo["dx"])
    y = torch.arange(ny, dtype=dtype) * dx
    u_in = u_mean * 4.0 * 1.5 * y * (GEOMETRY["ly"] - y) / GEOMETRY["ly"] ** 2

    m, mask_info = build_volume_fraction(D_cells, scale, sigma_max)
    m = m.to(dtype)
    solid = m >= 0.5
    fluid = ~solid
    ring = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )
    phi = signed_distance_from_fraction(m) if arm == "interface" else None
    wall = make_channel_wall_mask(ny, nx, solid, dev)
    cv, cv_bounds = control_volume_box(geo, cv_pad)
    dyn_p = 0.5 * u_mean**2 * D_cells

    rho0 = torch.ones((ny, nx), dtype=dtype)
    ux0 = u_in[:, None].expand(ny, nx).clone()
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))

    cd_series: list[float] = []
    fx_cv: list[float] = []
    fx_me: list[float] = []
    t0 = time.time()
    stop_reason = "max_steps"
    drift = float("nan")

    for step in range(1, max_steps + 1):
        f_start = f
        f_post = collide_mrt(f, tau)
        if arm == "hardwall":
            f_post = torch.where(solid.unsqueeze(0), f, f_post)
        f = stream(f_post)
        f = zou_he_inlet_velocity(f, u_in, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, wall)  # channel walls: hard in every arm

        if arm == "hardwall":
            fx, fy = compute_obstacle_forces(f, ring)
            f = bounce_back_cells(f, solid)
            me = float(fx)
        elif arm == "amplitude":
            fx, fy = partial_bounce_back_force(f, m)
            f = partial_bounce_back(f, m)
            me = float(fx)
        else:
            f = interface_bounce_back(f, f_post, phi)
            me = float("nan")

        if step % cv_every == 0:
            fxb, _fyb = momentum_balance_force_2d(f_start, f, f_post, cv, solid)
            fx_cv.append(fxb)
            fx_me.append(me)
            cd_series.append(fx_cv[-1] / dyn_p)

        if step % max(avg_window, 1) == 0 and len(cd_series) >= avg_window // cv_every:
            n = max(2, avg_window // cv_every)
            drift = _block_drift(cd_series[-n:])
            if progress:
                print(
                    f"  [{arm} D={D_cells} s={scale}] {step}/{max_steps} "
                    f"Cd~{sum(cd_series[-n:]) / n:.4f} drift~{drift:.2e} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        if (
            step >= min_steps
            and len(cd_series) >= 2 * (avg_window // cv_every)
            and drift == drift
            and drift <= plateau_tol
        ):
            stop_reason = "plateau"
            break

    n = max(2, min(len(cd_series), avg_window // cv_every))
    cd_cv = sum(cd_series[-n:]) / n
    me_vals = [v for v in fx_me[-n:] if v == v]
    cd_me = (sum(me_vals) / len(me_vals) / dyn_p) if me_vals else float("nan")
    half = max(1, n // 2)
    return {
        "arm": arm,
        "D_cells": int(D_cells),
        "scale": float(scale),
        "geometry": geo,
        "mask": mask_info,
        "lattice": {
            "u_mean": u_mean,
            "nu": nu_lb,
            "tau": tau,
            "re": u_mean * D_cells / nu_lb,
            "ma": u_mean / math.sqrt(1.0 / 3.0),
        },
        "steps": int(step),
        "stop_reason": stop_reason,
        "window_samples": int(n),
        "cd_cv": cd_cv,
        "cd_cv_first_half": sum(cd_series[-n : -half]) / max(half, 1),
        "cd_cv_second_half": sum(cd_series[-half:]) / max(half, 1),
        "cd_me": cd_me,
        "cd_plateau_drift": drift,
        "cv_bounds": cv_bounds,
        "n_solid_cells": int(solid.sum()),
        "n_half_filled_cells": int((ring & (m < 0.98)).sum()),
        "ref_cd": REFERENCE["cd"],
        "err_cd_cv_pct": (cd_cv - REFERENCE["cd"]) / REFERENCE["cd"] * 100.0,
        "err_cd_me_pct": (
            (cd_me - REFERENCE["cd"]) / REFERENCE["cd"] * 100.0 if cd_me == cd_me else float("nan")
        ),
        "wall_s": time.time() - t0,
        "dtype": str(dtype).replace("torch.", ""),
        "series": {"cd": cd_series},
    }


def _block_drift(series: Sequence[float], n_blocks: int = 4) -> float:
    """Relative spread of consecutive block means of the trailing window."""
    n = len(series)
    if n < n_blocks * 2:
        return float("nan")
    size = n // n_blocks
    start = n - size * n_blocks
    means = [sum(series[start + k * size : start + (k + 1) * size]) / size for k in range(n_blocks)]
    mean = sum(means) / n_blocks
    if mean == 0.0:
        return float("inf")
    return max(means) / mean - min(means) / mean


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def evaluate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the two ledger criteria to a set of arm/scale records."""
    by_key = {(r["arm"], round(r["scale"], 3), r["D_cells"]): r for r in records}
    resolutions = sorted({r["D_cells"] for r in records})
    table: list[dict[str, Any]] = []
    for D in resolutions:
        ref = by_key.get(("hardwall", 1.0, D))
        for arm in ARMS:
            if arm == "hardwall":
                continue
            base = by_key.get((arm, 1.0, D))
            half = by_key.get((arm, 0.5, D))
            if base is None or half is None:
                continue
            delta_scale = abs(half["cd_cv"] - base["cd_cv"]) / abs(base["cd_cv"]) * 100.0
            delta_hard = (
                abs(base["cd_cv"] - ref["cd_cv"]) / abs(ref["cd_cv"]) * 100.0 if ref else float("nan")
            )
            row = {
                "D_cells": D,
                "arm": arm,
                "cd_scale_1.0": base["cd_cv"],
                "cd_scale_0.5": half["cd_cv"],
                "delta_scale_pct": delta_scale,
                "cd_hardwall": ref["cd_cv"] if ref else float("nan"),
                "delta_hardwall_pct": delta_hard,
                "D1_scale_le_5pct": bool(delta_scale <= 5.0),
                "D2_hardwall_le_10pct": bool(delta_hard <= 10.0),
                "ref_cd": base["ref_cd"],
                "err_vs_ref_scale_1.0_pct": base["err_cd_cv_pct"],
            }
            table.append(row)
    ok = {
        arm: all(
            r["D1_scale_le_5pct"] and (r["D2_hardwall_le_10pct"] if r["D2_hardwall_le_10pct"] == r["D2_hardwall_le_10pct"] else False)
            for r in table
            if r["arm"] == arm
        )
        for arm in ("amplitude", "interface")
    }
    return {
        "criteria": {
            "D1": "|Cd(scale=0.5) - Cd(scale=1.0)| / Cd(scale=1.0) <= 5 %",
            "D2": "|Cd(arm, scale=1.0) - Cd(hardwall)| / Cd(hardwall) <= 10 %",
        },
        "table": table,
        "passes_all_resolutions": ok,
        "verdict": "PASS" if ok.get("interface") else "FAIL",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--D", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=60_000)
    ap.add_argument("--min-steps", type=int, default=8_000)
    ap.add_argument("--avg-window", type=int, default=2_000)
    ap.add_argument("--plateau-tol", type=float, default=1.0e-3)
    ap.add_argument("--cv-every", type=int, default=25)
    ap.add_argument("--cv-pad", type=int, default=12)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    res = run_arm(
        arm=args.arm,
        D_cells=args.D,
        scale=args.scale,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        avg_window=args.avg_window,
        plateau_tol=args.plateau_tol,
        cv_every=args.cv_every,
        cv_pad=args.cv_pad,
        sigma_max=args.sigma_max,
        progress=not args.quiet,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.arm}_D{args.D}_s{args.scale:g}.json"
    (args.out / name).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "series"}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())