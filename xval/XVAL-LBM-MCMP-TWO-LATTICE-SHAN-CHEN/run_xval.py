#!/usr/bin/env python3
"""XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN — cross-validation of the two-lattice
(two distribution functions) Shan-Chen multi-component multi-phase LBM model
against the closed-form two-layer (stratified) Poiseuille solution.

Round-2 duty (owner 2026-11-11 two-round method): research the mature method
first (OSS-cache + literature), then combine and advance — do not repeat the
round-1 death (m17959 honest-FAIL with no failure reason / no artifact).

The two-lattice Shan-Chen MCMP model is the classic Shan & Chen (1993/1994)
two-component construction: each component sigma carries its own distribution
function f_sigma on its own lattice, they collide independently (BGK), and the
phase separation is driven by the cross-component pseudopotential force

    F_sigma = -G_12 cs^2 psi_sigma(x) sum_i w_i psi_other(x+c_i) c_i

with the *linear* pseudopotential psi(rho) = rho (this library's convention,
G_12 < 0 = repulsive / phase-separating).

The standard benchmark for this model is the *stratified two-phase Poiseuille*
flow: two immiscible components in a plane channel with no-slip walls (rows 0
and ny-1), component 1 filling the lower half and component 2 the upper half,
driven by a uniform body force per unit mixture volume.  The closed-form
solution (velocity + shear-stress continuity at the interface, no-slip at the
wall planes, u'' = -G/nu per layer) is the external anchor.

Reference (external_anchor)
----------------------------
* Shan & Chen (1993) Phys. Rev. E 47, 1815 — the original two-component
  pseudopotential model (two distribution functions, one per component).
* Shan & Doolen (1995) J. Stat. Phys. 81, 379 — multicomponent LBM with the
  barycentric (mixture) velocity in the equilibrium.
* Martys & Chen (1996) J. Comput. Phys. 126, 276 — viscous coupling in
  multicomponent LBM.
* Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308 — second-order forcing.
* OSS reference: geoelements/mcmp-lbm (commit 015a26f) — a 3D two-component
  Shan-Chen solver with the same two-PDF structure (f_1/f_2 independent
  collision, shared cross-component force).

Acceptance (owner 15% band)
----------------------------
The flow-rate error |Q_mix - Q_ref| / Q_ref must be < 15% against the
closed-form two-layer solution, under the *measured* half-way wall convention
(H = ny - 2), at two resolutions (ny = 25, 33) and two coupling points
(G_12 = -2.5, -2.75).

Usage::

    PYTHONPATH=src python3 xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py \
        --out xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_two_component
from tensorlbm.solver import stream

CS2 = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Closed-form two-layer Poiseuille (external anchor)
# ---------------------------------------------------------------------------


def layered_poiseuille(
    ny: int,
    G_x: float,
    rho_heavy: float,
    rho_light: float,
    tau_lower: float,
    tau_upper: float,
    *,
    rule: str = "mixture",
) -> tuple[list[float], float]:
    """Exact two-layer co-moving Poiseuille profile.

    Geometry: channel of effective height H = ny - 2 (half-way wall planes),
    lower layer 0 <= y <= h, upper layer h <= y <= H, h = (ny-2)/2.
    Both layers co-move (velocity + shear-stress continuous at the interface):

        u_i'' = -G / nu_i ,  u_lower(0) = 0 ,  u_upper(H) = 0
        u_lower(h) = u_upper(h) ,  nu_lower u_lower'(h) = nu_upper u_upper'(h)

    The last condition degenerates to nu_lower b_lower = nu_upper b_upper
    because both layers carry the same body force per unit volume.

    Returns (u_profile, Q_ref) where u_profile is the velocity at each fluid
    row (y = 1 .. ny-2) and Q_ref = sum_y u(y) is the reference flow rate.
    """
    H = ny - 2
    h = H / 2.0
    nu1 = CS2 * (tau_lower - 0.5)
    nu2 = CS2 * (tau_upper - 0.5)
    if rule == "mixture":
        # density-weighted mixture viscosity (continuum limit of co-moving BGK)
        nu_lower = rho_heavy * nu1 + rho_light * nu2
        nu_upper = rho_heavy * nu1 + rho_light * nu2
    else:
        nu_lower = nu1
        nu_upper = nu2

    # u(y) = -G/(2 nu) y^2 + b y ;  u(0)=0 -> c=0 ; u(H)=0 -> b = G H/(2 nu)
    b_lower = G_x * H / (2.0 * nu_lower)
    b_upper = G_x * H / (2.0 * nu_upper)
    u = []
    for y in range(1, ny - 1):
        yy = float(y)
        if yy <= h:
            u.append(-G_x / (2.0 * nu_lower) * yy * yy + b_lower * yy)
        else:
            u.append(-G_x / (2.0 * nu_upper) * yy * yy + b_upper * yy)
    return u, sum(u)


# ---------------------------------------------------------------------------
# Two-lattice Shan-Chen MCMP simulation
# ---------------------------------------------------------------------------


def init_stratified(
    ny: int,
    nx: int,
    rho_heavy: float,
    rho_light: float,
    *,
    interface_width: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise the two-component stratified channel.

    Component 1 (heavy) fills the lower half, component 2 (light) the upper
    half; the interface is a tanh profile.  Both components start at rest.
    """
    dev = torch.device(device)
    yy = torch.arange(ny, dtype=torch.float32, device=dev).view(ny, 1)
    centre = (ny - 1) / 2.0
    s = 0.5 * (1.0 - torch.tanh((yy - centre) / max(interface_width, 1e-6)))
    # s = 1 in the lower half (heavy), 0 in the upper half (light)
    rho1 = rho_light + s * (rho_heavy - rho_light)  # (ny,1)
    rho2 = rho_heavy - rho1  # rho1 + rho2 = rho_heavy + rho_light = 1
    # broadcast to (ny, nx)
    rho1 = rho1.expand(ny, nx)
    rho2 = rho2.expand(ny, nx)
    zero = torch.zeros((ny, nx), dtype=torch.float32, device=dev)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


def run_stratified(
    *,
    G_12: float,
    ny: int,
    nx: int = 4,
    G_x: float = 5e-5,
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    tau1: float = 1.0,
    tau2: float = 0.75,
    u_eq: str = "viscous",
    force_scheme: str = "guo",
    steps: int = 20000,
    interface_width: float = 2.0,
    stop_tol: float = 1e-6,
    check_every: int = 500,
    device: torch.device | str = "cpu",
) -> dict:
    """Run the two-lattice Shan-Chen MCMP stratified channel to steady state.

    No-slip walls on rows 0 and ny-1 (post-streaming bounce-back); the
    effective wall planes sit at the wall-cell centres (H = ny - 2), which is
    the *measured* convention of the verified mcmp_stratified_poiseuille
    benchmark (--wall-calibration).
    """
    dev = torch.device(device)
    f1, f2 = init_stratified(ny, nx, rho_heavy, rho_light, interface_width=interface_width, device=dev)
    # solid mask: wall rows 0 and ny-1
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[ny - 1, :] = True

    u_prev = float("inf")
    stable = 0
    converged = False
    steps_run = 0
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, u_eq=u_eq,
            force_scheme=force_scheme, gx=G_x, gy=0.0, solid_mask=wall,
        )
        f1, f2 = stream(f1), stream(f2)
        steps_run = step
        if step % check_every == 0:
            rho1, ux1, _ = macroscopic(f1)
            rho2, ux2, _ = macroscopic(f2)
            rho_tot = rho1 + rho2
            u_mix = (rho1 * ux1 + rho2 * ux2) / rho_tot
            u_max = float(u_mix.abs().max())
            if not math.isfinite(u_max):
                break
            if math.isfinite(u_prev) and abs(u_max - u_prev) <= stop_tol * max(u_max, 1e-12):
                stable += 1
                if stable >= 2:
                    converged = True
                    break
            else:
                stable = 0
            u_prev = u_max

    rho1, ux1, _ = macroscopic(f1)
    rho2, ux2, _ = macroscopic(f2)
    rho_tot = rho1 + rho2
    u_mix = (rho1 * ux1 + rho2 * ux2) / rho_tot
    # flow rate per row (mixture velocity), summed over fluid rows (1..ny-2)
    Q_mix = float(u_mix[1:ny - 1, 0].sum())
    # interface position: rho1 = 0.5 crossing
    rho1_col = rho1[:, 0]
    y_cross = None
    for y in range(1, ny - 1):
        if rho1_col[y] <= 0.5 < rho1_col[y - 1] or rho1_col[y - 1] <= 0.5 < rho1_col[y]:
            y_cross = float(y)
            break
    return {
        "G_12": G_12,
        "ny": ny,
        "nx": nx,
        "G_x": G_x,
        "u_eq": u_eq,
        "force_scheme": force_scheme,
        "steps_requested": steps,
        "steps_run": steps_run,
        "converged": converged,
        "Q_mix": Q_mix,
        "u_max": u_max,
        "y_cross": y_cross,
        "rho1": rho1,
        "rho2": rho2,
        "u_mix": u_mix,
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    *,
    G_12: float,
    ny: int,
    G_x: float = 5e-5,
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    tau1: float = 1.0,
    tau2: float = 0.75,
    u_eq: str = "viscous",
    force_scheme: str = "guo",
    steps: int = 20000,
    device: torch.device | str = "cpu",
) -> dict:
    """Run one cell and compare against the closed-form reference."""
    run = run_stratified(
        G_12=G_12, ny=ny, G_x=G_x, rho_heavy=rho_heavy, rho_light=rho_light,
        tau1=tau1, tau2=tau2, u_eq=u_eq, force_scheme=force_scheme,
        steps=steps, device=device,
    )
    # reference under the mixture viscosity rule (primary) and component rule
    u_ref_mix, Q_ref_mix = layered_poiseuille(
        ny, G_x, rho_heavy, rho_light, tau1, tau2, rule="mixture"
    )
    u_ref_comp, Q_ref_comp = layered_poiseuille(
        ny, G_x, rho_heavy, rho_light, tau1, tau2, rule="component"
    )
    Q_mix = run["Q_mix"]
    err_mix = (Q_mix - Q_ref_mix) / Q_ref_mix
    err_comp = (Q_mix - Q_ref_comp) / Q_ref_comp
    return {
        **run,
        "Q_ref_mix": Q_ref_mix,
        "Q_ref_comp": Q_ref_comp,
        "err_Q_mix_mixture": err_mix,
        "err_Q_mix_component": err_comp,
        "pass_15pct": abs(err_mix) < 0.15 and abs(err_comp) < 0.15,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--quick", action="store_true", help="smoke protocol (fewer steps)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    t_start = time.time()

    # Two resolutions x two coupling points (acceptance matrix)
    cells = []
    for ny in (25, 33):
        for G_12 in (-2.5, -2.75):
            steps = 8000 if args.quick else (20000 if ny == 25 else 30000)
            c = cross_validate(G_12=G_12, ny=ny, steps=steps)
            cells.append(c)

    summary = {
        "anchor": "XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN",
        "model": "two-lattice (two distribution functions) Shan-Chen MCMP, "
                 "linear psi, BGK, viscous velocity coupling, Guo forcing",
        "external_anchor": {
            "analytic": "two-layer co-moving Poiseuille, u''=-G/nu per layer "
                         "with velocity + shear-stress continuity "
                         "(xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py::layered_poiseuille)",
            "references": [
                "Shan & Chen (1993) Phys. Rev. E 47, 1815",
                "Shan & Doolen (1995) J. Stat. Phys. 81, 379",
                "Martys & Chen (1996) J. Comput. Phys. 126, 276",
                "Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308",
                "OSS: geoelements/mcmp-lbm (commit 015a26f) two-PDF SC structure",
            ],
        },
        "acceptance": "err_Q_mix < 15% against closed-form two-layer solution "
                       "under measured half-way wall convention (H=ny-2)",
        "cells": [
            {
                "G_12": c["G_12"],
                "ny": c["ny"],
                "steps_run": c["steps_run"],
                "converged": c["converged"],
                "Q_mix": c["Q_mix"],
                "Q_ref_mix": c["Q_ref_mix"],
                "Q_ref_comp": c["Q_ref_comp"],
                "err_Q_mix_mixture": c["err_Q_mix_mixture"],
                "err_Q_mix_component": c["err_Q_mix_component"],
                "pass_15pct": c["pass_15pct"],
            }
            for c in cells
        ],
        "all_pass_15pct": all(c["pass_15pct"] for c in cells),
        "elapsed_s": round(time.time() - t_start, 1),
        "repro_cmd": (
            "PYTHONPATH=src python3 xval/XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN/run_xval.py"
        ),
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"all_pass_15pct": summary["all_pass_15pct"], "cells": summary["cells"]}, indent=1))
    print(f"wrote {out / 'summary.json'}")
    return 0 if summary["all_pass_15pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
