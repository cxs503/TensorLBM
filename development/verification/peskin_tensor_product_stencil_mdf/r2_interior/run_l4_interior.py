"""Round-2 XVAL (same anchor): end-to-end cylinder drag, solid-interior treatment.

Round 1 (``../run_xval.py``) closed the *operator* level (kernel/stencil identity,
MDF fixed point, no-slip) but the end-to-end L4 column was an honest FAIL: every
arm predicted ``Cd(Re=20) ~= 25`` against the literature ``2.05`` (Dennis & Chang
1970), i.e. ~12x.  Round 1 attributed this to the shared driver -- a
solid-interior *penalty* force (``interior_mask_force`` applied every step as a
Guo source) which leaves the solid permeable -- and explicitly deferred the fix.

This driver isolates that hypothesis instead of assuming it:

``interior`` modes
    ``penalty``   round-1 behaviour: ``F += interior_mask_force(u, mask, 0)``,
                  applied every step as a Guo source (baseline, reproduced here).
    ``none``      no interior treatment at all (markers only).
    ``hard``      hard solid: after the macroscopic step the solid cells are
                  reset to ``f_eq(rho_local, u = 0)`` -- momentum inside the body
                  is zeroed by *overwrite* instead of by a booked force, so the
                  drag is carried by the markers alone (the classic
                  "solid nodes" treatment used with diffuse-interface IBM).
    ``hard_rho1`` same, with ``rho = 1`` inside the body (incompressible solid).

Drag is measured on the Lagrangian markers (``Cd = -sum_j f_j,x / (0.5 rho U^2 D)``),
with the applied Eulerian force budget (``sum F_surf``, ``sum F_int``) recorded
separately so the two can be compared.

Every number in the report is read back from the JSON artifacts this script
writes (no hand-typing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
XV = HERE.parent
ROOT = XV.parents[2]

sys.path.insert(0, str(XV))
sys.path.insert(0, str(ROOT / "src"))

import run_xval as R1  # noqa: E402  (round-1 driver: helpers + vendored OSS operators)

from tensorlbm.boundaries import cylinder_mask, far_field_bc_2d  # noqa: E402
from tensorlbm.d2q9 import C as C2D  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.ibm_force_transfer import (  # noqa: E402
    build_stencil,
    direct_forcing,
    direct_forcing_mdf,
    solid_interior_reset,
)
from tensorlbm.ibm_openwater import CYLINDER_CD_2D  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

DT = torch.float64


def _solid_reset(f: torch.Tensor, solid: torch.Tensor, rho: float | None = None) -> torch.Tensor:
    """Hard solid-node treatment, via the shipped helper (no booked force)."""
    return solid_interior_reset(
        f, solid, lambda r: equilibrium(r, torch.zeros_like(r), torch.zeros_like(r)), rho=rho
    )


def coupled(
    re: float = 20.0,
    D: int = 8,
    h_marker: float = 0.5,
    n_iter: int = 10,
    n_steps: int = 6000,
    interior: str = "hard",
    scheme: str = "tl_mdf",
    nx_D: float = 30.0,
    ny_D: float = 20.0,
    u_in: float = 0.1,
    sample_every: int = 20,
    threads: int = 8,
    cd_hold_from: float | None = None,
) -> dict:
    torch.set_num_threads(threads)
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    nx, ny = int(nx_D * D), int(ny_D * D)
    cx, cy = float(nx) / 3.0, ny / 2.0
    Rrad = D / 2.0
    solid = cylinder_mask(nx, ny, cx, cy, Rrad, torch.device("cpu"))
    pts = R1.markers_circle(cx, cy, Rrad, h_marker)

    st = build_stencil(torch.tensor(pts, dtype=DT), (ny, nx), "4pt")
    u_target = torch.zeros((len(pts), 2), dtype=DT)

    rho = torch.ones((ny, nx), dtype=DT)
    f = equilibrium(rho, torch.full_like(rho, u_in), torch.zeros_like(rho))

    F = torch.zeros((2, ny, nx), dtype=DT)  # lagged body force (Guo source)
    hist = {"t": [], "cd": [], "slip": [], "f_surf_x": [], "f_int_x": [], "u_x_sum": [], "umax_solid": [], "cl": []}
    dyn = 0.5 * u_in**2 * D
    t0 = time.time()
    n_int_cells = int(solid.sum())
    for step in range(1, n_steps + 1):
        f = collide_mrt(f, tau)
        f = R1.guo_term(f, F[1], F[0], tau)  # F rows are (y, x)
        f = stream(f)
        f = far_field_bc_2d(f, u_in)
        if interior in ("hard", "hard_rho1"):
            f = _solid_reset(f, solid, None if interior == "hard" else 1.0)
        rho = f.sum(0)
        ux = (f * C2D[:, 0].to(DT).view(9, 1, 1)).sum(0) / rho
        uy = (f * C2D[:, 1].to(DT).view(9, 1, 1)).sum(0) / rho

        # --- surface operator -------------------------------------------------
        if scheme == "tl_mdf":
            fm_t, fg, _b, _t = direct_forcing_mdf(
                st, torch.stack([uy, ux]), u_target, None, n_mdf=n_iter
            )
        elif scheme == "tl_pointwise":
            fm_t, fg, _b, _t = direct_forcing(
                st, torch.stack([uy, ux]), u_target, None, n_correctors=n_iter
            )
        elif scheme == "oss_mdf":
            gf, fm, *_ = R1.oss_operators(pts, ny, nx, ux.numpy(), uy.numpy(), n_iter)
            fg = torch.tensor(np.stack([gf[1].T, gf[0].T]))
            fm_t = torch.tensor(np.asarray(fm)[:, ::-1].copy())
        else:
            raise ValueError(scheme)
        F_surf = fg
        # NB: the grid is stacked as (uy, ux), so marker-force component 0 is y
        # and component 1 is x (round 1 silently reversed the pair before taking
        # the drag; doing it here explicitly keeps lift available as well).
        fm_np = fm_t.numpy()[:, ::-1]  # -> (fx, fy)

        if interior == "penalty":
            F_int = R1.interior_mask_force(torch.stack([uy, ux]), solid, torch.zeros((2, ny, nx), dtype=DT))
        else:
            F_int = torch.zeros((2, ny, nx), dtype=DT)
        F = F_surf + F_int

        if step % sample_every == 0:
            hist["t"].append(step * u_in / D)
            hist["cd"].append(-float(fm_np.sum(axis=0)[0]) / dyn)
            hist["cl"].append(-float(fm_np.sum(axis=0)[1]) / dyn)
            slip_abs, _ = R1.iface_residual_pointwise(pts, ux.numpy(), uy.numpy(), fm_np[:, ::-1])
            hist["slip"].append(slip_abs / u_in)
            hist["f_surf_x"].append(float(F_surf[1].sum()))
            hist["f_int_x"].append(float(F_int[1].sum()))
            hist["u_x_sum"].append(float((rho * ux).sum()))
            hist["umax_solid"].append(float(ux[solid].abs().max()) if n_int_cells else 0.0)

    cd = np.asarray(hist["cd"])
    k = 0
    if cd_hold_from is None:
        half = cd[len(cd) // 2:]
        slip = np.asarray(hist["slip"])[len(cd) // 2:]
        fs = np.asarray(hist["f_surf_x"])[len(cd) // 2:]
        fi = np.asarray(hist["f_int_x"])[len(cd) // 2:]
        us = np.asarray(hist["umax_solid"])[len(cd) // 2:]
    else:
        k = int(np.searchsorted(np.asarray(hist["t"]), cd_hold_from))
        half = cd[k:]
        slip = np.asarray(hist["slip"])[k:]
        fs = np.asarray(hist["f_surf_x"])[k:]
        fi = np.asarray(hist["f_int_x"])[k:]
        us = np.asarray(hist["umax_solid"])[k:]
    ref = CYLINDER_CD_2D.get(int(re))
    cd_mean = float(np.mean(half))
    return {
        "case": {
            "re": re,
            "scheme": scheme,
            "interior": interior,
            "D": D,
            "grid_nx": nx,
            "grid_ny": ny,
            "blockage": D / ny,
            "u_in": u_in,
            "nu": nu,
            "tau": tau,
            "n_steps": n_steps,
            "h_marker": h_marker,
            "n_markers": len(pts),
            "n_iter": n_iter,
            "n_solid_cells": n_int_cells,
            "dtype": "float64",
        },
        "cd_mean_hold": cd_mean,
        "cd_std_hold": float(np.std(half)),
        "cd_last": float(cd[-1]),
        "cd_reference": ref,
        "cd_rel_error": None if ref is None else (cd_mean - ref) / ref,
        "pass_15pct": None if ref is None else bool(abs(cd_mean - ref) / ref < 0.15),
        "cl_mean_hold": float(np.mean(np.asarray(hist["cl"])[k:])),
        "slip_mean_hold": float(np.mean(slip)),
        "f_surf_x_hold": float(np.mean(fs)),
        "f_int_x_hold": float(np.mean(fi)),
        "interior_force_share": float(abs(np.mean(fi)) / (abs(np.mean(fs)) + abs(np.mean(fi)) + 1e-300)),
        "umax_in_solid_hold": float(np.mean(us)),
        "wall_s": time.time() - t0,
        "hist": hist,
    }


def run_matrix(cases: list[dict], out_path: Path) -> dict:
    results = []
    for c in cases:
        r = coupled(**c)
        results.append(r)
        print(
            f"[{r['case']['scheme']:12s} int={r['case']['interior']:10s} "
            f"D={r['case']['D']:<2d} h={r['case']['h_marker']:<4} "
            f"Re={r['case']['re']:<5}] Cd={r['cd_mean_hold']:.4f} "
            f"(ref {r['cd_reference']}) err={r['cd_rel_error']:+.1%} "
            f"slip={r['slip_mean_hold']:.2%} f_int/f_surf={r['interior_force_share']:.2f} "
            f"{r['wall_s']:.1f}s",
            flush=True,
        )
    out = {"cases": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screen")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.stage == "screen":
        cases = [
            dict(re=20.0, D=8, h_marker=0.5, interior="penalty", n_steps=3000),
            dict(re=20.0, D=8, h_marker=0.5, interior="none", n_steps=3000),
            dict(re=20.0, D=8, h_marker=0.5, interior="hard", n_steps=3000),
            dict(re=20.0, D=8, h_marker=0.5, interior="hard_rho1", n_steps=3000),
        ]
        out = HERE / "artifacts" / "screen.json"
    elif args.stage == "ladder":
        # grid-refinement + marker-refinement ladder at Re=20, plus the Re=40
        # cross-check; all with the hard-interior treatment the screen selects.
        cases = [
            dict(re=20.0, D=8, h_marker=0.5, interior="hard", n_steps=6000),
            dict(re=20.0, D=16, h_marker=1.0, interior="hard", n_steps=6000),
            dict(re=20.0, D=8, h_marker=1.0, interior="hard", n_steps=6000),
            dict(re=40.0, D=8, h_marker=0.5, interior="hard", n_steps=6000),
        ]
        out = HERE / "artifacts" / "ladder.json"
    else:
        raise SystemExit(f"unknown stage {args.stage}")
    if args.out:
        out = Path(args.out)
    run_matrix(cases, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())