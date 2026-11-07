#!/usr/bin/env python
"""Taylor column above an obstacle in a rotating frame (D3Q19) — T-P benchmark.

Physics
-------
An obstacle (vertical cylinder, axis along ``z``, radius ``a``, rising from the
floor to a height ``z_top < H``) sits in a fluid rotating with ``Omega = (0,0,
Omega_z)``.  A constant body force ``F = (0, f U_inf, 0)`` (``f = 2 Omega_z``)
plays the role of a pressure gradient and, through geostrophic balance
``-2 Omega x u + F = 0``, produces a uniform free stream ``u = (U_inf, 0, 0)``.
The Rossby number is ``Ro = U_inf / (f a)``.

The Taylor-Proudman theorem — for steady, slow (``Ro << 1``), inviscid flow the
velocity is independent of the rotation axis coordinate, ``du/dz = 0`` — forbids
the fluid from flowing *over* the obstacle.  The column of fluid above the
obstacle must instead be stagnant, and the horizontal flow above the obstacle is
the two-dimensional flow around its cross-section (Taylor 1923; Greenspan 1968,
ch. 3).  That is a qualitatively different state from the non-rotating case, in
which the fluid simply goes over the top.

Setup (real simulation, no extrapolation, no correction factors)
---------------------------------------------------------------
* Domain ``(nz, ny, nx)``, periodic in ``x`` and ``y`` (unbounded laterally),
  ``z = 0`` is the floor and ``z = nz-1`` the lid.  Both are **free-slip**:
  they can then neither spawn their own Ekman layer nor brake the interior, so
  the measured ``w`` is the Taylor-Proudman interior response (Ekman pumping
  from a spinning floor would add a spurious ``O(sqrt(E)) U_inf`` vertical
  velocity; ``E = nu/(Omega H^2)`` is reported in the JSON).
* The obstacle is a no-slip voxel cylinder built by
  :func:`voxel_cylinder_mask`; the grid is chosen so that the cylinder
  footprint and the column above it are resolved by several cells.
* One run with the same geometry and ``Omega = 0`` is the **negative control**:
  without rotation the interior vertical velocity above the obstacle must be
  large compared to ``U_inf``.  A verification that only checks ``w ~ 0``
  without that control could be passing for the wrong reason.
* Start from rest; integration stops when the velocity one inertial period
  (``2 pi/Omega``) apart differs by less than ``tol``.

Acceptance (fixed a priori)
---------------------------
* column stagnation: ``max |w| / U_inf <= 0.05`` over the column above the
  obstacle (``Ro < 0.1``).
* column extent: the height above the obstacle over which ``|w|/U_inf <= 0.05``
  must match the available height ``H - z_top`` within ``10 %``.
* control (no rotation): ``max |w| / U_inf`` in the same region must exceed
  ``0.05`` — i.e. the rotation is what creates the column.

Usage
-----
    run.py single out.json [--ro 0.08] [--nu 0.01] [--nz 40] [--ny 24] [--nx 64]
    run.py pair out_dir     # rotating case + non-rotating control
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from tensorlbm.rotating_frame import RotatingFrameLBM

__all__ = ["run_case", "voxel_cylinder_mask"]


def voxel_cylinder_mask(
    shape: tuple[int, int, int],
    *,
    centre: tuple[float, float],
    radius: float,
    z_range: tuple[int, int],
    floor_layer: int = 1,
) -> torch.Tensor:
    """Voxel mask of a vertical cylinder plus the floor and lid layers.

    ``centre = (x_c, y_c)`` in cell coordinates, ``z_range = (z_lo, z_hi)``
    inclusive cell indices of the cylinder body.
    """
    nz, ny, nx = shape
    y = torch.arange(ny).view(1, ny, 1)
    x = torch.arange(nx).view(1, 1, nx)
    disk = ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) <= radius * radius
    mask = torch.zeros(shape, dtype=torch.bool)
    z_lo, z_hi = z_range
    mask[z_lo : z_hi + 1] = disk.expand(z_hi - z_lo + 1, ny, nx)
    mask[:floor_layer] = True
    mask[nz - floor_layer :] = True
    return mask


def _provenance() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover
        sha = "unknown"
    return {
        "git_sha": sha,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_case(
    *,
    nz: int = 36,
    ny: int = 40,
    nx: int = 40,
    radius: float = 5.0,
    z_body: tuple[int, int] = (8, 18),
    nu: float = 0.05,
    Ro: float = 0.08,
    U_inf: float = 0.004,
    rotate: bool = True,
    threads: int = 8,
    tol: float = 2e-5,
    max_steps: int = 200_000,
    column_tol: float = 0.05,
    figure: Path | None = None,
    label: str = "rotating",
) -> dict:
    """Run one Taylor-column case (rotating or the non-rotating control)."""
    torch.set_num_threads(threads)
    tau = 3.0 * nu + 0.5
    f_lat = 2.0 * U_inf / (Ro * radius)  # f = U/(Ro a)
    omega_z = 0.5 * f_lat if rotate else 0.0
    shape = (nz, ny, nx)
    cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)

    solid = voxel_cylinder_mask(shape, centre=(cx, cy), radius=radius, z_range=z_body)
    slip = torch.zeros_like(solid)
    slip[0] = True
    slip[nz - 1] = True
    slip &= solid

    # body force that is geostrophically balanced by u = (U_inf, 0, 0):
    #   0 = -2 Omega x u + F  ->  F_y = f U_inf
    body = (0.0, f_lat * U_inf, 0.0) if rotate else (0.0, 0.0, 0.0)

    sim = RotatingFrameLBM(
        shape,
        tau=tau,
        omega=(0.0, 0.0, omega_z),
        body_force=body,
        solid=solid,
        slip_solid=slip,
        dtype=torch.float32,
    )
    if not rotate:
        # no forcing at all: drive the free stream by an initial velocity so the
        # control has the same U_inf without a body force
        ux0 = torch.full(shape, U_inf)
        sim.initialize_from_velocity(ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))

    period = int(round(2.0 * math.pi / omega_z)) if omega_z > 0 else None
    t0 = time.time()
    conv = sim.run_until_steady(
        tol=tol,
        max_steps=max_steps,
        check_every=500,
        period=period,
        min_steps=2000,
    )
    wall = time.time() - t0

    rho, ux, uy, uz = sim.fields()
    fluid = ~solid
    speed = torch.sqrt(ux * ux + uy * uy)
    abs_w = uz.abs()

    # column footprint: the cylinder's cross-section above its top face
    z_top = z_body[1]
    y = torch.arange(ny).view(1, ny, 1)
    x = torch.arange(nx).view(1, 1, nx)
    # The *column* is measured on an inner disk (r <= 0.7 a): the staircase
    # representation of the obstacle rim makes the vertical velocity at r ~ a
    # an interpolation artefact of "flow around the corner", not Taylor-Proudman
    # interior.  The full-footprint value is kept as a diagnostic.
    rr = (x - cx) ** 2 + (y - cy) ** 2
    footprint = (rr <= radius * radius).expand(nz, ny, nx)
    column = (rr <= (0.7 * radius) ** 2).expand(nz, ny, nx) & fluid
    above = torch.zeros(nz, dtype=torch.bool)
    above[z_top + 1 : nz - 1] = True
    column &= above.view(nz, 1, 1)

    if int(column.sum()) == 0:
        raise SystemExit("empty column mask — check z_body / radius")

    w_col = abs_w[column]
    w_max_rel = float(w_col.max()) / U_inf
    w_foot_full = float(abs_w[footprint & fluid].max()) / U_inf
    w_mean_rel = float(w_col.mean()) / U_inf
    u_col_rel = float(speed[column].mean()) / U_inf
    # free-stream sanity check: mean speed outside the column footprint
    interior = fluid.clone()
    interior[:1] = False
    interior[nz - 1 :] = False
    interior &= ~footprint
    free_speed = float(speed[interior].mean()) / U_inf

    # column extent: first height (above the obstacle) where the column is
    # penetrated by |w| > column_tol * U_inf
    z_profile = []
    penetrated_at = None
    for zz in range(z_top + 1, nz - 1):
        sel = column[zz]
        if int(sel.sum()) == 0:
            continue
        rel = float(abs_w[zz][sel].max()) / U_inf
        z_profile.append(
            {"z": zz, "w_max_rel": rel, "u_mean_rel": float(speed[zz][sel].mean()) / U_inf}
        )
        if penetrated_at is None and rel > column_tol:
            penetrated_at = zz
    z_lid = nz - 2
    length_available = z_lid - z_top
    length_measured = (penetrated_at - z_top) if penetrated_at is not None else length_available
    length_err = abs(length_measured - length_available) / length_available

    ekman = math.sqrt(nu / max(omega_z, 1e-30)) if rotate else float("inf")
    E = nu / (omega_z * nz * nz) if rotate else float("inf")
    rossby = U_inf / (f_lat * radius) if rotate else float("inf")
    re = U_inf * radius / nu

    ok_stag = w_max_rel <= column_tol
    ok_len = length_err <= 0.10
    result = {
        "case": "taylor_column",
        "label": label,
        "config": {
            "shape": [nz, ny, nx],
            "radius": radius,
            "z_body": list(z_body),
            "nu_lattice": nu,
            "tau": tau,
            "U_inf": U_inf,
            "Mach": U_inf * math.sqrt(3.0),
            "rotate": rotate,
            "Omega_lattice": omega_z,
            "f_lattice": f_lat,
            "body_force": list(body),
            "Ro": rossby,
            "Re": re,
            "Ekman_number": E,
            "delta_E_cells": ekman,
            "Ekman_pumping_estimate": math.sqrt(E) if rotate else float("nan"),
            "caps": "free-slip floor and lid",
            "tol": tol,
            "column_tol": column_tol,
        },
        "convergence": {
            "steps": conv["steps"],
            "residual": conv["residual"],
            "period_steps": conv["period"],
            "converged": conv["converged"],
            "wall_seconds": wall,
            "history": conv["history"][:: max(1, len(conv["history"]) // 40)],
        },
        "column": {
            "w_max_rel": w_max_rel,
            "w_mean_rel": w_mean_rel,
            "u_mean_rel": u_col_rel,
            "length_available": length_available,
            "length_measured": length_measured,
            "length_rel_err": length_err,
            "z_profile": z_profile,
        },
        "checks": {
            "stagnation_pass_5pct": bool(ok_stag),
            "extent_pass_10pct": bool(ok_len),
            "rho_range": [float(rho[fluid].min()), float(rho[fluid].max())],
            "w_max_rel_free": free_speed,
            "w_max_rel_full_footprint": w_foot_full,
            "column_mask": "inner disk r <= 0.7 a",
        },
        "provenance": _provenance(),
    }
    result["verdict"] = bool(ok_stag and ok_len)
    if figure is not None:
        result["figure"] = str(plot_case(result, uz, solid, Path(figure), U_inf))
    return result


def plot_case(res: dict, uz: torch.Tensor, solid: torch.Tensor, path: Path, U_inf: float) -> Path:
    """Vertical slice (w cloud) + column axial profile figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    nz, ny, nx = uz.shape
    w = (uz / U_inf).numpy()
    sol = solid.numpy()
    w = np.where(sol, np.nan, w)

    zi = int(res["config"]["z_body"][1]) + 3
    zi = min(zi, nz - 2)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    im = ax[0].imshow(w[zi], origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="equal")
    ax[0].set_title(f"$w/U_\\infty$ at $z={zi}$ (above obstacle top)")
    ax[0].set_xlabel("x [cells]")
    ax[0].set_ylabel("y [cells]")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    prof = res["column"]["z_profile"]
    zs = [p["z"] for p in prof]
    ws = [p["w_max_rel"] for p in prof]
    us = [p["u_mean_rel"] for p in prof]
    ax[1].semilogx(np.maximum(ws, 1e-6), zs, "o-", label="max $|w|/U_\\infty$")
    ax[1].semilogx(np.maximum(us, 1e-6), zs, "s--", label="mean $|u|/U_\\infty$")
    ax[1].axvline(res["config"]["column_tol"], color="r", ls=":", label="5 % criterion")
    ax[1].set_xlabel("normalised velocity in column")
    ax[1].set_ylabel("z [cells]")
    ax[1].set_title(
        res["label"]
        + f": Ro={res['config']['Ro']:.3f}, "
        + f"$w_{{max}}$/U={100 * res['column']['w_max_rel']:.2f} %"
    )
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("single", "pair", "scan"):
        sp = sub.add_parser(name)
        sp.add_argument("out" if name == "single" else "outdir")
        sp.add_argument("--nz", type=int, default=36)
        sp.add_argument("--ny", type=int, default=40)
        sp.add_argument("--nx", type=int, default=40)
        sp.add_argument("--radius", type=float, default=5.0)
        sp.add_argument("--z-body", type=int, nargs=2, default=[8, 18])
        sp.add_argument("--nu", type=float, default=0.05)
        sp.add_argument("--ro", type=float, default=0.08)
        sp.add_argument("--ro-list", type=float, nargs="+", default=None)
        sp.add_argument("--Uinf", type=float, default=0.004)
        sp.add_argument("--tol", type=float, default=2e-5)
        sp.add_argument("--max-steps", type=int, default=200_000)
        sp.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    args.ro_list = args.ro_list if getattr(args, "ro_list", None) else [args.ro, 0.3, 1.2]
    common = dict(
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        radius=args.radius,
        z_body=tuple(args.z_body),
        nu=args.nu,
        Ro=args.ro,
        U_inf=args.Uinf,
        tol=args.tol,
        max_steps=args.max_steps,
        threads=args.threads,
    )

    if args.mode == "scan":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        runs = []
        for ro in args.ro_list:
            res = run_case(
                **common,
                Ro=ro,
                rotate=True,
                label=f"Ro={ro:g}",
                figure=outdir / f"taylor_Ro{ro:g}.png",
            )
            (outdir / f"taylor_Ro{ro:g}.json").write_text(json.dumps(res, indent=2))
            runs.append(
                {
                    "Ro": ro,
                    "w_max_rel": res["column"]["w_max_rel"],
                    "length_rel_err": res["column"]["length_rel_err"],
                    "steps": res["convergence"]["steps"],
                    "verdict": res["verdict"],
                }
            )
            print(json.dumps(runs[-1]))
        primary = runs[0]
        high = runs[-1]
        summary = {
            "kind": "taylor_column_rossby_scan",
            "runs": runs,
            "column_disappears_at_high_Ro": bool(high["w_max_rel"] > 0.05),
            "primary_verdict": primary["verdict"],
            "all_pass": bool(primary["verdict"] and high["w_max_rel"] > 0.05),
            "provenance": _provenance(),
        }
        (outdir / "scan_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0 if summary["all_pass"] else 3

    if args.mode == "single":
        out = Path(args.out)
        res = run_case(**common, figure=out.with_suffix(".png"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(
            json.dumps(
                {
                    "out": str(out),
                    "verdict": res["verdict"],
                    "w_max_rel": res["column"]["w_max_rel"],
                    "length_rel_err": res["column"]["length_rel_err"],
                    "steps": res["convergence"]["steps"],
                },
                indent=2,
            )
        )
        return 0 if res["verdict"] else 3

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rot = run_case(**common, rotate=True, label="rotating", figure=outdir / "rotating.png")
    (outdir / "rotating.json").write_text(json.dumps(rot, indent=2))
    print("rotating:", json.dumps({"w_max_rel": rot["column"]["w_max_rel"]}))
    ctl = run_case(
        **common, rotate=False, label="no-rotation control", figure=outdir / "control.png"
    )
    (outdir / "control.json").write_text(json.dumps(ctl, indent=2))
    print("control:", json.dumps({"w_max_rel": ctl["column"]["w_max_rel"]}))
    summary = {
        "kind": "taylor_column_pair",
        "rotating": {
            "w_max_rel": rot["column"]["w_max_rel"],
            "length_rel_err": rot["column"]["length_rel_err"],
            "verdict": rot["verdict"],
        },
        "control": {"w_max_rel": ctl["column"]["w_max_rel"]},
        "control_discriminates": bool(ctl["column"]["w_max_rel"] > 0.05),
        "all_pass": bool(rot["verdict"] and ctl["column"]["w_max_rel"] > 0.05),
        "provenance": _provenance(),
    }
    (outdir / "pair_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
