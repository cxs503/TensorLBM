#!/usr/bin/env python
"""Independent (second-code-path) reproduction of the Zou-He channel case.

Why this file exists
--------------------
``REPORT.md`` reports a 0.35 % agreement between the library channel solver
(``tensorlbm.verification.zou_he_channel`` + ``tensorlbm.boundary.zou_he_channel``)
and the exact plane-Poiseuille profile.  A number produced by one code path can
always be an artefact of *that* code path -- of its lattice index table, its
streaming/pull convention, its averaging schedule or a sign convention that is
self-consistent but wrong.  This module re-solves the same physical case with a
**completely separate implementation**:

* pure NumPy, no import from ``tensorlbm`` (no shared code, no shared tables);
* textbook D2Q9 index table (``0=(0,0), 1=(1,0), 2=(0,1), 3=(-1,0), 4=(0,-1),
  5=(1,1), 6=(-1,1), 7=(-1,-1), 8=(1,-1)``) while the library uses the OSS
  permutation (``2=(-1,0), 3=(0,1), 7=(1,-1), 8=(-1,-1)``);
* the Zou-He closure re-derived here from mass + x-momentum + y-momentum
  (``_zou_he_west`` / ``_zou_he_east`` docstrings carry the algebra) instead of
  reusing the library's mirrored formulas;
* explicit Python loops over the 9 directions for streaming (the library uses a
  vectorised slicing scheme with a fluid mask).

If the two paths agree on ``u_x,max`` and on the profile to well under 1 %, the
headline number is a property of the *method* (D2Q9/BGK + half-way bounce-back +
Zou-He pressure ends), not of one implementation.

Only the ``pressure-pressure`` arm is implemented: it is the primary arm of the
cross-validation and the stricter test, because the mass flux there is
*emergent* (the library never supplies the profile).

Usage
-----
    python xval/.../independent.py --height 16 --length 128 --steps 8000 \
        --burn-in 2000 --avg-window 500 --out artifacts/xval_zouhe/indep_H16.json \
        --npz artifacts/xval_zouhe/indep_field_H16.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# --- textbook D2Q9 ---------------------------------------------------------
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array(
    [4.0 / 9.0]
    + [1.0 / 9.0] * 4
    + [1.0 / 36.0] * 4,
    dtype=np.float64,
)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium, shape (9, ny, nx)."""
    cu = C[:, 0, None, None] * ux[None] + C[:, 1, None, None] * uy[None]
    usq = (ux * ux + uy * uy)[None]
    feq = W[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq


def stream(f_post: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Pull streaming with half-way bounce-back on the two solid rows.

    Fluid rows are ``1 .. ny-2``; ``y = 0`` and ``y = ny-1`` are solid.  Source
    cells outside ``x`` (i.e. ``x < 0`` or ``x >= nx``) are *not* filled here:
    those are exactly the populations reconstructed by the Zou-He closure.

    Two windows are computed per direction, both in *destination* coordinates:

    * ``(dxs0:dxs1, dys0:dys1)`` -- destinations whose source cell is inside the
      fluid domain, i.e. ordinary pull streaming;
    * ``wy`` -- the single destination row whose y-source is the solid wall
      (``y = 1`` for ``cy = +1``, ``y = ny-2`` for ``cy = -1``), closed by
      half-way bounce-back of the opposite population in the same cell.

    Mixing the destination and source windows up is a silent, first-order error
    (it leaves whole columns unstreamed), so they are built separately here.
    """
    f = np.zeros_like(f_post)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        dy0, dy1 = 1, ny - 1                      # fluid rows (half-open)
        dx0, dx1 = (1, nx) if cx > 0 else ((0, nx - 1) if cx < 0 else (0, nx))
        # destination window whose *source* is inside the domain
        sx0, sx1 = max(0, dx0 - cx), min(nx, dx1 - cx)
        dxs0, dxs1 = sx0 + cx, sx1 + cx
        sy0, sy1 = max(1, dy0 - cy), min(ny - 1, dy1 - cy)
        dys0, dys1 = sy0 + cy, sy1 + cy
        if dxs1 > dxs0 and dys1 > dys0:
            f[i, dys0:dys1, dxs0:dxs1] = f_post[i, sy0:sy1, sx0:sx1]
        if cy != 0 and dxs1 > dxs0:
            # source row is the wall -> half-way bounce-back, same cell
            wy = 1 if cy > 0 else ny - 2
            f[i, wy, dxs0:dxs1] = f_post[OPP[i], wy, dxs0:dxs1]
    return f


def _zou_he_west(
    line: np.ndarray, rho: np.ndarray, uy: np.ndarray, ux: np.ndarray | None = None
) -> np.ndarray:
    """West (inlet) reconstruction, ``+x`` inflow.  Unknowns ``f1, f5, f8``.

    Known after streaming: ``f0, f2, f3, f4, f6, f7``.  Mass + normal momentum on
    the west layer read

        rho (1 - ux) = f0 + f2 + f4 + 2 (f3 + f6 + f7)      (mass + x-mom)
        f1 = f3 + (2/3) rho ux                              (non-eq. bounce-back)
        f5 - f8 = rho uy - (f2 + f6) + (f4 + f7)            (y-momentum)
        f5 + f8 = f6 + f7 + (1/3) rho ux                    (mass remainder)

    The first line gives ``ux`` when ``rho`` is prescribed (pressure mode) and
    ``rho`` when ``ux`` is (velocity mode); the last two give the diagonals.
    """
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = line
    mass_norm = f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)
    if ux is None:
        ux = 1.0 - mass_norm / rho
    else:
        rho = mass_norm / (1.0 - ux)
    f1 = f3 + (2.0 / 3.0) * rho * ux
    s = f6 + f7 + rho * ux / 3.0
    d = rho * uy - (f2 + f6) + (f4 + f7)
    f5 = 0.5 * (s + d)
    f8 = 0.5 * (s - d)
    line[0], line[1], line[2], line[3], line[4] = f0, f1, f2, f3, f4
    line[5], line[6], line[7], line[8] = f5, f6, f7, f8
    return line


def _zou_he_east(
    line: np.ndarray, rho: np.ndarray, uy: np.ndarray, ux: np.ndarray | None = None
) -> np.ndarray:
    """East (outlet) reconstruction, ``+x`` outflow.  Unknowns ``f3, f6, f7``.

        rho (1 + ux) = f0 + f2 + f4 + 2 (f1 + f5 + f8)
        f3 = f1 - (2/3) rho ux
        f6 - f7 = rho uy - (f2 + f5) + (f4 + f8)
        f6 + f7 = f5 + f8 - (1/3) rho ux
    """
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = line
    mass_norm = f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)
    if ux is None:
        ux = mass_norm / rho - 1.0
    else:
        rho = mass_norm / (1.0 + ux)
    f3 = f1 - (2.0 / 3.0) * rho * ux
    s = f5 + f8 - rho * ux / 3.0
    d = rho * uy - (f2 + f5) + (f4 + f8)
    f6 = 0.5 * (s + d)
    f7 = 0.5 * (s - d)
    line[0], line[1], line[2], line[3], line[4] = f0, f1, f2, f3, f4
    line[5], line[6], line[7], line[8] = f5, f6, f7, f8
    return line


def macros(f: np.ndarray):
    """Density and velocity.  Solid rows carry no populations (they are never a
    source for a fluid cell), so they are excluded from the divide instead of
    raising a divide-by-zero warning."""
    rho = f.sum(0)
    safe = np.where(rho > 0.0, rho, 1.0)
    ux = np.where(rho > 0.0, (C[:, 0, None, None] * f).sum(0) / safe, 0.0)
    uy = np.where(rho > 0.0, (C[:, 1, None, None] * f).sum(0) / safe, 0.0)
    ux[rho <= 0.0] = 0.0
    uy[rho <= 0.0] = 0.0
    return rho, ux, uy


def run(
    height: int = 16,
    length: int = 128,
    tau: float = 0.65,
    drho: float = 4.0e-3,
    steps: int = 8000,
    burn_in: int = 2000,
    avg_window: int = 500,
    ramp: int = 0,
    verbose: bool = False,
):
    ny, nx = height + 2, length
    nu = (tau - 0.5) / 3.0
    g_nom = drho / (3.0 * (nx - 1))
    y = np.arange(1, ny - 1, dtype=np.float64)
    yt = y - 0.5
    u_exact = g_nom / (2.0 * nu) * yt * (height - yt)

    # initial condition: exact parabola + linear density (library's "analytic")
    rho0 = np.ones((ny, nx))
    # linear ramp from ``1 + drho/2`` at x = 0 to ``1 - drho/2`` at x = nx-1
    rho0[1:-1, :] = 1.0 + drho * (0.5 - np.arange(nx) / (nx - 1.0))[None, :]
    ux0 = np.zeros((ny, nx))
    ux0[1:-1, :] = u_exact[:, None]
    f = equilibrium(rho0, ux0, np.zeros((ny, nx)))

    ux_sum = np.zeros((ny, nx))
    rho_sum = np.zeros((ny, nx))
    navg = 0
    q_col = None
    t0 = time.time()
    prof_prev = None
    drift_prof = []
    for step in range(1, steps + 1):
        rho, ux, uy = macros(f)
        rho[0, :] = 1.0        # solid rows: equilibrium density, populations unused
        rho[-1, :] = 1.0
        feq = equilibrium(rho, ux, uy)
        f = f - (f - feq) / tau
        f = stream(f, ny, nx)
        rampf = 1.0 if ramp <= 0 else min(1.0, step / float(ramp))
        rho_in = 1.0 + rampf * drho / 2.0
        rho_out = 1.0 - rampf * drho / 2.0
        _zou_he_west(f[:, 1:ny - 1, 0], np.full(ny - 2, rho_in), np.zeros(ny - 2))
        _zou_he_east(f[:, 1:ny - 1, -1], np.full(ny - 2, rho_out), np.zeros(ny - 2))
        if step > burn_in:
            ux_sum += ux
            rho_sum += rho
            navg += 1
            if navg % avg_window == 0:
                mean = ux_sum / navg
                p = mean[1:-1, :].mean(axis=1)
                if prof_prev is not None:
                    drift_prof.append(
                        {
                            "step": step,
                            "drift": float(
                                np.abs(p - prof_prev).max() / max(np.abs(p).max(), 1e-30)
                            ),
                        }
                    )
                prof_prev = p.copy()
        if step % max(1, steps // 8) == 0 or step == steps:
            q_col = (f[:, 1:-1, :] * C[:, 0, None, None]).sum(axis=(0, 1))
            if verbose:
                print(
                    f"  step {step:6d} q(x) spread "
                    f"{(q_col[1:-1].max() - q_col[1:-1].min()) / max(abs(q_col[1:-1].mean()), 1e-30):.3e}"
                )
    wall = time.time() - t0

    rho, ux, uy = macros(f)
    if navg:
        ux_avg = ux_sum / navg
        rho_avg = rho_sum / navg
    else:
        ux_avg, rho_avg = ux, rho
    q_col = (f[:, 1:-1, :] * C[:, 0, None, None]).sum(axis=(0, 1))

    # gradient from the computed mean density (least squares in x)
    rho_bar = rho_avg[1:-1, :].mean(axis=0)
    x = np.arange(nx, dtype=np.float64)
    slope = ((x - x.mean()) * (rho_bar - rho_bar.mean())).sum() / (
        (x - x.mean()) ** 2
    ).sum()
    g_meas = -slope / 3.0

    prof = ux_avg[1:-1, nx // 2]
    metrics = {}
    for tag, g in (("measured", g_meas), ("nominal", g_nom)):
        u_ref = g / (2.0 * nu) * yt * (height - yt)
        scale = np.abs(u_ref).max()
        diff = prof - u_ref
        metrics[tag] = {
            "u_max_sim": float(np.abs(prof).max()),
            "u_max_ref": float(scale),
            "rel_umax": float(abs(np.abs(prof).max() - scale) / scale),
            "rel_linf": float(np.abs(diff).max() / scale),
            "rel_l2": float(np.sqrt((diff ** 2).mean()) / scale),
            "shape_rel_l2": float(
                np.sqrt(((prof / np.abs(prof).max() - u_ref / scale) ** 2).mean())
            ),
        }

    return {
        "solver": "independent-numpy",
        "arm": "pressure-pressure",
        "height": height,
        "length": nx,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "drho": drho,
        "ma": float(np.abs(u_exact).max() * np.sqrt(3.0)),
        "steps": steps,
        "burn_in": burn_in,
        "avg_window": avg_window,
        "navg": navg,
        "wall_seconds": wall,
        "g_measured": float(g_meas),
        "g_nominal": float(g_nom),
        "g_rel_error": float(abs(g_meas - g_nom) / g_nom),
        "mass_defect": float(
            abs(q_col[0] - q_col[-1]) / max(abs(q_col[1:-1].mean()), 1e-30)
        ),
        "q_interior_spread": float(
            (q_col[1:-1].max() - q_col[1:-1].min()) / max(abs(q_col[1:-1].mean()), 1e-30)
        ),
        "metrics_vs_measured_gradient": metrics["measured"],
        "metrics_vs_nominal_gradient": metrics["nominal"],
        "profile_mid": prof.tolist(),
        "profile_analytic_measured_g": (
            g_meas / (2.0 * nu) * yt * (height - yt)
        ).tolist(),
        "drift_profile": drift_prof,
        "y_fluid": list(range(1, ny - 1)),
        "fields": {"ux": ux_avg, "rho": rho_avg},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--tau", type=float, default=0.65)
    p.add_argument("--drho", type=float, default=4.0e-3)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--burn-in", type=int, default=2000)
    p.add_argument("--avg-window", type=int, default=500)
    p.add_argument("--ramp", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--npz", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    res = run(
        height=a.height,
        length=a.length,
        tau=a.tau,
        drho=a.drho,
        steps=a.steps,
        burn_in=a.burn_in,
        avg_window=a.avg_window,
        ramp=a.ramp,
        verbose=True,
    )
    fields = res.pop("fields")
    m = res["metrics_vs_measured_gradient"]
    print(
        f"[independent H={a.height} nx={a.length} tau={a.tau} drho={a.drho}] "
        f"rel_umax={m['rel_umax']:.3e} rel_linf={m['rel_linf']:.3e} "
        f"shape={m['shape_rel_l2']:.3e} mass_defect={res['mass_defect']:.2e} "
        f"wall={res['wall_seconds']:.1f}s"
    )
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    if a.npz:
        os.makedirs(os.path.dirname(os.path.abspath(a.npz)), exist_ok=True)
        np.savez_compressed(
            a.npz,
            ux=fields["ux"],
            rho=fields["rho"],
            y=np.arange(res["ny"]),
            x=np.arange(res["length"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())