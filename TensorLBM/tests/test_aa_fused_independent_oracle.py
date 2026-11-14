"""Independent (from-scratch, NumPy, scatter-form) cross-check of the fused kernels.

Anchor: ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``.

Why a second reference is needed
--------------------------------
The existing equivalence evidence (`tests/test_aa_fused_push_pull.py`,
"fused == push, bit identical") compares two functions that share the same
helpers (``_collision_increment``, ``_pull``) and the same ``torch.roll``
shift convention.  That is a useful regression pin, but it is not an
independent implementation: a systematic error in the shared helpers -- wrong
velocity ordering, sign of a shifted direction, an off-by-one in the wrap --
would cancel exactly and still "pass".

This module therefore implements the same scheme a *second* time, from the
definitions only:

* NumPy float64, explicit ``for`` loops over nodes and directions (no vectorised
  shift operator, no ``roll``);
* **scatter** form (``f_i(x + c_i, t+1) <- f_i^post(x, t)``) instead of the
  pull form used by both TensorLBM kernels, so the streaming permutation is
  traversed in the opposite direction;
* halfway bounce-back written as "target = opposite of the outgoing direction",
  with the wall velocity sampled from the solid node;
* Guo forcing evaluated per direction as a scalar.

Agreement between the scatter reference and the pull kernels is therefore
evidence about the *scheme*, not about a shared code path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import tensorlbm.aa_fused_push_pull as AB
import tensorlbm.aa_fused_push_pull_inplace as AA

CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
W9 = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
      1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)

SCATTER_GATE = 1e-13


def scatter_step(f, tau, solid=None, wall_u=None, force=None):
    """One scatter-form D2Q9 BGK step on a periodic-in-x, walled-in-y domain."""
    omega = 1.0 / tau
    ny, nx = f.shape[1], f.shape[2]

    rho = np.zeros((ny, nx))
    mx = np.zeros((ny, nx))
    my = np.zeros((ny, nx))
    for i in range(9):
        rho += f[i]
        mx += CX[i] * f[i]
        my += CY[i] * f[i]
    if force is not None:
        mx = mx + 0.5 * force[0]
        my = my + 0.5 * force[1]
    ux, uy = mx / rho, my / rho

    post = np.zeros_like(f)
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        u2 = ux * ux + uy * uy
        feq = W9[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        post[i] = f[i] + omega * (feq - f[i])
        if force is not None:
            cF = CX[i] * force[0] + CY[i] * force[1]
            uF = ux * force[0] + uy * force[1]
            S = W9[i] * (3.0 * (cF - uF) + 9.0 * cu * cF)
            post[i] = post[i] + (1.0 - 0.5 * omega) * S

    # scatter: the population in direction i leaving (y, x) lands at (y+cy, x+cx)
    fnew = np.zeros_like(f)
    for y in range(ny):
        for x in range(nx):
            for i in range(9):
                yy = (y + CY[i]) % ny
                xx = (x + CX[i]) % nx
                fnew[i, yy, xx] = post[i, y, x]

    # halfway bounce-back: fluid node whose neighbour in direction i is solid
    # receives, in direction opp(i), the reflection of what left towards the wall
    if solid is not None:
        for y in range(ny):
            for x in range(nx):
                if solid[y, x]:
                    continue
                for i in range(1, 9):
                    yy = (y + CY[i]) % ny
                    xx = (x + CX[i]) % nx
                    if not solid[yy, xx]:
                        continue
                    o = OPP[i]
                    val = post[i, y, x]
                    if wall_u is not None:
                        val += 6.0 * W9[o] * rho[y, x] * (
                            CX[o] * wall_u[0, yy, xx] + CY[o] * wall_u[1, yy, xx])
                    fnew[o, y, x] = val
    return fnew


def _numpy_rest(ny, nx, perturb=False):
    rho = np.ones((ny, nx))
    if perturb:
        rho[ny // 2, nx // 3] += 0.02
    f = np.zeros((9, ny, nx))
    for i in range(9):
        f[i] = W9[i] * rho
    return f


def _to_numpy(f):
    return f.detach().cpu().numpy().copy()


def _compare(f_np, f_torch, fluid, n_steps):
    a = f_np[:, fluid]
    b = _to_numpy(f_torch)[:, fluid]
    return float(np.abs(a - b).max())


@pytest.mark.parametrize("steps", [1, 30])
def test_periodic_matches_scatter_oracle(steps):
    ny, nx, tau = 6, 5, 0.9
    f_np = _numpy_rest(ny, nx, perturb=True)
    f_ab = torch.tensor(f_np.copy())
    f_aa = f_ab.clone()
    fluid = np.ones((ny, nx), dtype=bool)
    for _ in range(steps):
        f_np = scatter_step(f_np, tau)
        f_ab = AB.fused_push_pull_step(f_ab, tau)
        AA.aa_fused_push_pull_step(f_aa, tau)
    assert _compare(f_np, f_ab, fluid, steps) < SCATTER_GATE
    assert _compare(f_np, f_aa, fluid, steps) < SCATTER_GATE


def test_walled_channel_matches_scatter_oracle():
    ny, nx, tau, steps = 8, 5, 0.9, 30
    solid_np = np.zeros((ny, nx), dtype=bool)
    solid_np[0, :] = solid_np[-1, :] = True
    solid = torch.tensor(solid_np)
    fluid = ~solid_np

    f_np = _numpy_rest(ny, nx, perturb=True)
    f_aa = torch.tensor(f_np.copy())
    for _ in range(steps):
        f_np = scatter_step(f_np, tau, solid=solid_np)
        AA.aa_fused_push_pull_step(f_aa, tau, solid=solid)
    assert _compare(f_np, f_aa, fluid, steps) < SCATTER_GATE


def test_moving_wall_matches_scatter_oracle():
    """The moving-wall sign is only pinned by a scatter reference written independently."""
    ny, nx, tau, steps, u_wall = 8, 5, 0.9, 30, 0.02
    solid_np = np.zeros((ny, nx), dtype=bool)
    solid_np[0, :] = solid_np[-1, :] = True
    wall_np = np.zeros((2, ny, nx))
    wall_np[0, -1, :] = u_wall
    solid = torch.tensor(solid_np)
    wall = torch.tensor(wall_np)
    fluid = ~solid_np

    f_np = _numpy_rest(ny, nx)
    f_aa = torch.tensor(f_np.copy())
    for _ in range(steps):
        f_np = scatter_step(f_np, tau, solid=solid_np, wall_u=wall_np)
        AA.aa_fused_push_pull_step(f_aa, tau, solid=solid, wall_u=wall)
    assert _compare(f_np, f_aa, fluid, steps) < SCATTER_GATE
    # and the wall must drive the fluid in +x, not -x (sign check)
    _, ux, _ = AA.macroscopic(f_aa)
    assert float(ux[~solid].mean()) > 0.0


def test_guo_forced_matches_scatter_oracle():
    ny, nx, nu, steps = 8, 5, 0.05, 30
    tau = 3.0 * nu + 0.5
    solid_np = np.zeros((ny, nx), dtype=bool)
    solid_np[0, :] = solid_np[-1, :] = True
    force_np = np.zeros((2, ny, nx))
    force_np[0] = 1e-6
    solid = torch.tensor(solid_np)
    force = torch.tensor(force_np)
    fluid = ~solid_np

    f_np = _numpy_rest(ny, nx)
    f_aa = torch.tensor(f_np.copy())
    for _ in range(steps):
        f_np = scatter_step(f_np, tau, solid=solid_np, force=force_np)
        AA.aa_fused_push_pull_step(f_aa, tau, solid=solid, force=force)
    assert _compare(f_np, f_aa, fluid, steps) < SCATTER_GATE


def test_scatter_oracle_is_a_permutation():
    """Sanity of the reference itself: a collisionless period is mass conserving."""
    ny, nx = 5, 4
    f = _numpy_rest(ny, nx, perturb=True)
    m0 = f.sum()
    for _ in range(10):
        f = scatter_step(f, 1e6)  # omega -> 0, streaming only
    assert abs(f.sum() - m0) / m0 < 1e-14