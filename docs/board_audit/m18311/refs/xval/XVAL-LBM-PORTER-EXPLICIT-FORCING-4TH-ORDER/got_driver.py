"""TensorLBM arm (uses the shipped ``tensorlbm.porter_forcing`` API) + control arms.

Two forcing schemes are provided on the same lattice, walls and initial field:

``porter``  the adopted four-piece explicit-forcing set (A1 literal / A2 sequential
            reading / guo), implemented through the library primitives;
``legacy``  control: the pre-Porter "original Shan-Chen" recipe (force folded into a
            per-component equilibrium velocity ``u_eq_i = u + tau_i F_i/rho_i``, no
            force PDF, no half force in the recovered velocity).

Also provides the analytic reference solutions used as the benchmark anchor.
"""

from __future__ import annotations

import numpy as np

from tensorlbm.porter_forcing import (
    equilibrium,
    momentum,
    physical_velocity,
    porter_collide,
    porter_force_pdf,
    stream_halfway_bb,
    viscosity_weighted_ueq,
)


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------


def layered_stokes(y, H, yi, nu_a, nu_b, g):
    """Steady two-layer Stokes profile in a channel ``y in [0, H]`` (force density ``g``).

    Layer A occupies ``y < yi`` with kinematic viscosity ``nu_a``, layer B
    ``y > yi`` with ``nu_b``; no slip at both walls, continuity of velocity and
    of the shear stress at the interface.  Solved as a 2x2 linear system (no
    hand-copied formula).
    """
    y = np.asarray(y, float)
    a = np.array([[yi, H - yi], [nu_a, -nu_b]])
    rhs = np.array([(g / (2 * nu_a) - g / (2 * nu_b)) * yi**2 + (g / (2 * nu_b)) * H**2, 0.0])
    c1, c3 = np.linalg.solve(a, rhs)
    c4 = (g / (2 * nu_b)) * H**2 - c3 * H
    u_a = -(g / (2 * nu_a)) * y**2 + c1 * y
    u_b = -(g / (2 * nu_b)) * y**2 + c3 * y + c4
    return np.where(y <= yi, u_a, u_b)


def interior_steps(ny: int) -> int:
    """Number of steps needed for the profile to settle (heuristic, recorded per run)."""
    return 200 * ny


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def run_channel(
    ny: int,
    taus: list[float],
    rho_init: list[np.ndarray],
    g: np.ndarray,
    steps: int,
    scheme: str = "porter",
    variant: str = "doc_literal",
    nx: int = 1,
    atol: float = 1e-8,
    check_every: int = 500,
    ux0=None,
):
    """Run a forced channel flow to (near) steady state.

    Args:
        ny: rows, rows 0 and ny-1 are solid (half-way bounce-back walls).
        taus: one relaxation time per component.
        rho_init: per-component initial density fields (ny, nx).
        g: per-component body acceleration (len == number of components).
        scheme: ``porter`` | ``legacy``.
        variant: ``doc_literal`` | ``doc_seq`` | ``guo`` (porter only).

    Returns:
        dict with ``ux`` profile (per fluid row), ``rho_tot``, ``steps``,
        ``mass_drift``, ``unstable`` flag and the convergence history.
    """
    ncomp = len(taus)
    rho = [np.array(r, float).reshape(ny, nx) for r in rho_init]
    solid = np.zeros((ny, nx), bool)
    solid[0, :] = True
    solid[-1, :] = True
    for r in rho:  # a solid row holds no fluid: drop it from the initial state and the mass ledger
        r[0, :] = 0.0
        r[-1, :] = 0.0
    zero = np.zeros((ny, nx))
    if ux0 is None:
        fs = [equilibrium(rho[s], zero, zero) for s in range(ncomp)]
    else:
        ux_field = np.asarray(ux0, float).reshape(ny, nx)
        fs = [equilibrium(rho[s], ux_field, zero) for s in range(ncomp)]
    mass0 = float(sum(r.sum() for r in rho))

    history: list[tuple[int, float]] = []
    prev = None
    n_done = steps
    unstable = False
    for t in range(1, steps + 1):
        jxs, jys = [], []
        for s in range(ncomp):
            jx, jy = momentum(fs[s])
            jxs.append(jx)
            jys.append(jy)
        fx = [g[s] * rho[s] for s in range(ncomp)]
        fy = [zero for _ in range(ncomp)]
        rho_tot = sum(rho)
        if scheme == "porter":
            ux, uy = physical_velocity(sum(jxs), sum(jys), sum(fx), sum(fy), rho_tot)
            ueqx, ueqy = viscosity_weighted_ueq(jxs, jys, fx, fy, rho, taus)
            feqs = [equilibrium(rho[s], ueqx, ueqy) for s in range(ncomp)]
            fpdfs = [porter_force_pdf(fx[s], fy[s], feqs[s], rho[s], ueqx, ueqy) for s in range(ncomp)]
            for s in range(ncomp):
                fs[s] = porter_collide(fs[s], feqs[s], fpdfs[s], taus[s], variant=variant)
        elif scheme == "legacy":
            ux, uy = sum(jxs) / np.maximum(rho_tot, 1e-30), sum(jys) / np.maximum(rho_tot, 1e-30)
            for s in range(ncomp):
                ueqx = ux + taus[s] * fx[s] / np.maximum(rho[s], 1e-12)
                ueqy = uy + taus[s] * fy[s] / np.maximum(rho[s], 1e-12)
                feq = equilibrium(rho[s], ueqx, ueqy)
                fs[s] = fs[s] + (feq - fs[s]) / taus[s]
        else:
            raise ValueError(scheme)
        fs = [stream_halfway_bb(fs[s], solid) for s in range(ncomp)]
        rho = [fs[s].sum(axis=0) for s in range(ncomp)]

        if not np.isfinite(float(np.max(np.abs(ux)))):
            unstable = True
            n_done = t
            break
        if t % check_every == 0:
            amp = max(float(np.max(np.abs(ux))), 1e-30)
            if prev is not None:
                delta = float(np.max(np.abs(ux - prev))) / amp
                history.append((t, delta))
                if delta < atol:
                    n_done = t
                    break
            prev = ux.copy()
    else:
        n_done = steps

    rho_tot = sum(rho)
    return {
        "ux": ux[:, 0].copy() if np.isfinite(ux).all() else np.full(ny, np.nan),
        "uy_max": float(np.max(np.abs(uy))) if np.isfinite(uy).all() else float("nan"),
        "rho_tot": rho_tot[:, 0].copy(),
        "steps": n_done,
        "mass_drift": float(abs(rho_tot.sum() - mass0) / mass0),
        "unstable": unstable,
        "history": history,
    }


def layers(ny: int, nx: int, width: float = 1.5, residual: float = 1e-3):
    """Initial density fields: component 0 below the mid-plane, component 1 above."""
    h = ny - 2
    yi = h / 2.0
    y = (np.arange(ny)[:, None] - 0.5) * np.ones((1, nx))  # wall distance
    phi = 0.5 * (1.0 - np.tanh((y - yi) / width))
    r0 = residual + (1.0 - residual) * phi
    r1 = residual + (1.0 - residual) * (1.0 - phi)
    return [r0, r1], h, yi