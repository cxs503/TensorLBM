"""External reference (OSS) side of XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64.

Reference implementation: **pylbm** (https://github.com/pylbm/pylbm), read-only
checkout at ``/root/oss-cache/pylbm`` (override with ``$PYLBM_ROOT``), commit
``f9f5d16``.  Everything about "the OSS way of doing a half-way moving-wall
bounce-back" below is taken from that source, not from TensorLBM:

| OSS recipe item | pylbm source |
|---|---|
| half-way wall == first fluid node at distance ``delta=0.5`` from the surface | ``pylbm/boundary.py:46`` (``self.distance = domain.distance[...]``), ``pylbm/geometry.py`` (distance annotation), ``tests/test_bouzidi.py:30-31`` (``y0 = 0.5 - delta``) |
| moving wall enters as a *wall-moment* function ``value(f, m, x, y)`` that sets ``m[rho], m[qx], m[qy]`` | ``pylbm/boundary.py:84-90`` (docstring), ``tests/test_linear_bounce_back.py:79-82`` |
| the wall equilibrium is evaluated at the wall point ``x_w = x_b + (1-delta) c_k`` | ``pylbm/boundary.py:358-433`` (``prepare_rhs``: ``s = 1 - distance``, ``x += s*v[k]*dx``) |
| bounce-back writes ``f[istore] = f[iload] + rhs`` (opposite population plus wall term) | ``pylbm/boundary.py:562-600`` (``BounceBack.generate``) |
| the wall term is the *antisymmetric equilibrium difference* at ``x_w``: ``rhs = feq_k(x_w) - feq_ksym(x_w)`` | ``pylbm/boundary.py:551-558`` (``BounceBack.set_rhs``) |
| generic-distance variants (Bouzidi/BFL/CLI/YLI/ParBFL) with the same wall term and alpha(delta) | ``pylbm/boundary.py:920-1190`` |
| scheme: d2Q9, moments ``(rho, qx, qy, ...)``, relaxation ``1/tau`` on all modes | ``demo/2D/Poiseuille.py:40-75`` |

For ``delta = 0.5`` and a d2Q9 equilibrium ``feq_q(x_w) = w_q rho_w (1 + 3 c_q.u_w + 4.5 (c_q.u_w)^2 - 1.5 u_w^2)``
the antisymmetric difference collapses to the textbook Ladd term

    rhs = feq_k(x_w) - feq_ksym(x_w) = 6 w_q rho_w (c_q . u_w)      (c_ksym = -c_q)

which is the term TensorLBM adds on the wall cell (with the *local* wall-cell
density instead of the imposed ``rho_w``).  That algebraic identity is checked
numerically here against the *live* ``bc.rhs`` array of a pylbm simulation
(``wall_term_from_source``), so the comparison never relies on a hand-typed
formula.

The NumPy backend of pylbm performs its periodic halo exchange through MPI even
on one rank, so this module plugs in a documented 1-rank serial stand-in for
``mpi4py`` (``_shim/mpi4py``) instead of requiring an MPI installation; the
shim implements self-send/self-recv as the subarray copy MPI would perform.
Its validity is checked by two independent facts reported with the results: the
Couette solution is exactly linear on the lattice, and the solution is
invariant to the periodic x extent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PYLBM_ROOT = Path(os.environ.get("PYLBM_ROOT", "/root/oss-cache/pylbm"))

if str(HERE / "_shim") not in sys.path:
    sys.path.insert(0, str(HERE / "_shim"))
if str(PYLBM_ROOT) not in sys.path:
    sys.path.insert(0, str(PYLBM_ROOT))

import sympy as sp  # noqa: E402

import pylbm  # noqa: E402

X, Y = sp.symbols("X, Y")
RHO, QX, QY = sp.symbols("rho, qx, qy")

# d2Q9, standard numbering (0 rest, 1..4 axis, 5..8 diagonals): the same
# velocity ordering pylbm's `list(range(9))` resolves to.
VELOCITIES = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
WEIGHTS = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def oss_commit() -> str:
    """Commit hash of the OSS checkout (provenance for the report)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PYLBM_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance only
        return f"unavailable ({exc})"


def _dico(N: int, tau: float, u_bottom: float, u_top: float, nx: int) -> dict:
    """pylbm input dictionary: Couette channel, periodic in x.

    ``N`` fluid rows at y = 0.5 ... N-0.5; the two walls sit exactly
    ``delta = 0.5`` outside the first/last fluid row, i.e. at y = 0 and y = N
    (``pylbm/boundary.py:46`` reads those distances from the geometry).
    """
    s = [0.0, 0.0, 0.0] + [1.0 / tau] * 6
    y0, y1 = 0.0, float(N)  # wall surfaces
    r2 = X**2 + Y**2

    def bottom(f, m, x, y):
        m[RHO] = 1.0
        m[QX] = u_bottom
        m[QY] = 0.0

    def top(f, m, x, y):
        m[RHO] = 1.0
        m[QX] = u_top
        m[QY] = 0.0

    geometry = {
        "box": {"x": [0.0, float(nx)], "y": [-1.0, N + 1.0], "label": [-1, -1, 0, 1]},
        "elements": [
            pylbm.Parallelogram((-1.0, y0 - 2.0), (float(nx) + 2.0, 0.0), (0.0, 2.0), label=0),
            pylbm.Parallelogram((-1.0, y1), (float(nx) + 2.0, 0.0), (0.0, 2.0), label=1),
        ],
        "space_step": 1.0,
        "scheme_velocity": 1.0,
    }
    return {
        **geometry,
        "schemes": [
            {
                "velocities": list(range(9)),
                "conserved_moments": [RHO, QX, QY],
                "polynomials": [
                    1,
                    X,
                    Y,
                    3 * r2 - 4,
                    (9 * r2**2 - 21 * r2 + 8) / 2,
                    3 * X * r2 - 5 * X,
                    3 * Y * r2 - 5 * Y,
                    X**2 - Y**2,
                    X * Y,
                ],
                "relaxation_parameters": s,
                "equilibrium": [
                    RHO,
                    QX,
                    QY,
                    -2 * RHO + 3 * (QX**2 + QY**2),
                    RHO - 3 * (QX**2 + QY**2),
                    -QX,
                    -QY,
                    QX**2 - QY**2,
                    QX * QY,
                ],
            }
        ],
        "init": {RHO: 1.0, QX: 0.0, QY: 0.0},
        "boundary_conditions": {
            0: {"method": {0: pylbm.bc.BounceBack}, "value": bottom},
            1: {"method": {0: pylbm.bc.BounceBack}, "value": top},
        },
        "generator": "numpy",
    }


def wall_term_from_source(N: int, tau: float, u_wall: float, nx: int = 2) -> dict:
    """Read the wall term the OSS code actually builds, straight from its BC object.

    Returns the per-link quantities of the bottom (moving) wall, which is the
    wall this comparison is about:
      ``rhs``      the ``feq_k - feq_ksym`` term pylbm adds (``boundary.py:551-558``)
      ``dirs``     lattice directions ``k`` of the written populations
      ``distance`` distances of those links to the wall surface
    """
    # initialize=True so that pylbm itself calls prepare_rhs() and fills the
    # wall term (``boundary.py:358-433``); the arrays below are read from that
    # state, never retyped.
    sol = pylbm.Simulation(_dico(N, tau, u_wall, 0.0, nx), initialize=True)
    method = sol.bc.methods[0]
    istore = np.asarray(method.istore)
    if istore.shape[1] != 3:  # fix_iload() stores (ncond, dim+1)
        istore = istore.T
    labels = np.asarray(method.ilabel)
    dirs = istore[:, 0][labels == 0]
    return {
        "rhs": np.asarray(method.rhs)[labels == 0].astype(np.float64),
        "feq": np.asarray(method.feq)[:, labels == 0].astype(np.float64),
        "dirs": dirs.astype(np.int64),
        "distance": np.asarray(method.distance)[labels == 0].astype(np.float64),
        "u_wall": float(u_wall),
        "tau": float(tau),
        "n_links": int(np.sum(labels == 0)),
    }


def couette(N: int, tau: float, u_wall: float, nx: int = 2, block: int = 500,
            tol: float = 1e-12, max_steps: int = 200_000) -> dict:
    """Steady 2D Couette in pylbm: bottom wall moves at ``u_wall``, top at rest.

    Returns the velocity profile of the ``N`` fluid rows, the analytic profile
    and the convergence record.  ``u = qx/rho`` with ``rho -> 1`` (the
    ``m[qx]`` moment of pylbm is the momentum; at Ma = U_w/sqrt(3) <= 0.03 the
    density correction is ~Ma^2 and is reported separately).
    """
    sol = pylbm.Simulation(_dico(N, tau, u_wall, 0.0, nx))
    u = np.array(sol.m[QX])[0, 1:N + 1]
    steps = 0
    prev = u.copy()
    converged = False
    while steps < max_steps:
        for _ in range(block):
            sol.one_time_step()
        steps += block
        u = np.array(sol.m[QX])[0, 1:N + 1]
        rho = np.array(sol.m[RHO])[0, 1:N + 1]
        delta = np.max(np.abs(u - prev)) / u_wall
        prev = u.copy()
        if delta < tol:
            converged = True
            break
    y = np.arange(N) + 0.5
    rho_now = np.array(sol.m[RHO])[0, 1:N + 1]
    return {
        "N": int(N),
        "tau": float(tau),
        "u_wall": float(u_wall),
        "nu": (tau - 0.5) / 3.0,
        "re_h": float(u_wall * N / ((tau - 0.5) / 3.0)),
        "steps": int(steps),
        "converged": bool(converged),
        "tol": float(tol),
        "block": int(block),
        "y": y.tolist(),
        "u": u.astype(float).tolist(),
        "u_ana": (u_wall * (1.0 - y / N)).tolist(),
        "rho": rho_now.astype(float).tolist(),
        "max_drho": float(np.max(np.abs(rho_now - 1.0))),
        "mass_per_column": float(np.sum(rho_now)),
        "impl": f"pylbm {pylbm.__version__} @ {oss_commit()} (BounceBack, delta=0.5, numpy generator)",
    }


if __name__ == "__main__":
    import json

    out = {
        "oss": {"pylbm": pylbm.__version__, "commit": oss_commit(), "root": str(PYLBM_ROOT)},
        "wall_term": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in wall_term_from_source(16, 0.8, 0.05).items()},
        "couette_N16_tau0.8_Uw0.05": couette(16, 0.8, 0.05, max_steps=40_000),
    }
    print(json.dumps(out, indent=2)[:4000])