"""Reference transcription of joconnor22/LIFE ``GridClass::regularisedBC`` (D2Q9).

Provenance (zero manual copying of numbers): ``oss_src/Grid.cpp`` in this
directory is a byte-for-byte copy of upstream ``src/Grid.cpp``
(md5 e6e4a647faa9606f9982a98303f6e54d).  Every step below is annotated with the
upstream line range it transcribes:

* L386-470  ``regularisedBC``            -> :func:`life_regularised_bc`
* L268-384  ``applyBCs`` (type dispatch) -> kind dispatch in the same function
* L504-540  ``getNormalVector``          -> caller supplied ``normal`` (axis+sign)
* L249-266  ``equilibrium`` (BGK branch) -> :func:`equilibrium`
* Utils.h L137-161 ``extrapolate(...,1)`` / L186-210 ``zeroGradient(...,2)``
* L52-105   ``lbmKernel`` ordering       -> driver: collide+stream, then BC, then macro

The transcription is deliberately dumb: numpy, one node at a time, in-place
updates in BC-vector order (ascending flat id), exactly like the upstream
serial reading order.  It is the *reference arm*, not production code.
"""

from __future__ import annotations

import numpy as np

# D2Q9 in TensorLBM's C2Q9 ordering (identical to LIFE's 9-velocity set)
C = np.array(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]],
    dtype=np.int64,
)
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
OPP = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=np.int64)
CS2 = 1.0 / 3.0

FLUID, WALL, VELOCITY, PRESSURE, FREE_SLIP = 0, 1, 2, 3, 4


def equilibrium(rho: float, ux: float, uy: float) -> np.ndarray:
    """Grid.cpp L249-266, BGK (2nd-order Hermite) branch."""
    feq = np.zeros(9)
    for v in range(9):
        cx, cy = C[v]
        feq[v] = rho * W[v] * (
            1.0
            + 3.0 * (cx * ux + cy * uy)
            + 4.5 * (ux * ux * (cx * cx - 1.0 / 3.0) + uy * uy * (cy * cy - 1.0 / 3.0))
            + 9.0 * cx * cy * ux * uy
        )
    return feq


def _clamp(i: int, n: int) -> int:
    return 0 if i < 0 else (n - 1 if i >= n else i)


def extrapolate(vec: np.ndarray, normal: np.ndarray, i: int, j: int) -> float:
    """Utils.h L151-161, first-order extrapolation (used for corner density)."""
    ny, nx = vec.shape
    i1 = _clamp(i + normal[0], ny)
    j1 = _clamp(j + normal[1], nx)
    i2 = _clamp(i + 2 * normal[0], ny)
    j2 = _clamp(j + 2 * normal[1], nx)
    return 2.0 * vec[i1, j1] - vec[i2, j2]


def zero_gradient(vec: np.ndarray, normal: np.ndarray, i: int, j: int) -> float:
    """Utils.h L200-210, second-order zero-gradient (tangential velocity)."""
    ny, nx = vec.shape
    i1 = _clamp(i + normal[0], ny)
    j1 = _clamp(j + normal[1], nx)
    i2 = _clamp(i + 2 * normal[0], ny)
    j2 = _clamp(j + 2 * normal[1], nx)
    return (4.0 / 3.0) * vec[i1, j1] - (1.0 / 3.0) * vec[i2, j2]


def life_regularised_bc(
    f: np.ndarray,
    kind: np.ndarray,
    normal: np.ndarray,
    u_bc: np.ndarray | None = None,
    rho_bc: np.ndarray | None = None,
    rho: np.ndarray | None = None,
    u: np.ndarray | None = None,
):
    """Apply LIFE's regularised BC at every non-fluid node.

    Parameters
    ----------
    f : (9, ny, nx) populations *after* collide+stream (pre-BC state).
    kind : (ny, nx) int, FLUID/WALL/VELOCITY/PRESSURE/FREE_SLIP.
    normal : (ny, nx, 2) int, outward lattice normal (0 or +-1 per axis).
    u_bc : (ny, nx, 2) prescribed velocity for VELOCITY nodes.
    rho_bc : (ny, nx) prescribed density for PRESSURE nodes.
    rho, u : (ny, nx) / (ny, nx, 2) macroscopic fields, updated in place copies.

    Returns
    -------
    f_out, rho_out, u_out, diagnostics
    """
    f = np.array(f, dtype=np.float64, copy=True)
    ny, nx = f.shape[1], f.shape[2]
    if rho is None or u is None:
        rho = f.sum(axis=0)
        u = np.stack(
            [(f * C[:, 0][:, None, None]).sum(axis=0), (f * C[:, 1][:, None, None]).sum(axis=0)],
            axis=-1,
        ) / rho[..., None]
    else:
        rho = np.array(rho, dtype=np.float64, copy=True)
        u = np.array(u, dtype=np.float64, copy=True)
    if u_bc is None:
        u_bc = np.zeros((ny, nx, 2))
    if rho_bc is None:
        rho_bc = np.ones((ny, nx))

    corner_nodes, edge_nodes = [], []
    for i in range(ny):
        for j in range(nx):
            t = int(kind[i, j])
            if t == FLUID:
                continue
            n = np.asarray(normal[i, j], dtype=np.int64)
            nd = 0 if n[0] != 0 else 1  # getNormalVector: normalDirection
            is_corner = bool(n[0] != 0 and n[1] != 0)
            (corner_nodes if is_corner else edge_nodes).append((i, j))

            # --- applyBCs type dispatch (L268-384) --------------------------
            if t == WALL:
                u[i, j, :] = 0.0
            elif t == VELOCITY:
                u[i, j, :] = u_bc[i, j]
            elif t == FREE_SLIP:
                u[i, j, nd] = 0.0
                for d in range(2):
                    if d != nd:
                        u[i, j, d] = zero_gradient(u[..., d], n, i, j)
            elif t == PRESSURE:
                rho[i, j] = rho_bc[i, j]
                for d in range(2):
                    if d != nd:
                        u[i, j, d] = zero_gradient(u[..., d], n, i, j)

            # --- regularisedBC (L386-470) ---------------------------------
            if is_corner:
                if t in (VELOCITY, WALL, FREE_SLIP):
                    rho[i, j] = extrapolate(rho, n, i, j)
            else:
                fplus = 0.0
                fzero = 0.0
                for v in range(9):
                    if C[v, nd] == -n[nd]:
                        fplus += f[v, i, j]
                    elif C[v, nd] == 0:
                        fzero += f[v, i, j]
                if t in (VELOCITY, WALL, FREE_SLIP):
                    rho[i, j] = (2.0 * fplus + fzero) / (1.0 - n[nd] * u[i, j, nd])
                elif t == PRESSURE:
                    u[i, j, nd] = n[nd] * (1.0 - (2.0 * fplus + fzero) / rho[i, j])

            feq = equilibrium(rho[i, j], u[i, j, 0], u[i, j, 1])
            Sxx = Syy = Sxy = 0.0
            for v in range(9):
                cx, cy = C[v]
                if is_corner:
                    if cx == n[0] or cy == n[1]:
                        if n[0] * cx + n[1] * cy == 0:
                            f[v, i, j] = feq[v]  # buried link
                        else:
                            f[v, i, j] = feq[v] + (f[OPP[v], i, j] - feq[OPP[v]])
                else:
                    if C[v, nd] == n[nd]:
                        f[v, i, j] = feq[v] + (f[OPP[v], i, j] - feq[OPP[v]])
                fneq = f[v, i, j] - feq[v]
                Sxx += cx * cx * fneq
                Syy += cy * cy * fneq
                Sxy += cx * cy * fneq

            # --- regularised stress reconstruction, all populations --------
            for v in range(9):
                cx, cy = C[v]
                f[v, i, j] = feq[v] + (W[v] / (2.0 * CS2 * CS2)) * (
                    (cx * cx - CS2) * Sxx + (cy * cy - CS2) * Syy + 2.0 * cx * cy * Sxy
                )

    rho_out = f.sum(axis=0)
    u_out = np.stack(
        [(f * C[:, 0][:, None, None]).sum(axis=0), (f * C[:, 1][:, None, None]).sum(axis=0)],
        axis=-1,
    ) / rho_out[..., None]
    diagnostics = {
        "corner_nodes": np.array(corner_nodes, dtype=np.int64).reshape(-1, 2),
        "edge_nodes": np.array(edge_nodes, dtype=np.int64).reshape(-1, 2),
        "rho": rho_out,
        "u": u_out,
    }
    return f, rho_out, u_out, diagnostics