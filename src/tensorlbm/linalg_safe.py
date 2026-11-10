"""Numerically verified small-matrix inverse (numpy LAPACK portability guard).

Why this module exists
----------------------
On the loong64 cluster nodes (numpy 2.0.1) ``np.linalg.inv`` / ``solve`` /
``pinv`` / ``svd`` return **garbage** for the 19x19 D3Q19 MRT matrix — measured
on node n315:

    np.linalg.inv(M)          -> NaN entries
    np.linalg.solve(M, I)     -> NaN
    np.linalg.pinv(M)         -> ||M pinv(M) - I|| = 2.1e0
    np.linalg.svd(M)          -> ||U S Vt - M||    = 1.4e1
    torch.linalg.inv(M64)     -> ||M inv(M) - I||  = 5.6e-16   (correct)

Because ``tensorlbm.solver3d`` built its MRT transform with ``np.linalg.inv``,
every MRT run on those nodes produced 1e16-scale populations and went NaN on
the first step (D3Q19 and D3Q27 alike).  The inverse is therefore computed with
torch's LAPACK and **verified against the identity**; numpy's result is only
accepted when it passes the same check.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["inverse_f64"]

_DEFAULT_TOL = 1e-10


def _residual(matrix: np.ndarray, inverse: np.ndarray) -> float:
    n = matrix.shape[0]
    return float(np.abs(matrix @ inverse - np.eye(n)).max())


def inverse_f64(matrix: np.ndarray, tol: float = _DEFAULT_TOL) -> np.ndarray:
    """Return ``matrix^-1`` in float64, verified to ``tol`` in the max norm.

    Tries torch (LAPACK) first, then numpy; raises ``RuntimeError`` when neither
    reproduces the identity, so a broken LAPACK can never be silently used.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    candidates: list[tuple[str, np.ndarray]] = []
    try:
        candidates.append(
            ("torch.linalg.inv", torch.linalg.inv(torch.tensor(matrix, dtype=torch.float64)).numpy())
        )
    except Exception:  # pragma: no cover - torch always available in tensorlbm
        pass
    try:
        candidates.append(("np.linalg.inv", np.linalg.inv(matrix)))
    except Exception:
        pass
    best = None
    for name, inv in candidates:
        if not np.all(np.isfinite(inv)):
            continue
        res = _residual(matrix, inv)
        if best is None or res < best[1]:
            best = (name, res, inv)
        if res < tol:
            return inv
    if best is None:
        raise RuntimeError("no finite inverse could be computed")
    name, res, inv = best
    raise RuntimeError(
        f"inverse could not be verified: best backend {name} gives "
        f"||M M^-1 - I|| = {res:.3e} > {tol:.1e}"
    )