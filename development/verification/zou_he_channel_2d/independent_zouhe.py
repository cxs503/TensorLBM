#!/usr/bin/env python3
"""Independent D2Q9 Zou/He reconstruction, solved as a linear system.

Purpose: check the closed-form reconstruction in
``src/tensorlbm/boundary/boundaries.py`` **without re-using its algebra**.

The repo functions hard-code the solved expressions.  Here the same boundary
value problem is posed in the form Zou & He (1997, Phys. Fluids 9:1591, §II)
write it, and solved numerically:

* unknowns are the populations entering the domain through the boundary layer
  plus one macroscopic unknown (``rho`` in velocity mode, ``ux`` in pressure
  mode);
* three equations are the discrete moments ``sum f = rho``,
  ``sum f c_x = rho ux``, ``sum f c_y = rho uy``;
* the fourth is Zou & He's non-equilibrium bounce-back on the *normal* link,
  ``f_i - f_opp = f_i^eq - f_opp^eq``.  For opposite directions the quadratic
  terms of the equilibrium cancel, so this equation is exactly
  ``f_i - f_opp = (2/3) rho ux`` -- it is a closure condition, not a
  truncation, and it is what makes the count 4 equations / 4 unknowns.

Nothing here is copied from the module under test: the lattice table and the
weights are read from the package (they define D2Q9, not the BC algebra), the
direction sets are derived by *counting* ``sign(c_x)``, and the solution comes
from ``numpy.linalg.solve``.  Agreement to round-off therefore means the
closed-form expressions are the solution of the stated BVP.

Only the west (velocity / pressure) and east (pressure) layers that
``boundaries.py`` actually implements are covered; north/south Zou/He is not
implemented in that module and is out of scope here.

CLI
---
    python3 independent_zouhe.py            # self-check vs the repo module
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.boundary.boundaries import (  # noqa: E402
    zou_he_inlet_pressure,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import C, W, OPPOSITE  # noqa: E402

DTYPE = torch.float64


def _eq(cx: float, cy: float, w: float, rho, ux, uy):
    cu = cx * ux + cy * uy
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))


def _dirs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = C.cpu().numpy().astype(float)
    w = W.cpu().numpy().astype(float)
    opp = OPPOSITE.cpu().numpy().astype(int)
    return c, w, opp


def solve_layer(
    f_layer: np.ndarray,
    side: str,
    *,
    mode: str,
    ux: float = 0.0,
    uy: float = 0.0,
    rho: float = 1.0,
) -> dict:
    """Solve the Zou/He BVP for one boundary line.

    Parameters
    ----------
    f_layer : (9, N) array of populations after streaming on the boundary line.
    side : "west" (in-flowing = c_x > 0) or "east" (in-flowing = c_x < 0).
    mode : "velocity" (ux, uy given, rho unknown) or "pressure" (rho, uy given,
        ux unknown).

    Returns
    -------
    dict with ``f`` (9, N) reconstructed layer, ``rho``/``ux``/``uy`` moments.
    """
    c, w, opp = _dirs()
    sign = +1.0 if side == "west" else -1.0
    incoming = [i for i in range(1, 9) if np.sign(c[i, 0]) == sign]
    known = [i for i in range(9) if i not in incoming]
    assert len(incoming) == 3, incoming

    n = f_layer.shape[1]
    out = f_layer.copy()
    for j in range(n):
        fk = f_layer[:, j]
        K = sum(fk[i] for i in known)                       # known mass
        Kx = sum(c[i, 0] * fk[i] for i in known)            # known x-momentum
        Ky = sum(c[i, 1] * fk[i] for i in known)            # known y-momentum

        # unknown vector z = [x_a, x_b, x_c] + [q]; q = rho (velocity) / ux (pressure)
        A = np.zeros((4, 4))
        b = np.zeros(4)
        # 1) mass: sum x + K = rho   (rho = q in velocity mode, = rho_in in pressure)
        A[0, :3] = 1.0
        b[0] = -K
        # 2) x-momentum: sum c_x x + Kx = rho ux
        for m, i in enumerate(incoming):
            A[1, m] = c[i, 0]
        b[1] = -Kx
        # 3) y-momentum: sum c_y x + Ky = rho uy
        for m, i in enumerate(incoming):
            A[2, m] = c[i, 1]
        b[2] = -Ky
        # 4) non-equilibrium bounce-back on the normal link:
        #    x_norm - f_opp = x_norm_eq - f_opp_eq = (2/3) rho (sign * ux),
        #    with sign = +1 on the west layer (c_norm = +x) and -1 on the east
        m_norm = incoming.index([i for i in incoming if c[i, 0] == sign][0])
        i_norm = incoming[m_norm]
        A[3, m_norm] = 1.0
        b[3] = fk[opp[i_norm]]

        if mode == "velocity":
            # q = rho.  mass: sum x - q = -K ; xmom: sum c_x x - q ux = -Kx ;
            # ymom: sum c_y x - q uy = -Ky ; NEBB: x_norm - (2/3) q ux = f_opp
            A[0, 3] = -1.0
            A[1, 3] = -ux
            A[2, 3] = -uy
            # normal link: c_norm.x = sign, so the NEBB RHS is (2/3) rho (sign*ux)
            A[3, 3] = -(2.0 / 3.0) * sign * ux
        else:
            # q = ux, rho prescribed.  mass: sum x = rho - K ;
            # xmom: sum c_x x - rho q = -Kx ; ymom: sum c_y x = rho uy - Ky ;
            # NEBB: x_norm - (2/3) rho q = f_opp
            A[0, 3] = 0.0
            b[0] = rho - K
            A[1, 3] = -rho
            b[1] = -Kx
            A[2, 3] = 0.0
            b[2] = rho * uy - Ky
            A[3, 3] = -(2.0 / 3.0) * rho * sign

        z = np.linalg.solve(A, b)
        for m, i in enumerate(incoming):
            out[i, j] = z[m]
        q = z[3]
        if mode == "velocity":
            rho_j, ux_j = q, ux
        else:
            rho_j, ux_j = rho, q
        uy_j = uy
        # moments of the reconstructed layer (all nine populations)
        assert abs(out[:, j].sum() - rho_j) < 1e-13 * max(1.0, abs(rho_j))
        assert abs((c[:, 0] * out[:, j]).sum() - rho_j * ux_j) < 1e-13
        assert abs((c[:, 1] * out[:, j]).sum() - rho_j * uy_j) < 1e-13
    return out


def check(ny: int = 18, nx: int = 9, seed: int = 19970317) -> dict:
    """Compare the independent solve with the repo's closed forms."""
    gen = torch.Generator().manual_seed(seed)
    f = torch.rand((9, ny, nx), generator=gen, dtype=DTYPE) + 0.1
    u_vec = torch.linspace(0.02, 0.09, ny, dtype=DTYPE)
    uy_vec = torch.linspace(-0.02, 0.02, ny, dtype=DTYPE)
    fw = f[:, :, 0].numpy()
    fe = f[:, :, -1].numpy()

    res: dict = {}

    # --- west velocity inlet, profile ---
    ux_p = u_vec.numpy()
    uy_p = uy_vec.numpy()
    ind = np.stack([
        solve_layer(fw[:, [j]], "west", mode="velocity", ux=ux_p[j], uy=uy_p[j])[:, 0]
        for j in range(ny)
    ], axis=1)
    rep = zou_he_inlet_velocity(f, u_vec, uy_vec)[:, :, 0].numpy()
    res["west_velocity"] = {
        "max_abs_diff": float(np.abs(ind - rep).max()),
        "max_rel_diff": float((np.abs(ind - rep) / (np.abs(ind) + 1e-30)).max()),
    }

    # --- west pressure inlet, scalar rho + profile uy ---
    rho_in = 1.004
    ind = np.stack([
        solve_layer(fw[:, [j]], "west", mode="pressure", rho=rho_in, uy=uy_p[j])[:, 0]
        for j in range(ny)
    ], axis=1)
    rep = zou_he_inlet_pressure(f, rho_in, uy_vec)[:, :, 0].numpy()
    res["west_pressure"] = {"max_abs_diff": float(np.abs(ind - rep).max())}

    # --- east pressure outlet, scalar rho + profile uy ---
    rho_out = 1.003
    ind = np.stack([
        solve_layer(fe[:, [j]], "east", mode="pressure", rho=rho_out, uy=uy_p[j])[:, 0]
        for j in range(ny)
    ], axis=1)
    rep = zou_he_outlet_pressure(f, rho_out, uy_vec)[:, :, -1].numpy()
    res["east_pressure"] = {"max_abs_diff": float(np.abs(ind - rep).max())}

    # --- known populations must be untouched ---
    c, _, opp = _dirs()
    touched = {
        "west_velocity": [1, 5, 8],
        "west_pressure": [1, 5, 8],
        "east_pressure": [3, 6, 7],
    }
    for k, dirs in touched.items():
        assert sorted(dirs) == sorted(
            [i for i in range(1, 9) if np.sign(c[i, 0]) == (+1 if k.startswith("west") else -1)]
        ), k
    return res


def main() -> int:
    out = check()
    print(json.dumps(out, indent=1))
    worst = max(v["max_abs_diff"] for v in out.values())
    print(f"worst |independent - repo| = {worst:.3e}")
    return 0 if worst < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())