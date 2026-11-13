"""Control-volume (momentum-budget) drag estimator.

Why this module exists
----------------------
The library's only drag path is the Ladd momentum-exchange sum in
``tensorlbm.boundaries.compute_obstacle_forces``.  Every Cd reported by the
square-cylinder case pool therefore inherits any bug in that single routine
(sign, missing pre/post-stream link set, force normalisation).  A case study
that scores Cd against literature cannot detect such a bug: the reference
comparison would silently absorb it.

This module re-derives the drag from a *different* physical statement — the
steady Reynolds-transport balance over a rectangular control volume (CV) that
encloses the body::

    D = - oint_CV (rho u_x u_j + p delta_xj) n_j dl          (2-D, unit depth)

with ``p = c_s^2 rho = rho/3`` in lattice units (D2Q9, ``c_s^2 = 1/3``) and
``n`` the outward normal.  The time mean over an integer number of shedding
periods removes the unsteady term.  Because the CV estimator never touches the
boundary-link force bookkeeping, agreement between the two estimators is a
non-trivial validation of the force instrumentation (and of the mass/momentum
conservation of the collide/stream/boundary chain).

Conventions
-----------
* Arrays are indexed ``[y, x]`` and are the *time means over one or more whole
  shedding periods* of ``rho*u_x*u_x``, ``rho*u_x*u_y`` and ``p`` (means of the
  products, not products of the means — the periodic fluctuation contribution
  is part of the balance).
* ``D > 0`` means a force on the body in the +x (free-stream) direction.
* Face areas are one lattice unit per cell, so face integrals are plain sums.
"""

from __future__ import annotations

import numpy as np

__all__ = ["momentum_balance_drag"]


def momentum_balance_drag(
    rhouux: np.ndarray,
    rhouxuy: np.ndarray,
    p: np.ndarray,
    u_in: float,
    side: float,
    rho0: float = 1.0,
) -> dict[str, object]:
    """Drag coefficient from the time-mean x-momentum balance on a CV.

    Parameters
    ----------
    rhouux, rhouxuy, p : ndarray ``[y, x]``
        Time-mean of ``rho*u_x*u_x``, ``rho*u_x*u_y`` and the pressure over the
        CV box (the arrays must already be cropped to the box).
    u_in : float
        Free-stream (inlet) velocity used for the dynamic-pressure
        normalisation, lattice units.
    side : float
        Reference length D of the body, lattice units.
    rho0 : float
        Reference density of the normalisation (1.0 in lattice units).

    Returns
    -------
    dict with ``cd``, ``drag``, ``net_flux`` and the four face contributions
    (same sign convention as ``net_flux``: positive = momentum leaving the CV).
    """
    rhouux = np.asarray(rhouux, dtype=np.float64)
    rhouxuy = np.asarray(rhouxuy, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if rhouux.ndim != 2 or rhouux.shape != p.shape or rhouux.shape != rhouxuy.shape:
        raise ValueError(
            f"shape mismatch: rhouux{rhouux.shape} rhouxuy{rhouxuy.shape} p{p.shape}"
        )
    if min(rhouux.shape) < 3:
        raise ValueError("control volume must be at least 3x3 cells")
    if u_in <= 0 or side <= 0:
        raise ValueError("u_in and side must be positive")

    # Outward-normal fluxes of the x-component of momentum.
    f_left = float(-(rhouux[:, 0] + p[:, 0]).sum())      # n = (-1, 0)
    f_right = float(+(rhouux[:, -1] + p[:, -1]).sum())   # n = (+1, 0)
    f_bottom = float(-(rhouxuy[0, :]).sum())             # n = ( 0,-1)
    f_top = float(+(rhouxuy[-1, :]).sum())               # n = ( 0,+1)

    net_flux = f_left + f_right + f_bottom + f_top
    drag = -net_flux  # force of the body on the fluid is -D (x-momentum sink)
    cd = drag / (0.5 * rho0 * u_in * u_in * side)
    return {
        "cd": cd,
        "drag": drag,
        "net_flux": net_flux,
        "faces": {
            "left": f_left,
            "right": f_right,
            "bottom": f_bottom,
            "top": f_top,
        },
        "cv_shape": list(rhouux.shape),
        "normalisation": {"u_in": u_in, "side": side, "rho0": rho0},
    }
