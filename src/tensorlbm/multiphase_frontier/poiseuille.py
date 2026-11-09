"""Pressure-driven generalised-Newtonian flow: LBM channel and radial pipe solver.

Two *independent* numerical solvers exercise the same constitutive law:

``channel_lbm_generalized_newtonian``
    D2Q9 BGK lattice Boltzmann on a plane channel, periodic in the streamwise
    direction, halfway bounce-back on both walls, Guo forcing for the constant
    pressure gradient.  The relaxation time is *local*:
    ``tau(x) = 1/2 + 3 nu_eff(gamma_dot(x))`` with the shear rate taken from the
    non-equilibrium stress tensor
    ``Pi^neq = sum_i e_i e_i (f_i - f_i^eq) = -2 rho cs^2 tau S``.
    ``nu_eff`` is lagged one time step (explicit generalised-Newtonian
    coupling); the fixed point of the iteration -- the steady state -- carries
    no lag bias, the transient does.

``pipe_finite_volume``
    1-D radial finite-volume solver of the steady momentum balance
    ``(1/r) d(r tau)/dr = G`` in conservative flux form with Picard iteration on
    the apparent viscosity.  Shares no code with the LBM and no code with the
    quadrature reference of :mod:`tensorlbm.multiphase_frontier.rheology`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "D2Q9_E",
    "D2Q9_W",
    "channel_lbm_generalized_newtonian",
    "channel_finite_volume",
    "pipe_finite_volume",
    "profile_error",
]

# D2Q9 ordering: 0 rest, 1..4 axis, 5..8 diagonals
D2Q9_E = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                   [0, 0, 1, 0, -1, 1, 1, -1, -1]], dtype=np.int64)
D2Q9_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
_EX = D2Q9_E[0].astype(float)[:, None, None]
_EY = D2Q9_E[1].astype(float)[:, None, None]
_W = D2Q9_W[:, None, None]


def _equilibrium(rho, ux, uy):
    eu = 3.0 * (_EX * ux[None] + _EY * uy[None])
    usq = 1.5 * (ux * ux + uy * uy)[None]
    return rho[None] * _W * (1.0 + eu + 0.5 * eu * eu - usq)


def channel_lbm_generalized_newtonian(
    model,
    ny: int = 65,
    nx: int = 8,
    grad_p: float = 2.0e-5,
    steps: int = 200000,
    tol: float = 1e-9,
    check_every: int = 500,
    tau_min: float = 0.55,
    nu_init: float | None = None,
    u_init: np.ndarray | None = None,
    gamma_floor: float = 1.0e-6,
    nu_update_every: int = 100,
    nu_relax: float = 0.5,
) -> dict:
    """Steady plane-Poiseuille flow of a generalised Newtonian fluid.

    Parameters
    ----------
    ny : number of lattice rows including the two solid rows; the wall sits
        halfway between the solid row and the first fluid row, so the channel
        width is ``H = ny - 2`` and the wall-normal coordinate of fluid node
        ``j`` is ``y = j - (ny-1)/2``.
    nx : streamwise rows kept small; the flow is x-independent at steady state
        and the profile is averaged over x.
    """
    ex, ey = D2Q9_E[0], D2Q9_E[1]
    w = _W

    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    f = _equilibrium(rho, ux, uy)
    Fx = np.full((ny, nx), float(grad_p))
    Fy = np.zeros((ny, nx))

    H = float(ny - 2)
    y = np.arange(ny, dtype=float) - 0.5 * (ny - 1)
    nu_floor = (tau_min - 0.5) / 3.0

    if nu_init is None:
        nu_init = float(model.mu(1e-3))
    nu_eff = np.full((ny, nx), max(nu_init, nu_floor))
    # A generalised-Newtonian liquid started from rest is a stiff fixed point
    # (zero shear rate -> maximum apparent viscosity -> no motion).  The field is
    # therefore initialised with the Newtonian parabola that carries the same
    # pressure gradient; the fixed point reached at convergence is independent
    # of this guess (verified by the ``u_init`` override in the tests).
    if u_init is None:
        u0 = grad_p / (2.0 * max(nu_init, nu_floor)) * (0.25 * H ** 2 - y ** 2)
        u_init = np.clip(u0, 0.0, None)
    ux = np.repeat(np.asarray(u_init, dtype=float)[:, None], nx, axis=1)
    rho = np.ones((ny, nx))
    f = _equilibrium(rho, ux, uy)
    # start the apparent viscosity from the shear rate of the initial profile
    g0 = np.abs(np.gradient(np.asarray(u_init, dtype=float)))
    nu_eff = np.maximum(np.repeat(model.mu(np.maximum(g0, gamma_floor))[:, None],
                                 nx, axis=1), nu_floor)

    u_prev = None
    rel = np.inf
    step = 0
    up = [i for i in range(1, 9) if ey[i] > 0]
    dn = [i for i in range(1, 9) if ey[i] < 0]
    for step in range(1, steps + 1):
        # ---- moments with the Guo half force -------------------------------
        rho = f.sum(axis=0)
        ux = (f * _EX).sum(axis=0) / rho + 0.5 * Fx / rho
        uy = (f * _EY).sum(axis=0) / rho + 0.5 * Fy / rho
        feq = _equilibrium(rho, ux, uy)
        # ---- shear rate from the velocity field ---------------------------
        # For a fully developed channel flow du/dy is the only non-zero
        # component of grad u, so the second invariant reduces to
        #   gamma_dot = |d u_x / d y|
        # evaluated by second-order central differences in the interior and
        # one-sided differences against the (no-slip) wall value at the two
        # boundary rows.  Using the velocity field rather than the
        # non-equilibrium stress avoids the O(u F) bias that Guo forcing
        # introduces into Pi^neq and keeps the explicit viscosity coupling
        # stable for strongly shear-thinning laws.
        # ghost cells mirror the no-slip wall (u = 0 halfway between row 0 and
        # row 1), so that the central difference at the first fluid row returns
        # the wall shear instead of the stale solid-row velocity
        ux[0] = -ux[1]
        ux[ny - 1] = -ux[ny - 2]
        gamma_dot = np.abs(np.gradient(ux, axis=0))
        gamma_dot = np.maximum(gamma_dot, gamma_floor)
        # ---- generalised-Newtonian viscosity (lagged, under-relaxed) -------
        # Under-relaxation keeps the explicit viscosity coupling neutral for
        # strongly shear-thinning laws (the fixed point is unchanged).
        # The apparent viscosity is refreshed only every ``nu_update_every``
        # steps: a *frozen*-viscosity BGK update is unconditionally stable in
        # tau, whereas a per-step explicit viscosity feedback makes a strongly
        # shear-thinning channel run away (measured: u_max overshoot to 0.070
        # then collapse to 0 for the power law).  Freezing + under-relaxation
        # keeps the same fixed point and removes that instability.
        if step % nu_update_every == 0:
            nu_eff = (1.0 - nu_relax) * nu_eff + nu_relax * np.maximum(
                model.mu(gamma_dot), nu_floor)
        tau_loc = 0.5 + 3.0 * nu_eff
        # ---- Guo-forced BGK collision --------------------------------------
        eu = _EX * ux[None] + _EY * uy[None]
        gf = (1.0 - 0.5 / tau_loc)[None]
        Si = (gf * w * (3.0 * (_EX - 3.0 * ux[None]) + 9.0 * eu * _EX) * Fx[None]
              + gf * w * (3.0 * (_EY - 3.0 * uy[None]) + 9.0 * eu * _EY) * Fy[None])
        fc = f - (f - feq) / tau_loc[None] + Si
        # ---- streaming (periodic in x) -------------------------------------
        for i in range(9):
            f[i] = np.roll(np.roll(fc[i], ey[i], axis=0), ex[i], axis=1)
        # ---- halfway bounce-back on both walls ------------------------------
        for i in up:
            f[i, 1, :] = fc[_OPP[i], 1, :]
        for i in dn:
            f[i, ny - 2, :] = fc[_OPP[i], ny - 2, :]
        if not np.isfinite(nu_eff).all():
            raise FloatingPointError(
                f"generalised-Newtonian channel diverged at step {step}; the "
                f"viscosity range of the model is outside the stable envelope")
        if step % check_every == 0:
            u_now = ((f * _EX).sum(axis=0) / f.sum(axis=0)).mean(axis=1)
            if u_prev is not None:
                rel = float(np.max(np.abs(u_now - u_prev))
                            / max(np.max(np.abs(u_now)), 1e-30))
                if rel < tol:
                    u_prev = u_now
                    break
            u_prev = u_now

    u_prof = ((f * _EX).sum(axis=0) / f.sum(axis=0)).mean(axis=1)
    return {
        "y": y,
        "u": u_prof,
        "H": H,
        "h": 0.5 * H,
        "grad_p": grad_p,
        "steps": step,
        "rel_change": float(rel),
        "nu": nu_eff.mean(axis=1),
        "u_max": float(u_prof[1:-1].max()),
        "mask": np.arange(1, ny - 1),
    }


def pipe_finite_volume(model, n_r: int = 400, R: float = 1.0, grad_p: float = 1.0,
                       iters: int = 5000, tol: float = 1e-13,
                       relax: float = 1.0) -> dict:
    """Steady pipe Poiseuille flow by a conservative radial finite-volume solver.

    Cell-centred unknowns ``u_j`` on ``r_j = (j+1/2) dr``; the momentum balance
    integrated over the annular control volume j reads::

        r_{j+1/2} tau_{j+1/2} - r_{j-1/2} tau_{j-1/2} = G (r_{j+1/2}^2 - r_{j-1/2}^2)/2

    with ``tau_f = mu(gamma_f) gamma_f`` evaluated **at the face** from the face
    shear rate ``gamma_f`` (the standard Picard linearisation for a
    generalised-Newtonian FV discretisation), ``r_{-1/2} = 0`` (axis, zero flux)
    and ``r_{N-1/2} = R`` where ``u = 0``.
    """
    dr = R / n_r
    r = (np.arange(n_r) + 0.5) * dr
    rface = (np.arange(n_r) + 1.0) * dr             # r_{j+1/2}, j = 0..N-1
    rface_in = np.concatenate([[0.0], rface[:-1]])  # r_{j-1/2}
    rhs = 0.5 * grad_p * (rface ** 2 - rface_in ** 2)

    u = np.zeros(n_r)
    scale = 1.0
    du = np.inf
    it = 0
    for it in range(1, iters + 1):
        g_face = np.empty(n_r)                       # shear rate at each face
        g_face[:-1] = (u[1:] - u[:-1]) / dr
        g_face[-1] = (0.0 - u[-1]) / (0.5 * dr)      # wall face, half cell
        mu_face = model.mu(np.abs(g_face))
        # flux coefficient: tau_face * r_face = mu_face * (u_{j+1}-u_j)/dr * r
        coef_hi = rface * mu_face / dr
        coef_hi[-1] = R * mu_face[-1] / (0.5 * dr)
        # the lower face of cell j is the upper face of cell j-1: its apparent
        # viscosity must be the one evaluated at that same face
        coef_lo = np.empty(n_r)
        coef_lo[0] = 0.0
        coef_lo[1:] = rface_in[1:] * mu_face[:-1] / dr
        lower = -coef_lo[1:].copy()
        upper = -coef_hi[:-1].copy()
        diag = coef_lo + coef_hi
        u_new = _thomas(lower, diag, upper, rhs)
        if it == 1:
            scale = max(np.max(np.abs(u_new)), 1e-30)
        d = float(np.max(np.abs(u_new - u)) / scale)
        u = (1.0 - relax) * u + relax * u_new
        du = d
        if d < tol:
            break
    g_cell = np.abs(np.gradient(u, dr))
    return {
        "r": r, "u": u, "R": R, "grad_p": grad_p, "iters": it,
        "rel_change": float(du), "nu": model.mu(g_cell), "u_max": float(u.max()),
    }


def channel_finite_volume(model, n_y: int = 128, h: float = 1.0,
                         grad_p: float = 1.0, iters: int = 5000,
                         tol: float = 1e-13) -> dict:
    """Steady plane-Poiseuille flow of a generalised Newtonian fluid, 1-D FV.

    The Cartesian counterpart of :func:`pipe_finite_volume` (same face-centred
    Picard linearisation, no radial weight)::

        tau_{j+1/2} - tau_{j-1/2} = -G dy,   tau_f = mu(gamma_f) gamma_f

    with no-slip faces at ``y = +-h``.  Independent of the LBM code path and of
    the quadrature reference.
    """
    dy = 2.0 * h / n_y
    y = -h + (np.arange(n_y) + 0.5) * dy
    yface = -h + (np.arange(n_y) + 1.0) * dy        # upper faces; lower = yface_in
    yface_in = np.concatenate([[-h], yface[:-1]])
    rhs = grad_p * (yface - yface_in)

    u = np.zeros(n_y)
    scale = 1.0
    du = np.inf
    it = 0
    for it in range(1, iters + 1):
        g_face = np.empty(n_y)
        g_face[:-1] = (u[1:] - u[:-1]) / dy
        g_face[-1] = (0.0 - u[-1]) / (0.5 * dy)      # top wall, half cell
        mu_face = model.mu(np.abs(g_face))
        coef_hi = mu_face / dy
        coef_hi[-1] = mu_face[-1] / (0.5 * dy)
        coef_lo = np.empty(n_y)
        coef_lo[0] = mu_face[0] / (0.5 * dy)         # bottom wall, half cell
        coef_lo[1:] = mu_face[:-1] / dy
        lower = -coef_lo[1:].copy()
        upper = -coef_hi[:-1].copy()
        diag = coef_lo + coef_hi
        u_new = _thomas(lower, diag, upper, rhs)
        if it == 1:
            scale = max(np.max(np.abs(u_new)), 1e-30)
        d = float(np.max(np.abs(u_new - u)) / scale)
        u = u_new
        du = d
        if d < tol:
            break
    g_cell = np.abs(np.gradient(u, dy))
    return {"y": y, "u": u, "h": h, "grad_p": grad_p, "iters": it,
            "rel_change": float(du), "nu": model.mu(g_cell),
            "u_max": float(u.max())}


def _thomas(lower, diag, upper, rhs) -> np.ndarray:
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / m
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def profile_error(u_num: np.ndarray, u_ref: np.ndarray, mask=None) -> dict:
    """Relative L2 and Linf profile errors, both normalised by ``max|u_ref|``."""
    u_num = np.asarray(u_num, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    if mask is not None:
        u_num, u_ref = u_num[mask], u_ref[mask]
    scale = max(np.max(np.abs(u_ref)), 1e-30)
    l2 = float(np.sqrt(np.mean((u_num - u_ref) ** 2)) / scale)
    linf = float(np.max(np.abs(u_num - u_ref)) / scale)
    return {"rel_l2": l2, "rel_linf": linf, "u_ref_max": float(scale)}