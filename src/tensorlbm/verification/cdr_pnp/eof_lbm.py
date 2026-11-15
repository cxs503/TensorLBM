"""Real numerical solver for electro-osmotic flow (EOF) in a slit microchannel.

Motivation
----------
The original ``pnp_solver.run_electroosmosis`` did **not** solve any flow
equation: it evaluated the (analytical) Helmholtz-Smoluchowski identity
``u(y) = u_hs * (1 - psi(y)/zeta)`` and then compared the *mean of that same
analytical expression* against ``u_hs``, reporting the difference as a
"numerical" error (gate C3).  That is circular - the gate could never fail
for physical reasons.  This module supplies a genuine solve so that C3 tests a
solver instead of an identity.

Physics
-------
Symmetric slit channel, wall-normal coordinate ``y in [0, H]``, both walls at
the same zeta potential.  The equilibrium EDL potential obeys the
Poisson-Boltzmann equation; in the Debye-Huckel (small zeta) limit

    psi(y) = zeta * cosh(xi/lambda_D) / cosh(H/(2 lambda_D)),   xi = y - H/2

The ionic charge density follows Poisson's equation, ``rho_e = -eps psi''``,
and the steady Navier-Stokes equations with a pure electro-osmotic body force
(no applied pressure gradient) read

    mu u'' = -rho_e E = eps E psi''.

Because the right-hand side is a total derivative, the equation integrates
exactly for *arbitrary* EDL structure (not only the Debye-Huckel limit):

    u(y) = (eps E / mu) (psi(y) - zeta) = u_hs * (1 - psi(y)/zeta),
    u_hs = -eps E zeta / mu   (Helmholtz-Smoluchowski velocity)

and both no-slip conditions ``u(0) = u(H) = 0`` are satisfied.  This makes
``u_hs * (1 - psi/zeta)`` an exact reference for the flow solver while the
flow solver itself is completely independent of it.

Two independent numerical flow solvers are provided:

``solve_stokes_fd``   finite-difference (Thomas algorithm) direct solution of
                      the discrete steady Stokes problem on the half-way
                      bounce-back grid (exact discrete solution, no iteration).
``solve_stokes_lbm``  D2Q9 BGK lattice-Boltzmann with Guo forcing, wall-normal
                      varying body force, half-way bounce-back walls, marched
                      to a steady state.

Units
-----
All quantities live in lattice units inside the LBM solver: ``nu = c_s^2
(tau - 1/2)``, channel height ``H_eff = ny - 2`` (walls sit half a cell inside
the boundary nodes), ``lambda_D = H_eff / kappa_H`` and the force amplitude is
chosen from the target plug velocity, ``a0 = u_hs * nu / lambda_D^2``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = [
    "debye_huckel_psi",
    "charge_density_from_potential",
    "eof_exact_velocity",
    "solve_stokes_fd",
    "solve_stokes_lbm",
    "eof_profile_errors",
    "run_eof_benchmark",
]

# D2Q9 (x-streaming is a no-op for this wall-normal 1-D problem).
_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
_W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)
# full reversal for no-slip half-way bounce-back: (cx, cy) -> (-cx, -cy).
# Reflecting only the wall-normal component would give a specular (free-slip)
# wall: a uniform body force would then accelerate the whole profile as a*t
# without ever generating wall shear (that failure mode was observed and is
# now locked by tests/test_eof_lbm_verification.py).
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
_CS2 = 1.0 / 3.0


def debye_huckel_psi(y, H: float, zeta: float, lambda_D: float) -> np.ndarray:
    """Equilibrium EDL potential of a symmetric slit (Debye-Huckel limit)."""
    y = np.asarray(y, dtype=np.float64)
    a = H / (2.0 * lambda_D)
    return zeta * np.cosh((y - 0.5 * H) / lambda_D) / np.cosh(a)


def charge_density_from_potential(psi, epsilon: float, lambda_D: float) -> np.ndarray:
    """Ionic charge density from Poisson's equation ``rho_e = -eps psi''``.

    For the Debye-Huckel profile ``psi'' = psi / lambda_D**2`` exactly.
    """
    return -epsilon * np.asarray(psi, dtype=np.float64) / (lambda_D**2)


def eof_exact_velocity(psi, zeta: float, u_hs: float) -> np.ndarray:
    """Exact EOF profile ``u = u_hs (1 - psi/zeta)`` (uniform zeta walls)."""
    return u_hs * (1.0 - np.asarray(psi, dtype=np.float64) / zeta)


def _wall_offsets(ny: int):
    """Physical positions of the fluid nodes; half-way BB walls at +-0.5."""
    H_eff = float(ny - 2)
    y_phys = (np.arange(1, ny - 1, dtype=np.float64) - 0.5)
    return y_phys, H_eff


def _force_profile(ny: int, kappa_H: float, a0: float):
    """Discrete body force ``a(y) = a0 * psi/psi_centre`` on fluid nodes."""
    y_phys, H_eff = _wall_offsets(ny)
    lam = H_eff / kappa_H
    g = np.cosh((y_phys - 0.5 * H_eff) / lam) / np.cosh(0.5 * H_eff / lam)
    return a0 * g, y_phys, H_eff, lam


def solve_stokes_fd(
    ny: int = 129,
    kappa_H: float = 10.0,
    nu: float = 1.0 / 6.0,
    u_hs: float = 0.02,
) -> Dict[str, object]:
    """Direct FD solve of ``nu u'' = -a(y)`` with no-slip half-way walls.

    Returns the discrete solution together with the grid data needed to build
    the analytical reference (``y_phys`` normalised to ``[0, 1]``).
    """
    ny = int(ny)
    if ny < 8:
        raise ValueError("ny must be >= 8")
    y_phys, H_eff, lam = _force_profile(ny, kappa_H, 0.0)[1:]
    a0 = u_hs * nu / lam**2
    a, _, _, _ = _force_profile(ny, kappa_H, a0)

    n = ny - 2  # interior (fluid) unknowns
    # nu (u_{j+1} - 2 u_j + u_{j-1}) = -a_j ; ghost u_0 = -u_1, u_{n+1} = -u_n
    diag = np.full(n, -2.0 * nu)
    lower = np.full(n - 1, nu)
    upper = np.full(n - 1, nu)
    diag[0] += -nu      # ghost wall at j=0 -> contributes -u_1
    diag[-1] += -nu     # ghost wall at j=n+1 -> contributes -u_n
    rhs = -a.copy()

    # Thomas algorithm (tridiagonal solve)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * cp[i - 1]
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom
        if i < n - 1:
            cp[i] = upper[i] / denom
    u = np.empty(n)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]

    xi = y_phys / H_eff
    return {
        "u": u,
        "y_phys": y_phys,
        "xi": xi,
        "H_eff": H_eff,
        "lambda_D": lam,
        "kappa_H": kappa_H,
        "nu": nu,
        "u_hs": u_hs,
        "a0": a0,
        "ny": ny,
    }


def _equilibrium(rho, ux, uy):
    cu = _CX[None, :] * ux[:, None] + _CY[None, :] * uy[:, None]
    usq = ux**2 + uy**2
    return _W[None, :] * rho[:, None] * (
        1.0 + cu / _CS2 + 0.5 * cu**2 / _CS2**2 - 0.5 * usq[:, None] / _CS2
    )


def solve_stokes_lbm(
    ny: int = 129,
    kappa_H: float = 10.0,
    tau: float = 1.0,
    u_hs: float = 0.02,
    max_steps: int = 400_000,
    tol: float = 1e-12,
    check_every: int = 500,
    init: str = "exact",
    force_scale: float = 1.0,
    force_sign: float = 1.0,
) -> Dict[str, object]:
    """D2Q9 BGK + Guo forcing solve of the EOF problem, marched to steady state.

    Parameters
    ----------
    ny : int
        Wall-normal node count (walls are nodes 0 and ny-1).
    kappa_H : float
        Dimensionless channel height ``H_eff / lambda_D``.
    tau : float
        BGK relaxation time; viscosity ``nu = (tau - 1/2)/3``.
    u_hs : float
        Target Helmholtz-Smoluchowski velocity in lattice units (sets the
        force amplitude, ``Ma = u_hs/c_s`` must stay small).
    init : {"exact", "rest"}
        Start from the analytical profile (fast) or from rest (proves the
        steady state is not an artefact of the initial condition).
    force_scale, force_sign : float
        Negative controls: scale / flip the body force.  ``force_scale=1``
        with ``force_sign=1`` is the physical case.

    Returns
    -------
    dict with ``u`` (steady velocity on fluid nodes), ``steps``, ``residual``
    (max |du| between successive checks), ``converged`` and the grid metadata.
    """
    ny = int(ny)
    if ny < 8:
        raise ValueError("ny must be >= 8")
    nu = (tau - 0.5) / 3.0
    y_phys, H_eff, lam = _force_profile(ny, kappa_H, 0.0)[1:]
    a0 = u_hs * nu / lam**2
    a_force, _, _, _ = _force_profile(ny, kappa_H, a0 * force_scale * force_sign)

    n = ny - 2
    if init == "exact":
        y = y_phys
        psi = debye_huckel_psi(y, H_eff, 1.0, lam)          # zeta = 1 normalised
        ux0 = u_hs * (1.0 - psi) * force_scale            # force_scale applied on top
    elif init == "rest":
        ux0 = np.zeros(n)
    else:
        raise ValueError("init must be 'exact' or 'rest'")

    rho = np.ones(ny)
    f = np.zeros((ny, 9))
    f[1:-1, :] = _equilibrium(rho[1:-1], ux0, np.zeros(n))

    inv_tau = 1.0 / tau
    # First-order Guo-type source, matching tensorlbm.turbulent_channel: the
    # source injects exactly sum_i c_ix S_i = rho * a_x (sum_i w_i c_ix^2 = 1/3).
    # Note: deliberately no "+F/2" velocity shift - mixing the two conventions
    # feeds back through f_eq and makes the momentum grow without bound.
    src_shape = 3.0 * _W[None, :] * _CX[None, :]  # (1, 9)
    u_prev = ux0.copy()
    converged = False
    steps = 0
    residual = np.inf
    for step in range(1, max_steps + 1):
        rho = f.sum(axis=1)
        ux = (f @ _CX)[1:-1] / rho[1:-1]
        uy = np.zeros(n)
        f_eq = np.zeros_like(f)
        f_eq[1:-1, :] = _equilibrium(rho[1:-1], ux, uy)
        f_post = f + inv_tau * (f_eq - f)
        f_post[1:-1, :] += src_shape * rho[1:-1, None] * a_force[:, None]
        # streaming in y + half-way bounce-back reflecting into the fluid node
        new = np.empty_like(f)
        for i in range(9):
            cyi = _CY[i]
            if cyi == 0:
                new[:, i] = f_post[:, i]
            elif cyi == 1:
                new[2:-1, i] = f_post[1:-2, i]
                new[ny - 2, _OPP[i]] = f_post[ny - 2, i]
            else:
                new[1:-2, i] = f_post[2:-1, i]
                new[1, _OPP[i]] = f_post[1, i]
        new[0, :] = 0.0
        new[ny - 1, :] = 0.0
        f = new
        if step % check_every == 0:
            rho = f.sum(axis=1)
            ux_new = (f @ _CX)[1:-1] / rho[1:-1]
            residual = float(np.max(np.abs(ux_new - u_prev)))
            u_prev = ux_new
            if residual < tol:
                converged = True
                steps = step
                break
        steps = step

    rho = f.sum(axis=1)
    ux = (f @ _CX)[1:-1] / rho[1:-1]
    # uy from the y-momentum (should vanish by symmetry)
    uy = (f @ _CY)[1:-1] / rho[1:-1]
    return {
        "u": ux,
        "u_y": uy,
        "y_phys": y_phys,
        "xi": y_phys / H_eff,
        "H_eff": H_eff,
        "lambda_D": lam,
        "kappa_H": kappa_H,
        "nu": nu,
        "tau": tau,
        "u_hs": u_hs,
        "a0": a0,
        "ny": ny,
        "steps": steps,
        "residual": residual,
        "converged": converged,
    }


def eof_profile_errors(u_num, u_ref, scale: float) -> Dict[str, float]:
    """Normalised max / L2 profile errors (``scale`` = reference plug velocity)."""
    u_num = np.asarray(u_num, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    d = u_num - u_ref
    return {
        "max_rel": float(np.max(np.abs(d)) / scale),
        "l2_rel": float(np.sqrt(np.mean(d**2)) / scale),
        "rms_rel": float(np.sqrt(np.mean(d**2)) / scale),
        "bias_rel": float(np.mean(d) / scale),
    }


def run_eof_benchmark(
    ny: int = 129,
    kappa_H: float = 10.0,
    tau: float = 1.0,
    u_hs: float = 0.02,
    solver: str = "lbm",
    max_steps: int = 400_000,
    tol: float = 1e-12,
    init: str = "exact",
    force_scale: float = 1.0,
    force_sign: float = 1.0,
) -> Dict[str, object]:
    """Solve EOF with ``solver`` and score it against the exact PB profile.

    Caveat (disclosed): for the physical case (``u_hs`` positive, unit force
    scale/sign) the analytical profile is the exact solution of both the
    continuous and - up to discretisation error - the discrete problem, so the
    reported error measures *discretisation* error only.  It is not a
    validation against an independent physical dataset.
    """
    if solver == "fd":
        res = solve_stokes_fd(ny=ny, kappa_H=kappa_H, u_hs=u_hs)
    elif solver == "lbm":
        res = solve_stokes_lbm(
            ny=ny, kappa_H=kappa_H, tau=tau, u_hs=u_hs,
            max_steps=max_steps, tol=tol, init=init,
            force_scale=force_scale, force_sign=force_sign,
        )
    else:
        raise ValueError("solver must be 'fd' or 'lbm'")

    lam = float(res["lambda_D"])
    H_eff = float(res["H_eff"])
    y_phys = np.asarray(res["y_phys"])
    psi = debye_huckel_psi(y_phys, H_eff, 1.0, lam)  # zeta-normalised
    # NB: the reference is the *physical* profile - it must not inherit
    # force_scale/force_sign, otherwise a deliberately wrong forcing (negative
    # control) would be scored against a wrong target and appear to match.
    u_exact = eof_exact_velocity(psi, 1.0, u_hs)
    errs = eof_profile_errors(res["u"], u_exact, abs(u_hs))
    out = dict(res)
    out.update(
        {
            "u_exact": u_exact,
            "psi_norm": psi,
            "errors": errs,
            "solver": solver,
            # plug-flow flatness in the channel core (kappa*y > 5)
            "flatness": _bulk_flatness(res["u"], y_phys, H_eff, kappa_H),
        }
    )
    return out


def _bulk_flatness(u, y_phys, H_eff: float, kappa_H: float) -> float:
    lam = H_eff / kappa_H
    core = np.abs(y_phys - 0.5 * H_eff) < max(0.5 * H_eff - 5.0 * lam, 0.05 * H_eff)
    if not np.any(core):
        return float("nan")
    uc = u[core]
    mean = float(np.mean(uc))
    if abs(mean) < 1e-30:
        return float("nan")
    return float(np.max(np.abs(uc - mean)) / abs(mean))