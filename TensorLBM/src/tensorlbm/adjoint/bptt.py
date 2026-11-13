"""Full backpropagation-through-time (BPTT) reference gradient for the
D2Q9 BGK LBM forward model in :mod:`tensorlbm.adjoint.lbm2d`.

Derivation (standard adjoint of the BGK update; cf. Griewank & Walther,
"Evaluating Derivatives of Computer Programs", 2nd ed., Springer 2008,
ch. 2; Williams & Zipser 1989 for the BPTT scheme):

Forward step, with y = x - e_i:

    f^{n+1}_i(y + e_i) = f_i^n(y) - (1/tau) [ f_i^n(y) - feq_i(y) ]

Jacobian of one step:

    df^{n+1}_i(x)/df^n_j(y) = delta_{x, y+e_i}
        * [ delta_ij - (1/tau) ( delta_ij - dfeq_i/df_j |_y ) ]

Adjoint (lambda is the costate, lambda^{n+1} known):

    lambda^n_j(y) = mu_j(y) - (1/tau) [ mu_j(y) - (B^T mu)_j(y) ]
    mu_j(y)        = lambda^{n+1}_j(y + e_j)          (anti-streaming)
    (B^T mu)_j     = sum_i (dfeq_i/df_j) mu_i

Note: B = dfeq/df is NOT symmetric, so B^T mu != B mu; the adjoint
recursion requires the transpose (see b_action_t).

with the 9x9 collision-Jacobian action (cell-wise, from macro fields):

    (B mu)_i = dfeq_drho_i * s1
               + dfeq_dux_i * (mx - ux*s1) + dfeq_duy_i * (my - uy*s1)
    s1 = sum_j mu_j,   m_a = sum_j c_ja mu_j

    (the rho from du_a/df_j = (c_ja - u_a)/rho cancels the rho in
    dfeq_i/du_a = W_i rho A_ia, so no 1/rho factor remains)

Parameter derivative (tau enters only the collision):

    df^{n+1}_i(x)/dtau = (1/tau^2) ( f_i^n(x-e_i) - feq_i^n(x-e_i) )

    dJ/dtau = sum_n sum_y (1/tau^2) ( f_i^n(y) - feq_i^n(y) ) mu_i(y)

The BPTT reference stores every forward state (memory O(T*N*Q)) and is
the ground truth the out-of-core (OOTC) variant must reproduce.
"""

from __future__ import annotations

import numpy as np

from .lbm2d import EX, EY, W, macro, equilibrium, step

__all__ = ["anti_stream", "b_action", "b_action_t", "seed_terminal",
           "bptt_gradient"]


def anti_stream(lam: np.ndarray) -> np.ndarray:
    """mu_j(y) = lambda_j(y + e_j): roll by -e_j (periodic)."""
    out = np.empty_like(lam)
    for i in range(9):
        out[i] = np.roll(lam[i], shift=(-EX[i], -EY[i]), axis=(0, 1))
    return out


def b_action(f: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Apply the collision Jacobian B = dfeq/df to mu at the state f.

    Returns (B mu) with (B mu)_i = sum_j (dfeq_i/df_j) mu_j, evaluated
    cell-wise from the macroscopic fields of f.
    """
    rho, ux, uy = macro(f)
    uc = EX[:, None, None] * ux[None] + EY[:, None, None] * uy[None]
    u2 = ux * ux + uy * uy
    dfeq_drho = W[:, None, None] * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * u2[None])
    dfeq_dux = W[:, None, None] * (3.0 * EX[:, None, None]
                                   + 9.0 * EX[:, None, None] * uc - 3.0 * ux[None])
    dfeq_duy = W[:, None, None] * (3.0 * EY[:, None, None]
                                   + 9.0 * EY[:, None, None] * uc - 3.0 * uy[None])
    s1 = mu.sum(axis=0)
    mx = (mu * EX[:, None, None]).sum(axis=0)
    my = (mu * EY[:, None, None]).sum(axis=0)
    return (dfeq_drho * s1[None]
            + dfeq_dux * (mx - ux * s1)
            + dfeq_duy * (my - uy * s1))


def b_action_t(f: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Apply the TRANSPOSED collision Jacobian B^T = (dfeq/df)^T to mu.

    The adjoint recursion needs (B^T mu)_j = sum_i (dfeq_i/df_j) mu_i,
    NOT (B mu)_j = sum_i (dfeq_j/df_i) mu_i -- B is not symmetric in
    general, so the two differ.  With

        dfeq_i/df_j = W_i P_i + W_i A_i (c_j - u)/rho,
        P_i = 1 + 3 uc_i + 4.5 uc_i^2 - 1.5 u^2,
        A_i = 3 c_i + 9 c_i uc_i - 3 u   (per component),

    (B^T mu)_j = S0 + (c_jx - ux) Sx + (c_jy - uy) Sy
    with S0 = sum_i W_i P_i mu_i,
         Sx = sum_i W_i (3 c_ix + 9 c_ix uc_i - 3 ux) mu_i,
         Sy = sum_i W_i (3 c_iy + 9 c_iy uc_i - 3 uy) mu_i.
    (no 1/rho: it cancels between dfeq/du and du/df)
    """
    rho, ux, uy = macro(f)
    uc = EX[:, None, None] * ux[None] + EY[:, None, None] * uy[None]
    u2 = ux * ux + uy * uy
    P = 1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * u2[None]
    Ax = 3.0 * EX[:, None, None] + 9.0 * EX[:, None, None] * uc - 3.0 * ux[None]
    Ay = 3.0 * EY[:, None, None] + 9.0 * EY[:, None, None] * uc - 3.0 * uy[None]
    S0 = (W[:, None, None] * P * mu).sum(axis=0)
    Sx = (W[:, None, None] * Ax * mu).sum(axis=0)
    Sy = (W[:, None, None] * Ay * mu).sum(axis=0)
    return (S0[None]
            + (EX[:, None, None] - ux[None]) * Sx[None]
            + (EY[:, None, None] - uy[None]) * Sy[None])


def seed_terminal(ux_t: np.ndarray, uy_t: np.ndarray, rho_t: np.ndarray,
                ux_ref: np.ndarray, uy_ref: np.ndarray) -> np.ndarray:
    """Seed lambda^T for J = 1/2 ||u_T - u_ref||^2.

    dJ/df^T_j = (ux_T-ux_ref)(EX_j - ux_T)/rho + (uy_T-uy_ref)(EY_j - uy_T)/rho
    """
    dux = ux_t - ux_ref
    duy = uy_t - uy_ref
    lam = np.empty((9,) + ux_t.shape, dtype=np.float64)
    for i in range(9):
        lam[i] = (dux * (EX[i] - ux_t) + duy * (EY[i] - uy_t)) / rho_t
    return lam


def bptt_gradient(tau: float, T: int, f0: np.ndarray,
                 ux_ref: np.ndarray, uy_ref: np.ndarray):
    """Exact BPTT gradient dJ/dtau with full trajectory storage.

    J(tau) = 1/2 || u_T(tau) - u_ref ||^2  (terminal velocity mismatch).

    Returns (dJ_dtau, J, (ux_T, uy_T), states) with states[n] = f^n.
    """
    # ---- forward, store everything ----
    states = [f0]
    f = f0
    for _ in range(T):
        f = step(f, tau)
        states.append(f)

    # ---- terminal seed ----
    rho_t, ux_t, uy_t = macro(states[T])
    lam = seed_terminal(ux_t, uy_t, rho_t, ux_ref, uy_ref)
    J = 0.5 * float(np.sum((ux_t - ux_ref) ** 2 + (uy_t - uy_ref) ** 2))

    # ---- backward ----
    dJ_dtau = 0.0
    for n in range(T - 1, -1, -1):
        f_n = states[n]
        mu = anti_stream(lam)
        rho, ux, uy = macro(f_n)
        feq = equilibrium(rho, ux, uy)
        dJ_dtau += float(np.sum((f_n - feq) * mu) / (tau * tau))
        lam = mu - (mu - b_action_t(f_n, mu)) / tau

    return dJ_dtau, J, (ux_t, uy_t), states
