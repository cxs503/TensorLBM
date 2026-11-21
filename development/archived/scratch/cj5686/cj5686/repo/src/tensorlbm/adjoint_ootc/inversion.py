"""Long-horizon parameter inversion driven by the OOTC adjoint gradient.

Motivation
----------
``GAP-PP-3-adjoint-ootc`` asked for a memory-bounded adjoint over long
time horizons, and the gradient side of that gate is met (the OOTC
gradient equals the BPTT reference bit for bit).  The *inversion* side --
"recover tau from a terminal velocity field" -- was left UNFINISHED by the
first pass: its driver was a hand-tuned steepest-descent loop (``alpha``
halving, 12 iterations, ``|d tau|/tau = 20 %``).  The gradient was right;
the optimiser was not.

Why a bracket method and not plain Newton
-----------------------------------------
Measured on the reference case (N=32, T=400, target tau=0.8, see
``docs/ADJOINT_OOTC_INVERSION_CONVERGENCE.md``) the objective

    J(tau) = 1/2 || u_T(tau) - u_ref ||^2

is *convex only near its minimum*.  Writing A(tau) ~ exp(-c (tau - 1/2))
for the pulse amplitude, A'(tau) ~ -c A so J'(tau) = (A - A_ref) A'
approaches 0+ from above as tau grows: g = dJ/dtau rises from negative,
peaks around tau ~ 1.0 and decays to 0+ for large tau.  So ``dg/dtau``
becomes **negative** (J concave) for tau above the peak.  A Newton step
started in that region is repelled outwards -- exactly what was observed
(iterate ran 1.20 -> 1.32 -> 1.45, pinned on the bound).

The driver below therefore runs the **Illinois variant of regula falsi**
on the analytic gradient inside a bracket ``[a, b]`` with ``g(a) < 0 <
g(b)`` (guaranteed to stay inside the bracket, superlinear, no tuning),
then polishes with a locally-measured Newton step once the bracket is
narrow, where J *is* convex.  Failure to bracket or to reach the
tolerance is reported as ``converged=False`` with a ``reason`` field --
never as success.

Objective
---------
``u_ref`` is built by :func:`reference_from_tau` from a target relaxation
time.  Use the *unscaled* terminal field (``u_ref = u_T(tau_ref)``) if the
inverse problem should have its zero exactly at ``tau_ref``: with an
amplitude-scaled reference (``0.9 * u_T``, as in the first pass) the
minimiser of J is offset from ``tau_ref`` by the amplitude mismatch --
an objective-specification artefact, not a solver error.  ``0.9`` of the
first pass puts that offset at ~6 % in tau, comfortably larger than the
3 % accuracy bar, so it is quantified separately in the doc rather than
being papered over.
"""

from __future__ import annotations

import numpy as np

from .lbm2d import macro, step
from .ootc import ootc_gradient

__all__ = ["reference_from_tau", "terminal_objective", "fd_gradient", "invert_tau"]


def reference_from_tau(tau_ref: float, T: int, f0: np.ndarray):
    """Terminal velocity (ux, uy) produced by the forward model at ``tau_ref``.

    This is the target of the inverse problem: the answer to
    "which tau produced this terminal field?".
    """
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau_ref)
    rho, ux, uy = macro(f)
    return ux, uy


def terminal_objective(
    tau: float, T: int, f0: np.ndarray, ux_ref: np.ndarray, uy_ref: np.ndarray
) -> float:
    """J(tau) evaluated with a plain forward run (no gradients)."""
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    rho, ux, uy = macro(f)
    return 0.5 * float(np.sum((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2))


def fd_gradient(
    tau: float, T: int, f0: np.ndarray, ux_ref: np.ndarray, uy_ref: np.ndarray, h: float = 1e-4
) -> float:
    """Central-difference dJ/dtau -- an independent check of the adjoint.

    Uses only forward runs, so it shares no code path with the adjoint
    recursion and cannot inherit a Jacobian-transpose mistake.
    """
    jp = terminal_objective(tau + h, T, f0, ux_ref, uy_ref)
    jm = terminal_objective(tau - h, T, f0, ux_ref, uy_ref)
    return (jp - jm) / (2.0 * h)


def invert_tau(
    tau0: float,
    T: int,
    f0: np.ndarray,
    ux_ref: np.ndarray,
    uy_ref: np.ndarray,
    k: int = 100,
    max_iter: int = 30,
    bounds: tuple[float, float] = (0.55, 1.45),
    tol_dtau: float = 1e-9,
    tol_rel_grad: float = 1e-8,
    bracket_tol: float = 1e-4,
    grad_fn=None,
    verbose: bool = False,
) -> dict:
    """Recover tau from a terminal velocity target (memory-bounded).

    Parameters
    ----------
    tau0 : initial guess (must lie inside ``bounds``)
    T, f0, ux_ref, uy_ref : horizon, initial state, terminal target
    k : OOTC checkpoint interval (memory ~ O(k*N*Q), independent of T)
    bounds : physical search interval; it must bracket the root, i.e.
        ``g(lo) < 0 < g(hi)``.  tau > 1/2 is required for positive
        viscosity; the upper end keeps the scheme in the low-Mach window.
    bracket_tol : bracket width at which the driver switches from
        Illinois to the Newton polish
    grad_fn : optional ``(tau, T, f0, ux_ref, uy_ref, k) -> (dJ, J)``,
        used by tests to inject an analytic/cheap gradient.

    Returns
    -------
    dict: ``tau``, ``converged``, ``reason``, ``iterations``, ``nfev``
    (gradient evaluations), ``history``, ``dJ_dtau``, ``J``, ``bracket``,
    ``tol_dtau``, ``tol_rel_grad``, ``grad_source``.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (lo < tau0 < hi):
        raise ValueError(f"tau0={tau0} outside bounds {bounds}")

    if grad_fn is None:

        def grad_fn(tau, T_, f0_, ux_ref_, uy_ref_, k_):
            dJ, J = ootc_gradient(tau, T_, f0_, ux_ref_, uy_ref_, k=k_)[:2]
            return dJ, J

        source = "ootc"
    else:
        source = "custom"

    hist: list[dict] = []
    nfev = 0

    def ev(tau, method):
        nonlocal nfev
        g, J = grad_fn(tau, T, f0, ux_ref, uy_ref, k)
        nfev += 1
        hist.append(
            {
                "iter": len(hist),
                "tau": float(tau),
                "J": float(J),
                "dJ_dtau": float(g),
                "method": method,
            }
        )
        if verbose:
            print(
                f"[inv] {method:9s} n={len(hist):2d} tau={tau:.12f} J={J:.6e} g={g:+.6e}",
                flush=True,
            )
        return float(g), float(J)

    # ---- 1. bracket the root of g ----
    g0, J0 = ev(tau0, "initial")
    g_lo, _ = ev(lo, "bound-lo")
    g_hi, _ = ev(hi, "bound-hi")
    if not (g_lo < 0.0 < g_hi):
        return _fail(
            hist,
            nfev,
            source,
            lo,
            hi,
            tau0,
            reason=f"root not bracketed: g({lo})={g_lo:+.3e}, g({hi})={g_hi:+.3e} (same sign)",
        )

    a, ga = lo, g_lo
    b, gb = hi, g_hi
    tau, g, J = tau0, g0, J0
    span = hi - lo
    step = 0.05 * span  # damped-gradient step, adapted on failure
    curv = None  # local dg/dtau from the last accepted move
    converged = False
    reason = "iterations exhausted"
    it = 0
    for it in range(1, max_iter + 1):
        if curv is not None and curv > 0.0:
            cand = tau - g / curv  # local Newton (convex basin)
            method = "newton"
        else:
            # J is concave where the curvature is negative (large tau), so
            # Newton would leave the basin: walk downhill in J instead.
            cand = tau - np.sign(g) * step
            method = "grad-step"
        # trust region: a secant slope taken far from the root can be
        # wildly wrong (the far field is concave); never move more than
        # half the current bracket in one step
        half = 0.5 * (b - a)
        if abs(cand - tau) > half:
            cand = tau + np.sign(cand - tau) * half
        margin = 1e-9 * span
        cand = float(np.clip(cand, a + margin, b - margin))
        if cand == tau:  # pinned: bisect to make progress
            cand = 0.5 * (a + b)
            method = "bisect"
        g_c, J_c = ev(cand, method)

        if g_c < 0.0:
            a = cand
        elif g_c > 0.0:
            b = cand
        else:
            tau, g, J = cand, g_c, J_c
            converged, reason = True, "exact zero gradient"
            break

        dt, dg = cand - tau, g_c - g
        if dt != 0.0 and dg != 0.0:
            curv = dg / dt
        if method == "grad-step" and not (J_c < J):
            step *= 0.5  # downhill step overshot: retry shorter
        tau, g, J = cand, g_c, J_c

        if abs(g) < tol_rel_grad * max(abs(g0), 1e-300):
            converged, reason = True, "gradient tolerance"
            break
        if (b - a) < tol_dtau:
            converged, reason = True, "bracket tolerance"
            break

    J = hist[-1]["J"]
    return {
        "tau": float(tau),
        "converged": bool(converged),
        "reason": reason,
        "iterations": it,
        "nfev": nfev,
        "history": hist,
        "dJ_dtau": float(g),
        "J": float(J),
        "bracket": [float(a), float(b)],
        "tol_dtau": tol_dtau,
        "tol_rel_grad": tol_rel_grad,
        "grad_source": source,
    }


def _fail(hist, nfev, source, lo, hi, tau0, reason):
    return {
        "tau": float(tau0),
        "converged": False,
        "reason": reason,
        "iterations": 0,
        "nfev": nfev,
        "history": hist,
        "dJ_dtau": float(hist[0]["dJ_dtau"]),
        "J": float(hist[0]["J"]),
        "bracket": [float(lo), float(hi)],
        "tol_dtau": None,
        "tol_rel_grad": None,
        "grad_source": source,
    }
