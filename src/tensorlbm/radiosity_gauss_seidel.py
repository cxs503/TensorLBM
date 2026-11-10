"""Gauss-Seidel iterative radiosity solver for gray-diffuse enclosures.

Why an iterative solver next to the direct one
----------------------------------------------
:class:`tensorlbm.gray_enclosure_radiosity.GrayEnclosure` solves
``(I - R F) J = e`` with a dense LU factorisation.  That is the right tool for
a few dozen surfaces but it is O(N^3), hides the iteration physics, and gives
no handle on the convergence criterion that the ship-deck / cargo-tank
radiation problems actually need (large enclosures, temperature-dependent
emissivity, coupling to a conduction field).

This module solves the same system with **Gauss-Seidel sweeps**

.. math::
    J_i^{(k+1)} = (1-\omega) J_i^{(k)} + \omega\Bigl[
        \varepsilon_i \sigma T_i^4 + (1-\varepsilon_i)
        \sum_{j<i} F_{ij} J_j^{(k+1)} + (1-\varepsilon_i)
        \sum_{j>i} F_{ij} J_j^{(k)} \Bigr],

with optional over/under-relaxation :math:`\omega`, an explicit convergence
certificate and a spectral-radius diagnostic that predicts the convergence
rate before the first sweep.

Convergence theory (and why it is certifiable)
----------------------------------------------
``R F`` with ``R = diag(1 - eps)`` is non-negative and, for a closed enclosure
whose view factors obey the summation rule, row-substochastic with row sums
``1 - eps_i``.  As soon as one surface is not a perfect reflector
(``eps_i > 0``) at least one row is strictly sub-stochastic and
``rho(R F) <= max_i (1 - eps_i) < 1``: the fixed-point iteration is a
contraction and Gauss-Seidel converges for every initial guess.  The
degenerate ``eps_i = 0`` for **all** surfaces (a pure reradiating cavity) makes
``I - R F`` singular, which is *detected and reported* instead of iterated
forever: the radiosity of such a cavity is determined only up to an additive
constant, while the net flux ``q = 0`` is well defined.

Reradiating (adiabatic, ``T = None``) surfaces enter unchanged:
``eps = 0`` gives ``R_i = 1`` and ``e_i = 0``, i.e. the row ``q_i = 0``.

Audits returned with every solution
-----------------------------------
* ``final_residual`` — ``max_i |J_i^{(k+1)} - J_i^{(k)}| / max_i |J_i^{(k+1)}|``
  (the machine criterion of the acceptance gate, default ``<= 1e-6``),
* ``algebraic_residual`` — ``||(I - R F)J - e||_inf / ||e||_inf``,
* ``energy_residual`` — ``|sum_i A_i q_i| / sum_i A_i \u03b5_i \u03c3 T_i^4``, the
  power closure of the enclosure relative to its emitted power (zero
  analytically; the isothermal-cavity capstone test).  Note that normalising
  by ``sum A|q|`` degenerates when q -> 0 (isothermal cavity), so the emitted
  power is used instead,
* ``spectral_radius`` — ``rho(R F)``, the asymptotic contraction factor.

References
----------
Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed., ch. 5, 10.
Incropera et al. (2007). *Fundamentals of Heat and Mass Transfer*, 6th ed.,
    ch. 13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gray_enclosure_radiosity import STEFAN_BOLTZMANN, GrayEnclosure

__all__ = [
    "GaussSeidelSolution",
    "enclosure_vectors",
    "iteration_matrix",
    "spectral_radius",
    "solve_radiosity_gauss_seidel",
]


def enclosure_vectors(enclosure: GrayEnclosure) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(A, eps, T, F)`` arrays rebuilt from the enclosure's surface list.

    ``T`` is ``nan`` for reradiating (adiabatic) surfaces, matching
    ``GrayEnclosure``'s internal convention.
    """
    areas = np.array([float(s.area) for s in enclosure.surfaces], dtype=np.float64)
    eps = np.array([float(s.emissivity) for s in enclosure.surfaces], dtype=np.float64)
    T = np.array(
        [float(s.temperature) if s.temperature is not None else np.nan for s in enclosure.surfaces],
        dtype=np.float64,
    )
    F = np.asarray(enclosure.view_factors, dtype=np.float64)
    return areas, eps, T, F


def iteration_matrix(enclosure: GrayEnclosure) -> np.ndarray:
    """``M = diag(1 - eps) F`` — the fixed-point operator of the radiosity system."""
    _, eps, _, F = enclosure_vectors(enclosure)
    return (1.0 - eps)[:, None] * F


def spectral_radius(enclosure: GrayEnclosure) -> float:
    """``rho(diag(1-eps) F)`` — asymptotic Gauss-Seidel contraction factor.

    For a closed enclosure this is bounded by ``max_i (1 - eps_i)``; equality
    holds only in the special case of a cavity whose reflective row does not
    couple (open enclosure).  A value of exactly 1.0 flags the singular
    all-reradiating cavity.
    """
    M = iteration_matrix(enclosure)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


@dataclass
class GaussSeidelSolution:
    """Result of a Gauss-Seidel radiosity sweep sequence."""

    radiosity: np.ndarray  # J_i [W/m2]
    irradiation: np.ndarray  # G_i = sum_j F_ij J_j [W/m2]
    net_flux: np.ndarray  # q_i = J_i - G_i [W/m2], positive = leaving
    iterations: int
    final_residual: float  # max|dJ| / max|J|
    algebraic_residual: float  # ||(I-RF)J - e||inf / ||e||inf
    energy_residual: float  # |sum A q| / sum A eps sigma T^4 (emitted power)
    closure_residual: float  # max|q - eps(sigmaT^4 - G)| / max emitted
    spectral_radius: float
    relaxation: float
    tol: float
    converged: bool
    residual_history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        """JSON-friendly view (arrays as lists, history truncated to 25 samples)."""
        hist = self.residual_history
        if len(hist) > 25:
            idx = np.unique(np.linspace(0, len(hist) - 1, 25).astype(int))
            hist = [hist[i] for i in idx]
        return {
            "radiosity": [float(v) for v in self.radiosity],
            "irradiation": [float(v) for v in self.irradiation],
            "net_flux": [float(v) for v in self.net_flux],
            "iterations": int(self.iterations),
            "final_residual": float(self.final_residual),
            "algebraic_residual": float(self.algebraic_residual),
            "energy_residual": float(self.energy_residual),
            "closure_residual": float(self.closure_residual),
            "spectral_radius": float(self.spectral_radius),
            "relaxation": float(self.relaxation),
            "tol": float(self.tol),
            "converged": bool(self.converged),
            "residual_history": [float(v) for v in hist],
        }


def solve_radiosity_gauss_seidel(
    enclosure: GrayEnclosure,
    tol: float = 1e-6,
    max_iter: int = 20000,
    relaxation: float = 1.0,
    initial: np.ndarray | None = None,
    record_history: bool = True,
) -> GaussSeidelSolution:
    r"""Solve ``J = e + R F J`` by Gauss-Seidel sweeps.

    Parameters
    ----------
    enclosure : GrayEnclosure
        Surfaces and view factors (the same object the direct solver accepts;
        its summation / reciprocity audits run on construction).
    tol : float
        Convergence threshold on the *iteration* residual
        ``max|J^{k+1} - J^k| / max|J^{k+1}|``.  The MISSION criterion is
        ``tol <= 1e-6``.
    max_iter : int
        Sweep cap; exceeding it returns ``converged = False`` (honest failure,
        no exception) so callers can report the residual they actually got.
    relaxation : float
        ``omega`` in ``(0, 2)``; ``1`` is plain Gauss-Seidel.
    initial : array, optional
        Initial radiosity guess; defaults to the black-body emission
        ``eps sigma T^4`` (a physical, non-zero start).
    record_history : bool
        Keep the per-sweep residual sequence.

    Raises
    ------
    ValueError
        If the enclosure is a pure reradiating cavity (``eps_i = 0`` for all
        ``i``): ``I - R F`` is singular and the radiosity is not unique.
    """
    if not (0.0 < relaxation < 2.0):
        raise ValueError(f"relaxation must be in (0, 2), got {relaxation!r}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and > 0, got {tol!r}")

    A, eps, T, F = enclosure_vectors(enclosure)
    n = F.shape[0]
    if np.all(eps <= 0.0):
        raise ValueError(
            "all surfaces are reradiating (eps = 0): (I - R F) is singular and the "
            "radiosity is determined only up to an additive constant; the net flux "
            "is identically zero and there is nothing to iterate"
        )
    emitted = np.where(np.isnan(T), 0.0, eps * STEFAN_BOLTZMANN * np.nan_to_num(T) ** 4)
    R = 1.0 - eps

    J = np.array(emitted, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    if J.shape != (n,):
        raise ValueError(f"initial shape {J.shape} does not match ({n},)")
    if not np.all(np.isfinite(J)):
        raise ValueError("initial radiosity contains non-finite entries")

    history: list[float] = []
    converged = False
    iterations = 0
    res = math.inf
    for iterations in range(1, max_iter + 1):
        J_old = J.copy()
        for i in range(n):
            # Gauss-Seidel with the diagonal eliminated exactly: the self-view
            # term F_ii J_i of a concave surface is solved, not lagged.
            denom = 1.0 - R[i] * F[i, i]
            if abs(denom) < 1.0e-12:
                raise ValueError(
                    f"surface {i}: 1 - (1-eps) F_ii = {denom:.3e} ~ 0; the radiosity "
                    "equation is singular for this row"
                )
            G_off = float(F[i, :i] @ J[:i]) + float(F[i, i + 1 :] @ J_old[i + 1 :])
            target = (emitted[i] + R[i] * G_off) / denom
            J[i] = (1.0 - relaxation) * J_old[i] + relaxation * target
        scale = float(np.max(np.abs(J)))
        res = float(np.max(np.abs(J - J_old))) / (scale if scale > 0.0 else 1.0)
        if record_history:
            history.append(res)
        if res <= tol:
            converged = True
            break

    G = F @ J
    q = J - G
    e_norm = float(np.max(np.abs(emitted)))
    algebraic = float(np.max(np.abs(J - emitted - (R * G)))) / (e_norm if e_norm > 0.0 else 1.0)
    power = float(np.sum(A * q))
    abs_power = float(np.sum(A * np.abs(q)))
    # Normalise by the *emitted* power, the physical scale of the exchange.
    # Normalising by sum(A|q|) degenerates for an isothermal cavity where q -> 0
    # analytically (0/0 noise amplified to O(1) -- see the report's trap list).
    emitted_power = float(np.sum(A * emitted))
    denom = emitted_power if emitted_power > 0.0 else abs_power
    energy = abs(power) / denom if denom > 0.0 else 0.0

    sigma_t4 = np.where(np.isnan(T), np.nan, STEFAN_BOLTZMANN * np.nan_to_num(T) ** 4)
    with np.errstate(invalid="ignore"):
        closure = np.abs(q - eps * (sigma_t4 - G))
    closure = float(np.nanmax(closure)) if np.any(~np.isnan(closure)) else 0.0
    closure /= e_norm if e_norm > 0.0 else 1.0

    return GaussSeidelSolution(
        radiosity=J,
        irradiation=G,
        net_flux=q,
        iterations=iterations,
        final_residual=res,
        algebraic_residual=algebraic,
        energy_residual=energy,
        closure_residual=closure,
        spectral_radius=spectral_radius(enclosure),
        relaxation=float(relaxation),
        tol=float(tol),
        converged=converged,
        residual_history=history,
    )