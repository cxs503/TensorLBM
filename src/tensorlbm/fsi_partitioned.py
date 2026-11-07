"""Partitioned fluid-structure interaction: added-mass instability scaling
and interface quasi-Newton / Aitken stabilisation.

This module isolates the *coupling numerics* of partitioned FSI from any
particular flow solver, so that the added-mass instability can be studied
against closed-form convergence factors.

Model (1-D reference problem)
-----------------------------
A rigid structural degree of freedom (mass ``m_s``, stiffness ``k``, damping
``c``) is wetted by an incompressible fluid which reacts to the interface
acceleration with an added mass ``m_a``.  With ``d`` the interface
displacement and ``F_ext`` the external (e.g. vortex shedding) load, the exact
coupled equilibrium of the interface is

.. math::

    m_s \\ddot d + c \\dot d + k d = F_{ext} - m_a \\ddot d
    \\quad\\Longleftrightarrow\\quad
    (m_s + m_a)\\,\\ddot d + c\\dot d + k d = F_{ext}

A partitioned (staggered) solver replaces this by a fixed-point sub-iteration:
the fluid operator :math:`G` maps the interface acceleration to the
hydrodynamic force

.. math::  G(a) = F_{ext} - m_a a \qquad (\\text{definition of added mass}),

and the structure operator :math:`S` solves the structural equations with a
*fixed-point mass* ``M_iter``:

.. math::  S(F) = \\bigl(F + \\text{known}\\bigr) / M_{eff}.

Because both operators are linear, the interface error contracts with the
exact factor

.. math::

    \\rho(\\omega) = \\bigl| 1 - \\omega\\,(1 + \\beta_{eff}) \\bigr|,
    \\qquad \\beta_{eff} = \\frac{m_a}{M_{eff}},

with ``omega`` the interface relaxation factor (1 = no relaxation).  Two
important configurations are

* ``variant="explicit"``   : ``M_iter = m_s``      -> ``rho = beta`` exactly,
  ``beta_crit = 1``: the classical **added-mass instability** (divergent as
  soon as the added mass exceeds the structural mass).
* ``variant="wet_mass"``   : ``M_iter = m_s + m_a`` -> ``rho = beta/(1+beta)``.
  Convergent for every ``beta`` but ``rho -> 1`` as ``beta -> inf``; the
  remaining amplification is the fraction of the total dynamic mass carried
  by the fluid.  This is the regime in which interface accelerators
  (Aitken / IQN-ILS) are essential.

A third, *consistency preserving* family is provided by the partial-implicit
("theta-split") coupling, see :func:`theta_split_rho`.  There the added-mass
term is split between the structure operator and the fluid force so that the
fixed point is the exact one, :math:`\\rho(\\theta) = (1-\\theta)\\beta /
(1+\\theta\\beta)`; ``theta = 0`` reproduces the explicit scheme and
``theta = 1`` the fully implicit (Robin) coupling with ``rho = 0``.

Accelerators
------------
* :class:`NoRelaxation`          -- plain fixed point (``omega = 1``).
* :class:`ConstantRelaxation`    -- fixed ``omega``.
* :class:`AitkenRelaxation`      -- dynamic Aitken (Irwin & Kesavan 2005),
  vector form of :math:`\\omega_k = -\\omega_{k-1}
  \\frac{\\Delta r_{k-1}\\cdot(\\Delta r_k-\\Delta r_{k-1})}
  {\\lVert\\Delta r_k-\\Delta r_{k-1}\\rVert^2}`.
* :class:`AitkenRatio`           -- the scalar ratio form
  :math:`\\omega_k = \\omega_{k-1}(1-\\Delta r_k/\\Delta r_{k-1})`.
* :class:`IQNILS`                -- interface quasi-Newton with inverse
  Jacobian from least squares (Degroote et al. 2009, 2010).

Both the analytic factors and the numerically measured contraction factors are
exposed so that they can be compared at a tolerance (see
``scripts/run_fsi_partitioned_scaling.py``).

References
----------
Causin, P., Gerbeau, J. F., & Nobile, F. (2005). Added-mass effect in the
    design of partitioned algorithms for fluid-structure problems.
    *Comput. Methods Appl. Mech. Engrg.* 194, 4506-4527.
Forster, C., Wall, W. A., & Ramm, E. (2007). Artificial added mass
    instabilities in sequential staggered coupling of nonlinear structures
    and incompressible viscous flows. *Comput. Methods Appl. Mech. Engrg.*
    196, 1278-1293.
Irwin, K. C., & Kesavan, V. (2005). A simulation framework for
    partitioning-based approach to the analysis of fluid-structure
    interaction mechanisms. *Comput. Struct.* 83, 1177-1193.
Kuttler, U., & Wall, W. A. (2008). Fixed-point fluid-structure interaction
    solvers with dynamic relaxation. *Comput. Mech.* 43, 61-72.
Degroote, J., Bathe, K.-J., & Vierendeels, J. (2009). Performance of a new
    partitioned procedure versus two block-Newton methods in the simulation
    of fluid-structure interaction. *Comput. Fluids* 38, 1202-1213.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

__all__ = [
    "AddedMassModel1D",
    "added_mass_from_channel",
    "analytic_convergence_factor",
    "optimal_relaxation",
    "beta_critical",
    "theta_split_rho",
    "NoRelaxation",
    "ConstantRelaxation",
    "AitkenRelaxation",
    "AitkenRatio",
    "IQNILS",
    "make_accelerator",
    "PartitionedStepResult",
    "solve_partitioned_1d",
    "measure_contraction_1d",
    "detect_beta_critical",
]

Variant = Literal["explicit", "wet_mass"]


# ---------------------------------------------------------------------------
# 1-D added-mass model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddedMassModel1D:
    """One-degree-of-freedom structure wetted by an added mass ``m_added``.

    Attributes
    ----------
    m_struct:
        Structural mass ``m_s`` (kg).
    m_added:
        Added mass ``m_a`` (kg); see :func:`added_mass_from_channel`.
    stiffness:
        Structural stiffness ``k`` (N/m).
    damping:
        Structural damping ``c`` (N s/m).
    """

    m_struct: float
    m_added: float
    stiffness: float = 0.0
    damping: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.m_struct) or self.m_struct <= 0.0:
            raise ValueError(f"m_struct must be positive and finite, got {self.m_struct}")
        if not math.isfinite(self.m_added) or self.m_added < 0.0:
            raise ValueError(f"m_added must be non-negative and finite, got {self.m_added}")
        if self.stiffness < 0.0:
            raise ValueError("stiffness must be non-negative")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")

    # -- dimensionless added-mass ratio -----------------------------------
    @property
    def beta(self) -> float:
        """Added-mass ratio ``beta = m_a / m_s``."""
        return self.m_added / self.m_struct

    # -- analytic factors --------------------------------------------------
    def rho_explicit(self) -> float:
        """``rho = beta`` -- the naive staggered (explicit) interface gain."""
        return analytic_convergence_factor(self.beta, omega=1.0)

    def rho_semi_implicit(self) -> float:
        """``rho = beta/(1+beta)`` -- wet-mass (added-mass-in-solid) gain."""
        return beta_over_one_plus_beta(self.beta)

    def beta_critical(self) -> float:
        """Critical added-mass ratio of the explicit scheme (``beta_crit = 1``)."""
        return beta_critical(self.beta)


def beta_over_one_plus_beta(beta: float) -> float:
    """Return ``beta / (1 + beta)`` (the wet-mass interface gain)."""
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    return beta / (1.0 + beta)


def added_mass_from_channel(
    rho_fluid: float, area: float, length: float
) -> float:
    """Added mass of a 1-D incompressible fluid column: ``m_a = rho_f A L``.

    For a plane channel of length ``L`` and cross-section ``A`` closed by a
    rigid piston, the piston force for a prescribed acceleration ``a`` is
    ``F = -rho_f A L a`` (Causin et al. 2005, section 2), so the added mass is
    ``rho_f A L``.  This is the exact potential-flow result and is used as the
    reference "fluid solver" in the unit tests.
    """
    if rho_fluid <= 0.0 or area <= 0.0 or length <= 0.0:
        raise ValueError("rho_fluid, area and length must be positive")
    return rho_fluid * area * length


def analytic_convergence_factor(beta_eff: float, omega: float = 1.0) -> float:
    """Exact interface contraction factor ``|1 - omega (1 + beta_eff)|``.

    ``beta_eff = m_a / M_eff`` is the added-mass ratio *with respect to the
    mass operator actually used by the structure sub-solve*.
    """
    if beta_eff < 0.0:
        raise ValueError("beta_eff must be non-negative")
    if not (0.0 < omega <= 1.0 + 1e-12):
        raise ValueError("omega must lie in (0, 1]")
    return abs(1.0 - omega * (1.0 + beta_eff))


def optimal_relaxation(beta_eff: float) -> float:
    """Relaxation factor that annihilates the linear interface error.

    ``rho(omega) = |1 - omega (1 + beta_eff)| = 0  =>  omega* = 1/(1+beta_eff)``.
    This is the target value that Aitken relaxation is expected to recover and
    the factor IQN-ILS reproduces implicitly.
    """
    if beta_eff < 0.0:
        raise ValueError("beta_eff must be non-negative")
    return 1.0 / (1.0 + beta_eff)


def beta_critical(beta: float, *, rho_tol: float = 1.0) -> float:
    """Critical added-mass ratio for the *explicit* staggered scheme.

    The error of the unrelaxed explicit iteration grows with factor ``beta``,
    hence the divergence boundary is ``beta_crit = 1`` (``rho_tol = 1``).
    Providing ``rho_tol < 1`` returns the ``beta`` at which ``rho`` first
    exceeds that tolerance, i.e. ``beta_crit = rho_tol``.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not (0.0 < rho_tol <= 1.0):
        raise ValueError("rho_tol must lie in (0, 1]")
    return rho_tol


def theta_split_rho(beta: float, theta: float) -> float:
    """Contraction factor of the consistency-preserving partial-implicit scheme.

    The added-mass term is split as ``m_a = theta m_a + (1-theta) m_a``; the
    ``theta`` part is treated implicitly by the structure operator and the
    remainder explicitly through the fluid force.  The fixed point remains the
    exact one ``(m_s + m_a) a = F_ext`` and, defining
    ``M_eff = m_s + theta m_a``,

    .. math::  \\rho(\\theta) = \\frac{(1-\\theta)\\,m_a}{m_s + \\theta m_a}
               = \\frac{(1-\\theta)\\beta}{1 + \\theta\\beta}.

    ``theta = 0`` -> explicit scheme (``rho = beta``); ``theta = 1`` -> fully
    implicit Robin coupling (``rho = 0``).
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must lie in [0, 1]")
    return (1.0 - theta) * beta / (1.0 + theta * beta)


# ---------------------------------------------------------------------------
# Interface accelerators
# ---------------------------------------------------------------------------


class NoRelaxation:
    """Plain interface fixed point (``omega = 1``)."""

    name = "none"

    def __init__(self) -> None:
        self.omega = 1.0

    def reset(self) -> None:  # pragma: no cover - trivial
        self.omega = 1.0

    def step(self, x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the next interface iterate and the relaxation used."""
        self.omega = 1.0
        return x + r, 1.0


class ConstantRelaxation:
    """Fixed relaxation ``omega``."""

    name = "constant"

    def __init__(self, omega: float = 0.5) -> None:
        if not (0.0 < omega <= 1.0):
            raise ValueError("omega must lie in (0, 1]")
        self.omega = float(omega)

    def reset(self) -> None:  # pragma: no cover - trivial
        return None

    def step(self, x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float]:
        return x + self.omega * r, self.omega


class AitkenRelaxation:
    """Dynamic Aitken relaxation (vector form).

    .. math::

        \\omega_k = -\\omega_{k-1}\\,
        \\frac{r_{k-1}\\cdot\\bigl(r_k-r_{k-1}\\bigr)}
             {\\lVert r_k-r_{k-1}\\rVert^2}

    (Irwin & Kesavan 2005; Kuettler & Wall 2008) with ``r_k = H(x_k) - x_k``
    the interface residual and ``omega_0`` the initial relaxation.  For a
    linear interface operator the recursive secant estimate returns
    ``omega_k = 1/(1+beta_eff)`` exactly after the first update, i.e. the
    relaxation that annihilates the interface error.  The factor is clipped to
    ``[omega_min, omega_max]`` for robustness (default ``[0.05, 1.0]``).
    """

    name = "aitken"

    def __init__(
        self,
        omega0: float = 0.5,
        *,
        omega_min: float = 0.05,
        omega_max: float = 1.0,
    ) -> None:
        if not (0.0 < omega0 <= 1.0):
            raise ValueError("omega0 must lie in (0, 1]")
        self.omega0 = float(omega0)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.reset()

    def reset(self) -> None:
        self.omega = self.omega0
        self._r_prev: np.ndarray | None = None
        self._dr_prev: np.ndarray | None = None

    def step(self, x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float]:
        r = np.asarray(r, dtype=float)
        if self._r_prev is not None:
            dr = r - self._r_prev
            denom = float(dr @ dr)
            if denom > 1e-300:
                omega = -self.omega * float(self._r_prev @ dr) / denom
                if math.isfinite(omega) and omega > 0.0:
                    self.omega = min(max(omega, self.omega_min), self.omega_max)
        self._r_prev = r.copy()
        return x + self.omega * r, self.omega


class AitkenRatio:
    """Scalar ratio form of Aitken: ``omega_k = omega_{k-1}(1 - dr_k/dr_{k-1})``.

    Kept for comparison with :class:`AitkenRelaxation`; it is *not* the
    standard dynamic relaxation and does not converge to ``1/(1+beta)`` in
    general (see ``docs/fsi_partitioned_added_mass.md``).
    """

    name = "aitken_ratio"

    def __init__(self, omega0: float = 0.5, *, omega_min: float = 0.05, omega_max: float = 1.0) -> None:
        self.omega0 = float(omega0)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.reset()

    def reset(self) -> None:
        self.omega = self.omega0
        self._dr_prev: float | None = None

    def step(self, x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float]:
        r = np.asarray(r, dtype=float)
        dr = float(np.linalg.norm(r))
        if self._dr_prev is not None and self._dr_prev > 1e-300:
            omega = self.omega * (1.0 - dr / self._dr_prev)
            if math.isfinite(omega):
                self.omega = min(max(abs(omega), self.omega_min), self.omega_max)
        self._dr_prev = dr
        return x + self.omega * r, self.omega


class IQNILS:
    """Interface quasi-Newton with inverse Jacobian from least squares.

    The interface equation solved in the sub-iteration is ``r(x) = H(x) - x``
    with ``H`` the composition of the fluid and structure operators.  Given
    stored pairs ``(dx_j, dr_j = r(x+dx_j) - r(x))`` and the current residual
    ``r_k``, IQN-ILS seeks coefficients ``c`` minimising

    .. math::  \\lVert r_k - \\sum_j c_j \\Delta r_j \\rVert_2

    and updates ``x_{k+1} = x_k - sum_j c_j dx_j``.  For a linear interface
    operator a single stored pair already gives the exact secant, so the
    method converges in very few iterations (this is the well known property
    exploited in Degroote et al. 2009).

    Parameters
    ----------
    mixing:
        Optional blend ``x_{k+1} = x_k - mixing * dx`` (safety factor, ``1``
        by default = pure quasi-Newton).
    max_pairs:
        Maximum number of stored secant pairs (recycling of the oldest).
    drop_tol:
        Columns of the residual-difference matrix with norm below
        ``drop_tol * ||r_k||`` are discarded (rank filtering).
    """

    name = "iqn_ils"

    def __init__(
        self,
        *,
        mixing: float = 1.0,
        max_pairs: int = 20,
        drop_tol: float = 1e-10,
        step_limit: float | None = None,
    ) -> None:
        if not (0.0 < mixing <= 1.0):
            raise ValueError("mixing must lie in (0, 1]")
        if max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        self.mixing = float(mixing)
        self.max_pairs = int(max_pairs)
        self.drop_tol = float(drop_tol)
        self.step_limit = step_limit
        self.reset()

    def reset(self) -> None:
        self._xs: list[np.ndarray] = []
        self._rs: list[np.ndarray] = []
        self.omega = 1.0
        self.last_rank = 0

    # -- helpers -----------------------------------------------------------
    def _pairs(self) -> tuple[np.ndarray, np.ndarray]:
        dX = np.column_stack([self._xs[i + 1] - self._xs[i] for i in range(len(self._xs) - 1)])
        dR = np.column_stack([self._rs[i + 1] - self._rs[i] for i in range(len(self._rs) - 1)])
        return dX, dR

    def step(self, x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(x, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        # record the new sample (residual of the *current* iterate)
        self._xs.append(x.copy())
        self._rs.append(r.copy())
        if len(self._xs) > self.max_pairs + 1:
            self._xs.pop(0)
            self._rs.pop(0)

        if len(self._xs) < 2:
            self.last_rank = 0
            self.omega = 1.0
            return x + r, 1.0

        dX, dR = self._pairs()
        r_norm = float(np.linalg.norm(r))
        keep = [j for j in range(dR.shape[1]) if np.linalg.norm(dR[:, j]) > self.drop_tol * max(r_norm, 1e-300)]
        if not keep:
            self.last_rank = 0
            return x + r, 1.0
        dR = dR[:, keep]
        dX = dX[:, keep]
        c, *_ = np.linalg.lstsq(dR, r, rcond=None)
        self.last_rank = int(np.linalg.matrix_rank(dR))
        dx = dX @ c
        dx = self.mixing * dx
        if self.step_limit is not None:
            dx_norm = float(np.linalg.norm(dx))
            limit = self.step_limit * max(float(np.linalg.norm(x)), 1e-300)
            if dx_norm > limit > 0.0:
                dx *= limit / dx_norm
        # effective relaxation implied by the quasi-Newton step
        dx_r = float(dx @ r) / max(float(r @ r), 1e-300)
        self.omega = abs(dx_r) if math.isfinite(dx_r) else 1.0
        return x - dx, self.omega


def make_accelerator(name: str, **kwargs) -> object:
    """Factory for the accelerators (``"none"``, ``"constant"``, ``"aitken"``,
    ``"aitken_ratio"``, ``"iqn_ils"``)."""
    table = {
        "none": NoRelaxation,
        "constant": ConstantRelaxation,
        "aitken": AitkenRelaxation,
        "aitken_ratio": AitkenRatio,
        "iqn_ils": IQNILS,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unknown accelerator {name!r}; choose from {sorted(table)}")
    return table[key](**kwargs)


# ---------------------------------------------------------------------------
# Partitioned 1-D solver
# ---------------------------------------------------------------------------


@dataclass
class PartitionedStepResult:
    """Result of a time-accurate partitioned run (1-DOF structure)."""

    t: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    iterations: np.ndarray
    step_residuals: np.ndarray
    residual_history: list[list[float]] = field(default_factory=list)
    omega_history: list[list[float]] = field(default_factory=list)
    converged: bool = True
    effective_beta: float = 0.0

    @property
    def max_iterations(self) -> int:
        return int(self.iterations.max()) if self.iterations.size else 0


def _structure_mass(
    model: AddedMassModel1D,
    dt: float,
    variant: Variant,
    gamma: float,
    beta_n: float,
) -> float:
    """Fixed-point mass ``M_iter`` of the structure sub-solve (Newmark)."""
    if variant == "explicit":
        m_iter = model.m_struct
    elif variant == "wet_mass":
        m_iter = model.m_struct + model.m_added
    else:  # pragma: no cover - guarded by type hints
        raise ValueError(f"unknown variant {variant!r}")
    return m_iter + model.damping * gamma * dt + model.stiffness * beta_n * dt * dt


def solve_partitioned_1d(
    model: AddedMassModel1D,
    *,
    n_steps: int,
    dt: float,
    variant: Variant = "explicit",
    forcing: Callable[[float], float] | None = None,
    accelerator: object | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    gamma: float = 0.5,
    beta_n: float = 0.25,
    a_init: float | None = None,
) -> PartitionedStepResult:
    """Time-accurate partitioned solve of the 1-DOF added-mass model.

    Each time step performs a Dirichlet-Neumann sub-iteration:

    1. ``a^k`` (interface acceleration) is prescribed to the fluid, which
       returns ``F^k = F_ext - m_a a^k`` (exact added-mass fluid).
    2. The structure is advanced with the implicit Newmark scheme using
       ``F^k`` and the fixed-point mass ``M_iter`` (see ``variant``).
    3. The interface residual ``r = a^{k+1} - a^k`` is passed to the
       accelerator which produces the next interface iterate.

    Parameters
    ----------
    model:
        Structural/added-mass model.
    variant:
        ``"explicit"`` (``M_iter = m_s``, the naive staggered scheme) or
        ``"wet_mass"`` (``M_iter = m_s + m_a``).
    forcing:
        External load ``F_ext(t)``; defaults to zero.
    accelerator:
        One of the objects from :func:`make_accelerator`; ``None`` keeps the
        plain fixed point.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not (0.0 < gamma <= 1.0) or not (0.0 <= beta_n <= 1.0):
        raise ValueError("invalid Newmark parameters")

    # Newmark effective mass: M_iter + c*gamma*dt + k*beta_N*dt^2 (already
    # contained in _structure_mass), so rho = m_a / m_eff is the exact
    # single-DOF interface gain.
    m_eff = _structure_mass(model, dt, variant, gamma, beta_n)
    beta_eff = model.m_added / m_eff if m_eff > 0.0 else float("inf")

    acc = accelerator
    if acc is None:
        acc = NoRelaxation()

    d = 0.0
    v = 0.0
    a_prev = 0.0
    if a_init is not None:
        a_prev = float(a_init)

    ts, ds, vs, accs, iters, step_res = [], [], [], [], [], []
    res_hist: list[list[float]] = []
    om_hist: list[list[float]] = []
    converged = True

    for n in range(n_steps):
        t_next = (n + 1) * dt
        f_ext = float(forcing(t_next)) if forcing is not None else 0.0

        # known (explicit) part of the Newmark update
        c_known = -(
            model.damping * (v + dt * (1.0 - gamma) * a_prev)
            + model.stiffness * (d + dt * v + dt * dt * (0.5 - beta_n) * a_prev)
        )

        x = np.array([a_prev], dtype=float)
        if hasattr(acc, "reset"):
            acc.reset()

        resid_series: list[float] = []
        omega_series: list[float] = []
        step_conv = False
        a_new = a_prev
        for k in range(max_iter):
            a_k = float(x[0])
            f_fluid = f_ext - model.m_added * a_k      # fluid operator (added mass)
            a_s = (f_fluid + c_known) / m_eff          # structure operator (Newmark)
            r = np.array([a_s - a_k], dtype=float)
            resid_series.append(float(abs(r[0])))
            if abs(float(r[0])) < tol:
                step_conv = True
                a_new = a_s
                break
            x_new, omega = acc.step(x, r)
            omega_series.append(float(omega))
            if not np.all(np.isfinite(x_new)) or abs(float(x_new[0])) > 1e12:
                step_conv = False
                a_new = float(x_new[0])
                x = x_new
                break
            x = x_new
            a_new = float(x[0])
        else:
            step_conv = False

        a_next = float(a_new)
        # Newmark kinematics with the converged (or last) acceleration
        d_next = d + dt * v + dt * dt * ((0.5 - beta_n) * a_prev + beta_n * a_next)
        v_next = v + dt * ((1.0 - gamma) * a_prev + gamma * a_next)

        ts.append(t_next)
        ds.append(d_next)
        vs.append(v_next)
        accs.append(a_next)
        iters.append(len(resid_series))
        step_res.append(resid_series[-1] if resid_series else 0.0)
        res_hist.append(resid_series)
        om_hist.append(omega_series)
        converged = converged and step_conv

        d, v, a_prev = d_next, v_next, a_next

    return PartitionedStepResult(
        t=np.asarray(ts),
        displacement=np.asarray(ds),
        velocity=np.asarray(vs),
        acceleration=np.asarray(accs),
        iterations=np.asarray(iters),
        step_residuals=np.asarray(step_res),
        residual_history=res_hist,
        omega_history=om_hist,
        converged=converged,
        effective_beta=beta_eff,
    )


# ---------------------------------------------------------------------------
# Numerical measurement of the contraction factor / critical beta
# ---------------------------------------------------------------------------


def measure_contraction_1d(
    model: AddedMassModel1D,
    *,
    dt: float,
    variant: Variant = "explicit",
    accelerator: object | None = None,
    a_init: float = 1.0,
    n_iter: int = 40,
    skip: int = 3,
    window: int = 8,
) -> tuple[float, list[float], list[float]]:
    """Numerically measure the interface contraction factor ``rho``.

    A single sub-iteration problem is set up (as in the first time step of
    :func:`solve_partitioned_1d`, ``F_ext = 0``, ``d = v = 0``) and started
    from a perturbed interface guess ``a_init``.  The measured factor is the
    geometric mean of the residual ratios over an asymptotic window,

    .. math::  \\rho_{num} = \\exp\\Bigl(\\frac{1}{W}\\sum_{i}
               \\ln \\frac{\\lVert r_{i+1}\\rVert}{\\lVert r_i\\rVert}\\Bigr).

    Returns ``(rho_num, residual_history, omega_history)``.
    """
    if n_iter < skip + window + 1:
        raise ValueError("n_iter too small for the requested window")
    m_eff = _structure_mass(model, dt, variant, 0.5, 0.25)
    acc = accelerator if accelerator is not None else NoRelaxation()
    if hasattr(acc, "reset"):
        acc.reset()

    x = np.array([float(a_init)])
    res: list[float] = []
    oms: list[float] = []
    for _ in range(n_iter):
        a_k = float(x[0])
        f_fluid = -model.m_added * a_k          # F_ext = 0
        a_s = f_fluid / m_eff                   # c_known = 0 at the first step
        r = np.array([a_s - a_k])
        res.append(abs(float(r[0])))
        x, omega = acc.step(x, r)
        oms.append(float(omega))
    ratios: list[float] = []
    exact_convergence = False
    for i in range(len(res) - 1):
        if res[i] <= 1e-300:
            break
        if res[i + 1] <= 1e-300:
            exact_convergence = True
            break
        ratios.append(res[i + 1] / res[i])
    tail = ratios[skip: skip + window]
    if not tail:
        tail = ratios[-window:]
    if not tail:
        rho = 0.0 if exact_convergence else float("nan")
    else:
        rho = float(math.exp(sum(math.log(v) for v in tail) / len(tail)))
        if exact_convergence:
            rho = 0.0
    return rho, res, oms


def detect_beta_critical(
    *,
    dt: float = 0.01,
    m_struct: float = 1.0,
    variant: Variant = "explicit",
    beta_lo: float = 0.1,
    beta_hi: float = 8.0,
    tol: float = 1e-4,
    max_iter: int = 200,
    n_refine: int = 30,
) -> dict:
    """Locate the numerically observed divergence boundary in ``beta``.

    For each ``beta`` the sub-iteration is run with the plain fixed point; the
    step is classified as *divergent* when the residual does not fall below
    ``tol`` within ``max_iter`` or the interface variable leaves the
    physically admissible range.  A bisection refines the transition,
    giving the measured ``beta_crit`` to compare with the analytic value
    ``beta_crit = 1`` (explicit scheme).
    """
    def diverges(beta: float) -> bool:
        model = AddedMassModel1D(m_struct=m_struct, m_added=beta * m_struct)
        rho, res, _ = measure_contraction_1d(
            model, dt=dt, variant=variant, n_iter=30, skip=2, window=10
        )
        return (not math.isfinite(rho)) or rho >= 1.0

    if diverges(beta_lo):
        return {"beta_crit": beta_lo, "bracket": [beta_lo, beta_hi], "monotone": False}
    if not diverges(beta_hi):
        return {"beta_crit": beta_hi, "bracket": [beta_lo, beta_hi], "monotone": True}

    lo, hi = beta_lo, beta_hi
    for _ in range(n_refine):
        mid = 0.5 * (lo + hi)
        if diverges(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return {
        "beta_crit": 0.5 * (lo + hi),
        "bracket": [lo, hi],
        "analytic": 1.0,
    }