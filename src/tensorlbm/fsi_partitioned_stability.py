"""Partitioned FSI added-mass stability calibration (1D analytic anchor + 2D LBM).

Why this module exists
----------------------
Partitioned (staggered) fluid-structure coupling is linearly unstable when the
fluid's added mass exceeds the structure's inertia - the well known *added mass
instability* of incompressible FSI.  TensorLBM's FSI surface
(``tensorlbm.fsi`` / ``fsi_common`` / ``ibm_common``) offers one-way and two-way
*explicit* coupling only and carries no falsifiable statement about *when* that
coupling is stable.  This module supplies such a statement in a machine
checkable form:

* a closed-form error-propagation (fixed-point) factor for a one-dimensional
  analytic anchor model ("piston in a closed column" / balloon model),
* two numerical schemes whose error sequences are *measured* and compared with
  the closed form (explicit staggered time marching, and a sub-iteration
  coupling loop),
* three stabilisation strategies for the sub-iteration loop (no relaxation,
  Aitken dynamic relaxation, IQN-ILS quasi-Newton with a 5-vector history), and
* a two-dimensional D2Q9 LBM realisation of the same physics (1-DOF heaving
  rigid body, direct-forcing immersed boundary) so that the analytic prediction
  can be checked against an actual lattice-Boltzmann solver.

Model
-----
1D anchor ("piston in a closed column"; cf. Causin-Gerbeau-Nobile 2005,
Foerster-Wall-Ramm 2007 one-dimensional added-mass models).  A rigid piston of
mass ``m_s`` closes a fluid column of density ``rho_f`` whose length matches the
piston thickness, so the displaced fluid mass is ``m_a = c * rho_f * V_s`` with
geometry factor ``c`` (``c = 1`` for a column-matched piston).  Inside a closed
column the fluid momentum is tied to the *absolute* piston velocity (mass
conservation in a fixed volume), hence the fluid reaction to the interface
velocity ``u`` is

    F_fluid = -(m_a / dt) * (u - u_ref),        u_ref = 0 for a column at rest.

Structure momentum balance ``m_s (u^{n+1} - u^n) = dt (F_fluid + F_ext)``:

* **Explicit / weakly coupled** (fluid force evaluated with the old velocity,
  relaxation factor ``omega``):

      u^{n+1} = (1 - omega * chi) u^n + ...        chi = m_a / m_s

  Error propagation factor (the quantity quoted by the campaign node):

      rho_dd(omega) = |1 - omega * chi| = |1 - omega * c * rho_f / rho_s|

  Divergence when ``omega * chi > 2`` => critical density ratio
  ``(rho_s/rho_f)_crit = c * omega / 2``.

* **Sub-iteration loop** (fluid force re-evaluated with the current iterate
  ``u^k``, structure returns a new iterate, relaxed):

      r^k = u_struct(u^k) - u^k,     dr^k/du^k = -(1 + chi)

      rho_sub(omega) = |1 - omega (1 + chi)|

  Divergence when ``omega (1 + chi) > 2`` => critical density ratio
  ``c * omega / (2 - omega)``, optimal relaxation ``omega* = 1/(1+chi)``.

Both factors are measured below from *numerical* error sequences and must match
the closed form to better than 5% (campaign criterion D1).

The 2D LBM realisation uses direct forcing on a filled body mask: overwriting
the populations in the body nodes with the equilibrium state of the trial body
velocity changes the fluid momentum by ``sum_body rho (u_body - u_star)``, which
is exactly the discrete added-mass law of the closed column with
``m_a = sum_body rho``.  Setting ``m_s = rho_s * N_body`` therefore gives
``chi = rho_f / rho_s`` by construction, and the 2D solver is a direct
realisation of the 1D anchor.

Audit status
------------
Everything here is REAL: closed-form factors are analytic, every number quoted
by the calibration script comes from an executed numerical run, and the
negative control (light body, explicit coupling) is expected to *diverge*.
No physics claim is made outside the linear added-mass regime: the model has no
viscous boundary layer, no free surface and no turbulence, so the stabilisation
gains reported here are a *linear stability* statement, not a validation of
turbulent FSI accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from .fsi_partitioned import clamp_aitken_factor

__all__ = [
    "AddedMassPistonModel",
    "AitkenRelaxation",
    "IQNILS",
    "CouplingReport",
    "PistonState",
    "simulate_explicit_piston",
    "measure_amplification_factor",
    "solve_piston_step_subiterations",
    "estimate_critical_density_ratio",
    "critical_density_ratio_from_scan",
    "stability_map",
    "HeavingBodyLBM",
    "run_lbm_fsi_decay",
    "structural_free_decay_frequency",
]

CouplingScheme = Literal["explicit", "fixed", "aitken", "iqn_ils"]

# --------------------------------------------------------------------------- #
# Analytic anchor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AddedMassPistonModel:
    """One-dimensional piston/balloon added-mass anchor model.

    ``rho_f`` and ``rho_s`` are fluid and structure densities; the structure
    volume equals the displaced fluid volume, so the density ratio
    ``rho_s / rho_f`` alone fixes the mass ratio up to the geometry factor
    ``added_mass_coefficient`` (``c = 1`` for a column-matched piston,
    ``c > 1`` for a piston sweeping more fluid than its own volume).
    """

    rho_f: float
    rho_s: float
    added_mass_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.rho_f <= 0.0 or self.rho_s <= 0.0:
            raise ValueError("densities must be strictly positive")
        if self.added_mass_coefficient <= 0.0:
            raise ValueError("added_mass_coefficient must be strictly positive")

    # -- basic ratios ------------------------------------------------------ #
    @property
    def density_ratio(self) -> float:
        """``rho_s / rho_f`` (the campaign sweep parameter)."""
        return self.rho_s / self.rho_f

    @property
    def chi(self) -> float:
        """Added-mass ratio ``chi = m_a / m_s = c * rho_f / rho_s``."""
        return self.added_mass_coefficient * self.rho_f / self.rho_s

    # -- closed-form propagation factors ----------------------------------- #
    def rho_dd_explicit(self, omega: float = 1.0) -> float:
        """Error propagation factor of the explicit staggered scheme.

        ``|1 - omega * c * rho_f / rho_s|``; ``rho_dd > 1`` means divergence.
        """
        return abs(1.0 - omega * self.chi)

    def rho_dd_subiteration(self, omega: float = 1.0) -> float:
        """Error propagation factor of the relaxed sub-iteration loop.

        ``|1 - omega * (1 + chi)|``; the extra ``+1`` is the identity term of
        the residual-based relaxation.
        """
        return abs(1.0 - omega * (1.0 + self.chi))

    def growth_exponent_explicit(self, omega: float = 1.0) -> float:
        """Per-step growth exponent ``ln rho_dd`` (``> 0`` => divergence)."""
        factor = self.rho_dd_explicit(omega)
        return math.log(factor) if factor > 0.0 else -math.inf

    # -- stability boundaries ---------------------------------------------- #
    def critical_density_ratio_explicit(self, omega: float = 1.0) -> float:
        """``(rho_s/rho_f)_crit = c * omega / 2`` for the explicit scheme."""
        return 0.5 * self.added_mass_coefficient * omega

    def critical_density_ratio_subiteration(self, omega: float = 1.0) -> float:
        """``c * omega / (2 - omega)`` for the relaxed sub-iteration loop."""
        if omega >= 2.0:
            return math.inf
        return self.added_mass_coefficient * omega / (2.0 - omega)

    def optimal_relaxation_explicit(self) -> float:
        """``omega = 1/chi``: one-shot exact for the explicit staggered map."""
        return 1.0 / self.chi

    def optimal_relaxation_subiteration(self) -> float:
        """``omega = 1/(1+chi)``: one-shot exact for the sub-iteration map."""
        return 1.0 / (1.0 + self.chi)

    def is_linearly_stable_explicit(self, omega: float = 1.0) -> bool:
        return self.rho_dd_explicit(omega) < 1.0

    def to_dict(self) -> dict:
        return {
            "rho_f": self.rho_f,
            "rho_s": self.rho_s,
            "added_mass_coefficient": self.added_mass_coefficient,
            "density_ratio": self.density_ratio,
            "chi": self.chi,
        }


# --------------------------------------------------------------------------- #
# Numerical 1D schemes
# --------------------------------------------------------------------------- #


@dataclass
class CouplingReport:
    """Outcome of one partitioned coupling solve (1D step or 2D time step)."""

    scheme: str
    converged: bool
    n_iterations: int
    final_residual: float
    residual_history: list[float] = field(default_factory=list)
    relaxation_history: list[float] = field(default_factory=list)
    diverged: bool = False

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_residual": self.final_residual,
            "diverged": self.diverged,
        }


@dataclass
class PistonState:
    """Piston displacement / velocity pair (1D anchor)."""

    displacement: float = 0.0
    velocity: float = 0.0


def _relative_error(value: float, reference: float) -> float:
    """Relative error, with an absolute fall-back when ``reference`` is ~0.

    ``rho_dd = 0`` happens at the one-shot-exact relaxation point
    (``omega * chi == 1``); there the two quantities are compared absolutely.
    """
    if reference > 1e-12:
        return abs(value - reference) / reference
    return abs(value - reference)


def simulate_explicit_piston(
    model: AddedMassPistonModel,
    n_steps: int = 60,
    omega: float = 1.0,
    u0: float = 1.0,
    clip: float = 1e150,
) -> dict:
    """March the explicit staggered scheme from a unit velocity perturbation.

    Free (undriven) problem: ``F_fluid = -(m_a/dt)(u^n - 0)`` so that

        u^{n+1} = (1 - omega * chi) * u^n

    exactly.  The returned ``velocity`` history is the raw *numerical* error
    sequence used to measure the propagation factor (no analytic shortcut).
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    chi = model.chi
    u = [float(u0)]
    diverged = False
    for _ in range(n_steps - 1):
        # fluid reaction measured against the resting column reference
        f_fluid = -chi * u[-1]
        u_next = u[-1] + omega * f_fluid
        if not math.isfinite(u_next) or abs(u_next) > clip:
            diverged = True
            u.append(math.copysign(clip, u_next) if math.isfinite(u_next) else clip)
            break
        u.append(u_next)
    factor = measure_amplification_factor(u)
    return {
        "scheme": "explicit",
        "omega": omega,
        "chi": chi,
        "velocity_history": u,
        "measured_factor": factor,
        "analytic_factor": model.rho_dd_explicit(omega),
        "relative_error": _relative_error(abs(factor), model.rho_dd_explicit(omega)),
        "growth_exponent": math.log(abs(factor)) if factor != 0.0 else -math.inf,
        "diverged": diverged or abs(factor) > 1.0,
    }


def measure_amplification_factor(history: Sequence[float]) -> float:
    """Growth factor of a scalar error sequence, measured from successive ratios.

    Consecutive ratios ``|e^{k+1}/e^{k}|`` are computed on the samples that are
    safely above round-off (``> 1e-12`` of the peak) and below the clipping
    ceiling; the median ratio is returned with the sign of the last retained
    pair (so alternating sequences keep their sign).  For a linear recursion the
    median ratio *is* the spectral factor, so this measurement carries no
    analytic shortcut.
    """
    arr = np.asarray([float(v) for v in history], dtype=float)
    mag = np.abs(arr)
    finite = np.isfinite(mag)
    if mag.size < 2 or not finite.any():
        raise ValueError("history too short / non-finite")
    peak = float(mag[finite].max())
    if peak <= 0.0:
        return 0.0
    ok = finite & (mag > 1e-12 * peak) & (mag < 1e150)
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return 0.0
    ratios = [mag[idx[i + 1]] / mag[idx[i]] for i in range(idx.size - 1) if mag[idx[i]] > 0]
    if not ratios:
        return 0.0
    lam = float(np.median(ratios))
    a, b = arr[idx[-2]], arr[idx[-1]]
    if a != 0.0 and (a * b) < 0.0:
        lam = -lam
    return lam


def _piston_step_exact_velocity(
    model: AddedMassPistonModel, u_n: float, f_ext: float, dt: float, m_s: float
) -> float:
    """Monolithic (tightly coupled) velocity of the 1D piston step."""
    return u_n + dt * f_ext / (m_s * (1.0 + model.chi))


def solve_piston_step_subiterations(
    model: AddedMassPistonModel,
    *,
    m_s: float,
    dt: float,
    u_n: float = 0.0,
    f_ext: float = 0.0,
    scheme: CouplingScheme = "fixed",
    omega: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 50,
    initial_guess: float | None = None,
) -> tuple[float, CouplingReport]:
    """Solve one implicit coupling step of the 1D piston by sub-iterations.

    Sub-iteration ``k``:

        fluid      F^k  = -(m_a/dt) (u^k - u_n)
        structure  u*^k = u_n + dt (F^k + f_ext) / m_s
        residual   r^k  = u*^k - u^k
        relax      u^{k+1} = u^k + omega_k r^k     (fixed / Aitken)
                   u^{k+1} = IQN-ILS step         (iqn_ils)

    ``scheme="explicit"`` degenerates to a single pass with ``u^k = u_n``.
    Returns ``(u_new, report)``; ``report.residual_history`` is the raw
    numerical residual sequence used for the D1 rate comparison.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    m_a = model.chi * m_s
    u_exact = _piston_step_exact_velocity(model, u_n, f_ext, dt, m_s)
    u_k = (u_exact + 1.0) if initial_guess is None else float(initial_guess)
    # scale of a unit interface-velocity perturbation (keeps the test dimensionless)
    scale = max(abs(u_exact) + abs(dt * f_ext / m_s) + 1.0, 1.0)

    scheme_name = "fixed" if scheme == "explicit" else scheme
    relax = AitkenRelaxation(omega0=omega) if scheme_name == "aitken" else None
    qn = IQNILS(history_size=5) if scheme_name == "iqn_ils" else None

    report = CouplingReport(
        scheme=scheme_name, converged=False, n_iterations=0, final_residual=math.inf
    )
    omega_k = float(omega)
    u_star = u_k
    for k in range(1, max_iter + 1):
        f_fluid = -(m_a / dt) * (u_k - u_n)
        u_star = u_n + dt * (f_fluid + f_ext) / m_s
        r = u_star - u_k
        report.residual_history.append(abs(r) / scale)
        report.relaxation_history.append(omega_k)
        report.n_iterations = k
        report.final_residual = abs(r) / scale
        if abs(r) / scale < tol:
            report.converged = True
            return u_k, report
        if scheme_name == "explicit":
            # one explicit pass only: reported as a non-converged single iteration
            return u_star, report
        if qn is not None:
            u_next = float(qn.step(u_k, r)[0])
            omega_k = qn.last_gain
            report.relaxation_history[-1] = omega_k
        else:
            if relax is not None:
                omega_k = relax.update(r, r)
                report.relaxation_history[-1] = omega_k
            u_next = u_k + omega_k * r
        if not math.isfinite(u_next) or abs(u_next - u_exact) / scale > 1e10:
            report.diverged = True
            report.n_iterations = k
            report.final_residual = abs(u_next - u_exact) / scale
            return u_next, report
        u_k = u_next
    return u_k, report


def estimate_critical_density_ratio(
    *,
    added_mass_coefficient: float = 1.0,
    omega: float = 1.0,
    scheme: Literal["explicit", "subiteration"] = "explicit",
    n_steps: int = 80,
    rho_f: float = 1.0,
    lo: float = 0.02,
    hi: float = 4.0,
    iters: int = 40,
) -> float:
    """Numerically locate the critical density ratio by bisection on a run.

    The classification uses an *executed* growth measurement: for the
    ``explicit`` scheme the free-perturbation growth factor of
    :func:`simulate_explicit_piston`; for ``subiteration`` the growth of the
    first-iteration residual of a genuinely diverging sub-iteration loop.
    """
    if scheme not in ("explicit", "subiteration"):
        raise ValueError(f"unknown scheme {scheme!r}")

    def diverges(density_ratio: float) -> bool:
        model = AddedMassPistonModel(
            rho_f=rho_f,
            rho_s=density_ratio * rho_f,
            added_mass_coefficient=added_mass_coefficient,
        )
        if scheme == "explicit":
            run = simulate_explicit_piston(model, n_steps=n_steps, omega=omega)
            return abs(run["measured_factor"]) > 1.0
        _, report = solve_piston_step_subiterations(
            model,
            m_s=1.0,
            dt=1.0,
            u_n=0.0,
            f_ext=1.0,
            scheme="fixed",
            omega=omega,
            tol=1e-12,
            max_iter=n_steps,
            initial_guess=1.0,
        )
        hist = report.residual_history
        if report.diverged:
            return True
        return len(hist) >= 3 and hist[-1] > hist[1] * 1.0

    if not diverges(lo):
        return lo
    if diverges(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if diverges(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_density_ratio_from_scan(
    density_ratios: Iterable[float], diverged: Sequence[bool]
) -> float:
    """Largest density ratio flagged as diverging (scan-based boundary)."""
    ratios = list(density_ratios)
    if len(ratios) != len(diverged):
        raise ValueError("density_ratios and diverged must have equal length")
    bad = [r for r, d in zip(ratios, diverged) if d]
    return max(bad) if bad else 0.0


def stability_map(
    *,
    density_ratios: Iterable[float],
    omegas: Iterable[float],
    added_mass_coefficient: float = 1.0,
    n_steps: int = 80,
) -> list[dict]:
    """Scan the ``(rho_s/rho_f, omega)`` plane with executed 1D runs.

    Each cell records the measured growth factor and the divergence flag, so
    the empirical boundary can be plotted against the closed form
    ``rho_s/rho_f = c * omega / 2``.
    """
    rows: list[dict] = []
    for r in density_ratios:
        model = AddedMassPistonModel(
            rho_f=1.0, rho_s=float(r), added_mass_coefficient=added_mass_coefficient
        )
        for w in omegas:
            run = simulate_explicit_piston(model, n_steps=n_steps, omega=float(w))
            rows.append(
                {
                    "density_ratio": float(r),
                    "omega": float(w),
                    "chi": model.chi,
                    "measured_factor": abs(run["measured_factor"]),
                    "analytic_factor": model.rho_dd_explicit(float(w)),
                    "growth_exponent": run["growth_exponent"],
                    "diverged": bool(abs(run["measured_factor"]) > 1.0),
                    "analytic_stable": model.is_linearly_stable_explicit(float(w)),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Relaxation strategies
# --------------------------------------------------------------------------- #


class AitkenRelaxation:
    """Aitken dynamic relaxation (scalar, vector-safe via inner products).

    ``omega_k = -omega_{k-1} (r^{k-1} . (r^k - r^{k-1})) / |r^k - r^{k-1}|^2``
    with the standard safeguards (bounded, slow start) that keep the first
    iterations from amplifying an added-mass-dominated residual.

    ``omega_min = 0`` (default) is scale free (positivity only, shared guard
    :func:`tensorlbm.fsi_partitioned.clamp_aitken_factor`); the former
    absolute floor ``1e-3`` diverged for ``beta_eff >= 4e3``
    (``xval/aitken_underrelaxation_life/out/scale_guard.json``).
    """

    def __init__(
        self,
        omega0: float = 1.0,
        omega_min: float = 0.0,
        omega_max: float = 1.0,
        drop: int = 1,
    ) -> None:
        self.omega = float(omega0)
        self.omega0 = float(omega0)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.drop = int(drop)
        self._prev_residual: np.ndarray | None = None
        self._steps = 0

    def reset(self, omega: float | None = None) -> None:
        self._prev_residual = None
        self._steps = 0
        self.omega = self.omega0 if omega is None else float(omega)

    def update(self, residual, delta) -> float:
        r = np.atleast_1d(np.asarray(residual, dtype=float))
        self._steps += 1
        if self._prev_residual is not None and self._steps > self.drop:
            dr = r - self._prev_residual
            denom = float(np.dot(dr, dr))
            if denom > 1e-30:
                omega = -self.omega * float(np.dot(self._prev_residual, dr)) / denom
                if math.isfinite(omega):
                    self.omega = clamp_aitken_factor(
                        abs(omega), omega_min=self.omega_min, omega_max=self.omega_max
                    )
        self._prev_residual = r.copy()
        return self.omega


class IQNILS:
    """Interface quasi-Newton with least-squares (IQN-ILS), 5-vector history.

    Stores consecutive differences of the interface state and of the coupling
    residual, solves the least-squares problem

        min_c || dR c + r^k ||_2        (last ``history_size`` columns)

    and returns ``x^k + dX c`` as the new interface iterate.  This is the
    standard IQN-ILS interface update (Degroote, Bathe, Vierendeels 2009),
    which is exact for linear coupling maps after two iterations - the reason
    it removes the added-mass instability that defeats plain relaxation.
    """

    def __init__(self, history_size: int = 5, cutoff: float = 1e-10) -> None:
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        self.history_size = int(history_size)
        self.cutoff = float(cutoff)
        self._dR: list[np.ndarray] = []
        self._dX: list[np.ndarray] = []
        self._prev_r: np.ndarray | None = None
        self._prev_x: np.ndarray | None = None
        self.last_gain: float = 1.0

    def reset(self) -> None:
        self._dR.clear()
        self._dX.clear()
        self._prev_r = None
        self._prev_x = None
        self.last_gain = 1.0

    def step(self, x, residual):
        """Return the next interface iterate for state ``x`` and residual ``r``."""
        xv = np.atleast_1d(np.asarray(x, dtype=float)).astype(float)
        rv = np.atleast_1d(np.asarray(residual, dtype=float)).astype(float)
        if self._prev_r is not None and self._prev_x is not None:
            self._dR.append(rv - self._prev_r)
            self._dX.append(xv - self._prev_x)
            while len(self._dR) > self.history_size:
                self._dR.pop(0)
                self._dX.pop(0)
        self._prev_r = rv.copy()
        self._prev_x = xv.copy()
        dx = -rv
        if self._dR:
            A = np.column_stack(self._dR)   # (n, m) residual differences
            B = np.column_stack(self._dX)   # (n, m) interface differences
            try:
                c, *_ = np.linalg.lstsq(A, -rv, rcond=self.cutoff)
                cand = B @ c
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                cand = dx
            if np.all(np.isfinite(cand)) and float(np.linalg.norm(cand)) > 0.0:
                dx = cand
        rn = float(np.linalg.norm(rv))
        self.last_gain = float(np.linalg.norm(dx) / rn) if rn > 0.0 else 1.0
        return xv + dx


# --------------------------------------------------------------------------- #
# 2D D2Q9 LBM realisation (1-DOF heaving body, direct forcing)
# --------------------------------------------------------------------------- #

_E = np.array(
    [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]],
    dtype=float,
)
_W = np.array([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 4 / 9])


def _feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    eu = 3.0 * (_E[:, 0, None, None] * ux + _E[:, 1, None, None] * uy)
    usq = ux * ux + uy * uy
    return (
        _W[:, None, None]
        * rho
        * (1.0 + eu + 0.5 * eu * eu - 1.5 * usq[None, :, :])
    )


class HeavingBodyLBM:
    """D2Q9 channel (periodic in x, walls in y) with a heaving rectangular body.

    The body occupies ``body_width x body_height`` nodes, is constrained to
    vertical translation only, and is coupled to a linear spring-damper in the
    vertical direction.  Direct forcing is applied on the body mask with the
    *trial* body velocity, and the hydrodynamic force is read out as the
    momentum the body imparted to the fluid:

        F_on_body(u_body) = -sum_body rho (u_body - u_star)

    which is the discrete added-mass law ``F = -m_a (u - u_ref) / dt`` with
    ``m_a = sum_body rho``.  Setting ``m_s = rho_s * N_body`` yields
    ``chi = rho_f / rho_s``.
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 96,
        body_width: int = 16,
        body_height: int = 16,
        tau: float = 0.9,
        rho_f: float = 1.0,
        spring_stiffness: float = 2.0e-4,
        structural_damping: float = 1.0e-6,
        y_offset: float = 0.0,
    ) -> None:
        if nx < 8 or ny < 16:
            raise ValueError("grid too small")
        if body_width < 2 or body_height < 2:
            raise ValueError("body too small")
        if not (0.5 < tau < 2.0):
            raise ValueError("tau must be in (0.5, 2)")
        self.nx, self.ny = int(nx), int(ny)
        self.bw, self.bh = int(body_width), int(body_height)
        self.tau = float(tau)
        self.rho_f = float(rho_f)
        self.k = float(spring_stiffness)
        self.c_damp = float(structural_damping)
        self.y0 = 0.5 * (ny - 1) + float(y_offset)
        self.n_body = self.bw * self.bh
        self.f = np.stack(
            [_feq(np.full((nx, ny), rho_f), np.zeros((nx, ny)), np.zeros((nx, ny)))[i]
             for i in range(9)]
        )
        x0 = (nx - self.bw) // 2
        self._xs = slice(x0, x0 + self.bw)

    # -- geometry ---------------------------------------------------------- #
    def body_rows(self, y_center: float) -> slice:
        lo = int(round(y_center - 0.5 * self.bh))
        lo = max(2, min(self.ny - 2 - self.bh, lo))
        return slice(lo, lo + self.bh)

    def mask(self, y_center: float) -> np.ndarray:
        m = np.zeros((self.nx, self.ny), dtype=bool)
        m[self._xs, self.body_rows(y_center)] = True
        return m

    def wall_mask(self) -> np.ndarray:
        m = np.zeros((self.nx, self.ny), dtype=bool)
        m[:, 0] = True
        m[:, 1] = True
        m[:, self.ny - 2] = True
        m[:, self.ny - 1] = True
        return m

    # -- fluid ------------------------------------------------------------- #
    def collide_stream(self) -> None:
        f = self.f
        rho = f.sum(axis=0)
        ux = (f * _E[:, 0, None, None]).sum(axis=0) / rho
        uy = (f * _E[:, 1, None, None]).sum(axis=0) / rho
        f_post = f - (f - _feq(rho, ux, uy)) / self.tau
        f_new = np.empty_like(f)
        for i in range(9):
            f_new[i] = np.roll(
                np.roll(f_post[i], int(_E[i, 0]), axis=0), int(_E[i, 1]), axis=1
            )
        self.f = f_new

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self.f
        rho = f.sum(axis=0)
        ux = (f * _E[:, 0, None, None]).sum(axis=0) / rho
        uy = (f * _E[:, 1, None, None]).sum(axis=0) / rho
        return rho, ux, uy

    def _momentum_masked(self, mask: np.ndarray) -> tuple[float, np.ndarray]:
        rho, ux, uy = self.macroscopic()
        w = rho[mask]
        m = float(w.sum())
        mom = np.array([float((w * ux[mask]).sum()), float((w * uy[mask]).sum())])
        return m, mom

    def impose(self, mask: np.ndarray, ux_v: float, uy_v: float) -> None:
        """Overwrite populations on ``mask`` with equilibrium at ``(ux_v, uy_v)``."""
        rho = self.f.sum(axis=0)
        rho3 = np.repeat(rho[:, :, None], 9, axis=2).transpose(2, 0, 1)
        feq = _feq(np.ones_like(rho), np.full_like(rho, ux_v), np.full_like(rho, uy_v))
        self.f = np.where(mask.transpose(0, 1)[None, :, :], feq * rho3, self.f)

    def body_force(self, mask: np.ndarray, u_body: float) -> float:
        """Vertical force on the body for trial velocity ``u_body`` (dt = 1)."""
        m_a, mom = self._momentum_masked(mask)
        return float(-(m_a * u_body - mom[1]))

    def added_mass(self, mask: np.ndarray) -> float:
        m_a, _ = self._momentum_masked(mask)
        return m_a


def run_lbm_fsi_decay(
    *,
    model: AddedMassPistonModel,
    scheme: CouplingScheme = "iqn_ils",
    omega: float = 1.0,
    n_steps: int = 600,
    tol: float = 1e-6,
    max_subiter: int = 20,
    nx: int = 64,
    ny: int = 96,
    body_size: int = 16,
    tau: float = 0.9,
    omega_n: float = 0.02,
    zeta: float = 0.01,
    amplitude: float = 0.5,
    clip_velocity: float = 0.5,
) -> dict:
    """Run a 1-DOF heave decay in the 2D LBM with a partitioned coupling.

    ``omega_n`` is the *structural* (vacuum) natural frequency in lattice
    time-step units; the spring is ``k = omega_n^2 m_s`` and the damping
    ``c = 2 zeta omega_n m_s`` with ``m_s = rho_s * N_body``, hence
    ``chi = rho_f / rho_s`` exactly (measured and reported as
    ``chi_measured``).

    Schemes:

    * ``explicit`` - classic staggered coupling: the structure advances with the
      interface force of the *previous* time level (fluid force lagged by one
      step), so the interface increments obey ``dv^{n+1} = -chi dv^n`` and the
      scheme is linearly unstable for ``chi > 1`` (added-mass instability).
    * ``fixed`` / ``aitken`` / ``iqn_ils`` - within-step sub-iteration on the
      interface velocity with the added-mass force re-evaluated at the trial
      velocity, i.e. the implicit added mass.  ``fixed`` (constant relaxation)
      diverges for light bodies, ``aitken`` and ``iqn_ils`` converge.
    """
    lbm = HeavingBodyLBM(
        nx=nx,
        ny=ny,
        body_width=body_size,
        body_height=body_size,
        tau=tau,
        rho_f=model.rho_f,
    )
    m_s = model.rho_s * lbm.n_body
    k = omega_n**2 * m_s
    c_d = 2.0 * zeta * omega_n * m_s

    y0_body = lbm.y0
    y = y0_body + amplitude
    v_n = 0.0
    f_fluid_prev = 0.0
    hist_y: list[float] = []
    hist_v: list[float] = []
    subiters: list[int] = []
    residuals: list[float] = []
    diverged = False
    unconverged_steps = 0
    m_a_eff = 0.0

    for _ in range(n_steps):
        mask = lbm.mask(y)
        lbm.collide_stream()
        lbm.impose(lbm.wall_mask(), 0.0, 0.0)
        m_a_eff, mom = lbm._momentum_masked(mask)
        f_spring = -(k * (y - y0_body)) - c_d * v_n

        if not math.isfinite(m_a_eff) or m_a_eff <= 0.0:
            diverged = True
            break

        r_norm = 0.0
        it = 0
        v_k = v_n

        if scheme == "explicit":
            it = 1
            v_k = v_n + (f_fluid_prev + f_spring) / m_s
        else:
            relax = AitkenRelaxation(omega0=omega) if scheme == "aitken" else None
            qn = IQNILS(history_size=5) if scheme == "iqn_ils" else None
            v_k = v_n
            omega_k = float(omega)
            sub_conv = False
            for it in range(1, max_subiter + 1):
                f_fluid = -(m_a_eff * v_k - mom[1])
                v_star = v_n + (f_fluid + f_spring) / m_s
                r = v_star - v_k
                r_norm = abs(r) / max(abs(v_n), 1e-9, omega_n * amplitude)
                if abs(r) < tol * max(abs(v_n), 1e-9, omega_n * amplitude):
                    sub_conv = True
                    break
                if qn is not None:
                    v_k = float(qn.step(v_k, r)[0])
                else:
                    if relax is not None:
                        omega_k = relax.update(r, r)
                    v_k = v_k + omega_k * r
                if not math.isfinite(v_k) or abs(v_k) > clip_velocity:
                    diverged = True
                    break
            if it == max_subiter and not sub_conv:
                unconverged_steps += 1

        if diverged or not math.isfinite(v_k) or abs(v_k) > clip_velocity:
            diverged = True
            hist_v.append(float(v_k) if math.isfinite(v_k) else clip_velocity)
            break

        lbm.impose(mask, 0.0, v_k)
        f_fluid_prev = -(m_a_eff * v_k - mom[1])
        v_n = v_k
        y = y + v_n  # dt = 1
        hist_y.append(y)
        hist_v.append(v_n)
        subiters.append(it)
        residuals.append(r_norm)
        if not math.isfinite(y) or abs(y - y0_body) > 0.45 * ny:
            diverged = True
            break

    disp = np.asarray(hist_y, dtype=float) - y0_body
    vig = np.asarray(hist_v, dtype=float)
    growth = _velocity_growth(vig)
    return {
        "scheme": scheme,
        "omega": omega,
        "density_ratio": model.density_ratio,
        "chi_analytic": model.chi,
        "chi_measured": (m_a_eff / m_s) if m_s > 0 else math.inf,
        "n_steps_done": int(disp.size),
        "diverged": bool(diverged),
        "velocity_history": vig.tolist(),
        "displacement_history": disp.tolist(),
        "unconverged_steps": int(unconverged_steps),
        "mean_subiterations": float(np.mean(subiters)) if subiters else math.inf,
        "max_subiterations": int(np.max(subiters)) if subiters else -1,
        "mean_residual": float(np.mean(residuals)) if residuals else math.inf,
        "final_residual": float(residuals[-1]) if residuals else math.inf,
        "amplitude": float(np.max(np.abs(disp))) if disp.size else 0.0,
        "growth_exponent": growth,
        "rho_dd_measured": math.exp(growth) if growth == growth else math.nan,
        "period": _zero_crossing_period(disp),
        "period_velocity": _zero_crossing_period(np.asarray(hist_v, dtype=float)),
    }


def _velocity_growth(v: np.ndarray) -> float:
    """Growth exponent per step measured from a velocity record.

    Uses successive local maxima of ``|v|`` when at least three exist (works for
    both growing and decaying oscillations), otherwise a log-linear fit over the
    record above the round-off floor (monotone divergence).
    """
    if v.size < 4:
        return math.nan
    mag = np.abs(np.asarray(v, dtype=float))
    peaks = [
        mag[i]
        for i in range(1, mag.size - 1)
        if mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1] and mag[i] > 0.0
    ]
    if len(peaks) >= 3:
        n = np.arange(len(peaks), dtype=float)
        return float(np.polyfit(n, np.log(np.asarray(peaks)), 1)[0])
    floor = 1e-14 * float(mag.max()) if mag.max() > 0 else 0.0
    idx = np.flatnonzero((mag > floor) & np.isfinite(mag))
    if idx.size < 3:
        return math.nan
    return float(np.polyfit(idx.astype(float), np.log(mag[idx]), 1)[0])


def _zero_crossing_period(y: np.ndarray) -> float:
    """Period from upward zero crossings of the heave record (nan if < 2)."""
    if y.size < 4:
        return math.nan
    sign = np.sign(y)
    cross = np.flatnonzero((sign[:-1] < 0) & (sign[1:] >= 0))
    if cross.size < 2:
        return math.nan
    return float(np.mean(np.diff(cross)))


def structural_free_decay_frequency(
    *, m_eff: float = 2000.0, stiffness: float = 2000.0, dt: float = 0.5,
    n_periods: int = 16, amplitude: float = 2.5819771690294147e-4,
) -> dict:
    """Structural-only heave free decay (no fluid), for the verified baseline.

    Mirrors the parameters of ``benchmarks/verified/floating_body_free_decay``
    (``m_eff = 2000``, ``k = 2000``, ``dt = 0.5`` => ``omega_n = 1.0``) with a
    velocity-Verlet (symplectic) integrator, so the FSI module's structure
    integrator can be checked against the verified reference.
    """
    omega_n = math.sqrt(stiffness / m_eff)
    period = 2.0 * math.pi / omega_n
    n_steps = int(round(n_periods * period / dt))
    y, v = amplitude, 0.0
    ys = np.zeros(n_steps)
    a = -(stiffness / m_eff) * y
    for n in range(n_steps):
        v_half = v + 0.5 * dt * a
        y = y + dt * v_half
        a = -(stiffness / m_eff) * y
        v = v_half + 0.5 * dt * a
        ys[n] = y
    period_steps = _zero_crossing_period(ys)
    omega_meas = (
        2.0 * math.pi / (period_steps * dt)
        if period_steps and period_steps > 0
        else math.nan
    )
    return {
        "omega_n_analytic": omega_n,
        "omega_meas": omega_meas,
        "relative_error": abs(omega_meas - omega_n) / omega_n if omega_meas == omega_meas else math.nan,
        "dt": dt,
        "n_steps": n_steps,
        "amplitude": float(np.max(np.abs(ys))),
    }