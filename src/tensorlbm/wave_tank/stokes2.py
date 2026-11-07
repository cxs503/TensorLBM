"""Second-order Stokes wave theory, finite depth, lattice units.

All lengths are in lattice cells, ``g`` is the gravity acceleration in lattice
units (lattice ``dt = dx = 1``), and the vertical coordinate ``y`` is measured
from the still-water level (SWL) with ``y < 0`` below the surface.

Free surface::

    eta(x, t) = a cos(theta) + (k a^2 / 4) * ((3 - s^2) / s^3) * cos(2 theta)
    theta     = k x - omega t,      s = tanh(k h)

which is the campaign-specified second-order form (equivalently Dean &
Dalrymple, *Water Wave Mechanics*, 2nd-order Stokes expansion).

Velocity field (first order + the phase-locked 2-omega potential)::

    u = a w cosh k(y+h)/sinh(kh) cos(theta)  + 2 k C cosh 2k(y+h) cos(2 theta)
    w = a w sinh k(y+h)/sinh(kh) sin(theta)  + 2 k C sinh 2k(y+h) sin(2 theta)

``C`` is fixed by the *kinematic* free-surface condition evaluated at the
second order (see :meth:`Stokes2Wave.second_order_potential`).  In deep water
``C -> 0``, recovering the classical result that the second-order velocity
correction in deep water appears only through the displacement of the sampling
point, not through an explicit 2-omega potential.

Dispersion: ``omega^2 = g k tanh(k h)`` (exact linear gravity-wave relation);
the solver in :func:`dispersion_solve` inverts it for ``k``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Stokes2Wave", "dispersion_solve"]


def dispersion_solve(
    omega: float,
    g: float,
    h: float,
    *,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> float:
    """Return the wavenumber ``k > 0`` with ``omega^2 = g k tanh(k h)``.

    Newton iteration on ``f(k) = g k tanh(kh) - omega^2`` with the
    deep-water guess ``k0 = omega^2 / g``.  The relation is monotone in ``k``
    for ``k > 0`` so Newton converges from below monotonically.
    """
    if omega <= 0.0 or g <= 0.0 or h <= 0.0:
        raise ValueError("omega, g and h must be positive")
    k = omega * omega / g  # deep-water guess (underestimates k in finite depth)
    for _ in range(max_iter):
        kh = k * h
        th = math.tanh(kh)
        f = g * k * th - omega * omega
        # d/dk [g k tanh(kh)] = g [tanh(kh) + k h sech^2(kh)]
        df = g * (th + kh * (1.0 - th * th))
        step = f / df
        k -= step
        if k <= 0.0:  # keep the iterate physical
            k = 0.5 * (k + step)
        if abs(step) <= tol * max(k, 1e-30):
            break
    return k


@dataclass(frozen=True)
class Stokes2Wave:
    """Second-order Stokes wave in finite depth (lattice units).

    Attributes:
        h: still-water depth.
        a: first-order amplitude (half wave height).
        k: wavenumber from the dispersion relation.
        omega: angular frequency.
        g: gravity acceleration.
    """

    h: float
    a: float
    k: float
    omega: float
    g: float

    # -- construction ------------------------------------------------------

    @classmethod
    def from_kh(
        cls,
        kh: float,
        eps: float,
        h: float,
        g: float,
    ) -> "Stokes2Wave":
        """Build from the dimensionless depth ``kh`` and steepness ``eps = k a``.

        ``k = kh / h`` and ``omega = sqrt(g k tanh kh)`` follow from the
        dispersion relation, hence ``a = eps / k``.
        """
        if kh <= 0.0 or h <= 0.0 or g <= 0.0:
            raise ValueError("kh, h and g must be positive")
        k = kh / h
        sigma = math.tanh(kh)
        omega = math.sqrt(g * k * sigma)
        a = eps / k
        return cls(h=h, a=a, k=k, omega=omega, g=g)

    @classmethod
    def from_omega(
        cls,
        omega: float,
        a: float,
        h: float,
        g: float,
    ) -> "Stokes2Wave":
        """Build from frequency and amplitude (wavenumber from dispersion)."""
        return cls(h=h, a=a, k=dispersion_solve(omega, g, h), omega=omega, g=g)

    # -- dimensionless helpers --------------------------------------------

    @property
    def sigma(self) -> float:
        """``tanh(kh)``."""
        return math.tanh(self.k * self.h)

    @property
    def kh(self) -> float:
        return self.k * self.h

    @property
    def eps(self) -> float:
        """Steepness ``k a``."""
        return self.k * self.a

    @property
    def celerity(self) -> float:
        """Linear phase speed ``c = omega / k = sqrt((g/k) tanh kh)``."""
        return self.omega / self.k

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.k

    @property
    def eta2(self) -> float:
        """Second-harmonic free-surface amplitude ``(k a^2/4)(3-s^2)/s^3``."""
        s = self.sigma
        return 0.25 * self.k * self.a * self.a * (3.0 - s * s) / (s ** 3)

    @property
    def eta_mean(self) -> float:
        """Second-order mean set-down ``-k a^2 / (2 sinh 2kh)`` (not imposed)."""
        return -self.k * self.a * self.a / (2.0 * math.sinh(2.0 * self.k * self.h))

    # -- kinematics --------------------------------------------------------

    def theta(self, x, t):
        """Wave phase ``k x - omega t`` (broadcasts over numpy arrays)."""
        return self.k * x - self.omega * t

    def eta(self, x, t):
        """Second-order free-surface elevation."""
        th = self.theta(x, t)
        return self.a * _cos(th) + self.eta2 * _cos(2.0 * th)

    def second_order_potential(self) -> float:
        """Amplitude ``C`` of the 2-omega velocity potential.

        Derived from the second-order kinematic free-surface condition
        ``w = eta_t + u eta_x`` at ``y = eta`` (Taylor-expanded to ``O(a^2)``)::

            2 k C sinh(2kh) = 2 omega eta2 - a^2 omega k coth(kh)

        where the two ``a^2 omega k coth(kh)/2`` contributions come from the
        surface-displacement term ``eta w_y`` and from ``u eta_x`` (they sit on
        opposite sides of the kinematic condition and therefore add).  In deep
        water the two terms cancel exactly, ``C -> 0``.
        """
        kh = self.k * self.h
        s = math.tanh(kh)
        coth = 1.0 / s
        num = 2.0 * self.omega * self.eta2 - self.a * self.a * self.omega * self.k * coth
        den = 2.0 * self.k * math.sinh(2.0 * kh)
        return num / den

    def velocity(self, x, y, t):
        """Return ``(u, w)`` first + second order velocity at ``(x, y)``.

        ``y`` is measured from the SWL (negative below the surface).  Fields
        are evaluated with the linearised (SWL) arguments, which is the
        standard second-order Stokes representation; the ``O(a)`` mismatch of
        sampling height is itself an ``O(a^2)`` effect already contained in the
        expansion.  Callers that impose the field on a moving interface should
        therefore blend it with the liquid fraction (see
        :mod:`tensorlbm.wave_tank.tank`).
        """
        kh = self.k * self.h
        s = math.tanh(kh)
        ch = math.cosh(kh)
        sh = math.sinh(kh)
        th = self.theta(x, t)
        # first order
        ky1 = self.k * (y + self.h)
        c1 = self.a * self.omega / sh
        u1 = c1 * _cosh(ky1) * _cos(th)
        w1 = c1 * _sinh(ky1) * _sin(th)
        # second order (phase-locked 2-omega potential)
        C = self.second_order_potential()
        ky2 = 2.0 * self.k * (y + self.h)
        u2 = 2.0 * self.k * C * _cosh(ky2) * _cos(2.0 * th)
        w2 = 2.0 * self.k * C * _sinh(ky2) * _sin(2.0 * th)
        del ch
        return u1 + u2, w1 + w2

    def pressure(self, x, y, t, rho: float = 1.0):
        """Hydrostatic + first-order dynamic pressure at ``(x, y)``.

        ``p = rho g (eta - y)``  (hydrostatic, exact for a linear surface).
        """
        return rho * self.g * (self.eta(x, t) - y)

    # -- verification ------------------------------------------------------

    def kinematic_residual(self, x=0.0, t=0.0) -> float:
        """Relative residual of the kinematic FSBC at ``y = eta``.

        Evaluates ``w - (eta_t + u eta_x)`` at the actual surface, normalised
        by ``a omega``.  A consistent second-order truncation leaves an
        ``O(a^3)`` residual, i.e. ``O((ka)^2)`` after this normalisation; a
        wrong second-order coefficient leaves ``O(ka)``.  Used by
        ``tests/test_wave_tank.py`` as a machine-checkable verification of the
        theory module.
        """
        eta = float(self.eta(x, t))
        u, w = self.velocity(x, eta, t)
        dt = 1e-4 * self.period
        eta_t = (float(self.eta(x, t + dt)) - float(self.eta(x, t - dt))) / (2.0 * dt)
        dx = 1e-4 * self.wavelength
        eta_x = (float(self.eta(x + dx, t)) - float(self.eta(x - dx, t))) / (2.0 * dx)
        return abs(w - (eta_t + u * eta_x)) / (self.a * self.omega)


# -- array-agnostic helpers ------------------------------------------------


def _cos(x):
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return _np.cos(x)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch as _torch

        if isinstance(x, _torch.Tensor):
            return _torch.cos(x)
    except Exception:  # pragma: no cover
        pass
    return math.cos(x)


def _sin(x):
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return _np.sin(x)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch as _torch

        if isinstance(x, _torch.Tensor):
            return _torch.sin(x)
    except Exception:  # pragma: no cover
        pass
    return math.sin(x)


def _cosh(x):
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return _np.cosh(x)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch as _torch

        if isinstance(x, _torch.Tensor):
            return _torch.cosh(x)
    except Exception:  # pragma: no cover
        pass
    return math.cosh(x)


def _sinh(x):
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return _np.sinh(x)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch as _torch

        if isinstance(x, _torch.Tensor):
            return _torch.sinh(x)
    except Exception:  # pragma: no cover
        pass
    return math.sinh(x)