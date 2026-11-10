"""Analytic reference layer for rarefied/slip flows (non-dimensionalised).

Conventions (fixed for the whole ``rarefied`` package)
------------------------------------------------------
* Units: ``m = 1``, ``R T = 1`` (so the isothermal sound speed is ``c_s = 1``),
  density normalised to ``rho_0 = 1``.  Channel height ``H`` (default ``1``).
* The gas is a **BGK** gas: viscosity ``mu = tau p`` so that the kinematic
  viscosity is ``nu = tau R T = tau``.
* Mean free path (Bird's viscosity-based, hard-sphere-equivalent definition)::

      lambda = tau * sqrt(pi R T / 2)          (=> tau = lambda / sqrt(pi RT/2))

  This is the same ``lambda`` DSMC literature uses when quoting Knudsen minima.
* Knudsen number: ``Kn = lambda / H`` (channel **height**, not half-height).
  ``kn_half = 2 Kn`` is reported too where the half-height convention matters.
* Tangential momentum accommodation coefficient ``sigma in (0, 1]``: a fraction
  ``sigma`` of the molecules reflects diffusely, ``(1 - sigma)`` specularly.
  ``sigma = 1`` is a fully diffuse wall, ``sigma -> 0`` approaches a perfectly
  specular (free-slip) wall.

Closed forms used as ladder rungs
---------------------------------
Maxwell first-order slip length::

      l_s = (2 - sigma) / sigma * lambda

Navier-Stokes + first-order slip, force-driven Poiseuille (body force ``g``
along x, zero pressure gradient; the linearised problem is equivalent to
pressure-driven Poiseuille with ``dp/dx = -rho g``)::

      u(y) = g / (2 nu) * [ H^2/4 - (y - H/2)^2 ] + g H l_s / (2 nu)
      Q    = g H^3 / (12 nu) * [ 1 + 6 l_s / H ]        (mass flow per unit width)

Navier-Stokes + slip plane Couette flow (bottom wall at rest, top wall ``U``)::

      u(y) = U (y + l_s) / (H + 2 l_s)

Free-molecular limit (diffuse walls, ``Kn -> inf``, body force ``g``, no
intermolecular collisions).  Molecules leave a wall with a half-Maxwellian, fly
ballistically over a distance ``s`` in a flight time ``t = s/|v_y|`` and pick up
``g t`` before being thermalised by the opposite wall.  Averaging over the
flux-weighted half-Maxwellian (``<1/|v_y|>_+ = sqrt(pi/(2RT))``)::

      u(y) = (g/2) [ y <1/v_y>_+ + (H - y) <1/v_y>_+ ] = g H / 2 * sqrt(pi/(2RT))

  i.e. a **plug profile** with

      Q_FM = rho_0 g H^2 / 2 * sqrt(pi / (2 R T)).

  Consistency check that makes this formula trustworthy: with
  ``Q_NS = g H^3/(12 nu)`` and ``tau = lambda sqrt(2/(pi RT))`` one gets
  ``Q_FM / Q_NS = 6 Kn`` exactly, the classical free-molecular / no-slip
  Poiseuille ratio.
"""

from __future__ import annotations

import math

import numpy as np

# tau = lambda * BGK_TAU_FACTOR  (with RT = 1)
BGK_TAU_FACTOR = math.sqrt(2.0 / math.pi)


def mean_free_path(tau: float, RT: float = 1.0) -> float:
    """Viscosity-based mean free path of a BGK gas: ``lambda = tau sqrt(pi RT/2)``."""
    return tau * math.sqrt(math.pi * RT / 2.0)


def tau_from_kn(kn: float, H: float = 1.0, RT: float = 1.0) -> float:
    """BGK relaxation time for a prescribed ``Kn = lambda/H``."""
    return kn * H * math.sqrt(2.0 / (math.pi * RT))


def kn_from_tau(tau: float, H: float = 1.0, RT: float = 1.0) -> float:
    """Inverse of :func:`tau_from_kn` (``Kn = lambda/H``)."""
    return mean_free_path(tau, RT) / H


def maxwell_slip_length(sigma: float, lam: float) -> float:
    """Maxwell first-order slip length ``l_s = (2 - sigma)/sigma * lambda``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0 (sigma -> 0 is free slip / infinite l_s)")
    return (2.0 - sigma) / sigma * lam


def maxwell_slip_coefficient(sigma: float) -> float:
    """Slip coefficient ``(2 - sigma)/sigma`` = ``l_s / lambda``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return (2.0 - sigma) / sigma


def ns_slip_poiseuille_profile(y, H: float, nu: float, g: float, ls: float = 0.0):
    """NS + first-order-slip velocity profile of force-driven plane Poiseuille."""
    y = np.asarray(y, dtype=float)
    return g / (2.0 * nu) * (H * H / 4.0 - (y - H / 2.0) ** 2) + g * H * ls / (2.0 * nu)


def ns_slip_flow_rate(H: float, nu: float, g: float, ls: float = 0.0) -> float:
    """Mass flow per unit width, ``Q = g H^3/(12 nu) (1 + 6 l_s/H)`` (rho_0 = 1)."""
    return g * H ** 3 / (12.0 * nu) * (1.0 + 6.0 * ls / H)


def ns_slip_couette_profile(y, H: float, U: float, ls: float = 0.0):
    """NS + slip plane Couette profile, bottom wall at rest, top wall ``U``."""
    y = np.asarray(y, dtype=float)
    return U * (y + ls) / (H + 2.0 * ls)


def free_molecular_flow_rate(H: float = 1.0, g: float = 1.0, RT: float = 1.0,
                             rho: float = 1.0) -> float:
    """Free-molecular force-driven mass flow per unit width (diffuse walls)."""
    return 0.5 * rho * g * H * H * math.sqrt(math.pi / (2.0 * RT))


def free_molecular_plug_velocity(H: float = 1.0, g: float = 1.0, RT: float = 1.0) -> float:
    """Uniform free-molecular velocity ``g H/2 sqrt(pi/(2RT))`` of the plug profile."""
    return 0.5 * g * H * math.sqrt(math.pi / (2.0 * RT))


def free_molecular_equivalent_slip_length(kn: float, H: float = 1.0) -> float:
    """Slip length that would reproduce ``Q_FM`` inside the NS+slip formula.

    Matching ``Q_FM`` to ``Q_NS (1 + 6 l_s/H)`` gives ``1 + 6 l_s/H = 6 Kn``,
    i.e. ``l_s = H (Kn - 1/6)``.  For ``Kn > 1/6`` this grows linearly, which is
    unphysical as a boundary condition: free-molecular transport is a *plug*
    profile, not a slip-extended parabola.  The saturation of ``Q`` in the
    transitional range is exactly what produces the Knudsen minimum.
    """
    return H * (kn - 1.0 / 6.0)


def knudsen_minimum_reference() -> dict:
    """Bookkeeping for the Knudsen-minimum rung of the ladder.

    The location of the minimum of the normalised mass flow rate
    ``G(Kn) = Q(Kn)/Q_FM`` is **definition dependent** (linear in the mean free
    path convention).  With the ``lambda`` of this module (viscosity based,
    ``Kn = lambda/H``) the classical kinetic-theory/DSMC result for plane
    Poiseuille places the minimum at ``Kn ~ 1``; the ladder asserts only the
    interval ``Kn_min in [0.5, 2]`` from the task definition and reports the
    number it actually measures.
    """
    return {
        "definition": "Kn = lambda/H, lambda = tau*sqrt(pi*RT/2) (Bird viscosity based)",
        "expected_interval": [0.5, 2.0],
        "half_height_convention": "kn_half = 2*Kn",
    }