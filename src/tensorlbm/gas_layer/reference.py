"""Analytical references for plane-channel flow (single-phase calibration).

All formulas are for a plane channel of height ``h`` (lattice units) driven by a
uniform body-force acceleration ``g`` along ``x`` with no-slip walls, laminar,
incompressible.  They define the *reference against which the lattice solver is
graded*; nothing here is fitted to the lattice data.

Conventions
-----------
* ``nu``  : kinematic viscosity, ``nu = cs^2 (tau - 1/2)``.
* ``h``   : channel height (distance between the two half-way wall nodes).
* ``y``   : wall-normal coordinate measured **from the bottom wall**.
* ``U``   : bulk (cross-section averaged) streamwise velocity.
* ``tau_w``: wall shear stress on one wall, ``tau_w = rho g h / 2``.
* ``Cf``  : ``2 tau_w / (rho U^2)`` (the definition fixed by the mission).

Exact laminar results
---------------------
``u(y)   = g y (h - y) / (2 nu)``
``u_max  = g h^2 / (8 nu)``, ``U = 2/3 u_max = g h^2 / (12 nu)``
``tau_w  = rho g h / 2 = 6 mu U / h``
``Cf     = 12 nu / (U h) = 12 / Re_h``,  ``Re_h = U h / nu``
``u_tau  = sqrt(tau_w / rho)``, ``Re_tau = u_tau (h/2) / nu``
``u+     = y+ (1 - y+ / (2 Re_tau))``  -- exact laminar law at *any* ``Re_tau``.

The last identity is what makes a ``Re_tau = 180`` reference profile available
without a turbulent simulation: at ``Re_tau = 180`` the exact laminar profile is
the reference, and the log law ``u+ = ln(y+)/kappa + B`` is printed alongside as
*context only* (a 2-D lattice channel is laminar; see the module docstring of
:mod:`tensorlbm.gas_layer.single_phase` for why the log law cannot be a gate at
this resolution).
"""

from __future__ import annotations

import math

#: von Karman constant and log-law intercept, used for context plots only.
KAPPA = 0.41
LOG_LAW_B = 5.2


def laminar_profile(g: float, nu: float, h: float, y: float):
    """Exact laminar velocity at wall distance ``y`` (scalar or tensor)."""
    return g * y * (h - y) / (2.0 * nu)


def laminar_mean_velocity(g: float, nu: float, h: float) -> float:
    """Bulk velocity of the exact laminar channel: ``g h^2 / (12 nu)``."""
    return g * h * h / (12.0 * nu)


def laminar_max_velocity(g: float, nu: float, h: float) -> float:
    """Centre-line velocity of the exact laminar channel: ``g h^2 / (8 nu)``."""
    return g * h * h / (8.0 * nu)


def laminar_wall_shear(g: float, rho: float, h: float) -> float:
    """Wall shear stress of one wall: ``rho g h / 2`` (momentum balance)."""
    return 0.5 * rho * g * h


def friction_coefficient(U: float, h: float, nu: float, rho: float = 1.0) -> float:
    """``Cf = 2 tau_w / (rho U^2)`` for laminar channel flow, ``= 12/Re_h``."""
    re_h = U * h / nu
    return 12.0 / re_h


def friction_coefficient_from_shear(tau_w: float, U: float, rho: float = 1.0) -> float:
    """``Cf`` from a measured wall shear (the definition used for the gate)."""
    return 2.0 * tau_w / (rho * U * U)


def skin_friction_ittc57(re_length: float) -> float:
    """ITTC-1957 flat-plate friction coefficient (physical units, turbulent).

    Used only by the economic ledger, to convert a *drag-reduction ratio*
    computed in the lattice into a physical wall shear stress.
    """
    if re_length <= 0.0:
        raise ValueError("Re must be positive")
    return 0.075 / (math.log10(re_length) - 2.0) ** 2


def utau(tau_w: float, rho: float) -> float:
    """Friction velocity ``sqrt(tau_w/rho)``."""
    return math.sqrt(tau_w / rho)


def re_tau(u_tau: float, half_height: float, nu: float) -> float:
    """Friction Reynolds number based on the channel half-height."""
    return u_tau * half_height / nu


def laminar_u_plus(y_plus: float, re_tau_value: float):
    """Exact laminar law ``u+ = y+ (1 - y+/(2 Re_tau))`` (scalar or tensor)."""
    return y_plus * (1.0 - y_plus / (2.0 * re_tau_value))


def log_law_u_plus(y_plus: float):
    """Log law ``u+ = ln(y+)/kappa + B`` (context only, not a gate)."""
    return math.log(max(y_plus, 1e-12)) / KAPPA + LOG_LAW_B