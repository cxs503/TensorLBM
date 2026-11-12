"""Canonical Hasimoto (1959) drag reference for a simple-cubic sphere array.

Why this module exists
----------------------
``K = F / (6 pi mu a U)`` — the drag enhancement of a sphere in a periodic
simple-cubic (SC) array up to the analytic-free (Hasimoto) confinement
correction — is used as an *analytic instrument* by several independent paths
of this repository:

* :mod:`tensorlbm.d3q27_recovery` (periodic-box Stokes sphere acceptance),
* :mod:`tensorlbm.force_control_surface` (a-posteriori array comparison),
* :mod:`tensorlbm.particle_suspension` (dilute suspension closures),
* :mod:`tensorlbm.sc_sphere_array_drag` and
  ``benchmarks/case_pool/sc_sphere_array_drag`` (array drag benchmark),
* ``benchmarks/sphere_curved_drag/driver.py`` (the third, correlation-free
  force instrument used to adjudicate ``GAP-LBM-SPHERE``).

Before this module each site carried its own truncated copy of the series and
they disagreed by up to 17 % (see ``docs/ARRAY_DRAG_REFERENCE_AUDIT.md``).
Here the series is written **once**, in volume fraction — the form in which it
is two-source locked in ``benchmarks/verified/permeability3d/NOTES.md`` §1.6 —
and every ``lambda = a / L_cell`` user is served by the *exact* geometric
conversion ``phi = 4 pi lambda^3 / 3``.  No coefficient is re-derived or
rounded by hand at the call site.

Series (Hasimoto 1959, J. Fluid Mech. 5:317; SC sedimentation)

    U / U0 = 1 - 1.7601 phi^(1/3) + phi - 1.5593 phi^2
             + 3.9799 phi^(8/3) - 3.0734 phi^(10/3) + O(phi^(11/3))
    K      = U0 / U = 1 / (U / U0)

Validity: dilute expansion, ``phi <= 0.216`` keeps it within 1 % of the locked
SC table (``benchmarks/verified/permeability3d/NOTES.md`` §1.6); at
``phi = 0.343`` it is 7.6 % low, i.e. outside its convergence radius.  Callers
that need ``phi > 0.216`` must use a tabulated reference, not this series.

Converted to ``lambda`` powers (derived here, not typed anywhere),

    U / U0 = 1 - 2.83730 lambda + 4.18879 lambda^3 - 27.359 lambda^6
             + 162.2 lambda^8 - 313.6 lambda^10 + ...

``lambda_series_coefficients`` returns exactly these numbers from the
volume-fraction series, which is how the audit proves that the various
lambda-form truncations in the repository are the *same* series.
"""

from __future__ import annotations

import math

#: ``(coefficient, exponent of phi)`` of ``U / U0``; Hasimoto (1959) SC series.
U_RATIO_PHI_TERMS: tuple[tuple[float, float], ...] = (
    (-1.7601, 1.0 / 3.0),
    (1.0, 1.0),
    (-1.5593, 2.0),
    (3.9799, 8.0 / 3.0),
    (-3.0734, 10.0 / 3.0),
)

#: ``phi = 4 pi a^3 / (3 L^3)`` for one sphere per cubic cell of side ``L``.
PHI_PER_LAMBDA_CUBED: float = 4.0 * math.pi / 3.0

#: Upper end of the dilute series' useful range (1 % vs the locked SC table).
SERIES_MAX_PHI: float = 0.216

#: Same bound expressed as ``lambda = a / L``.
SERIES_MAX_LAMBDA: float = (3.0 * SERIES_MAX_PHI / (4.0 * math.pi)) ** (1.0 / 3.0)

#: The full series written in ``phi``: ``c = float(phi)``; see module docstring.
def phi_from_lambda(lam: float) -> float:
    """Solid volume fraction of an SC array with ``lam = a / L_cell``."""
    return PHI_PER_LAMBDA_CUBED * float(lam) ** 3


def lambda_from_phi(phi: float) -> float:
    """Inverse of :func:`phi_from_lambda`."""
    return (float(phi) / PHI_PER_LAMBDA_CUBED) ** (1.0 / 3.0)


def u_ratio_from_phi(phi: float) -> float:
    """``U / U0`` (sedimentation velocity ratio) of the Hasimoto SC series."""
    c = float(phi)
    if c < 0.0:
        raise ValueError(f"volume fraction must be non-negative, got {phi}")
    return 1.0 + sum(coef * c**power for coef, power in U_RATIO_PHI_TERMS)


def array_drag_factor_from_phi(phi: float) -> float:
    """``K = F / (6 pi mu a U)`` = ``U0 / U``, as a function of volume fraction."""
    ratio = u_ratio_from_phi(phi)
    if ratio <= 0.0:
        raise ValueError(f"Hasimoto series leaves its radius of convergence at phi={phi}")
    return 1.0 / ratio


def array_drag_factor_from_lambda(lam: float) -> float:
    """``K = F / (6 pi mu a U)`` as a function of ``lam = a / L_cell``.

    Exact geometric conversion to the locked volume-fraction form; no
    re-rounding of coefficients happens here.
    """
    return array_drag_factor_from_phi(phi_from_lambda(lam))


def array_drag_factor_from_lambda_truncated(lam: float) -> float:
    """Historical ``lambda``-form truncation of the same series, through ``lambda^6``.

    ``U/U0 = 1 - 2.83730 lam + 4.18879 lam^3 - 27.359 lam^6`` with the
    coefficients produced by :func:`lambda_series_coefficients`.  Kept so that
    call sites which expect a closed-form ``lambda`` polynomial (and their
    locked regression numbers) can state *which* truncation they carry; it is
    0.05 % below :func:`array_drag_factor_from_lambda` at ``lam = 1/6``.
    """
    lam = float(lam)
    c = lambda_series_coefficients()
    den = 1.0 + c["lambda1"] * lam + c["lambda3"] * lam**3 + c["lambda6"] * lam**6
    if den <= 0.0:
        raise ValueError(f"lambda series out of range at lam={lam}")
    return 1.0 / den


def lambda_series_coefficients() -> dict[str, float]:
    """Coefficients of the ``lambda`` form of the ``U/U0`` series (derived).

    Keys are ``"lambda"``, ``"lambda3"`` … ; values are
    ``coef * PHI_PER_LAMBDA_CUBED**power`` for each ``phi**power`` term, so the
    identity is checked against the volume-fraction series without any typed
    constant beyond the literature's ``phi`` coefficients.
    """
    out: dict[str, float] = {}
    for coef, power in U_RATIO_PHI_TERMS:
        # phi**power = (PHI_PER_LAMBDA_CUBED * lambda**3)**power, so the
        # lambda exponent is 3*power: 1/3 -> 1, 1 -> 3, 2 -> 6, 8/3 -> 8, 10/3 -> 10.
        out[f"lambda{int(round(3 * power))}"] = coef * PHI_PER_LAMBDA_CUBED**power
    return out


def is_series_valid(phi: float) -> bool:
    """True while the truncated series is within 1 % of the locked SC table."""
    return 0.0 <= float(phi) <= SERIES_MAX_PHI


__all__ = [
    "PHI_PER_LAMBDA_CUBED",
    "SERIES_MAX_LAMBDA",
    "SERIES_MAX_PHI",
    "U_RATIO_PHI_TERMS",
    "array_drag_factor_from_lambda",
    "array_drag_factor_from_lambda_truncated",
    "array_drag_factor_from_phi",
    "is_series_valid",
    "lambda_from_phi",
    "lambda_series_coefficients",
    "phi_from_lambda",
    "u_ratio_from_phi",
]