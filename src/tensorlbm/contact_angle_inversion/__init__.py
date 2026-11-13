"""Contact-angle inversion for SC-LBM wetting walls.

Implements the Huang-et-al. (2007) insight that the simple Young relation
``cos(theta) = (psi_w - psi_v) / (psi_l - psi_v)`` is systematically wrong
because the wall pseudopotential acts on a single lattice row while the
fluid-fluid interaction spans the entire diffuse interface.

The practical remedy is a **calibration-based inversion**:
1. Sweep ``f_w = psi_w / psi_l`` on the target grid and measure ``theta(f_w)``.
2. Fit a smooth curve (cubic spline in ``cos(theta)`` vs ``f_w``).
3. Invert: given ``theta_target``, return ``f_w`` via the fitted curve.

This module provides:
- :func:`calibrate` — run the sweep and return the calibration object.
- :func:`invert` — given a calibration and target angle, return ``f_w``.
- :func:`forward` — given a calibration and ``f_w``, return predicted ``theta``.
- :class:`CalibrationResult` — holds the fitted curve and metadata.

References
----------
Huang, H., Thorne, D. T., Schaap, M. G., & Sukop, M. C. (2007).
"Proposed approximation for contact angles in Shan-and-Chen lattice
Boltzmann multicomponent simulations." *Physical Review E*, 76(6), 066701.

Benzi, R., Biferale, L., Sbragaglia, M., Succi, S., & Toschi, F. (2006).
"Mesoscopic modeling of a two-phase flow in the presence of boundaries:
the contact angle." *Physical Review E*, 74(2), 021509.
"""

from __future__ import annotations

from .calibration import (
    CalibrationResult,
    calibrate,
    forward,
    invert,
    young_prediction,
)

__all__ = [
    "CalibrationResult",
    "calibrate",
    "forward",
    "invert",
    "young_prediction",
]
