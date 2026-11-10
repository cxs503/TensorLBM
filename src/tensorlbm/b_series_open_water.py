"""Wageningen B-series open-water polynomial — reference data provider.

Purpose
-------
This module is *reference data only*: it evaluates the published regression
polynomials of Oosterveld & van Oossanen (1975) for the Wageningen B-series
propeller family.  It is used by the actuator-line (ALM) work in
:mod:`tensorlbm.actuator_line` as the independent yardstick for ``K_T`` /
``K_Q`` — it is **not** a model of anything, so agreement with it is a genuine
prediction test rather than a circular one.

Provenance
----------
The coefficient tables below reproduce the classical 39-term ``K_T`` and
47-term ``K_Q`` polynomials published by

    Oosterveld, M. & van Oossanen, P. (1975), "Further computer-analyzed data
    of the Wageningen B-screw series", International Shipbuilding Progress,
    22(251).

They were previously carried in this repository inside
``src/tensorlbm/propeller_cad.py`` (commit ``b33f474b``) and were dropped by a
later refactor that re-added a slimmer ``propeller_cad``; this module restores
them verbatim as a standalone, documented reference surface so that propeller
validation work has a citable baseline again.  Polynomial form:

    K_T = sum_n C_n J^{s_J} (P/D)^{s_P} (A_E/A_0)^{s_E} Z^{s_Z}
    K_Q = sum_n C_n J^{s_J} (P/D)^{s_P} (A_E/A_0)^{s_E} Z^{s_Z}

Validity envelope (Oosterveld & van Oossanen): J in [0, 1.5],
P/D in [0.5, 1.4], A_E/A_0 in [0.30, 1.05], Z in [2, 7].  The regression is
built from model-scale open-water tests at R_n = 2e6 for the blade sections;
full-scale corrections (Lerbs / ITTC-78 blade friction) are *not* applied.

Public API
----------
- :class:`BSeriesPropeller`   — immutable reference propeller definition.
- :func:`b_series_open_water` — dict-returning convenience wrapper.
- :data:`KT_COEFFS`, :data:`KQ_COEFFS` — coefficient tables (rows:
  ``(C, s_J, s_PD, s_EAR, s_Z)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "KT_COEFFS",
    "KQ_COEFFS",
    "BSeriesPropeller",
    "b_series_open_water",
]

# Each row: (C, exponent of J, exponent of P/D, exponent of A_E/A_0, exponent of Z)
KT_COEFFS = np.array([
    (0.00880496, 0, 0, 0, 0),
    (-0.204554, 1, 0, 0, 0),
    (0.166351, 0, 1, 0, 0),
    (0.158114, 0, 2, 0, 0),
    (-0.147581, 2, 0, 1, 0),
    (-0.481497, 1, 1, 1, 0),
    (0.415437, 0, 2, 1, 0),
    (0.0144043, 0, 0, 0, 1),
    (-0.0530054, 2, 0, 0, 1),
    (0.0143481, 0, 1, 0, 1),
    (0.0606826, 1, 1, 0, 1),
    (-0.0125894, 0, 0, 1, 1),
    (0.0109689, 1, 0, 1, 1),
    (-0.133698, 0, 3, 0, 0),
    (0.00638407, 0, 6, 0, 0),
    (-0.00132718, 2, 6, 0, 0),
    (0.168496, 3, 0, 1, 0),
    (-0.0507214, 0, 0, 2, 0),
    (0.0854559, 2, 0, 2, 0),
    (-0.0504475, 3, 0, 2, 0),
    (0.010465, 1, 6, 2, 0),
    (-0.00648272, 2, 6, 2, 0),
    (-0.00841728, 0, 3, 0, 1),
    (0.0168424, 1, 3, 0, 1),
    (-0.00102296, 3, 3, 0, 1),
    (-0.0317791, 0, 3, 1, 1),
    (0.018604, 1, 0, 2, 1),
    (-0.00410798, 0, 2, 2, 1),
    (-0.000606848, 0, 0, 0, 2),
    (-0.0049819, 1, 0, 0, 2),
    (0.0025983, 2, 0, 0, 2),
    (-0.000560528, 3, 0, 0, 2),
    (-0.00163652, 1, 2, 0, 2),
    (-0.000328787, 1, 6, 0, 2),
    (0.000116502, 2, 6, 0, 2),
    (0.000690904, 0, 0, 1, 2),
    (0.00421749, 0, 3, 1, 2),
    (0.0000565229, 3, 6, 1, 2),
    (-0.00146564, 0, 3, 2, 2),
], dtype=np.float64)

KQ_COEFFS = np.array([
    (0.00379368, 0, 0, 0, 0),
    (0.00886523, 2, 0, 0, 0),
    (-0.032241, 1, 1, 0, 0),
    (0.00344778, 0, 2, 0, 0),
    (-0.0408811, 0, 1, 1, 0),
    (-0.108009, 1, 1, 1, 0),
    (-0.0885381, 2, 1, 1, 0),
    (0.188561, 0, 2, 1, 0),
    (-0.00370871, 1, 0, 0, 1),
    (0.00513696, 0, 1, 0, 1),
    (0.0209449, 1, 1, 0, 1),
    (0.00474319, 2, 1, 0, 1),
    (-0.00723408, 2, 0, 1, 1),
    (0.00438388, 1, 1, 1, 1),
    (-0.0269403, 0, 2, 1, 1),
    (0.0558082, 3, 0, 1, 0),
    (0.0161886, 0, 3, 1, 0),
    (0.00318086, 1, 3, 1, 0),
    (0.015896, 0, 0, 2, 0),
    (0.0471729, 1, 0, 2, 0),
    (0.0196283, 3, 0, 2, 0),
    (-0.0502782, 0, 1, 2, 0),
    (-0.030055, 3, 1, 2, 0),
    (0.0417122, 2, 2, 2, 0),
    (-0.0397722, 0, 3, 2, 0),
    (-0.00350024, 0, 6, 2, 0),
    (-0.0106854, 3, 0, 0, 1),
    (0.00110903, 3, 3, 0, 1),
    (-0.000313912, 0, 6, 0, 1),
    (0.0035985, 3, 0, 1, 1),
    (-0.00142121, 0, 6, 1, 1),
    (-0.00383637, 1, 0, 2, 1),
    (0.0126803, 0, 2, 2, 1),
    (-0.00318278, 2, 3, 2, 1),
    (0.00334268, 0, 6, 2, 1),
    (-0.00183491, 1, 1, 0, 2),
    (0.000112451, 3, 2, 0, 2),
    (-0.0000297228, 3, 6, 0, 2),
    (0.000269551, 1, 0, 1, 2),
    (0.00083265, 2, 0, 1, 2),
    (0.00155334, 0, 2, 1, 2),
    (0.000302683, 0, 6, 1, 2),
    (-0.0001843, 0, 0, 2, 2),
    (-0.000425399, 0, 3, 2, 2),
    (0.0000869243, 3, 3, 2, 2),
    (-0.000465899, 0, 6, 2, 2),
    (0.0000554194, 1, 6, 2, 2),
], dtype=np.float64)


def _eval_poly(coeffs: np.ndarray, j_val: float, p_d: float, ae_a0: float, z_blades: int) -> float:
    """Evaluate one B-series polynomial (vectorised over the coefficient table)."""
    c = coeffs[:, 0]
    s_j = coeffs[:, 1]
    s_p = coeffs[:, 2]
    s_e = coeffs[:, 3]
    s_z = coeffs[:, 4]
    return float(
        np.sum(
            c
            * (float(j_val) ** s_j)
            * (float(p_d) ** s_p)
            * (float(ae_a0) ** s_e)
            * (float(z_blades) ** s_z)
        )
    )


@dataclass(frozen=True)
class BSeriesPropeller:
    """A Wageningen B-series propeller defined by (P/D, A_E/A_0, Z).

    Parameters
    ----------
    p_d : float
        Pitch ratio P/D at 0.7R (the B-series regression variable).
    ae_a0 : float
        Expanded blade area ratio A_E/A_0.
    z_blades : int
        Number of blades Z.
    """

    p_d: float = 1.0
    ae_a0: float = 0.70
    z_blades: int = 4

    def __post_init__(self) -> None:
        if not 0.5 <= self.p_d <= 1.4:
            raise ValueError(f"P/D={self.p_d} outside B-series validity [0.5, 1.4]")
        if not 0.30 <= self.ae_a0 <= 1.05:
            raise ValueError(f"A_E/A_0={self.ae_a0} outside B-series validity [0.30, 1.05]")
        if not 2 <= int(self.z_blades) <= 7:
            raise ValueError(f"Z={self.z_blades} outside B-series validity [2, 7]")

    @property
    def label(self) -> str:
        return f"B{int(self.z_blades)}-{int(round(self.ae_a0 * 100))} (P/D={self.p_d:g})"

    def kt(self, j_val: float) -> float:
        """Thrust coefficient ``K_T(J)`` (not clipped: can go slightly negative)."""
        return _eval_poly(KT_COEFFS, j_val, self.p_d, self.ae_a0, int(self.z_blades))

    def kq(self, j_val: float) -> float:
        """Torque coefficient ``K_Q(J)`` (may be clipped at a tiny positive value)."""
        return max(_eval_poly(KQ_COEFFS, j_val, self.p_d, self.ae_a0, int(self.z_blades)), 1e-12)

    def eta0(self, j_val: float) -> float:
        """Open-water efficiency ``eta_0 = J K_T / (2 pi K_Q)``."""
        if j_val <= 0.0:
            return 0.0
        return j_val * self.kt(j_val) / (2.0 * math.pi * self.kq(j_val))

    def j_at_zero_thrust(self, j_lo: float = 0.0, j_hi: float = 1.5) -> float:
        """Advance ratio at which ``K_T`` vanishes (bisection)."""
        if self.kt(j_lo) <= 0.0:
            return float(j_lo)
        if self.kt(j_hi) >= 0.0:
            return float(j_hi)
        for _ in range(80):
            j_mid = 0.5 * (j_lo + j_hi)
            if self.kt(j_mid) > 0.0:
                j_lo = j_mid
            else:
                j_hi = j_mid
        return 0.5 * (j_lo + j_hi)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "P_D": float(self.p_d),
            "Ae_A0": float(self.ae_a0),
            "Z": int(self.z_blades),
            "J_zero_thrust": self.j_at_zero_thrust(),
        }


def b_series_open_water(
    j_val: float,
    p_d: float,
    ae_a0: float,
    z_blades: int,
    *,
    rho: float = 1025.0,
    n_rps: float | None = None,
    diameter: float | None = None,
) -> dict[str, float]:
    """Convenience wrapper: open-water coefficients of one B-series propeller.

    Returns a dict with ``J``, ``KT``, ``KQ``, ``eta_0`` and — when ``n_rps``
    and ``diameter`` are supplied — dimensional ``T_N`` [N] and ``Q_Nm`` [N m].
    """
    prop = BSeriesPropeller(p_d=p_d, ae_a0=ae_a0, z_blades=int(z_blades))
    kt = prop.kt(j_val)
    kq = prop.kq(j_val)
    out: dict[str, float] = {
        "J": float(j_val),
        "KT": kt,
        "KQ": kq,
        "eta_0": prop.eta0(j_val),
    }
    if n_rps is not None and diameter is not None:
        out["T_N"] = rho * float(n_rps) ** 2 * float(diameter) ** 4 * kt
        out["Q_Nm"] = rho * float(n_rps) ** 2 * float(diameter) ** 5 * kq
        out["P_kW"] = 2.0 * math.pi * float(n_rps) * out["Q_Nm"] / 1000.0
    return out