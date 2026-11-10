"""Digitised experimental references used by the P22W2-N09 acceptance criteria.

The numbers are the same NACA0015 / Clark-Y digests that the repo already
carries in ``tensorlbm.cavitation_airfoil.reference_data()``; they are pulled
through that function (single source of truth, no hand copy) and re-expressed
here with an explicit provenance record so that every acceptance number in
``validation.json`` can be traced back to a citation.

Provenance
----------
``SHEET_REFERENCE``  NACA0015, alpha = 5 deg, water tunnel, sheet cavitation
    length versus cavitation number.  Digest of Brennen (1994) *Hydrodynamics
    of Pumps* / Brennen (1995) *Cavitation and Bubble Dynamics* ch. 7, with the
    Merkle-Feng-Buelow (1998) numerical solution carried alongside.
``CLOUD_REFERENCE``  Clark-Y, alpha = 6 deg: cloud shedding Strouhal number
    ``St_c = f c / U`` and cavity length.  Franc & Michel (1985) JFM 154, 63-90;
    Franc (2001) CAV2001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cavitation_airfoil import reference_data

__all__ = [
    "CLOUD_REFERENCE",
    "SHEET_REFERENCE",
    "SheetReference",
    "CloudReference",
    "sigma_critical_reference",
]


@dataclass(frozen=True)
class SheetReference:
    """Sheet-cavity length reference for NACA0015 at 5 deg."""

    case: str
    sigma: np.ndarray
    l_cav_exp: np.ndarray
    l_cav_numeric: np.ndarray
    band_l_cav: float
    source: str

    def l_cav_curve(self, sigma: float) -> float:
        """Campaign curve ``L/c(sigma)`` from a log-log fit of the digitised points."""
        return float(np.exp(np.polyval(np.polyfit(np.log(self.sigma), np.log(self.l_cav_exp), 1), np.log(sigma))))

    def derivative(self, sigma: float, h: float = 1e-3) -> float:
        """``d(L/c)/d(sigma)`` of the digitised curve (log-log slope carried back)."""
        return (self.l_cav_curve(sigma + h) - self.l_cav_curve(sigma - h)) / (2.0 * h)


@dataclass(frozen=True)
class CloudReference:
    """Cloud-shedding reference (Clark-Y, 6 deg)."""

    case: str
    sigma: np.ndarray
    st_exp: np.ndarray
    l_cav_exp: np.ndarray
    source: str
    note: str


_raw = reference_data()

SHEET_REFERENCE = SheetReference(
    case=_raw["sheet"]["case"],
    sigma=np.asarray(_raw["sheet"]["sigma"], dtype=float),
    l_cav_exp=np.asarray(_raw["sheet"]["l_cav_exp"], dtype=float),
    l_cav_numeric=np.asarray(_raw["sheet"]["l_cav_merkle"], dtype=float),
    band_l_cav=float(_raw["sheet"]["band_l_cav"]),
    source=_raw["sheet"]["source"],
)

CLOUD_REFERENCE = CloudReference(
    case=_raw["cloud"]["case"],
    sigma=np.asarray(_raw["cloud"]["sigma"], dtype=float),
    st_exp=np.asarray(_raw["cloud"]["st_exp"], dtype=float),
    l_cav_exp=np.asarray(_raw["cloud"]["l_cav_exp"], dtype=float),
    source=_raw["cloud"]["source"],
    note=_raw["cloud"]["note"],
)


def sigma_critical_reference(l_over_c_crit: float = 0.5) -> float:
    """Critical cavitation number implied by the digitised ``L/c(sigma)`` data.

    The sheet-to-cloud transition is the re-entrant-jet criterion
    ``L/c = 0.5``; solving it on the *experimental* curve gives the measured
    transition number.  The analytic log-log inversion of
    ``L/c = A sigma^-n`` is

    .. math::  \\sigma_{crit} = \\left(\\frac{A}{(L/c)_{crit}}\\right)^{1/n},

    so the reference is derived from measured points only -- nothing is typed in
    by hand and no model value enters it.
    """
    s = np.log(SHEET_REFERENCE.sigma)
    l = np.log(SHEET_REFERENCE.l_cav_exp)
    slope, intercept = np.polyfit(s, l, 1)
    return float(np.exp((np.log(l_over_c_crit) - intercept) / slope))