"""Ice-keel roughness geometry: ridge statistics -> ``k_s``, ``lambda``, LBM ribs.

Physical picture
----------------
The underside of a drifting ice sheet is not a plane: pressure-ridge keels
leave a population of deep, narrow obstacles whose drafts are comparable to
(or larger than) the level-ice thickness, separated by surviving level ice.
Two geometric scales control the turbulent boundary layer below the ice
(Makin 1982; Wadhams 2000; Lepparanta 2011):

``k_s``
    the *roughness height* -- the draft of the keels measured from the local
    level-ice underside.  It is the geometric analogue of Nikuradse's sand
    grain diameter, so it is reported in metres and converted to wall units
    ``k_s^+ = k_s u_tau / nu`` by the solver.
``lambda``
    the *spacing* of the keels (ridge spacing along the drift direction).
    Only the ratio ``lambda/k_s`` matters for the shape of the roughness
    function: small ratios are *d*-type (skimming flow over the cavities),
    large ratios approach isolated *k*-type elements that follow the
    sand-grain law.

The mission sweeps ``lambda/k_s = 2, 5, 10`` -- from ``d``-type (keels so
closely packed that the flow skims over the inter-keel water) to well
separated ``k``-type roughness.

The LBM channel resolves an idealised version of this geometry as spanwise
square ribs of height and width ``k_s`` at pitch ``lambda``
(:func:`tensorlbm.roughness_elements.transverse_rib_elements`).  The mapping is
explicitly an idealisation: real keels are 3-D, randomly spaced and partly
submerged in slush; the rib study isolates the *scaling* ``C_w(k_s^+,
lambda/k_s)`` that any ice-drag closure needs, not the exact geometry of one
keel field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from ..roughness_elements import RoughnessElementSet, transverse_rib_elements

__all__ = [
    "IceKeelRoughness",
    "LAMBDA_OVER_KS_LEVELS",
    "ice_keel_roughness",
    "default_ice_keel",
    "rib_elements_for_case",
    "keel_roughness_uncertainty",
]

#: The three spacing ratios of the campaign (d-type -> isolated k-type).
LAMBDA_OVER_KS_LEVELS: tuple[float, ...] = (2.0, 5.0, 10.0)


@dataclass(frozen=True)
class IceKeelRoughness:
    """Geometric roughness scales of an ice underside.

    Attributes
    ----------
    ks_m:
        Roughness (keel) height in metres above the level-ice underside.
    lambda_m:
        Mean spacing between keels in metres.
    keel_draft_m, sail_height_m:
        The underlying ridge statistics (reported for provenance).
    source:
        Where the numbers come from (dataset / publication / assumption).
    """

    ks_m: float
    lambda_m: float
    keel_draft_m: float = float("nan")
    sail_height_m: float = float("nan")
    source: str = ""

    def __post_init__(self) -> None:
        if not (self.ks_m > 0.0 and self.lambda_m > 0.0):
            raise ValueError("ks_m and lambda_m must be positive")

    @property
    def lambda_over_ks(self) -> float:
        return self.lambda_m / self.ks_m

    def as_dict(self) -> dict[str, Any]:
        return {
            "ks_m": float(self.ks_m),
            "lambda_m": float(self.lambda_m),
            "lambda_over_ks": float(self.lambda_over_ks),
            "keel_draft_m": float(self.keel_draft_m),
            "sail_height_m": float(self.sail_height_m),
            "source": self.source,
        }


def ice_keel_roughness(
    keel_drafts_m: Iterable[float],
    spacings_m: Iterable[float],
    *,
    level_ice_draft_m: float = 0.0,
    sail_heights_m: Iterable[float] | None = None,
    source: str = "survey",
) -> IceKeelRoughness:
    """Convert a ridge-keel population into ``(k_s, lambda)``.

    ``k_s`` is taken as the **root-mean-square keel draft above the level-ice
    underside**, i.e. the second moment of the obstacle-height distribution,
    which is the definition that reproduces Nikuradse's grain diameter for a
    random granular bed (the alternative, the maximum draft, is dominated by
    single deep keels and is reported as an uncertainty bound by
    :func:`keel_roughness_uncertainty`).

    ``lambda`` is the mean spacing of the keels along the drift direction.

    Parameters
    ----------
    keel_drafts_m:
        Draft of each mapped keel, measured from the water line (positive).
    spacings_m:
        Distance to the neighbouring keel along the drift direction, metres.
    level_ice_draft_m:
        Draft of the undisturbed level ice around the ridges; subtracted from
        the keel drafts so that ``k_s`` is measured from the ice underside.
    """
    drafts = np.asarray([float(v) for v in keel_drafts_m], dtype=float)
    spac = np.asarray([float(v) for v in spacings_m], dtype=float)
    if drafts.size == 0 or spac.size == 0:
        raise ValueError("need at least one keel draft and one spacing")
    if np.any(drafts <= level_ice_draft_m):
        raise ValueError("every keel draft must exceed level_ice_draft_m")
    if np.any(spac <= 0.0):
        raise ValueError("spacings must be positive")
    heights = drafts - float(level_ice_draft_m)
    ks = float(np.sqrt(np.mean(heights**2)))
    lam = float(np.mean(spac))
    sail = float(np.nanmean(np.asarray(list(sail_heights_m), dtype=float))) if sail_heights_m is not None else float("nan")
    return IceKeelRoughness(ks_m=ks, lambda_m=lam, keel_draft_m=float(np.mean(drafts)),
                            sail_height_m=sail, source=source)


def keel_roughness_uncertainty(
    keel_drafts_m: Iterable[float],
    spacings_m: Iterable[float],
    *,
    level_ice_draft_m: float = 0.0,
) -> dict[str, float]:
    """Spread of ``k_s`` and ``lambda/k_s`` induced by the estimator choice.

    ``k_s`` from the RMS draft (used) versus the maximum draft (upper bound)
    and the mean draft (lower bound); ``lambda/k_s`` from the mean versus the
    minimum spacing.  Reported so that the campaign can state which part of
    the ``C_w`` band is geometric uncertainty rather than turbulence physics.
    """
    drafts = np.asarray([float(v) for v in keel_drafts_m], dtype=float) - float(level_ice_draft_m)
    spac = np.asarray([float(v) for v in spacings_m], dtype=float)
    ks_rms = float(np.sqrt(np.mean(drafts**2)))
    ks_max = float(drafts.max())
    ks_mean = float(drafts.mean())
    lam = float(spac.mean())
    return {
        "ks_rms_m": ks_rms,
        "ks_max_m": ks_max,
        "ks_mean_m": ks_mean,
        "lambda_mean_m": lam,
        "lambda_min_m": float(spac.min()),
        "lambda_over_ks_rms": lam / ks_rms,
        "lambda_over_ks_lower": float(spac.min()) / ks_max,
        "lambda_over_ks_upper": float(spac.max()) / ks_mean,
    }


def default_ice_keel(*, lambda_over_ks: float = 5.0) -> IceKeelRoughness:
    """Reference ice underside used for the closure tables.

    ``k_s = 0.35 m`` is a mid-range keel height above a 1.5 m level-ice
    underside (Wadhams 2000 reports keel drafts of 5-15 m for large ridges but
    the *area-averaged* underside relief seen by the boundary layer is the
    surviving level ice plus the shallow keel population, i.e. O(0.1-1 m));
    ``lambda`` follows from the requested ratio.  These numbers only set the
    axes of the closure table -- the closure itself is a function of
    ``k_s^+`` and ``lambda/k_s`` and is therefore scale free.
    """
    ks = 0.35
    return IceKeelRoughness(ks_m=ks, lambda_m=float(lambda_over_ks) * ks,
                            keel_draft_m=1.85, sail_height_m=0.6,
                            source="reference underside: ks=0.35 m above 1.5 m level ice, "
                                   "ratio as requested (Wadhams 2000 keel-draft range)")


def rib_elements_for_case(
    nx: int,
    ny: int,
    nz: int,
    ks_lattice: float,
    lambda_over_ks: float,
    *,
    width_over_ks: float = 1.0,
    phase: int = 0,
) -> RoughnessElementSet:
    """Voxelised spanwise ribs with height ``k_s`` and pitch ``lambda``.

    Thin wrapper around
    :func:`tensorlbm.roughness_elements.transverse_rib_elements` that makes
    the ice-keel parameterisation explicit (``pitch_over_ks`` is the
    ``lambda/k_s`` of the closure) and refuses cases whose rib height would
    not fit the channel.
    """
    if not (ks_lattice >= 1.0):
        raise ValueError(f"ks_lattice={ks_lattice:g} < 1 lattice node: the keel would be unresolved")
    if not (lambda_over_ks >= 1.5):
        raise ValueError(f"lambda/ks={lambda_over_ks:g} < 1.5: keels would merge into a wall")
    if ks_lattice > 0.45 * (nz - 2) / 2.0 * 2.0:  # keep the module's own guard authoritative
        pass
    return transverse_rib_elements(
        nx, ny, nz, float(ks_lattice),
        pitch_over_ks=float(lambda_over_ks),
        width_over_ks=float(width_over_ks),
        phase=int(phase),
    )


def geometry_summary(elements: RoughnessElementSet | None) -> Mapping[str, Any]:
    """JSON-ready geometry provenance for a case record."""
    if elements is None:
        return {"kind": "smooth", "ks_lattice": 0.0}
    st = dict(elements.stats())
    p = dict(elements.params)
    st["pitch_over_ks"] = p.get("pitch_over_ks")
    st["n_ribs"] = p.get("n_ribs")
    st["rib_width"] = p.get("width")
    st["rib_height"] = p.get("height")
    return st