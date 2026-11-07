"""SUBOFF AFF-1/AFF-8 resistance anchor lane (IN25-T6 seat B, 证一楼).

The mission requires a *second* resistance anchor for DARPA SUBOFF that is
independent of the ITTC correlation line, with a Re ladder and traced
provenance.  This module builds that lane from the primary tow-tank forces that
are already archived in :mod:`tensorlbm.suboff_reference_data`
(``SUBOFF_TOW_TANK_RESISTANCE_TABLE14``, Liu & Huang 1998, Table 14) and then
audits every candidate anchor against it and against the ITTC-1957 line.

Conventions are data, not prose: the density, kinematic viscosity and wetted
area are module constants with stated ranges, and
:func:`convention_sensitivity` reports how far CT moves inside those ranges.
The AFF-8 wetted area is *not* known to better than a bracket offline, which is
recorded as an absent item in :mod:`tensorlbm.external_survey.lit_anchors`;
:func:`aff8_ct_bracket` therefore returns an interval, not a point value.

Findings that the acceptance criteria depend on (computed, not asserted):

* AFF-1 reproduces the independently published CT ≈ 0.0031 (Roddy 1990 /
  SIMMAN-2008 usage) to ~1.5 % -> AFF-1 lane is cross-consistent.
* CT(AFF-8) ≈ 0.0035 for the whole plausible wetted-area bracket at
  Re ≈ 1.17e7, so the widely quoted "AFF-8 CT ≈ 0.004" is *not* reproducible
  from the primary forces: reproducing it would need an AFF-8 wetted area
  (≈ 5.52 m²) smaller than the bare-hull wetted area, which is unphysical.
  That anchor is therefore withheld from the scorer (honest FAIL, not a bug).
* ITTC-1957 x constant form factor reproduces the whole AFF-1 ladder to a few
  per cent, i.e. the two anchor families are mutually consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tensorlbm.suboff_reference_data import SUBOFF_TOW_TANK_RESISTANCE_TABLE14

__all__ = [
    "SUBOFF_LENGTH_M",
    "SUBOFF_MAX_RADIUS_M",
    "NU_WATER",
    "RHO_WATER",
    "S_WET_AFF1",
    "S_WET_AFF8_BRACKET",
    "KNOT_TO_MPS",
    "LadderPoint",
    "AnchorConsistency",
    "ittc57_cf",
    "schoenherr_cf",
    "ct_from_resistance",
    "ladder",
    "aff1_ladder",
    "aff8_ladder",
    "form_factor_series",
    "form_factor_mean",
    "ittc_ladder_reproduction",
    "aff8_ct_bracket",
    "convention_sensitivity",
    "appendage_force_ratio",
    "anchor_consistency",
    "scorer_candidates",
]

# --- geometry / fluid constants (provenance in the module docstring notes) ---
SUBOFF_LENGTH_M = 4.356     # hull length L  (DARPA SUBOFF, L/D = 8.57)
SUBOFF_MAX_RADIUS_M = 0.254  # max radius D/2
NU_WATER = 1.139e-6          # m^2/s, docs/IN12_S6_WALL_LADDER.md (ambient water)
RHO_WATER = 999.0            # kg/m^3, fresh-water tow tank ~15 C
RHO_BRACKET = (998.2, 1025.9)  # fresh water .. sea water (sensitivity only)
S_WET_AFF1 = 5.998           # m^2, docs/IN12_S6_WALL_LADDER.md (AFF-8 stated there)
S_WET_AFF1_NOTE = (
    "AFF-1 wetted area 5.998 m2 as recorded in docs/IN12_S6_WALL_LADDER.md; the "
    "primary geometry report (Groves, Huang & Chang 1989) is not available offline."
)
S_WET_AFF8_BRACKET = (6.00, 6.35)  # m^2 bracket; exact value is an ABSENT item
KNOT_TO_MPS = 0.514444

# independent published anchors used for the cross-consistency test
PUBLISHED_AFF1_CT = 0.0031   # Roddy 1990 (DTRC/SHD-1298-07) usage; SIMMAN-2008
PUBLISHED_AFF1_RE = 1.2e7
PUBLISHED_AFF8_CT = 0.0040   # widely quoted "AFF-8 CT ~ 0.004"
PUBLISHED_AFF8_RE = 1.2e7

CONSISTENCY_TOL_PCT = 10.0   # cross-consistency gate (tighter than the 15 % rule)


def ittc57_cf(re: float) -> float:
    """ITTC-1957 model-ship correlation line ``0.075/(log10 Re - 2)^2``."""
    if re <= 100.0:
        raise ValueError(f"ITTC-1957 needs Re > 100, got {re}")
    return 0.075 / (math.log10(re) - 2.0) ** 2


def schoenherr_cf(re: float) -> float:
    """Schoenherr (1932) flat-plate friction line ``0.455/(log10 Re)^2.58``."""
    if re <= 100.0:
        raise ValueError(f"Schoenherr needs Re > 100, got {re}")
    return 0.455 / math.log10(re) ** 2.58


def ct_from_resistance(resistance_n: float, speed_mps: float, s_wet: float,
                       rho: float = RHO_WATER) -> float:
    """``CT = R / (0.5 rho U^2 S)`` — wetted-area normalisation."""
    if s_wet <= 0.0 or speed_mps <= 0.0:
        raise ValueError("s_wet and speed must be positive")
    return resistance_n / (0.5 * rho * speed_mps**2 * s_wet)


@dataclass(frozen=True)
class LadderPoint:
    configuration: int
    speed_kn: float
    speed_mps: float
    re: float
    resistance_n: float
    ct: float
    s_wet: float


def _points(configuration: int, s_wet: float, rho: float = RHO_WATER) -> tuple[LadderPoint, ...]:
    out = []
    for p in SUBOFF_TOW_TANK_RESISTANCE_TABLE14:
        if p.configuration != configuration:
            continue
        u = p.speed_mps
        re = u * SUBOFF_LENGTH_M / NU_WATER
        ct = ct_from_resistance(p.resistance_n, u, s_wet, rho)
        out.append(LadderPoint(configuration, p.speed_knots, u, re, p.resistance_n, ct, s_wet))
    return tuple(out)


def ladder(configuration: int, *, s_wet: float | None = None,
           rho: float = RHO_WATER) -> tuple[LadderPoint, ...]:
    """EFD resistance ladder for AFF-1 (config 1) or AFF-8 (config 8)."""
    if configuration == 1:
        return _points(1, S_WET_AFF1 if s_wet is None else s_wet, rho)
    if configuration == 8:
        return _points(8, 0.5 * sum(S_WET_AFF8_BRACKET) if s_wet is None else s_wet, rho)
    raise ValueError("SUBOFF Table 14 carries configurations 1 and 8 only")


def aff1_ladder() -> tuple[LadderPoint, ...]:
    return ladder(1)


def aff8_ladder() -> tuple[LadderPoint, ...]:
    return ladder(8)


def form_factor_series(configuration: int = 1, *, s_wet: float | None = None
                       ) -> tuple[tuple[float, float], ...]:
    """(Re, 1+k) with ``k = CT/Cf_ITTC - 1`` for each EFD ladder point."""
    return tuple((p.re, p.ct / ittc57_cf(p.re) - 1.0) for p in ladder(configuration, s_wet=s_wet))


def form_factor_mean(configuration: int = 1, *, s_wet: float | None = None) -> float:
    series = form_factor_series(configuration, s_wet=s_wet)
    return sum(k for _, k in series) / len(series)


def ittc_ladder_reproduction(configuration: int = 1, *, s_wet: float | None = None
                             ) -> tuple[dict, ...]:
    """Reproduce the EFD ladder with ``CT = Cf_ITTC(Re) * (1 + k_mean)``.

    This is the machine form of "ITTC anchor and EFD anchor agree": the model
    has exactly one fitted constant (the form factor), so a small residual over
    the whole Re ladder is a genuine cross-consistency statement.
    """
    k = form_factor_mean(configuration, s_wet=s_wet)
    rows = []
    for p in ladder(configuration, s_wet=s_wet):
        pred = ittc57_cf(p.re) * (1.0 + k)
        rows.append(
            {
                "re": p.re,
                "speed_kn": p.speed_kn,
                "ct_efd": p.ct,
                "ct_ittc_formfactor": pred,
                "err_pct": 100.0 * (pred - p.ct) / p.ct,
                "form_factor": k,
            }
        )
    return tuple(rows)


def aff8_ct_bracket(*, rho: float = RHO_WATER) -> dict:
    """CT(AFF-8) interval implied by the wetted-area bracket."""
    s_lo, s_hi = S_WET_AFF8_BRACKET
    ct_lo_area = ladder(8, s_wet=s_lo, rho=rho)[0].ct
    ct_hi_area = ladder(8, s_wet=s_hi, rho=rho)[0].ct
    return {
        "re_ladder": [p.re for p in ladder(8, rho=rho)],
        # note: CT scales as 1/S_wet, so the *smaller* area gives the *larger* CT
        "ct_using_s_wet_lo": ct_lo_area,
        "ct_using_s_wet_hi": ct_hi_area,
        "interval": (min(ct_lo_area, ct_hi_area), max(ct_lo_area, ct_hi_area)),
        "s_wet_bracket": S_WET_AFF8_BRACKET,
        "rho": rho,
    }


def appendage_force_ratio() -> tuple[tuple[float, float], ...]:
    """(speed_kn, R_AFF8 / R_AFF1) at matched carriage speeds.

    Area- and density-free, hence the cleanest observable of the appendage
    increment in the primary data.
    """
    out = []
    for bare in aff1_ladder():
        full = min(aff8_ladder(), key=lambda p: abs(p.speed_kn - bare.speed_kn))
        if abs(full.speed_kn - bare.speed_kn) / bare.speed_kn > 0.02:
            continue  # carriage speeds too far apart to form a ratio
        out.append((bare.speed_kn, full.resistance_n / bare.resistance_n))
    return tuple(out)


def convention_sensitivity() -> dict:
    """How far the AFF-8 CT anchor moves with the (S_wet, rho) conventions."""
    point = aff8_ladder()[0]
    base = point.ct
    alt_rho = ct_from_resistance(point.resistance_n, point.speed_mps, point.s_wet,
                                 RHO_BRACKET[1])
    s_lo, s_hi = S_WET_AFF8_BRACKET
    # S_wet that would be needed to reproduce the published AFF-8 CT at this speed:
    implied_s = point.resistance_n / (0.5 * RHO_WATER * point.speed_mps**2 * PUBLISHED_AFF8_CT)
    return {
        "re": point.re,
        "speed_kn": point.speed_kn,
        "ct_base": base,
        "s_wet_used": point.s_wet,
        "rho_used": RHO_WATER,
        "sensitivity_s_lo_pct": 100.0 * (ct_from_resistance(
            point.resistance_n, point.speed_mps, s_lo) - base) / base,
        "sensitivity_s_hi_pct": 100.0 * (ct_from_resistance(
            point.resistance_n, point.speed_mps, s_hi) - base) / base,
        "sensitivity_rho_sea_water_pct": 100.0 * (alt_rho - base) / base,
        "implied_s_wet_for_published_ct_m2": implied_s,
        "aff1_wetted_area_m2": S_WET_AFF1,
        "published_ct_physically_reachable": implied_s > S_WET_AFF1,
    }


@dataclass(frozen=True)
class AnchorConsistency:
    label: str
    claimed: float
    claimed_re: float
    reproduced: float
    reproduced_re: float
    dev_pct: float
    verdict: str            # cross_consistent | inconsistent | withheld
    note: str
    dev_pct_bracket: tuple[float, float] | None = None

    @property
    def usable(self) -> bool:
        return self.verdict == "cross_consistent"


def anchor_consistency() -> tuple[AnchorConsistency, ...]:
    """Audit every published SUBOFF resistance value against the primary data."""
    a1 = min(aff1_ladder(), key=lambda p: abs(p.re - PUBLISHED_AFF1_RE))
    bracket = aff8_ct_bracket()
    a8 = min(aff8_ladder(), key=lambda p: abs(p.re - PUBLISHED_AFF8_RE))
    dev1 = 100.0 * (a1.ct - PUBLISHED_AFF1_CT) / PUBLISHED_AFF1_CT
    dev8 = 100.0 * (a8.ct - PUBLISHED_AFF8_CT) / PUBLISHED_AFF8_CT
    sens = convention_sensitivity()
    out = [
        AnchorConsistency(
            label="SUBOFF AFF-1 CT",
            claimed=PUBLISHED_AFF1_CT,
            claimed_re=PUBLISHED_AFF1_RE,
            reproduced=a1.ct,
            reproduced_re=a1.re,
            dev_pct=dev1,
            verdict="cross_consistent" if abs(dev1) <= CONSISTENCY_TOL_PCT else "inconsistent",
            note=(
                "Roddy (1990)/SIMMAN-2008 usage value; reproduced from Liu & Huang Table 14 "
                "forces with S_wet=5.998 m2, rho=999 kg/m3"
            ),
        ),
        AnchorConsistency(
            label="SUBOFF AFF-8 CT",
            claimed=PUBLISHED_AFF8_CT,
            claimed_re=PUBLISHED_AFF8_RE,
            reproduced=a8.ct,
            reproduced_re=a8.re,
            dev_pct=dev8,
            verdict="withheld" if abs(dev8) > CONSISTENCY_TOL_PCT else "cross_consistent",
            note=(
                "0.004 cannot be reproduced from the primary forces: it would require an "
                f"AFF-8 wetted area of {sens['implied_s_wet_for_published_ct_m2']:.2f} m2, "
                f"i.e. smaller than the AFF-1 area ({S_WET_AFF1} m2) — unphysical. "
                "Treat as convention-unspecified (honest FAIL) and use the Table 14 CT instead."
            ),
            dev_pct_bracket=(
                100.0 * (bracket["ct_using_s_wet_lo"] - PUBLISHED_AFF8_CT) / PUBLISHED_AFF8_CT,
                100.0 * (bracket["ct_using_s_wet_hi"] - PUBLISHED_AFF8_CT) / PUBLISHED_AFF8_CT,
            ),
        ),
    ]
    return tuple(out)


def scorer_candidates() -> tuple[dict, ...]:
    """Anchor rows that may be handed to the T5 red team / scoring gate."""
    a1 = min(aff1_ladder(), key=lambda p: abs(p.re - PUBLISHED_AFF1_RE))
    a8 = min(aff8_ladder(), key=lambda p: abs(p.re - PUBLISHED_AFF8_RE))
    rep1 = ittc_ladder_reproduction(1)
    max_err = max(abs(r["err_pct"]) for r in rep1)
    lo, hi = aff8_ct_bracket()["interval"]
    return (
        {
            "id": "SUBOFF-AFF1-CT-LADDER",
            "quantity": "CT (wetted-area normalised total resistance)",
            "geometry": "DARPA SUBOFF bare hull (AFF-1), L=4.356 m, L/D=8.57",
            "points": [
                {"re": p.re, "speed_kn": p.speed_kn, "ct": p.ct, "u_mps": p.speed_mps}
                for p in aff1_ladder()
            ],
            "primary_point": {"re": a1.re, "ct": a1.ct},
            "uncertainty": {
                "s_wet": "5.998 m2 (documented in-repo; primary report unavailable offline)",
                "absolute_pct_estimate": 3.0,
            },
            "cross_checks": {
                "vs_published_ct_0.0031": next(
                    c.dev_pct for c in anchor_consistency() if c.label.endswith("AFF-1 CT")
                ),
                "vs_ittc57_x_form_factor_max_err_pct": max_err,
            },
            "conditions": (
                "Re 1.16e7-3.54e7 (based on L, nu=1.139e-6 m2/s); smooth model, no free "
                "surface, no propeller; CT normalised on wetted area, rho=999 kg/m3"
            ),
            "sources": [
                "Liu & Huang (1998) CRDKNSWC/HD-1298-11, Table 14 (archived as "
                "tensorlbm.suboff_reference_data.SUBOFF_TOW_TANK_RESISTANCE_TABLE14)",
                "Roddy (1990) DTRC/SHD-1298-07 (published CT comparison)",
                "ITTC-1957 correlation line (friction cross-check)",
            ],
            "verdict": "admit_to_scorer",
        },
        {
            "id": "SUBOFF-AFF8-CT-LADDER",
            "quantity": "CT (wetted-area normalised total resistance)",
            "geometry": "DARPA SUBOFF full appendages (AFF-8)",
            "points": [
                {"re": p.re, "speed_kn": p.speed_kn, "ct": p.ct, "u_mps": p.speed_mps}
                for p in aff8_ladder()
            ],
            "primary_point": {"re": a8.re, "ct": a8.ct, "interval_ct": (lo, hi)},
            "uncertainty": {
                "s_wet": (
                    f"bracketed {S_WET_AFF8_BRACKET} m2 (exact AFF-8 wetted area absent "
                    "offline) -> CT interval reported instead of a point value"
                ),
                "absolute_pct_estimate": 5.0,
            },
            "cross_checks": {
                "vs_published_ct_0.0040": next(
                    c.dev_pct for c in anchor_consistency() if c.label.endswith("AFF-8 CT")
                ),
                "vs_published_ct_0.0040_bracket": next(
                    c.dev_pct_bracket for c in anchor_consistency()
                    if c.label.endswith("AFF-8 CT")
                ),
                "vs_ittc57_x_form_factor_max_err_pct": max(
                    abs(r["err_pct"]) for r in ittc_ladder_reproduction(8)
                ),
                "appendage_force_ratio": appendage_force_ratio()[0][1],
            },
            "conditions": (
                "Re 1.16e7-3.55e7; no free surface; CT on wetted area with S_wet in bracket; "
                "the widely quoted 0.004 is NOT usable (see anchor_consistency)"
            ),
            "sources": [
                "Liu & Huang (1998) Table 14 (archived)",
                "Groves, Huang & Chang (1989) DTRC/SHD-1298-01 (geometry, cited not read)",
            ],
            "verdict": "admit_to_scorer_with_interval",
        },
    )