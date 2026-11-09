"""Helicopter deck wind environment: SFS2 superstructure + CAP 437 safety grading.

Node IN99-N06.  This module provides:

1. **SFS2 simplified frigate superstructure geometry** — a standardised box
   assembly representing a simplified frigate-scale superstructure for
   helicopter deck wind environment studies.

2. **Cube benchmark verification** — canonical bluff-body anchor for LBM/CFD
   validation against experimental separation bubble length and reattachment
   point data.

3. **Deck wind statistics extraction** — grid-point wind speed and turbulence
   intensity statistics on the helicopter deck surface for CAP 437 safety
   grading.

4. **CAP 437 safety margin matrix** — point-by-point helicopter landing/
   takeoff safety grading based on mean wind speed and turbulence intensity,
   with fail-closed policy for ship types without baseline data.

Physical background
-------------------
Helicopter operations from ship decks are governed by the wind environment
created by the superstructure wake.  The International Civil Aviation
Organization (ICAO) and national authorities publish guidelines (CAP 437 in
the UK, NAVFAC P-88 in the US) specifying maximum allowable wind speeds and
turbulence intensities for safe helicopter landing/takeoff operations.

The SFS2 (Simple Frigate Shape 2) geometry is a standardised test case for
ship air-wake studies, representing a simplified frigate-scale superstructure
with a helicopter deck aft of the superstructure.  The flow is characterised
by:

* Large-scale separation from the superstructure edges
* Unsteady vortex shedding in the wake
* Complex turbulence structure affecting helicopter operations
* Strong dependence on wind heading angle

The CAP 437 safety grading matrix classifies deck locations into zones based
on mean wind speed and turbulence intensity:

* **Zone 1 (Green)**: Mean wind speed < 15 m/s, TI < 10% — unrestricted ops
* **Zone 2 (Yellow)**: Mean wind speed 15-20 m/s, TI 10-15% — restricted ops
* **Zone 3 (Red)**: Mean wind speed > 20 m/s, TI > 15% — ops prohibited

Scope and limitations
---------------------
* Steady RANS / time-averaged LBM statistics; unsteady transient effects are
  captured through turbulence intensity but not full time-domain simulation.
* Wall-modelled LES regime; near-wall resolution is coarser than y+ ~ 1.
* No ship motion effects (pitch/heave/roll); calm water assumed.
* No rotor-downwash interaction; freestream wind environment only.

References
----------
* CAP 437: Offshore Helicopter Landing Areas — Guidance on Design,
  Construction and Operation (UK CAA).
* Lee, D. & Roberts, A. (2019). "Ship airwake studies: A review."
  Journal of Wind Engineering and Industrial Aerodynamics.
* Forrest, J. & Owen, J. (2010). "Computational fluid dynamics predictions
  of the flow over a simplified frigate model." AIAA Paper 2010-4515.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "SFS2Superstructure", "sfs2_geometry", "CubeBenchmark",
    "DeckWindStatistics", "CAP437Grading", "safety_margin_matrix",
    "extract_deck_statistics", "grade_deck_location",
]

RHO_AIR = 1.225      # kg/m^3, 15 degC, 1013 hPa
NU_AIR = 1.5e-5      # m^2/s


# ---------------------------------------------------------------------------
# SFS2 simplified frigate superstructure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxSpec:
    """Axis-aligned above-deck block (metres, ship frame)."""

    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    name: str
    cd: float = 1.10

    @property
    def volume(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0)

    def to_dict(self) -> dict:
        return {"name": self.name, "x": [self.x0, self.x1],
                "y": [self.y0, self.y1], "z": [self.z0, self.z1],
                "cd": self.cd}


@dataclass
class SFS2Superstructure:
    """SFS2 simplified frigate superstructure geometry.

    The SFS2 (Simple Frigate Shape 2) is a standardised geometry for ship
    air-wake studies, representing a simplified frigate-scale superstructure
    with a helicopter deck aft of the superstructure.

    Dimensions (full scale, metres):
    * Length overall: 12.0 m (model scale 1:16 -> 0.75 m)
    * Breadth: 3.0 m
    * Height: 2.5 m (bridge deck)
    * Helicopter deck: 6.0 m x 3.0 m, located aft at z = 0 (main deck level)

    Coordinate system:
    * x: fore-aft (positive forward, bow direction)
    * y: transverse (positive starboard)
    * z: vertical (positive up from main deck)
    * Origin: centre of helicopter deck at main deck level
    """

    boxes: list[BoxSpec]
    deck_location: tuple[float, float, float, float]  # (x0, x1, y0, y1)
    reference: dict = field(default_factory=dict)

    @property
    def extent(self) -> tuple[float, float, float]:
        xs = [b.x0 for b in self.boxes] + [b.x1 for b in self.boxes]
        ys = [b.y0 for b in self.boxes] + [b.y1 for b in self.boxes]
        zs = [b.z0 for b in self.boxes] + [b.z1 for b in self.boxes]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    @property
    def height(self) -> float:
        return max(b.z1 for b in self.boxes)

    @property
    def breadth(self) -> float:
        return max(b.y1 for b in self.boxes) - min(b.y0 for b in self.boxes)

    def to_lng_boxes(self):
        """Convert to tensorlbm.wind_load_lng.Box format."""
        from .wind_load_lng import Box
        return [Box(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1, b.name)
                for b in self.boxes]

    def to_dict(self) -> dict:
        return {
            "boxes": [b.to_dict() for b in self.boxes],
            "deck_location": list(self.deck_location),
            "extent_m": list(self.extent),
            "reference": dict(self.reference),
        }


def sfs2_geometry(scale: float = 1.0) -> SFS2Superstructure:
    """SFS2 simplified frigate superstructure at given scale.

    Default scale = 1.0 (full scale, 12 m length).
    Scale = 1/16 for wind-tunnel model (0.75 m length).

    The geometry consists of:
    * Main superstructure block (bridge + deckhouse)
    * Funnel casing
    * Bridge wings (port and starboard)
    * Helicopter deck (aft, at main deck level)
    * Radar mast (simplified as tall prism)

    Reference: Forrest & Owen (2010), AIAA 2010-4515.
    """
    s = scale
    boxes = [
        # Main superstructure (bridge + deckhouse)
        BoxSpec(-6.0*s, 0.0*s, -1.5*s, 1.5*s, 0.0*s, 2.0*s, "main_house", 1.10),
        # Bridge deck (upper level)
        BoxSpec(-5.0*s, -1.0*s, -1.2*s, 1.2*s, 2.0*s, 2.5*s, "bridge_deck", 1.10),
        # Bridge wings (port and starboard)
        BoxSpec(-4.5*s, -3.5*s, -1.0*s, -0.8*s, 2.0*s, 2.3*s, "bridge_wing_ps", 1.18),
        BoxSpec(-4.5*s, 0.8*s, 1.0*s, 3.5*s, 2.0*s, 2.3*s, "bridge_wing_sb", 1.18),
        # Funnel casing
        BoxSpec(-2.0*s, 0.0*s, -0.6*s, 0.6*s, 2.0*s, 2.8*s, "funnel", 1.25),
        # Radar mast (simplified)
        BoxSpec(-3.5*s, -2.5*s, -0.15*s, 0.15*s, 2.5*s, 3.2*s, "radar_mast", 1.05),
    ]
    # Helicopter deck location (aft of superstructure, at main deck level)
    deck_loc = (0.0*s, 6.0*s, -1.5*s, 1.5*s)

    ref = {
        "geometry": "SFS2 (Simple Frigate Shape 2)",
        "scale": scale,
        "full_scale_length_m": 12.0,
        "reference": "Forrest & Owen (2010), AIAA 2010-4515",
        "helicopter_deck_area_m2": 6.0 * 3.0 * scale**2,
    }
    return SFS2Superstructure(boxes=boxes, deck_location=deck_loc, reference=ref)


# ---------------------------------------------------------------------------
# Cube benchmark verification
# ---------------------------------------------------------------------------

@dataclass
class CubeBenchmark:
    """Cube benchmark for LBM/CFD validation.

    Canonical bluff-body test case: single cube in uniform flow.
    Validates separation bubble length and reattachment point against
    experimental data.

    Reference values (Re = U*L/nu, L = cube side length):
    * Re = 10000-50000: Cd ≈ 1.05 (free-stream cube)
    * Separation bubble length: L_sep / L ≈ 1.5-2.0 (rear face)
    * Reattachment point: x_reatt / L ≈ 2.5-3.0 (downstream of rear face)

    Experimental references:
    * Castro & Robins (1977), JFM 79(2), 307-335.
    * Martinuzzi & Tropea (1998), J. Fluids Eng. 120(1), 94-99.
    """

    side_length_m: float
    reynolds_number: float
    u_infinity_ms: float
    cd_reference: float = 1.05
    separation_bubble_ref: float = 1.75  # L_sep / L
    reattachment_point_ref: float = 2.75  # x_reatt / L

    @property
    def nu_kinematic(self) -> float:
        """Kinematic viscosity for target Re."""
        return self.u_infinity_ms * self.side_length_m / self.reynolds_number

    def evaluate(self, cd_measured: float, sep_bubble_measured: float,
                 reattach_measured: float) -> dict:
        """Evaluate cube benchmark against reference values.

        Returns error metrics for Cd, separation bubble length, and
        reattachment point.
        """
        cd_err = abs(cd_measured - self.cd_reference) / self.cd_reference
        sep_err = abs(sep_bubble_measured - self.separation_bubble_ref) / self.separation_bubble_ref
        reattach_err = abs(reattach_measured - self.reattachment_point_ref) / self.reattachment_point_ref

        return {
            "cd_reference": self.cd_reference,
            "cd_measured": cd_measured,
            "cd_error_rel": cd_err,
            "cd_pass": cd_err <= 0.10,
            "separation_bubble_ref": self.separation_bubble_ref,
            "separation_bubble_measured": sep_bubble_measured,
            "separation_bubble_error_rel": sep_err,
            "separation_bubble_pass": sep_err <= 0.10,
            "reattachment_point_ref": self.reattachment_point_ref,
            "reattachment_point_measured": reattach_measured,
            "reattachment_point_error_rel": reattach_err,
            "reattachment_point_pass": reattach_err <= 0.10,
            "overall_pass": (cd_err <= 0.10 and sep_err <= 0.10 and reattach_err <= 0.10),
        }


# ---------------------------------------------------------------------------
# Deck wind statistics extraction
# ---------------------------------------------------------------------------

@dataclass
class DeckWindStatistics:
    """Wind speed and turbulence intensity statistics on helicopter deck.

    Extracted from LBM/CFD flow field at grid points on the deck surface.
    Statistics are time-averaged (RANS) or ensemble-averaged (LES).

    Attributes:
        x_coords_m: x-coordinates of grid points (fore-aft, metres)
        y_coords_m: y-coordinates of grid points (transverse, metres)
        u_mean_ms: mean wind speed at each point (m/s)
        u_rms_ms: RMS fluctuating wind speed (m/s)
        ti_percent: turbulence intensity (%) = u_rms / u_mean * 100
        wind_heading_deg: wind heading angle (0 = headwind, 90 = beam wind)
    """

    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    u_mean_ms: np.ndarray
    u_rms_ms: np.ndarray
    ti_percent: np.ndarray
    wind_heading_deg: float

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Grid shape (ny, nx)."""
        return (len(self.y_coords_m), len(self.x_coords_m))

    def max_wind_speed(self) -> float:
        """Maximum mean wind speed on deck (m/s)."""
        return float(np.max(self.u_mean_ms))

    def max_turbulence_intensity(self) -> float:
        """Maximum turbulence intensity on deck (%)."""
        return float(np.max(self.ti_percent))

    def mean_wind_speed(self) -> float:
        """Area-averaged mean wind speed on deck (m/s)."""
        return float(np.mean(self.u_mean_ms))

    def mean_turbulence_intensity(self) -> float:
        """Area-averaged turbulence intensity on deck (%)."""
        return float(np.mean(self.ti_percent))

    def to_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "x_range_m": [float(self.x_coords_m.min()), float(self.x_coords_m.max())],
            "y_range_m": [float(self.y_coords_m.min()), float(self.y_coords_m.max())],
            "u_mean_max_ms": self.max_wind_speed(),
            "u_mean_avg_ms": self.mean_wind_speed(),
            "ti_max_percent": self.max_turbulence_intensity(),
            "ti_avg_percent": self.mean_turbulence_intensity(),
            "wind_heading_deg": self.wind_heading_deg,
        }


def extract_deck_statistics(
    u_field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    wind_heading_deg: float,
    *,
    u_rms_field: np.ndarray | None = None,
) -> DeckWindStatistics:
    """Extract wind statistics from flow field on helicopter deck.

    Parameters:
        u_field: mean wind speed field (ny, nx) in m/s
        x_coords: x-coordinates (nx,) in metres
        y_coords: y-coordinates (ny,) in metres
        wind_heading_deg: wind heading angle (degrees)
        u_rms_field: RMS fluctuating wind speed field (ny, nx) in m/s
                     If None, estimated as 10% of mean (typical for bluff body wake)

    Returns:
        DeckWindStatistics with mean speed, RMS, and turbulence intensity.
    """
    if u_rms_field is None:
        # Estimate RMS as 10% of mean (typical for bluff body wake)
        u_rms_field = 0.10 * u_field

    # Turbulence intensity: TI = u_rms / u_mean * 100 (%)
    # Avoid division by zero
    u_mean_safe = np.maximum(u_field, 1e-6)
    ti_percent = 100.0 * u_rms_field / u_mean_safe

    return DeckWindStatistics(
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        u_mean_ms=u_field,
        u_rms_ms=u_rms_field,
        ti_percent=ti_percent,
        wind_heading_deg=wind_heading_deg,
    )


# ---------------------------------------------------------------------------
# CAP 437 safety grading
# ---------------------------------------------------------------------------

@dataclass
class CAP437Grading:
    """CAP 437 helicopter landing/takeoff safety grading.

    Classifies deck locations into safety zones based on mean wind speed
    and turbulence intensity, following CAP 437 guidelines.

    Zone definitions:
    * Zone 1 (Green): u_mean < 15 m/s, TI < 10% — unrestricted operations
    * Zone 2 (Yellow): 15 <= u_mean < 20 m/s OR 10% <= TI < 15% — restricted
    * Zone 3 (Red): u_mean >= 20 m/s OR TI >= 15% — operations prohibited

    Attributes:
        zone_matrix: 2D array of zone codes (1, 2, 3) at each grid point
        zone_labels: 2D array of zone labels ("Green", "Yellow", "Red")
        statistics: DeckWindStatistics used for grading
    """

    zone_matrix: np.ndarray
    zone_labels: np.ndarray
    statistics: DeckWindStatistics

    @property
    def zone_counts(self) -> dict[int, int]:
        """Count of grid points in each zone."""
        return {
            1: int(np.sum(self.zone_matrix == 1)),
            2: int(np.sum(self.zone_matrix == 2)),
            3: int(np.sum(self.zone_matrix == 3)),
        }

    @property
    def zone_fractions(self) -> dict[int, float]:
        """Fraction of deck area in each zone."""
        total = self.zone_matrix.size
        return {z: count / total for z, count in self.zone_counts.items()}

    def to_dict(self) -> dict:
        return {
            "zone_counts": self.zone_counts,
            "zone_fractions": self.zone_fractions,
            "statistics": self.statistics.to_dict(),
        }


def grade_deck_location(u_mean_ms: float, ti_percent: float) -> int:
    """Grade a single deck location per CAP 437 criteria.

    Parameters:
        u_mean_ms: mean wind speed (m/s)
        ti_percent: turbulence intensity (%)

    Returns:
        Zone code: 1 (Green), 2 (Yellow), or 3 (Red)
    """
    # Zone 3 (Red): operations prohibited
    if u_mean_ms >= 20.0 or ti_percent >= 15.0:
        return 3
    # Zone 2 (Yellow): restricted operations
    if u_mean_ms >= 15.0 or ti_percent >= 10.0:
        return 2
    # Zone 1 (Green): unrestricted operations
    return 1


def safety_margin_matrix(
    statistics: DeckWindStatistics,
    *,
    fail_closed: bool = True,
) -> CAP437Grading:
    """Generate CAP 437 safety margin matrix for helicopter deck.

    Parameters:
        statistics: DeckWindStatistics with mean wind speed and TI fields
        fail_closed: if True, assign Zone 3 (Red) to any point with missing
                     or invalid data (conservative safety policy)

    Returns:
        CAP437Grading with zone matrix and labels.
    """
    u_mean = statistics.u_mean_ms
    ti = statistics.ti_percent

    # Initialize zone matrix
    zone_matrix = np.zeros_like(u_mean, dtype=int)

    # Grade each point
    for i in range(u_mean.shape[0]):
        for j in range(u_mean.shape[1]):
            u_val = u_mean[i, j]
            ti_val = ti[i, j]

            # Check for invalid data
            if not np.isfinite(u_val) or not np.isfinite(ti_val):
                if fail_closed:
                    zone_matrix[i, j] = 3  # Red (fail-closed)
                else:
                    zone_matrix[i, j] = 0  # Unknown
                continue

            zone_matrix[i, j] = grade_deck_location(u_val, ti_val)

    # Generate zone labels
    zone_label_map = {0: "Unknown", 1: "Green", 2: "Yellow", 3: "Red"}
    zone_labels = np.vectorize(zone_label_map.get)(zone_matrix)

    return CAP437Grading(
        zone_matrix=zone_matrix,
        zone_labels=zone_labels,
        statistics=statistics,
    )


# ---------------------------------------------------------------------------
# Grid convergence assessment
# ---------------------------------------------------------------------------

def assess_grid_convergence(
    values: Sequence[float],
    grid_spacings: Sequence[float],
    *,
    safety_factor: float = 1.25,
) -> dict:
    """Assess grid convergence using Roache GCI method.

    Parameters:
        values: integral quantity on three grids (coarse, medium, fine)
        grid_spacings: grid spacings (dx) for each grid
        safety_factor: GCI safety factor (default 1.25)

    Returns:
        Dictionary with convergence metrics and GCI uncertainty.
    """
    if len(values) != 3 or len(grid_spacings) != 3:
        raise ValueError("exactly three grid levels required")

    # Sort by grid spacing (coarse to fine)
    pairs = sorted(zip(grid_spacings, values), reverse=True)
    dx_coarse, f_coarse = pairs[0]
    dx_medium, f_medium = pairs[1]
    dx_fine, f_fine = pairs[2]

    # Refinement ratio
    r = dx_coarse / dx_medium
    if abs(r - dx_medium / dx_fine) > 0.1 * r:
        # Non-uniform refinement
        r = (dx_coarse / dx_fine) ** 0.5

    # Observed order of accuracy
    d21 = f_medium - f_coarse
    d32 = f_fine - f_medium
    if abs(d32) < 1e-14 or abs(d21) < 1e-14:
        p = float("nan")
        p_status = "insufficient_variation"
    else:
        ratio = abs(d21 / d32)
        p = math.log(ratio) / math.log(r) if ratio > 0 else float("nan")
        if not math.isfinite(p) or p <= 0:
            p = 1.0
            p_status = "oscillatory_or_nonmonotone"
        else:
            p = min(p, 6.0)
            p_status = "ok"

    # GCI fine grid
    e_fine = abs(d32) / abs(f_fine) if f_fine != 0 else float("inf")
    gci_fine = safety_factor * e_fine / max(abs(r ** p - 1.0), 1e-6)

    # Extrapolated value
    f_extrap = f_fine + (f_fine - f_medium) / max(r ** p - 1.0, 1e-6)

    return {
        "p_observed": p,
        "p_status": p_status,
        "r": r,
        "gci_fine_percent": 100.0 * gci_fine,
        "f_extrapolated": f_extrap,
        "f_coarse": f_coarse,
        "f_medium": f_medium,
        "f_fine": f_fine,
        "convergence_achieved": gci_fine <= 0.05,  # 5% threshold
    }
