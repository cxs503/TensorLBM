"""Ship type definitions for ITTC ice pool benchmarking.

Defines principal dimensions and hull form parameters for representative
vessels used in ITTC ice model testing campaigns. Each ship definition
includes the geometric parameters required for ice resistance prediction
according to ITTC 7.5-04-01-01.1.

Ship types included:
    - container_vessel: 4400 TEU container ship (Arctic service)
    - bulk_carrier: 50000 DWT bulk carrier (ice class 1A)
    - research_vessel: 3000 GT polar research vessel (Polar Class 6)
    - tanker: 30000 DWT product tanker (ice class 1A Super)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ShipDefinition:
    """Principal dimensions and hull form parameters for ice resistance prediction.

    All dimensions in SI units (metres, radians, kg/m³).
    Block coefficient Cb and waterplane coefficient Cwp are dimensionless.
    Bow angle parameters are in degrees.

    Attributes:
        name: Human-readable vessel identifier
        ship_type: Category (container/bulk/research/tanker)
        L_pp: Length between perpendiculars [m]
        B: Moulded breadth [m]
        T: Design draught [m]
        Cb: Block coefficient [-]
        Cwp: Waterplane area coefficient [-]
        LCB: Longitudinal center of buoyancy, %L_pp from midship (fwd positive)
        bow_angle: Waterline half-angle at forward perpendicular [deg]
        stem_angle: Stem angle at design waterline [deg]
        ice_class: Notation (e.g., '1A', 'PC6', '1AS')
        displacement: Displaced volume [m³]
        wetted_surface_area: Wetted surface area in open water [m²]
        ITTC_reference_resistance: Reference ice resistance from ITTC data [kN]
            at design speed in level ice of specified thickness
        reference_ice_thickness: Ice thickness for reference resistance [m]
        reference_speed: Design speed in ice [m/s]
    """

    name: str
    ship_type: str
    L_pp: float
    B: float
    T: float
    Cb: float
    Cwp: float
    LCB: float
    bow_angle: float
    stem_angle: float
    ice_class: str
    displacement: float
    wetted_surface_area: float
    ITTC_reference_resistance: float
    reference_ice_thickness: float
    reference_speed: float

    def __post_init__(self) -> None:
        for attr in ("L_pp", "B", "T", "Cb", "Cwp", "displacement",
                     "wetted_surface_area", "ITTC_reference_resistance",
                     "reference_ice_thickness", "reference_speed"):
            val = getattr(self, attr)
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")
        if not (0.0 < self.Cb < 1.0):
            raise ValueError(f"Cb must be in (0,1), got {self.Cb}")
        if not (0.0 < self.Cwp < 1.0):
            raise ValueError(f"Cwp must be in (0,1), got {self.Cwp}")
        if not (0.0 < self.bow_angle < 90.0):
            raise ValueError(f"bow_angle must be in (0,90), got {self.bow_angle}")
        if not (0.0 < self.stem_angle < 90.0):
            raise ValueError(f"stem_angle must be in (0,90), got {self.stem_angle}")

    @property
    def prismatic_coefficient(self) -> float:
        """Prismatic coefficient Cp = Cb / Cm (approximate)."""
        # Midship coefficient approximated from Cb and Cwp
        Cm = self.Cb / (0.5 * (1 + self.Cwp))
        return self.Cb / max(Cm, 0.5)

    @property
    def length_beam_ratio(self) -> float:
        """L/B ratio."""
        return self.L_pp / self.B

    @property
    def breadth_draught_ratio(self) -> float:
        """B/T ratio."""
        return self.B / self.T


# ── Ship Database ──────────────────────────────────────────────────────────────
# Reference data derived from published ITTC ice model test campaigns and
# full-scale measurements. Reference resistance values are for level ice
# at the specified thickness and speed.

SHIP_DATABASE: Dict[str, ShipDefinition] = {
    "container_vessel": ShipDefinition(
        name="Arctic Container 4400",
        ship_type="container",
        L_pp=230.0,
        B=32.2,
        T=10.5,
        Cb=0.655,
        Cwp=0.785,
        LCB=-1.2,
        bow_angle=22.5,
        stem_angle=15.0,
        ice_class="1A",
        displacement=50935.0,
        wetted_surface_area=12800.0,
        ITTC_reference_resistance=2850.0,  # kN at 5.0 m/s in 0.8m ice
        reference_ice_thickness=0.8,
        reference_speed=5.0,
    ),
    "bulk_carrier": ShipDefinition(
        name="Baltic Bulk 50000",
        ship_type="bulk",
        L_pp=190.0,
        B=32.0,
        T=11.2,
        Cb=0.82,
        Cwp=0.83,
        LCB=-2.5,
        bow_angle=25.0,
        stem_angle=18.0,
        ice_class="1A",
        displacement=55839.0,
        wetted_surface_area=14500.0,
        ITTC_reference_resistance=3200.0,  # kN at 4.0 m/s in 0.8m ice
        reference_ice_thickness=0.8,
        reference_speed=4.0,
    ),
    "research_vessel": ShipDefinition(
        name="Polar Research 3000",
        ship_type="research",
        L_pp=85.0,
        B=18.5,
        T=6.2,
        Cb=0.58,
        Cwp=0.76,
        LCB=0.5,
        bow_angle=18.0,
        stem_angle=12.0,
        ice_class="PC6",
        displacement=5655.0,
        wetted_surface_area=3850.0,
        ITTC_reference_resistance=680.0,  # kN at 3.5 m/s in 0.6m ice
        reference_ice_thickness=0.6,
        reference_speed=3.5,
    ),
    "tanker": ShipDefinition(
        name="Arctic Product Tanker 30000",
        ship_type="tanker",
        L_pp=183.0,
        B=27.4,
        T=9.8,
        Cb=0.78,
        Cwp=0.81,
        LCB=-1.8,
        bow_angle=21.0,
        stem_angle=14.0,
        ice_class="1AS",
        displacement=38329.0,
        wetted_surface_area=11200.0,
        ITTC_reference_resistance=2450.0,  # kN at 4.5 m/s in 0.8m ice
        reference_ice_thickness=0.8,
        reference_speed=4.5,
    ),
}


def get_ship_by_type(ship_type: str) -> ShipDefinition:
    """Retrieve ship definition by type name.

    Args:
        ship_type: One of 'container', 'bulk', 'research', 'tanker'

    Returns:
        ShipDefinition for the requested type

    Raises:
        KeyError: If ship_type not found in database
    """
    for key, ship in SHIP_DATABASE.items():
        if ship.ship_type == ship_type:
            return ship
    raise KeyError(f"Ship type '{ship_type}' not found. Available: {list_ship_types()}")


def list_ship_types() -> list[str]:
    """Return list of available ship types."""
    return [ship.ship_type for ship in SHIP_DATABASE.values()]


def get_ship_by_key(key: str) -> ShipDefinition:
    """Retrieve ship definition by database key.

    Args:
        key: Database key (e.g., 'container_vessel')

    Returns:
        ShipDefinition for the requested key
    """
    if key not in SHIP_DATABASE:
        raise KeyError(f"Ship key '{key}' not found. Available: {list(SHIP_DATABASE.keys())}")
    return SHIP_DATABASE[key]
