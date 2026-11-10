"""External validation anchors for channel refreezing (with honest provenance).

Every anchor carries

* ``provenance``   -- literature citation string,
* ``confidence``   -- ``"recalled"`` when the digit string was entered from
                      literature memory without a machine-readable source in
                      this sandbox (the same convention as
                      ``references/air_lubrication_reference_points.json``),
* ``sha256``       -- digest of the numeric payload so any later correction is
                      detectable,
* ``verification_gap`` -- what must be re-checked against the original paper.

Only the *Stefan/Neumann* anchor is exact (it is mathematics, not literature).
The Weeks & Ackley, Odden and Lewis anchors are recalled values and the node
report states explicitly that they must be re-verified before the corresponding
criterion is treated as externally certified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Anchor:
    key: str
    quantity: str
    value: float
    units: str
    band: tuple[float, float] | None
    provenance: str
    confidence: str
    verification_gap: str

    def digest(self) -> str:
        payload = json.dumps(
            {
                "key": self.key,
                "quantity": self.quantity,
                "value": self.value,
                "units": self.units,
                "band": list(self.band) if self.band else None,
                "provenance": self.provenance,
                "confidence": self.confidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def deviation(self, model_value: float) -> float:
        """Relative deviation |model - anchor| / |anchor|."""
        return abs(float(model_value) - self.value) / abs(self.value)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["band"] = list(self.band) if self.band else None
        d["sha256"] = self.digest()
        return d


def _anchor(**kw) -> Anchor:
    return Anchor(**kw)


# --- Stefan / Neumann (exact; provided for bookkeeping) --------------------
STEFAN_NEUMANN = _anchor(
    key="stefan_neumann_1889",
    quantity="one_phase_planar_freezing_front",
    value=float("nan"),  # the *law*, not a single number; deviation vs the solver
    units="m",
    band=None,
    provenance=(
        "Neumann (1860) / Stefan (1891) similarity solution; see Carslaw & Jaeger "
        "(1959) ch. XI and P155W1-N01 report for the numerical verification in "
        "this repository (benchmarks/verified/stefan_phase_change)."
    ),
    confidence="exact",
    verification_gap="none: exact solution, re-derived in stefan.neumann_lambda",
)

# --- Weeks & Ackley (1986) -------------------------------------------------
WEEKS_ACKLEY_GROWTH_CONSTANT = _anchor(
    key="weeks_ackley_1986_lead_growth_constant",
    quantity="effective_degree_day_growth_constant_of_wind_mixed_lead_ice",
    value=3.0,          # cm / (degC^0.5 day^0.5)
    units="cm/(degC^0.5 day^0.5)",
    band=(2.5, 3.5),
    provenance=(
        "Weeks & Ackley (1986), 'The growth, structure and properties of sea ice', "
        "CRREL Monograph 82-1: effective growth constant for young ice in "
        "wind-mixed leads, C = h/sqrt(FDD) ~ 3 cm/degC^0.5/day^0.5 (quoted band "
        "2.5-3.5); pure-conduction lake/sea ice typically 3.5."
    ),
    confidence="recalled",
    verification_gap=(
        "digit string entered from literature memory (no machine-readable source "
        "in the sandbox); re-check the original figure/table before treating "
        "criterion (b) as externally certified."
    ),
)

WEEKS_ACKLEY_LEAD_OCEAN_FLUX = _anchor(
    key="weeks_ackley_1986_lead_ocean_heat_flux",
    quantity="wind_driven_oceanic_heat_flux_to_lead_ice",
    value=25.0,         # W/m^2 at U_10 = 5 m/s
    units="W/m^2",
    band=(15.0, 40.0),
    provenance=(
        "Weeks & Ackley (1986) discuss wind-mixed lead oceanic heat fluxes in the "
        "tens of W/m^2 (order 20-30 W/m^2), the mechanism that keeps young leads "
        "thin relative to pure conductive growth."
    ),
    confidence="recalled",
    verification_gap="same as above (recalled); band is deliberately wide.",
)

# --- Odden (1998) full-scale -------------------------------------------------
ODDEN_1998_THICKNESS = _anchor(
    key="odden_1998_full_scale_thickness",
    quantity="ice_thickness_after_90_days_of_continuous_freezing",
    value=0.62,         # m
    units="m",
    band=(0.50, 0.75),
    provenance=(
        "Odden (1998), full-scale ice-growth monitoring in a Norwegian fjord "
        "(recalled): seasonal first-year ice thickness of order 0.5-0.7 m after "
        "~90 freezing days under mean air temperatures of about -6 degC."
    ),
    confidence="recalled",
    verification_gap=(
        "the original time series is not available offline; the comparison uses a "
        "90-day integral value plus a +/-0.12 m band, not point-by-point data."
    ),
)

# --- Lewis (1993) navigation window ----------------------------------------
LEWIS_1993_WINDOW = _anchor(
    key="lewis_1993_navigation_window",
    quantity="time_to_reach_0.30_m_ice_in_a_refreezing_channel",
    value=8.0,          # days
    units="day",
    band=(6.0, 10.0),
    provenance=(
        "Lewis (1993), icebreaker escort / navigation-window guidance (recalled): "
        "a refreezing channel under -20 degC air and ~5 m/s wind becomes "
        "navigation-limiting (around 0.3 m of ice) in about 8 days."
    ),
    confidence="recalled",
    verification_gap=(
        "recalled operational figure; criterion (d) therefore checks the model "
        "against a recalled band and against the model's own ODE/LBM cross-check."
    ),
)

ANCHORS = {
    a.key: a
    for a in (
        STEFAN_NEUMANN,
        WEEKS_ACKLEY_GROWTH_CONSTANT,
        WEEKS_ACKLEY_LEAD_OCEAN_FLUX,
        ODDEN_1998_THICKNESS,
        LEWIS_1993_WINDOW,
    )
}


def anchor_table() -> list[dict]:
    """Serialisable anchor list (includes sha256 of each numeric payload)."""
    return [a.as_dict() for a in ANCHORS.values()]


__all__ = [
    "Anchor",
    "STEFAN_NEUMANN",
    "WEEKS_ACKLEY_GROWTH_CONSTANT",
    "WEEKS_ACKLEY_LEAD_OCEAN_FLUX",
    "ODDEN_1998_THICKNESS",
    "LEWIS_1993_WINDOW",
    "ANCHORS",
    "anchor_table",
]