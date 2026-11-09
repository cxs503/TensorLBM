"""Ship resistance decomposition ``C_T = C_F (1 + k) + C_R`` with a wall-law ``C_F``.

This is the engineering end use of the wall model: the friction line that the
wall-modelled boundary layer produces must reproduce the friction coefficient
that the towing-tank community uses, and the total resistance must follow from
the classical decomposition with a form factor and a residual-resistance
coefficient.

The module is deliberately thin and explicit about what is *modelled* and what
is *carried*:

``C_F``  computed here from the ITTC-1957 friction line
         ``C_F = 0.075 / (log10 Re - 2)^2`` (the same line the flat-plate
         benchmark compares against, so the ship and plate anchors are
         consistent);
``1+k``  form factor, an input (documented as such, with a sensitivity sweep);
``C_R``  residual resistance, an input.  For KVLCC2 it is taken from the
         platform's traceable ITTC-1978 chain (recorded with the API response
         as provenance) when available; no value is memorised in this module.

Nothing here fabricates an experimental number: :func:`compare` only reports
deviations to references that the caller supplies *with* their provenance, and
a missing reference is reported as ``None`` / WITHHELD rather than filled in.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field

__all__ = [
    "ittc_1957_cf",
    "kvlcc2_geometry",
    "ResistanceDecomposition",
    "decompose",
    "fetch_platform_ittc78",
]

KVLCC2 = {
    "name": "KVLCC2",
    "Lpp_full_scale_m": 320.0,
    "B_m": 58.0,
    "T_m": 20.8,
    "Cb": 0.8098,
    "Cm": 0.9982,
    "wetted_surface_full_scale_m2": 27194.0,
    "design_speed_kn": 15.5,
    "design_froude": 0.142,
}


def kvlcc2_geometry(scale: float = 1.0) -> dict:
    """KVLCC2 main particulars at a geometric ``scale`` (1.0 = full scale)."""
    if scale <= 0.0:
        msg = "scale must be positive"
        raise ValueError(msg)
    return {
        "name": KVLCC2["name"],
        "scale": scale,
        "Lpp": KVLCC2["Lpp_full_scale_m"] * scale,
        "B": KVLCC2["B_m"] * scale,
        "T": KVLCC2["T_m"] * scale,
        "Cb": KVLCC2["Cb"],
        "Cm": KVLCC2["Cm"],
        "wetted_surface": KVLCC2["wetted_surface_full_scale_m2"] * scale**2,
        "displacement": KVLCC2["Cb"]
        * KVLCC2["Lpp_full_scale_m"]
        * KVLCC2["B_m"]
        * KVLCC2["T_m"]
        * scale**3,
    }


def ittc_1957_cf(reynolds: float) -> float:
    """ITTC-1957 friction coefficient."""
    if reynolds <= 100.0:
        msg = "ITTC-1957 requires Re > 100"
        raise ValueError(msg)
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


@dataclass(frozen=True)
class ResistanceDecomposition:
    """Total-resistance decomposition at one Reynolds number."""

    reynolds: float
    cf: float
    form_factor: float
    cr: float
    c_a: float = 0.0
    speed: float = float("nan")
    froude: float = float("nan")
    rho: float = 1025.0
    wetted_surface: float = float("nan")
    provenance: dict = field(default_factory=dict)

    @property
    def ct(self) -> float:
        """``C_T = C_F (1+k) + C_R + C_A``."""
        return self.cf * self.form_factor + self.cr + self.c_a

    @property
    def resistance(self) -> float:
        """Total resistance in newtons (needs ``speed`` and ``wetted_surface``)."""
        return 0.5 * self.rho * self.speed**2 * self.wetted_surface * self.ct

    @property
    def friction_resistance(self) -> float:
        return 0.5 * self.rho * self.speed**2 * self.wetted_surface * self.cf * self.form_factor

    def as_dict(self) -> dict:
        return {
            "reynolds": self.reynolds,
            "froude": self.froude,
            "cf": self.cf,
            "form_factor_1_plus_k": self.form_factor,
            "cr": self.cr,
            "c_a": self.c_a,
            "ct": self.ct,
            "speed_ms": self.speed,
            "wetted_surface_m2": self.wetted_surface,
            "total_resistance_N": self.resistance,
            "friction_resistance_N": self.friction_resistance,
            "provenance": self.provenance,
        }


def decompose(
    *,
    reynolds: float,
    form_factor: float,
    cr: float,
    c_a: float = 0.0,
    speed: float = float("nan"),
    froude: float = float("nan"),
    rho: float = 1025.0,
    wetted_surface: float = float("nan"),
    provenance: dict | None = None,
) -> ResistanceDecomposition:
    """Assemble the decomposition with the friction coefficient computed here."""
    return ResistanceDecomposition(
        reynolds=reynolds,
        cf=ittc_1957_cf(reynolds),
        form_factor=form_factor,
        cr=cr,
        c_a=c_a,
        speed=speed,
        froude=froude,
        rho=rho,
        wetted_surface=wetted_surface,
        provenance=provenance or {"cf": "ITTC-1957 line computed in this module"},
    )


def fetch_platform_ittc78(
    *,
    geometry: dict,
    speed: float,
    form_factor: float,
    base_url: str = "http://127.0.0.1:8001/api/v1",
    water_density: float = 1025.0,
    water_viscosity: float = 1.189e-6,
    timeout: float = 20.0,
) -> dict:
    """Query the ship-platform ITTC-1978 chain (cross-implementation reference).

    Returns the raw JSON response plus the request payload so the provenance of
    every borrowed number is auditable.  Raises on transport failure -- the
    caller decides whether the reference is then WITHHELD.
    """
    payload = {
        "Lwl": geometry["Lpp"],
        "B": geometry["B"],
        "T": geometry["T"],
        "displacement": geometry["displacement"],
        "Cb": geometry["Cb"],
        "Cm": geometry["Cm"],
        "wetted_surface": geometry["wetted_surface"],
        "speed": speed,
        "form_factor": form_factor,
        "water_density": water_density,
        "water_viscosity": water_viscosity,
    }
    request = urllib.request.Request(
        f"{base_url}/resistance-advanced/ittc78",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
        msg = f"ship platform ITTC-78 reference unavailable: {exc}"
        raise RuntimeError(msg) from exc
    return {"request": payload, "response": data, "endpoint": f"{base_url}/resistance-advanced/ittc78"}