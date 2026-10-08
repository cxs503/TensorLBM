"""Unit-explicit force bookkeeping for icebreaker simulations.

This module is deliberately solver-agnostic. TensorLBM supplies the fluid load
on the hull; TensorDEM supplies the ice-contact load on the hull. The ledger
keeps these contributions separate to prevent accidental double counting.

All force components are in newtons, moments in newton-metres, time in seconds.
The longitudinal resistance is positive when the total force on the hull points
opposite to the configured forward direction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class WrenchSI:
    """A force/moment vector acting on the hull, expressed in SI units."""

    fx_n: float = 0.0
    fy_n: float = 0.0
    fz_n: float = 0.0
    mx_nm: float = 0.0
    my_nm: float = 0.0
    mz_nm: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

    def __add__(self, other: "WrenchSI") -> "WrenchSI":
        if not isinstance(other, WrenchSI):
            return NotImplemented
        return WrenchSI(
            fx_n=self.fx_n + other.fx_n,
            fy_n=self.fy_n + other.fy_n,
            fz_n=self.fz_n + other.fz_n,
            mx_nm=self.mx_nm + other.mx_nm,
            my_nm=self.my_nm + other.my_nm,
            mz_nm=self.mz_nm + other.mz_nm,
        )

    def scaled(self, factor: float) -> "WrenchSI":
        """Return the wrench multiplied by a finite scalar."""
        if not math.isfinite(factor):
            raise ValueError("factor must be finite")
        return WrenchSI(**{key: value * factor for key, value in asdict(self).items()})


@dataclass(frozen=True)
class IcebreakingForceLedger:
    """One timestamp of force decomposition for a rigid or prescribed hull.

    Every wrench is the load exerted on the hull (not the reaction exerted by
    the hull on the fluid/ice). forward_axis is a unit vector in the global
    coordinate frame pointing in the vessel's positive travel direction.
    """

    time_s: float
    fluid_on_hull: WrenchSI = WrenchSI()
    ice_contact_on_hull: WrenchSI = WrenchSI()
    other_on_hull: WrenchSI = WrenchSI()
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        if len(self.forward_axis) != 3 or not all(
            math.isfinite(component) for component in self.forward_axis
        ):
            raise ValueError("forward_axis must contain three finite components")
        norm = math.sqrt(sum(component * component for component in self.forward_axis))
        if not math.isclose(norm, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("forward_axis must be a unit vector")

    @property
    def total_on_hull(self) -> WrenchSI:
        """Sum of fluid, ice-contact, and explicitly modeled other loads."""
        return self.fluid_on_hull + self.ice_contact_on_hull + self.other_on_hull

    @property
    def resistance_n(self) -> float:
        """Signed longitudinal resistance; positive means opposing forward travel."""
        total = self.total_on_hull
        ax, ay, az = self.forward_axis
        return -(total.fx_n * ax + total.fy_n * ay + total.fz_n * az)

    def as_record(self) -> Mapping[str, float]:
        """Flat, CSV-friendly record with explicit units in field names."""
        record: dict[str, float] = {"time_s": self.time_s}
        for prefix, wrench in (
            ("fluid", self.fluid_on_hull),
            ("ice_contact", self.ice_contact_on_hull),
            ("other", self.other_on_hull),
            ("total", self.total_on_hull),
        ):
            for key, value in asdict(wrench).items():
                record[f"{prefix}_{key}"] = value
        record["resistance_n"] = self.resistance_n
        return record
