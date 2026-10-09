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
from typing import Mapping, Sequence


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



def integrate_surface_traction(
    positions_m: "Sequence[Sequence[float]]",
    tractions_pa: "Sequence[Sequence[float]]",
    area_weights_m2: "Sequence[float]",
    origin_m: "Sequence[float]" = (0.0, 0.0, 0.0),
) -> WrenchSI:
    """Integrate sampled surface traction into a hull wrench in SI units.

    positions_m are global sample coordinates, tractions_pa are the traction
    vectors acting on the hull in pascals, and area_weights_m2 are the matching
    surface quadrature weights. The returned moment is about origin_m. This
    helper does not compute wall stress or surface quadrature weights; callers
    must provide samples from their chosen boundary-force method.
    """
    if not (len(positions_m) == len(tractions_pa) == len(area_weights_m2)):
        raise ValueError("positions, tractions, and area weights must have equal lengths")
    if len(origin_m) != 3 or not all(math.isfinite(float(v)) for v in origin_m):
        raise ValueError("origin_m must contain three finite coordinates")
    total_force = [0.0, 0.0, 0.0]
    total_moment = [0.0, 0.0, 0.0]
    for index, (position, traction, area) in enumerate(
        zip(positions_m, tractions_pa, area_weights_m2)
    ):
        if len(position) != 3 or len(traction) != 3:
            raise ValueError(f"sample {index} position and traction must be 3-vectors")
        values = [float(v) for v in (*position, *traction, area)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"sample {index} contains non-finite values")
        if area < 0.0:
            raise ValueError(f"sample {index} area weight must be non-negative")
        force = [float(traction[k]) * float(area) for k in range(3)]
        arm = [float(position[k]) - float(origin_m[k]) for k in range(3)]
        moment = [
            arm[1] * force[2] - arm[2] * force[1],
            arm[2] * force[0] - arm[0] * force[2],
            arm[0] * force[1] - arm[1] * force[0],
        ]
        for axis in range(3):
            total_force[axis] += force[axis]
            total_moment[axis] += moment[axis]
    return WrenchSI(
        fx_n=total_force[0], fy_n=total_force[1], fz_n=total_force[2],
        mx_nm=total_moment[0], my_nm=total_moment[1], mz_nm=total_moment[2],
    )
