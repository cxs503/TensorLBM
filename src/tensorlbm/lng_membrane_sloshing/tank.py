"""174 000 m^3 LNG carrier: prismatic membrane cargo tank, fills and similarity.

Geometry (design point of a two-stroke 174k m^3 LNG carrier with four
NO96/Mark III prismatic membrane tanks; ``4 x 43 472 m^3 = 173 888 m^3`` gross
tank volume):

======================  =========  =============================================
tank length ``L``        44.0 m     sloshing direction (ship's transverse, ``x``)
tank breadth ``B``       38.0 m     longitudinal (implies ``4L = 176 m`` tank block)
tank height ``H``        26.0 m     inner bottom -> tank top
gross volume / tank      43 472 m^3 ``L*B*H`` (chamfered hopper/stool excluded)
number of tanks          4          ``173 888 m^3`` total
======================  =========  =============================================

The 2-D sloshing slice used by the campaign is ``L x H = 44 m x 26 m``; the
breadth enters only through the volume check.  Partial fills are specified as a
fraction of the *tank height*, ``h = fill * H``, which is the way filling
limits are quoted (``fill = 0.15 / 0.30 / 0.50`` are inside the unrestricted
range of the IGC/GCC sloshing assessment).

Liquid properties are those of LNG at its atmospheric boiling point.  As in
every LBM sloshing study, the lattice cannot resolve the molecular viscosity
(``Re ~ 1e8`` at full scale); the numerical Reynolds number reached at the
grids used here is reported together with every case and the resulting
over-damping is quoted as a limitation.

Similarity: the lattice pair ``(dx, dt)`` is chosen to (i) hold the sloshing
Froude number, i.e. ``g_lat = g dt^2/dx`` follows from the chosen ``dt``, and
(ii) respect the weak-compressibility budget of the free-surface model,
``3 g_lat h_lat <= 0.25`` (pressure variation small against ``c_s^2 rho``).
Refining ``dx`` at constant lattice impact velocity then refines ``dx`` and
``dt`` together, which is what the GCI study needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..sloshing_resonance.tank2d import LatticeScaling, SloshingTank

__all__ = [
    "LNG_PROPERTIES",
    "LNGProperties",
    "MembraneTank174k",
    "PanelModel",
    "DEFAULT_FILLS",
    "scaling_for_case",
    "similarity_report",
]

#: Partial fills (fraction of tank height) studied by the campaign.
DEFAULT_FILLS: tuple[float, ...] = (0.15, 0.30, 0.50)


@dataclass(frozen=True)
class LNGProperties:
    """LNG at 111 K / 1 bar (boiling point, saturated liquid)."""

    density: float = 470.0          # kg/m^3  (450-470 at 111 K)
    kinematic_viscosity: float = 3.2e-7  # m^2/s (mu ~ 1.5e-4 Pa s / 470)
    sound_speed: float = 1250.0     # m/s
    temperature: float = 111.15     # K
    gas_density: float = 1.8        # kg/m^3 (methane vapour, 111 K, 1 bar)

    def check(self) -> None:
        if self.density <= 0.0 or self.sound_speed <= 0.0:
            raise ValueError("density and sound speed must be positive")
        if self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic viscosity must be positive")


@dataclass(frozen=True)
class MembraneTank174k:
    """Prismatic membrane cargo tank of a 174 000 m^3 LNG carrier."""

    length: float = 44.0
    breadth: float = 38.0
    height: float = 26.0
    n_tanks: int = 4
    lng: LNGProperties = LNGProperties()

    @property
    def gross_volume(self) -> float:
        """Gross volume of one tank [m^3]."""
        return self.length * self.breadth * self.height

    @property
    def total_volume(self) -> float:
        """Gross volume of the whole cargo containment [m^3]."""
        return self.n_tanks * self.gross_volume

    def fill_height(self, fill: float) -> float:
        """Still-water depth ``h = fill * H`` [m]."""
        if not (0.0 < fill < 1.0):
            raise ValueError("fill must be in (0, 1)")
        return fill * self.height

    def liquid_volume(self, fill: float) -> float:
        """Liquid volume at a given fill fraction [m^3], one tank."""
        return self.length * self.breadth * self.fill_height(fill)

    def check(self, *, rel_tol: float = 0.01) -> dict[str, float]:
        """Geometry / capacity self-check (the ``Cb-L-B`` gate of this node).

        Verifies that (i) all dimensions are positive and ordered like a real
        membrane tank (``H < B <= L``), (ii) the quoted volume matches the
        product ``L B H`` to ``rel_tol``, and (iii) the sum over tanks matches
        the 174 000 m^3 ship capacity.  Returns the audit numbers.
        """
        problems: list[str] = []
        for name, value in (
            ("length", self.length),
            ("breadth", self.breadth),
            ("height", self.height),
        ):
            if value <= 0.0:
                problems.append(f"{name} must be positive")
        if not (self.height < self.breadth <= self.length):
            problems.append("expected H < B <= L for a prismatic membrane tank")
        volume = self.gross_volume
        quoted = 43_472.0
        if abs(volume - quoted) / quoted > rel_tol:
            problems.append(f"tank volume {volume:.0f} m^3 deviates from {quoted:.0f} m^3")
        total = self.total_volume
        if abs(total - 173_888.0) / 173_888.0 > rel_tol:
            problems.append(f"total capacity {total:.0f} m^3 is not 174 000 m^3 class")
        if problems:
            raise ValueError("; ".join(problems))
        return {
            "length": self.length,
            "breadth": self.breadth,
            "height": self.height,
            "gross_volume": volume,
            "total_volume": total,
            "slenderness_L_over_H": self.length / self.height,
        }

    @property
    def sloshing_tank(self) -> SloshingTank:
        """The 2-D ``L x H`` slice as a :class:`SloshingTank`."""
        return SloshingTank(
            length=self.length,
            height=self.height,
            g=9.81,
            rho_liquid=self.lng.density,
            rho_gas=self.lng.gas_density,
        )


@dataclass(frozen=True)
class PanelModel:
    """Linear static pressure -> stress transfer of the loaded tank wall.

    The sloshing pressure crosses the insulation and loads the inner-hull
    plating, which is idealised as a clamped square plate (Timoshenko &
    Woinowsky-Krieger, *Theory of Plates and Shells*, table 11; maximum bending
    stress coefficient ``0.5`` at ``nu = 0.3``):

        sigma = 0.5 * p * (span / thickness)^2

    Defaults: inner-hull plating ``t = 22 mm`` between frames ``span = 0.8 m``.
    This is a *declared* linear transfer function (no FE, no dynamics); the
    fatigue comparison of the campaign is insensitive to it because both
    estimators use the same stress history.
    """

    span: float = 0.8
    thickness: float = 0.022
    coefficient: float = 0.5
    sn_class: str = "IIW FAT80"

    def check(self) -> None:
        if self.span <= 0.0 or self.thickness <= 0.0:
            raise ValueError("span and thickness must be positive")

    def stress_mpa(self, pressure_pa):
        """Wall-panel bending stress [MPa] for a pressure [Pa] array."""
        import numpy as np

        self.check()
        return (
            self.coefficient
            * np.asarray(pressure_pa, dtype=float)
            * (self.span / self.thickness) ** 2
            / 1.0e6
        )


def scaling_for_case(
    tank: MembraneTank174k,
    *,
    fill: float,
    cells_per_height: int,
    amplitude: float,
    frequency: float,
    tau: float = 0.55,
    u_lat_target: float = 0.045,
    compressibility_budget: float = 0.25,
) -> LatticeScaling:
    """Pick ``(dx, dt)`` for one sloshing case (Froude + weak-compressibility).

    ``dt`` is the *smaller* of

    * ``u_lat_target * dx / u_char`` with ``u_char = amplitude * omega`` (the
      excitation velocity scale), which fixes the lattice Mach number, and
    * the value allowed by the hydrostatic budget
      ``3 g_lat h_lat <= compressibility_budget``.

    Using one or the other per case keeps every run inside both constraints;
    the refinement family then holds ``dx`` and ``dt`` in fixed ratio.
    """
    if cells_per_height < 8:
        raise ValueError("cells_per_height must be >= 8")
    dx = tank.height / float(cells_per_height)
    h = tank.fill_height(fill)
    h_cells = h / dx
    g_lat_max = compressibility_budget / (3.0 * h_cells)
    dt_budget = math.sqrt(g_lat_max * dx / 9.81)
    u_char = max(amplitude * 2.0 * math.pi * frequency, 1e-9)
    dt_mach = u_lat_target * dx / u_char
    dt = min(dt_budget, dt_mach)
    scaling = LatticeScaling(
        dx=dx,
        dt=dt,
        g_lattice=9.81 * dt * dt / dx,
        tau=tau,
        u_lat=u_char * dt / dx,
        u_char=u_char,
    )
    scaling.check(h_cells)
    return scaling


def similarity_report(
    tank: MembraneTank174k, scaling: LatticeScaling, *, fill: float
) -> dict[str, float]:
    """Dimensionless numbers of one lattice realisation (honesty metrics)."""
    h = tank.fill_height(fill)
    h_cells = h / scaling.dx
    nu_phys = scaling.nu_physical(tank.sloshing_tank)
    c_wave = math.sqrt(9.81 * h)
    return {
        "dx_m": scaling.dx,
        "dt_s": scaling.dt,
        "depth_cells": h_cells,
        "g_lattice": scaling.g_lattice,
        "u_lat": scaling.u_lat,
        "compressibility_ratio": 3.0 * scaling.g_lattice * h_cells,
        "nu_lattice": scaling.nu_lattice,
        "nu_physical_equivalent": nu_phys,
        "reynolds_cell": c_wave * h / nu_phys,
        "reynolds_lng": c_wave * h / tank.lng.kinematic_viscosity,
        "viscosity_ratio": nu_phys / tank.lng.kinematic_viscosity,
        "steps_per_period": (
            1.0
            / (
                math.sqrt(9.81 * math.pi / tank.length * math.tanh(math.pi * h / tank.length))
                / (2.0 * math.pi)
                * scaling.dt
            )
        ),
    }


#: Convenience module-level LNG property set (boiling point, 1 bar).
LNG_PROPERTIES = LNGProperties()
