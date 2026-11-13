"""JSON case schema for LBM simulations, inspired by 2b-t/LB-t settings.json.

The 2b-t/LB-t project defines simulation cases via a JSON schema with sections:
  - ``physics``: density, kinematic viscosity
  - ``discretisation``: NX, NY, NZ
  - ``boundaryConditions``: list of boundary condition objects
  - ``initialConditions``: initial velocity field
  - ``geometry``: bounding box, models
  - ``lbmSettings``: D, Q, collision operator
  - ``output``: data format, write interval
  - ``times``: start, end, warm-up

This module provides a Python equivalent that:
1. Parses a JSON case definition into a ``CaseSchema`` dataclass
2. Bootstraps a ``LatticeBridge`` from the schema (using the same
   operating-point logic as ``bootstrap_bridge``)
3. Validates the schema against physical constraints (Ma < 0.1, τ ∈ [0.51, 2.0])

Reference: 2b-t/LB-t ``examples/test_1/settings.json``
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from tensorlbm.units.bridge import LatticeBridge, bootstrap_bridge


@dataclass
class Physics:
    """Physical properties of the fluid (SI units)."""

    density: float  # [kg/m³]
    kinematic_viscosity: float  # [m²/s]


@dataclass
class Discretisation:
    """Grid resolution."""

    NX: int
    NY: int = 1
    NZ: int = 1


@dataclass
class BoundaryCondition:
    """A single boundary condition entry."""

    boundary_type: str  # "slip", "noSlip", "velocity", "pressure", "periodic"
    reference_plane: str = ""  # "left", "right", "top", "bottom", "front", "back"
    velocity: dict[str, float] | None = None  # {"x": ..., "y": ..., "z": ...}
    density: float | None = None


@dataclass
class InitialConditions:
    """Initial velocity field."""

    velocity: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})


@dataclass
class LbmSettings:
    """Lattice Boltzmann method settings."""

    D: int = 2  # spatial dimensions
    Q: int = 9  # velocity count
    collision_operator: str = "bgk"
    floating_point_size: int = 8  # bytes


@dataclass
class OutputSettings:
    """Output configuration."""

    data_format: str = "vtk"
    output_folder: str = "output"
    write_interval: int = 1
    first_output: int = 0


@dataclass
class Times:
    """Simulation time control."""

    start_time: float = 0.0
    end_time: float = 100.0
    warm_up: float = 0.0


@dataclass
class CaseSchema:
    """Complete case definition schema for an LBM simulation.

    Mirrors the 2b-t/LB-t ``settings.json`` structure.  Can be loaded from
    JSON, validated, and used to bootstrap a ``LatticeBridge``.

    Attributes
    ----------
    physics : Physics
        Fluid physical properties.
    discretisation : Discretisation
        Grid resolution.
    boundary_conditions : list[BoundaryCondition]
        Boundary condition list.
    initial_conditions : InitialConditions
        Initial velocity field.
    lbm_settings : LbmSettings
        LBM solver settings.
    output : OutputSettings
        Output configuration.
    times : Times
        Time control.
    characteristic_length : float
        Characteristic length [m] for unit conversion.
    characteristic_velocity : float
        Characteristic velocity [m/s] for unit conversion.
    u_lb : float
        Target lattice velocity (default 0.05).
    """

    physics: Physics
    discretisation: Discretisation
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    lbm_settings: LbmSettings = field(default_factory=LbmSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    times: Times = field(default_factory=Times)
    characteristic_length: float = 1.0  # [m]
    characteristic_velocity: float = 1.0  # [m/s]
    u_lb: float = 0.05

    # --- JSON I/O ---
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CaseSchema:
        """Parse a case schema from a dictionary (e.g. loaded from JSON)."""
        phys = data.get("physics", {})
        disc = data.get("discretisation", {})
        bcs_raw = data.get("boundaryConditions", [])
        ic = data.get("initialConditions", {})
        lbm = data.get("lbmSettings", {})
        out = data.get("output", {})
        tms = data.get("times", {})

        boundary_conditions = []
        for bc in bcs_raw:
            boundary_conditions.append(BoundaryCondition(
                boundary_type=bc.get("boundaryType", "noSlip"),
                reference_plane=bc.get("referencePlane", ""),
                velocity=bc.get("velocity"),
                density=bc.get("density"),
            ))

        return cls(
            physics=Physics(
                density=phys.get("density", 1.0),
                kinematic_viscosity=phys.get("kinematicViscosity", 1e-6),
            ),
            discretisation=Discretisation(
                NX=disc.get("NX", 32),
                NY=disc.get("NY", 1),
                NZ=disc.get("NZ", 1),
            ),
            boundary_conditions=boundary_conditions,
            initial_conditions=InitialConditions(
                velocity=ic.get("velocity", {"x": 0.0, "y": 0.0, "z": 0.0}),
            ),
            lbm_settings=LbmSettings(
                D=lbm.get("D", 2),
                Q=lbm.get("Q", 9),
                collision_operator=lbm.get("collisionOperator", {}).get("type", "bgk"),
                floating_point_size=lbm.get("floatingPointSize", 8),
            ),
            output=OutputSettings(
                data_format=out.get("dataFormat", "vtk"),
                output_folder=out.get("outputFolder", "output"),
                write_interval=out.get("writeInterval", 1),
                first_output=out.get("firstOutput", 0),
            ),
            times=Times(
                start_time=tms.get("startTime", 0.0),
                end_time=tms.get("endTime", 100.0),
                warm_up=tms.get("warmUp", 0.0),
            ),
            characteristic_length=data.get("characteristicLength", 1.0),
            characteristic_velocity=data.get("characteristicVelocity", 1.0),
            u_lb=data.get("u_lb", 0.05),
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> CaseSchema:
        """Load a case schema from a JSON file."""
        with open(path) as f:
            return cls.from_json(json.load(f))

    def to_json(self) -> dict[str, Any]:
        """Serialise the case schema to a JSON-compatible dictionary."""
        bcs = []
        for bc in self.boundary_conditions:
            entry: dict[str, Any] = {
                "boundaryType": bc.boundary_type,
                "referencePlane": bc.reference_plane,
            }
            if bc.velocity is not None:
                entry["velocity"] = bc.velocity
            if bc.density is not None:
                entry["density"] = bc.density
            bcs.append(entry)

        return {
            "physics": {
                "density": self.physics.density,
                "kinematicViscosity": self.physics.kinematic_viscosity,
            },
            "discretisation": {
                "NX": self.discretisation.NX,
                "NY": self.discretisation.NY,
                "NZ": self.discretisation.NZ,
            },
            "boundaryConditions": bcs,
            "initialConditions": {
                "velocity": self.initial_conditions.velocity,
            },
            "lbmSettings": {
                "D": self.lbm_settings.D,
                "Q": self.lbm_settings.Q,
                "collisionOperator": {"type": self.lbm_settings.collision_operator},
                "floatingPointSize": self.lbm_settings.floating_point_size,
            },
            "output": {
                "dataFormat": self.output.data_format,
                "outputFolder": self.output.output_folder,
                "writeInterval": self.output.write_interval,
                "firstOutput": self.output.first_output,
            },
            "times": {
                "startTime": self.times.start_time,
                "endTime": self.times.end_time,
                "warmUp": self.times.warm_up,
            },
            "characteristicLength": self.characteristic_length,
            "characteristicVelocity": self.characteristic_velocity,
            "u_lb": self.u_lb,
        }

    def to_json_file(self, path: Path | str) -> None:
        """Write the case schema to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


def bootstrap_from_schema(schema: CaseSchema) -> LatticeBridge:
    """Bootstrap a ``LatticeBridge`` from a ``CaseSchema``.

    Uses the same operating-point logic as ``bootstrap_bridge``:
    ``dx = L / nx``, ``dt = u_lb * dx / u_phys``, with Ma and τ checks.

    Parameters
    ----------
    schema : CaseSchema
        Case definition with physical properties, grid, and target u_lb.

    Returns
    -------
    LatticeBridge
        Bootstrapped unit conversion bridge.

    Raises
    ------
    ValueError
        If Ma ≥ 0.1 or τ ∉ [0.51, 2.0].
    """
    return bootstrap_bridge(
        length_phys=schema.characteristic_length,
        velocity_phys=schema.characteristic_velocity,
        nu_phys=schema.physics.kinematic_viscosity,
        nx=schema.discretisation.NX,
        rho0=schema.physics.density,
        u_lb=schema.u_lb,
    )
