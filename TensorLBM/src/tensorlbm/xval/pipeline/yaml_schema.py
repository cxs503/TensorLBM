"""YAML schema definition and validation for LBM cross-validation cases.

Defines the structure of YAML case files and provides loading/validation
functions. The schema covers geometry, physics, boundary conditions,
solver settings, and acceptance criteria.

Schema version: 1.0
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Data classes for YAML case structure
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GeometryConfig:
    """Geometry definition for an LBM case."""
    type: str  # "channel", "cavity", "cylinder", "sphere", "custom"
    dimensions: str  # "2D" or "3D"
    nx: int = 0
    ny: int = 0
    nz: int = 0
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometryConfig":
        return cls(
            type=d.get("type", "channel"),
            dimensions=d.get("dimensions", "2D"),
            nx=d.get("nx", 64),
            ny=d.get("ny", 64),
            nz=d.get("nz", 1),
            lx=d.get("lx", 1.0),
            ly=d.get("ly", 1.0),
            lz=d.get("lz", 1.0),
            description=d.get("description", ""),
        )


@dataclasses.dataclass
class PhysicsConfig:
    """Physics model configuration."""
    model: str  # "bgk", "mrt", "cumulant", "shanchen", "free_surface"
    lattice: str  # "D2Q9", "D3Q19", "D3Q27"
    reynolds: Optional[float] = None
    mach: Optional[float] = None
    viscosity: Optional[float] = None
    density: float = 1.0
    gravity: Optional[List[float]] = None
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicsConfig":
        return cls(
            model=d.get("model", "bgk"),
            lattice=d.get("lattice", "D2Q9"),
            reynolds=d.get("reynolds"),
            mach=d.get("mach"),
            viscosity=d.get("viscosity"),
            density=d.get("density", 1.0),
            gravity=d.get("gravity"),
            description=d.get("description", ""),
        )


@dataclasses.dataclass
class BoundaryConfig:
    """Boundary condition specification."""
    walls: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    inlet: Optional[Dict[str, Any]] = None
    outlet: Optional[Dict[str, Any]] = None
    symmetry: List[str] = dataclasses.field(default_factory=list)
    moving_boundary: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundaryConfig":
        return cls(
            walls=d.get("walls", []),
            inlet=d.get("inlet"),
            outlet=d.get("outlet"),
            symmetry=d.get("symmetry", []),
            moving_boundary=d.get("moving_boundary"),
        )


@dataclasses.dataclass
class SolverConfig:
    """Solver runtime configuration."""
    max_steps: int = 10000
    convergence_threshold: float = 1e-6
    output_interval: int = 100
    checkpoint_interval: int = 0
    seed: int = 42
    precision: str = "float64"
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverConfig":
        return cls(
            max_steps=d.get("max_steps", 10000),
            convergence_threshold=d.get("convergence_threshold", 1e-6),
            output_interval=d.get("output_interval", 100),
            checkpoint_interval=d.get("checkpoint_interval", 0),
            seed=d.get("seed", 42),
            precision=d.get("precision", "float64"),
            description=d.get("description", ""),
        )


@dataclasses.dataclass
class ReferenceConfig:
    """Analytical or experimental reference for verification."""
    type: str  # "analytical", "experimental", "numerical_benchmark"
    citation: str = ""
    doi: str = ""
    solution_fn: Optional[str] = None  # Python function name for analytical solution
    data_file: Optional[str] = None  # Path to experimental data file
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReferenceConfig":
        return cls(
            type=d.get("type", "analytical"),
            citation=d.get("citation", ""),
            doi=d.get("doi", ""),
            solution_fn=d.get("solution_fn"),
            data_file=d.get("data_file"),
            description=d.get("description", ""),
        )


@dataclasses.dataclass
class AcceptanceCriteria:
    """Acceptance thresholds for verification metrics."""
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    stopping_rule: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcceptanceCriteria":
        return cls(
            metrics=d.get("metrics", {}),
            stopping_rule=d.get("stopping_rule", ""),
        )


@dataclasses.dataclass
class ResolutionScan:
    """Resolution convergence scan specification."""
    variable: str = "nx"
    values: List[int] = dataclasses.field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResolutionScan":
        return cls(
            variable=d.get("variable", "nx"),
            values=d.get("values", []),
            description=d.get("description", ""),
        )


@dataclasses.dataclass
class XValYAMLCase:
    """Complete YAML case definition for LBM cross-validation."""
    schema_version: str = SCHEMA_VERSION
    anchor: str = ""
    title: str = ""
    description: str = ""
    category: str = "analytical"  # analytical, experimental, benchmark
    geometry: Optional[GeometryConfig] = None
    physics: Optional[PhysicsConfig] = None
    boundary: Optional[BoundaryConfig] = None
    solver: Optional[SolverConfig] = None
    reference: Optional[ReferenceConfig] = None
    acceptance: Optional[AcceptanceCriteria] = None
    resolution_scan: Optional[ResolutionScan] = None
    tags: List[str] = dataclasses.field(default_factory=list)
    external_anchor: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "XValYAMLCase":
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            anchor=d.get("anchor", ""),
            title=d.get("title", ""),
            description=d.get("description", ""),
            category=d.get("category", "analytical"),
            geometry=GeometryConfig.from_dict(d.get("geometry", {})),
            physics=PhysicsConfig.from_dict(d.get("physics", {})),
            boundary=BoundaryConfig.from_dict(d.get("boundary", {})),
            solver=SolverConfig.from_dict(d.get("solver", {})),
            reference=ReferenceConfig.from_dict(d.get("reference", {})),
            acceptance=AcceptanceCriteria.from_dict(d.get("acceptance", {})),
            resolution_scan=ResolutionScan.from_dict(d.get("resolution_scan", {})),
            tags=d.get("tags", []),
            external_anchor=d.get("external_anchor", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"schema_version", "anchor", "title", "geometry", "physics", "solver"}

def validate_yaml_case(data: Dict[str, Any]) -> List[str]:
    """Validate a YAML case dictionary against the schema.

    Returns a list of validation errors (empty if valid).
    """
    errors = []

    # Check required top-level keys
    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    # Check schema version
    if "schema_version" in data:
        if data["schema_version"] != SCHEMA_VERSION:
            errors.append(
                f"Unsupported schema version '{data['schema_version']}', "
                f"expected '{SCHEMA_VERSION}'"
            )

    # Validate geometry
    if "geometry" in data:
        g = data["geometry"]
        if "type" not in g:
            errors.append("geometry.type is required")
        if "dimensions" not in g:
            errors.append("geometry.dimensions is required")
        if g.get("dimensions") not in ("2D", "3D"):
            errors.append("geometry.dimensions must be '2D' or '3D'")

    # Validate physics
    if "physics" in data:
        p = data["physics"]
        if "model" not in p:
            errors.append("physics.model is required")
        if "lattice" not in p:
            errors.append("physics.lattice is required")
        valid_lattices = {"D2Q9", "D3Q15", "D3Q19", "D3Q27"}
        if p.get("lattice") not in valid_lattices:
            errors.append(f"physics.lattice must be one of {valid_lattices}")

    # Validate solver
    if "solver" in data:
        s = data["solver"]
        if "max_steps" in s:
            if not isinstance(s["max_steps"], int) or s["max_steps"] <= 0:
                errors.append("solver.max_steps must be a positive integer")

    # Validate acceptance criteria
    if "acceptance" in data:
        a = data["acceptance"]
        if "metrics" in a:
            if not isinstance(a["metrics"], dict):
                errors.append("acceptance.metrics must be a dictionary")
            else:
                for k, v in a["metrics"].items():
                    if not isinstance(v, (int, float)) or v <= 0:
                        errors.append(f"acceptance.metrics['{k}'] must be a positive number")

    return errors


def load_yaml_case(path: Union[str, Path]) -> XValYAMLCase:
    """Load and validate a YAML case file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    XValYAMLCase
        Parsed and validated case definition.

    Raises
    ------
    ValueError
        If the YAML file is invalid or fails schema validation.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML case file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping, got {type(data).__name__}")

    errors = validate_yaml_case(data)
    if errors:
        raise ValueError(
            f"YAML case validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return XValYAMLCase.from_dict(data)


def load_yaml_cases_from_dir(dir_path: Union[str, Path]) -> List[XValYAMLCase]:
    """Load all YAML case files from a directory.

    Parameters
    ----------
    dir_path : str or Path
        Directory containing YAML case files.

    Returns
    -------
    list[XValYAMLCase]
        List of parsed case definitions.
    """
    dir_path = Path(dir_path)
    cases = []
    for yaml_file in sorted(dir_path.glob("*.yaml")):
        try:
            case = load_yaml_case(yaml_file)
            cases.append(case)
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: skipping {yaml_file}: {e}")
    for yaml_file in sorted(dir_path.glob("*.yml")):
        try:
            case = load_yaml_case(yaml_file)
            cases.append(case)
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: skipping {yaml_file}: {e}")
    return cases
