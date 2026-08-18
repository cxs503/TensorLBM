"""General-purpose LBM simulation framework (XFlow-style) — Phase 1 platform fusion.

Provides a unified interface for setting up and running LBM simulations
from geometry + physics, routing ALL computation through the 9 common
interface modules:

  1. Geometry:  ``read_stl`` + ``voxelize_stl`` (stl_geometry.py) for STL,
                or parametric mask builders for cylinder/sphere/suboff/naca
  2. Near-wall: ``get_near_wall_3d(solid)`` (drag_pressure.py)
  3. Mesh:      ``SurfaceMesh.from_xxx(solid, near, ...)`` (drag_pressure.py)
  4. Main loop: ``lbm_step_correct(f, solid, collide_fn, tau, ...)`` (lbm_step_correct.py)
  5. Force:     ``drag_pressure_integration`` + ``drag_friction_integration`` (drag_pressure.py)
  6. St:        ``detect_strouhal(cl_hist, ...)`` (postprocess.py)
  7. BC:        ``far_field_bc_3d`` + ``bounce_back_cells_3d`` (boundaries3d.py)
  8. Wall fn:   ``wall_function_3d`` (wall_model.py) — for high Re, REPLACES BB
  9. MEM:       ``momentum_exchange_standard`` (momentum_exchange.py) — optional comparison

Auto parameter selection:
  - Auto domain size:  geometry bbox × 3 (blockage < 10 %)
  - Auto collision:    Re < 1000 → MRT,  Re ≥ 1000 → MRT + Smagorinsky
  - Auto wall treat.:  Re < 10000 → bounce-back,  Re ≥ 10000 → wall function
  - Auto warmup:       domain_size² × 0.5
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .unit_converter import LBMUnitConverter
from .io import save_vtk_binary, save_hdf5

# ── Common interface module imports (the 9 modules) ──────────────────────
# 1. Geometry (stl_geometry.py)
from .stl_geometry import read_stl, voxelize_stl, make_sphere_stl, make_cylinder_stl, make_naca_stl

# 2 + 3 + 5. Near-wall, SurfaceMesh, force integration (drag_pressure.py)
from .drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
    drag_total,
)

# 4. Main loop (lbm_step_correct.py)
from .lbm_step_correct import lbm_step_correct

# 6. Strouhal (postprocess.py)
from .postprocess import detect_strouhal

# 7. Boundary conditions (boundaries3d.py)
from .boundaries3d import far_field_bc_3d, bounce_back_cells_3d

# 8. Wall function (wall_model.py)
from .wall_model import wall_function_3d

# 9. Momentum exchange (momentum_exchange.py)
from .momentum_exchange import momentum_exchange_standard


# ── Enums ──────────────────────────────────────────────────────────────────


class LatticeModel(str, Enum):
    D2Q9 = "d2q9"
    D3Q19 = "d3q19"
    D3Q27 = "d3q27"


class CollisionModel(str, Enum):
    BGK = "bgk"
    MRT = "mrt"
    TRT = "trt"
    SMAGORINSKY_BGK = "smagorinsky_bgk"
    SMAGORINSKY_MRT = "smagorinsky_mrt"
    AUTO = "auto"  # auto-select based on Re


class GeometrySource(str, Enum):
    STL_FILE = "stl_file"
    PARAMETRIC_SPHERE = "parametric_sphere"
    PARAMETRIC_CYLINDER = "parametric_cylinder"
    PARAMETRIC_SUBOFF = "parametric_suboff"
    PARAMETRIC_NACA = "parametric_naca"
    PARAMETRIC_HULL = "parametric_hull"
    POLYGON_2D = "polygon_2d"
    NONE = "none"  # empty channel


class BoundaryType(str, Enum):
    ZOU_HE_INLET = "zou_he_inlet"
    ZOU_HE_OUTLET = "zou_he_outlet"
    FAR_FIELD = "far_field"
    PERIODIC = "periodic"
    WALL_BOUNCE_BACK = "wall_bounce_back"
    WALL_FREE_SLIP = "wall_free_slip"


class OutputFormat(str, Enum):
    VTK = "vtk"
    HDF5 = "hdf5"
    NPY = "npy"


class WallTreatment(str, Enum):
    """Wall treatment strategy for solid boundaries."""

    BOUNCE_BACK = "bounce_back"  # half-way bounce-back (Re < 10000)
    WALL_FUNCTION = "wall_function"  # log-law wall function (Re ≥ 10000)
    AUTO = "auto"  # auto-select based on Re


class ForceMethod(str, Enum):
    """Force computation method."""

    PRESSURE_FRICTION = "pressure_friction"  # drag_pressure + drag_friction
    MOMENTUM_EXCHANGE = "momentum_exchange"  # MEM (Ladd 1994)
    BOTH = "both"  # compute both for comparison


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class BoundaryCondition:
    """Boundary condition for one face of the domain."""

    face: Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    type: BoundaryType = BoundaryType.FAR_FIELD
    velocity: tuple[float, ...] = (0.0, 0.0, 0.0)  # physical units m/s
    pressure: float = 101325.0  # Pa (for outlet)


@dataclass
class GeometryConfig:
    """Geometry definition for the simulation."""

    source: GeometrySource = GeometrySource.NONE
    stl_path: str | None = None
    # Parametric shapes
    sphere_radius: float = 0.5  # physical m
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cylinder_radius: float = 0.5
    cylinder_length: float = 2.0
    cylinder_axis: Literal["x", "y", "z"] = "z"
    hull_type: str = "wigley"
    hull_length: float = 4.356
    # SUBOFF
    suboff_length: float = 4.356
    suboff_radius: float | None = None  # auto from r_over_l if None
    # NACA airfoil
    naca_chord: float = 1.0
    naca_thickness: float = 0.12  # NACA 0012
    naca_camber: float = 0.0  # symmetric
    naca_camber_pos: float = 0.40
    # 2D polygon (list of [x,y] vertices in physical m)
    polygon_vertices: list[list[float]] = field(default_factory=list)
    # STL units
    stl_units: Literal["m", "mm", "lu"] = "m"


@dataclass
class PhysicsConfig:
    """Physical conditions for the simulation."""

    density: float = 1000.0  # kg/m³ (water)
    viscosity: float = 1.0e-6  # m²/s (water at 20°C)
    inlet_velocity: float = 1.0  # m/s
    reference_length: float = 1.0  # m (characteristic length for Re)
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s²


@dataclass
class SolverConfig:
    """LBM solver settings."""

    lattice: LatticeModel = LatticeModel.D3Q19
    collision: CollisionModel = CollisionModel.AUTO
    # Resolution: number of lattice cells along reference_length
    resolution: int = 48
    # Domain padding around geometry (in multiples of reference_length)
    # When empty/auto, domain = geometry bbox × 3 (blockage < 10%)
    domain_padding: tuple[float, float, float, float, float, float] | None = None
    # (x_min, x_max, y_min, y_max, z_min, z_max) — None = auto
    # Time stepping
    max_steps: int = 5000
    warmup_steps: int | None = None  # None = auto (domain_size² × 0.5)
    snapshot_interval: int = 100
    force_sample_interval: int = 10
    # Smagorinsky constant (if using Smagorinsky)
    smagorinsky_cs: float = 0.1
    # Target Mach number for auto tau calculation
    target_mach: float = 0.05
    # Device
    device: str = "cpu"
    # Wall treatment
    wall_treatment: WallTreatment = WallTreatment.AUTO
    # Force method
    force_method: ForceMethod = ForceMethod.PRESSURE_FRICTION
    # Pressure extrapolation for drag: 'none', 'linear', 'quadratic'
    pressure_extrap: str = "none"
    # p0 method for pressure drag: 'near_wall', 'far_field', 'domain_avg', 'inlet'
    p0_method: str = "near_wall"
    # Friction formula: 'standard', '2nd_order', 'central', 'lagrange', 'bfl'
    friction_formula: str = "standard"
    # Mass correction
    mass_correction: bool = True
    mass_correction_interval: int = 200


@dataclass
class OutputConfig:
    """Output settings."""

    directory: str = "/tmp/tensorlbm_sim"
    formats: list[OutputFormat] = field(default_factory=lambda: [OutputFormat.NPY])
    save_macroscopic: bool = True  # rho, ux, uy, uz
    save_forces: bool = True  # Cd, Cl, etc.


@dataclass
class GeneralSimConfig:
    """Complete configuration for a general LBM simulation."""

    name: str = "unnamed"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    boundaries: list[BoundaryCondition] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)

    def auto_boundaries(self) -> list[BoundaryCondition]:
        """Generate default boundary conditions for external flow."""
        if self.boundaries:
            return self.boundaries
        # Default: far-field inlet on x_min, zero-gradient outlet on x_max,
        # far-field (free-stream) on lateral faces
        v = self.physics.inlet_velocity
        return [
            BoundaryCondition(face="x_min", type=BoundaryType.FAR_FIELD, velocity=(v, 0.0, 0.0)),
            BoundaryCondition(face="x_max", type=BoundaryType.FAR_FIELD),
            BoundaryCondition(face="y_min", type=BoundaryType.FAR_FIELD),
            BoundaryCondition(face="y_max", type=BoundaryType.FAR_FIELD),
            BoundaryCondition(face="z_min", type=BoundaryType.FAR_FIELD),
            BoundaryCondition(face="z_max", type=BoundaryType.FAR_FIELD),
        ]

    @property
    def reynolds_number(self) -> float:
        """Compute Reynolds number from physics config."""
        p = self.physics
        return p.inlet_velocity * p.reference_length / p.viscosity


# ── Simulation Engine ──────────────────────────────────────────────────────


class GeneralSimEngine:
    """XFlow-style general LBM simulation engine — Phase 1 platform fusion.

    Takes a GeneralSimConfig, automatically:
    - Converts physical units to lattice units
    - Voxelises geometry into obstacle mask (via stl_geometry / parametric)
    - Computes near-wall mask + SurfaceMesh with normals (drag_pressure)
    - Auto-selects collision, wall treatment, warmup
    - Runs the LBM solver via lbm_step_correct
    - Computes forces via drag_pressure/friction_integration (or MEM)
    - Detects Strouhal number via detect_strouhal
    - Saves results

    ALL computation routes through the 9 common interface modules.
    """

    # Thresholds for auto parameter selection
    RE_MRT_SMAG_THRESHOLD = 1000  # Re ≥ 1000 → MRT + Smagorinsky
    RE_WALL_FN_THRESHOLD = 10000  # Re ≥ 10000 → wall function
    AUTO_DOMAIN_FACTOR = 3  # domain = geometry bbox × 3

    def __init__(self, config: GeneralSimConfig) -> None:
        self.config = config
        self.uc: LBMUnitConverter | None = None
        self.solid: torch.Tensor | None = None  # solid mask (nz, ny, nx)
        self.near: torch.Tensor | None = None  # near-wall mask
        self.mesh: SurfaceMesh | None = None  # surface mesh with normals
        self.f: torch.Tensor | None = None  # distribution function
        self.step_count: int = 0
        self.forces_log: list[dict] = []
        self.snapshots: list[dict] = []
        # Auto-selected parameters (filled during setup)
        self._auto_collision: CollisionModel | None = None
        self._auto_wall_treatment: WallTreatment | None = None
        self._auto_warmup: int | None = None
        self._auto_domain: tuple[int, int, int] | None = None
        # STL data (for from_stl mesh construction)
        self._stl_vertices: np.ndarray | None = None
        self._stl_faces: np.ndarray | None = None
        self._stl_normals: np.ndarray | None = None

    # ── Auto parameter selection ───────────────────────────────────────

    def _auto_select_collision(self, Re: float) -> CollisionModel:
        """Auto-select collision operator based on Reynolds number.

        Re < 1000  → MRT (laminar, stable)
        Re ≥ 1000  → MRT + Smagorinsky (turbulent, LES)
        """
        if Re < self.RE_MRT_SMAG_THRESHOLD:
            return CollisionModel.MRT
        return CollisionModel.SMAGORINSKY_MRT

    def _auto_select_wall_treatment(self, Re: float) -> WallTreatment:
        """Auto-select wall treatment based on Reynolds number.

        Re < 10000  → bounce-back (half-way BB, accurate for resolved walls)
        Re ≥ 10000  → wall function (log-law, for high-Re unresolved walls)
        """
        if Re < self.RE_WALL_FN_THRESHOLD:
            return WallTreatment.BOUNCE_BACK
        return WallTreatment.WALL_FUNCTION

    def _auto_warmup_steps(self, domain_size: int) -> int:
        """Auto-compute warmup steps: domain_size² × 0.5."""
        return max(100, int(domain_size**2 * 0.5))

    def _auto_domain_size(self, geo_bbox: tuple[float, ...]) -> tuple[int, int, int]:
        """Auto-compute domain size: geometry bbox × 3 (blockage < 10%).

        Returns (nx, ny, nz) in lattice units.
        """
        phys = self.config.physics
        sol = self.config.solver
        dx = phys.reference_length / sol.resolution

        # Geometry bounding box in physical units
        gx_min, gx_max, gy_min, gy_max, gz_min, gz_max = geo_bbox
        geo_w = gx_max - gx_min
        geo_h = gy_max - gy_min
        geo_d = gz_max - gz_min

        # Domain = geometry bbox × factor (ensures blockage < 10%)
        factor = self.AUTO_DOMAIN_FACTOR
        # Flow direction is x; add extra upstream/downstream padding
        domain_w = geo_w * factor + geo_w * 0.5  # extra streamwise
        domain_h = geo_h * factor
        domain_d = geo_d * factor

        nx = max(int(round(domain_w / dx)), sol.resolution)
        ny = max(int(round(domain_h / dx)), sol.resolution)
        nz = max(int(round(domain_d / dx)), 4) if sol.lattice != LatticeModel.D2Q9 else 4

        return nx, ny, nz

    # ── Setup ──────────────────────────────────────────────────────────

    def setup(self) -> dict[str, Any]:
        """Phase 1: Setup domain, geometry, unit conversion, mesh.

        Routes through:
          - stl_geometry.read_stl + voxelize_stl (STL path)
          - parametric mask builders (cylinder/sphere/suboff/naca)
          - drag_pressure.get_near_wall_3d
          - drag_pressure.SurfaceMesh.from_xxx
        """
        cfg = self.config
        sol = cfg.solver
        phys = cfg.physics

        # 1. Reynolds number + auto parameter selection
        Re = cfg.reynolds_number

        # Auto collision
        if sol.collision == CollisionModel.AUTO:
            self._auto_collision = self._auto_select_collision(Re)
        else:
            self._auto_collision = sol.collision

        # Auto wall treatment
        if sol.wall_treatment == WallTreatment.AUTO:
            self._auto_wall_treatment = self._auto_select_wall_treatment(Re)
        else:
            self._auto_wall_treatment = sol.wall_treatment

        # 2. Unit conversion
        self.uc = LBMUnitConverter(
            re=Re,
            l_phys=phys.reference_length,
            u_phys=phys.inlet_velocity,
            nu_phys=phys.viscosity,
            nx=sol.resolution,
        )

        # 3. Compute domain size
        geo_bbox = self._geometry_bounding_box()
        if sol.domain_padding is not None:
            # Use explicit padding
            dx = phys.reference_length / sol.resolution
            pad = sol.domain_padding
            domain_phys = (
                geo_bbox[0] - pad[0] * phys.reference_length,
                geo_bbox[1] + pad[1] * phys.reference_length,
                geo_bbox[2] - pad[2] * phys.reference_length,
                geo_bbox[3] + pad[3] * phys.reference_length,
                geo_bbox[4] - pad[4] * phys.reference_length,
                geo_bbox[5] + pad[5] * phys.reference_length,
            )
            nx = int(round((domain_phys[1] - domain_phys[0]) / dx))
            ny = int(round((domain_phys[3] - domain_phys[2]) / dx))
            nz = (
                int(round((domain_phys[5] - domain_phys[4]) / dx))
                if sol.lattice != LatticeModel.D2Q9
                else 4
            )
            self.domain_phys = domain_phys
        else:
            # Auto domain size
            nx, ny, nz = self._auto_domain_size(geo_bbox)
            dx = phys.reference_length / sol.resolution
            # Compute domain_phys from grid size
            self.domain_phys = (
                geo_bbox[0] - (nx * dx - (geo_bbox[1] - geo_bbox[0])) / 2,
                geo_bbox[0] + nx * dx - (nx * dx - (geo_bbox[1] - geo_bbox[0])) / 2,
                geo_bbox[2] - (ny * dx - (geo_bbox[3] - geo_bbox[2])) / 2,
                geo_bbox[2] + ny * dx - (ny * dx - (geo_bbox[3] - geo_bbox[2])) / 2,
                geo_bbox[4] - (nz * dx - (geo_bbox[5] - geo_bbox[4])) / 2,
                geo_bbox[4] + nz * dx - (nz * dx - (geo_bbox[5] - geo_bbox[4])) / 2,
            )

        self._auto_domain = (nx, ny, nz)
        self.nx, self.ny, self.nz = nx, ny, nz

        # Auto warmup
        if sol.warmup_steps is None:
            domain_size = max(nx, ny, nz)
            self._auto_warmup = self._auto_warmup_steps(domain_size)
        else:
            self._auto_warmup = sol.warmup_steps

        # 4. Voxelise geometry → solid mask
        device = torch.device(sol.device)
        self.solid = self._build_solid_mask(nx, ny, nz, device)

        # 5. Near-wall mask (common module: drag_pressure.get_near_wall_3d)
        if self.solid is not None and self.solid.any():
            self.near = get_near_wall_3d(self.solid)
        else:
            self.near = torch.zeros_like(self.solid)

        # 6. SurfaceMesh with normals (common module: drag_pressure.SurfaceMesh)
        if self.solid is not None and self.solid.any():
            self.mesh = self._build_surface_mesh(device)
        else:
            self.mesh = None

        # 7. Initialise distribution function (D3Q19 equilibrium)
        u_in_lb = self.uc.u_lb
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
        ux0 = torch.full((nz, ny, nx), u_in_lb, dtype=torch.float32, device=device)
        if self.solid is not None:
            ux0[self.solid] = 0.0
        uy0 = torch.zeros_like(ux0)
        uz0 = torch.zeros_like(ux0)
        from .d3q19 import equilibrium3d

        self.f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
        self._initial_mass = float(rho0.sum().item())

        return {
            "status": "setup_complete",
            "Re": Re,
            "tau": self.uc.tau,
            "u_lb": self.uc.u_lb,
            "nu_lb": self.uc.nu_lb,
            "Ma": self.uc.ma,
            "domain_lu": (nx, ny, nz),
            "domain_phys_m": tuple(round(d, 3) for d in self.domain_phys),
            "obstacle_cells": int(self.solid.sum()) if self.solid is not None else 0,
            "near_wall_cells": int(self.near.sum()) if self.near is not None else 0,
            "total_cells": nx * ny * nz,
            "device": str(device),
            "auto_collision": self._auto_collision.value if self._auto_collision else None,
            "auto_wall_treatment": self._auto_wall_treatment.value
            if self._auto_wall_treatment
            else None,
            "auto_warmup": self._auto_warmup,
            "modules_used": [
                "stl_geometry.read_stl",
                "stl_geometry.voxelize_stl",
                "drag_pressure.get_near_wall_3d",
                "drag_pressure.SurfaceMesh",
                "lbm_step_correct.lbm_step_correct",
                "drag_pressure.drag_pressure_integration",
                "drag_pressure.drag_friction_integration",
                "postprocess.detect_strouhal",
                "boundaries3d.far_field_bc_3d",
                "boundaries3d.bounce_back_cells_3d",
                "wall_model.wall_function_3d",
                "momentum_exchange.momentum_exchange_standard",
            ],
        }

    # ── Run ────────────────────────────────────────────────────────────

    def run(self, steps: int | None = None) -> dict[str, Any]:
        """Phase 2: Run simulation for given steps.

        Routes through:
          - lbm_step_correct.lbm_step_correct (main loop)
          - boundaries3d.far_field_bc_3d (far-field BC)
          - boundaries3d.bounce_back_cells_3d (solid BB, via lbm_step_correct)
          - wall_model.wall_function_3d (high-Re wall function, replaces BB)
          - drag_pressure.drag_pressure_integration + drag_friction_integration
          - momentum_exchange.momentum_exchange_standard (optional)
        """
        if self.f is None:
            raise RuntimeError("Call setup() first")

        cfg = self.config
        sol = cfg.solver
        n_steps = steps or sol.max_steps
        device = torch.device(sol.device)
        tau = self.uc.tau
        nu_lb = self.uc.nu_lb
        u_in = self.uc.u_lb

        # Select collision operator
        collide_fn, collide_kwargs = self._get_collide_fn()

        # Build far-field BC function (common module: boundaries3d.far_field_bc_3d)
        bc_config = self._build_bc_config()
        far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

        # Mass correction
        correct_mass_fn = None
        target_mass = None
        if sol.mass_correction:
            try:
                from .solver3d import correct_mass3d

                correct_mass_fn = correct_mass3d
                target_mass = self._initial_mass
            except ImportError:
                pass

        # Wall treatment selection
        use_wall_function = (
            self._auto_wall_treatment == WallTreatment.WALL_FUNCTION
            and self.solid is not None
            and self.solid.any()
        )

        # dpS: dynamic pressure × reference area
        dpS = self._compute_dpS()

        # Run loop
        for step in range(1, n_steps + 1):
            if use_wall_function:
                # High-Re: wall function replaces bounce-back
                # wall_model.wall_function_3d applies Guo body force + returns drag
                f, _, _ = wall_function_3d(
                    self.f,
                    self.solid,
                    nu_lb,
                    near_mask=self.near,
                )
                # Then standard step without BB (far-field BC handles boundaries)
                self.f = lbm_step_correct(
                    self.f,
                    collide_fn,
                    tau,
                    self.solid
                    if self.solid is not None
                    else torch.zeros_like(self.f[0], dtype=torch.bool),
                    u_in,
                    far_field_fn,
                    correct_mass_fn=correct_mass_fn,
                    target_mass=target_mass,
                    step=step,
                    mass_interval=sol.mass_correction_interval,
                    **collide_kwargs,
                )
            else:
                # Standard: lbm_step_correct with half-way bounce-back
                self.f = lbm_step_correct(
                    self.f,
                    collide_fn,
                    tau,
                    self.solid
                    if self.solid is not None
                    else torch.zeros_like(self.f[0], dtype=torch.bool),
                    u_in,
                    far_field_fn,
                    correct_mass_fn=correct_mass_fn,
                    target_mass=target_mass,
                    step=step,
                    mass_interval=sol.mass_correction_interval,
                    **collide_kwargs,
                )

            self.step_count += 1

            # Sample forces
            if cfg.output.save_forces and step % sol.force_sample_interval == 0:
                self._sample_forces(dpS, nu_lb)

            # Save snapshot
            if cfg.output.save_macroscopic and step % sol.snapshot_interval == 0:
                self._save_snapshot()

            # Divergence guard
            if not torch.isfinite(self.f).all():
                break

        return {
            "status": "completed",
            "steps": self.step_count,
            "snapshots": len(self.snapshots),
            "force_samples": len(self.forces_log),
            "diverged": not torch.isfinite(self.f).all().item(),
        }

    # ── Results ────────────────────────────────────────────────────────

    def results(self, output_format: OutputFormat | None = None) -> dict[str, Any]:
        """Phase 3: Collect and export results.

        Includes:
          - Cd/Cl from drag_pressure/friction_integration
          - Strouhal number from detect_strouhal
          - MEM comparison (if force_method == BOTH)
        """
        cfg = self.config
        out_dir = Path(cfg.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        formats = [output_format] if output_format else cfg.output.formats

        # Save snapshots
        saved_files = []
        for i, snap in enumerate(self.snapshots):
            for fmt in formats:
                if fmt == OutputFormat.NPY:
                    for key, arr in snap.items():
                        p = out_dir / f"snapshot_{i}_{key}.npy"
                        np.save(str(p), arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr)
                        saved_files.append(str(p))
                elif fmt == OutputFormat.VTK:
                    p = out_dir / f"snapshot_{i}.vtk"
                    ux = snap.get("ux")
                    uy = snap.get("uy")
                    uz = snap.get("uz")
                    rho = snap.get("rho")
                    save_vtk_binary(str(p), ux=ux, uy=uy, uz=uz, rho=rho)
                    saved_files.append(str(p))
                elif fmt == OutputFormat.HDF5:
                    try:
                        import h5py  # noqa: F401
                    except ImportError:
                        raise ImportError("h5py not available — HDF5 export requires h5py")
                    p = out_dir / f"snapshot_{i}.h5"
                    ux = snap.get("ux")
                    uy = snap.get("uy")
                    uz = snap.get("uz")
                    rho = snap.get("rho")
                    save_hdf5(str(p), step=i, ux=ux, uy=uy, uz=uz, rho=rho)
                    saved_files.append(str(p))

        # Save forces
        if self.forces_log:
            forces_path = out_dir / "forces.csv"
            with open(forces_path, "w") as fh:
                keys = self.forces_log[0].keys()
                fh.write(",".join(keys) + "\n")
                for entry in self.forces_log:
                    fh.write(",".join(str(entry[k]) for k in keys) + "\n")
            saved_files.append(str(forces_path))

        # Compute Cd/Cl from force history
        cd_cl = {}
        if self.forces_log and len(self.forces_log) > 10:
            recent = self.forces_log[-min(100, len(self.forces_log)) :]
            cd_p_mean = np.mean([e.get("cd_pressure", 0) for e in recent])
            cd_f_mean = np.mean([e.get("cd_friction", 0) for e in recent])
            cd_tot_mean = np.mean([e.get("cd_total", 0) for e in recent])
            cl_mean = np.mean([e.get("cl", 0) for e in recent])
            cd_cl["Cd_pressure"] = float(cd_p_mean)
            cd_cl["Cd_friction"] = float(cd_f_mean)
            cd_cl["Cd_total"] = float(cd_tot_mean)
            cd_cl["Cl"] = float(cl_mean)

            # MEM comparison (if available)
            mem_vals = [e.get("cd_mem", None) for e in recent]
            if any(v is not None for v in mem_vals):
                cd_cl["Cd_MEM"] = float(np.mean([v for v in mem_vals if v is not None]))

        # Strouhal number (common module: postprocess.detect_strouhal)
        st = None
        if self.forces_log and len(self.forces_log) > 20:
            cl_hist = [e.get("cl", 0.0) for e in self.forces_log]
            D = cfg.physics.reference_length
            u_in_lb = self.uc.u_lb if self.uc else 0.05
            st = detect_strouhal(
                cl_hist,
                sample_rate=1.0,
                u_ref=u_in_lb,
                length_ref=cfg.solver.resolution,
                min_cycles=3,
            )
        cd_cl["St"] = st

        return {
            "status": "results_ready",
            "output_dir": str(out_dir),
            "saved_files": saved_files,
            "total_snapshots": len(self.snapshots),
            "total_force_samples": len(self.forces_log),
            "Cd_Cl": cd_cl,
            "modules_used": [
                "stl_geometry.read_stl",
                "stl_geometry.voxelize_stl",
                "drag_pressure.get_near_wall_3d",
                "drag_pressure.SurfaceMesh",
                "lbm_step_correct.lbm_step_correct",
                "drag_pressure.drag_pressure_integration",
                "drag_pressure.drag_friction_integration",
                "postprocess.detect_strouhal",
                "boundaries3d.far_field_bc_3d",
                "boundaries3d.bounce_back_cells_3d",
                "wall_model.wall_function_3d",
                "momentum_exchange.momentum_exchange_standard",
            ],
        }

    # ── Internal helpers: geometry ─────────────────────────────────────

    def _geometry_bounding_box(self) -> tuple[float, ...]:
        """Compute physical bounding box of geometry."""
        geo = self.config.geometry
        if geo.source == GeometrySource.NONE:
            L = self.config.physics.reference_length
            return (-L, L, -L / 2, L / 2, -L / 2, L / 2)
        elif geo.source == GeometrySource.PARAMETRIC_SPHERE:
            r = geo.sphere_radius
            cx, cy, cz = geo.sphere_center
            return (cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)
        elif geo.source == GeometrySource.PARAMETRIC_CYLINDER:
            r = geo.cylinder_radius
            L = geo.cylinder_length
            if geo.cylinder_axis == "x":
                return (-L / 2, L / 2, -r, r, -r, r)
            elif geo.cylinder_axis == "y":
                return (-r, r, -L / 2, L / 2, -r, r)
            else:
                return (-r, r, -r, r, -L / 2, L / 2)
        elif geo.source == GeometrySource.PARAMETRIC_SUBOFF:
            L = geo.suboff_length
            R = geo.suboff_radius or L * (1.0 / (2.0 * 8.57))
            return (-L / 2, L / 2, -R, R, -R, R)
        elif geo.source == GeometrySource.PARAMETRIC_NACA:
            c = geo.naca_chord
            t = geo.naca_thickness
            return (0.0, c, -c * t, c * t, -c * t, c * t)
        elif geo.source == GeometrySource.STL_FILE:
            # Read STL and compute bounding box (common module: stl_geometry.read_stl)
            vertices, faces, face_normals = read_stl(geo.stl_path)
            self._stl_vertices = vertices
            self._stl_faces = faces
            self._stl_normals = face_normals
            verts = vertices
            if geo.stl_units == "mm":
                verts = verts / 1000.0
            return (
                float(verts[:, 0].min()),
                float(verts[:, 0].max()),
                float(verts[:, 1].min()),
                float(verts[:, 1].max()),
                float(verts[:, 2].min()),
                float(verts[:, 2].max()),
            )
        elif geo.source == GeometrySource.PARAMETRIC_HULL:
            L = geo.hull_length
            R = L * 0.1
            return (-L * 0.2, L * 1.2, -R, R, -R, R)
        else:
            L = self.config.physics.reference_length
            return (-L, L, -L / 2, L / 2, -L / 2, L / 2)

    def _build_solid_mask(self, nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
        """Build solid mask using common modules.

        STL: stl_geometry.read_stl + voxelize_stl
        Parametric: mask builders (cylinder/sphere/suboff/naca)
        """
        geo = self.config.geometry
        if geo.source == GeometrySource.NONE:
            return torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

        elif geo.source == GeometrySource.STL_FILE:
            # Common module: stl_geometry.voxelize_stl
            # STL data already loaded in _geometry_bounding_box
            if self._stl_vertices is None:
                vertices, faces, face_normals = read_stl(geo.stl_path)
                self._stl_vertices = vertices
                self._stl_faces = faces
                self._stl_normals = face_normals

            verts = self._stl_vertices
            if geo.stl_units == "mm":
                verts = verts / 1000.0

            # Compute origin and spacing for voxelization
            dp = self.domain_phys
            dx = self.config.physics.reference_length / self.config.solver.resolution
            origin = (dp[0], dp[2], dp[4])
            spacing = (dx, dx, dx)

            solid = voxelize_stl(
                verts,
                self._stl_faces,
                grid_shape=(nx, ny, nz),
                origin=origin,
                spacing=spacing,
            )
            return solid.to(device)

        elif geo.source == GeometrySource.PARAMETRIC_SPHERE:
            return self._build_sphere_solid(nx, ny, nz, device)

        elif geo.source == GeometrySource.PARAMETRIC_CYLINDER:
            return self._build_cylinder_solid(nx, ny, nz, device)

        elif geo.source == GeometrySource.PARAMETRIC_SUBOFF:
            return self._build_suboff_solid(nx, ny, nz, device)

        elif geo.source == GeometrySource.PARAMETRIC_NACA:
            return self._build_naca_solid(nx, ny, nz, device)

        else:
            return torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

    def _build_sphere_solid(self, nx, ny, nz, device):
        """Boolean solid mask for a sphere."""
        geo = self.config.geometry
        dx = self.config.physics.reference_length / self.config.solver.resolution
        R_lb = geo.sphere_radius / dx
        cx_lb = (geo.sphere_center[0] - self.domain_phys[0]) / dx
        cy_lb = (geo.sphere_center[1] - self.domain_phys[2]) / dx
        cz_lb = (geo.sphere_center[2] - self.domain_phys[4]) / dx
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return ((xx - cx_lb) ** 2 + (yy - cy_lb) ** 2 + (zz - cz_lb) ** 2) < R_lb**2

    def _build_cylinder_solid(self, nx, ny, nz, device):
        """Boolean solid mask for a cylinder extruded along an axis."""
        geo = self.config.geometry
        dx = self.config.physics.reference_length / self.config.solver.resolution
        R_lb = geo.cylinder_radius / dx
        L_lb = geo.cylinder_length / dx
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        if geo.cylinder_axis == "x":
            cx = nx * 0.25
            cy = ny * 0.5
            mask = (yy - cy) ** 2 + (zz - nz / 2) ** 2 <= R_lb**2
            mask = mask & (xx >= cx - L_lb / 2) & (xx <= cx + L_lb / 2)
        elif geo.cylinder_axis == "y":
            cx = nx * 0.25
            cz = nz * 0.5
            mask = (xx - cx) ** 2 + (zz - cz) ** 2 <= R_lb**2
            mask = mask & (yy >= ny / 2 - L_lb / 2) & (yy <= ny / 2 + L_lb / 2)
        else:  # z
            cx = nx * 0.25
            cy = ny * 0.5
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= R_lb**2
            mask = mask & (zz >= nz / 2 - L_lb / 2) & (zz <= nz / 2 + L_lb / 2)
        return mask

    def _build_suboff_solid(self, nx, ny, nz, device):
        """Boolean solid mask for SUBOFF hull (via suboff_cad.build_suboff_mask)."""
        from .suboff_cad import build_suboff_mask, SuboffConfig

        geo = self.config.geometry
        length_lb = geo.suboff_length / (
            self.config.physics.reference_length / self.config.solver.resolution
        )
        radius_lb = geo.suboff_radius
        if radius_lb is not None:
            radius_lb = radius_lb / (
                self.config.physics.reference_length / self.config.solver.resolution
            )
        solid, _stats = build_suboff_mask(
            hull_type="bare_hull",
            nx=nx,
            ny=ny,
            nz=nz,
            cx=nx * 0.25,
            cy=ny * 0.5,
            cz=nz * 0.5,
            length=length_lb,
            radius=radius_lb,
            config=SuboffConfig(),
            device=str(device),
        )
        return solid

    def _build_naca_solid(self, nx, ny, nz, device):
        """Boolean solid mask for NACA airfoil (via stl_geometry.make_naca_stl + voxelize_stl)."""
        geo = self.config.geometry
        dx = self.config.physics.reference_length / self.config.solver.resolution
        chord_lb = geo.naca_chord / dx
        x_le = nx * 0.25
        y_mid = ny * 0.5
        z0 = 0.0
        z1 = float(nz)

        # Generate NACA STL (common module: stl_geometry.make_naca_stl)
        vertices, faces = make_naca_stl(
            chord=chord_lb,
            x_le=x_le,
            y_mid=y_mid,
            z0=z0,
            z1=z1,
            thickness_ratio=geo.naca_thickness,
        )
        # Voxelize (common module: stl_geometry.voxelize_stl)
        solid = voxelize_stl(
            vertices,
            faces,
            grid_shape=(nx, ny, nz),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        return solid.to(device)

    # ── Internal helpers: surface mesh ────────────────────────────────

    def _build_surface_mesh(self, device: torch.device) -> SurfaceMesh | None:
        """Build SurfaceMesh with normals (common module: drag_pressure.SurfaceMesh).

        Selects the appropriate from_xxx classmethod based on geometry source.
        """
        geo = self.config.geometry
        sol = self.config.solver
        dx = self.config.physics.reference_length / sol.resolution

        if geo.source == GeometrySource.STL_FILE:
            # Common module: SurfaceMesh.from_stl (via stl_geometry.SurfaceMesh_from_stl)
            dp = self.domain_phys
            origin = (dp[0], dp[2], dp[4])
            spacing = (dx, dx, dx)
            return SurfaceMesh.from_stl(
                self.solid,
                self.near,
                self._stl_vertices,
                self._stl_faces,
                self._stl_normals,
                origin,
                spacing,
            )

        elif geo.source == GeometrySource.PARAMETRIC_SPHERE:
            cx_lb = (geo.sphere_center[0] - self.domain_phys[0]) / dx
            cy_lb = (geo.sphere_center[1] - self.domain_phys[2]) / dx
            cz_lb = (geo.sphere_center[2] - self.domain_phys[4]) / dx
            R_lb = geo.sphere_radius / dx
            return SurfaceMesh.from_sphere(self.solid, self.near, cx_lb, cy_lb, cz_lb, R_lb)

        elif geo.source == GeometrySource.PARAMETRIC_CYLINDER:
            cx_lb = self.nx * 0.25
            cy_lb = self.ny * 0.5
            R_lb = geo.cylinder_radius / dx
            cz_lb = self.nz / 2.0
            return SurfaceMesh.from_cylinder(
                self.solid,
                self.near,
                cx_lb,
                cy_lb,
                R_lb,
                axis=geo.cylinder_axis,
                cz=cz_lb,
            )

        elif geo.source == GeometrySource.PARAMETRIC_SUBOFF:
            length_lb = geo.suboff_length / dx
            radius_lb = (geo.suboff_radius or geo.suboff_length * (1.0 / (2.0 * 8.57))) / dx
            cx_lb = self.nx * 0.25
            cy_lb = self.ny * 0.5
            cz_lb = self.nz * 0.5
            return SurfaceMesh.from_suboff(
                self.solid,
                self.near,
                cx_lb,
                cy_lb,
                cz_lb,
                length_lb,
                radius_lb,
            )

        elif geo.source == GeometrySource.PARAMETRIC_NACA:
            x_le = self.nx * 0.25
            y_c = self.ny * 0.5
            chord_lb = geo.naca_chord / dx
            return SurfaceMesh.from_naca(
                self.solid,
                self.near,
                x_le,
                y_c,
                chord_lb,
                m=geo.naca_camber,
                p=geo.naca_camber_pos,
                t=geo.naca_thickness,
            )

        else:
            # Fallback: gradient-based normals
            return SurfaceMesh.from_gradient(self.solid, self.near)

    # ── Internal helpers: collision, BC, force ─────────────────────────

    def _get_collide_fn(self):
        """Select collision operator based on auto-selection or config.

        Returns (collide_fn, collide_kwargs).
        """
        sol = self.config.solver
        collision = self._auto_collision or sol.collision
        cs = sol.smagorinsky_cs

        from .solver3d import collide_mrt3d, collide_bgk3d
        from .turbulence import collide_smagorinsky_mrt3d

        if collision == CollisionModel.BGK:
            return collide_bgk3d, {}
        elif collision == CollisionModel.MRT:
            return collide_mrt3d, {}
        elif collision == CollisionModel.SMAGORINSKY_MRT:
            return collide_smagorinsky_mrt3d, {"C_s": cs}
        elif collision == CollisionModel.SMAGORINSKY_BGK:
            from .turbulence import collide_smagorinsky_bgk3d

            return collide_smagorinsky_bgk3d, {"C_s": cs}
        else:
            # Default to MRT
            return collide_mrt3d, {}

    def _build_bc_config(self) -> dict:
        """Build bc_config dict for far_field_bc_3d.

        Determines far-field and periodic faces based on geometry axis.
        """
        geo = self.config.geometry
        sol = self.config.solver

        # For 2D extruded geometries (cylinder along z, naca along z),
        # make z-direction periodic
        if geo.source in (GeometrySource.PARAMETRIC_CYLINDER, GeometrySource.PARAMETRIC_NACA):
            if geo.cylinder_axis == "z" or geo.source == GeometrySource.PARAMETRIC_NACA:
                return {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
            elif geo.cylinder_axis == "y":
                return {"far_field_faces": ["z-", "z+"], "periodic_faces": ["y-", "y+"]}
            else:  # x
                return {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}

        # For 3D geometries (sphere, suboff, STL), all lateral faces are far-field
        if self.nz > 4:
            return {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
        else:
            return {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    def _compute_dpS(self) -> float:
        """Compute dynamic pressure × reference area (dpS) for drag normalization."""
        cfg = self.config
        u_in = self.uc.u_lb if self.uc else 0.05
        geo = cfg.geometry

        if geo.source == GeometrySource.PARAMETRIC_CYLINDER:
            # 2D extruded: diameter × span
            D = geo.cylinder_radius * 2
            span = self.nz  # lattice units
            A_frontal = D * span
        elif geo.source == GeometrySource.PARAMETRIC_SPHERE:
            R = geo.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
            A_frontal = math.pi * R**2
        elif geo.source == GeometrySource.PARAMETRIC_SUBOFF:
            R = geo.suboff_radius or geo.suboff_length * (1.0 / (2.0 * 8.57))
            R = R / (cfg.physics.reference_length / cfg.solver.resolution)
            A_frontal = math.pi * R**2
        elif geo.source == GeometrySource.PARAMETRIC_NACA:
            t = geo.naca_thickness
            c = geo.naca_chord / (cfg.physics.reference_length / cfg.solver.resolution)
            A_frontal = c * t * self.nz
        else:
            # Generic: reference_length²
            A_frontal = cfg.solver.resolution**2

        return 0.5 * u_in**2 * A_frontal

    def _sample_forces(self, dpS: float, nu_lb: float):
        """Sample hydrodynamic forces via common modules.

        Primary: drag_pressure_integration + drag_friction_integration
        Optional: momentum_exchange_standard (if force_method == BOTH)
        """
        cfg = self.config
        sol = cfg.solver
        if self.mesh is None or self.solid is None:
            return

        entry: dict[str, Any] = {"step": self.step_count}

        # Common module: drag_pressure.drag_pressure_integration
        fx_p, fy_p, fz_p = drag_pressure_integration(
            self.f,
            self.mesh,
            dpS,
            extrap=sol.pressure_extrap,
            p0_method=sol.p0_method,
            solid=self.solid,
        )
        # Common module: drag_pressure.drag_friction_integration
        fx_f, fy_f, fz_f = drag_friction_integration(
            self.f,
            self.mesh,
            dpS,
            nu_lb,
            formula=sol.friction_formula,
        )

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        entry["cd_pressure"] = cd_p
        entry["cd_friction"] = cd_f
        entry["cd_total"] = cd_tot
        entry["cl"] = cl
        entry["fx"] = fx_p + fx_f
        entry["fy"] = fy_p + fy_f
        entry["fz"] = fz_p + fz_f

        # Optional: MEM comparison (common module: momentum_exchange.momentum_exchange_standard)
        if sol.force_method in (ForceMethod.MOMENTUM_EXCHANGE, ForceMethod.BOTH) and self.near is not None:
            fx_mem, fy_mem, fz_mem = momentum_exchange_standard(
                self.f,
                self.solid,
                self.near,
            )
            entry["cd_mem"] = fx_mem / dpS if dpS > 0 else 0.0
            entry["fx_mem"] = fx_mem
            entry["fy_mem"] = fy_mem
            entry["fz_mem"] = fz_mem
            if sol.force_method == ForceMethod.MOMENTUM_EXCHANGE:
                # MEM is the primary force: report it as the total
                entry["cd_pressure"] = 0.0
                entry["cd_friction"] = 0.0
                entry["cd_total"] = entry["cd_mem"]
                entry["fx"] = fx_mem
                entry["fy"] = fy_mem
                entry["fz"] = fz_mem

        self.forces_log.append(entry)

    def _save_snapshot(self):
        """Save macroscopic fields snapshot."""
        from .d3q19 import macroscopic3d

        rho, ux, uy, uz = macroscopic3d(self.f)
        self.snapshots.append(
            {
                "rho": rho.clone(),
                "ux": ux.clone(),
                "uy": uy.clone(),
                "uz": uz.clone(),
            }
        )
