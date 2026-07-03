"""General-purpose LBM simulation framework (XFlow-style).

Provides a unified interface for setting up and running LBM simulations
from geometry + physics, without case-specific code.

Workflow:
  1. Define geometry (STL import or parametric shape)
  2. Set physical conditions (velocity, viscosity, density)
  3. Auto-generate LBM domain (voxelise, unit conversion, boundary setup)
  4. Run simulation (BGK/MRT/Smagorinsky, 2D/3D)
  5. Output results (snapshots, forces, VTK/HDF5 export)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .unit_converter import LBMUnitConverter
from .preprocess_geo import voxelize_stl_3d, compute_q_generic_3d, poly_to_mask_2d
from .io import save_vtk_binary, save_hdf5


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


class GeometrySource(str, Enum):
    STL_FILE = "stl_file"
    PARAMETRIC_SPHERE = "parametric_sphere"
    PARAMETRIC_CYLINDER = "parametric_cylinder"
    PARAMETRIC_HULL = "parametric_hull"
    POLYGON_2D = "polygon_2d"
    NONE = "none"  # empty channel


class BoundaryType(str, Enum):
    ZOU_HE_INLET = "zou_he_inlet"
    ZOU_HE_OUTLET = "zou_he_outlet"
    PERIODIC = "periodic"
    WALL_BOUNCE_BACK = "wall_bounce_back"
    WALL_FREE_SLIP = "wall_free_slip"


class OutputFormat(str, Enum):
    VTK = "vtk"
    HDF5 = "hdf5"
    NPY = "npy"


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class BoundaryCondition:
    """Boundary condition for one face of the domain."""
    face: Literal["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    type: BoundaryType = BoundaryType.ZOU_HE_INLET
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
    collision: CollisionModel = CollisionModel.SMAGORINSKY_MRT
    # Resolution: number of lattice cells along reference_length
    resolution: int = 48
    # Domain padding around geometry (in multiples of reference_length)
    domain_padding: tuple[float, float, float, float, float, float] = (
        2.0, 4.0, 1.0, 1.0, 1.0, 1.0
    )  # (x_min, x_max, y_min, y_max, z_min, z_max)
    # Time stepping
    max_steps: int = 5000
    warmup_steps: int = 1000
    snapshot_interval: int = 100
    force_sample_interval: int = 10
    # Smagorinsky constant (if using Smagorinsky)
    smagorinsky_cs: float = 0.1
    # Target Mach number for auto tau calculation
    target_mach: float = 0.05
    # Device
    device: str = "cpu"


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
        # Default: inlet on x_min, outlet on x_max, walls elsewhere
        v = self.physics.inlet_velocity
        return [
            BoundaryCondition(face="x_min", type=BoundaryType.ZOU_HE_INLET, velocity=(v, 0.0, 0.0)),
            BoundaryCondition(face="x_max", type=BoundaryType.ZOU_HE_OUTLET),
            BoundaryCondition(face="y_min", type=BoundaryType.WALL_FREE_SLIP),
            BoundaryCondition(face="y_max", type=BoundaryType.WALL_FREE_SLIP),
            BoundaryCondition(face="z_min", type=BoundaryType.WALL_FREE_SLIP),
            BoundaryCondition(face="z_max", type=BoundaryType.WALL_FREE_SLIP),
        ]


# ── Simulation Engine ──────────────────────────────────────────────────────

class GeneralSimEngine:
    """XFlow-style general LBM simulation engine.

    Takes a GeneralSimConfig, automatically:
    - Converts physical units to lattice units
    - Voxelises geometry into obstacle mask
    - Sets up boundary conditions
    - Runs the LBM solver
    - Saves results
    """

    def __init__(self, config: GeneralSimConfig) -> None:
        self.config = config
        self.uc: LBMUnitConverter | None = None
        self.obstacle_mask: torch.Tensor | None = None
        self.q_field: torch.Tensor | None = None
        self.wall_mask: torch.Tensor | None = None
        self.f: torch.Tensor | None = None  # distribution function
        self.step_count: int = 0
        self.forces_log: list[dict] = []
        self.snapshots: list[dict] = []

    def setup(self) -> dict[str, Any]:
        """Phase 1: Setup domain, geometry, unit conversion."""
        cfg = self.config
        sol = cfg.solver
        phys = cfg.physics

        # 1. Unit conversion
        Re = phys.inlet_velocity * phys.reference_length / phys.viscosity
        self.uc = LBMUnitConverter(
            re=Re,
            l_phys=phys.reference_length,
            u_phys=phys.inlet_velocity,
            nu_phys=phys.viscosity,
            nx=sol.resolution,
        )

        # 2. Compute domain size in lattice units
        # Spatial conversion: dx = l_phys / resolution (physical m per lattice cell)
        dx = phys.reference_length / sol.resolution
        # Domain = geometry bounding box + padding
        geo_bbox = self._geometry_bounding_box()
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
        nz = int(round((domain_phys[5] - domain_phys[4]) / dx)) if sol.lattice != LatticeModel.D2Q9 else 1

        self.nx, self.ny, self.nz = nx, ny, nz
        self.domain_phys = domain_phys

        # 3. Voxelise geometry
        device = torch.device(sol.device)
        if sol.lattice == LatticeModel.D2Q9:
            self.obstacle_mask = self._voxelise_2d(nx, ny, device)
        else:
            self.obstacle_mask = self._voxelise_3d(nx, ny, nz, device)

        # 4. Compute q-field for Bouzidi BC
        if self.obstacle_mask is not None and sol.lattice != LatticeModel.D2Q9:
            self.q_field, _ = compute_q_generic_3d(
                self.obstacle_mask,
                device,
            )

        # 5. Wall mask (channel walls)
        if sol.lattice != LatticeModel.D2Q9:
            from .boundaries3d import make_channel_wall_mask_3d
            self.wall_mask = make_channel_wall_mask_3d(nz, ny, nx, self.obstacle_mask, device).to(device)

        # 6. Initialise distribution function
        if sol.lattice == LatticeModel.D2Q9:
            from .d2q9 import equilibrium
            rho0 = torch.ones((ny, nx), dtype=torch.float32, device=device)
            ux0 = torch.zeros((ny, nx), dtype=torch.float32, device=device)
            uy0 = torch.zeros((ny, nx), dtype=torch.float32, device=device)
            self.f = equilibrium(rho0, ux0, uy0)
        elif sol.lattice == LatticeModel.D3Q19:
            from .d3q19 import equilibrium3d
            rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
            ux0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
            uy0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
            uz0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
            self.f = equilibrium3d(rho0, ux0, uy0, uz0)
        elif sol.lattice == LatticeModel.D3Q27:
            from .d3q27 import equilibrium27
            rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
            ux0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
            uy0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
            uz0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
            self.f = equilibrium27(rho0, ux0, uy0, uz0)

        return {
            "status": "setup_complete",
            "Re": Re,
            "tau": self.uc.tau,
            "u_lb": self.uc.u_lb,
            "Ma": self.uc.ma,
            "domain_lu": (nx, ny, nz),
            "domain_phys_m": tuple(round(d, 3) for d in domain_phys),
            "obstacle_cells": int(self.obstacle_mask.sum()) if self.obstacle_mask is not None else 0,
            "total_cells": nx * ny * nz,
            "device": str(device),
        }

    def run(self, steps: int | None = None) -> dict[str, Any]:
        """Phase 2: Run simulation for given steps."""
        if self.f is None:
            raise RuntimeError("Call setup() first")

        cfg = self.config
        sol = cfg.solver
        n_steps = steps or sol.max_steps
        device = torch.device(sol.device)
        tau = self.uc.tau

        # Select collision operator
        collide_fn = self._get_collide_fn(tau)

        # Select stream function
        if sol.lattice == LatticeModel.D2Q9:
            from .solver import stream
            stream_fn = stream
        else:
            from .solver3d import stream3d
            stream_fn = stream3d

        # Boundary condition functions
        bc_fn = self._get_bc_fn()

        # Inlet velocity in lattice units
        u_in_lb = self.uc.u_lb

        # Run loop
        for step in range(n_steps):
            # Collide
            self.f = collide_fn(self.f)

            # Apply boundary conditions
            self.f = bc_fn(self.f, u_in_lb)

            # Stream
            self.f = stream_fn(self.f)

            # Bounce-back on obstacle
            if self.obstacle_mask is not None:
                if sol.lattice == LatticeModel.D2Q9:
                    from .boundaries import bounce_back_cells
                    self.f = bounce_back_cells(self.f, self.obstacle_mask)
                else:
                    from .boundaries3d import bounce_back_cells_3d
                    self.f = bounce_back_cells_3d(self.f, self.obstacle_mask)

            self.step_count += 1

            # Sample forces
            if cfg.output.save_forces and step % sol.force_sample_interval == 0:
                self._sample_forces()

            # Save snapshot
            if cfg.output.save_macroscopic and step % sol.snapshot_interval == 0:
                self._save_snapshot()

        return {
            "status": "completed",
            "steps": self.step_count,
            "snapshots": len(self.snapshots),
            "force_samples": len(self.forces_log),
        }

    def results(self, output_format: OutputFormat | None = None) -> dict[str, Any]:
        """Phase 3: Collect and export results."""
        cfg = self.config
        out_dir = Path(cfg.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use requested format or config default
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
                        raise ImportError("h5py not available — HDF5 export requires h5py (needs libhdf5)")
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

        # Compute Cd/Cl if forces available
        cd_cl = {}
        if self.forces_log and len(self.forces_log) > 10:
            recent = self.forces_log[-10:]
            fx_mean = np.mean([e.get("fx", 0) for e in recent])
            fy_mean = np.mean([e.get("fy", 0) for e in recent])
            rho_phys = cfg.physics.density
            u_phys = cfg.physics.inlet_velocity
            L_phys = cfg.physics.reference_length
            A_ref = L_phys ** 2  # frontal area approximation
            cd_cl["Cd"] = 2 * self.uc.lb_to_phys(fx_mean) / (rho_phys * u_phys**2 * A_ref) if A_ref > 0 else 0
            cd_cl["Cl"] = 2 * self.uc.lb_to_phys(fy_mean) / (rho_phys * u_phys**2 * A_ref) if A_ref > 0 else 0

        return {
            "status": "results_ready",
            "output_dir": str(out_dir),
            "saved_files": saved_files,
            "total_snapshots": len(self.snapshots),
            "total_force_samples": len(self.forces_log),
            "Cd_Cl": cd_cl,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _geometry_bounding_box(self) -> tuple[float, ...]:
        """Compute physical bounding box of geometry (x_min, x_max, y_min, y_max, z_min, z_max)."""
        geo = self.config.geometry
        if geo.source == GeometrySource.NONE:
            L = self.config.physics.reference_length
            return (-L, L, -L/2, L/2, -L/2, L/2)
        elif geo.source == GeometrySource.PARAMETRIC_SPHERE:
            r = geo.sphere_radius
            cx, cy, cz = geo.sphere_center
            return (cx-r, cx+r, cy-r, cy+r, cz-r, cz+r)
        elif geo.source == GeometrySource.PARAMETRIC_CYLINDER:
            r = geo.cylinder_radius
            L = geo.cylinder_length
            if geo.cylinder_axis == "x":
                return (-L/2, L/2, -r, r, -r, r)
            elif geo.cylinder_axis == "y":
                return (-r, r, -L/2, L/2, -r, r)
            else:
                return (-r, r, -r, r, -L/2, L/2)
        elif geo.source == GeometrySource.STL_FILE:
            # Load STL and compute bounding box
            from .ship_cad3d import import_mesh_stl
            mesh = import_mesh_stl(geo.stl_path, units=geo.stl_units)
            verts = mesh.vertices
            if geo.stl_units == "mm":
                verts = verts / 1000.0  # convert to meters
            return (
                float(verts[:, 0].min()), float(verts[:, 0].max()),
                float(verts[:, 1].min()), float(verts[:, 1].max()),
                float(verts[:, 2].min()), float(verts[:, 2].max()),
            )
        elif geo.source == GeometrySource.PARAMETRIC_HULL:
            L = geo.hull_length
            R = L * 0.1  # approximate radius
            return (-L*0.2, L*1.2, -R, R, -R, R)
        else:
            L = self.config.physics.reference_length
            return (-L, L, -L/2, L/2, -L/2, L/2)

    def _voxelise_3d(self, nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
        """Voxelise geometry into 3D obstacle mask."""
        geo = self.config.geometry
        if geo.source == GeometrySource.NONE:
            return torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
        elif geo.source == GeometrySource.STL_FILE:
            # Convert domain_phys to STL units for bbox_override
            dp = self.domain_phys
            if geo.stl_units == "mm":
                # domain_phys is in meters, convert to mm
                bbox = (dp[0]*1000, dp[1]*1000, dp[2]*1000, dp[3]*1000, dp[4]*1000, dp[5]*1000)
            else:
                bbox = (dp[0], dp[1], dp[2], dp[3], dp[4], dp[5])
            mask = voxelize_stl_3d(
                geo.stl_path,
                nx, ny, nz,
                device=device,
                bbox_override=bbox,
            )
            return mask
        elif geo.source == GeometrySource.PARAMETRIC_SPHERE:
            dx = self.config.physics.reference_length / self.config.solver.resolution
            r_lb = geo.sphere_radius / dx
            cx_lb = (geo.sphere_center[0] - self.domain_phys[0]) / dx
            cy_lb = (geo.sphere_center[1] - self.domain_phys[2]) / dx
            cz_lb = (geo.sphere_center[2] - self.domain_phys[4]) / dx
            z, y, x = torch.meshgrid(
                torch.arange(nz, device=device, dtype=torch.float32),
                torch.arange(ny, device=device, dtype=torch.float32),
                torch.arange(nx, device=device, dtype=torch.float32),
                indexing="ij",
            )
            return ((x - cx_lb)**2 + (y - cy_lb)**2 + (z - cz_lb)**2 <= r_lb**2)
        elif geo.source == GeometrySource.PARAMETRIC_CYLINDER:
            dx = self.config.physics.reference_length / self.config.solver.resolution
            r_lb = geo.cylinder_radius / dx
            L_lb = geo.cylinder_length / dx
            z, y, x = torch.meshgrid(
                torch.arange(nz, device=device, dtype=torch.float32),
                torch.arange(ny, device=device, dtype=torch.float32),
                torch.arange(nx, device=device, dtype=torch.float32),
                indexing="ij",
            )
            if geo.cylinder_axis == "x":
                return ((y - ny/2)**2 + (z - nz/2)**2 <= r_lb**2) & (x >= nx/2 - L_lb/2) & (x <= nx/2 + L_lb/2)
            elif geo.cylinder_axis == "y":
                return ((x - nx/2)**2 + (z - nz/2)**2 <= r_lb**2) & (y >= ny/2 - L_lb/2) & (y <= ny/2 + L_lb/2)
            else:
                return ((x - nx/2)**2 + (y - ny/2)**2 <= r_lb**2) & (z >= nz/2 - L_lb/2) & (z <= nz/2 + L_lb/2)
        else:
            return torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

    def _voxelise_2d(self, nx: int, ny: int, device: torch.device) -> torch.Tensor:
        """Voxelise geometry into 2D obstacle mask."""
        geo = self.config.geometry
        if geo.source == GeometrySource.NONE:
            return torch.zeros((ny, nx), dtype=torch.bool, device=device)
        elif geo.source == GeometrySource.POLYGON_2D and geo.polygon_vertices:
            # Convert physical coords to lattice units
            dx = self.config.physics.reference_length / self.config.solver.resolution
            x_min, x_max, y_min, y_max = self.domain_phys[:4]
            verts_lu = [
                ((vx - x_min) / dx, (vy - y_min) / dx)
                for vx, vy in geo.polygon_vertices
            ]
            mask = poly_to_mask_2d(verts_lu, ny=ny, nx=nx, device=device)
            return mask
        else:
            # Fallback: use 3D voxelisation and take middle slice
            mask_3d = self._voxelise_3d(nx, ny, 1, device)
            return mask_3d[0]

    def _get_collide_fn(self, tau: float):
        """Select collision operator based on config."""
        sol = self.config.solver
        cs = sol.smagorinsky_cs

        if sol.lattice == LatticeModel.D2Q9:
            if sol.collision == CollisionModel.BGK:
                from .solver import collide_bgk
                return lambda f: collide_bgk(f, tau)
            elif sol.collision == CollisionModel.SMAGORINSKY_BGK:
                from .turbulence import collide_smagorinsky_bgk
                return lambda f: collide_smagorinsky_bgk(f, tau, cs)
            else:
                from .solver import collide_bgk
                return lambda f: collide_bgk(f, tau)
        else:
            if sol.collision == CollisionModel.BGK:
                from .solver3d import collide_bgk3d
                return lambda f: collide_bgk3d(f, tau)
            elif sol.collision == CollisionModel.SMAGORINSKY_MRT:
                from .turbulence import collide_smagorinsky_mrt3d
                return lambda f: collide_smagorinsky_mrt3d(f, tau, cs)
            elif sol.collision == CollisionModel.SMAGORINSKY_BGK:
                from .turbulence import collide_smagorinsky_bgk3d
                return lambda f: collide_smagorinsky_bgk3d(f, tau, cs)
            elif sol.collision == CollisionModel.MRT:
                from .solver3d import collide_mrt3d
                return lambda f: collide_mrt3d(f, tau)
            else:
                from .solver3d import collide_bgk3d
                return lambda f: collide_bgk3d(f, tau)

    def _get_bc_fn(self):
        """Select boundary condition function based on config."""
        sol = self.config.solver
        bcs = self.config.auto_boundaries()

        if sol.lattice == LatticeModel.D2Q9:
            from .boundaries import apply_simple_channel_boundaries
            dev = torch.device(sol.device)
            wm = self.wall_mask if self.wall_mask is not None else torch.zeros((self.ny, self.nx), dtype=torch.bool, device=dev)
            om = self.obstacle_mask if self.obstacle_mask is not None else torch.zeros((self.ny, self.nx), dtype=torch.bool, device=dev)
            return lambda f, u_in: apply_simple_channel_boundaries(f, u_in, wm, om)
        else:
            from .boundaries3d import apply_zou_he_channel_boundaries_3d
            wm = self.wall_mask
            om = self.obstacle_mask
            return lambda f, u_in: apply_zou_he_channel_boundaries_3d(f, u_in, wm, om)

    def _sample_forces(self):
        """Sample hydrodynamic forces on obstacle."""
        if self.obstacle_mask is None:
            return
        sol = self.config.solver
        if sol.lattice == LatticeModel.D2Q9:
            from .boundaries import compute_obstacle_forces
            fx, fy = compute_obstacle_forces(self.f, self.obstacle_mask)
            self.forces_log.append({"step": self.step_count, "fx": float(fx), "fy": float(fy)})
        else:
            from .obstacles import compute_obstacle_forces_3d
            from .d3q19 import macroscopic3d
            rho, ux, uy, uz = macroscopic3d(self.f)
            fx, fy, fz = compute_obstacle_forces_3d(self.f, self.obstacle_mask)
            self.forces_log.append({"step": self.step_count, "fx": float(fx), "fy": float(fy), "fz": float(fz)})

    def _save_snapshot(self):
        """Save macroscopic fields snapshot."""
        sol = self.config.solver
        if sol.lattice == LatticeModel.D2Q9:
            from .d2q9 import macroscopic
            rho, ux, uy = macroscopic(self.f)
            self.snapshots.append({
                "rho": rho.clone(), "ux": ux.clone(), "uy": uy.clone(),
            })
        elif sol.lattice == LatticeModel.D3Q19:
            from .d3q19 import macroscopic3d
            rho, ux, uy, uz = macroscopic3d(self.f)
            self.snapshots.append({
                "rho": rho.clone(), "ux": ux.clone(), "uy": uy.clone(), "uz": uz.clone(),
            })
        elif sol.lattice == LatticeModel.D3Q27:
            from .d3q27 import macroscopic_d3q27
            rho, ux, uy, uz = macroscopic_d3q27(self.f)
            self.snapshots.append({
                "rho": rho.clone(), "ux": ux.clone(), "uy": uy.clone(), "uz": uz.clone(),
            })
