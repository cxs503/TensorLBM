from .backward_facing_step import BackwardFacingStepConfig, run_backward_facing_step
from .boundaries import (
    apply_simple_channel_boundaries,
    apply_zou_he_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from .boundaries3d import (
    apply_simple_channel_boundaries_3d,
    apply_zou_he_channel_boundaries_3d,
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from .boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    bounce_back_cells_27,
    compute_obstacle_forces_27,
    make_channel_wall_mask_27,
    zou_he_inlet_velocity_27,
    zou_he_outlet_pressure_27,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .config_io import load_config_json, save_config_json
from .cylinder_flow import CylinderFlowConfig, compute_vorticity, run_cylinder_flow
from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from .d3q19 import C as C3D
from .d3q19 import OPPOSITE as OPPOSITE3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import C as C27
from .d3q27 import OPPOSITE as OPPOSITE27
from .d3q27 import W as W27
from .d3q27 import (
    collide_bgk27,
    collide_mrt27,
    collide_smagorinsky_bgk27,
    collide_smagorinsky_mrt27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .d3q27_sphere_flow import SphereFlowD3Q27Config, run_sphere_flow_d3q27
from .lid_driven_cavity import (
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    LidDrivenCavityConfig,
    compare_ghia,
    run_lid_driven_cavity,
)
from .obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
    wigley_hull_mask,
)
from .postprocess import (
    compute_pressure,
    compute_q_criterion,
    compute_recirculation_length,
    compute_vorticity_3d,
    extract_wake_profile,
)
from .ship_flow import ShipHullFlowConfig, run_ship_hull_flow
from .solver import collide_bgk, collide_mrt, stream
from .solver3d import collide_bgk3d, collide_mrt3d, stream3d
from .sphere_flow import SphereFlowConfig, run_sphere_flow
from .turbulence import (
    collide_smagorinsky_bgk,
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_mrt,
    collide_smagorinsky_mrt3d,
)
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device
from .wave_bc import (
    airy_wave_velocity_3d,
    apply_wave_inlet_3d,
    zou_he_inlet_velocity_profile_3d,
)
from ._version import __version__

__all__ = [
    # Version
    "__version__",
    # D2Q9 lattice
    "C",
    "W",
    "OPPOSITE",
    "equilibrium",
    "macroscopic",
    # 2D boundaries
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "compute_obstacle_forces",
    "zou_he_inlet_velocity",
    "zou_he_outlet_pressure",
    "apply_simple_channel_boundaries",
    "apply_zou_he_channel_boundaries",
    # 2D solvers
    "collide_bgk",
    "collide_mrt",
    "stream",
    # 2D runner
    "CylinderFlowConfig",
    "run_cylinder_flow",
    "compute_vorticity",
    # D3Q19 lattice
    "C3D",
    "W3D",
    "OPPOSITE3D",
    "equilibrium3d",
    "macroscopic3d",
    # 3D boundaries (D3Q19)
    "sphere_mask",
    "make_channel_wall_mask_3d",
    "bounce_back_cells_3d",
    "zou_he_inlet_velocity_3d",
    "zou_he_outlet_pressure_3d",
    "apply_simple_channel_boundaries_3d",
    "apply_zou_he_channel_boundaries_3d",
    # 3D solvers (D3Q19)
    "collide_bgk3d",
    "collide_mrt3d",
    "stream3d",
    # 3D runner (D3Q19)
    "SphereFlowConfig",
    "run_sphere_flow",
    # D3Q27 lattice
    "C27",
    "W27",
    "OPPOSITE27",
    "equilibrium27",
    "macroscopic27",
    # D3Q27 boundaries
    "bounce_back_cells_27",
    "make_channel_wall_mask_27",
    "zou_he_inlet_velocity_27",
    "zou_he_outlet_pressure_27",
    "apply_zou_he_channel_boundaries_27",
    "compute_obstacle_forces_27",
    # D3Q27 solvers
    "collide_bgk27",
    "collide_mrt27",
    "collide_smagorinsky_bgk27",
    "collide_smagorinsky_mrt27",
    "stream27",
    "correct_mass27",
    # D3Q27 runner
    "SphereFlowD3Q27Config",
    "run_sphere_flow_d3q27",
    # Ship/ocean engineering – geometry and force diagnostics
    "wigley_hull_mask",
    "compute_obstacle_forces_3d",
    "compute_obstacle_moments_3d",
    # Ship/ocean engineering – Smagorinsky LES turbulence
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_mrt",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    # Ship/ocean engineering – wave boundary conditions
    "airy_wave_velocity_3d",
    "zou_he_inlet_velocity_profile_3d",
    "apply_wave_inlet_3d",
    # Ship hull flow runner
    "ShipHullFlowConfig",
    "run_ship_hull_flow",
    # CFD benchmarks
    "LidDrivenCavityConfig",
    "run_lid_driven_cavity",
    "GHIA_RE100",
    "GHIA_RE400",
    "GHIA_RE1000",
    "compare_ghia",
    "BackwardFacingStepConfig",
    "run_backward_facing_step",
    # Post-processing utilities
    "compute_vorticity_3d",
    "extract_wake_profile",
    "compute_recirculation_length",
    "compute_q_criterion",
    "compute_pressure",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    # Config I/O
    "save_config_json",
    "load_config_json",
    # Shared utilities
    "DiagnosticPoint",
    "resolve_device",
    "prepare_run_dir",
]

