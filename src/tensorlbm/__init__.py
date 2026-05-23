from ._version import __version__
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
    apply_water_entry_boundaries_3d,
    apply_zou_he_channel_boundaries_3d,
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
    make_tank_wall_mask_3d,
    sphere_mask,
    zou_he_inlet_velocity_3d,
    zou_he_inlet_velocity_z,
    zou_he_outlet_pressure_3d,
    zou_he_outlet_pressure_z,
)
from .boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    bounce_back_cells_27,
    make_channel_wall_mask_27,
    zou_he_inlet_velocity_27,
    zou_he_outlet_pressure_27,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .config_io import load_config
from .cylinder_flow import CylinderFlowConfig, compute_vorticity, run_cylinder_flow
from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .d3q19 import OPPOSITE as OPPOSITE3D
from .d3q19 import C as C3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import OPPOSITE as OPPOSITE27
from .d3q27 import C as C27
from .d3q27 import W as W27
from .d3q27 import collide_bgk27, collide_mrt27, equilibrium27, macroscopic27, stream27
from .dam_break import DamBreakConfig, run_dam_break
from .interpolated_bc import bouzidi_bounce_back, compute_q_circle
from .io import save_hdf5, save_vtk, save_vtk_binary
from .logging_config import configure_logging, logger
from .multiphase import (
    collide_sc_two_component,
    collide_sc_single_component,
    color_gradient_step,
    free_energy_step,
    init_free_energy_g,
    psi_exp,
    psi_linear,
    psi_power,
    sc_single_component_force,
    sc_two_component_force,
)
from .multiphase3d import (
    collide_sc_single_component_3d,
    collide_sc_two_component_3d,
    sc_two_component_force_3d,
)
from .multiphase_water_entry import MultiphaseWaterEntryConfig, run_multiphase_water_entry
from .obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
    wigley_hull_mask,
)
from .postprocess import (
    compute_pressure_coefficient,
    compute_q_criterion,
    compute_recirculation_length,
    compute_vorticity_3d,
    extract_velocity_profile,
    extract_wake_profile,
)
from .protocols import BoundaryCondition, CollisionOperator
from .ship_flow import ShipHullFlowConfig, run_ship_hull_flow
from .solver import collide_bgk, collide_mrt, correct_mass, stream
from .solver3d import collide_bgk3d, collide_mrt3d, correct_mass3d, stream3d
from .sphere_flow import SphereFlowConfig, run_sphere_flow
from .sphere_water_entry import SphereWaterEntryConfig, run_sphere_water_entry
from .turbulence import (
    collide_smagorinsky_bgk,
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_bgk27,
    collide_smagorinsky_mrt,
    collide_smagorinsky_mrt3d,
    collide_smagorinsky_mrt27,
)
from .utils import (
    DiagnosticPoint,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)
from .wave_bc import (
    airy_wave_velocity_3d,
    apply_wave_inlet_3d,
    zou_he_inlet_velocity_profile_3d,
)

__all__ = [
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
    "correct_mass",
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
    # 3D boundaries
    "sphere_mask",
    "make_channel_wall_mask_3d",
    "make_tank_wall_mask_3d",
    "bounce_back_cells_3d",
    "zou_he_inlet_velocity_3d",
    "zou_he_inlet_velocity_z",
    "zou_he_outlet_pressure_3d",
    "zou_he_outlet_pressure_z",
    "apply_simple_channel_boundaries_3d",
    "apply_zou_he_channel_boundaries_3d",
    "apply_water_entry_boundaries_3d",
    # 3D solvers
    "collide_bgk3d",
    "collide_mrt3d",
    "stream3d",
    "correct_mass3d",
    # 3D runners
    "SphereFlowConfig",
    "run_sphere_flow",
    # Single-phase sphere water entry (3D)
    "SphereWaterEntryConfig",
    "run_sphere_water_entry",
    "wigley_hull_mask",
    "compute_obstacle_forces_3d",
    "compute_obstacle_moments_3d",
    # Turbulence
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_mrt",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    # Wave BC
    "airy_wave_velocity_3d",
    "zou_he_inlet_velocity_profile_3d",
    "apply_wave_inlet_3d",
    # Marine / ship
    "ShipHullFlowConfig",
    "run_ship_hull_flow",
    # Multiphase models – D2Q9
    "psi_linear",
    "psi_exp",
    "psi_power",
    "sc_two_component_force",
    "collide_sc_two_component",
    "sc_single_component_force",
    "collide_sc_single_component",
    "color_gradient_step",
    "free_energy_step",
    "init_free_energy_g",
    # Multiphase models – D3Q19
    "sc_two_component_force_3d",
    "collide_sc_two_component_3d",
    "collide_sc_single_component_3d",
    # Dam-break benchmark
    "DamBreakConfig",
    "run_dam_break",
    # Multiphase water-entry benchmark
    "MultiphaseWaterEntryConfig",
    "run_multiphase_water_entry",
    # Shared utilities
    "DiagnosticPoint",
    "resolve_device",
    "prepare_run_dir",
    "get_reproducibility_metadata",
    "save_checkpoint",
    "load_checkpoint",
    "save_vtk",
    "save_vtk_binary",
    "save_hdf5",
    "extract_velocity_profile",
    "extract_wake_profile",
    "compute_recirculation_length",
    "compute_pressure_coefficient",
    "compute_q_criterion",
    "compute_vorticity_3d",
    "CollisionOperator",
    "BoundaryCondition",
    "load_config",
    # D3Q27 lattice
    "C27",
    "W27",
    "OPPOSITE27",
    "equilibrium27",
    "macroscopic27",
    "collide_bgk27",
    "stream27",
    "collide_mrt27",
    "collide_smagorinsky_bgk27",
    "collide_smagorinsky_mrt27",
    # D3Q27 boundaries
    "bounce_back_cells_27",
    "zou_he_inlet_velocity_27",
    "zou_he_outlet_pressure_27",
    "make_channel_wall_mask_27",
    "apply_zou_he_channel_boundaries_27",
    # Interpolated BC
    "bouzidi_bounce_back",
    "compute_q_circle",
    # Logging
    "logger",
    "configure_logging",
]
