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
from .cylinder_flow import CylinderFlowConfig, compute_vorticity, run_cylinder_flow
from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from .d3q19 import C as C3D
from .d3q19 import OPPOSITE as OPPOSITE3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .dam_break import DamBreakConfig, run_dam_break
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
from .obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
    wigley_hull_mask,
)
from .ship_flow import ShipHullFlowConfig, run_ship_hull_flow
from .solver import collide_bgk, collide_mrt, stream
from .solver3d import collide_bgk3d, collide_mrt3d, stream3d
from .sphere_flow import SphereFlowConfig, run_sphere_flow
from .sphere_water_entry import SphereWaterEntryConfig, run_sphere_water_entry
from .turbulence import (
    collide_smagorinsky_bgk,
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_mrt3d,
)
from .utils import DiagnosticPoint, prepare_run_dir, resolve_device
from .wave_bc import (
    airy_wave_velocity_3d,
    apply_wave_inlet_3d,
    zou_he_inlet_velocity_profile_3d,
)

__all__ = [
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
    # 3D boundaries
    "sphere_mask",
    "make_channel_wall_mask_3d",
    "bounce_back_cells_3d",
    "zou_he_inlet_velocity_3d",
    "zou_he_outlet_pressure_3d",
    "apply_simple_channel_boundaries_3d",
    "apply_zou_he_channel_boundaries_3d",
    # 3D solvers
    "collide_bgk3d",
    "collide_mrt3d",
    "stream3d",
    # 3D runner
    "SphereFlowConfig",
    "run_sphere_flow",
    # Ship/ocean engineering – geometry and force diagnostics
    "wigley_hull_mask",
    "compute_obstacle_forces_3d",
    "compute_obstacle_moments_3d",
    # Ship/ocean engineering – Smagorinsky LES turbulence
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    # Ship/ocean engineering – wave boundary conditions
    "airy_wave_velocity_3d",
    "zou_he_inlet_velocity_profile_3d",
    "apply_wave_inlet_3d",
    # Ship hull flow runner
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
    # Sphere water-entry benchmark
    "SphereWaterEntryConfig",
    "run_sphere_water_entry",
    # Shared utilities
    "DiagnosticPoint",
    "resolve_device",
    "prepare_run_dir",
]
