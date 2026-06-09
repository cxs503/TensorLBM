from ._version import __version__
from .adaptive_refinement import (
    AdaptationSchedule,
    AdaptiveSolver2D,
    AdaptiveSolver3D,
    AMRPatch2D,
    AMRPatch3D,
    gradient_indicator_2d,
    gradient_indicator_3d,
    mark_cells_for_refinement,
    nonequilibrium_indicator_2d,
    nonequilibrium_indicator_3d,
    vorticity_indicator_2d,
    vorticity_indicator_3d,
)
from .ai import (
    AIPipelineResult,
    EddyViscosityDataset,
    EddyViscosityMLP,
    FlowFieldTransformer,
    FlowTransformerArch,
    FlowTransformerTrainConfig,
    LBMDatabase,
    TrainConfig,
    build_flow_token_batch,
    collide_ai_les_bgk,
    extract_les_samples_2d,
    load_dataset_pt,
    load_flow_transformer_model,
    load_model,
    predict_nu_t_2d,
    predict_tau_eff_2d,
    reconstruct_flow_field,
    run_ai_dns_pipeline,
    run_ai_les_pipeline,
    save_dataset_pt,
    save_flow_transformer_model,
    save_model,
    strain_rate_tensor_2d,
    train_eddy_viscosity_model,
    train_flow_transformer_self_supervised,
)
from .backward_facing_step import (
    BackwardFacingStepConfig,
    make_bfs_solid_mask,
    measure_reattachment_length,
    run_backward_facing_step,
)
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
from .config_io import load_config, load_config_json, load_config_yaml, save_config_json
from .constants import D2Q9
from .cylinder_flow import CylinderFlowConfig, compute_vorticity, run_cylinder_flow
from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .d3q19 import OPPOSITE as OPPOSITE3D
from .d3q19 import C as C3D
from .d3q19 import W as W3D
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import OPPOSITE as OPPOSITE27
from .d3q27 import C as C27
from .d3q27 import W as W27
from .d3q27 import (
    collide_bgk27,
    collide_mrt27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .d3q27_sphere_flow import SphereFlowD3Q27Config, run_sphere_flow_d3q27
from .dam_break import DamBreakConfig, run_dam_break
from .hull_free_surface import HullFreeSurfaceConfig, run_hull_free_surface
from .ibm import (
    ibm_apply_body_force_2d,
    ibm_apply_body_force_3d,
    ibm_delta_4pt,
    ibm_delta_hat,
    ibm_direct_forcing,
    ibm_direct_forcing_3d,
    ibm_force_spread,
    ibm_force_spread_3d,
    ibm_velocity_interpolate,
    ibm_velocity_interpolate_3d,
)
from .interpolated_bc import (
    bouzidi_bounce_back,
    bouzidi_bounce_back_3d,
    compute_q_circle,
    compute_q_sphere,
)
from .io import save_hdf5, save_vtk, save_vtk_binary, save_xdmf
from .lid_driven_cavity import (
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    LidDrivenCavityConfig,
    compare_ghia,
    make_cavity_wall_mask,
    run_lid_driven_cavity,
    zou_he_moving_lid,
)
from .logging_config import configure_logging, logger
from .multiphase import (
    collide_sc_single_component,
    collide_sc_two_component,
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
    color_gradient_step_3d,
    free_energy_step_3d,
    init_free_energy_g_3d,
    sc_two_component_force_3d,
)
from .multiphase_benchmarks import (
    FreeEnergyDropletConfig,
    MultiphaseBenchmarkSuiteConfig,
    Spinodal3DConfig,
    SpinodaleConfig,
    StaticDroplet3DConfig,
    StaticDropletConfig,
    TwoPhaseChannelCompareConfig,
    run_free_energy_droplet,
    run_multiphase_benchmark_suite,
    run_spinodal_decomposition,
    run_spinodal_decomposition_3d,
    run_static_droplet,
    run_static_droplet_3d,
    run_two_phase_channel_compare,
)
from .multiphase_water_entry import MultiphaseWaterEntryConfig, run_multiphase_water_entry
from .non_newtonian import (
    apparent_viscosity_power_law,
    collide_power_law_bgk,
    strain_rate_magnitude_2d,
)
from .obstacles import (
    compute_obstacle_forces_3d,
    compute_obstacle_forces_27,
    compute_obstacle_moments_3d,
    wigley_hull_mask,
)
from .pipeline_flow import (
    PipelineFlowConfig,
    make_pipeline_wall_mask,
    measure_strouhal,
    run_pipeline_flow,
)
from .porous_media import (
    CapillaryInvasionConfig,
    LaplaceTestConfig,
    PorousDrainageConfig,
    TwoPhasePoiseuilleConfig,
    apply_wall_wettability_sc,
    make_random_cylinder_medium,
    make_tube_array_medium,
    run_capillary_invasion,
    run_laplace_test,
    run_porous_drainage,
    run_two_phase_poiseuille,
)
from .porous_media3d import (
    PorousDrainageConfig3D,
    make_random_sphere_medium,
    make_tube_array_medium_3d,
    run_porous_drainage_3d,
)
from .postprocess import (
    RunningStats,
    compute_added_mass_2d,
    compute_added_mass_3d,
    compute_divergence,
    compute_drag_lift_coefficients,
    compute_enstrophy_2d,
    compute_kinetic_energy,
    compute_lambda2_criterion,
    compute_pressure_coefficient,
    compute_q_criterion,
    compute_recirculation_length,
    compute_strouhal_fft,
    compute_velocity_magnitude,
    compute_vorticity_2d,
    compute_vorticity_3d,
    extract_velocity_profile,
    extract_wake_profile,
)
from .preprocess_geo import (
    compute_q_generic_3d,
    poly_to_mask_2d,
    poly_to_mask_and_q_2d,
    random_porosity_mask_2d,
    random_porosity_mask_3d,
    voxelize_stl_3d,
)
from .protocols import BoundaryCondition, CollisionOperator
from .rotating_cylinder import (
    RotatingCylinderConfig,
    moving_wall_bounce_back,
    rotating_wall_velocity,
    run_rotating_cylinder,
)
from .ship_cad import (  # noqa: I001
    ShipHullType,
    export_hull_stl,
    generate_hull_body_plan,
    generate_hull_previews,
    generate_hull_sideprofile,
    generate_hull_waterplane,
    hull_block_coefficient,
    hull_statistics,
    kcs_hull_mask,
    series60_hull_mask,
    ship_lbm_parameters,
    ship_resistance_estimate,
    theoretical_block_coefficient,
)
from .ship_cad import (
    build_hull_mask as build_ship_hull_mask,
)
from .ship_cad3d import (
    CADGeometryEngine,
    TriangleMesh,
    create_parametric_hull_mesh,
    export_mesh_gltf,
    export_mesh_stl_ascii,
    import_mesh_stl,
)
from .ship_flow import ShipHullFlowConfig, run_ship_hull_flow
from .simulation import LBMSimulation
from .sloshing_tank import (
    SloshingTankConfig,
    faltinsen_natural_frequency,
    make_sloshing_wall_mask,
    run_sloshing_tank,
)
from .solver import collide_bgk, collide_mrt, collide_rlbm, collide_trt, correct_mass, stream
from .solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    collide_rlbm3d,
    collide_trt3d,
    correct_mass3d,
    stream3d,
)
from .sphere_flow import SphereFlowConfig, run_sphere_flow
from .sphere_water_entry import SphereWaterEntryConfig, run_sphere_water_entry
from .suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    export_suboff_stl,
    generate_suboff_previews,
    suboff_hull_mask,
    suboff_radius_profile,
    suboff_statistics,
)
from .offshore_cad import (
    OffshoreStructureType,
    monopile_mask,
    jacket_mask,
    spar_mask,
    semi_sub_mask,
    build_offshore_mask,
    offshore_statistics,
    generate_offshore_previews,
    export_offshore_stl,
)
from .propeller_cad import (
    wageningen_b_series,
    optimal_advance_ratio,
    propeller_design,
    propeller_disk_mask,
)
from .suboff_resistance import (
    SuboffResistanceBenchmarkConfig,
    run_suboff_resistance_benchmark,
)
from .thermal import (
    C_D2Q5,
    W_D2Q5,
    apply_buoyancy_force,
    collide_thermal_bgk,
    equilibrium_thermal,
    macroscopic_thermal,
    stream_thermal,
)
from .thermal3d import (
    C_D3Q7,
    W_D3Q7,
    ThermalCavity3DConfig,
    apply_buoyancy_force_3d,
    collide_thermal_bgk_3d,
    equilibrium_thermal_3d,
    macroscopic_thermal_3d,
    run_thermal_cavity_3d,
    stream_thermal_3d,
)
from .turbulence import (
    collide_dynamic_smagorinsky_bgk,
    collide_dynamic_smagorinsky_bgk3d,
    collide_smagorinsky_bgk,
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_bgk27,
    collide_smagorinsky_mrt,
    collide_smagorinsky_mrt3d,
    collide_smagorinsky_mrt27,
    collide_vreman_bgk,
    collide_vreman_bgk3d,
    collide_vreman_bgk27,
    collide_wale_bgk,
    collide_wale_bgk3d,
    collide_wale_bgk27,
)
from .turbulent_channel import (
    TurbulentChannelConfig,
    log_law_velocity,
    run_turbulent_channel,
    viscous_sublayer_velocity,
)
from .unit_converter import LBMUnitConverter
from .utils import (
    DiagnosticPoint,
    configure_cpu_threads,
    flow_step_image_path,
    get_reproducibility_metadata,
    legacy_snapshot_image_path,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)
from .wave_bc import (
    airy_wave_velocity_3d,
    apply_jonswap_inlet_3d,
    apply_wave_inlet_3d,
    jonswap_spectrum,
    jonswap_wave_velocity_3d,
    zou_he_inlet_velocity_profile_3d,
)

__all__ = [
    "__version__",
    # Adaptive mesh refinement (AMR)
    "AdaptationSchedule",
    "AMRPatch2D",
    "AMRPatch3D",
    "AdaptiveSolver2D",
    "AdaptiveSolver3D",
    "nonequilibrium_indicator_2d",
    "vorticity_indicator_2d",
    "gradient_indicator_2d",
    "nonequilibrium_indicator_3d",
    "vorticity_indicator_3d",
    "gradient_indicator_3d",
    "mark_cells_for_refinement",
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
    "collide_rlbm",
    "collide_trt",
    "stream",
    "correct_mass",
    # 2D runner
    "CylinderFlowConfig",
    "run_cylinder_flow",
    "compute_vorticity",
    # Rotating cylinder (Magnus effect)
    "RotatingCylinderConfig",
    "run_rotating_cylinder",
    "rotating_wall_velocity",
    "moving_wall_bounce_back",
    # Lid-driven cavity benchmark
    "LidDrivenCavityConfig",
    "run_lid_driven_cavity",
    "zou_he_moving_lid",
    "make_cavity_wall_mask",
    "compare_ghia",
    "GHIA_RE100",
    "GHIA_RE400",
    "GHIA_RE1000",
    # Backward-facing step benchmark
    "BackwardFacingStepConfig",
    "run_backward_facing_step",
    "make_bfs_solid_mask",
    "measure_reattachment_length",
    # Near-bed pipeline flow benchmark
    "PipelineFlowConfig",
    "run_pipeline_flow",
    "make_pipeline_wall_mask",
    "measure_strouhal",
    # Sloshing tank benchmark
    "SloshingTankConfig",
    "run_sloshing_tank",
    "make_sloshing_wall_mask",
    "faltinsen_natural_frequency",
    # Turbulent channel flow benchmark
    "TurbulentChannelConfig",
    "run_turbulent_channel",
    "log_law_velocity",
    "viscous_sublayer_velocity",
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
    "collide_rlbm3d",
    "collide_trt3d",
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
    "compute_obstacle_forces_27",
    "compute_obstacle_moments_3d",
    # Turbulence
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_mrt",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    # WALE turbulence
    "collide_wale_bgk",
    "collide_wale_bgk3d",
    "collide_wale_bgk27",
    # Vreman turbulence
    "collide_vreman_bgk",
    "collide_vreman_bgk3d",
    "collide_vreman_bgk27",
    # Wave BC
    "airy_wave_velocity_3d",
    "zou_he_inlet_velocity_profile_3d",
    "apply_wave_inlet_3d",
    # Marine / ship
    "ShipHullFlowConfig",
    "run_ship_hull_flow",
    # Ship CAD module
    "ShipHullType",
    "series60_hull_mask",
    "kcs_hull_mask",
    "hull_block_coefficient",
    "hull_statistics",
    "theoretical_block_coefficient",
    "generate_hull_body_plan",
    "generate_hull_waterplane",
    "generate_hull_sideprofile",
    "generate_hull_previews",
    "export_hull_stl",
    "build_ship_hull_mask",
    "ship_lbm_parameters",
    "ship_resistance_estimate",
    # SUBOFF submarine CAD module
    "SuboffHullType",
    "SuboffConfig",
    "suboff_radius_profile",
    "suboff_hull_mask",
    "build_suboff_mask",
    "suboff_statistics",
    "generate_suboff_previews",
    "export_suboff_stl",
    "SuboffResistanceBenchmarkConfig",
    "run_suboff_resistance_benchmark",
    # Offshore structures CAD module
    "OffshoreStructureType",
    "monopile_mask",
    "jacket_mask",
    "spar_mask",
    "semi_sub_mask",
    "build_offshore_mask",
    "offshore_statistics",
    "generate_offshore_previews",
    "export_offshore_stl",
    # Propeller performance (Wageningen B-series)
    "wageningen_b_series",
    "optimal_advance_ratio",
    "propeller_design",
    "propeller_disk_mask",
    "CADGeometryEngine",
    "TriangleMesh",
    "create_parametric_hull_mesh",
    "import_mesh_stl",
    "export_mesh_stl_ascii",
    "export_mesh_gltf",
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
    "color_gradient_step_3d",
    "init_free_energy_g_3d",
    "free_energy_step_3d",
    # Multiphase benchmark suite
    "StaticDropletConfig",
    "run_static_droplet",
    "FreeEnergyDropletConfig",
    "run_free_energy_droplet",
    "StaticDroplet3DConfig",
    "run_static_droplet_3d",
    "SpinodaleConfig",
    "run_spinodal_decomposition",
    "Spinodal3DConfig",
    "run_spinodal_decomposition_3d",
    "TwoPhaseChannelCompareConfig",
    "run_two_phase_channel_compare",
    "MultiphaseBenchmarkSuiteConfig",
    "run_multiphase_benchmark_suite",
    # Dam-break benchmark
    "DamBreakConfig",
    "run_dam_break",
    # Multiphase water-entry benchmark
    "MultiphaseWaterEntryConfig",
    "run_multiphase_water_entry",
    # Porous-media gas-water displacement benchmarks
    "make_random_cylinder_medium",
    "make_tube_array_medium",
    "apply_wall_wettability_sc",
    "LaplaceTestConfig",
    "run_laplace_test",
    "CapillaryInvasionConfig",
    "run_capillary_invasion",
    "TwoPhasePoiseuilleConfig",
    "run_two_phase_poiseuille",
    "PorousDrainageConfig",
    "run_porous_drainage",
    # 3D porous-media benchmarks
    "make_random_sphere_medium",
    "make_tube_array_medium_3d",
    "PorousDrainageConfig3D",
    "run_porous_drainage_3d",
    # Immersed Boundary Method (IBM)
    "ibm_delta_hat",
    "ibm_delta_4pt",
    "ibm_velocity_interpolate",
    "ibm_force_spread",
    "ibm_direct_forcing",
    "ibm_apply_body_force_2d",
    # Shared utilities
    "DiagnosticPoint",
    "configure_cpu_threads",
    "resolve_device",
    "prepare_run_dir",
    "get_reproducibility_metadata",
    "flow_step_image_path",
    "legacy_snapshot_image_path",
    "write_legacy_snapshot_alias",
    "save_checkpoint",
    "load_checkpoint",
    "save_vtk",
    "save_vtk_binary",
    "save_hdf5",
    "save_xdmf",
    "extract_velocity_profile",
    "extract_wake_profile",
    "compute_recirculation_length",
    "compute_pressure_coefficient",
    "compute_q_criterion",
    "compute_lambda2_criterion",
    "compute_vorticity_2d",
    "compute_vorticity_3d",
    "compute_velocity_magnitude",
    "compute_kinetic_energy",
    "compute_enstrophy_2d",
    "compute_divergence",
    "compute_drag_lift_coefficients",
    "compute_strouhal_fft",
    "compute_added_mass_2d",
    "compute_added_mass_3d",
    "RunningStats",
    "CollisionOperator",
    "BoundaryCondition",
    "load_config",
    "load_config_yaml",
    "save_config_json",
    "load_config_json",
    # D3Q27 lattice
    "C27",
    "W27",
    "OPPOSITE27",
    "equilibrium27",
    "macroscopic27",
    "collide_bgk27",
    "stream27",
    "correct_mass27",
    "collide_mrt27",
    "collide_smagorinsky_bgk27",
    "collide_smagorinsky_mrt27",
    # D3Q27 boundaries
    "bounce_back_cells_27",
    "zou_he_inlet_velocity_27",
    "zou_he_outlet_pressure_27",
    "make_channel_wall_mask_27",
    "apply_zou_he_channel_boundaries_27",
    # D3Q27 runner
    "SphereFlowD3Q27Config",
    "run_sphere_flow_d3q27",
    # Interpolated BC
    "bouzidi_bounce_back",
    "compute_q_circle",
    "bouzidi_bounce_back_3d",
    "compute_q_sphere",
    # Logging
    "logger",
    "configure_logging",
    # Minimal D2Q9 scaffold
    "D2Q9",
    "LBMSimulation",
    # Pre-processing geometry
    "poly_to_mask_2d",
    "poly_to_mask_and_q_2d",
    "voxelize_stl_3d",
    "random_porosity_mask_2d",
    "random_porosity_mask_3d",
    "compute_q_generic_3d",
    # Unit converter
    "LBMUnitConverter",
    # Non-Newtonian (power-law) rheology
    "strain_rate_magnitude_2d",
    "apparent_viscosity_power_law",
    "collide_power_law_bgk",
    # Thermal LBM (D2Q9 + D2Q5 double-distribution)
    "C_D2Q5",
    "W_D2Q5",
    "equilibrium_thermal",
    "collide_thermal_bgk",
    "stream_thermal",
    "macroscopic_thermal",
    "apply_buoyancy_force",
    "C_D3Q7",
    "W_D3Q7",
    "ThermalCavity3DConfig",
    "equilibrium_thermal_3d",
    "collide_thermal_bgk_3d",
    "stream_thermal_3d",
    "macroscopic_thermal_3d",
    "apply_buoyancy_force_3d",
    "run_thermal_cavity_3d",
    # IBM 3D
    "ibm_velocity_interpolate_3d",
    "ibm_force_spread_3d",
    "ibm_direct_forcing_3d",
    "ibm_apply_body_force_3d",
    # Dynamic Smagorinsky
    "collide_dynamic_smagorinsky_bgk",
    "collide_dynamic_smagorinsky_bgk3d",
    # Wave BC additions
    "jonswap_spectrum",
    "jonswap_wave_velocity_3d",
    "apply_jonswap_inlet_3d",
    # Free-surface hull benchmark
    "HullFreeSurfaceConfig",
    "run_hull_free_surface",
    # AI turbulence (HPC + AI demo)
    "EddyViscosityDataset",
    "EddyViscosityMLP",
    "LBMDatabase",
    "TrainConfig",
    "AIPipelineResult",
    "extract_les_samples_2d",
    "strain_rate_tensor_2d",
    "save_dataset_pt",
    "load_dataset_pt",
    "save_model",
    "load_model",
    "train_eddy_viscosity_model",
    "predict_nu_t_2d",
    "predict_tau_eff_2d",
    "collide_ai_les_bgk",
    "run_ai_dns_pipeline",
    "run_ai_les_pipeline",
    # Transformer-based self-supervised flow model
    "FlowTransformerArch",
    "FlowTransformerTrainConfig",
    "FlowFieldTransformer",
    "build_flow_token_batch",
    "train_flow_transformer_self_supervised",
    "save_flow_transformer_model",
    "load_flow_transformer_model",
    "reconstruct_flow_field",
]
