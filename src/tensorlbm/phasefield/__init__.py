"""Common, tensorised three-dimensional phase-field building blocks."""

from .diagnostics import phase_volume_smoothed, phase_volume_threshold
from .evolution_adapter import (
    COLLISION_ONLY_STAGE,
    D3Q19_POPULATIONS,
    NO_STREAMING_BOUNDARY_WITHHELD,
    FreeEnergyCollisionOnlyConfig,
    FreeEnergyCollisionOnlyDiagnostic,
    FreeEnergyCollisionOnlyResult,
    FreeEnergyCollisionOnlyState,
    initialize_free_energy_collision_only_state,
    run_free_energy_collision_only,
)
from .evolution_stream_loop import (
    ADAPTER_STREAM_LOOP_STAGE,
    FreeEnergyAdapterStreamLoopConfig,
    FreeEnergyAdapterStreamLoopDiagnostic,
    FreeEnergyAdapterStreamLoopResult,
    collision_then_adapter_stream,
    run_free_energy_adapter_stream_loop,
)
from .ch_validation import (
    FreeEnergyCHDiagnosticResult,
    FreeEnergyCHStepDiagnostic,
    FreeEnergyCHValidationConfig,
    run_closed_periodic_free_energy_diagnostic,
    uniform_phase_capillary_force,
)
from .free_energy import DoubleWellFreeEnergy, force_minus_phi_grad_mu, force_mu_grad_phi
from .operators import central_gradient_3d, laplacian_3d
from .phase_inventory_flux import (
    ADAPTER_STREAM_DIAGNOSTIC_ONLY,
    AdapterStreamBoundaryCrossing,
    PhaseInventoryFluxDiagnostic,
    PhaseInventoryFluxStep,
    adapter_stream_boundary_crossing,
    diagnose_adapter_stream_phase_inventory_flux,
)
from .static_droplet import (
    DropletGeometryDiagnostic,
    KortewegForceDiagnostic,
    LaplaceStyleDiagnostic,
    StaticDropletDiagnosticResult,
    diagnose_static_droplet,
    estimate_droplet_radius,
    initialize_static_droplet,
    periodic_chemical_potential_and_korteweg_force,
)

__all__ = [
    "ADAPTER_STREAM_LOOP_STAGE",
    "ADAPTER_STREAM_DIAGNOSTIC_ONLY",
    "AdapterStreamBoundaryCrossing",
    "COLLISION_ONLY_STAGE",
    "D3Q19_POPULATIONS",
    "DoubleWellFreeEnergy",
    "DropletGeometryDiagnostic",
    "FreeEnergyAdapterStreamLoopConfig",
    "FreeEnergyAdapterStreamLoopDiagnostic",
    "FreeEnergyAdapterStreamLoopResult",
    "FreeEnergyCHDiagnosticResult",
    "FreeEnergyCHStepDiagnostic",
    "FreeEnergyCHValidationConfig",
    "FreeEnergyCollisionOnlyConfig",
    "FreeEnergyCollisionOnlyDiagnostic",
    "FreeEnergyCollisionOnlyResult",
    "FreeEnergyCollisionOnlyState",
    "KortewegForceDiagnostic",
    "LaplaceStyleDiagnostic",
    "NO_STREAMING_BOUNDARY_WITHHELD",
    "PhaseInventoryFluxDiagnostic",
    "PhaseInventoryFluxStep",
    "StaticDropletDiagnosticResult",
    "adapter_stream_boundary_crossing",
    "central_gradient_3d",
    "collision_then_adapter_stream",
    "diagnose_adapter_stream_phase_inventory_flux",
    "diagnose_static_droplet",
    "estimate_droplet_radius",
    "force_minus_phi_grad_mu",
    "force_mu_grad_phi",
    "initialize_free_energy_collision_only_state",
    "initialize_static_droplet",
    "laplacian_3d",
    "phase_volume_smoothed",
    "phase_volume_threshold",
    "periodic_chemical_potential_and_korteweg_force",
    "run_closed_periodic_free_energy_diagnostic",
    "run_free_energy_adapter_stream_loop",
    "run_free_energy_collision_only",
    "uniform_phase_capillary_force",
]
