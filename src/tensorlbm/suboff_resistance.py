"""SUBOFF resistance benchmark with iterative voxel-refinement control."""
from __future__ import annotations

import math
import hashlib
import json
import argparse
from dataclasses import dataclass, field

import torch

from .adaptive_refinement import (
    AdaptationSchedule,
    AdaptiveSolver3D,
    nonequilibrium_indicator_3d,
)
from .boundaries3d import (
    apply_zou_he_channel_boundaries_3d,
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from .d3q19 import C, equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import stream3d
from .suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask, suboff_statistics
from .turbulence import collide_smagorinsky_mrt3d
from .utils import resolve_device
from .rans_ke import KESolver
from .wall_model import apply_wall_model_bounce_back
from .wall_function_admission import WallFunctionRunRequest, require_wall_function_run
from .wall_function_contract import WallFunctionCapability


@dataclass(frozen=True)
class SuboffResistanceBenchmarkConfig:
    """Configuration for SUBOFF resistance-coefficient benchmark."""

    hull_type: str = SuboffHullType.BARE_HULL.value
    length_m: float = 4.356
    radius_m: float | None = None
    speed_ms: float = 2.5
    nu_m2s: float = 1.0e-6
    rho_kgm3: float = 1000.0
    base_length_lu: float = 48.0
    max_iterations: int = 3
    target_error_pct: float = 3.0
    numerics_max_coefficient_change_pct: float = 3.0
    device: str = "cpu"
    lbm_u_in: float = 0.06
    lbm_tau: float = 0.58
    lbm_steps: int = 60
    lbm_warmup_steps: int = 20
    lbm_sample_interval: int = 5
    conservation_max_relative_mass_drift: float = 1.0e-3
    conservation_max_relative_momentum_drift: float = 1.0e-3
    smagorinsky_cs: float = 0.1
    max_length_lu: float = 80.0
    use_adaptive_mesh: bool = False
    use_wall_model: bool = False
    use_rans_ke: bool = False
    adaptive_l1_pad: int = 4
    adaptive_l2_margin: int = 1
    adaptive_l2_pad: int = 2
    adaptive_interval: int = 5
    adaptive_refine_threshold: float = 1.0e-4
    adaptive_coarsen_threshold: float = 1.0e-6
    adaptive_max_patches: int = 8
    geometry: SuboffConfig = field(default_factory=SuboffConfig)
    # --- snapshot export (for ML training data) ---
    save_snapshots: bool = False
    snapshot_dir: str = "./suboff_snapshots"
    snapshot_start_step: int = 0
    snapshot_end_step: int = 0
    snapshot_interval: int = 1
    snapshot_crop_size: int = 100
    # Operator budgets are expensive full-field float64 reductions.  They are
    # opt-in diagnostics, never part of the normal benchmark hot path.
    momentum_budget_diagnostic: bool = False
    momentum_budget_interval: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "hull_type", SuboffHullType(self.hull_type).value)
        if self.length_m <= 0.0:
            raise ValueError("length_m must be > 0")
        if self.speed_ms <= 0.0:
            raise ValueError("speed_ms must be > 0")
        if self.nu_m2s <= 0.0:
            raise ValueError("nu_m2s must be > 0")
        if self.base_length_lu < 20.0:
            raise ValueError("base_length_lu must be >= 20")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.target_error_pct <= 0.0:
            raise ValueError("target_error_pct must be > 0")
        if (not math.isfinite(self.numerics_max_coefficient_change_pct)
                or self.numerics_max_coefficient_change_pct <= 0.0):
            raise ValueError("numerics_max_coefficient_change_pct must be finite and > 0")
        if not (0.0 < self.lbm_u_in < 0.15):
            raise ValueError("lbm_u_in must be in (0, 0.15)")
        if self.lbm_tau <= 0.5:
            raise ValueError("lbm_tau must be > 0.5")
        if self.lbm_steps < 10:
            raise ValueError("lbm_steps must be >= 10")
        if self.lbm_warmup_steps < 0:
            raise ValueError("lbm_warmup_steps must be >= 0")
        if self.lbm_sample_interval < 1:
            raise ValueError("lbm_sample_interval must be >= 1")
        if (not math.isfinite(self.conservation_max_relative_mass_drift)
                or self.conservation_max_relative_mass_drift < 0.0):
            raise ValueError("conservation_max_relative_mass_drift must be finite and >= 0")
        if (not math.isfinite(self.conservation_max_relative_momentum_drift)
                or self.conservation_max_relative_momentum_drift < 0.0):
            raise ValueError("conservation_max_relative_momentum_drift must be finite and >= 0")
        if self.max_length_lu < 20.0:
            raise ValueError("max_length_lu must be >= 20")
        if self.adaptive_l1_pad < 0:
            raise ValueError("adaptive_l1_pad must be >= 0")
        if self.adaptive_l2_margin < 0:
            raise ValueError("adaptive_l2_margin must be >= 0")
        if self.adaptive_l2_pad < 0:
            raise ValueError("adaptive_l2_pad must be >= 0")
        if self.adaptive_interval < 1:
            raise ValueError("adaptive_interval must be >= 1")
        if self.adaptive_refine_threshold <= 0.0:
            raise ValueError("adaptive_refine_threshold must be > 0")
        if self.adaptive_coarsen_threshold <= 0.0:
            raise ValueError("adaptive_coarsen_threshold must be > 0")
        if self.adaptive_coarsen_threshold >= self.adaptive_refine_threshold:
            raise ValueError(
                "adaptive_coarsen_threshold must be < adaptive_refine_threshold"
            )
        if self.adaptive_max_patches < 1:
            raise ValueError("adaptive_max_patches must be >= 1")
        if self.momentum_budget_interval < 1:
            raise ValueError("momentum_budget_interval must be >= 1")
        if self.use_wall_model:
            # Cold-path admission: the loop below must not decide capability.
            require_wall_function_run(WallFunctionRunRequest(
                capability=WallFunctionCapability.MOVING_BOUNCE_BACK,
                lattice="D3Q19",
                physics="single_phase_incompressible",
                collision="MRT_SMAGORINSKY",
                geometry="static_voxel_solid",
                backend="torch",
                adaptive_mesh=self.use_adaptive_mesh,
            ))

    @property
    def resolved_radius_m(self) -> float:
        if self.radius_m is not None:
            return float(self.radius_m)
        return self.geometry.r_over_l * self.length_m


def _ittc57_friction_coefficient(reynolds: float) -> float:
    if reynolds <= 100.0:
        raise ValueError("Reynolds number too low for ITTC-1957 formula")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


def _laminar_friction_coefficient(reynolds: float) -> float:
    """Blasius laminar flat-plate Cf.

    Cf = 1.328 / sqrt(Re)

    Valid for Re < 5e5 (laminar boundary layer).
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be > 0")
    return 1.328 / math.sqrt(max(reynolds, 1.0))


def _scale_cd_to_physical(
    cd_sim: float,
    re_lbm: float,
    re_phys: float,
) -> float:
    """Scale simulated Cd to physical Reynolds number using form-factor method.

    Assumes the form factor (1+k) = Cd/Cf is geometry-dependent and
    approximately Re-independent.  Then:

      Cf_sim   = 1.328 / sqrt(Re_lbm)      (laminar flat-plate)
      ff       = Cd_sim / Cf_sim            (form factor)
      Cf_phys  = ITTC-57 at Re_phys         (turbulent)
      Cd_pred  = Cf_phys * ff               (predicted physical Cd)

    Returns predicted Cd_phys (wetted-area reference).
    """
    cf_sim = _laminar_friction_coefficient(re_lbm)
    ff = cd_sim / max(cf_sim, 1e-10)
    cf_phys = _ittc57_friction_coefficient(re_phys)
    return cf_phys * ff


def _force_scale_factor(
    rho_phys: float, u_phys: float, l_phys: float,
    rho_lu: float, u_lu: float, l_lu: float,
) -> float:
    """Scale lattice force to physical force.

    F_phys = F_lu * (rho_phys/rho_lu) * (u_phys/u_lu)^2 * (l_phys/l_lu)^2
    """
    return rho_phys / rho_lu * (u_phys / u_lu) ** 2 * (l_phys / l_lu) ** 2


def _appendage_factor(hull_type: SuboffHullType) -> float:
    if hull_type == SuboffHullType.BARE_HULL:
        return 1.0
    if hull_type == SuboffHullType.WITH_SAIL:
        return 1.05
    return 1.12


def _voxel_wetted_area(mask: torch.Tensor, dx: float) -> float:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D tensor")

    m = mask
    area_faces = torch.tensor(0, dtype=torch.int64, device=m.device)
    area_faces += m[:, :, 0].sum()
    area_faces += m[:, :, -1].sum()
    area_faces += m[:, 0, :].sum()
    area_faces += m[:, -1, :].sum()
    area_faces += m[0, :, :].sum()
    area_faces += m[-1, :, :].sum()
    area_faces += (m[:, :, 1:] != m[:, :, :-1]).sum()
    area_faces += (m[:, 1:, :] != m[:, :-1, :]).sum()
    area_faces += (m[1:, :, :] != m[:-1, :, :]).sum()
    return float(area_faces.item()) * dx * dx


def voxel_wetted_area_x_slab(
    mask: torch.Tensor,
    dx: float,
    *,
    has_left_neighbor: bool,
    has_right_neighbor: bool,
) -> float:
    """Return wetted area of an x-decomposed physical slab.

    Interior rank cuts are communication interfaces, not exposed solid faces.
    Only a slab touching a physical x-domain boundary contributes that end
    face, making the summed area partition invariant.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D tensor")

    m = mask
    area_faces = torch.tensor(0, dtype=torch.int64, device=m.device)
    if not has_left_neighbor:
        area_faces += m[:, :, 0].sum()
    if not has_right_neighbor:
        area_faces += m[:, :, -1].sum()
    area_faces += m[:, 0, :].sum()
    area_faces += m[:, -1, :].sum()
    area_faces += m[0, :, :].sum()
    area_faces += m[-1, :, :].sum()
    area_faces += (m[:, :, 1:] != m[:, :, :-1]).sum()
    area_faces += (m[:, 1:, :] != m[:, :-1, :]).sum()
    area_faces += (m[1:, :, :] != m[:-1, :, :]).sum()
    return float(area_faces.item()) * dx * dx


def _crop_central_region(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Crop the central [crop_size]*3 region of a [nz, ny, nx] tensor."""
    _, ny, nx = tensor.shape[-3:]
    if crop_size >= min(ny, nx):
        return tensor
    z = tensor.shape[0] if tensor.ndim == 3 else 1
    sy = (ny - crop_size) // 2
    sx = (nx - crop_size) // 2
    sz = (max(z, crop_size) - crop_size) // 2
    if tensor.ndim == 3:
        return tensor[sz:sz + crop_size, sy:sy + crop_size, sx:sx + crop_size]
    return tensor[:, sy:sy + crop_size, sx:sx + crop_size]


def _lattice_momentum(f: torch.Tensor) -> torch.Tensor:
    """Return total D3Q19 population momentum in lattice momentum units.

    Components use the fixed (x, y, z) lattice axes.  A positive budget entry
    means the named operator added positive-axis fluid momentum.  Float64
    accumulation makes this an operator-budget observation, not a rho*u
    reconstruction artifact.
    """
    directions = C.to(device=f.device, dtype=torch.float64).view(19, 3, 1, 1, 1)
    return (f.to(torch.float64).unsqueeze(1) * directions).sum(dim=(0, 2, 3, 4))


def _masked_lattice_momentum(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """First population moment retained at cell centres selected by ``mask``.

    This is storage only.  It deliberately does not assign streamed
    populations, reconstructed Zou/He populations, or cell-reset bounce-back
    populations to physical links.
    """
    if mask.shape != f.shape[1:]:
        raise ValueError("mask must have spatial shape (nz, ny, nx) matching f")
    directions = C.to(device=f.device, dtype=torch.float64).view(19, 3, 1, 1, 1)
    weighted = f.to(torch.float64).unsqueeze(1) * directions
    return (weighted * mask.to(torch.float64).view(1, 1, *mask.shape)).sum(dim=(0, 2, 3, 4))


def _fluid_only_same_phase_control_volume_classification(
    operator_samples: list[dict[str, object]],
    *,
    wall_mask: torch.Tensor,
    solid_mask: torch.Tensor,
    full_per_step: bool,
) -> dict[str, object]:
    """Fail closed when this loop cannot define a physical fluid-only CV ledger.

    The implementation streams all cells through periodic ``roll`` and then
    mutates selected cells in Zou/He and bounce-back operators. It retains no
    per-population source/target link or overwrite provenance, so a numerical
    fluid-only residual would be a false closure.
    """
    fluid_mask = ~(wall_mask | solid_mask)
    overlap = wall_mask & solid_mask
    storage_samples: list[dict[str, object]] = []
    for sample in operator_samples:
        step = sample["step"]
        if not isinstance(step, int):
            raise ValueError("operator sample step must be an int")
        storage_samples.append({
            "step": step,
            "time_interval": {"start": f"retained_state[{step - 1}]",
                              "end": f"retained_state[{step}]"},
            "value": sample["fluid_only_storage_change"],
            "units": "lattice momentum (rho_lu * dx_lu^3 / dt_lu)",
        })
    return {
        "status": "not_definable",
        "kind": "fluid_only_same_phase_discrete_control_volume_momentum_ledger",
        "sample_phase": "retained_state_before_collision_to_post_complete_d3q19_bc_population_state",
        "coverage": "full_per_step" if full_per_step else "sampled",
        "fluid_mask": {
            "status": "measured",
            "definition": "not (channel_wall_mask or suboff_solid_mask)",
            "fluid_cell_count": int(fluid_mask.sum().item()),
            "wall_cell_count": int(wall_mask.sum().item()),
            "solid_cell_count": int(solid_mask.sum().item()),
            "wall_solid_overlap_cell_count": int(overlap.sum().item()),
            "solid_and_wall_excluded_from_storage": True,
        },
        "storage": {"status": "measured", "samples": storage_samples,
                    "meaning": "cell-centred fluid-mask population first-moment change only"},
        "stream_ownership": {
            "status": "not_definable",
            "implementation": "stream3d periodic torch.roll over every population cell",
            "missing_provenance": ["per_population_pre_stream_source_cell", "fluid_cv_crossing_link_classification"],
            "reason": "periodic_stream_has_no_recorded_source_target_link_ownership",
        },
        "zou_he_overwrite_ownership": {
            "status": "not_definable",
            "operators": ["zou_he_inlet_velocity_3d", "zou_he_outlet_pressure_3d"],
            "missing_provenance": ["overwritten_population_link_owner", "fluid_cv_boundary_link_classification"],
            "reason": "cell_plane_reconstruction_overwrites_populations_without_link_provenance",
        },
        "wall_solid_linkwise_exchange": {
            "status": "not_definable",
            "operators": ["bounce_back_cells_3d(wall_mask)", "bounce_back_cells_3d(solid_mask)"],
            "reason": "cell_based_population_reset_has_no_fluid_solid_link_pairing",
        },
        "control_volume_residual": {
            "status": "not_definable", "value": None,
            "reason": "required_link_owned_transport_and_boundary_impulse_terms_are_not_observable",
        },
        "prohibitions": [
            "cell_based_reset_delta_is_not_physical_traction",
            "do_not_use_full_array_operator_identity_as_fluid_only_control_volume_closure",
            "do_not_substitute_face_population_flux_for_discrete_link_crossing_term",
        ],
    }


def d3q19_x_face_momentum_flux(f: torch.Tensor) -> dict[str, list[float]]:
    """Measure population momentum flux through the D3Q19 inlet/outlet faces.

    ``Phi_alpha = sum_face sum_q f_q c_q,alpha c_q,x n_x`` is integrated on
    the two x-normal control faces.  The inlet uses the outward normal ``-x``;
    the outlet uses ``+x``.  Thus returned components are outward transport,
    and ``net_outward`` is inlet plus outlet.  Values are instantaneous lattice
    force / momentum-per-time: ``rho_lu * dx_lu**4 / dt_lu**2``.

    This is intentionally distinct from a Zou/He population-state delta.  It
    observes kinetic transport at a face and cannot close an operator-delta
    budget without a time-discrete control-volume formulation.
    """
    if f.ndim != 4 or f.shape[0] != 19:
        raise ValueError("f must have D3Q19 shape (19, nz, ny, nx)")
    directions = C.to(device=f.device, dtype=torch.float64)
    # q-by-component contribution to Pi_(component,x), accumulated on a face.
    contribution = directions * directions[:, 0:1]
    inlet_populations = f[:, :, :, 0].to(torch.float64).sum(dim=(1, 2))
    outlet_populations = f[:, :, :, -1].to(torch.float64).sum(dim=(1, 2))
    inlet = -(inlet_populations.unsqueeze(1) * contribution).sum(dim=0)
    outlet = (outlet_populations.unsqueeze(1) * contribution).sum(dim=0)
    net = inlet + outlet
    return {
        "inlet_outward": [float(value) for value in inlet.cpu().tolist()],
        "outlet_outward": [float(value) for value in outlet.cpu().tolist()],
        "net_outward": [float(value) for value in net.cpu().tolist()],
    }


def _same_time_control_volume_momentum_evidence(
    operator_samples: list[dict[str, object]],
    face_flux_samples: list[dict[str, object]],
    *,
    full_per_step: bool,
) -> dict[str, object]:
    """Bind retained-state operator and face-transport observations by step.

    This is evidence at one discrete sample phase, not a control-volume
    closure.  The loop does not expose every streaming or physical-boundary
    control-volume term required to calculate a residual.
    """
    flux_by_step = {sample.get("step"): sample for sample in face_flux_samples}
    missing_terms = [
        "streaming_face_crossing_term",
        "wall_control_volume_boundary_term",
        "solid_control_volume_boundary_term",
    ]
    samples: list[dict[str, object]] = []
    for operator_sample in operator_samples:
        step = operator_sample.get("step")
        face_sample = flux_by_step.get(step)
        if not isinstance(step, int) or not isinstance(face_sample, dict):
            return {
                "status": "withheld",
                "reason": "operator_and_face_flux_samples_are_not_phase_matched",
                "missing_terms": missing_terms,
                "samples": [],
            }
        samples.append({
            "step": step,
            "time_interval": {"start": f"retained_state[{step - 1}]",
                              "end": f"retained_state[{step}]"},
            "sample_phase": "post_complete_d3q19_bc_population_state",
            "storage_change": {
                "status": "measured",
                "value": operator_sample["fluid_momentum_delta"],
                "units": "lattice momentum (rho_lu * dx_lu^3 / dt_lu)",
            },
            "measured_x_face_transport": {
                "status": "measured",
                "time_index": f"retained_state[{step}]",
                "value": face_sample["net_outward"],
                "units": "lattice momentum flux / force (rho_lu * dx_lu^4 / dt_lu^2)",
                "sign_convention": "positive is net outward transport through x-normal faces",
            },
            "operator_state_deltas": {
                "status": "measured",
                "values": {name: operator_sample[name] for name in (
                    "collision", "streaming", "inlet_boundary", "outlet_boundary",
                    "wall_exchange", "solid_exchange", "unexplained_residual",
                )},
                "units": "lattice momentum (rho_lu * dx_lu^3 / dt_lu)",
                "meaning": "population-state deltas over the stated interval; not face-flux terms",
            },
            "control_volume_residual": {
                "status": "withheld",
                "value": None,
                "reason": "required_control_volume_terms_unavailable",
                "missing_terms": missing_terms,
            },
        })
    return {
        "status": "measured" if full_per_step else "sampled",
        "kind": "same_discrete_time_layer_control_volume_momentum_evidence",
        "control_volume": "entire retained D3Q19 lattice population domain; x-normal inlet and outlet faces only",
        "coverage": "full_per_step" if full_per_step else "sampled",
        "sample_phase": "post_complete_d3q19_bc_population_state",
        "closure": {
            "status": "withheld",
            "reason": "face_flux_is_not_a_population_delta_and_control_volume_terms_are_incomplete",
            "missing_terms": missing_terms,
        },
        "samples": samples,
    }


def _export_snapshot(
    f: torch.Tensor,
    step: int,
    config: SuboffResistanceBenchmarkConfig,
) -> None:
    """Export a single flow-field snapshot as 4 NPY files (p, ux, uy, uz)."""
    import os
    import numpy as np
    base = config.snapshot_dir
    dirs = (f"{base}/p", f"{base}/ux", f"{base}/uy", f"{base}/uz")
    rho, ux, uy, uz = macroscopic3d(f)
    rho_c = _crop_central_region(rho, config.snapshot_crop_size)
    ux_c = _crop_central_region(ux, config.snapshot_crop_size)
    uy_c = _crop_central_region(uy, config.snapshot_crop_size)
    uz_c = _crop_central_region(uz, config.snapshot_crop_size)
    idx = (step - config.snapshot_start_step) // config.snapshot_interval
    for arr, d in [(rho_c, dirs[0]), (ux_c, dirs[1]),
                    (uy_c, dirs[2]), (uz_c, dirs[3])]:
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{idx}.npy"), arr.cpu().numpy().astype(np.float32))


def _run_suboff_lbm_drag(
    *,
    config: SuboffResistanceBenchmarkConfig,
    hull_type: SuboffHullType,
    nx: int,
    ny: int,
    nz: int,
    length_lu: float,
    radius_lu: float,
) -> tuple[float, float, dict[str, object]]:
    device = resolve_device(config.device)
    mask, _stats = build_suboff_mask(
        hull_type=hull_type,
        nx=nx,
        ny=ny,
        nz=nz,
        length=length_lu,
        radius=radius_lu,
        config=config.geometry,
        device=str(device),
    )
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=device)

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, config.lbm_u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    # k-ε RANS solver initialization
    ke_solver: KESolver | None = None
    if config.use_rans_ke:
        nu_laminar = (config.lbm_tau - 0.5) / 3.0
        ke_solver = KESolver(nu=nu_laminar)
        _, ux_init, uy_init, uz_init = macroscopic3d(f)
        ke_solver.initialize(ux_init, uy_init, uz_init)

    form_stats_lu = suboff_statistics(hull_type, length_lu, radius_lu, config.geometry)
    ref_area_lu = float(form_stats_lu["wetted_area_lu2"])
    dyn_pressure_lu = 0.5 * config.lbm_u_in**2 * max(ref_area_lu, 1e-12)
    drag_samples: list[float] = []
    # Runtime numerical evidence is measured inside the actual solver loop.
    completed_steps = 0
    finite_population_checks = 0
    finite_density_checks = 0
    all_populations_finite = True
    all_densities_finite = True
    density_min = float("inf")
    density_max = float("-inf")
    # Conservation is sampled from the actual population state, including the
    # initialized lattice state before any update.  This is observational data,
    # not an inference from a finite drag coefficient.
    initial_mass = float(f.sum().item())
    initial_rho, initial_ux, initial_uy, initial_uz = macroscopic3d(f)
    initial_momentum = torch.stack(tuple(
        (initial_rho * velocity).sum()
        for velocity in (initial_ux, initial_uy, initial_uz)
    ))
    initial_population_momentum = (
        _lattice_momentum(f) if config.momentum_budget_diagnostic else None
    )
    initial_momentum_norm = float(torch.linalg.vector_norm(initial_momentum).item())
    final_mass = initial_mass
    final_momentum_norm = initial_momentum_norm
    max_abs_mass_drift = 0.0
    max_relative_mass_drift = 0.0
    max_abs_momentum_drift = 0.0
    max_relative_momentum_drift = 0.0
    mass_sample_count = 1
    # These full-field float64 reductions are deliberately allocated and used
    # only in explicit diagnostic mode.  A sample is an actual operator-state
    # snapshot; unsampled steps are never synthesized or represented as zero.
    momentum_budget_samples: list[dict[str, object]] = []
    # Face fluxes are sampled alongside the operator snapshots, but remain a
    # separate transport observation rather than a population-delta channel.
    face_flux_samples: list[dict[str, object]] = []
    face_flux_totals = {
        "inlet_outward": torch.zeros(3, dtype=torch.float64, device=device),
        "outlet_outward": torch.zeros(3, dtype=torch.float64, device=device),
        "net_outward": torch.zeros(3, dtype=torch.float64, device=device),
    }
    momentum_budget_totals = {
        "collision": torch.zeros(3, dtype=torch.float64, device=device),
        "streaming": torch.zeros(3, dtype=torch.float64, device=device),
        "inlet_boundary": torch.zeros(3, dtype=torch.float64, device=device),
        "outlet_boundary": torch.zeros(3, dtype=torch.float64, device=device),
        "wall_exchange": torch.zeros(3, dtype=torch.float64, device=device),
        "solid_exchange": torch.zeros(3, dtype=torch.float64, device=device),
        "unexplained_residual": torch.zeros(3, dtype=torch.float64, device=device),
    }
    coarse_cells = int(nx * ny * nz)

    adaptive_solver: AdaptiveSolver3D | None = None
    adaptive_cells: list[float] = []
    if config.use_adaptive_mesh:
        adaptive_solver = AdaptiveSolver3D(
            f,
            schedule=AdaptationSchedule(
                interval=config.adaptive_interval,
                warmup=config.lbm_warmup_steps,
                max_patches=config.adaptive_max_patches,
                refine_threshold=config.adaptive_refine_threshold,
                coarsen_threshold=config.adaptive_coarsen_threshold,
            ),
            mask=mask,
        )

    for step in range(1, config.lbm_steps + 1):
        sample_budget = (config.momentum_budget_diagnostic
                         and (step - 1) % config.momentum_budget_interval == 0)
        momentum_before = _lattice_momentum(f) if sample_budget else None
        fluid_mask = ~(wall_mask | mask)
        fluid_momentum_before = _masked_lattice_momentum(f, fluid_mask) if sample_budget else None
        if config.use_rans_ke and ke_solver is not None:
            _, ux, uy, uz = macroscopic3d(f)
            nu_t = ke_solver.step(ux, uy, uz, mask)
            nu_eff = (config.lbm_tau - 0.5) / 3.0 + nu_t.mean().item()
            tau_eff = min(max(3.0 * nu_eff + 0.5, 0.501), 2.0)
            f = collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)
        else:
            f = collide_smagorinsky_mrt3d(f, tau=config.lbm_tau, C_s=config.smagorinsky_cs)
        momentum_after_collision = _lattice_momentum(f) if sample_budget else None
        f = stream3d(f)
        momentum_after_stream = _lattice_momentum(f) if sample_budget else None
        fx, _, _ = compute_obstacle_forces_3d(f, mask)
        if config.use_wall_model:
            _, ux, uy, uz = macroscopic3d(f)
            nu = (config.lbm_tau - 0.5) / 3.0
            f = apply_wall_model_bounce_back(f, mask, ux, uy, uz, nu)
        if sample_budget:
            momentum_after_solid_preboundary = _lattice_momentum(f)
            f = zou_he_inlet_velocity_3d(f, config.lbm_u_in)
            momentum_after_inlet = _lattice_momentum(f)
            f = zou_he_outlet_pressure_3d(f)
            momentum_after_outlet = _lattice_momentum(f)
            f = bounce_back_cells_3d(f, wall_mask)
            momentum_after_wall = _lattice_momentum(f)
            if not config.use_wall_model:
                f = bounce_back_cells_3d(f, mask)
            momentum_after_boundaries = _lattice_momentum(f)
            assert (momentum_before is not None and momentum_after_collision is not None
                    and momentum_after_stream is not None)
            # Exact global first moments of the full retained population array
            # before and after each operator. This is an operator-domain ledger,
            # explicitly not a physical fluid control-volume balance.
            collision_delta = momentum_after_collision - momentum_before
            streaming_delta = momentum_after_stream - momentum_after_collision
            inlet_delta = momentum_after_inlet - momentum_after_solid_preboundary
            outlet_delta = momentum_after_outlet - momentum_after_inlet
            wall_delta = momentum_after_wall - momentum_after_outlet
            solid_delta = (momentum_after_solid_preboundary - momentum_after_stream
                           + momentum_after_boundaries - momentum_after_wall)
            total_delta = momentum_after_boundaries - momentum_before
            assert fluid_momentum_before is not None
            fluid_momentum_after = _masked_lattice_momentum(f, fluid_mask)
            fluid_only_storage_change = fluid_momentum_after - fluid_momentum_before
            explained_delta = (collision_delta + streaming_delta + inlet_delta + outlet_delta
                               + wall_delta + solid_delta)
            operator_deltas = {
                "collision": collision_delta, "streaming": streaming_delta,
                "inlet_boundary": inlet_delta, "outlet_boundary": outlet_delta,
                "wall_exchange": wall_delta, "solid_exchange": solid_delta,
                "unexplained_residual": total_delta - explained_delta,
            }
            for name, delta in operator_deltas.items():
                momentum_budget_totals[name] += delta
            # Measure after the complete inlet/outlet/wall/solid BC sequence,
            # on exactly the population state retained for the next step.
            face_flux = d3q19_x_face_momentum_flux(f)
            for name, values in face_flux.items():
                face_flux_totals[name] += torch.tensor(values, dtype=torch.float64, device=device)
            face_flux_samples.append({"step": step, **face_flux})
            momentum_budget_samples.append({
                "step": step,
                "fluid_momentum_delta": [float(value) for value in total_delta.cpu().tolist()],
                "fluid_only_storage_change": [
                    float(value) for value in fluid_only_storage_change.cpu().tolist()
                ],
                "face_flux": face_flux,
                **{name: [float(value) for value in delta.cpu().tolist()]
                   for name, delta in operator_deltas.items()},
                "operator_domain_ledger": {
                    "status": "measured",
                    "domain": "entire retained D3Q19 population array",
                    "sample_phase": "post_complete_d3q19_bc_population_state",
                    "operator_identity": {
                        "status": "measured",
                        "equation": (
                            "fluid_momentum_delta = collision + streaming + inlet_boundary "
                            "+ outlet_boundary + wall_exchange + solid_exchange "
                            "+ unexplained_residual"
                        ),
                        "residual": [float(value) for value in operator_deltas[
                            "unexplained_residual"].cpu().tolist()],
                        "meaning": (
                            "exact same-phase full-array population-momentum identity; "
                            "not a physical control-volume closure"
                        ),
                    },
                    "streaming": {
                        "fluid_momentum_change": [float(value) for value in streaming_delta.cpu().tolist()],
                        "implementation": "periodic torch.roll permutation",
                        "expected": "zero_global_population_momentum_delta",
                        "meaning": (
                            "stream3d is a periodic population permutation; any nonzero "
                            "floating-point reduction remainder is not a boundary flux"
                        ),
                    },
                    "wall_impulse": {
                        "fluid_momentum_change": [float(value) for value in wall_delta.cpu().tolist()],
                        "reaction_on_wall": [float(value) for value in (-wall_delta).cpu().tolist()],
                        "sign_convention": (
                            "fluid_momentum_change is added to fluid; reaction_on_wall "
                            "is the equal-and-opposite operator reaction"
                        ),
                        "scope": "bounce_back_cells_3d(wall_mask) over all retained populations",
                    },
                    "solid_impulse": {
                        "fluid_momentum_change": [float(value) for value in solid_delta.cpu().tolist()],
                        "reaction_on_solid": [float(value) for value in (-solid_delta).cpu().tolist()],
                        "sign_convention": (
                            "fluid_momentum_change is added to fluid; reaction_on_solid "
                            "is the equal-and-opposite operator reaction"
                        ),
                        "scope": (
                            "sum of all retained-population solid operators in this step "
                            "(wall model when enabled and obstacle bounce-back when enabled)"
                        ),
                    },
                },
            })
        else:
            # Preserve the established normal-path operator grouping exactly.
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=config.lbm_u_in, wall_mask=wall_mask,
                obstacle_mask=torch.zeros_like(mask) if config.use_wall_model else mask,
            )
        # Record direct per-step state observations. A finite final drag alone
        # cannot stand in for this numerical evidence.
        completed_steps += 1
        populations_finite = bool(torch.isfinite(f).all().item())
        finite_population_checks += 1
        all_populations_finite = all_populations_finite and populations_finite
        rho_step, ux_step, uy_step, uz_step = macroscopic3d(f)
        densities_finite = bool(torch.isfinite(rho_step).all().item())
        finite_density_checks += 1
        all_densities_finite = all_densities_finite and densities_finite
        if densities_finite:
            density_min = min(density_min, float(rho_step.min().item()))
            density_max = max(density_max, float(rho_step.max().item()))
        mass = float(f.sum().item())
        momentum = torch.stack(tuple(
            (rho_step * velocity).sum() for velocity in (ux_step, uy_step, uz_step)
        ))
        momentum_norm = float(torch.linalg.vector_norm(momentum).item())
        final_mass = mass
        final_momentum_norm = momentum_norm
        abs_mass_drift = abs(mass - initial_mass)
        abs_momentum_drift = abs(momentum_norm - initial_momentum_norm)
        max_abs_mass_drift = max(max_abs_mass_drift, abs_mass_drift)
        max_relative_mass_drift = max(
            max_relative_mass_drift, abs_mass_drift / max(abs(initial_mass), 1.0e-30),
        )
        max_abs_momentum_drift = max(max_abs_momentum_drift, abs_momentum_drift)
        max_relative_momentum_drift = max(
            max_relative_momentum_drift,
            abs_momentum_drift / max(abs(initial_momentum_norm), 1.0e-30),
        )
        mass_sample_count += 1
        # --- snapshot export ---
        if config.save_snapshots and step >= config.snapshot_start_step and step <= config.snapshot_end_step and (step - config.snapshot_start_step) % config.snapshot_interval == 0:
            _export_snapshot(f, step, config)
        if adaptive_solver is not None:
            adaptive_solver.coarse_f = f
            if adaptive_solver.should_adapt(step):
                rho, ux, uy, uz = macroscopic3d(f)
                indicator = nonequilibrium_indicator_3d(f, rho, ux, uy, uz)
                adaptive_solver.adapt(indicator)
            adaptive_cells.append(float(adaptive_solver.total_cells))
        if step > config.lbm_warmup_steps and (
            step % config.lbm_sample_interval == 0 or step == config.lbm_steps
        ):
            drag_samples.append(float(fx.item()))

    sampled_indices = list(range(1, config.lbm_steps + 1, config.momentum_budget_interval)
                           if config.momentum_budget_diagnostic else [])
    full_budget_coverage = (config.momentum_budget_diagnostic
                            and len(sampled_indices) == config.lbm_steps)
    momentum_budget_summary: dict[str, object] = {
        "status": "measured" if config.momentum_budget_diagnostic else "disabled",
        "mode": "operator_state_snapshots" if config.momentum_budget_diagnostic else "disabled",
        "interval": config.momentum_budget_interval if config.momentum_budget_diagnostic else None,
        "sampled_step_indices": sampled_indices,
        "sample_count": len(sampled_indices),
        "coverage": "full_per_step" if full_budget_coverage else (
            "sampled" if config.momentum_budget_diagnostic else "disabled"),
        "units": "lattice momentum per time step (rho_lu * dx_lu^4 / dt_lu)",
        "sign_convention": "positive component is momentum added to fluid along positive (x,y,z) lattice axis",
        "boundary_flux": {
            "status": "measured" if config.momentum_budget_diagnostic else "disabled",
            "kind": "face_integrated_population_momentum_flux" if config.momentum_budget_diagnostic else None,
            "sampling_state": "post_complete_d3q19_bc_population_state" if config.momentum_budget_diagnostic else None,
            "samples": face_flux_samples,
            "coverage": "full_per_step" if full_budget_coverage else (
                "sampled" if config.momentum_budget_diagnostic else "disabled"
            ),
            "sample_sum": {
                name: [float(value) for value in total.cpu().tolist()]
                for name, total in face_flux_totals.items()
            },
            "sample_sum_semantics": (
                "sum_of_observed_instantaneous_fluxes_only; not a time integral "
                "or full-run flux when coverage is sampled"
            ),
            "units": "lattice momentum flux / force (rho_lu * dx_lu^4 / dt_lu^2)",
            "sign_convention": (
                "outward control-volume transport: inlet normal is -x, "
                "outlet normal is +x; net_outward=inlet_outward+outlet_outward"
            ),
            "closure": {
                "status": "withheld",
                "reason": "face_flux_is_not_a_bc_population_delta",
            },
        },
        "body_force": {"status": "unavailable", "reason": "no body-force operator is enabled in this SUBOFF loop"},
        "samples": momentum_budget_samples,
        "cumulative_sampled": {
            **{name: [float(value) for value in total.cpu().tolist()]
               for name, total in momentum_budget_totals.items()},
        },
    }
    sampled_cumulative = momentum_budget_summary["cumulative_sampled"]
    assert isinstance(sampled_cumulative, dict)
    sampled_cumulative["closure_residual"] = list(sampled_cumulative["unexplained_residual"])
    if full_budget_coverage:
        assert initial_population_momentum is not None
        momentum_total = _lattice_momentum(f)
        sampled_cumulative["fluid_momentum_delta"] = [
            float(value) for value in (momentum_total - initial_population_momentum).cpu().tolist()
        ]
    else:
        momentum_budget_summary["closure"] = {
            "status": "withheld",
            "reason": "unsampled_steps_preclude_full_run_closure",
        }
    momentum_budget_summary["same_time_control_volume"] = (
        _same_time_control_volume_momentum_evidence(
            momentum_budget_samples, face_flux_samples,
            full_per_step=full_budget_coverage,
        ) if config.momentum_budget_diagnostic else {
            "status": "disabled", "reason": "operator_budget_disabled",
        }
    )
    momentum_budget_summary["fluid_only_same_phase_control_volume"] = (
        _fluid_only_same_phase_control_volume_classification(
            momentum_budget_samples,
            wall_mask=wall_mask,
            solid_mask=mask,
            full_per_step=full_budget_coverage,
        ) if config.momentum_budget_diagnostic else {
            "status": "disabled", "reason": "operator_budget_disabled",
        }
    )

    if adaptive_cells:
        active_cells = int(round(sum(adaptive_cells) / len(adaptive_cells)))
        finest_uniform_cells = int(coarse_cells * 9)
    else:
        active_cells = coarse_cells
        finest_uniform_cells = coarse_cells

    if not drag_samples:
        drag_samples.append(0.0)
    fx_lu = float(sum(drag_samples) / len(drag_samples))
    cd = abs(fx_lu) / dyn_pressure_lu
    mesh_stats = {
        "adaptive": bool(config.use_adaptive_mesh),
        "coarse_cells": coarse_cells,
        "active_cells": active_cells,
        "finest_uniform_cells": finest_uniform_cells,
        "cell_saving_pct": (
            (1.0 - float(active_cells) / max(float(finest_uniform_cells), 1.0)) * 100.0
        ),
        "runtime_evidence": {
            "requested_steps": config.lbm_steps,
            "completed_steps": completed_steps,
            "finite_population_checks": finite_population_checks,
            "finite_density_checks": finite_density_checks,
            "all_populations_finite": all_populations_finite,
            "all_densities_finite": all_densities_finite,
            "density_min": density_min if math.isfinite(density_min) else None,
            "density_max": density_max if math.isfinite(density_max) else None,
            "initial_lattice_mass": initial_mass,
            "final_lattice_mass": final_mass,
            "max_abs_mass_drift": max_abs_mass_drift,
            "max_relative_mass_drift": max_relative_mass_drift,
            "initial_lattice_momentum_norm": initial_momentum_norm,
            "final_lattice_momentum_norm": final_momentum_norm,
            "max_abs_momentum_drift": max_abs_momentum_drift,
            "max_relative_momentum_drift": max_relative_momentum_drift,
            "mass_sample_count": mass_sample_count,
            "sampled_step_count": completed_steps,
            "momentum_budget": momentum_budget_summary,
        },
    }
    return cd, fx_lu, mesh_stats


def run_suboff_resistance_benchmark(
    config: SuboffResistanceBenchmarkConfig,
) -> dict[str, object]:
    """Run iterative SUBOFF resistance benchmark and report convergence."""
    hull_type = SuboffHullType(config.hull_type)
    radius_m = config.resolved_radius_m
    if radius_m <= 0.0:
        raise ValueError("radius_m must be > 0")

    reynolds = config.speed_ms * config.length_m / config.nu_m2s
    cf = _ittc57_friction_coefficient(reynolds)
    form_factor = _appendage_factor(hull_type)
    ref_area = math.pi * radius_m**2

    form_stats = suboff_statistics(hull_type, config.length_m, radius_m, config.geometry)
    ref_area = float(form_stats["wetted_area_lu2"])
    cd_analytical = cf * form_factor
    resistance_analytical_n = 0.5 * config.rho_kgm3 * config.speed_ms**2 * ref_area * cd_analytical

    iterations: list[dict[str, object]] = []
    final_error = float("inf")
    best_cd = float("nan")
    best_resistance_n = float("nan")
    best_cd_lbm_re_analytical = float("nan")
    richardson_cd = float("nan")
    prev_cd: float | None = None
    refinement_ratio = 2.0
    order = 1.0

    for k in range(1, config.max_iterations + 1):
        scale = 2.0 ** float(k - 1)
        length_lu = min(config.base_length_lu * scale, config.max_length_lu)
        radius_lu = (radius_m / config.length_m) * length_lu
        nx = max(int(round(length_lu * 1.8)), int(round(length_lu + 12)))
        ny = max(int(round(radius_lu * 16.0)), 32)
        nz = ny

        # Lattice Reynolds number for this resolution
        nu_lu = (config.lbm_tau - 0.5) / 3.0
        re_lbm = config.lbm_u_in * length_lu / nu_lu
        # Analytical Cd at the lattice Re (laminar flat-plate friction only)
        cf_lbm = _laminar_friction_coefficient(re_lbm)
        cd_lbm_analytical = cf_lbm * form_factor

        mask, stats = build_suboff_mask(
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length=length_lu,
            radius=radius_lu,
            config=config.geometry,
            device=config.device,
        )

        dx = config.length_m / length_lu
        wetted_voxel = _voxel_wetted_area(mask, dx)
        cd_sim, fx_lu, mesh_stats = _run_suboff_lbm_drag(
            config=config,
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length_lu=length_lu,
            radius_lu=radius_lu,
        )
        # Scale lattice force to physical force
        f_scale = _force_scale_factor(
            config.rho_kgm3, config.speed_ms, config.length_m,
            1.0, config.lbm_u_in, length_lu,
        )
        resistance_sim_n = fx_lu * f_scale
        error_pct: float | None = None
        if prev_cd is not None:
            richardson_cd = cd_sim + (cd_sim - prev_cd) / (refinement_ratio**order - 1.0)
            error_pct = abs(cd_sim - richardson_cd) / max(abs(richardson_cd), 1e-12) * 100.0

        best_cd = cd_sim
        best_resistance_n = resistance_sim_n
        best_cd_lbm_re_analytical = cd_lbm_analytical
        if error_pct is not None:
            final_error = error_pct
        iterations.append(
            {
                "iteration": k,
                "grid": {"nx": nx, "ny": ny, "nz": nz},
                "length_lu": length_lu,
                "radius_lu": radius_lu,
                "cell_size_m": dx,
                "solid_fraction": (
                    float(stats["solid_cells"]) / max(float(stats["total_cells"]), 1.0)
                ),
                "wetted_area_m2": wetted_voxel,
                "re_lbm": re_lbm,
                "cd": cd_sim,
                "cd_lbm_analytical": cd_lbm_analytical,
                "resistance_n": resistance_sim_n,
                "drag_lu": fx_lu,
                "lbm": {
                    "u_in": config.lbm_u_in,
                    "tau": config.lbm_tau,
                    "steps": config.lbm_steps,
                },
                "mesh": mesh_stats,
                "runtime_evidence": mesh_stats["runtime_evidence"],
                "cd_richardson": richardson_cd if prev_cd is not None else None,
                "error_pct": error_pct,
            }
        )
        prev_cd = cd_sim
        # A spatial convergence assertion needs three independently executed
        # grids: coarse/fine differences alone have no observed order or trend.
        if (k >= 3 and error_pct is not None
                and error_pct <= config.target_error_pct):
            break

    return {
        "name": "suboff_resistance",
        "hull_type": hull_type.value,
        "target_error_pct": config.target_error_pct,
        "final_error_pct": final_error,
        "target_met": final_error <= config.target_error_pct,
        "reynolds": reynolds,
        "cf_ittc57": cf,
        "reference": {
            "wetted_area_m2": ref_area,
            "cd_analytical": cd_analytical,
            "resistance_analytical_n": resistance_analytical_n,
            "cd_richardson": richardson_cd if math.isfinite(richardson_cd) else None,
            "re_lbm_analytical_cd": best_cd_lbm_re_analytical if math.isfinite(best_cd_lbm_re_analytical) else None,
        },
        "simulated": {
            "cd": best_cd,
            "resistance_n": best_resistance_n,
        },
        "adaptive_mesh": {
            "enabled": bool(config.use_adaptive_mesh),
            "active_cells_mean": (
                float(sum(float(i["mesh"]["active_cells"]) for i in iterations))
                / max(float(len(iterations)), 1.0)
            ),
            "finest_uniform_cells_mean": (
                float(sum(float(i["mesh"]["finest_uniform_cells"]) for i in iterations))
                / max(float(len(iterations)), 1.0)
            ),
            "cell_saving_pct_mean": (
                float(sum(float(i["mesh"]["cell_saving_pct"]) for i in iterations))
                / max(float(len(iterations)), 1.0)
            ),
        },
        "iterations": iterations,
    }


def _refinement_level_observation(iteration: dict[str, object], index: int) -> dict[str, object]:
    """Canonical per-level solver evidence, suitable for hash binding."""
    runtime = iteration.get("runtime_evidence")
    grid = iteration.get("grid")
    if not isinstance(runtime, dict) or not isinstance(grid, dict):
        raise RuntimeError("SUBOFF refinement level has no runtime/grid evidence")
    required = ("requested_steps", "completed_steps", "finite_population_checks",
                "finite_density_checks", "density_min", "density_max")
    values = {name: runtime.get(name) for name in required}
    numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(float(value)) for value in values.values())
    finite_pass = (numeric
                   and values["completed_steps"] == values["requested_steps"]
                   and values["finite_population_checks"] == values["completed_steps"]
                   and values["finite_density_checks"] == values["completed_steps"]
                   and runtime.get("all_populations_finite") is True
                   and runtime.get("all_densities_finite") is True
                   and float(values["density_min"]) > 0.0
                   and float(values["density_min"]) <= float(values["density_max"]))
    coefficient = iteration.get("cd")
    record: dict[str, object] = {
        "level": index + 1,
        "grid": dict(grid),
        "cell_size_m": iteration.get("cell_size_m"),
        "coefficient": coefficient,
        "completion": {"requested_steps": values["requested_steps"],
                       "completed_steps": values["completed_steps"],
                       "pass": numeric and values["completed_steps"] == values["requested_steps"]},
        "finite": {"pass": finite_pass,
                   "population_checks": values["finite_population_checks"],
                   "density_checks": values["finite_density_checks"],
                   "density_min": values["density_min"], "density_max": values["density_max"]},
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    record["evidence_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return record


def run_suboff_resistance_runtime(
    config: SuboffResistanceBenchmarkConfig,
) -> dict[str, object]:
    """Execute the local SUBOFF runner and retain only directly observed facts.

    A normal return or a finite drag is not validation evidence.  Downstream
    artifact/gate code must keep unobserved preflight, numerical, conservation,
    and physics assertions withheld.
    """
    result = run_suboff_resistance_benchmark(config)
    simulated = result.get("simulated")
    iterations = result.get("iterations")
    coefficient = simulated.get("cd") if isinstance(simulated, dict) else None
    if not isinstance(iterations, list) or not iterations or not all(isinstance(item, dict) for item in iterations):
        raise RuntimeError("SUBOFF runner returned no iteration evidence")
    typed_iterations = [item for item in iterations if isinstance(item, dict)]
    refinement_levels = [_refinement_level_observation(item, index)
                         for index, item in enumerate(typed_iterations)]
    final_iteration = typed_iterations[-1]
    runtime_evidence = final_iteration.get("runtime_evidence")
    if not isinstance(runtime_evidence, dict):
        raise RuntimeError("SUBOFF runner returned no runtime numerical evidence")
    if (not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool)
            or not math.isfinite(float(coefficient))):
        raise RuntimeError("SUBOFF runner returned no finite measured resistance coefficient")
    requested_steps = runtime_evidence.get("requested_steps")
    completed_steps = runtime_evidence.get("completed_steps")
    population_checks = runtime_evidence.get("finite_population_checks")
    density_checks = runtime_evidence.get("finite_density_checks")
    density_min = runtime_evidence.get("density_min")
    density_max = runtime_evidence.get("density_max")
    initial_mass = runtime_evidence.get("initial_lattice_mass")
    final_mass = runtime_evidence.get("final_lattice_mass")
    max_abs_mass_drift = runtime_evidence.get("max_abs_mass_drift")
    max_relative_mass_drift = runtime_evidence.get("max_relative_mass_drift")
    initial_momentum = runtime_evidence.get("initial_lattice_momentum_norm")
    final_momentum = runtime_evidence.get("final_lattice_momentum_norm")
    max_abs_momentum_drift = runtime_evidence.get("max_abs_momentum_drift")
    max_relative_momentum_drift = runtime_evidence.get("max_relative_momentum_drift")
    mass_sample_count = runtime_evidence.get("mass_sample_count")
    sampled_step_count = runtime_evidence.get("sampled_step_count")
    momentum_budget = runtime_evidence.get("momentum_budget")
    grid = final_iteration.get("grid")
    numerical_fields = (requested_steps, completed_steps, population_checks, density_checks, density_min, density_max)
    evidence_is_numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                              and math.isfinite(float(value)) for value in numerical_fields)
    final_level_numerics_pass = (evidence_is_numeric and requested_steps == config.lbm_steps
                     and completed_steps == requested_steps and population_checks == completed_steps
                     and density_checks == completed_steps
                     and runtime_evidence.get("all_populations_finite") is True
                     and runtime_evidence.get("all_densities_finite") is True
                     and float(density_min) > 0.0 and float(density_min) <= float(density_max))
    coefficient_changes_pct: list[float] = []
    coefficients: list[float] = []
    for level in refinement_levels:
        value = level.get("coefficient")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))):
            coefficients.append(float(value))
        else:
            coefficients = []
            break
    if len(coefficients) == len(refinement_levels):
        for coarse, fine in zip(coefficients, coefficients[1:]):
            coefficient_changes_pct.append(
                abs(fine - coarse) / max(abs(fine), 1.0e-30) * 100.0
            )
    coefficient_change_pct = coefficient_changes_pct[-1] if coefficient_changes_pct else None
    observed_order: float | None = None
    if len(coefficients) >= 3:
        coarse_delta = abs(coefficients[-2] - coefficients[-3])
        fine_delta = abs(coefficients[-1] - coefficients[-2])
        if coarse_delta > 0.0 and fine_delta > 0.0:
            observed_order = math.log(coarse_delta / fine_delta) / math.log(2.0)
    monotonic: bool | None = None
    if len(coefficients) >= 3:
        deltas = [right - left for left, right in zip(coefficients, coefficients[1:])]
        monotonic = all(delta >= 0.0 for delta in deltas) or all(delta <= 0.0 for delta in deltas)
    levels_complete_finite = all(
        level["completion"]["pass"] is True and level["finite"]["pass"] is True
        for level in refinement_levels
    )
    convergence_pass = (len(refinement_levels) >= 3 and levels_complete_finite
                        and coefficient_change_pct is not None and math.isfinite(coefficient_change_pct)
                        and coefficient_change_pct <= config.numerics_max_coefficient_change_pct)
    numerics_pass = final_level_numerics_pass and convergence_pass
    domain_pass = (isinstance(grid, dict) and all(isinstance(grid.get(axis), int) and grid[axis] > 0
                  for axis in ("nx", "ny", "nz")))
    lattice_mach = config.lbm_u_in / math.sqrt(1.0 / 3.0)
    mach_pass = math.isfinite(lattice_mach) and lattice_mach < 0.25
    preflight_checks = {
        "config": {"pass": True, "lbm_tau": config.lbm_tau, "lbm_u_in": config.lbm_u_in},
        "domain": {"pass": domain_pass, "grid": grid},
        "mach": {"pass": mach_pass, "lattice_mach": lattice_mach, "limit": 0.25},
    }
    if not final_level_numerics_pass:
        raise RuntimeError("SUBOFF runner returned invalid runtime numerical evidence")
    if not domain_pass or not mach_pass:
        raise RuntimeError("SUBOFF runner failed runtime preflight")
    conservation_fields = (
        initial_mass, final_mass, max_abs_mass_drift, max_relative_mass_drift,
        initial_momentum, final_momentum, max_abs_momentum_drift,
        max_relative_momentum_drift,
    )
    conservation_is_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in conservation_fields
    )
    conservation_sampled = (
        isinstance(mass_sample_count, int) and not isinstance(mass_sample_count, bool)
        and isinstance(sampled_step_count, int) and not isinstance(sampled_step_count, bool)
        and mass_sample_count == completed_steps + 1 and sampled_step_count == completed_steps
    )
    conservation_pass = (
        conservation_is_numeric and conservation_sampled and float(initial_mass) > 0.0
        and float(final_mass) > 0.0 and float(max_abs_mass_drift) >= 0.0
        and float(max_relative_mass_drift) >= 0.0 and float(max_abs_momentum_drift) >= 0.0
        and float(max_relative_momentum_drift) >= 0.0
        and float(max_relative_mass_drift) <= config.conservation_max_relative_mass_drift
        and float(max_relative_momentum_drift) <= config.conservation_max_relative_momentum_drift
    )
    mass_pass = (conservation_is_numeric
                 and float(max_relative_mass_drift) <= config.conservation_max_relative_mass_drift)
    momentum_pass = (conservation_is_numeric
                     and float(max_relative_momentum_drift) <= config.conservation_max_relative_momentum_drift)
    mass_normalized = (float(max_relative_mass_drift) / max(config.conservation_max_relative_mass_drift, 1.0e-30)
                       if conservation_is_numeric else None)
    momentum_normalized = (float(max_relative_momentum_drift) / max(config.conservation_max_relative_momentum_drift, 1.0e-30)
                           if conservation_is_numeric else None)
    if mass_normalized is None or momentum_normalized is None:
        dominant_channel, attribution_reason = "withheld", "non_finite_conservation_observation"
    elif math.isclose(mass_normalized, momentum_normalized, rel_tol=1.0e-12, abs_tol=0.0):
        dominant_channel, attribution_reason = "balanced", "equal_normalized_bound_utilization"
    elif mass_normalized > momentum_normalized:
        dominant_channel = "mass"
        attribution_reason = "mass_bound_exceeded" if not mass_pass else "mass_is_larger_normalized_drift"
    else:
        dominant_channel = "momentum"
        attribution_reason = "momentum_bound_exceeded" if not momentum_pass else "momentum_is_larger_normalized_drift"
    budget_diagnostic = (isinstance(momentum_budget, dict)
                         and momentum_budget.get("status") == "measured"
                         and isinstance(momentum_budget.get("samples"), list))
    budget_observed = (budget_diagnostic
                       and momentum_budget.get("coverage") == "full_per_step"
                       and len(momentum_budget["samples"]) == completed_steps)
    operator_attribution: dict[str, object]
    if not budget_diagnostic:
        operator_attribution = {"status": "withheld", "reason": "operator_budget_disabled"}
    else:
        cumulative = momentum_budget.get("cumulative_sampled")
        if not isinstance(cumulative, dict):
            operator_attribution = {"status": "withheld", "reason": "momentum_budget_cumulative_unavailable"}
        else:
            channels = ("collision", "inlet_boundary", "outlet_boundary", "wall_exchange", "solid_exchange")
            norms: dict[str, float] = {}
            for channel in channels:
                value = cumulative.get(channel)
                if (not isinstance(value, list) or len(value) != 3
                        or not all(isinstance(component, (int, float)) and math.isfinite(float(component))
                                   for component in value)):
                    norms = {}
                    break
                norms[channel] = math.sqrt(sum(float(component) ** 2 for component in value))
            if not norms:
                operator_attribution = {"status": "withheld", "reason": "non_finite_operator_budget"}
            else:
                dominant_operator = max(norms, key=lambda channel: norms[channel])
                operator_attribution = {
                    "status": "measured" if budget_observed else "sampled",
                    "dominant_operator": dominant_operator,
                    "reason": ("largest_cumulative_operator_momentum_norm"
                               if budget_observed else "largest_sampled_operator_momentum_norm"),
                    "coverage": momentum_budget.get("coverage"),
                    "cumulative_norms": norms,
                    "boundary_flux": momentum_budget.get("boundary_flux"),
                    "body_force": momentum_budget.get("body_force"),
                }
    return {
        "schema": "suboff-resistance-runtime-observation-v1",
        "case": "suboff_runtime",
        "runner": "tensorlbm.suboff_resistance.run_suboff_resistance_benchmark",
        "completion": {
            "state": "COMPLETED",
            "requested_steps": int(requested_steps),
            "completed_steps": int(completed_steps),
            "evidence": "per_step_runtime_observation",
        },
        "resistance": {
            "coefficient": float(coefficient),
            "basis": "runner_simulated_cd",
            "status": "measured",
        },
        "preflight": {"status": "measured", "pass": all(check["pass"] for check in preflight_checks.values()),
                      "checks": preflight_checks},
        "numerics": {"status": "measured", "pass": numerics_pass,
                     "requested_steps": int(requested_steps), "completed_steps": int(completed_steps),
                     "finite_population_checks": int(population_checks), "finite_density_checks": int(density_checks),
                     "all_populations_finite": runtime_evidence["all_populations_finite"],
                     "all_densities_finite": runtime_evidence["all_densities_finite"],
                     "density_min": float(density_min), "density_max": float(density_max),
                     "refinement_kind": "grid", "required_levels": 3,
                     "refinement_levels": refinement_levels,
                     "coefficient_change_pct": coefficient_change_pct,
                     "coefficient_changes_pct": coefficient_changes_pct,
                     "observed_order": observed_order,
                     "monotonicity": {"status": "measured", "pass": monotonic,
                                      "coefficient_sequence": coefficients},
                     "convergence": {
                        "pass": convergence_pass,
                        "criterion": "three_or_more_complete_finite_grid_levels_and_final_coefficient_change_pct_at_or_below_limit",
                        "coefficient_change_pct_limit": config.numerics_max_coefficient_change_pct,
                         "observed_levels": len(refinement_levels),
                         "levels_complete_finite": levels_complete_finite,
                     }},
        "conservation": {
            "status": "measured",
            "pass": conservation_pass,
            "initial_lattice_mass": float(initial_mass) if conservation_is_numeric else None,
            "final_lattice_mass": float(final_mass) if conservation_is_numeric else None,
            "max_abs_mass_drift": float(max_abs_mass_drift) if conservation_is_numeric else None,
            "max_relative_mass_drift": float(max_relative_mass_drift) if conservation_is_numeric else None,
            "initial_lattice_momentum_norm": float(initial_momentum) if conservation_is_numeric else None,
            "final_lattice_momentum_norm": float(final_momentum) if conservation_is_numeric else None,
            "max_abs_momentum_drift": float(max_abs_momentum_drift) if conservation_is_numeric else None,
            "max_relative_momentum_drift": float(max_relative_momentum_drift) if conservation_is_numeric else None,
            "mass_sample_count": mass_sample_count,
            "sampled_step_count": sampled_step_count,
            "max_relative_mass_drift_limit": config.conservation_max_relative_mass_drift,
            "max_relative_momentum_drift_limit": config.conservation_max_relative_momentum_drift,
            "source_attribution": {
                "status": "measured" if conservation_is_numeric else "withheld",
                "dominant_channel": dominant_channel,
                "reason": attribution_reason,
                "mass": {"max_relative_drift": float(max_relative_mass_drift) if conservation_is_numeric else None,
                         "limit": config.conservation_max_relative_mass_drift, "pass": mass_pass,
                         "normalized_bound_utilization": mass_normalized},
                "momentum": {"max_relative_drift": float(max_relative_momentum_drift) if conservation_is_numeric else None,
                             "limit": config.conservation_max_relative_momentum_drift, "pass": momentum_pass,
                             "normalized_bound_utilization": momentum_normalized,
                             "operator_budget": momentum_budget if budget_diagnostic else {
                                 "status": "disabled", "reason": "operator_budget_disabled"},
                             "per_step_budget": momentum_budget if budget_observed else None,
                             "operator_attribution": operator_attribution},
            },
        },
        "physics": {"status": "withheld", "pass": False,
                    "reason": "no_independent_physics_validation"},
    }


def main() -> int:
    """Run a small, real three-grid SUBOFF diagnostic and print its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-length-lu", type=float, default=20.0)
    parser.add_argument("--max-length-lu", type=float, default=80.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--sample-interval", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = SuboffResistanceBenchmarkConfig(
        base_length_lu=args.base_length_lu, max_length_lu=args.max_length_lu,
        max_iterations=3, lbm_steps=args.steps, lbm_warmup_steps=args.warmup_steps,
        lbm_sample_interval=args.sample_interval, device=args.device,
    )
    print(json.dumps(run_suboff_resistance_runtime(config), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
