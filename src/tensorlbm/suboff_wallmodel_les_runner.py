"""SUBOFF high-Re wall-model LES runner (flat-plate-validated architecture).

Ports the two-viscosity BFL wall-stress architecture validated by
:mod:`tensorlbm.flat_plate_wall_model` to the SUBOFF hull (AFF-8 target):

* **collision viscosity** ``collision_nu = u·L/resolved_reynolds`` — finite
  ``tau = 0.5 + 3·collision_nu`` so near-wall momentum is replenished
  (the collapse mechanism of the ``wall_function_common`` path at
  ``tau ≈ 0.5`` is removed by construction);
* **Smagorinsky SGS** ``C_s = 0.05`` inside ``collide_cumulant_d3q19``;
* **BFL halfway links + wall-function Guo force** on the SUBOFF mask
  (``bfl_wall_function_3d``, wall-model-slip mode, cosine ramp);
* **positivity limiter** on collided and final populations;
* **non-equilibrium far-field** outer BC + **sponge** on outflow/lateral
  faces;
* **control-volume force observer** as the primary drag estimator,
  cross-checked against wall-traction + BFL link force (flat-plate
  observer-agreement contract).

Drag normalisation uses the **analytic bare-hull wetted area** (surface of
revolution of the SUBOFF radius profile), matching the experimental
reference-area convention locked in ``refs_lock`` (WSA = 5.998 m² vs CAD
analytic 5.9933 m², 0.08%); appendage own area is excluded for FULL hulls
exactly as in the experiment.

Loop per step (identical operator order to the flat-plate benchmark)::

    collide(cumulant, C_s) → [positivity] → keep solid uncollided
    → stream → outer far-field
    → bfl_wall_function_3d(wall_nu, ramp) → [positivity]
    → sponge → outer far-field → CV observer

This runner composes existing operators only; no solver hot path changes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .boundaries3d import bounce_back_cells_3d
from .checkpoint_io import atomic_torch_save
from .collision_d3q19_advanced import collide_bgk_d3q19
from .control_volume_force import box_control_volume, observe_control_volume_force
from .cumulant import collide_cumulant_d3q19
from .d3q19 import C, equilibrium3d, macroscopic3d
from .drag_pressure import SurfaceMesh, get_near_wall_3d
from .external_open_boundary import non_equilibrium_far_field_bc_3d
from .flat_plate_wall_model import _ramp
from .force_convergence import assess_force_stationarity
from .population_positivity import limit_nonequilibrium_for_positivity
from .solver3d import collide_mrt3d, stream3d
from .sponge_layer import apply_equilibrium_difference_sponge, build_sponge_sigma_3d
from .suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    suboff_radius_profile,
)
from .turbulence import collide_smagorinsky_bgk3d
from .wall_model import WALL_TRACTION_SOURCE_SCHEME, bfl_wall_function_3d

try:
    from .suboff_wallfn_fullgrid_runner import _hybrid_mesh
except ImportError:  # pragma: no cover - campaign tree always provides it
    _hybrid_mesh = None  # type: ignore[assignment]

__all__ = [
    "SuboffWallModelLesConfig",
    "analytic_bare_hull_area",
    "run_suboff_wallmodel_les",
    "write_artifact",
]


HULL_TYPES: tuple[str, ...] = ("BARE_HULL", "WITH_SAIL", "FULL")
MESH_SOURCES: tuple[str, ...] = ("suboff", "gradient", "hybrid")


def analytic_bare_hull_area(length: float, samples: int = 200_001) -> float:
    """Wetted area of the axisymmetric bare hull by profile integration.

    Surface of revolution:  A = ∫ 2π r(x) √(1 + r'(x)²) dx  over the hull
    axial extent.  At ``length = 240`` this yields 18187.9… lu²
    (= 0.31576·L²), the quantity cross-checked against the experimental
    WSA = 5.998 m² in ``refs_lock`` (0.08% agreement).
    """
    xi = np.linspace(0.0, 1.0, samples)
    max_radius = SuboffConfig().r_over_l * length
    r = np.asarray(suboff_radius_profile(xi), dtype=np.float64) * max_radius
    dr_dxi = np.gradient(r, xi)
    integrand = 2.0 * np.pi * r * np.sqrt(1.0 + (dr_dxi / length) ** 2)
    x = xi * length
    return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x)))


@dataclass(frozen=True)
class SuboffWallModelLesConfig:
    """One SUBOFF wall-model LES run at a target (physical) Reynolds.

    Defaults: AFF-8 FULL hull on 480×240×240 at the first locked anchor
    Re = 1.166e7 (V = 3.050 m/s).
    """

    # geometry / grid
    nx: int = 400
    ny: int = 200
    nz: int = 200
    hull_length: float = 200.0
    hull_type: str = "FULL"
    mesh_source: str = "hybrid"
    # physics ladder (target physical Re; collision runs at resolved Re)
    reynolds: float = 1.166e7
    resolved_reynolds: float = 1.0e5
    lattice_speed: float = 0.06
    # wall model
    wall_law: str = "reichardt"
    wall_mode: str = "wall_model_slip"
    ramp_steps: int = 1000
    stress_exchange_distance: float | None = None
    # diagnostic add-on: dump final-step macroscopic fields (CPU) for
    # wall-normal profile analysis; JSON artifact stays clean
    final_field_dump_path: str | None = None
    smagorinsky_cs: float = 0.05
    positivity_limiter: bool = True
    y_plus_lower_bound: float = 30.0
    y_plus_upper_bound: float = 5000.0
    # Wall-model internals: LIBRARY DEFAULTS (chunk=19, full macroscopic) —
    # RETRACTION 2026-10-10: an earlier probe pair (p8 defaults-small-grid
    # clean vs p11/p12 knobs-big-grid oscillating) led to the (wrong)
    # conviction that these memory-saving knobs cause the period-2 force
    # oscillation.  p13/p14 (library defaults, same big grid) oscillate
    # identically, so the knobs are exonerated; the mode is grid/time-scale
    # dependent.  Defaults kept = flat-plate-validated configuration.
    guo_forcing: bool = True
    guo_direction_chunk_size: int = 19
    low_memory_macroscopic: bool = False
    # Thick-solid interior convention (p16 A/B 2026-10-10):
    #   "marine_bounce" — collide everywhere, then bounce_back_cells_3d on
    #     solid cells after the wall function (validated marine convention;
    #     seals the interior so populations cannot free-fly through the
    #     thick hull).  Kills the period-2 momentum mode: probe p16 V3 decay
    #     5.90->3.88 raw units over 1000 steps vs flat-plate leak variant.
    #   "flatplate_where" — original flat-plate port (restore uncollided
    #     populations at solid cells, no interior seal).  RETAINED for A/B;
    #     develops the period-2 mode on thick bodies (p13/p14).
    solid_handling: str = "marine_bounce"
    # Collision kernel (p22 isolation 2026-10-10): the cumulant basis sustains
    # a global period-2 kinetic mode at tau~=0.5004 in this forced setup
    # (cumulant+C_s=0.05 AND C_s=0 both oscillate; BGK at the same tau kills
    # the mode within ~10 steps of a resume).  Operator toggles exonerated:
    # wall model on/off, Guo on/off, sponge on/off, single/double outer,
    # positivity on/off, solid handling both conventions.
    #   "bgk_smag" -> collide_smagorinsky_bgk3d (default: BGK relaxation
    #     kills the period-2 mode, Smagorinsky supplies the missing SGS)
    #   "mrt" -> collide_mrt3d (kills mode only via 20%+ positivity clipping
    #     at tau~=0.5 — physically corrupt, rejected p23/p24)
    #   "cumulant" -> collide_cumulant_d3q19 (oscillates here; kept for A/B)
    #   "bgk" -> collide_bgk_d3q19 (diagnostic; no SGS at this tau)
    collision_kernel: str = "bgk_smag"
    # numerics
    sponge_width: int = 24
    sponge_strength: float = 0.2
    cv_margin: int = 10
    # run control
    steps: int = 12000
    warmup_steps: int = 6000
    report_interval: int = 500
    wall_diagnostic_interval: int = 50
    checkpoint_interval: int = 2000
    checkpoint_path: str | None = None
    resume: bool = False
    # reference (locked ladder; informative at probe, gated at prereg)
    ct_reference: float = 0.003667
    reference_source: str = "refs_lock EFD anchor 1 (CFD Letters 2024 Table 3)"
    device: str = "cuda:5"

    def validate(self) -> None:
        if self.hull_type.upper() not in HULL_TYPES:
            raise ValueError(f"hull_type must be one of {HULL_TYPES}")
        if self.mesh_source not in MESH_SOURCES:
            raise ValueError(f"mesh_source must be one of {MESH_SOURCES}")
        if self.hull_type.upper() != "BARE_HULL" and self.mesh_source == "suboff":
            raise ValueError(
                "mesh_source='suboff' is only valid for hull_type='BARE_HULL'; "
                "use 'hybrid' or 'gradient' for appendage configurations"
            )
        if self.wall_law not in {"log", "reichardt", "musker"}:
            raise ValueError("unsupported wall law")
        if self.wall_mode not in {"stationary", "wall_model_slip", "spalding_exchange"}:
            raise ValueError("unsupported wall mode")
        if min(self.nx, self.ny, self.nz) < 16:
            raise ValueError("grid too small")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("invalid averaging window")
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        if self.stress_exchange_distance is not None and (self.stress_exchange_distance <= 0.0):
            raise ValueError("stress_exchange_distance must be positive")
        if not 0.0 <= self.smagorinsky_cs < 0.5:
            raise ValueError("smagorinsky_cs must lie in [0,0.5)")
        if not 0.0 < self.lattice_speed < 0.15:
            raise ValueError("lattice_speed must lie in (0,0.15)")
        if self.reynolds <= 0.0 or self.resolved_reynolds <= 0.0:
            raise ValueError("Reynolds numbers must be positive")
        if self.hull_length <= 0.0:
            raise ValueError("hull_length must be positive")
        if self.y_plus_lower_bound <= 0.0 or (self.y_plus_upper_bound <= self.y_plus_lower_bound):
            raise ValueError("invalid y+ applicability band")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")

    @property
    def wall_nu(self) -> float:
        """Wall-law viscosity at the physical Reynolds number."""
        return self.lattice_speed * self.hull_length / self.reynolds

    @property
    def collision_nu(self) -> float:
        """Collision viscosity at the resolved Reynolds number."""
        return self.lattice_speed * self.hull_length / self.resolved_reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.collision_nu


def _halfway_links(solid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BFL boundary-link mask and halfway q-field (q = 0.5) for Q19."""
    q_count = 19
    masks = torch.zeros((q_count, *solid.shape), dtype=torch.bool, device=solid.device)
    q_field = torch.full(masks.shape, 0.5, dtype=torch.float32, device=solid.device)
    for direction in range(1, q_count):
        cx, cy, cz = (int(value) for value in C[direction].tolist())
        neighbour = torch.roll(solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
        masks[direction] = ~solid & neighbour
    return masks, q_field


def _build_normals(
    config: SuboffWallModelLesConfig,
    solid: torch.Tensor,
    near: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
) -> SurfaceMesh:
    """Surface normals per mesh_source (outward from body into fluid)."""
    max_radius = SuboffConfig().r_over_l * config.hull_length
    if config.mesh_source == "gradient":
        mesh = SurfaceMesh.from_gradient(solid, near)
    else:
        mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, config.hull_length, max_radius)
        if config.mesh_source == "hybrid":
            if _hybrid_mesh is None:
                raise RuntimeError("_hybrid_mesh unavailable")
            mesh = _hybrid_mesh(mesh, solid, near, cx, cy, cz, config.hull_length, max_radius)
    return mesh


def run_suboff_wallmodel_les(
    config: SuboffWallModelLesConfig | None = None,
) -> dict[str, Any]:
    """Run SUBOFF wall-model LES and return the machine artifact dict."""
    if config is None:
        config = SuboffWallModelLesConfig()
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # --- geometry (CPU build, then move) ---
    cx = config.nx * 0.35
    cy = config.ny / 2.0
    cz = config.nz / 2.0
    solid, _stats = build_suboff_mask(
        hull_type=SuboffHullType[config.hull_type.upper()],
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=config.hull_length,
        device="cpu",
    )
    solid = solid.to(device)
    near = get_near_wall_3d(solid)
    mesh = _build_normals(config, solid, near, cx, cy, cz)
    normal_x = mesh.nx_n.to(device)
    normal_y = mesh.ny_n.to(device)
    normal_z = mesh.nz_n.to(device)
    near = near.to(device)

    bfl_mask, bfl_q = _halfway_links(solid)
    solid_q = solid.unsqueeze(0).expand((19, *solid.shape))

    shape = tuple(solid.shape)
    s_ref = analytic_bare_hull_area(config.hull_length)
    denominator = 0.5 * config.lattice_speed**2 * s_ref

    # --- control volume: hull bounding box + margin, strictly interior ---
    xs, ys, zs = torch.nonzero(solid, as_tuple=True)
    cv = box_control_volume(
        shape,
        x0=max(1, int(xs.min().item()) - config.cv_margin),
        x1=min(config.nx - 2, int(xs.max().item()) + config.cv_margin),
        y0=max(1, int(ys.min().item()) - config.cv_margin),
        y1=min(config.ny - 2, int(ys.max().item()) + config.cv_margin),
        z0=max(1, int(zs.min().item()) - config.cv_margin),
        z1=min(config.nz - 2, int(zs.max().item()) + config.cv_margin),
        device=device,
    )

    sigma = build_sponge_sigma_3d(
        shape,
        width=config.sponge_width,
        max_strength=config.sponge_strength,
        device=device,
        faces=("x+", "y-", "y+", "z-", "z+"),
    )

    # --- initial populations: uniform stream, zero inside solid ---
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, config.lattice_speed)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    ux[solid] = 0.0
    f = equilibrium3d(rho, ux, uy, uz, device=device)
    del rho, ux, uy, uz

    def outer(state: torch.Tensor) -> torch.Tensor:
        return non_equilibrium_far_field_bc_3d(
            state,
            u_in=config.lattice_speed,
            faces=("x-", "x+", "y-", "y+", "z-", "z+"),
        )

    friction_history: list[float] = []
    cv_history: list[float] = []
    bfl_total_history: list[float] = []
    ct_time_series: list[dict[str, float]] = []
    wall_y_plus_min_history: list[float] = []
    wall_y_plus_mean_history: list[float] = []
    wall_y_plus_max_history: list[float] = []
    wall_rejected_fraction_history: list[float] = []
    maximum_limited_fraction = 0.0
    start_step = 0
    diverged_at: int | None = None

    checkpoint = Path(config.checkpoint_path) if config.checkpoint_path else None
    if config.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("schema") != "tensorlbm-suboff-wallmodel-checkpoint-v1":
            raise ValueError("checkpoint schema mismatch")
        if state.get("configuration") != _checkpoint_configuration(config, shape):
            raise ValueError("checkpoint configuration does not match run")
        f = state["populations"].to(device=device)
        start_step = int(state["step"])
        friction_history = state["friction_history"].tolist()
        cv_history = state["control_volume_history"].tolist()
        bfl_total_history = state["bfl_total_history"].tolist()
        wall_y_plus_min_history = state["wall_y_plus_min_history"].tolist()
        wall_y_plus_mean_history = state["wall_y_plus_mean_history"].tolist()
        wall_y_plus_max_history = state["wall_y_plus_max_history"].tolist()
        wall_rejected_fraction_history = state["wall_rejected_fraction_history"].tolist()
        maximum_limited_fraction = float(state["maximum_limited_fraction"])
        if start_step >= config.steps:
            raise ValueError("checkpoint already reached requested steps")

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {
                "schema": "tensorlbm-suboff-wallmodel-checkpoint-v1",
                "configuration": _checkpoint_configuration(config, shape),
                "step": step,
                "populations": f.detach().cpu(),
                "friction_history": torch.tensor(friction_history, dtype=torch.float64),
                "control_volume_history": torch.tensor(cv_history, dtype=torch.float64),
                "bfl_total_history": torch.tensor(bfl_total_history, dtype=torch.float64),
                "wall_y_plus_min_history": torch.tensor(
                    wall_y_plus_min_history, dtype=torch.float64
                ),
                "wall_y_plus_mean_history": torch.tensor(
                    wall_y_plus_mean_history, dtype=torch.float64
                ),
                "wall_y_plus_max_history": torch.tensor(
                    wall_y_plus_max_history, dtype=torch.float64
                ),
                "wall_rejected_fraction_history": torch.tensor(
                    wall_rejected_fraction_history, dtype=torch.float64
                ),
                "maximum_limited_fraction": maximum_limited_fraction,
            },
            checkpoint,
        )

    for step in range(start_step + 1, config.steps + 1):
        old = f
        if config.collision_kernel == "cumulant":
            collided = collide_cumulant_d3q19(f, config.tau, C_s=config.smagorinsky_cs)
        elif config.collision_kernel == "bgk":
            collided = collide_bgk_d3q19(f, config.tau)
        elif config.collision_kernel == "bgk_smag":
            collided = collide_smagorinsky_bgk3d(f, config.tau, C_s=config.smagorinsky_cs)
        else:  # mrt
            collided = collide_mrt3d(f, config.tau)
        if config.positivity_limiter:
            collided, diagnostic = limit_nonequilibrium_for_positivity(collided)
            maximum_limited_fraction = max(maximum_limited_fraction, diagnostic.limited_fraction)
        if config.solid_handling == "flatplate_where":
            post = torch.where(solid_q, old, collided)
        else:  # marine_bounce: collide everywhere; interior sealed below
            post = collided
        del collided  # keep only (old, post, f) alive through the wall stage
        f = outer(stream3d(post))
        collect_wall_diagnostics = (
            step > config.warmup_steps and step % config.wall_diagnostic_interval == 0
        )
        wall_result = bfl_wall_function_3d(
            f,
            post,
            solid,
            config.wall_nu,
            bfl_mask,
            bfl_q,
            near_mask=near,
            wall_normals=(normal_x, normal_y, normal_z),
            bfl_wall_mode=config.wall_mode,
            wall_activation=_ramp(step, config.ramp_steps),
            wall_law=config.wall_law,
            stress_exchange_distance=config.stress_exchange_distance,
            return_wall_diagnostics=collect_wall_diagnostics,
            y_plus_lower_bound=config.y_plus_lower_bound,
            y_plus_upper_bound=config.y_plus_upper_bound,
            use_guo=config.guo_forcing,
            guo_direction_chunk_size=config.guo_direction_chunk_size,
            use_low_memory_macroscopic=config.low_memory_macroscopic,
            pressure_gradient_periodic_axes=(),
        )
        if collect_wall_diagnostics:
            f, friction, bfl_force, wall_diagnostics = wall_result
        else:
            f, friction, bfl_force = wall_result
        if config.solid_handling == "marine_bounce":
            f = bounce_back_cells_3d(f, solid)
        if config.positivity_limiter:
            f, diagnostic = limit_nonequilibrium_for_positivity(f)
            maximum_limited_fraction = max(maximum_limited_fraction, diagnostic.limited_fraction)
        f = apply_equilibrium_difference_sponge(
            f, sigma, velocity_target=(config.lattice_speed, 0.0, 0.0)
        )
        f = outer(f)
        cv_result = observe_control_volume_force(old, f, post, cv, solid=solid, periodic_axes=())
        cv_force = float(cv_result.force_on_body[0].item())
        cv_import = float(cv_result.momentum_import[0].item())
        cv_change = float(cv_result.fluid_momentum_change[0].item())
        if step > config.warmup_steps:
            friction_history.append(float(friction))
            cv_history.append(cv_force)
            bfl_total_history.append(float(friction) + float(bfl_force))
            ct_time_series.append(
                {
                    "step": step,
                    "ct_fric": float(friction) / denominator,
                    "ct_wall_total": (float(friction) + float(bfl_force)) / denominator,
                    "ct_cv": cv_force / denominator,
                    "ct_cv_import": cv_import / denominator,
                    "ct_cv_change": cv_change / denominator,
                }
            )
            if collect_wall_diagnostics:
                if wall_diagnostics.y_plus_mean is not None:
                    wall_y_plus_min_history.append(wall_diagnostics.y_plus_min)
                    wall_y_plus_mean_history.append(wall_diagnostics.y_plus_mean)
                    wall_y_plus_max_history.append(wall_diagnostics.y_plus_max)
                rejected = wall_diagnostics.rejected_fraction
                if rejected is not None:
                    wall_rejected_fraction_history.append(rejected)
        if not bool(torch.isfinite(f).all()):
            diverged_at = step
            break
        if config.report_interval and step % config.report_interval == 0:
            recent = ct_time_series[-min(len(ct_time_series), 10) :]
            mean_ct_cv = (
                sum(entry["ct_cv"] for entry in recent) / len(recent) if recent else math.nan
            )
            print(
                f"suboff_wm step={step}/{config.steps} "
                f"recent_Ct_cv={mean_ct_cv:.6f} "
                f"max_limited={maximum_limited_fraction:.3e}",
                flush=True,
            )
        if (
            checkpoint is not None
            and config.checkpoint_interval
            and step % config.checkpoint_interval == 0
        ):
            save_checkpoint(step)

    if checkpoint is not None and diverged_at is None:
        save_checkpoint(config.steps)

    if config.final_field_dump_path and diverged_at is None:
        rho, ux, uy, uz = macroscopic3d(f)
        atomic_torch_save(
            {
                "step": config.steps,
                "rho": rho.detach().to("cpu"),
                "u": torch.stack((ux, uy, uz)).detach().to("cpu"),
                "solid": solid.to("cpu"),
            },
            config.final_field_dump_path,
        )

    # --- summary (window: post-warmup samples) ---
    ct_fric = sum(e["ct_fric"] for e in ct_time_series) / max(len(ct_time_series), 1)
    ct_cv = sum(e["ct_cv"] for e in ct_time_series) / max(len(ct_time_series), 1)
    ct_wall = sum(e["ct_wall_total"] for e in ct_time_series) / max(len(ct_time_series), 1)
    cv_mean = sum(cv_history) / max(len(cv_history), 1)
    bfl_mean = sum(bfl_total_history) / max(len(bfl_total_history), 1)
    observer_difference = abs(cv_mean - bfl_mean) / max(abs(cv_mean), 1e-30) * 100.0
    stationarity = (
        assess_force_stationarity(
            [e["ct_cv"] for e in ct_time_series],
            block_size=max(1, len(ct_time_series) // 8),
        )
        if ct_time_series
        else None
    )
    reference_error = (
        abs(ct_cv - config.ct_reference) / config.ct_reference * 100.0
        if config.ct_reference > 0.0
        else None
    )
    maximum_rejected_fraction = max(wall_rejected_fraction_history, default=0.0)

    artifact: dict[str, Any] = {
        "schema": "tensorlbm.suboff-wallmodel-les/v1",
        "configuration": {
            "hull_type": config.hull_type.upper(),
            "mesh_source": config.mesh_source,
            "shape_zyx": list(shape),
            "hull_length": config.hull_length,
            "reynolds": config.reynolds,
            "resolved_reynolds": config.resolved_reynolds,
            "lattice_speed": config.lattice_speed,
            "wall_nu": config.wall_nu,
            "collision_nu": config.collision_nu,
            "tau": config.tau,
            "steps": config.steps,
            "warmup_steps": config.warmup_steps,
            "ramp_steps": config.ramp_steps,
            "wall_law": config.wall_law,
            "wall_mode": config.wall_mode,
            "smagorinsky_cs": config.smagorinsky_cs,
            "positivity_limiter": config.positivity_limiter,
            "y_plus_bounds": [
                config.y_plus_lower_bound,
                config.y_plus_upper_bound,
            ],
            "sponge_width": config.sponge_width,
            "sponge_strength": config.sponge_strength,
            "cv_margin": config.cv_margin,
            "stress_exchange_distance": config.stress_exchange_distance,
            "final_field_dump_path": config.final_field_dump_path,
            "guo_forcing": config.guo_forcing,
            "guo_direction_chunk_size": config.guo_direction_chunk_size,
            "low_memory_macroscopic": config.low_memory_macroscopic,
            "solid_handling": config.solid_handling,
            "collision_kernel": config.collision_kernel,
            "device": str(config.device),
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "resumed_from_step": start_step,
            "reference_area_convention": "analytic bare hull (refs_lock)",
        },
        "result": {
            "reference_area": s_ref,
            "Ct_fric": ct_fric,
            "Ct_cv": ct_cv,
            "Ct_wall_total": ct_wall,
            "ct_reference": config.ct_reference,
            "reference_error_pct": reference_error,
            "observer_difference_pct": observer_difference,
            "drag_stationarity": (stationarity.to_dict() if stationarity else None),
            "maximum_positivity_limited_fraction": maximum_limited_fraction,
            "wall_stress_applicability": {
                "samples": len(wall_y_plus_mean_history),
                "y_plus_min": (
                    min(
                        (v for v in wall_y_plus_min_history if v is not None),
                        default=None,
                    )
                ),
                "y_plus_mean": (
                    sum(v for v in wall_y_plus_mean_history if v is not None)
                    / max(
                        sum(1 for v in wall_y_plus_mean_history if v is not None),
                        1,
                    )
                    if any(v is not None for v in wall_y_plus_mean_history)
                    else None
                ),
                "y_plus_max": (
                    max(
                        (v for v in wall_y_plus_max_history if v is not None),
                        default=None,
                    )
                ),
                "maximum_rejected_fraction": maximum_rejected_fraction,
            },
            "finite": diverged_at is None,
            "diverged_at_step": diverged_at,
            "steps_completed": (config.steps if diverged_at is None else diverged_at - 1),
        },
        "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
        "ct_time_series": ct_time_series,
        "y_plus_history": {
            "min": wall_y_plus_min_history,
            "mean": wall_y_plus_mean_history,
            "max": wall_y_plus_max_history,
            "rejected_fraction": wall_rejected_fraction_history,
        },
    }
    return artifact


def _checkpoint_configuration(
    config: SuboffWallModelLesConfig, shape: tuple[int, int, int]
) -> dict[str, Any]:
    return {
        "shape_zyx": list(shape),
        "hull_type": config.hull_type.upper(),
        "mesh_source": config.mesh_source,
        "hull_length": config.hull_length,
        "reynolds": config.reynolds,
        "resolved_reynolds": config.resolved_reynolds,
        "lattice_speed": config.lattice_speed,
        "wall_law": config.wall_law,
        "wall_mode": config.wall_mode,
        "stress_exchange_distance": config.stress_exchange_distance,
        "final_field_dump_path": config.final_field_dump_path,
        "guo_forcing": config.guo_forcing,
        "guo_direction_chunk_size": config.guo_direction_chunk_size,
        "low_memory_macroscopic": config.low_memory_macroscopic,
        "solid_handling": config.solid_handling,
        "collision_kernel": config.collision_kernel,
        "ramp_steps": config.ramp_steps,
        "sponge_width": config.sponge_width,
        "sponge_strength": config.sponge_strength,
        "cv_margin": config.cv_margin,
        "smagorinsky_cs": config.smagorinsky_cs,
        "positivity_limiter": config.positivity_limiter,
        "y_plus_bounds": [
            config.y_plus_lower_bound,
            config.y_plus_upper_bound,
        ],
        "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
    }


def write_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Write the artifact as a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(artifact, sort_keys=True, indent=2),
        encoding="utf-8",
    )
