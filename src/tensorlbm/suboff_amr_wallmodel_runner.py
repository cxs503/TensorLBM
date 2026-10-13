"""SUBOFF high-Re wall-model LES runner on nested static block AMR.

Port of :mod:`tensorlbm.suboff_wallmodel_les_runner` (the uniform-grid
formal-tier architecture) to the conservative nested static-block runtime
(:class:`tensorlbm.static_block_amr.NestedStaticBlockAMR3D`):

* **per-level collision** ``bgk_smag`` (``collide_smagorinsky_bgk3d``,
  ``C_s = 0.05``) with the convective refinement tau chain
  ``tau_child = 0.5 + 2 (tau_parent - 0.5)``;
* **exact CAD regeneration on every fine level** (``build_fine_suboff_mask``
  / ``build_nested_fine_suboff_mask`` — never a voxel copy) with each
  level's own BFL halfway links, near-wall mask and hybrid surface normals
  (built by the shared :mod:`tensorlbm.wall_chain` module);
* **wall model where the hull surface is owned**: the finest level owns the
  hull (``wall_model_slip`` + ``reichardt``, wall viscosity at the physical
  Reynolds, cosine ramp).  Coarser levels whose hull is strictly interior
  to the child block are geometrically vacuous for wall treatment — the
  runner machine-counts the boundary links outside the child box and
  records the count instead of allocating dormant wall fields;
* **root layer** keeps the uniform-tier outer treatment: non-equilibrium
  far-field BC (before/after sponge) + equilibrium-difference sponge;
* **positivity limiter and marine_bounce interior seal on every level**;
* **control-volume force observer per level + composite**: the finest-level
  hull-hugging CV is the primary drag observer (same estimator family as
  the uniform tier), the root-level envelope CV is the independent
  cross-check, BFL wall totals are recorded from the active wall level;
* **period-2 hygiene** via :mod:`tensorlbm.kinetic_hygiene`: global
  x-momentum of the finest level per fine substep and of every level per
  root step (the campaign's convicted kinetic ghost mode probe; AMR
  interfaces are new injection surfaces).

Layout note (owner pivot 2026-10-11): the campaign refinement layout is
moving to concentric surface shells; this runner's per-level wall chain,
collision and observers are layout-agnostic by construction
(:class:`LevelContext` list + advance closure), and ``build_amr_contexts``
instantiates the executable box-based nested-block configuration — the A/B,
interface-validation and memory-pilot tier of Phase 0 — while the shell
layout line is adjudicated from ``amr_layout_highre.json``.

Common-module homes (owner directive 2026-10-11): wall chain
``tensorlbm/wall_chain.py``, period-2 probes ``tensorlbm/kinetic_hygiene.py``,
nested shell planning ``tensorlbm/amr_shell_planning.py``, union CV
``tensorlbm/control_volume_force.py`` (union_box_control_volume), analytic
wetted area ``tensorlbm/suboff_cad.py``.  This runner only orchestrates.

Level-split multi-GPU execution is deliberately NOT implemented (Phase 0
directive): ``device_per_level`` is accepted but must name one single
device for all levels.

The composition is factored so Phase 0 harnesses drive the exact runner
operator chain: :func:`build_amr_contexts` builds geometry/hierarchy/
contexts, :func:`make_advance` returns the runtime advance closure, and
:func:`run_suboff_amr_wallmodel` drives the production loop.  A driver
that calls :func:`make_advance` directly owns ``RunDiagnostics.step`` and
the per-step clearing of the ``finest_*`` buffers.  The production loop
additionally takes a ``pre_restrict_hook`` (default ``None`` = legacy
loop) forwarding the engine's pre-restriction stage to a hybrid octree-
shell composition; the hybrid hosting switches (``fine_wall_chains``,
``seal_solid_mode``/``seal_erode_passes``, ``root_bounce_seal``) default
to the legacy monolith wiring.

Observer brackets (F-6, diagnosis ``diag_negative_force_20261011``):
``observer_mode`` selects the control-volume measurement bracket.
``"legacy"`` (default, the v1/v2 formal records) keeps the per-substep
before/out pairs — which bracket only the L1 substep advances, so the
wall impulse applied inside the pre-restrict hook window (leaf BFL +
restriction write-back + reflux) lands between the measurement pairs and
is structurally invisible, and with ``fine_wall_chains=False`` the
friction/pressure reporters stay wired to the absent block wall chain
(literal 0.0).  ``"full_step"`` ADDITIONALLY brackets the full root step
(finest and root-envelope levels: entry-state snapshot before the substep
burst/hook, readout after ``hierarchy.step`` returns) so the hook edit
becomes part of the measured momentum change, and feeds the friction/
pressure split from an optional :class:`LeafForceLedgerMailbox` posted by
the host hook.  Measurement only — no physics or stepping change; every
legacy artifact key keeps producing the legacy numbers, new keys carry
the full-step readings, and each derived Ct series records its observer
(``result.ct_series_observers``).

This runner composes existing operators only; no solver hot path changes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as nnf

from .boundaries3d import bounce_back_cells_3d
from .checkpoint_io import atomic_torch_save
from .control_volume_force import (
    box_control_volume,
    fluid_momentum_change,
    observe_control_volume_force,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .external_open_boundary import non_equilibrium_far_field_bc_3d
from .flat_plate_wall_model import _ramp
from .force_convergence import assess_force_stationarity
from .kinetic_hygiene import global_x_momentum, period2_mx_probe
from .population_positivity import limit_nonequilibrium_for_positivity
from .solver3d import stream3d
from .sponge_layer import apply_equilibrium_difference_sponge, build_sponge_sigma_3d
from .static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
    convective_refined_tau,
)
from .suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    suboff_analytic_wetted_area,
)
from .suboff_static_amr import (
    build_fine_suboff_mask,
    build_nested_fine_suboff_mask,
    plan_nested_suboff_static_amr,
    plan_suboff_static_amr,
)
from .turbulence import collide_smagorinsky_bgk3d
from .wall_chain import (
    LevelWallChain,
    build_level_wall_chain,
    count_boundary_links_outside_box,
)
from .wall_model import WALL_TRACTION_SOURCE_SCHEME, bfl_wall_function_3d

__all__ = [
    "SuboffAMRWallModelConfig",
    "CT_SERIES_OBSERVERS",
    "LEAF_AREA_FACTOR",
    "LEAF_LEDGER_CONVENTION",
    "LeafForceLedgerMailbox",
    "LevelContext",
    "RunDiagnostics",
    "build_amr_contexts",
    "make_advance",
    "run_suboff_amr_wallmodel",
    "write_artifact",
]


@dataclass(frozen=True)
class SuboffAMRWallModelConfig:
    """One SUBOFF high-Re wall-model LES run on nested static block AMR.

    Defaults: AFF-8 FULL hull, root 400x200x200 (L=200, cx=0.35 nx — the
    uniform formal-tier base for direct A/B), anchor-1 Reynolds, one L1
    hull+wake 2:1 block, single card cuda:6.
    """

    # geometry / grid (root level)
    nx: int = 400
    ny: int = 200
    nz: int = 200
    hull_length: float = 200.0
    hull_type: str = "FULL"
    mesh_source: str = "hybrid"
    center_x_fraction: float = 0.35
    # physics ladder (target physical Re; collision runs at resolved Re)
    reynolds: float = 1.166e7
    resolved_reynolds: float = 1.0e5
    lattice_speed: float = 0.06
    # wall model
    wall_law: str = "reichardt"
    wall_mode: str = "wall_model_slip"
    ramp_steps: int = 1000
    smagorinsky_cs: float = 0.05
    positivity_limiter: bool = True
    # v3.3 symmetry repair (layer-2, adjudicated 2026-10-12): the octree
    # leaf field gets the same second (post-wall) positivity pass every
    # block advance gets.  The flag is read by the hybrid host's
    # pre-restrict hook and passed through to step_octree_shell's
    # leaf_post_wall_positivity (the runner itself never calls the shell
    # stepper); the runner records the pass's clamp-share series when the
    # host also passes a LeafPostWallPositivityMailbox.  Default False =
    # the legacy bit-identical leaf path (no tensor ops added).
    leaf_post_wall_positivity: bool = False
    y_plus_lower_bound: float = 30.0
    y_plus_upper_bound: float = 5000.0
    guo_forcing: bool = True
    guo_direction_chunk_size: int = 19
    low_memory_macroscopic: bool = False
    solid_handling: str = "marine_bounce"
    stress_exchange_distance: float | None = None
    # AMR block planning
    levels: int = 1  # 1 = L1-only, 2 = L1 + nested L2 core
    l1_wall_margin: int = 6  # root-lu
    l1_wake_cells: int = 40  # root-lu
    l2_wall_margin_l1_cells: int = 6  # L1-lu around the exact L1 hull
    reflux: bool = True
    maximum_reflux_correction_fraction: float = 0.2
    wall_model_on_root: bool = False  # vacuous by geometry; A/B toggle
    # numerics
    sponge_width: int = 24
    sponge_strength: float = 0.2
    cv_margin: int = 10  # root-lu request; finest CV may clamp
    # hybrid-ii hosting switches (F-5; defaults reproduce the legacy path)
    fine_wall_chains: bool = (
        True  # False: fine levels build no wall chain (leaf shell owns the wall)
    )
    seal_solid_mode: str = (
        "full"  # "full" legacy | "eroded_core" bounce seal on the eroded interior
    )
    seal_erode_passes: int = (
        6  # erode passes for seal_solid_mode="eroded_core" (hybrid ii: BL 4 + transition 1 + 1)
    )
    root_bounce_seal: bool = (
        True  # False: no root-level bounce-back (twin composition; restriction owns those cells)
    )
    # observer bracket (F-6; default reproduces the legacy path bit-for-bit):
    # "legacy" = per-substep before/out pairs (the v1/v2 formal records;
    # structurally blind to the pre-restrict hook edit — Q2 of
    # diag_negative_force_20261011 — and Ct_fric/Ct_wall_total stay 0.0
    # under fine_wall_chains=False, Q3), "full_step" = additionally bracket
    # the FULL root step and feed friction/pressure from the leaf force
    # ledger mailbox.  Measurement only; stepping is untouched either way.
    observer_mode: str = "legacy"
    # run control
    steps: int = 12000
    warmup_steps: int = 6000
    report_interval: int = 500
    wall_diagnostic_interval: int = 50
    checkpoint_interval: int = 2000
    checkpoint_path: str | None = None
    resume: bool = False
    final_field_dump_path: str | None = None
    # reference (locked ladder; informative at probe, gated at prereg)
    ct_reference: float = 0.003667
    reference_source: str = "refs_lock EFD anchor 1 (CFD Letters 2024 Table 3)"
    # devices: single-card only in Phase 0 (multi-card level split is a
    # design-only deliverable; pending the distributed support survey)
    device: str = "cuda:6"
    device_per_level: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.hull_type.upper() not in {"BARE_HULL", "WITH_SAIL", "FULL"}:
            raise ValueError("unsupported hull_type")
        if self.mesh_source not in {"suboff", "gradient", "hybrid"}:
            raise ValueError("unsupported mesh_source")
        if self.wall_law not in {"log", "reichardt", "musker"}:
            raise ValueError("unsupported wall law")
        if self.wall_mode not in {"stationary", "wall_model_slip", "spalding_exchange"}:
            raise ValueError("unsupported wall mode")
        if self.levels not in (1, 2):
            raise ValueError("levels must be 1 (L1-only) or 2 (nested L2)")
        if self.solid_handling not in {"marine_bounce", "flatplate_where"}:
            raise ValueError("unsupported solid_handling")
        if self.seal_solid_mode not in {"full", "eroded_core"}:
            raise ValueError("unsupported seal_solid_mode")
        if self.seal_erode_passes < 1:
            raise ValueError("seal_erode_passes must be >= 1")
        if self.observer_mode not in {"legacy", "full_step"}:
            raise ValueError("observer_mode must be 'legacy' or 'full_step'")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("invalid averaging window")
        if self.l1_wall_margin < 2 or self.l1_wake_cells < 0:
            raise ValueError("l1 margins out of range")
        if self.l2_wall_margin_l1_cells < 2:
            raise ValueError("l2 margin must be >= 2 L1 cells")
        if not 0.0 <= self.smagorinsky_cs < 0.5:
            raise ValueError("smagorinsky_cs must lie in [0,0.5)")
        if not 0.0 < self.lattice_speed < 0.15:
            raise ValueError("lattice_speed must lie in (0,0.15)")
        if self.reynolds <= 0.0 or self.resolved_reynolds <= 0.0:
            raise ValueError("Reynolds numbers must be positive")
        if self.resume and not self.checkpoint_path:
            raise ValueError("resume requires checkpoint_path")
        devices = self.device_per_level if self.device_per_level else (self.device,)
        if len(set(devices)) != 1:
            raise ValueError(
                "Phase 0 single-card rule: device_per_level must name one "
                "device for every level (multi-GPU level split is design-only)"
            )

    @property
    def wall_nu_root(self) -> float:
        return self.lattice_speed * self.hull_length / self.reynolds

    @property
    def collision_nu_root(self) -> float:
        return self.lattice_speed * self.hull_length / self.resolved_reynolds

    @property
    def tau_root(self) -> float:
        return 0.5 + 3.0 * self.collision_nu_root

    def tau_by_level(self) -> tuple[float, ...]:
        taus = [self.tau_root]
        for _ in range(self.levels):
            taus.append(convective_refined_tau(taus[-1]))
        return tuple(taus)

    def wall_nu_by_level(self) -> tuple[float, ...]:
        """Wall-law lattice viscosity per level (physical Re, level length)."""
        return tuple(
            self.lattice_speed * (self.hull_length * 2**level) / self.reynolds
            for level in range(self.levels + 1)
        )


_UNSET = object()  # sentinel: bounce_seal not given -> default to the level solid


class LevelContext:
    """Per-level geometry, wall chain and observers (single device).

    ``bounce_seal`` is the mask the marine bounce-back seal acts on:
    unset -> the level solid (legacy behaviour), ``None`` -> no bounce-back
    at this level, a tensor -> bounce-back confined to that mask (the CV
    accounting keeps using ``solid``).
    """

    def __init__(
        self,
        level: int,
        solid: torch.Tensor | None,
        chain: LevelWallChain | None,
        physical_mask: torch.Tensor,
        control_volume: torch.Tensor | None,
        cv_margin_cells: int | None,
        solid_handling: str,
        bounce_seal: Any = _UNSET,
    ) -> None:
        self.level = level
        self.solid = solid
        self.chain = chain
        self.physical_mask = physical_mask
        self.control_volume = control_volume
        self.cv_margin_cells = cv_margin_cells
        self.solid_handling = solid_handling
        self.bounce_seal = solid if bounce_seal is _UNSET else bounce_seal
        self.solid_q = solid.unsqueeze(0).expand((19, *solid.shape)) if solid is not None else None

    @property
    def has_wall_chain(self) -> bool:
        return self.chain is not None

    @property
    def near(self) -> torch.Tensor:
        assert self.chain is not None
        return self.chain.near

    @property
    def normals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.chain is not None
        return self.chain.normals

    @property
    def bfl_mask(self) -> torch.Tensor:
        assert self.chain is not None
        return self.chain.bfl_mask

    @property
    def bfl_q(self) -> torch.Tensor:
        assert self.chain is not None
        return self.chain.bfl_q

    @property
    def wall_nu(self) -> float:
        assert self.chain is not None
        return self.chain.wall_nu


class RunDiagnostics:
    """Mutable per-run observation buffers shared with the advance closure.

    A driver that calls :func:`make_advance` directly must set ``step``
    before each hierarchy step and clear the ``finest_*`` per-step buffers;
    the history buffers must persist across steps (and resume).
    """

    def __init__(self) -> None:
        self.step = 0
        self.maximum_limited_fraction = 0.0
        self.finest_cv_substeps: list[float] = []
        self.finest_friction_substeps: list[float] = []
        self.finest_pressure_substeps: list[float] = []
        self.root_cv_per_step: list[float] = []
        # full_step-observer buffers (observer_mode="full_step"; unused and
        # empty in legacy mode).  The ``*_import_*`` buffers follow the same
        # per-step clearing contract as ``finest_cv_substeps``; the
        # ``*_per_root`` buffers persist across steps.
        self.finest_cv_import_x_substeps: list[float] = []
        self.root_cv_import_x_step: list[float] = []
        self.finest_cv_full_step_per_root: list[float] = []
        self.root_cv_full_step_per_root: list[float] = []
        self.leaf_ledger_link_x_per_root: list[float] = []
        self.leaf_ledger_shear_x_per_root: list[float] = []
        # v3.3 leaf post-wall positivity pass series (per root step, max
        # over the root step's leaf substeps; filled only when the driver
        # passes a LeafPostWallPositivityMailbox)
        self.leaf_post_wall_clamp_share: list[float] = []
        self.leaf_post_wall_rho_nonpositive_count: list[int] = []
        self.mx_per_level_per_step: list[list[float]] = []
        self.mx_finest_substeps: list[float] = []
        self.y_plus_min: list[float] = []
        self.y_plus_mean: list[float] = []
        self.y_plus_max: list[float] = []
        self.rejected_fraction: list[float] = []
        self.reflux_residual_abs: list[float] = []
        self.reflux_max_correction_fraction: list[float] = []
        self.reflux_limited_directions: list[int] = []


# ---------------------------------------------------------------------------
# observer provenance + leaf force ledger (F-6)
# ---------------------------------------------------------------------------
# Provenance tags: every derived Ct series records which observer produced
# it; the artifact mirrors this mapping at result.ct_series_observers so a
# driver (gate6 in particular) can key on the tag, not the column name.
OBSERVER_LEGACY_SUBSTEP = "legacy_substep_bracket"
OBSERVER_FULL_ROOT_STEP = "full_root_step_bracket"
OBSERVER_LEAF_LEDGER = "leaf_force_ledger_cadence_p"
OBSERVER_BLOCK_CHAIN = "legacy_block_wall_chain_branch"

CT_SERIES_OBSERVERS: dict[str, str] = {
    "ct_cv": OBSERVER_LEGACY_SUBSTEP,
    "ct_root_envelope_cv": OBSERVER_LEGACY_SUBSTEP,
    "ct_fric": OBSERVER_BLOCK_CHAIN,
    "ct_wall_total": OBSERVER_BLOCK_CHAIN,
    "ct_cv_full_step": OBSERVER_FULL_ROOT_STEP,
    "ct_root_envelope_cv_full_step": OBSERVER_FULL_ROOT_STEP,
    "ct_fric_leaf": OBSERVER_LEAF_LEDGER,
    "ct_wall_total_leaf": OBSERVER_LEAF_LEDGER,
    "ct_leaf_total_cadence_p": OBSERVER_LEAF_LEDGER,
}

# leaf cell linear size relative to the root lattice (L1 2:1 host, leaf 2:1
# under it): leaf-unit areas are root-unit areas * 16 (the driver cadence-P
# denominator convention).
LEAF_AREA_FACTOR = 16.0

LEAF_LEDGER_CONVENTION: dict[str, str] = {
    "link_force": (
        "body frame, +x drag on the hull: the leaf field loses exactly this "
        "x momentum per accumulation window (m2 evidence: "
        "link_field_vs_ledger_residual == 0.0); BFL link momentum exchange "
        "of the slip wall"
    ),
    "shear_force": (
        "force ON THE FLUID from the wall-law traction: body-frame friction "
        "drag = -shear_force (m2 evidence: friction_drag_x_diag_body_"
        "convention == -shear_force_x_diag)"
    ),
    "cadence": (
        "per-root-step accumulations posted by the host hook inside its "
        "pre_restrict_hook firing (cadence P: one post per firing, after "
        "the leaf burst, before the probe reset; the runner sums all posts "
        "of the root step)"
    ),
    "units": (
        "leaf lattice units; leaf cell = root/4 so leaf-unit area = "
        "root-unit area * 16 (the ct_*_leaf denominators use S_ref * 16); "
        "no substep-count division — matches the formal driver ct_leaf "
        "convention"
    ),
    "pending": (
        "library-side convention unification (leaf_wall link/shear sign "
        "frames, probe.total_force mixed sum) is pending upstream; re-map "
        "the ct_*_leaf keys through this block when it lands"
    ),
}


class LeafForceLedgerMailbox:
    """Race-free per-root-step mailbox for the leaf wall-chain force split.

    The hybrid host (the driver owning the octree shell and its
    :class:`~tensorlbm.octree_boundary.leaf_wall.LeafWallForceProbe`)
    ``post``\\ s the ledger split inside its ``pre_restrict_hook`` — where
    it already snapshots the probe, before resetting it — and the runner
    ``drain``\\ s once per root step after ``hierarchy.step`` returns.  All
    entries posted since the last drain are summed, so a two-interface
    hierarchy (hook firing once per finest burst) is covered by the same
    contract.  Values are x components in leaf lattice units; the frame
    conventions are fixed in :data:`LEAF_LEDGER_CONVENTION` and recorded
    verbatim in the artifact.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[float, float]] = []
        self.convention: dict[str, str] = dict(LEAF_LEDGER_CONVENTION)

    def post(self, link_force_x: Any, shear_force_x: Any) -> None:
        """Record one hook firing's leaf ledger x split (leaf LU)."""
        link = (
            float(link_force_x.item())
            if isinstance(link_force_x, torch.Tensor)
            else float(link_force_x)
        )
        shear = (
            float(shear_force_x.item())
            if isinstance(shear_force_x, torch.Tensor)
            else float(shear_force_x)
        )
        self._pending.append((link, shear))

    def drain(self) -> tuple[float, float] | None:
        """Sum and clear the entries since the last drain (``None`` if empty)."""
        if not self._pending:
            return None
        link = sum(entry[0] for entry in self._pending)
        shear = sum(entry[1] for entry in self._pending)
        self._pending.clear()
        return link, shear


class LeafPostWallPositivityMailbox:
    """Per-root-step mailbox for the leaf post-wall positivity pass (v3.3).

    The octree-shell host (the driver owning the shell in its
    ``pre_restrict_hook``) passes this mailbox's :attr:`entries` list to
    :func:`tensorlbm.octree_boundary.stepping.step_octree_shell` as
    ``leaf_post_wall_diagnostics`` — one dict per leaf substep is appended
    there — and the runner ``drain``\\ s once per root step after
    ``hierarchy.step`` returns, reducing the root step's substeps to their
    max clamp share and max non-positive-density clamp count.  The same
    contract as :class:`LeafForceLedgerMailbox`, plus a ``history`` tap:
    every drained (share, count) pair is retained so the driver can record
    the full per-root-step series without racing the runner's drain.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, float]] = []
        self.history: list[tuple[float, int]] = []

    def drain(self) -> tuple[float, int] | None:
        """Max and clear the entries since the last drain (``None`` if empty)."""
        if not self.entries:
            return None
        share = max(float(entry["limited_fraction"]) for entry in self.entries)
        count = max(int(entry["rho_nonpositive_count"]) for entry in self.entries)
        self.entries.clear()
        self.history.append((share, count))
        return share, count


def _cv_crop_window(control_volume: torch.Tensor) -> tuple[slice, slice, slice]:
    """Crop window (CV bounding box plus one halo ring) for snapshots.

    The halo ring keeps the cropped control volume strictly interior so
    the :mod:`tensorlbm.control_volume_force` validators accept the crop,
    and drops the entry-snapshot memory from the full allocated grid to
    the CV box (the full-step bracket only needs the fluid-momentum change
    inside the CV).
    """
    zs, ys, xs = torch.nonzero(control_volume, as_tuple=True)
    z0, z1 = int(zs.min().item()), int(zs.max().item()) + 1
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
    return (slice(z0 - 1, z1 + 1), slice(y0 - 1, y1 + 1), slice(x0 - 1, x1 + 1))


def _crop_populations(
    f: torch.Tensor,
    window: tuple[slice, slice, slice],
) -> torch.Tensor:
    """Spatial crop of a (Q, nz, ny, nx) population tensor (a view)."""
    return f[:, window[0], window[1], window[2]]


def _pad_with_ghost(physical: torch.Tensor, ghost: int) -> torch.Tensor:
    """Place a physical fine mask into its ghost-padded allocated grid."""
    padded = torch.zeros(
        tuple(size + 2 * ghost for size in physical.shape),
        dtype=physical.dtype,
        device=physical.device,
    )
    padded[ghost:-ghost, ghost:-ghost, ghost:-ghost] = physical
    return padded


def _root_envelope_cv_box(
    root_solid: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    margin: int,
    nx: int,
    ny: int,
    nz: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    """Root envelope control volume: hull bounding box +- ``margin``.

    ``root_solid`` has (nz, ny, nx) layout, so the torch.nonzero columns
    arrive in (z, y, x) order — the x bounds come from the THIRD column.
    (F-5 fix: the original inline construction bound x from the first
    column, i.e. the z range, and vice versa.)
    """

    zs_c, ys_c, xs_c = torch.nonzero(root_solid, as_tuple=True)
    bounds = {
        "x": [
            max(1, int(xs_c.min().item()) - margin),
            min(nx - 2, int(xs_c.max().item()) + margin),
        ],
        "y": [
            max(1, int(ys_c.min().item()) - margin),
            min(ny - 2, int(ys_c.max().item()) + margin),
        ],
        "z": [
            max(1, int(zs_c.min().item()) - margin),
            min(nz - 2, int(zs_c.max().item()) + margin),
        ],
    }
    cv = box_control_volume(
        shape,
        x0=bounds["x"][0],
        x1=bounds["x"][1],
        y0=bounds["y"][0],
        y1=bounds["y"][1],
        z0=bounds["z"][0],
        z1=bounds["z"][1],
        device=device,
    )
    return cv, bounds


def _erode_solid(solid: torch.Tensor, passes: int) -> torch.Tensor:
    """Binary erosion (3x3x3, one cell per pass) of a solid mask."""
    cur = solid.clone()
    for _ in range(passes):
        shrink = (
            (
                -nnf.max_pool3d(
                    (~cur).float().unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                > -0.5
            )
            .squeeze(0)
            .squeeze(0)
        )
        cur = cur & shrink
        if not bool(cur.any()):
            return cur
    return cur


def build_amr_contexts(
    config: SuboffAMRWallModelConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Build geometry, the nested AMR hierarchy and per-level contexts.

    Returns a dict with ``hierarchy``, ``contexts`` (index = level),
    ``plans``, ``sponge``, ``root_cv``, geometry stats and bookkeeping
    integers, consumed by :func:`make_advance` and the artifact builder.
    """
    cad = SuboffConfig()
    hull = SuboffHullType[config.hull_type.upper()]
    center = (config.nx * config.center_x_fraction, config.ny / 2.0, config.nz / 2.0)
    root_solid, root_stats = build_suboff_mask(
        hull,
        config.nx,
        config.ny,
        config.nz,
        cx=center[0],
        cy=center[1],
        cz=center[2],
        length=config.hull_length,
        config=cad,
        device="cpu",
    )
    root_solid = root_solid.to(device)
    plan_l1 = plan_suboff_static_amr(
        root_solid,
        coarse_hull_length=config.hull_length,
        wall_margin=config.l1_wall_margin,
        wake_cells=config.l1_wake_cells,
    )
    l1_solid_physical, l1_stats = build_fine_suboff_mask(
        plan_l1,
        hull_type=hull,
        coarse_center=center,
        config=cad,
        device=device,
    )
    nested_plan = None
    l2_solid_physical = None
    if config.levels == 2:
        nested_plan = plan_nested_suboff_static_amr(
            plan_l1,
            l1_solid_physical,
            wall_margin=config.l2_wall_margin_l1_cells,
            wake_cells=0,
        )
        l2_solid_physical, _ = build_nested_fine_suboff_mask(
            nested_plan,
            hull_type=hull,
            coarse_center=center,
            config=cad,
            device=device,
        )

    shape = (config.nz, config.ny, config.nx)
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, config.lattice_speed)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    ux[root_solid] = 0.0
    root_f = equilibrium3d(rho, ux, uy, uz, device=device)
    del rho, ux, uy, uz

    configs = [
        StaticBlockAMRConfig(
            plan_l1.box,
            tau_coarse=config.tau_root,
            reflux=config.reflux,
            maximum_reflux_correction_fraction=(config.maximum_reflux_correction_fraction),
        ),
    ]
    if nested_plan is not None:
        configs.append(
            StaticBlockAMRConfig(
                nested_plan.box_in_outer_allocated_coordinates,
                tau_coarse=convective_refined_tau(config.tau_root),
                reflux=True,
                maximum_reflux_correction_fraction=(config.maximum_reflux_correction_fraction),
            ),
        )
    hierarchy = NestedStaticBlockAMR3D(root_f, configs)

    taus = config.tau_by_level()
    wall_nus = config.wall_nu_by_level()
    ghost = hierarchy.interfaces[0].config.ghost

    contexts: list[LevelContext] = []

    # ---- level 0 (root) ------------------------------------------------
    root_owned_interior = torch.zeros_like(root_solid)
    b = plan_l1.box
    root_owned_interior[b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1] = True
    root_links_outside = count_boundary_links_outside_box(root_solid, root_owned_interior)
    root_wall = config.wall_model_on_root and root_links_outside > 0
    root_chain = None
    if root_wall:
        root_chain = build_level_wall_chain(
            root_solid,
            wall_nu=wall_nus[0],
            cx=center[0],
            cy=center[1],
            cz=center[2],
            length=config.hull_length,
            mesh_source=config.mesh_source,
        )

    root_cv, root_cv_bounds = _root_envelope_cv_box(
        root_solid,
        shape,
        margin=config.cv_margin,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        device=device,
    )

    contexts.append(
        LevelContext(
            level=0,
            solid=root_solid,
            chain=root_chain,
            physical_mask=torch.ones(shape, dtype=torch.bool, device=device),
            control_volume=root_cv,
            cv_margin_cells=config.cv_margin,
            solid_handling=config.solid_handling,
            bounce_seal=(root_solid if config.root_bounce_seal else None),
        )
    )

    # ---- fine levels ----------------------------------------------------
    solids_physical = [l1_solid_physical]
    if l2_solid_physical is not None:
        solids_physical.append(l2_solid_physical)

    for level, solid_physical in enumerate(solids_physical, start=1):
        allocated = hierarchy.level_populations[level].shape[1:]
        solid_g = _pad_with_ghost(solid_physical, ghost)
        if level == 1:
            local_cx = center[0] * 2.0 - plan_l1.box.x0 * 2.0 + ghost
            local_cy = center[1] * 2.0 - plan_l1.box.y0 * 2.0 + ghost
            local_cz = center[2] * 2.0 - plan_l1.box.z0 * 2.0 + ghost
            hull_length_level = config.hull_length * 2.0
        else:
            assert nested_plan is not None
            origin_x, origin_y, origin_z = nested_plan.fine_origin_coarse_xyz
            cumulative = nested_plan.cumulative_ratio
            local_cx = (center[0] - origin_x) * cumulative + ghost
            local_cy = (center[1] - origin_y) * cumulative + ghost
            local_cz = (center[2] - origin_z) * cumulative + ghost
            hull_length_level = nested_plan.effective_hull_length_cells
        chain = None
        if config.fine_wall_chains:
            chain = build_level_wall_chain(
                solid_g,
                wall_nu=wall_nus[level],
                cx=local_cx,
                cy=local_cy,
                cz=local_cz,
                length=hull_length_level,
                mesh_source=config.mesh_source,
            )

        physical_mask = torch.zeros(allocated, dtype=torch.bool, device=device)
        physical_mask[ghost:-ghost, ghost:-ghost, ghost:-ghost] = True
        is_finest = level == len(solids_physical)
        control_volume = None
        cv_margin_cells = None
        if is_finest:
            fs, fy, fx = torch.nonzero(solid_g, as_tuple=True)
            z_lo, y_lo, x_lo = (int(t.min().item()) for t in (fs, fy, fx))
            z_hi, y_hi, x_hi = (int(t.max().item()) + 1 for t in (fs, fy, fx))
            request = config.cv_margin * (2**level)
            nz_a, ny_a, nx_a = allocated
            floor = ghost + 1
            cv_margin_cells = min(
                request,
                z_lo - floor,
                floor + nz_a - 1 - z_hi,
                y_lo - floor,
                floor + ny_a - 1 - y_hi,
                x_lo - floor,
                floor + nx_a - 1 - x_hi,
            )
            if cv_margin_cells < 1:
                raise ValueError(
                    "finest control volume does not fit inside the physical "
                    "fine region (increase the block wall margin)"
                )
            control_volume = box_control_volume(
                allocated,
                x0=x_lo - cv_margin_cells,
                x1=x_hi + cv_margin_cells,
                y0=y_lo - cv_margin_cells,
                y1=y_hi + cv_margin_cells,
                z0=z_lo - cv_margin_cells,
                z1=z_hi + cv_margin_cells,
                device=device,
            )
            del fs, fy, fx
        # eroded-core bounce seal (hybrid-ii hosting): the hull boundary band
        # belongs to the leaf shell's BFL, so the runner's marine seal acts
        # only on the eroded interior core.  The finest CV box above stays
        # derived from the FULL CAD solid bbox (margin semantics unchanged);
        # the eroded core replaces the context solid (seal + CV exclusion).
        seal_solid = solid_g
        if is_finest and config.seal_solid_mode == "eroded_core":
            eroded = _erode_solid(solid_physical, config.seal_erode_passes)
            if not bool(eroded.any()):
                raise ValueError("eroded seal solid is empty (seal_erode_passes too large)")
            seal_solid = _pad_with_ghost(eroded, ghost)
        contexts.append(
            LevelContext(
                level=level,
                solid=seal_solid,
                chain=chain,
                physical_mask=physical_mask,
                control_volume=control_volume,
                cv_margin_cells=cv_margin_cells,
                solid_handling=config.solid_handling,
            )
        )

    sponge = build_sponge_sigma_3d(
        shape,
        width=config.sponge_width,
        max_strength=config.sponge_strength,
        device=device,
        faces=("x+", "y-", "y+", "z-", "z+"),
    )

    return {
        "hierarchy": hierarchy,
        "contexts": contexts,
        "plan_l1": plan_l1,
        "nested_plan": nested_plan,
        "root_stats": root_stats,
        "l1_stats": l1_stats,
        "sponge": sponge,
        "root_cv": root_cv,
        "root_cv_bounds": root_cv_bounds,
        "root_links_outside_box": root_links_outside,
        "root_wall_active": root_wall,
        "taus": taus,
        "wall_nus": wall_nus,
        "center": center,
        "ghost": ghost,
        "bounce_seal_cells_by_level": [
            (int(ctx.bounce_seal.sum().item()) if ctx.bounce_seal is not None else None)
            for ctx in contexts
        ],
        "solid_cells_by_level": [
            int(root_stats["solid_cells"]),
            int(l1_stats["solid_cells"]),
            *([int(l2_solid_physical.sum().item())] if l2_solid_physical is not None else []),
        ],
    }


def make_advance(
    config: SuboffAMRWallModelConfig,
    built: dict[str, Any],
    diagnostics: RunDiagnostics,
) -> Any:
    """Build the ``advance(f, tau, level, substep)`` closure for the runtime.

    Operator order per level mirrors the uniform formal tier::

        collide(bgk_smag) -> [positivity] -> (marine|flatplate solid)
        -> stream -> (root: far-field) -> wall model
        -> (marine: interior seal) -> [positivity]
        -> (root: sponge -> far-field) -> CV observers
    """
    contexts = built["contexts"]
    sponge = built["sponge"]
    n_levels = len(contexts)
    full_step_observers = config.observer_mode == "full_step"

    def outer(state: torch.Tensor) -> torch.Tensor:
        return non_equilibrium_far_field_bc_3d(
            state,
            u_in=config.lattice_speed,
            faces=("x-", "x+", "y-", "y+", "z-", "z+"),
        )

    def advance(f: torch.Tensor, tau: float, level: int, substep: int):
        context = contexts[level]
        before = f
        collided = collide_smagorinsky_bgk3d(f, tau, C_s=config.smagorinsky_cs)
        if config.positivity_limiter:
            collided, diagnostic = limit_nonequilibrium_for_positivity(collided)
            diagnostics.maximum_limited_fraction = max(
                diagnostics.maximum_limited_fraction,
                diagnostic.limited_fraction,
            )
        if config.solid_handling == "flatplate_where" and context.solid_q is not None:
            post = torch.where(context.solid_q, before, collided)
        else:  # marine_bounce: collide everywhere, seal after the wall stage
            post = collided
        out = stream3d(post)
        if level == 0:
            out = outer(out)

        activation = _ramp(diagnostics.step, config.ramp_steps)
        is_finest = level == n_levels - 1
        collect_wall = (
            is_finest
            and diagnostics.step > config.warmup_steps
            and diagnostics.step % config.wall_diagnostic_interval == 0
        )
        friction = pressure = 0.0
        if context.has_wall_chain:
            wall_result = bfl_wall_function_3d(
                out,
                post,
                context.solid,
                context.wall_nu,
                context.bfl_mask,
                context.bfl_q,
                near_mask=context.near,
                wall_normals=context.normals,
                bfl_wall_mode=config.wall_mode,
                wall_activation=activation,
                wall_law=config.wall_law,
                stress_exchange_distance=config.stress_exchange_distance,
                return_wall_diagnostics=collect_wall,
                y_plus_lower_bound=config.y_plus_lower_bound,
                y_plus_upper_bound=config.y_plus_upper_bound,
                use_guo=config.guo_forcing,
                guo_direction_chunk_size=config.guo_direction_chunk_size,
                use_low_memory_macroscopic=config.low_memory_macroscopic,
                pressure_gradient_periodic_axes=(),
            )
            if collect_wall:
                out, friction, pressure, wall_diagnostics = wall_result
                if wall_diagnostics.y_plus_mean is not None:
                    diagnostics.y_plus_min.append(wall_diagnostics.y_plus_min)
                    diagnostics.y_plus_mean.append(wall_diagnostics.y_plus_mean)
                    diagnostics.y_plus_max.append(wall_diagnostics.y_plus_max)
                if wall_diagnostics.rejected_fraction is not None:
                    diagnostics.rejected_fraction.append(wall_diagnostics.rejected_fraction)
            else:
                out, friction, pressure = wall_result
        if config.solid_handling == "marine_bounce" and context.bounce_seal is not None:
            out = bounce_back_cells_3d(out, context.bounce_seal)
        if config.positivity_limiter:
            out, diagnostic = limit_nonequilibrium_for_positivity(out)
            diagnostics.maximum_limited_fraction = max(
                diagnostics.maximum_limited_fraction,
                diagnostic.limited_fraction,
            )
        if level == 0:
            out = apply_equilibrium_difference_sponge(
                out, sponge, velocity_target=(config.lattice_speed, 0.0, 0.0)
            )
            out = outer(out)

        # observers -------------------------------------------------------
        if level == 0 and contexts[0].control_volume is not None:
            root_result = observe_control_volume_force(
                before,
                out,
                post,
                contexts[0].control_volume,
                solid=contexts[0].solid,
                periodic_axes=(),
            )
            diagnostics.root_cv_per_step.append(float(root_result.force_on_body[0].item()))
            if full_step_observers:
                diagnostics.root_cv_import_x_step.append(
                    float(root_result.momentum_import[0].item())
                )
        if level == n_levels - 1:
            diagnostics.mx_finest_substeps.append(global_x_momentum(f, context.physical_mask))
            if context.control_volume is not None:
                cv_result = observe_control_volume_force(
                    before,
                    out,
                    post,
                    context.control_volume,
                    solid=context.solid,
                    periodic_axes=(),
                )
                cv_force = float(cv_result.force_on_body[0].item())
                diagnostics.finest_cv_substeps.append(cv_force)
                if full_step_observers:
                    diagnostics.finest_cv_import_x_substeps.append(
                        float(cv_result.momentum_import[0].item())
                    )
                diagnostics.finest_friction_substeps.append(float(friction))
                diagnostics.finest_pressure_substeps.append(float(pressure))
        return AMRAdvanceResult(out, post)

    return advance


def run_suboff_amr_wallmodel(
    config: SuboffAMRWallModelConfig | None = None,
    *,
    pre_restrict_hook: Any = None,
    leaf_force_ledger: LeafForceLedgerMailbox | None = None,
    leaf_post_wall_mailbox: LeafPostWallPositivityMailbox | None = None,
) -> dict[str, Any]:
    """Run the nested static-AMR SUBOFF wall-model LES and return the artifact.

    ``pre_restrict_hook`` (default ``None`` = the legacy loop, unchanged)
    forwards the engine's pre-restriction stage to a hybrid composition:
    once per root step, after the finest level's last substep and before
    the fine-to-coarse restriction, the hook is called as
    ``pre_restrict_hook(fine_old, fine_new, fine_post)`` where
    ``fine_old``/``fine_new`` are interior (ghost-stripped) views of the
    finest level's populations at root-step entry / after the substep
    burst and ``fine_post`` lists the finest level's post-collision
    interior views, one per substep (the refluxing shell stage's
    ``l1_post`` contract).  ``fine_new`` may be mutated in place; the
    restriction then carries the mutated state into the coarse level.

    ``leaf_force_ledger`` (default ``None``) is only read under
    ``config.observer_mode == "full_step"``: a
    :class:`LeafForceLedgerMailbox` posted by ``pre_restrict_hook`` once
    per firing supplies the leaf wall-chain force split that feeds the
    ``ct_fric_leaf``/``ct_wall_total_leaf``/``ct_leaf_total_cadence_p``
    series (conventions recorded in the artifact).  Without a mailbox the
    leaf-ledger Ct keys are stored as NaN — an unmeasured reading, never a
    fake zero.

    ``leaf_post_wall_mailbox`` (default ``None``; v3.3) records the leaf
    post-wall positivity pass: the hook passes the mailbox's ``entries``
    list into :func:`step_octree_shell` as ``leaf_post_wall_diagnostics``
    (together with ``leaf_post_wall_positivity=`` the config flag of the
    same name), the stepper appends one dict per leaf substep, and this
    runner drains once per root step into the
    ``result.leaf_post_wall_positivity`` series (per-root-step max clamp
    share and non-positive-density clamp count).
    """
    if config is None:
        config = SuboffAMRWallModelConfig()
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    built = build_amr_contexts(config, device)
    hierarchy = built["hierarchy"]
    contexts = built["contexts"]
    diagnostics = RunDiagnostics()
    advance = make_advance(config, built, diagnostics)

    step_advance = advance
    step_hook: Any = None
    if pre_restrict_hook is not None:
        n_levels = len(contexts)
        ghost = built["ghost"]
        finest_post_views: list[torch.Tensor] = []

        def advance_with_capture(f: torch.Tensor, tau: float, level: int, substep: int):
            result = advance(f, tau, level, substep)
            if level == n_levels - 1:
                finest_post_views.append(
                    result.post_collision[:, ghost:-ghost, ghost:-ghost, ghost:-ghost]
                )
            return result

        def hook_adapter(fine_old: torch.Tensor, fine_new: torch.Tensor) -> None:
            try:
                pre_restrict_hook(
                    fine_old[:, ghost:-ghost, ghost:-ghost, ghost:-ghost],
                    fine_new[:, ghost:-ghost, ghost:-ghost, ghost:-ghost],
                    list(finest_post_views),
                )
            finally:
                finest_post_views.clear()

        step_advance = advance_with_capture
        step_hook = hook_adapter

    # full-step observer scaffolding (observer_mode="full_step"): static CV
    # crops for the entry-state snapshots.  The snapshots are per-root-step
    # copies of the CV-box region only (measurement; nothing downstream of
    # them feeds the physics).
    full_step_observers = config.observer_mode == "full_step"
    finest_window: tuple[slice, slice, slice] | None = None
    root_window: tuple[slice, slice, slice] | None = None
    finest_cv_crop = finest_solid_crop = None
    root_cv_crop = root_solid_crop = None
    finest_snapshot: torch.Tensor | None = None
    root_snapshot: torch.Tensor | None = None
    if full_step_observers:
        if contexts[-1].control_volume is not None:
            finest_window = _cv_crop_window(contexts[-1].control_volume)
            finest_cv_crop = contexts[-1].control_volume[finest_window].clone()
            finest_solid_crop = (
                contexts[-1].solid[finest_window].clone()
                if contexts[-1].solid is not None
                else None
            )
            finest_snapshot = _crop_populations(
                hierarchy.level_populations[-1], finest_window
            ).clone()
        if contexts[0].control_volume is not None:
            root_window = _cv_crop_window(contexts[0].control_volume)
            root_cv_crop = contexts[0].control_volume[root_window].clone()
            root_solid_crop = (
                contexts[0].solid[root_window].clone() if contexts[0].solid is not None else None
            )
            root_snapshot = _crop_populations(hierarchy.level_populations[0], root_window).clone()

    plan_l1 = built["plan_l1"]
    nested_plan = built["nested_plan"]
    shape = (config.nz, config.ny, config.nx)
    s_ref = suboff_analytic_wetted_area(config.hull_length)
    denominator = 0.5 * config.lattice_speed**2 * s_ref

    start_step = 0
    diverged_at: int | None = None
    ct_time_series: list[dict[str, Any]] = []

    checkpoint = Path(config.checkpoint_path) if config.checkpoint_path else None
    if config.resume:
        assert checkpoint is not None
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("schema") != "tensorlbm-suboff-amr-wallmodel-checkpoint-v1":
            raise ValueError("checkpoint schema mismatch")
        # observer_mode is compared separately so pre-F-6 checkpoints (no
        # observer_mode key) resume under the legacy mode unchanged, while a
        # mode switch across a resume is refused.
        saved_configuration = dict(state["configuration"])
        saved_observer_mode = saved_configuration.pop("observer_mode", "legacy")
        current_configuration = _checkpoint_configuration(config)
        current_observer_mode = current_configuration.pop("observer_mode")
        if (
            saved_configuration != current_configuration
            or saved_observer_mode != current_observer_mode
        ):
            raise ValueError("checkpoint configuration does not match run")
        hierarchy.restore_level_populations(
            [tensor.to(device) for tensor in state["level_populations"]]
        )
        start_step = int(state["step"])
        diagnostics.maximum_limited_fraction = float(state["maximum_limited_fraction"])
        ct_time_series = [dict(entry) for entry in state["ct_time_series"]]
        diagnostics.y_plus_min = list(state["y_plus_min"])
        diagnostics.y_plus_mean = list(state["y_plus_mean"])
        diagnostics.y_plus_max = list(state["y_plus_max"])
        diagnostics.rejected_fraction = list(state["rejected_fraction"])
        diagnostics.mx_per_level_per_step = [
            list(entry) for entry in state["mx_per_level_per_step"]
        ]
        diagnostics.mx_finest_substeps = list(state["mx_finest_substeps"])
        diagnostics.finest_cv_full_step_per_root = list(
            state.get("finest_cv_full_step_per_root", [])
        )
        diagnostics.root_cv_full_step_per_root = list(state.get("root_cv_full_step_per_root", []))
        diagnostics.leaf_ledger_link_x_per_root = list(state.get("leaf_ledger_link_x_per_root", []))
        diagnostics.leaf_ledger_shear_x_per_root = list(
            state.get("leaf_ledger_shear_x_per_root", [])
        )
        diagnostics.leaf_post_wall_clamp_share = list(state.get("leaf_post_wall_clamp_share", []))
        diagnostics.leaf_post_wall_rho_nonpositive_count = list(
            state.get("leaf_post_wall_rho_nonpositive_count", [])
        )
        if start_step >= config.steps:
            raise ValueError("checkpoint already reached requested steps")

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {
                "schema": "tensorlbm-suboff-amr-wallmodel-checkpoint-v1",
                "configuration": _checkpoint_configuration(config),
                "step": step,
                "level_populations": [
                    tensor.detach().cpu() for tensor in hierarchy.level_populations
                ],
                "maximum_limited_fraction": diagnostics.maximum_limited_fraction,
                "ct_time_series": ct_time_series,
                "y_plus_min": diagnostics.y_plus_min,
                "y_plus_mean": diagnostics.y_plus_mean,
                "y_plus_max": diagnostics.y_plus_max,
                "rejected_fraction": diagnostics.rejected_fraction,
                "mx_per_level_per_step": diagnostics.mx_per_level_per_step,
                "mx_finest_substeps": diagnostics.mx_finest_substeps,
                "finest_cv_full_step_per_root": diagnostics.finest_cv_full_step_per_root,
                "root_cv_full_step_per_root": diagnostics.root_cv_full_step_per_root,
                "leaf_ledger_link_x_per_root": diagnostics.leaf_ledger_link_x_per_root,
                "leaf_ledger_shear_x_per_root": diagnostics.leaf_ledger_shear_x_per_root,
                "leaf_post_wall_clamp_share": diagnostics.leaf_post_wall_clamp_share,
                "leaf_post_wall_rho_nonpositive_count": (
                    diagnostics.leaf_post_wall_rho_nonpositive_count
                ),
            },
            checkpoint,
        )

    for step in range(start_step + 1, config.steps + 1):
        diagnostics.step = step
        diagnostics.finest_cv_substeps.clear()
        diagnostics.finest_friction_substeps.clear()
        diagnostics.finest_pressure_substeps.clear()
        if full_step_observers:
            diagnostics.finest_cv_import_x_substeps.clear()
            diagnostics.root_cv_import_x_step.clear()
            if finest_snapshot is not None:
                finest_snapshot.copy_(
                    _crop_populations(hierarchy.level_populations[-1], finest_window)
                )
            if root_snapshot is not None:
                root_snapshot.copy_(_crop_populations(hierarchy.level_populations[0], root_window))
        ledgers = hierarchy.step(step_advance, pre_restrict_hook=step_hook)
        diagnostics.reflux_residual_abs.append(
            max(abs(float(ledger.residual.sum().item())) for ledger in ledgers)
        )
        diagnostics.reflux_max_correction_fraction.append(
            max(ledger.maximum_applied_correction_fraction for ledger in ledgers)
        )
        diagnostics.reflux_limited_directions.append(
            max(ledger.limited_directions for ledger in ledgers)
        )
        # v3.3 leaf post-wall positivity series: the host's hook appended
        # this root step's per-substep clamp diagnostics inside
        # step_octree_shell; drain once per root step (same cadence as the
        # leaf force ledger mailbox — the hook fires inside
        # hierarchy.step, this drain runs after it returns).
        if leaf_post_wall_mailbox is not None:
            drained_post_wall = leaf_post_wall_mailbox.drain()
            if drained_post_wall is not None:
                diagnostics.leaf_post_wall_clamp_share.append(drained_post_wall[0])
                diagnostics.leaf_post_wall_rho_nonpositive_count.append(drained_post_wall[1])
        diagnostics.mx_per_level_per_step.append(
            [
                global_x_momentum(level_f, ctx.physical_mask)
                for level_f, ctx in zip(hierarchy.level_populations, contexts)
            ]
        )
        # ---- full-step observers: read out after ALL advances and
        # write-backs (hook edit, restriction write-back, reflux) complete.
        # The finest-level bracket is exact: nothing else edits CV cells of
        # the finest level between the substep pairs.  The root-envelope
        # bracket additionally contains the restriction write-back (that is
        # the point) and any fine-level streaming across the root-CV faces
        # where the L1 box extends past it (see the artifact caveat).
        leaf_link_x = leaf_shear_x = math.nan
        if full_step_observers:
            if finest_snapshot is not None and diagnostics.finest_cv_substeps:
                change_finest = fluid_momentum_change(
                    finest_snapshot,
                    _crop_populations(hierarchy.level_populations[-1], finest_window),
                    finest_cv_crop,
                    solid=finest_solid_crop,
                    periodic_axes=(),
                )
                n_substeps = len(diagnostics.finest_cv_substeps)
                diagnostics.finest_cv_full_step_per_root.append(
                    (sum(diagnostics.finest_cv_import_x_substeps) - float(change_finest[0]))
                    / n_substeps
                )
            if root_snapshot is not None and diagnostics.root_cv_import_x_step:
                change_root = fluid_momentum_change(
                    root_snapshot,
                    _crop_populations(hierarchy.level_populations[0], root_window),
                    root_cv_crop,
                    solid=root_solid_crop,
                    periodic_axes=(),
                )
                diagnostics.root_cv_full_step_per_root.append(
                    diagnostics.root_cv_import_x_step[0] - float(change_root[0])
                )
            if leaf_force_ledger is not None:
                drained = leaf_force_ledger.drain()
                if drained is not None:
                    leaf_link_x, leaf_shear_x = drained
                    diagnostics.leaf_ledger_link_x_per_root.append(leaf_link_x)
                    diagnostics.leaf_ledger_shear_x_per_root.append(leaf_shear_x)
        if diagnostics.finest_cv_substeps:
            cv_force = sum(diagnostics.finest_cv_substeps) / len(diagnostics.finest_cv_substeps)
            friction = sum(diagnostics.finest_friction_substeps) / len(
                diagnostics.finest_friction_substeps
            )
            pressure = sum(diagnostics.finest_pressure_substeps) / len(
                diagnostics.finest_pressure_substeps
            )
        else:
            cv_force = friction = pressure = math.nan
        root_cv = diagnostics.root_cv_per_step[-1] if diagnostics.root_cv_per_step else math.nan
        if step > config.warmup_steps:
            entry: dict[str, Any] = {
                "step": step,
                "ct_fric": friction / denominator,
                "ct_wall_total": (friction + pressure) / denominator,
                "ct_cv": cv_force / denominator,
                "ct_root_envelope_cv": root_cv / denominator,
            }
            if full_step_observers:
                denominator_leaf = denominator * LEAF_AREA_FACTOR
                entry["observer_mode"] = config.observer_mode
                entry["ct_cv_full_step"] = (
                    diagnostics.finest_cv_full_step_per_root[-1] / denominator
                    if diagnostics.finest_cv_full_step_per_root
                    else math.nan
                )
                entry["ct_root_envelope_cv_full_step"] = (
                    diagnostics.root_cv_full_step_per_root[-1] / denominator
                    if diagnostics.root_cv_full_step_per_root
                    else math.nan
                )
                # leaf ledger, body frame: friction = -shear (fluid frame ->
                # body frame), pressure-like = link; conventions recorded at
                # result.leaf_ledger_convention (re-map point for the
                # pending library-side unification)
                entry["ct_fric_leaf"] = (-leaf_shear_x) / denominator_leaf
                entry["ct_wall_total_leaf"] = (leaf_link_x - leaf_shear_x) / denominator_leaf
                entry["ct_leaf_total_cadence_p"] = (leaf_link_x + leaf_shear_x) / denominator_leaf
            ct_time_series.append(entry)
        if not all(bool(torch.isfinite(level_f).all()) for level_f in hierarchy.level_populations):
            diverged_at = step
            break
        if config.report_interval and step % config.report_interval == 0:
            recent = ct_time_series[-min(len(ct_time_series), 10) :]
            mean_ct_cv = (
                sum(entry["ct_cv"] for entry in recent) / len(recent) if recent else math.nan
            )
            print(
                f"suboff_amr_wm step={step}/{config.steps} "
                f"recent_Ct_cv={mean_ct_cv:.6f} "
                f"max_limited={diagnostics.maximum_limited_fraction:.3e}",
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
        dump: dict[str, Any] = {
            "step": config.steps,
            "l1_box_root_xyz_halfopen": [
                plan_l1.box.x0,
                plan_l1.box.x1,
                plan_l1.box.y0,
                plan_l1.box.y1,
                plan_l1.box.z0,
                plan_l1.box.z1,
            ],
            "levels": [],
        }
        for level_f, context in zip(hierarchy.level_populations, contexts):
            rho_l, ux_l, uy_l, uz_l = macroscopic3d(level_f)
            dump["levels"].append(
                {
                    "rho": rho_l.detach().to("cpu"),
                    "u": torch.stack((ux_l, uy_l, uz_l)).detach().to("cpu"),
                    "solid": (
                        context.solid.detach().to("cpu") if context.solid is not None else None
                    ),
                    "physical_mask": context.physical_mask.detach().to("cpu"),
                }
            )
        atomic_torch_save(dump, config.final_field_dump_path)

    # --- summary (window: post-warmup samples) ---
    def _mean(key: str) -> float:
        values = [entry[key] for entry in ct_time_series]
        return sum(values) / max(len(values), 1)

    def _mean_optional(key: str) -> float:
        """Window mean of a series key that legacy entries do not carry.

        Absent keys read as NaN (unmeasured), never as a silent 0.0 — the
        Q3 lesson of diag_negative_force_20261011.
        """
        values = [entry[key] for entry in ct_time_series if key in entry]
        return sum(values) / len(values) if values else math.nan

    ct_fric = _mean("ct_fric")
    ct_cv = _mean("ct_cv")
    ct_wall = _mean("ct_wall_total")
    ct_root = _mean("ct_root_envelope_cv")
    ct_cv_full_step = _mean_optional("ct_cv_full_step")
    ct_root_full_step = _mean_optional("ct_root_envelope_cv_full_step")
    ct_fric_leaf = _mean_optional("ct_fric_leaf")
    ct_wall_leaf = _mean_optional("ct_wall_total_leaf")
    ct_leaf_total = _mean_optional("ct_leaf_total_cadence_p")
    observer_difference = abs(ct_cv - ct_wall) / max(abs(ct_cv), 1e-30) * 100.0
    stationarity = (
        assess_force_stationarity(
            [entry["ct_cv"] for entry in ct_time_series],
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
    period2 = period2_mx_probe(diagnostics.mx_finest_substeps)

    artifact: dict[str, Any] = {
        "schema": "tensorlbm.suboff-amr-wallmodel-les/v1",
        "configuration": {
            "hull_type": config.hull_type.upper(),
            "mesh_source": config.mesh_source,
            "shape_zyx": list(shape),
            "hull_length": config.hull_length,
            "center_x_fraction": config.center_x_fraction,
            "reynolds": config.reynolds,
            "resolved_reynolds": config.resolved_reynolds,
            "lattice_speed": config.lattice_speed,
            "wall_nu_root": config.wall_nu_root,
            "collision_nu_root": config.collision_nu_root,
            "tau_root": config.tau_root,
            "steps": config.steps,
            "warmup_steps": config.warmup_steps,
            "ramp_steps": config.ramp_steps,
            "wall_law": config.wall_law,
            "wall_mode": config.wall_mode,
            "smagorinsky_cs": config.smagorinsky_cs,
            "positivity_limiter": config.positivity_limiter,
            "leaf_post_wall_positivity": config.leaf_post_wall_positivity,
            "y_plus_bounds": [config.y_plus_lower_bound, config.y_plus_upper_bound],
            "sponge_width": config.sponge_width,
            "sponge_strength": config.sponge_strength,
            "cv_margin": config.cv_margin,
            "stress_exchange_distance": config.stress_exchange_distance,
            "guo_forcing": config.guo_forcing,
            "guo_direction_chunk_size": config.guo_direction_chunk_size,
            "low_memory_macroscopic": config.low_memory_macroscopic,
            "solid_handling": config.solid_handling,
            "fine_wall_chains": config.fine_wall_chains,
            "observer_mode": config.observer_mode,
            "seal_solid_mode": config.seal_solid_mode,
            "seal_erode_passes": config.seal_erode_passes,
            "root_bounce_seal": config.root_bounce_seal,
            "collision_kernel": "bgk_smag",
            "device": str(config.device),
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "resumed_from_step": start_step,
            "reference_area_convention": "analytic bare hull (refs_lock)",
        },
        "library_modules": {
            "wall_chain": "tensorlbm/wall_chain.py",
            "period2_probes": "tensorlbm/kinetic_hygiene.py",
            "nested_shell_planning": "tensorlbm/amr_shell_planning.py",
            "union_control_volume": ("tensorlbm/control_volume_force.py::union_box_control_volume"),
            "analytic_wetted_area": ("tensorlbm/suboff_cad.py::suboff_analytic_wetted_area"),
        },
        "amr": {
            "levels": config.levels,
            "l1_wall_margin_root_lu": config.l1_wall_margin,
            "l1_wake_cells_root_lu": config.l1_wake_cells,
            "l2_wall_margin_l1_cells": (
                config.l2_wall_margin_l1_cells if config.levels == 2 else None
            ),
            "l1_box_root_xyz_halfopen": [
                plan_l1.box.x0,
                plan_l1.box.x1,
                plan_l1.box.y0,
                plan_l1.box.y1,
                plan_l1.box.z0,
                plan_l1.box.z1,
            ],
            "l2_box_l1_allocated_xyz_halfopen": (
                [
                    nested_plan.box_in_outer_allocated_coordinates.x0,
                    nested_plan.box_in_outer_allocated_coordinates.x1,
                    nested_plan.box_in_outer_allocated_coordinates.y0,
                    nested_plan.box_in_outer_allocated_coordinates.y1,
                    nested_plan.box_in_outer_allocated_coordinates.z0,
                    nested_plan.box_in_outer_allocated_coordinates.z1,
                ]
                if nested_plan is not None
                else None
            ),
            "tau_by_level": list(built["taus"]),
            "wall_nu_by_level": list(built["wall_nus"]),
            "allocated_cells_by_level": [
                int(level_f[0].numel()) for level_f in hierarchy.level_populations
            ],
            "level_shapes_zyx": [
                list(level_f.shape[1:]) for level_f in hierarchy.level_populations
            ],
            "solid_cells_by_level": built["solid_cells_by_level"],
            "root_boundary_links_outside_l1_box": built["root_links_outside_box"],
            "root_wall_model_active": built["root_wall_active"],
            "wall_model_placement_note": (
                "wall model active on the finest level, which owns the hull "
                "surface; the root hull is strictly interior to the L1 box "
                "(boundary links with fluid cell outside the box = "
                f"{built['root_links_outside_box']}), so root wall treatment "
                "is geometrically vacuous"
            ),
            "finest_cv_margin_cells": contexts[-1].cv_margin_cells,
            "finest_cv_margin_root_lu_equivalent": (
                contexts[-1].cv_margin_cells / float(2 ** contexts[-1].level)
            ),
            "bounce_seal_cells_by_level": built["bounce_seal_cells_by_level"],
            "root_envelope_cv_bounds_xyz": built["root_cv_bounds"],
            "reflux_enabled": config.reflux,
        },
        "geometry": {
            "analytic_bare_hull_wetted_area_lu2": s_ref,
            "root_solid_cells": built["root_stats"]["solid_cells"],
            "root_appendage_cells": built["root_stats"]["appendage_solid_cells"],
            "l1_solid_cells": built["l1_stats"]["solid_cells"],
            "l1_appendage_cells": built["l1_stats"]["appendage_solid_cells"],
        },
        "result": {
            "reference_area": s_ref,
            "Ct_fric": ct_fric,
            "Ct_cv": ct_cv,
            "Ct_wall_total": ct_wall,
            "Ct_root_envelope_cv": ct_root,
            # ---- F-6 full-step observer keys (additive; the four legacy Ct
            # keys above keep producing the legacy numbers verbatim) ----
            "observer_mode": config.observer_mode,
            "ct_cv_full_step": ct_cv_full_step,
            "ct_root_envelope_cv_full_step": ct_root_full_step,
            "ct_fric_leaf": ct_fric_leaf,
            "ct_wall_total_leaf": ct_wall_leaf,
            "ct_leaf_total_cadence_p": ct_leaf_total,
            "ct_series_observers": dict(CT_SERIES_OBSERVERS),
            "leaf_ledger_convention": dict(LEAF_LEDGER_CONVENTION),
            "full_step_observer": {
                "finest_bracket": (
                    "root-step entry snapshot (before the substep burst and "
                    "the pre-restrict hook) -> state after hierarchy.step "
                    "returns (after the hook edit, restriction write-back "
                    "and reflux)"
                ),
                "finest_formula": (
                    "force_full = (sum_sub streaming_momentum_import(post_sub)"
                    " - fluid_momentum_change(entry, exit)) / n_substeps"
                ),
                "root_envelope_bracket": (
                    "same entry->exit bracket on the root level (single root "
                    "advance per root step, no substep division)"
                ),
                "root_envelope_caveat": (
                    "the L1 block may extend past the root CV box (wake "
                    "cells); fine-level streaming across the root-CV "
                    "downstream face enters the bracket as unimported "
                    "delta-M — cross-check observer only, the finest CV is "
                    "the primary"
                ),
                "normalization": (
                    "finest force divided by the substep count (same "
                    "mean-of-substeps convention as the legacy ct_cv); "
                    "leaf-ledger keys undivided (cadence P, driver ct_leaf "
                    "convention)"
                ),
                "full_step_samples": len(diagnostics.finest_cv_full_step_per_root),
                "root_full_step_samples": len(diagnostics.root_cv_full_step_per_root),
                "leaf_ledger_samples": len(diagnostics.leaf_ledger_link_x_per_root),
            },
            "composite_drag_observer": "finest-level hull-hugging CV (primary)",
            "ct_reference": config.ct_reference,
            "reference_error_pct": reference_error,
            "observer_difference_pct": observer_difference,
            "drag_stationarity": (stationarity.to_dict() if stationarity else None),
            "maximum_positivity_limited_fraction": (diagnostics.maximum_limited_fraction),
            # v3.3 leaf post-wall positivity pass (additive series; empty
            # when the pass is off — the default-off legacy path).  Block
            # limiter passes are counted by maximum_limited_fraction above.
            "leaf_post_wall_positivity": {
                "enabled": config.leaf_post_wall_positivity,
                "samples": len(diagnostics.leaf_post_wall_clamp_share),
                "clamp_share_max": (max(diagnostics.leaf_post_wall_clamp_share, default=None)),
                "clamp_share_last50_max": (
                    max(diagnostics.leaf_post_wall_clamp_share[-50:], default=None)
                ),
                "rho_nonpositive_count_max": (
                    max(diagnostics.leaf_post_wall_rho_nonpositive_count, default=0)
                ),
                "rho_nonpositive_count_total": (
                    sum(diagnostics.leaf_post_wall_rho_nonpositive_count)
                ),
                "series_convention": (
                    "per root step, max over the root step's leaf substeps "
                    "of the post-wall (post-BFL/Guo-traction) leaf "
                    "positivity pass in step_octree_shell"
                ),
            },
            "maximum_abs_reflux_mass_residual": (max(diagnostics.reflux_residual_abs, default=0.0)),
            "maximum_reflux_correction_fraction": (
                max(diagnostics.reflux_max_correction_fraction, default=0.0)
            ),
            "maximum_reflux_limited_directions": (
                max(diagnostics.reflux_limited_directions, default=0)
            ),
            "wall_stress_applicability": {
                "samples": len(diagnostics.y_plus_mean),
                "y_plus_min": min(diagnostics.y_plus_min, default=None),
                "y_plus_mean": (
                    sum(diagnostics.y_plus_mean) / len(diagnostics.y_plus_mean)
                    if diagnostics.y_plus_mean
                    else None
                ),
                "y_plus_max": max(diagnostics.y_plus_max, default=None),
                "maximum_rejected_fraction": (max(diagnostics.rejected_fraction, default=0.0)),
            },
            "period2_finest_mx": period2,
            "finite": diverged_at is None,
            "diverged_at_step": diverged_at,
            "steps_completed": (config.steps if diverged_at is None else diverged_at - 1),
        },
        "wall_traction_source_scheme": WALL_TRACTION_SOURCE_SCHEME,
        "ct_time_series": ct_time_series,
        "mx_finest_substeps": diagnostics.mx_finest_substeps,
        "mx_per_level_per_step": diagnostics.mx_per_level_per_step,
        "y_plus_history": {
            "min": diagnostics.y_plus_min,
            "mean": diagnostics.y_plus_mean,
            "max": diagnostics.y_plus_max,
            "rejected_fraction": diagnostics.rejected_fraction,
        },
    }
    return artifact


def _checkpoint_configuration(config: SuboffAMRWallModelConfig) -> dict[str, Any]:
    return {
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "hull_length": config.hull_length,
        "hull_type": config.hull_type.upper(),
        "mesh_source": config.mesh_source,
        "center_x_fraction": config.center_x_fraction,
        "reynolds": config.reynolds,
        "resolved_reynolds": config.resolved_reynolds,
        "lattice_speed": config.lattice_speed,
        "wall_law": config.wall_law,
        "wall_mode": config.wall_mode,
        "ramp_steps": config.ramp_steps,
        "smagorinsky_cs": config.smagorinsky_cs,
        "positivity_limiter": config.positivity_limiter,
        "leaf_post_wall_positivity": config.leaf_post_wall_positivity,
        "y_plus_bounds": [config.y_plus_lower_bound, config.y_plus_upper_bound],
        "sponge_width": config.sponge_width,
        "sponge_strength": config.sponge_strength,
        "cv_margin": config.cv_margin,
        "stress_exchange_distance": config.stress_exchange_distance,
        "guo_forcing": config.guo_forcing,
        "guo_direction_chunk_size": config.guo_direction_chunk_size,
        "low_memory_macroscopic": config.low_memory_macroscopic,
        "solid_handling": config.solid_handling,
        "fine_wall_chains": config.fine_wall_chains,
        "observer_mode": config.observer_mode,
        "seal_solid_mode": config.seal_solid_mode,
        "seal_erode_passes": config.seal_erode_passes,
        "root_bounce_seal": config.root_bounce_seal,
        "levels": config.levels,
        "l1_wall_margin": config.l1_wall_margin,
        "l1_wake_cells": config.l1_wake_cells,
        "l2_wall_margin_l1_cells": config.l2_wall_margin_l1_cells,
        "reflux": config.reflux,
        "maximum_reflux_correction_fraction": (config.maximum_reflux_correction_fraction),
        "wall_model_on_root": config.wall_model_on_root,
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
