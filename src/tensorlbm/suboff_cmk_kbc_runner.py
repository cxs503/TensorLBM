"""SUBOFF bare-hull CM/CUMULANT/KBC × SGS diagnostic runner.

Runs SUBOFF bare hull at Re=2×10⁶ with three advanced collision families
(CM, CUMULANT, KBC) coupled with three SGS models (Smagorinsky, WALE,
Vreman).  Uses :func:`collide_advanced_3d` for unified collision dispatch
and far-field boundary conditions.

The SGS coupling computes a per-cell effective relaxation time
``tau_eff = tau_0 + 3*nu_t`` from the local eddy viscosity, then passes
it to the collision kernel:

* **CM / CUMULANT** — accept tensor ``tau`` (element-wise ``omega = 1/tau``).
* **KBC** — uses scalar mean ``tau_eff`` because the entropy bisection
  initialiser (``torch.full``) requires a Python scalar.

This runner does **not** modify any solver hot path.  Only existing
operators are composed.  The optional ``use_triton_step=True`` switch
swaps the per-step chain for the fused Triton step from
:mod:`tensorlbm.backends.triton_backend` (single-GPU CUDA, CM/CUMULANT
only); the default path is byte-identical to the pre-switch runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .advanced_collision_contract import collide_advanced_3d
from .boundaries3d import far_field_bc_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import correct_mass3d, stream3d
from .suboff_cad import SuboffHullType, build_suboff_mask
from .suboff_resistance import _voxel_wetted_area
from .compile_utils import compile_step, validate_compile_mode
from .turbulence import (
    _neq_stress_norm_3d,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
)

__all__ = [
    "SuboffCmkKbcConfig",
    "run_suboff_cmk_kbc",
    "COMBINATIONS",
]

# The 9 test combinations: 3 collision families × 3 SGS models
COMBINATIONS: list[tuple[str, str]] = [
    ("CM", "smagorinsky"),
    ("CM", "wale"),
    ("CM", "vreman"),
    ("CUMULANT", "smagorinsky"),
    ("CUMULANT", "wale"),
    ("CUMULANT", "vreman"),
    ("KBC", "smagorinsky"),
    ("KBC", "wale"),
    ("KBC", "vreman"),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuboffCmkKbcConfig:
    """Configuration for a single SUBOFF CM/CUMULANT/KBC × SGS run.

    Defaults target Re=2×10⁶ on a 320×160×160 D3Q19 grid with 1000 steps
    on ``sdaa:0``.  For fast tests, override ``nx/ny/nz/n_steps/device``.
    """

    re: float = 2_000_000.0
    collision: str = "CM"
    turbulence_model: str = "smagorinsky"
    nx: int = 320
    ny: int = 160
    nz: int = 160
    n_steps: int = 1000
    u_in: float = 0.06
    hull_length: float = 192.0  # 0.6 * 320
    device: str = "sdaa:0"
    lattice: str = "D3Q19"
    boundary_type: str = "farfield"
    # Step-backend switch (opt-in, default off): False keeps the inline
    # PyTorch 5-op chain (collide/stream/force/BC/mass) below unchanged;
    # True routes the per-step work through the fused Triton step in
    # ``tensorlbm.backends.triton_backend``.  Single-GPU CUDA only, and
    # CM/CUMULANT collisions only (KBC entropy bisection is Triton-less).
    use_triton_step: bool = False
    # SGS model constants
    C_s: float = 0.1  # Smagorinsky
    C_w: float = 0.5  # WALE
    C_V: float = 0.025  # Vreman
    # ITTC reference
    reference_Ct: float = 0.00405
    reference_source: str = "ITTC-1957"
    # Loop-policy knobs (zero-risk: they never touch the populations).
    # Sample the momentum-exchange force every ``force_every`` steps.
    # The default ``1`` keeps the historical dense per-step series;
    # N > 1 thins ``force_time_series``/``ct_time_series`` to the steps
    # that are multiples of N (final ``Ct`` then refers to the last
    # *sampled* step, which may precede ``steps_completed``) and skips
    # the full-grid force reduction on the other steps.  Applies to the
    # default PyTorch path only — the fused Triton step produces forces
    # on every launch, so its series stays dense regardless.
    force_every: int = 1
    # Run the read-only isfinite watchdog every ``check_every`` steps
    # (plus always on the final step).  The historical runner checked
    # every step; the default 10 only delays NaN/Inf fault detection,
    # it never changes the numerics or the artifact of a finite run.
    check_every: int = 10
    # Step-backend switch (opt-in, default off): ``None`` keeps the inline
    # PyTorch 5-op chain (collide/stream/force/BC/mass) eager and
    # byte-identical to the historical runner; ``"default"`` or
    # ``"max-autotune-no-cudagraphs"`` wraps that same chain in
    # ``torch.compile`` as two guarded graphs (with and without the
    # every-10-step mass correction; the step index never enters the
    # compiled code object, so no per-step recompiles).  Measured on
    # RTX 5090 (fp32, force_every=1): CM+Smag 4.9x/3.9x/2.1x at
    # nx=96/128/256, CUMULANT+Smag 10.0x at nx=128; numerics match eager
    # to ~2e-6 relative over 110 steps (Inductor reorders fp32 ops).
    # Modes with cudagraphs are rejected: the LBM step feeds its own
    # output back, which cudagraph replay overwrites.  CM/CUMULANT only
    # (KBC's entropy bisection calls ``.item()`` per step) and mutually
    # exclusive with ``use_triton_step``.  Single-GPU CUDA only.
    compile_mode: str | None = None

    def __post_init__(self) -> None:
        if self.collision.upper() not in {"CM", "CUMULANT", "KBC"}:
            raise ValueError(f"collision must be CM, CUMULANT, or KBC; got {self.collision!r}")
        if self.turbulence_model.lower() not in {"smagorinsky", "wale", "vreman"}:
            raise ValueError(
                f"turbulence_model must be smagorinsky, wale, or vreman; "
                f"got {self.turbulence_model!r}"
            )
        if self.lattice.upper() != "D3Q19":
            raise ValueError(
                f"lattice must be D3Q19 (far_field_bc_3d is D3Q19-only); got {self.lattice!r}"
            )
        if self.nx < 16 or self.ny < 8 or self.nz < 8:
            raise ValueError("nx, ny, nz must be at least 16, 8, 8")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.u_in <= 0.0 or self.u_in >= 0.15:
            raise ValueError("u_in must be in (0, 0.15)")
        if self.re <= 0.0:
            raise ValueError("re must be > 0")
        if self.hull_length <= 0.0:
            raise ValueError("hull_length must be > 0")
        if self.use_triton_step and self.collision.upper() == "KBC":
            raise ValueError(
                "use_triton_step=True does not support KBC: the entropy "
                "bisection is only implemented in the PyTorch collision "
                "path; use CM or CUMULANT"
            )
        if self.force_every < 1:
            raise ValueError("force_every must be >= 1")
        if self.check_every < 1:
            raise ValueError("check_every must be >= 1")
        if self.compile_mode is not None:
            # Shared whitelist check (see tensorlbm.compile_utils for the
            # measured lessons behind the two allowed modes and the
            # cudagraphs rejection).
            validate_compile_mode(self.compile_mode)
            if self.use_triton_step:
                raise ValueError(
                    "compile_mode and use_triton_step are mutually exclusive"
                )
            if self.collision.upper() == "KBC":
                raise ValueError(
                    "compile_mode does not support KBC: the per-step entropy "
                    "bisection calls .item() and cannot be compiled"
                )

    @property
    def nu(self) -> float:
        """Kinematic viscosity (lattice units)."""
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        """Baseline shear relaxation time."""
        return 3.0 * self.nu + 0.5


# ---------------------------------------------------------------------------
# SGS-coupled collision
# ---------------------------------------------------------------------------


def _compute_sgs_tau_eff(
    f: torch.Tensor,
    config: SuboffCmkKbcConfig,
    tau_base: float,
) -> torch.Tensor:
    """Compute per-cell effective tau from the chosen SGS model.

    * **Smagorinsky** — uses the non-equilibrium stress Frobenius norm
      (LBM-specific, no velocity gradients needed).
    * **WALE / Vreman** — compute eddy viscosity from velocity gradients
      via central differences, then convert to tau_eff.
    """
    rho, ux, uy, uz = macroscopic3d(f)

    model = config.turbulence_model.lower()
    if model == "smagorinsky":
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        pi_norm = _neq_stress_norm_3d(f_neq)
        return _smagorinsky_tau(tau_base, pi_norm, rho, config.C_s)
    elif model == "wale":
        nu_t = _wale_nu_t_3d(ux, uy, uz, config.C_w)
        return _nu_t_to_tau_eff(tau_base, nu_t)
    elif model == "vreman":
        nu_t = _vreman_nu_t_3d(ux, uy, uz, config.C_V)
        return _nu_t_to_tau_eff(tau_base, nu_t)
    # Unreachable — validated in __post_init__
    raise ValueError(f"Unknown turbulence model: {config.turbulence_model}")


def _collide_with_sgs(
    f: torch.Tensor,
    config: SuboffCmkKbcConfig,
    tau_base: float,
) -> torch.Tensor:
    """Apply collision via ``collide_advanced_3d`` with SGS-coupled tau_eff.

    For CM and CUMULANT the per-cell tensor ``tau_eff`` is passed directly
    (both kernels use ``omega = 1/tau`` element-wise).

    For KBC the entropy bisection initialiser (``torch.full``) requires a
    Python scalar, so the spatial mean of ``tau_eff`` is used.
    """
    tau_eff = _compute_sgs_tau_eff(f, config, tau_base)

    if config.collision.upper() == "KBC":
        tau_scalar = float(tau_eff.mean().item())
        return collide_advanced_3d(
            config.lattice,
            config.collision,
            f,
            tau=tau_scalar,
        )
    # CM and CUMULANT accept tensor tau
    return collide_advanced_3d(
        config.lattice,
        config.collision,
        f,
        tau=tau_eff,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_suboff_cmk_kbc(
    config: SuboffCmkKbcConfig | None = None,
) -> dict[str, Any]:
    """Run SUBOFF bare-hull CM/CUMULANT/KBC × SGS and produce artifact.

    Returns a machine-readable dict with the fields specified by the task:
    ``Re, collision, turbulence_model, Ct, finite, steps_completed,
    boundary_type, device, reference_Ct, reference_source`` plus
    ``status="diagnostic_only"`` and ``physical_validation=False``.

    Loop-policy knobs (defaults preserve the historical behavior of the
    default PyTorch path): ``force_every=1`` records the momentum-exchange
    force on every step; ``force_every=N>1`` records it only on steps that
    are multiples of N, so ``force_time_series``/``ct_time_series`` become
    N-times sparser and ``Ct`` refers to the last sampled step.
    ``check_every`` (default 10) throttles the read-only isfinite watchdog
    to every N steps plus the final step — the historical runner checked
    every step; this only delays NaN/Inf fault detection, never the
    numerics of a finite run.
    """
    if config is None:
        config = SuboffCmkKbcConfig()

    device = torch.device(config.device)

    # --- 1. Build geometry ---
    cx = config.nx * 0.35
    cy = config.ny / 2.0
    cz = config.nz / 2.0
    solid, _stats = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=config.hull_length,
        device=str(device),
    )
    solid = solid.to(device)

    # Wetted area and dynamic pressure for Ct normalization
    wetted_area = _voxel_wetted_area(solid, 1.0)
    rho_lu = 1.0
    dynamic_pressure = 0.5 * rho_lu * config.u_in**2 * wetted_area

    # --- 2. Initialize populations ---
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    initial_mass = float(f.sum().item())

    tau_base = config.tau

    # --- 2c. Optional torch.compile step backend (opt-in; default off) ---
    # ``compile_mode=None`` (default) changes nothing below.  When set,
    # the *same* default-path chain (SGS collide, stream, momentum-
    # exchange force, far-field BC incl. bounce-back, mass correction)
    # is compiled as two whole-step graphs: one for plain steps and one
    # for steps that also run the every-10-step mass correction.  The
    # step counter stays outside the compiled code objects.
    compiled_step: tuple[Any, Any] | None = None
    if config.compile_mode is not None:
        if device.type != "cuda":
            raise RuntimeError(
                f"compile_mode requires a CUDA device; got {device}"
            )

        def _make_compiled_step(with_mass: bool):
            def _step(f: torch.Tensor):
                f = _collide_with_sgs(f, config, tau_base)
                f = stream3d(f)
                fx_c, fy_c, fz_c = compute_obstacle_forces_3d(f, solid)
                f = far_field_bc_3d(f, config.u_in, obstacle_mask=solid)
                if with_mass:
                    f = correct_mass3d(f, initial_mass)
                return f, fx_c, fy_c, fz_c

            # Shared wrapper (tensorlbm.compile_utils): validates the
            # mode and wraps with torch.compile.  Two guarded graphs are
            # built (plain / with mass correction) so the step index
            # never enters the compiled code objects.
            return compile_step(_step, config.compile_mode)

        compiled_step = (_make_compiled_step(False), _make_compiled_step(True))

    # --- 2b. Optional Triton-fused step backend (opt-in; default off) ---
    # With ``use_triton_step=False`` (default) nothing below changes: the
    # inline PyTorch chain in the loop is executed exactly as before.
    triton_state: Any = None
    solid_int8: torch.Tensor | None = None
    if config.use_triton_step:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise NotImplementedError(
                "use_triton_step=True is single-GPU only: the fused Triton "
                "step has no halo exchange nor force all-reduce.  Keep the "
                "default PyTorch path for distributed runs (or drive "
                "tensorlbm.triton_suboff_step_distributed explicitly)."
            )
        if device.type != "cuda":
            raise RuntimeError(
                f"use_triton_step=True requires a CUDA device; got {device}"
            )
        # Local import: Triton/CUDA must not become a hard dependency of
        # the default import path.
        from .backends.triton_backend import (
            TritonStepState,
            is_available as _triton_is_available,
            triton_suboff_step,
        )

        if not _triton_is_available():
            raise RuntimeError(
                "use_triton_step=True but tensorlbm.backends.triton_backend "
                "is unavailable (needs CUDA + the Triton kernels)"
            )
        solid_int8 = solid.to(torch.int8)
        triton_state = TritonStepState(f, device)

    # --- 3. Solver loop ---
    # Forces are accumulated device-side in ``force_buf`` (one row per
    # sampled step) and moved to the host once after the loop instead of
    # three per-step ``.item()`` round-trips.
    force_buf = torch.empty((config.n_steps, 3), device=device, dtype=f.dtype)
    force_rows = 0
    completed_steps = 0
    all_finite = True

    for step in range(1, config.n_steps + 1):
        force_this_step = config.use_triton_step  # fused step always returns forces
        if triton_state is not None:
            # Fused Triton path (PR4, opt-in): SGS tau_eff is still
            # computed in PyTorch with the very same
            # ``_compute_sgs_tau_eff`` as the default path; collide +
            # stream + wet-node bounce-back + Ladd force + 6-face BC +
            # mass correction run fused in one kernel launch (+ small
            # post-ops).  Collision family KBC was rejected in
            # ``__post_init__``.
            tau_eff = _compute_sgs_tau_eff(f, config, tau_base)
            f, fx_t, fy_t, fz_t = triton_suboff_step(
                f,
                solid_int8,
                config.u_in,
                step,
                mass_period=10,
                initial_mass=initial_mass,
                nu_lb=config.nu,
                Cs=config.C_s,
                collision=config.collision.upper(),
                state=triton_state,
                tau_eff=tau_eff,
            )
        else:
            if compiled_step is not None:
                # torch.compile backend (opt-in): the whole default-path
                # chain runs as one guarded graph.  Only the mass-
                # correction pattern (every 10 steps) selects between the
                # two compiled variants; forces are produced by the
                # graph itself (force_every=1 semantics for compiled
                # steps, matching the Triton step backend).
                f, fx_t, fy_t, fz_t = (
                    compiled_step[1](f) if step % 10 == 0 else compiled_step[0](f)
                )
                force_this_step = True
            else:
                # Collision with SGS
                f = _collide_with_sgs(f, config, tau_base)
                # Streaming (pull scheme)
                f = stream3d(f)
                # Force measurement (momentum exchange, before bounce-back).
                # ``force_every`` thins the sampling: on skipped steps the
                # full-grid reduction (and its host transfer) is skipped.
                if step % config.force_every == 0:
                    fx_t, fy_t, fz_t = compute_obstacle_forces_3d(f, solid)
                    force_this_step = True
                # Far-field BC (includes bounce-back on solid)
                f = far_field_bc_3d(f, config.u_in, obstacle_mask=solid)
                # Mass correction every 10 steps
                if step % 10 == 0:
                    f = correct_mass3d(f, initial_mass)

        # Accumulate this step's forces on the device (no host sync).
        if force_this_step:
            torch.stack((fx_t, fy_t, fz_t), out=force_buf[force_rows])
            force_rows += 1

        completed_steps = step
        # Read-only NaN/Inf watchdog, throttled to every ``check_every``
        # steps (plus the final step): only fault-detection latency
        # changes, never the populations.
        if step % config.check_every == 0 or step == config.n_steps:
            finite = bool(torch.isfinite(f).all().item())
            all_finite = all_finite and finite
            if not finite:
                break

    # --- 3b. One-shot host transfer + series construction ---
    # The sampled steps are exactly the multiples of ``force_every`` up
    # to ``completed_steps`` (Triton path and compiled path: every step,
    # because both step backends produce forces on every launch), in order.
    recorded_steps = (
        range(1, completed_steps + 1)
        if config.use_triton_step or compiled_step is not None
        else range(config.force_every, completed_steps + 1, config.force_every)
    )
    forces_host = force_buf[:force_rows].tolist()
    force_series: list[dict[str, Any]] = [
        {"step": s, "fx": row[0], "fy": row[1], "fz": row[2]}
        for s, row in zip(recorded_steps, forces_host)
    ]
    ct_series: list[dict[str, Any]] = [
        {"step": s, "ct": row[0] / dynamic_pressure if dynamic_pressure > 0 else 0.0}
        for s, row in zip(recorded_steps, forces_host)
    ]

    # --- 4. Build artifact ---
    ct_final = ct_series[-1]["ct"] if ct_series else 0.0
    artifact: dict[str, Any] = {
        "schema": "tensorlbm.suboff-cmk-kbc-sgs/v1",
        "status": "diagnostic_only",
        "physical_validation": False,
        "Re": config.re,
        "collision": config.collision.upper(),
        "turbulence_model": config.turbulence_model.lower(),
        "Ct": ct_final,
        "finite": all_finite,
        "steps_completed": completed_steps,
        "boundary_type": config.boundary_type,
        "device": "sdaa",
        "reference_Ct": config.reference_Ct,
        "reference_source": config.reference_source,
        "lattice": config.lattice,
        "grid": {"nx": config.nx, "ny": config.ny, "nz": config.nz},
        "u_in": config.u_in,
        "tau": tau_base,
        "nu": config.nu,
        "wetted_area": wetted_area,
        "dynamic_pressure": dynamic_pressure,
        "force_time_series": force_series,
        "ct_time_series": ct_series,
    }
    if config.use_triton_step:
        artifact["step_backend"] = "triton_fused"
    elif config.compile_mode is not None:
        artifact["step_backend"] = f"torch_compile:{config.compile_mode}"
    return artifact


def write_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Write the artifact as a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(artifact, sort_keys=True, indent=2),
        encoding="utf-8",
    )
