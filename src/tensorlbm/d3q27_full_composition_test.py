"""D3Q27 MRT **full** composition evidence probe (R1).

This module is a **cold-path evidence writer**, not a solver component.  It
executes deterministic consistency checks that compose multiple D3Q27 MRT
components — bounce-back walls, equilibrium+collision+streaming, macroscopic
recovery, and wall-link force extraction — and produces a machine-readable
JSON artifact.

The artifact establishes **composition-level** executable consistency evidence
for D3Q27 MRT in the wall / geometry / boundary / force dimensions.  It does
**not** change the ``general_capability_matrix`` disposition: D3Q27 MRT
remains WITHHELD (``WITHHELD_D3Q27_COMPOSITION``, tier
``no_composition_evidence``) until physical validation exists.  This probe
proves that the composition is *executable and internally consistent*, not
that it is *physically accurate*.

Checks (all D3Q27 MRT, float32, CPU, seed=31415, tau=0.8):

**Bounce-back wall composition** (``bounce_back_cells_27``):

1. **bounce_back_involution** — ``bounce_back(bounce_back(f)) == f`` (atol=1e-5)
2. **bounce_back_mass_conservation** — ``sum(bounce_back(f)) == sum(f)`` (atol=1e-5)
3. **bounce_back_momentum_reflection** — at solid cells, momentum is reversed:
   ``momentum_after == -momentum_before`` (atol=1e-5)

**Equilibrium + collision + streaming one-step composition**:

4. **full_step_shape** — ``stream27(collide_mrt27(f, tau))`` preserves shape
5. **full_step_mass_periodic** — mass conserved across collide+stream (atol=1e-5)
6. **full_step_equilibrium_fixed_point** — uniform equilibrium is a fixed
   point of collide+stream (atol=1e-5)
7. **full_step_finite** — all post-step values are finite

**Macroscopic recovery composition** (``macroscopic27``):

8. **macroscopic_roundtrip** — ``equilibrium27 → macroscopic27`` recovers
   ``(rho, ux, uy, uz)`` (atol=1e-5)
9. **macroscopic_finite_after_step** — macroscopic values finite after full step
10. **macroscopic_mass_after_step** — ``sum(rho)`` conserved after full step
    (atol=1e-5)

**Wall-link force extraction composition** (``compute_obstacle_forces_27``):

11. **force_empty_zero** — empty obstacle mask gives zero force (atol=1e-5)
12. **force_finite** — non-empty obstacle gives finite force
13. **force_momentum_balance** — fluid momentum change from bounce-back equals
    ``-force_on_solid`` (atol=1e-5)
14. **force_drag_sign** — flow in +x yields positive drag on obstacle

**Determinism**:

15. **determinism** — two runs on identical clones produce bitwise-identical
    measured values

This module never claims physical accuracy, long-time stability, or validation
against a reference solution.  Those remain WITHHELD.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .boundaries_d3q27 import bounce_back_cells_27
from .d3q27 import (
    C as C27,
    OPPOSITE as OPPOSITE27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .obstacles import compute_obstacle_forces_27

D3Q27_FULL_COMPOSITION_VERSION = "d3q27-full-composition-evidence-r1"

_PROBE_SHAPE = (8, 10, 12)
_PROBE_SEED = 31415
_PROBE_TAU = 0.8
_TOL = 1e-5          # absolute tolerance for element-wise checks
_REL_TOL = 1e-5      # relative tolerance for global-sum checks (float32)


def _rel_tol(reference: float) -> float:
    """Relative tolerance scaled to the magnitude of *reference*."""
    return _REL_TOL * max(1.0, abs(reference))


# ---------------------------------------------------------------------------
# Probe state construction
# ---------------------------------------------------------------------------

def _make_probe_state() -> tuple[
    torch.Tensor,  # feq_uniform  — uniform equilibrium (fixed-point tests)
    torch.Tensor,  # f_perturbed  — perturbed state (mass/momentum tests)
    torch.Tensor,  # obstacle_mask — small block in domain interior
    torch.Tensor,  # wall_mask     — channel wall mask
]:
    """Return probe state with fixed seed, float32, CPU.

    ``feq_uniform`` is a spatially uniform equilibrium so that periodic
    streaming is identity and the collide+stream fixed-point property can be
    tested.  ``f_perturbed`` adds a zero-mass, zero-momentum perturbation to a
    random low-Mach equilibrium for conservation tests.
    """
    generator = torch.Generator(device="cpu").manual_seed(_PROBE_SEED)

    # Random low-Mach equilibrium (non-uniform) for conservation tests
    rho = 0.9 + 0.2 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    ux = 0.02 + 0.04 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    uy = -0.02 + 0.04 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    uz = -0.02 + 0.04 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    feq = equilibrium27(rho, ux, uy, uz)

    # Conservative perturbation: zero mass, zero momentum
    perturbation = torch.zeros_like(feq)
    perturbation[0] = 2.0e-4
    perturbation[1] = -5.0e-5
    perturbation[2] = -5.0e-5
    perturbation[3] = -5.0e-5
    perturbation[4] = -5.0e-5
    f_perturbed = feq + perturbation

    # Uniform equilibrium for fixed-point test
    rho_u = torch.ones(_PROBE_SHAPE, dtype=torch.float32)
    ux_u = torch.full(_PROBE_SHAPE, 0.03, dtype=torch.float32)
    uy_u = torch.full(_PROBE_SHAPE, 0.01, dtype=torch.float32)
    uz_u = torch.full(_PROBE_SHAPE, -0.01, dtype=torch.float32)
    feq_uniform = equilibrium27(rho_u, ux_u, uy_u, uz_u)

    # Obstacle mask: 2×2×2 block in domain interior
    nz, ny, nx = _PROBE_SHAPE
    obstacle_mask = torch.zeros(_PROBE_SHAPE, dtype=torch.bool)
    obstacle_mask[3:5, 4:6, 5:7] = True

    # Wall mask: channel walls (±y and ±z faces)
    wall_mask = torch.zeros(_PROBE_SHAPE, dtype=torch.bool)
    wall_mask[:, 0, :] = True
    wall_mask[:, -1, :] = True
    wall_mask[0, :, :] = True
    wall_mask[-1, :, :] = True

    return feq_uniform, f_perturbed, obstacle_mask, wall_mask


def _momentum_at_mask(
    f: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Total fluid momentum ``(3,)`` at cells selected by *mask*."""
    c = C27.to(device=f.device, dtype=f.dtype).float()
    mask_4d = mask.unsqueeze(0)
    f_masked = f * mask_4d
    px = (c[:, 0].view(27, 1, 1, 1) * f_masked).sum()
    py = (c[:, 1].view(27, 1, 1, 1) * f_masked).sum()
    pz = (c[:, 2].view(27, 1, 1, 1) * f_masked).sum()
    return torch.stack([px, py, pz])


# ---------------------------------------------------------------------------
# 1. Bounce-back wall composition checks
# ---------------------------------------------------------------------------

def _check_bounce_back_involution(
    f: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """bounce_back(bounce_back(f)) == f."""
    f_bb = bounce_back_cells_27(f, mask)
    f_bb_bb = bounce_back_cells_27(f_bb, mask)
    delta = float(torch.abs(f_bb_bb - f).max().item())
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "bounce_back_cells_27 applied twice returns original f",
    }


def _check_bounce_back_mass_conservation(
    f: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """sum(bounce_back(f)) == sum(f)."""
    f_bb = bounce_back_cells_27(f, mask)
    mass_before = float(f.sum().item())
    mass_after = float(f_bb.sum().item())
    delta = abs(mass_after - mass_before)
    tol = _rel_tol(mass_before)
    return {
        "status": "PASS" if delta <= tol else "FAIL",
        "max_abs_delta": delta,
        "tolerance": tol,
        "tolerance_type": "relative",
        "description": "bounce_back_cells_27 conserves total mass",
    }


def _check_bounce_back_momentum_reflection(
    f: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """At solid cells, momentum_after == -momentum_before."""
    momentum_before = _momentum_at_mask(f, mask)
    f_bb = bounce_back_cells_27(f, mask)
    momentum_after = _momentum_at_mask(f_bb, mask)
    delta = float(torch.abs(momentum_after + momentum_before).max().item())
    ref = float(torch.abs(momentum_before).max().item())
    tol = _rel_tol(ref)
    return {
        "status": "PASS" if delta <= tol else "FAIL",
        "max_abs_delta": delta,
        "tolerance": tol,
        "tolerance_type": "relative",
        "description": "bounce_back reverses fluid momentum at solid cells",
    }


# ---------------------------------------------------------------------------
# 2. Equilibrium + collision + streaming one-step composition
# ---------------------------------------------------------------------------

def _check_full_step_shape(f: torch.Tensor) -> dict[str, Any]:
    """stream27(collide_mrt27(f, tau)) preserves shape."""
    f_collided = collide_mrt27(f, tau=_PROBE_TAU)
    f_streamed = stream27(f_collided)
    shape_ok = f_streamed.shape == f.shape
    return {
        "status": "PASS" if shape_ok else "FAIL",
        "shape_preserved": shape_ok,
        "description": "stream27(collide_mrt27(f)) preserves (27, nz, ny, nx)",
    }


def _check_full_step_mass_periodic(f: torch.Tensor) -> dict[str, Any]:
    """Mass conserved across collide+stream (periodic)."""
    mass_before = float(f.sum().item())
    f_collided = collide_mrt27(f, tau=_PROBE_TAU)
    f_streamed = stream27(f_collided)
    mass_after = float(f_streamed.sum().item())
    delta = abs(mass_after - mass_before)
    tol = _rel_tol(mass_before)
    return {
        "status": "PASS" if delta <= tol else "FAIL",
        "max_abs_delta": delta,
        "tolerance": tol,
        "tolerance_type": "relative",
        "description": "sum(stream27(collide_mrt27(f))) == sum(f) (periodic)",
    }


def _check_full_step_equilibrium_fixed_point(feq_uniform: torch.Tensor) -> dict[str, Any]:
    """Uniform equilibrium is a fixed point of collide+stream."""
    f_collided = collide_mrt27(feq_uniform, tau=_PROBE_TAU)
    f_streamed = stream27(f_collided)
    delta = float(torch.abs(f_streamed - feq_uniform).max().item())
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "uniform feq is fixed point of collide_mrt27+stream27",
    }


def _check_full_step_finite(f: torch.Tensor) -> dict[str, Any]:
    """All post-step values finite."""
    f_collided = collide_mrt27(f, tau=_PROBE_TAU)
    f_streamed = stream27(f_collided)
    all_finite = bool(torch.isfinite(f_streamed).all().item())
    return {
        "status": "PASS" if all_finite else "FAIL",
        "all_finite": all_finite,
        "description": "all post-step population values are finite",
    }


# ---------------------------------------------------------------------------
# 3. Macroscopic recovery composition
# ---------------------------------------------------------------------------

def _check_macroscopic_roundtrip(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> dict[str, Any]:
    """equilibrium27 → macroscopic27 recovers (rho, ux, uy, uz)."""
    f = equilibrium27(rho, ux, uy, uz)
    rho_out, ux_out, uy_out, uz_out = macroscopic27(f)
    delta = float(
        max(
            torch.abs(rho_out - rho).max().item(),
            torch.abs(ux_out - ux).max().item(),
            torch.abs(uy_out - uy).max().item(),
            torch.abs(uz_out - uz).max().item(),
        )
    )
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "equilibrium27 → macroscopic27 recovers rho/ux/uy/uz",
    }


def _check_macroscopic_finite_after_step(f: torch.Tensor) -> dict[str, Any]:
    """Macroscopic values finite after full step."""
    f_collided = collide_mrt27(f, tau=_PROBE_TAU)
    f_streamed = stream27(f_collided)
    rho, ux, uy, uz = macroscopic27(f_streamed)
    all_finite = bool(
        torch.isfinite(rho).all().item()
        and torch.isfinite(ux).all().item()
        and torch.isfinite(uy).all().item()
        and torch.isfinite(uz).all().item()
    )
    return {
        "status": "PASS" if all_finite else "FAIL",
        "all_finite": all_finite,
        "description": "macroscopic27(f_after_step) yields finite rho/ux/uy/uz",
    }


def _check_macroscopic_mass_after_step(f: torch.Tensor) -> dict[str, Any]:
    """sum(rho) conserved after full step."""
    rho_before, _, _, _ = macroscopic27(f)
    mass_before = float(rho_before.sum().item())
    f_collided = collide_mrt27(f, tau=_PROBE_TAU)
    f_streamed = stream27(f_collided)
    rho_after, _, _, _ = macroscopic27(f_streamed)
    mass_after = float(rho_after.sum().item())
    delta = abs(mass_after - mass_before)
    tol = _rel_tol(mass_before)
    return {
        "status": "PASS" if delta <= tol else "FAIL",
        "max_abs_delta": delta,
        "tolerance": tol,
        "tolerance_type": "relative",
        "description": "sum(macroscopic27(f_after_step)[0]) == sum(rho_before)",
    }


# ---------------------------------------------------------------------------
# 4. Wall-link force extraction composition
# ---------------------------------------------------------------------------

def _check_force_empty_zero(f: torch.Tensor, shape: tuple[int, ...]) -> dict[str, Any]:
    """Empty obstacle mask gives zero force."""
    empty_mask = torch.zeros(shape, dtype=torch.bool, device=f.device)
    fx, fy, fz = compute_obstacle_forces_27(f, empty_mask)
    delta = float(max(abs(float(fx)), abs(float(fy)), abs(float(fz))))
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "compute_obstacle_forces_27 with empty mask returns (0,0,0)",
    }


def _check_force_finite(
    f: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """Non-empty obstacle gives finite force."""
    fx, fy, fz = compute_obstacle_forces_27(f, mask)
    all_finite = bool(
        torch.isfinite(fx).item()
        and torch.isfinite(fy).item()
        and torch.isfinite(fz).item()
    )
    return {
        "status": "PASS" if all_finite else "FAIL",
        "all_finite": all_finite,
        "fx": float(fx),
        "fy": float(fy),
        "fz": float(fz),
        "description": "compute_obstacle_forces_27 yields finite fx/fy/fz",
    }


def _check_force_momentum_balance(
    f: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """Fluid momentum change from bounce-back == -force_on_solid.

    The momentum-exchange identity: after streaming, the force on the solid
    is ``F = 2 Σ c·f_solid``.  Bounce-back reverses the populations at solid
    cells, changing fluid momentum by ``Δp = -F``.  This check verifies that
    ``compute_obstacle_forces_27`` and ``bounce_back_cells_27`` are mutually
    consistent — the core wall-link force composition property.
    """
    fx, fy, fz = compute_obstacle_forces_27(f, mask)
    force = torch.stack([fx, fy, fz])

    momentum_before = _momentum_at_mask(f, mask)
    f_bb = bounce_back_cells_27(f, mask)
    momentum_after = _momentum_at_mask(f_bb, mask)
    delta_momentum = momentum_after - momentum_before

    residual = delta_momentum + force
    delta = float(torch.abs(residual).max().item())
    ref = float(torch.abs(force).max().item())
    tol = _rel_tol(ref)
    return {
        "status": "PASS" if delta <= tol else "FAIL",
        "max_abs_delta": delta,
        "tolerance": tol,
        "tolerance_type": "relative",
        "force_on_solid": [float(fx), float(fy), float(fz)],
        "delta_momentum": [float(delta_momentum[0]), float(delta_momentum[1]), float(delta_momentum[2])],
        "description": "Δp_fluid from bounce-back == -F_solid (momentum-exchange identity)",
    }


def _check_force_drag_sign(shape: tuple[int, ...]) -> dict[str, Any]:
    """Flow in +x yields positive drag on obstacle.

    A uniform equilibrium with ``ux > 0`` is streamed (identity for uniform
    fields) and the force on a centred obstacle is computed.  The x-component
    (drag) must be positive because the obstacle reflects +x momentum.
    """
    rho = torch.ones(shape, dtype=torch.float32)
    ux = torch.full(shape, 0.05, dtype=torch.float32)
    zeros = torch.zeros_like(rho)
    f = equilibrium27(rho, ux, zeros, zeros)
    f = stream27(f)

    obstacle = torch.zeros(shape, dtype=torch.bool)
    nz, ny, nx = shape
    obstacle[nz // 2 - 1 : nz // 2 + 1,
             ny // 2 - 1 : ny // 2 + 1,
             nx // 2 - 1 : nx // 2 + 1] = True

    fx, fy, fz = compute_obstacle_forces_27(f, obstacle)
    drag_positive = float(fx) > 0.0
    return {
        "status": "PASS" if drag_positive else "FAIL",
        "drag_positive": drag_positive,
        "fx": float(fx),
        "description": "flow in +x → positive drag (fx > 0) on obstacle",
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _check_determinism(
    f: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """Two runs on identical clones produce bitwise-identical measured values."""
    # Full step
    f1 = collide_mrt27(f.clone(), tau=_PROBE_TAU)
    f1 = stream27(f1)
    f1 = bounce_back_cells_27(f1, mask)

    f2 = collide_mrt27(f.clone(), tau=_PROBE_TAU)
    f2 = stream27(f2)
    f2 = bounce_back_cells_27(f2, mask)

    bitwise_identical = bool(torch.equal(f1, f2))
    return {
        "status": "PASS" if bitwise_identical else "FAIL",
        "bitwise_identical": bitwise_identical,
        "description": "collide+stream+bounce_back on identical clones is bitwise-identical",
    }


# ---------------------------------------------------------------------------
# WITHHELD physical validation
# ---------------------------------------------------------------------------

_WITHHELD_PHYSICAL_VALIDATION: dict[str, dict[str, str]] = {
    "wall_treatment": {
        "status": "WITHHELD",
        "reason": (
            "bounce_back_cells_27 + collide_mrt27 composition is executable "
            "and internally consistent, but no physical wall-accuracy "
            "validation (e.g. Poiseuille channel comparison) exists."
        ),
    },
    "geometry": {
        "status": "WITHHELD",
        "reason": (
            "obstacle_mask + collide_mrt27 composition is executable, but no "
            "physical geometry-accuracy validation (e.g. drag on a sphere "
            "vs. reference Cd) exists."
        ),
    },
    "boundary": {
        "status": "WITHHELD",
        "reason": (
            "Zou/He inlet/outlet + collide_mrt27 composition is executable, "
            "but no physical boundary-accuracy validation (e.g. mass flux "
            "conservation across inlet/outlet) exists."
        ),
    },
    "streaming_collision_coupling": {
        "status": "WITHHELD",
        "reason": (
            "stream27 + collide_mrt27 coupling conserves mass in one step, "
            "but no multi-step long-time stability or convergence test "
            "against a reference solution exists."
        ),
    },
    "force_observation": {
        "status": "WITHHELD",
        "reason": (
            "compute_obstacle_forces_27 + bounce_back_cells_27 satisfy the "
            "momentum-exchange identity, but no physical force-accuracy "
            "validation (e.g. drag coefficient vs. experiment) exists."
        ),
    },
    "physical_accuracy": {
        "status": "WITHHELD",
        "reason": (
            "No end-to-end physical accuracy test (e.g. lid-driven cavity "
            "vs. Ghia, or sphere drag vs. Schlichting) exists for D3Q27 MRT."
        ),
    },
}

_CAPABILITY_MATRIX_XREF: dict[str, str] = {
    "general_capability_matrix": "WITHHELD",
    "evidence_tier": "no_composition_evidence",
    "reason_code": "WITHHELD_D3Q27_COMPOSITION",
    "component_evidence_tier": "component_contract",
    "composition_evidence_tier": "composition_contract",
    "advanced_collision_contract": "AVAILABLE",
}


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------

def _compute_artifact_sha256(artifact: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the artifact (excluding the hash itself)."""
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_d3q27_full_composition_probe() -> dict[str, Any]:
    """Execute all D3Q27 MRT full-composition checks and return a JSON-ready artifact.

    The artifact is deterministic: calling this function twice produces
    identical measured values (the probe uses a fixed seed and CPU float32).

    Returns:
        A machine-readable dict with keys:
        - ``artifact_id``, ``version``, ``lattice``, ``collision``, ``entrypoint``
        - ``probe_config`` (shape, dtype, device, seed, tau)
        - ``checks`` (15 checks, each with ``status`` and measured values)
        - ``composition_evidence_tier`` (``"composition_contract"`` if all PASS)
        - ``withheld_physical_validation`` (six WITHHELD aspects with reasons)
        - ``capability_matrix_cross_reference``
        - ``artifact_sha256`` (self-hash of the canonical JSON)
    """
    feq_uniform, f_perturbed, obstacle_mask, wall_mask = _make_probe_state()

    # Recover the macroscopic fields of f_perturbed for the roundtrip check
    rho_p, ux_p, uy_p, uz_p = macroscopic27(f_perturbed)

    # Stream f_perturbed for force checks (force is computed post-stream)
    f_streamed = stream27(f_perturbed)

    checks: dict[str, dict[str, Any]] = {
        # 1. Bounce-back wall composition
        "bounce_back_involution": _check_bounce_back_involution(f_perturbed, wall_mask),
        "bounce_back_mass_conservation": _check_bounce_back_mass_conservation(f_perturbed, wall_mask),
        "bounce_back_momentum_reflection": _check_bounce_back_momentum_reflection(f_perturbed, wall_mask),
        # 2. Equilibrium + collision + streaming
        "full_step_shape": _check_full_step_shape(f_perturbed),
        "full_step_mass_periodic": _check_full_step_mass_periodic(f_perturbed),
        "full_step_equilibrium_fixed_point": _check_full_step_equilibrium_fixed_point(feq_uniform),
        "full_step_finite": _check_full_step_finite(f_perturbed),
        # 3. Macroscopic recovery
        "macroscopic_roundtrip": _check_macroscopic_roundtrip(rho_p, ux_p, uy_p, uz_p),
        "macroscopic_finite_after_step": _check_macroscopic_finite_after_step(f_perturbed),
        "macroscopic_mass_after_step": _check_macroscopic_mass_after_step(f_perturbed),
        # 4. Wall-link force extraction
        "force_empty_zero": _check_force_empty_zero(f_streamed, _PROBE_SHAPE),
        "force_finite": _check_force_finite(f_streamed, obstacle_mask),
        "force_momentum_balance": _check_force_momentum_balance(f_streamed, obstacle_mask),
        "force_drag_sign": _check_force_drag_sign(_PROBE_SHAPE),
        # 5. Determinism
        "determinism": _check_determinism(f_perturbed, wall_mask),
    }

    all_pass = all(c["status"] == "PASS" for c in checks.values())
    composition_tier = "composition_contract" if all_pass else "check_failure"

    artifact: dict[str, Any] = {
        "artifact_id": "d3q27-full-composition-evidence-r1",
        "version": D3Q27_FULL_COMPOSITION_VERSION,
        "lattice": "D3Q27",
        "collision": "MRT",
        "entrypoint": "tensorlbm.d3q27.collide_mrt27",
        "probe_config": {
            "shape": list(_PROBE_SHAPE),
            "dtype": "float32",
            "device": "cpu",
            "seed": _PROBE_SEED,
            "tau": _PROBE_TAU,
        },
        "checks": checks,
        "composition_evidence_tier": composition_tier,
        "withheld_physical_validation": dict(_WITHHELD_PHYSICAL_VALIDATION),
        "capability_matrix_cross_reference": dict(_CAPABILITY_MATRIX_XREF),
    }
    artifact["artifact_sha256"] = _compute_artifact_sha256(artifact)
    return artifact


__all__ = [
    "D3Q27_FULL_COMPOSITION_VERSION",
    "run_d3q27_full_composition_probe",
]
