"""D3Q27 MRT component-level composition evidence probe (R1).

This module is a **cold-path evidence writer**, not a solver component.  It
executes five deterministic consistency checks directly against
``tensorlbm.d3q27.collide_mrt27`` and produces a machine-readable JSON
artifact that establishes **component-level** executable consistency evidence
for D3Q27 MRT.

The artifact explicitly documents that complete wall/geometry/boundary/output
composition remains WITHHELD by the general capability matrix
(``WITHHELD_D3Q27_COMPOSITION``, tier ``no_composition_evidence``).  This probe
does not change that disposition; it proves that the component-level evidence
the matrix already references is real, executable, and reproducible.

Checks (all D3Q27 MRT, float32, CPU, seed=2718, tau=0.8):

1. **equilibrium fixed point** — ``collide_mrt27(feq) ≈ feq`` (atol=1e-6)
2. **mass invariant** — ``sum(collide_mrt27(f)) == sum(f)`` (atol=1e-6)
3. **momentum invariant** — ``rho*u`` before == after (atol=1e-6)
4. **finite output** — all post-collision values are finite
5. **determinism** — two runs on identical clones produce bitwise-identical output
6. **source hash** — SHA-256 of ``collide_mrt27`` source matches the locked value

This module never imports or executes streaming, boundary conditions,
geometry masks, force observation, or output pipelines.  Those are the
WITHHELD composition aspects documented in the artifact.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from math import isfinite
from types import FunctionType
from typing import Any

import torch

from .d3q27 import collide_mrt27, equilibrium27, macroscopic27

D3Q27_COMPOSITION_EVIDENCE_VERSION = "d3q27-composition-evidence-r1"

#: SHA-256 of ``inspect.getsource(collide_mrt27)``.  Matches the value locked
#: by ``tests/test_d3q19_d3q27_mrt_consistency.py`` so that both evidence
#: artifacts bind to the same collision implementation.
EXPECTED_SOURCE_SHA256 = "4b1b55bf7b2aae49857f22d261e75666765764f5eeeb37050f105a17bafc10b5"

_PROBE_SHAPE = (2, 3, 4)
_PROBE_SEED = 2718
_PROBE_TAU = 0.8
_TOL = 1e-6


def _source_sha256(function: FunctionType) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _make_probe_state() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (feq, f_perturbed) with fixed seed, float32, CPU.

    ``feq`` is the equilibrium distribution for a random low-Mach state.
    ``f_perturbed`` adds a per-cell conservative perturbation whose density
    and all three raw momentum components are zero for the D3Q27 stencil,
    matching the construction in the consistency audit.
    """
    generator = torch.Generator(device="cpu").manual_seed(_PROBE_SEED)
    rho = 0.9 + 0.2 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    ux = -0.035 + 0.07 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    uy = -0.035 + 0.07 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    uz = -0.035 + 0.07 * torch.rand(_PROBE_SHAPE, generator=generator, dtype=torch.float32)
    feq = equilibrium27(rho, ux, uy, uz)

    perturbation = torch.zeros_like(feq)
    perturbation[0] = 2.0e-4
    perturbation[1] = -5.0e-5
    perturbation[2] = -5.0e-5
    perturbation[3] = -5.0e-5
    perturbation[4] = -5.0e-5
    return feq, feq + perturbation


def _check_equilibrium_fixed_point(feq: torch.Tensor) -> dict[str, Any]:
    out = collide_mrt27(feq, tau=_PROBE_TAU)
    delta = float(torch.abs(out - feq).max().item())
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "collide_mrt27(feq, tau=0.8) == feq within atol=1e-6",
    }


def _check_mass_invariant(f: torch.Tensor) -> dict[str, Any]:
    mass_before = float(f.sum().item())
    out = collide_mrt27(f, tau=_PROBE_TAU)
    mass_after = float(out.sum().item())
    delta = abs(mass_after - mass_before)
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "sum(collide_mrt27(f)) == sum(f) within atol=1e-6",
    }


def _check_momentum_invariant(f: torch.Tensor) -> dict[str, Any]:
    rho_b, ux_b, uy_b, uz_b = macroscopic27(f)
    momentum_before = torch.stack((rho_b * ux_b, rho_b * uy_b, rho_b * uz_b))
    out = collide_mrt27(f, tau=_PROBE_TAU)
    rho_a, ux_a, uy_a, uz_a = macroscopic27(out)
    momentum_after = torch.stack((rho_a * ux_a, rho_a * uy_a, rho_a * uz_a))
    delta = float(torch.abs(momentum_after - momentum_before).max().item())
    return {
        "status": "PASS" if delta <= _TOL else "FAIL",
        "max_abs_delta": delta,
        "tolerance": _TOL,
        "description": "rho*u before == rho*u after within atol=1e-6 (all 3 components)",
    }


def _check_finite_output(f: torch.Tensor) -> dict[str, Any]:
    out = collide_mrt27(f, tau=_PROBE_TAU)
    all_finite = bool(torch.isfinite(out).all().item())
    return {
        "status": "PASS" if all_finite else "FAIL",
        "all_finite": all_finite,
        "description": "all post-collision population values are finite",
    }


def _check_determinism(f: torch.Tensor) -> dict[str, Any]:
    first = collide_mrt27(f.clone(), tau=_PROBE_TAU)
    second = collide_mrt27(f.clone(), tau=_PROBE_TAU)
    bitwise_identical = bool(torch.equal(first, second))
    return {
        "status": "PASS" if bitwise_identical else "FAIL",
        "bitwise_identical": bitwise_identical,
        "description": "two runs on identical clones produce bitwise-identical output (torch.equal)",
    }


def _check_source_hash() -> dict[str, Any]:
    actual = _source_sha256(collide_mrt27)
    match = actual == EXPECTED_SOURCE_SHA256
    return {
        "status": "PASS" if match else "FAIL",
        "sha256": actual,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "description": "SHA-256 of inspect.getsource(collide_mrt27) matches locked value",
    }


_WITHHELD_COMPOSITION: dict[str, dict[str, str]] = {
    "wall_treatment": {
        "status": "WITHHELD",
        "reason": (
            "bounce_back_cells_27 + collide_mrt27 coupling not verified as a "
            "complete wall-treatment composition; no integrated mass/momentum "
            "balance test across bounce-back + MRT collision exists."
        ),
    },
    "geometry": {
        "status": "WITHHELD",
        "reason": (
            "static_solid_mask + collide_mrt27 not verified as a complete "
            "geometry composition; no obstacle-mask + MRT integration test "
            "with closed mass/momentum balance exists."
        ),
    },
    "boundary": {
        "status": "WITHHELD",
        "reason": (
            "Zou/He inlet/outlet (zou_he_inlet_velocity_27, zou_he_outlet_pressure_27) "
            "+ collide_mrt27 not verified as a complete boundary composition; "
            "no inlet/outlet + MRT mass conservation test exists."
        ),
    },
    "output": {
        "status": "WITHHELD",
        "reason": (
            "macroscopic27 (rho/velocity extraction) + collide_mrt27 not verified "
            "as a complete output composition; no end-to-end output accuracy test "
            "against a reference solution exists."
        ),
    },
    "streaming_collision_coupling": {
        "status": "WITHHELD",
        "reason": (
            "stream27 + collide_mrt27 coupling not verified as a complete "
            "time-stepping composition; no multi-step mass/momentum conservation "
            "test across stream + collide exists."
        ),
    },
    "force_observation": {
        "status": "WITHHELD",
        "reason": (
            "compute_obstacle_forces_27 + collide_mrt27 not verified as a "
            "complete force-observation composition; no force + MRT integration "
            "test with momentum balance exists."
        ),
    },
}

_CAPABILITY_MATRIX_XREF: dict[str, str] = {
    "general_capability_matrix": "WITHHELD",
    "evidence_tier": "no_composition_evidence",
    "reason_code": "WITHHELD_D3Q27_COMPOSITION",
    "advanced_collision_contract": "AVAILABLE",
}


def _compute_artifact_sha256(artifact: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the artifact (excluding the hash itself)."""
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_d3q27_mrt_composition_probe() -> dict[str, Any]:
    """Execute all five D3Q27 MRT component-level checks and return a JSON-ready artifact.

    The artifact is deterministic: calling this function twice produces
    identical measured values (the probe uses a fixed seed and CPU float32).

    Returns:
        A machine-readable dict with keys:
        - ``artifact_id``, ``version``, ``lattice``, ``collision``, ``entrypoint``
        - ``probe_config`` (shape, dtype, device, seed, tau)
        - ``checks`` (six checks, each with ``status`` and measured values)
        - ``component_evidence_tier`` ("component_contract" if all PASS)
        - ``withheld_composition`` (six WITHHELD aspects with reasons)
        - ``capability_matrix_cross_reference``
        - ``artifact_sha256`` (self-hash of the canonical JSON)
    """
    feq, f_perturbed = _make_probe_state()

    checks: dict[str, dict[str, Any]] = {
        "equilibrium_fixed_point": _check_equilibrium_fixed_point(feq),
        "mass_invariant": _check_mass_invariant(f_perturbed),
        "momentum_invariant": _check_momentum_invariant(f_perturbed),
        "finite_output": _check_finite_output(f_perturbed),
        "determinism": _check_determinism(f_perturbed),
        "source_hash": _check_source_hash(),
    }

    all_pass = all(c["status"] == "PASS" for c in checks.values())
    component_tier = "component_contract" if all_pass else "check_failure"

    artifact: dict[str, Any] = {
        "artifact_id": "d3q27-mrt-composition-evidence-r1",
        "version": D3Q27_COMPOSITION_EVIDENCE_VERSION,
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
        "component_evidence_tier": component_tier,
        "withheld_composition": dict(_WITHHELD_COMPOSITION),
        "capability_matrix_cross_reference": dict(_CAPABILITY_MATRIX_XREF),
    }
    artifact["artifact_sha256"] = _compute_artifact_sha256(artifact)
    return artifact


__all__ = [
    "D3Q27_COMPOSITION_EVIDENCE_VERSION",
    "EXPECTED_SOURCE_SHA256",
    "run_d3q27_mrt_composition_probe",
]
