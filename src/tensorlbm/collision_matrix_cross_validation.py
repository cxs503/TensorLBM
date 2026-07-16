"""Reproducible numerical-consistency evidence for the advanced-collision matrix.

This diagnostic runner deliberately calls only the kernels that the public
advanced-collision contract marks AVAILABLE.  It records small deterministic
single-node probes; it does not make accuracy, physics, or ranking claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

import torch

from .advanced_collision_contract import collision_capability_matrix, collide_advanced_3d
from .d3q19 import C as C19
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import C as C27
from .d3q27 import equilibrium27, macroscopic27

EVIDENCE_SCHEMA_VERSION = "tensorlbm.collision-matrix-cross-validation/v1"
_PROBE_TOLERANCE = 2.0e-5
_PROBE_SHAPE = (2, 2, 3)


@dataclass(frozen=True)
class SourceProvenance:
    """Exact source bytes used by one runnable consistency probe."""

    module: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ConsistencyProbeResult:
    """One bounded numerical-consistency observation, not a validation claim."""

    name: Literal[
        "equilibrium_fixed_point",
        "mass_momentum_collision_invariants",
        "finite_non_equilibrium",
    ]
    status: Literal["PASS", "FAIL"]
    max_abs_error: float
    tolerance: float
    finite: bool
    detail: str


@dataclass(frozen=True)
class CollisionCombinationEvidence:
    """Evidence or explicit withholding for one capability-matrix cell."""

    lattice: Literal["D3Q19", "D3Q27"]
    family: Literal["BGK", "TRT", "RLBM", "MRT", "CM", "KBC", "CUMULANT"]
    status: Literal["PASS", "FAIL", "SKIPPED_WITHHELD"]
    entrypoint: str | None
    withheld_reason: str | None
    probes: tuple[ConsistencyProbeResult, ...]
    source_provenance: tuple[SourceProvenance, ...]


@dataclass(frozen=True)
class CollisionMatrixCrossValidationEvidence:
    """Machine-consumable first-round evidence for the collision capability matrix."""

    schema_version: str
    runner: str
    torch_version: str
    dtype: str
    device: str
    probe_shape: tuple[int, int, int]
    combinations: tuple[CollisionCombinationEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_provenance(lattice: str) -> tuple[SourceProvenance, ...]:
    package_root = Path(__file__).resolve().parent
    names = [
        "collision_matrix_cross_validation.py",
        "advanced_collision_contract.py",
        "d3q19.py",
        "solver3d.py",
    ]
    if lattice == "D3Q27":
        names = [
            "collision_matrix_cross_validation.py",
            "advanced_collision_contract.py",
            "d3q27.py",
        ]
    return tuple(
        SourceProvenance(
            module=f"tensorlbm.{path.stem}",
            path=str(path.relative_to(package_root.parent.parent)),
            sha256=sha256(path.read_bytes()).hexdigest(),
        )
        for name in names
        for path in [package_root / name]
    )


def _max_abs(value: torch.Tensor) -> float:
    return float(value.abs().max().item())


def _state(lattice: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = torch.tensor(
        [[[1.00, 1.01, 0.99], [1.02, 0.98, 1.00]], [[1.01, 0.99, 1.00], [1.00, 1.02, 0.98]]],
        dtype=torch.float32,
    )
    ux = torch.full(_PROBE_SHAPE, 0.03125)
    uy = torch.full(_PROBE_SHAPE, -0.0175)
    uz = torch.full(_PROBE_SHAPE, 0.0125)
    equilibrium = equilibrium3d if lattice == "D3Q19" else equilibrium27
    return equilibrium(rho, ux, uy, uz), rho, ux, uy, uz


def _macroscopic(lattice: str, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return macroscopic3d(f) if lattice == "D3Q19" else macroscopic27(f)


def _non_equilibrium(lattice: str) -> torch.Tensor:
    f, _, _, _, _ = _state(lattice)
    directions = (C19 if lattice == "D3Q19" else C27).to(dtype=f.dtype)
    coefficient = torch.linspace(-1.0, 1.0, f.shape[0], dtype=f.dtype).view(-1, 1, 1, 1)
    # Adding a density-scaled directional perturbation makes a finite,
    # deterministic non-equilibrium input without changing runner state.
    perturbation = 1.0e-4 * coefficient * (1.0 + directions[:, 0].view(-1, 1, 1, 1) ** 2)
    return f + perturbation


def _probe_available_kernel(lattice: Literal["D3Q19", "D3Q27"], family: str = "MRT") -> tuple[ConsistencyProbeResult, ...]:
    """Run deterministic consistency probes for one AVAILABLE kernel cell."""
    equilibrium_state, _, _, _, _ = _state(lattice)
    equilibrium_post = collide_advanced_3d(lattice, family, equilibrium_state, tau=0.8)
    fixed_error = _max_abs(equilibrium_post - equilibrium_state)

    non_equilibrium = _non_equilibrium(lattice)
    before = _macroscopic(lattice, non_equilibrium)
    after_population = collide_advanced_3d(lattice, family, non_equilibrium, tau=0.8)
    after = _macroscopic(lattice, after_population)
    invariant_error = max(_max_abs(after[index] - before[index]) for index in range(4))

    finite = bool(torch.isfinite(after_population).all().item())
    probes = (
        ConsistencyProbeResult(
            "equilibrium_fixed_point", "PASS" if fixed_error <= _PROBE_TOLERANCE else "FAIL",
            fixed_error, _PROBE_TOLERANCE, bool(torch.isfinite(equilibrium_post).all().item()),
            "post-collision equilibrium minus input equilibrium",
        ),
        ConsistencyProbeResult(
            "mass_momentum_collision_invariants", "PASS" if invariant_error <= _PROBE_TOLERANCE else "FAIL",
            invariant_error, _PROBE_TOLERANCE, finite,
            "maximum difference across rho, ux, uy, uz recovered before and after collision",
        ),
        ConsistencyProbeResult(
            "finite_non_equilibrium", "PASS" if finite else "FAIL", 0.0 if finite else float("inf"),
            0.0, finite, "all post-collision non-equilibrium populations are finite",
        ),
    )
    return probes


def run_collision_matrix_cross_validation() -> CollisionMatrixCrossValidationEvidence:
    """Run deterministic probes for AVAILABLE MRT cells and record withheld cells.

    The result contains the whole D3Q19/D3Q27 × MRT/CM/KBC matrix, while only
    available cells execute kernels.  Withheld cells are intentionally skipped,
    never represented as a numerical failure.
    """
    matrix = collision_capability_matrix()
    combinations: list[CollisionCombinationEvidence] = []
    for lattice in ("D3Q19", "D3Q27"):
        for family in ("MRT", "CM", "KBC", "CUMULANT"):
            capability = matrix[lattice][family]
            if not capability.available:
                combinations.append(CollisionCombinationEvidence(
                    lattice, family, "SKIPPED_WITHHELD", capability.entrypoint,
                    capability.status, (), (),
                ))
                continue
            probes = _probe_available_kernel(lattice, family)
            combinations.append(CollisionCombinationEvidence(
                lattice, family, "PASS" if all(item.status == "PASS" for item in probes) else "FAIL",
                capability.entrypoint, None, probes, _source_provenance(lattice),
            ))
    return CollisionMatrixCrossValidationEvidence(
        EVIDENCE_SCHEMA_VERSION,
        "tensorlbm.collision_matrix_cross_validation.run_collision_matrix_cross_validation",
        torch.__version__, "torch.float32", "cpu", _PROBE_SHAPE, tuple(combinations),
    )


def write_collision_matrix_evidence(path: str | Path) -> Path:
    """Write canonical JSON evidence with a hash of its hash-free payload."""
    output = Path(path)
    payload = run_collision_matrix_cross_validation().to_dict()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload["canonical_payload_sha256"] = sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


__all__ = [
    "EVIDENCE_SCHEMA_VERSION", "SourceProvenance", "ConsistencyProbeResult",
    "CollisionCombinationEvidence", "CollisionMatrixCrossValidationEvidence",
    "run_collision_matrix_cross_validation", "write_collision_matrix_evidence",
]
