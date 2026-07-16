"""Honest common contract for three-dimensional advanced collision families.

This module deliberately distinguishes callable, validated kernels from
experimental approximations elsewhere in the package.  In particular,
``advanced_collision.collide_cascaded_d3q27`` is a second-order regularized
reconstruction (its higher central moments are not implemented) and its KBC
routine uses a caller-supplied blend rather than an entropy solve.  They are
therefore *not* advertised here as CM/KBC kernels.

BGK, TRT, and RLBM are registered as AVAILABLE for both D3Q19 and D3Q27
because validated, contract-tested kernels exist for every combination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .d3q27 import collide_bgk27, collide_mrt27, collide_rlbm27, collide_trt27
from .solver3d import collide_bgk3d, collide_mrt3d, collide_rlbm3d, collide_trt3d

LatticeName = Literal["D3Q19", "D3Q27"]
CollisionFamily = Literal["BGK", "TRT", "RLBM", "MRT", "CM", "KBC"]

WITHHELD_NO_D3Q19_CM_KERNEL = "WITHHELD_NO_D3Q19_CM_KERNEL"
WITHHELD_NO_D3Q19_KBC_KERNEL = "WITHHELD_NO_D3Q19_KBC_KERNEL"
WITHHELD_NO_D3Q27_CM_KERNEL = "WITHHELD_NO_D3Q27_CM_KERNEL"
WITHHELD_NO_D3Q27_KBC_KERNEL = "WITHHELD_NO_D3Q27_KBC_KERNEL"


class CollisionKernelWithheldError(NotImplementedError):
    """Raised when a requested family has no validated kernel for a lattice."""


@dataclass(frozen=True)
class CollisionCapability:
    """Availability and provenance of one lattice/family combination."""

    available: bool
    entrypoint: str | None
    status: str
    note: str


def collision_capability_matrix() -> dict[LatticeName, dict[CollisionFamily, CollisionCapability]]:
    """Return the audited D3Q19/D3Q27 collision capability matrix.

    ``available`` means an executable kernel exists under this common contract,
    not merely that a similarly named experimental implementation is present.
    """
    return {
        "D3Q19": {
            "BGK": CollisionCapability(True, "tensorlbm.solver3d.collide_bgk3d", "AVAILABLE", "Single-relaxation-time BGK; conserved moments are exact."),
            "TRT": CollisionCapability(True, "tensorlbm.solver3d.collide_trt3d", "AVAILABLE", "Two-relaxation-time with magic-parameter Λ; symmetric/anti-symmetric split via OPPOSITE."),
            "RLBM": CollisionCapability(True, "tensorlbm.solver3d.collide_rlbm3d", "AVAILABLE", "Regularized BGK; non-equilibrium projected onto 2nd-order Hermite subspace."),
            "MRT": CollisionCapability(True, "tensorlbm.solver3d.collide_mrt3d", "AVAILABLE", "19x19 MRT transform; conserved rows are explicit."),
            "CM": CollisionCapability(False, None, WITHHELD_NO_D3Q19_CM_KERNEL, "No standalone validated D3Q19 central-moment kernel."),
            "KBC": CollisionCapability(False, None, WITHHELD_NO_D3Q19_KBC_KERNEL, "No standalone entropy-solved D3Q19 KBC kernel."),
        },
        "D3Q27": {
            "BGK": CollisionCapability(True, "tensorlbm.d3q27.collide_bgk27", "AVAILABLE", "Single-relaxation-time BGK; conserved moments are exact."),
            "TRT": CollisionCapability(True, "tensorlbm.d3q27.collide_trt27", "AVAILABLE", "Two-relaxation-time with magic-parameter Λ; symmetric/anti-symmetric split via D3Q27 OPPOSITE (includes corner directions)."),
            "RLBM": CollisionCapability(True, "tensorlbm.d3q27.collide_rlbm27", "AVAILABLE", "Regularized BGK; 2nd-order Hermite projection with D3Q27 4th-order-isotropic weights."),
            "MRT": CollisionCapability(True, "tensorlbm.d3q27.collide_mrt27", "AVAILABLE", "27x27 full-rank Gram-Schmidt moment transform with explicit inverse."),
            "CM": CollisionCapability(False, None, WITHHELD_NO_D3Q27_CM_KERNEL, "Existing cascaded routine is regularized second-order reconstruction; higher central moments are not implemented."),
            "KBC": CollisionCapability(False, None, WITHHELD_NO_D3Q27_KBC_KERNEL, "Existing KBC-labelled routine uses a prescribed blend and has no entropy minimization."),
        },
    }


def _normalise_lattice(lattice: str) -> LatticeName:
    value = lattice.upper()
    if value not in {"D3Q19", "D3Q27"}:
        raise ValueError("lattice must be 'D3Q19' or 'D3Q27'")
    return value  # type: ignore[return-value]


def _normalise_family(family: str) -> CollisionFamily:
    value = family.upper().replace("-", "_")
    aliases = {
        "BGK": "BGK", "SRT": "BGK",
        "TRT": "TRT", "TWO_RELAXATION_TIME": "TRT",
        "RLBM": "RLBM", "REGULARIZED": "RLBM", "REGULARISED": "RLBM",
        "MRT": "MRT",
        "CM": "CM", "CASCADED": "CM",
        "KBC": "KBC", "ENTROPIC_KBC": "KBC",
    }
    if value not in aliases:
        raise ValueError("family must be BGK/SRT, TRT, RLBM/regularized, MRT, CM/cascaded, or KBC/entropic_kbc")
    return aliases[value]  # type: ignore[return-value]


def _select_kernel(lattice_name: LatticeName, family_name: CollisionFamily) -> Callable[..., torch.Tensor]:
    """Return the validated kernel callable for a lattice/family pair."""
    table: dict[LatticeName, dict[str, Callable[..., torch.Tensor]]] = {
        "D3Q19": {
            "BGK": collide_bgk3d,
            "TRT": collide_trt3d,
            "RLBM": collide_rlbm3d,
            "MRT": collide_mrt3d,
        },
        "D3Q27": {
            "BGK": collide_bgk27,
            "TRT": collide_trt27,
            "RLBM": collide_rlbm27,
            "MRT": collide_mrt27,
        },
    }
    return table[lattice_name][family_name]


def collide_advanced_3d(lattice: str, family: str, f: torch.Tensor, *, tau: float, **rates: float) -> torch.Tensor:
    """Execute a validated common collision kernel or explicitly withhold it.

    BGK, TRT, RLBM, and MRT are executable for both D3Q19 and D3Q27.
    CM and KBC are explicitly withheld.

    * ``tau`` is the relaxation time for BGK, RLBM, and MRT, and the symmetric
      relaxation time *τ₊* for TRT.
    * For TRT, ``lambda_trt`` may be passed as a keyword rate (default 3/16).
    * For MRT, keyword rates ``s_e``, ``s_eps``, ``s_q``, ``s_pi`` are passed
      through unchanged.
    """
    lattice_name = _normalise_lattice(lattice)
    family_name = _normalise_family(family)
    expected_q = 19 if lattice_name == "D3Q19" else 27
    if f.ndim != 4 or f.shape[0] != expected_q:
        raise ValueError(f"{lattice_name} populations must have shape ({expected_q}, nz, ny, nx)")
    if tau <= 0.5:
        raise ValueError("tau must be greater than 0.5")
    capability = collision_capability_matrix()[lattice_name][family_name]
    if not capability.available:
        raise CollisionKernelWithheldError(f"{capability.status}: {capability.note}")
    kernel = _select_kernel(lattice_name, family_name)
    if family_name == "TRT":
        lambda_trt = float(rates.pop("lambda_trt", 3.0 / 16.0))
        return kernel(f, tau_plus=tau, lambda_trt=lambda_trt)
    return kernel(f, tau=tau, **rates)


__all__ = [
    "CollisionCapability", "CollisionKernelWithheldError", "WITHHELD_NO_D3Q19_CM_KERNEL",
    "WITHHELD_NO_D3Q19_KBC_KERNEL", "WITHHELD_NO_D3Q27_CM_KERNEL",
    "WITHHELD_NO_D3Q27_KBC_KERNEL", "collision_capability_matrix", "collide_advanced_3d",
]
