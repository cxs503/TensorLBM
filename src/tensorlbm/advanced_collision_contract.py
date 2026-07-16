"""Honest common contract for three-dimensional advanced collision families.

This module deliberately distinguishes callable, validated kernels from
experimental approximations elsewhere in the package.  In particular,
``advanced_collision.collide_cascaded_d3q27`` is a second-order regularized
reconstruction (its higher central moments are not implemented) and its KBC
routine uses a caller-supplied blend rather than an entropy solve.  They are
therefore *not* advertised here as CM/KBC kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .d3q27 import collide_mrt27
from .solver3d import collide_mrt3d

LatticeName = Literal["D3Q19", "D3Q27"]
CollisionFamily = Literal["MRT", "CM", "KBC"]

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
            "MRT": CollisionCapability(True, "tensorlbm.solver3d.collide_mrt3d", "AVAILABLE", "19x19 MRT transform; conserved rows are explicit."),
            "CM": CollisionCapability(False, None, WITHHELD_NO_D3Q19_CM_KERNEL, "No standalone validated D3Q19 central-moment kernel."),
            "KBC": CollisionCapability(False, None, WITHHELD_NO_D3Q19_KBC_KERNEL, "No standalone entropy-solved D3Q19 KBC kernel."),
        },
        "D3Q27": {
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
    aliases = {"MRT": "MRT", "CM": "CM", "CASCADED": "CM", "KBC": "KBC", "ENTROPIC_KBC": "KBC"}
    if value not in aliases:
        raise ValueError("family must be MRT, CM/cascaded, or KBC/entropic_kbc")
    return aliases[value]  # type: ignore[return-value]


def collide_advanced_3d(lattice: str, family: str, f: torch.Tensor, *, tau: float, **rates: float) -> torch.Tensor:
    """Execute a validated common collision kernel or explicitly withhold it.

    Currently MRT is executable for D3Q19 and D3Q27.  Keyword rates map to the
    native MRT API (``s_e``, ``s_eps``, ``s_q``, ``s_pi``); they are passed
    through unchanged after the lattice direction dimension is checked.
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
    kernel: Callable[..., torch.Tensor] = collide_mrt3d if lattice_name == "D3Q19" else collide_mrt27
    return kernel(f, tau=tau, **rates)


__all__ = [
    "CollisionCapability", "CollisionKernelWithheldError", "WITHHELD_NO_D3Q19_CM_KERNEL",
    "WITHHELD_NO_D3Q19_KBC_KERNEL", "WITHHELD_NO_D3Q27_CM_KERNEL",
    "WITHHELD_NO_D3Q27_KBC_KERNEL", "collision_capability_matrix", "collide_advanced_3d",
]
