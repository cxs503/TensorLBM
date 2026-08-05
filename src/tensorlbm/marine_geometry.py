"""Phase-independent static voxel geometry and D3Q19 wall-link compilation.

The contracts in this module deliberately contain no free-surface, phase-field,
or solver state.  Tensor voxel fields always use ``(z, y, x)`` indexing while
D3Q19 velocities are read from the authoritative ``d3q19.C`` in ``(x, y, z)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Final

import torch

from .d3q19 import C

_AXIS_ZYX_FROM_C_XYZ: Final[torch.Tensor] = torch.tensor((2, 1, 0), dtype=torch.int64)


def _origin_xyz(value: object) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("origin must be an (x, y, z) tuple of length 3")
    result: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) or not isfinite(coordinate):
            raise ValueError("origin must contain finite numeric coordinates")
        result.append(float(coordinate))
    return tuple(result)  # type: ignore[return-value]


def _geometry_hash(mask: torch.Tensor, body_id: str, origin: tuple[float, float, float], units: str, source_id: str) -> str:
    """Return a stable provenance digest for the static geometry contract."""
    canonical = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = sha256()
    digest.update(b"GeometryAsset/R1\0")
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.numpy().tobytes())
    for item in (body_id, repr(origin), units, source_id):
        digest.update(b"\0")
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class GeometryAsset:
    """Immutable static body geometry shared by surface and submerged workflows.

    ``solid_mask`` is a bool tensor indexed ``(z, y, x)``.  The constructor
    copies it into private storage and the public property returns a fresh clone,
    so neither the input nor a later reader can mutate the frozen asset.
    """

    _solid_mask: torch.Tensor = field(repr=False, compare=False)
    body_id: str
    origin: tuple[float, float, float]
    units: str
    source_id: str
    source_hash: str

    def __init__(
        self,
        solid_mask: torch.Tensor,
        body_id: str,
        origin: tuple[float, float, float],
        units: str,
        source_id: str,
        source_hash: str | None = None,
    ) -> None:
        if not isinstance(solid_mask, torch.Tensor) or solid_mask.ndim != 3 or solid_mask.dtype != torch.bool:
            raise ValueError("solid_mask must be a 3D bool torch.Tensor indexed (z, y, x)")
        if not isinstance(body_id, str) or not body_id:
            raise ValueError("body_id must be a non-empty string")
        if not isinstance(units, str) or not units:
            raise ValueError("units must be a non-empty string")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        resolved_origin = _origin_xyz(origin)
        mask = solid_mask.detach().clone()
        calculated = _geometry_hash(mask, body_id, resolved_origin, units, source_id)
        if source_hash is not None and (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(c not in "0123456789abcdef" for c in source_hash.lower())
        ):
            raise ValueError("source_hash must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "_solid_mask", mask)
        object.__setattr__(self, "body_id", body_id)
        object.__setattr__(self, "origin", resolved_origin)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_hash", source_hash or calculated)

    @property
    def solid_mask(self) -> torch.Tensor:
        """Return a detached clone; callers cannot mutate the frozen snapshot."""
        return self._solid_mask.detach().clone()


@dataclass(frozen=True, slots=True)
class D3Q19WallLinks:
    """Link-owned, solid-to-fluid D3Q19 interfaces for one static asset.

    ``owner_zyx`` names solid voxels; ``neighbor_zyx`` names their adjacent
    fluid voxels.  Direction is the corresponding row in ``d3q19.C``.  Domain
    exits are excluded, never wrapped, so periodicity cannot be inferred from a
    static mask alone.
    """

    owner_zyx: torch.Tensor
    neighbor_zyx: torch.Tensor
    direction: torch.Tensor
    body_id: str
    source_hash: str
    lattice_id: str = "D3Q19"

    def __post_init__(self) -> None:
        count = self.direction.numel()
        if self.lattice_id != "D3Q19":
            raise ValueError("D3Q19WallLinks requires lattice_id='D3Q19'")
        if self.owner_zyx.dtype != torch.int64 or self.neighbor_zyx.dtype != torch.int64:
            raise ValueError("wall-link coordinates must be int64")
        if self.owner_zyx.shape != (count, 3) or self.neighbor_zyx.shape != (count, 3):
            raise ValueError("wall-link coordinates must have shape (n, 3)")
        if self.direction.dtype != torch.int64 or self.direction.ndim != 1 or bool(((self.direction < 1) | (self.direction > 18)).any()):
            raise ValueError("direction must contain D3Q19 moving directions 1..18")

    @property
    def count(self) -> int:
        return int(self.direction.numel())

    @property
    def has_link_ownership(self) -> bool:
        """True because every entry explicitly owns a solid-to-fluid link."""
        return True


def compile_d3q19_wall_links(asset: GeometryAsset) -> D3Q19WallLinks:
    """Compile non-wrapping solid-to-fluid links from ``asset.solid_mask``.

    The result derives all directions from the authoritative ``d3q19.C`` and
    explicitly maps each velocity from xyz to tensor zyx axes.
    """
    if not isinstance(asset, GeometryAsset):
        raise TypeError("asset must be a GeometryAsset")
    mask = asset.solid_mask
    shape = torch.tensor(mask.shape, dtype=torch.int64, device=mask.device)
    owners = mask.nonzero(as_tuple=False).to(dtype=torch.int64)
    directions_zyx = C[1:].to(device=mask.device, dtype=torch.int64).index_select(1, _AXIS_ZYX_FROM_C_XYZ.to(mask.device))
    owner_parts: list[torch.Tensor] = []
    neighbor_parts: list[torch.Tensor] = []
    direction_parts: list[torch.Tensor] = []
    for q, delta_zyx in enumerate(directions_zyx, start=1):
        candidate = owners + delta_zyx
        in_bounds = ((candidate >= 0) & (candidate < shape)).all(dim=1)
        if not bool(in_bounds.any()):
            continue
        owner = owners[in_bounds]
        neighbor = candidate[in_bounds]
        fluid = ~mask[tuple(neighbor.T)]
        if bool(fluid.any()):
            owner_parts.append(owner[fluid])
            neighbor_parts.append(neighbor[fluid])
            direction_parts.append(torch.full((int(fluid.sum().item()),), q, dtype=torch.int64, device=mask.device))
    if owner_parts:
        owner_zyx = torch.cat(owner_parts)
        neighbor_zyx = torch.cat(neighbor_parts)
        direction = torch.cat(direction_parts)
    else:
        owner_zyx = torch.empty((0, 3), dtype=torch.int64, device=mask.device)
        neighbor_zyx = torch.empty((0, 3), dtype=torch.int64, device=mask.device)
        direction = torch.empty((0,), dtype=torch.int64, device=mask.device)
    return D3Q19WallLinks(owner_zyx, neighbor_zyx, direction, asset.body_id, asset.source_hash or "")


# A concise alias for consumers that do not need to mention the stencil twice.
WallLinks = D3Q19WallLinks

__all__ = ["D3Q19WallLinks", "GeometryAsset", "WallLinks", "compile_d3q19_wall_links"]
