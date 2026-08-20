"""Lid-driven cavity case — aligned with ``benchmarks/verified/cavity/3d``.

Configuration and step chain reproduce the verified benchmark
``benchmarks/verified/cavity/3d/run.py`` (spanwise-periodic 3D cavity,
Re=400, MRT collision, Ghia 1982 reference):

* three stationary walls (``x=0``, ``x=nx-1``, ``y=0``) via PRE-streaming
  half-way bounce-back reading the pre-collision state;
* moving lid (``y=ny-1``) via ``boundaries3d.zou_he_moving_lid_3d``;
* spanwise (z) periodic streaming;
* ``tau = 3 * u_lid * nx / Re + 0.5`` (MRT shear rate = 1/tau).
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import torch

from ..boundary_registry import BCPhase, BCKind, BoundaryCondition
from .base import CaseBase, CaseUnits
from .registry import register_case


@register_case("cavity")
class LidCavityCase(CaseBase):
    """Spanwise-periodic lid-driven cavity (D3Q19, MRT)."""

    name: ClassVar[str] = "cavity"
    lattice: ClassVar[str] = "D3Q19"
    collision: ClassVar[str] = "mrt"
    description: ClassVar[str] = (
        "Lid-driven cavity, spanwise-periodic 3D (verified: Ghia 1982 Re=400, "
        "benchmarks/verified/cavity/3d). Walls x=0/x=nx-1/y=0 pre-streaming "
        "half-way bounce-back, lid y=ny-1 Zou-He, MRT."
    )

    def __init__(
        self,
        resolution: int | Sequence[int] = 96,
        re: float = 400.0,
        *,
        u_lid: float = 0.06,
        span: int | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        collision: str | None = None,
    ) -> None:
        self.u_lid = float(u_lid)
        self.span = span
        super().__init__(
            resolution, re, device=device, dtype=dtype, collision=collision
        )

    @classmethod
    def default_params(cls) -> dict:
        return {"resolution": 96, "re": 400.0, "u_lid": 0.06}

    def make_resolution(self, resolution: int | Sequence[int]) -> Sequence[int]:
        if isinstance(resolution, (int,)):
            if resolution < 8:
                raise ValueError(f"resolution must be >= 8, got {resolution}")
            nz = self.span if self.span is not None else max(4, resolution // 4)
            return (nz, resolution, resolution)
        if len(resolution) != 3:
            raise ValueError("resolution sequence must be (nz, ny, nx)")
        return tuple(resolution)

    def make_units(self, re: float, resolution: Sequence[int]) -> CaseUnits:
        # tau = 3 * u_lid * nx / re + 0.5  (verified run.py formula; the
        # reference length is the cavity width nx).
        return CaseUnits.from_reference(re=re, u_lb=self.u_lid, n_ref=float(resolution[2]))

    def wall_mask(self) -> torch.Tensor:
        """Stationary walls: x=0, x=nx-1 and the bottom y=0 plane."""
        nz, ny, nx = self.resolution
        wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=self.device)
        wall[:, :, 0] = True
        wall[:, :, -1] = True
        wall[:, 0, :] = True
        return wall

    def solid_mask(self) -> torch.Tensor:
        # The stationary walls are treated as solid for the missing-mask
        # derivation (near-wall fluid cells pull from them).
        return self.wall_mask()

    def periodic_axes(self) -> dict[str, bool]:
        return {"x": False, "y": False, "z": True}

    def boundary_conditions(self) -> list[BoundaryCondition]:
        return [
            BoundaryCondition(
                BCKind.BOUNCE_BACK,
                phase=BCPhase.PRE_STREAMING,
                mask=self.wall_mask(),
                name="cavity_stationary_walls",
            ),
            BoundaryCondition(
                BCKind.MOVING_LID,
                phase=BCPhase.POST_STREAMING,
                face="y+",
                params={"u_lid": self.u_lid},
                name="cavity_lid",
            ),
        ]

    def initial_pu(self):
        nz, ny, nx = self.resolution
        rho = torch.ones((nz, ny, nx), dtype=self.dtype, device=self.device)
        u = torch.zeros((nz, ny, nx), dtype=self.dtype, device=self.device)
        return rho, u, u.clone(), u.clone()
