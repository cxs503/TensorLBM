"""Circular-pipe Hagen-Poiseuille case — aligned with
``benchmarks/verified/poiseuille_3d_pipe``.

Configuration and step chain reproduce the verified benchmark's velocity
mode: Zou-He velocity inlet at ``x=0``, Zou-He pressure outlet at
``x=nx-1``, no-slip pipe wall (cells with ``d > R``) via POST-streaming
half-way bounce-back, BGK collision, parabolic initial profile with
``U_max = 2*u_in`` over the effective radius ``R_eff = R + 0.5``.

``nu = 2 * u_in * R_eff / Re`` (Re based on mean velocity and effective
diameter), hence ``tau = 0.5 + 3 * nu``.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import torch

from ..boundary_registry import BCPhase, BCKind, BoundaryCondition
from .base import CaseBase, CaseUnits
from .registry import register_case


@register_case("poiseuille")
class PipePoiseuilleCase(CaseBase):
    """3D circular-pipe Poiseuille flow (D3Q19, BGK, analytic solution)."""

    name: ClassVar[str] = "poiseuille"
    lattice: ClassVar[str] = "D3Q19"
    collision: ClassVar[str] = "bgk"
    # The wall bounce-back intentionally reprocesses the solid cells of
    # the inlet/outlet planes AFTER the plane BCs (verified worker order),
    # so the inlet/outlet ∩ wall overlap is order-dependent by design.
    strict_bc_overlap: ClassVar[bool] = False
    description: ClassVar[str] = (
        "Hagen-Poiseuille pipe flow (verified: analytic parabola, "
        "benchmarks/verified/poiseuille_3d_pipe). Zou-He velocity inlet / "
        "pressure outlet, bounce-back pipe wall, BGK."
    )

    def __init__(
        self,
        resolution: int | Sequence[int] = 20,
        re: float = 50.0,
        *,
        u_in: float = 0.05,
        l_over_r: int = 6,
        device=None,
        dtype: torch.dtype = torch.float32,
        collision: str | None = None,
    ) -> None:
        self.u_in = float(u_in)
        self.l_over_r = int(l_over_r)
        super().__init__(
            resolution, re, device=device, dtype=dtype, collision=collision
        )

    @classmethod
    def default_params(cls) -> dict:
        return {"resolution": 20, "re": 50.0, "u_in": 0.05, "l_over_r": 6}

    @property
    def radius(self) -> int:
        """Pipe radius *R* in cells (= the ``resolution`` argument)."""
        return self._radius

    def make_resolution(self, resolution: int | Sequence[int]) -> Sequence[int]:
        if isinstance(resolution, (int,)):
            if resolution < 4:
                raise ValueError(f"resolution (pipe radius R) must be >= 4, got {resolution}")
            r = int(resolution)
        else:
            if len(resolution) != 3:
                raise ValueError("resolution sequence must be (nz, ny, nx)")
            r = int(resolution[1]) // 2 - 1  # invert ny = 2R + 3
        self._radius = r
        # Cross-section with a 1-cell solid margin on each side; pipe axis
        # at (yc, zc) = (R + 1, R + 1); flow along +x.
        ny = nz = 2 * r + 3
        nx = self.l_over_r * r
        return (nz, ny, nx)

    def make_units(self, re: float, resolution: Sequence[int]) -> CaseUnits:
        # nu = 2 * u_in * R_eff / re with R_eff = R + 0.5 (half-way
        # bounce-back wall position) — exactly the verified run.py value.
        d_eff = 2 * self._radius + 1
        return CaseUnits.from_reference(re=re, u_lb=self.u_in, n_ref=float(d_eff))

    @property
    def r_eff(self) -> float:
        return self._radius + 0.5

    def _cross_section(self):
        nz, ny, nx = self.resolution
        yc = zc = self._radius + 1
        iz = torch.arange(nz, device=self.device, dtype=torch.float32).view(-1, 1)
        iy = torch.arange(ny, device=self.device, dtype=torch.float32).view(1, -1)
        d = torch.sqrt((iy - yc) ** 2 + (iz - zc) ** 2)
        fluid = d <= self._radius
        return d, fluid

    def wall_mask(self) -> torch.Tensor:
        """Solid pipe wall: cells outside the circle ``d <= R``."""
        nz, ny, nx = self.resolution
        _d, fluid = self._cross_section()
        wall = (~fluid).unsqueeze(-1).expand(nz, ny, nx).contiguous()
        return wall

    def solid_mask(self) -> torch.Tensor:
        return self.wall_mask()

    def boundary_conditions(self) -> list[BoundaryCondition]:
        # Application order = verified worker order: inlet → outlet →
        # wall bounce-back (the wall legitimately reprocesses the solid
        # cells of the inlet/outlet planes afterwards).
        return [
            BoundaryCondition(
                BCKind.ZOU_HE_INLET_VELOCITY,
                phase=BCPhase.POST_STREAMING,
                face="x-",
                params={"u_in": self.u_in},
                name="poiseuille_inlet",
            ),
            BoundaryCondition(
                BCKind.ZOU_HE_OUTLET_PRESSURE,
                phase=BCPhase.POST_STREAMING,
                face="x+",
                params={"rho_out": 1.0},
                name="poiseuille_outlet",
            ),
            BoundaryCondition(
                BCKind.BOUNCE_BACK,
                phase=BCPhase.POST_STREAMING,
                mask=self.wall_mask(),
                name="poiseuille_wall",
            ),
        ]

    def initial_pu(self):
        # Parabolic init over the effective radius (verified run.py):
        # ux = U_max * (1 - (d/R_eff)^2) inside the fluid, 0 in the wall.
        nz, ny, nx = self.resolution
        d, fluid = self._cross_section()
        d3 = d.unsqueeze(-1)
        ux = torch.where(
            fluid.unsqueeze(-1),
            (2.0 * self.u_in) * (1.0 - d3**2 / self.r_eff**2),
            torch.zeros_like(d3),
        ).expand(nz, ny, nx)
        ux = ux.contiguous().to(self.dtype)
        rho = torch.ones((nz, ny, nx), dtype=self.dtype, device=self.device)
        zero = torch.zeros_like(rho)
        return rho, ux, zero.clone(), zero.clone()
