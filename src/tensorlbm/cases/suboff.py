"""Small SUBOFF bare-hull channel case — aligned with ``examples/ai4s_export.py``.

Configuration and step chain reproduce the AI4S pilot runner
(``examples/ai4s_export.py::run_config``): analytic SUBOFF bare-hull mask
from :func:`tensorlbm.suboff_cad.build_suboff_mask` centred at
``cx = 0.35 * nx`` with hull length ``0.6 * nx``; free-stream equilibrium
inlet, zero-gradient outlet, Dirichlet lateral faces and obstacle
bounce-back via :func:`tensorlbm.boundaries3d.far_field_bc_3d`; mass
correction every 10 steps; ``tau = 3 * u_in * L / Re + 0.5`` with
``L = 0.6 * nx`` (lattice units).
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import torch

from ..boundary_registry import BCKind, BCPhase, BoundaryCondition
from .base import CaseBase, CaseUnits
from .registry import register_case


@register_case("suboff_n128")
class SuboffChannelCase(CaseBase):
    """SUBOFF bare hull in a free-stream channel (D3Q19).

    ``hull_type`` selects the DARPA configuration (bare_hull / with_sail /
    full) and ``sail_scale`` / ``fin_scale`` multiply the appendages' own
    dimensions about their DARPA anchors (1.0 = exact geometry), so the
    geometry axis is scannable as plain numeric sweep params. All three
    flow through :class:`~tensorlbm.suboff_cad.SuboffConfig`.
    """

    name: ClassVar[str] = "suboff_n128"
    lattice: ClassVar[str] = "D3Q19"
    collision: ClassVar[str] = "cumulant"
    description: ClassVar[str] = (
        "SUBOFF bare-hull external flow, AI4S pilot scale "
        "(examples/ai4s_export.py). Analytic hull via suboff_cad, far-field "
        "BC on all six faces with obstacle bounce-back, mass correction every "
        "10 steps."
    )
    mass_correction_interval = 10

    def __init__(
        self,
        resolution: int | Sequence[int] = 128,
        re: float = 420.0,
        *,
        u_in: float = 0.10,
        hull_type: str = "bare_hull",
        sail_scale: float = 1.0,
        fin_scale: float = 1.0,
        device=None,
        dtype: torch.dtype = torch.float32,
        collision: str | None = None,
    ) -> None:
        self.u_in = float(u_in)
        self.hull_type = hull_type
        self.sail_scale = float(sail_scale)
        self.fin_scale = float(fin_scale)
        super().__init__(resolution, re, device=device, dtype=dtype, collision=collision)

    @classmethod
    def default_params(cls) -> dict:
        return {"resolution": 128, "re": 420.0, "u_in": 0.10, "collision": "cumulant"}

    @property
    def hull_length(self) -> float:
        return 0.6 * self.resolution[2]

    def make_resolution(self, resolution: int | Sequence[int]) -> Sequence[int]:
        if isinstance(resolution, (int,)):
            if resolution < 16:
                raise ValueError(f"resolution must be >= 16, got {resolution}")
            return (resolution // 2, resolution // 2, resolution)
        if len(resolution) != 3:
            raise ValueError("resolution sequence must be (nz, ny, nx)")
        return tuple(resolution)

    def make_units(self, re: float, resolution: Sequence[int]) -> CaseUnits:
        # tau = 3 * nu + 0.5 with nu = u_in * hull_length / re
        # (ai4s_export PilotConfig formula; fractional reference length).
        return CaseUnits.from_reference(re=re, u_lb=self.u_in, n_ref=self.hull_length)

    def build_solid(self) -> torch.Tensor:
        from ..suboff_cad import SuboffConfig, build_suboff_mask

        nz, ny, nx = self.resolution
        solid, _stats = build_suboff_mask(
            hull_type=self.hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            cx=nx * 0.35,
            cy=ny / 2.0,
            cz=nz / 2.0,
            length=self.hull_length,
            config=SuboffConfig(sail_scale=self.sail_scale, fin_scale=self.fin_scale),
            device=str(self.device),
        )
        return solid

    def solid_mask(self) -> torch.Tensor:
        if not hasattr(self, "_solid"):
            self._solid = self.build_solid()
        return self._solid

    def boundary_conditions(self) -> list[BoundaryCondition]:
        return [
            BoundaryCondition(
                BCKind.FAR_FIELD,
                phase=BCPhase.POST_STREAMING,
                params={
                    "u_in": self.u_in,
                    "obstacle_mask": self.solid_mask(),
                    "faces": ["x-", "x+", "y-", "y+", "z-", "z+"],
                },
                name="suboff_far_field",
            ),
        ]

    def initial_pu(self):
        # rho = 1 with ux = u_in in the fluid, 0 inside the hull
        # (ai4s_export run_config initial condition).
        nz, ny, nx = self.resolution
        rho = torch.ones((nz, ny, nx), dtype=self.dtype, device=self.device)
        ux = torch.full_like(rho, self.u_in)
        ux[self.solid_mask()] = 0.0
        zero = torch.zeros_like(rho)
        return rho, ux, zero.clone(), zero.clone()

    def metadata(self) -> dict:
        meta = super().metadata()
        meta["hull_length"] = self.hull_length
        meta["hull_type"] = self.hull_type
        meta["sail_scale"] = self.sail_scale
        meta["fin_scale"] = self.fin_scale
        return meta
