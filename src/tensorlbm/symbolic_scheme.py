"""pylbm-inspired symbolic LBM scheme builder.

Absorbed architectural point
-----------------------------
pylbm (BSD-3-Clause) defines LBM simulations through two declarative
structures:

1. **Geometry dict** — walls, inlets, outlets, and internal obstacles are
   described as Python dicts with shape / position / boundary-type keys.
2. **Scheme dict** — the lattice (D2Q9 / D3Q19), relaxation time, and
   boundary-condition assignments are declared symbolically.

The framework then *generates* the solver: it builds the solid mask,
initialises equilibrium, wires the BC handlers, and exposes a single
``step()`` method.

Why this matters for TensorLBM
------------------------------
Before this module, every TensorLBM example / test manually:
- creates ``solid`` / ``fluid`` masks,
- calls ``equilibrium()`` to initialise ``f``,
- applies BCs by calling ``zou_he_inlet_velocity()`` etc.,
- loops ``collide -> stream -> BC``.

The symbolic scheme collapses those four steps into a declarative
specification.  The *mechanism* (declarative geometry + auto-generated
solver) is absorbed from pylbm; the *implementation* is original PyTorch
code that reuses TensorLBM's existing ``d2q9``, ``boundaries``, and
``solver`` primitives.

Provenance
----------
- pylbm geometry/scheme pattern: ``pylbm.Geometry`` / ``pylbm.Scheme``
  (BSD-3-Clause, https://github.com/loicgouarin/pylbm)
- No code is copied; only the *architectural pattern* is transplanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from . import d2q9
from .boundaries import (
    bounce_back_cells,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from .solver import collide_bgk, stream


@dataclass
class GeometrySpec:
    """Declarative geometry description (pylbm-inspired).

    Note: ``"left"`` and ``"right"`` walls are treated as **BC columns**
    (inlet/outlet), not solid walls.  Only ``"top"`` and ``"bottom"``
    produce bounce-back solid cells.  This matches the TensorLBM channel
    convention where Zou/He BCs occupy the boundary columns.
    """

    shape: Tuple[int, int]
    walls: List[str] = field(default_factory=list)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    periodic: Tuple[bool, bool] = (False, False)

    def build_solid_mask(self) -> torch.Tensor:
        """Build a ``(ny, nx)`` boolean solid mask from the spec.

        Only ``"top"`` and ``"bottom"`` walls produce solid cells.
        ``"left"`` and ``"right"`` are BC columns (handled by Zou/He).
        """
        nx, ny = self.shape
        solid = torch.zeros((ny, nx), dtype=torch.bool)

        # Only top/bottom are solid walls; left/right are BC columns
        if "top" in self.walls:
            solid[0, :] = True
        if "bottom" in self.walls:
            solid[-1, :] = True

        for obs in self.obstacles:
            obs_type = obs.get("type", "circle")
            if obs_type == "circle":
                cx, cy = obs["center"]
                r = obs["radius"]
                yy, xx = torch.meshgrid(
                    torch.arange(ny, dtype=torch.float32),
                    torch.arange(nx, dtype=torch.float32),
                    indexing="ij",
                )
                solid |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            elif obs_type == "rectangle":
                cx, cy = obs["center"]
                w, h = obs["size"]
                x0 = max(0, int(cx - w / 2))
                x1 = min(nx, int(cx + w / 2))
                y0 = max(0, int(cy - h / 2))
                y1 = min(ny, int(cy + h / 2))
                solid[y0:y1, x0:x1] = True

        return solid


@dataclass
class SchemeSpec:
    """Declarative LBM scheme description (pylbm-inspired)."""

    lattice: str = "D2Q9"
    tau: Optional[float] = None
    reynolds: Optional[float] = None
    u_in: Optional[float] = None
    char_length: Optional[float] = None
    bc_inlet: str = "zou_he"
    bc_outlet: str = "zou_he"

    def compute_tau(self, geometry: GeometrySpec) -> float:
        """Compute tau from Re and u_in if not provided."""
        if self.tau is not None:
            return self.tau
        if self.reynolds is None or self.u_in is None:
            raise ValueError("tau or (reynolds + u_in) must be provided")
        L = self.char_length or float(geometry.shape[1])
        nu = self.u_in * L / self.reynolds
        return 3.0 * nu + 0.5


class SymbolicScheme:
    """Auto-generated LBM solver from declarative geometry + scheme specs.

    Instead of manually wiring masks, BCs, and the collide-stream loop,
    the user declares *what* they want and this class generates *how* to
    do it — the core pylbm architectural pattern absorbed into TensorLBM.
    """

    def __init__(
        self,
        lattice: str = "D2Q9",
        geometry: Optional[Dict[str, Any]] = None,
        scheme: Optional[Dict[str, Any]] = None,
        reynolds: Optional[float] = None,
        u_in: Optional[float] = None,
        char_length: Optional[float] = None,
        tau: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self._dtype = dtype
        self._device = device or torch.device("cpu")

        if geometry is None:
            geometry = {}
        self._geom = GeometrySpec(
            shape=tuple(geometry.get("shape", (128, 64))),
            walls=geometry.get("walls", []),
            obstacles=geometry.get("obstacles", []),
            periodic=tuple(geometry.get("periodic", (False, False))),
        )

        if scheme is None:
            scheme = {}
        self._scheme = SchemeSpec(
            lattice=scheme.get("lattice", lattice),
            tau=scheme.get("tau", tau),
            reynolds=scheme.get("reynolds", reynolds),
            u_in=scheme.get("u_in", u_in),
            char_length=scheme.get("char_length", char_length),
            bc_inlet=scheme.get("bc_inlet", "zou_he"),
            bc_outlet=scheme.get("bc_outlet", "zou_he"),
        )

        if self._scheme.lattice != "D2Q9":
            raise ValueError(
                f"Unsupported lattice: {self._scheme.lattice} (only D2Q9 for now)"
            )

        self._tau = self._scheme.compute_tau(self._geom)
        self._solid: Optional[torch.Tensor] = None
        self._f: Optional[torch.Tensor] = None
        self._initialized = False

    @property
    def shape(self) -> Tuple[int, int]:
        return self._geom.shape

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def solid(self) -> torch.Tensor:
        if self._solid is None:
            raise RuntimeError("Call init() first")
        return self._solid

    @property
    def f(self) -> torch.Tensor:
        if self._f is None:
            raise RuntimeError("Call init() first")
        return self._f

    def init(self, rho0: float = 1.0) -> "SymbolicScheme":
        """Build solid mask and initialise equilibrium distribution."""
        nx, ny = self.shape
        self._solid = self._geom.build_solid_mask().to(self._device)
        u_in = self._scheme.u_in if self._scheme.u_in is not None else 0.05

        rho = torch.full((ny, nx), rho0, dtype=self._dtype, device=self._device)
        ux = torch.full((ny, nx), u_in, dtype=self._dtype, device=self._device)
        uy = torch.zeros((ny, nx), dtype=self._dtype, device=self._device)

        # Zero velocity on solid
        ux[self._solid] = 0.0
        uy[self._solid] = 0.0

        feq = d2q9.equilibrium(rho, ux, uy, device=self._device)
        self._f = feq.clone()
        self._f[:, self._solid] = 0.0
        self._initialized = True
        return self

    def step(self) -> "SymbolicScheme":
        """One collide-stream-BC step."""
        if not self._initialized:
            raise RuntimeError("Call init() first")

        f = self._f
        solid = self._solid
        u_in = self._scheme.u_in if self._scheme.u_in is not None else 0.05

        f = collide_bgk(f, self._tau)
        f = stream(f)
        f = self._apply_bc(f, solid, u_in)

        self._f = f
        return self

    def _apply_bc(
        self, f: torch.Tensor, solid: torch.Tensor, u_in: float
    ) -> torch.Tensor:
        """Apply boundary conditions based on the scheme spec."""
        ny, nx = self.shape

        # Inlet BC (left column, x=0)
        if "left" in self._geom.walls:
            if self._scheme.bc_inlet == "zou_he":
                f = zou_he_inlet_velocity(f, u_in)
            else:
                rho = torch.full((ny,), 1.0, dtype=self._dtype, device=self._device)
                ux = torch.full((ny,), u_in, dtype=self._dtype, device=self._device)
                uy = torch.zeros((ny,), dtype=self._dtype, device=self._device)
                feq = d2q9.equilibrium(rho, ux, uy, device=self._device)
                f[:, :, 0] = feq

        # Outlet BC (right column, x=-1)
        if "right" in self._geom.walls:
            if self._scheme.bc_outlet == "zou_he":
                f = zou_he_outlet_pressure(f, rho_out=1.0)

        # Bounce-back on solid (top/bottom walls + obstacles)
        f = bounce_back_cells(f, solid)
        return f

    def macroscopic(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rho, ux, uy) fields."""
        if not self._initialized:
            raise RuntimeError("Call init() first")
        return d2q9.macroscopic(self._f, device=self._device)

    def run(self, n_steps: int) -> "SymbolicScheme":
        """Run n_steps collide-stream-BC cycles."""
        for _ in range(n_steps):
            self.step()
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the scheme specification."""
        return {
            "lattice": self._scheme.lattice,
            "geometry": {
                "shape": list(self._geom.shape),
                "walls": self._geom.walls,
                "obstacles": self._geom.obstacles,
                "periodic": list(self._geom.periodic),
            },
            "scheme": {
                "tau": self._tau,
                "reynolds": self._scheme.reynolds,
                "u_in": self._scheme.u_in,
                "char_length": self._scheme.char_length,
                "bc_inlet": self._scheme.bc_inlet,
                "bc_outlet": self._scheme.bc_outlet,
            },
        }
