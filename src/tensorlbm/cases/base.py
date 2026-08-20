"""Case base class for the TensorLBM case registry.

The design follows lettuce's ``ExtFlow``
(``lettuce/ext/_flows/_ext_flow.py``, MIT licence, Copyright (c) 2019
Andreas Kraemer): a flow is described by three small hooks —
``make_resolution`` / ``make_units`` / ``initial_pu`` — plus *optional*
boundary hooks (``pre_boundaries`` / ``post_boundaries``), so that the
~82 worker scripts in the repository root can be absorbed incrementally
into named, enumerable cases without rewriting their physics.

Differences from lettuce (adapted for TensorLBM, listed per the MIT
licence's notice requirement):

* units are expressed in lattice units via :class:`CaseUnits` (same
  attribute surface as :class:`tensorlbm.unit_converter.LBMUnitConverter`:
  ``re``/``u_lb``/``nu_lb``/``tau``/``ma``) because several production
  cases anchor ``tau`` to non-integer lattice lengths;
* boundaries are declared through the integer-id BC registry
  (:mod:`tensorlbm.boundary_registry`) instead of a ``Boundary`` ABC;
* the default :meth:`CaseBase.make_step` composes the exact production
  chain collide → pre-boundaries → stream → post-boundaries from the
  public solver operators, so a registry-run case is bit-identical to
  the equivalent worker/benchmark loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

import torch

from ..boundary_registry import (
    BoundaryCondition,
    boundary_condition_registry,
    apply_boundary_conditions,
    build_bc_mask,
    check_bc_consistency,
)
from ..unit_converter import LBMUnitConverter  # noqa: F401  (re-exported for case authors)

__all__ = ["CaseBase", "CaseUnits"]


@dataclass(frozen=True)
class CaseUnits:
    """Lattice-unit conversion for one case (same surface as
    :class:`~tensorlbm.unit_converter.LBMUnitConverter`).

    ``nu_lb = u_lb * n_ref / re`` and ``tau = 0.5 + nu_lb / cs²``; the
    reference length ``n_ref`` may be fractional (e.g. a hull spanning
    ``0.6 * nx`` cells), which the integer-``nx`` ``LBMUnitConverter``
    cannot express.
    """

    re: float
    u_lb: float
    n_ref: float
    nu_lb: float
    tau: float
    ma: float

    @classmethod
    def from_reference(
        cls, re: float, u_lb: float, n_ref: float, *, u_lb_label: str = "u_lb"
    ) -> "CaseUnits":
        """Derive ``nu_lb``/``tau``/``ma`` from (Re, u_lb, n_ref)."""
        if re <= 0 or u_lb <= 0 or n_ref <= 0:
            raise ValueError(f"re, {u_lb_label} and n_ref must be positive")
        nu_lb = u_lb * n_ref / re
        tau = 0.5 + nu_lb / (1.0 / 3.0)
        return cls(re=re, u_lb=u_lb, n_ref=n_ref, nu_lb=nu_lb, tau=tau, ma=u_lb * (3.0**0.5))


class CaseBase(ABC):
    """Base class for named simulation cases.

    Subclasses implement the three lettuce-``ExtFlow`` hooks
    (:meth:`make_resolution`, :meth:`make_units`, :meth:`initial_pu`)
    and declare their boundaries via :meth:`boundary_conditions`.  The
    default :meth:`make_step` then produces the standard TensorLBM
    chain; cases with extra per-step behaviour (e.g. periodic mass
    correction) override :meth:`make_step` or set
    :attr:`mass_correction_interval`.
    """

    #: Registry key (must be unique); set by every concrete case.
    name: ClassVar[str] = ""
    #: Lattice model label ("D3Q19" or "D3Q27").
    lattice: ClassVar[str] = "D3Q19"
    #: Default collision family label (see :meth:`collide`).
    collision: ClassVar[str] = "bgk"
    #: One-line description for ``list_cases`` / platform enumeration.
    description: ClassVar[str] = ""
    #: Apply ``correct_mass3d`` every N steps (0 = never).
    mass_correction_interval: int = 0
    #: Reject overlapping BC cell sets at construction.  Cases whose BC
    #: order is intentionally order-dependent (e.g. the verified
    #: Poiseuille benchmark, whose bounce-back wall reprocesses the solid
    #: cells of the inlet/outlet planes) set this to ``False`` and get a
    #: warning instead (XLB semantics).
    strict_bc_overlap: ClassVar[bool] = True

    def __init__(
        self,
        resolution: int | Sequence[int],
        re: float,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        collision: str | None = None,
        register_bcs: bool = True,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        if collision is not None:
            self.collision = collision
        res = self.make_resolution(resolution)
        res = tuple(int(d) for d in res)
        if len(res) != 3 or any(d <= 0 for d in res):
            raise ValueError(f"make_resolution must return positive (nz, ny, nx), got {res}")
        self.resolution: tuple[int, int, int] = res  # type: ignore[assignment]
        self.re = float(re)
        self.units: CaseUnits = self.make_units(self.re, self.resolution)
        self._bcs: list[BoundaryCondition] | None = None
        self._bc_masks: dict | None = None
        if register_bcs:
            self._register_boundary_conditions()

    # -- lettuce ExtFlow hooks --------------------------------------------

    @abstractmethod
    def make_resolution(self, resolution: int | Sequence[int]) -> Sequence[int]:
        """Expand *resolution* into the grid ``(nz, ny, nx)``."""
        raise NotImplementedError

    @abstractmethod
    def make_units(self, re: float, resolution: Sequence[int]) -> CaseUnits:
        """Derive lattice units (``tau`` in particular) for this case."""
        raise NotImplementedError

    @abstractmethod
    def initial_pu(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial ``(rho, ux, uy, uz)`` fields of shape ``(nz, ny, nx)``."""
        raise NotImplementedError

    # -- optional hooks -----------------------------------------------------

    def solid_mask(self) -> torch.Tensor | None:
        """Boolean solid field (``None`` when the case has no obstacle)."""
        return None

    def boundary_conditions(self) -> list[BoundaryCondition]:
        """Boundary conditions of this case (order = application order)."""
        return []

    def periodic_axes(self) -> dict[str, bool]:
        """Per-axis periodicity for the missing-direction mask."""
        return {"x": False, "y": False, "z": False}

    # -- boundary registry plumbing ----------------------------------------

    def _register_boundary_conditions(self) -> None:
        """Register this case's BCs and build the per-phase integer id masks.

        One mask per application phase: PRE and POST BCs may legally share
        cells (e.g. the cavity lid plane meets the stationary walls on its
        edge lines), and a single combined mask would hide those cells
        from the earlier phase's ``bc_mask == id`` selection.
        """
        from ..boundary_registry import BCPhase

        self._bcs = self.boundary_conditions()
        for bc in self._bcs:
            if bc not in boundary_condition_registry:
                boundary_condition_registry.register(bc)
        check_bc_consistency(
            self._bcs, self.resolution, device=self.device,
            strict_overlap=self.strict_bc_overlap,
        )
        self._bc_masks = {
            phase: build_bc_mask(
                self.resolution, self._bcs, phase=phase, device=self.device
            )
            for phase in (BCPhase.PRE_STREAMING, BCPhase.POST_STREAMING)
        }
        # A same-phase overlap shadows the earlier BC's cells (last wins);
        # only intentional when strict_bc_overlap is False (warned above).
        for bc in self._bcs:
            if not bool((self._bc_masks[bc.phase] == bc.id).any()):
                raise ValueError(
                    f"boundary condition {bc.name!r} is fully shadowed by a later "
                    f"{bc.phase.value} BC sharing its cells"
                )

    @property
    def bcs(self) -> list[BoundaryCondition]:
        """The registered boundary conditions of this case."""
        if self._bcs is None:
            self._register_boundary_conditions()
        return self._bcs  # type: ignore[return-value]

    def bc_mask_for(self, phase: BCPhase | str) -> torch.Tensor:
        """Integer id field of one application phase (0 = no BC in that phase)."""
        from ..boundary_registry import BCPhase

        if self._bc_masks is None:
            self._register_boundary_conditions()
        return self._bc_masks[BCPhase(phase)]

    @property
    def bc_mask(self) -> torch.Tensor:
        """Combined integer id field over all BCs (last BC wins on shared
        cells).  Diagnostic / single-phase use; the step pipeline uses
        :meth:`bc_mask_for` so cross-phase shared cells stay selectable.
        """
        if self._bc_masks is None:
            self._register_boundary_conditions()
        return build_bc_mask(self.resolution, self._bcs, device=self.device)

    def pre_boundaries(self, f: torch.Tensor, f_pre: torch.Tensor) -> torch.Tensor:
        """Apply PRE_STREAMING BCs (between collision and streaming)."""
        return apply_boundary_conditions(
            f, self.bcs, phase="pre_streaming",
            bc_mask=self.bc_mask_for("pre_streaming"), f_pre=f_pre,
        )

    def post_boundaries(self, f: torch.Tensor) -> torch.Tensor:
        """Apply POST_STREAMING BCs (after streaming)."""
        return apply_boundary_conditions(
            f, self.bcs, phase="post_streaming",
            bc_mask=self.bc_mask_for("post_streaming"),
        )

    # -- step composition ---------------------------------------------------

    def collide(self, f: torch.Tensor) -> torch.Tensor:
        """Collision step dispatched by :attr:`collision`."""
        family = self.collision.lower()
        if family == "bgk":
            from ..solver3d import collide_bgk3d

            return collide_bgk3d(f, self.units.tau)
        if family == "mrt":
            from ..solver3d import collide_mrt3d

            return collide_mrt3d(f, self.units.tau)
        from ..advanced_collision_contract import collide_advanced_3d

        return collide_advanced_3d(self.lattice, family, f, tau=self.units.tau)

    def stream(self, f: torch.Tensor) -> torch.Tensor:
        """Streaming step (periodic pull scheme, cached-index gather)."""
        if self.lattice == "D3Q27":
            from ..d3q27 import stream27_roll

            return stream27_roll(f)
        from ..solver3d import stream3d

        return stream3d(f)

    def make_step(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the full one-step closure for this case.

        The chain is the production order used by the verified
        benchmarks: collide → (pre-streaming BCs with the pre-collision
        state) → stream → (post-streaming BCs).
        """
        collide = self.collide
        pre = self.pre_boundaries
        stream = self.stream
        post = self.post_boundaries

        def step(f: torch.Tensor) -> torch.Tensor:
            f_pre = f
            f = collide(f)
            f = pre(f, f_pre)
            f = stream(f)
            return post(f)

        return step

    # -- conveniences ---------------------------------------------------------

    def initial_f(self) -> torch.Tensor:
        """Equilibrium populations from :meth:`initial_pu`."""
        if self.lattice == "D3Q27":
            from ..d3q27 import equilibrium27

            rho, ux, uy, uz = self.initial_pu()
            return equilibrium27(rho, ux, uy, uz)
        from ..d3q19 import equilibrium3d

        rho, ux, uy, uz = self.initial_pu()
        return equilibrium3d(rho, ux, uy, uz, device=self.device)

    def grid(self) -> tuple[int, int, int]:
        """Grid ``(nz, ny, nx)``."""
        return self.resolution

    def metadata(self) -> dict[str, Any]:
        """JSON-safe metadata describing this case instance."""
        nz, ny, nx = self.resolution
        return {
            "case": self.name,
            "lattice": self.lattice,
            "collision": self.collision,
            "boundary_type": ",".join(sorted({bc.kind.value for bc in self.bcs})),
            "re": self.re,
            "u_in": self.units.u_lb,
            "nu": self.units.nu_lb,
            "tau": self.units.tau,
            "nx": nx,
            "ny": ny,
            "nz": nz,
        }

    def missing_mask(self) -> torch.Tensor:
        """Missing-direction mask for this case (XLB stream-the-boolean method)."""
        from ..boundary_registry import derive_missing_mask

        return derive_missing_mask(
            self.resolution,
            solid_mask=self.solid_mask(),
            periodic=self.periodic_axes(),
            lattice=self.lattice,
            device=self.device,
        )
