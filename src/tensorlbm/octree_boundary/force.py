"""P3 force closure for the octree boundary shell.

Implements design doc §3.5 / §4 (P3 acceptance): the momentum-exchange
(MEM) force accumulated over the shell substeps with the per-leaf weight
``2^-(d_max - d_leaf)``, plus an independent control-volume (CV) force
whose sampling surface is kept clear of the shell interface and the AMR
interface filter shell by a **fail-closed clearance gate**.

Conventions
-----------
* ``ShellForceLedger.mem_force`` is the time-averaged MEM force over one
  root step in **leaf lattice units**: ``sum_substeps w_leaf * F_sub / 2^d``
  (``UniformSubcycleAverager`` convention — the finest level advances
  ``2^d`` uniformly spaced substeps per root step and the force is their
  mean).  With ``d_max=1`` and all leaves at depth 1 the weight is 1 and
  ``mem_force = (F_s0 + F_s1) / 2``.
* ``ShellForceLedger.cv_force`` is observed once per root step on the L1
  block tensor (L1 lattice units).
* The two forces live on different lattices (leaf ``dx=0.5`` vs L1
  ``dx=1``).  :func:`convert_leaf_force_to_l1` maps the leaf force into L1
  units with the convective scaling ``F_l1 = (dx_l^4 / dt_l^2) F_leaf``
  (``rho0 = 1``), which is the exact dimensional conversion of the link
  impulse.  The acceptance comparison is done on the dimensionless Cd, so
  the example normalises each force with its own lattice's dynamic area
  (radius in that lattice's units) — the lattice factors cancel exactly.
* The CV clearance gate (fail-closed): the outer one-cell surface of the
  control volume must not intersect the shell-covered mask, the AMR
  interface filter shell, or the body, and the covered region + body must
  be fully enclosed.  Any violation raises ``ValueError`` — the CV is
  never silently moved.
"""

from __future__ import annotations

import math

import torch

from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)


def substep_force_weights(octree) -> torch.Tensor:
    """Per-leaf substep weight ``2^-(d_max - d_leaf)`` (see bfl module)."""
    return 2.0 ** (-(octree.d_max - octree.leaf_level.to(torch.float64)))


def convert_leaf_force_to_l1(
    leaf_force: torch.Tensor,
    dx_leaf: float,
    dt_leaf: float,
) -> torch.Tensor:
    """Convert a leaf-lattice force to L1 lattice units.

    ``F_phys = rho0 (dx^4/dt^2) F_lu`` in any lattice (rho0 = 1); the
    conversion factor between two convectively scaled lattices is
    ``(dx_l^4/dt_l^2) / (dx_c^4/dt_c^2)``, with the L1 lattice at
    ``dx_c = dt_c = 1``.
    """
    scale = (float(dx_leaf) ** 4) / (float(dt_leaf) ** 2)
    return torch.as_tensor(leaf_force, dtype=torch.float64) * scale


class ShellForceLedger:
    """MEM accumulation over substeps + one CV observation per root step.

    One ledger instance covers one L1 root step; call
    :meth:`reset` between root steps (or construct a fresh instance).
    """

    def __init__(
        self,
        octree,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.d_max = int(octree.d_max)
        self.n_substeps = 1 << self.d_max
        self._accum = torch.zeros(
            3,
            dtype=dtype,
            device=octree.f_leaf.device,
        )
        self.cv_force: torch.Tensor | None = None
        self.cv_samples = 0
        self.substep_count = 0

    def add_substep_force(self, force: torch.Tensor) -> None:
        """Accumulate one substep's (already per-leaf weighted) MEM force."""
        f = torch.as_tensor(
            force,
            dtype=self._accum.dtype,
            device=self._accum.device,
        )
        if f.shape != (3,):
            raise ValueError(f"force must be (3,), got {tuple(f.shape)}")
        if not bool(torch.isfinite(f).all()):
            raise FloatingPointError("non-finite MEM force substep")
        self._accum += f
        self.substep_count += 1

    @property
    def mem_force(self) -> torch.Tensor:
        """Time-averaged MEM force over the root step (leaf lattice units).

        Raises:
            RuntimeError: when fewer than ``2^d`` substeps were accumulated.
        """
        if self.substep_count != self.n_substeps:
            raise RuntimeError(
                "MEM force requires exactly 2^d substep samples "
                f"(d={self.d_max}): observed {self.substep_count}",
            )
        return self._accum / self.n_substeps

    def observe_cv_force(
        self,
        f_old: torch.Tensor,
        f_new: torch.Tensor,
        f_post_collision: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        control_volume: torch.Tensor,
        *,
        solid: torch.Tensor | None = None,
        wall_mom_l1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Observe the control-volume force once per root step (L1 units).

        ``f_post_collision`` may be a single L1 post-collision state or the
        sequence of all L1 substep states of the root step.  The streaming
        momentum import is linear in the post-collision populations, so a
        sequence is summed first — this is required when the L1 block
        advances more than one substep per root step (the hybrid hierarchy
        advances ``ratio`` substeps), otherwise the import is undercounted
        by the missing substeps and the force is biased low.

        **P3 CV fix (momentum decomposition, see ``scripts/diag_cv_components``
        / ``scripts/diag_momentum_audit``):** ``f_new`` must be the live L1
        tensor snapshot taken after the block's own advance and **before**
        :func:`tensorlbm.octree_boundary.stepping.step_octree_shell` rewrites
        it in place.  The shell's restriction and reflux corrections are the
        internal bookkeeping that projects the leaf state (which already
        carries the wall force) back onto the L1 grid — counting their
        in-place momentum deltas as fluid momentum change double-counts the
        wall transfer and overestimates the CV drag.  The wall force itself
        is re-added explicitly via ``wall_mom_l1`` (the per-root-step leaf
        MEM force in L1 units, :meth:`wall_momentum_l1`), which is the
        physical momentum the shell injects into the L1 balance::

            F_cv = import - dP_L1advance + wall_mom_l1

        ``wall_mom_l1`` defaults to ``None`` (legacy behaviour: no explicit
        wall term, ``f_new`` taken as-is).
        """
        if isinstance(f_post_collision, (list, tuple)):
            if len(f_post_collision) == 0:
                raise ValueError("f_post_collision sequence must not be empty")
            summed = f_post_collision[0]
            for post in f_post_collision[1:]:
                summed = summed + post
            f_post_collision = summed
        result = observe_control_volume_force(
            f_old,
            f_new,
            f_post_collision,
            control_volume,
            solid=solid,
        )
        force = result.force_on_body
        if wall_mom_l1 is not None:
            w = torch.as_tensor(
                wall_mom_l1,
                dtype=self._accum.dtype,
                device=self._accum.device,
            )
            if w.shape != (3,):
                raise ValueError(f"wall_mom_l1 must be (3,), got {tuple(w.shape)}")
            if not bool(torch.isfinite(w).all()):
                raise FloatingPointError("non-finite wall_mom_l1")
            force = force + w
        self.cv_force = force.to(self._accum.dtype)
        self.cv_samples += 1
        return self.cv_force

    def wall_momentum_l1(self, dx_leaf: float, dt_leaf: float) -> torch.Tensor:
        """Per-root-step leaf wall force converted to L1 lattice units.

        This is the momentum the shell wall injects into the L1 control
        volume per root step (``mem_force`` is the substep-averaged MEM
        force; the convective conversion
        :func:`convert_leaf_force_to_l1` maps it into L1 units with the
        same dynamic-area normalisation the CV uses, so ``Cd_cv == Cd_mem``
        when the two closures agree).  Requires exactly ``2^d`` substeps
        (see :attr:`mem_force`).
        """
        return convert_leaf_force_to_l1(
            self.mem_force,
            dx_leaf,
            dt_leaf,
        )

    def deviation_pct(
        self,
        *,
        dx_leaf: float,
        dt_leaf: float,
        axis: int = 0,
    ) -> float | None:
        """MEM vs CV deviation on the streamwise axis in a common frame."""
        if self.cv_force is None or self.substep_count != self.n_substeps:
            return None
        mem_l1 = convert_leaf_force_to_l1(
            self.mem_force,
            dx_leaf,
            dt_leaf,
        )
        cv = self.cv_force
        return (
            abs(float(mem_l1[axis]) - float(cv[axis])) / max(abs(float(cv[axis])), 1.0e-30) * 100.0
        )

    def reset(self) -> None:
        """Start a fresh root step (keeps the device/dtype state)."""
        self._accum.zero_()
        self.cv_force = None
        self.cv_samples = 0
        self.substep_count = 0


def _cv_surface(cv: torch.Tensor) -> torch.Tensor:
    """Outer one-cell shell of a (box) control-volume mask."""
    interior = cv.clone()
    for shift in (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ):
        interior &= torch.roll(cv, shift, dims=(0, 1, 2))
    return cv & ~interior


def build_shell_control_volume(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    shell_band: float,
    margin: int,
    *,
    covered: torch.Tensor,
    filter_shell: torch.Tensor | None = None,
    solid: torch.Tensor | None = None,
    ghost: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Control volume around the body on the L1 with-ghost tensor (fail-closed).

    Args:
        shape: L1 with-ghost tensor shape ``(nz, ny, nx)``.
        center: sphere centre in with-ghost indices.
        radius: sphere radius in L1 cell units.
        shell_band: shell thickness (``bl_thickness + transition``) in L1
            cells — the CV box must enclose it.
        margin: extra cells beyond ``radius + shell_band``.
        covered: shell-covered mask on the L1 *physical* (ghost-free) grid.
        filter_shell: AMR interface filter blend (``> 0`` = filter shell);
            ``None`` skips the filter gate.
        solid: body mask on the with-ghost grid (used for the gates and the
            CV momentum balance); may be ``None`` to skip the body gates.
        ghost: ghost width of the L1 block (1).

    Returns:
        The ``bool`` control-volume mask (strictly interior).

    Raises:
        ValueError: fail-closed — CV surface intersects the shell interface,
            the interface filter shell or the body; the covered region or
            the body is not fully enclosed; or the CV degenerates.
    """
    if covered.shape != tuple(s - 2 * ghost for s in shape):
        raise ValueError(
            "covered mask must have the physical (ghost-free) L1 shape",
        )
    nz, ny, nx = shape
    half = int(math.ceil(radius + shell_band)) + int(margin)

    def _lo(c: float) -> int:
        # keep the CV surface >= 1 physical cell away from the L1 block
        # boundary (the with-ghost grid's outermost ghost/interface cells)
        return int(max(ghost + 1, math.floor(c - half)))

    def _hi(c: float, limit: int) -> int:
        return int(min(limit - ghost - 2, math.ceil(c + half) + 1))

    x0, x1 = _lo(center[0]), _hi(center[0], nx)
    y0, y1 = _lo(center[1]), _hi(center[1], ny)
    z0, z1 = _lo(center[2]), _hi(center[2], nz)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError(
            "control volume degenerates on the L1 block — enlarge the L1 "
            "block or reduce --cv-margin",
        )
    cv = box_control_volume(
        (int(nz), int(ny), int(nx)),
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        z0=z0,
        z1=z1,
        device=device,
    )
    surface = _cv_surface(cv)

    covered_g = torch.zeros(shape, dtype=torch.bool, device=device)
    covered_g[ghost:-ghost, ghost:-ghost, ghost:-ghost] = covered
    if bool((surface & covered_g).any()):
        raise ValueError(
            "CV surface intersects the shell interface (fail-closed) — increase --cv-margin",
        )
    if bool((covered_g & ~cv).any()):
        raise ValueError(
            "shell-covered region is not fully enclosed by the CV "
            "(fail-closed) — increase --cv-margin",
        )
    if filter_shell is not None:
        if filter_shell.shape != tuple(shape):
            raise ValueError("filter_shell must match the with-ghost shape")
        if bool((surface & (filter_shell > 0)).any()):
            raise ValueError(
                "CV surface intersects the AMR interface filter shell "
                "(fail-closed) — enlarge the L1 block or shrink the CV",
            )
    if solid is not None:
        if solid.shape != tuple(shape) or solid.dtype is not torch.bool:
            raise ValueError("solid must be a bool with-ghost mask")
        if bool((surface & solid).any()):
            raise ValueError(
                "CV surface intersects the body (fail-closed)",
            )
        if bool((solid & ~cv).any()):
            raise ValueError(
                "body is not fully enclosed by the CV (fail-closed)",
            )
    return cv


__all__ = [
    "ShellForceLedger",
    "build_shell_control_volume",
    "convert_leaf_force_to_l1",
    "substep_force_weights",
]
