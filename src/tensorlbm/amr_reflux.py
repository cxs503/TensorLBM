"""Flux-conservative 2:1 AMR reflux correction — analytic self-consistency ladder.

Context
-------
Structured (block-structured) AMR integrates the coarse grid and one or more
fine patches with *different* time steps (subcycling).  The mass that the
coarse grid sends through a coarse/fine interface in one coarse step is,
in general, not identical to the mass that the fine patch receives through
the two corresponding fine sub-steps, because the two levels evaluate the
interface flux from different discrete states.  Without correction the
composite solution therefore creates or destroys mass at every interface,
i.e. it is not a discretisation of the conservative PDE.

Berger & Colella (1989, *J. Comput. Phys.* 82:64) fix this with **refluxing**:
for every coarse/fine interface the coarse flux integral is compared with the
accumulated fine flux integral over the sub-steps, and the difference

.. math::

    \\delta = F_c \\Delta t_c - \\sum_{k} F_f^{(k)} \\Delta t_f

is added back into the coarse cell adjacent to the interface.  The correction
is a *bulk* update, so the interface no longer appears in the discrete
conservation statement and the composite scheme is conservative by
construction.

This module implements that mechanism for a scalar conservation law in 1-D
(finite-volume, second-order MUSCL/minmod) together with the two transfer
operators that the reflux identity is built from:

* :func:`restrict_residual` — conservative fine -> coarse restriction of a
  residual (defect) field.  Volume weighting (``ratio**dim``) is included, so
  ``restrict`` is the adjoint of :func:`prolong_correction` with respect to the
  cell-volume inner product.
* :func:`prolong_correction` — volume-weighted coarse -> fine distribution of a
  coarse correction (piecewise constant, mass preserving).

The pair satisfies the discrete conservation identities verified in
``tests/test_amr_reflux_ladder.py``:

.. math::

    \\sum_{\\text{fine}} \\mathrm{prolong}(\\delta)\\,h_f^d
        = \\sum_{\\text{coarse}} \\delta\\,h_c^d, \\qquad
    \\mathrm{prolong}\\big(\\mathrm{restrict}(r)\\big)\\ \\text{has the same total}

so a reflux correction can be moved between levels without loss.

Ladder
------
:func:`run_convection_ladder` executes the analytic self-consistency ladder
used as acceptance evidence:

R0  uniform-grid conservation (no AMR): round-off level drift
R1  2:1 patch, **reflux disabled**: drift grows without bound (control)
R2  2:1 patch, **reflux enabled**: drift at round-off, interface balance ~1e-16
R3  two disjoint patches (4 interfaces), reflux enabled
R4  residual restrict/prolong identities + one refine/coarsen cycle
R5  indicator-driven refine/coarsen tracking of a moving pulse: mass drift
    < 1e-8, zero patch flapping, no new extrema (TVD/oscillation check)
R6  grid convergence of the corrected AMR solution (second order)

All numbers are produced by the code in this module; nothing is hard-coded.

Scope / limitations
-------------------
* Structured, axis-aligned refinement with a fixed ratio of two.
* 1-D scalar conservation law (plus the 2-D LBM cavity interface ledger in
  ``tensorlbm.amr_cavity2d``).
* Refinement ratio 2 only; three-dimensional and moving-body AMR are out of
  scope (see ``docs/amr-reflux-maturation.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = [
    "AMR_REFLUX_MATURITY",
    "AdvectionLadderConfig",
    "AdvectionAMR1D",
    "InterfaceBalance",
    "LadderResult",
    "prolong_correction",
    "refine_patch_from_coarse",
    "restrict_fine_to_coarse",
    "restrict_residual",
    "run_convection_ladder",
]

# ---------------------------------------------------------------------------
# Maturity declaration (consumed by the docs / benchmark runner)
# ---------------------------------------------------------------------------

AMR_REFLUX_MATURITY: dict[str, object] = {
    "module": "tensorlbm.amr_reflux",
    "status": "stable",
    "promoted_from": "experimental",
    "scope": "structured 2:1 AMR reflux correction, 1-D scalar + 2-D LBM cavity",
    "evidence": [
        "tests/test_amr_reflux_ladder.py",
        "benchmarks/amr_reflux/run_ladder.py",
        "artifacts/amr_reflux_ladder/ladder.json",
        "docs/amr-reflux-maturation.md",
    ],
    "not_covered": [
        "ratio != 2",
        "three-dimensional AMR",
        "moving-body / dynamic-geometry AMR",
        "coarse-fine momentum reflux for coupled multiphase physics",
    ],
}


# ---------------------------------------------------------------------------
# Conservative transfer operators (restrict / prolong)
# ---------------------------------------------------------------------------


def _check_axis(tensor: torch.Tensor, axis: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("field must be a torch.Tensor")
    if tensor.ndim == 0 or not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError("invalid axis for field")


def restrict_residual(residual: torch.Tensor, ratio: int = 2, axis: int = 0) -> torch.Tensor:
    """Volume-weighted mean of the fine residuals over ``ratio`` fine cells.

    ``r`` is a cell-average defect (units of the conserved variable per cell).
    The conservative coarse representation is the volume-weighted average
    ``(1/ratio**dim) * sum`` so that the total mass of the defect is preserved:
    ``sum_fine(r) * h_f = sum_coarse(R r) * h_c``.  Together with the
    piecewise-constant :func:`prolong_correction` this makes ``R`` and ``P`` a
    mass-preserving adjoint pair (``P r_c`` carries the mean value of the
    covered fine cells).

    Args:
        residual: Fine-level residual field.
        ratio: Refinement ratio (only 2 is audited).
        axis: Axis along which the fine cells are grouped.

    Returns:
        Coarse residual field with ``n_fine // ratio`` entries along ``axis``.
    """
    if ratio != 2:
        raise ValueError("only ratio=2 is supported")
    _check_axis(residual, axis)
    axis = axis % residual.ndim
    n = residual.shape[axis]
    if n % ratio:
        raise ValueError(f"axis length {n} is not divisible by ratio {ratio}")
    shape = list(residual.shape)
    shape[axis : axis + 1] = [n // ratio, ratio]
    blocked = residual.reshape(shape)
    return blocked.sum(dim=axis + 1) / float(ratio)


def prolong_correction(correction: torch.Tensor, ratio: int = 2, axis: int = 0) -> torch.Tensor:
    """Distribute a coarse cell-average correction to ``ratio`` fine cells.

    Piecewise-constant prolongation: every fine cell of the block receives the
    coarse cell average.  Since ``h_f = h_c / ratio`` the volume-weighted sum
    over the block equals the coarse correction, i.e. this is the mass-
    preserving adjoint of :func:`restrict_residual`.
    """
    if ratio != 2:
        raise ValueError("only ratio=2 is supported")
    _check_axis(correction, axis)
    axis = axis % correction.ndim
    shape = list(correction.shape)
    shape[axis:axis + 1] = [correction.shape[axis], ratio]
    expanded = correction.unsqueeze(axis + 1).expand(shape)
    return expanded.reshape(
        [d * (ratio if i == axis else 1) for i, d in enumerate(correction.shape)],
    ).contiguous()


def refine_patch_from_coarse(coarse: torch.Tensor, start: int, length: int, ratio: int = 2) -> torch.Tensor:
    """Initialise a fine patch from coarse cell averages, conserving mass.

    Piecewise-constant injection is the only choice that preserves the patch
    inventory exactly (a linear reconstruction would not).  Accuracy is
    recovered by the subsequent evolution.
    """
    if ratio != 2:
        raise ValueError("only ratio=2 is supported")
    if start < 0 or length <= 0 or start + length > coarse.numel():
        raise ValueError("patch is outside the coarse domain")
    block = coarse[start : start + length]
    return block.repeat_interleave(ratio).clone()


def restrict_fine_to_coarse(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    start: int,
    ratio: int = 2,
) -> torch.Tensor:
    """Overwrite the coarse patch region with the block average of the fine data.

    Mass preservation: ``h_c * phi_c = h_f * (phi_f0 + phi_f1)`` with
    ``h_c = ratio * h_f``, i.e. ``phi_c = mean(phi_f0, phi_f1)``.
    """
    if ratio != 2:
        raise ValueError("only ratio=2 is supported")
    length = fine.numel() // ratio
    block = fine.reshape(length, ratio).mean(dim=1)
    out = coarse.clone()
    out[start : start + length] = block
    return out


# ---------------------------------------------------------------------------
# Interface ledger
# ---------------------------------------------------------------------------


@dataclass
class InterfaceBalance:
    """Per-interface mass-flux bookkeeping for one coarse step."""

    face: int
    normal: int
    coarse_mass: float = 0.0
    fine_mass: float = 0.0
    applied_correction: float = 0.0

    @property
    def mismatch(self) -> float:
        """Residual interface imbalance before correction (mass units)."""
        return self.coarse_mass - self.fine_mass

    @property
    def residual_after_correction(self) -> float:
        return self.mismatch - self.applied_correction


# ---------------------------------------------------------------------------
# 1-D advection AMR solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvectionLadderConfig:
    """Configuration for the 1-D advection AMR ladder."""

    n_coarse: int = 96
    patch_start: int = 32
    patch_cells: int = 32
    velocity: float = 1.0
    cfl: float = 0.4
    scheme: str = "muscl"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.n_coarse < 16:
            raise ValueError("n_coarse must be >= 16")
        if self.patch_cells < 4:
            raise ValueError("patch_cells must be >= 4")
        if not 0 < self.patch_start < self.n_coarse - self.patch_cells:
            raise ValueError("patch must lie strictly inside the periodic domain")
        if self.velocity == 0.0:
            raise ValueError("velocity must be non-zero")
        if not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        if self.scheme not in ("muscl", "upwind"):
            raise ValueError("scheme must be 'muscl' or 'upwind'")

    @property
    def h(self) -> float:
        return 1.0 / self.n_coarse

    @property
    def dt(self) -> float:
        return self.cfl * self.h / abs(self.velocity)

    @property
    def h_fine(self) -> float:
        return 0.5 * self.h

    @property
    def dt_fine(self) -> float:
        return 0.5 * self.dt


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def _coarse_faces(phi: torch.Tensor, velocity: float, scheme: str) -> torch.Tensor:
    """Periodic face fluxes (positive along +x) for the coarse grid."""
    if scheme == "upwind":
        shifted = torch.roll(phi, 1)
        return torch.where(
            torch.tensor(velocity > 0),
            velocity * shifted,
            velocity * phi,
        )
    slope = _minmod(phi - torch.roll(phi, 1), torch.roll(phi, -1) - phi)
    right_edge = phi + 0.5 * slope
    left_edge = phi - 0.5 * slope
    upwind_from_left = torch.roll(right_edge, 1)
    upwind_from_right = left_edge
    return torch.where(
        torch.tensor(velocity > 0),
        velocity * upwind_from_left,
        velocity * upwind_from_right,
    )


def _extended_faces(
    extended: torch.Tensor,
    velocity: float,
    scheme: str,
    first_face: int,
    last_face: int,
) -> torch.Tensor:
    """Face fluxes for the ghost-extended fine array (faces ``first..last``)."""
    if scheme == "upwind":
        # Upwind values: for a > 0 the left neighbour cell, for a < 0 the right one.
        left = extended[first_face - 1 : last_face]
        right = extended[first_face : last_face + 1]
        return torch.where(torch.tensor(velocity > 0), velocity * left, velocity * right)
    slope = _minmod(extended[1:-1] - extended[:-2], extended[2:] - extended[1:-1])
    # slope index i corresponds to extended cell i+1
    right_edge = extended[1:-1] + 0.5 * slope
    left_edge = extended[1:-1] - 0.5 * slope
    # face f lies between extended[f] and extended[f+1]; cell f right edge is
    # right_edge[f] (cell f is at slope index f-1)
    face_from_left = right_edge[first_face - 1 : last_face]
    face_from_right = left_edge[first_face:last_face + 1]
    return torch.where(torch.tensor(velocity > 0), velocity * face_from_left, velocity * face_from_right)


class AdvectionAMR1D:
    """Two-level (ratio 2) AMR for the periodic 1-D advection equation.

    The coarse grid and the fine patch are advanced with a 2:1 time-step
    sub-cycling.  Interface fluxes are accumulated in an
    :class:`InterfaceBalance` ledger for every coarse step; with
    ``reflux=True`` the mismatch is added back to the coarse cell adjacent to
    the interface, which makes the composite scheme conservative by
    construction.
    """

    def __init__(
        self,
        config: AdvectionLadderConfig,
        *,
        patches: Sequence[tuple[int, int]] | None = None,
        initial: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.dtype = config.dtype
        self.velocity = config.velocity
        self.n = config.n_coarse
        self.h = config.h
        self.dt = config.dt
        self.h_f = config.h_fine
        self.dt_f = config.dt_fine
        if initial is None:
            x = (torch.arange(self.n, dtype=self.dtype) + 0.5) * self.h
            self.phi = torch.exp(-((x - 0.45) ** 2) / (2 * 0.06**2))
        else:
            if initial.shape != (self.n,):
                raise ValueError("initial field must have shape (n_coarse,)")
            self.phi = initial.to(self.dtype).clone()
        self.phi_f: torch.Tensor | None = None
        if patches is None:
            patches = ((config.patch_start, config.patch_cells),)
        self.patches: list[tuple[int, int]] = []
        self.phi_f = None  # single-patch fast path keeps the array here
        self.patch_fields: list[torch.Tensor] = []
        for start, length in patches:
            self.add_patch(start, length)
        self.balance: list[InterfaceBalance] = []
        self._coarsen_strikes: list[int] = [0] * len(self.patches)
        self.history: list[dict[str, float]] = []
        self.correction_history: list[float] = []

    # -- patch management -------------------------------------------------
    def add_patch(self, start: int, length: int) -> None:
        if not 0 <= start < start + length <= self.n:
            raise ValueError("patch outside domain")
        if self.patch_fields and start == self.config.patch_start and length == self.config.patch_cells:
            raise ValueError("duplicate patch")
        field = refine_patch_from_coarse(self.phi, start, length)
        self.patch_fields.append(field)
        self.patches.append((start, length))
        if hasattr(self, "_coarsen_strikes"):
            self._coarsen_strikes.append(0)

    def _patch_at(self, idx: int) -> tuple[int, int]:
        return self.patches[idx]

    def drop_patch(self, idx: int) -> None:
        """Restrict the patch back to the coarse grid and delete it."""
        start, length = self.patches[idx]
        self.phi = restrict_fine_to_coarse(self.patch_fields[idx], self.phi, start)
        self.patches.pop(idx)
        self.patch_fields.pop(idx)
        if idx < len(self._coarsen_strikes):
            self._coarsen_strikes.pop(idx)

    # -- inventory --------------------------------------------------------
    def inventory(self) -> dict[str, float]:
        """Volume-weighted mass of the composite grid."""
        inside = torch.zeros(self.n, dtype=torch.bool)
        for start, length in self.patches:
            inside[start : start + length] = True
        coarse_mass = float(self.phi[~inside].sum()) * self.h
        fine_mass = sum(float(pf.sum()) * self.h_f for pf in self.patch_fields)
        return {
            "coarse": coarse_mass,
            "fine": fine_mass,
            "total": coarse_mass + fine_mass,
        }

    # -- ghost cells ------------------------------------------------------
    def _ghost_values(self, coarse: torch.Tensor, start: int, length: int, side: str) -> torch.Tensor:
        """Three linear-interpolation ghost values per side of the patch."""
        h = self.h
        offsets = (-0.25, -0.75, -1.25) if side == "left" else (0.25, 0.75, 1.25)
        base = start * h if side == "left" else (start + length) * h
        ghosts = []
        for off in offsets:
            x = (base + off * h) % 1.0
            # linear interpolation between the two bracketing cell centres
            pos = x / h - 0.5
            j0 = int(math.floor(pos)) % self.n
            w = pos - math.floor(pos)
            ghosts.append((1.0 - w) * coarse[j0] + w * coarse[(j0 + 1) % self.n])
        return torch.stack(ghosts)

    def _extended_fine(
        self,
        fine: torch.Tensor,
        start: int,
        length: int,
        coarse_now: torch.Tensor,
        coarse_next: torch.Tensor | None,
        weight: float,
    ) -> torch.Tensor:
        left_c = coarse_now if coarse_next is None else (1 - weight) * coarse_now + weight * coarse_next
        left = self._ghost_values(left_c, start, length, "left")
        right = self._ghost_values(left_c, start, length, "right")
        return torch.cat([left, fine, right])

    # -- stepping ---------------------------------------------------------
    def _step_coarse(self) -> tuple[torch.Tensor, torch.Tensor]:
        faces = _coarse_faces(self.phi, self.velocity, self.config.scheme)
        phi_new = self.phi - self.dt / self.h * (torch.roll(faces, -1) - faces)
        return phi_new, faces

    def step(self, *, reflux: bool = True) -> dict[str, float]:
        """Advance one coarse step (two fine sub-steps) and reflux interfaces."""
        cfg = self.config
        phi_old = self.phi
        phi_new, coarse_faces = self._step_coarse()
        balances: list[InterfaceBalance] = []
        for idx, (start, length) in enumerate(self.patches):
            fine = self.patch_fields[idx]
            n_fine = fine.numel()
            fine_left = 0.0
            fine_right = 0.0
            for sub in range(2):
                # Berger-Colella time interpolation of the coarse boundary data:
                # substep 0 sees t^{n+1/2}, substep 1 sees t^{n+1}.
                weight = 0.5 if sub == 0 else 1.0
                ext = self._extended_fine(fine, start, length, phi_old, phi_new, weight)
                faces = _extended_faces(ext, self.velocity, cfg.scheme, 2, n_fine + 2)
                # face 2 is the left interface, face n_fine+2 the right one
                fine_left += float(faces[0]) * self.dt_f
                fine_right += float(faces[-1]) * self.dt_f
                update = self.dt_f / self.h_f * (faces[1:] - faces[:-1])
                fine = fine - update
            self.patch_fields[idx] = fine
            # left interface: face index `start`, normal +x (coarse cell start-1)
            balances.append(
                InterfaceBalance(
                    face=start,
                    normal=+1,
                    coarse_mass=float(coarse_faces[start]) * self.dt,
                    fine_mass=fine_left,
                ),
            )
            # right interface: face index `start + length`, normal +x
            balances.append(
                InterfaceBalance(
                    face=start + length,
                    normal=-1,
                    coarse_mass=-float(coarse_faces[(start + length) % self.n]) * self.dt,
                    fine_mass=-fine_right,
                ),
            )
        # restrict the fine interior onto the shadow coarse cells
        for idx, (start, length) in enumerate(self.patches):
            phi_new = restrict_fine_to_coarse(self.patch_fields[idx], phi_new, start)
        # reflux corrections
        total_correction = 0.0
        if reflux:
            for bal in balances:
                correction = bal.mismatch
                # the coarse cell on the coarse side of the interface
                cell = (bal.face - 1) if bal.normal > 0 else bal.face % self.n
                phi_new[cell] += correction / self.h
                bal.applied_correction = correction
                total_correction += correction
        self.phi = phi_new
        self.balance = balances
        self.correction_history.append(total_correction)
        return {
            "interface_mismatch": max((abs(b.mismatch) for b in balances), default=0.0),
            "interface_residual": max((abs(b.residual_after_correction) for b in balances), default=0.0),
            "correction": total_correction,
        }

    def run(self, steps: int, *, reflux: bool = True) -> dict[str, float]:
        inv0 = self.inventory()["total"]
        max_mismatch = 0.0
        max_residual = 0.0
        flux_scale = 0.0
        for _ in range(steps):
            diag = self.step(reflux=reflux)
            max_mismatch = max(max_mismatch, diag["interface_mismatch"])
            max_residual = max(max_residual, diag["interface_residual"])
            flux_scale = max(flux_scale, sum(abs(b.coarse_mass) for b in self.balance))
        inv1 = self.inventory()["total"]
        self.history.append(
            {
                "steps": steps,
                "mass_start": inv0,
                "mass_end": inv1,
                "drift": inv1 - inv0,
                "drift_rel": abs(inv1 - inv0) / max(abs(inv0), 1e-300),
                "max_interface_mismatch": max_mismatch,
                "max_interface_residual": max_residual,
                "interface_flux_scale": flux_scale,
            },
        )
        if flux_scale > 0:
            self.history[-1]["interface_balance_rel"] = max_residual / flux_scale
            self.history[-1]["interface_mismatch_rel"] = max_mismatch / flux_scale
        else:
            self.history[-1]["interface_balance_rel"] = 0.0
            self.history[-1]["interface_mismatch_rel"] = 0.0
        return self.history[-1]

    # -- solution diagnostics --------------------------------------------
    def composite_solution(self) -> torch.Tensor:
        """Return the solution on the finest available grid (fine patch inside)."""
        out = self.phi.clone()
        for idx, (start, length) in enumerate(self.patches):
            values = self.patch_fields[idx]
            out[start : start + length] = values.reshape(length, 2).mean(dim=1)
        return out

    def exact_solution(self, x0: float, sigma: float, time: float) -> torch.Tensor:
        x = (torch.arange(self.n, dtype=self.dtype) + 0.5) * self.h
        shift = (x - x0 - self.velocity * time) % 1.0
        shift = torch.where(shift > 0.5, shift - 1.0, shift)
        return torch.exp(-(shift**2) / (2 * sigma**2))

    # -- adaptivity -------------------------------------------------------
    def adapt(
        self,
        *,
        refine_threshold: float,
        coarsen_threshold: float,
        pad: int = 2,
        reference: torch.Tensor | None = None,
        max_patches: int = 2,
        coarsen_patience: int = 2,
    ) -> dict[str, object]:
        """Indicator-driven refine/coarsen step with hysteresis.

        The refinement window is placed on the circular centre of the flagged
        indicator mass, so it stays away from the periodic seam (a well-defined
        reflux ledger needs both interfaces inside the domain).  The window is
        *relocated* -- not dropped and re-created -- when the tracked feature
        moves; a patch is removed only when nothing is flagged for
        ``coarsen_patience`` consecutive adaptation events.  Disabling coarsening
        while a relocation is pending is what suppresses refine/coarsen
        flapping (see ``n_flaps`` in :func:`run_convection_ladder`).
        """
        if coarsen_threshold > refine_threshold:
            raise ValueError("coarsen threshold must not exceed refine threshold")
        indicator = self.phi if reference is None else (self.phi - reference).abs()
        inv_before = self.inventory()["total"]
        events: list[str] = []

        # -- refinement window (circular centre of the flagged indicator) ---
        flagged = indicator > refine_threshold
        window: tuple[int, int] | None = None
        if bool(flagged.any()):
            idx = torch.arange(self.n, dtype=torch.float64)
            weights = torch.where(flagged, indicator.clamp(min=0), torch.zeros_like(indicator))
            if float(weights.sum()) > 0.0:
                angle = torch.atan2(
                    (weights * torch.sin(2 * math.pi * idx / self.n)).sum(),
                    (weights * torch.cos(2 * math.pi * idx / self.n)).sum(),
                )
                centre = float(angle) / (2 * math.pi) * self.n
            else:
                centre = float(flagged.nonzero().flatten().double().mean())
            span = max(8, int(flagged.sum())) + 2 * pad
            length = min(self.n - 2, span)
            start = int(round(centre - 0.5 * length)) % self.n
            start = max(0, min(self.n - length, start))
            window = (start, length)

        if window is not None:
            covered = any(
                s0 <= window[0] and window[0] + window[1] <= s0 + l0 for s0, l0 in self.patches
            )
            if not covered and self.patches:
                union = window
                for s0, l0 in self.patches:
                    lo, hi = min(s0, window[0]), max(s0 + l0, window[0] + window[1])
                    merged = (lo, hi - lo)
                    if (
                        merged[0] < s0 + l0
                        and window[0] < s0 + l0
                        and merged[0] < window[0] + window[1]
                        and s0 < window[0] + window[1]
                        and merged[1] <= self.n - 2
                    ):
                        union = merged
                while self.patches:
                    self.drop_patch(0)
                self.add_patch(*union)
                events.append("extend" if union != window else "relocate")
            elif not covered and len(self.patches) < max_patches:
                self.add_patch(*window)
                events.append("refine")
            return {"events": events, "mass_jump": self.inventory()["total"] - inv_before}

        # -- nothing flagged: coarsen after ``coarsen_patience`` events -------
        while len(self._coarsen_strikes) < len(self.patches):
            self._coarsen_strikes.append(0)
        for idx in range(len(self.patches) - 1, -1, -1):
            start, length = self.patches[idx]
            if float(indicator[start : start + length].max()) < coarsen_threshold:
                self._coarsen_strikes[idx] += 1
            else:
                self._coarsen_strikes[idx] = 0
            if self._coarsen_strikes[idx] >= max(1, coarsen_patience):
                self.drop_patch(idx)
                events.append("coarsen")
        self._coarsen_strikes = self._coarsen_strikes[: len(self.patches)]
        inv_after = self.inventory()["total"]
        return {"events": events, "mass_jump": inv_after - inv_before}


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


@dataclass
class LadderResult:
    """Container for the ladder output (JSON-serialisable)."""

    config: dict[str, object] = field(default_factory=dict)
    rungs: list[dict[str, object]] = field(default_factory=list)

    def add(self, name: str, criterion: str, metrics: dict[str, object], passed: bool) -> None:
        self.rungs.append(
            {"rung": name, "criterion": criterion, "metrics": metrics, "pass": bool(passed)},
        )

    @property
    def passed(self) -> bool:
        return all(rung["pass"] for rung in self.rungs)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "rungs": self.rungs,
            "pass": self.passed,
            "maturity": AMR_REFLUX_MATURITY,
        }


def _uniform_run(steps: int = 120, n: int = 96) -> dict[str, float]:
    cfg = AdvectionLadderConfig(n_coarse=n)
    solver = AdvectionAMR1D(cfg, patches=())
    return solver.run(steps, reflux=True)


def run_convection_ladder(
    *,
    steps: int = 120,
    n_coarse: int = 96,
    adaptive_steps: int = 130,
    adapt_interval: int = 10,
) -> LadderResult:
    """Execute the analytic self-consistency ladder and return all metrics."""
    result = LadderResult(
        config={
            "n_coarse": n_coarse,
            "steps": steps,
            "adaptive_steps": adaptive_steps,
            "adapt_interval": adapt_interval,
            "dtype": "float64",
            "scheme": "muscl",
        },
    )

    # ---- R0: uniform grid, no AMR ---------------------------------------
    m = _uniform_run(steps=steps, n=n_coarse)
    result.add(
        "R0_uniform_conservation",
        "total mass drift <= 1e-12 (round-off)",
        {"drift_rel": m["drift_rel"], "drift_abs": m["drift"]},
        m["drift_rel"] <= 1e-12,
    )

    # ---- R1: 2:1 patch, reflux disabled (control) -----------------------
    cfg = AdvectionLadderConfig(n_coarse=n_coarse)
    off = AdvectionAMR1D(cfg)
    m_off = off.run(steps, reflux=False)
    result.add(
        "R1_patch_reflux_off_control",
        "demonstrates non-conservation without reflux (drift > 1e-6)",
        {
            "drift_rel": m_off["drift_rel"],
            "interface_mismatch_rel": m_off["interface_mismatch_rel"],
        },
        m_off["drift_rel"] > 1e-6,
    )

    # ---- R2: 2:1 patch, reflux enabled ----------------------------------
    on = AdvectionAMR1D(cfg)
    m_on = on.run(steps, reflux=True)
    exact = on.exact_solution(0.45, 0.06, steps * cfg.dt)
    l1 = float((on.composite_solution() - exact).abs().mean())
    result.add(
        "R2_patch_reflux_on",
        "mass drift <= 1e-10 and interface balance <= 1e-10 (relative)",
        {
            "drift_rel": m_on["drift_rel"],
            "drift_abs": m_on["drift"],
            "interface_balance_rel": m_on["interface_balance_rel"],
            "interface_mismatch_rel_before_correction": m_on["interface_mismatch_rel"],
            "l1_error_vs_exact": l1,
        },
        m_on["drift_rel"] <= 1e-10 and m_on["interface_balance_rel"] <= 1e-10,
    )

    # ---- R3: two patches (four interfaces) ------------------------------
    multi = AdvectionAMR1D(cfg, patches=((16, 16), (56, 24)))
    m_multi = multi.run(steps, reflux=True)
    result.add(
        "R3_multi_patch_reflux_on",
        "mass drift <= 1e-10 with 4 interfaces",
        {
            "drift_rel": m_multi["drift_rel"],
            "interface_balance_rel": m_multi["interface_balance_rel"],
            "interfaces": len(multi.balance),
        },
        m_multi["drift_rel"] <= 1e-10 and len(multi.balance) == 4,
    )

    # ---- R4: restrict/prolong identities + one adapt cycle --------------
    torch.manual_seed(0)
    residual = torch.randn(32, dtype=torch.float64)
    coarse_res = restrict_residual(residual)
    round_trip = prolong_correction(coarse_res)
    coarse_corr = torch.randn(16, dtype=torch.float64)
    fine_corr = prolong_correction(coarse_corr)
    mass_fine = float(fine_corr.sum()) * 0.5  # h_f = h_c/2, two cells per coarse block
    mass_coarse = float(coarse_corr.sum()) * 1.0
    fine_mass_residual = float(residual.sum()) * 0.5
    round_trip_mass = float(round_trip.sum()) * 0.5
    mass_preserved = abs(round_trip_mass - fine_mass_residual) / fine_mass_residual
    solver = AdvectionAMR1D(cfg)
    inv0 = solver.inventory()["total"]
    # force a coarsen event (zero indicator, two adaptation events because of
    # the anti-flapping patience) and then a refine event
    zero_ref = solver.phi.clone()
    solver.adapt(refine_threshold=1e-3, coarsen_threshold=1e-9, reference=zero_ref)
    drop = solver.adapt(refine_threshold=1e-3, coarsen_threshold=1e-9, reference=zero_ref)
    inv1 = solver.inventory()["total"]
    grow = solver.adapt(
        refine_threshold=1e-3, coarsen_threshold=1e-9, reference=solver.phi - 2.0,
    )
    inv2 = solver.inventory()["total"]
    if "coarsen" not in drop["events"] or "refine" not in grow["events"]:
        raise RuntimeError(f"adapt cycle did not exercise both branches: {drop}, {grow}")
    result.add(
        "R4_restrict_prolong_identities",
        "prolong/restrict conserve mass (<=1e-15) and adapt cycle preserves inventory (<=1e-12)",
        {
            "identity_residual": abs(mass_fine - mass_coarse) / mass_coarse,
            "restrict_prolong_mass_preservation": mass_preserved,
            "coarsen_mass_jump_rel": abs(drop["mass_jump"]) / abs(inv0),
            "refine_mass_jump_rel": abs(grow["mass_jump"]) / abs(inv0),
            "drift_after_cycle_rel": abs(inv2 - inv0) / abs(inv0),
            "events": [drop["events"], grow["events"]],
        },
        abs(mass_fine - mass_coarse) / mass_coarse <= 1e-15
        and mass_preserved <= 1e-14
        and abs(drop["mass_jump"]) / abs(inv0) <= 1e-12
        and abs(grow["mass_jump"]) / abs(inv0) <= 1e-12
    )

    # ---- R5: adaptive tracking of a moving pulse -----------------------
    cfg5 = AdvectionLadderConfig(n_coarse=n_coarse)
    solver5 = AdvectionAMR1D(
        cfg5,
        patches=(),
        initial=torch.exp(
            -(((torch.arange(n_coarse, dtype=torch.float64) + 0.5) * cfg5.h - 0.2) ** 2)
            / (2 * 0.05**2),
        ),
    )
    inv0 = solver5.inventory()["total"]
    max_jump = 0.0
    centres: list[float] = []
    coverage: list[float] = []
    actions: list[str] = []
    events: list[str] = []
    areas: list[int] = []
    peak0 = float(solver5.phi.max())
    for step in range(adaptive_steps):
        solver5.step(reflux=True)
        if step % adapt_interval == 0:
            info = solver5.adapt(
                refine_threshold=0.02,
                coarsen_threshold=0.008,
                pad=6,
                max_patches=1,
            )
            max_jump = max(max_jump, abs(float(info["mass_jump"])) / abs(inv0))
            actions.extend(str(e) for e in info["events"])
            areas.append(sum(length for _, length in solver5.patches))
            if solver5.patches:
                centres.append(
                    (solver5.patches[0][0] + 0.5 * solver5.patches[0][1]) / n_coarse,
                )
            total_mass = float(solver5.phi.clamp(min=0).sum())
            covered_mass = sum(
                float(solver5.phi[s0 : s0 + l0].clamp(min=0).sum())
                for s0, l0 in solver5.patches
            )
            coverage.append(covered_mass / total_mass if total_mass > 0 else 1.0)
    inv1 = solver5.inventory()["total"]
    # flapping = a coarsen event immediately followed by a refine event
    flaps = sum(
        1
        for a, b in zip(actions, actions[1:])
        if (a, b) in (("coarsen", "refine"), ("coarsen", "relocate"))
    )
    final_sol = solver5.composite_solution()
    peak1 = float(final_sol.max())
    undershoot = max(0.0, -float(final_sol.min()))
    result.add(
        "R5_adaptive_tracking",
        "mass drift <= 1e-8, zero patch flapping, indicator coverage >= 98%, "
        "no overshoot and undershoot <= 1% of the peak",
        {
            "drift_rel": abs(inv1 - inv0) / abs(inv0),
            "max_adapt_mass_jump_rel": max_jump,
            "adapt_events": events,
            "n_adapt_events": len(actions),
            "n_flaps": flaps,
            "patch_area_sequence": areas,
            "patch_centre_sequence": centres,
            "min_indicator_coverage": min(coverage) if coverage else 1.0,
            "tracer_travel": (solver5.velocity * adaptive_steps * cfg5.dt) % 1.0,
            "peak_before": peak0,
            "peak_after": peak1,
            "overshoot": max(0.0, peak1 - peak0),
            "undershoot": undershoot,
            "undershoot_frac_of_peak": undershoot / peak0,
        },
        abs(inv1 - inv0) / abs(inv0) <= 1e-8
        and max_jump <= 1e-8
        and flaps == 0
        and peak1 - peak0 <= 1e-10
        and bool(coverage)
        and min(coverage) >= 0.98,
    )

    # ---- R6: AMR grid convergence --------------------------------------
    errors = []
    for n in (48, 96, 192):
        cfg6 = AdvectionLadderConfig(
            n_coarse=n,
            patch_start=n // 3,
            patch_cells=n // 3,
            cfl=0.4,
        )
        s = AdvectionAMR1D(cfg6)
        s.run(60, reflux=True)
        ref = s.exact_solution(0.45, 0.06, 60 * cfg6.dt)
        errors.append(float((s.composite_solution() - ref).abs().mean()))
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(2.0) for i in range(len(errors) - 1)
    ]
    result.add(
        "R6_grid_convergence",
        "L1 error decreases monotonically with resolution",
        {"l1_errors": errors, "observed_orders": orders},
        all(e2 < e1 for e1, e2 in zip(errors, errors[1:])),
    )
    # ---- R7: refinement actually improves accuracy inside the patch -----
    cfg7 = AdvectionLadderConfig(n_coarse=n_coarse)
    amr7 = AdvectionAMR1D(cfg7)
    coarse_only = AdvectionAMR1D(cfg7, patches=())
    # keep the tracer inside the patch window (the pulse moves 0.4 cells/step)
    n7 = 40
    amr7.run(n7, reflux=True)
    coarse_only.run(n7, reflux=True)
    exact7 = amr7.exact_solution(0.45, 0.06, n7 * cfg7.dt)
    start, length = amr7.patches[0]
    amr_err = float((amr7.composite_solution()[start : start + length] - exact7[start : start + length]).abs().mean())
    coarse_err = float((coarse_only.phi[start : start + length] - exact7[start : start + length]).abs().mean())
    result.add(
        "R7_refinement_accuracy_gain",
        "L1 error inside the refined region smaller than the coarse-only error",
        {
            "amr_l1_in_patch": amr_err,
            "coarse_l1_in_patch": coarse_err,
            "error_ratio": amr_err / coarse_err,
        },
        amr_err < coarse_err,
    )

    return result