"""Per-margin-block "pointer-shift" streaming -- port of the TCLB formulation.

The reference implementation is CFD-GO/TCLB (`src/conf.R`, `src/LatticeAccess.inc.cpp.Rt`,
`src/Lattice.cu.Rt`).  TCLB does *not* reorder populations with an index table.
Instead the population field of every lattice direction is stored as a set of
**margin blocks** (``block1`` .. ``block27``, one per combination of
low/interior/high span along each axis) and a neighbour read is a fixed
*linear offset inside a block*::

    con = load.field("val", f, p, dp, con)          # LatticeAccess.inc.cpp.Rt:316-336
    ... = MContextPocket.Margin[[m]]Name[Offsets[m] - MContextOffset[[m]]]

``move.pocket()`` (``LatticeAccess.inc.cpp.Rt:69``) advances the *base pointer*
of the block (``pocket.blockN += delta``) whenever the loop crosses into another
block, so the address arithmetic collapses to a constant offset from a moved
pointer -- hence "pointer-shift streaming".  The block offsets themselves come
from ``offsets()`` in ``src/conf.R:897`` (``offset.p`` / ``get_offsets`` /
``put_offsets``), and the extent of each field is derived from the stencil
(``Fields$minx = ... min(dx, ...)``, ``src/conf.R:143``).

This module ports the *mechanism* -- not TCLB's generated C++ -- onto
TensorLBM's ``(q, ...)`` torch layout:

* :func:`margin_bands` -- the 3-way per-axis split that makes up the blocks;
* :func:`pointer_shift_plan` -- the block copy list, each entry a pure
  ``(dst_slice, src_slice, flat_offset)`` with a **constant** flat offset
  (the property that makes pointer shifting legal);
* :func:`stream_pointer_shift` -- the transport operator built from the plan
  (views only: no index tensors, no ``roll``, no gather);
* :func:`check_plan_invariant` -- exhaustive verifier of that constancy
  property against a cell-by-cell reference.

Semantics are identical to :func:`tensorlbm.solver.stream` (2D) and
:func:`tensorlbm.solver3d.stream3d_gather` (3D), i.e. pull/gather::

    out[q, y, x]    = f[q, y - c_y, x - c_x]                  # 2D
    out[q, z, y, x] = f[q, z - c_z, y - c_y, x - c_x]         # 3D

with *periodic* wrap on every axis (TCLB's own periodic case) and an optional
solid mask for the bounce-back arm handled outside the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "PointerShiftPlan",
    "ShiftCopy",
    "check_plan_invariant",
    "flat_offset",
    "margin_bands",
    "plan_to_dict",
    "pointer_shift_plan",
    "stream_pointer_shift",
    "stream_pointer_shift_flat",
]


# --------------------------------------------------------------------- blocks
def margin_bands(n: int) -> tuple[slice, slice, slice]:
    """Split ``range(n)`` into the three margin spans TCLB keeps per axis.

    ``lo`` (1 cell), ``mid`` (``n - 2`` cells) and ``hi`` (1 cell) mirror the
    ``tab3`` selections of ``offsets()`` (``src/conf.R:917-923``): a block is
    identified by which of the three spans each axis falls into.
    """
    if n < 3:
        raise ValueError(f"margin decomposition needs >= 3 cells per axis, got {n}")
    return slice(0, 1), slice(1, n - 1), slice(n - 1, n)


def _axis_segments(n: int, c: int) -> list[tuple[slice, slice]]:
    """``(dst, src)`` slice pairs along one axis for shift ``c`` (periodic).

    Exactly one pair per "margin choice": the wrapped strip (if ``c != 0``) and
    the in-range remainder.  Every pair has equal length, which is what makes
    the flat-index difference constant inside the pair.
    """
    if c == 0:
        return [(slice(0, n), slice(0, n))]
    if c > 0:
        return [(slice(1, n), slice(0, n - 1)), (slice(0, 1), slice(n - 1, n))]
    return [(slice(0, n - 1), slice(1, n)), (slice(n - 1, n), slice(0, 1))]


def flat_offset(shape: Sequence[int], dst: Sequence[slice], src: Sequence[slice]) -> int:
    """Row-major flat-index difference ``flat(src) - flat(dst)`` of two boxes.

    Raises if the boxes have different extents (then no single constant offset
    exists) -- the invariant TCLB's block layout has to satisfy.
    """
    if len(shape) != len(dst) or len(shape) != len(src):
        raise ValueError("shape / slice tuples must have the same rank")
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    off = 0
    for st, d, s in zip(strides, dst, src):
        if d.stop - d.start != s.stop - s.start:
            raise ValueError(
                f"box extents differ along an axis: dst {d.stop - d.start} vs src {s.stop - s.start}"
            )
        off += (s.start - d.start) * st
    return int(off)


@dataclass(frozen=True)
class ShiftCopy:
    """One block copy of the streaming operator."""

    q: int
    dst: tuple[slice, ...]
    src: tuple[slice, ...]
    flat_offset: int
    wrapped_axes: tuple[int, ...] = field(default_factory=tuple)

    @property
    def extents(self) -> tuple[int, ...]:
        return tuple(d.stop - d.start for d in self.dst)

    @property
    def n_cells(self) -> int:
        n = 1
        for e in self.extents:
            n *= e
        return n


@dataclass(frozen=True)
class PointerShiftPlan:
    """Full transport plan: one :class:`ShiftCopy` list per lattice direction."""

    shape: tuple[int, ...]  # field shape without the leading q axis
    lveloc: tuple[tuple[int, ...], ...]
    copies: tuple[ShiftCopy, ...]
    per_axis_blocks: tuple[int, ...] = ()

    def by_direction(self) -> dict[int, list[ShiftCopy]]:
        out: dict[int, list[ShiftCopy]] = {}
        for cp in self.copies:
            out.setdefault(cp.q, []).append(cp)
        return out

    def n_blocks(self, q: int) -> int:
        return sum(1 for cp in self.copies if cp.q == q)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "q": len(self.lveloc),
            "dims": len(self.shape),
            "copies": len(self.copies),
            "blocks_per_direction": {int(q): self.n_blocks(q) for q in range(len(self.lveloc))},
            "cells_covered": {
                int(q): int(sum(cp.n_cells for cp in self.copies if cp.q == q))
                for q in range(len(self.lveloc))
            },
        }


def _lveloc_array(lveloc: Any) -> np.ndarray:
    lv = lveloc.detach().cpu().numpy() if isinstance(lveloc, torch.Tensor) else np.asarray(lveloc)
    lv = lv.astype(np.int64, copy=False)
    if lv.ndim != 2 or lv.shape[1] not in (2, 3):
        raise ValueError(f"velocity table must be (q, 2) or (q, 3), got {lv.shape}")
    return lv


def pointer_shift_plan(lveloc: Any, shape: Sequence[int]) -> PointerShiftPlan:
    """Build the per-margin-block transport plan for ``lveloc`` on ``shape``.

    ``shape`` is the *spatial* shape in field axis order (``(ny, nx)`` for 2D,
    ``(nz, ny, nx)`` for 3D); velocity components are read in the same order,
    matching :func:`tensorlbm.solver.stream` / :func:`tensorlbm.solver3d.stream3d_gather`.
    """
    lv = _lveloc_array(lveloc)
    nd = len(shape)
    for n in shape:
        if int(n) < 3:
            raise ValueError(
                f"per-margin decomposition needs >= 3 cells per axis, got {tuple(shape)}"
            )
    if lv.shape[1] < nd:
        raise ValueError(f"velocity table has {lv.shape[1]} components, field has {nd} axes")
    c_tab = [(int(c[0]), int(c[1]), int(c[2]) if lv.shape[1] > 2 else 0) for c in lv]
    # field axis order is reversed w.r.t. (x, y, z): last axis is x in both layouts
    copies: list[ShiftCopy] = []
    for q, c_xyz in enumerate(c_tab):
        c_axes = tuple(reversed(c_xyz[:nd]))  # (x,y,z)[:nd] reversed -> (..., y, x)
        if len(c_axes) < nd:  # 2D table with 3D shape or vice versa
            c_axes = (0,) * (nd - len(c_axes)) + c_axes
        seg = [_axis_segments(n, int(c)) for n, c in zip(shape, c_axes)]
        # cartesian product over axes (row-major order matches the flat layout)
        for combo in _product(seg):
            dst = tuple(p[0] for p in combo)
            src = tuple(p[1] for p in combo)
            wrapped = tuple(i for i, c in enumerate(c_axes) if int(c) != 0)
            copies.append(
                ShiftCopy(q=q, dst=dst, src=src, flat_offset=flat_offset(shape, dst, src),
                          wrapped_axes=wrapped)
            )
    return PointerShiftPlan(
        shape=tuple(int(n) for n in shape),
        lveloc=tuple(tuple(int(v) for v in row) for row in lv.tolist()),
        copies=tuple(copies),
        per_axis_blocks=tuple(3 for _ in shape),
    )


def _product(parts: Iterable[Sequence[Any]]) -> Iterable[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = [()]
    for part in parts:
        out = [prev + (item,) for prev in out for item in part]
    return out


# ------------------------------------------------------------------ transport
def stream_pointer_shift(
    f: torch.Tensor,
    lveloc: Any,
    *,
    plan: PointerShiftPlan | None = None,
) -> torch.Tensor:
    """Periodic streaming by per-margin block copies (pointer-shift port).

    ``f`` has shape ``(q, *spatial)``.  Every destination cell is written
    exactly once (the blocks partition the domain), using slice views with the
    precomputed constant offsets -- no index tensors, no ``torch.roll``.
    """
    if f.dim() < 2:
        raise ValueError(f"expected (q, *spatial) field, got {tuple(f.shape)}")
    shape = tuple(int(n) for n in f.shape[1:])
    if plan is None:
        plan = pointer_shift_plan(lveloc, shape)
    elif plan.shape != shape:
        raise ValueError(f"plan shape {plan.shape} does not match field {shape}")
    if f.shape[0] != len(plan.lveloc):
        raise ValueError(f"field has {f.shape[0]} populations, plan has {len(plan.lveloc)}")
    out = torch.empty_like(f)
    for cp in plan.copies:
        out[(cp.q,) + cp.dst] = f[(cp.q,) + cp.src]
    return out


def stream_pointer_shift_flat(
    f: torch.Tensor,
    lveloc: Any,
    *,
    plan: PointerShiftPlan | None = None,
) -> torch.Tensor:
    """Same operator, executed as flat (1-D) offset copies.

    This is the closest torch analogue of what TCLB's generated C++ does: each
    block copy becomes ``out_flat[k : k+n] = f_flat[k + off : k + off + n]``,
    i.e. a *single* constant pointer offset per block.  It exists to show that
    the plan's ``flat_offset`` is sufficient to address every block (and to
    price the address arithmetic), not as a faster path.
    """
    if f.dim() < 2:
        raise ValueError(f"expected (q, *spatial) field, got {tuple(f.shape)}")
    shape = tuple(int(n) for n in f.shape[1:])
    if plan is None:
        plan = pointer_shift_plan(lveloc, shape)
    q = f.shape[0]
    out = torch.empty_like(f)
    oflat = out.reshape(q, -1)
    fflat = f.reshape(q, -1)
    strides = _strides(shape)
    for cp in plan.copies:
        start = sum(d.start * st for d, st in zip(cp.dst, strides))
        n = cp.n_cells
        # only meaningful when the box is contiguous in flat memory
        contiguous = _is_contiguous_box(shape, cp.dst)
        if contiguous:
            oflat[cp.q, start:start + n] = fflat[cp.q, start + cp.flat_offset: start + cp.flat_offset + n]
        else:
            out[(cp.q,) + cp.dst] = f[(cp.q,) + cp.src]
    return out


def _strides(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _is_contiguous_box(shape: Sequence[int], box: Sequence[slice]) -> bool:
    """True if the box is one contiguous run in row-major flat memory."""
    return all(b.start == 0 and b.stop == n for n, b in zip(shape[1:], box[1:])) and True


# ------------------------------------------------------------- verification
def check_plan_invariant(
    f: torch.Tensor,
    lveloc: Any,
    *,
    plan: PointerShiftPlan | None = None,
) -> dict[str, Any]:
    """Exhaustively verify the pointer-shift invariant on ``f``.

    Three claims are checked, all cell-by-cell:

    1. **partition** -- every destination cell of every direction is written by
       exactly one block (no overlap, no hole);
    2. **constancy** -- inside a block, ``flat_index(src) - flat_index(dst)``
       equals the block's recorded ``flat_offset`` for *every* cell (this is
       the property that licenses replacing per-cell index arithmetic with a
       moved base pointer);
    3. **correctness** -- the resulting field equals a direct gather reference.
    """
    shape = tuple(int(n) for n in f.shape[1:])
    plan = plan or pointer_shift_plan(lveloc, shape)
    q = f.shape[0]
    strides = _strides(shape)

    # 1. partition
    cover = torch.zeros((q,) + shape, dtype=torch.int32)
    for cp in plan.copies:
        cover[(cp.q,) + cp.dst] += 1
    partition_ok = bool((cover == 1).all())
    holes = int((cover == 0).sum())
    overlaps = int((cover > 1).sum())

    # 2. constancy, cell by cell (vectorised over the block)
    max_offset_error = 0
    for cp in plan.copies:
        dst_idx = _flat_indices(shape, strides, cp.dst)
        src_idx = _flat_indices(shape, strides, cp.src)
        diff = (src_idx - dst_idx - cp.flat_offset).abs()
        if diff.numel():
            max_offset_error = max(max_offset_error, int(diff.max()))

    # 3. correctness vs an index-table gather reference
    ref = _gather_reference(f, plan.lveloc)
    got = stream_pointer_shift(f, lveloc, plan=plan)
    max_abs_diff = float((got - ref).abs().max()) if f.numel() else 0.0

    return {
        "partition_ok": partition_ok,
        "holes": holes,
        "overlaps": overlaps,
        "max_offset_error": max_offset_error,
        "max_abs_diff_vs_gather": max_abs_diff,
        "copies": len(plan.copies),
        "cells_covered": int(cover.sum()),
        "cells_expected": int(q * int(np.prod(shape))),
    }


def _flat_indices(shape: Sequence[int], strides: Sequence[int], box: Sequence[slice]) -> torch.Tensor:
    """Flat indices of every cell of ``box`` in row-major order."""
    grids = torch.meshgrid(
        *[torch.arange(b.start, b.stop) * st for b, st in zip(box, strides)], indexing="ij"
    )
    out = grids[0]
    for g in grids[1:]:
        out = out + g
    return out.reshape(-1)


def _gather_reference(f: torch.Tensor, lveloc: Sequence[Sequence[int]]) -> torch.Tensor:
    """Independent reference: per-direction shifted slices built from ``torch.roll``."""
    nd = f.dim() - 1
    rows = []
    for c_xyz in lveloc:
        c = list(c_xyz)[:nd][::-1]  # axis order (..., y, x)
        d = f[len(rows)]
        for axis, ca in enumerate(c):
            if int(ca):
                d = torch.roll(d, shifts=int(ca), dims=axis)
        rows.append(d)
    return torch.stack(rows, dim=0)


def plan_to_dict(plan: PointerShiftPlan) -> dict[str, Any]:
    """Serialisable form of a plan (used by the XVAL driver artefacts)."""
    return {
        **plan.summary(),
        "copies": [
            {
                "q": cp.q,
                "dst": [[d.start, d.stop] for d in cp.dst],
                "src": [[s.start, s.stop] for s in cp.src],
                "flat_offset": cp.flat_offset,
                "n_cells": cp.n_cells,
            }
            for cp in plan.copies
        ],
    }