#!/usr/bin/env python3
"""Boxed (non-periodic) streaming for closed-cavity 2D domains.

Problem this solves
-------------------
:func:`tensorlbm.solver.stream` is **fully periodic** in both directions: it is a
single advanced-index gather with ``% ny`` / ``% nx``.  If a solid wall occupies
the first or the last row, that wall lies *on the periodic seam*, and the seam
becomes a **conduit between the two walls**: the gather fills the seam-adjacent
solid row with populations taken from the *opposite* side of the domain, the
bounce-back step then reflects them, and the wall-adjacent fluid row receives
them as if the local wall had emitted them.

In a sessile-drop / contact-angle validation this is not cosmetic.  The wetting
phase near the bottom wall is re-injected at the top of the box, which produces
(i) mirror copies of the contact-line structure at the far wall (at the drop's
own abscissa) and (ii) a slow leak of phase out of the drop.  Both were reported
as an unexplained "far-wall blob" in
``docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE-R2-CONVENTION.md``.

What this module does
---------------------
:func:`stream_boxed` is a complete streaming step with half-way bounce-back at
every fluid link whose upstream cell is solid **or outside the grid**:

    f_k(x, t+1) = f*_opp(k)(x, t)     if the upstream of k at x is not fluid
    f_k(x, t+1) = f*_k(x - c_k, t)    otherwise

Solid cells are set to zero (they are never advected).  Nothing crosses the
periodic seam, so the two walls are hydraulically independent and the fluid mass
is conserved exactly.  The scheme is the same one implemented in
:func:`tensorlbm.amr_composite2d.stream_walls` (numpy, previously merged); the
torch version here is verified against it in
``tests/test_stream_boxed.py``.

Note the semantics change: this operator uses the *current* post-collision
populations for the reflected links, whereas the stock "reflect inside the solid
cell after streaming" convention of :mod:`tensorlbm.solver` is lagged by one
step.  Both are standard; do not mix them inside one run.  :func:`stream_boxed`
leaves the field bit-identical to ``stream`` + reflection when no boundary line
is solid (see ``tests/test_stream_boxed.py::test_compat_no_seam_wall``).
"""
from __future__ import annotations

import torch

from .solver import C, OPPOSITE_2D, stream

__all__ = ["stream_boxed", "seam_leak_fraction"]

_OPP = tuple(int(v) for v in OPPOSITE_2D)  # canonical D2Q9 map, solver.OPPOSITE_2D


def _shift_mask(mask: torch.Tensor, cx: int, cy: int) -> torch.Tensor:
    """``out[y, x] = mask[y - cy, x - cx]``, ``False`` outside the grid."""
    out = torch.zeros_like(mask)
    ny, nx = mask.shape
    ys_dst = slice(max(0, cy), ny + min(0, cy))
    ys_src = slice(max(0, -cy), ny + min(0, -cy))
    xs_dst = slice(max(0, cx), nx + min(0, cx))
    xs_src = slice(max(0, -cx), nx + min(0, -cx))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def stream_boxed(f: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    """Streaming with half-way bounce-back and **no** periodic seam coupling.

    Args:
        f: post-collision populations, shape ``(9, ny, nx)``.
        solid: boolean solid mask, shape ``(ny, nx)``.

    Returns:
        The streamed populations.  The caller must **not** apply the stock
        solid-cell reflection again.
    """
    if f.dim() != 3 or f.shape[1:] != solid.shape:
        raise ValueError(f"f shape {tuple(f.shape)} incompatible with solid {tuple(solid.shape)}")
    fluid = ~solid
    out = stream(f)
    for k in range(9):
        cx, cy = int(C[k, 0]), int(C[k, 1])
        if cx == 0 and cy == 0:
            continue
        upstream_fluid = _shift_mask(fluid, cx, cy)
        bad = fluid & ~upstream_fluid
        if bool(bad.any()):
            out[k] = torch.where(bad, f[_OPP[k]], out[k])
    return torch.where(solid.unsqueeze(0), torch.zeros_like(out), out)


def seam_leak_fraction(f: torch.Tensor, solid: torch.Tensor) -> float:
    """Share of the wall-adjacent fluid inflow that comes from the far side.

    Diagnostic for the periodic seam: with :func:`tensorlbm.solver.stream` (plus
    the stock reflection) a wall on the seam feeds the wall-adjacent fluid row
    with populations gathered from the *opposite* boundary.  The return value is
    the L1 mass fraction of those far-side contributions; it is ``0.0`` for
    :func:`stream_boxed` by construction and grows with the phase contrast at the
    wall.
    """
    ny = solid.shape[0]
    reflected = stream(f)[list(_OPP)]
    boxed = stream_boxed(f, solid)
    fluid = ~solid
    rows = []
    if bool(solid[0].any()) and ny > 2:
        rows.append(1)
    if bool(solid[ny - 1].any()) and ny > 2:
        rows.append(ny - 2)
    if not rows:
        return 0.0
    sel = [slice(None), rows, slice(None)]
    leak = (reflected[sel] - boxed[sel]).abs().sum()
    total = reflected[sel].abs().sum().clamp(min=1e-300)
    return float(leak / total)