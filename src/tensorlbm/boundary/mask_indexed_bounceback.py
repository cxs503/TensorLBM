"""Mask-indexed bounce-back (OSS cross-validation ``XVAL-LBM-MASK-INDEXED-BOUNCEBACK``).

Reference implementation (zero hand-copy, read from the checkout)
-----------------------------------------------------------------
``haimingz/vivsim`` @ ``fb84b12f02e18f3b9dfdf738d6a1a318d81539f9`` (JAX):

* ``vivsim/lbm/boundary/bb.py:98-110`` — the whole 2-D rule::

      def obstacle_bounce_back(f, mask):
          return f.at[:, mask].set(f[:, mask][OPP_DIRS])

  i.e. **post-streaming**, on the boolean obstacle mask, *all* populations of an
  obstacle cell are replaced by the opposite-direction populations of the same
  cell as they are after streaming (``at[].set`` evaluates the right-hand side
  on the pre-assignment array, so it is one simultaneous permuting write);
* ``vivsim/lbm3d/boundary/bb.py:56-59`` — the D3Q19 twin, same one-liner;
* ``vivsim/lbm/lattice.py`` / ``bb.py:12`` — ``OPP_DIRS`` is the lattice
  opposite-direction table (``lbm/lattice.py`` ``D2Q9.opp_dirs``);
* ``vivsim/lbm/basic.py:46-79`` — the streaming step is a periodic pull shift
  (``shift_x_pos`` = ``concatenate`` of a one-cell shifted slice), so at step
  ``n+1`` the post-stream content of an obstacle cell ``S`` is what the
  neighbour ``S - c_i`` held after collision at step ``n``;
* the call is made after streaming and before the next collision; no wall
  position, no interpolation distance and no wall velocity is supplied — the
  scheme carries no sub-grid information (moving walls are a *separate*,
  non-mask routine, ``bb.py:16-80``);
* ``vivsim`` stores the *shifted* populations ``f_i = f_i^full - w_i``
  (``lbm/basic.py:20``), which is harmless here because ``w_i = w_opp(i)`` for
  D2Q9/D3Q19, so mirroring commutes with the shift.

What this module adds
---------------------
The OSS form is a boolean-mask gather + scatter.  In ``torch`` that form costs
``O(Q*N_total)`` (the mask gather materialises an index list over the whole
domain).  Here the same permutation is expressed through a **precomputed flat
index pair** over the obstacle cells only (``dst``/``src`` into the flattened
``(Q, *spatial)`` array), so the work per application is ``O(Q*N_solid)`` — the
index pair is geometry-only and is built once per geometry change, exactly like
the OSS mask is built once.  Two explicit reference arms are kept in the same
module so that the equivalence is testable rather than asserted:

``boolean_mask_bounce_back``
    Literal transplant of the OSS one-liner (boolean-mask select + scatter).
``mask_indexed_bounce_back`` / ``mask_indexed_bounce_back_``
    Index-pair form (out-of-place / in-place on a caller-owned buffer).

Semantics of the scheme (proved by ``tests/test_mask_indexed_bounceback.py``)
----------------------------------------------------------------------------
Applying the swap to the *post-stream* populations of obstacle cells delivers,
at the next step and on the adjacent fluid node ``F``, ``f_i(F) = f_opp(i)^post-coll(F)``
— i.e. it is **exactly the half-way bounce-back** with the no-slip plane on the
face between ``F`` and the obstacle cell ``S`` (half a lattice unit from each
centre).  The reflection is *stored* on the solid node but the wall is halfway:
the "solid-node" phrasing describes where the write happens, not where the wall
is.  ``tests/test_mask_indexed_bounceback.py::test_equivalence_with_fluid_side_halfway``
verifies the identity field-by-field against an independently written
fluid-side half-way bounce-back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "SolidMaskIndex",
    "build_solid_mask_index",
    "boolean_mask_bounce_back",
    "mask_indexed_bounce_back",
    "mask_indexed_bounce_back_",
]


@dataclass(frozen=True)
class SolidMaskIndex:
    """Precomputed flat index pair for the mask-indexed bounce-back.

    ``dst`` and ``src`` are 1-D ``int64`` tensors of length ``Q * count``
    addressing the flattened ``(Q, *spatial)`` population array: for every
    obstacle cell ``j`` and every direction ``i`` the entry copies
    ``flat[src] -> flat[dst]`` with ``dst = i * N + j`` and
    ``src = opp[i] * N + j``.
    """

    shape: tuple[int, ...]
    count: int
    n_total: int
    dst: torch.Tensor
    src: torch.Tensor

    @property
    def solid_fraction(self) -> float:
        return self.count / float(self.n_total) if self.n_total else 0.0

    @property
    def index_nbytes(self) -> int:
        return int(self.dst.numel() * self.dst.element_size() + self.src.numel() * self.src.element_size())


def build_solid_mask_index(
    mask: torch.Tensor,
    opp: torch.Tensor | list[int] | tuple[int, ...],
) -> SolidMaskIndex:
    """Build the flat index pair for a boolean solid mask.

    Args:
        mask: boolean array of shape ``spatial`` (``True`` = obstacle/solid).
        opp: opposite-direction table of length ``Q``.

    Returns:
        :class:`SolidMaskIndex` addressing ``(Q, *spatial)`` flattened arrays.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a boolean tensor, got {mask.dtype}")
    spatial = tuple(int(s) for s in mask.shape)
    if not spatial:
        raise ValueError("mask must have at least one spatial dimension")
    opp_t = torch.as_tensor(opp, dtype=torch.int64, device=mask.device)
    q = int(opp_t.numel())
    n_total = int(mask.numel())
    solid = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1).to(torch.int64)
    count = int(solid.numel())
    dirs = torch.arange(q, dtype=torch.int64, device=mask.device).view(q, 1)
    dst = (dirs * n_total + solid.view(1, count)).reshape(-1)
    src = (opp_t.view(q, 1) * n_total + solid.view(1, count)).reshape(-1)
    return SolidMaskIndex(shape=spatial, count=count, n_total=n_total, dst=dst, src=src)


def _check_layout(f: torch.Tensor, spatial: tuple[int, ...]) -> None:
    if tuple(int(s) for s in f.shape[1:]) != spatial:
        raise ValueError(f"field spatial shape {tuple(f.shape[1:])} != mask shape {spatial}")


def boolean_mask_bounce_back(
    f: torch.Tensor,
    mask: torch.Tensor,
    opp: torch.Tensor | list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Literal transplant of the OSS rule (``vivsim/lbm/boundary/bb.py:98-110``).

    ``f.at[:, mask].set(f[:, mask][OPP_DIRS])`` → clone + simultaneous
    permuting write on the boolean mask.  Reference arm: O(Q*N_total).
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a boolean tensor, got {mask.dtype}")
    spatial = tuple(int(s) for s in mask.shape)
    _check_layout(f, spatial)
    opp_t = torch.as_tensor(opp, dtype=torch.int64, device=f.device)
    if int(opp_t.numel()) != int(f.shape[0]):
        raise ValueError(f"opp table length {int(opp_t.numel())} != Q {int(f.shape[0])}")
    mirrored = f[opp_t][:, mask]  # gather only on the mask (index list materialised)
    out = f.clone()
    out[:, mask] = mirrored
    return out


def mask_indexed_bounce_back(f: torch.Tensor, index: SolidMaskIndex) -> torch.Tensor:
    """Out-of-place mask-indexed bounce-back (``O(Q*N_total)`` copy + ``O(Q*N_solid)`` work)."""
    out = f.clone()
    return mask_indexed_bounce_back_(out, index)


def mask_indexed_bounce_back_(f: torch.Tensor, index: SolidMaskIndex) -> torch.Tensor:
    """In-place mask-indexed bounce-back: work is exactly ``O(Q*N_solid)``.

    The caller owns ``f`` (must not be a view aliased elsewhere); returns ``f``.
    """
    _check_layout(f, index.shape)
    if index.count == 0:
        return f
    flat = f.reshape(-1)
    flat.index_copy_(0, index.dst, flat.take(index.src))
    return f