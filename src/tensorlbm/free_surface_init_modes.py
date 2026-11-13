"""Birth-state (re)initialisation modes for free-surface LBM cells.

Anchor: ``XVAL-LBM-AVERAGE-SURROUND-INIT``.  OSS counterpart:
``Maarten-vd-Sande/lbm`` @ ``573a01c``, ``lbm/update_types.py::average_surround``
(``mass=0; rho=0; u=0;`` then ``rho,u`` averaged over the FLUID|INTERFACE
8-neighbours of the *previous* step and ``set_equi`` called on the average).

Two modes are provided for a cell that is *born* (GAS -> INTERFACE) or
*reborn* (envelope shell) inside ``koerner_step_2d`` / ``free_surface_step``:

``fixed_rho``
    Historical TensorLBM behaviour: equilibrium at a prescribed reference
    density (``rho_liquid``) with the neighbour-averaged velocity.
``average_surround``
    OSS semantics: equilibrium at the neighbour-averaged density *and*
    neighbour-averaged velocity.

Transplant deltas (deliberate, see ``docs/XVAL_AVERAGE_SURROUND_INIT.md``):

1. **Isolated cell fallback.**  OSS leaves ``rho = 0`` when ``c == 0``, so
   ``set_equi`` writes an all-zero population (``fill = mass / rho`` then
   divides by zero).  Here a cell with no active neighbour falls back to the
   caller-supplied reference density (u = 0), matching OSS only for ``c > 0``.
2. **Mass ledger.**  OSS sets ``mass = 0`` on the new cell and the cell's fill
   is only rebuilt by later redistribution.  TensorLBM keeps the credited
   conversion mass and renormalises the fill with the refilled density, so
   ``rho`` and ``mass`` stay consistent.
3. **Temporal staggering.**  OSS averages ``rho_prev/u_prev`` (pre-stream
   state); the torch path averages whatever macroscopic fields the caller
   hands in (post-collision state of the same step).
4. **Stencil.**  D2Q9 8-neighbour average (OSS) is generalised to the 18
   moving D3Q19 links; the average is over *active* (LIQUID|INTERFACE) cells
   only, matching the OSS ``cell_type & (FLUID | INTERFACE)`` filter.
5. **Vectorisation.**  OSS is a per-cell numba loop; here it is a stack of
   ``torch.roll`` gathers with no host synchronisation.

The same module also carries a NumPy ``float64`` *reference oracle*
(:func:`average_surround_reference`) that is written independently of the
torch kernels: it is the cross-validation ``ref`` side of the XVAL benchmark
and the invariant oracle of the unit tests.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

INIT_MODE_FIXED_RHO = "fixed_rho"
INIT_MODE_AVERAGE_SURROUND = "average_surround"
#: ``legacy`` keeps each solver's pre-existing per-call-site behaviour.
INIT_MODE_LEGACY = "legacy"

#: Selectable modes for an individual birth call.
INIT_MODES: tuple[str, ...] = (INIT_MODE_FIXED_RHO, INIT_MODE_AVERAGE_SURROUND)
#: Selectable modes for a whole solver step (adds the historical asymmetry).
SOLVER_INIT_MODES: tuple[str, ...] = INIT_MODES + (INIT_MODE_LEGACY,)


def normalize_init_mode(mode: object, *, allow_legacy: bool = False) -> str:
    """Validate ``mode`` and return its canonical spelling.

    Raises ``ValueError`` for anything outside the registry so a typo can never
    silently fall back to a different birth state.
    """
    allowed = SOLVER_INIT_MODES if allow_legacy else INIT_MODES
    if not isinstance(mode, str):
        raise ValueError(f"init_mode must be a str, got {type(mode).__name__}")
    canon = mode.strip().lower()
    if canon not in allowed:
        raise ValueError(f"init_mode must be one of {allowed}, got {mode!r}")
    return canon


def resolve_birth_mode(solver_mode: str, *, site: str) -> str:
    """Map a solver-level mode to the mode used at one birth call site.

    ``legacy`` preserves the historical per-site default (the 2D solver emitted
    births at two sites with *different* densities: envelope shells at the
    neighbour average, conversion births at ``rho_liquid``).
    """
    mode = normalize_init_mode(solver_mode, allow_legacy=True)
    if mode == INIT_MODE_LEGACY:
        if site not in LEGACY_SITE_DEFAULTS:
            raise ValueError(
                f"unknown birth site {site!r}; known: {tuple(LEGACY_SITE_DEFAULTS)}"
            )
        return normalize_init_mode(LEGACY_SITE_DEFAULTS[site])
    return mode


# ---------------------------------------------------------------------------
# Stencils (unit offsets; the average is sign-symmetric so orientation is
# irrelevant, but the list must be the complete moving set)
# ---------------------------------------------------------------------------

#: ``(dy, dx)`` pull-source offsets in the :mod:`tensorlbm.d2q9` link order
#: (q = 1..8), so a stack over this tuple reproduces the historical kernel
#: summation order exactly.
D2Q9_SHIFTS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

D3Q19_SHIFTS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 1),
    (0, 1, -1),
    (0, -1, 1),
    (0, -1, -1),
    (1, 0, 1),
    (1, 0, -1),
    (-1, 0, 1),
    (-1, 0, -1),
    (1, 1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (-1, -1, 0),
)

#: Default birth site semantics of ``koerner_step_2d`` (see ``resolve_birth_mode``).
LEGACY_SITE_DEFAULTS: dict[str, str] = {
    "envelope": INIT_MODE_AVERAGE_SURROUND,
    "conversion": INIT_MODE_FIXED_RHO,
}


# ---------------------------------------------------------------------------
# Torch kernel: neighbour average over active cells
# ---------------------------------------------------------------------------


def _shift_to_neighbour(field: torch.Tensor, offset: Sequence[int]) -> torch.Tensor:
    """Value of ``field`` at ``x - offset`` broadcast back onto ``x``.

    Pull-source convention (``offset`` is the lattice link *leading away* from
    the cell), identical to ``free_surface_lbm_2d._roll_from_pull``: the 8-link
    set is closed under negation, so the average over the set is independent of
    the sign convention, but keeping the pull order makes the summation order —
    and therefore the float32 result — bit-identical to the historical kernel.
    """
    ndim = len(offset)
    dims = tuple(range(field.dim() - ndim, field.dim()))
    return torch.roll(field, shifts=tuple(int(o) for o in offset), dims=dims)


def neighbour_average(
    rho: torch.Tensor,
    velocities: Sequence[torch.Tensor],
    active: torch.Tensor,
    shifts: Sequence[Sequence[int]] = D2Q9_SHIFTS,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Arithmetic mean of ``rho``/``velocities`` over *active* neighbours.

    Mirrors ``average_surround``'s loop body
    (``if cell_type[ix_next, iy_next] & (CT.FLUID | CT.INTERFACE)``): only
    active neighbours enter both the numerator and the divisor.

    Returns ``(rho_avg, (u_avg...), count)``.  ``rho_avg``/``u_avg`` are
    *defined* everywhere (divisor clamped to 1) and are meaningful only where
    ``count > 0``; callers decide the ``count == 0`` fallback.
    """
    mask = active.to(rho.dtype)
    stacked = torch.stack([_shift_to_neighbour(mask, off) for off in shifts])
    count = stacked.sum(dim=0)

    def _mean(field: torch.Tensor) -> torch.Tensor:
        acc = torch.stack(
            [_shift_to_neighbour(field, off) * stacked[q] for q, off in enumerate(shifts)]
        )
        return acc.sum(dim=0) / count.clamp(min=1.0)

    mean_rho = _mean(rho)
    mean_vels = tuple(_mean(v) for v in velocities)
    return mean_rho, mean_vels, count


# ---------------------------------------------------------------------------
# Torch kernel: birth population
# ---------------------------------------------------------------------------


def birth_population(
    f: torch.Tensor,
    mask: torch.Tensor,
    rho_ref: torch.Tensor | float,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None,
    *,
    mode: str,
    equilibrium_fn: Callable[..., torch.Tensor],
    flags: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    liquid_flag: int = 1,
    interface_flag: int = 2,
    shifts: Sequence[Sequence[int]] = D2Q9_SHIFTS,
) -> torch.Tensor:
    """Refill ``mask`` cells with the birth equilibrium for ``mode``.

    ``flags``/``active`` default to the active set implied by ``flags``; pass
    ``active`` explicitly when the caller already has it.
    """
    mode = normalize_init_mode(mode)
    if uz is None:
        shifts = D2Q9_SHIFTS if len(shifts) == 8 else tuple(shifts)
    if active is None:
        if flags is None:
            raise ValueError("birth_population needs either flags or active")
        active = (flags == liquid_flag) | (flags == interface_flag)

    rho_macro = f.sum(0)
    vels = (ux, uy) if uz is None else (ux, uy, uz)
    mean_rho, mean_vels, count = neighbour_average(rho_macro, vels, active, shifts)

    if isinstance(rho_ref, torch.Tensor):
        ref = rho_ref.to(f.dtype)
    else:
        ref = torch.full_like(ux, float(rho_ref))

    if mode == INIT_MODE_AVERAGE_SURROUND:
        # OSS: average rho over active neighbours; fall back to the caller's
        # reference density where the cell is isolated (delta 1 in the header).
        rho_field = torch.where(count > 0, mean_rho, ref)
    else:  # INIT_MODE_FIXED_RHO
        rho_field = ref

    # Both modes average the velocity (OSS does; so did the historical path).
    mean_ux, mean_uy = mean_vels[0], mean_vels[1]
    if uz is None:
        feq = equilibrium_fn(rho_field, mean_ux, mean_uy)
    else:
        feq = equilibrium_fn(rho_field, mean_ux, mean_uy, mean_vels[2])
    return torch.where(mask.unsqueeze(0), feq, f)


# ---------------------------------------------------------------------------
# NumPy float64 reference oracle (independent of the torch path)
# ---------------------------------------------------------------------------


def average_surround_reference(
    rho: np.ndarray,
    velocities: Sequence[np.ndarray],
    active: np.ndarray,
    *,
    fallback_rho: float = 1.0,
    shifts: Sequence[Sequence[int]] = D2Q9_SHIFTS,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """``float64`` reference implementation of the OSS ``average_surround`` body.

    Written with explicit index arithmetic (no ``roll``), so it cannot share a
    bug with :func:`neighbour_average`.  Non-periodic borders are treated as
    *inactive* (counted as 0), matching OSS's ``range(1, 9)`` neighbour probe
    which never sees outside the array.
    """
    rho = np.asarray(rho, dtype=np.float64)
    vel = [np.asarray(v, dtype=np.float64) for v in velocities]
    act = np.asarray(active).astype(bool)
    out_rho = np.zeros_like(rho)
    out_vel = [np.zeros_like(v) for v in vel]
    count = np.zeros(rho.shape, dtype=np.float64)

    ndim = rho.ndim
    for off in shifts:
        if len(off) != ndim:
            raise ValueError(f"offset {off} has wrong rank for field of ndim {ndim}")
        src_slices = []
        dst_slices = []
        valid = True
        for o in off:
            if o > 0:
                src_slices.append(slice(o, None))
                dst_slices.append(slice(None, -o))
            elif o < 0:
                src_slices.append(slice(None, o))
                dst_slices.append(slice(-o, None))
            else:
                src_slices.append(slice(None))
                dst_slices.append(slice(None))
            if o and rho.shape[len(src_slices) - 1] <= abs(o):
                valid = False
        if not valid:
            continue
        src = tuple(src_slices)
        dst = tuple(dst_slices)
        m = act[src]
        count[dst] += m
        out_rho[dst] += np.where(m, rho[src], 0.0)
        for k, v in enumerate(vel):
            out_vel[k][dst] += np.where(m, v[src], 0.0)

    has = count > 0
    out_rho = np.where(has, out_rho / np.maximum(count, 1.0), float(fallback_rho))
    out_vel = [np.where(has, v / np.maximum(count, 1.0), 0.0) for v in out_vel]
    return out_rho, out_vel, count