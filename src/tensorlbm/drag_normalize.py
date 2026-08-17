"""Common Cd (drag) normalisation helpers for AMR/octree solvers.

Consolidates the drag-normalisation arithmetic that used to be copy-pasted
across 12+ amr/octree example scripts (with mutually inconsistent area
formulas — count-weighted mean / wall-link / ``2^-(1+d_max)`` — which caused
the "area phantom" Cd inflation bug).  Single source of truth:

* :func:`compute_wall_link_dx`      — wall-lattice leaf dx (coarse cells),
  taken from the leaf levels that actually carry the BFL wall links
  (dcf899e fix: force lives on the finest wall-adjacent leaves, so the
  shell-wide count-weighted mean dx is WRONG for area purposes).
* :func:`leaf_radius_from_dx`       — body radius expressed in the wall
  lattice units (``radius_coarse / dx_leaf``).
* :func:`dynamic_area`              — ``0.5 * u^2 * pi * radius_leaf^2``,
  the reference dynamic-pressure area for ``Cd = F / dynamic_area``.
* :func:`compute_blockage_factor`   — wind-tunnel blockage correction
  (ported from examples/octree_integrated_validate.py, 69d027c).

Conventions (matching examples/octree_integrated_validate.py):

* Leaf dx relative to the COARSE grid: ``2^-(1+l)`` on the L1-block path
  (host grid = 2x coarse), ``2^-l`` on the legacy two-level path
  (host grid = coarse), where ``l`` is the leaf level (1 or 2).
* The BFL wall links all live on the finest wall-adjacent leaves
  (level-2 for d_max=2); ``dx_leaf_area`` is the mean wall-link-leaf dx,
  falling back to the finest level ``2^-(1+d_max)`` / ``2^-d_max`` when no
  leaf carries a wall link.
* ``Cd = F_mem / dynamic_area`` with ``dynamic_area = 0.5 u^2 pi R_leaf^2``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

# Wind-tunnel blockage correction knobs (69d027c semantics).
#   beta = D / Ly   (D = body diameter, Ly = domain width = ny)
#   simple           (Maskell/Bartz):   f = 1 / (1 - beta)^2
#   glauert          (Glauert-type):    f = 1 / sqrt(1 - beta^2)
#   glauert_classic  (classic Glauert): f = 1 / (1 - 1.5 * beta)
#   off:             f = 1.0
# Hard gate: beta >= BLOCKAGE_HARD_GATE auto-escalates 'simple' -> 'glauert'
# (the conservative higher-order form for severe blockage).  Enlarging the
# domain to Ly >= 8D keeps beta <= 12.5% (BLOCKAGE_WARN_RATIO).
BLOCKAGE_HARD_GATE = 0.15    # beta >= 15%: severe blockage -> warn + escalate
BLOCKAGE_WARN_RATIO = 0.125  # beta > 12.5%: soft advisory (Ly >= 8D rule)


def compute_wall_link_dx(octree, l1_block: bool = True) -> float:
    """Mean wall-link leaf dx in coarse cells (the area-correct resolution).

    Selects the leaves that actually carry BFL wall links
    (``bfl_mask.any(dim=0)``) and averages their dx: the wall force lives on
    the finest wall-adjacent leaves, so the shell-wide count-weighted mean dx
    (which mixes in the wall-free level-1 outer band) would shrink the area
    and inflate Cd by ``(dx_wall/dx_count)^2`` — the R10 d2 Cd 9.33 root
    cause (dcf899e).  Falls back to the finest leaf dx
    (``2^-(1+d_max)`` L1 / ``2^-d_max`` legacy) when no wall links exist.

    Args:
        octree: object exposing ``leaf_level`` (n_leaf, int64, levels 1|2),
            ``bfl_mask`` (Q, n_leaf, bool) and ``d_max`` (int) — e.g. the
            ``OctreeGrid`` from ``tensorlbm.octree_boundary.geometry``.
        l1_block: True for the L1 middle-block path (host grid = 2x coarse,
            leaf dx = ``2^-(1+l)``), False for the legacy two-level path
            (host grid = coarse, leaf dx = ``2^-l``).

    Returns:
        Mean wall-link leaf dx in coarse cells (float).
    """
    wall_lv = octree.leaf_level[octree.bfl_mask.any(dim=0)]
    if wall_lv.numel():
        dx_wall_leaf = 2.0 ** (-(1.0 + wall_lv.to(torch.float64))) \
            if l1_block else 2.0 ** (-wall_lv.to(torch.float64))
        return float(dx_wall_leaf.mean().item())
    # No wall links: fall back to the finest leaf resolution (the old
    # ``dx_leaf_old`` formula, which coincides with the finest leaf dx).
    return float(2.0 ** (-(1 + octree.d_max))) if l1_block \
        else float(2.0 ** (-octree.d_max))


def leaf_radius_from_dx(radius_coarse: float, dx_leaf: float) -> float:
    """Body radius in wall-lattice (leaf) units.

    ``dx_leaf`` is the wall-link leaf dx in coarse cells (see
    :func:`compute_wall_link_dx`); the radius expressed in the lattice that
    actually carries the wall force is ``radius_coarse / dx_leaf``.
    """
    return radius_coarse / dx_leaf


def dynamic_area(u: float, radius_leaf: float) -> float:
    """Reference dynamic-pressure area ``0.5 * u^2 * pi * radius_leaf^2``.

    The Cd normalisation denominator: ``Cd = F / dynamic_area`` with the
    force measured in the wall (leaf) lattice and ``radius_leaf`` the body
    radius in the same lattice.  All lattice factors cancel exactly, so the
    dimensionless Cd is resolution-independent (see
    ``octree_boundary/force.py`` conventions).
    """
    return 0.5 * u ** 2 * math.pi * radius_leaf ** 2


def compute_blockage_factor(
    beta: float,
    mode: str,
    hard_gate: float = BLOCKAGE_HARD_GATE,
) -> Tuple[float, str, bool]:
    """Wind-tunnel blockage correction factor for the confined domain.

    Ported from ``examples/octree_integrated_validate.py`` (69d027c).
    Returns ``(corr_factor, bc_note, escalated)``.

    * ``off``              -> f = 1.0 (never escalated)
    * ``simple``           -> f = 1/(1-beta)^2     (Maskell/Bartz)
    * ``glauert``          -> f = 1/sqrt(1-beta^2) (Glauert-type)
    * ``glauert_classic``  -> f = 1/(1-1.5*beta)   (classic Glauert, legacy)

    Hard gate: when ``beta >= hard_gate`` (default 0.15) the ``simple`` mode
    is auto-escalated to ``glauert`` and ``escalated`` is True; ``off`` is
    always respected.  Pass ``hard_gate=1.0`` to disable the gate and obtain
    the raw ``simple`` factor (e.g. for formula-level unit tests).

    Examples (R6 sphere: D=12, Ly=64 -> beta=0.1875): simple f=1.5148
    (raw), glauert f=1.0181.
    """
    if mode == "off" or beta <= 0.0:
        return 1.0, "off", False
    if mode == "glauert":
        f = 1.0 / math.sqrt(max(1.0 - beta * beta, 1e-12))
        return f, "glauert 1/sqrt(1-beta^2)", False
    if mode == "glauert_classic":
        # Guard the 1-1.5*beta pole (beta >= 2/3 is unphysical here, but
        # never divide by <= 0).
        f = 1.0 / max(1.0 - 1.5 * beta, 1e-3)
        return f, "glauert_classic 1/(1-1.5*beta)", False
    # "simple" (default): the hard gate auto-escalates to Glauert.
    if beta >= hard_gate:
        f = 1.0 / math.sqrt(max(1.0 - beta * beta, 1e-12))
        return (
            f,
            "simple->glauert 1/sqrt(1-beta^2) "
            "(auto-escalated: beta>=15% hard gate)",
            True,
        )
    return 1.0 / (1.0 - beta) ** 2, "simple 1/(1-beta)^2", False
