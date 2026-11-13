"""Analytic-solution / relative-L2 verification harness (XVAL port).

Provenance of the convention
----------------------------
This module ports the *error-evaluation practice* of the open-source code that
accompanies Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva & Viggen, *The Lattice
Boltzmann Method: Principles and Practice* (Springer 2016), MIT licence,
``lbm-principles-practice/code``.  The two files that define the convention are

* ``chapter13/matlab/vortexdecay.m``  -- the Taylor-Green decay driver: samples
  the field every ``NMSG`` steps and prints ``t, E, L2rho, L2ux, L2uy``;
* ``chapter13/matlab/taylorgreen.m``  -- the closed-form solution the sampled
  field is compared against.

Reading those sources (not the book text) fixes the exact recipe implemented
below:

1. **Coordinates are cell centred**: ``x = (1:NX) - 0.5`` and ``y = (1:NY) - 0.5``
   in 1-based MATLAB, i.e. node ``i`` sits at ``x_i = i + 0.5`` (0-based) -- see
   :func:`cell_centred_coordinates`.
2. **Relative L2 norm is normalised by the reference *fluctuation***, not by the
   raw reference field::

       L2rho = sqrt( sum((rho - rho_a)^2) / sum((rho_a - rho0)^2) )
       L2ux  = sqrt( sum((ux  - ux_a )^2) / sum( ux_a^2 ) )
       L2uy  = sqrt( sum((uy  - uy_a )^2) / sum( uy_a^2 ) )

   For density the base is the rest density ``rho0``; for velocity it is zero,
   so both cases are the one formula implemented in :func:`relative_l2` with
   ``ref_base``.
3. **The "error" is reported as a sum of squares as well as a normalised norm**
   (``rhoe2``/``uxe2``/``uye2`` in the MATLAB driver); :class:`FieldError`
   carries both so a caller can quote whichever the ledger asks for.
4. **Energy diagnostic** is ``E = sum(rho * (u^2 + v^2))`` -- no factor 1/2 and
   no division by the node count (it is an extensive quantity).  See
   :func:`kinetic_energy`.
5. **Sampling happens at fixed step intervals** and the analytic solution is
   evaluated at the *same* step index (``t`` counts collisions already done),
   which matters: see ``docs/XVAL_m17416_TGV2D_ANALYTIC_L2_HARNESS_20261113.md``
   §5 for the off-by-one-step error that this convention pins down.
6. **Node restriction**: in wall-bounded benchmarks of the same OSS repository
   (``chapter5/*.m``, ``chapter6/force_poiseuille_*.m``) the norm is summed only
   over fluid nodes, the wall/virtual rows being excluded.  :func:`relative_l2`
   therefore takes an optional boolean ``mask`` (``True`` = count this node).

What is *not* ported: the MATLAB driver's ``circshift`` streaming, its specific
``tau = 1`` / ``u_max = 0.04/scale`` parameter ladder and its console format.
Those are case settings, not evaluation convention; the XVAL benchmark under
``benchmarks/verified/tgv2d_analytic_l2_xval/`` reproduces them separately so
the two arms can be compared at identical settings.

The harness is deliberately solver agnostic: it accepts ``numpy`` arrays or
``torch`` tensors and never touches populations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FieldError",
    "TGVAnalytic",
    "cell_centred_coordinates",
    "fluid_mask",
    "kinetic_energy",
    "relative_l2",
    "steady_criterion",
    "tgv2d_analytic",
    "tgv2d_decay_time",
]


# --------------------------------------------------------------------------- #
# analytic solutions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TGVAnalytic:
    """Closed-form 2-D Taylor-Green vortex state (Krüger et al., ch. 13).

    ``u  = -u_max sqrt(ky/kx) cos(kx x) sin(ky y) exp(-t/td)``
    ``v  =  u_max sqrt(kx/ky) sin(kx x) cos(ky y) exp(-t/td)``
    ``P  = -0.25 rho0 u_max^2 ((ky/kx) cos(2 kx x) + (kx/ky) cos(2 ky y)) exp(-2t/td)``
    ``rho = rho0 + 3 P``

    The last line is the lattice equation of state ``p = rho cs^2`` with
    ``cs^2 = 1/3``; it is what makes the density field, not just the velocity
    field, checkable.  ``P`` is the pressure in *lattice* units (``rho0 = 1``).
    """

    t: float
    kx: float
    ky: float
    td: float
    rho0: float
    u_max: float
    rho: Any
    ux: Any
    uy: Any
    p: Any

    @property
    def t_over_td(self) -> float:
        return self.t / self.td


def tgv2d_decay_time(nx: int, ny: int, nu: float) -> float:
    """Viscous decay time ``td = 1 / (nu (kx^2 + ky^2))`` with ``k = 2 pi / N``.

    Bit-for-bit the expression used by the OSS driver for its ``t/td`` label.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"grid must be positive, got nx={nx}, ny={ny}")
    if nu <= 0:
        raise ValueError(f"viscosity must be positive, got nu={nu}")
    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    return 1.0 / (nu * (kx * kx + ky * ky))


def cell_centred_coordinates(nx: int, ny: int, *, as_torch: bool = False):
    """Node coordinates ``x_i = i + 0.5``, ``y_j = j + 0.5`` (OSS convention)."""
    x = np.arange(nx, dtype=np.float64) + 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(x, y)
    if as_torch:  # pragma: no cover - convenience only
        import torch

        return torch.as_tensor(xx), torch.as_tensor(yy)
    return xx, yy


def tgv2d_analytic(
    t: float,
    nx: int,
    ny: int,
    nu: float,
    *,
    rho0: float = 1.0,
    u_max: float = 0.04,
    as_torch: bool = False,
    device: Any = None,
    dtype: Any = None,
) -> TGVAnalytic:
    """Evaluate the exact Taylor-Green state at integer step ``t``.

    Parameters mirror ``chapter13/matlab/taylorgreen.m`` one-to-one
    (``t``, ``NX``, ``NY``, ``nu``, ``rho0``, ``u_max``) so a port can be
    diffed against the source line by line.  ``nx != ny`` is supported (the
    OSS driver only ever uses a square box, hence the ``sqrt(ky/kx)`` factors).
    """
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    td = 1.0 / (nu * (kx * kx + ky * ky))
    decay = math.exp(-t / td)
    decay2 = math.exp(-2.0 * t / td)
    ratio_k = math.sqrt(ky / kx)
    ratio_k_inv = math.sqrt(kx / ky)

    xx, yy = cell_centred_coordinates(nx, ny)
    ux = -u_max * ratio_k * np.cos(kx * xx) * np.sin(ky * yy) * decay
    uy = u_max * ratio_k_inv * np.sin(kx * xx) * np.cos(ky * yy) * decay
    p = (
        -0.25
        * rho0
        * u_max
        * u_max
        * (ratio_k * np.cos(2.0 * kx * xx) + ratio_k_inv * np.cos(2.0 * ky * yy))
        * decay2
    )
    rho = rho0 + 3.0 * p

    if as_torch:  # pragma: no cover - exercised by the torch arm
        import torch

        conv = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        return TGVAnalytic(t, kx, ky, td, rho0, u_max, conv(rho), conv(ux), conv(uy), conv(p))
    return TGVAnalytic(t, kx, ky, td, rho0, u_max, rho, ux, uy, p)


# --------------------------------------------------------------------------- #
# error norms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldError:
    """Relative L2 error of one field, OSS normalisation.

    ``rel_l2 = sqrt(ssq / ref_ssq)`` with
    ``ssq = sum_mask (num - ref)^2`` and ``ref_ssq = sum_mask (ref - ref_base)^2``.
    """

    rel_l2: float
    ssq: float
    ref_ssq: float
    n_nodes: int

    def as_dict(self) -> dict[str, float]:
        return {
            "rel_l2": self.rel_l2,
            "ssq": self.ssq,
            "ref_ssq": self.ref_ssq,
            "n_nodes": float(self.n_nodes),
        }


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):  # torch tensor
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _as_mask(mask: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    m = _to_numpy(mask).astype(bool)
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} does not match field shape {shape}")
    if not m.any():
        raise ValueError("mask selects no nodes")
    return m


def relative_l2(
    num: Any,
    ref: Any,
    *,
    ref_base: float = 0.0,
    mask: Any = None,
) -> FieldError:
    """Relative L2 norm of ``num - ref`` over the selected nodes.

    Implements the OSS convention (see module docstring): the denominator uses
    the reference *fluctuation about* ``ref_base``.  Use ``ref_base=rho0`` for a
    density field and the default ``0.0`` for a velocity component -- exactly
    the pair of calls the OSS driver makes (``sumrhoa2`` vs ``sumuxa2``).

    Restricting the sum to fluid nodes is done by passing ``mask`` (``True`` for
    nodes that count).  Wall and virtual layers must be excluded explicitly;
    this function cannot know about them.
    """
    a = _to_numpy(num).astype(np.float64)
    b = _to_numpy(ref).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: num {a.shape} vs ref {b.shape}")
    m = _as_mask(mask, a.shape)
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite values in num field")
    if not np.all(np.isfinite(b)):
        raise ValueError("non-finite values in ref field")

    diff = a - b
    ref_fluct = b - ref_base
    if m is None:
        ssq = float(np.sum(diff * diff))
        ref_ssq = float(np.sum(ref_fluct * ref_fluct))
        n_nodes = int(diff.size)
    else:
        ssq = float(np.sum(diff[m] * diff[m]))
        ref_ssq = float(np.sum(ref_fluct[m] * ref_fluct[m]))
        n_nodes = int(m.sum())

    if ref_ssq <= 0.0:
        raise ValueError(
            "reference fluctuation is identically zero over the selected nodes; "
            "relative L2 is undefined (use an absolute norm instead)"
        )
    return FieldError(rel_l2=math.sqrt(ssq / ref_ssq), ssq=ssq, ref_ssq=ref_ssq, n_nodes=n_nodes)


def kinetic_energy(rho: Any, ux: Any, uy: Any, *, mask: Any = None) -> float:
    """``E = sum(rho (u^2 + v^2))`` -- OSS extensive energy diagnostic."""
    r = _to_numpy(rho).astype(np.float64)
    u = _to_numpy(ux).astype(np.float64)
    v = _to_numpy(uy).astype(np.float64)
    if not (r.shape == u.shape == v.shape):
        raise ValueError("rho/ux/uy shapes must match")
    e = r * (u * u + v * v)
    m = _as_mask(mask, e.shape)
    return float(np.sum(e if m is None else e[m]))


def fluid_mask(solid: Any, *, extra_excluded_rows: int = 0) -> np.ndarray:
    """Boolean mask of countable nodes: ``not solid``, minus boundary rows.

    ``extra_excluded_rows`` drops that many rows at the top and bottom of the
    array -- the wall/wet-node rows that the OSS wall-bounded benchmarks leave
    out of the norm.  Purely geometric: no physics is assumed.
    """
    s = _to_numpy(solid).astype(bool)
    if s.ndim != 2:
        raise ValueError(f"solid mask must be 2-D, got shape {s.shape}")
    m = ~s
    if extra_excluded_rows:
        k = int(extra_excluded_rows)
        if 2 * k >= m.shape[0]:
            raise ValueError("extra_excluded_rows leaves no countable rows")
        m = m.copy()
        m[:k, :] = False
        m[-k:, :] = False
    return m


def steady_criterion(mean_history: list[float], *, tol: float = 1e-12) -> dict[str, Any]:
    """OSS steady-state stopping rule on a scalar history (e.g. mean ``u_x``).

    The OSS benchmarks iterate until ``|mean(u)/mean(u_old) - 1| < 1e-12``.
    Returning ``converged`` plus the last residual keeps the decision auditable
    instead of a silent ``break`` inside a solver loop.
    """
    if len(mean_history) < 2:
        raise ValueError("need at least two samples to form a relative change")
    old, new = float(mean_history[-2]), float(mean_history[-1])
    if old == 0.0:
        raise ValueError("previous mean is zero; relative-change criterion undefined")
    resid = abs(new / old - 1.0)
    return {"converged": bool(resid < tol), "residual": resid, "tol": tol, "n_samples": len(mean_history)}