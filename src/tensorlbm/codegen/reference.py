"""Independent NumPy reference kernels for cross-validating generated code.

These deliberately use a *different* code path from the generator:

* moments/equilibrium are written with stacked velocity arrays and broadcasting
  (``tensordot``/``einsum``-style), not unrolled statements;
* the TRT split is obtained by averaging the full equilibrium against its
  opposite direction (``feq_plus = (feq + feq_opp)/2``), not from the analytic
  even/odd expressions the template bakes in.

Agreement between the two is therefore a real check of the emitted algebra, and
disagreement points at the generator rather than at the mathematics.
"""

from __future__ import annotations

import numpy as np

from .protocol import CS2, D2Q9, StencilSpec

_OPP_CACHE: dict[str, np.ndarray] = {}


def _opp_index(stencil: StencilSpec) -> np.ndarray:
    key = stencil.name
    if key not in _OPP_CACHE:
        _OPP_CACHE[key] = np.asarray(stencil.opposite, dtype=int)
    return _OPP_CACHE[key]


def _arrays(stencil: StencilSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ex = np.asarray(stencil.ex, dtype=float).reshape(-1, 1, 1)
    ey = np.asarray(stencil.ey, dtype=float).reshape(-1, 1, 1)
    w = np.asarray(stencil.w, dtype=float).reshape(-1, 1, 1)
    return ex, ey, w


def reference_moments(
    f: np.ndarray, stencil: StencilSpec = D2Q9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ex = np.asarray(stencil.ex, dtype=float)
    ey = np.asarray(stencil.ey, dtype=float)
    rho = f.sum(axis=0)
    ux = np.tensordot(ex, f, axes=(0, 0)) / rho
    uy = np.tensordot(ey, f, axes=(0, 0)) / rho
    return rho, ux, uy


def reference_equilibrium(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    stencil: StencilSpec = D2Q9,
) -> np.ndarray:
    ex, ey, w = _arrays(stencil)
    cu = ex * ux[None] + ey * uy[None]
    usq = ux * ux + uy * uy
    return w * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def reference_collide(
    f: np.ndarray,
    *,
    family: str = "BGK",
    tau: float = 0.6,
    magic: float = 3.0 / 16.0,
    stencil: StencilSpec = D2Q9,
) -> np.ndarray:
    rho, ux, uy = reference_moments(f, stencil)
    feq = reference_equilibrium(rho, ux, uy, stencil)
    s_plus = 1.0 / tau
    if family == "BGK":
        return f - s_plus * (f - feq)
    if family != "TRT":
        raise ValueError(f"unknown family {family!r}")
    s_minus = 1.0 / (magic / (1.0 / s_plus - 0.5) + 0.5)
    opp = _opp_index(stencil)
    f_opp = f[opp]
    feq_opp = feq[opp]
    fp, fm = 0.5 * (f + f_opp), 0.5 * (f - f_opp)
    feqp, feqm = 0.5 * (feq + feq_opp), 0.5 * (feq - feq_opp)
    return (fp - s_plus * (fp - feqp)) + (fm - s_minus * (fm - feqm))


def reference_stream(f_post: np.ndarray, stencil: StencilSpec = D2Q9) -> np.ndarray:
    out = np.empty_like(f_post)
    for i in range(stencil.q):
        out[i] = np.roll(f_post[i], (stencil.ey[i], stencil.ex[i]), axis=(0, 1))
    return out


def reference_step(
    f: np.ndarray,
    *,
    family: str = "BGK",
    tau: float = 0.6,
    magic: float = 3.0 / 16.0,
    stencil: StencilSpec = D2Q9,
) -> np.ndarray:
    return reference_stream(
        reference_collide(f, family=family, tau=tau, magic=magic, stencil=stencil),
        stencil,
    )


def shear_viscosity(tau: float) -> float:
    """Lattice shear viscosity of the supported components."""
    return (tau - 0.5) * CS2


__all__ = [
    "reference_collide",
    "reference_equilibrium",
    "reference_moments",
    "reference_step",
    "reference_stream",
    "shear_viscosity",
]