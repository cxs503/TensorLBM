"""Pure-numpy core of the D3Q19 *whole step* pointwise reference.

No lbmpy, no torch.  The frozen-reference gate
(``tests/test_xval_pointwise_d3q19_refdata.py``) imports this module so the
regression runs on machines where lbmpy is absent (round 2 showed that an
lbmpy-dependent gate is a dead gate outside the dev container).

Contents
--------
* D3Q19 velocity set / exact binary64 weights, read from the repo (single
  source of truth for the *model*, never a hand-typed table).
* ``definition_stream3d`` -- the textbook pull streaming
  ``f_i(x, t+1) = f_i^post(x - c_i, t)`` implemented as per-direction
  ``np.roll``; it shares no code with any in-repo streaming kernel.
* ``discrete_equilibrium`` / ``continuous_equilibrium`` -- the two equilibrium
  conventions lbmpy can hand out (``continuous_equilibrium=False/True``),
  re-derived here in numpy so the convention arm is checkable without lbmpy.
* ``compare`` / ``moments`` -- the pointwise metric and the conserved moments.
* Physics helpers: 3-D shear wave and Arnold-Beltrami-Childress (Beltrami)
  flow, both with closed-form decay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

LATTICE = "D3Q19"
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# lattice bookkeeping (read from the repo)
# ---------------------------------------------------------------------------
def velocities() -> np.ndarray:
    """D3Q19 velocities in TensorLBM direction order, ``(19, 3)`` float64."""
    from tensorlbm.d3q19 import C  # noqa: PLC0415

    return np.asarray(C, dtype=np.float64)


def weights() -> np.ndarray:
    """Exact binary64 D3Q19 weights in TensorLBM order (``W_EXACT64``)."""
    from tensorlbm.d3q19 import W_EXACT64  # noqa: PLC0415

    return np.asarray(W_EXACT64, dtype=np.float64)


def opposite_index() -> np.ndarray:
    """Index of the opposite direction, derived from the velocity tuples."""
    vel = velocities().astype(int)
    lut = {tuple(v): i for i, v in enumerate(vel)}
    return np.array([lut[tuple(-v)] for v in vel], dtype=int)


# ---------------------------------------------------------------------------
# streaming (definition, independent of the repo)
# ---------------------------------------------------------------------------
def definition_stream3d(f: np.ndarray) -> np.ndarray:
    """Pull streaming ``f_i(x) <- f_i^post(x - c_i)`` on a periodic box.

    Layout ``(19, nz, ny, nx)``; rolls are applied on the three spatial axes
    only.  Round 2 pinned this operator against 6 in-repo kernels (eps = 0), so
    it is used as the streaming half of the reference step here as well.
    """
    c = velocities().astype(int)
    out = np.empty_like(f)
    for i in range(f.shape[0]):
        cx, cy, cz = c[i]
        out[i] = np.roll(f[i], shift=(-cz, -cy, -cx), axis=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# equilibria (the two conventions)
# ---------------------------------------------------------------------------
def _cu(u: np.ndarray) -> np.ndarray:
    """``c_i . u`` per node, shape ``(19, *spatial)``."""
    return np.tensordot(velocities(), u, axes=(1, 0))


def discrete_equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Discrete (moment-conserving) 2nd-order Maxwellian, TensorLBM convention."""
    w = weights().reshape((19,) + (1,) * rho.ndim)
    cu = _cu(u)
    u2 = np.einsum("k...,k...->...", u, u)
    return w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def continuous_equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous Maxwellian ``rho (2 pi cs^2)^(-d/2) exp(-(c-u)^2 / 2cs^2)``.

    This is exactly what lbmpy's ``ContinuousHydrodynamicMaxwellian`` is, and it
    is what ``create_srt(..., continuous_equilibrium=True)`` puts into the
    collision rule.  It is *not* moment conserving on a discrete stencil
    (``sum_i f_i^eq != rho``), which is why it may only be used as a reference
    after an explicit convention decision -- see the report's arm D4.
    """
    pref = (1.0 / (2.0 * np.pi * CS2)) ** (1.5)
    c = velocities().reshape((19, 3) + (1,) * (u.ndim - 1))
    diff = c - u[None, ...]              # (19, 3, *spatial)
    d2 = (diff * diff).sum(axis=1)       # (19, *spatial)
    return pref * rho[None, ...] * np.exp(-d2 / (2.0 * CS2))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compare(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Pointwise comparison metrics for two ``(19, *spatial)`` fields."""
    diff = got - ref
    scale = float(np.abs(ref).max())
    return {
        "eps_inf": float(np.abs(diff).max() / scale),
        "eps_l2": float(np.sqrt((diff ** 2).mean()) / np.sqrt((ref ** 2).mean())),
        "max_abs": float(np.abs(diff).max()),
        "ref_scale": scale,
    }


def moments(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rho, u)`` from a ``(19, *spatial)`` field."""
    rho = f.sum(axis=0)
    j = np.einsum("ki,k...->i...", velocities(), f)
    return rho, j / rho


def conservation(f_prev: np.ndarray, f_next: np.ndarray) -> dict[str, float]:
    """Relative mass / momentum drift of one step, normalised by total mass."""
    rho0, u0 = moments(f_prev)
    rho1, u1 = moments(f_next)
    m0 = float(rho0.sum())
    mass = abs(float(rho1.sum()) - m0) / m0
    j0 = (u0 * rho0[None]).reshape(3, -1).sum(axis=1)
    j1 = (u1 * rho1[None]).reshape(3, -1).sum(axis=1)
    mom = float(np.abs(j1 - j0).max() / m0)
    return {"mass_drift": mass, "momentum_drift": mom}


# ---------------------------------------------------------------------------
# physics helpers (both with closed-form decay)
# ---------------------------------------------------------------------------
def shear_wave_3d(shape: tuple[int, int, int], u0: float, k: float) -> np.ndarray:
    """``u_x = u0 sin(k z)`` on the periodic box ``(nz, ny, nx)``.

    Exact solution of the Stokes problem: ``u_x(t) = u0 sin(k z) e^{-nu k^2 t}``.
    """
    nz, ny, nx = shape
    z = np.arange(nz)          # lattice index: k is already in lattice units
    u = np.zeros((3, nz, ny, nx))
    profile = u0 * np.sin(k * z)
    u[0] = profile[:, None, None]
    rho = np.ones((nz, ny, nx))
    return discrete_equilibrium(rho, u)


def shear_amplitude(f: np.ndarray) -> float:
    """Peak ``u_x`` of a 3-D shear wave (independent of phase)."""
    _, u = moments(f)
    return float(np.abs(u[0]).max())


def abc_field(n: int, amp: float) -> np.ndarray:
    """Arnold-Beltrami-Childress field at local equilibrium.

    ``u = A (sin z + cos y, sin x + cos z, cos x + sin y)`` with ``curl u = u``;
    hence ``(u.grad)u = grad(|u|^2/2)`` and ``lap u = -u``, so the field decays
    *exactly* as ``u(t) = u(0) e^{-nu t}`` under Navier-Stokes when the
    wavenumber is 1.  On the ``n^3`` lattice the sampled mode has lattice
    wavenumber ``k = 2 pi / n``, so the expected lattice decay rate is
    ``nu k^2`` (that is what the driver compares against).  It is a genuine
    3-D anchor: all three velocity components are non-zero and the vorticity is
    not aligned with any axis.
    """
    t = (np.arange(n) + 0.5) * (2.0 * np.pi / n)
    z = t[:, None, None]
    y = t[None, :, None]
    x = t[None, None, :]
    u = np.empty((3, n, n, n))
    u[0] = amp * (np.sin(z) + np.cos(y))
    u[1] = amp * (np.sin(x) + np.cos(z))
    u[2] = amp * (np.cos(x) + np.sin(y))
    rho = np.ones((n, n, n))
    return discrete_equilibrium(rho, u)


def abc_amplitude(f: np.ndarray) -> float:
    """Peak velocity magnitude of the ABC field."""
    _, u = moments(f)
    return float(np.sqrt((u ** 2).sum(axis=0)).max())