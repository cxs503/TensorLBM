"""Pure-numpy core of the D3Q27 *whole step* pointwise reference (round 4).

No lbmpy, no torch on the reference side.  The frozen-bundle gate
(``tests/test_xval_pointwise_d3q27_refdata.py``) imports this module so the
regression runs on machines where lbmpy is absent -- rounds 1-3 showed that an
lbmpy-dependent gate is a dead gate outside the dev container.

Contents
--------
* D3Q27 velocity set / exact binary64 weights, read from the repo
  (``tensorlbm.d3q27``: single source of truth for the *model*, never a
  hand-typed table).
* ``definition_stream27`` -- textbook pull streaming
  ``f_i(x, t+1) = f_i^post(x - c_i, t)`` as per-direction ``np.roll``; shares no
  code with ``tensorlbm.d3q27.stream27_roll``.
* ``discrete_equilibrium`` / ``continuous_equilibrium`` -- the two equilibrium
  conventions lbmpy can hand out, re-derived in numpy.
* ``compare`` / ``moments`` / ``drift`` -- pointwise metric, conserved moments,
  per-step conservation drift.
* Physics helpers: 3-D shear wave with closed-form decay.
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

LATTICE = "D3Q27"
Q = 27
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# lattice bookkeeping (read from the repo)
# ---------------------------------------------------------------------------
def velocities() -> np.ndarray:
    """D3Q27 velocities in TensorLBM direction order, ``(27, 3)`` int."""
    from tensorlbm.d3q27 import C  # noqa: PLC0415

    return np.asarray(C.detach().cpu().numpy(), dtype=int)


def weights() -> np.ndarray:
    """Exact binary64 D3Q27 weights in TensorLBM order (``W_EXACT64``)."""
    from tensorlbm.d3q27 import W_EXACT64  # noqa: PLC0415

    return np.asarray(W_EXACT64.detach().cpu().numpy(), dtype=np.float64)


def opposite_index() -> np.ndarray:
    """Index of the opposite direction, derived from the velocity tuples."""
    vel = velocities()
    lut = {tuple(v): i for i, v in enumerate(vel)}
    return np.array([lut[tuple(-v)] for v in vel], dtype=int)


# ---------------------------------------------------------------------------
# streaming (definition, independent of the repo kernels)
# ---------------------------------------------------------------------------
def definition_stream27(f: np.ndarray) -> np.ndarray:
    """Pull streaming ``f_i(x) <- f_i^post(x - c_i)`` on a periodic box.

    Layout ``(27, nz, ny, nx)``; rolls act on the three spatial axes only.
    """
    c = velocities()
    out = np.empty_like(f)
    for i in range(f.shape[0]):
        cx, cy, cz = c[i]
        out[i] = np.roll(f[i], shift=(-cz, -cy, -cx), axis=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# equilibria (the two conventions)
# ---------------------------------------------------------------------------
def _cu(u: np.ndarray) -> np.ndarray:
    """``c_i . u`` per node, shape ``(27, *spatial)``."""
    return np.tensordot(velocities().astype(np.float64), u, axes=(1, 0))


def discrete_equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Discrete (moment-conserving) 2nd-order Maxwellian, TensorLBM convention."""
    w = weights().reshape((Q,) + (1,) * rho.ndim)
    cu = _cu(u)
    u2 = np.einsum("k...,k...->...", u, u)
    return w * rho[None, ...] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None, ...])


def continuous_equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous Maxwellian ``rho (2 pi cs^2)^(-3/2) exp(-(c-u)^2 / 2cs^2)``."""
    pref = (1.0 / (2.0 * np.pi * CS2)) ** 1.5
    c = velocities().astype(np.float64).reshape((Q, 3) + (1,) * (u.ndim - 1))
    diff = c - u[None, ...]
    d2 = (diff * diff).sum(axis=1)
    return pref * rho[None, ...] * np.exp(-d2 / (2.0 * CS2))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compare(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Pointwise comparison metrics for two ``(27, *spatial)`` fields."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        raise ValueError(f"shape mismatch {got.shape} vs {ref.shape}")
    diff = got - ref
    scale = float(np.abs(ref).max())
    denom = np.maximum(np.abs(ref), 1e-300)
    return {
        "eps_scaled_inf": float(np.abs(diff).max() / max(scale, 1e-300)),
        "eps_rel_pointwise_max": float((np.abs(diff) / denom).max()),
        "eps_rel_rms": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-300)),
        "abs_diff_max": float(np.abs(diff).max()),
        "scale_abs_max": scale,
    }


def moments(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rho, u)`` of a ``(27, *spatial)`` field, ``u = j/rho`` (compressible)."""
    rho = f.sum(axis=0)
    j = np.tensordot(velocities().astype(np.float64), f, axes=(0, 0))
    u = j / np.maximum(rho, 1e-300)[None, ...]
    return rho, u


def drift(f_prev: np.ndarray, f_next: np.ndarray) -> dict[str, float]:
    """Relative mass and momentum drift of one step (conservation check)."""
    rho0, u0 = moments(f_prev)
    rho1, u1 = moments(f_next)
    m0 = float(rho0.sum())
    mass = abs(float(rho1.sum()) - m0) / m0
    j0 = (u0 * rho0[None]).reshape(3, -1).sum(axis=1)
    j1 = (u1 * rho1[None]).reshape(3, -1).sum(axis=1)
    mom = float(np.abs(j1 - j0).max() / m0)
    return {"mass_drift": mass, "momentum_drift": mom}


# ---------------------------------------------------------------------------
# physics helpers (closed-form decay)
# ---------------------------------------------------------------------------
def shear_wave_3d(shape: tuple[int, int, int], u0: float, k: float) -> np.ndarray:
    """``u_x = u0 sin(k z)`` on the periodic box ``(nz, ny, nx)``.

    Exact Stokes solution ``u_x(t) = u0 sin(k z) e^{-nu k^2 t}``.
    """
    nz, ny, nx = shape
    z = np.arange(nz)
    u = np.zeros((3, nz, ny, nx))
    u[0] = (u0 * np.sin(k * z))[:, None, None]
    rho = np.ones((nz, ny, nx))
    return discrete_equilibrium(rho, u)


def shear_amplitude(f: np.ndarray) -> float:
    """Peak ``u_x`` of a 3-D shear wave (independent of phase)."""
    _, u = moments(f)
    return float(np.abs(u[0]).max())


def lattice_wavenumber(n: int) -> float:
    """Wavenumber of the lowest mode representable on ``n`` lattice nodes."""
    return 2.0 * np.pi / n