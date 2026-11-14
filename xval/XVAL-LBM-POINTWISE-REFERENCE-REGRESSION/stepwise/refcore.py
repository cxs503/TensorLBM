"""Pure-numpy core of the pointwise full-step reference (NO lbmpy, NO torch).

Split out of ``stepwise_ref.py`` so that the frozen-reference regression gate
(``tests/test_xval_pointwise_stepwise_refdata.py``) can run on machines where
lbmpy is not installed -- that is the whole point of freezing the reference
data.  Anything that needs lbmpy lives in ``stepwise_ref.py``.

Contents
--------
* TensorLBM direction order / weights (read from ``tensorlbm.d2q9``).
* ``definition_stream``: the textbook pull streaming, per direction ``np.roll``.
* ``shear_wave_field`` / ``shear_amplitude``: the physics arm.
* ``compare``: the pointwise metric used by every arm.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

D2Q9 = "D2Q9"


def tensorlbm_velocities(lattice: str = D2Q9) -> np.ndarray:
    """Velocity set in TensorLBM direction order (``(Q, dim)``)."""
    if lattice == "D2Q9":
        from tensorlbm.d2q9 import C  # noqa: PLC0415

        return np.asarray(C, dtype=np.float64)
    if lattice == "D3Q19":
        from tensorlbm.d3q19 import C as C19  # noqa: PLC0415

        return np.asarray(C19, dtype=np.float64)
    raise ValueError(lattice)


def weights(lattice: str = D2Q9) -> np.ndarray:
    if lattice == "D2Q9":
        from tensorlbm.d2q9 import W_EXACT  # noqa: PLC0415

        return np.asarray(W_EXACT, dtype=np.float64)
    if lattice == "D3Q19":
        from tensorlbm.d3q19 import W_EXACT64  # noqa: PLC0415

        return np.asarray(W_EXACT64, dtype=np.float64)
    raise ValueError(lattice)


def equilibrium_field(lattice: str, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Second-order discrete Maxwellian, TensorLBM order, ``(Q, *spatial)``."""
    vel = tensorlbm_velocities(lattice)
    w = weights(lattice).reshape((len(vel),) + (1,) * rho.ndim)
    cu = np.tensordot(vel, u, axes=(1, 0))
    u2 = (u ** 2).sum(axis=0, keepdims=True)
    return rho * w * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2)


def definition_stream(f: np.ndarray, lattice: str = D2Q9) -> np.ndarray:
    """Pull streaming straight from the definition, periodic wrap.

    ``out[q, y, x] = f[q, y - c_qy, x - c_qx]`` implemented with ``np.roll``
    (``roll(a, s)`` yields ``out[i] = a[i - s]``), direction-major.
    """
    f = np.asarray(f, dtype=np.float64)
    vel = tensorlbm_velocities(lattice).astype(int)
    out = np.empty_like(f)
    for q in range(f.shape[0]):
        out[q] = np.roll(f[q], shift=(int(vel[q, 1]), int(vel[q, 0])), axis=(0, 1))
    return out


def shear_wave_field(ly: int = 64, lx: int = 8, *, u0: float = 2e-2, k_mode: int = 1,
                     rho0: float = 1.0, lattice: str = D2Q9) -> np.ndarray:
    """Divergence-free shear wave ``u_x = u0 sin(k y)``, uniform density."""
    y = np.arange(ly, dtype=np.float64)
    ky = 2.0 * np.pi * k_mode / ly
    ux = u0 * np.sin(ky * y)[:, None] * np.ones((1, lx))
    rho = np.full((ly, lx), rho0)
    u = np.zeros((2, ly, lx))
    u[0] = ux
    return equilibrium_field(lattice, rho, u)


def shear_amplitude(f: np.ndarray, lattice: str = D2Q9, *, k_mode: int = 1) -> float:
    """Amplitude of the ``sin(k y)`` mode of ``j_x`` (mode-1 projection)."""
    mom = moments(np.asarray(f, dtype=np.float64), lattice)
    ly = mom["j"].shape[1]
    ky = 2.0 * np.pi * k_mode / ly
    y = np.arange(ly, dtype=np.float64)
    return float(2.0 * (mom["j"][0] * np.sin(ky * y)[:, None]).mean(axis=0).mean())


def moments(f: np.ndarray, lattice: str = D2Q9) -> dict[str, np.ndarray]:
    vel = tensorlbm_velocities(lattice)
    return {"rho": f.sum(axis=0), "j": np.tensordot(vel, f, axes=(0, 0))}


def compare(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Pointwise deviation metrics between a kernel output and a reference."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
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


def sha256_file_bytes(arr: np.ndarray) -> str:
    """sha256 of an array's C-contiguous float64 bytes (per-array provenance)."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()