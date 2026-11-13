"""Reference objects for XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE.

Four independent objects are built here so that the library kernel
(:func:`tensorlbm.multiphase.sc_two_component_force`) can be judged against
references that do not share its code path:

``oss_force``
    NumPy transcription of the open-source *forward-gather* stencil
    ``F_sigma = -g psi_sigma sum_i w_i c_i psi_sigma'(x + c_i)`` as published in
    ``omalaspinas/palabos`` (``src/multiPhysics/shanChenProcessor3D.hh:304``,
    force accumulated with the isotropic ``t_i`` weights, ``psi`` supplied by a
    policy function - ``PsiIsRho`` is the linear choice) and in
    ``sorush-khajepor/listLBM`` (``mcmp_mod.f90``).  Inside TensorLBM the same
    interaction is written as a *reverse* gather
    ``-G_12 rho_sigma sum_i w_i c_i rho_sigma'(x - c_i)``, so the convention map
    is ``g_oss = -G_12`` (OSS positive = repulsive, TensorLBM negative =
    repulsive); :func:`oss_force` lets that map be checked elementwise instead
    of asserted.

``spectral_force``
    Continuum reference.  The cross interaction of the two-component
    Shan-Chen model is, to leading order in the lattice spacing,

        ``F_sigma = -G_12 psi_sigma grad psi_sigma'``  with the D2Q9 weights
        supplying ``-cs^2 grad`` inside the reverse gather, i.e.
        ``F_sigma = +G_12 cs^2 psi_sigma grad psi_sigma'``

    (derivation in ``REPORT.md`` section 2).  On a periodic field the exact
    gradient of the *sampled* field is available from the Fourier symbol
    ``i k`` (:func:`spectral_grad`), which is exact for band-limited fields and
    super-algebraically accurate for smooth periodic ones.  With
    ``psi = rho`` (this library's MCMP choice) the reference force is
    ``G_12 cs^2 rho1 * spectral_grad(rho2)``.

``symbol_ratio``
    Closed-form symbol of the discrete stencil.  For a single Fourier mode
    ``rho2 = A sin(k.x)`` with a constant companion density, the discrete force
    equals the continuum force multiplied by

        ``S(k) = sum_i w_i (c_i.khat) sin(k.c_i) / (cs^2 |k|)``

    so the relative L2 error of the operator on that mode is ``|1 - S(k)|``.
    On the axis this reduces to ``sin(k)/k`` (the well-known second-order
    underestimation of a central difference), which makes the measured error
    predictable in closed form rather than merely tabulated.

All functions are pure NumPy, float64, periodic wrap - no torch, no library
import, so an implementation defect in the solver cannot be inherited here.
"""

from __future__ import annotations

import numpy as np

CS2 = 1.0 / 3.0

# D2Q9 (identical ordering to tensorlbm.d2q9.C / .W, restated here on purpose:
# a reference that imported the tested module's constants would not be able to
# catch a wrong weight table).
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.float64,
)
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64)


# ---------------------------------------------------------------------------
# OSS stencil (forward gather) and its variants
# ---------------------------------------------------------------------------


def _shift(field: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Periodic ``field(x + c)`` for integer ``c`` (rows = y, cols = x)."""
    return np.roll(field, shift=(-int(cy), -int(cx)), axis=(0, 1))


def oss_force(
    psi1: np.ndarray,
    psi2: np.ndarray,
    g_oss: float,
    *,
    weights: np.ndarray | None = None,
    directions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``F_1 = -g psi_1 sum_i w_i c_i psi_2(x + c_i)`` (Palabos/listLBM form).

    ``weights``/``directions`` overrides exist only for the mutation controls of
    the falsifiability study (``run_xval.py::ARMS['mutation']``); the default
    call is the published stencil.
    """
    w = W if weights is None else weights
    c = C if directions is None else directions
    acc = np.zeros(psi2.shape + (2,), dtype=np.float64)
    for wi, ci in zip(w, c):
        acc += wi * ci * _shift(psi2, ci[0], ci[1])[..., None]
    acc *= -g_oss * psi1[..., None]
    return acc[..., 0].copy(), acc[..., 1].copy()


# ---------------------------------------------------------------------------
# Continuum reference: exact (spectral) gradient
# ---------------------------------------------------------------------------


def spectral_grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact periodic derivative of the sampled field (Fourier symbol ``i k``)."""
    ny, nx = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=1.0)
    fhat = np.fft.fft2(field)
    dx = np.real(np.fft.ifft2(1j * kx[None, :] * fhat))
    dy = np.real(np.fft.ifft2(1j * ky[:, None] * fhat))
    return dx, dy


def spectral_force(
    rho1: np.ndarray, rho2: np.ndarray, G_12: float
) -> tuple[np.ndarray, np.ndarray]:
    """Continuum cross force ``F_1 = G_12 cs^2 rho_1 grad rho_2`` (library sign)."""
    d2x, d2y = spectral_grad(rho2)
    return G_12 * CS2 * rho1 * d2x, G_12 * CS2 * rho1 * d2y


# ---------------------------------------------------------------------------
# Closed-form symbol of the discrete stencil
# ---------------------------------------------------------------------------


def symbol_ratio(kvec: tuple[float, float]) -> float:
    """``S(k)``: discrete/continuum ratio of the cross-force operator at mode k."""
    kvec_arr = np.asarray(kvec, dtype=np.float64)
    kmag = float(np.linalg.norm(kvec_arr))
    if kmag == 0.0:
        return 1.0
    khat = kvec_arr / kmag
    proj = C @ khat  # c_i . khat
    return float(np.sum(W * proj * np.sin(C @ kvec_arr)) / (CS2 * kmag))


# ---------------------------------------------------------------------------
# Analytic test fields
# ---------------------------------------------------------------------------


def mode_field(n: int, kvec: tuple[float, float], amplitude: float = 0.25, base: float = 0.5):
    """``rho = base + amplitude sin(k.x)`` on an ``n x n`` periodic grid."""
    y, x = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")
    phase = kvec[0] * x + kvec[1] * y
    return base + amplitude * np.sin(phase)


def sloshing_fields(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two incommensurate smooth fields (non-constant companion density).

    Used for the momentum-defect arm: a constant ``rho_1`` makes the *standard*
    force sum vanish trivially, which would hide the action-reaction defect.
    """
    y, x = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")
    k1 = 2.0 * np.pi / n
    rho1 = 0.55 + 0.20 * np.sin(k1 * x) + 0.12 * np.cos(3.0 * k1 * y)
    rho2 = 0.45 + 0.18 * np.cos(2.0 * k1 * x + 0.7) + 0.10 * np.sin(k1 * y)
    return rho1, rho2