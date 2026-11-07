"""Decoupled viscosity-ratio mapping for two-phase / pseudopotential LBM.

Why this module exists
----------------------
``benchmarks/pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md``
declared the Hadamard-Rybczynski (HR) benchmark *dead* on the grounds of
library gap **G2**:

    "No module separates viscosity ratio from density ratio with acceptable
     accuracy: CG family locks lambda=rho; SC-MCMP unequal-tau is structurally
     biased; Allen-Cahn has the knobs but the phase dynamics are
     non-functional."

The lock is easy to see.  In a single-relaxation-time (BGK) kernel the
*dynamic* viscosity of a cell is

    mu(x) = rho(x) c_s^2 (tau(x) - 1/2).                                   (1)

If ``tau`` is one constant for the whole lattice -- the only thing the
two-phase kernels expose -- then ``mu_d/mu_c = rho_d/rho_c``: the viscosity
ratio ``lambda`` is *slaved* to the density ratio.  The HR family is
parameterised by ``lambda`` at fixed mass-density contrast, so it was
un-runnable rather than un-implemented.

What this module provides
-------------------------
1. ``phase_viscosity`` -- viscosity field from a phase indicator, ratio
   ``lambda`` by construction (linear / harmonic / geometric averaging).
2. ``relaxation_time`` / ``pseudopotential_relaxation`` -- per-cell inversion of
   (1),

        tau(x) = 1/2 + mu(phi(x)) / (rho(x) c_s^2),                        (2)

   so ``mu_d/mu_c = lambda`` is imposed *independently* of ``rho_d/rho_c``.
3. ``legacy_uniform_relaxation`` -- the pre-fix locked mapping, kept so the
   difference can be *measured* instead of asserted.
4. ``shear_wave_decay`` -- periodic D2Q9 BGK transverse-wave probe that
   *measures* the viscosity a ``tau`` field actually realises, i.e. an
   independent check of (2) on the real lattice kernel (not an identity).

Conventions: ``phi = 0`` continuous phase, ``phi = 1`` dispersed phase,
``lambda = mu_d/mu_c``, ``mu`` dynamic viscosity, ``rho`` mass density,
``cs2`` lattice speed of sound squared (1/3 for D2Q9/D3Q19).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

CS2 = 1.0 / 3.0

D2Q9_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
D2Q9_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64)


def _check_lam(lam: float) -> float:
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lambda must be finite and >= 0, got {lam}")
    return lam


def phase_viscosity(phi, mu_c: float, lam: float, scheme: str = "linear"):
    """Dynamic viscosity field from a phase indicator ``phi``.

    ``phi = 0`` continuous, ``phi = 1`` dispersed, ratio ``lam`` by
    construction.  ``scheme='harmonic'`` is the series (interface-consistent)
    average and the right choice when the indicator is smeared over a cell;
    ``scheme='linear'`` is the arithmetic average; ``scheme='geometric'`` the
    logarithmic one.
    """
    phi = np.asarray(phi, dtype=np.float64)
    lam = _check_lam(lam)
    if not np.isfinite(mu_c) or mu_c <= 0.0:
        raise ValueError(f"mu_c must be positive, got {mu_c}")
    mu_d = lam * mu_c
    if scheme == "linear":
        out = mu_c + (mu_d - mu_c) * phi
    elif scheme == "harmonic":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 1.0 / ((1.0 - phi) / mu_c + phi / max(mu_d, 1e-300))
        out = np.where(phi <= 0.0, mu_c, out)
        out = np.where(phi >= 1.0, mu_d, out)
    elif scheme == "geometric":
        out = mu_c * np.exp(np.clip(phi, 0.0, 1.0) * np.log(max(lam, 1e-300)))
        out = np.where(phi <= 0.0, mu_c, out)
        out = np.where(phi >= 1.0, mu_d, out)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return np.asarray(out, dtype=np.float64)


def relaxation_time(rho, mu, cs2: float = CS2):
    """Invert ``mu = rho cs2 (tau - 1/2)`` for ``tau`` (eq. 2)."""
    rho = np.asarray(rho, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    tau = 0.5 + mu / (rho * cs2)
    if np.any(tau <= 0.5):
        raise ValueError("computed relaxation time <= 1/2 (non-positive viscosity)")
    return tau


def pseudopotential_relaxation(
    phi,
    rho,
    mu_c: float,
    lam: float,
    cs2: float = CS2,
    scheme: str = "harmonic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Decoupled per-cell relaxation field for the two-phase engine.

    Returns ``(tau, mu)``: ``mu`` is the target dynamic viscosity everywhere and
    ``tau`` is the relaxation time realising it at the *local* density, so
    ``mu_d/mu_c = lambda`` holds for any ``rho_d/rho_c``.
    """
    mu = phase_viscosity(phi, mu_c, lam, scheme=scheme)
    tau = relaxation_time(rho, mu, cs2=cs2)
    return tau, mu


def legacy_uniform_relaxation(rho, mu_c: float, lam: float, cs2: float = CS2) -> np.ndarray:
    """The pre-fix (locked) mapping: a single constant ``tau`` for both phases.

    ``tau`` is chosen so that the *continuous* phase has viscosity ``mu_c``; the
    dispersed phase then automatically has ``mu_d/mu_c = rho_d/rho_c`` and
    ``lambda`` is not a free knob (gap G2).
    """
    rho = np.asarray(rho, dtype=np.float64)
    lam = _check_lam(lam)  # accepted and ignored on purpose: documents the lock
    nu_c = mu_c / float(np.amin(rho))
    return np.full(rho.shape, 0.5 + nu_c / cs2, dtype=np.float64)


def viscosity_ratio_from_tau(tau, rho, phi, cs2: float = CS2) -> float:
    """Measure the realised ``mu_d/mu_c`` of a ``(tau, rho, phi)`` triple."""
    tau = np.asarray(tau, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mu = rho * cs2 * (tau - 0.5)
    disp = mu[phi >= 0.999]
    cont = mu[phi <= 0.001]
    if disp.size == 0 or cont.size == 0:
        raise ValueError("both pure phases must be present in the mask")
    return float(np.mean(disp) / np.mean(cont))


# ---------------------------------------------------------------------------
# Lattice probe: measure the viscosity a tau field really produces
# ---------------------------------------------------------------------------
def _feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    cu = 3.0 * (D2Q9_C[:, 0, None, None] * ux + D2Q9_C[:, 1, None, None] * uy)
    usq = 1.5 * (ux * ux + uy * uy)
    return D2Q9_W[:, None, None] * rho[None, :, :] * (
        1.0 + cu + 0.5 * cu * cu - usq[None, :, :]
    )


def shear_wave_decay(
    tau_field,
    ny: int = 64,
    nx: int = 4,
    amp: float = 1e-3,
    k_mode: int = 1,
    steps: int = 400,
    sample_every: int = 10,
    band: Tuple[int, int] = (0, 0),
) -> Dict[str, object]:
    """Transverse-wave decay in a periodic D2Q9 BGK lattice.

    ``tau_field`` shape ``(ny,)`` -- one relaxation time per row.  Density is
    uniform ``rho = 1`` (no acoustics in the transverse mode), so the measured
    decay rate is ``nu k^2`` with ``nu = cs2 (tau - 1/2)`` per row.  If ``band``
    is a non-empty ``(j0, j1)`` slice, the amplitude is projected on that band
    only, which is how a two-phase viscosity contrast is probed without the
    interface contaminating the fit.
    """
    tau_field = np.asarray(tau_field, dtype=np.float64)
    if tau_field.shape != (ny,):
        raise ValueError("tau_field must have shape (ny,)")
    if np.any(tau_field <= 0.5):
        raise ValueError("tau_field must exceed 1/2")
    j0, j1 = band
    if j1 <= j0:
        j0, j1 = 0, ny
    y = np.arange(ny)
    k = 2.0 * np.pi * k_mode / ny
    rho = np.ones((nx, ny))
    ux = amp * np.cos(k * y)[None, :] * np.ones((nx, 1))
    uy = np.zeros((nx, ny))
    f = _feq(rho, ux, uy)
    tau = np.broadcast_to(tau_field[None, :], (nx, ny))
    probe = np.zeros(ny)
    probe[j0:j1] = np.cos(k * y[j0:j1])

    def amplitude() -> float:
        return float(2.0 * (ux * probe[None, :]).sum() / max(1, j1 - j0) / nx)

    series: List[float] = [amplitude()]
    times: List[int] = [0]
    for step in range(1, steps + 1):
        feq = _feq(rho, ux, uy)
        f = f + (feq - f) / tau[None, :, :]
        for i in range(9):
            f[i] = np.roll(np.roll(f[i], D2Q9_C[i, 0], axis=0), D2Q9_C[i, 1], axis=1)
        rho = f.sum(axis=0)
        ux = (f * D2Q9_C[:, 0, None, None]).sum(axis=0) / rho
        uy = (f * D2Q9_C[:, 1, None, None]).sum(axis=0) / rho
        if step % sample_every == 0:
            series.append(amplitude())
            times.append(step)

    amp_series = np.asarray(series, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    n = len(amp_series)
    lo, hi = max(1, n // 4), n - 1  # skip start-up, keep the settled exponential
    if hi - lo < 3:
        lo, hi = 1, n
    slope = np.polyfit(times_arr[lo:hi], np.log(amp_series[lo:hi]), 1)[0]
    nu = -slope / (k * k)
    return {
        "nu": float(nu),
        "decay_rate": float(-slope),
        "k": float(k),
        "amplitude": amp_series,
        "times": times_arr,
        "band": (j0, j1),
    }


def measure_uniform_viscosity(tau: float, ny: int = 64, steps: int = 800, sample_every: int = 20) -> float:
    """Lattice-measured kinematic viscosity of a uniform relaxation time.

    This is the calibration of eq. (1) on the real BGK kernel: the returned
    value should equal ``cs2 (tau - 1/2)``.  It is what makes the decoupling
    claim testable rather than an algebraic tautology -- ``mu = rho * nu`` with
    ``nu`` measured, and ``rho`` from the two-phase density field.
    """
    out = shear_wave_decay(
        np.full(ny, float(tau)), ny=ny, steps=steps, sample_every=sample_every
    )
    return float(out["nu"])
