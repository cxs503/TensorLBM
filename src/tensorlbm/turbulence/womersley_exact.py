"""Exact Womersley solution for pulsatile (oscillatory pressure-driven) pipe flow.

The module provides the *analytic* solution of the unsteady Stokes problem for
fully developed flow in a circular pipe driven by a harmonic pressure gradient.
It replaces the boundary-layer approximation that has historically been used by
:func:`tensorlbm.turbulence.inlet_profiles.womersley_profile` (peak-relative
deviation of that approximation from the exact solution is 2 %–35 % over the
range Wo ∈ [1, 10]; see ``docs/womersley_exact_verification_20261108.md``).

Derivation and normalisation
----------------------------
With axial velocity :math:`u(r,t)` and pressure gradient
:math:`-\\partial p/\\partial x = G e^{i\\omega t}`, the momentum equation for a
fully developed pipe flow reduces to

.. math::

    i\\omega U = \\frac{G}{\\rho} + \\nu\\left(U'' + \\frac{1}{r}U'\\right),

whose regular solution is :math:`U(r) \\propto J_0(\\lambda r)` with
:math:`\\lambda^2 = i\\omega/\\nu`.  Non-dimensionalising with the pipe radius
:math:`R` and :math:`\\tilde r = r/R` gives the complex shape

.. math::

    \\hat W(\\tilde r) =
    \\frac{1 - J_0(\\alpha \\tilde r)/J_0(\\alpha)}
         {1 - 2 J_1(\\alpha)/\\left(\\alpha J_0(\\alpha)\\right)},
    \\qquad \\alpha = R\\sqrt{\\omega/\\nu}\\,\\sqrt{i} = \\mathrm{Wo}\\,e^{i\\pi/4}.

The denominator is fixed by **prescribing the instantaneous flow rate** rather
than the pressure gradient:
:math:`\\int_0^1 \\hat W(\\tilde r)\\,2\\tilde r\\,\\mathrm d\\tilde r \\equiv 1`,
so :math:`\\hat W` is the response to a unit *bulk velocity* (bulk = spatial
mean over the cross-section).  This is the form needed for an inlet boundary
condition, where the flow rate is known and the pressure gradient is not.

Consequently the profile with time-mean bulk velocity ``u_mean`` and unit
oscillatory bulk amplitude is

.. math::

    u(\\tilde r, \\phi) = 2 u_{mean}(1 - \\tilde r^2)
        + u_{mean}\\,\\mathrm{Re}\\left\\{\\hat W(\\tilde r)e^{i\\phi}\\right\\},

i.e. a Hagen–Poiseuille mean part plus the exact oscillatory part that carries
bulk flux ``u_mean cos φ``.

Physical limits used as verification anchors (all reproduced by
``tests/test_womersley_exact_verification.py``):

* ``Wo → 0``: :math:`\\hat W \\to 2(1-\\tilde r^2)(1 - i\\,\\mathrm{Wo}^2/6)`;
  the quasi-steady (parabolic, in-phase) profile is recovered with relative
  error ``Wo²/48`` — second order, with an *analytically predicted*
  coefficient.
* ``Wo → ∞``: the core is a plug moving in phase with the flow rate
  (:math:`\\hat W(0) \\to 1`) and the wall shear **leads** the flow rate
  by ``45°`` (:math:`\\arg\\,\\mathrm d\\hat W/\\mathrm d\\tilde r|_{\\tilde r=1}
  \\to +45°`) — the classical Womersley asymptote.
* Intermediate ``Wo``: the **centreline lags** the flow rate (up to ≈ −18.5°
  at Wo ≈ 5) while the wall shear leads it.  This sign structure is only
  correct with the ``α = Wo e^{-iπ/4}`` convention; using ``e^{+iπ/4}``
  conjugates the solution (identical amplitudes, every phase flipped) —
  the error was caught by the Bessel-free finite-difference check, see
  ``docs/womersley_exact_verification_20261108.md``.

Numerical notes
---------------
* Bessel functions of complex argument are taken from :mod:`scipy.special` when
  available; otherwise a self-contained power-series implementation
  (:func:`_j0_j1_series`) is used.  The series loses accuracy through
  cancellation for ``Wo ≳ 50`` (≈1e-10 at Wo = 50, ≈1e-4 at Wo = 100).
* ``|J0(α)|`` grows exponentially with ``Wo`` while ``|J0(α r̃)|`` grows more
  slowly, so the *ratio* ``J0(αr̃)/J0(α)`` is well scaled; the only unstable
  operation is ``1 - J0(αr̃)/J0(α)`` for ``r̃ → 1``, which is evaluated with a
  two-term Taylor expansion about ``r̃ = 1`` when ``|α|(1-r̃) < 1e-5``.  The
  crossover is where the two error sources balance (cancellation
  ``≈2e-16/|α|(1-r̃)`` vs truncation ``≈0.3(|α|(1-r̃))²``; both ≈1e-11 there).
  A wide Taylor window (e.g. ``|α|(1-r̃) < 0.5``) is *wrong* at the percent level
  and was caught by the flux-identity check.
* Validity range: ``1e-3 ≤ Wo ≤ 100`` (``WO_MAX_TRUSTED``).  Nothing here
  depends on a lattice; the output is in the same units as the input velocities.

References
----------
Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow
    and viscous drag in arteries when the pressure gradient is known.
    *J. Physiol.* 127, 553–563.
Nichols, W. W. & O'Rourke, M. F. (2005). *McDonald's Blood Flow in Arteries*,
    5th ed., Hodder Arnold — ch. 2 (Womersley solution and its asymptotes).
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "WO_MAX_TRUSTED",
    "womersley_complex_profile",
    "womersley_diagnostics",
    "womersley_profile_exact",
]

#: Largest Womersley number for which the implementation is verified
#: (see module docstring "Numerical notes").
WO_MAX_TRUSTED = 100.0

# With the e^{+i omega t} convention the regular radial eigenvalue is
# k^2 = -i omega / nu, hence alpha = R k = Wo * sqrt(-i) = Wo e^{-i pi/4}.
# (Using sqrt(+i) instead conjugates the whole solution: amplitudes are
# unchanged but every phase lag changes sign -- caught by the Bessel-free
# finite-difference check in artifacts/womersley_exact_20261108.)
_SQRT_MINUS_I = complex(1.0, -1.0) / math.sqrt(2.0)  # e^{-i pi/4}

# numpy >= 2 renamed trapz -> trapezoid; keep both working.
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ---------------------------------------------------------------------------
# Bessel functions of complex argument
# ---------------------------------------------------------------------------


def _j0_j1_series(z: complex, tol: float = 1e-16, max_terms: int = 2000):
    """Complex ``(J0, J1)`` from the ascending power series.

    Self-contained fallback used when :mod:`scipy` is unavailable.  Accuracy is
    limited by cancellation for large ``|z|`` (see module docstring).
    """
    h = complex(z) / 2.0
    term0 = 1.0 + 0.0j
    sum0 = term0
    term1 = h
    sum1 = term1
    for m in range(1, max_terms + 1):
        term0 = -term0 * h * h / (m * m)
        sum0 += term0
        term1 = -term1 * h * h / (m * (m + 1.0))
        sum1 += term1
        if abs(term0) < tol * abs(sum0) and abs(term1) < tol * abs(sum1):
            return sum0, sum1
    return sum0, sum1


def _scipy_available() -> bool:
    try:
        import scipy.special  # noqa: F401, PLC0415
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


def _j0_j1(z: complex):
    """Complex ``(J0, J1)`` — scipy when importable, power series otherwise."""
    if _scipy_available():
        from scipy.special import jv  # noqa: PLC0415

        z = complex(z)
        return complex(jv(0, z)), complex(jv(1, z))
    return _j0_j1_series(z)


def _j0_array(z: np.ndarray) -> np.ndarray:
    """Vectorised complex ``J0`` over an array (scipy, else per-element series)."""
    z = np.asarray(z, dtype=complex)
    if _scipy_available():
        from scipy.special import jv  # noqa: PLC0415

        return np.asarray(jv(0, z), dtype=complex)
    flat = z.ravel()
    out = np.array([_j0_j1_series(complex(v))[0] for v in flat], dtype=complex)
    return out.reshape(z.shape)


# ---------------------------------------------------------------------------
# Complex shape  W-hat(r/R)
# ---------------------------------------------------------------------------


def _alpha(wo: float, harmonic: int = 1) -> complex:
    """Womersley parameter ``α = Wo √k e^{-iπ/4}`` for harmonic *harmonic* = k."""
    return float(wo) * math.sqrt(float(harmonic)) * _SQRT_MINUS_I



def womersley_complex_profile(rr, wo: float, harmonic: int = 1) -> np.ndarray:
    """Complex, flux-normalised Womersley shape :math:`\\hat W(\\tilde r)`.

    Args:
        rr:       Radial stations :math:`\\tilde r = r/R \\in [0, 1]`
                  (scalar or array-like).  ``0`` = centreline, ``1`` = wall.
        wo:       Womersley number :math:`\\mathrm{Wo} = R\\sqrt{\\omega/\\nu}`
                  of the *fundamental*.
        harmonic: Harmonic index ``k`` (default 1 = fundamental).  The
                  parameter used is ``Wo*sqrt(k)``, i.e. frequency ``k*omega``.

    Returns:
        Complex :class:`numpy.ndarray` with the same shape as ``rr``.  Its
        cross-section integral :math:`\\int_0^1 \\hat W 2\\tilde r\\,d\\tilde r`
        equals ``1`` for every ``wo`` (unit bulk velocity).
    """
    rr_arr = np.atleast_1d(np.asarray(rr, dtype=float))
    if np.any(~np.isfinite(rr_arr)):
        raise ValueError("rr must be finite")
    if np.any(rr_arr < 0.0) or np.any(rr_arr > 1.0):
        raise ValueError("rr must lie in [0, 1] (r/R)")
    wo = float(wo)
    if not math.isfinite(wo) or wo <= 0.0:
        raise ValueError(f"Womersley number must be positive and finite, got {wo!r}")
    if wo > 10.0 * WO_MAX_TRUSTED:  # pragma: no cover - guard rail
        raise ValueError(
            f"Wo={wo} exceeds the verified range (Wo <= {WO_MAX_TRUSTED}); "
            "the Bessel ratio is numerically unrepresentable far beyond it"
        )

    if wo < 1e-3:
        # Quasi-steady limit: W-hat -> 2(1 - r^2) (see module docstring).
        return (2.0 * (1.0 - rr_arr**2)).astype(complex)

    a = _alpha(wo, harmonic)
    j0a, j1a = _j0_j1(a)
    g1 = j1a / j0a
    g = j1a / (a * j0a)
    denom = 1.0 - 2.0 * g

    eps = 1.0 - rr_arr
    # Crossover chosen where the two error sources balance: the direct form loses
    # ~2e-16/(eps*|a|) to cancellation, the 2-term Taylor expansion truncates at
    # ~0.3*(eps*|a|)^2.  Both are ~1e-11 at eps*|a| = 1e-5.
    near = np.abs(a) * eps < 1e-5
    out = np.empty(rr_arr.shape, dtype=complex)
    if np.any(~near):
        r_far = rr_arr[~near]
        out[~near] = (1.0 - _j0_array(a * r_far) / j0a) / denom
    if np.any(near):
        e = eps[near]
        out[near] = (-e * a * g1 + 0.5 * e * e * a * a * (1.0 - g)) / denom
    return out


# ---------------------------------------------------------------------------
# Diagnostics (classic Womersley amplitude / phase relations)
# ---------------------------------------------------------------------------


def womersley_diagnostics(wo: float, harmonic: int = 1) -> dict:
    """Amplitude and phase of the classic Womersley quantities.

    All quantities are relative to the *flow rate* (which is by construction
    the phase reference of this normalisation), so the centreline and wall
    shear phases are directly comparable to the textbook Womersley diagrams.

    Returns:
        dict with keys ``wo``, ``harmonic``, ``flux`` (≡1, numerical check),
        ``centerline_amplitude``, ``centerline_phase_deg``,
        ``wall_shear_amplitude``, ``wall_shear_phase_deg``.

    Notes:
        * ``centerline_*`` is :math:`\\hat W(0)`: amplitude →1 and phase →0 as
          ``Wo → ∞`` (core moves as a plug in phase with the flow rate).
        * ``wall_shear_*`` is the derivative :math:`d\\hat W/d\\tilde r` at :math:`\\tilde r=1`
          (sign included, hence proportional to the wall shear stress on the
          outward normal): amplitude grows ∝ Wo and the phase tends to
          **+45°** as ``Wo → ∞`` — the classical Womersley result that the wall
          shear *leads* the flow rate by 45° at high Wo.
        * The centreline phase is **negative** at intermediate ``Wo`` (about
          −18.5° near Wo = 5): the core lags the flow rate while the Stokes
          layer leads it.
    """
    wo = float(wo)
    if not math.isfinite(wo) or wo <= 0.0:
        raise ValueError(f"Womersley number must be positive and finite, got {wo!r}")
    if wo < 1e-3:
        w0 = 2.0 + 0.0j
        dw = 0.0 + 0.0j
    else:
        a = _alpha(wo, harmonic)
        j0a, j1a = _j0_j1(a)
        g1 = j1a / j0a
        g = j1a / (a * j0a)
        denom = 1.0 - 2.0 * g
        w0 = (1.0 - 1.0 / j0a) / denom
        dw = -a * g1 / denom  # d/dr~ [ (1 - J0(a r~)/J0(a)) ] / D  at r~ = 1
    grid = np.linspace(0.0, 1.0, 200001)
    flux = _TRAPZ(womersley_complex_profile(grid, wo, harmonic) * 2.0 * grid, grid)
    return {
        "wo": wo,
        "harmonic": int(harmonic),
        "flux": float(flux.real),
        "centerline_amplitude": float(abs(w0)),
        "centerline_phase_deg": float(math.degrees(np.angle(w0))),
        "wall_shear_amplitude": float(abs(dw)),
        "wall_shear_phase_deg": float(math.degrees(np.angle(dw))),
    }


# ---------------------------------------------------------------------------
# Torch-facing profiles (drop-in companion of inlet_profiles.womersley_profile)
# ---------------------------------------------------------------------------


def womersley_profile_exact(
    n: int,
    u_mean: float,
    wo: float,
    phase: float = 0.0,
    n_harmonics: int = 4,
    amplitude: float = 1.0,
    clamp_nonneg: bool = False,
    dtype: torch.dtype | None = None,
    device=None,
) -> torch.Tensor:
    """Instantaneous **exact** Womersley velocity profile across a diameter.

    Same calling convention as
    :func:`tensorlbm.turbulence.inlet_profiles.womersley_profile` (``n`` cell
    centres across the diameter, ``(n,)`` tensor), but the oscillatory part is
    the analytic Bessel solution instead of a Stokes-layer approximation.

    Args:
        n:           Number of cells across the diameter.
        u_mean:      Time-averaged bulk velocity (the mean part is exactly the
                     Hagen–Poiseuille profile with bulk ``u_mean``).
        wo:          Womersley number of the fundamental (``1e-3 … 100``).
        phase:       Phase angle ``ωt`` in radians.
        n_harmonics: Number of harmonics ``k = 1 … n_harmonics``; harmonic ``k``
                     has frequency ``kω`` and bulk amplitude ``amplitude/k²``.
        amplitude:   Bulk amplitude of the fundamental oscillation, in units of
                     ``u_mean`` (0.5 → the bulk velocity oscillates between
                     0.5 and 1.5 times the mean).
        clamp_nonneg: Clip negative values at zero (default ``False``: the
                     analytic profile *may* reverse near the wall at large
                     ``phase`` amplitude — that backflow is physical and
                     clipping it destroys the analytic property).
        dtype:       Torch dtype of the result (default: default float dtype).
        device:      Torch device of the result.

    Returns:
        Tensor of shape ``(n,)`` in the same units as ``u_mean``.

    Raises:
        ValueError: on ``n < 3``, non-positive ``wo``, or ``n_harmonics < 1``.
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"n must be >= 3 grid cells, got {n}")
    if int(n_harmonics) < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
    if not math.isfinite(float(wo)) or float(wo) <= 0.0:
        raise ValueError(f"Womersley number must be positive and finite, got {wo!r}")

    y = np.linspace(0.5, n - 0.5, n)
    rr = np.abs(2.0 * y / n - 1.0)

    u = 2.0 * float(u_mean) * (1.0 - rr * rr)
    for k in range(1, int(n_harmonics) + 1):
        amp_k = float(amplitude) / (k * k)
        shape = womersley_complex_profile(rr, wo, harmonic=k)
        u = u + float(u_mean) * amp_k * np.real(shape * np.exp(1j * k * float(phase)))

    out = torch.as_tensor(u, dtype=dtype if dtype is not None else torch.get_default_dtype())
    if device is not None:
        out = out.to(device)
    if clamp_nonneg:
        out = out.clamp(min=0.0)
    return out