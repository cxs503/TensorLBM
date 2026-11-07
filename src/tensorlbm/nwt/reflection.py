"""Wave reflection analysis for the numerical wave tank.

Primary estimator: the **two-point method of Goda & Suzuki (1976)** — the
standard technique used to separate incident and reflected wave trains from two
simultaneous surface-elevation records.  A least-squares extension over an
arbitrary number of probes (Goda 2000; Mansard & Funke 1980) is provided as an
independent cross-check.

Signal model
------------
With ``x`` measured in the direction of propagation and the probes at ``x_j``,

.. math::

    \\eta(x_j,t) = a_i \\cos(\\omega t - k x_j + \\phi_i)
                 + a_r \\cos(\\omega t + k x_j + \\phi_r)

i.e. in complex form, using :math:`\\eta = \\Re\\{Z_j e^{i\\omega t}\\}`,

.. math::

    Z_j = A_i e^{-i k x_j} + A_r e^{i k x_j},

where ``A_i`` and ``A_r`` are the (complex) incident and reflected amplitudes.
For two probes separated by :math:`\\Delta x = x_2 - x_1` the 2x2 system is
solved in closed form (Goda & Suzuki 1976):

.. math::

    A_i = \\frac{Z_1 e^{i k \\Delta x} - Z_2}{2 i \\sin k\\Delta x},
    \\quad
    A_r = \\frac{Z_2 - Z_1 e^{-i k \\Delta x}}{2 i \\sin k\\Delta x}

and the reflection coefficient is :math:`K_R = |A_r| / |A_i|`.

The pair is ill-conditioned when :math:`\\sin k\\Delta x \\to 0` (i.e.
``Delta x -> m lambda / 2``); :func:`two_point_reflection` reports the
conditioning factor so a caller can reject a badly placed pair instead of
publishing a garbage ``K_R``.
"""

from __future__ import annotations

import cmath
import math

import torch

__all__ = [
    "complex_amplitude",
    "two_point_reflection",
    "least_squares_reflection",
    "wavenumber_by_model_fit",
]


def _as_float_series(eta) -> torch.Tensor:
    if isinstance(eta, torch.Tensor):
        return eta.reshape(-1).to(torch.float64)
    return torch.as_tensor(eta, dtype=torch.float64).reshape(-1)


def complex_amplitude(eta, dt: float, omega: float, with_mean: bool = True) -> complex:
    """Least-squares complex amplitude ``Z`` with ``eta ~ Re{Z e^{i omega t}}``.

    Fits ``c cos(omega t) + s sin(omega t)`` (optionally plus a constant over
    the record) by ordinary least squares, so the record length need not be an
    integer number of periods — this removes spectral leakage at the harmonic of
    interest.  Returns ``Z = c - i s``.
    """
    y = _as_float_series(eta)
    n = y.numel()
    t = torch.arange(n, dtype=torch.float64) * dt
    cols = [torch.cos(omega * t), torch.sin(omega * t)]
    if with_mean:
        cols.append(torch.ones_like(t))
    A = torch.stack(cols, dim=1)
    coef, *_ = torch.linalg.lstsq(A, y.unsqueeze(1))
    c, s = float(coef[0, 0]), float(coef[1, 0])
    return complex(c, -s)


def _pair_solution(Z1: complex, Z2: complex, k: float, dx: float) -> tuple[complex, complex, float]:
    ph = k * dx
    sinph = math.sin(ph)
    if abs(sinph) < 1e-9:
        raise ValueError("degenerate probe spacing: sin(k dx) = 0")
    eip = cmath.exp(1j * ph)
    Ai = (Z1 * eip - Z2) / (2j * sinph)
    Ar = (Z2 - Z1 * eip.conjugate()) / (2j * sinph)
    return Ai, Ar, abs(sinph)


def two_point_reflection(
    eta1,
    eta2,
    x1: float,
    x2: float,
    dt: float,
    omega: float,
    k: float,
) -> dict:
    """Goda & Suzuki (1976) two-point incident/reflected separation.

    Args:
        eta1, eta2: surface-elevation records at the two probes (same length).
        x1, x2: probe positions (same length unit as ``k``).
        dt: sampling interval.
        omega: angular frequency of the analysed component.
        k: wave number of the analysed component (from the NWT dispersion
            measurement; sign gives the propagation direction).

    Returns:
        dict with ``a_incident``, ``a_reflected``, ``KR``, ``condition``
        (``|sin k dx|`` — reject pairs below ~0.1) and the complex split.
    """
    Z1 = complex_amplitude(eta1, dt, omega)
    Z2 = complex_amplitude(eta2, dt, omega)
    Ai, Ar, cond = _pair_solution(Z1, Z2, k, x2 - x1)
    return {
        "a_incident": abs(Ai),
        "a_reflected": abs(Ar),
        "KR": abs(Ar) / abs(Ai) if abs(Ai) > 0 else float("inf"),
        "condition": cond,
        "A_incident": Ai,
        "A_reflected": Ar,
        "phase_incident": cmath.phase(Ai),
        "phase_reflected": cmath.phase(Ar),
    }


def least_squares_reflection(
    eta_probes,
    x_probes,
    dt: float,
    omega: float,
    k: float,
) -> dict:
    """Least-squares incident/reflected separation over an arbitrary probe set.

    Solves ``Z_j = A_i e^{-i k x_j} + A_r e^{i k x_j}`` in the least-squares
    sense (real normal equations on the 4 unknowns ``Re/Im`` of ``A_i, A_r``).
    With more than two probes this is over-determined and insensitive to a
    single bad probe; it is used as an independent cross-check of
    :func:`two_point_reflection`.
    """
    x = [float(v) for v in x_probes]
    Z = [complex_amplitude(e, dt, omega) for e in eta_probes]
    rows: list[list[float]] = []
    rhs: list[float] = []
    for xj, zj in zip(x, Z):
        ca, sa = math.cos(-k * xj), math.sin(-k * xj)
        cb, sb = math.cos(k * xj), math.sin(k * xj)
        # Re{ (Ai_r + i Ai_i)(ca + i sa) + (Ar_r + i Ar_i)(cb + i sb) }
        rows.append([ca, -sa, cb, -sb])
        rhs.append(zj.real)
        # Im{ ... }
        rows.append([sa, ca, sb, cb])
        rhs.append(zj.imag)
    M = torch.tensor(rows, dtype=torch.float64)
    b = torch.tensor(rhs, dtype=torch.float64)
    sol, *_ = torch.linalg.lstsq(M, b.unsqueeze(1))
    Ai = complex(float(sol[0, 0]), float(sol[1, 0]))
    Ar = complex(float(sol[2, 0]), float(sol[3, 0]))
    resid = float((M @ sol.squeeze(1) - b).abs().max())
    return {
        "a_incident": abs(Ai),
        "a_reflected": abs(Ar),
        "KR": abs(Ar) / abs(Ai) if abs(Ai) > 0 else float("inf"),
        "A_incident": Ai,
        "A_reflected": Ar,
        "max_residual": resid,
    }


def wavenumber_by_model_fit(
    eta_probes,
    x_probes,
    dt: float,
    omega: float,
    k_guess: float,
    span: float = 0.15,
    n_scan: int = 241,
    n_refine: int = 40,
) -> dict:
    """Wave number by fitting the full incident+reflected model to all probes.

    With ``N >= 3`` probes the two-parameter model
    ``Z_j = A_i exp(-i k x_j) + A_r exp(+i k x_j)`` is over-determined, so the
    residual of the complex least-squares fit depends on ``k``.  Minimising that
    residual over ``k`` yields a wave-number estimate that is **immune to the
    standing-wave distortion** that biases a raw phase regression: the reflected
    train is a free parameter of the model, not a contaminant.

    A coarse scan over ``k_guess (1 +/- span)`` locates the global minimum and a
    parabolic refinement on the final grid polishes it.  Returns the wave number,
    the residual, the separated amplitudes and the residual at the coarse guess
    (a model-mismatch diagnostic: harmonics, noise or a non-planar wave field all
    inflate it).
    """
    x = [float(v) for v in x_probes]
    Z = [complex_amplitude(e, dt, omega) for e in eta_probes]
    if len(x) < 3:
        raise ValueError("model fitting needs at least three probes")

    def residual(k: float) -> tuple[float, complex, complex]:
        rows: list[list[float]] = []
        rhs: list[float] = []
        for xj, zj in zip(x, Z):
            ca, sa = math.cos(-k * xj), math.sin(-k * xj)
            cb, sb = math.cos(k * xj), math.sin(k * xj)
            rows.append([ca, -sa, cb, -sb])
            rhs.append(zj.real)
            rows.append([sa, ca, sb, cb])
            rhs.append(zj.imag)
        M = torch.tensor(rows, dtype=torch.float64)
        b = torch.tensor(rhs, dtype=torch.float64)
        sol, *_ = torch.linalg.lstsq(M, b.unsqueeze(1))
        res = float(torch.sqrt(((M @ sol.squeeze(1) - b) ** 2).mean()))
        return res, complex(float(sol[0, 0]), float(sol[1, 0])), complex(float(sol[2, 0]), float(sol[3, 0]))

    lo, hi = k_guess * (1.0 - span), k_guess * (1.0 + span)
    ks = [lo + (hi - lo) * i / (n_scan - 1) for i in range(n_scan)]
    res = [residual(k)[0] for k in ks]
    i_best = int(min(range(n_scan), key=lambda i: res[i]))
    a, b, c = (res[max(i_best - 1, 0)], res[i_best], res[min(i_best + 1, n_scan - 1)])
    dk = (ks[1] - ks[0]) if n_scan > 1 else 0.0
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom > 1e-30 else 0.0
    k_best = ks[i_best] + delta * dk
    # golden-section style local refine around the parabola vertex
    width = 2.0 * dk
    left, right = k_best - width, k_best + width
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c1 = right - gr * (right - left)
    c2 = left + gr * (right - left)
    f1, f2 = residual(c1)[0], residual(c2)[0]
    for _ in range(n_refine):
        if f1 < f2:
            right, c2, f2 = c2, c1, f1
            c1 = right - gr * (right - left)
            f1 = residual(c1)[0]
        else:
            left, c1, f1 = c1, c2, f2
            c2 = left + gr * (right - left)
            f2 = residual(c2)[0]
    k_ref = 0.5 * (left + right)
    res_ref, Ai, Ar = residual(k_ref)
    res0, Ai0, _ = residual(k_guess)
    return {
        "k_measured": k_ref,
        "residual": res_ref,
        "residual_at_guess": res0,
        "a_incident": abs(Ai),
        "a_reflected": abs(Ar),
        "KR": abs(Ar) / abs(Ai) if abs(Ai) > 0 else float("inf"),
        "A_incident": Ai,
        "A_reflected": Ar,
        "scan_min_k": ks[i_best],
    }
