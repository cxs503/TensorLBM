"""Quantitative verification of the 2-D acoustic pulse (cylindrical spreading).

Why this module exists
----------------------
``tensorlbm.acoustic_wave.run_2d_gaussian_pulse`` produced a 2-D Gaussian
pressure pulse that was only ever used as a *visualisation*: ``run_full_
verification`` stores ``pressure_2d_history`` and never compares it to a
reference.  The 1-D sinusoidal branch carries the quantitative gate, the
2-D branch carried none.  A pulse is however the physical content of the
"声脉冲传播 / acoustic pulse" case anchor, and it exercises physics that the
plane-wave benchmark cannot: two-dimensional (cylindrical) spreading, a
travelling wavefront and the ``r**-1/2`` envelope.

This module supplies the missing, *independent* reference and the metrics:

* ``exact_pulse_radial``   -- semi-analytic solution of the linearised 2-D wave
  equation obtained from the exact Hankel (Fourier-Bessel) representation.
  No LBM code path is used, so it is a genuine referee for the solver.
  It is itself validated two ways: it reproduces the initial Gaussian at
  ``t = 0`` (analytic identity) and it is cross-checked against a fully
  independent finite-difference solution of ``p_tt = c^2 lap(p)`` on a
  refined grid (``fd_wave_pulse``).
* ``simulate_pulse_2d``    -- D2Q9 / BGK run through the public library
  kernels (``solver.collide_bgk`` + ``solver.stream``), isothermal acoustics
  ``p = c_s^2 (rho - 1)``.
* ``travel_time_speed``    -- wavefront speed from cross-correlation travel
  times between annular probes (a time-domain instrument, independent of any
  spectral fit).
* ``envelope_exponent``    -- log-log slope of the peak amplitude vs radius.
* ``waveform_error``       -- relative L2 / max deviation of the radial
  waveform against the exact solution.

The *same* metric functions are applied to the exact solution, so every
reported number is a like-for-like comparison rather than a fitted constant.

Conventions: lattice units, ``c_s = 1/sqrt(3)``, ``nu = (tau - 1/2)/3``.
The initial condition is a static pressure perturbation
``p(r, 0) = A c_s^2 exp(-r^2 / (2 sigma^2))`` with ``p_t(r, 0) = 0``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream

CS = 1.0 / math.sqrt(3.0)

__all__ = [
    "CS",
    "gaussian_pressure",
    "exact_pulse_radial",
    "fd_wave_pulse",
    "radial_profile",
    "simulate_pulse_2d",
    "travel_time_speed",
    "envelope_exponent",
    "waveform_error",
    "anisotropy_residual",
]


# ---------------------------------------------------------------------------
# Bessel function J0 (avoid a hard scipy dependency)
# ---------------------------------------------------------------------------
def _bessel_j0(x: np.ndarray) -> np.ndarray:
    """Bessel J0 via scipy when available, else the NumPy polynomial fallback."""
    try:  # pragma: no cover - scipy is present in CI, kept optional on purpose
        from scipy.special import j0 as _j0

        return _j0(x)
    except Exception:  # pragma: no cover
        x = np.asarray(x, dtype=np.float64)
        ax = np.abs(x)
        out = np.empty_like(ax)
        small = ax < 8.0
        # Abramowitz & Stegun 9.4.1 / 9.4.3 style rational approximations
        y = ax[small] ** 2
        p1 = (
            -1.0 * y**6 * 1.0 / 147456.0
            + y**5 / 147456.0 * 1.0
        )  # low-order term only; replaced below
        # Chebyshev-like series (Numerical Recipes bessj0)
        y = ax[small] ** 2
        ans = (
            57568490574.0
            + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))
        ) / (
            57568490411.0
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))))
        )
        out[small] = ans
        del p1
        big = ~small
        if big.any():
            z = 8.0 / ax[big]
            y2 = z**2
            xx = ax[big] - 0.785398164
            ans1 = math.sqrt(0.636619772 / ax[big]) * (
                np.cos(xx)
                * (
                    1.0
                    + y2
                    * (
                        -0.1098628627e-2
                        + y2 * (0.2734510407e-4 + y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6))
                    )
                )
                - z
                * np.sin(xx)
                * (
                    -0.1562499995e-1
                    + y2 * (0.1430488765e-3 + y2 * (-0.6911147651e-5 + y2 * (0.7621095161e-6 - y2 * 0.934935152e-7)))
                )
            )
            out[big] = ans1
        return out


# ---------------------------------------------------------------------------
# Exact (semi-analytic) reference
# ---------------------------------------------------------------------------
def gaussian_pressure(nx: int, ny: int, sigma: float, amplitude: float, center=None) -> np.ndarray:
    """Initial pressure perturbation ``p = A c_s^2 exp(-r^2/2sigma^2)``."""
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    cx, cy = center
    y, x = np.meshgrid(np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing="ij")
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amplitude * CS**2 * np.exp(-r2 / (2.0 * sigma**2))


def exact_pulse_radial(
    r,
    t: float,
    sigma: float,
    amplitude: float,
    c: float = CS,
    phase_per_step: float = 0.02,
    n_k_max: int = 400000,
) -> np.ndarray:
    """Exact linear 2-D wave-equation solution at radius ``r`` and time ``t``.

    ``p(r, t) = int_0^inf P(k) J0(k r) cos(c k t) k dk`` with the Hankel
    transform of the Gaussian initial condition ``P(k) = A c_s^2 sigma^2
    exp(-k^2 sigma^2 / 2)`` (analytic: the Hankel transform of a Gaussian is a
    Gaussian).  The integral is evaluated by composite Simpson quadrature with
    the step chosen from the largest joint phase ``(r_max + c t) K``.
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    amp = amplitude * c**2 * sigma**2  # P(k) prefactor
    k_max = 8.0 / sigma  # Gaussian tail e^-32 ~ 1e-14
    r_max = float(np.max(r)) if r.size else 0.0
    # trapezoid on an analytic, exponentially decaying integrand: the joint
    # oscillation is bounded by (r_max + c t) * k_max, so resolve the phase.
    phase = (r_max + abs(c * t)) * k_max + 1.0
    dk = phase_per_step / phase
    n_k = int(math.ceil(k_max / dk))
    n_k = min(n_k, n_k_max)
    n_k = max(n_k, 200)
    k = np.linspace(0.0, k_max, n_k + 1)
    dk_eff = k[1] - k[0]
    base = amp * np.exp(-(k**2) * sigma**2 / 2.0) * np.cos(c * k * t) * k
    jac = _bessel_j0(np.outer(r, k))
    trap = np.full_like(k, dk_eff)
    trap[0] = trap[-1] = 0.5 * dk_eff
    val = jac @ (base * trap)
    # Richardson guard: one refinement level, used only as an error estimate
    k2 = np.linspace(0.0, k_max, 2 * n_k + 1)
    dk2 = k2[1] - k2[0]
    base2 = amp * np.exp(-(k2**2) * sigma**2 / 2.0) * np.cos(c * k2 * t) * k2
    trap2 = np.full_like(k2, dk2)
    trap2[0] = trap2[-1] = 0.5 * dk2
    val2 = _bessel_j0(np.outer(r, k2)) @ (base2 * trap2)
    exact_pulse_radial.last_rel_change = float(np.max(np.abs(val2 - val)) / max(1e-300, np.max(np.abs(val2))))
    return val2


def fd_wave_pulse(
    n: int = 257,
    sigma: float = 8.0,
    amplitude: float = 1.0,
    courant: float = 0.5,
    n_steps: int = 100,
) -> dict:
    """Independent second-order finite-difference solution of ``p_tt = c^2 lap p``.

    Used only to validate :func:`exact_pulse_radial` (a referee needs its own
    referee).  Units are chosen so that ``c = 1`` and the grid spacing is 1;
    compare with ``exact_pulse_radial(..., c=1.0)`` at ``t = courant*n_steps``.
    """
    if not 0.0 < courant <= 1.0 / math.sqrt(2.0):
        raise ValueError("courant must be in (0, 1/sqrt(2)] for the explicit scheme")
    c = 1.0
    amp = float(amplitude)
    y, x = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")
    ctr = (n - 1) / 2.0
    r2 = (x - ctr) ** 2 + (y - ctr) ** 2
    p0 = amp * np.exp(-r2 / (2.0 * sigma**2))
    lam2 = (courant) ** 2

    def lap(p):
        out = np.zeros_like(p)
        out[1:-1, 1:-1] = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * p[1:-1, 1:-1]
        return out

    p_prev = p0
    p = p0 + 0.5 * lam2 * lap(p0)  # p_t(0) = 0 -> symmetric first step
    for _ in range(n_steps - 1):
        p_next = 2.0 * p - p_prev + lam2 * lap(p)
        p_prev, p = p, p_next
    ygrid, xgrid = y, x
    r = np.sqrt((xgrid - ctr) ** 2 + (ygrid - ctr) ** 2)
    return {
        "field": p,
        "radius": r,
        "t": courant * n_steps,
        "c": c,
        "amplitude": amp,
        "sigma": sigma,
        "n": n,
        "courant": courant,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Radial profiling helpers
# ---------------------------------------------------------------------------
def radial_profile(field: np.ndarray, center=None, nbins=None, rmax=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthal mean of a 2-D field in unit-radius bins.

    Returns ``(r_centers, mean, spread)`` where ``spread`` is the max-minus-min
    azimuthal deviation inside each bin -- a direct measure of the lattice
    rotational-symmetry error (a continuum solution is exactly axisymmetric).
    """
    ny, nx = field.shape
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    cx, cy = center
    y, x = np.meshgrid(np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing="ij")
    r = np.hypot(x - cx, y - cy)
    if rmax is None:
        rmax = 0.5 * min(nx, ny)
    if nbins is None:
        nbins = int(math.floor(rmax)) + 1
    idx = np.floor(r).astype(np.int64)
    ok = (r < rmax) & (idx >= 0)
    r_c, mean, spread, count = [], [], [], []
    for b in range(nbins):
        sel = ok & (idx == b)
        if not sel.any():
            continue
        vals = field[sel]
        r_c.append(float(r[sel].mean()))
        mean.append(float(vals.mean()))
        spread.append(float(vals.max() - vals.min()))
        count.append(int(sel.sum()))
    return np.asarray(r_c), np.asarray(mean), np.asarray(spread)


def anisotropy_residual(field: np.ndarray, center=None, r_lo: float = 5.0, r_hi=None) -> float:
    """Max azimuthal spread relative to the radial peak amplitude (percent)."""
    r, mean, spread = radial_profile(field, center=center, rmax=r_hi)
    sel = (r >= r_lo) & (np.abs(mean) > 1e-30)
    if not sel.any():
        return float("nan")
    ref = float(np.max(np.abs(mean)))
    return float(100.0 * np.max(spread[sel & (np.abs(mean) > 1e-3 * ref)]) / ref)


# ---------------------------------------------------------------------------
# LBM simulation
# ---------------------------------------------------------------------------
def simulate_pulse_2d(
    nx: int = 384,
    ny: int = 384,
    tau: float = 0.52,
    n_steps: int = 180,
    sigma: float = 12.0,
    amplitude: float = 1e-4,
    sample_every: int = 10,
    probe_radii=(30.0, 50.0, 70.0, 90.0),
    snapshots_at=(),
    dtype=torch.float64,
    device=None,
    threads: int | None = 4,
) -> dict:
    """D2Q9/BGK 2-D Gaussian acoustic pulse through the public library kernels.

    Isothermal acoustics: ``p = c_s^2 (rho - 1)``.  The domain is periodic but
    the run must stop before the wavefront reaches the walls (asserted through
    ``front_radius`` in the returned metadata).
    """
    if device is None:
        device = torch.device("cpu")
    if threads:
        torch.set_num_threads(int(threads))
    centre = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    y, x = torch.meshgrid(
        torch.arange(ny, dtype=dtype, device=device),
        torch.arange(nx, dtype=dtype, device=device),
        indexing="ij",
    )
    r2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
    rho = 1.0 + amplitude * torch.exp(-r2 / (2.0 * sigma**2))
    zeros = torch.zeros_like(rho)
    f = equilibrium(rho, zeros, zeros, device=device)

    probe_r = list(probe_radii)
    np_r = np.asarray(probe_r, dtype=np.float64)
    times = [0]
    probe_series = [[0.0] * len(probe_r)]
    snaps = {}
    # initial probes
    p0 = (rho.cpu().numpy() - 1.0) * CS**2
    r_grid, mean0, _ = radial_profile(p0, center=centre, rmax=max(probe_r) + 2.0)
    for j, rr in enumerate(probe_r):
        k = int(np.argmin(np.abs(r_grid - rr)))
        probe_series[0][j] = float(mean0[k])

    for step in range(1, n_steps + 1):
        f = stream(collide_bgk(f, tau))
        if step in tuple(snapshots_at):
            rho_s, _, _ = macroscopic(f)
            snaps[int(step)] = (rho_s.cpu().numpy() - 1.0) * CS**2
        if step % sample_every == 0 or step == n_steps:
            rho_s, _, _ = macroscopic(f)
            p = (rho_s.cpu().numpy() - 1.0) * CS**2
            rg, mean, _ = radial_profile(p, center=centre, rmax=max(probe_r) + 2.0)
            times.append(step)
            probe_series.append(
                [float(mean[int(np.argmin(np.abs(rg - rr)))]) for rr in probe_r]
            )

    rho_f, ux_f, uy_f = macroscopic(f)
    p_final = (rho_f.cpu().numpy() - 1.0) * CS**2
    return {
        "pressure_final": p_final,
        "snapshots": snaps,
        "times": np.asarray(times, dtype=np.float64),
        "probe_radii": np_r,
        "probe_series": np.asarray(probe_series, dtype=np.float64),
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "sigma": sigma,
        "amplitude": amplitude,
        "n_steps": n_steps,
        "front_radius": CS * n_steps,
        "max_speed": float(np.max(np.hypot(ux_f.numpy(), uy_f.numpy()))),
        "finite": bool(torch.isfinite(rho_f).all()),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def travel_time_speed(times, series, radii) -> dict:
    """Wavefront speed from cross-correlation travel times between probes.

    ``series[i, j]`` is the signal at ``radii[j]`` and ``times[i]``.  For each
    probe pair the lag maximising the cross-correlation gives ``dt`` and hence
    ``c = (r_b - r_a) / dt``.  A time-domain instrument: no spectral fit.
    """
    times = np.asarray(times, dtype=np.float64)
    series = np.asarray(series, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    dt_sample = float(np.median(np.diff(times)))
    pairs = []
    for a in range(series.shape[1]):
        for b in range(a + 1, series.shape[1]):
            sa = series[:, a] - series[:, a].mean()
            sb = series[:, b] - series[:, b].mean()
            if not (np.any(sa) and np.any(sb)):
                continue
            full = np.correlate(sb, sa, mode="full")
            lags = np.arange(-len(sa) + 1, len(sa), dtype=np.float64)
            ipk = int(np.argmax(full))
            lag = lags[ipk]
            if 0 < ipk < len(full) - 1:  # parabolic sub-sample refinement
                y0, y1, y2 = full[ipk - 1], full[ipk], full[ipk + 1]
                denom = (y0 - 2.0 * y1 + y2)
                if denom != 0.0:
                    lag = lag + 0.5 * (y0 - y2) / denom
            if lag <= 0:
                continue
            dt = lag * dt_sample
            pairs.append(
                {
                    "r_a": float(radii[a]),
                    "r_b": float(radii[b]),
                    "lag_samples": float(lag),
                    "dt": float(dt),
                    "c": float((radii[b] - radii[a]) / dt),
                }
            )
    if not pairs:
        return {"c_meas": float("nan"), "pairs": []}
    cs = np.array([p["c"] for p in pairs], dtype=np.float64)
    return {
        "c_meas": float(cs.mean()),
        "c_spread_pct": float(100.0 * (cs.max() - cs.min()) / cs.mean()),
        "pairs": pairs,
    }


def envelope_exponent(r, peak, r_lo: float = 20.0, r_hi=None) -> dict:
    """Log-log slope of peak amplitude vs radius (2-D monopole: ``-1/2``)."""
    r = np.asarray(r, dtype=np.float64)
    peak = np.asarray(peak, dtype=np.float64)
    sel = (r >= r_lo) & (peak > 0.0)
    if r_hi is not None:
        sel &= r <= r_hi
    if sel.sum() < 3:
        return {"exponent": float("nan"), "n": int(sel.sum())}
    lx = np.log(r[sel])
    ly = np.log(peak[sel])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "exponent": float(slope),
        "prefactor": float(math.exp(intercept)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(sel.sum()),
    }


def waveform_error(p_num, p_ref, r=None, r_lo=5.0, r_hi=None) -> dict:
    """Relative L2 and max deviation of a radial waveform against the reference."""
    p_num = np.asarray(p_num, dtype=np.float64)
    p_ref = np.asarray(p_ref, dtype=np.float64)
    mask = np.ones_like(p_ref, dtype=bool)
    if r is not None:
        r = np.asarray(r, dtype=np.float64)
        mask = r >= r_lo
        if r_hi is not None:
            mask &= r <= r_hi
    a = p_num[mask]
    b = p_ref[mask]
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return {"l2_rel_pct": float("nan"), "max_rel_pct": float("nan"), "n": int(mask.sum())}
    peak = float(np.max(np.abs(b)))
    return {
        "l2_rel_pct": float(100.0 * np.linalg.norm(a - b) / denom),
        "max_rel_pct": float(100.0 * np.max(np.abs(a - b)) / peak),
        "peak_ref": peak,
        "peak_num": float(np.max(np.abs(a))),
        "n": int(mask.sum()),
    }