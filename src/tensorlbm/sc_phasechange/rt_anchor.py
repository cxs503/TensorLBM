"""Rayleigh-Taylor linear-growth anchor on top of :mod:`tensorlbm.rayleigh_taylor`.

Physical problem
----------------
Heavy fluid over light fluid in gravity, single-mode interface perturbation of
wavelength ``lam`` and amplitude ``a(t)``.  In the linear regime

.. math::

    a(t) = a_0 e^{\\sigma t}, \\qquad
    \\sigma = \\sqrt{A g k}\\quad(\\text{inviscid}),
    \\qquad
    \\sigma = \\sqrt{A g k + \\nu^2 k^4} - \\nu k^2\\quad(\\text{equal-} \\nu),

with ``A = (rho_h - rho_l)/(rho_h + rho_l)`` the Atwood number and
``k = 2 pi / lam`` (Rayleigh 1883; Taylor 1950; Chandrasekhar 1961 §92).

Why this anchor is not run on the Shan-Chen pseudopotential
-----------------------------------------------------------
The campaign archive ``benchmarks/pending/rayleigh_taylor`` (2026-09-19)
scanned the SCMP parameter surface with gravity and found **no** growth window:
the gravity ceiling set by the vapour spinodal (``g_ceiling = 7.1e-6`` at
``lam = 280``) sits below the capillary cutoff, and all 10 cases returned
``sigma_sim <= 0`` or ``NaN``.  The anchor therefore uses the Boussinesq
two-layer LBM (one buoyancy force per unit mass ``F_y = -A g phi``, mean
hydrostatic part removed), which is the same sharp-interface RT problem in the
limit where the density *difference* drives the flow but the *mean* density
stays at ``rho_0 = 1``.  That keeps the Atwood sweep (0.1 / 0.33 / 0.5)
physically meaningful at ``Ma < 0.01``.

Diagnostics
-----------
The interface is tracked by the conserved heavy-fluid column height
``N(x) = sum_y (1 + phi)/2`` and its first Fourier mode -- no thresholding, so
the measurement is noise-free and fully vectorised (the scalar per-column
zero-crossing search in the library is O(ny * nx) python and costs ~50 ms per
sample at ``lam = 64``, which dominates the runtime of a long run).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..rayleigh_taylor import (
    RTConfig,
    analytic_growth_rate,
    init_rt_state,
    rt_step,
    viscous_growth_rate,
)

__all__ = ["RTAnchorCase", "run_rt_case", "run_rt_atwood_sweep"]


@dataclass
class RTAnchorCase:
    atwood: float
    wavelength: int
    gravity: float
    sigma_sim: float
    sigma_analytic: float
    sigma_viscous: float
    rel_error_pct: float
    rel_error_viscous_pct: float
    fit_r2: float
    steps_run: int
    fit_window: tuple[float, float] = (0.0, 0.0)
    samples: list[tuple[float, float]] = field(default_factory=list)


def _mode_amplitude(phi: torch.Tensor, lam: int) -> float:
    """Magnitude of the first Fourier mode of the heavy-fluid column height.

    ``N(x) = sum_y (1+phi)/2`` is the heavy-fluid column height, a conserved
    moment of the phase field (no thresholding).  For ``phi = tanh((y-y_int)/d)``
    with the heavy fluid on top, ``N = n_y - y_int``, so a *positive* crest of
    ``y_int`` shows up as a negative Fourier coefficient: the magnitude is
    returned, the phase (sign) carries no growth information.
    """
    ny, nx = phi.shape
    column = 0.5 * (1.0 + phi).sum(dim=0)
    fluc = column - column.mean()
    x = torch.arange(nx, dtype=column.dtype, device=column.device)
    basis = torch.cos(2.0 * math.pi * x / float(lam))
    return float(abs((2.0 / nx) * (fluc * basis).sum()))


def run_rt_case(
    atwood: float,
    lam: int = 64,
    gravity: float = 2.0e-3,
    n_steps: int = 6000,
    sample_every: int = 10,
    layers: float = 3.0,
    a_lo_frac: float = 0.01,
    a_hi_frac: float = 0.06,
    stop_a_frac: float = 0.16,
    device: str = "cpu",
) -> RTAnchorCase:
    """Run one single-mode RT case and fit the exponential growth rate."""
    cfg = RTConfig(
        lam=lam,
        atwood=float(atwood),
        g=float(gravity),
        layers=layers,
        n_steps=n_steps,
        sample_every=sample_every,
        stop_a_frac=stop_a_frac,
        device=device,
    )
    cfg.validate()
    torch.set_num_threads(1)
    f, g, phi = init_rt_state(cfg)

    times: list[float] = []
    amps: list[float] = []
    steps = 0
    for step in range(1, n_steps + 1):
        f, g = rt_step(f, g, cfg)
        steps = step
        if step % sample_every == 0:
            a = _mode_amplitude(g.sum(dim=0), lam)
            times.append(float(step))
            amps.append(a)
            if a > stop_a_frac * lam:
                break

    amp = np.array(amps)
    t = np.array(times)
    a_lo = a_lo_frac * lam
    a_hi = a_hi_frac * lam
    win = (amp >= a_lo) & (amp <= a_hi)
    if win.sum() < 3:
        # fall back to a fixed fraction of the sampled range above the noise floor
        start = int(np.argmax(amp > a_lo)) if np.any(amp > a_lo) else 0
        win = np.zeros_like(amp, dtype=bool)
        win[start : start + max(3, len(amp) // 4)] = True
    tt = t[win]
    ll = np.log(np.clip(amp[win], 1e-30, None))
    slope, intercept = np.polyfit(tt, ll, 1)
    pred = slope * tt + intercept
    ss_res = float(((ll - pred) ** 2).sum())
    ss_tot = float(((ll - ll.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    sigma_a = analytic_growth_rate(atwood, gravity, lam)
    sigma_v = viscous_growth_rate(atwood, gravity, lam, cfg.nu)
    return RTAnchorCase(
        atwood=float(atwood),
        wavelength=int(lam),
        gravity=float(gravity),
        sigma_sim=float(slope),
        sigma_analytic=float(sigma_a),
        sigma_viscous=float(sigma_v),
        rel_error_pct=100.0 * (float(slope) - sigma_a) / sigma_a,
        rel_error_viscous_pct=100.0 * (float(slope) - sigma_v) / sigma_v,
        fit_r2=float(r2),
        steps_run=int(steps),
        fit_window=(float(tt[0]), float(tt[-1])) if tt.size else (0.0, 0.0),
        samples=[(float(a), float(b)) for a, b in zip(times, amps)],
    )


def run_rt_atwood_sweep(
    atwoods: tuple[float, ...] = (0.1, 1.0 / 3.0, 0.5),
    lam: int = 64,
    gravity: float = 2.0e-3,
    n_steps: int = 6000,
    progress=None,
) -> dict:
    """Growth-rate anchor for several Atwood numbers at fixed wavelength."""
    cases = []
    for at in atwoods:
        case = run_rt_case(at, lam=lam, gravity=gravity, n_steps=n_steps)
        cases.append(case)
        if progress:
            progress(case)
    return dict(
        wavelength=lam,
        gravity=gravity,
        atwoods=list(atwoods),
        cases=[c.__dict__ for c in cases],
        max_abs_error_pct=max(abs(c.rel_error_pct) for c in cases),
        all_fits_ok=all(c.fit_r2 > 0.98 for c in cases),
    )
