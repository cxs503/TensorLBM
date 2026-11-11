"""Standing capillary wave: ``omega(k)`` and ``gamma(k)`` against the classical solution.

Physics
-------
Two fluid layers of equal density and viscosity, no gravity, a small-amplitude
standing wave of the interface ``y_i(x) = y0 + A cos(kx)`` (Kelvin 1871;
Landau & Lifshitz, *Fluid Mechanics* 2nd ed. §62):

    omega^2 = sigma k^3 tanh(k H) / (rho_l + rho_g)                       (1)
    gamma   = 2 nu k^2                                                   (2)

with ``H`` the depth of *one* layer (distance from the interface to the symmetry
plane, here ``ny/2`` because the box is periodic in ``y``): the ``tanh(kH)``
factor is the finite-depth correction, which reduces to the deep-water form
``omega^2 = sigma k^3/(rho_l+rho_g)`` for ``kH >> 1``.  Both are closed form in
parameters that enter the *model input* (``sigma``, ``nu``, ``rho``); there is
nothing to tune, so this is a blind-prediction test of the capillary force chain.

Why the measurement is not a single damped-sinusoid fit
------------------------------------------------------
The impulsive initial state (a displaced ``tanh`` interface, zero velocity) is
not exactly an eigenstate of the discrete operator: the discrete interface
relaxes its shape over the first ~``1/3`` of a period while a second
(interface-preserving) flow mode is excited, so a single damped sinusoid fitted
over the whole record is biased.  The frequency is therefore extracted three
independent ways and *all* are reported:

* ``omega_extrema`` (primary) -- from the spacing of successive extrema of
  ``a(t)``.  For ``a = A e^{-gamma t} cos(omega t + phi)`` the extrema sit at
  ``omega t + phi = n pi``, so the spacing is *exactly* ``pi/omega`` for any
  damping -- this is the only estimator that stays unbiased when the record is
  damped as strongly as a lattice capillary wave is;
* ``omega_cross`` -- from the late-time zero-crossing intervals of ``a(t)``
  (no functional form assumed, only periodicity);
* windowed damped-sinusoid fits (``early`` / ``mid`` / ``late``), which also
  give ``gamma`` and ``R^2``;
* the spectral peak of ``a(t)`` (reported as ``omega_fft``).

Measurement chain
-----------------
The interface height ``h(x, t)`` is the ``phi = 0.5`` contour located by linear
interpolation per column (:func:`tensorlbm.capillary_wave.interface_height`),
projected on ``cos(kx)``.  ``interface_sharpness`` (``max |grad phi|``) monitors
the interface thickness, so a frequency shift caused by a thickening interface
can be told apart from a momentum error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from ..capillary_wave import (
    damped_sinusoid_fit,
    damping_theory,
    dispersion_theory,
    interface_height,
    modal_amplitude,
)
from ..free_surface_tension.csf import gradient
from ..free_surface_tension.solver import CapillarySolver, SolverConfig

__all__ = [
    "WaveConfig",
    "damped_fit_windows",
    "interface_sharpness",
    "run_dispersion_scan",
    "run_standing_wave",
    "standing_wave_phi",
    "zero_crossing_period",
]


@dataclass
class WaveConfig:
    """Geometry, fluid properties and measurement settings of one wave case.

    ``advection`` selects the interface transport of the solver: ``"sl"``
    (semi-Lagrangian cubic, low numerical diffusion) is the default;
    ``"upwind"`` (conservative flux form) is available because it changes the
    discrete dissipation and therefore ``gamma``.
    """

    nx: int = 128
    ny: int = 64
    mode: int = 2
    amplitude: float = 0.5
    sigma: float = 0.05
    rho_l: float = 1.0
    rho_g: float = 1.0
    tau: float = 0.6
    width: float = 2.0
    steps: int = 6000
    sample: int = 5
    burn: int = 0
    advection: str = "sl"
    dtype: str = "float64"
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def k(self) -> float:
        return 2.0 * math.pi * self.mode / self.nx

    @property
    def nu(self) -> float:
        """Kinematic viscosity of the LBGK collision, ``nu = (tau - 1/2)/3``."""
        return (self.tau - 0.5) / 3.0

    @property
    def steepness(self) -> float:
        return self.amplitude * self.k

    @property
    def depth(self) -> float:
        """Layer depth ``H``: interface to symmetry plane (``ny/2``, periodic box)."""
        return 0.5 * self.ny

    def omega_theory(self) -> float:
        """Finite-depth capillary dispersion (1); deep-water limit for ``kH >> 1``."""
        k = self.k
        return dispersion_theory(self.sigma, k, self.rho_l, self.rho_g) * math.sqrt(
            math.tanh(k * self.depth)
        )

    def omega_theory_deep(self) -> float:
        """Deep-water limit of (1) -- kept for the finite-depth comparison."""
        return dispersion_theory(self.sigma, self.k, self.rho_l, self.rho_g)

    def gamma_theory(self) -> float:
        return damping_theory(self.nu, self.k)

    def validate(self) -> None:
        if self.nx % 2 or self.ny % 2:
            raise ValueError("nx and ny must be even (periodic interface at ny/2)")
        if self.steepness > 0.2:
            raise ValueError(
                f"A*k = {self.steepness:.3f} leaves the linear regime (limit 0.2)"
            )
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for a positive viscosity")
        if self.advection not in ("sl", "upwind"):
            raise ValueError(f"unknown advection {self.advection!r}")


def standing_wave_phi(config: WaveConfig, *, device: str = "cpu") -> torch.Tensor:
    """``(ny, nx)`` interface indicator of the perturbed layered interface.

    ``phi = 0.5 + 0.5 tanh(2 (y_i(x) - y) / W)`` with ``y_i = ny/2 + A cos(kx)``
    (``phi = 1`` below the interface) -- the same convention as
    :func:`tensorlbm.csf.tanh_flat_phi`, perturbed by one cosine mode.
    """
    config.validate()
    x = torch.arange(config.nx, dtype=torch.float64, device=device).view(1, -1)
    y = torch.arange(config.ny, dtype=torch.float64, device=device).view(-1, 1)
    yi = 0.5 * config.ny + config.amplitude * torch.cos(config.k * x)
    return 0.5 + 0.5 * torch.tanh(2.0 * (yi - y) / config.width)


def interface_sharpness(phi: torch.Tensor, *, trim: int = 1) -> float:
    """``max |grad phi|`` away from the periodic seam -- ``1/W`` for an ideal ``tanh``.

    TRAP (measured): the box is periodic in ``y`` and the liquid band ends where
    ``phi`` jumps from 1 (row 0) to 0 (row -1), so a plain central-difference
    gradient returns ``|grad phi| = 0.5`` *at the seam* for **every** interface
    width -- the diagnostic then reports a constant 0.5 while the interface may be
    smearing freely.  The ``trim`` rows at both ends are therefore excluded; the
    physical interface sits at ``ny/2`` and is unaffected.
    """
    if trim <= 0:
        return float(torch.sqrt(sum(g * g for g in gradient(phi)) + 1e-30).max())
    g2 = sum(g * g for g in gradient(phi))
    core = g2[trim:-trim] if g2.shape[0] > 2 * trim else g2
    return float(torch.sqrt(core + 1e-30).max())


def zero_crossing_period(
    t: Sequence[float], a: Sequence[float]
) -> dict[str, Any]:
    """Period estimate from the zero crossings of ``a(t)``.

    Returns the crossings, the half-period list and ``omega`` from the mean of
    the *later half* of the half-periods (the early ones are still affected by
    the shape-relaxation transient).  No functional form is assumed, so this is
    an independent check of the damped-sinusoid fit.
    """
    tt = [float(v) for v in t]
    aa = [float(v) for v in a]
    cross: list[float] = []
    for i in range(1, len(aa)):
        if aa[i - 1] == 0.0:
            continue
        if aa[i - 1] * aa[i] < 0.0:
            cross.append(tt[i - 1] + (tt[i] - tt[i - 1]) * abs(aa[i - 1]) / (
                abs(aa[i - 1]) + abs(aa[i])
            ))
    periods = [2.0 * (cross[i + 1] - cross[i]) for i in range(len(cross) - 1)]
    out: dict[str, Any] = {"crossings": cross, "periods": periods}
    if len(periods) >= 2:
        tail = periods[len(periods) // 2 :]
        out["period_mean"] = sum(tail) / len(tail)
        out["omega"] = 2.0 * math.pi / out["period_mean"]
        out["n_periods"] = len(periods)
    return out


def extrema_half_period(
    t: Sequence[float], a: Sequence[float], *, smooth: int = 5
) -> dict[str, Any]:
    """Half period from the spacing of successive extrema of ``a(t)``.

    A light moving average (``smooth`` samples) suppresses the lattice-scale
    jitter that the interface tracker leaves on the record; the first and last
    spacing are dropped because the moving average is one-sided there.  The
    *median* is returned (robust to a single missed extremum) together with the
    spread, so a contaminated record is visible instead of silently averaged.
    """
    import numpy as np

    tt = np.asarray(t, dtype=float)
    aa = np.asarray(a, dtype=float)
    if aa.size < 3 * smooth:
        return {"omega": float("nan"), "n_half_periods": 0}
    ker = np.ones(smooth) / smooth
    asm = np.convolve(aa, ker, mode="same")
    idx = [
        i
        for i in range(2, asm.size - 2)
        if (asm[i] - asm[i - 1]) * (asm[i + 1] - asm[i]) < 0.0
    ]
    if len(idx) < 3:
        return {"omega": float("nan"), "n_half_periods": 0}
    spac = [tt[idx[i + 1]] - tt[idx[i]] for i in range(len(idx) - 1)]
    spac = spac[1:-1] if len(spac) > 3 else spac
    med = float(np.median(spac))
    return {
        "half_period": med,
        "omega": math.pi / med if med > 0 else float("nan"),
        "n_half_periods": len(spac),
        "spread": float(np.std(spac)),
        "extrema_t": [float(tt[i]) for i in idx],
    }


def damped_fit_windows(
    t: Sequence[float],
    a: Sequence[float],
    *,
    omega_theory: float,
    gamma_theory: float,
    windows: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Damped-sinusoid fits on several windows of the amplitude record.

    The bounds are scaled by the analytic values (3 sigma wide in ``omega``, 20
    in ``gamma``) so a fit cannot silently lock onto a harmonic; every fit
    reports its ``R^2`` so an unconverged window is visible.
    """
    if windows is None:
        windows = {"early": (0.05, 0.45), "mid": (0.30, 0.75), "late": (0.50, 1.0)}
    n = len(t)
    out: dict[str, dict[str, float]] = {}
    for name, (f0, f1) in windows.items():
        i0 = max(int(f0 * n), 0)
        i1 = min(int(f1 * n), n)
        if i1 - i0 < 6:
            out[name] = {"error": "window too short"}
            continue
        try:
            fit = damped_sinusoid_fit(
                t[i0:i1],
                a[i0:i1],
                omega_bounds=(0.6 * omega_theory, 1.6 * omega_theory),
                gamma_bounds=(0.0, 20.0 * gamma_theory),
            )
        except ValueError as exc:  # pragma: no cover - defensive
            out[name] = {"error": str(exc)}
            continue
        out[name] = {
            "omega": fit["omega"],
            "omega_rel_err": fit["omega"] / omega_theory - 1.0,
            "gamma": fit["gamma"],
            "gamma_rel_err": fit["gamma"] / gamma_theory - 1.0,
            "r2": fit["r2"],
            "amplitude": fit["amplitude"],
            "window": [i0, i1],
        }
    return out


def _fft_peak(t: Sequence[float], a: Sequence[float], *, lo: float, hi: float) -> float:
    """Dominant angular frequency of ``a(t)`` inside ``[lo, hi]`` (zero padded FFT)."""
    import numpy as np

    tt = np.asarray(t, dtype=float)
    aa = np.asarray(a, dtype=float) - float(np.mean(a))
    if tt.size < 8:
        return float("nan")
    dt = float(np.mean(np.diff(tt)))
    nfft = 1 << 16
    spec = np.abs(np.fft.rfft(aa * np.hanning(tt.size), n=nfft))
    freq = np.fft.rfftfreq(nfft, d=dt)  # cycles per lattice step
    omega = 2.0 * np.pi * freq
    band = (omega >= lo) & (omega <= hi)
    if not band.any():
        return float("nan")
    return float(omega[band][np.argmax(spec[band])])


def run_standing_wave(config: WaveConfig, *, device: str = "cpu") -> dict[str, Any]:
    """Run one standing-wave case and return the measured ``(omega, gamma)``.

    ``sigma = 0`` is the **negative control**: with the capillary force switched
    off the interface must not oscillate at all, so only the amplitude history
    (and its maximum deviation) is reported; a wave there would mean the signal
    is a numerical artefact, not a capillary response.
    """
    config.validate()
    solver = CapillarySolver(
        (config.ny, config.nx),
        SolverConfig(
            sigma=config.sigma,
            tau_liquid=config.tau,
            tau_gas=config.tau,
            rho0=0.5 * (config.rho_l + config.rho_g),
            width=config.width,
            init_laplace=False,
            advection=config.advection,
            dtype=torch.float64 if config.dtype == "float64" else torch.float32,
        ),
    )
    phi = standing_wave_phi(config, device=device)
    solver.set_phi(phi, init_laplace=False)
    if config.burn:
        solver.step(config.burn, advect=True)

    k = config.k
    ts: list[int] = []
    amps: list[float] = []
    sharp: list[float] = []
    umax: list[float] = []
    done = 0
    while done < config.steps:
        n = min(config.sample, config.steps - done)
        solver.step(n, advect=True)
        done += n
        ts.append(config.burn + done)
        amps.append(modal_amplitude(solver.phi.unsqueeze(0), k))
        if len(amps) % 20 == 1:
            sharp.append(interface_sharpness(solver.phi))
        umax.append(solver.max_speed())

    out: dict[str, Any] = {
        "case": "standing_capillary_wave",
        "label": config.label,
        "params": {
            "nx": config.nx,
            "ny": config.ny,
            "mode": config.mode,
            "k": k,
            "kW": k * config.width,
            "k_half_depth": k * 0.5 * config.ny,
            "amplitude": config.amplitude,
            "steepness_Ak": config.steepness,
            "sigma": config.sigma,
            "nu": config.nu,
            "tau": config.tau,
            "width": config.width,
            "steps": config.steps,
            "sample": config.sample,
            "burn": config.burn,
            "advection": config.advection,
        },
        "series": {"t": ts, "a": amps, "sharpness": sharp, "umax": umax},
        "sharpness_initial": sharp[0] if sharp else float("nan"),
        "sharpness_final": sharp[-1] if sharp else float("nan"),
        "mass_drift_rel": abs(float(solver.phi_mass()) - float(phi.sum())) / float(phi.sum()),
    }

    if config.sigma == 0.0:
        out["negative_control"] = {
            "amp_initial": amps[0],
            "amp_max_dev": max(abs(a - amps[0]) for a in amps),
            "amp_final": amps[-1],
            "umax": max(umax),
        }
        return out

    wth = config.omega_theory()
    gth = config.gamma_theory()
    out["theory"] = {"omega": wth, "gamma": gth, "period": 2.0 * math.pi / wth}
    out["fit_windows"] = damped_fit_windows(ts, amps, omega_theory=wth, gamma_theory=gth)
    # the smoothing window is tied to the *predicted* half period (15% of it), so
    # it removes lattice-scale jitter without touching the physical oscillation
    sample_dt = float(ts[1] - ts[0]) if len(ts) > 1 else 1.0
    smooth = max(5, int(0.15 * (math.pi / config.omega_theory()) / sample_dt))
    smooth = smooth + 1 if smooth % 2 == 0 else smooth
    ext = extrema_half_period(ts, amps, smooth=smooth)
    ext["smooth_samples"] = smooth
    ext["smooth_steps"] = smooth * sample_dt
    out["extrema"] = ext
    out["crossing"] = zero_crossing_period(ts, amps)
    out["omega_deep_theory"] = config.omega_theory_deep()
    out["omega_fft"] = _fft_peak(ts, amps, lo=0.6 * wth, hi=1.6 * wth)
    if "omega" in out["crossing"]:
        out["omega_cross"] = out["crossing"]["omega"]
        out["omega_cross_rel_err"] = out["omega_cross"] / wth - 1.0
    out["omega_fft_rel_err"] = out["omega_fft"] / wth - 1.0
    # the primary estimator: extrema spacing (unbiased under strong damping)
    primary = ext.get("omega", float("nan"))
    out["omega_extrema"] = primary
    out["metrics"] = {
        "omega_meas": primary,
        "omega_rel_err": primary / wth - 1.0,
        "sigma_eff": primary**2 * (config.rho_l + config.rho_g) / k**3,
        "sigma_eff_rel_err": primary**2 * (config.rho_l + config.rho_g) / k**3 / config.sigma - 1.0,
    }
    return out


def run_dispersion_scan(configs: Sequence[WaveConfig], *, device: str = "cpu") -> dict[str, Any]:
    """Run every case of a dispersion scan and return per-case + summary metrics."""
    results = [run_standing_wave(cfg, device=device) for cfg in configs]
    errors = [
        r["metrics"]["omega_rel_err"]
        for r in results
        if r.get("metrics", {}).get("omega_rel_err") == r.get("metrics", {}).get("omega_rel_err")
    ]
    return {
        "cases": results,
        "summary": {
            "n_cases": len(results),
            "max_abs_omega_rel_err": max((abs(e) for e in errors), default=float("nan")),
        },
    }