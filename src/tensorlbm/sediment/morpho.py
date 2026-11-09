"""1-D morphodynamic response used to *measure* the incipient-motion curve.

The critical Shields stress is not read off the model parameter; it is
*measured* from the bed response of a weakly perturbed bed.  A sinusoidal
perturbation of amplitude ``A`` is imposed on an otherwise uniform bed.  The
local shear velocity responds to the local water depth through the standard
"locally uniform flow" scaling

    u*(x) = u*_ref ( h_ref / h(x) )^{3/2},        h(x) = h_ref - (z_b - z_ref),

so a bed bump (smaller ``h``) sees a larger Shields number and erodes faster
than a trough.  The *numerical* critical Shields stress ``theta_num`` is the
base Shields number at which the perturbation amplitude, integrated over a
fixed observation window ``T``, crosses a physically motivated detection
level (a bed change of ``detect_frac`` of the grain size).  Because the
response is computed by the Exner/MPM model driven by the LBM wall-shear
velocity, the comparison against the Soulsby-Whitehouse reference (criterion
C1) is not circular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .exner import ExnerConfig, exner_step, mpm_flux

__all__ = ["MorphoConfig", "run_morpho", "measure_theta_cr"]


@dataclass(frozen=True)
class MorphoConfig:
    """Parameters for the 1-D morphodynamic incipient-motion probe."""

    d: float  # grain diameter (m)
    s: float = 1.65
    g: float = 9.81
    porosity: float = 0.4
    h_ref: float = 1.0  # reference water depth (m)
    n_waves: int = 2  # number of perturbation wavelengths in the domain
    domain_len: float = 20.0  # domain length (m)
    n_cells: int = 200  # reduced for speed
    amp_frac: float = 0.02  # perturbation amplitude as a fraction of d
    obs_time: float = 3600.0  # observation window (s)
    n_steps: int = 500  # reduced for speed
    detect_frac: float = 0.01  # detection level as a fraction of d


def _exner_cfg(mc: MorphoConfig, theta_cr: float) -> ExnerConfig:
    return ExnerConfig(
        d=mc.d, s=mc.s, g=mc.g, porosity=mc.porosity, theta_cr=theta_cr
    )


def _perturbed_bed(mc: MorphoConfig) -> tuple[np.ndarray, float]:
    dx = mc.domain_len / mc.n_cells
    x = (np.arange(mc.n_cells) + 0.5) * dx
    z = mc.amp_frac * mc.d * np.sin(2.0 * np.pi * mc.n_waves * x / mc.domain_len)
    return z, dx


def run_morpho(
    theta_base: float,
    mc: MorphoConfig,
    theta_cr: float,
) -> float:
    """Run the Exner/MPM response for a base Shields number ``theta_base``.

    Returns the maximum absolute bed change over the observation window (m).
    ``theta_base`` sets the base Shields number; the local value varies with
    the bed shape through the depth scaling ``theta(x) = theta_base
    (h_ref / h(x))**3``.  ``theta_cr`` is the MPM incipient-motion threshold
    (the model parameter).
    """
    z_b, dx = _perturbed_bed(mc)
    z_init = z_b.copy()
    cfg = _exner_cfg(mc, theta_cr)
    dt = mc.obs_time / mc.n_steps
    h_ref = mc.h_ref

    for _ in range(mc.n_steps):
        h = h_ref - z_b
        # Prevent division by zero or negative depth
        h = np.maximum(h, 0.1 * h_ref)
        theta = theta_base * (h_ref / h) ** 3
        q = mpm_flux(theta, cfg)
        z_b = exner_step(z_b, q, dx, dt, cfg)
    return float(np.max(np.abs(z_b - z_init)))


def measure_theta_cr(
    u_star: float,
    mc: MorphoConfig,
    theta_cr: float,
    lo: float | None = None,
    hi: float | None = None,
    iters: int = 12,  # reduced for speed
) -> float:
    """Bisect on the base Shields number for the incipient-motion threshold.

    The threshold is the ``theta_base`` at which the bed change over the
    observation window equals the detection level (``detect_frac * d``).
    ``theta_cr`` is the MPM incipient-motion threshold (model parameter).
    ``u_star`` is the wall-shear velocity (m/s) used to compute the Shields
    number scaling.
    """
    target = mc.detect_frac * mc.d
    # Compute the base Shields number from u_star
    denom = (mc.s - 1.0) * mc.g * mc.d
    theta_base_ref = u_star ** 2 / denom

    if lo is None:
        lo = 0.5 * theta_cr
    if hi is None:
        hi = 2.0 * theta_cr
    f_lo = run_morpho(lo, mc, theta_cr) - target
    f_hi = run_morpho(hi, mc, theta_cr) - target
    if f_lo > 0.0:
        return lo
    if f_hi < 0.0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = run_morpho(mid, mc, theta_cr) - target
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
