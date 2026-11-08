"""Real LBM hold-out test for wave probe inversion.

Uses the TensorLBM wave tank to generate probe data, then applies the
inversion protocol determined by the identifiability scan. Compares
the measured phase speed with the closed-form dispersion relation.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@dataclass
class LBMHoldoutResult:
    """Result of LBM hold-out test.

    Attributes
    ----------
    kh : float
        Non-dimensional wavenumber.
    c_measured : float
        Measured phase speed from LBM simulation.
    c_theory : float
        Theoretical phase speed from dispersion relation.
    c_error : float
        Relative error |c_measured - c_theory| / c_theory.
    R_measured : float
        Measured reflection coefficient.
    protocol : dict
        Probe protocol used (dx/λ, s/λ, L/λ, n_probes).
    passed : bool
        Whether the test passed (c_error < 3%).
    """
    kh: float
    c_measured: float
    c_theory: float
    c_error: float
    R_measured: float
    protocol: dict
    passed: bool

    def to_dict(self) -> dict:
        return {
            "kh": self.kh,
            "c_measured": self.c_measured,
            "c_theory": self.c_theory,
            "c_error": self.c_error,
            "R_measured": self.R_measured,
            "protocol": self.protocol,
            "passed": self.passed,
        }


def run_lbm_holdout(
    kh: float = 1.5,
    eps: float = 0.08,
    protocol: Optional[dict] = None,
) -> LBMHoldoutResult:
    """Run LBM hold-out test.

    Parameters
    ----------
    kh : float
        Non-dimensional wavenumber.
    eps : float
        Wave steepness ka.
    protocol : dict, optional
        Probe protocol. If None, uses the minimum protocol from identifiability.

    Returns
    -------
    LBMHoldoutResult
    """
    from tensorlbm.wave_tank.tank import WaveTankConfig, run_case
    from tensorlbm.wave_tank.stokes2 import dispersion_solve
    from tensorlbm.wave_tank.gauges import fit_harmonic, phase_speed_from_phase

    # Default protocol based on identifiability results
    if protocol is None:
        protocol = {
            "dx_over_lambda": 1/20,
            "s_over_lambda": 0.5,
            "L_over_lambda": 2.0,
            "n_probes": 5,
        }

    # Configure wave tank
    # Use the standard depth for this kh
    DEPTH = {0.7: 12.0, 1.5: 24.0, 3.0: 40.0}
    h = DEPTH.get(kh, 24.0)
    g = 6.5e-3
    nu = 0.008

    # Compute wave parameters
    k = kh / h
    omega = math.sqrt(g * (1 - 0.1) * k * math.tanh(k * h))  # g_eff = g*(1-rho_g/rho_l)
    wavelength = 2 * math.pi / k
    period = 2 * math.pi / omega
    c_theory = omega / k

    # Set up probe positions based on protocol
    s = protocol["s_over_lambda"] * wavelength
    L = protocol["L_over_lambda"] * wavelength
    n_probes = protocol.get("n_probes", 5)

    # Probe fraction (relative to mid zone)
    lam_mid = protocol["L_over_lambda"] + 0.5  # add margin
    probe_frac = tuple([0.1 + i * protocol["s_over_lambda"] for i in range(n_probes)])
    probe_frac = tuple([p / lam_mid for p in probe_frac])  # normalize

    cfg = WaveTankConfig(
        kh=kh,
        eps=eps,
        h=h,
        g=g,
        nu=nu,
        absorb=True,
        lam_mid=lam_mid,
        probe_frac=probe_frac,
        warm_periods=6.0,
        meas_periods=6.0,
        sample_every=2,
    )

    # Run simulation
    print(f"Running LBM hold-out: kh={kh}, eps={eps}")
    print(f"  Protocol: s/λ={protocol['s_over_lambda']:.2f}, L/λ={protocol['L_over_lambda']:.1f}")
    print(f"  Probes: {n_probes} at fractions {probe_frac}")

    result = run_case(cfg, quiet=True)

    # Extract probe time series
    probe_eta = result.series["probe_eta"]  # shape: (n_probes, n_times)
    t = result.series["t"]
    probe_cols = result.series["probe_cols"]

    # Compute physical probe positions
    n_in = cfg.geometry()["n_in"]
    x_probes = [n_in + col for col in probe_cols]

    # Fit harmonics at each probe
    omega_fit = cfg.wave.omega
    c_probes = []
    for i in range(n_probes):
        c = fit_harmonic(t, probe_eta[i], omega_fit)
        c_probes.append(c)

    c_probes = np.array(c_probes)

    # Run multi-probe inversion
    from .inversion import MultiProbeLeastSquares
    inv = MultiProbeLeastSquares(omega_fit, k, 0.0)
    inv_result = inv.invert(np.array(x_probes, dtype=float), c_probes)

    c_measured = inv_result.c_hat
    R_measured = inv_result.R_hat
    c_error = abs(c_measured - c_theory) / c_theory

    print(f"  c_theory = {c_theory:.6f}")
    print(f"  c_measured = {c_measured:.6f}")
    print(f"  c_error = {c_error:.4f} ({c_error*100:.2f}%)")
    print(f"  R_measured = {R_measured:.4f}")

    passed = c_error < 0.03

    return LBMHoldoutResult(
        kh=kh,
        c_measured=c_measured,
        c_theory=c_theory,
        c_error=c_error,
        R_measured=R_measured,
        protocol=protocol,
        passed=passed,
    )


def run_old_protocol_negative_control(
    kh: float = 1.5,
    eps: float = 0.08,
) -> LBMHoldoutResult:
    """Run negative control with old (bad) protocol.

    The old protocol used only 2 probes with small spacing, which should
    give ≥7% phase speed error due to reflection contamination.
    """
    # Old protocol: 2 probes, small spacing s/λ = 0.1
    protocol = {
        "dx_over_lambda": 1/10,
        "s_over_lambda": 0.1,
        "L_over_lambda": 0.5,
        "n_probes": 2,
    }
    return run_lbm_holdout(kh, eps, protocol)
