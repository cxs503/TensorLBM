"""Verification tests for the spectral Saffman--Taylor (integrating factor) solver.

Covers the claims made in
``docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR.md``:

1. the exponential integrating factor is exact for the diffusion operator
   (relative L2 error vs the analytic Fourier decay < 1e-8);
2. the port reproduces an independent NumPy transcription of the OSS
   ``spectrum_method`` scheme to round-off (R = 0 and R > 0);
3. mass is conserved and the discrete velocity is divergence free;
4. the transverse mode growth rate scales linearly with the log-viscosity ratio
   R (first-order coupling) once the exact diffusive decay k^2 is removed;
5. the reported elliptic defect (OSS frozen-coefficient symbol vs the exact
   curl-Darcy relation) is O(1) for R > 0 and zero for R = 0.

Run: ``python -m pytest tests/test_spectral_saffman_taylor.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent
                   / "xval" / "XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR"))

from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D  # noqa: E402
import oss_spectral_reference as ossref                              # noqa: E402

NY, NX, ASPECT, PE = 32, 64, 2.0, 200.0


def _sim(**kw):
    cfg = dict(ny=NY, nx=NX, aspect=ASPECT, pe=PE, dt=0.1, mode_index=3)
    cfg.update(kw)
    return SpectralSaffmanTaylor2D(**cfg)


def _rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_wavenumbers_match_oss_convention():
    sim = _sim()
    assert np.isclose(sim.km[1], 2 * np.pi / (ASPECT * PE))
    assert np.isclose(sim.kn[1], 2 * np.pi / PE)
    assert float(sim._k2[0, 0]) == 0.0                    # zero mode handled separately


def test_integrating_factor_is_exact_for_diffusion():
    """R = 0: the scheme must reproduce exp(-k^2 t) to round-off."""
    x = np.arange(NX) * (ASPECT * PE / NX)
    y = np.arange(NY) * (PE / NY)
    m, n = 2, 3
    km, kn = 2 * np.pi * m / (ASPECT * PE), 2 * np.pi * n / PE
    k2 = km ** 2 + kn ** 2
    c0 = 0.5 + 0.25 * np.cos(km * x)[None, :] * np.cos(kn * y)[:, None]
    sim = _sim(viscosity_ratio=0.0)
    sim.set_concentration(c0)
    sim.run(5.0)
    exact = 0.5 + 0.25 * np.exp(-k2 * 5.0) * np.cos(km * x)[None, :] * np.cos(kn * y)[:, None]
    assert _rel_l2(sim.numpy_state(), exact) < 1e-8


@pytest.mark.parametrize("R", [0.0, 2.0])
def test_port_matches_numpy_oss_transcription(R):
    """The torch port must reproduce the OSS scheme to round-off."""
    ic = ossref.oss_single_mode_ic(NY, NX, ASPECT, PE, x0=0.5 * ASPECT * PE,
                                   width=4.0, amplitude=1e-3, n_mode=3)
    sim = _sim(viscosity_ratio=R)
    sim.set_concentration(ic)
    ref = ossref.OSSCompute(ny=NY, nx=NX, aspect=ASPECT, pe=PE,
                            viscosity_ratio=R, dt=0.1)
    ref.set_c(ic)
    n = 20
    for _ in range(n):
        sim.step()
        ref.step()
    assert _rel_l2(sim.numpy_state(), ref.c) < 1e-10


def test_mass_conservation_and_divergence_free():
    ic = ossref.oss_random_front_ic(NY, NX, seed=7)
    sim = _sim(viscosity_ratio=1.0)
    sim.set_concentration(ic)
    m0 = float(sim.c.mean())
    sim.run(5.0)
    assert abs(float(sim.c.mean()) - m0) / m0 < 1e-12
    assert sim.divergence_error() < 1e-10


def test_growth_rate_scales_linearly_with_R():
    """(sigma + k^2)/R is R-independent for the first-order coupling."""
    x0 = 0.5 * ASPECT * PE
    k2 = (2 * np.pi * 3 / PE) ** 2
    coef = []
    for R in (0.2, 1.0):
        sim = _sim(viscosity_ratio=R, dt=0.05)
        sim.set_concentration(sim.single_mode_ic(x0=x0, width=4.0, amplitude=1e-3,
                                                 n_mode=3))
        ts, amps = [], []
        for _ in range(81):
            d = sim.diagnostics()
            ts.append(d.t)
            amps.append(d.mode_amplitude)
            sim.step()
        ts, amps = np.array(ts), np.array(amps)
        mask = (ts >= 0.5) & (ts <= 4.0)
        sigma = float(np.polyfit(ts[mask], np.log(amps[mask]), 1)[0])
        coef.append((sigma + k2) / R)
    assert abs(coef[0] - coef[1]) / abs(coef[0]) < 0.15


def test_elliptic_defect_reported():
    """The OSS frozen symbol violates the exact curl-Darcy relation for R > 0."""
    ic = ossref.oss_random_front_ic(NY, NX, seed=11)
    sim0 = _sim(viscosity_ratio=0.0)
    sim0.set_concentration(ic)
    assert sim0.elliptic_residual()["residual_l2"] == 0.0
    sim = _sim(viscosity_ratio=1.0)
    sim.set_concentration(ic)
    assert sim.elliptic_residual()["residual_l2"] > 0.1