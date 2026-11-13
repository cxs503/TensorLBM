"""Regression tests for the XVAL Shan-Chen Young-Laplace calibration.

Anchor: ``XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`` (round 1).

These tests pin the two facts the cross-validation rests on:

1. the TensorLBM SC-MP kernel (``sc_surface_tension.step_sc``) is *algebraically*
   identical to the documented OSS port (``oss_port.OSSShanChen``) -- five
   identical steps from the same field agree to machine precision;
2. the OSS pressure closure ``p = rho/3 + G psi^2/6`` and TensorLBM's
   ``pressure_eos`` are the same function, while the model-free
   ``pressure_flux`` readout is a genuinely different measurement;
3. the upstream uninitialised-radius defect (``chapter9/shanchen.cpp:358``) is
   turned into an exception by the port's guardrail.

They are deliberately a smoke-level subset (a few hundred steps); the calibrated
sweep lives in ``xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/run_xval.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
XVAL = ROOT / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

from oss_port import (OSSShanChen, RadiusEstimatorFailed, calibrate,  # noqa: E402
                      closure_pressure)
import run_xval as X  # noqa: E402

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.sc_surface_tension import (pressure_eos, pressure_flux,  # noqa: E402
                                          step_sc)

torch.set_num_threads(2)

N = 64
G_EFF = -4.7


def _same_initial_field(R: float = 16.0) -> np.ndarray:
    rho = X.sharp_droplet(R)
    return equilibrium(torch.tensor(rho, dtype=torch.float64),
                       torch.zeros(N, N, dtype=torch.float64),
                       torch.zeros(N, N, dtype=torch.float64)).numpy()


def test_kernel_is_algebraically_identical_to_oss_port():
    """5 identical steps: max |rho_OSS - rho_TL| must be at float64 round-off."""
    f = torch.tensor(_same_initial_field(16.0), dtype=torch.float64)
    o = OSSShanChen(N, X.TAU, G_EFF, X.RHO0, X.RHO_L_INIT, X.RHO_G_INIT)
    o.f = f.numpy().copy()
    for _ in range(5):
        o.step()
        f = step_sc(f, G_eff=G_EFF, tau=X.TAU, stencil=X.STENCIL)
    assert np.max(np.abs(o.rho() - f.sum(0).numpy())) < 1e-12


def test_closure_pressure_matches_oss_formula():
    """TensorLBM ``pressure_eos`` == OSS closure ``rho/3 + G psi^2/6``."""
    rng = np.random.default_rng(7)
    rho = torch.tensor(0.1 + 2.0 * rng.random((8, 8)), dtype=torch.float64)
    ours = pressure_eos(rho, G_EFF).numpy()
    oss = closure_pressure(rho.numpy(), G_EFF, X.RHO0)
    assert np.max(np.abs(ours - oss)) < 1e-14


def test_flux_readout_differs_from_closure():
    """The momentum-flux readout is a different measurement from the closure."""
    f = torch.tensor(_same_initial_field(16.0), dtype=torch.float64)
    for _ in range(20):
        f = step_sc(f, G_eff=G_EFF, tau=X.TAU, stencil=X.STENCIL)
    rho = f.sum(0)
    d = float(np.max(np.abs(pressure_flux(f, G_EFF).numpy()
                            - pressure_eos(rho, G_EFF).numpy())))
    assert d > 0.0


def test_calibrator_agrees_between_oss_and_driver():
    """The OSS calibrator and the driver's re-implementation agree on one field."""
    f = torch.tensor(_same_initial_field(16.0), dtype=torch.float64)
    for _ in range(50):
        f = step_sc(f, G_eff=G_EFF, tau=X.TAU, stencil=X.STENCIL)
    rho = f.sum(0).numpy()
    p = closure_pressure(rho, G_EFF, X.RHO0)
    a = calibrate(f.numpy(), G_EFF, X.RHO0)
    b = X.calibrate_arrays(rho, p)
    assert a["dp"] == pytest.approx(b["dp"], rel=1e-12, abs=1e-15)
    assert a["r"] == pytest.approx(b["r"], rel=1e-12)
    assert a["gamma"] == pytest.approx(b["gamma"], rel=1e-12)


def test_radius_guardrail_replaces_upstream_undefined_value():
    """No rho = rho_av crossing -> exception, never an undefined gamma."""
    flat = np.full((N, N), 1.0)
    with pytest.raises(RadiusEstimatorFailed):
        calibrate(np.full((9, N, N), 1.0 / 9.0), G_EFF)
    # and the driver's wrapper turns it into an explicit failure record
    rec = X.calib_safe(flat, closure_pressure(flat, G_EFF))
    assert rec["gamma"] is None
    assert rec["status"].startswith("radius_estimator_failed")


def test_steady_state_run_gives_positive_laplace_gamma():
    """The calibrated steady state (4000 steps, R=16) has dp > 0 and gamma > 0.

    Note: the first ~200 steps are a violent transient of the (deliberately
    off-coexistence, rho_l=2.1) initial condition -- the centre box is
    momentarily gas, so the OSS calibrator is only meaningful at steady state.
    """
    s = OSSShanChen(N, X.TAU, G_EFF, X.RHO0, X.RHO_L_INIT, X.RHO_G_INIT)
    s.init_droplet(16.0)
    s.run(4000)
    rho = s.rho()
    cal = X.calib_safe(rho, closure_pressure(rho, G_EFF, X.RHO0))
    assert cal["status"] == "ok"
    assert cal["dp"] > 0.0
    assert cal["gamma"] > 0.0
    assert cal["rho_l"] > cal["rho_g"]