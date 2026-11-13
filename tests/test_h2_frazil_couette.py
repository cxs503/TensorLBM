"""Guards for the reworked H2 Couette face (``scripts/run_h2_frazil_couette.py``).

The face exists because the first attempt at a particle-resolved relative
viscosity was audited L1 (``docs/AUDIT_m14882_H2_FRAZIL_RHEOLOGY.md``): a
body-force channel was mis-labelled Couette, the pure-fluid calibration was off
by 2.37x, ``eta_r`` was a ratio of *nominal* rates and the closures were
compared at the declared rather than the achieved concentration.

These tests pin the four properties that made the old face wrong, at a box small
enough to run in seconds:

1. the cell is moving-wall Couette, and its ``c = 0`` case reproduces the exact
   solution ``tau = rho nu gamma_dot`` (both stress estimators),
2. ``eta_r`` is a ratio of *measured* stresses at *measured* rates,
3. the closure argument is the measured lattice volume fraction,
4. the criteria thresholds are the declared ones (nothing is loosened in code).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from tensorlbm import frazil_powerlaw_mature as face  # noqa: E402

BOX = {"n_xy": 10, "n_z": 14, "a_cells": 1.5}
NU = 0.2


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_h2_frazil_couette", ROOT / "scripts" / "run_h2_frazil_couette.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pure_couette_reproduces_exact_stress_both_estimators():
    """``c = 0``: the flux and the per-wall momentum exchange both give ``rho nu gamma``."""
    rec = face.pure_reference(n_xy=BOX["n_xy"], n_z=BOX["n_z"], nu=NU, gamma_dot=1.0e-3,
                              steps=80, warmup=40)
    assert abs(rec["flux_rel_err"]) < 0.02, rec
    assert abs(rec["wall_rel_err"]) < 0.02, rec
    # measured rate is the profile slope, not the nominal rate
    assert rec["gamma_dot"] == pytest.approx(1.0e-3, rel=0.02)


def test_eta_ratio_uses_measured_stresses_and_rates():
    """A pure fluid must give ``eta_r = 1`` however the two rates differ."""
    h2 = _load_script()
    pure_a = {"gamma_dot": 1.0e-3, "tau_flux": 2.0e-4, "tau_wall": 2.0e-4}
    same = {"gamma_dot": 1.0e-3, "tau_flux": 2.0e-4, "tau_wall": 2.0e-4}
    faster = {"gamma_dot": 2.0e-3, "tau_flux": 4.0e-4 * 1.01, "tau_wall": 4.0e-4 * 1.01}
    assert h2._eta_r(same, pure_a, "tau_flux") == pytest.approx(1.0)
    # 1 % extra stress at twice the (measured) rate => 1 % extra viscosity,
    # whereas the old nominal-rate ratio would have reported 2.02 -> 102 % error
    assert h2._eta_r(faster, pure_a, "tau_flux") == pytest.approx(1.01, rel=1e-12)


def test_suspension_eta_r_is_measured_at_lattice_fraction():
    """A dilute frozen suspension gives ``eta_r > 1`` at the *measured* ``phi``."""
    h2 = _load_script()
    parts, info = h2._suspension(0.05, BOX)
    assert 0.0 < info["phi_lattice"] < 0.08
    pure = face.pure_reference(n_xy=BOX["n_xy"], n_z=BOX["n_z"], nu=NU,
                               gamma_dot=1.0e-3, steps=80, warmup=40)
    rec = face.measure_point(parts, n_xy=BOX["n_xy"], n_z=BOX["n_z"], nu=NU,
                             gamma_dot=1.0e-3, steps=120, warmup=60)
    eta = h2._eta_r(rec, pure, "tau_flux")
    assert np.isfinite(eta)
    assert eta > 1.0
    # the coarse 1.5-cell crystal may overstate eta_r, but not by the 3-8x of the
    # audited face: the smoke box must stay within a factor 2 of Einstein
    assert eta < 2.0 * (1.0 + 2.5 * info["phi_lattice"])


def test_closure_anchors_are_the_flat_plate_closures():
    assert float(face.krieger_dougherty(0.0)) == pytest.approx(1.0)
    assert float(face.krieger_dougherty(0.20)) == pytest.approx(
        (1.0 - 0.20 / 0.6) ** (-2.5 * 0.6), rel=1e-12)
    assert float(face.einstein_batchelor_green(0.05)) == pytest.approx(
        1.0 + 2.5 * 0.05 + 6.2 * 0.05 ** 2, rel=1e-12)


def test_criteria_thresholds_are_declared_not_loosened():
    h2 = _load_script()
    assert h2.CRITERIA["c1_pure_couette"]["threshold"] == pytest.approx(0.02)
    assert h2.CRITERIA["c2_dilute_einstein"]["threshold"] == pytest.approx(0.15)
    assert h2.CRITERIA["c3_krieger_dougherty"]["threshold"] == pytest.approx(0.15)
    assert h2.CRITERIA["c4_rate_independence"]["threshold"] == pytest.approx(0.10)
    # the audited face reported gamma_meas/gamma_target = 2.37; the pure control
    # of the reworked face is the guard, so it must sit below that by an order
    assert h2.CRITERIA["c1_pure_couette"]["threshold"] < 0.237