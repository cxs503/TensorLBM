"""Invariant tests for XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP.

Each test asserts a *property* of the two-component Shan-Chen cross-force
operator, not merely that code runs:

* the D2Q9 moment algebra that makes the stencil a second-order, isotropic
  gradient (symbolic arm),
* the closed-form transfer function of that gradient and its ``1 - k^2/6``
  truncation, against an independent FFT derivative (spectral arm),
* bit-level agreement with the OSS (lbmpy) transcription of the same operator,
* exact telescoping of the cross force (Newton's third law on a periodic box),
* the sign of the mixture equation of state, decided by flat-interface
  mechanical equilibrium,
* consistency with the archived cross-validation artifacts when present.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

XVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "development", "verification", "mcmp_cross_force_operator")
sys.path.insert(0, XVAL)

import arms  # noqa: E402

CS2 = 1.0 / 3.0


# --------------------------------------------------------------------------
# arm S -- symbolic
# --------------------------------------------------------------------------

def test_symbolic_moments_and_closed_form():
    s = arms.arm_symbol()
    assert s["table_matches_library_d2q9"], "the sympy D2Q9 table must match tensorlbm.d2q9"
    assert s["M2_equals_cs2_delta"], "second moment must be cs^2 * identity"
    assert s["M3_all_zero"], "third moment vanishes -> the stencil is a 2nd-order gradient"
    assert s["M4_equals_cs4_delta_delta"], "fourth moment must be cs^4 (delta delta + 2 perms)"
    assert s["stencil_closed_form_x_verified"], "closed form must reproduce the sympy sum"
    assert s["gain_leading_cs2_float"] == pytest.approx(CS2, rel=1e-12)
    # k^2 coefficient of the gain is -cs^2/6, i.e. gain = cs^2 (1 - k^2/6 + ...)
    assert s["gain_k2_coefficient_float"] == pytest.approx(-CS2 / 6.0, rel=1e-12)


# --------------------------------------------------------------------------
# arm P -- spectral
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spectral():
    return arms.arm_spectral(n=64, m_max=5)


def test_spectral_gain_matches_closed_form(spectral):
    """Numerical lattice gain == closed form on every reciprocal-lattice mode."""
    assert spectral["max_rel_diff_lattice_vs_closed_form"] < 1e-6
    # the FFT derivative is the independent reference; its gain must be exactly 1
    assert spectral["max_rel_diff_fft_gain_vs_unity"] < 1e-10


def test_spectral_gain_approaches_cs2_and_is_second_order(spectral):
    rows = sorted(spectral["rows"], key=lambda r: r["k"])
    assert rows[0]["gain_over_cs2"] == pytest.approx(1.0, abs=2e-3)
    # observed relative gain error follows k^2/6 in the resolved band
    for r in rows:
        if r["k"] <= 0.5:
            assert r["rel_err_vs_continuum"] == pytest.approx(r["k"] ** 2 / 6.0, rel=0.05)
    order = spectral["refinement_order"]
    assert order, "refinement ladder must produce at least one slope"
    assert all(abs(o - 2.0) < 0.15 for o in order), f"expected 2nd order, got {order}"


# --------------------------------------------------------------------------
# arm O -- OSS transcription
# --------------------------------------------------------------------------

def test_oss_operator_identity():
    o = arms.arm_oss(n=24)
    assert o["max_rel_rms_diff"] < 1e-12, o["cases"]


def test_uniform_field_gives_zero_cross_force():
    n = 16
    r1 = torch.full((n, n), 1.7, dtype=torch.float64)
    r2 = torch.full((n, n), 0.4, dtype=torch.float64)
    fx1, fy1, fx2, fy2 = arms.sc_two_component_force(r1, r2, -2.5)
    assert max(float(v.abs().max()) for v in (fx1, fy1, fx2, fy2)) < 1e-12


def test_cross_force_telescopes_exactly():
    """sum(F_1 + F_2) = 0 on a periodic box (Newton's third law of the operator)."""
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    n = 32
    r1 = torch.as_tensor(1.0 + 0.5 * rng.standard_normal((n, n)), dtype=torch.float64)
    r2 = torch.as_tensor(1.0 + 0.5 * rng.standard_normal((n, n)), dtype=torch.float64)
    F = arms.sc_two_component_force(r1, r2, -2.5)
    scale = max(float(F[j].abs().max()) for j in range(4)) * n * n
    integ = max(abs(float((F[0] + F[2]).sum())), abs(float((F[1] + F[3]).sum())))
    assert integ / scale < 1e-12, f"third law residual {integ / scale:.3e}"


# --------------------------------------------------------------------------
# equation of state: which sign makes a flat interface uniform
# --------------------------------------------------------------------------

def test_pressure_conventions_differ_by_the_interaction_term():
    """The two candidate EOS conventions differ by exactly 2|G| cs^2 psi1 psi2.

    A symmetric flat-interface *state* cannot discriminate them (the ideal parts
    are equal and the interaction term has the same magnitude with either sign),
    so the sign is decided by the relaxed flat-interface run archived by
    ``run_xval.py`` (see ``test_archived_results_consistency``) and by the
    momentum balance of the discrete kernel.
    """
    rng = np.random.default_rng(5)
    n = 24
    r1 = torch.as_tensor(1.0 + 0.3 * rng.standard_normal((n, n)), dtype=torch.float64)
    r2 = torch.as_tensor(1.0 + 0.3 * rng.standard_normal((n, n)), dtype=torch.float64)
    G = -2.5
    diff = (arms._mcmp_pressure(r1, r2, G, convention="library")
            - arms._mcmp_pressure(r1, r2, G, convention="derived"))
    expect = 2.0 * G * CS2 * r1 * r2
    assert float((diff - expect).abs().max()) < 1e-14


# --------------------------------------------------------------------------
# archived artifacts (physics arm) -- skips when the xval has not been run
# --------------------------------------------------------------------------

def test_archived_results_consistency():
    path = os.path.join(XVAL, "artifacts", "xval_results.json")
    if not os.path.exists(path):
        pytest.skip("run_xval.py has not been executed in this checkout")
    d = json.load(open(path))
    assert d["arm_symbol"]["stencil_closed_form_x_verified"]
    assert d["arm_oss"]["max_rel_rms_diff"] < 1e-12
    cases = [c for c in d["arm_physics"]["cases"] if c.get("status") == "ok"]
    assert cases, "physics arm produced no finished case"
    for c in cases:
        assert c["interface_contrast"] > 0.5, "a real interface must be present"
        # pressure must be uniform across the flat interface (3% gate)
        assert c["p_uniformity_rel"] < 0.03, (c["G"], c["p_uniformity_rel"])
        # and the wrong sign must be far worse -- this is the discriminating test
        assert c["p_uniformity_rel"] * 5 < c["p_uniformity_library_convention"]
        assert c["boltzmann_relation_residual_rel"] < 0.03, (c["G"], c["boltzmann_relation_residual_rel"])
        if math.isfinite(c["newton_third_law_residual_rel"]):
            assert c["newton_third_law_residual_rel"] < 1e-12
    assert math.isfinite(cases[0]["operator_error_on_interface_rel"])