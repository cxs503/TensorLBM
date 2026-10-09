"""Unit tests for the shared Re ↔ tau conversion module.

Convention under test (D 口径): the characteristic length is the DIAMETER
``L_ref = 2R``, i.e. ``Re = u·2R/ν`` and ``tau = 0.5 + 3·(u·2R/Re)``.
"""

from __future__ import annotations

import pytest

from tensorlbm.lbm_re_tau import nu_from_re, nu_from_tau, re_from_tau, tau_from_re


def test_tau_from_re_radius6_expected_0_5216() -> None:
    # Re=100, u=0.06, R=6 → L_ref=12 → tau = 0.5 + 3*(0.06*12/100) = 0.5216
    assert tau_from_re(0.06, 2 * 6, 100) == pytest.approx(0.5216, abs=1e-12)


def test_tau_from_re_radius10_expected_0_536() -> None:
    # Re=100, u=0.06, R=10 → L_ref=20 → tau = 0.5 + 3*(0.06*20/100) = 0.536
    assert tau_from_re(0.06, 2 * 10, 100) == pytest.approx(0.536, abs=1e-12)


def test_tau_matches_legacy_2r_inline_formula() -> None:
    u, R, Re = 0.06, 10.0, 100.0
    assert tau_from_re(u, 2 * R, Re) == pytest.approx(0.5 + 3.0 * (u * 2.0 * R / Re))


def test_re_from_tau_round_trip() -> None:
    u, R, Re = 0.06, 6.0, 100.0
    tau = tau_from_re(u, 2 * R, Re)
    assert re_from_tau(u, 2 * R, tau) == pytest.approx(Re, rel=1e-12)


def test_re_from_tau_reference_values() -> None:
    # tau=0.53, u=0.05, D=40 → Re = 3*0.05*40/0.03 = 200
    assert re_from_tau(0.05, 40.0, 0.53) == pytest.approx(200.0)


def test_nu_from_re_matches_tau_chain() -> None:
    u, R, Re = 0.06, 10.0, 100.0
    nu = nu_from_re(u, 2 * R, Re)  # = u*2R/Re = 0.012
    assert nu == pytest.approx(0.012, abs=1e-12)
    # nu = (tau - 0.5)/3 must be consistent with tau_from_re
    assert nu == pytest.approx(nu_from_tau(tau_from_re(u, 2 * R, Re)), abs=1e-12)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        tau_from_re(0.06, 0.0, 100)  # L_ref must be > 0
    with pytest.raises(ValueError):
        tau_from_re(0.06, 12.0, -100)  # Re must be > 0
    with pytest.raises(ValueError):
        re_from_tau(0.06, 12.0, 0.5)  # tau must be > 0.5
    with pytest.raises(ValueError):
        nu_from_tau(0.4)
