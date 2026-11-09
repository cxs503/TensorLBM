"""Tests for the mixed bounce-back / specular (partial-slip) wall module.

Covers the four claims the module makes:

1. algebra: the mixed operator reduces to bounce-back at ``sigma=0`` and to
   specular reflection at ``sigma=1``, and is a convex combination in between;
2. plane Poiseuille: the discrete steady profile is exactly parabolic, the
   no-slip limit matches the analytic parabola and the inverted slip length is
   linear in ``sigma/(1-sigma)``;
3. circular pipe: the axisymmetric geometric-source solver reproduces
   Hagen-Poiseuille at ``sigma=0`` and the same ``b(sigma)`` law;
4. turbulent plate: the slip drag-reduction law is monotone and the ledger
   is self-consistent.

All tolerances are the report criteria (C1..C6).
"""

import math

import numpy as np
import pytest

from tensorlbm.boundary import partial_slip_wall as ps

TAU = 0.8
NU = ps.nu_from_tau(TAU)
G = 1.0e-5


# ---------------------------------------------------------------------------
# algebra of the mixed wall operator
# ---------------------------------------------------------------------------
def test_mirror_table_is_involution_and_keeps_cx():
    for q in range(9):
        m = ps.MIRROR_Y[q]
        assert ps.MIRROR_Y[m] == q
        assert ps.C2[m][0] == ps.C2[q][0]      # tangential velocity kept
        assert ps.C2[m][1] == -ps.C2[q][1]     # normal velocity flipped


def test_mixed_wall_sigma0_is_bounce_back():
    rng = np.random.default_rng(0)
    f = rng.random((9, 5))
    row = ps.mixed_wall_row(f, 0.0, 1, top=False)
    opp = np.array(ps.OPP2)
    for q in range(9):
        if ps.C2[q][1] > 0:                     # points back into the fluid
            assert row[q] == pytest.approx(f[opp[q], 1])


def test_mixed_wall_sigma1_is_specular():
    rng = np.random.default_rng(1)
    f = rng.random((9, 5))
    row = ps.mixed_wall_row(f, 1.0, 1, top=False)
    mir = np.array(ps.MIRROR_Y)
    for q in range(9):
        if ps.C2[q][1] > 0:
            assert row[q] == pytest.approx(f[mir[q], 1])


def test_mixed_wall_is_convex_combination():
    rng = np.random.default_rng(2)
    f = rng.random((9, 5))
    s = 0.37
    mix = ps.mixed_wall_row(f, s, 1, top=False)
    bb = ps.mixed_wall_row(f, 0.0, 1, top=False)
    sp = ps.mixed_wall_row(f, 1.0, 1, top=False)
    assert np.allclose(mix, (1.0 - s) * bb + s * sp)


# ---------------------------------------------------------------------------
# plane channel
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def no_slip_channel():
    return ps.channel_poiseuille(0.0, ny=17, tau=TAU, g=G, steps=80000, tol=1e-9)


def test_sigma0_recovers_no_slip_analytic(no_slip_channel):
    """C4: sigma=0 -> no-slip analytic profile within 0.5%."""
    res = no_slip_channel
    assert res.converged
    yc = float(np.mean(res.coord))
    ana = ps.analytic_plane_profile(res.coord, res.length, NU, G, 0.0, center=yc)
    dev = float(np.max(np.abs(res.u - ana)) / ana.max())
    assert dev <= 5e-3
    inv = ps.invert_slip_length(res)
    assert abs(inv["b"]) / res.length <= 5e-3


def test_profile_curvature_is_exact(no_slip_channel):
    """The discrete steady profile is exactly quadratic: B/B_theory = 1."""
    inv = ps.invert_slip_length(no_slip_channel)
    assert inv["curvature_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert inv["fit_residual"] < 1e-9


def test_slip_length_increases_with_sigma():
    bs = []
    for sigma in (0.0, 0.25, 0.5, 0.75):
        res = ps.channel_poiseuille(sigma, ny=17, tau=TAU, g=G, steps=80000, tol=1e-9)
        bs.append(ps.invert_slip_length(res)["b"])
    assert all(b2 > b1 for b1, b2 in zip(bs, bs[1:]))


def test_slip_law_form_and_inverse():
    """b = a sigma/(1-sigma) + c reproduces the data and inverts exactly."""
    sig = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    b = 0.3 * sig / (1.0 - sig) - 0.004
    law = ps.fit_slip_law(sig, b)
    assert law.r2 >= 0.999999
    assert law.monotone
    for s in (0.2, 0.55, 0.77):
        assert law.sigma_of_b(law.b_of_sigma(s)) == pytest.approx(s, rel=1e-10)


def test_slip_law_flags_non_monotone_data():
    sig = np.array([0.0, 0.2, 0.5, 0.8])
    b = np.array([0.0, 0.3, 0.1, 0.9])
    law = ps.fit_slip_law(sig, b)
    assert not law.monotone


def test_planar_closed_loop_inversion():
    """C1: prescribe b via the law, re-invert from the profile, error <= 2%."""
    sig = np.array([0.0, 0.25, 0.5, 0.75, 0.9])
    b = 0.3 * sig / (1.0 - sig)
    law = ps.fit_slip_law(sig, b)
    for target in (1.0, 2.0):
        sigma = law.sigma_of_b(target)
        res = ps.channel_poiseuille(sigma, ny=17, tau=TAU, g=G, steps=120000, tol=1e-9)
        got = ps.invert_slip_length(res)["b"]
        assert abs(got - target) / target <= 0.02


def test_sigma1_is_plug_flow_and_has_no_steady_state():
    """C5: sigma=1 is free slip -> plug flow, body force never balances."""
    res = ps.channel_poiseuille(1.0, ny=17, tau=TAU, g=G, steps=4000,
                                tol=1e-14, track_momentum=True)
    spread = (res.u.max() - res.u.min()) / res.u.max()
    assert spread < 1e-3                    # plug flow
    assert not res.converged                # no steady state under forcing
    assert res.momentum_growth > 0.0        # momentum keeps growing


# ---------------------------------------------------------------------------
# circular pipe (axisymmetric, geometric source)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def no_slip_pipe():
    return ps.pipe_poiseuille(0.0, nr=25, tau=TAU, g=G, steps=200000, tol=1e-9)


def test_pipe_sigma0_matches_hagen_poiseuille(no_slip_pipe):
    res = no_slip_pipe
    assert res.converged
    u_max = G * res.length ** 2 / (4.0 * NU)
    assert abs(res.u[0] - u_max) / u_max <= 0.01
    ana = ps.analytic_pipe_profile(res.coord, res.length, NU, G, 0.0)
    assert float(np.max(np.abs(res.u - ana)) / ana.max()) <= 0.01


def test_pipe_slip_inversion_closed_loop(no_slip_pipe):
    """C2: pipe closed-loop inversion error <= 5%."""
    base = ps.invert_slip_length(no_slip_pipe)["b"]
    sig = np.array([0.0, 0.25, 0.5, 0.75, 0.9])
    b = 0.3 * sig / (1.0 - sig)
    law = ps.fit_slip_law(sig, b)
    target = 1.0
    sigma = law.sigma_of_b(target)
    res = ps.pipe_poiseuille(sigma, nr=25, tau=TAU, g=G, steps=200000, tol=1e-9)
    got = ps.invert_slip_length(res)["b"]
    assert abs(got - target) / target <= 0.05
    # the raw offset is the O(1) axis layer of the annulus fit; it is stable
    assert abs((got - base) - target) / target <= 0.05


# ---------------------------------------------------------------------------
# turbulent flat plate + ledger
# ---------------------------------------------------------------------------
def test_plate_anchors_are_textbook():
    assert ps.plate_local_cf_noslip(1e6) == pytest.approx(0.0592 * 1e6 ** -0.2)
    assert ps.plate_average_cf_noslip(1e7) == pytest.approx(0.074 * 1e7 ** -0.2)
    # delta+ for a 5 m model at 3 m/s and for a ship must bracket the DNS range
    assert 1e2 < ps.delta_plus(6.5e6) < 1e4
    assert ps.delta_plus(1.3e9) > 1e4


def test_drag_reduction_monotone():
    """C6: dCf(b+) monotone for b+ in {0,5,10,20} at both scales."""
    for re_l in (1.0e8, 1.297e9):
        out = ps.drag_reduction_curve([0.0, 5.0, 10.0, 20.0], re_l)
        assert out["monotone"]
        vals = [r["dCf_local_rel"] for r in out["rows"]]
        assert vals[0] == 0.0
        assert all(v2 > v1 for v1, v2 in zip(vals, vals[1:]))
        assert 0.0 < vals[-1] < 0.2


def test_drag_reduction_zero_at_bplus_zero():
    for re_l in (1e6, 1e8, 1e9):
        assert ps.slip_drag_reduction(0.0, re_l) == 0.0


def test_virtual_wall_model_is_negligible():
    """The mean-profile virtual-wall model cannot explain DNS drag reduction."""
    out = ps.virtual_wall_drag_reduction(b_plus_values=(10.0,), re_theta=1.0e4)
    assert out["rows"][0]["dCf_rel"] < 1e-2      # < 1% at b+=10


def test_loglaw_closure_is_consistent():
    rows = ps.validate_loglaw_closure()["rows"]
    for r in rows[:2]:                            # low-Re_theta end: within 10%
        assert abs(r["rel_err_vs_powerlaw"]) < 0.10


def test_ledger_is_self_consistent():
    led = ps.net_benefit_ledger([0.0, 10.0, 50.0], cost_coeffs=(1e-4,))
    assert led["Re_L"] > 1e9
    assert led["P_friction_W"] > 0
    assert "WITHHELD" in led["cost_model_status"]
    assert led["cc_critical_m_s_per_bplus"] > 0
    bp = [r["b_plus"] for r in led["rows"] if r["b_plus"] > 0]
    psaved = [r["P_saved_W"] for r in led["rows"] if r["b_plus"] > 0]
    assert all(p2 > p1 for p1, p2 in zip(psaved, psaved[1:]))
    # physical slip length follows b = b+ nu / u_tau and must stay sub-millimetre
    bmm = [r["b_physical_m"] * 1e3 for r in led["rows"] if r["b_plus"] == 10.0][0]
    assert 0.01 < bmm < 0.2
    # a 1e-4 m/s flux coefficient costs far more than any saving
    assert not led["breakeven"]["0.0001"]["viable"]


def test_sigma_limits_documented():
    lim = ps.sigma_limits()
    assert "bounce-back" in lim["sigma=0"]
    assert "specular" in lim["sigma=1"]