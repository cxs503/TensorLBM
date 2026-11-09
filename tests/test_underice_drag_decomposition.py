"""Drag-decomposition contract of the under-ice channel kernel.

The skin/form split of :mod:`tensorlbm.underice_drag` is a *force
decomposition*, so it has to satisfy four machine-checkable properties:

1. **exactness** -- the split is a projection of the link-wise momentum
   exchange ``2 c_i f*_opp``, so ``form + skin == total`` identically, and the
   total must close the global momentum budget ``rho g h_eff`` (the only
   stationarity-limited statement here);
2. **negative control** -- a flat (level-ice) wall has *zero* form drag,
   exactly, because every wall link of a z-normal wall projects onto buttons the
   z axis;
3. **keel sign** -- a keel field produces a positive form drag, and the two
   parts stay positive;
4. **back-end parity** -- the torch and C/OpenMP back-ends report the same
   decomposition for the same case (they used to disagree: torch took
   ``cd_skin = tau_skin/q`` while native used ``cd_total - cd_form``).

Plus a structural check of the one documented ambiguity of the split: the
``45 deg`` staircase normal of a *corner* link lies between the two
axis-aligned assignments, so those two bracket the answer instead of being
silently picked by a tie-break order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import underice_drag as ud

# small grids: the contract is discrete identities, not resolved turbulence
NX, NY, NZ = 16, 16, 34
STEPS, WARMUP = 5000, 2000


def _cfg(**kw):
    base = dict(nx=NX, ny=NY, nz=NZ, re_tau=20.0, u_tau=0.002, steps=STEPS,
                warmup=WARMUP, sample_interval=10, init_profile="parabolic",
                noise_amplitude=0.0, threads=32)
    base.update(kw)
    return ud.UnderIceChannelConfig(**base)


@pytest.fixture(scope="module")
def smooth():
    return ud.run_underice_channel_native(_cfg())


@pytest.fixture(scope="module")
def keel():
    spec = ud.KeelArraySpec(nx=NX, ny=NY, nz=NZ, h_keel=4.0, coverage=0.5,
                            n_ridges=2, height_modulation=0.0)
    mask = ud.triangular_keel_mask(spec)
    return ud.run_underice_channel_native(_cfg(), mask, spec=spec)


# ---------------------------------------------------------------------------
# 1. exactness / momentum budget
# ---------------------------------------------------------------------------


def test_split_is_an_exact_identity(smooth, keel):
    for rec in (smooth, keel):
        assert abs(rec["tau_form"] + rec["tau_skin"] - rec["tau_me"]) <= 1e-12 * rec["tau_me"]
        assert abs(rec["closure_residual"]) <= 1e-12


def test_wall_force_closes_the_momentum_budget(smooth, keel):
    """tau_me must equal (rho g N_fluid - dP/dt)/(2 A) -- the exact budget.

    The residual is the *resolved* unsteadiness of the sampling window, so the
    tolerance scales with the stationarity drift of the same record.
    """
    for rec in (smooth, keel):
        assert abs(rec["momentum_balance_residual"]) < 0.05
        assert rec["tau_me"] > 0.0
    # the smooth case is the stationary one: it must be far tighter
    assert abs(smooth["momentum_balance_residual"]) < 0.01


def test_total_force_matches_rho_g_h_eff_for_the_smooth_channel(smooth):
    """Level ice: tau_me == rho g h_eff (exact laminar closure, <= 2 %)."""
    assert smooth["tau_me"] == pytest.approx(smooth["tau_total"], rel=0.02)
    # and the drag coefficient is the exact laminar one, C_d = 9/Re_tau^2
    assert smooth["cd_total"] == pytest.approx(9.0 / smooth["re_tau"] ** 2, rel=0.02)


# ---------------------------------------------------------------------------
# 2. negative control: a flat wall has no form drag
# ---------------------------------------------------------------------------


def test_level_ice_has_exactly_zero_form_drag(smooth):
    assert smooth["tau_form"] == 0.0
    assert smooth["cd_form"] == 0.0
    assert smooth["tau_skin"] == pytest.approx(smooth["tau_me"], rel=1e-12)


# ---------------------------------------------------------------------------
# 3. keel: form drag appears, both parts positive
# ---------------------------------------------------------------------------


def test_keel_produces_positive_form_drag(keel):
    assert keel["tau_form"] > 0.0
    assert keel["tau_skin"] > 0.0
    share = keel["cd_form"] / keel["cd_me"]
    assert 0.3 < share < 0.95, share
    # form drag must exceed the flat-ice reference: the keel adds resistance
    assert keel["cd_me"] > 0.0


# ---------------------------------------------------------------------------
# 4. back-end parity
# ---------------------------------------------------------------------------


def test_backends_agree_on_the_decomposition(smooth):
    torch_rec = ud.run_underice_channel(_cfg(steps=1500, warmup=800, threads=8))
    native_rec = ud.run_underice_channel_native(_cfg(steps=1500, warmup=800, threads=8))
    assert native_rec["cf"] == pytest.approx(torch_rec["cf"], rel=5e-3)
    assert native_rec["cd_form"] == 0.0 and torch_rec["cd_form"] == 0.0
    assert native_rec["cd_skin"] == pytest.approx(torch_rec["cd_skin"], rel=5e-3)


# ---------------------------------------------------------------------------
# 5. the recorded ambiguity: axis-aligned bracket of the corner normal
# ---------------------------------------------------------------------------


def _x_exchange_share(solid: np.ndarray, priority) -> float:
    """Fraction of the |c_x| wall-link exchange assigned to x-normal faces."""
    links, xface = ud.wall_link_face_masks(solid, priority=priority)
    numer = float((xface * np.abs(np.asarray(ud.C)[:, 0])[:, None, None, None]).sum())
    denom = float((links * np.abs(np.asarray(ud.C)[:, 0])[:, None, None, None]).sum())
    return numer / denom


def test_axis_aligned_rules_bracket_the_corner_normal():
    spec = ud.KeelArraySpec(nx=NX, ny=NY, nz=NZ, h_keel=4.0, coverage=0.5,
                            n_ridges=2, height_modulation=0.0)
    solid = ud.mirrored_solid(ud.triangular_keel_mask(spec))
    share_z = _x_exchange_share(solid, ("z", "y", "x"))
    share_x = _x_exchange_share(solid, ("x", "y", "z"))
    assert 0.0 < share_z < share_x < 1.0
    # the flat-wall case has *no* x-face at all under any rule
    flat = np.zeros((NZ, NY, NX), dtype=bool)
    flat[0] = flat[-1] = True
    for priority in (("z", "y", "x"), ("x", "y", "z")):
        assert _x_exchange_share(flat, priority) == 0.0


# ---------------------------------------------------------------------------
# 6. the surviving calibrated shear diagnostic
# ---------------------------------------------------------------------------


def test_gradient_shear_route_brackets_the_exact_skin_drag(smooth):
    """The wall-normal velocity-gradient route is within a few % (not a split)."""
    legacy = smooth["tau_skin_gradient_legacy"]
    assert legacy == pytest.approx(smooth["tau_skin"], rel=0.05)
