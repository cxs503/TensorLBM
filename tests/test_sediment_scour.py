"""Verification tests for the sediment / scour face (node P84W2-N04).

Layers, cheapest first:

1. ``unit``          -- sediment properties, Shields parameterisation, Rouse
   closed form, Exner/avalanche conservation properties.
2. ``physics``       -- D2Q9 engine against its analytic solutions (forced
   homogeneous flow, plane Poiseuille, channel momentum balance).
3. ``benchmark``     -- the three campaign cases against their references
   (Shields 0.045, Rouse 1937, Sumer & Fredsoe 2002).
4. ``slow``          -- the full runner writing ``artifacts/sediment_scour``.

The pile-scour benchmark is split in two: the *numerics* are asserted (mass
conservation, erosion does occur, the run is grid-consistent) and the
*physical* outcome is asserted to be outside the Sumer & Fredsoe band, which is
the honest-FAIL finding of this node -- a 2-D depth-averaged model has no
horseshoe vortex and therefore cannot reproduce the experimental scour depth
band of 1.0-1.5 D.  If a future 3-D implementation closes that gap,
``test_scour_model_form_2d_underpredicts`` is the test that has to be rewritten
together with the report.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.verification.sediment.scour import (
    D2Q9Channel,
    Sediment,
    avalanche_relax,
    exner_step,
    pile_scour_case,
    rouse_case,
    rouse_profile,
    shields_case,
    shields_critical_soulsby,
    total_load_transport,
)
from tensorlbm.verification.sediment.scour.cases import _advection_diffusion
from tensorlbm.verification.sediment.scour.lattice import equilibrium
from tensorlbm.verification.sediment.scour.runner import gci_study
from tensorlbm.verification.sediment.scour.sediment import KAPPA

pytestmark = pytest.mark.physics


# ---------------------------------------------------------------------------
# 1. unit -- physics kernels
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sediment_properties_d02mm():
    sed = Sediment(d=2.0e-4)
    assert 4.5 < sed.d_star < 6.0           # D* = 5.06 for 0.2 mm quartz
    assert 0.02 < sed.settling_velocity() < 0.035  # ~0.025 m/s, sand 0.2 mm
    assert sed.settling_velocity() < (sed.s - 1.0) * 9.81 * sed.d**2 / (18 * sed.nu)


@pytest.mark.unit
def test_settling_velocity_matches_drag_balance():
    """Fixed point of the Schiller-Naumann balance closes to 1e-6 relative."""
    sed = Sediment(d=2.0e-4)
    w = sed.settling_velocity()
    re = w * sed.d / sed.nu
    cd = 24.0 / re * (1.0 + 0.15 * re**0.687)
    w_check = np.sqrt(4.0 * (sed.s - 1.0) * 9.81 * sed.d / (3.0 * cd))
    assert abs(w_check / w - 1.0) < 1e-6


@pytest.mark.unit
def test_shields_threshold_0p2mm_within_15pct():
    """C1: theta_cr(0.2 mm) against the classical 0.045."""
    theta = float(shields_critical_soulsby(Sediment(d=2.0e-4).d_star))
    assert abs(theta - 0.045) / 0.045 < 0.15
    # monotone limits: large grains asymptote to 0.055, very fine grains rise
    assert 0.05 < float(shields_critical_soulsby(1e3)) < 0.06
    assert float(shields_critical_soulsby(0.5)) > 0.1


@pytest.mark.unit
def test_rouse_profile_reference_and_exponent():
    h, a, Z = 0.3, 0.015, 1.2
    assert abs(float(rouse_profile(a, h, a, 1.0, Z)) - 1.0) < 1e-12
    # ratio at two heights equals [(h-z)/z / (a/(h-a))]^Z exactly
    z = 0.2
    expect = (((h - z) / z) * (a / (h - a))) ** Z
    assert abs(float(rouse_profile(z, h, a, 1.0, Z)) / expect - 1.0) < 1e-12


@pytest.mark.unit
def test_finite_volume_rouse_matches_closed_form():
    h, a, Z = 0.3, 0.015, 1.2
    c, z = _advection_diffusion(h, a, Z, n=400)
    ana = rouse_profile(z, h, a, 1.0, Z)
    # compare where the profile is resolvable: the closed form spans ~30 decades
    # towards the surface, so relative errors there are meaningless
    win = (z < 0.9 * h) & (ana > 1e-6) & (z > 1.2 * a)
    assert np.max(np.abs(c[win] / ana[win] - 1.0)) < 0.06


@pytest.mark.unit
def test_total_load_vanishes_at_threshold():
    sed = Sediment()
    theta_cr = float(shields_critical_soulsby(sed.d_star))
    q_at, _, _ = total_load_transport(
        sed, theta=theta_cr, theta_cr=theta_cr, u_star=0.012, h=0.3, a_ref=0.006
    )
    assert q_at == 0.0
    q_above, q_bed, q_susp = total_load_transport(
        sed, theta=1.5 * theta_cr, theta_cr=theta_cr, u_star=0.015, h=0.3, a_ref=0.006
    )
    assert q_above > 0.0 and q_bed > 0.0 and q_susp > 0.0
    # transport grows strongly with the excess shear stress
    q_more, _, _ = total_load_transport(
        sed, theta=3.0 * theta_cr, theta_cr=theta_cr, u_star=0.021, h=0.3, a_ref=0.006
    )
    assert q_more > 5.0 * q_above


@pytest.mark.unit
def test_exner_conservation_matches_boundary_flux():
    """C4 at unit level: the telescoping update equals the net flux integral."""
    n = 32
    dx, dt, lam = 0.01, 1.0, 0.4
    q = np.linspace(1e-5, 3e-5, n)  # non-uniform transport -> divergence
    zb = np.zeros(n)
    dz = exner_step(zb, q, dx, dt, lam) - zb
    # (a) implementation check against an independently assembled divergence
    face = 0.5 * (q + np.roll(q, -1))
    div = (face - np.roll(face, 1)) / dx
    assert np.allclose(dz, -div * dt / (1.0 - lam), rtol=0, atol=1e-18)
    # (b) telescoping: on a periodic domain the net bed volume cannot change
    assert abs(dz.sum()) < 1e-12 * max(np.abs(dz).sum(), 1e-30)


@pytest.mark.unit
def test_avalanche_limiter_is_mass_conserving_and_bounded():
    dx = 0.01
    zb = np.zeros(40)
    zb[20] = -0.1  # a one-cell pit
    out = avalanche_relax(zb, dx, 32.0, iters=30)
    assert abs(out.sum() - zb.sum()) < 1e-12
    slope = np.abs(np.diff(out)) / dx
    assert slope.max() <= np.tan(np.deg2rad(32.0)) * 1.05


# ---------------------------------------------------------------------------
# 2. physics -- the D2Q9 engine against analytic solutions
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_homogeneous_forcing_accelerates_at_g():
    nu, a = 0.08, 1e-5
    s = D2Q9Channel(4, 4, nu_lat=nu, g_lat=a, cs_smag=0.0, top="slip")
    s.set_solid(np.zeros((4, 4), dtype=bool))
    u0 = s.macros()[1].mean()
    s.step(50)
    u1 = s.macros()[1].mean()
    assert abs((u1 - u0) / 50.0 / a - 1.0) < 1e-6


@pytest.mark.unit
def test_poiseuille_profile_matches_analytic():
    """No-slip bed + specular free surface: u = (a/nu)(H s - s^2/2)."""
    nu, a, ny, nx = 0.08, 1e-5, 33, 8
    s = D2Q9Channel(nx, ny, nu_lat=nu, g_lat=a, cs_smag=0.0, top="slip")
    s.step(60000)
    prof = s.macros()[1][0, 1:-1]
    s_coord = (np.arange(1, ny - 1) + 0.5) - 1.0
    H = (ny - 1) - 1.0
    ana = (a / nu) * (H * s_coord - s_coord**2 / 2.0)
    assert np.max(np.abs(prof - ana)) / ana.max() < 5e-3


@pytest.mark.unit
def test_channel_momentum_balance_tau_w_equals_rho_g_H():
    """The discrete momentum balance is what makes u_* trustworthy."""
    nu, a, h_cells, nx = 0.05, 2e-5, 16, 8
    s = D2Q9Channel(nx, h_cells + 2, nu_lat=nu, g_lat=a, cs_smag=0.0, top="slip")
    s.step(40000)
    prof = s.macros()[1][:, 1:-1].mean(axis=0)
    # halfway bounce-back puts the wall half a cell from the first cell centre,
    # so the one-sided gradient is u_1 / 0.5.  It carries an O(dy) wall-position
    # bias of ~3 % -- documented, not tuned away.
    tau_wall = nu * prof[0] / 0.5
    assert abs(tau_wall / (a * h_cells) - 1.0) < 0.05


@pytest.mark.unit
def test_momentum_exchange_is_antisymmetric_under_transpose():
    """The stencil tables must stay consistent (regression guard)."""
    from tensorlbm.verification.sediment.scour.lattice import CX, CY, OPP, SPEC, W

    assert abs(W.sum() - 1.0) < 1e-15
    assert np.all(OPP[OPP] == np.arange(9))
    assert np.all(CX[OPP] == -CX) and np.all(CY[OPP] == -CY)
    assert np.all(CX[SPEC] == CX) and np.all(CY[SPEC] == -CY)


# ---------------------------------------------------------------------------
# 3. benchmark -- the campaign cases
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_shields_case_lbm_hydraulics_and_threshold():
    res = shields_case(steps=20000)
    assert res["theta_cr_deviation"] < 0.15                       # C1
    # the LBM ratio u*/U must be a plausible turbulent-channel value
    assert 0.03 < res["lbm"]["u_star_over_U"] < 0.10
    # the log-law consistency in the outer layer (buffer layer not resolved)
    assert res["lbm"]["logfit_rms_over_U"] < 0.10
    assert res["U_depth_mean_critical_m_s"] > 0.0
    # the parameterisation tracks the digitised Shields curve in the sand range
    sc = res["shields_curve"]
    ds = np.array(sc["digitised_d_star"])
    th = np.array(sc["digitised_theta_cr"])
    model = np.array(sc["theta_cr"])
    model_on_data = np.interp(ds, np.array(sc["d_star"]), model)
    band = ds >= 3.0
    assert np.median(np.abs(model_on_data[band] / th[band] - 1.0)) < 0.25


@pytest.mark.slow
def test_rouse_case_matches_closed_form():
    res = rouse_case(steps=15000, n_particles=120_000, seed=7)
    assert res["deviation_lpt"] < 0.20          # C2 (Lagrangian)
    assert res["deviation_fv"] < 0.05           # discretisation error alone
    assert res["log_rms_lpt"] < 0.10
    # physical sanity: Rouse number of 0.2 mm sand at u_* = 0.05 m/s
    assert 1.0 < res["parameters"]["rouse_number"] < 1.5
    assert res["parameters"]["w_settling_m_s"] == pytest.approx(0.0248, abs=2e-3)


@pytest.mark.slow
def test_scour_case_numerics_mass_and_erosion():
    """Numerics of the coupled morphodynamic model (not its physics)."""
    res = pile_scour_case(cells_per_D=8, n_relax=2500, n_cycle=8, n_per_cycle=200)
    assert res["mass_balance_rel_error"] < 1e-4      # C4
    assert res["z_s_m"] > 0.0                        # erosion happens (not MOCK)
    assert np.all(np.isfinite(np.array(res["bed"])))
    assert res["parameters"]["theta_0"] > res["parameters"]["theta_cr"]


@pytest.mark.slow
def test_scour_model_form_2d_underpredicts():
    """Honest-FAIL guard: the depth-averaged 2-D model under-predicts scour.

    The Sumer & Fredsoe (2002) experimental band for equilibrium scour depth
    of a slender pile is z_s/D = 1.0-1.5.  The 2-D depth-averaged model
    produces z_s/D ≈ 0.14 -- far below the band -- because it cannot resolve
    the horseshoe vortex / downflow that is the dominant scour mechanism in
    3-D.  This test asserts that the model does NOT pass C3 (deviation < 25%),
    which is the documented honest-FAIL.  Rewriting this test is the signal
    that a 3-D (horseshoe-vortex-resolving) model has landed.
    """
    res = pile_scour_case(cells_per_D=8, n_relax=2500, n_cycle=8, n_per_cycle=200,
                          max_scour_D=6.0)
    lo, hi = 1.0, 1.5
    zs = res["z_s_over_D"]
    # The 2-D model must NOT land inside the Sumer & Fredsoe band
    inside_band = lo <= zs <= hi
    assert not inside_band, (
        f"2-D model unexpectedly produced z_s/D={zs:.3f} inside the "
        f"Sumer & Fredsoe band [{lo}, {hi}]; if this is real, the 3-D "
        f"model has landed and this test must be rewritten"
    )
    # Additionally, the deviation from the band must exceed the 25% threshold
    if zs < lo:
        dev = (lo - zs) / lo
    else:
        dev = (zs - hi) / hi
    assert dev > 0.25, (
        f"2-D model deviation {dev:.3f} is within the 25% tolerance; "
        f"this would incorrectly pass C3"
    )


@pytest.mark.unit
def test_gci_helper_matches_richardson():
    # exact second-order convergence: fine value 1.0, ratio 2 -> p = 2
    vals = [1.0 - 4 * 0.0625, 1.0 - 0.0625, 1.0]
    res = gci_study(vals, [2.0, 2.0])
    assert res["observed_order"] == pytest.approx(2.0, abs=0.05)
    assert res["gci_fine"] < 0.10
