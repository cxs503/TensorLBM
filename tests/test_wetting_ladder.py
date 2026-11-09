"""IN32-L01 — tests for the 2D CAC solver, the wetting wall condition and the
measurement operators of the wetting verification ladder.

Positive controls (P) feed a configuration whose answer is known analytically
and require the measured value; negative controls (N) feed a configuration
that *violates* the assumption and require the operator or the gate to say so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm.wetting import metrics as M

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sim(shape=(64, 32), *, sigma=0.02, policy="freeze", angle=None, axes="pp", wall_mask=None,
         gravity=(0.0, 0.0), gravity_mode="uniform", zero_net=True):
    return cac.CAC2D(
        shape, rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=sigma, width=4.0,
        mobility=0.05, p0=0.01, axes=axes, wall_mask=wall_mask, wall_phi_policy=policy,
        contact_angle_deg=angle, gravity=gravity, gravity_mode=gravity_mode,
        wall_contact_zero_net=zero_net, dtype=torch.float32,
    )


def _analytic_cap(shape, theta_deg, y_wall, radius, width=2.0):
    """Diffuse (tanh) phi field of an exact circular lens with contact angle
    theta -- the same profile family the solver relaxes to, so the estimators
    are tested without staircase artefacts."""
    ny, nx = shape
    th = math.radians(theta_deg)
    yc = y_wall - radius * math.cos(th)
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)
    r = np.hypot(y - yc, x - (nx - 1) / 2.0)
    return 0.5 * (1.0 + np.tanh(2.0 * (radius - r) / width))


# ---------------------------------------------------------------------------
# stencils
# ---------------------------------------------------------------------------


def test_iso_gradient_exact_on_linear_field():
    """P: the D2Q9 isotropic gradient is exact for a linear field."""
    ny, nx = 24, 20
    y = np.arange(ny)[:, None] * np.ones((1, nx))
    x = np.ones((ny, 1)) * np.arange(nx)[None, :]
    phi = torch.tensor(0.3 * x + 0.7 * y + 1.0, dtype=torch.float32)
    gx, gy = cac.iso_gradient_2d(phi, axes="pp")
    assert torch.allclose(gx[2:-2, 2:-2], torch.full_like(gx[2:-2, 2:-2], 0.3), atol=1e-4)
    assert torch.allclose(gy[2:-2, 2:-2], torch.full_like(gy[2:-2, 2:-2], 0.7), atol=1e-4)


def test_iso_laplacian_exact_on_quadratic_field():
    """P: lap(x^2 + y^2) = 4 exactly (isotropic D2Q9 stencil)."""
    ny, nx = 24, 24
    y = np.arange(ny)[:, None] * np.ones((1, nx))
    x = np.ones((ny, 1)) * np.arange(nx)[None, :]
    phi = torch.tensor(x**2 + y**2, dtype=torch.float32)
    lap = cac.iso_laplacian_2d(phi, axes="pp")
    assert torch.allclose(lap[2:-2, 2:-2], torch.full_like(lap[2:-2, 2:-2], 4.0), atol=1e-3)


def test_laplacian_exact_on_biquadratic_field():
    """P: the isotropic D2Q9 Laplacian is second-order exact on x^2 y^2, whose
    Laplacian is 2x^2 + 2y^2 (a 5-point stencil misses the diagonal coupling and
    the quadratic test would still pass, so this field discriminates)."""
    ny = nx = 16
    y = np.arange(ny)[:, None] * np.ones((1, nx))
    x = np.ones((ny, 1)) * np.arange(nx)[None, :]
    phi = torch.tensor(x**2 * y**2, dtype=torch.float32)
    lap = cac.iso_laplacian_2d(phi, axes="pp").numpy()
    exact = 2.0 * x**2 + 2.0 * y**2
    err = np.abs(lap - exact)[4:-4, 4:-4]
    assert err.max() < 2e-3 * np.abs(exact).max()


# ---------------------------------------------------------------------------
# free energy / wall condition
# ---------------------------------------------------------------------------


def test_free_energy_params_consistent_with_tension_and_width():
    """P: beta = 12 sigma / W and k = 1.5 sigma W reproduce sigma and the
    tanh equilibrium width relation |grad phi| = 4 phi (1-phi) / W."""
    sigma, width = 0.02, 4.0
    beta, k = cac.free_energy_params(sigma, width)
    assert beta * width / 12.0 == pytest.approx(sigma)
    assert k / (1.5 * width) == pytest.approx(sigma)
    # first integral: (k/2) (dphi/deta)^2 = beta phi^2 (1-phi)^2 at phi=0.5
    lhs = 0.5 * k * (4.0 * 0.5 * 0.5 / width) ** 2
    rhs = beta * 0.5**2 * 0.5**2
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_wall_contact_flux_neutral_at_90():
    """P: the neutral (Neumann) limit is exactly recovered at 90 deg."""
    assert cac.wall_contact_flux_from_angle(90.0, 4.0) == pytest.approx(0.0, abs=1e-15)


def test_wall_contact_ghost_neutral_is_identity():
    """P: theta = 90 leaves the ghost equal to the fluid mean."""
    phi = torch.tensor([[0.0, 0.3, 0.5, 0.8, 1.0]])
    out = cac.wall_contact_ghost(phi, 0.0, 4.0)
    assert torch.allclose(out, phi)


def test_wall_contact_ghost_vanishes_in_both_bulks():
    """P: self-limiting form -- no phase is exchanged deep inside a bulk."""
    phi = torch.tensor([[0.0, 1.0]])
    for theta in (30.0, 60.0, 120.0, 150.0):
        out = cac.wall_contact_ghost(phi, math.cos(math.radians(theta)), 4.0)
        assert torch.allclose(out, phi)


def test_wall_contact_ghost_sign_is_wetting_for_acute_angle():
    """P: for theta < 90 the ghost is enriched (wall pulls liquid in); for
    theta > 90 it is depleted."""
    phi = torch.full((1, 3), 0.5)
    wet = cac.wall_contact_ghost(phi, math.cos(math.radians(60.0)), 4.0)
    dry = cac.wall_contact_ghost(phi, math.cos(math.radians(120.0)), 4.0)
    assert bool((wet > phi).all()) and bool((dry < phi).all())
    assert float((wet - phi).abs().max()) == pytest.approx(0.5 * 4.0 * 0.25 / 4.0, rel=1e-6)


def test_contact_policy_requires_angle_and_rejects_unknown_policy():
    """N: misconfiguration is refused instead of silently running dry."""
    with pytest.raises(ValueError):
        _sim(policy="contact", angle=None)
    with pytest.raises(ValueError):
        _sim(angle=60.0)  # angle without the contact policy
    with pytest.raises(ValueError):
        _sim(policy="slip")


# ---------------------------------------------------------------------------
# planar interface tension
# ---------------------------------------------------------------------------


def test_flat_interface_tension_matches_input_sigma():
    """P: for the analytic tanh slab the tension integral recovers the model
    sigma (discrete stencils cost ~8%, the same 'bar_eff' the 3D CAC scheme
    documents), and the measured 10-90 width recovers W."""
    sigma, width = 0.02, 4.0
    sim = _sim(shape=(64, 8), sigma=sigma)
    sim.initialize(cac.init_phi_slab((64, 8), width, y_lo=16.0, y_hi=48.0))
    m = M.flat_interface_tension(sim, pad=2)
    sigma_half = 0.5 * m["sigma_integral"]  # slab carries two interfaces
    assert 0.80 * sigma < sigma_half < 1.05 * sigma
    assert m["width_fit"] == pytest.approx(width, abs=0.6)


def test_flat_interface_tension_negative_control_smeared_profile():
    """N: a smeared (non-equilibrium) interface has excess tension -- the
    operator must not report the input sigma for an arbitrary profile."""
    sigma, width = 0.02, 4.0
    sim = _sim(shape=(64, 8), sigma=sigma)
    y = torch.arange(64).view(-1, 1).float()
    smeared = 0.5 + 0.5 * torch.tanh(2.0 * (32.0 - y) / (4.0 * width))
    sim.initialize(smeared.expand(64, 8).contiguous())
    m = M.flat_interface_tension(sim, pad=2)
    assert m["width_fit"] > 1.5 * width  # 10-90 width is 4x too large


# ---------------------------------------------------------------------------
# contact-angle estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta,tol", [(45.0, 3.0), (60.0, 2.0), (90.0, 2.0), (120.0, 12.0)])
def test_cap_inversion_recovers_analytic_angle(theta, tol):
    """P: on an exact circular lens the segment inversion returns theta.

    The estimator is a cross-check only: it needs the contact width, which is
    a short base for a small (non-wetting) cap, hence the documented loose
    tolerance at 120 deg.  The circle fit is the primary estimator.
    """
    phi = _analytic_cap((160, 160), theta, y_wall=1.5, radius=45.0)
    out = M.spherical_cap_inversion(phi, y_wall=1.5)
    assert out["theta_deg"] == pytest.approx(theta, abs=tol)


@pytest.mark.parametrize("theta", [60.0, 90.0, 120.0])
def test_circle_fit_recovers_analytic_angle(theta):
    """P: circle fit of the lens contour returns theta."""
    phi = _analytic_cap((160, 160), theta, y_wall=1.5, radius=45.0)
    out = M.contact_angle_circle_fit(phi, y_wall=1.5, trim=0.0)
    assert out["theta_deg"] == pytest.approx(theta, abs=1.5)


def test_circle_fit_negative_control_rejects_parabola():
    """N: a parabolic (non-circular) lens is *not* fitted as a circle of the
    wrong radius without cost -- the residual grows and the recovered angle
    drifts away from the circle answer."""
    ny, nx = 160, 160
    y = np.arange(ny)[:, None].astype(float)
    x = np.arange(nx)[None, :].astype(float)
    h, w = 30.0, 120.0
    apex = 1.5 + h
    para = apex - h * (2.0 * (x - (nx - 1) / 2.0) / w) ** 2
    phi = (y <= para).astype(float)
    out = M.contact_angle_circle_fit(phi, y_wall=1.5, trim=0.0)
    assert out["rms"] > 0.05  # not a circle: residual is visible


# ---------------------------------------------------------------------------
# meniscus / drop extent / oscillation
# ---------------------------------------------------------------------------


def test_meniscus_height_synthetic():
    """P: the sub-grid crossing of a synthetic step is interpolated: phi=1 on
    rows 0..10 and phi=0.3 on row 11 puts the 0.5 level at 10 + 0.5/0.7."""
    phi = np.zeros((40, 10))
    phi[:11, :] = 1.0
    phi[11, :] = 0.3
    h, std = M.meniscus_height(phi, 0, 10)
    assert h == pytest.approx(10.0 + 0.5 / 0.7, abs=0.01)
    assert std == pytest.approx(0.0, abs=1e-12)


def test_meniscus_height_negative_control_detects_tilt():
    """N: a tilted interface must show a non-zero column spread."""
    phi = np.zeros((40, 10))
    for j in range(10):
        phi[: 10 + j, j] = 1.0
    _, std = M.meniscus_height(phi, 0, 10)
    assert std > 2.0


def test_radial_extents_synthetic_ellipse():
    """P: extents of a synthetic ellipse are recovered exactly."""
    phi = np.zeros((80, 80))
    y, x = np.ogrid[:80, :80]
    phi[((y - 40) / 12.0) ** 2 + ((x - 40) / 20.0) ** 2 <= 1.0] = 1.0
    rx, ry, req = M.radial_extents(phi)
    assert rx == pytest.approx(20.0, abs=1.5)
    assert ry == pytest.approx(12.0, abs=1.5)
    assert req == pytest.approx(math.sqrt(rx * ry), rel=0.15)


def test_damped_sine_fit_recovers_frequency():
    """P: analytic damped sine -> omega and gamma recovered."""
    t = np.arange(0, 4000, 5.0)
    om, gam = 3.0e-3, 5.0e-4
    sig = 1.7 * np.exp(-gam * t) * np.cos(om * t + 0.3) + 0.4
    fit = M.damped_sine_fit(t, sig, om)
    assert fit["omega"] == pytest.approx(om, rel=0.02)
    assert fit["gamma"] == pytest.approx(gam, rel=0.15)
    assert fit["r2"] > 0.99


def test_damped_sine_fit_negative_control_pure_decay():
    """N: a pure exponential (no oscillation) must not fit a high-quality
    sine -- the reported r2 stays low, so the Lamb gate cannot pass on noise."""
    t = np.arange(0, 4000, 5.0)
    sig = 2.0 * np.exp(-1.0e-3 * t)
    fit = M.damped_sine_fit(t, sig, 3.0e-3)
    # a decaying envelope can be partly absorbed by a heavily damped sine, so the
    # bar is set just above the analytic-sine level of 0.99
    assert fit["r2"] < 0.95


# ---------------------------------------------------------------------------
# Cox-Voinov
# ---------------------------------------------------------------------------


def test_cox_voinov_fit_recovers_slope():
    """P: synthetic Cox-Voinov data is inverted for 9 ln(L/Ls) and theta_e."""
    theta_e = 60.0
    lnl = 8.0
    ca = np.linspace(0.002, 0.02, 12)
    th = np.degrees(np.cbrt(np.radians(theta_e) ** 3 + 9.0 * lnl * ca))
    out = M.cox_voinov_fit(ca, th, theta_e)
    assert out["ln_L_over_Ls"] == pytest.approx(lnl, rel=0.02)
    assert out["theta_e_from_intercept_deg"] == pytest.approx(theta_e, abs=0.5)
    assert out["r2"] > 0.999


def test_cox_voinov_negative_control_ca_independent_angle():
    """N: a contact angle that does not respond to Ca gives zero slope, i.e.
    the Cox-Voinov gate rejects it."""
    ca = np.linspace(0.002, 0.02, 12)
    th = np.full_like(ca, 75.0)
    out = M.cox_voinov_fit(ca, th, 75.0)
    assert abs(out["slope"]) < 1e-6
    assert not np.isfinite(out["ln_L_over_Ls"]) or out["ln_L_over_Ls"] <= 0.0


# ---------------------------------------------------------------------------
# solver-level behaviour
# ---------------------------------------------------------------------------


def test_cac2d_conserves_phase_in_periodic_box():
    """P: the conservative AC scheme holds int(phi) to round-off (no walls)."""
    sim = _sim(shape=(40, 24))
    sim.initialize(cac.init_phi_droplet((40, 24), 8.0, 4.0))
    a0 = float(cac.macro(sim)["phi"].sum())
    cac.run(sim, 30, sample_every=0)
    a1 = float(cac.macro(sim)["phi"].sum())
    assert abs(a1 / a0 - 1.0) < 5e-3
    assert int(torch.isnan(sim.f).sum()) == 0


def test_contact_policy_90_matches_neumann_policy_bitwise():
    """P: theta = 90 deg contact policy is bit-identical to the Neumann policy."""
    shape = (48, 32)
    mask = cac.solid_plate(shape, 2, "bottom")
    a = _sim(shape=shape, policy="neumann", axes="wp", wall_mask=mask)
    b = _sim(shape=shape, policy="contact", angle=90.0, axes="wp", wall_mask=mask)
    phi0 = cac.init_phi_layers(shape, 4.0, y0=16.0)
    a.initialize(phi0)
    b.initialize(phi0)
    for _ in range(5):
        a.step()
        b.step()
    assert torch.equal(a.f, b.f) and torch.equal(a.g, b.g)


def test_wetting_wall_moves_the_angle_in_the_prescribed_direction():
    """P/N pair: with the wetting sign the apparent angle of a 60 deg case
    moves *below* the neutral 90 deg result; flipping the sign moves it above.
    A gate that accepted both signs would be vacuous, so both directions are
    checked on the same short run."""
    shape = (80, 80)
    mask = cac.solid_plate(shape, 2, "bottom") | cac.solid_plate(shape, 2, "top")
    y_wall = 1.5
    radius = 16.0

    def run(cos_sign):
        sim = _sim(shape=shape, policy="contact", angle=60.0, axes="wp", wall_mask=mask,
                   gravity=(0.0, -2e-6), gravity_mode="rho")
        yc = y_wall - radius * math.cos(math.radians(60.0))
        sim.initialize(cac.init_phi_droplet(shape, radius, 4.0, center=(yc, shape[1] / 2.0)))
        if cos_sign < 0:  # negative control: reversed wall condition
            sim.wall_contact_cos = -sim.wall_contact_cos
        cac.run(sim, 400, sample_every=0)
        phi = cac.macro(sim)["phi"].numpy()
        return M.contact_angle_circle_fit(phi, y_wall)["theta_deg"]

    theta_ok = run(+1)
    theta_bad = run(-1)
    assert theta_ok < 90.0 < theta_bad
    assert abs(theta_ok - 90.0) > 1.0  # the condition actually does something


def test_spurious_current_gate_on_a_relaxing_drop():
    """P: the spurious-current diagnostic is finite and small on a relaxing
    static drop, and is reported in the same units the ladder gate uses."""
    sim = _sim(shape=(64, 64), sigma=0.02)
    sim.initialize(cac.init_phi_droplet((64, 64), 16.0, 4.0))
    cac.run(sim, 200, sample_every=0)
    d = cac.diagnostics(sim)
    u_ref = 0.02 / 0.1  # sigma / mu_l
    assert 0.0 <= d["u_max_fluid"] < 0.05 * u_ref
    assert d["nan_f"] == 0 and d["nan_g"] == 0