"""Wall-closure tests for the diffuse-interface (``contact``/``wall_order``) wall BC.

Anchor: XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER.

The production CAC solver used to impose a Young angle by overwriting the
populations *at the solid nodes* with ``feq(phi_ghost)``.  That is a phase
source: for ``cos(theta) != 0`` the source is not compensated and the drop is
drained through the plate (measured: 91 % of the phase field lost in 800 steps
at ``theta = 60 deg``), so the integration arm of the cross-validation produced
no usable angle.  ``wall_contact_closure="potential"`` keeps the ghost value for
the gradient/Laplacian stencils only (the surface-energy route) and leaves the
solid populations to halfway bounce-back, which conserves ``int(phi)``.

These tests lock both the defect (legacy closure) and the repair (potential
closure), and the end-to-end ``wall_order`` arm used by the xval harness.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm.wetting import metrics as M
from tensorlbm.wetting.wall_order_parameter_2d import analytic_contact_angle


SHAPE = (96, 96)
RADIUS = 18.0
WIDTH = 4.0
SIGMA = 0.02
GRAVITY = 2.0e-6


def _sim(theta_deg: float | None, closure: str, psi_wall: float | None = None):
    mask = cac.solid_plate(SHAPE, 2, "bottom") | cac.solid_plate(SHAPE, 2, "top")
    kw = dict(
        rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=SIGMA, width=WIDTH,
        mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, -GRAVITY),
        gravity_mode="rho", axes="wp", wall_mask=mask,
        wall_contact_closure=closure, device=torch.device("cpu"), dtype=torch.float32,
    )
    if psi_wall is None:
        kw.update(wall_phi_policy="contact", contact_angle_deg=theta_deg)
    else:
        kw.update(wall_phi_policy="wall_order", wall_order_parameter=psi_wall,
                  wall_order_parameter_top=-1.0)
    return cac.CAC2D(SHAPE, **kw)


def _run(sim, steps: int):
    y_wall = 1.5
    theta0 = sim.contact_angle_deg if sim.contact_angle_deg is not None else 90.0
    y0 = y_wall - RADIUS * math.cos(math.radians(theta0))
    phi0 = cac.init_phi_droplet(SHAPE, RADIUS, WIDTH, center=(y0, SHAPE[1] / 2.0))
    sim.initialize(phi0)
    fluid = sim.fluid_mask
    m0 = float((phi0 * fluid).sum())
    diags = cac.run(sim, steps, sample_every=steps)[-1]
    phi = sim.f.sum(dim=0).numpy()
    m1 = float(torch.from_numpy(phi)[fluid].sum())
    return phi, (m1 - m0) / m0, diags, y_wall


def test_unknown_closure_rejected():
    with pytest.raises(ValueError):
        _sim(90.0, "predictor-corrector")


@pytest.mark.parametrize("theta", [60.0, 120.0])
def test_potential_closure_conserves_mass(theta):
    """P: the potential closure conserves int(phi) to <1 % over 600 steps."""
    _, drift, diags, _ = _run(_sim(theta, "potential"), 600)
    assert abs(drift) < 0.01, drift
    assert math.isfinite(diags["u_max_fluid"])


def test_potential_closure_recovers_acute_young_angle():
    """P: theta = 60 deg on the plate is recovered within the 15 % gate."""
    phi, drift, _, y_wall = _run(_sim(60.0, "potential"), 1200)
    fit = M.contact_angle_circle_fit(phi, y_wall)
    cap = M.spherical_cap_inversion(phi, y_wall)
    assert abs(fit["theta_deg"] - 60.0) / 60.0 < 0.05
    assert abs(cap["theta_deg"] - 60.0) / 60.0 < 0.10
    assert abs(drift) < 0.01


def test_legacy_population_closure_drains_acute_drop():
    """Characterisation of the defect (must not regress silently): the
    population closure loses most of the phase field of a wetting drop."""
    _, drift, _, _ = _run(_sim(60.0, "population"), 600)
    assert drift < -0.3, drift


def test_neutral_wall_is_closure_independent():
    """Control: at theta = 90 deg the wall flux vanishes, both closures agree."""
    phi_p, drift_p, _, y_wall = _run(_sim(90.0, "population"), 400)
    phi_q, drift_q, _, _ = _run(_sim(90.0, "potential"), 400)
    assert abs(drift_p) < 0.01 and abs(drift_q) < 0.01
    th_p = M.contact_angle_circle_fit(phi_p, y_wall)["theta_deg"]
    th_q = M.contact_angle_circle_fit(phi_q, y_wall)["theta_deg"]
    assert abs(th_p - 90.0) < 3.0 and abs(th_q - 90.0) < 3.0


def test_wall_order_arm_end_to_end():
    """P: psi_w -> emergent angle through the production solver, 15 % gate."""
    psi_wall = 0.35
    theta_young = math.degrees(analytic_contact_angle(psi_wall))
    phi, drift, _, y_wall = _run(_sim(None, "potential", psi_wall=psi_wall), 1200)
    fit = M.contact_angle_circle_fit(phi, y_wall)
    cap = M.spherical_cap_inversion(phi, y_wall)
    theta = float((fit["theta_deg"] + cap["theta_deg"]) / 2.0)
    assert abs(theta - theta_young) / theta_young < 0.15, (theta, theta_young)
    assert abs(drift) < 0.01
