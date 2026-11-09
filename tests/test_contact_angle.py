"""Wetting (contact-angle) boundary condition and its measurement (P143W2-N01).

What is testable without a long relaxation:

* Young's relation ``cos(theta) = (psi_w - psi_g)/(psi_l - psi_g)`` and its
  inverse are exact inverses of each other, with the correct limits;
* the initial sessile-drop profile carries the requested 2-D area;
* the measurement operator (``contact_angle_from_mask``, a *linear* least-squares
  fit of the cap profile) is validated against analytic caps, which is what fixes
  its instrument bias (reported in the campaign artifact);
* a short wall-bounded relaxation conserves mass (the half-way bounce-back does
  not leak) and produces a finite, in-range angle.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.sc_surface_tension import (  # noqa: E402
    RHO_L,
    RHO_V,
    build_stencil,
    contact_angle_from_mask,
    contact_angle_from_wall_psi,
    psi_exp,
    sessile_rho,
    step_sc,
    wall_psi_from_contact_angle,
)


def test_young_relation_round_trip():
    for th in (10.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0):
        psi_w = wall_psi_from_contact_angle(th)
        back = contact_angle_from_wall_psi(psi_w)
        assert back == pytest.approx(th, abs=1e-3)


def test_young_relation_limits():
    psi_l = float(psi_exp(torch.tensor(RHO_L)))
    psi_g = float(psi_exp(torch.tensor(RHO_V)))
    assert wall_psi_from_contact_angle(0.0) == pytest.approx(psi_l, abs=1e-9)
    assert wall_psi_from_contact_angle(90.0) == pytest.approx(psi_g, abs=1e-9)
    assert wall_psi_from_contact_angle(180.0) == pytest.approx(2 * psi_g - psi_l, abs=1e-9)
    # monotone decrease of the wall potential with the angle
    ws = [wall_psi_from_contact_angle(t) for t in (0, 45, 90, 135, 180)]
    assert all(ws[i] > ws[i + 1] for i in range(len(ws) - 1))


@pytest.mark.parametrize("theta", [45.0, 60.0, 90.0, 120.0])
def test_sessile_initial_profile_has_the_requested_area(theta):
    ny, nx, wall = 96, 160, 2
    area = math.pi * 10.0 ** 2
    rho = sessile_rho(ny, nx, float(wall - 1), theta, area)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:wall, :] = True
    solid[ny - wall:, :] = True
    fluid = ~solid
    # the diffuse profile is integrated with the liquid fraction
    frac = torch.clamp((rho - RHO_V) / (RHO_L - RHO_V), 0.0, 1.0)
    got = float((frac * fluid).sum())
    assert abs(got - area) / area < 0.08, (got, area)


@pytest.mark.parametrize("theta_true", [45.0, 60.0, 90.0, 120.0])
def test_contact_angle_operator_on_analytic_caps(theta_true):
    ny, nx, wall, R = 96, 160, 2, 30.0
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:wall, :] = True
    solid[ny - wall:, :] = True
    y = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    x = torch.arange(nx, dtype=torch.float32).view(1, nx).expand(ny, nx)
    zc = -R * math.cos(math.radians(theta_true))
    mask = ((x - 0.5 * (nx - 1)) ** 2 + (y - (wall - 1 + 0.5 + zc)) ** 2) < R * R
    rho = torch.where(mask, torch.tensor(1.9), torch.tensor(0.15))
    q = contact_angle_from_mask(rho, solid)
    assert q["status"] == "ok", q
    assert abs(q["theta_deg"] - theta_true) < 2.5, (q["theta_deg"], theta_true)
    assert q["R_fit"] == pytest.approx(R, rel=0.05)


def test_angle_moves_the_right_way_with_wall_potential():
    """Denser wall potential -> more wetting -> smaller angle (synthetic caps)."""
    ny, nx, wall = 96, 160, 2
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:wall, :] = True
    solid[ny - wall:, :] = True
    y = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    x = torch.arange(nx, dtype=torch.float32).view(1, nx).expand(ny, nx)
    angles = []
    for th in (120.0, 75.0, 45.0):
        R = 30.0
        zc = -R * math.cos(math.radians(th))
        mask = ((x - 0.5 * (nx - 1)) ** 2 + (y - (wall - 1 + 0.5 + zc)) ** 2) < R * R
        rho = torch.where(mask, torch.tensor(1.9), torch.tensor(0.15))
        angles.append(contact_angle_from_mask(rho, solid)["theta_deg"])
    assert angles[0] > angles[1] > angles[2], angles


def test_short_wall_relaxation_conserves_mass():
    ny, nx, wall = 48, 80, 2
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:wall, :] = True
    solid[ny - wall:, :] = True
    psi_w = wall_psi_from_contact_angle(60.0)
    rho0 = sessile_rho(ny, nx, float(wall - 1), 60.0, math.pi * 9.0 ** 2)
    rho0 = torch.where(solid, torch.full_like(rho0, RHO_L), rho0)
    z = torch.zeros_like(rho0)
    f = equilibrium(rho0, z, z)
    fluid = ~solid
    m0 = float((rho0 * fluid).sum())
    st = build_stencil("d2q9")
    for _ in range(300):
        f = step_sc(f, stencil=st, solid=solid, wall_psi=psi_w, bounce_back=True)
    rho = f.sum(dim=0)
    assert bool(torch.isfinite(rho).all())
    m1 = float((rho * fluid).sum())
    assert abs(m1 - m0) / m0 < 1e-3, (m0, m1)
    q = contact_angle_from_mask(rho, solid)
    assert q["status"] in ("ok", "insufficient-rows", "no-cap-height")
    if q["status"] == "ok":
        assert 0.0 < q["theta_deg"] < 180.0


def _gate() -> dict | None:
    p = REPO / "artifacts" / "sc_surface_tension" / "laplace_gate.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_contact_artifact_is_honest():
    """The gate must report the contact-angle status it actually achieved, and the
    *instrument* (analytic-cap) bias must be inside the 3 deg tolerance even when
    the realised model angle is not."""
    gate = _gate()
    if gate is None:
        pytest.skip("campaign artifact not present")
    c = gate["contact"]
    assert c["n_valid"] >= 3, c["n_valid"]
    assert c["instrument_bias_max_abs_deg"] < 3.0, c["instrument_bias_max_abs_deg"]
    # honest reporting: the pass flags must be present and consistent
    crit = gate["criteria"]
    assert "c_pass_3deg_young" in crit
    if crit["c_pass_3deg_young"]:
        assert c["young_max_abs_error_deg"] < 3.0
    else:
        assert c["young_max_abs_error_deg"] >= 3.0