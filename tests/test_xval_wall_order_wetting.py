"""Tests for the wall-order-parameter wetting route (XVAL anchor).

Three independent layers, each able to fail on its own:

1. **Math** — the Young law ``cos(theta) = (3 psi_w - psi_w^3)/2`` is re-derived
   here by numerically integrating the truncated-profile wall excess energy
   difference (an *independent route* to the same relation: energy integral vs
   the module's closed form), and the module's ``tanh`` profile / surface tension
   identities are checked.
2. **Module** — the static relaxation of
   :mod:`tensorlbm.wetting.wall_order_parameter_2d` reproduces the law at
   ``psi_w = 0`` (neutral wall) and carries the degeneracy flag for shallow caps.
3. **Integration** — the production solver :class:`tensorlbm.cac_lbm2d.CAC2D`
   accepts ``wall_phi_policy="wall_order"``, derives the wall flux from the same
   ``young_cos_theta`` and produces wall ghosts *bit-identical* to the
   ``"contact"`` policy with the corresponding prescribed angle.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm import cac_lbm2d as cac  # noqa: E402
from tensorlbm.wetting import wall_order_parameter_2d as W  # noqa: E402
from tensorlbm.wetting import ladder as L  # noqa: E402


# ---------------------------------------------------------------------------
# 1. math: independent derivation of the law
# ---------------------------------------------------------------------------
def _wall_excess_energy(psi_wall: float, psi_bulk: float, sigma: float, xi: float) -> float:
    """Truncated-profile wall excess energy ``G = int_0^inf [f + k/2 psi'^2 - f_bulk] dy``.

    Using the first integral ``k/2 psi'^2 = f - f_bulk`` the integrand is
    ``2 (f - f_bulk)`` and ``dy = xi dpsi / (1 - psi^2)`` for the ``tanh`` profile,
    so ``G = (a xi / 2) int (1 - psi^2) dpsi`` between the wall value and the bulk.
    """
    a, _kappa = W.free_energy_parameters(sigma, xi)
    lo, hi = (psi_wall, psi_bulk) if psi_bulk > psi_wall else (psi_bulk, psi_wall)
    psi = np.linspace(lo, hi, 20001, endpoint=True)
    integrand = 1.0 - psi**2
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float((a * xi / 2.0) * trapz(integrand, psi))


@pytest.mark.parametrize("psi_wall", [-0.9, -0.5, -0.1, 0.0, 0.1, 0.35, 0.5, 0.7, 0.9])
def test_young_law_from_wall_excess_energy(psi_wall: float) -> None:
    """``sigma cos(theta)`` equals the wall excess energy difference (independent route)."""
    sigma, xi = 0.05, 3.0
    diff = _wall_excess_energy(psi_wall, -1.0, sigma, xi) - _wall_excess_energy(psi_wall, 1.0, sigma, xi)
    assert abs(diff / sigma - W.young_cos_theta(psi_wall)) < 2e-5


def test_young_law_limits() -> None:
    assert abs(math.degrees(W.analytic_contact_angle(0.0)) - 90.0) < 1e-12
    assert abs(W.analytic_contact_angle(1.0)) < 1e-9
    assert abs(math.degrees(W.analytic_contact_angle(-1.0)) - 180.0) < 1e-9
    assert abs(math.degrees(W.analytic_contact_angle(0.7)) - 28.5381) < 1e-3
    with pytest.raises(ValueError):
        W.analytic_contact_angle(1.5)


def test_profile_and_surface_tension_identities() -> None:
    sigma, xi = 0.04, 2.5
    a, kappa = W.free_energy_parameters(sigma, xi)
    assert abs(W.model_surface_tension(kappa, xi) - sigma) < 1e-12
    assert abs(a - 2.0 * kappa / xi**2) < 1e-12
    # the tanh profile implied by the free energy has a 10-90 thickness of 2 atanh(0.8) xi
    yy = np.linspace(-8.0, 8.0, 40001) * xi
    pp = np.tanh(yy / xi)
    y80 = float(np.interp(0.8, pp, yy))
    ym80 = float(np.interp(-0.8, pp, yy))
    assert abs((y80 - ym80) - 2.0 * math.atanh(0.8) * xi) < 0.02 * xi


def _lap1d(f: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(f)
    out[1:-1] = f[2:] - 2.0 * f[1:-1] + f[:-2]
    return out


# ---------------------------------------------------------------------------
# 2. module: static relaxation
# ---------------------------------------------------------------------------
def test_relaxation_reproduces_neutral_wall() -> None:
    cfg = W.WallOrderWettingConfig(ny=48, nx=64, psi_wall=0.0, psi_wall_top=-1.0,
                                   interface_width=3.0, surface_tension=0.05)
    out = W.relax_to_equilibrium(cfg, radius=12.0, centre_y=12.0, max_iter=4000, omega=0.4)
    m = W.contact_angle_from_field(out["psi"], psi_wall=0.0, interface_width=3.0)
    assert abs(m["theta_cap_deg"] - 90.0) < 5.0
    assert out["mass_error"] < 1e-12


def test_cap_measurability_flag_separates_window() -> None:
    """``psi_w = 0.5`` keeps the cap measurable, ``psi_w = 0.7`` does not (measured)."""
    measured = {}
    for psi_wall in (0.5, 0.7):
        cfg = W.WallOrderWettingConfig(ny=56, nx=80, psi_wall=psi_wall, psi_wall_top=-1.0,
                                       interface_width=3.0, surface_tension=0.05)
        out = W.relax_to_equilibrium(cfg, radius=16.0, centre_y=16.0, max_iter=20000, omega=0.4)
        measured[psi_wall] = W.contact_angle_from_field(out["psi"], psi_wall=psi_wall,
                                                        interface_width=3.0)
    assert measured[0.5]["cap_measurable"] is True
    assert measured[0.5]["layer_fraction"] < 0.5
    assert measured[0.7]["cap_measurable"] is False
    assert measured[0.7]["layer_fraction"] > 0.9


# ---------------------------------------------------------------------------
# 3. integration: production CAC solver
# ---------------------------------------------------------------------------
def test_cac_wall_order_policy_derives_the_young_angle() -> None:
    sim = cac.CAC2D((32, 32), rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=0.02, width=4.0,
                    wall_mask=cac.solid_plate((32, 32), 2, "bottom"),
                    wall_phi_policy="wall_order", wall_order_parameter=0.35, axes="wp")
    assert abs(sim.wall_contact_cos - W.young_cos_theta(0.35)) < 1e-12
    assert abs(sim.contact_angle_deg - math.degrees(W.analytic_contact_angle(0.35))) < 1e-9
    with pytest.raises(ValueError):
        cac.CAC2D((32, 32), rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=0.02, width=4.0,
                  wall_phi_policy="wall_order")
    with pytest.raises(ValueError):
        cac.CAC2D((32, 32), rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=0.02, width=4.0,
                  wall_phi_policy="neumann", wall_order_parameter=0.2)


def test_cac_wall_order_ghost_matches_prescribed_angle_policy() -> None:
    """Same emergent angle -> bit-identical wall ghosts as ``wall_phi_policy='contact'``."""
    psi_wall = 0.35
    theta = math.degrees(W.analytic_contact_angle(psi_wall))
    mask = cac.solid_plate((40, 40), 2, "bottom") | cac.solid_plate((40, 40), 2, "top")
    kw = dict(rho_l=1.0, rho_g=0.1, mu_l=0.1, mu_g=0.01, sigma=0.02, width=4.0, axes="wp",
              wall_mask=mask)
    a = cac.CAC2D((40, 40), wall_phi_policy="contact", contact_angle_deg=theta, **kw)
    b = cac.CAC2D((40, 40), wall_phi_policy="wall_order", wall_order_parameter=psi_wall,
                  wall_order_parameter_top=psi_wall * 1.0, **kw)
    phi = cac.init_phi_droplet((40, 40), 10.0, 4.0, center=(2.0, 20.0)).to(torch.float32)
    a.initialize(phi)
    b.initialize(phi)
    a.refresh_solid_ghosts()
    b.refresh_solid_ghosts()
    assert torch.allclose(a.f, b.f, rtol=0.0, atol=1e-7)


def test_ladder_contact_wall_order_entry_runs() -> None:
    """The ladder entry point (integration caller) returns a JSON-serialisable dict."""
    res = L.contact_wall_order(0.35, steps=60, shape=(48, 48), radius=8.0)
    assert res["case"] == "contact_wall_order"
    assert res["inputs"]["wall_phi_policy"] == "wall_order"
    assert abs(res["theta_young_deg"] - math.degrees(W.analytic_contact_angle(0.35))) < 1e-9
    assert math.isfinite(res["u_max"])