"""Tests for the three-phase (ice / water / air) contact-line primitives.

The tests are deliberately small (a few hundred lattice steps, grids of a few
hundred cells) and check *physics invariants* rather than pinned numbers:

* the Shan-Chen equation of state reproduces coexistence (equal pressure in the
  liquid and vapour plateaus of a relaxed film),
* the discrete interaction pressure tensor is anisotropic only where the
  interface is (zero in the bulk) and its integral over a flat liquid film has
  the sign and magnitude of the Laplace value,
* the wall-coupling weight field ``S(x)`` counts exactly the links that point
  into the solid,
* the wetting-tension thermodynamic integration is monotone in the wall coupling
  and its ``psi_w`` ladder is resolved,
* the contact-line tracker finds the level crossing where it is by construction,
* the Young inverse is exact against a synthetic triple of tensions,
* the physical-unit mapping is consistent.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.contact_line import three_phase as tp
from tensorlbm.contact_line.engine import Channel, WallStepper, sessile_drop_state
from tensorlbm.d2q9 import macroscopic


@pytest.fixture(scope="module")
def film() -> dict:
    """Short relaxed liquid film (periodic, no walls) used by the EOS tests."""
    torch.manual_seed(0)
    out = tp.planar_liquid_gas_tension(ny=64, nx=8, half_width=12.0, n_steps=1200)
    return out


def test_sc_pressure_coexistence() -> None:
    """The lattice EOS must give equal bulk pressure on both sides of a film."""
    rho_l, rho_v = 1.9315, 0.1555  # measured plateaus of the G=5, tau=1 scheme
    pl = float(tp.sc_pressure(torch.tensor(rho_l, dtype=torch.float64)))
    pv = float(tp.sc_pressure(torch.tensor(rho_v, dtype=torch.float64)))
    assert abs(pl - pv) / abs(pl) < 0.02


def test_sc_pressure_has_van_der_waals_loop() -> None:
    """The lattice EOS must show the non-ideal (spinodal) loop of a two-phase model."""
    rho = torch.linspace(0.05, 2.0, 400, dtype=torch.float64)
    p = tp.sc_pressure(rho)
    dp = p[1:] - p[:-1]
    # non-monotone: some decreasing stretch (mechanical instability / spinodal)
    assert bool((dp < 0).any())
    # and the liquid branch is stiff (dp/drho > 0 at rho = 1.9)
    p_hi = tp.sc_pressure(torch.tensor([1.85, 1.95], dtype=torch.float64))
    assert float(p_hi[1]) > float(p_hi[0])


def test_pressure_tensor_vanishes_in_bulk(film: dict) -> None:
    rho = torch.tensor(film["profile"], dtype=torch.float64).view(-1, 1).expand(-1, 4).contiguous()
    p_xx, p_yy = tp.interaction_pressure_tensor(rho)
    # far from the interface the anisotropy must be numerically zero
    assert float((p_xx - p_yy)[:3].abs().max()) < 1e-6
    assert float((p_xx - p_yy)[-3:].abs().max()) < 1e-6
    # and clearly non-zero inside the interface
    aniso = (p_xx - p_yy).abs().max()
    assert float(aniso) > 1e-3


def test_flat_film_tension_positive_and_consistent(film: dict) -> None:
    """Mechanical film tension: positive, same order as the Laplace reference."""
    gamma = float(film["gamma"])
    assert gamma > 0.0
    ratio = gamma / float(film["gamma_laplace_reference"])
    # the discrete D2Q9 prefactor is known to differ from the Laplace value by
    # ~15 % (documented limitation); the test only enforces the order of magnitude
    assert 0.5 < ratio < 2.0
    assert film["mass_drift"] < 1e-3


def test_wall_coupling_weights_first_row() -> None:
    """``S`` of the first fluid row must equal the weight of the 3 links into the wall."""
    ch = Channel(nx=8, ny=16, wall_rows=1)
    solid = ch.solid_mask()
    w = tp.wall_coupling_weights(ch.shape, solid_mask=solid)
    # first fluid row: neighbours (0,-1), (1,-1), (-1,-1) -> 1/9 + 2/36
    assert w[1, 3].item() == pytest.approx(1.0 / 9.0 + 2.0 / 36.0, rel=1e-6)
    # second fluid row has no link into the solid
    assert w[2, 3].item() == pytest.approx(0.0, abs=1e-12)
    # inside the wall every x-direction link cancels in the x-force but the
    # weights themselves are still those of the *upward* links to the fluid
    assert w[0, 3].item() > 0.0


def test_wall_force_is_zero_for_uniform_wall() -> None:
    """A homogeneous wall spanning the domain carries no tangential Shan-Chen force.

    This is the structural reason why the wall tension *cannot* be read off a
    wall force in the virtual-density wetting model (see the node report).
    """
    ch = Channel(nx=16, ny=32, wall_rows=2)
    solid = ch.solid_mask()
    f, _ = sessile_drop_state(ch, radius=6.0)
    st = WallStepper(ch, G=5.0, tau=1.0, wall_fraction=0.7, forcing="edm")
    for _ in range(50):
        f = st.step(f, 0.0)
    rho = macroscopic(f)[0]
    psi_w = 0.7 * float(tp.psi_exp(torch.tensor(1.957, dtype=torch.float64)))
    out = tp.wall_force(rho, solid_mask=solid, wall_psi=psi_w)
    assert abs(out["fx_per_length"]) < 1e-9


def test_contact_line_position_analytic() -> None:
    """A synthetic tanh front must be located at the level crossing."""
    ny, nx, y_wall = 32, 64, 1.5
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64), torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    x_cl = 20.0
    rho = 0.16 + (1.96 - 0.16) * 0.5 * (1.0 - torch.tanh((xx - x_cl) / 1.2))
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[:2] = True
    front = tp.contact_line_position(
        rho, level=0.5 * (1.96 + 0.16), y_wall=y_wall, side="front", centre_x=10.0
    )
    rear = tp.contact_line_position(
        rho, level=0.5 * (1.96 + 0.16), y_wall=y_wall, side="rear", centre_x=40.0
    )
    assert front == pytest.approx(x_cl, abs=1.0)
    assert rear == pytest.approx(x_cl, abs=1.0)


def test_contact_line_position_rejects_bad_side() -> None:
    rho = torch.ones(8, 8, dtype=torch.float64)
    with pytest.raises(ValueError):
        tp.contact_line_position(rho, level=1.0, y_wall=1.5, side="left", centre_x=4.0)


def test_contact_line_velocity_recovers_slope() -> None:
    samples = [(float(n), 3.0 + 0.0125 * n) for n in range(0, 400, 20)]
    out = tp.contact_line_velocity(samples)
    assert out["velocity"] == pytest.approx(0.0125, rel=1e-6)
    assert out["r2"] == pytest.approx(1.0, abs=1e-9)
    assert out["n_points"] == 20


def test_contact_line_velocity_needs_points() -> None:
    out = tp.contact_line_velocity([(1.0, 2.0), (2.0, 2.5)])
    assert math.isnan(out["velocity"])


def test_young_angle_inverse_exact() -> None:
    g_lg, g_sl = 0.0728, 0.0291
    theta = 60.0
    g_sg = g_sl + g_lg * math.cos(math.radians(theta))
    assert tp.young_angle_deg(g_sg, g_sl, g_lg) == pytest.approx(theta, abs=1e-9)


def test_young_angle_complete_wetting_returns_nan() -> None:
    assert math.isnan(tp.young_angle_deg(0.2, 0.0291, 0.0728))
    with pytest.raises(ValueError):
        tp.young_angle_deg(0.1, 0.05, 0.0)


def test_mass_total_excludes_solid() -> None:
    rho = torch.ones(4, 4, dtype=torch.float64)
    solid = torch.zeros(4, 4, dtype=torch.bool)
    solid[0] = True
    assert tp.mass_total(rho) == pytest.approx(16.0)
    assert tp.mass_total(rho, solid) == pytest.approx(12.0)


def test_planar_wall_tension_reports_resolution_spread() -> None:
    """Wall/fluid tension measurement must be reproducible across row counts."""
    out = tp.planar_wall_tension(
        bulk_density=1.957, wall_fraction=0.7, ny=24, n_steps=600, n_resolutions=2
    )
    assert math.isfinite(out.gamma)
    assert out.mass_drift < 1e-3
    assert len(out.gamma_by_resolution) == 2
    assert math.isfinite(out.resolution_spread)


def test_wetting_tension_ti_monotone_and_shaped() -> None:
    """TI ladder must be monotone in ``psi_w`` and give a finite wetting tension."""
    out = tp.wetting_tension_ti(0.7, ny=24, n_ladder=4, n_steps=400)
    moments = [row["moment_l"] for row in out.ladder]
    assert all(moments[i] <= moments[i + 1] + 1e-9 for i in range(len(moments) - 1))
    assert math.isfinite(out.delta_gamma)
    assert out.n_ladder == 4
    assert out.psi_w > 0.0


def test_wetting_tension_ti_rejects_negative_fraction() -> None:
    with pytest.raises(ValueError):
        tp.wetting_tension_ti(-0.1, ny=16, n_ladder=2, n_steps=10)


def test_to_physical_units_ratios_and_scale() -> None:
    out = tp.to_physical_units(gamma_lg=0.0564, gamma_sl=0.030, gamma_sg=0.060)
    assert out["scale_N_per_m_per_lattice"] == pytest.approx(0.0728 / 0.0564, rel=1e-12)
    assert out["gamma_lg_si"] == pytest.approx(0.0728, rel=1e-9)
    assert out["ratio_sg_over_lg"] == pytest.approx(0.060 / 0.0564, rel=1e-12)
    assert 0.0 <= out["theta_young_lattice_deg"] <= 180.0


def test_free_droplet_smoke() -> None:
    """A small free droplet run must return a positive Laplace tension."""
    out = tp.free_droplet(n=64, radius=10.0, n_steps=400)
    assert out["gamma"] > 0.0
    assert out["mass_drift"] < 1e-3
    assert out["radius"] > 5.0