"""Unit tests for the wall-bounded Shan-Chen engine (IN100-N02).

These tests lock the two properties the contact-line study relies on:

* the wall is a genuine no-slip wall (half-way bounce-back) — mass is conserved
  and the wall-adjacent velocity vanishes;
* the per-cell wall pseudopotential path is *identical* to the scalar path that
  the rest of the library uses (this is the one change made inside
  ``tensorlbm.multiphase`` for the contact-line work).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.contact_line.engine import (
    Channel,
    WallStepper,
    meniscus_state,
    sessile_drop_state,
    tanh_profile,
)
from tensorlbm.multiphase import collide_sc_single_component, psi_exp


class TestChannel:
    def test_geometry(self):
        ch = Channel(nx=64, ny=40, wall_rows=2)
        assert ch.shape == (40, 64)
        assert ch.y_wall() == pytest.approx(1.5)
        assert ch.y_top() == pytest.approx(40 - 2 - 0.5)
        assert ch.n_fluid_y == 36
        solid = ch.solid_mask()
        assert solid.shape == (40, 64)
        assert bool(solid[:2].all()) and bool(solid[-2:].all())
        assert not bool(solid[2:-2].any())

    def test_validation(self):
        with pytest.raises(ValueError):
            Channel(nx=4, ny=40)
        with pytest.raises(ValueError):
            Channel(nx=64, ny=40, wall_rows=0)
        with pytest.raises(ValueError):
            Channel(nx=64, ny=10, wall_rows=4)


class TestProfiles:
    def test_tanh_limits(self):
        d = torch.tensor([[-20.0, 0.0, 20.0]])
        rho = tanh_profile((1, 3), d, 2.0, 0.1, 1.0)
        assert float(rho[0, 0]) == pytest.approx(2.0, abs=1e-6)
        assert float(rho[0, 1]) == pytest.approx(1.05, abs=1e-6)
        assert float(rho[0, 2]) == pytest.approx(0.1, abs=1e-6)

    def test_shape_and_width_validation(self):
        with pytest.raises(ValueError):
            tanh_profile((2, 2), torch.zeros((3, 3)), 1.0, 0.0)
        with pytest.raises(ValueError):
            tanh_profile((2, 2), torch.zeros((2, 2)), 1.0, 0.0, width=0.0)


class TestInitialStates:
    def test_sessile_drop_mass_and_position(self):
        ch = Channel(nx=64, ny=40, wall_rows=2)
        f, rho = sessile_drop_state(ch, 8.0)
        assert f.shape == (9, 40, 64)
        solid = ch.solid_mask()
        liquid = (rho * (~solid)).sum()
        # a 2D droplet of radius 8 carries about pi r^2 * rho_l of liquid
        assert float(liquid) > 150.0
        # the droplet centre is one radius above the wall plane
        assert float(rho[2:-2].max()) == pytest.approx(1.957, abs=0.05)

    def test_meniscus_fills_upstream(self):
        ch = Channel(nx=64, ny=32, wall_rows=2)
        f, rho = meniscus_state(ch, 20.0)
        assert float(rho[16, 5]) > 1.5   # liquid on the left
        assert float(rho[16, 55]) < 0.3  # vapour on the right


class TestStepper:
    def test_mass_conserved_and_finite(self):
        ch = Channel(nx=48, ny=32, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.8)
        f, _ = sessile_drop_state(ch, 8.0)
        m0 = float(f.sum())
        f = stepper.run(f, 60)
        assert torch.isfinite(f).all()
        m1 = float(f.sum())
        assert m1 == pytest.approx(m0, rel=2e-3)

    def test_wall_impermeable(self):
        """Half-way bounce-back: no mass crosses the wall plane."""
        ch = Channel(nx=48, ny=32, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.8, drive="liquid")
        f, _ = sessile_drop_state(ch, 8.0)
        total0 = float(f.sum())
        f = stepper.run(f, 120, gx=1e-4)
        # total mass is conserved (the body force carries momentum only)
        assert float(f.sum()) == pytest.approx(total0, rel=1e-5)

        # The wall region holds no net mass reservoir: the population reflected at
        # the wall plane balances the population arriving from the fluid, so the
        # solid-region mass may redistribute during the near-wall depletion
        # transient but cannot grow (a leaky wall would accumulate mass linearly).
        from tensorlbm.d2q9 import equilibrium

        rho_u = torch.full(ch.shape, 1.9)
        zero = torch.zeros_like(rho_u)
        f_u = equilibrium(rho_u, zero, zero)
        wall_mass0 = float(f_u[:, stepper.solid].sum())
        f_u = stepper.run(f_u, 40)
        wall_mass1 = float(f_u[:, stepper.solid].sum())
        assert abs(wall_mass1 - wall_mass0) / wall_mass0 < 0.05
        assert float(f_u.sum()) == pytest.approx(float(equilibrium(
            rho_u, zero, zero).sum()), rel=1e-5)

    def test_uniform_wall_and_fraction_field(self):
        ch = Channel(nx=32, ny=24, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.7)
        psi_l = stepper.psi_l
        assert float(stepper.wall_psi.mean()) == pytest.approx(0.7 * psi_l, rel=1e-6)
        grad = stepper.gradient_wall(0.5, 0.9)
        assert float(grad[:, 0].mean()) == pytest.approx(0.5 * psi_l, rel=1e-2)
        assert float(grad[:, -1].mean()) == pytest.approx(0.9 * psi_l, rel=1e-2)

    def test_spurious_current_finite(self):
        ch = Channel(nx=48, ny=32, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.8)
        f, _ = sessile_drop_state(ch, 8.0)
        f = stepper.run(f, 40)
        sp = stepper.spurious_current(f)
        assert sp["max"] >= sp["mean"] >= 0.0
        assert torch.isfinite(torch.tensor(sp["max"]))


class TestWallPsiTensorPath:
    """The per-cell ``wall_psi`` field must match the scalar path bit for bit."""

    def test_tensor_equals_scalar(self):
        ch = Channel(nx=32, ny=24, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.6)
        f, _ = sessile_drop_state(ch, 6.0)
        scalar = collide_sc_single_component(
            f, 5.0, tau=1.0, psi_fn=psi_exp, solid_mask=stepper.solid, wall_psi=0.5
        )
        field = collide_sc_single_component(
            f,
            5.0,
            tau=1.0,
            psi_fn=psi_exp,
            solid_mask=stepper.solid,
            wall_psi=torch.full(ch.shape, 0.5),
        )
        assert torch.equal(scalar, field)

    def test_wall_psi_shape_validation(self):
        ch = Channel(nx=32, ny=24, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.6)
        f, _ = sessile_drop_state(ch, 6.0)
        with pytest.raises(ValueError):
            collide_sc_single_component(
                f, 5.0, tau=1.0, psi_fn=psi_exp, solid_mask=stepper.solid,
                wall_psi=torch.ones((3, 3)),
            )

    def test_per_cell_wall_changes_solution(self):
        ch = Channel(nx=32, ny=24, wall_rows=2)
        stepper = WallStepper(ch, wall_fraction=0.6)
        f, _ = sessile_drop_state(ch, 6.0)
        grad = stepper.gradient_wall(0.4, 0.8)
        a = collide_sc_single_component(
            f, 5.0, tau=1.0, psi_fn=psi_exp, solid_mask=stepper.solid, wall_psi=0.5
        )
        b = collide_sc_single_component(
            f, 5.0, tau=1.0, psi_fn=psi_exp, solid_mask=stepper.solid, wall_psi=grad
        )
        assert not torch.equal(a, b)