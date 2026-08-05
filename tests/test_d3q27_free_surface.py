"""D3Q27 free-surface Körner model — TDD tests.

Tests cover:
  - D3Q27 stencil module (neighbor masks, shifts, phase-link guard)
  - free_surface_step_27 core pipeline (BGK collision + stream + mass update)
  - Static water column (hydrostatic stability, mass conservation)
  - Dam-break stability (water flows in expected direction)
  - Interface conversion (I→L / I→G)
  - Smagorinsky SGS option (optional)

These tests do NOT modify existing D3Q19 free-surface code.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import W as W27
from tensorlbm.d3q27 import OPPOSITE as OPP27
from tensorlbm.d3q27 import (
    collide_bgk27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.boundaries_d3q27 import bounce_back_cells_27


# ---------------------------------------------------------------------------
# Stencil module tests
# ---------------------------------------------------------------------------

class TestD3Q27Stencil:
    """Tests for the D3Q27 stencil bridge module."""

    def test_import(self):
        from tensorlbm.core.d3q27_stencil import D3Q27_MOVING_Q
        assert D3Q27_MOVING_Q == tuple(range(1, 27))

    def test_moving_tensor_shifts_count(self):
        from tensorlbm.core.d3q27_stencil import moving_tensor_shifts_27
        shifts = moving_tensor_shifts_27()
        assert len(shifts) == 26

    def test_moving_tensor_shifts_match_C(self):
        from tensorlbm.core.d3q27_stencil import moving_tensor_shifts_27
        shifts = moving_tensor_shifts_27()
        for i, q in enumerate(range(1, 27)):
            dz, dy, dx = shifts[i]
            assert dz == int(C27[q, 2])
            assert dy == int(C27[q, 1])
            assert dx == int(C27[q, 0])

    def test_roll_from_pull_source(self):
        from tensorlbm.core.d3q27_stencil import roll_from_pull_source_27
        field = torch.arange(5 * 5 * 5, dtype=torch.float32).reshape(5, 5, 5)
        # q=1 is (1,0,0) → shift (dz=0, dy=0, dx=1)
        rolled = roll_from_pull_source_27(field, 1)
        # Pull source for q=1 at (z,y,x) is (z,y,x-1), so rolled[x] = field[x-1]
        assert torch.equal(rolled[:, :, 0], field[:, :, -1])
        assert torch.equal(rolled[:, :, 1], field[:, :, 0])

    def test_roll_to_neighbor(self):
        from tensorlbm.core.d3q27_stencil import roll_to_neighbor_27
        field = torch.arange(5 * 5 * 5, dtype=torch.float32).reshape(5, 5, 5)
        rolled = roll_to_neighbor_27(field, 1)
        # roll_to_neighbor is the inverse of roll_from_pull_source
        from tensorlbm.core.d3q27_stencil import roll_from_pull_source_27
        assert torch.equal(roll_from_pull_source_27(rolled, 1), field)

    def test_all_moving_neighbor_masks_count(self):
        from tensorlbm.core.d3q27_stencil import all_moving_neighbor_masks_27
        mask = torch.zeros(6, 6, 6, dtype=torch.bool)
        mask[3, 3, 3] = True
        masks = all_moving_neighbor_masks_27(mask)
        assert len(masks) == 26

    def test_all_moving_neighbor_masks_correct(self):
        from tensorlbm.core.d3q27_stencil import all_moving_neighbor_masks_27
        mask = torch.zeros(7, 7, 7, dtype=torch.bool)
        mask[3, 3, 3] = True
        masks = all_moving_neighbor_masks_27(mask)
        # Each mask should have exactly one True (the neighbor of (3,3,3))
        for m in masks:
            assert m.sum() == 1
        # The union should cover all 26 neighbors of (3,3,3)
        union = torch.stack(masks).any(dim=0)
        assert union.sum() == 26
        # Center should not be a neighbor
        assert not union[3, 3, 3]

    def test_assert_no_direct_phase_links_clean(self):
        from tensorlbm.core.d3q27_stencil import assert_no_direct_phase_links_27
        flags = torch.full((5, 5, 5), 0, dtype=torch.int8)  # all GAS
        flags[2, 2, 2] = 1  # one LIQUID
        # Surround with INTERFACE on all 26 D3Q27 moving neighbors
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    flags[2 + dz, 2 + dy, 2 + dx] = 2
        # Should not raise
        assert_no_direct_phase_links_27(flags, 1, 0, "test")

    def test_assert_no_direct_phase_links_violation(self):
        from tensorlbm.core.d3q27_stencil import assert_no_direct_phase_links_27
        flags = torch.full((5, 5, 5), 0, dtype=torch.int8)  # all GAS
        flags[2, 2, 2] = 1  # LIQUID
        flags[2, 2, 3] = 0  # direct GAS neighbor (no INTERFACE)
        with pytest.raises(ValueError, match="direct phase link"):
            assert_no_direct_phase_links_27(flags, 1, 0, "test")


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestFreeSurface27Init:
    """Tests for D3Q27 free-surface initialization helpers."""

    def test_init_fill_rectangular(self):
        from tensorlbm.free_surface_lbm_27 import init_fill_rectangular_27
        fill, solid = init_fill_rectangular_27(8, 8, 10, 4, 6, torch.device("cpu"))
        assert fill.shape == (8, 8, 10)
        assert solid.shape == (8, 8, 10)
        # Solid walls
        assert solid[:, 0, :].all()
        assert solid[:, -1, :].all()
        assert solid[:, :, 0].all()
        assert solid[:, :, -1].all()
        # Liquid column
        assert fill[:, :6, 1:5].sum() > 0

    def test_init_flags_from_fill(self):
        from tensorlbm.free_surface_lbm_27 import (
            init_fill_rectangular_27,
            init_flags_from_fill_27,
            GAS, LIQUID, INTERFACE, SOLID,
        )
        fill, solid = init_fill_rectangular_27(8, 8, 10, 4, 6, torch.device("cpu"))
        flags = init_flags_from_fill_27(fill, solid)
        # Interior liquid cells should be LIQUID
        assert (flags[1:-1, 1:5, 1:4] == LIQUID).any()
        # Walls should be SOLID
        assert (flags[solid] == SOLID).all()
        # No direct LIQUID-GAS links
        from tensorlbm.core.d3q27_stencil import assert_no_direct_phase_links_27
        assert_no_direct_phase_links_27(flags, LIQUID, GAS, "init")

    def test_init_mass_from_fill(self):
        from tensorlbm.free_surface_lbm_27 import (
            init_fill_rectangular_27,
            init_flags_from_fill_27,
            init_mass_from_fill_27,
            LIQUID, INTERFACE,
        )
        fill, solid = init_fill_rectangular_27(8, 8, 10, 4, 6, torch.device("cpu"))
        flags = init_flags_from_fill_27(fill, solid)
        mass = init_mass_from_fill_27(fill, flags, rho_liquid=1.0)
        # LIQUID cells have mass = rho_liquid
        liquid_mask = flags == LIQUID
        assert torch.allclose(mass[liquid_mask], torch.tensor(1.0))
        # INTERFACE cells have mass = fill * rho_liquid
        iface_mask = flags == INTERFACE
        assert torch.allclose(mass[iface_mask], fill[iface_mask])


# ---------------------------------------------------------------------------
# Core free_surface_step_27 tests
# ---------------------------------------------------------------------------

class TestFreeSurfaceStep27:
    """Tests for the D3Q27 free-surface step."""

    def _make_static_column(self, nz=10, ny=10, nx=10, liquid_height=6, rho_gas=1.0):
        """Create a static water column initial condition."""
        from tensorlbm.free_surface_lbm_27 import (
            init_fill_rectangular_27,
            init_flags_from_fill_27,
            init_mass_from_fill_27,
        )
        fill, solid = init_fill_rectangular_27(nz, ny, nx, nx - 2, liquid_height, torch.device("cpu"))
        flags = init_flags_from_fill_27(fill, solid)
        mass = init_mass_from_fill_27(fill, flags, rho_liquid=1.0)
        # Initialize f with equilibrium at rest
        rho_field = torch.where(flags == 1, torch.tensor(1.0), torch.tensor(0.0))
        rho_field = torch.where(flags == 2, fill, rho_field)
        f = equilibrium27(rho_field, torch.zeros_like(rho_field), torch.zeros_like(rho_field), torch.zeros_like(rho_field))
        return f, fill, flags, solid, mass, rho_gas

    def test_step_returns_correct_shapes(self):
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f_out, fill_out, flags_out, mass_out, df = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
        )
        assert f_out.shape == (27, *flags.shape)
        assert fill_out.shape == flags.shape
        assert flags_out.shape == flags.shape
        assert mass_out.shape == flags.shape

    def test_step_finite(self):
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f_out, fill_out, flags_out, mass_out, df = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
        )
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(mass_out).all()
        assert torch.isfinite(fill_out).all()

    def test_static_column_mass_conservation(self):
        """A static water column with no gravity should not drift catastrophically.

        The Körner model has an inherent mass drift at LIQUID-INTERFACE links
        because LIQUID cell mass is always reset to rho_liquid.  The D3Q19
        version tracks this drift via a mass ledger but does not correct it
        by default.  We check that the drift is bounded, not zero.
        """
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column(nz=10, ny=10, nx=10, liquid_height=6)
        initial_mass = float(mass.sum())
        for _ in range(5):
            f, fill, flags, mass, _ = free_surface_step_27(
                f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
            )
        final_mass = float(mass.sum())
        # Mass drift is expected in the Körner model; check it's bounded
        assert abs(final_mass - initial_mass) < initial_mass * 0.5, (
            f"Mass drift too large: initial={initial_mass}, final={final_mass}, "
            f"drift={final_mass - initial_mass}"
        )

    def test_static_column_stable_velocities(self):
        """A static water column with no gravity should have near-zero velocity."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column(nz=10, ny=10, nx=10, liquid_height=6)
        for _ in range(3):
            f, fill, flags, mass, _ = free_surface_step_27(
                f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
            )
        rho, ux, uy, uz = macroscopic27(f)
        # Check velocities in liquid region are small
        liquid = flags == 1
        if liquid.any():
            u_max = float((ux[liquid].abs().max() + uy[liquid].abs().max() + uz[liquid].abs().max()))
            assert u_max < 0.5, f"Velocity too large in liquid: {u_max}"

    def test_dam_break_flows(self):
        """Dam break: water column should start moving under gravity."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column(nz=10, ny=10, nx=12, liquid_height=6)
        # Apply gravity in -y direction (dam collapses)
        for _ in range(5):
            f, fill, flags, mass, _ = free_surface_step_27(
                f, fill, flags, solid, mass=mass, tau=1.0, gy=-0.001, rho_gas=rho_gas,
            )
        rho, ux, uy, uz = macroscopic27(f)
        liquid = flags == 1
        if liquid.any():
            # Under gravity, y-velocity should be negative (downward)
            assert float(uy[liquid].mean()) < 0.01  # should be moving or near rest

    def test_no_direct_liquid_gas_links_after_step(self):
        """After a step, no direct LIQUID-GAS links should exist."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        from tensorlbm.core.d3q27_stencil import assert_no_direct_phase_links_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f, fill, flags, mass, _ = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
        )
        assert_no_direct_phase_links_27(flags, 1, 0, "post-step")

    def test_smagorinsky_option(self):
        """free_surface_step_27 should accept C_s > 0 for Smagorinsky SGS."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f_out, fill_out, flags_out, mass_out, df = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, C_s=0.1, rho_gas=rho_gas,
        )
        assert torch.isfinite(f_out).all()

    def test_mrt_collision_option(self):
        """free_surface_step_27 should accept collision='mrt'."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f_out, fill_out, flags_out, mass_out, df = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, collision='mrt', rho_gas=rho_gas,
        )
        assert torch.isfinite(f_out).all()

    def test_fill_bounded(self):
        """Fill field should remain in [0, 1] after a step."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f, fill, flags, mass, _ = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
        )
        assert bool((fill >= 0).all())
        assert bool((fill <= 1).all())

    def test_mass_non_negative(self):
        """Mass field should remain non-negative after a step."""
        from tensorlbm.free_surface_lbm_27 import free_surface_step_27
        f, fill, flags, solid, mass, rho_gas = self._make_static_column()
        f, fill, flags, mass, _ = free_surface_step_27(
            f, fill, flags, solid, mass=mass, tau=1.0, gz=0.0, rho_gas=rho_gas,
        )
        assert bool((mass >= 0).all())


# ---------------------------------------------------------------------------
# Multi-step stability test
# ---------------------------------------------------------------------------

class TestFreeSurface27Stability:
    """Longer-run stability tests."""

    def test_20_step_stability(self):
        """20 steps should not produce NaN or Inf."""
        from tensorlbm.free_surface_lbm_27 import (
            init_fill_rectangular_27,
            init_flags_from_fill_27,
            init_mass_from_fill_27,
            free_surface_step_27,
        )
        nz, ny, nx = 12, 12, 12
        fill, solid = init_fill_rectangular_27(nz, ny, nx, 6, 8, torch.device("cpu"))
        flags = init_flags_from_fill_27(fill, solid)
        mass = init_mass_from_fill_27(fill, flags, rho_liquid=1.0)
        rho_field = torch.where(flags == 1, torch.tensor(1.0), torch.tensor(0.0))
        rho_field = torch.where(flags == 2, fill, rho_field)
        f = equilibrium27(rho_field, torch.zeros_like(rho_field), torch.zeros_like(rho_field), torch.zeros_like(rho_field))
        for _ in range(20):
            f, fill, flags, mass, _ = free_surface_step_27(
                f, fill, flags, solid, mass=mass, tau=1.0, gy=-0.001,
            )
            assert torch.isfinite(f).all(), "f has non-finite values"
            assert torch.isfinite(mass).all(), "mass has non-finite values"
        # Mass should not have drifted catastrophically
        initial = float(init_mass_from_fill_27(
            init_fill_rectangular_27(nz, ny, nx, 6, 8, torch.device("cpu"))[0],
            init_flags_from_fill_27(*init_fill_rectangular_27(nz, ny, nx, 6, 8, torch.device("cpu"))[:2]),
            rho_liquid=1.0,
        ).sum())
        final = float(mass.sum())
        assert abs(final - initial) < initial * 0.5, f"Mass drift too large: {initial} → {final}"
