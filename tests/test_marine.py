"""Tests for ship and ocean engineering extensions.

Covers:
- obstacles.py  : Wigley hull mask, 3D force/moment diagnostics
- turbulence.py : Smagorinsky BGK/MRT collision operators
- wave_bc.py    : Airy wave velocity profile and inlet BC
- ship_flow.py  : ShipHullFlowConfig validation
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import (
    airy_wave_velocity_3d,
    apply_wave_inlet_3d,
    collide_smagorinsky_bgk,
    collide_smagorinsky_bgk3d,
    collide_smagorinsky_mrt3d,
    compute_obstacle_forces_3d,
    compute_obstacle_moments_3d,
    equilibrium,
    equilibrium3d,
    macroscopic,
    macroscopic3d,
    make_channel_wall_mask_3d,
    stream,
    stream3d,
    wigley_hull_mask,
    zou_he_inlet_velocity_profile_3d,
)
from tensorlbm.ship_flow import ShipHullFlowConfig


# ---------------------------------------------------------------------------
# Wigley hull mask
# ---------------------------------------------------------------------------

class TestWigleyHullMask:
    def test_shape(self) -> None:
        mask = wigley_hull_mask(
            nx=40, ny=20, nz=16,
            cx=20.0, cy=10.0, cz_keel=2.0,
            length=20.0, beam=4.0, draft=8.0,
            device=torch.device("cpu"),
        )
        assert mask.shape == (16, 20, 40)
        assert mask.dtype == torch.bool

    def test_non_empty(self) -> None:
        """Hull must contain at least one solid cell."""
        mask = wigley_hull_mask(
            nx=40, ny=20, nz=16,
            cx=20.0, cy=10.0, cz_keel=2.0,
            length=20.0, beam=4.0, draft=8.0,
            device=torch.device("cpu"),
        )
        assert mask.any()

    def test_symmetric_in_y(self) -> None:
        """Wigley hull is port-starboard symmetric about cy."""
        mask = wigley_hull_mask(
            nx=40, ny=20, nz=16,
            cx=20.0, cy=9.5, cz_keel=2.0,
            length=20.0, beam=4.0, draft=8.0,
            device=torch.device("cpu"),
        )
        # Reflect about cy = 9.5 → index flip in y
        assert torch.equal(mask, mask.flip(dims=[1]))

    def test_hull_within_domain(self) -> None:
        """All hull cells must lie strictly inside the grid."""
        nz, ny, nx = 16, 20, 40
        mask = wigley_hull_mask(
            nx=nx, ny=ny, nz=nz,
            cx=20.0, cy=10.0, cz_keel=2.0,
            length=18.0, beam=3.0, draft=8.0,
            device=torch.device("cpu"),
        )
        # No hull cells at domain boundaries (x=0 or x=nx-1)
        assert not mask[:, :, 0].any()
        assert not mask[:, :, -1].any()


# ---------------------------------------------------------------------------
# 3D force diagnostics
# ---------------------------------------------------------------------------

class TestObstacleForces3d:
    def _zero_flow_f(self, nz: int, ny: int, nx: int) -> torch.Tensor:
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        return stream3d(f)

    def test_empty_mask_gives_zero_forces(self) -> None:
        nz, ny, nx = 6, 8, 10
        f = self._zero_flow_f(nz, ny, nx)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fx, fy, fz = compute_obstacle_forces_3d(f, mask)
        assert float(fx) == pytest.approx(0.0)
        assert float(fy) == pytest.approx(0.0)
        assert float(fz) == pytest.approx(0.0)

    def test_returns_finite_values(self) -> None:
        nz, ny, nx = 8, 10, 16
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.05)
        f = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f = stream3d(f)
        mask = wigley_hull_mask(
            nx=nx, ny=ny, nz=nz,
            cx=8.0, cy=5.0, cz_keel=1.0,
            length=6.0, beam=2.0, draft=4.0,
            device=torch.device("cpu"),
        )
        fx, fy, fz = compute_obstacle_forces_3d(f, mask)
        assert math.isfinite(float(fx))
        assert math.isfinite(float(fy))
        assert math.isfinite(float(fz))


class TestObstacleMoments3d:
    def test_empty_mask_gives_zero_moments(self) -> None:
        nz, ny, nx = 6, 8, 10
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mx, my, mz = compute_obstacle_moments_3d(f, mask, 5.0, 4.0, 3.0)
        assert float(mx) == pytest.approx(0.0)
        assert float(my) == pytest.approx(0.0)
        assert float(mz) == pytest.approx(0.0)

    def test_returns_finite_values(self) -> None:
        nz, ny, nx = 8, 10, 16
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.05)
        f = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f = stream3d(f)
        mask = wigley_hull_mask(
            nx=nx, ny=ny, nz=nz,
            cx=8.0, cy=5.0, cz_keel=1.0,
            length=6.0, beam=2.0, draft=4.0,
            device=torch.device("cpu"),
        )
        mx, my, mz = compute_obstacle_moments_3d(f, mask, 8.0, 5.0, 3.0)
        assert math.isfinite(float(mx))
        assert math.isfinite(float(my))
        assert math.isfinite(float(mz))


# ---------------------------------------------------------------------------
# Smagorinsky turbulence models
# ---------------------------------------------------------------------------

class TestSmagorinskyBGK2D:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 10
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_smagorinsky_bgk(f, tau=0.6)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_conserves_mass_and_momentum(self) -> None:
        """Smagorinsky BGK must preserve local density and momentum."""
        ny, nx = 8, 10
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        f = equilibrium(rho, ux, uy)
        f_new = collide_smagorinsky_bgk(f, tau=0.7)
        rho_new, ux_new, uy_new = macroscopic(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-5)
        assert torch.allclose(ux_new, ux, atol=1e-5)
        assert torch.allclose(uy_new, uy, atol=1e-5)

    def test_at_equilibrium_is_identity(self) -> None:
        ny, nx = 8, 10
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.full_like(rho, 0.02)
        feq = equilibrium(rho, ux, uy)
        f_out = collide_smagorinsky_bgk(feq, tau=0.6)
        assert torch.allclose(f_out, feq, atol=1e-5)


class TestSmagorinskyBGK3D:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_smagorinsky_bgk3d(f, tau=0.6)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_conserves_mass_and_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium3d(rho, ux, uy, uz)
        f_new = collide_smagorinsky_bgk3d(f, tau=0.7)
        rho_new, ux_new, uy_new, uz_new = macroscopic3d(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-4)
        assert torch.allclose(ux_new, ux, atol=1e-4)
        assert torch.allclose(uy_new, uy, atol=1e-4)
        assert torch.allclose(uz_new, uz, atol=1e-4)

    def test_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_smagorinsky_bgk3d(feq, tau=0.6)
        assert torch.allclose(f_out, feq, atol=1e-4)


class TestSmagorinskyMRT3D:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_smagorinsky_mrt3d(f, tau=0.6)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_conserves_mass_and_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium3d(rho, ux, uy, uz)
        f_new = collide_smagorinsky_mrt3d(f, tau=0.7)
        rho_new, ux_new, uy_new, uz_new = macroscopic3d(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-4)
        assert torch.allclose(ux_new, ux, atol=1e-4)
        assert torch.allclose(uy_new, uy, atol=1e-4)
        assert torch.allclose(uz_new, uz, atol=1e-4)

    def test_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_smagorinsky_mrt3d(feq, tau=0.6)
        assert torch.allclose(f_out, feq, atol=1e-4)


# ---------------------------------------------------------------------------
# Wave boundary conditions
# ---------------------------------------------------------------------------

class TestAiryWaveVelocity3D:
    def test_output_shape(self) -> None:
        ux, uy, uz = airy_wave_velocity_3d(
            nz=10, ny=8, step=100,
            u_mean=0.05, wave_amp=0.005, wave_period=200.0,
            wave_k=0.05, water_depth=8.0, z_bed=1.0,
            device=torch.device("cpu"),
        )
        assert ux.shape == (10, 8)
        assert uy.shape == (10, 8)
        assert uz.shape == (10, 8)

    def test_finite_values(self) -> None:
        ux, uy, uz = airy_wave_velocity_3d(
            nz=10, ny=8, step=50,
            u_mean=0.04, wave_amp=0.003, wave_period=150.0,
            wave_k=0.06, water_depth=8.0, z_bed=0.0,
            device=torch.device("cpu"),
        )
        assert torch.isfinite(ux).all()
        assert torch.isfinite(uy).all()
        assert torch.isfinite(uz).all()

    def test_mean_velocity_included(self) -> None:
        """When wave_amp=0, ux should equal u_mean everywhere."""
        u_mean = 0.05
        ux, _, _ = airy_wave_velocity_3d(
            nz=8, ny=6, step=0,
            u_mean=u_mean, wave_amp=0.0, wave_period=100.0,
            wave_k=0.05, water_depth=6.0, z_bed=0.0,
            device=torch.device("cpu"),
        )
        # wave_amp=0 means no oscillation; at step=0 cos(0)=1 so contribution is 0
        assert torch.allclose(ux, torch.full_like(ux, u_mean), atol=1e-6)

    def test_uy_is_zero(self) -> None:
        _, uy, _ = airy_wave_velocity_3d(
            nz=8, ny=6, step=30,
            u_mean=0.05, wave_amp=0.003, wave_period=100.0,
            wave_k=0.05, water_depth=6.0, z_bed=0.0,
            device=torch.device("cpu"),
        )
        assert float(uy.abs().max()) == pytest.approx(0.0)


class TestZouHeInletVelocityProfile3D:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 6, 8, 12
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        ux_in = torch.full((nz, ny), 0.05)
        uy_in = torch.zeros(nz, ny)
        uz_in = torch.zeros(nz, ny)
        f_out = zou_he_inlet_velocity_profile_3d(f, ux_in, uy_in, uz_in)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_prescribes_mean_ux_at_inlet(self) -> None:
        """After applying the BC, macroscopic ux at x=0 should match ux_in."""
        nz, ny, nx = 6, 8, 12
        rho0 = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
        from tensorlbm import collide_bgk3d, stream3d
        f = collide_bgk3d(f, tau=0.6)
        f = stream3d(f)

        u_in_val = 0.06
        ux_in = torch.full((nz, ny), u_in_val)
        uy_in = torch.zeros(nz, ny)
        uz_in = torch.zeros(nz, ny)
        f_out = zou_he_inlet_velocity_profile_3d(f, ux_in, uy_in, uz_in)
        _, ux_out, _, _ = macroscopic3d(f_out)
        assert torch.allclose(ux_out[:, :, 0], torch.full((nz, ny), u_in_val), atol=2e-4)


class TestApplyWaveInlet3D:
    def test_preserves_shape_and_finite(self) -> None:
        nz, ny, nx = 8, 10, 16
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho), torch.zeros_like(rho))
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        wall_mask = make_channel_wall_mask_3d(nz, ny, nx, obstacle, device=torch.device("cpu"))
        f_out = apply_wave_inlet_3d(
            f, step=10,
            wall_mask=wall_mask, obstacle_mask=obstacle,
            u_mean=0.05, wave_amp=0.003, wave_period=100.0,
            wave_k=0.05, water_depth=float(nz), z_bed=0.0,
        )
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# ShipHullFlowConfig validation
# ---------------------------------------------------------------------------

class TestShipHullFlowConfig:
    def test_valid_config_does_not_raise(self) -> None:
        cfg = ShipHullFlowConfig(
            nx=80, ny=40, nz=30,
            hull_length=40.0, hull_beam=4.0, hull_draft=6.0,
            u_in=0.05, re=100.0, n_steps=10, output_interval=5,
        )
        cfg.validate()  # should not raise

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"nx": 4}, "nx, ny"),
            ({"ny": 2}, "nx, ny"),
            ({"nz": 2}, "nx, ny"),
            ({"n_steps": 0}, "n_steps"),
            ({"output_interval": 0}, "output_interval"),
            ({"u_in": -0.01}, "u_in"),
            ({"re": 0.0}, "u_in"),
            ({"hull_length": 0.0}, "hull_length"),
            ({"hull_beam": 0.0}, "hull_length"),
            ({"hull_draft": 0.0}, "hull_length"),
            ({"hull_length": 200.0}, "hull_length must be less than nx"),
            ({"hull_beam": 200.0}, "hull_beam must be less than ny"),
            ({"hull_draft": 200.0}, "hull_draft must be less than nz"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        base = dict(
            nx=80, ny=40, nz=30,
            hull_length=40.0, hull_beam=4.0, hull_draft=6.0,
            u_in=0.05, re=100.0, n_steps=10, output_interval=5,
        )
        base.update(overrides)
        cfg = ShipHullFlowConfig(**base)
        with pytest.raises(ValueError, match=match):
            cfg.validate()

    def test_tau_too_small_raises(self) -> None:
        cfg = ShipHullFlowConfig(u_in=1e-9, re=1e12, hull_length=40.0)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_froude_finite(self) -> None:
        cfg = ShipHullFlowConfig(u_in=0.05, wave_k=0.05)
        assert math.isfinite(cfg.froude)
        assert cfg.froude > 0.0

    def test_resolved_run_name_contains_key_params(self) -> None:
        cfg = ShipHullFlowConfig(nx=80, ny=40, nz=30, re=200.0, u_in=0.05, n_steps=100)
        name = cfg.resolved_run_name()
        assert "nx80" in name
        assert "re200" in name
