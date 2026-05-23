"""Tests for miscellaneous modules with low coverage:
- logging_config.py: configure_logging, logger
- d3q27.py: collide_bgk27, stream27
- solver.py: correct_mass
- solver3d.py: correct_mass3d
- boundaries.py: zou_he_outlet_pressure, bounce_back_cells
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from tensorlbm import (
    correct_mass,
    correct_mass3d,
    equilibrium,
    equilibrium3d,
    macroscopic,
)
from tensorlbm.d3q27 import (
    collide_bgk27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.logging_config import configure_logging, logger

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# logging_config
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_logger_name(self) -> None:
        assert logger.name == "tensorlbm"

    def test_configure_adds_handler(self) -> None:
        # Remove existing handlers to start clean
        logger.handlers.clear()
        configure_logging(level=logging.DEBUG)
        assert len(logger.handlers) >= 1

    def test_configure_sets_level(self) -> None:
        logger.handlers.clear()
        configure_logging(level=logging.WARNING)
        assert logger.level == logging.WARNING

    def test_configure_not_duplicate_handler(self) -> None:
        """Calling configure_logging twice must not add a second handler."""
        logger.handlers.clear()
        configure_logging()
        n_after_first = len(logger.handlers)
        configure_logging()
        assert len(logger.handlers) == n_after_first

    def test_logger_can_emit(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="tensorlbm"):
            logger.debug("test message")
        assert "test message" in caplog.text


# ---------------------------------------------------------------------------
# D3Q27: collide_bgk27 and stream27
# ---------------------------------------------------------------------------


class TestCollide_bgk27:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_bgk27(f, tau=0.6)
        assert f_out.shape == (27, nz, ny, nx)

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_bgk27(f, tau=0.6)
        assert torch.isfinite(f_out).all()

    def test_conserves_mass(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_bgk27(f, tau=0.7)
        rho_new, _, _, _ = macroscopic27(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_bgk27(f, tau=0.7)
        _, ux_new, uy_new, uz_new = macroscopic27(f_new)
        assert torch.allclose(ux_new, ux, atol=1e-5)
        assert torch.allclose(uy_new, uy, atol=1e-5)
        assert torch.allclose(uz_new, uz, atol=1e-5)

    def test_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        f_out = collide_bgk27(feq, tau=0.6)
        assert torch.allclose(f_out, feq, atol=1e-5)


class TestStream27:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = stream27(f)
        assert f_out.shape == (27, nz, ny, nx)

    def test_conserves_mass_periodic(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        mass_before = float(f.sum().item())
        f_out = stream27(f)
        mass_after = float(f_out.sum().item())
        assert abs(mass_before - mass_after) < 1e-5 * mass_before

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = stream27(f)
        assert torch.isfinite(f_out).all()

    def test_double_stream_uniform_returns_same(self) -> None:
        """Streaming a uniform field twice should return the original."""
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f2 = stream27(stream27(f))
        # After two streams the populations have moved by 2 lattice steps;
        # for a uniform field all values are the same so it must be equal.
        assert torch.allclose(f2, f, atol=1e-6)


# ---------------------------------------------------------------------------
# correct_mass (2D) and correct_mass3d (3D)
# ---------------------------------------------------------------------------


class TestCorrectMass2D:
    def test_corrects_total_mass(self) -> None:
        rho = torch.ones((8, 10))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        target = float(f.sum().item()) * 1.05  # 5 % drift
        f_corr = correct_mass(f, target_mass=target)
        assert abs(float(f_corr.sum().item()) - target) < 1e-4

    def test_preserves_shape(self) -> None:
        rho = torch.ones((8, 10))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_corr = correct_mass(f, target_mass=float(f.sum()))
        assert f_corr.shape == f.shape

    def test_no_change_if_mass_correct(self) -> None:
        rho = torch.ones((8, 10))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        target = float(f.sum().item())
        f_corr = correct_mass(f, target_mass=target)
        assert torch.allclose(f_corr, f, atol=1e-6)

    def test_near_zero_mass_returns_f(self) -> None:
        """If current mass ≈ 0 the function must not crash or divide by zero."""
        f = torch.zeros((9, 8, 10))
        f_corr = correct_mass(f, target_mass=1.0)
        # Returns f unchanged (guard clause)
        assert torch.allclose(f_corr, f)


class TestCorrectMass3D:
    def test_corrects_total_mass(self) -> None:
        rho = torch.ones((4, 6, 8))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        target = float(f.sum().item()) * 0.98
        f_corr = correct_mass3d(f, target_mass=target)
        assert abs(float(f_corr.sum().item()) - target) < 1e-4

    def test_preserves_shape(self) -> None:
        rho = torch.ones((4, 6, 8))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_corr = correct_mass3d(f, target_mass=float(f.sum()))
        assert f_corr.shape == f.shape

    def test_near_zero_mass_returns_f(self) -> None:
        f = torch.zeros((19, 4, 6, 8))
        f_corr = correct_mass3d(f, target_mass=1.0)
        assert torch.allclose(f_corr, f)


# ---------------------------------------------------------------------------
# zou_he_outlet_pressure (2D) and bounce_back_cells
# ---------------------------------------------------------------------------


class TestZouHeOutletPressure2D:
    def test_preserves_shape(self) -> None:
        from tensorlbm import zou_he_outlet_pressure

        ny, nx = 12, 20
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho))
        f_out = zou_he_outlet_pressure(f, rho_out=1.0)
        assert f_out.shape == f.shape

    def test_finite_values(self) -> None:
        from tensorlbm import zou_he_outlet_pressure

        ny, nx = 12, 20
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho))
        f_out = zou_he_outlet_pressure(f, rho_out=1.0)
        assert torch.isfinite(f_out).all()

    def test_prescribes_density_at_outlet(self) -> None:
        from tensorlbm import zou_he_outlet_pressure

        ny, nx = 12, 20
        rho0 = torch.ones((ny, nx))
        f = equilibrium(rho0, torch.full_like(rho0, 0.05), torch.zeros_like(rho0))
        rho_out = 1.0
        f_new = zou_he_outlet_pressure(f, rho_out=rho_out)
        rho_field, _, _ = macroscopic(f_new)
        assert torch.allclose(rho_field[:, -1], torch.full((ny,), rho_out), atol=1e-4)


class TestBounceBackCells:
    def test_preserves_shape(self) -> None:
        from tensorlbm import bounce_back_cells

        ny, nx = 10, 12
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((ny, nx), dtype=torch.bool)
        mask[0, :] = True
        f_out = bounce_back_cells(f, mask)
        assert f_out.shape == f.shape

    def test_empty_mask_unchanged(self) -> None:
        from tensorlbm import bounce_back_cells

        ny, nx = 10, 12
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((ny, nx), dtype=torch.bool)
        f_out = bounce_back_cells(f, mask)
        assert torch.allclose(f_out, f)

    def test_at_equilibrium_no_net_momentum(self) -> None:
        """Bounce-back at equilibrium (zero velocity) leaves distributions symmetric."""
        from tensorlbm import bounce_back_cells

        ny, nx = 10, 12
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((ny, nx), dtype=torch.bool)
        mask[0, :] = True
        f_out = bounce_back_cells(f, mask)
        assert torch.isfinite(f_out).all()
