"""Unit tests for the DDES/SAS hybrid turbulence module (G21 orphan verification).

Closes G21 gap: ``ddes.py`` was implemented but had zero test coverage.
These tests verify mathematical invariants of each public function without
requiring a full simulation run.

Test matrix
-----------
T1  Shielding function range: f_d ∈ [0, 1] for all inputs.
T2  Shielding function limits: f_d → 0 when r_d → ∞ (BL), f_d → 1 when r_d → 0 (far-field).
T3  Length scale bounds: l_LES ≤ l_DDES ≤ l_RANS when l_RANS > l_LES.
T4  Eddy viscosity positivity: ν_t ≥ 0 always.
T5  Eddy viscosity clamping: ν_t ≤ ν_t_max.
T6  LES mode: f_d ≡ 1, ν_t = (C_S Δ)² |S|.
T7  Collision stability: τ_eff > 0.5 always.
T8  SAS source term non-negativity: Q_SAS ≥ 0.
T9  Strain-rate magnitude: |S| ≥ 0, zero for uniform flow.
T10 Gradient magnitude: |∇φ| ≥ 0.
T11 Laplacian: zero for constant field.
T12 DDESResult fields: all ∈ [0, 1] for fractions.
T13 apply_ddes_collision: output shape matches input, mass conserved to ~1e-10.
T14 run_ddes_diagnostics: returns valid DDESResult.

All computations: CPU, float64, no GPU required.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.ddes import (
    DDESConfig,
    DDESResult,
    apply_ddes_collision,
    ddes_eddy_viscosity,
    ddes_length_scale,
    ddes_shielding_function,
    run_ddes_diagnostics,
    sas_source_term,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE = torch.float64


def _make_field(ny: int = 32, nx: int = 64, val: float = 0.0) -> torch.Tensor:
    return torch.full((ny, nx), val, dtype=DTYPE)


def _make_shear_flow(ny: int = 32, nx: int = 64, shear: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear shear: u = shear * y, v = 0."""
    y = torch.arange(ny, dtype=DTYPE).unsqueeze(1)
    ux = shear * y.expand(ny, nx)
    uy = torch.zeros_like(ux)
    return ux, uy


def _make_parabolic(ny: int = 32, nx: int = 64, u_max: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Parabolic profile: u = 4*u_max*y*(1-y), v = 0."""
    y = torch.arange(ny, dtype=DTYPE).unsqueeze(1) / (ny - 1)
    ux = 4.0 * u_max * y * (1.0 - y)
    ux = ux.expand(ny, nx)
    uy = torch.zeros_like(ux)
    return ux, uy


def _make_d_wall(ny: int, nx: int) -> torch.Tensor:
    """Wall distance: min(y, ny-1-y) for channel flow."""
    y = torch.arange(ny, dtype=DTYPE).unsqueeze(1)
    dw = torch.minimum(y, torch.tensor(ny - 1, dtype=DTYPE) - y)
    return dw.expand(ny, nx)


# ---------------------------------------------------------------------------
# T1: Shielding function range
# ---------------------------------------------------------------------------


class TestShieldingRange:
    def test_fd_in_unit_interval_random(self) -> None:
        nu = 1e-5
        nu_t = torch.rand(20, 40, dtype=DTYPE) * 0.1
        S_mag = torch.rand(20, 40, dtype=DTYPE) * 0.5 + 0.01
        d_wall = torch.rand(20, 40, dtype=DTYPE) * 10.0 + 0.5
        f_d = ddes_shielding_function(nu, nu_t, S_mag, d_wall)
        assert (f_d >= 0.0).all(), "f_d has negative values"
        assert (f_d <= 1.0).all(), "f_d exceeds 1.0"

    def test_fd_in_unit_interval_zero_strain(self) -> None:
        nu = 1e-5
        nu_t = torch.zeros(16, 32, dtype=DTYPE)
        S_mag = torch.full((16, 32), 1e-20, dtype=DTYPE)
        d_wall = torch.ones(16, 32, dtype=DTYPE) * 5.0
        f_d = ddes_shielding_function(nu, nu_t, S_mag, d_wall)
        assert (f_d >= 0.0).all()
        assert (f_d <= 1.0).all()


# ---------------------------------------------------------------------------
# T2: Shielding function limits
# ---------------------------------------------------------------------------


class TestShieldingLimits:
    def test_fd_near_zero_for_large_rd(self) -> None:
        """Large r_d (BL region) → f_d ≈ 0."""
        nu = 1e-5
        nu_t = torch.zeros(8, 8, dtype=DTYPE)
        S_mag = torch.full((8, 8), 0.001, dtype=DTYPE)  # small strain
        d_wall = torch.full((8, 8), 0.1, dtype=DTYPE)  # very close to wall
        # r_d = (1e-5) / (0.001 * 0.41^2 * 0.01) ≈ 1e-5 / 1.68e-6 ≈ 6
        # (8*6)^3 = 384^3 ≈ 5.7e7 → tanh ≈ 1 → f_d ≈ 0
        f_d = ddes_shielding_function(nu, nu_t, S_mag, d_wall)
        assert float(f_d.mean()) < 0.1, f"f_d should be ≈0 in BL, got {f_d.mean()}"

    def test_fd_near_one_for_small_rd(self) -> None:
        """Small r_d (far-field) → f_d ≈ 1."""
        nu = 1e-5
        nu_t = torch.zeros(8, 8, dtype=DTYPE)
        S_mag = torch.full((8, 8), 1.0, dtype=DTYPE)  # large strain
        d_wall = torch.full((8, 8), 100.0, dtype=DTYPE)  # far from wall
        # r_d = 1e-5 / (1.0 * 0.41^2 * 10000) ≈ 1e-5 / 1681 ≈ 6e-9
        # (8*6e-9)^3 ≈ 1e-22 → tanh ≈ 0 → f_d ≈ 1
        f_d = ddes_shielding_function(nu, nu_t, S_mag, d_wall)
        assert float(f_d.mean()) > 0.99, f"f_d should be ≈1 in far-field, got {f_d.mean()}"


# ---------------------------------------------------------------------------
# T3: Length scale bounds
# ---------------------------------------------------------------------------


class TestLengthScaleBounds:
    def test_l_ddes_between_l_rans_and_l_les(self) -> None:
        l_rans = torch.full((10, 20), 2.0, dtype=DTYPE)
        f_d = torch.full((10, 20), 0.5, dtype=DTYPE)
        delta_max = 1.0
        c_des = 0.65
        l_les = c_des * delta_max  # 0.65

        l_ddes = ddes_length_scale(torch.ones(10, 20), f_d, delta_max, l_rans, c_des)

        # l_DDES = l_RANS - f_d * max(0, l_RANS - l_LES)
        # = 2.0 - 0.5 * max(0, 2.0 - 0.65) = 2.0 - 0.5 * 1.35 = 1.325
        expected = 2.0 - 0.5 * (2.0 - 0.65)
        assert torch.allclose(l_ddes, torch.full_like(l_ddes, expected), atol=1e-12)
        assert (l_ddes >= l_les - 1e-12).all(), "l_DDES below l_LES"
        assert (l_ddes <= l_rans + 1e-12).all(), "l_DDES above l_RANS"

    def test_l_ddes_equals_l_rans_when_fd_zero(self) -> None:
        l_rans = torch.full((8, 8), 3.0, dtype=DTYPE)
        f_d = torch.zeros(8, 8, dtype=DTYPE)
        l_ddes = ddes_length_scale(torch.ones(8, 8), f_d, 1.0, l_rans)
        assert torch.allclose(l_ddes, l_rans, atol=1e-12)

    def test_l_ddes_equals_l_les_when_fd_one_and_l_rans_gt_l_les(self) -> None:
        l_rans = torch.full((8, 8), 5.0, dtype=DTYPE)
        f_d = torch.ones(8, 8, dtype=DTYPE)
        c_des = 0.65
        l_ddes = ddes_length_scale(torch.ones(8, 8), f_d, 1.0, l_rans, c_des)
        l_les = c_des * 1.0
        assert torch.allclose(l_ddes, torch.full_like(l_ddes, l_les), atol=1e-12)


# ---------------------------------------------------------------------------
# T4-T5: Eddy viscosity positivity and clamping
# ---------------------------------------------------------------------------


class TestEddyViscosity:
    def test_nu_t_non_negative_shear(self) -> None:
        ux, uy = _make_shear_flow(32, 64, shear=0.01)
        d_wall = _make_d_wall(32, 64)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)
        nu_t, f_d, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg)
        assert (nu_t >= 0.0).all(), "ν_t has negative values"

    def test_nu_t_non_negative_parabolic(self) -> None:
        ux, uy = _make_parabolic(32, 64, u_max=0.1)
        d_wall = _make_d_wall(32, 64)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)
        nu_t, f_d, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg)
        assert (nu_t >= 0.0).all()

    def test_nu_t_clamped(self) -> None:
        ux, uy = _make_shear_flow(16, 32, shear=10.0)  # extreme shear
        d_wall = _make_d_wall(16, 32)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0, nu_t_max=0.1)
        nu_t, f_d, nu_t_mean, nu_t_max_val, shield_frac = ddes_eddy_viscosity(ux, uy, d_wall, cfg)
        assert float(nu_t.max()) <= cfg.nu_t_max + 1e-12, f"ν_t max {nu_t.max()} > {cfg.nu_t_max}"


# ---------------------------------------------------------------------------
# T6: LES mode
# ---------------------------------------------------------------------------


class TestLESMode:
    def test_les_fd_is_unity(self) -> None:
        ux, uy = _make_shear_flow(16, 32, shear=0.05)
        d_wall = _make_d_wall(16, 32)
        cfg = DDESConfig(mode="les", nu_molecular=1e-5, dx=1.0, c_s=0.1)
        nu_t, f_d, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg)
        assert torch.allclose(f_d, torch.ones_like(f_d), atol=1e-12)

    def test_les_nu_t_formula(self) -> None:
        """ν_t = (C_S Δ)² |S| for LES mode."""
        ux, uy = _make_shear_flow(32, 64, shear=0.02)
        d_wall = _make_d_wall(32, 64)
        c_s = 0.12
        dx = 1.0
        cfg = DDESConfig(mode="les", nu_molecular=1e-5, dx=dx, c_s=c_s)
        nu_t, f_d, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg)

        # For linear shear u = shear*y: |S| = |dudy| = shear (since dvdx=0, dudx=0, dvdy=0)
        # Actually |S| = sqrt(2 * 2 * S12^2) = sqrt(4 * (shear/2)^2) = shear
        from tensorlbm.ddes import _strain_rate_magnitude

        S_mag = _strain_rate_magnitude(ux, uy)
        expected = (c_s * dx) ** 2 * S_mag
        expected = torch.clamp(expected, max=cfg.nu_t_max)
        assert torch.allclose(nu_t, expected, atol=1e-10), "LES ν_t formula mismatch"


# ---------------------------------------------------------------------------
# T7: Collision stability
# ---------------------------------------------------------------------------


class TestCollisionStability:
    def test_tau_above_half(self) -> None:
        """τ_eff = 0.5 + 3(ν_mol + ν_t) must be > 0.5."""
        ux, uy = _make_shear_flow(16, 32, shear=0.01)
        d_wall = _make_d_wall(16, 32)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)
        nu_t, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg)
        tau_eff = 0.5 + 3.0 * (cfg.nu_molecular + nu_t)
        assert (tau_eff > 0.5).all(), f"τ_eff has values ≤ 0.5, min={tau_eff.min()}"


# ---------------------------------------------------------------------------
# T8: SAS source term non-negativity
# ---------------------------------------------------------------------------


class TestSASSourceTerm:
    def test_q_sas_non_negative(self) -> None:
        ux, uy = _make_parabolic(32, 64, u_max=0.1)
        k = torch.full((32, 64), 0.01, dtype=DTYPE)
        omega = torch.full((32, 64), 0.5, dtype=DTYPE)
        cfg = DDESConfig(mode="sas", nu_molecular=1e-5, dx=1.0)
        Q = sas_source_term(ux, uy, k, omega, cfg)
        assert (Q >= 0.0).all(), "Q_SAS has negative values"


# ---------------------------------------------------------------------------
# T9-T11: Differential operators
# ---------------------------------------------------------------------------


class TestDifferentialOperators:
    def test_strain_rate_non_negative(self) -> None:
        from tensorlbm.ddes import _strain_rate_magnitude

        ux, uy = _make_shear_flow(16, 32, shear=0.01)
        S_mag = _strain_rate_magnitude(ux, uy)
        assert (S_mag >= 0.0).all()

    def test_strain_rate_zero_for_uniform(self) -> None:
        from tensorlbm.ddes import _strain_rate_magnitude

        ux = torch.full((16, 32), 0.05, dtype=DTYPE)
        uy = torch.zeros(16, 32, dtype=DTYPE)
        S_mag = _strain_rate_magnitude(ux, uy)
        # Central differences of constant → 0, but +1e-20 regularization
        assert float(S_mag.max()) < 1e-9

    def test_gradient_magnitude_non_negative(self) -> None:
        from tensorlbm.ddes import _gradient_magnitude

        field = torch.rand(16, 32, dtype=DTYPE)
        g = _gradient_magnitude(field)
        assert (g >= 0.0).all()

    def test_laplacian_zero_for_constant(self) -> None:
        from tensorlbm.ddes import _laplacian

        field = torch.full((16, 32), 3.14, dtype=DTYPE)
        lap = _laplacian(field)
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-12)


# ---------------------------------------------------------------------------
# T12: DDESResult fractions in [0, 1]
# ---------------------------------------------------------------------------


class TestDDESResult:
    def test_fractions_in_unit_interval(self) -> None:
        ux, uy = _make_parabolic(32, 64, u_max=0.1)
        d_wall = _make_d_wall(32, 64)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)
        result = run_ddes_diagnostics(ux, uy, d_wall, cfg)
        assert 0.0 <= result.shield_fraction <= 1.0
        assert 0.0 <= result.rans_fraction <= 1.0
        assert 0.0 <= result.les_fraction <= 1.0
        assert abs(result.rans_fraction + result.les_fraction - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# T13: apply_ddes_collision shape and mass conservation
# ---------------------------------------------------------------------------


class TestApplyDDESCollision:
    def test_output_shape_matches(self) -> None:
        from tensorlbm.d2q9 import equilibrium

        ny, nx = 16, 32
        rho = torch.full((ny, nx), 1.0, dtype=DTYPE)
        ux, uy = _make_shear_flow(ny, nx, shear=0.01)
        d_wall = _make_d_wall(ny, nx)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)

        f = equilibrium(rho, ux, uy)
        f_out, nu_t = apply_ddes_collision(f, ux, uy, rho, d_wall, cfg)

        assert f_out.shape == f.shape

    def test_mass_conserved(self) -> None:
        """Sum of populations should be conserved by BGK collision."""
        from tensorlbm.d2q9 import equilibrium

        ny, nx = 16, 32
        rho = torch.full((ny, nx), 1.0, dtype=DTYPE)
        ux, uy = _make_shear_flow(ny, nx, shear=0.01)
        d_wall = _make_d_wall(ny, nx)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)

        f = equilibrium(rho, ux, uy)
        f_out, nu_t = apply_ddes_collision(f, ux, uy, rho, d_wall, cfg)

        mass_in = float(f.sum(dim=0).sum())
        mass_out = float(f_out.sum(dim=0).sum())
        rel_err = abs(mass_out - mass_in) / mass_in
        assert rel_err < 1e-10, f"Mass not conserved: rel_err={rel_err}"


# ---------------------------------------------------------------------------
# T14: run_ddes_diagnostics returns valid result
# ---------------------------------------------------------------------------


class TestRunDDESDiagnostics:
    def test_returns_valid_result(self) -> None:
        ux, uy = _make_parabolic(32, 64, u_max=0.1)
        d_wall = _make_d_wall(32, 64)
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0)
        result = run_ddes_diagnostics(ux, uy, d_wall, cfg)

        assert isinstance(result, DDESResult)
        assert result.nu_t_mean >= 0.0
        assert result.nu_t_max >= result.nu_t_mean
        assert 0.0 <= result.shield_fraction <= 1.0
        assert 0.0 <= result.rans_fraction <= 1.0
        assert 0.0 <= result.les_fraction <= 1.0
