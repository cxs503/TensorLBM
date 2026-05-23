"""Unit tests for the wave_bc module (Airy wave BC + sponge layer)."""
from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.wave_bc import (
    airy_wave_velocity_3d,
    apply_sponge_layer_3d,
    apply_wave_inlet_3d,
    zou_he_inlet_velocity_profile_3d,
)


# ---------------------------------------------------------------------------
# airy_wave_velocity_3d
# ---------------------------------------------------------------------------

def test_airy_wave_velocity_3d_shape() -> None:
    ux, uz = airy_wave_velocity_3d(
        t=0, amplitude=0.01, wavelength=40.0, depth=16.0,
        iz_still_water=20, nz=24, ny=8, device=torch.device("cpu"),
    )
    assert ux.shape == (24, 8)
    assert uz.shape == (24, 8)


def test_airy_wave_velocity_3d_zero_above_waterline() -> None:
    """Velocity must be exactly zero above the still-water level."""
    iz_sw = 16
    ux, uz = airy_wave_velocity_3d(
        t=0, amplitude=0.02, wavelength=40.0, depth=16.0,
        iz_still_water=iz_sw, nz=24, ny=4, device=torch.device("cpu"),
    )
    assert torch.all(ux[iz_sw + 1:] == 0.0), "ux above waterline should be zero"
    assert torch.all(uz[iz_sw + 1:] == 0.0), "uz above waterline should be zero"


def test_airy_wave_velocity_3d_nonzero_below_surface() -> None:
    """Velocity must be non-zero below the waterline when amplitude > 0."""
    ux, _ = airy_wave_velocity_3d(
        t=0, amplitude=0.03, wavelength=40.0, depth=15.0,
        iz_still_water=20, nz=24, ny=4, device=torch.device("cpu"),
    )
    # At least one cell below waterline should have non-zero ux
    assert ux[:20].abs().max() > 0.0


def test_airy_wave_velocity_3d_time_variation() -> None:
    """Profiles at t=0 and t=T/2 should differ (time-dependent)."""
    kwargs = dict(amplitude=0.02, wavelength=40.0, depth=15.0,
                  iz_still_water=20, nz=24, ny=4, device=torch.device("cpu"))
    import math
    k = 2.0 * math.pi / 40.0
    omega = math.sqrt(k * math.tanh(k * 15.0) / 3.0)
    half_period = math.pi / omega if omega > 0 else 100.0
    ux0, _ = airy_wave_velocity_3d(t=0, **kwargs)
    ux_half, _ = airy_wave_velocity_3d(t=half_period, **kwargs)
    # At t=T/2, cos(−ωt) = cos(π) = −1, so profiles should differ
    assert not torch.allclose(ux0, ux_half, atol=1e-6)


def test_airy_wave_velocity_3d_uniform_in_y() -> None:
    """For a plane wave, profiles must be uniform in the y-direction."""
    ux, uz = airy_wave_velocity_3d(
        t=5.0, amplitude=0.01, wavelength=40.0, depth=15.0,
        iz_still_water=16, nz=20, ny=6, device=torch.device("cpu"),
    )
    # All y-columns should be identical
    assert torch.allclose(ux[:, 0:1].expand_as(ux), ux)
    assert torch.allclose(uz[:, 0:1].expand_as(uz), uz)


# ---------------------------------------------------------------------------
# zou_he_inlet_velocity_profile_3d
# ---------------------------------------------------------------------------

def test_zou_he_inlet_profile_shape() -> None:
    rho = torch.ones(4, 8, 16)
    ux = torch.full((4, 8, 16), 0.05)
    f = equilibrium3d(rho, ux, torch.zeros_like(ux), torch.zeros_like(ux))
    ux_p = torch.full((4, 8), 0.05)
    uy_p = torch.zeros(4, 8)
    uz_p = torch.zeros(4, 8)
    f_out = zou_he_inlet_velocity_profile_3d(f, ux_p, uy_p, uz_p)
    assert f_out.shape == f.shape


def test_zou_he_inlet_profile_finite() -> None:
    torch.manual_seed(0)
    f = torch.rand(19, 6, 8, 12) * 0.05 + 0.05
    ux_p = torch.full((6, 8), 0.06)
    uy_p = torch.zeros(6, 8)
    uz_p = torch.zeros(6, 8)
    f_out = zou_he_inlet_velocity_profile_3d(f, ux_p, uy_p, uz_p)
    assert torch.isfinite(f_out).all()


def test_zou_he_inlet_profile_prescribes_velocity() -> None:
    """After applying the BC, macroscopic ux at x=0 must match the profile."""
    nz, ny, nx = 6, 8, 12
    rho0 = torch.ones(nz, ny, nx)
    f = equilibrium3d(rho0, torch.zeros_like(rho0),
                      torch.zeros_like(rho0), torch.zeros_like(rho0))
    # Perturb away from equilibrium
    from tensorlbm.solver3d import collide_bgk3d, stream3d
    f = collide_bgk3d(f, tau=0.6)
    f = stream3d(f)

    u_target = 0.07
    ux_p = torch.full((nz, ny), u_target)
    uy_p = torch.zeros(nz, ny)
    uz_p = torch.zeros(nz, ny)
    f_out = zou_he_inlet_velocity_profile_3d(f, ux_p, uy_p, uz_p)

    _, ux_out, _, _ = macroscopic3d(f_out)
    assert torch.allclose(ux_out[:, :, 0], torch.full((nz, ny), u_target), atol=2e-4)


# ---------------------------------------------------------------------------
# apply_sponge_layer_3d
# ---------------------------------------------------------------------------

def test_sponge_layer_no_change_outside() -> None:
    """Sponge layer must not modify cells at x < ix_start."""
    nz, ny, nx = 4, 6, 32
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.05)
    f = equilibrium3d(rho, ux, torch.zeros_like(ux), torch.zeros_like(ux))
    f_orig = f.clone()

    rho_t = torch.ones_like(rho)
    f_out = apply_sponge_layer_3d(f, rho_t,
                                  torch.zeros_like(ux), torch.zeros_like(ux),
                                  torch.zeros_like(ux),
                                  ix_start=24)
    assert torch.allclose(f_out[:, :, :, :24], f_orig[:, :, :, :24])


def test_sponge_layer_moves_f_towards_target() -> None:
    """At the outlet, f must move towards the target equilibrium."""
    nz, ny, nx = 4, 6, 32
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.1)
    f = equilibrium3d(rho, ux, torch.zeros_like(ux), torch.zeros_like(ux))

    rho_t = torch.ones_like(rho)
    feq_t = equilibrium3d(rho_t, torch.zeros_like(ux),
                          torch.zeros_like(ux), torch.zeros_like(ux))
    f_out = apply_sponge_layer_3d(f, rho_t,
                                  torch.zeros_like(ux), torch.zeros_like(ux),
                                  torch.zeros_like(ux),
                                  ix_start=20, sigma_max=0.5)

    diff_before = (f[:, :, :, -1] - feq_t[:, :, :, -1]).abs().mean()
    diff_after = (f_out[:, :, :, -1] - feq_t[:, :, :, -1]).abs().mean()
    assert float(diff_after) < float(diff_before)


def test_sponge_layer_ix_start_beyond_domain() -> None:
    """If ix_start >= nx-1, f must be returned unchanged."""
    nz, ny, nx = 4, 6, 16
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.05)
    f = equilibrium3d(rho, ux, torch.zeros_like(ux), torch.zeros_like(ux))
    f_orig = f.clone()
    f_out = apply_sponge_layer_3d(f, rho,
                                  torch.zeros_like(ux), torch.zeros_like(ux),
                                  torch.zeros_like(ux),
                                  ix_start=nx)
    assert torch.allclose(f_out, f_orig)


# ---------------------------------------------------------------------------
# apply_wave_inlet_3d
# ---------------------------------------------------------------------------

def test_apply_wave_inlet_3d_shape() -> None:
    nz, ny, nx = 24, 8, 32
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.05)
    f = equilibrium3d(rho, ux, torch.zeros_like(ux), torch.zeros_like(ux))
    f_out = apply_wave_inlet_3d(
        f, t=0, amplitude=0.01, wavelength=40.0, depth=16.0,
        iz_still_water=20, mean_ux=0.05,
    )
    assert f_out.shape == f.shape


def test_apply_wave_inlet_3d_finite() -> None:
    nz, ny, nx = 24, 8, 32
    torch.manual_seed(0)
    f = torch.rand(19, nz, ny, nx) * 0.05 + 0.05
    f_out = apply_wave_inlet_3d(
        f, t=10.0, amplitude=0.01, wavelength=40.0, depth=16.0,
        iz_still_water=20, mean_ux=0.04,
    )
    assert torch.isfinite(f_out).all()
