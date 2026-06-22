"""Core library tests for the four gap-closure improvements.

These tests run without FastAPI and exercise the pure-Python/PyTorch modules:
- synthetic_inflow  (DFSEM + Digital Filter)
- roughness         (sand-grain wall BC)
- sponge_bc         (absorbing outlet layer)
- turbulence_stats  (Reynolds stresses, TKE, Tu)
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# synthetic_inflow
# ---------------------------------------------------------------------------

def test_dfsem_imports_from_tensorlbm() -> None:
    from tensorlbm import DFSEMInlet, DigitalFilterInlet  # noqa: F401


def test_dfsem_produces_finite_values() -> None:
    from tensorlbm import DFSEMInlet

    ny = 32
    gen = DFSEMInlet(ny=ny, nz=1, u_mean=torch.full((ny, 1), 0.1),
                      uu=1e-4, vv=1e-4, ww=1e-4, length_scale=4.0,
                      n_eddies=50, seed=0)
    u, v, w = gen.sample()
    assert torch.isfinite(u).all()
    assert torch.isfinite(v).all()
    assert torch.isfinite(w).all()


def test_dfm_produces_finite_values() -> None:
    from tensorlbm import DigitalFilterInlet

    gen = DigitalFilterInlet(ny=32, nz=1, uu=1e-4, vv=1e-4, ww=1e-4,
                              length_scale=4.0, seed=0)
    u, v, w = gen.sample()
    assert torch.isfinite(u).all()
    assert torch.isfinite(v).all()


def test_dfsem_stress_ordering() -> None:
    """Larger prescribed stress → larger RMS fluctuation."""
    from tensorlbm import DFSEMInlet

    ny = 64
    base_kwargs = dict(ny=ny, nz=1, u_mean=torch.full((ny, 1), 0.1),
                       length_scale=4.0, n_eddies=100)

    gen_lo = DFSEMInlet(uu=1e-5, vv=1e-5, ww=1e-5, seed=42, **base_kwargs)
    gen_hi = DFSEMInlet(uu=1e-3, vv=1e-3, ww=1e-3, seed=42, **base_kwargs)

    u_lo, _, _ = gen_lo.sample()
    u_hi, _, _ = gen_hi.sample()
    assert u_hi.abs().mean() > u_lo.abs().mean()


# ---------------------------------------------------------------------------
# roughness
# ---------------------------------------------------------------------------

def test_roughness_imports_from_tensorlbm() -> None:
    from tensorlbm import (  # noqa: F401
        apply_rough_wall_bounce_back,
        compute_rough_wall_slip_velocity,
        roughness_b_correction,
    )


def test_roughness_correction_monotone() -> None:
    from tensorlbm import roughness_b_correction

    ks_plus = torch.linspace(0.1, 500, 100)
    delta_b = roughness_b_correction(ks_plus)
    # Non-decreasing (up to floating-point tolerance)
    assert (delta_b[1:] >= delta_b[:-1] - 1e-5).all()


def test_roughness_smooth_zero_b() -> None:
    from tensorlbm import roughness_b_correction

    # ks+ < 2.25 is hydraulically smooth
    ks_plus = torch.tensor([0.1, 0.5, 2.0])
    delta_b = roughness_b_correction(ks_plus)
    assert (delta_b.abs() < 1e-5).all()


# ---------------------------------------------------------------------------
# sponge_bc
# ---------------------------------------------------------------------------

def test_sponge_imports_from_tensorlbm() -> None:
    from tensorlbm import (  # noqa: F401
        apply_target_sponge_2d,
        apply_target_sponge_3d,
        apply_viscous_sponge_2d,
        apply_viscous_sponge_3d,
        build_mean_equilibrium_2d,
        build_mean_equilibrium_3d,
        sponge_profile,
    )


def test_sponge_profile_zero_before_x0() -> None:
    from tensorlbm import sponge_profile

    profile = sponge_profile(nx=100, x0=80, x1=99, amplitude=0.5)
    assert (profile[:80] == 0.0).all()


def test_sponge_profile_max_at_x1() -> None:
    from tensorlbm import sponge_profile

    profile = sponge_profile(nx=100, x0=80, x1=99, amplitude=0.7, exponent=1.0)
    assert abs(float(profile[99]) - 0.7) < 1e-5


def test_viscous_sponge_2d_conserves_shape() -> None:
    from tensorlbm import (
        apply_viscous_sponge_2d,
        equilibrium,
        macroscopic,
        sponge_profile,
    )

    ny, nx = 16, 32
    rho = torch.ones(ny, nx)
    ux = torch.full((ny, nx), 0.1)
    uy = torch.zeros(ny, nx)
    f = equilibrium(rho, ux, uy)
    sponge = sponge_profile(nx=nx, x0=24, x1=31)
    f_out = apply_viscous_sponge_2d(f, rho, ux, uy, tau0=0.8, sponge=sponge)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()


def test_target_sponge_2d_full_damping() -> None:
    """beta=1 everywhere → f_out == f_target."""
    from tensorlbm import apply_target_sponge_2d

    ny, nx = 8, 16
    f = torch.rand(9, ny, nx)
    f_target = torch.zeros_like(f)
    sponge = torch.ones(nx)
    f_out = apply_target_sponge_2d(f, f_target, sponge)
    assert torch.allclose(f_out, f_target, atol=1e-6)


def test_target_sponge_3d_no_damping() -> None:
    """beta=0 → f_out == f."""
    from tensorlbm import apply_target_sponge_3d

    f = torch.rand(19, 4, 4, 8)
    f_target = torch.zeros_like(f)
    sponge = torch.zeros(8)
    f_out = apply_target_sponge_3d(f, f_target, sponge)
    assert torch.allclose(f_out, f, atol=1e-6)


# ---------------------------------------------------------------------------
# turbulence_stats
# ---------------------------------------------------------------------------

def test_turbstats_imports_from_tensorlbm() -> None:
    from tensorlbm import (  # noqa: F401
        TurbulenceStatsAccumulator,
        compute_reynolds_stresses,
        compute_turbulence_intensity,
        compute_turbulence_length_scale,
        turbulence_stats_from_checkpoints,
    )


def test_turbstats_accumulates_correctly() -> None:
    from tensorlbm import TurbulenceStatsAccumulator

    acc = TurbulenceStatsAccumulator()
    # All-constant velocity → variance should be ~0
    for _ in range(5):
        acc.update(torch.full((8, 8), 0.1), torch.full((8, 8), 0.02))

    assert acc.count == 5
    assert acc.uu.max().item() < 1e-10
    assert (acc.tke < 1e-10).all()


def test_turbstats_variance_positive_with_fluctuations() -> None:
    from tensorlbm import TurbulenceStatsAccumulator

    acc = TurbulenceStatsAccumulator()
    torch.manual_seed(0)
    for _ in range(20):
        acc.update(torch.rand(8, 8) * 0.1, torch.rand(8, 8) * 0.02)

    assert (acc.uu > 0).any()


def test_turbstats_reynolds_stresses_fn() -> None:
    from tensorlbm import compute_reynolds_stresses

    ux_mean = torch.full((4, 4), 0.1)
    uy_mean = torch.zeros(4, 4)
    result = compute_reynolds_stresses(
        ux_mean, uy_mean,
        ux_rms=torch.full((4, 4), 0.01),
        uy_rms=torch.full((4, 4), 0.005),
    )
    assert (result["uu"] > 0).all()
    assert (result["tke"] > 0).all()
    assert (result["tu_percent"] > 0).all()


def test_turbstats_length_scale_synthetic_signal() -> None:
    from tensorlbm import compute_turbulence_length_scale

    # Sinusoidal signal with known period
    t = torch.linspace(0, 4 * torch.pi, 256)
    signal = torch.sin(t)
    L = compute_turbulence_length_scale(signal, dt=1.0, u_conv=1.0)
    # Integral length scale should be > 0 for periodic signal
    assert L > 0.0


def test_turbstats_accumulator_to_dict_complete() -> None:
    from tensorlbm import TurbulenceStatsAccumulator

    acc = TurbulenceStatsAccumulator()
    acc.update(torch.rand(4, 4), torch.rand(4, 4))
    d = acc.to_dict()

    required_keys = {
        "n_samples", "mean_u", "mean_v", "uu", "vv", "uv",
        "tke", "skewness_u", "flatness_u",
    }
    assert required_keys.issubset(d.keys())


def test_turbstats_3d_ww_nonzero() -> None:
    from tensorlbm import TurbulenceStatsAccumulator

    acc = TurbulenceStatsAccumulator(is_3d=True)
    torch.manual_seed(5)
    for _ in range(5):
        uz = torch.rand(4, 4, 4) * 0.05
        acc.update(torch.rand(4, 4, 4) * 0.1,
                   torch.rand(4, 4, 4) * 0.02,
                   uz)

    assert acc.ww is not None
    assert (acc.ww >= 0).all()
