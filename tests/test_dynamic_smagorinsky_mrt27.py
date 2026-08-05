"""Contract + integration tests for the dynamic Smagorinsky MRT (D3Q27) closure.

This is the last turbulence collision gap: D3Q27 + MRT + dynamic Smagorinsky.
It combines the dynamic Smagorinsky procedure (Germano identity with test
filtering to compute a global C_s) — already implemented for D3Q27 BGK —
with the D3Q27 MRT collision operator whose stress modes 5–9 are overridden
by the per-cell effective relaxation rate, exactly as in
:func:`collide_smagorinsky_mrt27`.

Contract tests verify operator algebra (shape, finiteness, mass/momentum
conservation, equilibrium fixed-point, non-equilibrium relaxation), NOT
turbulence physics correctness.  A small sphere-flow integration test
exercises the operator inside a stream–collide–boundary loop to confirm
end-to-end finiteness and drag-force computation.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    make_channel_wall_mask_27,
)
from tensorlbm.d3q27 import equilibrium27, macroscopic27, stream27
from tensorlbm.obstacles import compute_obstacle_forces_27
from tensorlbm.turbulence import collide_dynamic_smagorinsky_mrt27


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_d3q27_state(seed: int = 99) -> torch.Tensor:
    """Build a small D3Q27 equilibrium distribution with mild shear."""
    torch.manual_seed(seed)
    shape = (4, 6, 8)
    rho = 1.0 + 0.01 * torch.rand(shape)
    ux = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    uy = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    uz = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    return equilibrium27(rho, ux, uy, uz)


# ---------------------------------------------------------------------------
# Contract tests — operator algebra
# ---------------------------------------------------------------------------

def test_dyn_smag_mrt27_shape() -> None:
    f = _make_d3q27_state()
    assert collide_dynamic_smagorinsky_mrt27(f, tau=0.7).shape == f.shape


def test_dyn_smag_mrt27_output_finite() -> None:
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(f, tau=0.7)
    assert torch.isfinite(fout).all()


def test_dyn_smag_mrt27_preserves_mass() -> None:
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(f, tau=0.7)
    rho_before = macroscopic27(f)[0]
    rho_after = macroscopic27(fout)[0]
    torch.testing.assert_close(rho_after, rho_before, rtol=1e-6, atol=1e-7)


def test_dyn_smag_mrt27_preserves_momentum() -> None:
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(f, tau=0.7)
    _, ux_b, uy_b, uz_b = macroscopic27(f)
    _, ux_a, uy_a, uz_a = macroscopic27(fout)
    torch.testing.assert_close(ux_a, ux_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(uy_a, uy_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(uz_a, uz_b, rtol=1e-6, atol=1e-7)


def test_dyn_smag_mrt27_equilibrium_fixed_point() -> None:
    """Equilibrium distribution must be a collision fixed point."""
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(f, tau=0.7)
    torch.testing.assert_close(fout, f, rtol=1e-6, atol=1e-7)


def test_dyn_smag_mrt27_relaxes_non_equilibrium() -> None:
    """A non-equilibrium perturbation must move toward equilibrium."""
    f = _make_d3q27_state()
    f_neq = f + 1e-3 * torch.randn_like(f)
    fout = collide_dynamic_smagorinsky_mrt27(f_neq, tau=0.7)
    feq = f
    err_before = (f_neq - feq).abs().mean()
    err_after = (fout - feq).abs().mean()
    assert err_after < err_before


def test_dyn_smag_mrt27_accepts_mrt_rates() -> None:
    """MRT relaxation rates should be accepted as keyword arguments."""
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(
        f, tau=0.7, s_e=1.19, s_eps=1.4, s_q=1.2
    )
    assert fout.shape == f.shape
    assert torch.isfinite(fout).all()


def test_dyn_smag_mrt27_s_pi_defaults_to_s_e() -> None:
    """s_pi=None should default to s_e without error."""
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(
        f, tau=0.7, s_e=1.19, s_eps=1.4, s_q=1.2, s_pi=None
    )
    assert torch.isfinite(fout).all()


def test_dyn_smag_mrt27_lambda_clip_accepted() -> None:
    """lambda_clip keyword should be accepted (dynamic procedure parameter)."""
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(f, tau=0.7, lambda_clip=0.0)
    assert torch.isfinite(fout).all()


def test_dyn_smag_mrt27_filter_width_accepted() -> None:
    """filter_width keyword should be accepted (test-filter width)."""
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_mrt27(f, tau=0.7, filter_width=3)
    assert torch.isfinite(fout).all()


# ---------------------------------------------------------------------------
# Consistency: dynamic MRT27 vs dynamic BGK27 on equilibrium
# ---------------------------------------------------------------------------

def test_dyn_smag_mrt27_consistent_with_bgk27_on_equilibrium() -> None:
    """Both dynamic D3Q27 variants should leave equilibrium unchanged."""
    from tensorlbm.turbulence import collide_dynamic_smagorinsky_bgk27

    torch.manual_seed(7)
    shape = (3, 5, 7)
    rho = torch.ones(shape)
    ux = 0.01 * torch.rand(shape)
    uy = 0.01 * torch.rand(shape)
    uz = 0.01 * torch.rand(shape)
    f27 = equilibrium27(rho, ux, uy, uz)
    out_mrt = collide_dynamic_smagorinsky_mrt27(f27, tau=0.7)
    out_bgk = collide_dynamic_smagorinsky_bgk27(f27, tau=0.7)
    torch.testing.assert_close(out_mrt, f27, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(out_bgk, f27, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Integration: mini sphere flow with dynamic Smagorinsky MRT D3Q27
# ---------------------------------------------------------------------------

def test_dyn_smag_mrt27_sphere_flow_finite() -> None:
    """A few stream–collide–boundary steps past a sphere must stay finite.

    This exercises the new collision operator inside a realistic D3Q27
    channel-flow loop (Zou/He inlet, bounce-back walls, sphere obstacle,
    momentum-exchange drag) without modifying the production solver.
    """
    nz, ny, nx = 8, 12, 16
    radius = 2.0
    u_in = 0.05
    re = 20.0
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    device = torch.device("cpu")
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    obstacle = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    wall_mask = make_channel_wall_mask_27(nz, ny, nx, obstacle, device=device)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[obstacle] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))

    fx = fy = fz = torch.tensor(0.0)
    for _ in range(5):
        f = collide_dynamic_smagorinsky_mrt27(f, tau=tau)
        f = stream27(f)
        fx, fy, fz = compute_obstacle_forces_27(f, obstacle)
        f = apply_zou_he_channel_boundaries_27(
            f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=obstacle
        )

    rho, ux, uy, uz = macroscopic27(f)
    # Mask velocity inside the obstacle (production pattern)
    ux = ux.masked_fill(obstacle, 0.0)
    assert torch.isfinite(f).all()
    assert torch.isfinite(rho).all()
    assert torch.isfinite(ux).all()
    # Drag force on the sphere must be finite and non-NaN
    assert math.isfinite(float(fx))
    assert math.isfinite(float(fy))
    assert math.isfinite(float(fz))
    # Velocity inside the obstacle should be zero after masking
    assert ux[obstacle].abs().max() < 1e-6
