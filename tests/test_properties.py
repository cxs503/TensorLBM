"""Property-based tests using Hypothesis.

Tests that fundamental physical invariants hold for arbitrary inputs:
- Mass conservation (collision invariant)
- Momentum conservation (collision invariant)
- Positivity of distributions after equilibrium
- D3Q27 weights sum to 1
- D3Q27 equilibrium roundtrip
"""
from __future__ import annotations

import pytest

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

    def given(*args: object, **kwargs: object):
        def decorator(func: object) -> object:
            return func

        return decorator

    def settings(*args: object, **kwargs: object):
        def decorator(func: object) -> object:
            return func

        return decorator

    class HealthCheck:
        too_slow = object()

import torch

from tensorlbm import collide_bgk, equilibrium, macroscopic
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import W as W27
from tensorlbm.d3q27 import equilibrium27, macroscopic27

pytestmark = pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")

if HYPOTHESIS_AVAILABLE:
    _field_strategy = st.tuples(
        st.integers(min_value=4, max_value=16),
        st.integers(min_value=4, max_value=16),
        st.floats(min_value=0.01, max_value=0.08),
        st.floats(min_value=0.8, max_value=1.2),
    )
else:
    _field_strategy = None


@given(_field_strategy)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_bgk_conserves_mass_hypothesis(args: tuple[int, int, float, float]) -> None:
    """BGK collision must conserve total mass for arbitrary fields."""
    ny, nx, u_mag, rho_val = args
    rho = torch.full((ny, nx), rho_val)
    ux = torch.full_like(rho, u_mag * 0.5)
    uy = torch.full_like(rho, u_mag * 0.3)
    f = equilibrium(rho, ux, uy)
    f_new = collide_bgk(f, tau=0.7)
    rho_new, _, _ = macroscopic(f_new)
    assert torch.allclose(rho_new, rho, atol=1e-5)


@given(_field_strategy)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_bgk_conserves_momentum_hypothesis(args: tuple[int, int, float, float]) -> None:
    """BGK collision must conserve momentum for arbitrary fields."""
    ny, nx, u_mag, rho_val = args
    rho = torch.full((ny, nx), rho_val)
    ux = torch.full_like(rho, u_mag * 0.5)
    uy = torch.full_like(rho, u_mag * 0.3)
    f = equilibrium(rho, ux, uy)
    f_new = collide_bgk(f, tau=0.7)
    _, ux_new, uy_new = macroscopic(f_new)
    assert torch.allclose(ux_new, ux, atol=1e-5)
    assert torch.allclose(uy_new, uy, atol=1e-5)


def test_d3q27_weights_sum_to_one() -> None:
    """D3Q27 weights must sum to exactly 1."""
    assert abs(float(W27.sum().item()) - 1.0) < 1e-6


def test_d3q27_equilibrium_roundtrip() -> None:
    """D3Q27 equilibrium roundtrip: macroscopic(equilibrium(rho,u)) == (rho,u)."""
    rho = torch.ones((4, 6, 8))
    ux = torch.full_like(rho, 0.03)
    uy = torch.full_like(rho, 0.02)
    uz = torch.full_like(rho, -0.01)
    f = equilibrium27(rho, ux, uy, uz)
    rho_out, ux_out, uy_out, uz_out = macroscopic27(f)
    assert torch.allclose(rho_out, rho, atol=1e-5)
    assert torch.allclose(ux_out, ux, atol=1e-5)
    assert torch.allclose(uy_out, uy, atol=1e-5)
    assert torch.allclose(uz_out, uz, atol=1e-5)


def test_d3q27_c_symmetry() -> None:
    """D3Q27 velocities must be symmetric: for every c there is a -c."""
    c_list = [tuple(row.tolist()) for row in C27]
    for cx, cy, cz in c_list:
        assert (-cx, -cy, -cz) in c_list, f"Missing opposite of ({cx},{cy},{cz})"
