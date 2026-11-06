"""Tests for the default ``torch_5op`` SUBOFF step implementation.

:mod:`tensorlbm.backends.torch_5op_step` holds the canonical
stream → force → far-field-BC → mass-correction chain that the SUBOFF runner
dispatches through the backends layer.  It had no test under ``tests/backends/``
at all, so a silent reordering (e.g. force reduction after the BC, or mass
correction on the wrong cadence) would change production drag numbers with no
red test.  The reference implementation below is the *inline production chain*
rebuilt from the same kernel primitives; equality with it is the contract.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.backends.torch_5op_step import step_torch_5op
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d, stream3d

N = 8
U_IN = 0.05
MASS_PERIOD = 10


def _uniform_flow(n: int = N, u_in: float = U_IN) -> torch.Tensor:
    rho = torch.ones((n, n, n), dtype=torch.float32)
    ux = torch.full_like(rho, u_in)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, ux, zero, zero)


def _blob_mask(n: int = N) -> torch.Tensor:
    solid = torch.zeros((n, n, n), dtype=torch.bool)
    solid[3:5, 3:5, 3:5] = True
    return solid


def _reference_chain(
    f: torch.Tensor,
    solid: torch.Tensor,
    u_in: float,
    step: int,
    mass_period: int,
    initial_mass: float,
    *,
    do_mass_correction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pre-refactor inline chain, spelled out step by step."""
    streamed = stream3d(f)
    fx, fy, fz = compute_obstacle_forces_3d(streamed, solid)
    out = far_field_bc_3d(streamed, u_in, obstacle_mask=solid)
    if do_mass_correction and step % mass_period == 0:
        out = correct_mass3d(out, initial_mass)
    return out, fx, fy, fz


# ---------------------------------------------------------------------------
# Positive: exact equivalence with the inline chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", [1, 7, MASS_PERIOD])
def test_step_matches_inline_production_chain_bitwise(step: int) -> None:
    f = _uniform_flow()
    solid = _blob_mask()
    initial_mass = float(f.sum())
    got = step_torch_5op(f, solid, U_IN, step, MASS_PERIOD, initial_mass)
    expected = _reference_chain(f, solid, U_IN, step, MASS_PERIOD, initial_mass)
    assert torch.equal(got[0], expected[0])
    for got_force, expected_force in zip(got[1:], expected[1:], strict=True):
        assert torch.equal(got_force, expected_force)


def test_step_returns_four_tensors_with_stable_shapes_and_dtype() -> None:
    f = _uniform_flow()
    result = step_torch_5op(f, _blob_mask(), U_IN, 1, MASS_PERIOD, float(f.sum()))
    assert isinstance(result, tuple) and len(result) == 4
    assert result[0].shape == f.shape
    assert result[0].dtype is torch.float32
    for force in result[1:]:
        assert force.shape == ()
        assert force.dtype is torch.float32


def test_step_does_not_mutate_its_input() -> None:
    f = _uniform_flow()
    snapshot = f.clone()
    step_torch_5op(f, _blob_mask(), U_IN, 3, MASS_PERIOD, float(f.sum()))
    assert torch.equal(f, snapshot)


def test_step_is_deterministic() -> None:
    f = _uniform_flow()
    solid = _blob_mask()
    initial_mass = float(f.sum())
    first = step_torch_5op(f, solid, U_IN, 1, MASS_PERIOD, initial_mass)
    second = step_torch_5op(f, solid, U_IN, 1, MASS_PERIOD, initial_mass)
    assert torch.equal(first[0], second[0])
    for a, b in zip(first[1:], second[1:], strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Positive: physics/BC semantics of the individual ops
# ---------------------------------------------------------------------------


def test_far_field_bc_overwrites_inlet_and_lateral_planes() -> None:
    f = _uniform_flow()
    out, *_ = step_torch_5op(f, _blob_mask(), U_IN, 1, MASS_PERIOD, float(f.sum()))
    uniform_eq = equilibrium3d(
        torch.ones((1, 1, 1)),
        torch.full((1, 1, 1), U_IN),
        torch.zeros((1, 1, 1)),
        torch.zeros((1, 1, 1)),
    ).expand(19, N, N, N)
    assert torch.allclose(out[:, :, :, 0], uniform_eq[:, :, :, 0])  # inlet x-
    assert torch.allclose(out[:, :, 0, :], uniform_eq[:, :, 0, :])  # y-
    assert torch.allclose(out[:, :, -1, :], uniform_eq[:, :, -1, :])  # y+
    assert torch.allclose(out[:, 0, :, :], uniform_eq[:, 0, :, :])  # z-
    assert torch.allclose(out[:, -1, :, :], uniform_eq[:, -1, :, :])  # z+
    # Outlet is zero-gradient (copy of the neighbouring plane).
    assert torch.allclose(out[:, :, :, -1], out[:, :, :, -2])


def test_forces_are_zero_without_obstacle() -> None:
    f = _uniform_flow()
    solid = torch.zeros((N, N, N), dtype=torch.bool)
    _, fx, fy, fz = step_torch_5op(f, solid, U_IN, 1, MASS_PERIOD, float(f.sum()))
    assert float(fx) == 0.0
    assert float(fy) == 0.0
    assert float(fz) == 0.0


def test_streamwise_drag_is_positive_and_transverse_force_vanishes_on_axis() -> None:
    f = _uniform_flow()
    _, fx, fy, fz = step_torch_5op(f, _blob_mask(), U_IN, 1, MASS_PERIOD, float(f.sum()))
    assert float(fx) > 0.0
    # The axisymmetric blob has no lateral momentum exchange.
    assert float(fy) == pytest.approx(0.0, abs=1e-7)
    assert float(fz) == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Positive/negative: mass-correction cadence
# ---------------------------------------------------------------------------


def test_mass_correction_fires_only_on_period_multiples() -> None:
    f = _uniform_flow()
    solid = _blob_mask()
    initial_mass = float(f.sum())
    doubled = 2.0 * initial_mass
    on_cadence, *_ = step_torch_5op(f, solid, U_IN, MASS_PERIOD, MASS_PERIOD, doubled)
    off_cadence, *_ = step_torch_5op(f, solid, U_IN, MASS_PERIOD - 1, MASS_PERIOD, doubled)
    assert float(on_cadence.sum()) == pytest.approx(doubled, rel=1e-6)
    # One step early the correction must not fire at all: mass stays at ~initial.
    assert float(off_cadence.sum()) == pytest.approx(initial_mass, rel=1e-6)
    assert float(off_cadence.sum()) != pytest.approx(doubled, rel=1e-6)


def test_mass_correction_retargets_the_requested_mass() -> None:
    f = _uniform_flow()
    solid = _blob_mask()
    target = 2.0 * float(f.sum())
    corrected, *_ = step_torch_5op(f, solid, U_IN, MASS_PERIOD, MASS_PERIOD, target)
    assert float(corrected.sum()) == pytest.approx(target, rel=1e-6)


def test_do_mass_correction_false_skips_the_rescale() -> None:
    f = _uniform_flow()
    solid = _blob_mask()
    initial_mass = float(f.sum())
    skipped, *_ = step_torch_5op(
        f, solid, U_IN, MASS_PERIOD, MASS_PERIOD, initial_mass, do_mass_correction=False
    )
    reference = _reference_chain(
        f, solid, U_IN, MASS_PERIOD, MASS_PERIOD, initial_mass, do_mass_correction=False
    )
    assert torch.equal(skipped, reference[0])
    # A wildly wrong target must be inert when the flag is off.
    inert, *_ = step_torch_5op(
        f, solid, U_IN, MASS_PERIOD, MASS_PERIOD, 10.0 * initial_mass, do_mass_correction=False
    )
    assert torch.equal(inert, skipped)


def test_zero_mass_period_raises_instead_of_silently_streaming() -> None:
    """Negative: ``mass_period=0`` is an unguarded modulo.

    The modulus is evaluated lazily (short-circuit ``and``) only when the flag
    is on, so the failure is a hard ``ZeroDivisionError`` rather than a quietly
    skipped correction.  Pinned here so that any future guard is a deliberate
    decision.
    """
    f = _uniform_flow()
    with pytest.raises(ZeroDivisionError):
        step_torch_5op(f, _blob_mask(), U_IN, 1, 0, float(f.sum()))


def test_zero_mass_period_is_harmless_when_correction_is_disabled() -> None:
    f = _uniform_flow()
    out = step_torch_5op(f, _blob_mask(), U_IN, 1, 0, float(f.sum()), do_mass_correction=False)
    assert bool(torch.isfinite(out[0]).all())


# ---------------------------------------------------------------------------
# Positive: short-run stability of the whole chain
# ---------------------------------------------------------------------------


def test_ten_step_loop_is_finite_and_mass_bounded() -> None:
    initial = _uniform_flow()
    solid = _blob_mask()
    initial_mass = float(initial.sum())
    f = initial.clone()
    drag = []
    for step in range(1, 11):
        f, fx, _, _ = step_torch_5op(f, solid, U_IN, step, MASS_PERIOD, initial_mass)
        drag.append(float(fx))
    assert bool(torch.isfinite(f).all())
    assert float(f.sum()) == pytest.approx(initial_mass, rel=1e-6)
    assert all(value > 0.0 for value in drag), drag


def test_non_square_domain_supported() -> None:
    shape = (6, 7, 9)
    rho = torch.ones(shape, dtype=torch.float32)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, torch.full_like(rho, U_IN), zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[2:4, 2:4, 2:4] = True
    out, fx, _, _ = step_torch_5op(f, solid, U_IN, 1, MASS_PERIOD, float(f.sum()))
    assert out.shape == f.shape
    assert float(fx) > 0.0
