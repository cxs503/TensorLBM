from __future__ import annotations

import torch

from tensorlbm.nonequilibrium_wall_selector import NonequilibriumWallSelector


def _state(parameter: float, *, u_tau: float = 0.02, y_plus: float = 100.0):
    return (
        torch.tensor((parameter,)),
        torch.tensor((u_tau,)),
        torch.tensor((y_plus,)),
    )


def test_selector_requires_persistent_adverse_gradient_and_hysteresis() -> None:
    selector = NonequilibriumWallSelector(
        enter_observations=3,
        exit_observations=2,
    )
    for _ in range(2):
        active, _ = selector.update(*_state(2.0))
        assert not bool(active.item())
    active, diagnostics = selector.update(*_state(2.0))
    assert bool(active.item())
    assert diagnostics.newly_activated_nodes == 1

    # Between exit and entry thresholds, the existing selection is retained.
    active, _ = selector.update(*_state(0.75))
    assert bool(active.item())
    active, _ = selector.update(*_state(0.1))
    assert bool(active.item())
    active, diagnostics = selector.update(*_state(0.1))
    assert not bool(active.item())
    assert diagnostics.newly_deactivated_nodes == 1


def test_selector_low_shear_and_invalid_evidence_fail_closed_immediately() -> None:
    selector = NonequilibriumWallSelector(enter_observations=1)
    active, _ = selector.update(*_state(2.0))
    assert bool(active.item())

    active, diagnostics = selector.update(*_state(2.0, u_tau=1.0e-12))
    assert not bool(active.item())
    assert diagnostics.low_shear_rejected_nodes == 1

    selector.reset()
    active, _ = selector.update(
        *_state(2.0),
        valid=torch.tensor((False,)),
    )
    assert not bool(active.item())


def test_selector_rejects_out_of_policy_y_plus_and_favourable_gradient() -> None:
    selector = NonequilibriumWallSelector(enter_observations=1)
    active, diagnostics = selector.update(*_state(2.0, y_plus=1500.0))
    assert not bool(active.item())
    assert diagnostics.y_plus_rejected_nodes == 1

    active, _ = selector.update(*_state(-20.0))
    assert not bool(active.item())


def test_selector_requires_reset_before_shape_change() -> None:
    selector = NonequilibriumWallSelector()
    selector.update(*_state(0.0))
    try:
        selector.update(
            torch.zeros(2),
            torch.ones(2),
            torch.full((2,), 100.0),
        )
    except ValueError as error:
        assert "reset" in str(error)
    else:
        raise AssertionError("selector accepted an implicit state resize")
