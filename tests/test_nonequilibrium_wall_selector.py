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


def test_selector_checkpoint_preserves_temporal_hysteresis() -> None:
    original = NonequilibriumWallSelector(
        enter_observations=3,
        exit_observations=2,
    )
    original.update(*_state(2.0))
    original.update(*_state(2.0))
    state = original.state_dict()

    # Returned tensors are detached copies, not mutable views of live state.
    assert isinstance(state["enter_count"], torch.Tensor)
    state["enter_count"].zero_()
    active, diagnostics = original.update(*_state(2.0))
    assert bool(active.item())
    assert diagnostics.newly_activated_nodes == 1

    restored = NonequilibriumWallSelector(
        enter_observations=3,
        exit_observations=2,
    )
    # Restore the actual two-observation state and cross the entry threshold.
    original.reset()
    original.update(*_state(2.0))
    original.update(*_state(2.0))
    restored.load_state_dict(original.state_dict())
    active, diagnostics = restored.update(*_state(2.0))
    assert bool(active.item())
    assert diagnostics.newly_activated_nodes == 1


def test_selector_checkpoint_rejects_configuration_and_counter_mismatch() -> None:
    source = NonequilibriumWallSelector(enter_observations=3)
    source.update(*_state(2.0))
    state = source.state_dict()

    incompatible = NonequilibriumWallSelector(enter_observations=4)
    try:
        incompatible.load_state_dict(state)
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("selector accepted incompatible configuration")

    corrupt = source.state_dict()
    assert isinstance(corrupt["enter_count"], torch.Tensor)
    corrupt["enter_count"].fill_(4)
    try:
        source.load_state_dict(corrupt)
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("selector accepted an impossible counter")


def test_uninitialised_selector_checkpoint_round_trip() -> None:
    source = NonequilibriumWallSelector()
    restored = NonequilibriumWallSelector()
    restored.load_state_dict(source.state_dict())
    assert restored.active is None
