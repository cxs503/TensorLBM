"""Contract tests for a pure paired common-flux transaction."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.free_surface_common_flux_transaction import (
    CommonFluxTransaction,
    FluxTransactionError,
)


def _transaction(*, active: bool = True, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> CommonFluxTransaction:
    return CommonFluxTransaction(active=active, atol=atol, rtol=rtol)


def test_plan_stage_validate_commit_preserves_inputs_and_returns_independent_buffers() -> None:
    interface = torch.tensor([1.25, 2.50], dtype=torch.float64)
    counterpart = torch.tensor([3.75, 4.50], dtype=torch.float64)
    interface_before = interface.clone()
    counterpart_before = counterpart.clone()

    planned = _transaction().plan(
        interface,
        counterpart,
        interface_delta=torch.tensor([0.30, -0.10], dtype=torch.float64),
        counterpart_delta=torch.tensor([-0.30, 0.10], dtype=torch.float64),
    )
    staged = planned.stage()
    validation = staged.validate()
    committed_interface, committed_counterpart = validation.commit()

    assert validation.valid
    assert validation.residual == pytest.approx(0.0)
    assert torch.equal(interface, interface_before)
    assert torch.equal(counterpart, counterpart_before)
    assert torch.equal(committed_interface, torch.tensor([1.55, 2.40], dtype=torch.float64))
    assert torch.equal(committed_counterpart, torch.tensor([3.45, 4.60], dtype=torch.float64))
    assert committed_interface.data_ptr() != interface.data_ptr()
    assert committed_counterpart.data_ptr() != counterpart.data_ptr()
    committed_interface.add_(10.0)
    assert torch.equal(interface, interface_before)


def test_inactive_transaction_fails_closed_before_staging() -> None:
    with pytest.raises(FluxTransactionError, match="active"):
        _transaction(active=False).plan(
            torch.tensor([1.0]), torch.tensor([2.0]),
            interface_delta=torch.tensor([0.1]), counterpart_delta=torch.tensor([-0.1]),
        )


def test_commit_rejects_material_unpaired_residual_without_mutating_inputs() -> None:
    interface = torch.tensor([1.0, 2.0])
    counterpart = torch.tensor([3.0, 4.0])
    planned = _transaction(atol=1.0e-7, rtol=0.0).plan(
        interface,
        counterpart,
        interface_delta=torch.tensor([0.4, 0.0]),
        counterpart_delta=torch.tensor([-0.3, 0.0]),
    )
    validation = planned.stage().validate()

    assert not validation.valid
    assert validation.residual == pytest.approx(0.1)
    with pytest.raises(FluxTransactionError, match="residual"):
        validation.commit()
    assert torch.equal(interface, torch.tensor([1.0, 2.0]))
    assert torch.equal(counterpart, torch.tensor([3.0, 4.0]))


def test_non_finite_non_floating_or_shape_mismatch_inputs_fail_closed() -> None:
    transaction = _transaction()
    cases = [
        (
            torch.tensor([1.0]), torch.tensor([2.0]),
            torch.tensor([float("nan")]), torch.tensor([0.0]),
        ),
        (
            torch.tensor([1], dtype=torch.int64), torch.tensor([2.0]),
            torch.tensor([0.0]), torch.tensor([0.0]),
        ),
        (
            torch.tensor([1.0, 2.0]), torch.tensor([3.0]),
            torch.tensor([0.0, 0.0]), torch.tensor([0.0]),
        ),
    ]
    for interface, counterpart, interface_delta, counterpart_delta in cases:
        with pytest.raises(FluxTransactionError):
            transaction.plan(interface, counterpart, interface_delta, counterpart_delta)


@pytest.mark.parametrize("bad_inventory", [[1.0], None, "not-a-tensor"])
@pytest.mark.parametrize("inventory_name", ["interface", "counterpart"])
def test_non_tensor_inventories_fail_closed_with_domain_error(
    inventory_name: str, bad_inventory: object,
) -> None:
    inputs: dict[str, object] = {
        "interface": torch.tensor([1.0]),
        "counterpart": torch.tensor([2.0]),
        "interface_delta": torch.tensor([0.1]),
        "counterpart_delta": torch.tensor([-0.1]),
    }
    inputs[inventory_name] = bad_inventory

    with pytest.raises(FluxTransactionError, match=inventory_name):
        _transaction().plan(**inputs)  # type: ignore[arg-type]


def test_tolerance_uses_atol_plus_rtol_times_staged_scale_and_rejects_invalid_policy() -> None:
    transaction = _transaction(atol=0.01, rtol=0.01)
    valid = transaction.plan(
        torch.tensor([100.0]), torch.tensor([100.0]),
        torch.tensor([1.02]), torch.tensor([-1.00]),
    ).stage().validate()
    invalid = transaction.plan(
        torch.tensor([100.0]), torch.tensor([100.0]),
        torch.tensor([1.03]), torch.tensor([-1.00]),
    ).stage().validate()

    # Scale is the largest absolute aggregate endpoint flux (1.02 here),
    # rather than either inventory total or a global correction magnitude.
    assert valid.tolerance == pytest.approx(0.0202)
    assert valid.valid
    assert invalid.residual == pytest.approx(0.03)
    assert not invalid.valid
    with pytest.raises(FluxTransactionError, match="tolerance"):
        _transaction(atol=-1.0)
    with pytest.raises(FluxTransactionError, match="tolerance"):
        _transaction(rtol=float("inf"))
