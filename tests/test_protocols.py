"""Tests for tensorlbm.protocols – CollisionOperator and BoundaryCondition."""

from __future__ import annotations

import torch

from tensorlbm.protocols import BoundaryCondition, CollisionOperator

# ---------------------------------------------------------------------------
# Helper concrete implementations that satisfy the protocols
# ---------------------------------------------------------------------------


class _BGKCollide:
    """Minimal BGK collision operator satisfying CollisionOperator."""

    def __call__(self, f: torch.Tensor, tau: float) -> torch.Tensor:
        return f  # identity for testing purposes


class _NoBoundary:
    """Minimal boundary condition satisfying BoundaryCondition."""

    def __call__(self, f: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return f


class _NotACollide:
    """Object that does NOT satisfy CollisionOperator (wrong signature)."""

    def apply(self, f: torch.Tensor) -> torch.Tensor:
        return f


# ---------------------------------------------------------------------------
# CollisionOperator protocol tests
# ---------------------------------------------------------------------------


class TestCollisionOperatorProtocol:
    def test_conforming_class_is_instance(self) -> None:
        """A class with the correct __call__ signature should be recognised."""
        assert isinstance(_BGKCollide(), CollisionOperator)

    def test_non_conforming_class_is_not_instance(self) -> None:
        """An object without __call__(f, tau) should not satisfy the protocol."""
        assert not isinstance(_NotACollide(), CollisionOperator)

    def test_plain_function_satisfies_protocol(self) -> None:
        """A free function with the right signature must satisfy the protocol."""

        def my_collide(f: torch.Tensor, tau: float) -> torch.Tensor:
            return f

        assert isinstance(my_collide, CollisionOperator)

    def test_lambda_satisfies_protocol(self) -> None:
        op = lambda f, tau: f  # noqa: E731
        assert isinstance(op, CollisionOperator)

    def test_operator_can_be_called(self) -> None:
        """The conforming operator must be callable and return a tensor."""
        op = _BGKCollide()
        f = torch.ones((9, 4, 4))
        result = op(f, tau=0.6)
        assert result.shape == f.shape

    def test_protocol_is_runtime_checkable(self) -> None:
        """CollisionOperator must be decorated with @runtime_checkable."""
        # isinstance() on Protocol only works if runtime_checkable is set
        try:
            isinstance(object(), CollisionOperator)
        except TypeError:
            raise AssertionError("CollisionOperator is not runtime_checkable") from None


# ---------------------------------------------------------------------------
# BoundaryCondition protocol tests
# ---------------------------------------------------------------------------


class TestBoundaryConditionProtocol:
    def test_conforming_class_is_instance(self) -> None:
        assert isinstance(_NoBoundary(), BoundaryCondition)

    def test_non_conforming_class_is_not_instance(self) -> None:
        assert not isinstance(_NotACollide(), BoundaryCondition)

    def test_plain_function_satisfies_protocol(self) -> None:
        def my_bc(f: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return f

        assert isinstance(my_bc, BoundaryCondition)

    def test_boundary_can_be_called(self) -> None:
        bc = _NoBoundary()
        f = torch.ones((9, 4, 4))
        result = bc(f)
        assert result.shape == f.shape

    def test_boundary_accepts_kwargs(self) -> None:
        """BoundaryCondition must accept extra keyword arguments."""
        bc = _NoBoundary()
        f = torch.ones((9, 4, 4))
        mask = torch.zeros((4, 4), dtype=torch.bool)
        result = bc(f, wall_mask=mask, u_in=0.05)
        assert result.shape == f.shape

    def test_protocol_is_runtime_checkable(self) -> None:
        try:
            isinstance(object(), BoundaryCondition)
        except TypeError:
            raise AssertionError("BoundaryCondition is not runtime_checkable") from None
