"""Abstract Protocol interfaces for TensorLBM extension points.

These :class:`typing.Protocol` classes define the structural contracts that
custom collision operators and boundary conditions must satisfy. Users can
register their own implementations without modifying the core library, as
long as their classes conform to the protocol signatures.

Usage example
-------------
.. code-block:: python

    from tensorlbm.protocols import CollisionOperator

    class MyCustomCollide:
        def __call__(self, f: torch.Tensor, tau: float) -> torch.Tensor:
            ...  # your collision logic
            return f_out

    # MyCustomCollide satisfies CollisionOperator; no subclassing needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class CollisionOperator(Protocol):
    """Structural protocol for D2Q9 / D3Q19 collision operators.

    Any callable that accepts a distribution tensor *f* and a relaxation
    time *tau* and returns an updated tensor of the same shape satisfies
    this protocol.
    """

    def __call__(self, f: torch.Tensor, tau: float) -> torch.Tensor:
        """Apply the collision step.

        Args:
            f: Pre-collision distribution tensor.
            tau: BGK relaxation time (τ > 0.5).

        Returns:
            Post-collision distribution tensor with the same shape as *f*.
        """
        ...


@runtime_checkable
class BoundaryCondition(Protocol):
    """Structural protocol for boundary-condition functions.

    A boundary condition is a callable that modifies an in-flight
    distribution tensor *f* in-place or returns a new tensor.
    """

    def __call__(self, f: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Apply the boundary condition.

        Args:
            f: Distribution tensor (any shape).
            **kwargs: Boundary-specific keyword arguments (e.g. ``u_in``,
                ``wall_mask``, ``obstacle_mask``).

        Returns:
            Updated distribution tensor.
        """
        ...


__all__ = ["CollisionOperator", "BoundaryCondition"]
