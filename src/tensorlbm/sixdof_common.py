"""Common 6-DOF rigid-body module — solver-agnostic rigid-body step.

This module extracts the validated Symplectic-Euler rigid-body integrator
(:func:`tensorlbm.sixdof.step_sixdof`) behind a uniform public interface that
is **not bound to any specific solver**.  It accepts a state vector and a
force/moment vector and returns the advanced state, so it can be composed
with IBM, FSI, or any collision/turbulence loop.

Public contract
----------------
``rigid_body_step(state, force, dt, *, body) -> RigidBodyState``

    * ``state``  – :class:`RigidBodyState` (pos, vel, quat, omega_body).
    * ``force``  – force/moment vector.  Accepted shapes:
        - ``(6,)``        ``[fx, fy, fz, mx, my, mz]``,
        - :class:`FluidForcesMoments` (passed through),
        - ``(3,)``        translational force only (moments = 0).
    * ``dt``     – time step [s].
    * ``body``   – :class:`SixDOFBody` physical properties.
    Returns the advanced :class:`RigidBodyState`.

This module does **not** modify the solver hot path.  It wraps the existing
``sixdof.step_sixdof`` kernel and adds a uniform state container.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .sixdof import (
    FluidForcesMoments,
    SixDOFBody,
    _update_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    step_sixdof,
)

__all__ = [
    "SixDOFIntegratorName",
    "SixDOFCapabilityWithheldError",
    "RigidBodyState",
    "rigid_body_step",
    "rigid_body_state_to_euler",
]

SixDOFIntegratorName = Literal["symplectic_euler", "cummins"]


class SixDOFCapabilityWithheldError(NotImplementedError):
    """Raised when a 6-DOF capability request lacks a validated integrator."""


# --------------------------------------------------------------------------- #
# Uniform state container
# --------------------------------------------------------------------------- #


@dataclass
class RigidBodyState:
    """Uniform rigid-body state container (solver-agnostic).

    Attributes:
        pos:         Centre-of-mass position in the world frame, shape ``(3,)`` [m].
        vel:         Linear velocity in the world frame, shape ``(3,)`` [m/s].
        quat:        Unit quaternion ``(w, x, y, z)``, shape ``(4,)``.
        omega_body:  Angular velocity in the body frame, shape ``(3,)`` [rad/s].
    """

    pos: torch.Tensor
    vel: torch.Tensor
    quat: torch.Tensor
    omega_body: torch.Tensor

    def __post_init__(self) -> None:
        for name, val, expected in (
            ("pos", self.pos, (3,)),
            ("vel", self.vel, (3,)),
            ("quat", self.quat, (4,)),
            ("omega_body", self.omega_body, (3,)),
        ):
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(val).__name__}.")
            if val.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(val.shape)}.")

    def clone(self) -> "RigidBodyState":
        return RigidBodyState(
            self.pos.clone(),
            self.vel.clone(),
            self.quat.clone(),
            self.omega_body.clone(),
        )

    @classmethod
    def zero(cls, dtype: torch.dtype = torch.float64) -> "RigidBodyState":
        return cls(
            torch.zeros(3, dtype=dtype),
            torch.zeros(3, dtype=dtype),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype),
            torch.zeros(3, dtype=dtype),
        )


# --------------------------------------------------------------------------- #
# Force coercion
# --------------------------------------------------------------------------- #


def _coerce_force(force: object) -> FluidForcesMoments:
    """Coerce a force input into :class:`FluidForcesMoments`."""
    if isinstance(force, FluidForcesMoments):
        return force
    if isinstance(force, torch.Tensor):
        t = force.detach().to(torch.float64)
        if t.shape == (6,):
            return FluidForcesMoments(
                fx=float(t[0]), fy=float(t[1]), fz=float(t[2]),
                mx=float(t[3]), my=float(t[4]), mz=float(t[5]),
            )
        if t.shape == (3,):
            return FluidForcesMoments(
                fx=float(t[0]), fy=float(t[1]), fz=float(t[2]),
                mx=0.0, my=0.0, mz=0.0,
            )
        raise ValueError(
            f"force tensor must have shape (6,) or (3,); got {tuple(t.shape)}."
        )
    raise TypeError(
        f"force must be FluidForcesMoments or a torch.Tensor; got {type(force).__name__}."
    )


# --------------------------------------------------------------------------- #
# Public step interface
# --------------------------------------------------------------------------- #


def rigid_body_step(
    state: RigidBodyState,
    force: torch.Tensor | FluidForcesMoments,
    dt: float,
    *,
    body: SixDOFBody,
    integrator: SixDOFIntegratorName = "symplectic_euler",
) -> RigidBodyState:
    """Advance a rigid body by one time step.

    Args:
        state:     Current :class:`RigidBodyState`.
        force:     Force/moment vector ``(6,)`` or ``(3,)``, or a
                   :class:`FluidForcesMoments`.
        dt:        Time step size [s].
        body:      :class:`SixDOFBody` physical properties.
        integrator: Integration scheme.  ``"symplectic_euler"`` (default) uses
                   the validated :func:`step_sixdof` kernel.  ``"cummins"`` is
                   withheld for the common interface (requires radiation data).

    Returns:
        Advanced :class:`RigidBodyState`.
    """
    if integrator != "symplectic_euler":
        raise SixDOFCapabilityWithheldError(
            f"WITHHELD_INTEGRATOR: {integrator!r} is not available through the "
            f"common rigid_body_step interface; use 'symplectic_euler'."
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive; got {dt}.")
    fluid = _coerce_force(force)
    pos, vel, quat, omega = step_sixdof(
        state.pos, state.vel, state.quat, state.omega_body, fluid, body, dt
    )
    return RigidBodyState(pos=pos, vel=vel, quat=quat, omega_body=omega)


def rigid_body_state_to_euler(
    state: RigidBodyState,
) -> tuple[float, float, float]:
    """Return ``(roll, pitch, yaw)`` in radians for a :class:`RigidBodyState`."""
    R = quaternion_to_rotation_matrix(state.quat)
    return rotation_matrix_to_euler(R)
