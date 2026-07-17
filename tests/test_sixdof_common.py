"""Contract tests for the common 6-DOF rigid-body module.

These tests verify operator algebra (shape, zero-force identity, constant-force
momentum, DOF constraints, quaternion normalisation), NOT rigid-body physics
correctness or fluid-structure coupling validation.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.sixdof import SixDOFBody
from tensorlbm.sixdof_common import (
    RigidBodyState,
    SixDOFCapabilityWithheldError,
    rigid_body_state_to_euler,
    rigid_body_step,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_body(
    mass: float = 1.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SixDOFBody:
    return SixDOFBody(
        mass=mass,
        ixx=1.0, iyy=1.0, izz=1.0,
        gravity=gravity,
    )


# --------------------------------------------------------------------------- #
# Shape tests
# --------------------------------------------------------------------------- #


class TestRigidBodyStepShape:
    def test_output_state_shapes(self) -> None:
        state = RigidBodyState.zero()
        body = _make_body()
        force = torch.zeros(6, dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.01, body=body)
        assert new_state.pos.shape == (3,)
        assert new_state.vel.shape == (3,)
        assert new_state.quat.shape == (4,)
        assert new_state.omega_body.shape == (3,)

    def test_state_clone_preserves_values(self) -> None:
        state = RigidBodyState(
            pos=torch.tensor([1.0, 2.0, 3.0]),
            vel=torch.tensor([0.1, 0.2, 0.3]),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            omega_body=torch.tensor([0.0, 0.0, 0.0]),
        )
        clone = state.clone()
        assert torch.allclose(clone.pos, state.pos)
        assert torch.allclose(clone.vel, state.vel)
        assert torch.allclose(clone.quat, state.quat)
        assert torch.allclose(clone.omega_body, state.omega_body)

    def test_zero_state_has_correct_defaults(self) -> None:
        state = RigidBodyState.zero()
        assert torch.allclose(state.pos, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(state.vel, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(state.quat, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(state.omega_body, torch.zeros(3, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Zero-force identity: zero force + zero gravity → state unchanged (velocity)
# --------------------------------------------------------------------------- #


class TestRigidBodyZeroForceIdentity:
    def test_zero_force_zero_gravity_preserves_velocity(self) -> None:
        """With no force and no gravity, velocity should be unchanged; position
        advances by vel × dt."""
        state = RigidBodyState(
            pos=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            vel=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))
        force = torch.zeros(6, dtype=torch.float64)
        dt = 0.01
        new_state = rigid_body_step(state, force, dt=dt, body=body)
        # Velocity unchanged.
        assert torch.allclose(new_state.vel, state.vel, atol=1e-10)
        # Position = pos + vel × dt.
        assert torch.allclose(new_state.pos, state.pos + state.vel * dt, atol=1e-10)


# --------------------------------------------------------------------------- #
# Constant-force momentum: F = ma → Δv = F/m × dt
# --------------------------------------------------------------------------- #


class TestRigidBodyConstantForce:
    def test_constant_force_produces_known_acceleration(self) -> None:
        """A constant force F on a mass m produces acceleration a = F/m, so
        Δv = a × dt and Δpos = (v + Δv) × dt."""
        mass = 2.0
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=mass, gravity=(0.0, 0.0, 0.0))
        force = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        dt = 0.1
        new_state = rigid_body_step(state, force, dt=dt, body=body)
        # a = F/m = 10/2 = 5; Δv = 5 × 0.1 = 0.5
        assert new_state.vel[0].item() == pytest.approx(0.5, abs=1e-10)
        # pos = (v_new) × dt = 0.5 × 0.1 = 0.05
        assert new_state.pos[0].item() == pytest.approx(0.05, abs=1e-10)

    def test_gravity_produces_known_free_fall(self) -> None:
        """Gravity g on mass m produces force F = mg, acceleration a = g."""
        mass = 1.0
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=mass, gravity=(0.0, -9.81, 0.0))
        force = torch.zeros(6, dtype=torch.float64)  # no fluid force
        dt = 0.1
        new_state = rigid_body_step(state, force, dt=dt, body=body)
        # v_y = -9.81 × 0.1 = -0.981
        assert new_state.vel[1].item() == pytest.approx(-0.981, abs=1e-10)


# --------------------------------------------------------------------------- #
# DOF constraints
# --------------------------------------------------------------------------- #


class TestRigidBodyDOFConstraints:
    def test_fix_surge_freezes_x_velocity(self) -> None:
        """When fix_surge is True, the x-velocity should not change even with
        an x-force."""
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = SixDOFBody(mass=1.0, fix_surge=True, gravity=(0.0, 0.0, 0.0))
        force = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        # x-velocity should be frozen at 0.5 (constraint zeroes the force).
        assert new_state.vel[0].item() == pytest.approx(0.5, abs=1e-10)

    def test_fix_heave_freezes_z(self) -> None:
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = SixDOFBody(mass=1.0, fix_heave=True, gravity=(0.0, 0.0, -9.81))
        force = torch.zeros(6, dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        # z-velocity should remain 0 despite gravity.
        assert new_state.vel[2].item() == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# Quaternion normalisation
# --------------------------------------------------------------------------- #


class TestRigidBodyQuaternion:
    def test_quaternion_stays_unit(self) -> None:
        """After a step, the quaternion should remain unit-norm."""
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        )
        body = _make_body()
        force = torch.zeros(6, dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.01, body=body)
        assert new_state.quat.norm().item() == pytest.approx(1.0, abs=1e-10)

    def test_zero_omega_preserves_identity_quaternion(self) -> None:
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body()
        force = torch.zeros(6, dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.01, body=body)
        assert torch.allclose(new_state.quat, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-10)


# --------------------------------------------------------------------------- #
# Force coercion: (3,) and (6,) and FluidForcesMoments
# --------------------------------------------------------------------------- #


class TestRigidBodyForceCoercion:
    def test_three_vector_force(self) -> None:
        """A (3,) force should be treated as translational only (moments=0)."""
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        force = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        assert new_state.vel[0].item() == pytest.approx(0.1, abs=1e-10)

    def test_six_vector_force(self) -> None:
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        force = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        assert new_state.vel[0].item() == pytest.approx(0.1, abs=1e-10)


# --------------------------------------------------------------------------- #
# Fail-closed: unknown integrator
# --------------------------------------------------------------------------- #


class TestRigidBodyFailClosed:
    def test_unknown_integrator_raises(self) -> None:
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        force = torch.zeros(6, dtype=torch.float64)
        with pytest.raises(SixDOFCapabilityWithheldError, match="WITHHELD_INTEGRATOR"):
            rigid_body_step(state, force, dt=0.01, body=body, integrator="rk4")

    def test_negative_dt_raises(self) -> None:
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        force = torch.zeros(6, dtype=torch.float64)
        with pytest.raises(ValueError, match="dt must be positive"):
            rigid_body_step(state, force, dt=-0.01, body=body)


# --------------------------------------------------------------------------- #
# Euler angle extraction
# --------------------------------------------------------------------------- #


class TestRigidBodyEulerExtraction:
    def test_identity_quaternion_gives_zero_euler(self) -> None:
        state = RigidBodyState.zero(dtype=torch.float64)
        roll, pitch, yaw = rigid_body_state_to_euler(state)
        assert roll == pytest.approx(0.0, abs=1e-10)
        assert pitch == pytest.approx(0.0, abs=1e-10)
        assert yaw == pytest.approx(0.0, abs=1e-10)
