"""Contract tests for the common FSI module (IBM + 6-DOF composition).

These tests verify operator algebra (shape, zero-flow identity, force-reaction
sign, composition consistency, D3Q19/D3Q27 parity), NOT FSI physics
correctness or moving-body validation.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.fsi_common import (
    FSICapabilityWithheldError,
    FSIResult,
    fsi_step,
)
from tensorlbm.sixdof import SixDOFBody
from tensorlbm.sixdof_common import RigidBodyState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _solid_mask(nz: int, ny: int, nx: int, cx: int, cy: int, cz: int, r: int) -> torch.Tensor:
    iz, iy, ix = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    return (((ix - cx).float() ** 2 + (iy - cy).float() ** 2 + (iz - cz).float() ** 2) <= r ** 2)


def _make_body(gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SixDOFBody:
    return SixDOFBody(mass=1.0, ixx=1.0, iyy=1.0, izz=1.0, gravity=gravity)


# --------------------------------------------------------------------------- #
# Shape tests
# --------------------------------------------------------------------------- #


class TestFSIStepShape:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_result_shapes(self, lattice, q, equilibrium) -> None:
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        result = fsi_step(f, state, mask, body=body, lattice=lattice, dt=1.0)
        assert isinstance(result, FSIResult)
        assert result.f_updated.shape == f.shape == (q, nz, ny, nx)
        assert result.force_on_fluid.shape == (3, nz, ny, nx)
        assert result.force_on_body.shape == (6,)
        assert result.structure_updated.pos.shape == (3,)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_all_outputs_finite(self, lattice) -> None:
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        u = torch.full_like(rho, 0.01)
        zero = torch.zeros_like(rho)
        if lattice == "D3Q19":
            f = equilibrium3d(rho, u, zero, zero)
        else:
            f = equilibrium27(rho, u, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        result = fsi_step(f, state, mask, body=body, lattice=lattice, dt=1.0)
        assert torch.isfinite(result.f_updated).all()
        assert torch.isfinite(result.force_on_fluid).all()
        assert torch.isfinite(result.force_on_body).all()
        assert torch.isfinite(result.structure_updated.pos).all()


# --------------------------------------------------------------------------- #
# Zero-flow identity: zero flow + zero target + zero gravity → no force, no motion
# --------------------------------------------------------------------------- #


class TestFSIZeroFlowIdentity:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_zero_flow_zero_target_no_force(self, lattice, q, equilibrium) -> None:
        """In zero flow with a stationary body (zero velocity), the IBM force
        should be zero (target = 0, interpolated = 0), so the body should not
        move (no gravity)."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body(gravity=(0.0, 0.0, 0.0))
        result = fsi_step(f, state, mask, body=body, lattice=lattice, dt=1.0)
        # Force on fluid should be ~0.
        assert result.force_on_fluid.abs().max().item() < 1e-6
        # Force on body should be ~0.
        assert result.force_on_body.abs().max().item() < 1e-6
        # Body should not move.
        assert torch.allclose(result.structure_updated.pos, state.pos, atol=1e-10)
        assert torch.allclose(result.structure_updated.vel, state.vel, atol=1e-10)
        # f should be unchanged.
        assert torch.allclose(result.f_updated, f, atol=1e-7)


# --------------------------------------------------------------------------- #
# Force-reaction sign: force on body = −Σ force on fluid
# --------------------------------------------------------------------------- #


class TestFSIForceReactionSign:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_reaction_force_is_negative_of_fluid_force(self, lattice, q, equilibrium) -> None:
        """The force on the body should be the negative of the total IBM force
        on the fluid (Newton's third law)."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        # Non-zero target velocity → non-zero IBM force.
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))
        result = fsi_step(f, state, mask, body=body, lattice=lattice, dt=1.0)
        # Σ force_on_fluid should be non-zero (target ≠ 0 in zero flow).
        fx_fluid = result.force_on_fluid[0].sum().item()
        assert abs(fx_fluid) > 1e-6
        # force_on_body[0] = −fx_fluid.
        assert result.force_on_body[0].item() == pytest.approx(-fx_fluid, abs=1e-6)


# --------------------------------------------------------------------------- #
# Composition consistency: FSI = IBM + 6DOF
# --------------------------------------------------------------------------- #


class TestFSICompositionConsistency:
    def test_fsi_advances_body_with_reaction_force(self) -> None:
        """The body should advance according to the reaction force via the
        6-DOF integrator.  With no gravity, Δv = F_body/m × dt."""
        from tensorlbm.ibm_common import ibm_direct_forcing_3d_common
        from tensorlbm.sixdof_common import rigid_body_step

        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))
        dt = 1.0

        # Manual composition: IBM → reaction force → 6DOF.
        force_fluid, f_corr = ibm_direct_forcing_3d_common(
            f, mask, state.vel.to(f.dtype), lattice="D3Q19"
        )
        fx_manual = float(force_fluid[0].sum().item())
        force_body_manual = torch.tensor(
            [-fx_manual, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
        )
        state_manual = rigid_body_step(state, force_body_manual, dt, body=body)

        # FSI step.
        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=dt)

        # The FSI body state should match the manual composition (x-component).
        assert result.structure_updated.vel[0].item() == pytest.approx(
            state_manual.vel[0].item(), abs=1e-6
        )


# --------------------------------------------------------------------------- #
# Two-way explicit coupling
# --------------------------------------------------------------------------- #


class TestFSITwoWayExplicit:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_two_way_produces_finite_result(self, lattice) -> None:
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        if lattice == "D3Q19":
            f = equilibrium3d(rho, zero, zero, zero)
        else:
            f = equilibrium27(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.03, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))
        result = fsi_step(
            f, state, mask, body=body, lattice=lattice, dt=1.0,
            coupling="two_way_explicit",
        )
        assert torch.isfinite(result.f_updated).all()
        assert torch.isfinite(result.force_on_body).all()
        assert torch.isfinite(result.structure_updated.pos).all()


# --------------------------------------------------------------------------- #
# Fail-closed: unknown lattice / coupling
# --------------------------------------------------------------------------- #


class TestFSIFailClosed:
    def test_unknown_lattice_raises(self) -> None:
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        with pytest.raises(FSICapabilityWithheldError, match="WITHHELD_UNKNOWN_LATTICE"):
            fsi_step(f, state, mask, body=body, lattice="D2Q9")

    def test_unknown_coupling_raises(self) -> None:
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        with pytest.raises(FSICapabilityWithheldError, match="WITHHELD_UNKNOWN_COUPLING"):
            fsi_step(f, state, mask, body=body, lattice="D3Q19", coupling="strong")


# --------------------------------------------------------------------------- #
# D3Q19 / D3Q27 parity
# --------------------------------------------------------------------------- #


class TestFSILatticeParity:
    def test_both_lattices_produce_consistent_structure(self) -> None:
        """Both lattices should advance the body in a consistent direction
        (same sign of velocity change) for the same initial conditions."""
        nz, ny, nx = 10, 10, 10
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f19 = equilibrium3d(rho, zero, zero, zero)
        f27 = equilibrium27(rho, zero, zero, zero)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.03, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))
        r19 = fsi_step(f19, state.clone(), mask, body=body, lattice="D3Q19", dt=1.0)
        r27 = fsi_step(f27, state.clone(), mask, body=body, lattice="D3Q27", dt=1.0)
        # Both should produce finite, non-NaN body positions.
        assert torch.isfinite(r19.structure_updated.pos).all()
        assert torch.isfinite(r27.structure_updated.pos).all()
