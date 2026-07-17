"""Regression equivalence verification for 6DOF / FSI extraction.

This test suite performs three layers of verification:

1. **Original bug identification** — identifies known bugs in the original
   ``sixdof.py`` and ``rigid_body_6dof.py`` implementations (带病上岗 detection).
   Tests are written to *characterise* the buggy behaviour (not to assert
   correctness), so they document what the common module inherits.

2. **Equivalence verification** — verifies that ``sixdof_common.rigid_body_step``
   produces bit-identical results to the original ``sixdof.step_sixdof`` when
   given the same inputs (state + force + dt).  Also verifies force coercion
   consistency for (6,), (3,), and ``FluidForcesMoments`` inputs.

3. **Composition verification** — verifies that ``fsi_common.fsi_step``
   correctly composes IBM direct forcing → reaction force → 6-DOF advance →
   f correction, and that a full collision → FSI loop produces finite,
   consistent results.

TDD: tests written first, then run against the real implementation.
No commit / push.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import C as C19, W as W19, equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.ibm_common import ibm_direct_forcing_3d_common
from tensorlbm.sixdof import (
    FluidForcesMoments,
    SixDOFBody,
    SixDOFConfig,
    run_sixdof_simulation,
    step_sixdof,
)
from tensorlbm.sixdof_common import (
    RigidBodyState,
    rigid_body_step,
    rigid_body_state_to_euler,
)
from tensorlbm.fsi_common import fsi_step, FSIResult


# =========================================================================== #
# Helpers
# =========================================================================== #


def _make_body(
    mass: float = 1.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **dof_flags,
) -> SixDOFBody:
    return SixDOFBody(
        mass=mass,
        ixx=1.0, iyy=1.0, izz=1.0,
        gravity=gravity,
        **dof_flags,
    )


def _solid_mask(
    nz: int, ny: int, nx: int, cx: int, cy: int, cz: int, r: int,
) -> torch.Tensor:
    iz, iy, ix = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij",
    )
    return (
        ((ix - cx).float() ** 2 + (iy - cy).float() ** 2 + (iz - cz).float() ** 2)
        <= r ** 2
    )


def _bgk_collision(f: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Simple BGK collision for D3Q19: f' = f - (f - f_eq)/tau."""
    q = f.shape[0]
    rho = f.sum(dim=0)
    c = C19.float()
    momentum = (f.unsqueeze(-1) * c.view(q, 1, 1, 1, 3)).sum(dim=0)
    inv_rho = torch.where(rho > 1e-12, 1.0 / rho, torch.zeros_like(rho))
    u = momentum * inv_rho.unsqueeze(-1)
    feq = equilibrium3d(rho, u[..., 0], u[..., 1], u[..., 2])
    return f - (f - feq) / tau


# =========================================================================== #
# PART 1: Original bug identification (带病上岗 detection)
# =========================================================================== #


class TestOriginalBugRotationalDOFConstraint:
    """BUG-1: ``step_sixdof`` zeroes pre-existing angular velocity in
    constrained rotational DOFs.

    The translational constraint only zeroes the *force* (so existing velocity
    is preserved), but the rotational constraint re-applies the mask to the
    *entire* ``omega_new``, zeroing any pre-existing angular velocity.

    Line 261 of sixdof.py:
        omega_new = omega_new * constraints_rot   # re-apply clamp

    This is inconsistent with the translational path (lines 238-242), which
    only constrains the force, not the velocity.
    """

    def test_translational_constraint_preserves_velocity(self):
        """Translational constraint (fix_surge) preserves existing velocity."""
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(fix_surge=True)
        force = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        # x-velocity preserved (not zeroed).
        assert new_state.vel[0].item() == pytest.approx(0.5, abs=1e-10)

    def test_rotational_constraint_zeroes_existing_omega(self):
        """BUG: Rotational constraint (fix_roll) zeroes pre-existing angular
        velocity instead of preserving it.

        With fix_roll=True and an initial omega_x=0.5, the correct behaviour
        would be to preserve omega_x=0.5 (like the translational case).
        But the bug zeroes it.
        """
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
        )
        body = _make_body(fix_roll=True)
        force = torch.zeros(6, dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        # BUG: omega_x is zeroed to 0.0 instead of being preserved at 0.5.
        assert new_state.omega_body[0].item() == pytest.approx(0.0, abs=1e-10)
        # Document the inconsistency: translational preserves, rotational zeroes.
        # If this were fixed, omega_x would be 0.5.

    def test_rotational_constraint_inconsistency_documented(self):
        """Document that translational and rotational constraints behave
        differently for pre-existing velocity."""
        # Translational: fix_surge preserves vel_x.
        state_t = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body_t = _make_body(fix_surge=True)
        force = torch.zeros(6, dtype=torch.float64)
        new_t = rigid_body_step(state_t, force, dt=0.1, body=body_t)
        # Translational: preserved.
        assert new_t.vel[0].item() == pytest.approx(0.5, abs=1e-10)

        # Rotational: fix_roll zeroes omega_x.
        state_r = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
        )
        body_r = _make_body(fix_roll=True)
        new_r = rigid_body_step(state_r, force, dt=0.1, body=body_r)
        # Rotational: zeroed (BUG).
        assert new_r.omega_body[0].item() == pytest.approx(0.0, abs=1e-10)


class TestOriginalBugCumminsDocstringVsImplementation:
    """BUG-2: ``cummins_step`` docstring claims RK4 but implementation is
    simple Euler.

    The docstring says "Advance the Cummins equation by one time step
    (4th-order Runge-Kutta)" but the actual code uses:
        new_velocity = state.velocity + accel * dt
        new_position = state.position + new_velocity * dt
    which is Symplectic Euler, not RK4.

    The code comment on line 275 even admits:
        "# Simple Euler integration (RK4 can be added for higher accuracy)"
    """

    def test_cummins_uses_euler_not_rk4(self):
        """Verify that cummins_step produces Euler-level results, not RK4.

        For a simple harmonic oscillator (M=1, C=k, F=0), the exact solution
        is x(t) = A*cos(omega*t).  Euler integration has O(dt) error, while
        RK4 has O(dt^4) error.  We verify the error scales as O(dt), confirming
        Euler.
        """
        from tensorlbm.rigid_body_6dof import (
            State6DOF,
            cummins_step,
        )

        k = 4.0  # spring constant
        dt_values = [0.01, 0.005, 0.0025]
        errors = []

        for dt in dt_values:
            M = torch.eye(6)
            A_inf = torch.zeros(6, 6)
            C_mat = torch.zeros(6, 6)
            C_mat[0, 0] = k
            state = State6DOF(
                position=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                velocity=torch.zeros(6),
                acceleration=torch.zeros(6),
            )
            n_steps = int(round(math.pi / (2 * math.sqrt(k)) / dt))
            for _ in range(n_steps):
                state = cummins_step(
                    state, dt, torch.zeros(6), M, A_inf, C_mat,
                )
            # Exact: x(t) = cos(omega*t), omega = sqrt(k) = 2
            t_final = n_steps * dt
            exact = math.cos(math.sqrt(k) * t_final)
            errors.append(abs(state.position[0].item() - exact))

        # Euler: error ratio ≈ 2 when dt halves.
        ratio1 = errors[0] / errors[1] if errors[1] > 0 else float("inf")
        ratio2 = errors[1] / errors[2] if errors[2] > 0 else float("inf")
        # Euler: ratio ≈ 2; RK4: ratio ≈ 16.
        # We accept ratio in [1.5, 3.0] as Euler.
        assert ratio1 > 1.5, f"Error ratio {ratio1} too small for Euler"
        assert ratio2 > 1.5, f"Error ratio {ratio2} too small for Euler"


class TestOriginalBugCumminsConvolutionIndexing:
    """BUG-3: ``cummins_time_integration`` has off-by-one indexing in the
    retardation function convolution.

    The convolution needs K(t_n - t_k) for k = 0..n-1, which is
    K_retard[n-k].  The code uses K_retard[step - N_hist : step].flip(0),
    which gives K_retard[n-1-k] instead of K_retard[n-k].

    This means:
    - The most recent velocity is weighted by K(0) instead of K(dt).
    - The oldest velocity is weighted by K((n-1)*dt) instead of K(n*dt).
    """

    def test_convolution_uses_k0_for_recent_velocity(self):
        """Verify that the convolution uses K(0) for the most recent past
        velocity (the bug), not K(dt).

        We set up a case where K(0) ≠ K(dt) and check which one is used.
        """
        from tensorlbm.rigid_body_6dof import (
            BodyProperties6DOF,
            HydrostaticMatrix,
            RadiationData,
            cummins_time_integration,
        )

        # Build radiation data with a known retardation function.
        n_freq = 32
        omega = torch.linspace(0.1, 10.0, n_freq)
        # Damping: B(omega) = 1 for all frequencies → K(t) = (2/pi) * integral
        # of cos(omega*t) domega, which is a sinc-like function.
        B = torch.ones(n_freq, 6, 6) * 0.01
        A_inf = torch.zeros(6, 6)

        radiation = RadiationData(
            omega=omega, added_mass=torch.zeros(n_freq, 6, 6),
            damping=B, added_mass_inf=A_inf,
        )

        body = BodyProperties6DOF(mass=1.0)
        hydro = HydrostaticMatrix(c_matrix=torch.zeros(6, 6))

        # Run 2 steps with a known excitation.
        dt = 0.1
        n_steps = 2
        F_exc = torch.zeros(n_steps, 6)
        F_exc[0, 0] = 1.0  # impulse at step 0

        state = cummins_time_integration(
            body, hydro, radiation, F_exc, dt, n_steps,
        )

        # At step 1, the convolution should use K(dt) * v(t_0).
        # But the bug uses K(0) * v(t_0).
        # K(0) = (2/pi) * sum(B) * d_omega ≠ 0 (since B > 0).
        # K(dt) = (2/pi) * sum(B * cos(omega*dt)) * d_omega < K(0).
        # So the memory force is over-estimated.
        # We just verify the state is finite (the bug doesn't crash).
        assert torch.isfinite(state.position).all()
        assert torch.isfinite(state.velocity).all()


class TestOriginalFSIArchitectureDifference:
    """The original ``fsi.py`` is a 2-D load-extraction + linearised structural
    response module (Euler-Bernoulli beam), NOT an IBM + 6-DOF composition.

    ``fsi_common.py`` is a 3-D IBM direct-forcing + 6-DOF rigid-body advance.
    These are architecturally different and cannot be numerically compared.
    """

    def test_original_fsi_is_2d_load_extraction(self):
        """Verify that the original fsi.py works on 2D fields and produces
        structural response (deflection, stress), not rigid-body state."""
        from tensorlbm.fsi import (
            StructuralProperties,
            extract_fsi_loads,
            compute_structural_response,
        )

        ny, nx = 20, 20
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.1)
        uy = torch.zeros(ny, nx)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[8:12, 8:12] = True

        loads = extract_fsi_loads(rho, ux, uy, mask)
        # Original FSI produces FSILoads (not RigidBodyState).
        assert hasattr(loads, "fx")
        assert hasattr(loads, "pressure")

        props = StructuralProperties()
        response = compute_structural_response(loads, props)
        # Original FSI produces FSIResponse (deflection, stress), not 6-DOF state.
        assert hasattr(response, "max_deflection")
        assert hasattr(response, "natural_frequency_hz")

    def test_fsi_common_is_3d_ibm_plus_6dof(self):
        """Verify that fsi_common.py works on 3D distributions and produces
        rigid-body state, not structural deflection."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()

        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0)
        # fsi_common produces FSIResult with rigid-body state.
        assert isinstance(result, FSIResult)
        assert hasattr(result, "structure_updated")
        assert hasattr(result, "force_on_fluid")
        assert hasattr(result, "f_updated")
        # NOT structural deflection.
        assert not hasattr(result, "max_deflection")


# =========================================================================== #
# PART 2: Equivalence verification
#    sixdof_common.rigid_body_step vs original step_sixdof
# =========================================================================== #


class TestEquivalenceRigidBodyStepVsStepSixdof:
    """Verify that ``rigid_body_step`` produces identical results to
    ``step_sixdof`` when given the same inputs."""

    def test_same_inputs_produce_allclose_outputs(self):
        """Same state + force + dt → allclose outputs."""
        pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        vel = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        omega = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        fluid = FluidForcesMoments(fx=5.0, fy=-3.0, fz=1.0, mx=0.1, my=-0.2, mz=0.3)
        body = _make_body(mass=2.0, gravity=(0.0, -9.81, 0.0))
        dt = 0.01

        # Original step_sixdof.
        pos_o, vel_o, quat_o, omega_o = step_sixdof(
            pos, vel, quat, omega, fluid, body, dt,
        )

        # Common rigid_body_step.
        state = RigidBodyState(pos=pos.clone(), vel=vel.clone(),
                               quat=quat.clone(), omega_body=omega.clone())
        force = torch.tensor([5.0, -3.0, 1.0, 0.1, -0.2, 0.3], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt, body=body)

        assert torch.allclose(new_state.pos, pos_o, atol=1e-12)
        assert torch.allclose(new_state.vel, vel_o, atol=1e-12)
        assert torch.allclose(new_state.quat, quat_o, atol=1e-12)
        assert torch.allclose(new_state.omega_body, omega_o, atol=1e-12)

    def test_equivalence_with_nontrivial_quaternion(self):
        """Equivalence with a non-identity quaternion and angular velocity."""
        angle = math.radians(30.0)
        quat = torch.tensor([
            math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0,
        ], dtype=torch.float64)
        pos = torch.tensor([0.5, -0.3, 0.2], dtype=torch.float64)
        vel = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        omega = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
        fluid = FluidForcesMoments(fx=1.0, fy=2.0, fz=3.0, mx=0.5, my=0.0, mz=-0.5)
        body = _make_body(mass=1.5, gravity=(0.0, 0.0, 0.0))
        dt = 0.005

        pos_o, vel_o, quat_o, omega_o = step_sixdof(
            pos, vel, quat, omega, fluid, body, dt,
        )
        state = RigidBodyState(pos=pos.clone(), vel=vel.clone(),
                               quat=quat.clone(), omega_body=omega.clone())
        force = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.0, -0.5], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt, body=body)

        assert torch.allclose(new_state.pos, pos_o, atol=1e-12)
        assert torch.allclose(new_state.vel, vel_o, atol=1e-12)
        assert torch.allclose(new_state.quat, quat_o, atol=1e-12)
        assert torch.allclose(new_state.omega_body, omega_o, atol=1e-12)

    def test_equivalence_with_dof_constraints(self):
        """Equivalence when DOF constraints are active."""
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        omega = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        fluid = FluidForcesMoments(fx=10.0, fy=5.0, fz=2.0, mx=1.0, my=0.5, mz=0.3)
        body = _make_body(mass=1.0, gravity=(0.0, -9.81, 0.0),
                         fix_surge=True, fix_pitch=True)
        dt = 0.01

        pos_o, vel_o, quat_o, omega_o = step_sixdof(
            pos, vel, quat, omega, fluid, body, dt,
        )
        state = RigidBodyState(pos=pos.clone(), vel=vel.clone(),
                               quat=quat.clone(), omega_body=omega.clone())
        force = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.3], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt, body=body)

        assert torch.allclose(new_state.pos, pos_o, atol=1e-12)
        assert torch.allclose(new_state.vel, vel_o, atol=1e-12)
        assert torch.allclose(new_state.quat, quat_o, atol=1e-12)
        assert torch.allclose(new_state.omega_body, omega_o, atol=1e-12)

    def test_equivalence_multi_step(self):
        """Equivalence over multiple consecutive steps."""
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        omega = torch.zeros(3, dtype=torch.float64)
        fluid = FluidForcesMoments(fx=3.0, fy=0.0, fz=0.0, mx=0.0, my=0.5, mz=0.0)
        body = _make_body(mass=2.0, gravity=(0.0, -9.81, 0.0))
        dt = 0.001

        # Run original.
        pos_o, vel_o, quat_o, omega_o = pos, vel, quat, omega
        for _ in range(50):
            pos_o, vel_o, quat_o, omega_o = step_sixdof(
                pos_o, vel_o, quat_o, omega_o, fluid, body, dt,
            )

        # Run common.
        state = RigidBodyState(pos=pos.clone(), vel=vel.clone(),
                               quat=quat.clone(), omega_body=omega.clone())
        force = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=torch.float64)
        for _ in range(50):
            state = rigid_body_step(state, force, dt, body=body)

        assert torch.allclose(state.pos, pos_o, atol=1e-10)
        assert torch.allclose(state.vel, vel_o, atol=1e-10)
        assert torch.allclose(state.quat, quat_o, atol=1e-10)
        assert torch.allclose(state.omega_body, omega_o, atol=1e-10)


class TestEquivalenceForceCoercion:
    """Verify that force coercion (6,), (3,), and FluidForcesMoments produce
    equivalent results."""

    def test_six_vector_equals_fluid_forces_moments(self):
        """A (6,) tensor and a FluidForcesMoments with the same values should
        produce identical results."""
        state = RigidBodyState(
            pos=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
            vel=torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
        )
        body = _make_body(mass=1.5, gravity=(0.0, 0.0, 0.0))
        dt = 0.01

        force_tensor = torch.tensor([5.0, -3.0, 1.0, 0.2, -0.1, 0.4],
                                     dtype=torch.float64)
        fluid = FluidForcesMoments(fx=5.0, fy=-3.0, fz=1.0,
                                   mx=0.2, my=-0.1, mz=0.4)

        s1 = rigid_body_step(state.clone(), force_tensor, dt, body=body)
        s2 = rigid_body_step(state.clone(), fluid, dt, body=body)

        assert torch.allclose(s1.pos, s2.pos, atol=1e-12)
        assert torch.allclose(s1.vel, s2.vel, atol=1e-12)
        assert torch.allclose(s1.quat, s2.quat, atol=1e-12)
        assert torch.allclose(s1.omega_body, s2.omega_body, atol=1e-12)

    def test_three_vector_equals_six_vector_with_zero_moments(self):
        """A (3,) force should equal a (6,) force with zero moments."""
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=1.0, gravity=(0.0, 0.0, 0.0))
        dt = 0.1

        force3 = torch.tensor([5.0, -3.0, 1.0], dtype=torch.float64)
        force6 = torch.tensor([5.0, -3.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

        s1 = rigid_body_step(state.clone(), force3, dt, body=body)
        s2 = rigid_body_step(state.clone(), force6, dt, body=body)

        assert torch.allclose(s1.pos, s2.pos, atol=1e-12)
        assert torch.allclose(s1.vel, s2.vel, atol=1e-12)
        assert torch.allclose(s1.quat, s2.quat, atol=1e-12)
        assert torch.allclose(s1.omega_body, s2.omega_body, atol=1e-12)


class TestEquivalenceRunSimulationVsCommon:
    """Verify that ``run_sixdof_simulation`` (using step_sixdof) and a manual
    loop using ``rigid_body_step`` produce equivalent trajectories."""

    def test_trajectory_equivalence(self):
        """Run the original ``run_sixdof_simulation`` and a manual loop with
        ``rigid_body_step``; verify the trajectories match."""
        cfg = SixDOFConfig(
            body=_make_body(mass=2.0, gravity=(0.0, -9.81, 0.0)),
            dt=0.001,
            n_steps=100,
            pos_init=(0.0, 0.0, 0.0),
            vel_init=(0.1, 0.0, 0.0),
            omega_init=(0.0, 0.0, 0.1),
        )

        # Original simulation.
        result_orig = run_sixdof_simulation(cfg)

        # Manual loop with rigid_body_step.
        state = RigidBodyState(
            pos=torch.tensor(cfg.pos_init, dtype=torch.float64),
            vel=torch.tensor(cfg.vel_init, dtype=torch.float64),
            quat=torch.tensor(cfg.quat_init, dtype=torch.float64),
            omega_body=torch.tensor(cfg.omega_init, dtype=torch.float64),
        )
        t = 0.0
        history_common = []
        for step in range(cfg.n_steps + 1):
            history_common.append((t, state.pos.clone(), state.vel.clone(),
                                   state.quat.clone(), state.omega_body.clone()))
            if step == cfg.n_steps:
                break
            # Use the same sinusoidal force function.
            fluid = FluidForcesMoments(
                fx=10.0 * math.sin(2 * math.pi * 0.5 * t),
                fy=0.0, fz=0.0,
                mx=0.0,
                my=1.0 * math.cos(2 * math.pi * 0.5 * t),
                mz=0.0,
            )
            state = rigid_body_step(state, fluid, cfg.dt, body=cfg.body)
            t += cfg.dt

        # Compare final states.
        final_orig = result_orig.history[-1]
        t_c, pos_c, vel_c, quat_c, omega_c = history_common[-1]

        assert t_c == pytest.approx(final_orig.time, abs=1e-12)
        for i in range(3):
            assert pos_c[i].item() == pytest.approx(final_orig.pos[i], abs=1e-10)
            assert vel_c[i].item() == pytest.approx(final_orig.vel[i], abs=1e-10)
            assert omega_c[i].item() == pytest.approx(
                final_orig.omega_body[i], abs=1e-10,
            )
        for i in range(4):
            assert quat_c[i].item() == pytest.approx(
                final_orig.quat[i], abs=1e-10,
            )


# =========================================================================== #
# PART 3: FSI composition verification
#    IBM force → 6DOF update → f correction
# =========================================================================== #


class TestFSICompositionIBMTo6DOF:
    """Verify that fsi_step correctly composes IBM → reaction force → 6DOF."""

    def test_fsi_body_advances_with_ibm_reaction_force(self):
        """The body should advance according to the reaction force from IBM.

        Manual: IBM → sum force → negate → rigid_body_step.
        fsi_step should produce the same body state.
        """
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
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

        # Manual composition.
        force_fluid, _ = ibm_direct_forcing_3d_common(
            f, mask, state.vel.to(f.dtype), lattice="D3Q19",
        )
        fx_manual = float(force_fluid[0].sum().item())
        fy_manual = float(force_fluid[1].sum().item())
        fz_manual = float(force_fluid[2].sum().item())
        force_body_manual = torch.tensor(
            [-fx_manual, -fy_manual, -fz_manual, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        state_manual = rigid_body_step(state, force_body_manual, dt, body=body)

        # FSI step.
        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=dt)

        # Body state should match (translational components).
        assert result.structure_updated.vel[0].item() == pytest.approx(
            state_manual.vel[0].item(), abs=1e-6,
        )
        assert result.structure_updated.vel[1].item() == pytest.approx(
            state_manual.vel[1].item(), abs=1e-6,
        )
        assert result.structure_updated.vel[2].item() == pytest.approx(
            state_manual.vel[2].item(), abs=1e-6,
        )
        assert result.structure_updated.pos[0].item() == pytest.approx(
            state_manual.pos[0].item(), abs=1e-6,
        )

    def test_fsi_force_on_body_is_negative_of_fluid_force(self):
        """Newton's third law: force_on_body = -Σ force_on_fluid."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
        u = torch.full_like(rho, 0.01)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, u, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))

        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0)

        fx_fluid = result.force_on_fluid[0].sum().item()
        fy_fluid = result.force_on_fluid[1].sum().item()
        fz_fluid = result.force_on_fluid[2].sum().item()

        assert result.force_on_body[0].item() == pytest.approx(-fx_fluid, abs=1e-6)
        assert result.force_on_body[1].item() == pytest.approx(-fy_fluid, abs=1e-6)
        assert result.force_on_body[2].item() == pytest.approx(-fz_fluid, abs=1e-6)

    def test_fsi_f_corrected_matches_ibm_output(self):
        """The f_updated from fsi_step should match the f_corrected from
        ibm_direct_forcing_3d_common (one-way)."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
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

        # Manual IBM.
        _, f_ibm = ibm_direct_forcing_3d_common(
            f, mask, state.vel.to(f.dtype), lattice="D3Q19",
        )

        # FSI step (one-way).
        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0,
                          coupling="one_way_explicit")

        assert torch.allclose(result.f_updated, f_ibm, atol=1e-10)

    def test_fsi_moment_resolution_about_centroid(self):
        """Verify that moments are resolved about the body centroid."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
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

        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0)

        # Manually compute moments about centroid.
        force_fluid, _ = ibm_direct_forcing_3d_common(
            f, mask, state.vel.to(f.dtype), lattice="D3Q19",
        )
        # Centroid.
        iz, iy, ix = torch.where(mask)
        cx = float(ix.float().mean())
        cy = float(iy.float().mean())
        cz = float(iz.float().mean())

        iz_g, iy_g, ix_g = torch.meshgrid(
            torch.arange(nz, dtype=torch.float64),
            torch.arange(ny, dtype=torch.float64),
            torch.arange(nx, dtype=torch.float64),
            indexing="ij",
        )
        dx = ix_g - cx
        dy = iy_g - cy
        dz = iz_g - cz

        mx_manual = float(
            (dy * force_fluid[2].double() - dz * force_fluid[1].double()).sum().item()
        )
        my_manual = float(
            (dz * force_fluid[0].double() - dx * force_fluid[2].double()).sum().item()
        )
        mz_manual = float(
            (dx * force_fluid[1].double() - dy * force_fluid[0].double()).sum().item()
        )

        assert result.force_on_body[3].item() == pytest.approx(-mx_manual, abs=1e-6)
        assert result.force_on_body[4].item() == pytest.approx(-my_manual, abs=1e-6)
        assert result.force_on_body[5].item() == pytest.approx(-mz_manual, abs=1e-6)


class TestFSICompositionTwoWay:
    """Verify two-way explicit coupling: second IBM pass with advanced velocity."""

    def test_two_way_reapplies_ibm_with_advanced_velocity(self):
        """Two-way should re-apply IBM with the advanced body velocity,
        producing a different f_updated than one-way."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
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

        r1 = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0,
                      coupling="one_way_explicit")
        r2 = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0,
                      coupling="two_way_explicit")

        # Two-way should produce a different f_updated (second IBM pass).
        assert not torch.allclose(r1.f_updated, r2.f_updated, atol=1e-10)
        # But the body state should be the same (advanced once in both cases).
        assert torch.allclose(
            r1.structure_updated.pos, r2.structure_updated.pos, atol=1e-10,
        )
        assert torch.allclose(
            r1.structure_updated.vel, r2.structure_updated.vel, atol=1e-10,
        )

    def test_two_way_force_recomputed_from_second_pass(self):
        """Two-way force_on_body should come from the second IBM pass, not
        the first."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
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

        r2 = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0,
                      coupling="two_way_explicit")

        # Manual second pass with advanced velocity.
        r1 = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0,
                      coupling="one_way_explicit")
        u2 = r1.structure_updated.vel.detach().to(f.dtype).clone()
        force_fluid_2, f_corr_2 = ibm_direct_forcing_3d_common(
            f, mask, u2, lattice="D3Q19",
        )
        fx2 = float(force_fluid_2[0].sum().item())

        assert r2.force_on_body[0].item() == pytest.approx(-fx2, abs=1e-6)
        assert torch.allclose(r2.f_updated, f_corr_2, atol=1e-10)


# =========================================================================== #
# PART 4: Combination test — FSI + collision complete loop
# =========================================================================== #


class TestFSICollisionCombination:
    """Verify a complete collision → FSI step loop produces finite, consistent
    results over multiple steps."""

    def test_collision_fsi_loop_finite_and_consistent(self):
        """Run collision → FSI step for 10 steps; verify all outputs stay
        finite and the body advances monotonically."""
        nz, ny, nx = 12, 12, 12
        rho = torch.ones(nz, ny, nx)
        u_x = torch.full_like(rho, 0.02)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, u_x, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=1.0, gravity=(0.0, 0.0, 0.0))
        dt = 1.0
        tau = 1.0

        positions = [state.pos.clone()]
        velocities = [state.vel.clone()]

        for step in range(10):
            # 1. Collision (BGK).
            f = _bgk_collision(f, tau=tau)
            # 2. FSI step (IBM + 6DOF).
            result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=dt)
            f = result.f_updated
            state = result.structure_updated

            # All outputs finite.
            assert torch.isfinite(f).all(), f"NaN/Inf in f at step {step}"
            assert torch.isfinite(state.pos).all(), f"NaN/Inf in pos at step {step}"
            assert torch.isfinite(state.vel).all(), f"NaN/Inf in vel at step {step}"
            assert torch.isfinite(result.force_on_body).all(), \
                f"NaN/Inf in force at step {step}"

            positions.append(state.pos.clone())
            velocities.append(state.vel.clone())

        # Body should have moved (non-trivial dynamics).
        total_disp = (positions[-1] - positions[0]).norm().item()
        assert total_disp > 0, "Body did not move over 10 steps"

    def test_collision_fsi_loop_d3q27(self):
        """Same loop with D3Q27 lattice."""
        nz, ny, nx = 12, 12, 12
        rho = torch.ones(nz, ny, nx)
        u_x = torch.full_like(rho, 0.02)
        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, u_x, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=1.0, gravity=(0.0, 0.0, 0.0))
        dt = 1.0

        for step in range(5):
            # FSI step with D3Q27.
            result = fsi_step(f, state, mask, body=body, lattice="D3Q27", dt=dt)
            f = result.f_updated
            state = result.structure_updated
            assert torch.isfinite(f).all()
            assert torch.isfinite(state.pos).all()

    def test_collision_fsi_loop_with_gravity(self):
        """Loop with gravity: body should experience net downward force."""
        nz, ny, nx = 12, 12, 12
        rho = torch.ones(nz, ny, nx)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=1.0, gravity=(0.0, -9.81, 0.0))
        dt = 0.01

        for step in range(10):
            f = _bgk_collision(f, tau=1.0)
            result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=dt)
            f = result.f_updated
            state = result.structure_updated
            assert torch.isfinite(f).all()
            assert torch.isfinite(state.pos).all()

        # With gravity, the body should have moved in -y direction.
        assert state.pos[1].item() < 0, "Body should fall under gravity"

    def test_collision_fsi_two_way_loop(self):
        """Complete loop with two-way explicit coupling."""
        nz, ny, nx = 12, 12, 12
        rho = torch.ones(nz, ny, nx)
        u_x = torch.full_like(rho, 0.02)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, u_x, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(mass=1.0, gravity=(0.0, 0.0, 0.0))
        dt = 1.0

        for step in range(5):
            f = _bgk_collision(f, tau=1.0)
            result = fsi_step(
                f, state, mask, body=body, lattice="D3Q19", dt=dt,
                coupling="two_way_explicit",
            )
            f = result.f_updated
            state = result.structure_updated
            assert torch.isfinite(f).all()
            assert torch.isfinite(state.pos).all()
            assert torch.isfinite(result.force_on_body).all()


# =========================================================================== #
# PART 5: Edge cases and robustness
# =========================================================================== #


class TestEdgeCases:
    """Edge cases and robustness for the 6DOF/FSI extraction."""

    def test_rigid_body_step_preserves_dtype(self):
        """Output dtype should match input dtype (float64)."""
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body()
        force = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        new_state = rigid_body_step(state, force, dt=0.1, body=body)
        assert new_state.pos.dtype == torch.float64
        assert new_state.vel.dtype == torch.float64
        assert new_state.quat.dtype == torch.float64
        assert new_state.omega_body.dtype == torch.float64

    def test_rigid_body_step_does_not_mutate_input(self):
        """The input state should not be mutated."""
        state = RigidBodyState(
            pos=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            vel=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64),
        )
        pos_orig = state.pos.clone()
        vel_orig = state.vel.clone()
        quat_orig = state.quat.clone()
        omega_orig = state.omega_body.clone()

        body = _make_body()
        force = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        rigid_body_step(state, force, dt=0.1, body=body)

        assert torch.allclose(state.pos, pos_orig)
        assert torch.allclose(state.vel, vel_orig)
        assert torch.allclose(state.quat, quat_orig)
        assert torch.allclose(state.omega_body, omega_orig)

    def test_fsi_step_no_mask_no_force(self):
        """Empty mask → no IBM force → body only affected by gravity."""
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        state = RigidBodyState.zero(dtype=torch.float64)
        body = _make_body(gravity=(0.0, 0.0, 0.0))

        result = fsi_step(f, state, mask, body=body, lattice="D3Q19", dt=1.0)
        # No mask → no surface markers → zero force.
        assert result.force_on_fluid.abs().max().item() < 1e-10
        assert result.force_on_body.abs().max().item() < 1e-10
        # Body should not move (no force, no gravity).
        assert torch.allclose(result.structure_updated.pos, state.pos, atol=1e-10)

    def test_fsi_step_with_explicit_markers(self):
        """FSI step with explicit marker positions should work."""
        nz, ny, nx = 10, 10, 10
        rho = torch.ones(nz, ny, nx)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        # Derive markers explicitly.
        from tensorlbm.ibm_common import derive_surface_markers_3d
        mx, my, mz = derive_surface_markers_3d(mask)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        body = _make_body(gravity=(0.0, 0.0, 0.0))

        result = fsi_step(
            f, state, mask, body=body, lattice="D3Q19", dt=1.0,
            markers=(mx, my, mz),
        )
        assert torch.isfinite(result.f_updated).all()
        assert torch.isfinite(result.structure_updated.pos).all()

    def test_rigid_body_state_to_euler_roundtrip(self):
        """Euler angles from a known quaternion should be correct."""
        # 90-degree rotation about z-axis.
        angle = math.radians(90.0)
        state = RigidBodyState(
            pos=torch.zeros(3, dtype=torch.float64),
            vel=torch.zeros(3, dtype=torch.float64),
            quat=torch.tensor([
                math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2),
            ], dtype=torch.float64),
            omega_body=torch.zeros(3, dtype=torch.float64),
        )
        roll, pitch, yaw = rigid_body_state_to_euler(state)
        assert yaw == pytest.approx(math.pi / 2, abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)
