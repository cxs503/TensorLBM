"""Six-degrees-of-freedom (6-DOF) rigid-body dynamics for TensorLBM.

Provides rigid-body motion prediction coupled with LBM fluid forces, enabling
simulation of freely moving bodies such as:

* Ship/vessel motion in waves (surge, sway, heave, roll, pitch, yaw)
* Automotive aerodynamic stability (lift-off, crosswind response)
* Projectile flight dynamics
* Dropped / launched objects in fluid

This corresponds to the **rigid-body 6-DOF** capability in XFlow and the
**moving geometry** (ALE) feature in PowerFlow.

Mathematical model
------------------
The rigid body is described by:
  - Position of the centre of mass  x_G ∈ ℝ³
  - Linear velocity                 v_G ∈ ℝ³
  - Quaternion orientation          q = (q_w, q_x, q_y, q_z), |q| = 1
  - Angular velocity (body frame)   ω_B ∈ ℝ³

The equations of motion (Newton–Euler):
    m ẍ_G  = F_fluid + F_gravity + F_constraint
    I_B ω̇_B = M_fluid − ω_B × (I_B ω_B)

where I_B is the inertia tensor in body coordinates.

Integration uses the **Symplectic Euler** scheme (semi-implicit) which is
momentum-conserving and robust for fluid–structure coupling:
  1. Update linear velocity:  v_G(t+dt) = v_G(t) + dt * F / m
  2. Update angular velocity: ω_B(t+dt) = ω_B(t) + dt * I_B⁻¹ (M − ω×Iω)
  3. Update position:         x_G(t+dt) = x_G(t) + dt * v_G(t+dt)
  4. Update quaternion:       q(t+dt) via quaternion product with exp(ω dt/2)

Degrees of freedom can be selectively **constrained** (frozen) to model
partially restrained bodies (e.g., a ship with heave/pitch/roll only).

References
----------
Ferziger, J. H., Perić, M., & Street, R. L. (2002). *Computational Methods
for Fluid Dynamics* (3rd ed.). Springer.
Hughes, T. J. R. (2000). *The Finite Element Method*. Dover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = [
    "SixDOFBody",
    "SixDOFConfig",
    "FluidForcesMoments",
    "SixDOFState",
    "SixDOFResult",
    "step_sixdof",
    "run_sixdof_simulation",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_euler",
]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SixDOFBody:
    """Physical properties of the rigid body."""
    mass: float = 1.0                               # kg
    # Inertia tensor diagonal (body frame) [kg·m²]
    ixx: float = 1.0
    iyy: float = 1.0
    izz: float = 1.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)  # m/s²
    # Constrained DOF flags (True = freeze that DOF)
    fix_surge: bool = False    # x translation
    fix_sway: bool = False     # y translation
    fix_heave: bool = False    # z translation
    fix_roll: bool = False     # x rotation
    fix_pitch: bool = False    # y rotation
    fix_yaw: bool = False      # z rotation

    def inertia_tensor(self) -> torch.Tensor:
        """Return the 3×3 body-frame inertia tensor."""
        return torch.tensor(
            [
                [self.ixx, -self.ixy, -self.ixz],
                [-self.ixy, self.iyy, -self.iyz],
                [-self.ixz, -self.iyz, self.izz],
            ],
            dtype=torch.float64,
        )


@dataclass
class SixDOFConfig:
    """Simulation configuration."""
    body: SixDOFBody = field(default_factory=SixDOFBody)
    dt: float = 1e-3                    # time step [s]
    n_steps: int = 100
    # Initial conditions
    pos_init: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel_init: tuple[float, float, float] = (0.0, 0.0, 0.0)
    omega_init: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s, body frame
    # Initial quaternion (w, x, y, z) – defaults to identity (no rotation)
    quat_init: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class FluidForcesMoments:
    """Integrated fluid forces and moments about the centre of mass."""
    fx: float = 0.0   # N
    fy: float = 0.0   # N
    fz: float = 0.0   # N
    mx: float = 0.0   # N·m
    my: float = 0.0   # N·m
    mz: float = 0.0   # N·m


@dataclass
class SixDOFState:
    """Instantaneous state of the rigid body."""
    time: float
    pos: list[float]          # [x, y, z] in world frame [m]
    vel: list[float]          # [vx, vy, vz] in world frame [m/s]
    quat: list[float]         # [qw, qx, qy, qz] (unit quaternion)
    omega_body: list[float]   # [ωx, ωy, ωz] in body frame [rad/s]
    euler_deg: list[float]    # [roll, pitch, yaw] in degrees


@dataclass
class SixDOFResult:
    """Time-series output of a 6-DOF simulation."""
    history: list[SixDOFState]
    max_displacement: float      # m
    max_velocity: float          # m/s
    max_roll_deg: float
    max_pitch_deg: float


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def _quat_multiply(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q[0], q[1], q[2], q[3]
    w2, x2, y2, z2 = p[0], p[1], p[2], p[3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion (w, x, y, z) to a 3×3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.stack([
        torch.stack([1-2*(y**2+z**2),   2*(x*y-z*w),   2*(x*z+y*w)]),
        torch.stack([  2*(x*y+z*w), 1-2*(x**2+z**2),   2*(y*z-x*w)]),
        torch.stack([  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x**2+y**2)]),
    ])
    return R


def rotation_matrix_to_euler(R: torch.Tensor) -> tuple[float, float, float]:
    """Extract roll (φ), pitch (θ), yaw (ψ) in radians from a 3×3 rotation matrix."""
    # Using ZYX Euler convention
    r = R.double()
    pitch = math.asin(float(-r[2, 0].clamp(-1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw  = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        yaw  = 0.0
    return roll, pitch, yaw


def _update_quaternion(
    q: torch.Tensor, omega_body: torch.Tensor, dt: float
) -> torch.Tensor:
    """Integrate quaternion using the exponential map (first-order accurate)."""
    omega_norm = omega_body.norm().item()
    if omega_norm < 1e-12:
        return q / q.norm()

    angle_half = 0.5 * omega_norm * dt
    axis = omega_body / omega_norm
    dq = torch.cat([
        torch.tensor([math.cos(angle_half)]),
        axis * math.sin(angle_half),
    ])
    q_new = _quat_multiply(q, dq)
    return q_new / q_new.norm()


# ---------------------------------------------------------------------------
# Core integration step
# ---------------------------------------------------------------------------

def step_sixdof(
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega_body: torch.Tensor,
    fluid: FluidForcesMoments,
    body: SixDOFBody,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform one Symplectic Euler integration step.

    Returns (new_pos, new_vel, new_quat, new_omega_body).
    """
    m = body.mass
    g = torch.tensor(body.gravity, dtype=torch.float64)
    I = body.inertia_tensor()
    I_inv = torch.linalg.inv(I)

    # --- Forces in world frame ---
    F_fluid = torch.tensor([fluid.fx, fluid.fy, fluid.fz], dtype=torch.float64)
    F_grav = m * g
    F_total = F_fluid + F_grav

    # Apply translational DOF constraints
    constraints_lin = torch.tensor([
        0.0 if body.fix_surge else 1.0,
        0.0 if body.fix_sway  else 1.0,
        0.0 if body.fix_heave else 1.0,
    ], dtype=torch.float64)
    F_total = F_total * constraints_lin

    # --- Update linear velocity & position (Symplectic Euler) ---
    vel_new = vel + (F_total / m) * dt
    pos_new = pos + vel_new * dt

    # --- Moments in body frame ---
    M_fluid = torch.tensor([fluid.mx, fluid.my, fluid.mz], dtype=torch.float64)

    # Gyroscopic term: ω × (I ω)
    Iw = I @ omega_body
    gyro = torch.linalg.cross(omega_body, Iw)
    alpha_body = I_inv @ (M_fluid - gyro)

    # Apply rotational DOF constraints
    constraints_rot = torch.tensor([
        0.0 if body.fix_roll  else 1.0,
        0.0 if body.fix_pitch else 1.0,
        0.0 if body.fix_yaw   else 1.0,
    ], dtype=torch.float64)
    alpha_body = alpha_body * constraints_rot

    omega_new = omega_body + alpha_body * dt
    omega_new = omega_new * constraints_rot   # re-apply clamp

    # --- Update quaternion ---
    quat_new = _update_quaternion(quat, omega_new, dt)

    return pos_new, vel_new, quat_new, omega_new


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _simple_sinusoidal_forces(
    t: float,
    amplitude: float = 10.0,
    frequency: float = 0.5,
) -> FluidForcesMoments:
    """Generate a simple sinusoidal fluid force for demonstration purposes."""
    return FluidForcesMoments(
        fx=amplitude * math.sin(2 * math.pi * frequency * t),
        fy=0.0,
        fz=0.0,
        mx=0.0,
        my=amplitude * 0.1 * math.cos(2 * math.pi * frequency * t),
        mz=0.0,
    )


def run_sixdof_simulation(
    cfg: SixDOFConfig,
    fluid_forces_fn=None,
) -> SixDOFResult:
    """Run a 6-DOF rigid-body simulation.

    Parameters
    ----------
    cfg:
        Simulation configuration.
    fluid_forces_fn:
        Callable ``(t, pos, vel, quat, omega) -> FluidForcesMoments``.
        If None, a simple sinusoidal demonstration force is used.
    """
    if fluid_forces_fn is None:
        fluid_forces_fn = lambda t, *args: _simple_sinusoidal_forces(t)  # noqa: E731

    # Initialise state
    pos   = torch.tensor(cfg.pos_init,    dtype=torch.float64)
    vel   = torch.tensor(cfg.vel_init,    dtype=torch.float64)
    quat  = torch.tensor(cfg.quat_init,   dtype=torch.float64)
    omega = torch.tensor(cfg.omega_init,  dtype=torch.float64)
    quat  = quat / quat.norm()

    history: list[SixDOFState] = []
    t = 0.0

    for step in range(cfg.n_steps + 1):
        # Record state
        R = quaternion_to_rotation_matrix(quat)
        roll, pitch, yaw = rotation_matrix_to_euler(R)
        history.append(SixDOFState(
            time=t,
            pos=pos.tolist(),
            vel=vel.tolist(),
            quat=quat.tolist(),
            omega_body=omega.tolist(),
            euler_deg=[
                math.degrees(roll),
                math.degrees(pitch),
                math.degrees(yaw),
            ],
        ))

        if step == cfg.n_steps:
            break

        # Get fluid forces
        fluid = fluid_forces_fn(t, pos, vel, quat, omega)
        # Integrate
        pos, vel, quat, omega = step_sixdof(pos, vel, quat, omega, fluid, cfg.body, cfg.dt)
        t += cfg.dt

    # Summary statistics
    displacements = [
        math.sqrt(sum(x**2 for x in s.pos))
        for s in history
    ]
    velocities = [math.sqrt(sum(v**2 for v in s.vel)) for s in history]
    rolls  = [abs(s.euler_deg[0]) for s in history]
    pitches = [abs(s.euler_deg[1]) for s in history]

    return SixDOFResult(
        history=history,
        max_displacement=max(displacements),
        max_velocity=max(velocities),
        max_roll_deg=max(rolls),
        max_pitch_deg=max(pitches),
    )
