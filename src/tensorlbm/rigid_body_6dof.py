"""6-DOF rigid body motion solver for floating bodies in waves.

Implements the Cummins time-domain equation of motion for a floating body
with six degrees of freedom (surge, sway, heave, roll, pitch, yaw):

.. math::

    (M + A_\\infty) \\ddot{\\xi}(t) + \\int_0^t K(t-\\tau) \\dot{\\xi}(\\tau) d\\tau
    + C \\xi(t) = F_{exc}(t) + F_{ext}(t)

where:
- M is the body mass/inertia matrix (6×6)
- A_∞ is the infinite-frequency added mass matrix (6×6)
- K(t) is the retardation function matrix (6×6)
- C is the hydrostatic restoring matrix (6×6)
- F_exc(t) is the wave excitation force/moment vector (6,)
- F_ext(t) is any external force (mooring, PTO, etc.)

The retardation function is related to the frequency-domain radiation damping by:

.. math::

    K(t) = \\frac{2}{\\pi} \\int_0^\\infty B(\\omega) \\cos(\\omega t) d\\omega

References:
    Cummins, W.E. (1962). "The impulse response function and ship motions."
    Schiffstechnik, 9, 101-109.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BodyProperties6DOF:
    """Physical properties of a floating body for 6-DOF motion.

    Attributes:
        mass: Body mass [kg].
        cog: Centre of gravity (x, y, z) [m].
        inertia_matrix: 3×3 rotational inertia tensor about CoG [kg·m²].
        displacement_volume: Displaced water volume [m³].
        waterplane_area: Waterplane area A_wp [m²].
        water_density: Water density ρ [kg/m³].
        gravity: Gravitational acceleration g [m/s²].
    """

    mass: float
    cog: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_matrix: Optional[torch.Tensor] = None  # (3, 3)
    displacement_volume: float = 0.0
    waterplane_area: float = 0.0
    water_density: float = 1025.0
    gravity: float = 9.81

    def __post_init__(self):
        if self.inertia_matrix is None:
            # Default: uniform sphere approximation
            r_gyr = 0.3  # radius of gyration factor
            ixx = self.mass * (r_gyr * 2.0) ** 2
            iyy = self.mass * (r_gyr * 3.0) ** 2
            izz = self.mass * (r_gyr * 3.0) ** 2
            self.inertia_matrix = torch.diag(torch.tensor([ixx, iyy, izz]))

    def build_mass_matrix(self) -> torch.Tensor:
        """Build the 6×6 generalized mass matrix M.

        Returns:
            Tensor of shape (6, 6): [surge, sway, heave, roll, pitch, yaw].
        """
        M = torch.zeros(6, 6)
        m = self.mass
        # Translational DOFs
        M[0, 0] = m  # surge
        M[1, 1] = m  # sway
        M[2, 2] = m  # heave
        # Rotational DOFs
        if self.inertia_matrix is not None:
            M[3:6, 3:6] = self.inertia_matrix
        return M


@dataclass
class HydrostaticMatrix:
    """Hydrostatic restoring matrix C for a floating body.

    For a symmetric body at equilibrium, only heave, roll, and pitch
    have non-zero restoring coefficients:
    - C[2,2] = ρ g A_wp (heave)
    - C[3,3] = ρ g ∇ GM_T (roll)
    - C[4,4] = ρ g ∇ GM_L (pitch)

    Attributes:
        c_matrix: 6×6 restoring coefficient matrix.
    """

    c_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(6, 6))

    @classmethod
    def from_body(
        cls,
        body: BodyProperties6DOF,
        gm_transverse: float,
        gm_longitudinal: float,
    ) -> "HydrostaticMatrix":
        """Build hydrostatic matrix from body properties and metacentric heights.

        Args:
            body: Body physical properties.
            gm_transverse: Transverse metacentric height GM_T [m].
            gm_longitudinal: Longitudinal metacentric height GM_L [m].

        Returns:
            HydrostaticMatrix instance.
        """
        C = torch.zeros(6, 6)
        rho = body.water_density
        g = body.gravity
        # Heave restoring
        C[2, 2] = rho * g * body.waterplane_area
        # Roll restoring
        C[3, 3] = rho * g * body.displacement_volume * gm_transverse
        # Pitch restoring
        C[4, 4] = rho * g * body.displacement_volume * gm_longitudinal
        return cls(c_matrix=C)


@dataclass
class RadiationData:
    """Frequency-domain radiation data (added mass & damping).

    Attributes:
        omega: Angular frequency array [rad/s], shape (N_freq,).
        added_mass: Added mass A(ω), shape (N_freq, 6, 6).
        damping: Radiation damping B(ω), shape (N_freq, 6, 6).
        added_mass_inf: Infinite-frequency added mass A(∞), shape (6, 6).
    """

    omega: torch.Tensor
    added_mass: torch.Tensor  # (N_freq, 6, 6)
    damping: torch.Tensor  # (N_freq, 6, 6)
    added_mass_inf: torch.Tensor  # (6, 6)


def compute_retardation_function(
    radiation: RadiationData,
    t: torch.Tensor,
    n_omega_integration: int = 256,
) -> torch.Tensor:
    """Compute the retardation function K(t) from radiation damping.

    .. math::

        K_{ij}(t) = \\frac{2}{\\pi} \\int_0^\\infty B_{ij}(\\omega)
                     \\cos(\\omega t) d\\omega

    Uses trapezoidal integration over the available frequency range.

    Args:
        radiation: Frequency-domain radiation data.
        t: Time array, shape (N_t,).
        n_omega_integration: Number of frequency points for integration.

    Returns:
        Retardation function K(t), shape (N_t, 6, 6).
    """
    omega = radiation.omega  # (N_freq,)
    B = radiation.damping  # (N_freq, 6, 6)
    N_freq = omega.shape[0]
    N_t = t.shape[0]

    # Interpolate B to a finer frequency grid if needed
    if n_omega_integration > N_freq:
        omega_fine = torch.linspace(omega[0].item(), omega[-1].item(), n_omega_integration)
        B_fine = torch.zeros(n_omega_integration, 6, 6)
        for i in range(6):
            for j in range(6):
                B_fine[:, i, j] = torch.from_numpy(
                    __import__("numpy").interp(
                        omega_fine.numpy(), omega.numpy(), B[:, i, j].numpy()
                    )
                )
        omega_int = omega_fine
        B_int = B_fine
    else:
        omega_int = omega
        B_int = B

    d_omega = omega_int[1] - omega_int[0] if len(omega_int) > 1 else torch.tensor(1.0)

    # K(t) = (2/π) ∫ B(ω) cos(ωt) dω
    # Shape: (N_t, N_omega, 6, 6)
    cos_wt = torch.cos(omega_int.unsqueeze(0) * t.unsqueeze(1))  # (N_t, N_omega)
    # Integrate: sum over omega dimension
    K = (2.0 / math.pi) * torch.einsum("to,oij->tij", cos_wt, B_int) * d_omega

    return K


@dataclass
class State6DOF:
    """State vector for 6-DOF motion at a given time step.

    Attributes:
        position: Displacement ξ = [x, y, z, φ, θ, ψ], shape (6,).
        velocity: Velocity ξ̇ = [u, v, w, p, q, r], shape (6,).
        acceleration: Acceleration ξ̈, shape (6,).
        time_history: List of (time, position, velocity) tuples.
    """

    position: torch.Tensor = field(default_factory=lambda: torch.zeros(6))
    velocity: torch.Tensor = field(default_factory=lambda: torch.zeros(6))
    acceleration: torch.Tensor = field(default_factory=lambda: torch.zeros(6))
    time_history: list = field(default_factory=list)
    position_history: list = field(default_factory=list)
    velocity_history: list = field(default_factory=list)


def cummins_step(
    state: State6DOF,
    dt: float,
    F_exc: torch.Tensor,
    M: torch.Tensor,
    A_inf: torch.Tensor,
    C: torch.Tensor,
    K_history: Optional[torch.Tensor] = None,
    vel_history: Optional[torch.Tensor] = None,
    F_ext: Optional[torch.Tensor] = None,
) -> State6DOF:
    """Advance the Cummins equation by one time step (4th-order Runge-Kutta).

    Solves:
        (M + A_∞) ξ̈ = F_exc + F_ext - C ξ - ∫₀ᵗ K(t-τ) ξ̇(τ) dτ

    The convolution integral is approximated using stored velocity history.

    Args:
        state: Current 6-DOF state.
        dt: Time step size [s].
        F_exc: Excitation force vector at current time, shape (6,).
        M: Mass matrix, shape (6, 6).
        A_inf: Infinite-frequency added mass, shape (6, 6).
        C: Hydrostatic restoring matrix, shape (6, 6).
        K_history: Retardation function values K(t_n - t_k), shape (N_hist, 6, 6).
        vel_history: Past velocities ξ̇(t_k), shape (N_hist, 6).
        F_ext: External forces (mooring, etc.), shape (6,).

    Returns:
        Updated State6DOF.
    """
    if F_ext is None:
        F_ext = torch.zeros(6)

    M_total = M + A_inf  # (6, 6)
    M_inv = torch.linalg.inv(M_total)

    # Compute convolution integral (memory force)
    F_memory = torch.zeros(6)
    if K_history is not None and vel_history is not None:
        # ∫₀ᵗ K(t-τ) ξ̇(τ) dτ ≈ Σ K(t_n - t_k) ξ̇(t_k) dt
        # K_history: (N_hist, 6, 6), vel_history: (N_hist, 6)
        F_memory = torch.einsum("kij,kj->i", K_history, vel_history) * dt

    # Right-hand side: F = F_exc + F_ext - C*ξ - F_memory
    F_rhs = F_exc + F_ext - C @ state.position - F_memory

    # Acceleration: ξ̈ = M_total⁻¹ F_rhs
    accel = M_inv @ F_rhs

    # Simple Euler integration (RK4 can be added for higher accuracy)
    new_velocity = state.velocity + accel * dt
    new_position = state.position + new_velocity * dt

    new_state = State6DOF(
        position=new_position,
        velocity=new_velocity,
        acceleration=accel,
        time_history=state.time_history + [dt],
        position_history=state.position_history + [new_position.clone()],
        velocity_history=state.velocity_history + [new_velocity.clone()],
    )
    return new_state


def cummins_time_integration(
    body: BodyProperties6DOF,
    hydrostatics: HydrostaticMatrix,
    radiation: RadiationData,
    excitation_forces: torch.Tensor,
    dt: float,
    n_steps: int,
    F_ext_func=None,
    device: torch.device = torch.device("cpu"),
) -> State6DOF:
    """Full time-domain simulation of 6-DOF motion using Cummins equation.

    Args:
        body: Body physical properties.
        hydrostatics: Hydrostatic restoring data.
        radiation: Frequency-domain radiation data.
        excitation_forces: Time series of excitation forces, shape (n_steps, 6).
        dt: Time step [s].
        n_steps: Number of time steps.
        F_ext_func: Optional callable(step) -> Tensor(6,) for external forces.
        device: Computation device.

    Returns:
        Final State6DOF with full history.
    """
    M = body.build_mass_matrix().to(device)
    A_inf = radiation.added_mass_inf.to(device)
    C = hydrostatics.c_matrix.to(device)

    # Pre-compute retardation function
    t_retard = torch.arange(n_steps, device=device, dtype=torch.float32) * dt
    K_retard = compute_retardation_function(radiation, t_retard.cpu()).to(device)

    state = State6DOF(
        position=torch.zeros(6, device=device),
        velocity=torch.zeros(6, device=device),
        acceleration=torch.zeros(6, device=device),
    )

    # Store velocity history for convolution
    vel_history = []

    for step in range(n_steps):
        F_exc = excitation_forces[step].to(device)
        F_ext = F_ext_func(step) if F_ext_func is not None else torch.zeros(6, device=device)

        # Build convolution kernel and velocity history
        if len(vel_history) > 0:
            N_hist = len(vel_history)
            K_hist = K_retard[step - N_hist : step].flip(0)  # K(t_n - t_k)
            V_hist = torch.stack(vel_history)  # (N_hist, 6)
        else:
            K_hist = None
            V_hist = None

        state = cummins_step(
            state, dt, F_exc, M, A_inf, C,
            K_history=K_hist,
            vel_history=V_hist,
            F_ext=F_ext,
        )
        vel_history.append(state.velocity.clone())

        # Limit history length for efficiency (retardation decays)
        max_history = min(500, n_steps)
        if len(vel_history) > max_history:
            vel_history = vel_history[-max_history:]

    return state


def generate_jonswap_excitation(
    omega: torch.Tensor,
    spectrum: torch.Tensor,
    rao_complex: torch.Tensor,
    water_depth: float,
    n_components: int = 100,
    duration: float = 3600.0,
    dt: float = 0.01,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate irregular wave excitation forces from JONSWAP spectrum.

    Uses random phase method to decompose the spectrum into N regular
    wave components, then computes excitation forces using RAO.

    Args:
        omega: Frequency array [rad/s].
        spectrum: JONSWAP spectral density S(ω) [m²·s].
        rao_complex: Complex RAO for each DOF, shape (N_freq, 6).
        water_depth: Water depth [m].
        n_components: Number of wave components.
        duration: Simulation duration [s].
        dt: Time step [s].
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time_array, wave_elevation, excitation_forces):
        - time_array: shape (N_steps,)
        - wave_elevation: shape (N_steps,)
        - excitation_forces: shape (N_steps, 6)
    """
    torch.manual_seed(seed)

    # Select frequency components
    omega_min = omega[omega > 0.1].min().item() if (omega > 0.1).any() else 0.1
    omega_max = omega[-1].item()
    omega_comp = torch.linspace(omega_min, omega_max, n_components)
    d_omega = omega_comp[1] - omega_comp[0]

    # Interpolate spectrum to component frequencies
    S_comp = torch.zeros(n_components)
    for i, w in enumerate(omega_comp):
        idx = torch.argmin(torch.abs(omega - w))
        S_comp[i] = spectrum[idx]

    # Wave amplitudes from spectrum: a_i = sqrt(2 * S(ω_i) * dω)
    amplitudes = torch.sqrt(2.0 * S_comp * d_omega)

    # Random phases
    phases = 2.0 * math.pi * torch.rand(n_components)

    # Wave numbers from dispersion relation: ω² = gk·tanh(kh)
    g = 9.81
    k_comp = torch.zeros(n_components)
    for i, w in enumerate(omega_comp):
        # Newton iteration for dispersion relation
        k = w**2 / g  # deep water initial guess
        for _ in range(20):
            f = w**2 - g * k * math.tanh(k * water_depth)
            df = -g * (math.tanh(k * water_depth) + k * water_depth / math.cosh(k * water_depth) ** 2)
            k = k - f / df
        k_comp[i] = k

    # Time array
    n_steps = int(duration / dt)
    t = torch.arange(n_steps, dtype=torch.float32) * dt

    # Wave elevation: η(t) = Σ a_i cos(ω_i t + φ_i)
    eta = torch.zeros(n_steps)
    for i in range(n_components):
        eta += amplitudes[i] * torch.cos(omega_comp[i] * t + phases[i])

    # Excitation forces: F_exc(t) = Re{Σ RAO(ω_i) · a_i · exp(j(ω_i t + φ_i))}
    # Interpolate RAO to component frequencies
    excitation = torch.zeros(n_steps, 6)
    for i in range(n_components):
        idx = torch.argmin(torch.abs(omega - omega_comp[i]))
        rao_i = rao_complex[idx]  # (6,) complex
        for dof in range(6):
            excitation[:, dof] += (
                amplitudes[i]
                * (
                    rao_i[dof].real * torch.cos(omega_comp[i] * t + phases[i])
                    - rao_i[dof].imag * torch.sin(omega_comp[i] * t + phases[i])
                )
            )

    return t, eta, excitation
