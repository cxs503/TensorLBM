"""DEM solver for ice particle dynamics.

Implements time integration of particle equations of motion with
contact detection and force computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .particle import IceParticle, ParticleSystem
from .contact import (
    ContactModel,
    LinearSpringDashpot,
    detect_particle_particle_contact,
    detect_particle_wall_contact,
)


@dataclass
class DEMConfig:
    """Configuration for DEM solver.
    
    Parameters
    ----------
    time_step : float
        Time step size in seconds.
    gravity : torch.Tensor
        Gravitational acceleration vector (gx, gy, gz) in m/s².
    contact_model : ContactModel
        Contact mechanics model.
    domain_bounds : tuple of torch.Tensor
        Domain boundaries as (min_corner, max_corner) in meters.
    enable_wall_contacts : bool
        Whether to compute particle-wall contacts.
    drag_model : str
        Drag model type: "stokes", "schiller_naumann", or "none".
    fluid_density : float
        Fluid density in kg/m³ (for drag computation).
    fluid_viscosity : float
        Fluid dynamic viscosity in Pa·s (for drag computation).
    """
    time_step: float = 1e-4
    gravity: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, -9.81]))
    contact_model: ContactModel = field(default_factory=LinearSpringDashpot)
    domain_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    enable_wall_contacts: bool = True
    drag_model: str = "schiller_naumann"
    fluid_density: float = 1000.0
    fluid_viscosity: float = 1e-3
    
    def __post_init__(self):
        if isinstance(self.gravity, (list, tuple)):
            self.gravity = torch.tensor(self.gravity, dtype=torch.float64)


class DEMSolver:
    """DEM solver for ice particle dynamics.
    
    Integrates Newton's equations for translation and rotation:
    
        m * dv/dt = F_gravity + F_contact + F_drag + F_fluid
        I * dω/dt = T_contact + T_fluid
    
    Parameters
    ----------
    config : DEMConfig
        Solver configuration.
    """
    
    def __init__(self, config: DEMConfig):
        self.config = config
        self.system: Optional[ParticleSystem] = None
        self.time = 0.0
        self.step_count = 0
        
        # Storage for external forces (from fluid coupling)
        self._external_forces: Optional[torch.Tensor] = None
        self._external_torques: Optional[torch.Tensor] = None
        
        # History tracking
        self.history: List[dict] = []
    
    def initialize(self, particles: List[IceParticle]):
        """Initialize the particle system.
        
        Parameters
        ----------
        particles : list of IceParticle
            List of particles to simulate.
        """
        self.system = ParticleSystem(particles, self.config.gravity)
        self.time = 0.0
        self.step_count = 0
        self.history.clear()
    
    def set_external_forces(
        self,
        forces: torch.Tensor,
        torques: Optional[torch.Tensor] = None,
    ):
        """Set external forces and torques from fluid coupling.
        
        Parameters
        ----------
        forces : torch.Tensor
            External force on each particle, shape (N, 3) in Newtons.
        torques : torch.Tensor, optional
            External torque on each particle, shape (N, 3) in N·m.
        """
        self._external_forces = forces
        self._external_torques = torques
    
    def compute_drag_force(
        self,
        particle: IceParticle,
        fluid_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drag force on a particle.
        
        Parameters
        ----------
        particle : IceParticle
            The particle.
        fluid_velocity : torch.Tensor
            Fluid velocity at particle position in m/s.
        
        Returns
        -------
        torch.Tensor
            Drag force vector in Newtons.
        """
        if self.config.drag_model == "none":
            return torch.zeros(3, dtype=particle.position.dtype, device=particle.position.device)
        
        # Relative velocity
        v_rel = fluid_velocity - particle.velocity
        v_rel_mag = torch.norm(v_rel).item()
        
        if v_rel_mag < 1e-10:
            return torch.zeros(3, dtype=particle.position.dtype, device=particle.position.device)
        
        # Particle Reynolds number
        Re_p = (
            self.config.fluid_density
            * v_rel_mag
            * 2.0 * particle.radius
            / self.config.fluid_viscosity
        )
        
        # Drag coefficient
        if self.config.drag_model == "stokes":
            # Stokes drag: C_D = 24/Re
            C_D = 24.0 / max(Re_p, 1e-6)
        elif self.config.drag_model == "schiller_naumann":
            # Schiller-Naumann correlation
            if Re_p < 1000:
                C_D = (24.0 / max(Re_p, 1e-6)) * (1.0 + 0.15 * Re_p**0.687)
            else:
                C_D = 0.44
        else:
            raise ValueError(f"Unknown drag model: {self.config.drag_model}")
        
        # Drag force: F_D = 0.5 * C_D * ρ_f * A * |v_rel| * v_rel
        A = particle.cross_section_area
        f_drag = 0.5 * C_D * self.config.fluid_density * A * v_rel_mag * v_rel
        
        return f_drag
    
    def compute_contact_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute all contact forces.
        
        Returns
        -------
        forces : torch.Tensor
            Contact force on each particle, shape (N, 3) in Newtons.
        torques : torch.Tensor
            Contact torque on each particle, shape (N, 3) in N·m.
        """
        n_particles = len(self.system)
        device = self.system[0].position.device
        forces = torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        torques = torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        
        # Reset tangential history
        self.config.contact_model.reset_tangential_history()
        
        # Particle-particle contacts
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                p1 = self.system[i]
                p2 = self.system[j]
                
                in_contact, overlap, normal = detect_particle_particle_contact(p1, p2)
                
                if in_contact:
                    # Relative velocity at contact point
                    r1 = normal * p1.radius
                    r2 = -normal * p2.radius
                    v1_contact = p1.velocity + torch.linalg.cross(p1.angular_velocity, r1)
                    v2_contact = p2.velocity + torch.linalg.cross(p2.angular_velocity, r2)
                    v_rel = v1_contact - v2_contact
                    
                    # Compute contact force
                    contact_info = self.config.contact_model.compute_contact_force(
                        p1, p2, normal, overlap, v_rel
                    )
                    
                    # Apply Newton's third law
                    forces[i] += contact_info.total_force
                    forces[j] -= contact_info.total_force
                    
                    # Torques
                    torques[i] += torch.linalg.cross(r1, contact_info.total_force)
                    torques[j] += torch.linalg.cross(r2, -contact_info.total_force)
        
        # Particle-wall contacts
        if self.config.enable_wall_contacts and self.config.domain_bounds is not None:
            min_corner, max_corner = self.config.domain_bounds
            
            for i, particle in enumerate(self.system):
                # Check each wall (6 faces of bounding box)
                for dim in range(3):
                    # Min wall
                    wall_pos = min_corner.clone()
                    wall_pos[dim] = min_corner[dim]
                    wall_normal = torch.zeros(3, dtype=torch.float64, device=device)
                    wall_normal[dim] = 1.0
                    
                    in_contact, overlap, normal = detect_particle_wall_contact(
                        particle, wall_pos, wall_normal
                    )
                    if in_contact:
                        v_rel = particle.velocity
                        contact_info = self.config.contact_model.compute_contact_force(
                            particle, None, normal, overlap, v_rel
                        )
                        forces[i] += contact_info.total_force
                        r = -normal * particle.radius
                        torques[i] += torch.linalg.cross(r, contact_info.total_force)
                    
                    # Max wall
                    wall_pos = max_corner.clone()
                    wall_pos[dim] = max_corner[dim]
                    wall_normal = torch.zeros(3, dtype=torch.float64, device=device)
                    wall_normal[dim] = -1.0
                    
                    in_contact, overlap, normal = detect_particle_wall_contact(
                        particle, wall_pos, wall_normal
                    )
                    if in_contact:
                        v_rel = particle.velocity
                        contact_info = self.config.contact_model.compute_contact_force(
                            particle, None, normal, overlap, v_rel
                        )
                        forces[i] += contact_info.total_force
                        r = -normal * particle.radius
                        torques[i] += torch.linalg.cross(r, contact_info.total_force)
        
        return forces, torques
    
    def step(self, fluid_velocities: Optional[torch.Tensor] = None):
        """Advance one time step.
        
        Parameters
        ----------
        fluid_velocities : torch.Tensor, optional
            Fluid velocity at each particle position, shape (N, 3) in m/s.
            If a single vector (3,) is provided, it's broadcast to all particles.
            If None, no drag force is computed.
        """
        if self.system is None:
            raise RuntimeError("DEM solver not initialized. Call initialize() first.")
        
        n_particles = len(self.system)
        device = self.system[0].position.device
        dt = self.config.time_step
        
        # Compute forces
        contact_forces, contact_torques = self.compute_contact_forces()
        
        # Gravity
        gravity_forces = self.config.gravity.unsqueeze(0) * torch.tensor(
            [p.mass for p in self.system], dtype=torch.float64, device=device
        ).unsqueeze(1)
        
        # Drag forces
        drag_forces = torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        if fluid_velocities is not None:
            # Handle broadcasting: if single vector, expand to all particles
            if fluid_velocities.ndim == 1:
                fluid_velocities = fluid_velocities.unsqueeze(0).expand(n_particles, 3)
            
            for i, particle in enumerate(self.system):
                drag_forces[i] = self.compute_drag_force(particle, fluid_velocities[i])
        
        # External forces (from fluid coupling)
        external_forces = (
            self._external_forces
            if self._external_forces is not None
            else torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        )
        external_torques = (
            self._external_torques
            if self._external_torques is not None
            else torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        )
        
        # Total forces and torques
        total_forces = gravity_forces + contact_forces + drag_forces + external_forces
        total_torques = contact_torques + external_torques
        
        # Update velocities (semi-implicit Euler)
        for i, particle in enumerate(self.system):
            # Translation
            acceleration = total_forces[i] / particle.mass
            particle.velocity += acceleration * dt
            
            # Rotation
            angular_acceleration = total_torques[i] / particle.moment_of_inertia
            particle.angular_velocity += angular_acceleration * dt
            
            # Update positions
            particle.position += particle.velocity * dt
        
        # Update time
        self.time += dt
        self.step_count += 1
        
        # Record history
        self.history.append({
            "time": self.time,
            "kinetic_energy": self.system.total_kinetic_energy(),
            "momentum": self.system.total_momentum().tolist(),
        })
    
    def run(self, n_steps: int, fluid_velocities_fn=None):
        """Run simulation for multiple steps.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps.
        fluid_velocities_fn : callable, optional
            Function that takes (time, positions) and returns fluid velocities
            at particle positions, shape (N, 3).
        """
        for _ in range(n_steps):
            if fluid_velocities_fn is not None:
                positions = self.system.positions_tensor()
                fluid_velocities = fluid_velocities_fn(self.time, positions)
            else:
                fluid_velocities = None
            self.step(fluid_velocities)
