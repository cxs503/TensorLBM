"""LBM-DEM coupling via Immersed Boundary Method.

Couples the LBM fluid solver with DEM particle solver using the
immersed boundary method (IBM) for fluid-particle interaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .dem import DEMSolver, DEMConfig, IceParticle
from .ibm import (
    ibm_velocity_interpolate_3d,
    ibm_force_spread_3d,
    ibm_delta_4pt,
)


@dataclass
class LBMDemCouplingConfig:
    """Configuration for LBM-DEM coupling.
    
    Parameters
    ----------
    ibm_kernel : str
        IBM kernel type: "hat" or "4pt".
    no_slip_factor : float
        Factor for no-slip boundary enforcement (1.0 = full no-slip).
    enable_particle_rotation : bool
        Whether to compute torque from fluid and update particle rotation.
    """
    ibm_kernel: str = "4pt"
    no_slip_factor: float = 1.0
    enable_particle_rotation: bool = True


class LBMDemCoupling:
    """LBM-DEM coupling via immersed boundary method.
    
    The coupling works as follows:
    1. Interpolate fluid velocity to particle surface markers using IBM.
    2. Compute drag force on each particle from the velocity difference.
    3. Spread the reaction force back to the fluid grid using IBM.
    4. Apply the force to the LBM distribution function.
    
    Parameters
    ----------
    dem_solver : DEMSolver
        DEM solver for particle dynamics.
    coupling_config : LBMDemCouplingConfig
        Coupling configuration.
    """
    
    def __init__(
        self,
        dem_solver: DEMSolver,
        coupling_config: LBMDemCouplingConfig = LBMDemCouplingConfig(),
    ):
        self.dem = dem_solver
        self.config = coupling_config
        
        # Particle surface markers (for IBM)
        self._markers: List[torch.Tensor] = []
        self._marker_forces: List[torch.Tensor] = []
        
        # Statistics
        self.total_drag_force = torch.zeros(3, dtype=torch.float64)
        self.total_drag_torque = torch.zeros(3, dtype=torch.float64)
    
    def generate_surface_markers(
        self,
        particle: IceParticle,
        n_markers: int = 50,
    ) -> torch.Tensor:
        """Generate surface markers for a spherical particle.
        
        Parameters
        ----------
        particle : IceParticle
            The particle.
        n_markers : int
            Approximate number of surface markers.
        
        Returns
        -------
        torch.Tensor
            Marker positions, shape (M, 3) where M ≈ n_markers.
        """
        # Use Fibonacci sphere for uniform distribution
        indices = torch.arange(0, n_markers, dtype=torch.float64)
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        
        # Polar angle
        theta = torch.acos(1.0 - 2.0 * (indices + 0.5) / n_markers)
        # Azimuthal angle
        phi = 2.0 * math.pi * indices / golden_ratio
        
        # Convert to Cartesian
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        
        # Scale by radius and translate to particle position
        markers = particle.radius * torch.stack([x, y, z], dim=1)
        markers = markers + particle.position.unsqueeze(0)
        
        return markers
    
    def interpolate_velocity_to_markers(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        markers: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate fluid velocity to marker positions.
        
        Parameters
        ----------
        ux, uy, uz : torch.Tensor
            Fluid velocity components, shape (nz, ny, nx).
        markers : torch.Tensor
            Marker positions, shape (M, 3).
        
        Returns
        -------
        torch.Tensor
            Interpolated velocities at markers, shape (M, 3).
        """
        marker_x = markers[:, 0]
        marker_y = markers[:, 1]
        marker_z = markers[:, 2]
        
        u_mx, u_my, u_mz = ibm_velocity_interpolate_3d(
            ux, uy, uz,
            marker_x, marker_y, marker_z,
            kernel=self.config.ibm_kernel,
        )
        
        return torch.stack([u_mx, u_my, u_mz], dim=1)
    
    def compute_particle_forces(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        dx: float,
        dt: float,
        fluid_density: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute drag forces on all particles.
        
        Parameters
        ----------
        ux, uy, uz : torch.Tensor
            Fluid velocity components, shape (nz, ny, nx).
        dx : float
            Grid spacing in meters.
        dt : float
            Time step in seconds.
        fluid_density : float
            Fluid density in kg/m³.
        
        Returns
        -------
        forces : torch.Tensor
            Force on each particle, shape (N, 3) in Newtons.
        torques : torch.Tensor
            Torque on each particle, shape (N, 3) in N·m.
        grid_forces : tuple of torch.Tensor
            Force fields (fx, fy, fz) to apply to fluid, each shape (nz, ny, nx).
        """
        n_particles = len(self.dem.system)
        device = ux.device
        
        forces = torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        torques = torch.zeros((n_particles, 3), dtype=torch.float64, device=device)
        
        # Accumulate grid forces
        nz, ny, nx = ux.shape
        fx_grid = torch.zeros((nz, ny, nx), dtype=ux.dtype, device=device)
        fy_grid = torch.zeros((nz, ny, nx), dtype=ux.dtype, device=device)
        fz_grid = torch.zeros((nz, ny, nx), dtype=ux.dtype, device=device)
        
        for i, particle in enumerate(self.dem.system):
            # Generate surface markers
            markers = self.generate_surface_markers(particle, n_markers=50)
            n_markers = markers.shape[0]
            
            # Interpolate fluid velocity to markers
            u_fluid = self.interpolate_velocity_to_markers(ux, uy, uz, markers)
            
            # Particle velocity at markers (rigid body motion)
            r = markers - particle.position.unsqueeze(0)
            u_solid = particle.velocity.unsqueeze(0) + torch.cross(
                particle.angular_velocity.unsqueeze(0).expand(n_markers, 3),
                r,
            )
            
            # Velocity difference (desired velocity change)
            du = self.config.no_slip_factor * (u_solid - u_fluid)
            
            # Force on markers (direct forcing)
            # F_marker = ρ_f * V_marker * du / dt
            # where V_marker ≈ (4/3 π R³) / n_markers
            v_marker = particle.volume / n_markers
            f_marker = fluid_density * v_marker * du / dt
            
            # Total force on particle
            forces[i] = f_marker.sum(dim=0)
            
            # Torque on particle
            torques[i] = torch.cross(r, f_marker).sum(dim=0)
            
            # Spread force to grid using IBM
            marker_x = markers[:, 0]
            marker_y = markers[:, 1]
            marker_z = markers[:, 2]
            
            fx_marker = f_marker[:, 0]
            fy_marker = f_marker[:, 1]
            fz_marker = f_marker[:, 2]
            
            # Spread all force components at once
            fx_spread, fy_spread, fz_spread = ibm_force_spread_3d(
                fx_marker, fy_marker, fz_marker,
                marker_x, marker_y, marker_z,
                nz, ny, nx, kernel=self.config.ibm_kernel,
            )
            
            fx_grid += fx_spread
            fy_grid += fy_spread
            fz_grid += fz_spread
        
        # Update statistics
        self.total_drag_force = forces.sum(dim=0)
        self.total_drag_torque = torques.sum(dim=0)
        
        return forces, torques, (fx_grid, fy_grid, fz_grid)
    
    def apply_forces_to_lbm(
        self,
        f: torch.Tensor,
        fx_grid: torch.Tensor,
        fy_grid: torch.Tensor,
        fz_grid: torch.Tensor,
        tau: float,
        dx: float,
        dt: float,
    ) -> torch.Tensor:
        """Apply IBM body forces to LBM distribution function.
        
        Uses Guo forcing scheme for D3Q19.
        
        Parameters
        ----------
        f : torch.Tensor
            Distribution function, shape (19, nz, ny, nx).
        fx_grid, fy_grid, fz_grid : torch.Tensor
            Body force components, each shape (nz, ny, nx).
        tau : float
            Relaxation time.
        dx : float
            Grid spacing.
        dt : float
            Time step.
        
        Returns
        -------
        torch.Tensor
            Updated distribution function.
        """
        from .d3q19 import C, W
        
        device = f.device
        c = C.to(device).float()
        w = W.to(device).float()
        
        # Guo forcing factor
        guo_factor = 1.0 - 1.0 / (2.0 * tau)
        
        # Force term: (1 - 1/(2τ)) * w_i * 3 * (c_i · F)
        cx = c[:, 0].view(19, 1, 1, 1)
        cy = c[:, 1].view(19, 1, 1, 1)
        cz = c[:, 2].view(19, 1, 1, 1)
        w_view = w.view(19, 1, 1, 1)
        
        forcing = w_view * 3.0 * guo_factor * (
            cx * fx_grid.unsqueeze(0)
            + cy * fy_grid.unsqueeze(0)
            + cz * fz_grid.unsqueeze(0)
        )
        
        return f + forcing


def create_ice_particles(
    n_particles: int,
    radius_range: Tuple[float, float],
    position_range: Tuple[torch.Tensor, torch.Tensor],
    density: float = 917.0,
    seed: int = 0,
) -> List[IceParticle]:
    """Create a list of ice particles with random positions and sizes.
    
    Parameters
    ----------
    n_particles : int
        Number of particles.
    radius_range : tuple of float
        (min_radius, max_radius) in meters.
    position_range : tuple of torch.Tensor
        (min_pos, max_pos) for random positions.
    density : float
        Particle density in kg/m³.
    seed : int
        Random seed.
    
    Returns
    -------
    list of IceParticle
        List of created particles.
    """
    torch.manual_seed(seed)
    
    particles = []
    for _ in range(n_particles):
        # Random radius
        radius = radius_range[0] + torch.rand(1).item() * (radius_range[1] - radius_range[0])
        
        # Random position
        pos = position_range[0] + torch.rand(3) * (position_range[1] - position_range[0])
        
        # Initial velocity (zero)
        vel = torch.zeros(3, dtype=torch.float64)
        
        # Initial angular velocity (zero)
        omega = torch.zeros(3, dtype=torch.float64)
        
        particle = IceParticle(
            position=pos,
            velocity=vel,
            angular_velocity=omega,
            radius=radius,
            density=density,
        )
        particles.append(particle)
    
    return particles
