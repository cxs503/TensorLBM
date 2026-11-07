"""Propeller-ice interaction benchmark.

Simulates the interaction between a marine propeller (modeled as actuator disk)
and ice particles using LBM-DEM coupling.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .actuator_disk import ActuatorDiskConfig, apply_actuator_disk, _interp_kt, _interp_kq
from .d3q19 import equilibrium3d, macroscopic3d
from .solver3d import stream3d
from .turbulence import collide_smagorinsky_mrt3d
from .dem import DEMSolver, DEMConfig, IceParticle
from .ice_particle_ibm import LBMDemCoupling, LBMDemCouplingConfig, create_ice_particles
from .utils import get_reproducibility_metadata, prepare_run_dir, resolve_device


@dataclass
class PropellerIceConfig:
    """Configuration for propeller-ice interaction benchmark.
    
    Parameters
    ----------
    # Domain
    nx, ny, nz : int
        Domain size in lattice units.
    dx : float
        Grid spacing in meters.
    dt : float
        Time step in seconds.
    
    # Propeller
    propeller_diameter : float
        Propeller diameter in lattice units.
    propeller_rpm : float
        Propeller rotation speed in revolutions per second.
    propeller_position : tuple of float
        Propeller disk center (x, y, z) in lattice units.
    
    # Ice particles
    n_particles : int
        Number of ice particles.
    particle_radius_range : tuple of float
        (min_radius, max_radius) in meters.
    particle_concentration : float
        Particle volume concentration (0-1).
    
    # Flow
    inflow_velocity : float
        Inflow velocity in m/s.
    fluid_density : float
        Fluid density in kg/m³.
    fluid_viscosity : float
        Fluid kinematic viscosity in m²/s.
    
    # Simulation
    n_steps : int
        Total simulation steps.
    output_interval : int
        Steps between output.
    output_root : Path
        Output directory.
    device : str
        Computation device.
    """
    # Domain
    nx: int = 200
    ny: int = 100
    nz: int = 100
    dx: float = 0.01  # 1 cm
    dt: float = 1e-4
    
    # Propeller
    propeller_diameter: float = 48.0  # lattice units
    propeller_rpm: float = 10.0  # rev/s
    propeller_position: Tuple[float, float, float] = (60.0, 50.0, 50.0)
    
    # Ice particles
    n_particles: int = 10
    particle_radius_range: Tuple[float, float] = (0.05, 0.15)  # meters
    particle_concentration: float = 0.05
    
    # Flow
    inflow_velocity: float = 2.0  # m/s
    fluid_density: float = 1025.0  # seawater kg/m³
    fluid_viscosity: float = 1.5e-6  # m²/s
    
    # Simulation
    n_steps: int = 5000
    output_interval: int = 100
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    device: str = "cpu"
    
    def __post_init__(self):
        self.output_root = Path(self.output_root)
        self.device = self.device.lower()


class PropellerIceBenchmark:
    """Propeller-ice interaction benchmark.
    
    Simulates a marine propeller operating in ice-infested waters using
    LBM-DEM coupling with actuator disk propeller model.
    """
    
    def __init__(self, config: PropellerIceConfig):
        self.config = config
        self.device = resolve_device(config.device)
        
        # Initialize LBM fields
        self._init_lbm_fields()
        
        # Initialize DEM solver
        self._init_dem_solver()
        
        # Initialize coupling
        self.coupling = LBMDemCoupling(self.dem_solver, LBMDemCouplingConfig())
        
        # Statistics
        self.thrust_history: List[float] = []
        self.torque_history: List[float] = []
        self.particle_ingestion_count = 0
        self.total_impact_force = 0.0
        
        # Output directory
        self.run_dir: Optional[Path] = None
    
    def _init_lbm_fields(self):
        """Initialize LBM distribution function."""
        c = self.config
        nx, ny, nz = c.nx, c.ny, c.nz
        
        # Convert inflow velocity to lattice units
        u_lattice = c.inflow_velocity * c.dt / c.dx
        
        # Initialize with uniform inflow
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=self.device)
        ux0 = torch.full_like(rho0, u_lattice)
        uy0 = torch.zeros_like(rho0)
        uz0 = torch.zeros_like(rho0)
        
        self.f = equilibrium3d(rho0, ux0, uy0, uz0, device=self.device)
        
        # Compute relaxation time
        # ν = (τ - 0.5) * c_s² * dt / dx²
        # τ = 0.5 + ν * dx² / (c_s² * dt)
        c_s_sq = 1.0 / 3.0
        nu_lattice = c.fluid_viscosity * c.dt / c.dx**2
        self.tau = 0.5 + nu_lattice / c_s_sq
        
        print(f"LBM initialized: τ={self.tau:.3f}, u_in={u_lattice:.4f} lu")
    
    def _init_dem_solver(self):
        """Initialize DEM solver with ice particles."""
        c = self.config
        
        # Create particles upstream of propeller
        min_pos = torch.tensor([10.0 * c.dx, 30.0 * c.dx, 30.0 * c.dx], dtype=torch.float64)
        max_pos = torch.tensor([40.0 * c.dx, 70.0 * c.dx, 70.0 * c.dx], dtype=torch.float64)
        
        particles = create_ice_particles(
            c.n_particles,
            c.particle_radius_range,
            (min_pos, max_pos),
            density=917.0,  # Ice density
            seed=42,
        )
        
        # DEM configuration
        dem_config = DEMConfig(
            time_step=c.dt,
            gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
            domain_bounds=(
                torch.zeros(3, dtype=torch.float64),
                torch.tensor([c.nx * c.dx, c.ny * c.dx, c.nz * c.dx], dtype=torch.float64),
            ),
            enable_wall_contacts=True,
            drag_model="schiller_naumann",
            fluid_density=c.fluid_density,
            fluid_viscosity=c.fluid_density * c.fluid_viscosity,
        )
        
        self.dem_solver = DEMSolver(dem_config)
        self.dem_solver.initialize(particles)
        
        print(f"DEM initialized: {c.n_particles} ice particles")
    
    def compute_thrust_torque(self) -> Tuple[float, float]:
        """Compute propeller thrust and torque from actuator disk forces."""
        c = self.config
        cx, cy, cz = c.propeller_position
        D = c.propeller_diameter
        R = D / 2.0
        
        # Get velocity at propeller disk
        nz, ny, nx = self.f.shape[1:]
        rho, ux, uy, uz = macroscopic3d(self.f)
        
        # Average inflow at disk
        disk_mask = (
            (torch.arange(nx, device=self.device) - cx).abs() <= 1.0
        )
        u_in = ux[:, :, disk_mask].mean().item()
        
        # Compute advance ratio
        n_rps = c.propeller_rpm
        J = u_in / (n_rps * D) if n_rps > 0 else 0.0
        
        # Interpolate KT, KQ
        kt = _interp_kt(max(0.1, min(1.1, J)))
        kq = _interp_kq(max(0.1, min(1.1, J)))
        
        # Compute thrust and torque
        rho_phys = c.fluid_density
        n2d4 = n_rps**2 * D**4
        n2d5 = n_rps**2 * D**5
        
        thrust = kt * rho_phys * n2d4 * (c.dx**2 / c.dt**2)  # Convert to physical units
        torque = kq * rho_phys * n2d5 * (c.dx**3 / c.dt**2)
        
        return thrust, torque
    
    def step(self):
        """Advance one time step."""
        c = self.config
        
        # 1. LBM collision + streaming
        self.f = collide_smagorinsky_mrt3d(self.f, tau=self.tau, C_s=0.1)
        self.f = stream3d(self.f)
        
        # 2. Get macroscopic fields
        rho, ux, uy, uz = macroscopic3d(self.f)
        
        # 3. Apply actuator disk forces
        cx, cy, cz = c.propeller_position
        fx_ad, fy_ad, fz_ad = apply_actuator_disk(
            ux, uy, uz,
            cx, cy, cz,
            c.propeller_diameter,
            hub_diameter_ratio=0.18,
            rpm=c.propeller_rpm,
            rho=1.0,
        )
        
        # Apply forces to velocity
        ux = ux + fx_ad
        uy = uy + fy_ad
        uz = uz + fz_ad
        
        # 4. LBM-DEM coupling
        # Get fluid velocities at particle positions
        positions = self.dem_solver.system.positions_tensor()
        n_particles = len(positions)
        
        if n_particles > 0:
            # Interpolate fluid velocity to particles
            fluid_velocities = torch.zeros((n_particles, 3), dtype=torch.float64, device=self.device)
            for i, pos in enumerate(positions):
                ix = int(pos[0].item() / c.dx)
                iy = int(pos[1].item() / c.dx)
                iz = int(pos[2].item() / c.dx)
                
                if 0 <= ix < c.nx and 0 <= iy < c.ny and 0 <= iz < c.nz:
                    fluid_velocities[i, 0] = ux[iz, iy, ix].item() * c.dx / c.dt
                    fluid_velocities[i, 1] = uy[iz, iy, ix].item() * c.dx / c.dt
                    fluid_velocities[i, 2] = uz[iz, iy, ix].item() * c.dx / c.dt
            
            # Compute IBM forces
            forces, torques, (fx_ibm, fy_ibm, fz_ibm) = self.coupling.compute_particle_forces(
                ux, uy, uz, c.dx, c.dt, c.fluid_density
            )
            
            # Apply IBM forces to LBM
            self.f = self.coupling.apply_forces_to_lbm(
                self.f, fx_ibm, fy_ibm, fz_ibm, self.tau, c.dx, c.dt
            )
            
            # Apply forces to DEM
            self.dem_solver.set_external_forces(forces, torques)
        
        # 5. DEM step
        self.dem_solver.step(fluid_velocities if n_particles > 0 else None)
        
        # 6. Compute thrust and torque
        thrust, torque = self.compute_thrust_torque()
        self.thrust_history.append(thrust)
        self.torque_history.append(torque)
    
    def run(self) -> dict:
        """Run the full benchmark.
        
        Returns
        -------
        dict
            Results dictionary with statistics and metadata.
        """
        c = self.config
        
        # Prepare output directory
        self.run_dir = prepare_run_dir(
            c.output_root,
            "propeller_ice",
            f"pi_n{c.n_particles}_D{int(c.propeller_diameter)}",
            overwrite=True,
        )
        
        # Save config
        config_dict = {
            "nx": c.nx,
            "ny": c.ny,
            "nz": c.nz,
            "dx": c.dx,
            "dt": c.dt,
            "propeller_diameter": c.propeller_diameter,
            "propeller_rpm": c.propeller_rpm,
            "n_particles": c.n_particles,
            "inflow_velocity": c.inflow_velocity,
            "n_steps": c.n_steps,
        }
        (self.run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
        
        print(f"\n{'='*60}")
        print(f"Propeller-Ice Interaction Benchmark")
        print(f"{'='*60}")
        print(f"Domain: {c.nx}×{c.ny}×{c.nz}")
        print(f"Propeller: D={c.propeller_diameter} lu, RPM={c.propeller_rpm}")
        print(f"Ice particles: {c.n_particles}")
        print(f"Steps: {c.n_steps}")
        print(f"Output: {self.run_dir}")
        print(f"{'='*60}\n")
        
        t_start = time.perf_counter()
        
        for step in range(1, c.n_steps + 1):
            self.step()
            
            if step % c.output_interval == 0:
                # Compute statistics
                thrust_mean = sum(self.thrust_history[-c.output_interval:]) / c.output_interval
                torque_mean = sum(self.torque_history[-c.output_interval:]) / c.output_interval
                
                print(f"Step {step}/{c.n_steps}: "
                      f"T={thrust_mean:.2f} N, Q={torque_mean:.3f} N·m")
        
        t_elapsed = time.perf_counter() - t_start
        
        # Final statistics
        results = {
            "metadata": get_reproducibility_metadata(),
            "config": config_dict,
            "thrust_mean": sum(self.thrust_history) / len(self.thrust_history),
            "thrust_rms": (
                sum((t - sum(self.thrust_history)/len(self.thrust_history))**2 
                    for t in self.thrust_history) / len(self.thrust_history)
            )**0.5,
            "torque_mean": sum(self.torque_history) / len(self.torque_history),
            "elapsed_time_s": t_elapsed,
            "n_particles": c.n_particles,
        }
        
        # Save results
        (self.run_dir / "results.json").write_text(json.dumps(results, indent=2))
        
        print(f"\n{'='*60}")
        print(f"Benchmark complete!")
        print(f"Mean thrust: {results['thrust_mean']:.2f} N")
        print(f"Mean torque: {results['torque_mean']:.3f} N·m")
        print(f"Elapsed time: {t_elapsed:.1f} s")
        print(f"{'='*60}\n")
        
        return results


def run_propeller_ice_benchmark(config: Optional[PropellerIceConfig] = None) -> dict:
    """Run propeller-ice interaction benchmark.
    
    Parameters
    ----------
    config : PropellerIceConfig, optional
        Configuration. If None, uses default.
    
    Returns
    -------
    dict
        Results dictionary.
    """
    if config is None:
        config = PropellerIceConfig()
    
    benchmark = PropellerIceBenchmark(config)
    return benchmark.run()
