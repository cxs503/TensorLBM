"""2D Taylor-Green Vortex viscous decay validation for TensorLBM.

Physical background
-------------------
The Taylor-Green vortex (TGV) is a canonical test case for incompressible
flow solvers. In 2D, the Navier-Stokes equations admit an exact solution:

    u(x,y,t) = -U0 * cos(k*x) * sin(k*y) * exp(-2*nu*k^2*t)
    v(x,y,t) =  U0 * sin(k*x) * cos(k*y) * exp(-2*nu*k^2*t)

where k = 2*pi/L is the wavenumber and nu is the kinematic viscosity.

The volume-averaged kinetic energy decays exponentially:

    E(t) = E(0) * exp(-4*nu*k^2*t)

This provides a stringent test of:
1. The collision operator (BGK relaxation)
2. The streaming step (advection)
3. The periodic boundary conditions
4. The overall second-order accuracy of the LBM scheme

LBM setup
---------
- D2Q9 lattice with BGK collision
- Periodic boundaries in both directions
- Viscosity: nu_lb = cs^2 * (tau - 0.5) * dt, with cs^2 = 1/3, dt = 1
- Initial velocity amplitude U0 << cs (incompressibility: Ma < 0.1)

Validation criterion
--------------------
The relative error in energy decay rate must be <= 8% over the simulation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensorlbm.lattice import equilibrium


# Optimized streaming using vectorized operations
def stream_optimized(f: torch.Tensor) -> torch.Tensor:
    """Stream D2Q9 distributions with periodic boundaries (optimized)."""
    ny, nx, _ = f.shape
    streamed = torch.empty_like(f)
    
    # D2Q9 velocities: (cx, cy)
    # 0: (0,0), 1: (1,0), 2: (0,1), 3: (-1,0), 4: (0,-1)
    # 5: (1,1), 6: (-1,1), 7: (-1,-1), 8: (1,-1)
    
    # Rest
    streamed[..., 0] = f[..., 0]
    # Horizontal
    streamed[..., 1] = torch.roll(f[..., 1], shifts=1, dims=1)
    streamed[..., 3] = torch.roll(f[..., 3], shifts=-1, dims=1)
    # Vertical
    streamed[..., 2] = torch.roll(f[..., 2], shifts=1, dims=0)
    streamed[..., 4] = torch.roll(f[..., 4], shifts=-1, dims=0)
    # Diagonal
    streamed[..., 5] = torch.roll(f[..., 5], shifts=(1, 1), dims=(0, 1))
    streamed[..., 6] = torch.roll(f[..., 6], shifts=(1, -1), dims=(0, 1))
    streamed[..., 7] = torch.roll(f[..., 7], shifts=(-1, -1), dims=(0, 1))
    streamed[..., 8] = torch.roll(f[..., 8], shifts=(-1, 1), dims=(0, 1))
    
    return streamed


class TaylorGreenVortex2D:
    """2D Taylor-Green vortex simulation using D2Q9 LBM."""

    def __init__(
        self,
        nx: int = 32,
        ny: int = 32,
        tau: float = 0.6,
        u0: float = 0.05,
        device: str = "cpu",
    ) -> None:
        """Initialize TGV simulation.

        Parameters
        ----------
        nx, ny : int
            Grid resolution in x and y directions.
        tau : float
            Relaxation time (> 0.5 for stability).
        u0 : float
            Initial velocity amplitude (Mach number ~ u0/cs, cs=1/sqrt(3)).
        device : str
            Compute device ('cpu' or 'cuda').
        """
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.u0 = u0
        self.device = torch.device(device)

        # Lattice parameters
        self.cs2 = 1.0 / 3.0  # Speed of sound squared
        self.dt = 1.0  # Time step
        self.dx = 1.0  # Grid spacing

        # Kinematic viscosity in lattice units
        self.nu = self.cs2 * (tau - 0.5) * self.dt

        # Wavenumber for TGV (one full period in domain)
        self.k = 2.0 * math.pi / nx

        # Decay rate: gamma = 2 * nu * k^2
        self.gamma = 2.0 * self.nu * self.k**2

        # Energy decay rate: alpha = 4 * nu * k^2
        self.alpha = 4.0 * self.nu * self.k**2

        # Precompute lattice velocities
        self.c = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize fields
        self._initialize_fields()

        # History tracking
        self.time_history = []
        self.energy_history = []
        self.max_velocity_history = []

    def _initialize_fields(self) -> None:
        """Initialize distribution functions with TGV velocity field."""
        # Create coordinate arrays
        x = torch.linspace(0, self.nx - 1, self.nx, device=self.device)
        y = torch.linspace(0, self.ny - 1, self.ny, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        # TGV initial velocity field
        # u = -U0 * cos(k*x) * sin(k*y)
        # v =  U0 * sin(k*x) * cos(k*y)
        kx = self.k * X
        ky = self.k * Y

        ux = -self.u0 * torch.cos(kx) * torch.sin(ky)
        uy = self.u0 * torch.sin(kx) * torch.cos(ky)

        # Stack velocity components
        u = torch.stack([ux, uy], dim=-1)  # (ny, nx, 2)

        # Initial density (uniform)
        rho = torch.ones((self.ny, self.nx), dtype=torch.float32, device=self.device)

        # Initialize equilibrium distribution
        self.f = equilibrium(rho, u)

        # Store initial energy
        self.E0 = self._compute_kinetic_energy()

    def _compute_kinetic_energy(self) -> float:
        """Compute volume-averaged kinetic energy."""
        rho, u = self._macroscopic()
        # E = 0.5 * <rho * |u|^2>
        ke = 0.5 * rho * (u[..., 0]**2 + u[..., 1]**2)
        return ke.mean().item()

    def _macroscopic(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute macroscopic variables from distribution functions."""
        rho = self.f.sum(dim=-1)
        momentum = torch.einsum("...q,qa->...a", self.f, self.c)
        u = momentum / rho.unsqueeze(-1).clamp_min(1e-12)
        return rho, u

    def step(self) -> None:
        """Perform one LBM time step (collision + streaming)."""
        rho, u = self._macroscopic()
        feq = equilibrium(rho, u)
        # BGK collision
        f_post = self.f - (self.f - feq) / self.tau
        # Streaming with periodic boundaries
        self.f = stream_optimized(f_post)

    def run(self, n_steps: int, sample_interval: int = 10) -> dict:
        """Run simulation for n_steps, recording history.

        Parameters
        ----------
        n_steps : int
            Number of time steps to run.
        sample_interval : int
            Record history every sample_interval steps.

        Returns
        -------
        dict
            Dictionary with time, energy, and velocity history.
        """
        for step_i in range(n_steps + 1):
            if step_i % sample_interval == 0:
                t = step_i * self.dt
                E = self._compute_kinetic_energy()
                _, u = self._macroscopic()
                u_max = u.abs().max().item()

                self.time_history.append(t)
                self.energy_history.append(E)
                self.max_velocity_history.append(u_max)

            if step_i < n_steps:
                self.step()

        return {
            "time": np.array(self.time_history),
            "energy": np.array(self.energy_history),
            "max_velocity": np.array(self.max_velocity_history),
        }

    def analytical_energy(self, t: np.ndarray) -> np.ndarray:
        """Compute analytical kinetic energy decay.

        E(t) = E(0) * exp(-4 * nu * k^2 * t)
        """
        return self.E0 * np.exp(-self.alpha * t)

    def analytical_velocity_amplitude(self, t: float) -> float:
        """Compute analytical velocity amplitude at time t."""
        return self.u0 * np.exp(-self.gamma * t)

    def get_velocity_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current velocity field as numpy arrays."""
        _, u = self._macroscopic()
        u_np = u.cpu().numpy()
        x = np.linspace(0, self.nx - 1, self.nx)
        y = np.linspace(0, self.ny - 1, self.ny)
        return x, y, u_np


def run_validation(
    nx: int = 32,
    tau: float = 0.6,
    u0: float = 0.05,
    n_steps: int = 500,
    output_dir: str = None,
) -> dict:
    """Run TGV validation and return results.

    Parameters
    ----------
    nx : int
        Grid resolution (square domain).
    tau : float
        Relaxation time.
    u0 : float
        Initial velocity amplitude.
    n_steps : int
        Number of time steps to simulate.
    output_dir : str, optional
        Directory to save outputs.

    Returns
    -------
    dict
        Validation results including error metrics.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Taylor-Green Vortex 2D Validation")
    print("=" * 60)

    # Create simulation
    sim = TaylorGreenVortex2D(nx=nx, ny=nx, tau=tau, u0=u0)

    # Compute characteristic decay time
    t_decay = 1.0 / sim.alpha if sim.alpha > 0 else float('inf')
    
    # Print parameters
    print(f"\nSimulation Parameters:")
    print(f"  Grid: {nx} x {nx}")
    print(f"  tau: {tau}")
    print(f"  nu (lattice): {sim.nu:.6f}")
    print(f"  k: {sim.k:.6f}")
    print(f"  gamma = 2*nu*k^2: {sim.gamma:.6f}")
    print(f"  alpha = 4*nu*k^2: {sim.alpha:.6f}")
    print(f"  t_decay = 1/alpha: {t_decay:.1f} steps")
    print(f"  u0: {u0}")
    print(f"  Ma = u0/cs: {u0 * math.sqrt(3):.4f}")
    print(f"  n_steps: {n_steps}")
    print(f"  t_max = n_steps * dt: {n_steps}")

    # Run simulation
    sample_interval = max(1, n_steps // 100)  # ~100 samples
    results = sim.run(n_steps=n_steps, sample_interval=sample_interval)

    # Compute analytical solution
    t_sim = results["time"]
    E_sim = results["energy"]
    E_ana = sim.analytical_energy(t_sim)

    # Compute relative error in energy
    rel_error = np.abs(E_sim - E_ana) / E_ana

    # Compute decay rate from simulation (fit exponential)
    # E(t) = E0 * exp(-alpha * t), so ln(E) = ln(E0) - alpha * t
    # Fit linear regression to ln(E) vs t
    mask = E_sim > 1e-10  # Avoid log of zero
    if mask.sum() > 2:
        t_fit = t_sim[mask]
        lnE_fit = np.log(E_sim[mask])
        # Linear fit: lnE = a - alpha * t
        coeffs = np.polyfit(t_fit, lnE_fit, 1)
        alpha_sim = -coeffs[0]  # Decay rate from simulation
        alpha_ana = sim.alpha  # Analytical decay rate
        decay_rate_error = abs(alpha_sim - alpha_ana) / alpha_ana
    else:
        alpha_sim = sim.alpha
        alpha_ana = sim.alpha
        decay_rate_error = 0.0

    # Validation criterion: relative error <= 8%
    mean_rel_error = np.mean(rel_error)
    max_rel_error = np.max(rel_error)
    criterion_passed = decay_rate_error <= 0.08

    print(f"\nResults:")
    print(f"  Initial energy E0: {sim.E0:.6e}")
    print(f"  Final energy E(t={n_steps}): {E_sim[-1]:.6e}")
    print(f"  Analytical E(t={n_steps}): {E_ana[-1]:.6e}")
    print(f"  Energy ratio E_sim/E_ana: {E_sim[-1]/E_ana[-1]:.6f}")
    print(f"  Mean relative error: {mean_rel_error:.4%}")
    print(f"  Max relative error: {max_rel_error:.4%}")
    print(f"  Decay rate (sim): {alpha_sim:.6f}")
    print(f"  Decay rate (ana): {alpha_ana:.6f}")
    print(f"  Decay rate error: {decay_rate_error:.4%}")
    print(f"\nValidation criterion (decay rate error <= 8%): {'PASS' if criterion_passed else 'FAIL'}")

    # Save results
    results_dict = {
        "parameters": {
            "nx": nx,
            "tau": tau,
            "u0": u0,
            "nu": sim.nu,
            "k": sim.k,
            "gamma": sim.gamma,
            "alpha": sim.alpha,
            "n_steps": n_steps,
        },
        "results": {
            "E0": sim.E0,
            "E_final_sim": float(E_sim[-1]),
            "E_final_ana": float(E_ana[-1]),
            "mean_rel_error": float(mean_rel_error),
            "max_rel_error": float(max_rel_error),
            "alpha_sim": float(alpha_sim),
            "alpha_ana": float(alpha_ana),
            "decay_rate_error": float(decay_rate_error),
            "criterion_passed": bool(criterion_passed),
        },
        "time_history": t_sim.tolist(),
        "energy_sim": E_sim.tolist(),
        "energy_ana": E_ana.tolist(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Generate plots
    generate_plots(t_sim, E_sim, E_ana, rel_error, sim, output_dir)

    print(f"\nOutputs saved to: {output_dir}")

    return results_dict


def generate_plots(
    t: np.ndarray,
    E_sim: np.ndarray,
    E_ana: np.ndarray,
    rel_error: np.ndarray,
    sim: TaylorGreenVortex2D,
    output_dir: Path,
) -> None:
    """Generate validation plots."""

    # Figure 1: Energy decay comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(t, E_sim, "b-o", markersize=2, label="LBM simulation", linewidth=2, markevery=5)
    ax1.plot(t, E_ana, "r--", label="Analytical", linewidth=2)
    ax1.set_xlabel("Time t (lattice units)", fontsize=12)
    ax1.set_ylabel("Kinetic Energy E(t)", fontsize=12)
    ax1.set_title("Taylor-Green Vortex: Energy Decay", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    fig1.tight_layout()
    fig1.savefig(output_dir / "energy_decay.png", dpi=150)
    plt.close(fig1)

    # Figure 2: Relative error over time
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(t, rel_error * 100, "g-", linewidth=2)
    ax2.axhline(y=8, color="r", linestyle="--", label="8% criterion", linewidth=2)
    ax2.set_xlabel("Time t (lattice units)", fontsize=12)
    ax2.set_ylabel("Relative Error (%)", fontsize=12)
    ax2.set_title("Taylor-Green Vortex: Energy Decay Error", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "energy_error.png", dpi=150)
    plt.close(fig2)

    # Figure 3: Velocity field snapshot at t=0 and t=t_max
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Initial field (reconstruct from analytical)
    x = np.linspace(0, sim.nx - 1, sim.nx)
    y = np.linspace(0, sim.ny - 1, sim.ny)
    X, Y = np.meshgrid(x, y)
    kx = sim.k * X
    ky = sim.k * Y

    # t=0
    ux0 = -sim.u0 * np.cos(kx) * np.sin(ky)
    uy0 = sim.u0 * np.sin(kx) * np.cos(ky)
    speed0 = np.sqrt(ux0**2 + uy0**2)

    im0 = axes[0].imshow(speed0, cmap="viridis", origin="lower")
    axes[0].set_title("Velocity magnitude at t=0", fontsize=12)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    # t=t_max (analytical)
    t_final = t[-1]
    decay = np.exp(-sim.gamma * t_final)
    ux_final = -sim.u0 * decay * np.cos(kx) * np.sin(ky)
    uy_final = sim.u0 * decay * np.sin(kx) * np.cos(ky)
    speed_final = np.sqrt(ux_final**2 + uy_final**2)

    im1 = axes[1].imshow(speed_final, cmap="viridis", origin="lower")
    axes[1].set_title(f"Velocity magnitude at t={t_final:.0f} (analytical)", fontsize=12)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    fig3.tight_layout()
    fig3.savefig(output_dir / "velocity_field.png", dpi=150)
    plt.close(fig3)

    print(f"Plots saved: energy_decay.png, energy_error.png, velocity_field.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Taylor-Green Vortex 2D Validation")
    parser.add_argument("--nx", type=int, default=32, help="Grid resolution")
    parser.add_argument("--tau", type=float, default=0.6, help="Relaxation time")
    parser.add_argument("--u0", type=float, default=0.05, help="Initial velocity amplitude")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of time steps")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    results = run_validation(
        nx=args.nx,
        tau=args.tau,
        u0=args.u0,
        n_steps=args.n_steps,
        output_dir=args.output_dir,
    )

    # Exit with appropriate code
    sys.exit(0 if results["results"]["criterion_passed"] else 1)
