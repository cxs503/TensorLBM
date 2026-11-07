"""Stefan problem verification for ice melting in warm water.

This module provides verification of the Stefan problem using both:
1. Neumann analytical solution (reference)
2. Simple finite difference solver (verification baseline)
3. LBM solver framework (work in progress)

The Stefan problem is a classic moving boundary problem where ice melts
in warm water. The melting front position follows δ(t) = 2λ√(α·t).

Physics:
- One-phase Stefan problem: semi-infinite ice initially at T_melt
- Two-phase Stefan problem: ice initially at T_i < T_melt
- Neumann solution provides exact melting front position

References:
- Neumann (1868): Analytical solution
- Alexiades & Solomon (1993): Mathematical modeling of solid-liquid phase change
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf, erfc

# Physical constants for ice-water system
RHO_WATER = 1000.0  # kg/m³
RHO_ICE = 917.0  # kg/m³
CP_WATER = 4186.0  # J/(kg·K)
CP_ICE = 2090.0  # J/(kg·K)
K_WATER = 0.6  # W/(m·K)
K_ICE = 2.22  # W/(m·K)
L_LATENT = 334000.0  # J/kg (latent heat of fusion)
T_MELT = 0.0  # °C


@dataclass
class StefanProblemConfig:
    """Configuration for Stefan problem."""
    L: float = 0.1  # Characteristic length (m)
    T_warm: float = 10.0  # Warm water temperature (°C)
    T_initial: float = -10.0  # Initial ice temperature (°C)
    alpha_water: float = K_WATER / (RHO_WATER * CP_WATER)  # m²/s
    alpha_ice: float = K_ICE / (RHO_ICE * CP_ICE)  # m²/s
    
    @property
    def stefan_number(self) -> float:
        """Stefan number: Ste = c_p·(T_warm - T_melt) / L"""
        return CP_WATER * (self.T_warm - T_MELT) / L_LATENT


class NeumannSolution:
    """Neumann analytical solution for semi-infinite Stefan problem.
    
    The melting front position: δ(t) = 2λ√(α_w·t)
    where λ satisfies the transcendental equation.
    """
    
    def __init__(self, config: StefanProblemConfig):
        self.config = config
        self._lambda = None
    
    def _solve_lambda_one_phase(self, ste: float) -> float:
        """Solve for λ in one-phase Stefan problem using Newton iteration.
        
        Equation: λ·exp(λ²)·erf(λ) = Ste/(2√π)
        """
        lam = 0.1
        
        for _ in range(100):
            f = lam * np.exp(lam**2) * erf(lam) - ste / (2 * np.sqrt(np.pi))
            df = (np.exp(lam**2) * erf(lam) + 
                  2 * lam**2 * np.exp(lam**2) * erf(lam) + 
                  2 * lam / np.sqrt(np.pi))
            
            if abs(df) < 1e-15:
                break
            
            lam_new = lam - f / df
            if abs(lam_new - lam) < 1e-12:
                break
            lam = max(1e-10, lam_new)  # Keep positive
        
        return lam
    
    def _solve_lambda_two_phase(self, ste: float) -> float:
        """Solve for λ in two-phase Stefan problem (simplified)."""
        # Start with one-phase solution
        lam = self._solve_lambda_one_phase(ste)
        
        # Two-phase correction (approximate, applied once)
        kappa = self.config.alpha_ice / self.config.alpha_water
        ste_ice = CP_ICE * (T_MELT - self.config.T_initial) / L_LATENT
        
        # Apply correction factor once
        correction = 1.0 + ste_ice / (ste + 1e-10) * np.sqrt(kappa)
        lam = lam / np.sqrt(correction)
        
        return lam
    
    def compute_lambda(self) -> float:
        """Compute λ for the Stefan problem."""
        if self._lambda is not None:
            return self._lambda
        
        ste = self.config.stefan_number
        
        if self.config.T_initial >= T_MELT - 0.1:
            self._lambda = self._solve_lambda_one_phase(ste)
        else:
            self._lambda = self._solve_lambda_two_phase(ste)
        
        return self._lambda
    
    def melting_front_position(self, t: float) -> float:
        """Compute melting front position δ(t) = 2λ√(α_w·t)."""
        lam = self.compute_lambda()
        return 2 * lam * np.sqrt(self.config.alpha_water * t)
    
    def temperature_profile_water(self, x: float, t: float) -> float:
        """Temperature in water region (0 ≤ x < δ)."""
        lam = self.compute_lambda()
        delta = self.melting_front_position(t)
        
        if x >= delta or t <= 0:
            return T_MELT
        
        eta = x / (2 * np.sqrt(self.config.alpha_water * t))
        return self.config.T_warm * (1 - erf(eta) / erf(lam))
    
    def temperature_profile_ice(self, x: float, t: float) -> float:
        """Temperature in ice region (x ≥ δ)."""
        lam = self.compute_lambda()
        delta = self.melting_front_position(t)
        
        if x < delta or t <= 0:
            return T_MELT
        
        eta = x / (2 * np.sqrt(self.config.alpha_ice * t))
        lam_ice = lam * np.sqrt(self.config.alpha_water / self.config.alpha_ice)
        
        return (self.config.T_initial + 
                (T_MELT - self.config.T_initial) * erfc(eta) / erfc(lam_ice))


class FiniteDifferenceStefanSolver:
    """Simple finite difference solver for Stefan problem (1D).
    
    Uses enthalpy method on fixed grid.
    """
    
    def __init__(self, config: StefanProblemConfig, nx: int = 100, 
                 total_time: float = 100.0, n_steps: int = 1000):
        self.config = config
        self.nx = nx
        self.total_time = total_time
        self.n_steps = n_steps
        
        self.dx = 4.0 * config.L / nx  # Domain is 4L to capture semi-infinite
        self.dt = total_time / n_steps
        
        # Initialize fields
        self.x = np.linspace(0, 4 * config.L, nx)
        self.T = np.full(nx, config.T_initial)
        self.T[0] = config.T_warm  # Boundary condition
        
        self.f_l = np.zeros(nx)  # Liquid fraction
    
    def _compute_enthalpy(self, T: np.ndarray, f_l: np.ndarray) -> np.ndarray:
        """Compute enthalpy."""
        rho_cp = RHO_WATER * CP_WATER
        rho_L = RHO_WATER * L_LATENT
        return rho_cp * T + rho_L * f_l
    
    def _update_from_enthalpy(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update temperature and liquid fraction from enthalpy."""
        rho_cp_water = RHO_WATER * CP_WATER
        rho_cp_ice = RHO_ICE * CP_ICE
        rho_L = RHO_WATER * L_LATENT
        
        H_solid = rho_cp_ice * T_MELT
        H_liquid = H_solid + rho_L
        
        f_l = np.clip((H - H_solid) / rho_L, 0.0, 1.0)
        
        T = np.where(
            H < H_solid,
            H / rho_cp_ice,
            np.where(
                H > H_liquid,
                (H - rho_L) / rho_cp_water,
                np.full_like(H, T_MELT)
            )
        )
        
        return f_l, T
    
    def step(self) -> dict[str, float]:
        """Advance one time step."""
        # Compute enthalpy
        H = self._compute_enthalpy(self.T, self.f_l)
        
        # Diffusion (explicit FTCS)
        alpha_eff = self.config.alpha_water  # Simplified
        d2T_dx2 = (np.roll(self.T, -1) - 2 * self.T + np.roll(self.T, 1)) / self.dx**2
        d2T_dx2[0] = 0  # Boundary
        d2T_dx2[-1] = 0
        
        H_new = H + alpha_eff * RHO_WATER * CP_WATER * d2T_dx2 * self.dt
        
        # Update phase
        self.f_l, self.T = self._update_from_enthalpy(H_new)
        
        # Boundary conditions
        self.T[0] = self.config.T_warm
        self.f_l[0] = 1.0
        
        # Compute diagnostics
        melted_thickness = np.sum(self.f_l > 0.5) * self.dx
        
        return {
            "melted_thickness": melted_thickness,
            "max_T": self.T.max(),
            "min_T": self.T.min(),
        }
    
    def run(self, verbose: bool = False) -> dict[str, Any]:
        """Run the simulation."""
        history = []
        
        for step in range(self.n_steps):
            diag = self.step()
            time = (step + 1) * self.dt
            
            if verbose and step % max(1, self.n_steps // 10) == 0:
                print(f"Step {step}/{self.n_steps}, t={time:.2f}s, "
                      f"melted={diag['melted_thickness']:.6f}m")
            
            history.append({
                "step": step + 1,
                "time": time,
                **diag
            })
        
        return {
            "history": history,
            "final_T": self.T.copy(),
            "final_f_l": self.f_l.copy(),
            "x": self.x.copy(),
        }


def run_verification_case(
    T_warm: float,
    T_initial: float,
    output_dir: str = "runs/stefan_verification",
) -> dict[str, Any]:
    """Run a single verification case."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = StefanProblemConfig(T_warm=T_warm, T_initial=T_initial)
    
    # Analytical solution
    analytical = NeumannSolution(config)
    
    # Finite difference solver
    fd_solver = FiniteDifferenceStefanSolver(config, nx=200, total_time=100.0, n_steps=1000)
    fd_result = fd_solver.run(verbose=False)
    
    # Compare at final time
    t_final = 100.0
    delta_analytical = analytical.melting_front_position(t_final)
    delta_fd = fd_result["history"][-1]["melted_thickness"]
    
    rel_error = abs(delta_fd - delta_analytical) / (delta_analytical + 1e-10) * 100
    
    result = {
        "config": {
            "T_warm": T_warm,
            "T_initial": T_initial,
            "stefan_number": config.stefan_number,
            "lambda": analytical.compute_lambda(),
        },
        "analytical": {
            "melting_front_at_100s": delta_analytical,
        },
        "finite_difference": {
            "melting_front_at_100s": delta_fd,
        },
        "comparison": {
            "absolute_error": abs(delta_fd - delta_analytical),
            "relative_error_percent": rel_error,
        }
    }
    
    # Save result
    case_name = f"Tw{T_warm}_Ti{T_initial}"
    case_file = output_path / f"{case_name}.json"
    with open(case_file, "w") as f:
        json.dump(result, f, indent=2)
    
    return result


def run_parameter_sweep(
    T_warm_list: list[float] = [5.0, 10.0, 15.0, 20.0],
    T_initial_list: list[float] = [-5.0, -10.0, -20.0],
    output_dir: str = "runs/stefan_sweep",
) -> dict[str, Any]:
    """Run parameter sweep for Stefan problem."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for T_warm in T_warm_list:
        for T_initial in T_initial_list:
            case_name = f"Tw{T_warm}_Ti{T_initial}"
            print(f"\n{'='*60}")
            print(f"Running case: {case_name}")
            print(f"{'='*60}")
            
            result = run_verification_case(T_warm, T_initial, output_dir)
            results[case_name] = result
            
            print(f"λ = {result['config']['lambda']:.6f}")
            print(f"Analytical δ(100s) = {result['analytical']['melting_front_at_100s']:.6f} m")
            print(f"FD δ(100s) = {result['finite_difference']['melting_front_at_100s']:.6f} m")
            print(f"Relative error = {result['comparison']['relative_error_percent']:.2f}%")
    
    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Run a single test case
    result = run_verification_case(T_warm=10.0, T_initial=-10.0)
    
    print("\n" + "="*60)
    print("Verification Results:")
    print("="*60)
    print(f"Stefan number: {result['config']['stefan_number']:.6f}")
    print(f"λ (Neumann): {result['config']['lambda']:.6f}")
    print(f"\nAnalytical melting front at t=100s: {result['analytical']['melting_front_at_100s']:.6f} m")
    print(f"Finite difference at t=100s: {result['finite_difference']['melting_front_at_100s']:.6f} m")
    print(f"Relative error: {result['comparison']['relative_error_percent']:.2f}%")
